"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
"""
from typing import Callable, Union
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.models.diffusion_policy_nets as DPNets
from robomimic.models.learned_temporal import (
    TemporalEncoder,
    HistoryDecoder,
    ImageEmbeddingFusion,
    MultiImageHistoryDecoder,
    gather_history_latents,
    history_reconstruction_loss,
    multi_image_history_reconstruction_loss,
)
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

import random
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils


@register_algo_factory_func("diffusion_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    if algo_config.unet.enabled:
        return DiffusionPolicyUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


class DiffusionPolicyUNet(PolicyAlgo):
    def _create_optimizers(self):
        """Put LTE-only modules in DP's policy optimizer as well.

        The base DP config intentionally has one ``policy`` optimizer, while
        LTE adds modules outside ``nets['policy']``. Without this, they would
        receive gradients but never take an optimizer step.
        """
        super(DiffusionPolicyUNet, self)._create_optimizers()
        if getattr(self, "learned_temporal_enabled", False):
            extra_modules = [self.nets["temporal_encoder"]]
            if "temporal_image_fusion" in self.nets:
                extra_modules.append(self.nets["temporal_image_fusion"])
            if "history_decoder" in self.nets:
                extra_modules.append(self.nets["history_decoder"])
            extra_params = [param for module in extra_modules for param in module.parameters()]
            if extra_params:
                self.optimizers["policy"].add_param_group({"params": extra_params})
                # The base class constructed this scheduler before LTE added
                # its parameter group. Recreate it so every group receives a
                # scheduled learning rate during the first optimizer step.
                self.lr_schedulers["policy"] = TorchUtils.lr_scheduler_from_optim_params(
                    net_optim_params=self.optim_params["policy"],
                    net=self.nets["policy"],
                    optimizer=self.optimizers["policy"],
                )

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]
        temporal_cfg = self.obs_config.temporal_encodings.learned_encoder
        self.learned_temporal_enabled = bool(temporal_cfg.enabled)
        self.learned_temporal_cfg = temporal_cfg
        self.learned_temporal_uses_image_embeddings = self.learned_temporal_enabled and (
            temporal_cfg.input_signal == "image_embedding" or bool(temporal_cfg.use_image_embedding)
        )
        all_image_keys = [
            key for key in self.obs_shapes if ObsUtils.key_is_obs_modality(key, "rgb")
        ]
        if self.learned_temporal_uses_image_embeddings:
            requested_images = getattr(temporal_cfg, "num_images", 1)
            if requested_images is None:
                temporal_image_count = len(all_image_keys)
            elif (
                isinstance(requested_images, bool)
                or not isinstance(requested_images, int)
                or not 1 <= requested_images <= len(all_image_keys)
            ):
                raise ValueError(
                    "learned_encoder.num_images must be between 1 and the number "
                    "of configured RGB observations, or null for all"
                )
            else:
                temporal_image_count = requested_images
            self.temporal_image_keys = all_image_keys[:temporal_image_count]
        else:
            self.temporal_image_keys = all_image_keys
        temporal_encoder = None
        history_decoder = None
        temporal_dim = 0
        if self.learned_temporal_enabled:
            if self.learned_temporal_uses_image_embeddings and not self.temporal_image_keys:
                raise ValueError("LTE-IMG-NoT requires at least one rgb observation key")
            image_embedding_dims = []
            if self.learned_temporal_uses_image_embeddings:
                image_encoder = obs_encoder.nets["obs"]
                image_embedding_dims = [
                    int(image_encoder.obs_nets[key].output_shape(self.obs_shapes[key])[0])
                    for key in self.temporal_image_keys
                ]
                use_temporal_image_fusion = (
                    len(image_embedding_dims) > 1
                    and bool(getattr(temporal_cfg, "multi_image_fusion", False))
                )
                image_embedding_dim = (
                    int(temporal_cfg.image_embedding_fusion_dim)
                    if use_temporal_image_fusion else sum(image_embedding_dims)
                )
            else:
                image_embedding_dim = 0
            temporal_encoder = TemporalEncoder(
                time_dim=0 if self.learned_temporal_uses_image_embeddings else 1,
                signal_dim=0 if self.learned_temporal_uses_image_embeddings else 1,
                image_embedding_dim=image_embedding_dim,
                latent_dim=temporal_cfg.latent_dim,
                hidden_dim=temporal_cfg.hidden_dim,
                recurrent=temporal_cfg.recurrent,
            )
            temporal_dim = int(temporal_cfg.latent_dim)
            if temporal_cfg.history_reconstruction.enabled:
                if self.learned_temporal_uses_image_embeddings and use_temporal_image_fusion:
                    history_decoder = MultiImageHistoryDecoder(
                        latent_dim=temporal_cfg.latent_dim,
                        image_embedding_dims=image_embedding_dims,
                        hidden_dim=temporal_cfg.hidden_dim,
                    )
                else:
                    history_decoder = HistoryDecoder(
                        latent_dim=temporal_cfg.latent_dim,
                        signal_dim=(image_embedding_dim if self.learned_temporal_uses_image_embeddings else 1),
                        hidden_dim=temporal_cfg.hidden_dim,
                    )

        # create network object
        noise_pred_net = DPNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=(obs_dim + temporal_dim) * self.algo_config.horizon.observation_horizon
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "noise_pred_net": noise_pred_net
            })
        })
        if temporal_encoder is not None:
            nets["temporal_encoder"] = temporal_encoder
        if self.learned_temporal_uses_image_embeddings and use_temporal_image_fusion:
            nets["temporal_image_fusion"] = ImageEmbeddingFusion(
                image_embedding_dims,
                output_dim=int(temporal_cfg.image_embedding_fusion_dim),
                hidden_dim=int(temporal_cfg.image_embedding_fusion_hidden_dim),
            )
        if history_decoder is not None:
            nets["history_decoder"] = history_decoder

        nets = nets.float().to(self.device)
        
        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError()
        
        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)
                
        # set attrs
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
        self.temporal_latent_queue = None
        self.temporal_z_prev = None
        self._temporal_image_dataset = None
        self._temporal_validation_image_dataset = None
        self._history_image_embedding_cache = None
        self._history_image_cache_epoch = None

    def set_temporal_image_dataset(self, dataset, validation_dataset=None):
        """Attach the training dataset used to build LTE's detached GPU cache."""
        if self.learned_temporal_uses_image_embeddings:
            self._temporal_image_dataset = dataset
            self._temporal_validation_image_dataset = validation_dataset

    @staticmethod
    def _temporal_image_cache_should_refresh(cfg, cache, cache_epoch, epoch):
        """Mirror TemporalDiffusionPolicy's warm-up and periodic refresh schedule."""
        epoch_index = int(epoch) - 1
        if epoch_index < int(cfg.image_embedding_cache_start_epoch):
            return False
        if cache is None:
            return True
        warmup_end = int(cfg.image_embedding_cache_start_epoch) + int(cfg.image_embedding_cache_warmup_epochs)
        return epoch_index < warmup_end or (epoch_index - warmup_end) % int(cfg.image_embedding_cache_refresh_epochs) == 0

    def _encode_temporal_image_views(self, images, nets=None):
        """Run the policy's per-camera visual towers, without altering DP features."""
        nets = self.nets if nets is None else nets
        encoder = nets["policy"]["obs_encoder"].nets["obs"]
        embeddings = []
        for key, image in zip(self.temporal_image_keys, images):
            value = image
            for randomizer in encoder.obs_randomizers[key]:
                if randomizer is not None:
                    value = randomizer.forward_in(value)
            value = encoder.obs_nets[key](value)
            if encoder.activation is not None:
                value = encoder.activation(value)
            for randomizer in reversed(encoder.obs_randomizers[key]):
                if randomizer is not None:
                    value = randomizer.forward_out(value)
            embeddings.append(TensorUtils.flatten(value, begin_axis=1))
        return embeddings

    def _fetch_temporal_image_views(self, indices, dataset):
        if dataset is None:
            raise RuntimeError("LTE-IMG-NoT training requires its SequenceDataset to be attached")
        indices = indices.detach().cpu().numpy()
        batches = dataset.get_temporal_image_batch(indices, self.temporal_image_keys)
        tensors = [torch.from_numpy(batches[key]).to(self.device, dtype=torch.float32) for key in self.temporal_image_keys]
        with torch.no_grad():
            return [embedding.detach() for embedding in self._encode_temporal_image_views(tensors)]

    def _build_history_image_embedding_cache(self):
        dataset = self._temporal_image_dataset
        if dataset is None:
            raise RuntimeError("LTE-IMG-NoT image cache has no training dataset")
        embedding_chunks = [[] for _ in self.temporal_image_keys]
        with torch.no_grad():
            for start in range(0, dataset._temporal_num_images, 256):
                indices = torch.arange(start, min(start + 256, dataset._temporal_num_images), device=self.device)
                views = self._fetch_temporal_image_views(indices, dataset)
                for chunks, view in zip(embedding_chunks, views):
                    chunks.append(view)
        return [torch.cat(chunks, dim=0) for chunks in embedding_chunks]

    def _get_history_image_embeddings(self, history_indices, history_mask, epoch, validate):
        dataset = self._temporal_validation_image_dataset if validate else self._temporal_image_dataset
        use_cache = bool(self.learned_temporal_cfg.cache_image_embeddings) and not validate
        if use_cache and self._temporal_image_cache_should_refresh(
                self.learned_temporal_cfg, self._history_image_embedding_cache,
                self._history_image_cache_epoch, epoch):
            self._history_image_embedding_cache = self._build_history_image_embedding_cache()
            self._history_image_cache_epoch = int(epoch)
        batch_size, history_length = history_indices.shape
        valid_indices = history_indices[history_mask]
        if use_cache and self._history_image_embedding_cache is not None:
            encoded = [cache[valid_indices.long()] for cache in self._history_image_embedding_cache]
        else:
            encoded = self._fetch_temporal_image_views(valid_indices, dataset)
        views = []
        for values in encoded:
            history = torch.zeros(batch_size, history_length, values.shape[-1], device=self.device, dtype=values.dtype)
            history[history_mask] = values
            views.append(history)
        return views
    
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        # Datasets may load auxiliary velocity solely to form temporal history.
        # Keep it out of the regular observation pipeline and its statistics.
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in self.obs_shapes}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, :Tp, :]
        if self.learned_temporal_enabled:
            for key in (
                "temporal_history_time",
                "temporal_history_signal",
                "temporal_history_mask",
                "temporal_obs_history_indices",
            ):
                input_batch[key] = batch[key]
            if self.learned_temporal_uses_image_embeddings:
                input_batch["temporal_history_image_indices"] = batch["temporal_history_image_indices"]
        
        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError("'actions' must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.")
            self.action_check_done = True
        
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        
    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch["actions"].shape[0]
        
        
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNet, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]
            
            # encode obs
            inputs = {
                "obs": batch["obs"],
                "goal": batch["goal_obs"]
            }
            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
            
            obs_features = TensorUtils.time_distributed(inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
            assert obs_features.ndim == 3  # [B, T, D]

            loss_history = torch.zeros((), device=self.device)
            if self.learned_temporal_enabled:
                history_time = batch["temporal_history_time"]
                history_signal = batch["temporal_history_signal"]
                history_mask = batch["temporal_history_mask"].bool()
                history_indices = batch["temporal_obs_history_indices"].long()
                if self.learned_temporal_uses_image_embeddings:
                    history_views = self._get_history_image_embeddings(
                        batch["temporal_history_image_indices"], history_mask, epoch, validate
                    )
                    image_embeddings = (
                        self.nets["temporal_image_fusion"](history_views)
                        if "temporal_image_fusion" in self.nets else torch.cat(history_views, dim=-1)
                    )
                    history_latents = self.nets["temporal_encoder"].encode_history(
                        history_time[..., :0], valid_mask=history_mask,
                        image_embeddings=image_embeddings,
                    )
                else:
                    history_latents = self.nets["temporal_encoder"].encode_history(
                        history_time, history_signal, history_mask
                    )
                temporal_latents = gather_history_latents(history_latents, history_indices)
                obs_features = torch.cat([obs_features, temporal_latents], dim=-1)
                if "history_decoder" in self.nets:
                    if self.learned_temporal_uses_image_embeddings and "temporal_image_fusion" in self.nets:
                        loss_history = multi_image_history_reconstruction_loss(
                            self.nets["history_decoder"], temporal_latents, history_time,
                            history_views, history_indices,
                            self.learned_temporal_cfg.history_reconstruction.num_history_queries,
                        )
                    else:
                        loss_history = history_reconstruction_loss(
                            self.nets["history_decoder"], temporal_latents, history_time,
                            image_embeddings if self.learned_temporal_uses_image_embeddings else history_signal,
                            history_indices,
                            self.learned_temporal_cfg.history_reconstruction.num_history_queries,
                        )

            obs_cond = obs_features.flatten(start_dim=1)
            
            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=self.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=self.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # predict the noise residual
            noise_pred = self.nets["policy"]["noise_pred_net"](
                noisy_actions, timesteps, global_cond=obs_cond)
            
            # L2 loss
            loss_diffusion = F.mse_loss(noise_pred, noise)
            loss = loss_diffusion + (
                self.learned_temporal_cfg.history_reconstruction.lambda_history * loss_history
                if self.learned_temporal_enabled and "history_decoder" in self.nets else 0.0
            )
            
            # logging
            losses = {
                "l2_loss": loss,
                "diffusion_loss": loss_diffusion,
                "history_loss": loss_history,
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )
                
                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)
                
                step_info = {
                    "policy_grad_norms": policy_grad_norms
                }
                info.update(step_info)

        return info
    
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DiffusionPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
    
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue
        if self.learned_temporal_enabled:
            self.temporal_latent_queue = deque(maxlen=To)
            self.temporal_z_prev = None
    
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        # obs_dict: key: [1,D]
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        
        if self.learned_temporal_enabled:
            self._advance_temporal_state(obs_dict)

        if len(self.action_queue) == 0:
            # no actions left, run inference
            # [1,T,Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)
            
            # put actions into the queue
            self.action_queue.extend(action_sequence[0])
        
        # has action, execute from left to right
        # [Da]
        action = self.action_queue.popleft()
        
        # [1,Da]
        action = action.unsqueeze(0)
        return action

    def _advance_temporal_state(self, obs_dict):
        """Advance exactly once per environment observation, before action reuse."""
        def latest(value):
            # FrameStack supplies (B, T, D); plain environments supply (B, D).
            return value[:, -1] if value.ndim >= 3 else value
        nets = self.ema.averaged_model if self.ema is not None else self.nets
        with torch.no_grad():
            if self.learned_temporal_uses_image_embeddings:
                missing = [key for key in self.temporal_image_keys if key not in obs_dict]
                if missing:
                    raise KeyError("LTE-IMG-NoT requires image observations {}".format(missing))
                images = [latest(obs_dict[key]).to(device=self.device, dtype=torch.float32) for key in self.temporal_image_keys]
                views = self._encode_temporal_image_views(images, nets)
                image_embedding = (
                    nets["temporal_image_fusion"](views)
                    if "temporal_image_fusion" in nets else torch.cat(views, dim=-1)
                )
                time_t = torch.empty(image_embedding.shape[0], 0, device=self.device, dtype=image_embedding.dtype)
                self.temporal_z_prev = nets["temporal_encoder"](
                    time_t, z_prev=self.temporal_z_prev, image_embedding_t=image_embedding
                ).detach()
            else:
                if "time_signal" not in obs_dict:
                    raise KeyError("learned_encoder requires the synthetic observation key 'time_signal'")
                signal_key = "agent_velocity" if self.learned_temporal_cfg.input_signal == "velocity" else "direct_idleness"
                if signal_key not in obs_dict:
                    raise KeyError("learned_encoder requires observation key '{}'".format(signal_key))
                time_t = latest(obs_dict["time_signal"]).to(device=self.device, dtype=torch.float32)
                signal_t = latest(obs_dict[signal_key]).to(device=self.device, dtype=torch.float32)
                if signal_key == "agent_velocity":
                    max_velocity = self.obs_config.temporal_encodings.max_train_set_agent_velocity
                    if max_velocity is None or max_velocity <= 0:
                        raise ValueError("Velocity learned_encoder requires max_train_set_agent_velocity")
                    signal_t = 2.0 * signal_t / float(max_velocity) - 1.0
                self.temporal_z_prev = nets["temporal_encoder"](time_t, signal_t, self.temporal_z_prev).detach()
        self.temporal_latent_queue.append(self.temporal_z_prev)
        
    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model
        
        # encode obs
        inputs = {
            "obs": obs_dict,
            "goal": goal_dict
        }
        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                # adding time dimension if not present -- this is required as
                # frame stacking is not invoked when sequence length is 1
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
        obs_features = TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        if self.learned_temporal_enabled:
            if not self.temporal_latent_queue:
                raise RuntimeError("Temporal state must be advanced before inference")
            latents = list(self.temporal_latent_queue)
            latents = [latents[0]] * (To - len(latents)) + latents
            temporal_latents = torch.stack(latents, dim=1)
            obs_features = torch.cat([obs_features, temporal_latents], dim=-1)
        B = obs_features.shape[0]

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=self.device)
        naction = noisy_action
        
        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        # process action using Ta
        start = To - 1
        end = start + Ta
        action = naction[:,start:end]
        return action

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "optimizers": { k : self.optimizers[k].state_dict() for k in self.optimizers },
            "lr_schedulers": { k : self.lr_schedulers[k].state_dict() if self.lr_schedulers[k] is not None else None for k in self.lr_schedulers },
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict, load_optimizers=False):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
            load_optimizers (bool): whether to load optimizers and lr_schedulers from the model_dict;
                used when resuming training from a checkpoint
        """
        self.nets.load_state_dict(model_dict["nets"])

        # for backwards compatibility
        if "optimizers" not in model_dict:
            model_dict["optimizers"] = {}
        if "lr_schedulers" not in model_dict:
            model_dict["lr_schedulers"] = {}

        if model_dict.get("ema", None) is not None:
            self.ema.averaged_model.load_state_dict(model_dict["ema"])

        if load_optimizers:
            for k in model_dict["optimizers"]:
                self.optimizers[k].load_state_dict(model_dict["optimizers"][k])
            for k in model_dict["lr_schedulers"]:
                if model_dict["lr_schedulers"][k] is not None:
                    self.lr_schedulers[k].load_state_dict(model_dict["lr_schedulers"][k])


def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("This function requires pytorch >= 1.9.0")

    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module
