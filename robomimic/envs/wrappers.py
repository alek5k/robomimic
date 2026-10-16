"""
A collection of useful environment wrappers.
"""
from copy import deepcopy
import textwrap
import numpy as np
from collections import deque

import robomimic.envs.env_base as EB

from robomimic.utils.temporal_encodings import (
    calculate_growth_rate_parameter,
    progress_positional_encoding_transformer,
    progress_saturalising_encoding,
    IdlenessEncoder,
)

class EnvWrapper(object):
    """
    Base class for all environment wrappers in robomimic.
    """
    def __init__(self, env):
        """
        Args:
            env (EnvBase instance): The environment to wrap.
        """
        assert isinstance(env, EB.EnvBase) or isinstance(env, EnvWrapper)
        self.env = env

    @classmethod
    def class_name(cls):
        return cls.__name__

    def _warn_double_wrap(self):
        """
        Utility function that checks if we're accidentally trying to double wrap an env
        Raises:
            Exception: [Double wrapping env]
        """
        env = self.env
        while True:
            if isinstance(env, EnvWrapper):
                if env.class_name() == self.class_name():
                    raise Exception(
                        "Attempted to double wrap with Wrapper: {}".format(
                            self.__class__.__name__
                        )
                    )
                env = env.env
            else:
                break

    @property
    def unwrapped(self):
        """
        Grabs unwrapped environment

        Returns:
            env (EnvBase instance): Unwrapped environment
        """
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        else:
            return self.env

    def _to_string(self):
        """
        Subclasses should override this method to print out info about the 
        wrapper (such as arguments passed to it).
        """
        return ''

    def __repr__(self):
        """Pretty print environment."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        if self._to_string() != '':
            msg += textwrap.indent("\n" + self._to_string(), indent)
        msg += textwrap.indent("\nenv={}".format(self.env), indent)
        msg = header + '(' + msg + '\n)'
        return msg

    # this method is a fallback option on any methods the original env might support
    def __getattr__(self, attr):
        # using getattr ensures that both __getattribute__ and __getattr__ (fallback) get called
        # (see https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute)
        orig_attr = getattr(self.env, attr)
        if callable(orig_attr):

            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if id(result) == id(self.env):
                    return self
                return result

            return hooked
        else:
            return orig_attr


class FrameStackWrapper(EnvWrapper):
    """
    Wrapper for frame stacking observations during rollouts. The agent
    receives a sequence of past observations instead of a single observation
    when it calls @env.reset, @env.reset_to, or @env.step in the rollout loop.
    """
    def __init__(self, env, num_frames):
        """
        Args:
            env (EnvBase instance): The environment to wrap.
            num_frames (int): number of past observations (including current observation)
                to stack together. Must be greater than 1 (otherwise this wrapper would
                be a no-op).
        """
        assert num_frames > 1, "error: FrameStackWrapper must have num_frames > 1 but got num_frames of {}".format(num_frames)

        super(FrameStackWrapper, self).__init__(env=env)
        self.num_frames = num_frames

        # keep track of last @num_frames observations for each obs key
        self.obs_history = None

    def _get_initial_obs_history(self, init_obs):
        """
        Helper method to get observation history from the initial observation, by
        repeating it.

        Returns:
            obs_history (dict): a deque for each observation key, with an extra
                leading dimension of 1 for each key (for easy concatenation later)
        """
        obs_history = {}
        for k in init_obs:
            obs_history[k] = deque(
                [init_obs[k][None] for _ in range(self.num_frames)], 
                maxlen=self.num_frames,
            )
        return obs_history

    def _get_stacked_obs_from_history(self):
        """
        Helper method to convert internal variable @self.obs_history to a 
        stacked observation where each key is a numpy array with leading dimension
        @self.num_frames.
        """
        # concatenate all frames per key so we return a numpy array per key
        return { k : np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history }

    def cache_obs_history(self):
        self.obs_history_cache = deepcopy(self.obs_history)

    def uncache_obs_history(self):
        self.obs_history = self.obs_history_cache
        self.obs_history_cache = None

    def reset(self):
        """
        Modify to return frame stacked observation which is @self.num_frames copies of 
        the initial observation.

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
        """
        obs = self.env.reset()
        self.timestep = 0  # always zero regardless of timestep type
        self.update_obs(obs, reset=True)
        self.obs_history = self._get_initial_obs_history(init_obs=obs)
        return self._get_stacked_obs_from_history()

    def reset_to(self, state):
        """
        Modify to return frame stacked observation which is @self.num_frames copies of 
        the initial observation.

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
        """
        obs = self.env.reset_to(state)
        self.timestep = 0  # always zero regardless of timestep type
        self.update_obs(obs, reset=True)
        self.obs_history = self._get_initial_obs_history(init_obs=obs)
        return self._get_stacked_obs_from_history()

    def step(self, action):
        """
        Modify to update the internal frame history and return frame stacked observation,
        which will have leading dimension @self.num_frames for each key.

        Args:
            action (np.array): action to take

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, r, done, info = self.env.step(action)
        self.update_obs(obs, action=action, reset=False)
        # update frame history
        for k in obs:
            # make sure to have leading dim of 1 for easy concatenation
            self.obs_history[k].append(obs[k][None])
        obs_ret = self._get_stacked_obs_from_history()
        return obs_ret, r, done, info

    def update_obs(self, obs, action=None, reset=False):
        obs["timesteps"] = np.array([self.timestep])
        
        if reset:
            obs["actions"] = np.zeros(self.env.action_dimension)
        else:
            self.timestep += 1
            obs["actions"] = action[: self.env.action_dimension]

    def _to_string(self):
        """Info to pretty print."""
        return "num_frames={}".format(self.num_frames)


class TemporalEncodingWrapper(EnvWrapper):
    def __init__(self, env, temporal_encoding_config=None):
        super(TemporalEncodingWrapper, self).__init__(env=env)

        self.temporal_encoding_config = temporal_encoding_config or {}
        learned_cfg = self.temporal_encoding_config.get("learned_encoder", {})
        self._learned_temporal_enabled = bool(learned_cfg.get("enabled", False))
        self._learned_temporal_signal = learned_cfg.get("input_signal", "velocity")
        self._learned_temporal_uses_image = (
            self._learned_temporal_signal == "image_embedding"
            or bool(learned_cfg.get("use_image_embedding", False))
        )
        needs_velocity = any((
            self.temporal_encoding_config.get("use_idleness_encoding", False),
            self.temporal_encoding_config.get("use_direct_idleness", False),
            self._learned_temporal_enabled and not self._learned_temporal_uses_image,
        ))
        self._idleness_vmax = self.temporal_encoding_config.get("max_train_set_agent_velocity")
        if needs_velocity and self._idleness_vmax is None:
            raise ValueError("Temporal idleness features require max_train_set_agent_velocity")
        self._idleness_encoder = None
        if self.temporal_encoding_config.get("use_idleness_encoding", False):
            alpha = self.temporal_encoding_config.get("idleness_alpha", None)
            if alpha is None:
                alpha = calculate_growth_rate_parameter(max_steps=self.temporal_encoding_config["max_train_set_steps"])
            self._idleness_encoder = IdlenessEncoder(
                vmax=self._idleness_vmax,
                rest_thresh=self.temporal_encoding_config["idleness_rest_thresh"], alpha=alpha,
            )

        self._saturating_progress_omega = None
        if self.temporal_encoding_config.get("use_saturating_progress_encoding", False):
            omega = self.temporal_encoding_config.get("saturating_omega", None)
            if omega is None:
                max_steps = self.temporal_encoding_config.get("max_train_set_steps")
                if max_steps is None:
                    raise ValueError("Saturating progress requires max_train_set_steps")
                omega = calculate_growth_rate_parameter(max_steps=max_steps)
            self._saturating_progress_omega = omega

        self._timestep = 0

    def _update_temporal_encodings(self, obs):
        if "agent_velocity" in obs:
            # Native temporal environments expose the same scalar used when
            # their demonstrations were collected.
            velocity = float(np.asarray(obs["agent_velocity"]).reshape(-1)[0])
        else:
            mag_joint = np.linalg.norm(obs["robot0_joint_vel"])
            mag_gripper = np.linalg.norm(obs["robot0_gripper_qvel"])
            velocity = mag_joint + 0.1 * mag_gripper


        obs["agent_velocity"] = np.array([velocity], dtype=np.float32)

        if self.temporal_encoding_config.get("use_direct_time", False) or (
            self._learned_temporal_enabled and not self._learned_temporal_uses_image
        ):
            max_steps = self.temporal_encoding_config.get("max_train_set_steps")
            if max_steps is None or max_steps <= 0:
                raise ValueError("Direct time requires a positive max_train_set_steps")
            obs["time_signal"] = np.array(
                [2.0 * self._timestep / float(max_steps) - 1.0], dtype=np.float32
            )

        if (
            self.temporal_encoding_config.get("use_direct_idleness", False)
            or (self._learned_temporal_enabled and self._learned_temporal_signal == "idle_indicator")
        ):
            threshold = self.temporal_encoding_config.get("idleness_rest_thresh", 0.03)
            obs["direct_idleness"] = np.array(
                [float(abs(velocity) / self._idleness_vmax < threshold)], dtype=np.float32
            )

        if self.temporal_encoding_config.get("use_sinusoidal_progress_encoding", False):
            pe = progress_positional_encoding_transformer(t=self._timestep, d_model=self.temporal_encoding_config["sinusoidal_dim"])
            obs["sinusoidal_progress_encoding"] = pe.reshape(-1).astype(np.float32) # flatten to 1D array

        if self.temporal_encoding_config.get("use_saturating_progress_encoding", False):
            sp = progress_saturalising_encoding(t=self._timestep, omega=self._saturating_progress_omega)
            obs["saturating_progress_encoding"] = np.array([sp], dtype=np.float32)

        if self.temporal_encoding_config.get("use_idleness_encoding", False):
            obs["idleness"] = np.array([self._idleness_encoder.step(float(obs["agent_velocity"]))], dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        if self._idleness_encoder is not None:
            self._idleness_encoder.reset()
        self._timestep = 0
        self._update_temporal_encodings(obs)
        return obs

    def reset_to(self, state):
        obs = self.env.reset_to(state)
        if self._idleness_encoder is not None:
            self._idleness_encoder.reset()
        self._timestep = 0
        self._update_temporal_encodings(obs)
        return obs

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        self._timestep += 1
        self._update_temporal_encodings(obs)
        return obs, r, done, info
