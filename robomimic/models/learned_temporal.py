"""Causal learned temporal features used by Diffusion Policy.

The encoder consumes only a normalized timestamp, a temporal signal, and its
previous latent.  It is intentionally separate from observation encoders so
the state can be advanced once per environment observation at rollout time.
"""
import torch
import torch.nn as nn


class ImageEmbeddingFusion(nn.Module):
    """Fuse detached per-camera visual features into the LTE input."""

    def __init__(self, embedding_dims, output_dim, hidden_dim):
        super().__init__()
        self.embedding_dims = tuple(int(dim) for dim in embedding_dims)
        if not self.embedding_dims or any(dim <= 0 for dim in self.embedding_dims):
            raise ValueError("image embedding dimensions must be positive")
        self.output_dim = int(output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sum(self.embedding_dims), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self.output_dim),
        )

    def forward(self, embeddings):
        if len(embeddings) != len(self.embedding_dims):
            raise ValueError("wrong number of image embedding views")
        leading_shape = embeddings[0].shape[:-1]
        for embedding, expected_dim in zip(embeddings, self.embedding_dims):
            if embedding.shape[:-1] != leading_shape or embedding.shape[-1] != expected_dim:
                raise ValueError("image embeddings have incompatible shapes")
        return self.mlp(torch.cat(list(embeddings), dim=-1))


class TemporalEncoder(nn.Module):
    """Small recurrent residual MLP which produces a causal temporal latent."""

    def __init__(self, time_dim=1, signal_dim=1, image_embedding_dim=0,
                 latent_dim=32, hidden_dim=128, recurrent=True):
        super().__init__()
        self.time_dim = int(time_dim)
        self.signal_dim = int(signal_dim)
        self.image_embedding_dim = int(image_embedding_dim)
        self.latent_dim = int(latent_dim)
        self.recurrent = bool(recurrent)
        input_dim = self.time_dim + self.signal_dim + self.image_embedding_dim + (self.latent_dim if self.recurrent else 0)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self.latent_dim),
        )

    def initial_latent(self, batch_size, *, device=None, dtype=None):
        return torch.zeros(batch_size, self.latent_dim, device=device, dtype=dtype)

    def forward(self, time_t, signal_t=None, z_prev=None, image_embedding_t=None):
        if time_t.ndim != 2 or time_t.shape[-1] != self.time_dim:
            raise ValueError("time_t must have shape (B, time_dim)")
        inputs = [time_t]
        if self.signal_dim:
            if signal_t is None or signal_t.shape != (time_t.shape[0], self.signal_dim):
                raise ValueError("signal_t must have shape (B, signal_dim)")
            inputs.append(signal_t)
        if self.image_embedding_dim:
            if image_embedding_t is None or image_embedding_t.shape != (time_t.shape[0], self.image_embedding_dim):
                raise ValueError("image_embedding_t must have shape (B, image_embedding_dim)")
            inputs.append(image_embedding_t)
        if self.recurrent:
            if z_prev is None:
                z_prev = self.initial_latent(time_t.shape[0], device=time_t.device, dtype=time_t.dtype)
            if z_prev.shape != (time_t.shape[0], self.latent_dim):
                raise ValueError("z_prev has the wrong shape")
            inputs.append(z_prev)
            return torch.tanh(z_prev + self.mlp(torch.cat(inputs, dim=-1)))
        return torch.tanh(self.mlp(torch.cat(inputs, dim=-1)))

    def encode_history(self, times, signals=None, valid_mask=None, image_embeddings=None):
        """Encode padded episode prefixes and return one latent per history step."""
        if times.ndim != 3 or times.shape[-1] != self.time_dim:
            raise ValueError("times must have shape (B, T, time_dim)")
        if self.signal_dim and (signals is None or signals.shape != (*times.shape[:2], self.signal_dim)):
            raise ValueError("signals must have shape (B, T, signal_dim)")
        if self.image_embedding_dim and (image_embeddings is None or image_embeddings.shape != (*times.shape[:2], self.image_embedding_dim)):
            raise ValueError("image_embeddings must have shape (B, T, image_embedding_dim)")
        batch_size, history_len = times.shape[:2]
        if valid_mask is None:
            valid_mask = torch.ones(batch_size, history_len, dtype=torch.bool, device=times.device)
        z_prev = self.initial_latent(batch_size, device=times.device, dtype=times.dtype)
        latents = []
        for index in range(history_len):
            z_candidate = self(
                times[:, index],
                None if signals is None else signals[:, index],
                z_prev,
                None if image_embeddings is None else image_embeddings[:, index],
            )
            z_prev = torch.where(valid_mask[:, index].unsqueeze(-1), z_candidate, z_prev)
            latents.append(z_prev)
        return torch.stack(latents, dim=1)


class HistoryDecoder(nn.Module):
    """Training-only decoder for a temporal signal at a historical lag."""

    def __init__(self, latent_dim, signal_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(int(latent_dim) + 1, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(signal_dim)),
        )

    def forward(self, latent, lag):
        return self.mlp(torch.cat([latent, lag], dim=-1))


class MultiImageHistoryDecoder(nn.Module):
    """Reconstruct detached past visual features, with one head per camera."""

    def __init__(self, latent_dim, image_embedding_dims, hidden_dim=64):
        super().__init__()
        self.image_embedding_dims = tuple(int(dim) for dim in image_embedding_dims)
        self.trunk = nn.Sequential(nn.Linear(int(latent_dim) + 1, int(hidden_dim)), nn.SiLU())
        self.image_heads = nn.ModuleList(
            nn.Linear(int(hidden_dim), dim) for dim in self.image_embedding_dims
        )

    def forward(self, latent, lag):
        hidden = self.trunk(torch.cat([latent, lag], dim=-1))
        return [head(hidden) for head in self.image_heads]


def gather_history_latents(latents, indices):
    return torch.gather(
        latents, 1, indices.long().unsqueeze(-1).expand(-1, -1, latents.shape[-1])
    )


def history_reconstruction_loss(decoder, latents, history_times, history_signals, current_indices, num_queries):
    """Reconstruct uniformly sampled, strictly causal historical signal values."""
    if num_queries <= 0:
        raise ValueError("num_queries must be positive")
    current_times = torch.gather(history_times[..., 0], 1, current_indices.long())
    elapsed = (current_times - history_times[:, :1, 0]).clamp_min(0.0)
    bin_offsets = torch.arange(num_queries, device=elapsed.device, dtype=elapsed.dtype) / num_queries
    lags = (bin_offsets + torch.rand(*elapsed.shape, num_queries, device=elapsed.device, dtype=elapsed.dtype) / num_queries) * elapsed.unsqueeze(-1)
    query_times = current_times.unsqueeze(-1) - lags
    time_values = history_times[..., 0].contiguous()
    upper = torch.searchsorted(time_values, query_times.reshape(query_times.shape[0], -1).contiguous())
    upper = torch.minimum(upper, current_indices.long().unsqueeze(-1).expand_as(query_times).reshape(query_times.shape[0], -1))
    upper = upper.clamp(min=0, max=history_times.shape[1] - 1)
    lower = (upper - 1).clamp(min=0)
    def gather(values, index):
        return torch.gather(values, 1, index.unsqueeze(-1).expand(-1, -1, values.shape[-1]))
    low_time, high_time = gather(history_times, lower), gather(history_times, upper)
    low_signal, high_signal = gather(history_signals, lower), gather(history_signals, upper)
    weight = (query_times.reshape(query_times.shape[0], -1).unsqueeze(-1) - low_time) / (high_time - low_time).clamp_min(1e-8)
    targets = (low_signal + weight * (high_signal - low_signal)).reshape(*lags.shape, history_signals.shape[-1])
    predictions = decoder(latents.unsqueeze(-2).expand(-1, -1, num_queries, -1), lags.unsqueeze(-1))
    return nn.functional.mse_loss(predictions, targets)


def multi_image_history_reconstruction_loss(
        decoder, latents, history_times, history_image_embeddings, current_indices, num_queries):
    """Causally reconstruct historical per-camera embeddings from LTE states."""
    if num_queries <= 0:
        raise ValueError("num_queries must be positive")
    current_times = torch.gather(history_times[..., 0], 1, current_indices.long())
    elapsed = (current_times - history_times[:, :1, 0]).clamp_min(0.0)
    offsets = torch.arange(num_queries, device=elapsed.device, dtype=elapsed.dtype) / num_queries
    lags = (offsets + torch.rand(*elapsed.shape, num_queries, device=elapsed.device, dtype=elapsed.dtype) / num_queries) * elapsed.unsqueeze(-1)
    query_times = current_times.unsqueeze(-1) - lags
    time_values = history_times[..., 0].contiguous()
    upper = torch.searchsorted(time_values, query_times.reshape(query_times.shape[0], -1).contiguous())
    upper = torch.minimum(upper, current_indices.long().unsqueeze(-1).expand_as(query_times).reshape(query_times.shape[0], -1))
    upper = upper.clamp(min=0, max=history_times.shape[1] - 1)
    lower = (upper - 1).clamp(min=0)

    def gather(values, index):
        return torch.gather(values, 1, index.unsqueeze(-1).expand(-1, -1, values.shape[-1]))

    low_time, high_time = gather(history_times, lower), gather(history_times, upper)
    weight = (query_times.reshape(query_times.shape[0], -1).unsqueeze(-1) - low_time) / (high_time - low_time).clamp_min(1e-8)
    query_latents = latents.unsqueeze(-2).expand(-1, -1, num_queries, -1)
    predictions = decoder(query_latents, lags.unsqueeze(-1))
    losses = []
    for embeddings, prediction in zip(history_image_embeddings, predictions):
        targets = (gather(embeddings, lower) + weight * (gather(embeddings, upper) - gather(embeddings, lower)))
        targets = targets.reshape(*lags.shape, embeddings.shape[-1])
        losses.append(nn.functional.mse_loss(prediction, targets))
    return torch.stack(losses).mean()
