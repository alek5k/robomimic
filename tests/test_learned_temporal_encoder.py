"""Unit tests for the causal learned temporal encoder."""
import unittest

import torch

from robomimic.models.learned_temporal import (
    HistoryDecoder,
    ImageEmbeddingFusion,
    TemporalEncoder,
    gather_history_latents,
    history_reconstruction_loss,
)


class TestLearnedTemporalEncoder(unittest.TestCase):
    def test_future_history_does_not_change_past_latent(self):
        torch.manual_seed(1)
        encoder = TemporalEncoder(latent_dim=4, hidden_dim=8)
        times = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
        signals = torch.tensor([[[0.0], [1.0], [0.0], [1.0], [0.0]]])
        original = encoder.encode_history(times, signals)
        changed_future = signals.clone()
        changed_future[:, 3:] = 99.0
        changed = encoder.encode_history(times, changed_future)
        torch.testing.assert_close(original[:, :3], changed[:, :3])

    def test_history_loss_backpropagates(self):
        encoder = TemporalEncoder(latent_dim=4, hidden_dim=8)
        decoder = HistoryDecoder(latent_dim=4, signal_dim=1, hidden_dim=8)
        times = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
        signals = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
        indices = torch.tensor([[1, 3]])
        latents = gather_history_latents(encoder.encode_history(times, signals), indices)
        loss = history_reconstruction_loss(decoder, latents, times, signals, indices, 2)
        loss.backward()
        self.assertIsNotNone(encoder.mlp[0].weight.grad)

    def test_image_without_time_is_causal(self):
        torch.manual_seed(2)
        encoder = TemporalEncoder(
            time_dim=0, signal_dim=0, image_embedding_dim=3,
            latent_dim=4, hidden_dim=8,
        )
        images = torch.randn(1, 5, 3)
        no_time = torch.empty(1, 5, 0)
        original = encoder.encode_history(no_time, image_embeddings=images)
        changed_images = images.clone()
        changed_images[:, 3:] += 10.0
        changed = encoder.encode_history(no_time, image_embeddings=changed_images)
        torch.testing.assert_close(original[:, :3], changed[:, :3])

        fusion = ImageEmbeddingFusion((2, 3), output_dim=4, hidden_dim=8)
        self.assertEqual(fusion([torch.randn(2, 2), torch.randn(2, 3)]).shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
