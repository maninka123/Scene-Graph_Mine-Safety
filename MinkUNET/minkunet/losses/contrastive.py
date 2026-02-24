from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if z_a.shape != z_b.shape:
        raise ValueError(f"Embedding shape mismatch: {z_a.shape} vs {z_b.shape}")

    batch_size = z_a.shape[0]
    if batch_size < 2:
        raise ValueError("Contrastive batch size must be >= 2")

    # Compute logits in fp32 to avoid fp16 overflow when masking diagonals.
    z = torch.cat([z_a, z_b], dim=0).float()
    if not torch.isfinite(z).all():
        raise ValueError("Non-finite embeddings detected in contrastive head output.")
    similarity = torch.matmul(z, z.T) / temperature

    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, -1e4)

    positive_indices = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ]
    )

    return F.cross_entropy(similarity, positive_indices)
