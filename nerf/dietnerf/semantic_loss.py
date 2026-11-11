
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def semantic_consistency_loss(
    target_embeddings: Tensor, rendered_embedding: Tensor, target_indices: Tensor | None = None,
) -> Tensor:
    if target_embeddings.ndim == 2:
        target_embeddings = target_embeddings.unsqueeze(0)
    if rendered_embedding.ndim == 1:
        rendered_embedding = rendered_embedding.unsqueeze(0)
    if target_embeddings.ndim != 3 or rendered_embedding.ndim != 2:
        raise ValueError("expected target embeddings [N,D] or [B,N,D] and rendered embeddings [D] or [B,D]")
    if target_embeddings.shape[0] == 1 and rendered_embedding.shape[0] > 1:
        target_embeddings = target_embeddings.expand(rendered_embedding.shape[0], -1, -1)
    if target_embeddings.shape[0] != rendered_embedding.shape[0]:
        raise ValueError("target and rendered embedding batches must match")
    if target_embeddings.shape[-1] != rendered_embedding.shape[-1] or target_embeddings.shape[1] < 1:
        raise ValueError("embedding dimensions must match and the target pool cannot be empty")

    target_embeddings = target_embeddings.to(device=rendered_embedding.device)
    if target_indices is None:
        selected = torch.randint(target_embeddings.shape[1], (rendered_embedding.shape[0],), device=rendered_embedding.device)
    else:
        selected = target_indices.to(device=rendered_embedding.device, dtype=torch.long).reshape(-1)
        if selected.numel() == 1 and rendered_embedding.shape[0] > 1:
            selected = selected.expand(rendered_embedding.shape[0])
        if selected.shape[0] != rendered_embedding.shape[0] or (selected < 0).any() or (selected >= target_embeddings.shape[1]).any():
            raise ValueError("target_indices must select one valid target for each rendered embedding")
    batch_indices = torch.arange(rendered_embedding.shape[0], device=rendered_embedding.device)
    targets = target_embeddings[batch_indices, selected].detach()
    targets = F.normalize(targets.float(), dim=-1)
    rendered = F.normalize(rendered_embedding.float(), dim=-1)
    return -(targets * rendered).sum(dim=-1).mean()
