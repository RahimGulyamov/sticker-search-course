"""SigLIP encoders and a distributed, weighted multi-positive sigmoid loss."""
import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.distributed.nn.functional import all_gather


class FullSiglip(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def forward(self, pixel_values, input_ids, attention_mask=None):
        image = self.backbone.get_image_features(pixel_values=pixel_values)
        text = self.backbone.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return (F.normalize(image.float(), dim=-1), F.normalize(text.float(), dim=-1),
                self.backbone.logit_scale.float().clamp(max=math.log(1000)).exp(),
                self.backbone.logit_bias.float())


class SiglipPairLoss(nn.Module):
    """One local text row against every GPU's image candidates, with autograd gather.

    DDP averages parameter gradients. world_size/global_weight normalizes the
    local loss so the result matches a single-process global weighted loss.
    Repeated image columns have total weight one; unknown same-group matches
    are masked rather than treated as positive ground truth or false negatives.
    """
    def forward(self, images, texts, scale, bias, positive, valid, weights, column_weights):
        world = dist.get_world_size() if dist.is_initialized() else 1
        global_images = torch.cat(all_gather(images), dim=0) if world > 1 else images
        logits = texts.float() @ global_images.float().T * scale + bias
        positive, valid = positive.bool(), valid.bool()
        if positive.shape != logits.shape or valid.shape != logits.shape:
            raise ValueError("Pair-mask/logit shape mismatch")
        if not bool(positive.any(dim=1).all()) or bool((positive & ~valid).any()):
            raise ValueError("Each row needs at least one valid positive")
        terms = F.softplus(torch.where(positive, -logits, logits))
        per_row = (terms * valid * column_weights[None, :]).sum(dim=1)
        denominator = weights.detach().sum().clone()
        if world > 1:
            dist.all_reduce(denominator)
        return (per_row * weights).sum() * world / denominator.clamp_min(1e-8)


def gather_indices(indices):
    if not dist.is_initialized():
        return indices.tolist()
    gathered = [torch.empty_like(indices) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, indices)
    return torch.cat(gathered).tolist()


def pair_masks(local_indices, global_indices, pairs, positives, item_groups, device):
    from collections import Counter
    candidates = [pairs[j]["item_id"] for j in global_indices]
    counts = Counter(candidates)
    positive, valid = [], []
    for i in local_indices:
        p = positives[i]
        groups = {item_groups[pid] for pid in p}
        positive.append([pid in p for pid in candidates])
        valid.append([pid in p or item_groups[pid] not in groups for pid in candidates])
    return (torch.tensor(positive, dtype=torch.bool, device=device),
            torch.tensor(valid, dtype=torch.bool, device=device),
            torch.tensor([1 / counts[pid] for pid in candidates], device=device))
