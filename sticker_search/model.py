"""Small trainable residual projections over frozen multilingual CLIP features."""
import torch
from torch import nn
from torch.nn import functional as F


class Adapter(nn.Module):
    def __init__(self, dim=512, rank=64):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return F.normalize(x + self.up(F.gelu(self.down(x))), dim=-1)


class RetrievalModel(nn.Module):
    def __init__(self, dim=512, rank=64):
        super().__init__()
        self.query_adapter = Adapter(dim, rank)
        self.image_adapter = Adapter(dim, rank)
        self.fusion_logits = nn.Parameter(torch.log(torch.tensor([0.65, 0.25, 0.10])))

    def encode_query(self, query):
        return self.query_adapter(query)

    def encode_items(self, image, caption, ocr):
        weights = self.fusion_logits.softmax(0)
        # Missing branches get zero weight, not a penalty against items without text.
        present = torch.stack([image.norm(dim=-1)>0, caption.norm(dim=-1)>0, ocr.norm(dim=-1)>0], -1)
        w = present * weights
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
        return self.image_adapter(image) * w[:, :1] + caption * w[:, 1:2] + ocr * w[:, 2:3]

    def forward(self, query, image, caption, ocr):
        return self.encode_query(query) @ self.encode_items(image, caption, ocr).T


def multi_positive_loss(scores, positives, valid, temperature=0.07):
    """At least one known positive per query; related unlabelled siblings are masked."""
    if not positives.any(dim=1).all():
        raise ValueError("Each query needs at least one positive")
    logits = scores / temperature
    denominator = torch.logsumexp(logits.masked_fill(~valid, -torch.inf), dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)
    return (denominator - numerator).mean()


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = RetrievalModel(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint
