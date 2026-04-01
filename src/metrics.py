import torch
from torch import Tensor


def lalign(features: Tensor, labels: Tensor, alpha: int = 2) -> float:

    diff = features.unsqueeze(1) - features.unsqueeze(0)
    distances = diff.norm(dim=2).pow(alpha)

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float()

    mask.fill_diagonal_(0)

    if mask.sum() == 0:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    loss = (distances * mask).sum() / mask.sum()

    return loss.item()


def lunif(x: Tensor, t: float = 2) -> float:
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log().item()
