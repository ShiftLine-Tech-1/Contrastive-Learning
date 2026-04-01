import torch.nn.functional as F
from torch import nn


class Pooler(nn.Module):
    def __init__(self, pool_type="mean_pooling"):
        super().__init__()
        self.type = pool_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state

        if self.type=="mean_pooling":
            pooled_result = ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.type=="cls":
            pooled_result = last_hidden[:, 0]
        else:
            pass

        return pooled_result
