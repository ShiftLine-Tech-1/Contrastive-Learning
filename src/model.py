from torch import Tensor, nn
from transformers import XLMRobertaModel

from pooler import Pooler


class XLMRoBERTaSupCon(nn.Module):
    def __init__(self, proj_dim: int = 128) -> None:
        super().__init__()
        self.encoder: XLMRobertaModel = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.pooler = Pooler()
        self.projector = nn.Sequential(
            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.LeakyReLU(),
            nn.Linear(768, proj_dim),
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mean_embedding = self.pooler(attention_mask = attention_mask, outputs = out)
        proj = self.projector(mean_embedding)

        return {
            "embeddings": mean_embedding,
            "projection": proj,
        }
