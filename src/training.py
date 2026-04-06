from typing import TypedDict

import torch
from torch.nn import functional


class TrainStepOutput(TypedDict):
    loss: float
    grad_norm: float
    norm_embeddings: torch.Tensor


def train_step(model, optimizer, criterion, batch_X, batch_y) -> TrainStepOutput:
    model.train()
    device = next(model.parameters()).device

    optimizer.zero_grad()

    input_ids = batch_X["input_ids"].to(device) # dimension (BATCH_SIZE, 512)
    attention_mask = batch_X["attention_mask"].to(device) # dimension (BATCH_SIZE, 512)
    labels = batch_y.to(device) # dimension = (BATCH_SIZE)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        mean_embeddings = outputs["embeddings"]
        projections = outputs["projection"]

        loss = criterion(projections, labels)

    loss.backward()
    optimizer.step()

    norm_embeddings = functional.normalize(mean_embeddings)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))

    return TrainStepOutput(loss=loss.item(), grad_norm=grad_norm.item(), norm_embeddings=norm_embeddings)
