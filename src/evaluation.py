import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from pooler import Pooler


def evaluate(encoder, data_loader: DataLoader, pooler: Pooler):
    device = next(encoder.parameters()).device

    total_norm_embeddings = []
    total_labels = []

    encoder.eval()
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            input_ids = batch_x["input_ids"].to(device)  # dimension (BATCH_SIZE, 512)
            attention_mask = batch_x["attention_mask"].to(device)  # dimension (BATCH_SIZE, 512)

            out = encoder(input_ids=input_ids, attention_mask=attention_mask)
            mean_embeddings = pooler(attention_mask=attention_mask, outputs=out)

            norm_embeddings = functional.normalize(mean_embeddings)

            total_norm_embeddings.append(norm_embeddings.cpu())
            total_labels.append(batch_y.cpu())

    embeddings = torch.concat(total_norm_embeddings)
    labels = torch.concat(total_labels)

    return {"embeddings": embeddings, "labels": labels}
