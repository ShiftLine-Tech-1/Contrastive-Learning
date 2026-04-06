import torch

ROBERTA_MAX_SEQUENCE_LENGTH = 512


def collate_function(data: list[tuple[str, int]], tokenizer, *, label_encoding_map: dict | None = None):
    texts, labels = zip(*data, strict=True)

    if label_encoding_map is not None:
        labels = [label_encoding_map[label] for label in labels]

    encoding = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=ROBERTA_MAX_SEQUENCE_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return (
        {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        },
        torch.tensor(labels, dtype=torch.long),
    )
