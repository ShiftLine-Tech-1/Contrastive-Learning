
def freeze_layers_below(model, layers_to_freeze: int):
    for layer_number, layer in enumerate(model.encoder.layer):
        if layer_number >= layers_to_freeze:
            break

        for param in layer.parameters():
            param.requires_grad = False

def freeze_embeddings(model):
    for param in model.embeddings.parameters():
        param.requires_grad = False
