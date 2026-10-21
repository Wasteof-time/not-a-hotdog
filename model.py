import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_model(pretrained=True):
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 1),
    )
    return model


def set_trainable(model, backbone, last_n_blocks=0):
    for p in model.features.parameters():
        p.requires_grad = False
    if backbone:
        if last_n_blocks == 0:
            for p in model.features.parameters():
                p.requires_grad = True
        else:
            for p in model.features[-last_n_blocks:].parameters():
                p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
