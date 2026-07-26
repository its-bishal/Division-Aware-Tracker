
from .unet import DoubleConv
import torch as nn
import torch


MODEL = {
    'unet': DoubleConv,
}

def init_weights(model):
    """Initialize the weights of the model using Kaiming He initialization."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    return model


def get_model(model_name, in_channels, out_channels):
    if model_name not in MODEL:
        raise ValueError(f'Model {model_name} not found. Available models: {list(MODEL.keys())}')
    
    model_class = MODEL[model_name]
    model = model_class(in_channels, out_channels)
    model = init_weights(model)
    
    return model
