"""
VitFly-AirSim Models Module

This module contains all the deep learning models used in VitFly-AirSim:
- Vision Transformer (ViT)
- ViT+LSTM (Best performing model)
- Convolutional Networks
- LSTM Networks
- UNet architectures

Author: Adapted from original VitFly project (GRASP Lab, UPenn)
License: MIT
"""

from .vit_models import ViT, ViTLSTM
from .conv_models import ConvNet, LSTMNet, UNetConvLSTMNet
from .vit_submodules import (
    OverlapPatchMerging,
    EfficientSelfAttention,
    MixFFN,
    MixTransformerEncoderLayer
)

__all__ = [
    'ViT',
    'ViTLSTM', 
    'ConvNet',
    'LSTMNet',
    'UNetConvLSTMNet',
    'OverlapPatchMerging',
    'EfficientSelfAttention',
    'MixFFN',
    'MixTransformerEncoderLayer'
]