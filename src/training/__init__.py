"""
Training Module for VitFly-AirSim

This module contains all training-related code adapted for Windows and AirSim,
including data loading, training loops, and model evaluation.

Author: Adapted from original VitFly project
"""

try:
    from .data_loader import VitFlyDataLoader, create_data_loaders
    from .trainer import VitFlyTrainer
    from .evaluator import ModelEvaluator
except ImportError:
    from data_loader import VitFlyDataLoader, create_data_loaders
    from trainer import VitFlyTrainer
    from evaluator import ModelEvaluator

__all__ = [
    'VitFlyDataLoader',
    'create_data_loaders', 
    'VitFlyTrainer',
    'ModelEvaluator'
]