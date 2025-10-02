"""
Inference Module for VitFly-AirSim

This module provides model inference capabilities for real-time
drone control and simulation.

Author: Adapted from original VitFly project
"""

try:
    from .model_inference import ModelInference
except ImportError:
    from model_inference import ModelInference

__all__ = ['ModelInference']