"""
AirSim Interface Module for VitFly-AirSim

This module provides the interface between VitFly models and AirSim simulation,
replacing the original ROS-based communication with direct AirSim Python API calls.

Author: Adapted from original VitFly project for Windows/AirSim compatibility
"""

from .airsim_client import AirSimClient
from .drone_controller import DroneController
from .sensor_manager import SensorManager
from .obstacle_detector import ObstacleDetector
from .data_collector import DataCollector

__all__ = [
    'AirSimClient',
    'DroneController', 
    'SensorManager',
    'ObstacleDetector',
    'DataCollector'
]