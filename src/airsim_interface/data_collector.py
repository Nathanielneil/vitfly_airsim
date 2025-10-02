"""
Data Collector for VitFly-AirSim

This module handles data collection for training dataset generation,
replacing the original ROS-based data logging with AirSim-compatible collection.

Author: Adapted from original VitFly project
"""

import os
import cv2
import numpy as np
import pandas as pd
import time
import json
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path

try:
    from .airsim_client import AirSimClient
    from .sensor_manager import SensorManager
    from .obstacle_detector import ObstacleDetector
except ImportError:
    from airsim_client import AirSimClient
    from sensor_manager import SensorManager
    from obstacle_detector import ObstacleDetector


class DataCollector:
    """Collects training data from AirSim simulation"""
    
    def __init__(self, client: AirSimClient, sensor_manager: SensorManager, 
                 obstacle_detector: ObstacleDetector, output_dir: str = "data"):
        """Initialize data collector
        
        Args:
            client: AirSim client instance
            sensor_manager: Sensor manager instance
            obstacle_detector: Obstacle detector instance
            output_dir: Output directory for collected data
        """
        self.client = client
        self.sensor_manager = sensor_manager
        self.obstacle_detector = obstacle_detector
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data collection state
        self.is_collecting = False
        self.current_trajectory = None
        self.trajectory_data = []
        self.trajectory_counter = 0
        
        # Collection parameters
        self.collection_frequency = 10.0  # Hz
        self.collection_period = 1.0 / self.collection_frequency
        self.last_collection_time = 0.0
        
        # Data storage
        self.metadata_columns = [
            'timestamp', 'timestamp_ns', 'desired_velocity',
            'orientation_w', 'orientation_x', 'orientation_y', 'orientation_z',
            'position_x', 'position_y', 'position_z',
            'velocity_x', 'velocity_y', 'velocity_z',
            'cmd_velocity_x', 'cmd_velocity_y', 'cmd_velocity_z',
            'collective_thrust', 'body_rate_x', 'body_rate_y', 'body_rate_z',
            'collision'
        ]
        
    def start_trajectory_collection(self, trajectory_name: Optional[str] = None) -> bool:
        """Start collecting data for a new trajectory
        
        Args:
            trajectory_name: Optional custom trajectory name
            
        Returns:
            True if collection started successfully, False otherwise
        """
        if self.is_collecting:
            self.logger.warning("Already collecting data")
            return False
        
        try:
            # Generate trajectory name if not provided
            if trajectory_name is None:
                trajectory_name = f"trajectory_{self.trajectory_counter:06d}"
            
            self.current_trajectory = trajectory_name
            self.trajectory_data = []
            self.is_collecting = True
            self.last_collection_time = 0.0
            
            # Create trajectory directory
            traj_dir = self.output_dir / trajectory_name
            traj_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Started trajectory collection: {trajectory_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting trajectory collection: {e}")
            return False
    
    def collect_data_point(self, desired_velocity: float, 
                          velocity_command: Optional[tuple] = None) -> bool:
        """Collect a single data point
        
        Args:
            desired_velocity: Desired velocity in m/s
            velocity_command: Actual velocity command sent to drone (vx, vy, vz)
            
        Returns:
            True if data collected successfully, False otherwise
        """
        if not self.is_collecting:
            return False
        
        current_time = time.time()
        
        # Check if enough time has passed since last collection
        if current_time - self.last_collection_time < self.collection_period:
            return False
        
        try:
            # Get depth image
            depth_image = self.sensor_manager.get_depth_image()
            if depth_image is None:
                self.logger.warning("Failed to get depth image")
                return False
            
            # Get drone state
            position = self.client.get_position()
            velocity = self.client.get_velocity()
            orientation = self.client.get_orientation()
            collision_info = self.client.get_collision_info()
            
            if position is None or velocity is None or orientation is None:
                self.logger.warning("Failed to get drone state")
                return False
            
            # Create timestamp
            timestamp = current_time
            timestamp_ns = int(timestamp * 1e9)
            
            # Get IMU data for additional info
            imu_data = self.sensor_manager.get_imu_data()
            angular_velocity = (0.0, 0.0, 0.0)
            if imu_data is not None:
                angular_velocity = imu_data['angular_velocity']
            
            # Create data entry
            data_entry = {
                'timestamp': timestamp,
                'timestamp_ns': timestamp_ns,
                'desired_velocity': desired_velocity,
                'orientation_w': orientation[0],
                'orientation_x': orientation[1],
                'orientation_y': orientation[2],
                'orientation_z': orientation[3],
                'position_x': position[0],
                'position_y': position[1],
                'position_z': position[2],
                'velocity_x': velocity[0],
                'velocity_y': velocity[1],
                'velocity_z': velocity[2],
                'collective_thrust': 0.0,  # Not available in AirSim
                'body_rate_x': angular_velocity[0],
                'body_rate_y': angular_velocity[1],
                'body_rate_z': angular_velocity[2],
                'collision': collision_info.get('has_collided', False)
            }
            
            # Add velocity command if provided
            if velocity_command is not None:
                data_entry.update({
                    'cmd_velocity_x': velocity_command[0],
                    'cmd_velocity_y': velocity_command[1],
                    'cmd_velocity_z': velocity_command[2]
                })
            else:
                data_entry.update({
                    'cmd_velocity_x': 0.0,
                    'cmd_velocity_y': 0.0,
                    'cmd_velocity_z': 0.0
                })
            
            # Save depth image
            image_filename = f"{timestamp_ns}.png"
            traj_dir = self.output_dir / self.current_trajectory
            image_path = traj_dir / image_filename
            
            # Convert depth to uint16 for PNG saving (preserves precision)
            depth_uint16 = (depth_image * 65535).astype(np.uint16)
            cv2.imwrite(str(image_path), depth_uint16)
            
            # Add to trajectory data
            self.trajectory_data.append(data_entry)
            self.last_collection_time = current_time
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error collecting data point: {e}")
            return False
    
    def stop_trajectory_collection(self) -> bool:
        """Stop collecting data and save trajectory
        
        Returns:
            True if trajectory saved successfully, False otherwise
        """
        if not self.is_collecting:
            self.logger.warning("Not currently collecting data")
            return False
        
        try:
            if not self.trajectory_data:
                self.logger.warning("No data collected for trajectory")
                return False
            
            # Save metadata CSV
            traj_dir = self.output_dir / self.current_trajectory
            metadata_path = traj_dir / "data.csv"
            
            df = pd.DataFrame(self.trajectory_data)
            df.to_csv(metadata_path, index=False)
            
            # Save trajectory info
            trajectory_info = {
                'trajectory_name': self.current_trajectory,
                'num_frames': len(self.trajectory_data),
                'duration': self.trajectory_data[-1]['timestamp'] - self.trajectory_data[0]['timestamp'],
                'start_time': self.trajectory_data[0]['timestamp'],
                'end_time': self.trajectory_data[-1]['timestamp'],
                'collection_frequency': self.collection_frequency,
                'collision_occurred': any(entry['collision'] for entry in self.trajectory_data)
            }
            
            info_path = traj_dir / "trajectory_info.json"
            with open(info_path, 'w') as f:
                json.dump(trajectory_info, f, indent=2)
            
            self.logger.info(f"Saved trajectory: {self.current_trajectory} ({len(self.trajectory_data)} frames)")
            
            # Reset state
            self.is_collecting = False
            self.current_trajectory = None
            self.trajectory_data = []
            self.trajectory_counter += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving trajectory: {e}")
            return False
    
    def collect_expert_trajectory(self, desired_velocity: float, 
                                 max_duration: float = 30.0,
                                 collision_stop: bool = True) -> bool:
        """Collect a trajectory using expert obstacle avoidance
        
        Args:
            desired_velocity: Desired forward velocity
            max_duration: Maximum trajectory duration in seconds
            collision_stop: Whether to stop on collision
            
        Returns:
            True if trajectory collected successfully, False otherwise
        """
        # Start collection
        if not self.start_trajectory_collection():
            return False
        
        try:
            start_time = time.time()
            control_period = 0.1  # 10 Hz control
            
            while time.time() - start_time < max_duration:
                # Get current position
                position = self.client.get_position()
                if position is None:
                    break
                
                # Detect obstacles
                depth_image = self.sensor_manager.get_depth_image()
                if depth_image is not None:
                    obstacles = self.obstacle_detector.detect_obstacles_from_depth(depth_image)
                
                # Compute expert velocity command
                velocity_command = self.obstacle_detector.compute_expert_velocity_command(
                    position, desired_velocity
                )
                
                # Send velocity command
                self.client.client.moveByVelocityAsync(
                    velocity_command[0], velocity_command[1], velocity_command[2],
                    control_period, vehicle_name=self.client.drone_name
                )
                
                # Collect data point
                self.collect_data_point(desired_velocity, velocity_command)
                
                # Check for collision
                if collision_stop and self.client.check_collision():
                    self.logger.warning("Collision detected, stopping trajectory")
                    break
                
                time.sleep(control_period)
            
            # Stop collection
            return self.stop_trajectory_collection()
            
        except Exception as e:
            self.logger.error(f"Error collecting expert trajectory: {e}")
            return False
    
    def validate_trajectory_data(self, trajectory_name: str) -> Dict[str, Any]:
        """Validate collected trajectory data
        
        Args:
            trajectory_name: Name of trajectory to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            traj_dir = self.output_dir / trajectory_name
            
            if not traj_dir.exists():
                return {'valid': False, 'error': 'Trajectory directory not found'}
            
            # Check metadata file
            metadata_path = traj_dir / "data.csv"
            if not metadata_path.exists():
                return {'valid': False, 'error': 'Metadata file not found'}
            
            # Load metadata
            df = pd.read_csv(metadata_path)
            
            # Check for required columns
            missing_columns = set(self.metadata_columns) - set(df.columns)
            if missing_columns:
                return {'valid': False, 'error': f'Missing columns: {missing_columns}'}
            
            # Count images
            image_files = list(traj_dir.glob("*.png"))
            metadata_rows = len(df)
            
            # Validation results
            validation_results = {
                'valid': True,
                'trajectory_name': trajectory_name,
                'num_metadata_entries': metadata_rows,
                'num_image_files': len(image_files),
                'data_consistent': metadata_rows == len(image_files),
                'has_collision': df['collision'].any() if 'collision' in df.columns else False,
                'duration': df['timestamp'].max() - df['timestamp'].min() if len(df) > 1 else 0,
                'average_velocity': df['desired_velocity'].mean() if len(df) > 0 else 0
            }
            
            if not validation_results['data_consistent']:
                validation_results['warning'] = 'Mismatch between metadata entries and image files'
            
            return validation_results
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {e}'}
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get statistics about data collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            trajectories = [d for d in self.output_dir.iterdir() if d.is_dir()]
            
            total_trajectories = len(trajectories)
            total_frames = 0
            total_duration = 0.0
            collision_trajectories = 0
            
            for traj_dir in trajectories:
                validation = self.validate_trajectory_data(traj_dir.name)
                if validation['valid']:
                    total_frames += validation['num_metadata_entries']
                    total_duration += validation['duration']
                    if validation['has_collision']:
                        collision_trajectories += 1
            
            return {
                'total_trajectories': total_trajectories,
                'total_frames': total_frames,
                'total_duration': total_duration,
                'collision_trajectories': collision_trajectories,
                'valid_trajectories': total_trajectories,  # Simplified
                'output_directory': str(self.output_dir),
                'collection_frequency': self.collection_frequency,
                'currently_collecting': self.is_collecting,
                'current_trajectory': self.current_trajectory
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collection statistics: {e}")
            return {'error': str(e)}
    
    def export_for_training(self, export_dir: str = "training_data") -> bool:
        """Export collected data in format suitable for training
        
        Args:
            export_dir: Directory to export training data
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            export_path = Path(export_dir)
            export_path.mkdir(parents=True, exist_ok=True)
            
            trajectories = [d for d in self.output_dir.iterdir() if d.is_dir()]
            
            for traj_dir in trajectories:
                validation = self.validate_trajectory_data(traj_dir.name)
                
                if not validation['valid']:
                    self.logger.warning(f"Skipping invalid trajectory: {traj_dir.name}")
                    continue
                
                # Copy trajectory to export directory
                export_traj_dir = export_path / traj_dir.name
                export_traj_dir.mkdir(exist_ok=True)
                
                # Copy and convert images to training format
                for image_file in traj_dir.glob("*.png"):
                    # Load and convert depth image
                    depth_img = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)
                    
                    if depth_img is not None:
                        # Convert back to normalized float and save as PNG
                        depth_normalized = (depth_img.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
                        cv2.imwrite(str(export_traj_dir / image_file.name), depth_normalized)
                
                # Copy metadata
                metadata_src = traj_dir / "data.csv"
                metadata_dst = export_traj_dir / "data.csv"
                
                if metadata_src.exists():
                    # Load, process, and save metadata
                    df = pd.read_csv(metadata_src)
                    
                    # Ensure compatibility with original format
                    df.to_csv(metadata_dst, index=False)
            
            self.logger.info(f"Exported training data to: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting training data: {e}")
            return False