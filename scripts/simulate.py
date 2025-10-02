#!/usr/bin/env python3
"""
Simulation Script for VitFly-AirSim

This script runs VitFly models in AirSim simulation for testing and evaluation.

Author: Adapted from original VitFly project
"""

import os
import sys
import argparse
import logging
import yaml
import time
import signal
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

# Add individual module paths to bypass __init__.py import issues
sys.path.insert(0, str(src_path / 'airsim_interface'))
sys.path.insert(0, str(src_path / 'inference'))
sys.path.insert(0, str(src_path / 'utils'))

# Direct imports bypassing package structure
from airsim_client import AirSimClient
from drone_controller import DroneController
from sensor_manager import SensorManager
from obstacle_detector import ObstacleDetector
from data_collector import DataCollector
from model_inference import ModelInference
from visualization import SimulationVisualizer


class SimulationRunner:
    """Runs VitFly simulation with AirSim"""
    
    def __init__(self, config: dict):
        """Initialize simulation runner"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.running = False

        # Initialize components
        self.client = None
        self.controller = None
        self.sensor_manager = None
        self.obstacle_detector = None
        self.model_inference = None
        self.data_collector = None
        self.visualizer = None

        # Track collision state
        self.last_collision_timestamp = 0
        
    def setup(self):
        """Setup simulation components"""
        try:
            # Connect to AirSim
            self.logger.info("Connecting to AirSim...")
            self.client = AirSimClient(
                host=self.config.get('airsim_host', '127.0.0.1'),
                port=self.config.get('airsim_port', 41451)
            )
            
            if not self.client.connect():
                raise RuntimeError("Failed to connect to AirSim")
            
            # Initialize drone
            if not self.client.initialize_drone():
                raise RuntimeError("Failed to initialize drone")
            
            # Create components
            self.sensor_manager = SensorManager(self.client.client, self.client.drone_name)
            self.controller = DroneController(self.client, self.sensor_manager)
            self.obstacle_detector = ObstacleDetector(self.sensor_manager, self.config)
            
            # Load model if specified
            if self.config.get('use_model', False):
                self.model_inference = ModelInference(
                    model_path=self.config['model_path'],
                    model_type=self.config['model_type'],
                    device=self.config.get('device', 'auto')
                )
            
            # Setup data collection if requested
            if self.config.get('collect_data', False):
                self.data_collector = DataCollector(
                    self.client, self.sensor_manager, self.obstacle_detector,
                    output_dir=self.config.get('data_output_dir', 'collected_data')
                )
            
            # Setup visualization if requested
            if self.config.get('enable_visualization', True):
                self.visualizer = SimulationVisualizer()
            
            self.logger.info("Simulation setup completed")
            
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            raise
    
    def takeoff(self):
        """Takeoff sequence"""
        self.logger.info("Starting takeoff sequence...")

        if not self.client.takeoff():
            raise RuntimeError("Takeoff failed")

        # Wait for stabilization
        time.sleep(2)

        # Move to desired altitude if specified
        target_altitude = self.config.get('flight_altitude', -5.0)  # AirSim coordinates
        if abs(self.client.get_position()[2] - target_altitude) > 1.0:
            self.controller.move_to_position(0, 0, target_altitude, velocity=2.0)

        # Check and log collision state after takeoff
        # AirSim doesn't have a direct reset collision API, so we just track the timestamp
        collision_info = self.client.get_collision_info()
        if collision_info.get('has_collided', False):
            self.logger.warning(f"Collision state detected after takeoff (likely ground contact during takeoff): "
                              f"Object: {collision_info.get('object_name', 'unknown')}")
            # Set the initial timestamp to ignore this collision
            self.last_collision_timestamp = collision_info.get('time_stamp', 0)
            self.logger.info(f"Ignoring initial collision (timestamp: {self.last_collision_timestamp}) - will monitor new collisions only")

        self.logger.info("Takeoff completed")
    
    def run_expert_policy(self):
        """Run expert obstacle avoidance policy"""
        self.logger.info("Running expert policy simulation...")
        
        desired_velocity = self.config.get('desired_velocity', 5.0)
        max_duration = self.config.get('max_duration', 30.0)
        control_frequency = self.config.get('control_frequency', 10.0)
        
        start_time = time.time()
        control_period = 1.0 / control_frequency
        
        while self.running and (time.time() - start_time) < max_duration:
            loop_start = time.time()

            # Get current position
            position = self.client.get_position()
            if position is None:
                break

            # Detect obstacles
            obstacles = []
            depth_image = self.sensor_manager.get_depth_image()
            if depth_image is not None:
                obstacles = self.obstacle_detector.detect_obstacles_from_depth(depth_image)
            
            # Compute expert velocity command
            velocity_command = self.obstacle_detector.compute_expert_velocity_command(
                position, desired_velocity
            )
            
            # Send velocity command
            self.controller.set_velocity_command(*velocity_command, duration=control_period)
            
            # Data collection
            if self.data_collector and self.data_collector.is_collecting:
                self.data_collector.collect_data_point(desired_velocity, velocity_command)
            
            # Visualization
            if self.visualizer:
                self.visualizer.update(depth_image, velocity_command, obstacles)

            # Check for new collision (not just persistent collision state)
            collision_info = self.client.get_collision_info()
            if collision_info.get('has_collided', False):
                collision_time = collision_info.get('time_stamp', 0)
                # Only report if this is a new collision (different timestamp)
                if collision_time > self.last_collision_timestamp:
                    self.logger.warning(f"New collision detected! Object: {collision_info.get('object_name', 'unknown')}, "
                                      f"Penetration: {collision_info.get('penetration_depth', 0):.3f}m")
                    self.last_collision_timestamp = collision_time
                    if self.config.get('stop_on_collision', True):
                        break
            
            # Maintain control frequency
            elapsed = time.time() - loop_start
            if elapsed < control_period:
                time.sleep(control_period - elapsed)
        
        self.logger.info("Expert policy simulation completed")
    
    def run_model_inference(self):
        """Run model-based control"""
        if not self.model_inference:
            raise RuntimeError("Model inference not initialized")
        
        self.logger.info("Running model inference simulation...")
        
        desired_velocity = self.config.get('desired_velocity', 5.0)
        max_duration = self.config.get('max_duration', 30.0)
        control_frequency = self.config.get('control_frequency', 30.0)
        
        start_time = time.time()
        control_period = 1.0 / control_frequency
        
        while self.running and (time.time() - start_time) < max_duration:
            loop_start = time.time()
            
            # Get sensor data
            depth_image, success = self.sensor_manager.get_processed_depth_for_model()
            if not success:
                continue
            
            # Get drone state
            position = self.client.get_position()
            orientation = self.client.get_orientation()
            
            if position is None or orientation is None:
                continue
            
            # Run model inference
            velocity_command = self.model_inference.predict_velocity(
                depth_image, desired_velocity, orientation
            )
            
            # Send velocity command
            self.controller.set_velocity_command(*velocity_command, duration=control_period)
            
            # Data collection
            if self.data_collector and self.data_collector.is_collecting:
                self.data_collector.collect_data_point(desired_velocity, velocity_command)
            
            # Visualization
            if self.visualizer:
                # Convert depth for visualization
                vis_depth = depth_image.squeeze().cpu().numpy() if hasattr(depth_image, 'cpu') else depth_image.squeeze()
                self.visualizer.update_model_prediction(vis_depth, velocity_command)

            # Check for new collision (not just persistent collision state)
            collision_info = self.client.get_collision_info()
            if collision_info.get('has_collided', False):
                collision_time = collision_info.get('time_stamp', 0)
                # Only report if this is a new collision (different timestamp)
                if collision_time > self.last_collision_timestamp:
                    self.logger.warning(f"New collision detected! Object: {collision_info.get('object_name', 'unknown')}, "
                                      f"Penetration: {collision_info.get('penetration_depth', 0):.3f}m")
                    self.last_collision_timestamp = collision_time
                    if self.config.get('stop_on_collision', True):
                        break
            
            # Maintain control frequency
            elapsed = time.time() - loop_start
            if elapsed < control_period:
                time.sleep(control_period - elapsed)
        
        self.logger.info("Model inference simulation completed")
    
    def run_data_collection(self):
        """Run data collection mode"""
        if not self.data_collector:
            raise RuntimeError("Data collector not initialized")
        
        self.logger.info("Running data collection simulation...")
        
        num_trajectories = self.config.get('num_trajectories', 10)
        desired_velocity = self.config.get('desired_velocity', 5.0)
        trajectory_duration = self.config.get('trajectory_duration', 30.0)
        
        for i in range(num_trajectories):
            if not self.running:
                break
            
            self.logger.info(f"Collecting trajectory {i+1}/{num_trajectories}")
            
            # Reset simulation
            self.client.reset()
            time.sleep(2)
            self.takeoff()
            
            # Collect trajectory using expert policy
            success = self.data_collector.collect_expert_trajectory(
                desired_velocity=desired_velocity,
                max_duration=trajectory_duration,
                collision_stop=True
            )
            
            if success:
                self.logger.info(f"Trajectory {i+1} collected successfully")
            else:
                self.logger.warning(f"Failed to collect trajectory {i+1}")
        
        self.logger.info("Data collection completed")
    
    def run(self):
        """Main simulation loop"""
        self.running = True
        
        try:
            # Setup simulation
            self.setup()
            
            # Takeoff
            self.takeoff()
            
            # Start data collection if requested
            if self.data_collector and self.config.get('collect_data', False):
                self.data_collector.start_trajectory_collection()
            
            # Run simulation based on mode
            mode = self.config.get('mode', 'expert')
            
            if mode == 'expert':
                self.run_expert_policy()
            elif mode == 'model':
                self.run_model_inference()
            elif mode == 'data_collection':
                self.run_data_collection()
            else:
                raise ValueError(f"Unknown simulation mode: {mode}")
            
        except KeyboardInterrupt:
            self.logger.info("Simulation interrupted by user")
        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
            raise
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup simulation"""
        self.running = False
        
        if self.controller:
            self.controller.emergency_stop()
        
        if self.data_collector and self.data_collector.is_collecting:
            self.data_collector.stop_trajectory_collection()
        
        if self.client:
            self.client.land()
            self.client.disconnect()
        
        if self.visualizer:
            self.visualizer.close()
        
        self.logger.info("Simulation cleanup completed")


def load_config(config_path: str) -> dict:
    """Load configuration from file"""
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                import json
                config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run VitFly simulation')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to simulation configuration file')
    parser.add_argument('--mode', type=str,
                       choices=['expert', 'model', 'data_collection'],
                       help='Override simulation mode from config')
    parser.add_argument('--model-path', type=str,
                       help='Override model path from config')
    parser.add_argument('--model-type', type=str,
                       choices=['ViT', 'ViTLSTM', 'ConvNet', 'LSTMNet', 'UNet'],
                       help='Override model type from config')
    parser.add_argument('--desired-velocity', type=float,
                       help='Override desired velocity from config')
    parser.add_argument('--max-duration', type=float,
                       help='Override max duration from config')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.mode:
        config['mode'] = args.mode
    if args.model_path:
        config['model_path'] = args.model_path
        config['use_model'] = True
    if args.model_type:
        config['model_type'] = args.model_type
    if args.desired_velocity:
        config['desired_velocity'] = args.desired_velocity
    if args.max_duration:
        config['max_duration'] = args.max_duration
    
    # Create and run simulation
    runner = SimulationRunner(config)
    
    # Setup signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        runner.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        runner.run()
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()