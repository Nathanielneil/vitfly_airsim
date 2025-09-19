"""
AirSim Client for VitFly-AirSim

This module provides the main interface to AirSim, handling connection,
initialization, and basic drone operations.

Author: Adapted from original VitFly project
"""

import airsim
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging


class AirSimClient:
    """Main AirSim client for drone control and simulation management"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 41451):
        """Initialize AirSim client
        
        Args:
            host: AirSim server host address
            port: AirSim server port
        """
        self.host = host
        self.port = port
        self.client: Optional[airsim.MultirotorClient] = None
        self.drone_name = "Drone1"  # Default drone name
        self.is_connected = False
        self.home_position = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def connect(self, timeout: float = 30.0) -> bool:
        """Connect to AirSim server
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.logger.info(f"Connecting to AirSim at {self.host}:{self.port}")
            self.client = airsim.MultirotorClient(ip=self.host, port=self.port)
            
            # Wait for connection with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    self.client.confirmConnection()
                    self.is_connected = True
                    self.logger.info("Successfully connected to AirSim")
                    return True
                except:
                    time.sleep(1)
                    
            self.logger.error(f"Failed to connect to AirSim within {timeout} seconds")
            return False
            
        except Exception as e:
            self.logger.error(f"Error connecting to AirSim: {e}")
            return False
    
    def initialize_drone(self) -> bool:
        """Initialize drone for flight
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not self.is_connected:
            self.logger.error("Not connected to AirSim")
            return False
            
        try:
            # Enable API control
            self.client.enableApiControl(True, self.drone_name)
            
            # Arm the drone
            self.client.armDisarm(True, self.drone_name)
            
            # Get and store home position
            self.home_position = self.client.getMultirotorState(self.drone_name).kinematics_estimated.position
            
            self.logger.info("Drone initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing drone: {e}")
            return False
    
    def takeoff(self, timeout: float = 20.0) -> bool:
        """Take off to default altitude
        
        Args:
            timeout: Takeoff timeout in seconds
            
        Returns:
            True if takeoff successful, False otherwise
        """
        if not self.is_connected:
            self.logger.error("Not connected to AirSim")
            return False
            
        try:
            self.logger.info("Taking off...")
            future = self.client.takeoffAsync(timeout_sec=timeout, vehicle_name=self.drone_name)
            future.join()
            
            # Wait a bit for stabilization
            time.sleep(2)
            
            self.logger.info("Takeoff completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during takeoff: {e}")
            return False
    
    def land(self, timeout: float = 20.0) -> bool:
        """Land the drone
        
        Args:
            timeout: Landing timeout in seconds
            
        Returns:
            True if landing successful, False otherwise
        """
        if not self.is_connected:
            self.logger.error("Not connected to AirSim")
            return False
            
        try:
            self.logger.info("Landing...")
            future = self.client.landAsync(timeout_sec=timeout, vehicle_name=self.drone_name)
            future.join()
            
            self.logger.info("Landing completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during landing: {e}")
            return False
    
    def reset(self) -> bool:
        """Reset simulation to initial state
        
        Returns:
            True if reset successful, False otherwise
        """
        try:
            self.logger.info("Resetting simulation...")
            self.client.reset()
            time.sleep(2)  # Wait for reset to complete
            
            # Re-initialize after reset
            return self.initialize_drone()
            
        except Exception as e:
            self.logger.error(f"Error during reset: {e}")
            return False
    
    def get_drone_state(self) -> Optional[airsim.MultirotorState]:
        """Get current drone state
        
        Returns:
            MultirotorState object or None if error
        """
        if not self.is_connected:
            return None
            
        try:
            return self.client.getMultirotorState(self.drone_name)
        except Exception as e:
            self.logger.error(f"Error getting drone state: {e}")
            return None
    
    def get_position(self) -> Optional[Tuple[float, float, float]]:
        """Get current drone position
        
        Returns:
            Tuple of (x, y, z) coordinates or None if error
        """
        state = self.get_drone_state()
        if state is None:
            return None
            
        pos = state.kinematics_estimated.position
        return (pos.x_val, pos.y_val, pos.z_val)
    
    def get_orientation(self) -> Optional[Tuple[float, float, float, float]]:
        """Get current drone orientation as quaternion
        
        Returns:
            Tuple of (w, x, y, z) quaternion or None if error
        """
        state = self.get_drone_state()
        if state is None:
            return None
            
        quat = state.kinematics_estimated.orientation
        return (quat.w_val, quat.x_val, quat.y_val, quat.z_val)
    
    def get_velocity(self) -> Optional[Tuple[float, float, float]]:
        """Get current drone velocity
        
        Returns:
            Tuple of (vx, vy, vz) velocities or None if error
        """
        state = self.get_drone_state()
        if state is None:
            return None
            
        vel = state.kinematics_estimated.linear_velocity
        return (vel.x_val, vel.y_val, vel.z_val)
    
    def move_by_velocity(self, vx: float, vy: float, vz: float, 
                        duration: float = 1.0) -> bool:
        """Move drone by velocity commands
        
        Args:
            vx, vy, vz: Velocity components in m/s
            duration: Command duration in seconds
            
        Returns:
            True if command sent successfully, False otherwise
        """
        if not self.is_connected:
            return False
            
        try:
            self.client.moveByVelocityAsync(
                vx, vy, vz, duration, 
                vehicle_name=self.drone_name
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending velocity command: {e}")
            return False
    
    def emergency_stop(self) -> bool:
        """Emergency stop - immediately halt all movement
        
        Returns:
            True if stop successful, False otherwise
        """
        if not self.is_connected:
            return False
            
        try:
            # Stop all movement
            self.client.moveByVelocityAsync(0, 0, 0, 0.1, vehicle_name=self.drone_name)
            
            # Hover in place
            self.client.hoverAsync(vehicle_name=self.drone_name)
            
            self.logger.warning("Emergency stop executed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during emergency stop: {e}")
            return False
    
    def check_collision(self) -> bool:
        """Check if drone has collided
        
        Returns:
            True if collision detected, False otherwise
        """
        if not self.is_connected:
            return False
            
        try:
            collision_info = self.client.simGetCollisionInfo(self.drone_name)
            return collision_info.has_collided
            
        except Exception as e:
            self.logger.error(f"Error checking collision: {e}")
            return False
    
    def get_collision_info(self) -> Dict[str, Any]:
        """Get detailed collision information
        
        Returns:
            Dictionary with collision details
        """
        if not self.is_connected:
            return {}
            
        try:
            collision_info = self.client.simGetCollisionInfo(self.drone_name)
            return {
                'has_collided': collision_info.has_collided,
                'object_name': collision_info.object_name,
                'object_id': collision_info.object_id,
                'position': (
                    collision_info.position.x_val,
                    collision_info.position.y_val, 
                    collision_info.position.z_val
                ),
                'normal': (
                    collision_info.normal.x_val,
                    collision_info.normal.y_val,
                    collision_info.normal.z_val
                ),
                'impact_point': (
                    collision_info.impact_point.x_val,
                    collision_info.impact_point.y_val,
                    collision_info.impact_point.z_val
                ),
                'penetration_depth': collision_info.penetration_depth,
                'time_stamp': collision_info.time_stamp
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collision info: {e}")
            return {}
    
    def disconnect(self):
        """Disconnect from AirSim server"""
        if self.is_connected:
            try:
                # Disarm and disable API control
                self.client.armDisarm(False, self.drone_name)
                self.client.enableApiControl(False, self.drone_name)
                self.is_connected = False
                self.logger.info("Disconnected from AirSim")
            except Exception as e:
                self.logger.error(f"Error during disconnect: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()