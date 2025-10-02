"""
Drone Controller for VitFly-AirSim

This module provides high-level drone control functions including
velocity control, waypoint navigation, and safety management.

Author: Adapted from original VitFly project
"""

import airsim
import numpy as np
import time
from typing import Optional, Tuple, List, Dict, Any
import logging
import math

try:
    from .airsim_client import AirSimClient
    from .sensor_manager import SensorManager
except ImportError:
    from airsim_client import AirSimClient
    from sensor_manager import SensorManager


class DroneController:
    """High-level drone controller with safety features"""
    
    def __init__(self, client: AirSimClient, sensor_manager: SensorManager):
        """Initialize drone controller
        
        Args:
            client: AirSim client instance
            sensor_manager: Sensor manager instance
        """
        self.client = client
        self.sensor_manager = sensor_manager
        self.logger = logging.getLogger(__name__)
        
        # Control parameters
        self.max_velocity = 10.0  # m/s
        self.max_acceleration = 5.0  # m/s^2
        self.control_frequency = 30.0  # Hz
        self.control_period = 1.0 / self.control_frequency
        
        # Safety parameters
        self.min_altitude = -50.0  # Minimum altitude (AirSim coordinates)
        self.max_altitude = -2.0   # Maximum altitude (AirSim coordinates)
        self.safety_distance = 2.0  # Minimum distance from obstacles
        
        # Current state
        self.last_command_time = 0.0
        self.current_velocity = np.array([0.0, 0.0, 0.0])
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        
        # Safety flags
        self.safety_enabled = True
        self.emergency_stop_active = False
        
    def set_velocity_command(self, vx: float, vy: float, vz: float, 
                           duration: float = None) -> bool:
        """Send velocity command to drone with safety checks
        
        Args:
            vx, vy, vz: Velocity components in m/s (world frame)
            duration: Command duration (uses control_period if None)
            
        Returns:
            True if command sent successfully, False otherwise
        """
        if duration is None:
            duration = self.control_period
            
        # Safety checks
        if self.emergency_stop_active:
            self.logger.warning("Emergency stop active, ignoring velocity command")
            return False
        
        if self.safety_enabled:
            vx, vy, vz = self._apply_safety_limits(vx, vy, vz)
        
        # Apply velocity limits
        velocity_magnitude = math.sqrt(vx*vx + vy*vy + vz*vz)
        if velocity_magnitude > self.max_velocity:
            scale = self.max_velocity / velocity_magnitude
            vx *= scale
            vy *= scale
            vz *= scale
        
        # Send command
        success = self.client.move_by_velocity(vx, vy, vz, duration)
        
        if success:
            self.target_velocity = np.array([vx, vy, vz])
            self.last_command_time = time.time()
            
        return success
    
    def _apply_safety_limits(self, vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
        """Apply safety limits to velocity commands
        
        Args:
            vx, vy, vz: Desired velocity components
            
        Returns:
            Safe velocity components
        """
        # Get current position
        position = self.client.get_position()
        if position is None:
            self.logger.warning("Cannot get position for safety check")
            return 0.0, 0.0, 0.0
        
        x, y, z = position
        
        # Altitude safety checks (AirSim uses NED coordinates, so negative z is up)
        if z > self.min_altitude:  # Too low
            vz = max(0.0, vz)  # Only allow upward movement
            self.logger.warning(f"Altitude safety: too low ({z:.2f}), limiting downward velocity")
        
        if z < self.max_altitude:  # Too high
            vz = min(0.0, vz)  # Only allow downward movement
            self.logger.warning(f"Altitude safety: too high ({z:.2f}), limiting upward velocity")
        
        # Check for obstacles using depth camera
        depth_img = self.sensor_manager.get_depth_image()
        if depth_img is not None:
            # Check forward direction (center region of depth image)
            h, w = depth_img.shape
            center_region = depth_img[h//3:2*h//3, w//3:2*w//3]
            min_forward_distance = np.min(center_region) * 10.0  # Convert back to meters
            
            if min_forward_distance < self.safety_distance and vx > 0:
                vx = max(0.0, vx * 0.5)  # Reduce forward velocity
                self.logger.warning(f"Obstacle detected at {min_forward_distance:.2f}m, reducing forward velocity")
        
        return vx, vy, vz
    
    def hover(self) -> bool:
        """Make drone hover at current position
        
        Returns:
            True if hover command sent successfully, False otherwise
        """
        if not self.client.is_connected:
            return False
            
        try:
            self.client.client.hoverAsync(vehicle_name=self.client.drone_name)
            self.target_velocity = np.array([0.0, 0.0, 0.0])
            return True
        except Exception as e:
            self.logger.error(f"Error sending hover command: {e}")
            return False
    
    def emergency_stop(self) -> bool:
        """Execute emergency stop
        
        Returns:
            True if emergency stop executed successfully, False otherwise
        """
        self.emergency_stop_active = True
        success = self.client.emergency_stop()
        
        if success:
            self.target_velocity = np.array([0.0, 0.0, 0.0])
            self.logger.warning("Emergency stop executed")
        
        return success
    
    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop state
        
        Returns:
            True if reset successful, False otherwise
        """
        self.emergency_stop_active = False
        self.logger.info("Emergency stop reset")
        return True
    
    def move_to_position(self, x: float, y: float, z: float, 
                        velocity: float = 5.0, timeout: float = 30.0) -> bool:
        """Move to specific position
        
        Args:
            x, y, z: Target position coordinates
            velocity: Movement velocity in m/s
            timeout: Movement timeout in seconds
            
        Returns:
            True if movement successful, False otherwise
        """
        if not self.client.is_connected:
            return False
            
        try:
            future = self.client.client.moveToPositionAsync(
                x, y, z, velocity,
                timeout_sec=timeout,
                vehicle_name=self.client.drone_name
            )
            future.join()
            return True
            
        except Exception as e:
            self.logger.error(f"Error moving to position: {e}")
            return False
    
    def move_by_distance(self, dx: float, dy: float, dz: float,
                        velocity: float = 5.0) -> bool:
        """Move by relative distance
        
        Args:
            dx, dy, dz: Relative distance to move
            velocity: Movement velocity in m/s
            
        Returns:
            True if movement successful, False otherwise
        """
        # Get current position
        position = self.client.get_position()
        if position is None:
            return False
        
        # Calculate target position
        target_x = position[0] + dx
        target_y = position[1] + dy
        target_z = position[2] + dz
        
        return self.move_to_position(target_x, target_y, target_z, velocity)
    
    def follow_waypoints(self, waypoints: List[Tuple[float, float, float]],
                        velocity: float = 5.0, 
                        waypoint_timeout: float = 30.0) -> bool:
        """Follow a series of waypoints
        
        Args:
            waypoints: List of (x, y, z) waypoint coordinates
            velocity: Movement velocity in m/s
            waypoint_timeout: Timeout for each waypoint
            
        Returns:
            True if all waypoints reached successfully, False otherwise
        """
        for i, (x, y, z) in enumerate(waypoints):
            self.logger.info(f"Moving to waypoint {i+1}/{len(waypoints)}: ({x}, {y}, {z})")
            
            success = self.move_to_position(x, y, z, velocity, waypoint_timeout)
            if not success:
                self.logger.error(f"Failed to reach waypoint {i+1}")
                return False
                
            # Small pause between waypoints
            time.sleep(0.5)
            
        self.logger.info("All waypoints reached successfully")
        return True
    
    def get_distance_to_target(self, target_pos: Tuple[float, float, float]) -> Optional[float]:
        """Get distance to target position
        
        Args:
            target_pos: Target position (x, y, z)
            
        Returns:
            Distance in meters or None if error
        """
        current_pos = self.client.get_position()
        if current_pos is None:
            return None
        
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        dz = target_pos[2] - current_pos[2]
        
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def maintain_altitude(self, target_altitude: float, tolerance: float = 0.5) -> bool:
        """Maintain specific altitude
        
        Args:
            target_altitude: Target altitude (negative in AirSim coordinates)
            tolerance: Altitude tolerance in meters
            
        Returns:
            True if at target altitude within tolerance, False otherwise
        """
        position = self.client.get_position()
        if position is None:
            return False
        
        current_altitude = position[2]
        altitude_error = target_altitude - current_altitude
        
        if abs(altitude_error) <= tolerance:
            return True
        
        # Simple proportional control for altitude
        vz = np.clip(altitude_error * 2.0, -2.0, 2.0)  # Proportional gain of 2.0
        
        return self.set_velocity_command(0.0, 0.0, vz)
    
    def get_controller_status(self) -> Dict[str, Any]:
        """Get controller status information
        
        Returns:
            Dictionary with controller status
        """
        position = self.client.get_position()
        velocity = self.client.get_velocity()
        orientation = self.client.get_orientation()
        
        return {
            'position': position,
            'velocity': velocity,
            'orientation': orientation,
            'target_velocity': self.target_velocity.tolist(),
            'emergency_stop_active': self.emergency_stop_active,
            'safety_enabled': self.safety_enabled,
            'last_command_time': self.last_command_time,
            'control_frequency': self.control_frequency,
            'collision_detected': self.client.check_collision()
        }
    
    def enable_safety(self, enable: bool = True):
        """Enable or disable safety features
        
        Args:
            enable: Whether to enable safety features
        """
        self.safety_enabled = enable
        self.logger.info(f"Safety features {'enabled' if enable else 'disabled'}")
    
    def set_control_frequency(self, frequency: float):
        """Set control loop frequency
        
        Args:
            frequency: Control frequency in Hz
        """
        self.control_frequency = max(1.0, min(100.0, frequency))  # Clamp to reasonable range
        self.control_period = 1.0 / self.control_frequency
        self.logger.info(f"Control frequency set to {self.control_frequency} Hz")