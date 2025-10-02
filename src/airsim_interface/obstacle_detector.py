"""
Obstacle Detector for VitFly-AirSim

This module provides obstacle detection capabilities using depth images
and other sensor data, implementing the expert policy from the original VitFly project.

Author: Adapted from original VitFly project
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
import logging
import time

try:
    from .sensor_manager import SensorManager
except ImportError:
    from sensor_manager import SensorManager


class Obstacle:
    """Represents a detected obstacle"""
    
    def __init__(self, position: Tuple[float, float, float], radius: float, 
                 confidence: float = 1.0):
        """Initialize obstacle
        
        Args:
            position: Obstacle center position (x, y, z)
            radius: Obstacle radius in meters
            confidence: Detection confidence [0, 1]
        """
        self.position = position
        self.radius = radius
        self.confidence = confidence
        self.timestamp = time.time()


class ObstacleDetector:
    """Detects obstacles from sensor data and implements expert avoidance policy"""

    def __init__(self, sensor_manager: SensorManager, config: Optional[Dict] = None):
        """Initialize obstacle detector

        Args:
            sensor_manager: Sensor manager instance
            config: Optional configuration dictionary
        """
        self.sensor_manager = sensor_manager
        self.logger = logging.getLogger(__name__)

        # Detection parameters
        self.min_obstacle_distance = 0.5  # Minimum distance to consider as obstacle
        self.max_detection_range = 20.0   # Maximum detection range
        self.depth_threshold = 0.1        # Depth threshold for obstacle detection

        # Expert policy parameters (with config override)
        if config is None:
            config = {}
        self.obstacle_inflation_factor = config.get('obstacle_inflation_factor', 0.6)
        self.obstacle_distance_threshold = config.get('obstacle_distance_threshold', 8.0)
        self.grid_center_offset = config.get('grid_center_offset', 8.0)
        self.grid_displacement = config.get('grid_displacement', 0.5)
        self.x_displacement = config.get('x_displacement', 8.0)

        self.logger.info(f"Obstacle detector initialized with parameters:")
        self.logger.info(f"  - Inflation factor: {self.obstacle_inflation_factor}")
        self.logger.info(f"  - Distance threshold: {self.obstacle_distance_threshold}m")
        self.logger.info(f"  - Grid center offset: {self.grid_center_offset}m")
        self.logger.info(f"  - Grid displacement: {self.grid_displacement}m")
        self.logger.info(f"  - X displacement: {self.x_displacement}m")
        
        # Detected obstacles cache
        self.detected_obstacles: List[Obstacle] = []
        self.last_detection_time = 0.0
        
    def detect_obstacles_from_depth(self, depth_image: np.ndarray) -> List[Obstacle]:
        """Detect obstacles from depth image
        
        Args:
            depth_image: Depth image array (H, W) normalized to [0, 1]
            
        Returns:
            List of detected obstacles
        """
        obstacles = []
        
        try:
            # Convert normalized depth back to meters (assuming 10m max range)
            depth_meters = depth_image * 10.0
            
            # Find obstacles (close depth values)
            obstacle_mask = (depth_meters > self.min_obstacle_distance) & (depth_meters < self.max_detection_range)
            
            # Use morphological operations to group nearby obstacles
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            obstacle_mask = cv2.morphologyEx(obstacle_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            # Find contours of obstacles
            contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            h, w = depth_image.shape
            
            for contour in contours:
                # Get contour properties
                area = cv2.contourArea(contour)
                if area < 10:  # Skip very small detections
                    continue
                
                # Get bounding rectangle
                x, y, width, height = cv2.boundingRect(contour)
                
                # Calculate obstacle position in image coordinates
                center_x = x + width // 2
                center_y = y + height // 2
                
                # Get depth at obstacle center
                if 0 <= center_y < h and 0 <= center_x < w:
                    obstacle_depth = depth_meters[center_y, center_x]
                    
                    # Convert image coordinates to world coordinates (simplified)
                    # Assume camera FOV and convert pixel position to angle
                    # This is a simplified conversion - in practice would use camera intrinsics
                    fov_horizontal = 90.0  # degrees
                    fov_vertical = 60.0    # degrees
                    
                    angle_x = (center_x - w/2) / w * fov_horizontal * np.pi / 180.0
                    angle_y = (center_y - h/2) / h * fov_vertical * np.pi / 180.0
                    
                    # Convert to world position (relative to drone)
                    world_x = obstacle_depth * np.cos(angle_y) * np.cos(angle_x)
                    world_y = obstacle_depth * np.cos(angle_y) * np.sin(angle_x)
                    world_z = obstacle_depth * np.sin(angle_y)
                    
                    # Estimate obstacle radius from bounding box
                    pixel_radius = max(width, height) / 2
                    estimated_radius = pixel_radius * obstacle_depth / (w / 2) * np.tan(fov_horizontal * np.pi / 360.0)
                    estimated_radius = max(0.5, min(2.0, estimated_radius))  # Clamp radius
                    
                    # Create obstacle
                    obstacle = Obstacle(
                        position=(world_x, world_y, world_z),
                        radius=estimated_radius,
                        confidence=min(1.0, area / 1000.0)  # Confidence based on size
                    )
                    obstacles.append(obstacle)
            
            self.detected_obstacles = obstacles
            self.last_detection_time = time.time()
            
        except Exception as e:
            self.logger.error(f"Error detecting obstacles from depth: {e}")
        
        return obstacles
    
    def check_line_obstacle_collision(self, start: Tuple[float, float, float], 
                                    end: Tuple[float, float, float],
                                    obstacle: Obstacle) -> bool:
        """Check if a line intersects with an obstacle sphere
        
        Args:
            start: Line start point (x, y, z)
            end: Line end point (x, y, z)
            obstacle: Obstacle to check
            
        Returns:
            True if collision detected, False otherwise
        """
        try:
            x1, y1, z1 = start
            x2, y2, z2 = end
            x3, y3, z3 = obstacle.position
            r = obstacle.radius + self.obstacle_inflation_factor
            
            # Line-sphere intersection calculation
            b = 2 * ((x2 - x1) * (x1 - x3) + (y2 - y1) * (y1 - y3) + (z2 - z1) * (z1 - z3))
            a = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
            c = (x3**2 + y3**2 + z3**2 + x1**2 + y1**2 + z1**2 
                 - 2 * (x3 * x1 + y3 * y1 + z3 * z1) - r**2)
            
            discriminant = b**2 - 4 * a * c
            return discriminant >= 0
            
        except Exception as e:
            self.logger.error(f"Error checking line-obstacle collision: {e}")
            return True  # Conservative: assume collision if error
    
    def compute_expert_velocity_command(self, current_position: Tuple[float, float, float],
                                      desired_velocity: float) -> Tuple[float, float, float]:
        """Compute velocity command using expert obstacle avoidance policy
        
        Args:
            current_position: Current drone position (x, y, z)
            desired_velocity: Desired forward velocity in m/s
            
        Returns:
            Velocity command (vx, vy, vz)
        """
        if not self.detected_obstacles:
            # No obstacles detected, move forward
            return (desired_velocity, 0.0, 0.0)
        
        try:
            # Filter obstacles that are ahead and within range
            relevant_obstacles = [
                obs for obs in self.detected_obstacles
                if (obs.position[0] > 0 and 
                    obs.position[0] < self.obstacle_distance_threshold)
            ]
            
            if not relevant_obstacles:
                return (desired_velocity, 0.0, 0.0)
            
            # Create collision grid
            y_vals = np.arange(-self.grid_center_offset, 
                             self.grid_center_offset + self.grid_displacement, 
                             self.grid_displacement)
            z_vals = np.arange(-self.grid_center_offset, 
                             self.grid_center_offset + self.grid_displacement, 
                             self.grid_displacement)
            
            num_points = len(y_vals)
            collision_grid = np.zeros((num_points, num_points))
            waypoints_3d = np.zeros((num_points, num_points, 3))
            
            # Check each waypoint for collisions
            for yi, y in enumerate(y_vals):
                for zi, z in enumerate(z_vals):
                    waypoint = (self.x_displacement, y, z)
                    waypoints_3d[yi, zi] = waypoint
                    
                    # Check collision with all relevant obstacles
                    for obstacle in relevant_obstacles:
                        if self.check_line_obstacle_collision((0, 0, 0), waypoint, obstacle):
                            collision_grid[yi, zi] = 1
                            break
            
            # Find collision-free waypoint closest to center
            if collision_grid.sum() == collision_grid.size:
                # No collision-free path found
                self.logger.warning("No collision-free path found, stopping")
                return (0.5, 0.0, 0.25)
            
            # Find closest zero point to center
            center_idx = num_points // 2
            distances = np.sqrt((np.arange(num_points)[:, None] - center_idx)**2 + 
                              (np.arange(num_points)[None, :] - center_idx)**2)
            
            # Mask out collision points
            distances_masked = np.where(collision_grid == 0, distances, np.inf)
            
            # Find minimum distance point
            min_idx = np.unravel_index(np.argmin(distances_masked), distances_masked.shape)
            best_waypoint = waypoints_3d[min_idx[0], min_idx[1]]
            
            # Normalize to desired velocity magnitude
            waypoint_vector = best_waypoint / np.linalg.norm(best_waypoint)
            velocity_command = waypoint_vector * desired_velocity
            
            return tuple(velocity_command)
            
        except Exception as e:
            self.logger.error(f"Error computing expert velocity command: {e}")
            return (0.0, 0.0, 0.0)
    
    def get_obstacle_summary(self) -> Dict[str, Any]:
        """Get summary of detected obstacles
        
        Returns:
            Dictionary with obstacle summary
        """
        if not self.detected_obstacles:
            return {
                'count': 0,
                'closest_distance': None,
                'average_distance': None,
                'obstacles': []
            }
        
        distances = [np.linalg.norm(obs.position) for obs in self.detected_obstacles]
        
        obstacle_data = []
        for obs in self.detected_obstacles:
            obstacle_data.append({
                'position': obs.position,
                'radius': obs.radius,
                'confidence': obs.confidence,
                'distance': np.linalg.norm(obs.position),
                'age': time.time() - obs.timestamp
            })
        
        return {
            'count': len(self.detected_obstacles),
            'closest_distance': min(distances),
            'average_distance': np.mean(distances),
            'obstacles': obstacle_data,
            'last_detection_time': self.last_detection_time
        }
    
    def visualize_obstacles(self, depth_image: np.ndarray) -> np.ndarray:
        """Create visualization of detected obstacles
        
        Args:
            depth_image: Original depth image
            
        Returns:
            Visualization image with obstacles marked
        """
        # Convert depth to displayable format
        vis_image = (depth_image * 255).astype(np.uint8)
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
        
        try:
            # Mark detected obstacles
            h, w = depth_image.shape
            for i, obstacle in enumerate(self.detected_obstacles):
                # Project obstacle back to image coordinates (simplified)
                if obstacle.position[0] > 0:  # Only forward obstacles
                    # Simplified projection
                    fov_horizontal = 90.0 * np.pi / 180.0
                    fov_vertical = 60.0 * np.pi / 180.0
                    
                    angle_x = np.arctan2(obstacle.position[1], obstacle.position[0])
                    angle_y = np.arctan2(obstacle.position[2], obstacle.position[0])
                    
                    pixel_x = int(w/2 + angle_x / (fov_horizontal/2) * w/2)
                    pixel_y = int(h/2 + angle_y / (fov_vertical/2) * h/2)
                    
                    if 0 <= pixel_x < w and 0 <= pixel_y < h:
                        # Draw obstacle marker
                        cv2.circle(vis_image, (pixel_x, pixel_y), 10, (255, 0, 0), 2)
                        cv2.putText(vis_image, f"{i}", (pixel_x+15, pixel_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add obstacle count
            cv2.putText(vis_image, f"Obstacles: {len(self.detected_obstacles)}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        except Exception as e:
            self.logger.error(f"Error visualizing obstacles: {e}")
        
        return vis_image
    
    def clear_old_obstacles(self, max_age: float = 2.0):
        """Clear obstacles older than specified age
        
        Args:
            max_age: Maximum obstacle age in seconds
        """
        current_time = time.time()
        self.detected_obstacles = [
            obs for obs in self.detected_obstacles 
            if current_time - obs.timestamp < max_age
        ]