"""
Sensor Manager for VitFly-AirSim

This module handles all sensor data acquisition from AirSim including
depth cameras, RGB cameras, and IMU data.

Author: Adapted from original VitFly project
"""

import airsim
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any, List
import logging
import time


class SensorManager:
    """Manages sensor data acquisition from AirSim"""
    
    def __init__(self, client: airsim.MultirotorClient, drone_name: str = "Drone1"):
        """Initialize sensor manager
        
        Args:
            client: AirSim client instance
            drone_name: Name of the drone
        """
        self.client = client
        self.drone_name = drone_name
        self.logger = logging.getLogger(__name__)
        
        # Camera configurations
        # Note: In AirSim, depth is retrieved from the same camera using different ImageType
        self.depth_camera_name = "FrontCamera"  # Use FrontCamera for depth (ImageType 1)
        self.rgb_camera_name = "FrontCamera"

        # Image processing parameters
        # Note: AirSim is configured to match D435i native resolution (848x480)
        # Images are resized to model input size (90x60) during preprocessing
        self.native_width = 848   # D435i native width
        self.native_height = 480  # D435i native height
        self.target_height = 60   # Model input height
        self.target_width = 90    # Model input width
        self.depth_scale = 1000.0  # Convert to meters
        
    def get_depth_image(self, camera_name: Optional[str] = None) -> Optional[np.ndarray]:
        """Get depth image from AirSim
        
        Args:
            camera_name: Name of the depth camera
            
        Returns:
            Depth image as numpy array (H, W) or None if error
        """
        if camera_name is None:
            camera_name = self.depth_camera_name
            
        try:
            # Get depth image from AirSim using Images API (returns uncompressed data)
            responses = self.client.simGetImages([
                airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)
            ], vehicle_name=self.drone_name)

            if not responses or len(responses) == 0:
                self.logger.error(f"No response from simGetImages for camera: {camera_name}")
                return None

            response = responses[0]

            # Debug: Print response details
            self.logger.info(f"Depth image response - Width: {response.width}, Height: {response.height}")
            self.logger.info(f"Image data types available - float: {len(response.image_data_float) if response.image_data_float else 0}, "
                           f"uint8: {len(response.image_data_uint8) if response.image_data_uint8 else 0}")

            # Check if image data is available
            if response.width == 0 or response.height == 0:
                self.logger.error(f"Invalid image dimensions: {response.width}x{response.height}")
                return None

            # Get image data (try float first, then uint8)
            img_data = response.image_data_float
            if img_data is None or len(img_data) == 0:
                self.logger.warning("image_data_float is empty, trying image_data_uint8")
                img_data = response.image_data_uint8
                if img_data is None or len(img_data) == 0:
                    self.logger.error("Both image_data_float and image_data_uint8 are empty")
                    return None
                # Convert uint8 to float
                self.logger.info("Using uint8 depth data, converting to float")
                depth_img = np.array(img_data, dtype=np.float32)
                depth_img = depth_img.reshape(response.height, response.width)
                # Uint8 depth is usually normalized, scale back to meters
                depth_img = depth_img * 100.0 / 255.0  # Assume max 100m range
            else:
                # Convert to numpy array (already float32 from AirSim)
                depth_img = np.array(img_data, dtype=np.float32)
                depth_img = depth_img.reshape(response.height, response.width)

            # Handle invalid depth values
            depth_img[depth_img > 100] = 100  # Clamp far distances
            depth_img = np.nan_to_num(depth_img, nan=100.0, posinf=100.0, neginf=0.0)

            # Resize to target size for model input
            depth_img = cv2.resize(depth_img, (self.target_width, self.target_height))
            
            # Normalize to [0, 1] range for model input
            depth_img = np.clip(depth_img / 10.0, 0.0, 1.0)  # Assume 10m max range
            
            return depth_img.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error getting depth image: {e}")
            return None
    
    def get_rgb_image(self, camera_name: Optional[str] = None) -> Optional[np.ndarray]:
        """Get RGB image from AirSim
        
        Args:
            camera_name: Name of the RGB camera
            
        Returns:
            RGB image as numpy array (H, W, 3) or None if error
        """
        if camera_name is None:
            camera_name = self.rgb_camera_name
            
        try:
            # Get RGB image from AirSim
            response = self.client.simGetImage(
                camera_name,
                airsim.ImageType.Scene,
                vehicle_name=self.drone_name
            )
            
            if response is None or len(response) == 0:
                self.logger.warning("Empty RGB image response")
                return None
            
            # Convert to numpy array (D435i native resolution: 848x480)
            img_1d = np.frombuffer(response, dtype=np.uint8)
            img_rgb = img_1d.reshape(self.native_height, self.native_width, 3)

            # Convert BGR to RGB if needed
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            
            return img_rgb
            
        except Exception as e:
            self.logger.error(f"Error getting RGB image: {e}")
            return None
    
    def get_imu_data(self) -> Optional[Dict[str, Any]]:
        """Get IMU data from drone
        
        Returns:
            Dictionary with IMU data or None if error
        """
        try:
            imu_data = self.client.getImuData(vehicle_name=self.drone_name)
            
            return {
                'angular_velocity': (
                    imu_data.angular_velocity.x_val,
                    imu_data.angular_velocity.y_val,
                    imu_data.angular_velocity.z_val
                ),
                'linear_acceleration': (
                    imu_data.linear_acceleration.x_val,
                    imu_data.linear_acceleration.y_val,
                    imu_data.linear_acceleration.z_val
                ),
                'orientation': (
                    imu_data.orientation.w_val,
                    imu_data.orientation.x_val,
                    imu_data.orientation.y_val,
                    imu_data.orientation.z_val
                ),
                'time_stamp': imu_data.time_stamp
            }
            
        except Exception as e:
            self.logger.error(f"Error getting IMU data: {e}")
            return None
    
    def get_gps_data(self) -> Optional[Dict[str, Any]]:
        """Get GPS data from drone
        
        Returns:
            Dictionary with GPS data or None if error
        """
        try:
            gps_data = self.client.getGpsData(vehicle_name=self.drone_name)
            
            return {
                'latitude': gps_data.gnss.geo_point.latitude,
                'longitude': gps_data.gnss.geo_point.longitude,
                'altitude': gps_data.gnss.geo_point.altitude,
                'velocity': (
                    gps_data.gnss.velocity.x_val,
                    gps_data.gnss.velocity.y_val,
                    gps_data.gnss.velocity.z_val
                ),
                'time_stamp': gps_data.time_stamp
            }
            
        except Exception as e:
            self.logger.error(f"Error getting GPS data: {e}")
            return None
    
    def get_barometer_data(self) -> Optional[Dict[str, Any]]:
        """Get barometer data from drone
        
        Returns:
            Dictionary with barometer data or None if error
        """
        try:
            baro_data = self.client.getBarometerData(vehicle_name=self.drone_name)
            
            return {
                'altitude': baro_data.altitude,
                'pressure': baro_data.pressure,
                'qnh': baro_data.qnh,
                'time_stamp': baro_data.time_stamp
            }
            
        except Exception as e:
            self.logger.error(f"Error getting barometer data: {e}")
            return None
    
    def get_processed_depth_for_model(self) -> Optional[Tuple[np.ndarray, bool]]:
        """Get depth image processed specifically for VitFly model input
        
        Returns:
            Tuple of (processed_depth_image, success_flag) or (None, False) if error
        """
        depth_img = self.get_depth_image()
        if depth_img is None:
            return None, False
        
        try:
            # Ensure correct shape for model (1, 1, H, W) - batch and channel dims
            if len(depth_img.shape) == 2:
                depth_img = depth_img[np.newaxis, np.newaxis, :, :]  # Add batch and channel dims
            elif len(depth_img.shape) == 3:
                depth_img = depth_img[np.newaxis, :, :, :]  # Add batch dim
            
            return depth_img, True
            
        except Exception as e:
            self.logger.error(f"Error processing depth image for model: {e}")
            return None, False
    
    def create_debug_image(self, depth_img: np.ndarray, 
                          velocity_command: Tuple[float, float, float]) -> Optional[np.ndarray]:
        """Create debug image with velocity vector overlay
        
        Args:
            depth_img: Depth image array
            velocity_command: Velocity command (vx, vy, vz)
            
        Returns:
            Debug image with velocity arrow or None if error
        """
        try:
            # Convert depth to displayable format
            if depth_img.max() <= 1.0:
                display_img = (depth_img * 255).astype(np.uint8)
            else:
                display_img = depth_img.astype(np.uint8)
            
            # Convert to RGB for arrow drawing
            if len(display_img.shape) == 2:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
            
            # Draw velocity arrow
            h, w = display_img.shape[:2]
            center = (w // 2, h // 2)
            
            # Scale velocity for visualization
            scale = min(w, h) // 6
            arrow_end = (
                int(center[0] + velocity_command[1] * scale),  # y velocity -> x direction
                int(center[1] - velocity_command[2] * scale)   # z velocity -> y direction (inverted)
            )
            
            # Draw arrow
            cv2.arrowedLine(
                display_img, center, arrow_end,
                color=(0, 255, 0),  # Green arrow
                thickness=2,
                tipLength=0.3
            )
            
            # Add velocity text
            vel_text = f"V: ({velocity_command[0]:.2f}, {velocity_command[1]:.2f}, {velocity_command[2]:.2f})"
            cv2.putText(
                display_img, vel_text,
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )
            
            return display_img
            
        except Exception as e:
            self.logger.error(f"Error creating debug image: {e}")
            return None
    
    def save_sensor_data(self, filepath: str, 
                        include_rgb: bool = True, 
                        include_imu: bool = True) -> bool:
        """Save current sensor data to file
        
        Args:
            filepath: Base filepath for saving data
            include_rgb: Whether to save RGB image
            include_imu: Whether to save IMU data
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            timestamp = str(int(time.time() * 1000))
            
            # Save depth image
            depth_img = self.get_depth_image()
            if depth_img is not None:
                depth_path = f"{filepath}_depth_{timestamp}.npy"
                np.save(depth_path, depth_img)
            
            # Save RGB image if requested
            if include_rgb:
                rgb_img = self.get_rgb_image()
                if rgb_img is not None:
                    rgb_path = f"{filepath}_rgb_{timestamp}.png"
                    cv2.imwrite(rgb_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            
            # Save IMU data if requested
            if include_imu:
                imu_data = self.get_imu_data()
                if imu_data is not None:
                    imu_path = f"{filepath}_imu_{timestamp}.npy"
                    np.save(imu_path, imu_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving sensor data: {e}")
            return False
    
    def get_all_sensor_data(self) -> Dict[str, Any]:
        """Get all available sensor data in one call
        
        Returns:
            Dictionary with all sensor data
        """
        return {
            'depth_image': self.get_depth_image(),
            'rgb_image': self.get_rgb_image(),
            'imu_data': self.get_imu_data(),
            'gps_data': self.get_gps_data(),
            'barometer_data': self.get_barometer_data(),
            'timestamp': time.time()
        }