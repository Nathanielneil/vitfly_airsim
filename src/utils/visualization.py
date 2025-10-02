"""
Visualization Utilities for VitFly-AirSim

This module provides real-time visualization capabilities for
simulation monitoring and debugging.

Author: Adapted from original VitFly project
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import threading
import queue
import time
from typing import List, Tuple, Optional, Any
import logging


class SimulationVisualizer:
    """Real-time visualization for VitFly simulation"""
    
    def __init__(self, window_size: Tuple[int, int] = (800, 600), 
                 update_rate: float = 30.0):
        """Initialize visualizer
        
        Args:
            window_size: Size of visualization window
            update_rate: Update rate in Hz
        """
        self.window_size = window_size
        self.update_rate = update_rate
        self.logger = logging.getLogger(__name__)
        
        # Visualization state
        self.running = False
        self.current_depth = None
        self.current_velocity = None
        self.current_obstacles = []
        
        # Threading
        self.update_queue = queue.Queue(maxsize=10)
        self.vis_thread = None
        
        # OpenCV windows
        self.depth_window = "VitFly - Depth View"
        self.debug_window = "VitFly - Debug View"
        
    def start(self):
        """Start visualization"""
        if self.running:
            return
        
        self.running = True
        self.vis_thread = threading.Thread(target=self._visualization_loop, daemon=True)
        self.vis_thread.start()
        
        self.logger.info("Visualization started")
    
    def stop(self):
        """Stop visualization"""
        self.running = False

        # Give the visualization thread time to exit cleanly
        if self.vis_thread and self.vis_thread.is_alive():
            self.vis_thread.join(timeout=2.0)

        # Force close all OpenCV windows
        try:
            cv2.destroyAllWindows()
            # Call waitKey to process window close events
            cv2.waitKey(1)
        except Exception as e:
            self.logger.debug(f"Error closing OpenCV windows: {e}")

        self.logger.info("Visualization stopped")
    
    def update(self, depth_image: np.ndarray, velocity_command: Tuple[float, float, float],
              obstacles: Optional[List] = None):
        """Update visualization with new data
        
        Args:
            depth_image: Current depth image
            velocity_command: Current velocity command (vx, vy, vz)
            obstacles: List of detected obstacles
        """
        if not self.running:
            self.start()
        
        # Package data for visualization thread
        vis_data = {
            'depth_image': depth_image.copy() if depth_image is not None else None,
            'velocity_command': velocity_command,
            'obstacles': obstacles if obstacles is not None else [],
            'timestamp': time.time()
        }
        
        # Add to queue (drop old data if queue is full)
        try:
            self.update_queue.put_nowait(vis_data)
        except queue.Full:
            try:
                self.update_queue.get_nowait()  # Remove old data
                self.update_queue.put_nowait(vis_data)
            except queue.Empty:
                pass
    
    def update_model_prediction(self, depth_image: np.ndarray, velocity_command: Tuple[float, float, float]):
        """Update visualization with model prediction"""
        self.update(depth_image, velocity_command, [])
    
    def _visualization_loop(self):
        """Main visualization loop (runs in separate thread)"""
        update_period = 1.0 / self.update_rate
        
        while self.running:
            loop_start = time.time()
            
            try:
                # Get latest data
                vis_data = None
                while not self.update_queue.empty():
                    try:
                        vis_data = self.update_queue.get_nowait()
                    except queue.Empty:
                        break
                
                if vis_data is not None:
                    self._update_display(vis_data)
                
                # Handle window events
                cv2.waitKey(1)
                
            except Exception as e:
                self.logger.error(f"Visualization error: {e}")
            
            # Maintain update rate
            elapsed = time.time() - loop_start
            if elapsed < update_period:
                time.sleep(update_period - elapsed)
    
    def _update_display(self, vis_data: dict):
        """Update display with new data"""
        depth_image = vis_data['depth_image']
        velocity_command = vis_data['velocity_command']
        obstacles = vis_data['obstacles']
        
        if depth_image is None:
            return
        
        # Create depth visualization
        depth_vis = self._create_depth_visualization(depth_image, velocity_command)
        
        # Create debug visualization
        debug_vis = self._create_debug_visualization(depth_image, velocity_command, obstacles)
        
        # Display windows
        cv2.imshow(self.depth_window, depth_vis)
        cv2.imshow(self.debug_window, debug_vis)
    
    def _create_depth_visualization(self, depth_image: np.ndarray, 
                                  velocity_command: Tuple[float, float, float]) -> np.ndarray:
        """Create depth image visualization with velocity overlay"""
        
        # Normalize depth image for display
        if depth_image.max() <= 1.0:
            vis_image = (depth_image * 255).astype(np.uint8)
        else:
            vis_image = np.clip(depth_image, 0, 255).astype(np.uint8)
        
        # Convert to RGB for color overlays
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        h, w = vis_image.shape[:2]
        
        # Draw velocity vector
        center = (w // 2, h // 2)
        scale = min(w, h) // 6
        
        # Scale velocity for visualization (assuming max velocity ~10 m/s)
        vel_scale = 0.1
        arrow_end = (
            int(center[0] + velocity_command[1] * scale * vel_scale),  # y -> x
            int(center[1] - velocity_command[2] * scale * vel_scale)   # z -> y (inverted)
        )
        
        # Draw arrow
        cv2.arrowedLine(vis_image, center, arrow_end, (0, 255, 0), 2, tipLength=0.3)
        
        # Draw forward velocity indicator
        forward_length = int(velocity_command[0] * scale * vel_scale)
        forward_end = (center[0] + forward_length, center[1])
        cv2.arrowedLine(vis_image, center, forward_end, (255, 0, 0), 2, tipLength=0.3)
        
        # Add velocity text
        vel_text = f"Vel: ({velocity_command[0]:.2f}, {velocity_command[1]:.2f}, {velocity_command[2]:.2f})"
        cv2.putText(vis_image, vel_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add speed text
        speed = np.linalg.norm(velocity_command)
        speed_text = f"Speed: {speed:.2f} m/s"
        cv2.putText(vis_image, speed_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image
    
    def _create_debug_visualization(self, depth_image: np.ndarray,
                                  velocity_command: Tuple[float, float, float],
                                  obstacles: List) -> np.ndarray:
        """Create debug visualization with obstacle information"""
        
        # Create a larger canvas for debug info
        debug_height = 400
        debug_width = 600
        debug_image = np.zeros((debug_height, debug_width, 3), dtype=np.uint8)
        
        # Resize depth image to fit in debug window
        img_height = 200
        img_width = int(img_height * depth_image.shape[1] / depth_image.shape[0])
        
        depth_resized = cv2.resize(depth_image, (img_width, img_height))
        if depth_resized.max() <= 1.0:
            depth_resized = (depth_resized * 255).astype(np.uint8)
        
        if len(depth_resized.shape) == 2:
            depth_resized = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR)
        
        # Place depth image in debug canvas
        debug_image[10:10+img_height, 10:10+img_width] = depth_resized
        
        # Draw velocity command visualization
        vel_x = 10 + img_width + 20
        vel_y = 20
        
        cv2.putText(debug_image, "Velocity Command:", (vel_x, vel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(debug_image, f"X (forward): {velocity_command[0]:.3f}", 
                   (vel_x, vel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.putText(debug_image, f"Y (left): {velocity_command[1]:.3f}", 
                   (vel_x, vel_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(debug_image, f"Z (up): {velocity_command[2]:.3f}", 
                   (vel_x, vel_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        speed = np.linalg.norm(velocity_command)
        cv2.putText(debug_image, f"Speed: {speed:.3f} m/s", 
                   (vel_x, vel_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw obstacle information
        obs_y = vel_y + 120
        cv2.putText(debug_image, f"Obstacles: {len(obstacles)}", 
                   (vel_x, obs_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        for i, obstacle in enumerate(obstacles[:5]):  # Show max 5 obstacles
            if hasattr(obstacle, 'position') and hasattr(obstacle, 'radius'):
                obs_text = f"{i+1}: pos({obstacle.position[0]:.1f}, {obstacle.position[1]:.1f}, {obstacle.position[2]:.1f}), r={obstacle.radius:.1f}"
                cv2.putText(debug_image, obs_text, (vel_x, obs_y + 20 + i*15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw velocity vector in 2D representation
        vec_center_x = vel_x + 100
        vec_center_y = obs_y + 120
        vec_size = 50
        
        cv2.circle(debug_image, (vec_center_x, vec_center_y), vec_size, (100, 100, 100), 1)
        
        # Draw velocity components as vectors
        vel_scale = vec_size / 5.0  # Scale for visualization
        
        # Forward velocity (red)
        end_x = int(vec_center_x + velocity_command[0] * vel_scale)
        cv2.arrowedLine(debug_image, (vec_center_x, vec_center_y), (end_x, vec_center_y), 
                       (0, 0, 255), 2, tipLength=0.3)
        
        # Lateral velocity (green)
        end_y = int(vec_center_y - velocity_command[1] * vel_scale)  # Invert Y
        cv2.arrowedLine(debug_image, (vec_center_x, vec_center_y), (vec_center_x, end_y), 
                       (0, 255, 0), 2, tipLength=0.3)
        
        # Combined horizontal velocity (yellow)
        end_combined = (
            int(vec_center_x + velocity_command[0] * vel_scale),
            int(vec_center_y - velocity_command[1] * vel_scale)
        )
        cv2.arrowedLine(debug_image, (vec_center_x, vec_center_y), end_combined, 
                       (0, 255, 255), 1, tipLength=0.2)
        
        return debug_image
    
    def close(self):
        """Close visualization (alias for stop)"""
        self.stop()


class PerformanceVisualizer:
    """Visualize performance metrics"""
    
    def __init__(self, max_history: int = 1000):
        """Initialize performance visualizer"""
        self.max_history = max_history
        self.inference_times = []
        self.timestamps = []
        
        # Matplotlib setup
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('VitFly Performance Monitoring')
        
    def update_inference_time(self, inference_time: float):
        """Update inference time data"""
        self.inference_times.append(inference_time * 1000)  # Convert to ms
        self.timestamps.append(time.time())
        
        # Keep only recent data
        if len(self.inference_times) > self.max_history:
            self.inference_times.pop(0)
            self.timestamps.pop(0)
    
    def plot_performance(self):
        """Plot performance metrics"""
        if len(self.inference_times) < 2:
            return
        
        # Clear axes
        for ax in self.axes.flat:
            ax.clear()
        
        times = np.array(self.inference_times)
        
        # Inference time over time
        self.axes[0, 0].plot(self.timestamps, self.inference_times, 'b-', alpha=0.7)
        self.axes[0, 0].set_title('Inference Time Over Time')
        self.axes[0, 0].set_ylabel('Time (ms)')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # Inference time histogram
        self.axes[0, 1].hist(self.inference_times, bins=30, alpha=0.7)
        self.axes[0, 1].set_title('Inference Time Distribution')
        self.axes[0, 1].set_xlabel('Time (ms)')
        self.axes[0, 1].set_ylabel('Frequency')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # Rolling average
        window_size = min(50, len(times))
        if window_size > 1:
            rolling_avg = np.convolve(times, np.ones(window_size)/window_size, mode='valid')
            self.axes[1, 0].plot(self.timestamps[window_size-1:], rolling_avg, 'r-', linewidth=2)
            self.axes[1, 0].set_title(f'Rolling Average (window={window_size})')
            self.axes[1, 0].set_ylabel('Time (ms)')
            self.axes[1, 0].grid(True, alpha=0.3)
        
        # Statistics
        stats_text = f"""
        Mean: {np.mean(times):.2f} ms
        Median: {np.median(times):.2f} ms
        Std: {np.std(times):.2f} ms
        Min: {np.min(times):.2f} ms
        Max: {np.max(times):.2f} ms
        Frequency: {1000/np.mean(times):.1f} Hz
        """
        
        self.axes[1, 1].text(0.1, 0.5, stats_text, transform=self.axes[1, 1].transAxes,
                            fontsize=10, verticalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        self.axes[1, 1].set_title('Performance Statistics')
        self.axes[1, 1].set_xlim(0, 1)
        self.axes[1, 1].set_ylim(0, 1)
        self.axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def save_plot(self, filepath: str):
        """Save current plot to file"""
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    # Test visualization
    import time
    
    # Create visualizer
    vis = SimulationVisualizer()
    
    try:
        # Simulate data updates
        for i in range(100):
            # Create fake depth image
            depth = np.random.rand(60, 90).astype(np.float32)
            
            # Create fake velocity command
            velocity = (
                2.0 + np.sin(i * 0.1),
                np.cos(i * 0.05),
                0.1 * np.sin(i * 0.2)
            )
            
            # Update visualization
            vis.update(depth, velocity)
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("Stopping visualization test...")
    
    finally:
        vis.stop()