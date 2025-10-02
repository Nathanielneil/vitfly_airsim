#!/usr/bin/env python3
"""
Annotate Flight Video with Visualization Overlays

This script processes recorded AirSim flight videos and adds:
- Velocity vectors (arrows)
- Speed indicators
- Obstacle markers
- Flight statistics
- Trajectory path

Usage:
    python scripts/annotate_flight_video.py --video <input.mp4> --data <flight_data.csv> --output <output.mp4>

Author: VitFly-AirSim Project
"""

import argparse
import cv2
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FlightVideoAnnotator:
    """Annotate flight videos with telemetry data"""

    def __init__(self, video_path: str, data_path: str, output_path: str,
                 config: Optional[Dict] = None):
        """Initialize video annotator

        Args:
            video_path: Path to input video (from AirSim recording)
            data_path: Path to flight data CSV (position, velocity, etc.)
            output_path: Path to output annotated video
            config: Optional configuration dict
        """
        self.video_path = video_path
        self.data_path = data_path
        self.output_path = output_path
        self.config = config or {}

        # Video properties
        self.cap = None
        self.out = None
        self.fps = 0
        self.width = 0
        self.height = 0

        # Flight data
        self.flight_data = None
        self.current_frame = 0

        # Visualization settings
        self.arrow_scale = self.config.get('arrow_scale', 30.0)
        self.show_trajectory = self.config.get('show_trajectory', True)
        self.show_stats = self.config.get('show_stats', True)
        self.trajectory_color = (0, 255, 255)  # Cyan
        self.velocity_color = (0, 255, 0)  # Green

    def load_video(self):
        """Load input video"""
        logger.info(f"Loading video: {self.video_path}")
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video properties - FPS: {self.fps}, Size: {self.width}x{self.height}")

    def load_flight_data(self):
        """Load flight telemetry data"""
        logger.info(f"Loading flight data: {self.data_path}")

        if self.data_path.endswith('.csv'):
            self.flight_data = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.json'):
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                self.flight_data = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data format: {self.data_path}")

        logger.info(f"Loaded {len(self.flight_data)} data points")
        logger.info(f"Columns: {list(self.flight_data.columns)}")

    def setup_output_video(self):
        """Setup output video writer"""
        logger.info(f"Setting up output video: {self.output_path}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )

    def get_data_for_frame(self, frame_idx: int) -> Optional[Dict]:
        """Get flight data for current frame

        Args:
            frame_idx: Current frame index

        Returns:
            Dictionary with flight data or None
        """
        # Map frame to data index (assuming data was recorded at video FPS)
        if frame_idx >= len(self.flight_data):
            return None

        row = self.flight_data.iloc[frame_idx]
        return row.to_dict()

    def draw_velocity_arrow(self, frame: np.ndarray, velocity: Tuple[float, float, float],
                           position: Tuple[int, int] = None) -> np.ndarray:
        """Draw velocity vector as arrow on frame

        Args:
            frame: Video frame
            velocity: Velocity vector (vx, vy, vz)
            position: Arrow start position (default: center)

        Returns:
            Annotated frame
        """
        if position is None:
            position = (self.width // 2, self.height - 150)

        vx, vy, vz = velocity

        # Draw forward velocity (horizontal)
        forward_end = (
            int(position[0] + vx * self.arrow_scale),
            position[1]
        )
        cv2.arrowedLine(frame, position, forward_end, (255, 0, 0), 3, tipLength=0.2)

        # Draw lateral velocity (green)
        lateral_end = (
            int(position[0] + vy * self.arrow_scale),
            int(position[1] - vz * self.arrow_scale)
        )
        cv2.arrowedLine(frame, position, lateral_end, (0, 255, 0), 3, tipLength=0.2)

        # Draw resultant velocity (yellow)
        resultant_end = (
            int(position[0] + vx * self.arrow_scale + vy * self.arrow_scale),
            int(position[1] - vz * self.arrow_scale)
        )
        cv2.arrowedLine(frame, position, resultant_end, (0, 255, 255), 2, tipLength=0.15)

        # Draw center point
        cv2.circle(frame, position, 5, (255, 255, 255), -1)

        return frame

    def draw_stats_panel(self, frame: np.ndarray, data: Dict) -> np.ndarray:
        """Draw statistics panel on frame

        Args:
            frame: Video frame
            data: Flight data dictionary

        Returns:
            Annotated frame
        """
        # Create semi-transparent overlay
        overlay = frame.copy()
        panel_height = 200
        panel_width = 500

        # Dark panel background
        cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height),
                     (40, 40, 40), -1)

        # Blend with frame
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Add text information
        y_offset = 40
        line_height = 35

        # Extract data
        vx = data.get('velocity_x', 0.0)
        vy = data.get('velocity_y', 0.0)
        vz = data.get('velocity_z', 0.0)
        speed = np.sqrt(vx**2 + vy**2 + vz**2)

        pos_x = data.get('position_x', 0.0)
        pos_y = data.get('position_y', 0.0)
        pos_z = data.get('position_z', 0.0)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        # Velocity
        vel_text = f"Velocity: X={vx:+.2f} Y={vy:+.2f} Z={vz:+.2f} m/s"
        cv2.putText(frame, vel_text, (20, y_offset), font, font_scale, (255, 255, 255), font_thickness)

        # Speed
        speed_text = f"Speed: {speed:.2f} m/s"
        cv2.putText(frame, speed_text, (20, y_offset + line_height), font, font_scale, (0, 255, 255), font_thickness)

        # Position
        pos_text = f"Position: X={pos_x:.1f} Y={pos_y:.1f} Z={pos_z:.1f} m"
        cv2.putText(frame, pos_text, (20, y_offset + line_height * 2), font, font_scale, (255, 255, 255), font_thickness)

        # Timestamp
        timestamp = data.get('timestamp', 0.0)
        time_text = f"Time: {timestamp:.2f}s"
        cv2.putText(frame, time_text, (20, y_offset + line_height * 3), font, font_scale, (200, 200, 200), font_thickness)

        # Legend
        legend_y = y_offset + line_height * 4 + 10
        cv2.putText(frame, "Blue=Forward  Green=Lateral  Yellow=Resultant",
                   (20, legend_y), font, 0.5, (200, 200, 200), 1)

        return frame

    def draw_trajectory_trace(self, frame: np.ndarray, current_idx: int,
                            trail_length: int = 100) -> np.ndarray:
        """Draw trajectory trace on frame

        Args:
            frame: Video frame
            current_idx: Current data index
            trail_length: Number of past positions to show

        Returns:
            Annotated frame
        """
        if 'position_x' not in self.flight_data.columns:
            return frame

        # Get trajectory points
        start_idx = max(0, current_idx - trail_length)
        trajectory = self.flight_data.iloc[start_idx:current_idx + 1]

        if len(trajectory) < 2:
            return frame

        # Project 3D positions to 2D screen (top-down view in corner)
        map_size = 200
        map_x = self.width - map_size - 20
        map_y = 20

        # Create mini-map background
        cv2.rectangle(frame, (map_x, map_y), (map_x + map_size, map_y + map_size),
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (map_x, map_y), (map_x + map_size, map_y + map_size),
                     (150, 150, 150), 2)

        # Add label
        cv2.putText(frame, "Trajectory (Top View)", (map_x, map_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Get position bounds for scaling
        x_positions = trajectory['position_x'].values
        y_positions = trajectory['position_y'].values

        x_min, x_max = x_positions.min(), x_positions.max()
        y_min, y_max = y_positions.min(), y_positions.max()

        x_range = max(x_max - x_min, 1.0)
        y_range = max(y_max - y_min, 1.0)

        # Scale to fit in mini-map
        scale = min((map_size - 40) / x_range, (map_size - 40) / y_range)

        # Draw trajectory
        points = []
        for idx, row in trajectory.iterrows():
            px = int(map_x + map_size // 2 + (row['position_x'] - (x_min + x_max) / 2) * scale)
            py = int(map_y + map_size // 2 - (row['position_y'] - (y_min + y_max) / 2) * scale)
            points.append((px, py))

        # Draw line segments with fading effect
        for i in range(1, len(points)):
            alpha = i / len(points)  # Fade in towards current position
            color = tuple(int(c * alpha) for c in self.trajectory_color)
            cv2.line(frame, points[i-1], points[i], color, 2)

        # Draw current position
        if points:
            cv2.circle(frame, points[-1], 5, (255, 255, 255), -1)
            cv2.circle(frame, points[-1], 6, (0, 0, 255), 2)

        return frame

    def process_video(self):
        """Process entire video and add annotations"""
        logger.info("Starting video processing...")

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Total frames to process: {total_frames}")

        frame_idx = 0

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            # Get flight data for this frame
            data = self.get_data_for_frame(frame_idx)

            if data is not None:
                # Draw velocity arrow
                velocity = (
                    data.get('velocity_x', 0.0),
                    data.get('velocity_y', 0.0),
                    data.get('velocity_z', 0.0)
                )
                frame = self.draw_velocity_arrow(frame, velocity)

                # Draw stats panel
                if self.show_stats:
                    frame = self.draw_stats_panel(frame, data)

                # Draw trajectory trace
                if self.show_trajectory:
                    frame = self.draw_trajectory_trace(frame, frame_idx)

            # Write frame
            self.out.write(frame)

            # Progress indicator
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                logger.info(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames})")

            frame_idx += 1

        logger.info("Video processing completed!")

    def cleanup(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()

    def run(self):
        """Run the video annotation pipeline"""
        try:
            self.load_video()
            self.load_flight_data()
            self.setup_output_video()
            self.process_video()
            logger.info(f"Annotated video saved to: {self.output_path}")
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
        finally:
            self.cleanup()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Annotate flight videos with telemetry data'
    )
    parser.add_argument('--video', type=str, required=True,
                       help='Input video file from AirSim')
    parser.add_argument('--data', type=str, required=True,
                       help='Flight data CSV/JSON file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output annotated video file')
    parser.add_argument('--arrow-scale', type=float, default=30.0,
                       help='Scale factor for velocity arrows')
    parser.add_argument('--no-trajectory', action='store_true',
                       help='Disable trajectory trace')
    parser.add_argument('--no-stats', action='store_true',
                       help='Disable statistics panel')

    args = parser.parse_args()

    # Check input files exist
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)

    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)

    # Configuration
    config = {
        'arrow_scale': args.arrow_scale,
        'show_trajectory': not args.no_trajectory,
        'show_stats': not args.no_stats,
    }

    # Create annotator and run
    annotator = FlightVideoAnnotator(
        args.video,
        args.data,
        args.output,
        config
    )

    annotator.run()
    logger.info("Done!")


if __name__ == '__main__':
    main()
