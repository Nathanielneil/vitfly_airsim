"""
Model Inference for VitFly-AirSim

This module provides real-time model inference for drone control,
optimized for Windows and AirSim integration.

Author: Adapted from original VitFly project
"""

import torch
import numpy as np
import time
from typing import Tuple, Optional, Any
import logging

try:
    from .vitfly_models import ViT, LSTMNetVIT, ConvNet, LSTMNet, UNetConvLSTMNet
    # Create alias for compatibility
    ViTLSTM = LSTMNetVIT
except ImportError:
    import sys
    from pathlib import Path
    # Try to import from same directory
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from vitfly_models import ViT, LSTMNetVIT, ConvNet, LSTMNet, UNetConvLSTMNet
    # Create alias for compatibility
    ViTLSTM = LSTMNetVIT


class ModelInference:
    """Real-time model inference for drone control"""
    
    def __init__(self, model_path: str, model_type: str, device: str = 'auto'):
        """Initialize model inference
        
        Args:
            model_path: Path to trained model
            model_type: Type of model (ViT, ViTLSTM, etc.)
            device: Device to run inference on
        """
        self.model_path = model_path
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
        # Inference state
        self.hidden_state = None
        self.inference_times = []
        self.max_inference_history = 100
        
        # Performance monitoring
        self.total_inferences = 0
        self.total_inference_time = 0.0
        
    def _load_model(self) -> torch.nn.Module:
        """Load trained model for inference"""
        try:
            # Create model
            if self.model_type == 'ViT':
                model = ViT()
            elif self.model_type == 'ViTLSTM':
                model = ViTLSTM()
            elif self.model_type == 'ConvNet':
                model = ConvNet()
            elif self.model_type == 'LSTMNet':
                model = LSTMNet()
            elif self.model_type == 'UNet':
                model = UNetConvLSTMNet()
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Load weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            # Set to evaluation mode
            model = model.to(self.device)
            model.eval()
            
            # Warm up model for consistent timing
            self._warmup_model(model)
            
            self.logger.info(f"Loaded {self.model_type} model for inference")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _warmup_model(self, model: torch.nn.Module, warmup_iterations: int = 5):
        """Warm up model for consistent inference timing"""
        self.logger.info("Warming up model...")
        
        with torch.no_grad():
            for _ in range(warmup_iterations):
                # Create dummy inputs
                dummy_image = torch.randn(1, 1, 60, 90).to(self.device)
                dummy_velocity = torch.tensor([[5.0]]).to(self.device)
                dummy_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).to(self.device)
                
                inputs = [dummy_image, dummy_velocity, dummy_quaternion]
                
                # Add hidden state for LSTM models
                if 'LSTM' in self.model_type:
                    inputs.append(None)
                
                # Run inference
                _, _ = model(inputs)
        
        self.logger.info("Model warmup completed")
    
    def predict_velocity(self, depth_image: np.ndarray, desired_velocity: float,
                        orientation: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
        """Predict velocity command from sensor inputs
        
        Args:
            depth_image: Depth image array (H, W) or (1, 1, H, W)
            desired_velocity: Desired forward velocity
            orientation: Quaternion orientation (w, x, y, z)
            
        Returns:
            Velocity command (vx, vy, vz)
        """
        start_time = time.time()
        
        try:
            with torch.no_grad():
                # Prepare inputs
                inputs = self._prepare_inputs(depth_image, desired_velocity, orientation)
                
                # Run inference
                prediction, new_hidden_state = self.model(inputs)
                
                # Update hidden state for LSTM models
                if 'LSTM' in self.model_type:
                    self.hidden_state = new_hidden_state
                
                # Convert to velocity command
                velocity_command = self._process_prediction(prediction, desired_velocity)
                
                # Update performance metrics
                inference_time = time.time() - start_time
                self._update_performance_metrics(inference_time)
                
                return velocity_command
                
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            # Return safe default command
            return (0.0, 0.0, 0.0)
    
    def _prepare_inputs(self, depth_image: np.ndarray, desired_velocity: float,
                       orientation: Tuple[float, float, float, float]) -> list:
        """Prepare inputs for model inference"""
        
        # Convert depth image to tensor
        if isinstance(depth_image, np.ndarray):
            # Ensure correct shape (1, 1, H, W)
            if len(depth_image.shape) == 2:
                depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0)
            elif len(depth_image.shape) == 3:
                depth_tensor = torch.from_numpy(depth_image).unsqueeze(0)
            elif len(depth_image.shape) == 4:
                depth_tensor = torch.from_numpy(depth_image)
            else:
                raise ValueError(f"Invalid depth image shape: {depth_image.shape}")
        else:
            depth_tensor = depth_image
        
        # Ensure tensor is on correct device
        depth_tensor = depth_tensor.to(self.device).float()
        
        # Prepare desired velocity tensor
        velocity_tensor = torch.tensor([[desired_velocity]], dtype=torch.float32, device=self.device)
        
        # Prepare quaternion tensor
        quaternion_tensor = torch.tensor([list(orientation)], dtype=torch.float32, device=self.device)
        
        # Create input list
        inputs = [depth_tensor, velocity_tensor, quaternion_tensor]
        
        # Add hidden state for LSTM models
        if 'LSTM' in self.model_type:
            inputs.append(self.hidden_state)
        
        return inputs
    
    def _process_prediction(self, prediction: torch.Tensor, desired_velocity: float) -> Tuple[float, float, float]:
        """Process model prediction to velocity command"""
        
        # Convert to numpy
        pred_np = prediction.cpu().numpy().squeeze()
        
        # Clamp forward velocity to reasonable range
        pred_np[0] = np.clip(pred_np[0], -1.0, 1.0)
        
        # Normalize prediction vector
        pred_norm = np.linalg.norm(pred_np)
        if pred_norm > 0:
            pred_np = pred_np / pred_norm
        
        # Scale by desired velocity
        velocity_command = pred_np * desired_velocity
        
        # Apply safety limits
        max_velocity = 10.0  # Maximum velocity limit
        velocity_magnitude = np.linalg.norm(velocity_command)
        if velocity_magnitude > max_velocity:
            velocity_command = velocity_command * (max_velocity / velocity_magnitude)
        
        return tuple(velocity_command.tolist())
    
    def _update_performance_metrics(self, inference_time: float):
        """Update performance monitoring metrics"""
        self.total_inferences += 1
        self.total_inference_time += inference_time
        
        # Store recent inference times
        self.inference_times.append(inference_time)
        if len(self.inference_times) > self.max_inference_history:
            self.inference_times.pop(0)
        
        # Log performance periodically
        if self.total_inferences % 100 == 0:
            avg_time = self.total_inference_time / self.total_inferences
            recent_avg = np.mean(self.inference_times) if self.inference_times else 0
            self.logger.info(
                f"Inference stats: {self.total_inferences} inferences, "
                f"avg: {avg_time*1000:.2f}ms, recent avg: {recent_avg*1000:.2f}ms"
            )
    
    def reset_hidden_state(self):
        """Reset hidden state for LSTM models"""
        if 'LSTM' in self.model_type:
            self.hidden_state = None
            self.logger.debug("Hidden state reset")
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {'message': 'No inference data available'}
        
        recent_times = np.array(self.inference_times)
        
        return {
            'total_inferences': self.total_inferences,
            'average_time_ms': (self.total_inference_time / self.total_inferences * 1000) if self.total_inferences > 0 else 0,
            'recent_average_ms': float(np.mean(recent_times) * 1000),
            'recent_min_ms': float(np.min(recent_times) * 1000),
            'recent_max_ms': float(np.max(recent_times) * 1000),
            'recent_std_ms': float(np.std(recent_times) * 1000),
            'inference_frequency_hz': 1.0 / np.mean(recent_times) if len(recent_times) > 0 else 0,
            'model_type': self.model_type,
            'device': str(self.device)
        }
    
    def benchmark_inference(self, num_iterations: int = 100) -> dict:
        """Benchmark inference performance
        
        Args:
            num_iterations: Number of iterations to run
            
        Returns:
            Dictionary with benchmark results
        """
        self.logger.info(f"Running inference benchmark ({num_iterations} iterations)...")
        
        # Create dummy inputs
        dummy_depth = np.random.rand(60, 90).astype(np.float32)
        dummy_velocity = 5.0
        dummy_orientation = (1.0, 0.0, 0.0, 0.0)
        
        # Reset hidden state
        self.reset_hidden_state()
        
        # Warm up
        for _ in range(10):
            self.predict_velocity(dummy_depth, dummy_velocity, dummy_orientation)
        
        # Benchmark
        start_time = time.time()
        times = []
        
        for i in range(num_iterations):
            iter_start = time.time()
            self.predict_velocity(dummy_depth, dummy_velocity, dummy_orientation)
            iter_time = time.time() - iter_start
            times.append(iter_time)
        
        total_time = time.time() - start_time
        times = np.array(times)
        
        results = {
            'num_iterations': num_iterations,
            'total_time_s': total_time,
            'average_time_ms': float(np.mean(times) * 1000),
            'min_time_ms': float(np.min(times) * 1000),
            'max_time_ms': float(np.max(times) * 1000),
            'std_time_ms': float(np.std(times) * 1000),
            'throughput_hz': num_iterations / total_time,
            'model_type': self.model_type,
            'device': str(self.device)
        }
        
        self.logger.info(f"Benchmark completed: {results['average_time_ms']:.2f}ms avg, {results['throughput_hz']:.1f} Hz")
        
        return results


if __name__ == '__main__':
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    model_path = "models/vitlstm_best.pth"
    model_type = "ViTLSTM"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        exit(1)
    
    # Create inference engine
    inference = ModelInference(model_path, model_type)
    
    # Run benchmark
    benchmark_results = inference.benchmark_inference(100)
    
    print("Benchmark Results:")
    for key, value in benchmark_results.items():
        print(f"  {key}: {value}")
    
    # Test single inference
    dummy_depth = np.random.rand(60, 90).astype(np.float32)
    velocity_cmd = inference.predict_velocity(dummy_depth, 5.0, (1.0, 0.0, 0.0, 0.0))
    print(f"\nSample velocity command: {velocity_cmd}")