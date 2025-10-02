"""
Model Evaluator for VitFly-AirSim

This module provides evaluation functionality for trained VitFly models,
including metrics computation and visualization.

Author: Adapted from original VitFly project
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import json
import os

try:
    from ..models import ViT, ViTLSTM, ConvNet, LSTMNet, UNetConvLSTMNet
    from .data_loader import VitFlyDataLoader
except ImportError:
    import sys
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from models import ViT, ViTLSTM, ConvNet, LSTMNet, UNetConvLSTMNet
    from training.data_loader import VitFlyDataLoader


class ModelEvaluator:
    """Evaluates trained VitFly models"""
    
    def __init__(self, model_path: str, model_type: str, device: str = 'auto'):
        """Initialize evaluator
        
        Args:
            model_path: Path to trained model
            model_type: Type of model (ViT, ViTLSTM, etc.)
            device: Device to run evaluation on
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
        
    def _load_model(self) -> torch.nn.Module:
        """Load trained model"""
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
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            self.logger.info(f"Loaded {self.model_type} model from {self.model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def evaluate_dataset(self, data_loader: VitFlyDataLoader, 
                        split: str = 'val') -> Dict[str, Any]:
        """Evaluate model on dataset
        
        Args:
            data_loader: Data loader with evaluation data
            split: Which split to evaluate ('train' or 'val')
            
        Returns:
            Dictionary with evaluation metrics
        """
        if split == 'val':
            loader = data_loader.val_loader
        else:
            loader = data_loader.train_loader
        
        self.model.eval()
        
        total_loss = 0.0
        predictions = []
        targets = []
        velocities = []
        errors = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                # Move data to device
                images = batch['image'].to(self.device)
                desired_velocities = batch['desired_velocity'].to(self.device)
                quaternions = batch['quaternion'].to(self.device)
                velocity_commands = batch['velocity_command'].to(self.device)
                
                # Prepare model inputs
                model_inputs = [images, desired_velocities, quaternions]
                
                # Forward pass
                pred, _ = self.model(model_inputs)
                
                # Normalize targets
                normalized_targets = velocity_commands / desired_velocities.unsqueeze(-1)
                
                # Compute loss
                loss = F.mse_loss(pred, normalized_targets)
                total_loss += loss.item()
                
                # Store predictions and targets for analysis
                predictions.append(pred.cpu().numpy())
                targets.append(normalized_targets.cpu().numpy())
                velocities.append(desired_velocities.cpu().numpy())
                
                # Compute per-sample errors
                sample_errors = torch.mean((pred - normalized_targets) ** 2, dim=1).cpu().numpy()
                errors.extend(sample_errors)
        
        # Concatenate all predictions and targets
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        velocities = np.concatenate(velocities, axis=0)
        errors = np.array(errors)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, targets, velocities, errors, total_loss / len(loader))
        
        self.logger.info(f"Evaluation on {split} set completed")
        return metrics
    
    def _compute_metrics(self, predictions: np.ndarray, targets: np.ndarray,
                        velocities: np.ndarray, errors: np.ndarray, 
                        avg_loss: float) -> Dict[str, Any]:
        """Compute evaluation metrics
        
        Args:
            predictions: Model predictions (N, 3)
            targets: Ground truth targets (N, 3)
            velocities: Desired velocities (N, 1)
            errors: Per-sample MSE errors (N,)
            avg_loss: Average loss
            
        Returns:
            Dictionary with computed metrics
        """
        # Basic metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # Per-axis metrics
        mse_per_axis = np.mean((predictions - targets) ** 2, axis=0)
        rmse_per_axis = np.sqrt(mse_per_axis)
        mae_per_axis = np.mean(np.abs(predictions - targets), axis=0)
        
        # Correlation metrics
        correlations = []
        for i in range(3):
            corr = np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0.0)
        
        # Direction accuracy (for normalized vectors)
        pred_norms = np.linalg.norm(predictions, axis=1, keepdims=True)
        target_norms = np.linalg.norm(targets, axis=1, keepdims=True)
        
        pred_normalized = predictions / (pred_norms + 1e-8)
        target_normalized = targets / (target_norms + 1e-8)
        
        dot_products = np.sum(pred_normalized * target_normalized, axis=1)
        direction_errors = np.arccos(np.clip(dot_products, -1, 1))  # Angular error in radians
        avg_direction_error = np.mean(direction_errors)
        
        # Speed accuracy
        pred_speeds = np.linalg.norm(predictions, axis=1)
        target_speeds = np.linalg.norm(targets, axis=1)
        speed_error = np.mean(np.abs(pred_speeds - target_speeds))
        
        # Error distribution
        error_percentiles = np.percentile(errors, [25, 50, 75, 90, 95, 99])
        
        return {
            'loss': avg_loss,
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mse_per_axis': mse_per_axis.tolist(),
            'rmse_per_axis': rmse_per_axis.tolist(),
            'mae_per_axis': mae_per_axis.tolist(),
            'correlations': correlations,
            'direction_error_rad': float(avg_direction_error),
            'direction_error_deg': float(np.degrees(avg_direction_error)),
            'speed_error': float(speed_error),
            'error_percentiles': {
                'p25': float(error_percentiles[0]),
                'p50': float(error_percentiles[1]),
                'p75': float(error_percentiles[2]),
                'p90': float(error_percentiles[3]),
                'p95': float(error_percentiles[4]),
                'p99': float(error_percentiles[5])
            },
            'num_samples': len(predictions)
        }
    
    def evaluate_single_trajectory(self, trajectory_data: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Evaluate model on single trajectory
        
        Args:
            trajectory_data: List of trajectory data points
            
        Returns:
            Dictionary with trajectory evaluation results
        """
        self.model.eval()
        
        predictions = []
        targets = []
        hidden_state = None
        
        with torch.no_grad():
            for data_point in trajectory_data:
                # Move to device
                image = data_point['image'].unsqueeze(0).to(self.device)
                desired_velocity = data_point['desired_velocity'].unsqueeze(0).to(self.device)
                quaternion = data_point['quaternion'].unsqueeze(0).to(self.device)
                velocity_command = data_point['velocity_command']
                
                # Prepare inputs
                model_inputs = [image, desired_velocity, quaternion]
                if hidden_state is not None:
                    model_inputs.append(hidden_state)

                # Forward pass
                pred, hidden_state = self.model(model_inputs)

                # Store results
                predictions.append(pred.cpu().numpy()[0])
                normalized_target = velocity_command.cpu().numpy() / desired_velocity.cpu().numpy()[0]
                targets.append(normalized_target)

        predictions = np.array(predictions)
        targets = np.array(targets)

        # Compute trajectory metrics
        metrics = self._compute_metrics(
            predictions, targets,
            np.ones((len(predictions), 1)),  # Dummy velocities
            np.mean((predictions - targets) ** 2, axis=1),
            np.mean((predictions - targets) ** 2)
        )

        return {
            'metrics': metrics,
            'predictions': predictions.tolist(),
            'targets': targets.tolist(),
            'trajectory_length': len(predictions)
        }

    def visualize_predictions(self, predictions: np.ndarray, targets: np.ndarray,
                            save_path: Optional[str] = None) -> None:
        """Visualize prediction results

        Args:
            predictions: Model predictions (N, 3)
            targets: Ground truth targets (N, 3)
            save_path: Optional path to save plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axis_names = ['X (Forward)', 'Y (Left)', 'Z (Up)']

        # Scatter plots for each axis
        for i in range(3):
            ax = axes[0, i]
            ax.scatter(targets[:, i], predictions[:, i], alpha=0.6, s=1)
            ax.plot([targets[:, i].min(), targets[:, i].max()],
                   [targets[:, i].min(), targets[:, i].max()], 'r--', lw=2)
            ax.set_xlabel(f'True {axis_names[i]}')
            ax.set_ylabel(f'Predicted {axis_names[i]}')
            ax.set_title(f'{axis_names[i]} Velocity')
            ax.grid(True, alpha=0.3)

            # Add correlation coefficient
            corr = np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Error histograms
        for i in range(3):
            ax = axes[1, i]
            errors = predictions[:, i] - targets[:, i]
            ax.hist(errors, bins=50, alpha=0.7, density=True)
            ax.set_xlabel(f'{axis_names[i]} Error')
            ax.set_ylabel('Density')
            ax.set_title(f'{axis_names[i]} Error Distribution')
            ax.grid(True, alpha=0.3)

            # Add statistics
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            ax.axvline(mean_error, color='red', linestyle='--', label=f'Mean: {mean_error:.3f}')
            ax.text(0.05, 0.95, f'μ = {mean_error:.3f}\nσ = {std_error:.3f}',
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Visualization saved to {save_path}")

        plt.show()

    def create_evaluation_report(self, data_loader: VitFlyDataLoader,
                               output_dir: str) -> Dict[str, Any]:
        """Create comprehensive evaluation report

        Args:
            data_loader: Data loader for evaluation
            output_dir: Directory to save report

        Returns:
            Dictionary with full evaluation results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Evaluate on validation set
        val_metrics = self.evaluate_dataset(data_loader, 'val')

        # Evaluate on training set (subset)
        train_metrics = self.evaluate_dataset(data_loader, 'train')

        # Create visualizations
        # Re-run evaluation to get predictions for plotting
        predictions, targets = self._get_predictions_for_visualization(data_loader.val_loader)

        # Save prediction plots
        plot_path = output_path / 'prediction_analysis.png'
        self.visualize_predictions(predictions, targets, str(plot_path))

        # Create summary report
        report = {
            'model_info': {
                'model_type': self.model_type,
                'model_path': self.model_path,
                'device': str(self.device)
            },
            'validation_metrics': val_metrics,
            'training_metrics': train_metrics,
            'data_info': data_loader.get_data_statistics()
        }

        # Save report as JSON
        report_path = output_path / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Create readable summary
        summary_path = output_path / 'evaluation_summary.txt'
        self._create_text_summary(report, summary_path)

        self.logger.info(f"Evaluation report saved to {output_path}")
        return report

    def _get_predictions_for_visualization(self, data_loader) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions and targets for visualization"""
        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].to(self.device)
                desired_velocities = batch['desired_velocity'].to(self.device)
                quaternions = batch['quaternion'].to(self.device)
                velocity_commands = batch['velocity_command'].to(self.device)

                model_inputs = [images, desired_velocities, quaternions]
                pred, _ = self.model(model_inputs)

                normalized_targets = velocity_commands / desired_velocities.unsqueeze(-1)

                predictions.append(pred.cpu().numpy())
                targets.append(normalized_targets.cpu().numpy())

        return np.concatenate(predictions), np.concatenate(targets)

    def _create_text_summary(self, report: Dict[str, Any], summary_path: Path):
        """Create human-readable summary"""
        with open(summary_path, 'w') as f:
            f.write("VitFly Model Evaluation Summary\n")
            f.write("=" * 40 + "\n\n")

            # Model info
            f.write(f"Model Type: {report['model_info']['model_type']}\n")
            f.write(f"Model Path: {report['model_info']['model_path']}\n")
            f.write(f"Device: {report['model_info']['device']}\n\n")

            # Validation metrics
            val_metrics = report['validation_metrics']
            f.write("Validation Metrics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"MSE Loss: {val_metrics['mse']:.6f}\n")
            f.write(f"RMSE: {val_metrics['rmse']:.6f}\n")
            f.write(f"MAE: {val_metrics['mae']:.6f}\n")
            f.write(f"Direction Error: {val_metrics['direction_error_deg']:.2f} degrees\n")
            f.write(f"Speed Error: {val_metrics['speed_error']:.6f}\n\n")

            # Per-axis metrics
            f.write("Per-Axis RMSE:\n")
            f.write(f"  X (Forward): {val_metrics['rmse_per_axis'][0]:.6f}\n")
            f.write(f"  Y (Left): {val_metrics['rmse_per_axis'][1]:.6f}\n")
            f.write(f"  Z (Up): {val_metrics['rmse_per_axis'][2]:.6f}\n\n")

            # Correlations
            f.write("Correlations:\n")
            f.write(f"  X: {val_metrics['correlations'][0]:.3f}\n")
            f.write(f"  Y: {val_metrics['correlations'][1]:.3f}\n")
            f.write(f"  Z: {val_metrics['correlations'][2]:.3f}\n\n")

            # Data info
            data_info = report['data_info']
            f.write("Dataset Information:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Training Samples: {data_info['train_size']}\n")
            f.write(f"Validation Samples: {data_info['val_size']}\n")
            f.write(f"Number of Trajectories: {data_info['num_trajectories_train'] + data_info['num_trajectories_val']}\n")


if __name__ == '__main__':
    # Example evaluation
    logging.basicConfig(level=logging.INFO)

    # Configuration
    model_path = "outputs/vitfly_training_20240101_120000/checkpoints/best_model.pth"
    model_type = "ViTLSTM"
    data_dir = "data/training_data"
    output_dir = "evaluation_results"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        exit(1)

    # Create evaluator
    evaluator = ModelEvaluator(model_path, model_type)

    # Create data loader
    data_loader = VitFlyDataLoader(
        data_dir=data_dir,
        batch_size=32,
        val_split=0.2,
        short=0  # Use all data
    )

    # Run evaluation
    report = evaluator.create_evaluation_report(data_loader, output_dir)

    print("Evaluation completed!")
    print(f"Results saved to: {output_dir}")