"""
Trainer for VitFly-AirSim Models

This module provides the training functionality for VitFly models,
adapted from the original project for Windows and AirSim compatibility.

Author: Adapted from original VitFly project
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

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


class VitFlyTrainer:
    """Trainer for VitFly models"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.logger.info(f"Using device: {self.device}")
        
        # Create workspace
        self._setup_workspace()
        
        # Initialize model
        self.model = self._create_model()
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup data loaders
        self.data_loader = self._create_data_loader()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.workspace / 'tensorboard')
        
        # Load checkpoint if specified
        if config.get('load_checkpoint', False):
            self._load_checkpoint()
    
    def _setup_workspace(self):
        """Setup workspace directory"""
        from datetime import datetime
        
        # Create workspace name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        workspace_name = f"vitfly_training_{timestamp}"
        
        if 'workspace_suffix' in self.config:
            workspace_name += f"_{self.config['workspace_suffix']}"
        
        # Create workspace directory
        base_dir = Path(self.config.get('output_dir', 'outputs'))
        self.workspace = base_dir / workspace_name
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.workspace / 'checkpoints').mkdir(exist_ok=True)
        (self.workspace / 'logs').mkdir(exist_ok=True)
        
        # Save config
        import json
        with open(self.workspace / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Setup logging to file
        log_file = self.workspace / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Workspace created: {self.workspace}")
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration"""
        model_type = self.config.get('model_type', 'ViTLSTM')
        
        if model_type == 'ViT':
            model = ViT()
        elif model_type == 'ViTLSTM':
            model = ViTLSTM()
        elif model_type == 'ConvNet':
            model = ConvNet()
        elif model_type == 'LSTMNet':
            model = LSTMNet()
        elif model_type == 'UNet':
            model = UNetConvLSTMNet()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model.to(self.device)
        
        # Log model info
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Created {model_type} model with {num_params:,} parameters")
        
        return model
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer"""
        optimizer_type = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 0.0)
        
        if optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        self.logger.info(f"Created {optimizer_type} optimizer with lr={lr}")
        return optimizer
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        if not self.config.get('use_scheduler', False):
            return None
        
        scheduler_type = self.config.get('scheduler_type', 'cosine')
        
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('num_epochs', 100),
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('scheduler_step_size', 30),
                gamma=self.config.get('scheduler_gamma', 0.1)
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        
        self.logger.info(f"Created {scheduler_type} scheduler")
        return scheduler
    
    def _create_data_loader(self) -> VitFlyDataLoader:
        """Create data loader"""
        data_config = {
            'batch_size': self.config.get('batch_size', 32),
            'val_split': self.config.get('val_split', 0.2),
            'num_workers': self.config.get('num_workers', 4),
            'target_height': self.config.get('target_height', 60),
            'target_width': self.config.get('target_width', 90),
            'short': self.config.get('short', 0),
            'seed': self.config.get('seed', None)
        }
        
        data_loader = VitFlyDataLoader(
            data_dir=self.config['data_dir'],
            **data_config
        )
        
        # Log data statistics
        stats = data_loader.get_data_statistics()
        self.logger.info(f"Data loaded: {stats['train_size']} train, {stats['val_size']} val samples")
        
        return data_loader
    
    def _load_checkpoint(self):
        """Load model from checkpoint"""
        checkpoint_path = self.config.get('checkpoint_path')
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint path not found: {checkpoint_path}")
            return
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                # Full checkpoint with metadata
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'epoch' in checkpoint:
                    self.current_epoch = checkpoint['epoch']
                if 'global_step' in checkpoint:
                    self.global_step = checkpoint['global_step']
                if 'best_val_loss' in checkpoint:
                    self.best_val_loss = checkpoint['best_val_loss']
            else:
                # Just model state dict
                self.model.load_state_dict(checkpoint)
            
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.data_loader.train_loader):
            # Move data to device
            images = batch['image'].to(self.device)
            desired_velocities = batch['desired_velocity'].to(self.device)
            quaternions = batch['quaternion'].to(self.device)
            velocity_commands = batch['velocity_command'].to(self.device)
            
            # Prepare model inputs
            model_inputs = [images, desired_velocities, quaternions]
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions, hidden_state = self.model(model_inputs)
            
            # Normalize targets by desired velocity (like original implementation)
            normalized_targets = velocity_commands / desired_velocities.unsqueeze(-1)
            
            # Compute loss
            loss = F.mse_loss(predictions, normalized_targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log progress
            if batch_idx % self.config.get('log_interval', 100) == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.data_loader.train_loader)}, "
                    f"Loss: {loss.item():.6f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
            
            # Tensorboard logging
            if self.global_step % self.config.get('tensorboard_interval', 50) == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_time = time.time() - start_time
        
        return {
            'loss': avg_loss,
            'time': epoch_time,
            'num_batches': num_batches
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.data_loader.val_loader:
                # Move data to device
                images = batch['image'].to(self.device)
                desired_velocities = batch['desired_velocity'].to(self.device)
                quaternions = batch['quaternion'].to(self.device)
                velocity_commands = batch['velocity_command'].to(self.device)
                
                # Prepare model inputs
                model_inputs = [images, desired_velocities, quaternions]
                
                # Forward pass
                predictions, hidden_state = self.model(model_inputs)
                
                # Normalize targets
                normalized_targets = velocity_commands / desired_velocities.unsqueeze(-1)
                
                # Compute loss
                loss = F.mse_loss(predictions, normalized_targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'loss': avg_loss}
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.workspace / 'checkpoints' / f'checkpoint_epoch_{self.current_epoch:04d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.workspace / 'checkpoints' / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with validation loss: {self.best_val_loss:.6f}")
        
        # Save latest checkpoint
        latest_path = self.workspace / 'checkpoints' / 'latest_model.pth'
        torch.save(checkpoint, latest_path)
    
    def train(self):
        """Main training loop"""
        num_epochs = self.config.get('num_epochs', 100)
        val_interval = self.config.get('val_interval', 1)
        save_interval = self.config.get('save_interval', 10)
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                
                # Train epoch
                train_metrics = self.train_epoch()
                
                # Validation
                if epoch % val_interval == 0:
                    val_metrics = self.validate()
                    
                    # Check if best model
                    is_best = val_metrics['loss'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics['loss']
                    
                    # Logging
                    self.logger.info(
                        f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.6f}, "
                        f"Val Loss: {val_metrics['loss']:.6f}, "
                        f"Time: {train_metrics['time']:.2f}s"
                    )
                    
                    # Tensorboard
                    self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                    self.writer.add_scalar('train/epoch_loss', train_metrics['loss'], epoch)
                    
                    # Save checkpoint
                    if epoch % save_interval == 0 or is_best:
                        self.save_checkpoint(is_best)
                
                # Update scheduler
                if self.scheduler is not None:
                    self.scheduler.step()
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            raise
        
        finally:
            # Save final checkpoint
            self.save_checkpoint()
            
            total_time = time.time() - start_time
            self.logger.info(f"Training completed in {total_time:.2f}s")
            
            # Close tensorboard writer
            self.writer.close()
    
    def export_model(self, export_path: str):
        """Export trained model for inference
        
        Args:
            export_path: Path to save exported model
        """
        try:
            # Save just the model state dict for inference
            torch.save(self.model.state_dict(), export_path)
            self.logger.info(f"Model exported to: {export_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting model: {e}")


def create_trainer_from_config(config_path: str) -> VitFlyTrainer:
    """Create trainer from configuration file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Initialized trainer
    """
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return VitFlyTrainer(config)


if __name__ == '__main__':
    # Example training configuration
    config = {
        'model_type': 'ViTLSTM',
        'data_dir': 'data/training_data',
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'val_split': 0.2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'output_dir': 'outputs',
        'save_interval': 10,
        'val_interval': 1,
        'log_interval': 50,
        'tensorboard_interval': 25,
        'grad_clip': 1.0,
        'use_scheduler': True,
        'scheduler_type': 'cosine'
    }
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run trainer
    trainer = VitFlyTrainer(config)
    trainer.train()