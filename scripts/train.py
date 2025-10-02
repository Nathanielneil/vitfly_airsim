#!/usr/bin/env python3
"""
Training Script for VitFly-AirSim

This script provides the main entry point for training VitFly models
on Windows with AirSim data.

Author: Adapted from original VitFly project
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Direct import to avoid __init__.py relative import issues
from training.trainer import VitFlyTrainer


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                import json
                config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)


def validate_config(config: dict) -> dict:
    """Validate and set default values for configuration"""
    
    # Required fields
    required_fields = ['data_dir', 'model_type']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required config field missing: {field}")
    
    # Check if data directory exists
    if not os.path.exists(config['data_dir']):
        raise ValueError(f"Data directory not found: {config['data_dir']}")
    
    # Set default values
    defaults = {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'val_split': 0.2,
        'device': 'auto',
        'output_dir': 'outputs',
        'save_interval': 10,
        'val_interval': 1,
        'log_interval': 50,
        'tensorboard_interval': 25,
        'num_workers': 4,
        'target_height': 60,
        'target_width': 90,
        'grad_clip': 1.0,
        'weight_decay': 0.0,
        'use_scheduler': True,
        'scheduler_type': 'cosine',
        'optimizer': 'adam',
        'short': 0,
        'seed': None
    }
    
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
    
    # Handle device selection
    if config['device'] == 'auto':
        import torch
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return config


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train VitFly models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--data-dir', type=str,
                       help='Override data directory from config')
    parser.add_argument('--model-type', type=str,
                       choices=['ViT', 'ViTLSTM', 'ConvNet', 'LSTMNet', 'UNet'],
                       help='Override model type from config')
    parser.add_argument('--epochs', type=int,
                       help='Override number of epochs from config')
    parser.add_argument('--batch-size', type=int,
                       help='Override batch size from config')
    parser.add_argument('--lr', type=float,
                       help='Override learning rate from config')
    parser.add_argument('--device', type=str,
                       choices=['auto', 'cpu', 'cuda'],
                       help='Override device from config')
    parser.add_argument('--output-dir', type=str,
                       help='Override output directory from config')
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print configuration and exit without training')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    overrides = {}
    if args.data_dir:
        overrides['data_dir'] = args.data_dir
    if args.model_type:
        overrides['model_type'] = args.model_type
    if args.epochs:
        overrides['num_epochs'] = args.epochs
    if args.batch_size:
        overrides['batch_size'] = args.batch_size
    if args.lr:
        overrides['learning_rate'] = args.lr
    if args.device:
        overrides['device'] = args.device
    if args.output_dir:
        overrides['output_dir'] = args.output_dir
    if args.resume:
        overrides['load_checkpoint'] = True
        overrides['checkpoint_path'] = args.resume
    
    config.update(overrides)
    
    # Validate configuration
    try:
        config = validate_config(config)
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    # Print configuration
    logger.info("Training configuration:")
    for key, value in sorted(config.items()):
        logger.info(f"  {key}: {value}")
    
    if args.dry_run:
        logger.info("Dry run mode - exiting without training")
        return
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    try:
        # Create trainer
        logger.info("Creating trainer...")
        trainer = VitFlyTrainer(config)
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Export final model
        export_path = Path(config['output_dir']) / 'final_model.pth'
        trainer.export_model(str(export_path))
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()