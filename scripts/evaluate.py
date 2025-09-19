#!/usr/bin/env python3
"""
Evaluation Script for VitFly-AirSim

This script provides model evaluation functionality for trained VitFly models.

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

from training.evaluator import ModelEvaluator
from training.data_loader import VitFlyDataLoader


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate VitFly models')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model-type', type=str, required=True,
                       choices=['ViT', 'ViTLSTM', 'ConvNet', 'LSTMNet', 'UNet'],
                       help='Type of model to evaluate')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to evaluation data directory')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to run evaluation on')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--short', type=int, default=0,
                       help='Limit number of trajectories for testing')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating visualization plots')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Create evaluator
        logger.info(f"Loading {args.model_type} model from: {args.model_path}")
        evaluator = ModelEvaluator(args.model_path, args.model_type, args.device)
        
        # Create data loader
        logger.info(f"Loading evaluation data from: {args.data_dir}")
        data_loader = VitFlyDataLoader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            val_split=args.val_split,
            num_workers=args.num_workers,
            short=args.short
        )
        
        # Run evaluation
        logger.info("Running model evaluation...")
        report = evaluator.create_evaluation_report(data_loader, args.output_dir)
        
        # Print summary
        val_metrics = report['validation_metrics']
        logger.info("Evaluation Results Summary:")
        logger.info(f"  Validation MSE: {val_metrics['mse']:.6f}")
        logger.info(f"  Validation RMSE: {val_metrics['rmse']:.6f}")
        logger.info(f"  Validation MAE: {val_metrics['mae']:.6f}")
        logger.info(f"  Direction Error: {val_metrics['direction_error_deg']:.2f} degrees")
        logger.info(f"  Speed Error: {val_metrics['speed_error']:.6f}")
        logger.info(f"  Per-axis RMSE: {val_metrics['rmse_per_axis']}")
        logger.info(f"  Correlations: {val_metrics['correlations']}")
        
        logger.info(f"Detailed results saved to: {args.output_dir}")
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()