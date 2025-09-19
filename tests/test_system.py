#!/usr/bin/env python3
"""
System Integration Test for VitFly-AirSim

This script tests the complete system integration including
AirSim interface, data processing, and model inference.
"""

import sys
import os
import torch
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from models import ViTLSTM
from training.data_loader import VitFlyDataset, VitFlyDataLoader
from training.trainer import VitFlyTrainer
from training.evaluator import ModelEvaluator
from inference.model_inference import ModelInference


def create_dummy_data(data_dir: Path, num_trajectories: int = 3, traj_length: int = 10):
    """Create dummy training data for testing"""
    print(f"Creating dummy data in {data_dir}...")
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for traj_idx in range(num_trajectories):
        traj_name = f"trajectory_{traj_idx:06d}"
        traj_dir = data_dir / traj_name
        traj_dir.mkdir(exist_ok=True)
        
        # Create metadata
        metadata = []
        
        for frame_idx in range(traj_length):
            timestamp = time.time() + frame_idx * 0.1
            timestamp_ns = int(timestamp * 1e9)
            
            metadata.append({
                'timestamp': timestamp,
                'timestamp_ns': timestamp_ns,
                'desired_velocity': 5.0,
                'orientation_w': 1.0,
                'orientation_x': 0.0,
                'orientation_y': 0.0,
                'orientation_z': 0.0,
                'position_x': frame_idx * 0.5,
                'position_y': 0.0,
                'position_z': -5.0,
                'velocity_x': 2.0,
                'velocity_y': 0.1 * np.sin(frame_idx),
                'velocity_z': 0.0,
                'cmd_velocity_x': 1.0,
                'cmd_velocity_y': 0.1 * np.sin(frame_idx),
                'cmd_velocity_z': 0.0,
                'collective_thrust': 15.0,
                'body_rate_x': 0.0,
                'body_rate_y': 0.0,
                'body_rate_z': 0.0,
                'collision': False
            })
            
            # Create dummy depth image (60x90)
            depth_image = np.random.rand(60, 90).astype(np.float32)
            depth_uint16 = (depth_image * 65535).astype(np.uint16)
            
            import cv2
            image_path = traj_dir / f"{timestamp_ns}.png"
            cv2.imwrite(str(image_path), depth_uint16)
        
        # Save metadata
        import pandas as pd
        df = pd.DataFrame(metadata)
        df.to_csv(traj_dir / "data.csv", index=False)
    
    print(f"Created {num_trajectories} trajectories with {traj_length} frames each")


def test_data_loading():
    """Test data loading functionality"""
    print("\nTesting data loading...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / "test_data"
        create_dummy_data(data_dir)
        
        try:
            # Test dataset creation
            dataset = VitFlyDataset(str(data_dir), val_split=0.3, is_train=True)
            
            if len(dataset) == 0:
                print("  ✗ Dataset is empty")
                return False
            
            # Test data loading
            sample = dataset[0]
            required_keys = ['image', 'desired_velocity', 'quaternion', 'velocity_command']
            
            for key in required_keys:
                if key not in sample:
                    print(f"  ✗ Missing key in sample: {key}")
                    return False
            
            # Test shapes
            if sample['image'].shape != (1, 60, 90):
                print(f"  ✗ Wrong image shape: {sample['image'].shape}")
                return False
            
            print(f"  ✓ Data loading successful ({len(dataset)} samples)")
            
            # Test data loader
            data_loader = VitFlyDataLoader(str(data_dir), batch_size=2, short=2)
            
            for batch in data_loader.train_loader:
                if batch['image'].shape[0] != 2:
                    print(f"  ✗ Wrong batch size: {batch['image'].shape[0]}")
                    return False
                break
            
            print("  ✓ Data loader working correctly")
            return True
            
        except Exception as e:
            print(f"  ✗ Data loading failed: {e}")
            return False


def test_training_pipeline():
    """Test training pipeline"""
    print("\nTesting training pipeline...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / "training_data"
        output_dir = Path(temp_dir) / "outputs"
        
        create_dummy_data(data_dir, num_trajectories=5, traj_length=8)
        
        try:
            # Create minimal training config
            config = {
                'model_type': 'ViTLSTM',
                'data_dir': str(data_dir),
                'output_dir': str(output_dir),
                'batch_size': 2,
                'learning_rate': 1e-3,
                'num_epochs': 2,  # Very short for testing
                'val_split': 0.3,
                'device': 'cpu',  # Force CPU for testing
                'save_interval': 1,
                'val_interval': 1,
                'short': 3,  # Use only 3 trajectories
                'num_workers': 0  # Avoid multiprocessing issues in tests
            }
            
            # Create trainer
            trainer = VitFlyTrainer(config)
            
            # Check that model was created
            if trainer.model is None:
                print("  ✗ Model not created")
                return False
            
            # Test one training step
            trainer.current_epoch = 0
            train_metrics = trainer.train_epoch()
            
            if 'loss' not in train_metrics:
                print("  ✗ Training metrics missing loss")
                return False
            
            print(f"  ✓ Training step successful (loss: {train_metrics['loss']:.6f})")
            
            # Test validation
            val_metrics = trainer.validate()
            
            if 'loss' not in val_metrics:
                print("  ✗ Validation metrics missing loss")
                return False
            
            print(f"  ✓ Validation successful (loss: {val_metrics['loss']:.6f})")
            
            # Test model saving
            trainer.save_checkpoint()
            
            checkpoint_dir = trainer.workspace / 'checkpoints'
            if not any(checkpoint_dir.glob('*.pth')):
                print("  ✗ No checkpoint files saved")
                return False
            
            print("  ✓ Model saving successful")
            return True
            
        except Exception as e:
            print(f"  ✗ Training pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_model_inference():
    """Test model inference"""
    print("\nTesting model inference...")
    
    try:
        # Create a model and save it
        model = ViTLSTM()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pth"
            torch.save(model.state_dict(), model_path)
            
            # Test model inference
            inference = ModelInference(str(model_path), 'ViTLSTM', 'cpu')
            
            # Test single prediction
            dummy_depth = np.random.rand(60, 90).astype(np.float32)
            velocity_cmd = inference.predict_velocity(
                dummy_depth, 5.0, (1.0, 0.0, 0.0, 0.0)
            )
            
            if len(velocity_cmd) != 3:
                print(f"  ✗ Wrong velocity command length: {len(velocity_cmd)}")
                return False
            
            # Check if output is reasonable
            speed = np.linalg.norm(velocity_cmd)
            if speed > 20.0 or speed == 0.0:  # Sanity check
                print(f"  ✗ Unreasonable velocity command: {velocity_cmd}")
                return False
            
            print(f"  ✓ Model inference successful (velocity: {velocity_cmd})")
            
            # Test performance
            stats = inference.get_performance_stats()
            if 'total_inferences' not in stats:
                print("  ✗ Performance stats incomplete")
                return False
            
            print(f"  ✓ Performance monitoring working")
            
            # Test benchmark
            benchmark = inference.benchmark_inference(10)
            if benchmark['num_iterations'] != 10:
                print("  ✗ Benchmark failed")
                return False
            
            print(f"  ✓ Benchmark successful ({benchmark['average_time_ms']:.2f}ms avg)")
            return True
            
    except Exception as e:
        print(f"  ✗ Model inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation():
    """Test model evaluation"""
    print("\nTesting model evaluation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / "eval_data"
        model_path = Path(temp_dir) / "eval_model.pth"
        
        create_dummy_data(data_dir, num_trajectories=4, traj_length=6)
        
        try:
            # Create and save a model
            model = ViTLSTM()
            torch.save(model.state_dict(), model_path)
            
            # Create evaluator
            evaluator = ModelEvaluator(str(model_path), 'ViTLSTM', 'cpu')
            
            # Create data loader
            data_loader = VitFlyDataLoader(
                str(data_dir), batch_size=2, short=3, num_workers=0
            )
            
            # Test evaluation
            metrics = evaluator.evaluate_dataset(data_loader, 'val')
            
            required_metrics = ['loss', 'mse', 'rmse', 'mae']
            for metric in required_metrics:
                if metric not in metrics:
                    print(f"  ✗ Missing metric: {metric}")
                    return False
            
            print(f"  ✓ Evaluation successful (RMSE: {metrics['rmse']:.6f})")
            
            # Test report generation
            report_dir = Path(temp_dir) / "evaluation_report"
            report = evaluator.create_evaluation_report(data_loader, str(report_dir))
            
            if not (report_dir / 'evaluation_report.json').exists():
                print("  ✗ Evaluation report not created")
                return False
            
            print("  ✓ Evaluation report generated successfully")
            return True
            
        except Exception as e:
            print(f"  ✗ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_end_to_end_workflow():
    """Test complete end-to-end workflow"""
    print("\nTesting end-to-end workflow...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        data_dir = base_dir / "data"
        output_dir = base_dir / "outputs"
        
        create_dummy_data(data_dir, num_trajectories=6, traj_length=10)
        
        try:
            # Step 1: Train a model
            config = {
                'model_type': 'ConvNet',  # Use simpler model for speed
                'data_dir': str(data_dir),
                'output_dir': str(output_dir),
                'batch_size': 2,
                'learning_rate': 1e-3,
                'num_epochs': 3,
                'val_split': 0.3,
                'device': 'cpu',
                'save_interval': 1,
                'val_interval': 1,
                'short': 4,
                'num_workers': 0
            }
            
            trainer = VitFlyTrainer(config)
            trainer.train()
            
            # Check that model was saved
            latest_model = trainer.workspace / 'checkpoints' / 'latest_model.pth'
            if not latest_model.exists():
                print("  ✗ Trained model not found")
                return False
            
            print("  ✓ Step 1: Model training completed")
            
            # Step 2: Test inference with trained model
            inference = ModelInference(str(latest_model), 'ConvNet', 'cpu')
            
            dummy_depth = np.random.rand(60, 90).astype(np.float32)
            velocity_cmd = inference.predict_velocity(
                dummy_depth, 5.0, (1.0, 0.0, 0.0, 0.0)
            )
            
            print(f"  ✓ Step 2: Inference working (velocity: {velocity_cmd})")
            
            # Step 3: Evaluate the model
            data_loader = VitFlyDataLoader(
                str(data_dir), batch_size=2, short=3, num_workers=0
            )
            
            evaluator = ModelEvaluator(str(latest_model), 'ConvNet', 'cpu')
            metrics = evaluator.evaluate_dataset(data_loader, 'val')
            
            print(f"  ✓ Step 3: Evaluation completed (RMSE: {metrics['rmse']:.6f})")
            
            print("  🎉 End-to-end workflow successful!")
            return True
            
        except Exception as e:
            print(f"  ✗ End-to-end workflow failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Run all system tests"""
    print("=" * 60)
    print("VitFly-AirSim System Integration Tests")
    print("=" * 60)
    
    tests = [
        test_data_loading,
        test_training_pipeline,
        test_model_inference,
        test_evaluation,
        test_end_to_end_workflow
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"  FAILED: {test.__name__}")
        except Exception as e:
            print(f"  ERROR in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"System Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All system tests passed!")
        print("\nVitFly-AirSim is ready for use!")
        return True
    else:
        print("❌ Some system tests failed!")
        print("\nPlease check the failed tests and fix any issues.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)