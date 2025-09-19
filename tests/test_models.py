#!/usr/bin/env python3
"""
Test script for VitFly models

This script tests all the VitFly models to ensure they work correctly
on Windows with the new architecture.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from models import ViT, ViTLSTM, ConvNet, LSTMNet, UNetConvLSTMNet


def test_model_creation():
    """Test that all models can be created successfully"""
    print("Testing model creation...")
    
    models = {
        'ViT': ViT,
        'ViTLSTM': ViTLSTM,
        'ConvNet': ConvNet,
        'LSTMNet': LSTMNet,
        'UNet': UNetConvLSTMNet
    }
    
    for name, model_class in models.items():
        try:
            model = model_class()
            print(f"  ✓ {name} created successfully")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            return False
    
    return True


def test_model_forward_pass():
    """Test forward pass for all models"""
    print("\nTesting model forward pass...")
    
    # Create test inputs
    batch_size = 2
    depth_image = torch.randn(batch_size, 1, 60, 90)
    desired_velocity = torch.randn(batch_size, 1)
    quaternion = torch.randn(batch_size, 4)
    
    models = {
        'ViT': ViT(),
        'ViTLSTM': ViTLSTM(),
        'ConvNet': ConvNet(),
        'LSTMNet': LSTMNet(),
        'UNet': UNetConvLSTMNet()
    }
    
    for name, model in models.items():
        try:
            model.eval()
            
            # Prepare inputs
            inputs = [depth_image, desired_velocity, quaternion]
            
            # Add hidden state for LSTM models
            if 'LSTM' in name or name == 'UNet':
                inputs.append(None)
            
            # Forward pass
            with torch.no_grad():
                output, hidden_state = model(inputs)
            
            # Check output shape
            expected_shape = (batch_size, 3)
            if output.shape != expected_shape:
                print(f"  ✗ {name} output shape mismatch: got {output.shape}, expected {expected_shape}")
                return False
            
            # Check output range (should be reasonable velocity commands)
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"  ✗ {name} output contains NaN or Inf")
                return False
            
            print(f"  ✓ {name} forward pass successful (output shape: {output.shape})")
            
        except Exception as e:
            print(f"  ✗ {name} forward pass failed: {e}")
            return False
    
    return True


def test_model_parameters():
    """Test model parameter counts"""
    print("\nTesting model parameter counts...")
    
    expected_params = {
        'ViT': 3_101_199,
        'ViTLSTM': 3_563_663,
        'ConvNet': 235_269,
        'LSTMNet': 2_949_937,
        'UNet': 2_955_822
    }
    
    models = {
        'ViT': ViT(),
        'ViTLSTM': ViTLSTM(),
        'ConvNet': ConvNet(),
        'LSTMNet': LSTMNet(),
        'UNet': UNetConvLSTMNet()
    }
    
    tolerance = 0.05  # 5% tolerance for parameter count differences
    
    for name, model in models.items():
        try:
            actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            expected = expected_params[name]
            
            if abs(actual_params - expected) / expected > tolerance:
                print(f"  ⚠ {name} parameter count mismatch: got {actual_params:,}, expected {expected:,}")
            else:
                print(f"  ✓ {name} parameter count OK: {actual_params:,}")
                
        except Exception as e:
            print(f"  ✗ {name} parameter counting failed: {e}")
            return False
    
    return True


def test_model_sequential_inference():
    """Test sequential inference for LSTM models"""
    print("\nTesting sequential inference for LSTM models...")
    
    # Test data
    sequence_length = 5
    depth_images = torch.randn(sequence_length, 1, 1, 60, 90)
    desired_velocities = torch.ones(sequence_length, 1, 1) * 5.0
    quaternions = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(sequence_length, 1, 1)
    
    lstm_models = {
        'ViTLSTM': ViTLSTM(),
        'LSTMNet': LSTMNet(),
        'UNet': UNetConvLSTMNet()
    }
    
    for name, model in lstm_models.items():
        try:
            model.eval()
            hidden_state = None
            outputs = []
            
            with torch.no_grad():
                for i in range(sequence_length):
                    inputs = [
                        depth_images[i],
                        desired_velocities[i],
                        quaternions[i],
                        hidden_state
                    ]
                    
                    output, hidden_state = model(inputs)
                    outputs.append(output)
            
            # Check that outputs are different (showing memory)
            outputs = torch.stack(outputs, dim=0)
            
            if torch.allclose(outputs[0], outputs[-1], atol=1e-3):
                print(f"  ⚠ {name} outputs too similar (may not be using memory)")
            else:
                print(f"  ✓ {name} sequential inference working (outputs vary)")
                
        except Exception as e:
            print(f"  ✗ {name} sequential inference failed: {e}")
            return False
    
    return True


def test_model_device_compatibility():
    """Test model device compatibility"""
    print("\nTesting device compatibility...")
    
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    model = ViT()  # Test with one model
    
    for device in devices:
        try:
            model_dev = model.to(device)
            
            # Test input
            depth_image = torch.randn(1, 1, 60, 90).to(device)
            desired_velocity = torch.ones(1, 1).to(device)
            quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).to(device)
            
            inputs = [depth_image, desired_velocity, quaternion]
            
            with torch.no_grad():
                output, _ = model_dev(inputs)
            
            assert output.device.type == device
            print(f"  ✓ Device {device} compatibility OK")
            
        except Exception as e:
            print(f"  ✗ Device {device} compatibility failed: {e}")
            return False
    
    return True


def test_input_shapes():
    """Test various input shapes"""
    print("\nTesting input shape flexibility...")
    
    model = ViT()
    model.eval()
    
    test_cases = [
        # (batch_size, height, width)
        (1, 60, 90),   # Standard
        (2, 60, 90),   # Batch
        (1, 120, 180), # Different resolution
        (4, 30, 45),   # Smaller resolution
    ]
    
    for batch_size, height, width in test_cases:
        try:
            depth_image = torch.randn(batch_size, 1, height, width)
            desired_velocity = torch.ones(batch_size, 1)
            quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(batch_size, 1)
            
            inputs = [depth_image, desired_velocity, quaternion]
            
            with torch.no_grad():
                output, _ = model(inputs)
            
            expected_shape = (batch_size, 3)
            assert output.shape == expected_shape
            
            print(f"  ✓ Input shape ({batch_size}, {height}, {width}) OK")
            
        except Exception as e:
            print(f"  ✗ Input shape ({batch_size}, {height}, {width}) failed: {e}")
            return False
    
    return True


def main():
    """Run all model tests"""
    print("=" * 50)
    print("VitFly-AirSim Model Tests")
    print("=" * 50)
    
    tests = [
        test_model_creation,
        test_model_forward_pass,
        test_model_parameters,
        test_model_sequential_inference,
        test_model_device_compatibility,
        test_input_shapes
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
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All model tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)