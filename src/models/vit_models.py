"""
Vision Transformer Models for VitFly-AirSim

This module contains the Vision Transformer models adapted for quadrotor
obstacle avoidance, including both standalone ViT and ViT+LSTM variants.

Author: Adapted from original VitFly project (GRASP Lab, UPenn)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from .vit_submodules import MixTransformerEncoderLayer


def refine_inputs(X, target_height=60, target_width=90):
    """Refine and normalize input data for model inference
    
    Args:
        X: List containing [depth_image, desired_velocity, quaternion]
        target_height: Target height for depth image resizing
        target_width: Target width for depth image resizing
        
    Returns:
        Refined input list with proper shapes and default values
    """
    # Fill quaternion rotation if not given
    if len(X) < 3 or X[2] is None:
        # Default quaternion [w, x, y, z] = [1, 0, 0, 0] for identity rotation
        X = list(X)  # Make mutable copy
        while len(X) < 3:
            X.append(None)
        X[2] = torch.zeros((X[0].shape[0], 4)).float().to(X[0].device)
        X[2][:, 0] = 1.0  # Set w component to 1

    # Resize depth images if not of right shape
    if X[0].shape[-2] != target_height or X[0].shape[-1] != target_width:
        X[0] = F.interpolate(
            X[0], size=(target_height, target_width), 
            mode='bilinear', align_corners=False
        )

    return X


class ViT(nn.Module):
    """Vision Transformer + FC Network for quadrotor control
    
    This model uses Vision Transformer architecture to process depth images
    and outputs velocity commands for obstacle avoidance.
    
    Parameters: 3,101,199
    """
    
    def __init__(self, input_height=60, input_width=90):
        super().__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        
        # Vision Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(
                in_channels=1, out_channels=32, 
                patch_size=7, stride=4, padding=3, 
                n_layers=2, reduction_ratio=8, 
                num_heads=1, expansion_factor=8
            ),
            MixTransformerEncoderLayer(
                in_channels=32, out_channels=64, 
                patch_size=3, stride=2, padding=1, 
                n_layers=2, reduction_ratio=4, 
                num_heads=2, expansion_factor=8
            )
        ])
        
        # Feature processing layers
        self.decoder = nn.Linear(4608, 512)
        self.fc1 = spectral_norm(nn.Linear(517, 256))  # 512 + 1 + 4 = 517
        self.fc2 = spectral_norm(nn.Linear(256, 3))
        
        # Upsampling and processing layers
        self.upsample = nn.Upsample(
            size=(16, 24), mode='bilinear', align_corners=True
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.downsample = nn.Conv2d(48, 12, 3, padding=1)

    def forward(self, X):
        """Forward pass through ViT model
        
        Args:
            X: List containing [depth_image, desired_velocity, quaternion]
            
        Returns:
            tuple: (velocity_commands, None) - None for compatibility with LSTM models
        """
        X = refine_inputs(X, self.input_height, self.input_width)
        
        depth_image = X[0]  # Shape: (B, 1, H, W)
        desired_vel = X[1]  # Shape: (B, 1)
        quaternion = X[2]   # Shape: (B, 4)
        
        # Process through ViT encoder blocks
        embeddings = [depth_image]
        for block in self.encoder_blocks:
            embeddings.append(block(embeddings[-1]))
        
        # Combine multi-scale features
        features = embeddings[1:]  # Skip input image
        # Upsample smaller features and combine
        combined = torch.cat([
            self.pixel_shuffle(features[1]),  # Upsample 64-channel features
            self.upsample(features[0])        # Upsample 32-channel features
        ], dim=1)
        
        # Process combined features
        combined = self.downsample(combined)
        flattened = self.decoder(combined.flatten(1))
        
        # Concatenate with metadata (desired velocity and quaternion)
        metadata = torch.cat([desired_vel / 10.0, quaternion], dim=1).float()
        full_input = torch.cat([flattened, metadata], dim=1).float()
        
        # Final prediction layers
        x = F.leaky_relu(self.fc1(full_input))
        velocity_commands = self.fc2(x)
        
        return velocity_commands, None


class ViTLSTM(nn.Module):
    """Vision Transformer + LSTM Network for quadrotor control
    
    This is the best performing model combining ViT for spatial processing
    with LSTM for temporal sequence modeling.
    
    Parameters: 3,563,663
    """
    
    def __init__(self, input_height=60, input_width=90):
        super().__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        
        # Vision Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(
                in_channels=1, out_channels=32, 
                patch_size=7, stride=4, padding=3, 
                n_layers=2, reduction_ratio=8, 
                num_heads=1, expansion_factor=8
            ),
            MixTransformerEncoderLayer(
                in_channels=32, out_channels=64, 
                patch_size=3, stride=2, padding=1, 
                n_layers=2, reduction_ratio=4, 
                num_heads=2, expansion_factor=8
            )
        ])
        
        # Feature processing layers
        self.decoder = spectral_norm(nn.Linear(4608, 512))
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=517,     # 512 + 1 + 4 = 517
            hidden_size=128,
            num_layers=3,
            dropout=0.1,
            batch_first=True
        )
        
        # Output layer
        self.output_fc = spectral_norm(nn.Linear(128, 3))
        
        # Upsampling and processing layers
        self.upsample = nn.Upsample(
            size=(16, 24), mode='bilinear', align_corners=True
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.downsample = nn.Conv2d(48, 12, 3, padding=1)

    def forward(self, X):
        """Forward pass through ViT+LSTM model
        
        Args:
            X: List containing [depth_image, desired_velocity, quaternion, hidden_state]
            
        Returns:
            tuple: (velocity_commands, new_hidden_state)
        """
        X = refine_inputs(X, self.input_height, self.input_width)
        
        depth_image = X[0]  # Shape: (B, 1, H, W)
        desired_vel = X[1]  # Shape: (B, 1)
        quaternion = X[2]   # Shape: (B, 4)
        
        # Get hidden state if provided
        hidden_state = X[3] if len(X) > 3 and X[3] is not None else None
        
        # Process through ViT encoder blocks
        embeddings = [depth_image]
        for block in self.encoder_blocks:
            embeddings.append(block(embeddings[-1]))
        
        # Combine multi-scale features
        features = embeddings[1:]  # Skip input image
        combined = torch.cat([
            self.pixel_shuffle(features[1]),  # Upsample 64-channel features
            self.upsample(features[0])        # Upsample 32-channel features
        ], dim=1)
        
        # Process combined features
        combined = self.downsample(combined)
        flattened = self.decoder(combined.flatten(1))
        
        # Concatenate with metadata
        metadata = torch.cat([desired_vel / 10.0, quaternion], dim=1).float()
        lstm_input = torch.cat([flattened, metadata], dim=1).float()
        
        # Add sequence dimension for LSTM (batch_first=True)
        lstm_input = lstm_input.unsqueeze(1)  # (B, 1, features)
        
        # LSTM processing
        if hidden_state is not None:
            lstm_output, new_hidden_state = self.lstm(lstm_input, hidden_state)
        else:
            lstm_output, new_hidden_state = self.lstm(lstm_input)
        
        # Remove sequence dimension and get final output
        lstm_output = lstm_output.squeeze(1)  # (B, hidden_size)
        velocity_commands = self.output_fc(lstm_output)
        
        return velocity_commands, new_hidden_state


def get_model_info():
    """Get information about available models and their parameter counts"""
    
    models_info = {
        'ViT': {
            'class': ViT,
            'params': 3_101_199,
            'description': 'Vision Transformer with fully connected output'
        },
        'ViTLSTM': {
            'class': ViTLSTM,
            'params': 3_563_663,
            'description': 'Vision Transformer with LSTM temporal modeling (best performance)'
        }
    }
    
    return models_info


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model instantiation and parameter counting
    print("VitFly-AirSim Model Parameter Counts:")
    print("-" * 40)
    
    models_info = get_model_info()
    for name, info in models_info.items():
        model = info['class']().float()
        actual_params = count_parameters(model)
        expected_params = info['params']
        
        print(f"{name}:")
        print(f"  Expected: {expected_params:,}")
        print(f"  Actual:   {actual_params:,}")
        print(f"  Match:    {actual_params == expected_params}")
        print(f"  Description: {info['description']}")
        print()