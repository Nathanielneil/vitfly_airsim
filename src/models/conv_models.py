"""
Convolutional Neural Network Models for VitFly-AirSim

This module contains traditional CNN-based models including ConvNet, LSTMNet,
and UNet architectures for comparison with Vision Transformer models.

Author: Adapted from original VitFly project (GRASP Lab, UPenn)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from .vit_models import refine_inputs


class ConvNet(nn.Module):
    """Convolutional + Fully Connected Network
    
    Simple baseline model using traditional CNN architecture.
    Parameters: 235,269
    """
    
    def __init__(self, input_height=60, input_width=90):
        super().__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(4, 10, kernel_size=3, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(4)
        
        # Fully connected layers
        # Input size: image features (845) + desired_vel (1) + quaternion (4) = 850
        self.fc0 = nn.Linear(845, 256, bias=False)
        self.fc1 = nn.Linear(256, 64, bias=False)
        self.fc2 = nn.Linear(64, 32, bias=False)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, X):
        """Forward pass through ConvNet
        
        Args:
            X: List containing [depth_image, desired_velocity, quaternion]
            
        Returns:
            tuple: (velocity_commands, None) - None for compatibility with LSTM models
        """
        X = refine_inputs(X, self.input_height, self.input_width)
        
        depth_image = X[0]  # Shape: (B, 1, H, W)
        desired_vel = X[1]  # Shape: (B, 1)
        quaternion = X[2]   # Shape: (B, 4)
        
        # Convolutional processing
        x = F.relu(self.conv1(depth_image))
        x = self.bn1(x)
        x = -self.maxpool(-x)  # Negative max pooling
        x = F.relu(self.conv2(x))
        x = self.avgpool(x)
        
        # Flatten spatial features
        x = torch.flatten(x, 1)
        
        # Concatenate with metadata
        metadata = torch.cat([desired_vel * 0.1, quaternion], dim=1).float()
        x = torch.cat([x, metadata], dim=1).float()
        
        # Fully connected layers
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        velocity_commands = self.fc3(x)
        
        return velocity_commands, None


class LSTMNet(nn.Module):
    """LSTM + Convolutional Network
    
    Uses LSTM for temporal modeling with CNN feature extraction.
    Parameters: 2,949,937
    """
    
    def __init__(self, input_height=60, input_width=90):
        super().__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, stride=3, padding=1)
        self.conv2 = nn.Conv2d(4, 10, kernel_size=3, stride=2, padding=0)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(10)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=665,     # Feature size after conv + metadata
            hidden_size=395,
            num_layers=2,
            dropout=0.15,
            bias=False,
            batch_first=True
        )
        
        # Output layers
        self.fc1 = spectral_norm(nn.Linear(395, 64))
        self.fc2 = spectral_norm(nn.Linear(64, 16))
        self.fc3 = spectral_norm(nn.Linear(16, 3))

    def forward(self, X):
        """Forward pass through LSTMNet
        
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
        
        # Convolutional processing
        x = F.relu(self.conv1(depth_image))
        x = self.bn1(x)
        x = -self.maxpool(-x)  # Negative max pooling
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.avgpool(x)
        
        # Flatten and concatenate with metadata
        x = torch.flatten(x, 1)
        metadata = torch.cat([desired_vel * 0.1, quaternion], dim=1).float()
        lstm_input = torch.cat([x, metadata], dim=1).float()
        
        # Add sequence dimension for LSTM
        lstm_input = lstm_input.unsqueeze(1)  # (B, 1, features)
        
        # LSTM processing
        if hidden_state is not None:
            lstm_output, new_hidden_state = self.lstm(lstm_input, hidden_state)
        else:
            lstm_output, new_hidden_state = self.lstm(lstm_input)
        
        # Remove sequence dimension and get final output
        lstm_output = lstm_output.squeeze(1)
        x = F.leaky_relu(self.fc1(lstm_output))
        x = F.leaky_relu(self.fc2(x))
        velocity_commands = self.fc3(x)
        
        return velocity_commands, new_hidden_state


class UNetConvLSTMNet(nn.Module):
    """UNet + Convolutional + LSTM Network
    
    Uses UNet architecture for dense feature extraction combined with LSTM.
    Parameters: 2,955,822
    """
    
    def __init__(self, input_height=60, input_width=90):
        super().__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        
        # UNet encoder layers
        self.unet_e11 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.unet_e12 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.unet_pool1 = nn.MaxPool2d(kernel_size=2, stride=3)
        
        self.unet_e21 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.unet_e22 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.unet_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.unet_e31 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.unet_e32 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        
        # UNet decoder layers
        self.unet_upconv1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.unet_d11 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.unet_d12 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        
        self.unet_upconv2 = nn.ConvTranspose2d(8, 4, kernel_size=3, stride=3)
        self.unet_d21 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.unet_d22 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        
        self.unet_out = nn.Conv2d(4, 1, kernel_size=1)
        
        # Additional convolutional processing
        self.conv1 = nn.Conv2d(2, 4, kernel_size=5, stride=3)  # Input + UNet output
        self.conv2 = nn.Conv2d(4, 10, kernel_size=5, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(4)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=3065,    # Conv features + bottleneck features + metadata
            hidden_size=200,
            num_layers=2,
            dropout=0.15,
            bias=False,
            batch_first=True
        )
        
        # Output layers
        self.fc1 = spectral_norm(nn.Linear(200, 64))
        self.fc2 = spectral_norm(nn.Linear(64, 32))
        self.fc3 = spectral_norm(nn.Linear(32, 3))

    def forward(self, X):
        """Forward pass through UNet+LSTM
        
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
        
        # UNet encoder
        e1 = F.relu(self.unet_e12(F.relu(self.unet_e11(depth_image))))
        e1_pool = self.unet_pool1(e1)
        
        e2 = F.relu(self.unet_e22(F.relu(self.unet_e21(e1_pool))))
        e2_pool = self.unet_pool2(e2)
        
        e3 = F.relu(self.unet_e32(F.relu(self.unet_e31(e2_pool))))
        
        # UNet decoder
        d1 = F.relu(self.unet_d12(F.relu(self.unet_d11(
            torch.cat([self.unet_upconv1(e3), e2], dim=1)
        ))))
        
        d2 = F.relu(self.unet_d22(F.relu(self.unet_d21(
            torch.cat([self.unet_upconv2(d1), e1], dim=1)
        ))))
        
        unet_output = self.unet_out(d2)
        
        # Combine original image with UNet output
        combined = torch.cat([depth_image, unet_output], dim=1)
        
        # Additional convolutional processing
        conv_out = F.relu(self.conv1(combined))
        conv_out = self.bn1(conv_out)
        conv_out = -self.maxpool(-conv_out)  # Negative max pooling
        conv_out = F.relu(self.conv2(conv_out))
        conv_out = self.avgpool(conv_out)
        
        # Prepare LSTM input
        conv_features = torch.flatten(conv_out, 1)
        bottleneck_features = torch.flatten(e3, 1)
        metadata = torch.cat([desired_vel * 0.1, quaternion], dim=1).float()
        
        lstm_input = torch.cat([
            conv_features, bottleneck_features, metadata
        ], dim=1).float()
        
        # Add sequence dimension for LSTM
        lstm_input = lstm_input.unsqueeze(1)  # (B, 1, features)
        
        # LSTM processing
        if hidden_state is not None:
            lstm_output, new_hidden_state = self.lstm(lstm_input, hidden_state)
        else:
            lstm_output, new_hidden_state = self.lstm(lstm_input)
        
        # Remove sequence dimension and get final output
        lstm_output = lstm_output.squeeze(1)
        x = F.leaky_relu(self.fc1(lstm_output))
        x = F.leaky_relu(self.fc2(x))
        velocity_commands = self.fc3(x)
        
        return velocity_commands, new_hidden_state


def get_conv_models_info():
    """Get information about available convolutional models"""
    
    models_info = {
        'ConvNet': {
            'class': ConvNet,
            'params': 235_269,
            'description': 'Simple CNN baseline with fully connected output'
        },
        'LSTMNet': {
            'class': LSTMNet,
            'params': 2_949_937,
            'description': 'CNN + LSTM for temporal modeling'
        },
        'UNet': {
            'class': UNetConvLSTMNet,
            'params': 2_955_822,
            'description': 'UNet + CNN + LSTM for dense feature extraction'
        }
    }
    
    return models_info


if __name__ == '__main__':
    # Test model instantiation and parameter counting
    from .vit_models import count_parameters
    
    print("VitFly-AirSim Convolutional Model Parameter Counts:")
    print("-" * 50)
    
    models_info = get_conv_models_info()
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