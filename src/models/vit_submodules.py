"""
Vision Transformer Submodules for VitFly-AirSim

This module contains the submodules for ViT that were used in the paper 
"Utilizing vision transformer models for end-to-end vision-based
quadrotor obstacle avoidance" by Bhattacharya, et. al

Adapted for Windows and AirSim compatibility.

Source: https://github.com/git-dhruv/Segformer
Author: A Bhattacharya, et. al (GRASP Lab, University of Pennsylvania)
"""

import torch
import torch.nn as nn


class OverlapPatchMerging(nn.Module):
    """Overlap patch merging layer for Vision Transformer"""
    
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=patch_size, stride=stride, padding=padding
        )
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, patches):
        """Merge patches to reduce dimensions of input.

        Args:
            patches: tensor with shape (B, C, H, W) where
                B is the Batch size
                C is the number of Channels
                H and W are the Height and Width
                
        Returns:
            tuple: (merged_patches, H, W) where merged_patches has shape (B, N, C)
        """
        x = self.conv(patches)
        _, _, H, W = x.shape
        # Flatten spatial dimensions and transpose to (B, N, C)
        x = x.flatten(2).transpose(1, 2)
        x = self.layer_norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    """Efficient self-attention mechanism with spatial reduction"""
    
    def __init__(self, channels, reduction_ratio, num_heads):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels {channels} should be divided by num_heads {num_heads}."
        )

        self.heads = num_heads
        
        # Spatial reduction components
        self.reduction_conv = nn.Conv2d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=reduction_ratio, 
            stride=reduction_ratio
        )
        self.reduction_norm = nn.LayerNorm(channels)
        
        # Attention components
        self.key_value_proj = nn.Linear(channels, channels * 2)
        self.query_proj = nn.Linear(channels, channels)
        self.softmax = nn.Softmax(dim=-1)
        self.output_proj = nn.Linear(channels, channels)

    def forward(self, x, H, W):
        """Perform self attention with reduced sequence length

        Args:
            x: tensor of shape (B, N, C) where
                B is the batch size,
                N is the number of queries (equal to H * W)
                C is the number of channels
                
        Returns:
            tensor of shape (B, N, C)
        """
        B, N, C = x.shape
        
        # Spatial reduction for key and value
        # Reshape to spatial format for convolution
        x_spatial = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
        x_reduced = self.reduction_conv(x_spatial)
        x_reduced = x_reduced.reshape(B, C, -1).permute(0, 2, 1).contiguous()
        x_reduced = self.reduction_norm(x_reduced)
        
        # Extract key and value from reduced features
        key_value = self.key_value_proj(x_reduced)
        key_value = key_value.reshape(
            B, -1, 2, self.heads, C // self.heads
        ).permute(2, 0, 3, 1, 4).contiguous()
        key, value = key_value[0], key_value[1]  # (B, heads, N_reduced, C//heads)
        
        # Extract query from original features
        query = self.query_proj(x).reshape(
            B, N, self.heads, C // self.heads
        ).permute(0, 2, 1, 3).contiguous()  # (B, heads, N, C//heads)
        
        # Compute attention
        scale = (C // self.heads) ** 0.5
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / scale
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, value)
        attended = attended.transpose(1, 2).reshape(B, N, C)
        
        # Final projection
        output = self.output_proj(attended)
        return output


class MixFFN(nn.Module):
    """Mix Feed-Forward Network with depthwise convolution"""
    
    def __init__(self, channels, expansion_factor):
        super().__init__()
        expanded_channels = channels * expansion_factor
        
        # MLP layers
        self.linear1 = nn.Linear(channels, expanded_channels)
        self.depthwise_conv = nn.Conv2d(
            expanded_channels, expanded_channels, 
            kernel_size=3, padding=1, groups=expanded_channels
        )
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(expanded_channels, channels)

    def forward(self, x, H, W):
        """Apply Mix FFN to input features

        Args:
            x: tensor with shape (B, N, C) where
                B is the Batch size
                N is the sequence length (H*W)
                C is the number of Channels
                
        Returns:
            tensor with shape (B, N, C)
        """
        # First linear transformation
        x = self.linear1(x)
        B, N, C = x.shape
        
        # Reshape for depthwise convolution
        x = x.transpose(1, 2).view(B, C, H, W)
        
        # Apply depthwise convolution and activation
        x = self.gelu(self.depthwise_conv(x))
        
        # Reshape back and apply second linear transformation
        x = x.flatten(2).transpose(1, 2)
        x = self.linear2(x)
        return x


class MixTransformerEncoderLayer(nn.Module):
    """Mix Transformer Encoder Layer combining patch merging, attention, and FFN"""
    
    def __init__(self, in_channels, out_channels, patch_size, stride, padding, 
                 n_layers, reduction_ratio, num_heads, expansion_factor):
        super().__init__()
        
        # Patch merging layer
        self.patch_merge = OverlapPatchMerging(
            in_channels, out_channels, patch_size, stride, padding
        )
        
        # Multiple transformer layers
        self.attention_layers = nn.ModuleList([
            EfficientSelfAttention(out_channels, reduction_ratio, num_heads) 
            for _ in range(n_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            MixFFN(out_channels, expansion_factor) 
            for _ in range(n_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(out_channels) 
            for _ in range(n_layers)
        ])

    def forward(self, x):
        """Run one block of the mix vision transformer

        Args:
            x: tensor with shape (B, C, H, W) where
                B is the Batch size
                C is the number of Channels
                H and W are the Height and Width
                
        Returns:
            tensor with shape (B, C', H', W') after processing
        """
        B, C, H, W = x.shape
        
        # Apply patch merging
        x, H, W = self.patch_merge(x)  # Now x is (B, N, C')
        
        # Apply transformer layers
        for attention, ffn, norm in zip(
            self.attention_layers, self.ffn_layers, self.norm_layers
        ):
            # Self-attention with residual connection
            x = x + attention(x, H, W)
            
            # Feed-forward with residual connection
            x = x + ffn(x, H, W)
            
            # Layer normalization
            x = norm(x)
        
        # Reshape back to spatial format
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x