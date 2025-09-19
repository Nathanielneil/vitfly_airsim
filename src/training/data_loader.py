"""
Data Loader for VitFly-AirSim Training

This module handles loading and preprocessing of training data collected
from AirSim simulation, adapted from the original VitFly project.

Author: Adapted from original VitFly project
"""

import os
import cv2
import glob
import time
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict, Any
import logging
from pathlib import Path


class VitFlyDataset(Dataset):
    """PyTorch Dataset for VitFly training data"""
    
    def __init__(self, data_dir: str, val_split: float = 0.2, is_train: bool = True,
                 target_height: int = 60, target_width: int = 90, 
                 short: int = 0, seed: Optional[int] = None,
                 transform=None):
        """Initialize dataset
        
        Args:
            data_dir: Directory containing trajectory folders
            val_split: Validation split ratio
            is_train: Whether this is training set (True) or validation set (False)
            target_height: Target height for depth images
            target_width: Target width for depth images
            short: If > 0, limit number of trajectories for testing
            seed: Random seed for reproducibility
            transform: Optional transform to apply to images
        """
        self.data_dir = Path(data_dir)
        self.target_height = target_height
        self.target_width = target_width
        self.transform = transform
        self.logger = logging.getLogger(__name__)
        
        # Load and process data
        self.image_paths = []
        self.metadata = []
        self.trajectory_lengths = []
        self.desired_velocities = []
        self.quaternions = []
        self.velocity_commands = []
        
        self._load_data(val_split, is_train, short, seed)
        
    def _load_data(self, val_split: float, is_train: bool, short: int, seed: Optional[int]):
        """Load data from trajectory folders"""
        start_time = time.time()
        
        # Get all trajectory folders
        traj_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        if short > 0:
            traj_folders = traj_folders[:short]
        
        # Shuffle folders for train/val split
        if seed is not None:
            random.seed(seed)
        random.shuffle(traj_folders)
        
        # Split into train and validation
        num_val_trajs = int(val_split * len(traj_folders))
        if is_train:
            selected_folders = traj_folders[num_val_trajs:]
        else:
            selected_folders = traj_folders[:num_val_trajs]
        
        self.logger.info(f"Loading {'training' if is_train else 'validation'} data from {len(selected_folders)} trajectories")
        
        skipped_folders = 0
        skipped_images = 0
        
        for i, traj_folder in enumerate(selected_folders):
            if i % max(1, len(selected_folders) // 10) == 0:
                self.logger.info(f"Loading trajectory {i+1}/{len(selected_folders)}: {traj_folder.name}")
            
            # Load trajectory data
            success = self._load_trajectory(traj_folder)
            if not success:
                skipped_folders += 1
                continue
        
        self.logger.info(f"Data loading completed in {time.time() - start_time:.2f}s")
        self.logger.info(f"Loaded {len(self.image_paths)} images from {len(selected_folders) - skipped_folders} trajectories")
        if skipped_folders > 0:
            self.logger.warning(f"Skipped {skipped_folders} folders due to errors")
    
    def _load_trajectory(self, traj_folder: Path) -> bool:
        """Load single trajectory data
        
        Args:
            traj_folder: Path to trajectory folder
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Load metadata
            metadata_path = traj_folder / "data.csv"
            if not metadata_path.exists():
                self.logger.warning(f"No metadata file in {traj_folder.name}")
                return False
            
            df = pd.read_csv(metadata_path)
            
            if len(df) == 0:
                self.logger.warning(f"Empty metadata in {traj_folder.name}")
                return False
            
            # Get image files
            image_files = sorted(traj_folder.glob("*.png"))
            
            if len(image_files) != len(df):
                self.logger.warning(f"Mismatch in {traj_folder.name}: {len(image_files)} images vs {len(df)} metadata entries")
                # Use minimum length to avoid index errors
                min_length = min(len(image_files), len(df))
                image_files = image_files[:min_length]
                df = df.iloc[:min_length]
            
            # Extract required data
            traj_start_idx = len(self.image_paths)
            
            for idx, (img_path, (_, row)) in enumerate(zip(image_files, df.iterrows())):
                self.image_paths.append(img_path)
                
                # Extract metadata
                desired_vel = row.get('desired_velocity', 5.0)
                self.desired_velocities.append(desired_vel)
                
                # Extract quaternion (w, x, y, z)
                quat = [
                    row.get('orientation_w', 1.0),
                    row.get('orientation_x', 0.0), 
                    row.get('orientation_y', 0.0),
                    row.get('orientation_z', 0.0)
                ]
                self.quaternions.append(quat)
                
                # Extract velocity commands
                vel_cmd = [
                    row.get('cmd_velocity_x', 0.0),
                    row.get('cmd_velocity_y', 0.0),
                    row.get('cmd_velocity_z', 0.0)
                ]
                self.velocity_commands.append(vel_cmd)
                
                # Store full metadata row
                self.metadata.append(row.to_dict())
            
            # Store trajectory length
            self.trajectory_lengths.append(len(image_files))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading trajectory {traj_folder.name}: {e}")
            return False
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get single data item
        
        Args:
            idx: Data index
            
        Returns:
            Dictionary with image, metadata, and labels
        """
        # Load and process image
        image = self._load_image(idx)
        
        # Get metadata
        desired_vel = torch.tensor([self.desired_velocities[idx]], dtype=torch.float32)
        quaternion = torch.tensor(self.quaternions[idx], dtype=torch.float32)
        velocity_command = torch.tensor(self.velocity_commands[idx], dtype=torch.float32)
        
        return {
            'image': image,
            'desired_velocity': desired_vel,
            'quaternion': quaternion,
            'velocity_command': velocity_command,
            'metadata': self.metadata[idx]
        }
    
    def _load_image(self, idx: int) -> torch.Tensor:
        """Load and preprocess image
        
        Args:
            idx: Image index
            
        Returns:
            Preprocessed image tensor
        """
        try:
            image_path = self.image_paths[idx]
            
            # Load image (assuming it's saved as uint16 depth)
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                # Return dummy image
                return torch.zeros((1, self.target_height, self.target_width), dtype=torch.float32)
            
            # Convert to float and normalize
            if image.dtype == np.uint16:
                # Convert from uint16 back to [0, 1] range
                image = image.astype(np.float32) / 65535.0
            elif image.dtype == np.uint8:
                # Convert from uint8 to [0, 1] range
                image = image.astype(np.float32) / 255.0
            
            # Ensure single channel
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Resize to target size
            if image.shape != (self.target_height, self.target_width):
                image = cv2.resize(image, (self.target_width, self.target_height))
            
            # Convert to tensor and add channel dimension
            image_tensor = torch.from_numpy(image).unsqueeze(0).float()
            
            # Apply transform if provided
            if self.transform is not None:
                image_tensor = self.transform(image_tensor)
            
            return image_tensor
            
        except Exception as e:
            self.logger.error(f"Error loading image {idx}: {e}")
            return torch.zeros((1, self.target_height, self.target_width), dtype=torch.float32)
    
    def get_trajectory_data(self, traj_idx: int) -> Dict[str, Any]:
        """Get all data for a specific trajectory
        
        Args:
            traj_idx: Trajectory index
            
        Returns:
            Dictionary with trajectory data
        """
        if traj_idx >= len(self.trajectory_lengths):
            return {}
        
        # Calculate start and end indices for trajectory
        start_idx = sum(self.trajectory_lengths[:traj_idx])
        end_idx = start_idx + self.trajectory_lengths[traj_idx]
        
        trajectory_data = []
        for idx in range(start_idx, end_idx):
            trajectory_data.append(self[idx])
        
        return {
            'trajectory_index': traj_idx,
            'length': self.trajectory_lengths[traj_idx],
            'data': trajectory_data
        }


class VitFlyDataLoader:
    """Data loader wrapper for VitFly training"""
    
    def __init__(self, data_dir: str, batch_size: int = 32, 
                 val_split: float = 0.2, num_workers: int = 4,
                 target_height: int = 60, target_width: int = 90,
                 short: int = 0, seed: Optional[int] = None):
        """Initialize data loader
        
        Args:
            data_dir: Directory containing training data
            batch_size: Batch size for training
            val_split: Validation split ratio
            num_workers: Number of worker processes
            target_height: Target height for images
            target_width: Target width for images
            short: Limit number of trajectories for testing
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.target_height = target_height
        self.target_width = target_width
        self.short = short
        self.seed = seed
        
        self.logger = logging.getLogger(__name__)
        
        # Create datasets
        self.train_dataset = VitFlyDataset(
            data_dir, val_split, is_train=True,
            target_height=target_height, target_width=target_width,
            short=short, seed=seed
        )
        
        self.val_dataset = VitFlyDataset(
            data_dir, val_split, is_train=False,
            target_height=target_height, target_width=target_width,
            short=short, seed=seed
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        self.logger.info(f"Created data loaders: {len(self.train_dataset)} train, {len(self.val_dataset)} val")
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded data
        
        Returns:
            Dictionary with data statistics
        """
        # Sample some data for statistics
        sample_size = min(1000, len(self.train_dataset))
        sample_indices = random.sample(range(len(self.train_dataset)), sample_size)
        
        desired_vels = []
        velocity_commands = []
        
        for idx in sample_indices:
            data = self.train_dataset[idx]
            desired_vels.append(data['desired_velocity'].item())
            velocity_commands.append(data['velocity_command'].numpy())
        
        desired_vels = np.array(desired_vels)
        velocity_commands = np.array(velocity_commands)
        
        return {
            'train_size': len(self.train_dataset),
            'val_size': len(self.val_dataset),
            'num_trajectories_train': len(self.train_dataset.trajectory_lengths),
            'num_trajectories_val': len(self.val_dataset.trajectory_lengths),
            'avg_trajectory_length': np.mean(self.train_dataset.trajectory_lengths),
            'desired_velocity_stats': {
                'mean': float(np.mean(desired_vels)),
                'std': float(np.std(desired_vels)),
                'min': float(np.min(desired_vels)),
                'max': float(np.max(desired_vels))
            },
            'velocity_command_stats': {
                'mean': velocity_commands.mean(axis=0).tolist(),
                'std': velocity_commands.std(axis=0).tolist(),
                'min': velocity_commands.min(axis=0).tolist(),
                'max': velocity_commands.max(axis=0).tolist()
            }
        }


def create_data_loaders(data_dir: str, config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders
    
    Args:
        data_dir: Directory containing training data
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_loader_wrapper = VitFlyDataLoader(
        data_dir=data_dir,
        batch_size=config.get('batch_size', 32),
        val_split=config.get('val_split', 0.2),
        num_workers=config.get('num_workers', 4),
        target_height=config.get('target_height', 60),
        target_width=config.get('target_width', 90),
        short=config.get('short', 0),
        seed=config.get('seed', None)
    )
    
    return data_loader_wrapper.train_loader, data_loader_wrapper.val_loader


def collate_trajectory_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for trajectory-based batching
    
    Args:
        batch: List of data items
        
    Returns:
        Batched data dictionary
    """
    # Stack all tensors
    images = torch.stack([item['image'] for item in batch])
    desired_velocities = torch.stack([item['desired_velocity'] for item in batch])
    quaternions = torch.stack([item['quaternion'] for item in batch])
    velocity_commands = torch.stack([item['velocity_command'] for item in batch])
    
    return {
        'image': images,
        'desired_velocity': desired_velocities,
        'quaternion': quaternions,
        'velocity_command': velocity_commands
    }


if __name__ == '__main__':
    # Test data loading
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample data directory
    data_dir = "data/training_data"
    
    if os.path.exists(data_dir):
        print("Testing VitFly data loader...")
        
        data_loader_wrapper = VitFlyDataLoader(
            data_dir=data_dir,
            batch_size=4,
            val_split=0.2,
            short=2  # Test with just 2 trajectories
        )
        
        # Test train loader
        for i, batch in enumerate(data_loader_wrapper.train_loader):
            print(f"Batch {i}:")
            print(f"  Image shape: {batch['image'].shape}")
            print(f"  Desired velocity shape: {batch['desired_velocity'].shape}")
            print(f"  Quaternion shape: {batch['quaternion'].shape}")
            print(f"  Velocity command shape: {batch['velocity_command'].shape}")
            
            if i >= 2:  # Test just a few batches
                break
        
        # Print statistics
        stats = data_loader_wrapper.get_data_statistics()
        print("\nData statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    else:
        print(f"Data directory {data_dir} not found. Skipping test.")