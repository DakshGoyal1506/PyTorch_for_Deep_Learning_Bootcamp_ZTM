"""
Data utilities for Food-101 dataset handling and preprocessing.

This module contains functions for:
- Data downloading and extraction
- Data loading and preprocessing
- Creating DataLoaders for training and testing
"""
import torch
import requests
import tarfile
from pathlib import Path
from typing import Tuple, List
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image


class Food101Dataset(Dataset):
    """Custom dataset class for Food-101."""
    
    def __init__(self, file_list: List[str], images_dir: Path, transform=None):
        self.file_list = file_list
        self.images_dir = images_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Get image path and label
        image_path = self.file_list[idx]
        image_path_full = self.images_dir / f"{image_path}.jpg"
        
        # Extract class name from path
        class_name = image_path.split('/')[0]
        
        # Load image
        image = Image.open(image_path_full).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        # Get class index (you'd need a mapping from class names to indices)
        # For now, we'll return the class name and convert it later
        return image, class_name


def download_food101(data_dir: str = "data") -> Path:
    """
    Download and extract the Food-101 dataset.
    
    Args:
        data_dir (str): Directory to save the dataset
        
    Returns:
        Path: Path to the extracted dataset
    """
    data_path = Path(data_dir)
    dataset_path = data_path / "food-101"
    
    if dataset_path.exists():
        print(f"[INFO] Food-101 dataset already exists at {dataset_path}")
        return dataset_path
    
    # Create data directory
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Download URL for Food-101
    url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    filename = "food-101.tar.gz"
    filepath = data_path / filename
    
    print(f"[INFO] Downloading Food-101 dataset from {url}")
    print(f"[INFO] This is a large dataset (~5GB), please be patient...")
    
    # Download the dataset
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"[INFO] Download completed. Extracting...")
    
    # Extract the dataset
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(path=data_path)
    
    # Remove the tar file to save space
    filepath.unlink()
    
    print(f"[INFO] Food-101 dataset extracted to {dataset_path}")
    return dataset_path


def get_food101_class_names(dataset_path: Path) -> List[str]:
    """
    Get the class names from Food-101 dataset.
    
    Args:
        dataset_path (Path): Path to the Food-101 dataset
        
    Returns:
        List[str]: List of class names
    """
    meta_path = dataset_path / "meta"
    classes_file = meta_path / "classes.txt"
    
    if not classes_file.exists():
        raise FileNotFoundError(f"Classes file not found at {classes_file}")
    
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    return sorted(class_names)


def create_food101_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create train and test transforms for Food-101 dataset.
    
    Args:
        img_size (int): Size to resize images to
        
    Returns:
        Tuple[transforms.Compose, transforms.Compose]: Train and test transforms
    """
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Test transforms without augmentation
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def create_food101_dataloaders(
    dataset_path: Path,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 0  # Set to 0 for Windows compatibility
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create DataLoaders for Food-101 dataset.
    
    Args:
        dataset_path (Path): Path to the Food-101 dataset
        batch_size (int): Batch size for DataLoaders
        img_size (int): Image size for transforms
        num_workers (int): Number of workers for DataLoader
        
    Returns:
        Tuple[DataLoader, DataLoader, List[str]]: Train DataLoader, Test DataLoader, Class names
    """
    
    # Get transforms
    train_transform, test_transform = create_food101_transforms(img_size)
    
    # Create datasets
    train_dir = dataset_path / "images"
    test_dir = dataset_path / "images"  # Food-101 doesn't have separate test folder in images
    
    # Load the training and test splits
    meta_path = dataset_path / "meta"
    
    # Read train and test file lists
    with open(meta_path / "train.txt", 'r') as f:
        train_files = [line.strip() for line in f.readlines()]
    
    with open(meta_path / "test.txt", 'r') as f:
        test_files = [line.strip() for line in f.readlines()]
    
    # Get class names
    class_names = get_food101_class_names(dataset_path)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    # Use torchvision's ImageFolder for simplicity
    # First, let's try to use the standard ImageFolder approach
    try:
        train_dataset = datasets.ImageFolder(
            root=train_dir,
            transform=train_transform
        )
        
        test_dataset = datasets.ImageFolder(
            root=test_dir,
            transform=test_transform
        )
        
        # Create DataLoaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Get class names from the dataset
        class_names = train_dataset.classes
        
        print(f"[INFO] Created DataLoaders with {len(train_dataset)} training samples and {len(test_dataset)} test samples")
        print(f"[INFO] Number of classes: {len(class_names)}")
        
        return train_dataloader, test_dataloader, class_names
        
    except Exception as e:
        print(f"[ERROR] Could not create ImageFolder datasets: {e}")
        print("[INFO] Please ensure the dataset is properly organized in folders by class name")
        raise


def get_sample_batch(dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a sample batch from a DataLoader.
    
    Args:
        dataloader (DataLoader): DataLoader to sample from
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sample images and labels
    """
    return next(iter(dataloader))


if __name__ == "__main__":
    # Test the data utilities
    print("Testing Food-101 data utilities...")
    
    # Download the dataset (this will take a while on first run)
    dataset_path = download_food101("data")
    
    # Get class names
    class_names = get_food101_class_names(dataset_path)
    print(f"Number of classes: {len(class_names)}")
    print(f"First 5 classes: {class_names[:5]}")
    
    # Create data loaders
    train_dataloader, test_dataloader, class_names = create_food101_dataloaders(
        dataset_path=dataset_path,
        batch_size=32,
        img_size=224
    )
    
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Test batches: {len(test_dataloader)}")
    
    # Get a sample batch
    sample_images, sample_labels = get_sample_batch(train_dataloader)
    print(f"Sample batch shape: {sample_images.shape}")
    print(f"Sample labels shape: {sample_labels.shape}")
