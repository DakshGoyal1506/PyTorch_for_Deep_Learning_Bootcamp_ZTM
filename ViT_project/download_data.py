"""
Download Food-101 dataset script.

This script downloads and prepares the Food-101 dataset for training.
Run this script before training to ensure the dataset is available.
"""

import argparse
from pathlib import Path
from model.data_utils import download_food101, get_food101_class_names


def main():
    parser = argparse.ArgumentParser(description="Download Food-101 dataset")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data",
        help="Directory to save the dataset (default: data)"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Food-101 Dataset Download")
    print("=" * 50)
    
    # Download the dataset
    dataset_path = download_food101(args.data_dir)
    
    # Verify the download
    class_names = get_food101_class_names(dataset_path)
    
    print(f"\n✅ Dataset successfully downloaded and verified!")
    print(f"📁 Dataset location: {dataset_path}")
    print(f"🏷️  Number of classes: {len(class_names)}")
    print(f"📊 Total images: ~101,000 (1,000 per class)")
    print(f"🔄 Train/Test split: 750/250 per class")
    
    print(f"\n📋 First 10 food categories:")
    for i, class_name in enumerate(class_names[:10], 1):
        print(f"  {i:2d}. {class_name}")
    
    print(f"\n🎯 Ready for training! Run: python train.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
