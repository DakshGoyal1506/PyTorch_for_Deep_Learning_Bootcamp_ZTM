"""
Testing script for Vision Transformer model.

This script loads a trained model and evaluates it on the test set.
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List
import argparse
from tqdm import tqdm

from model.vit_model import create_vit_model
from model.data_utils import download_food101, create_food101_dataloaders


def evaluate_model(model: nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  device: torch.device,
                  class_names: List[str]) -> Dict:
    """
    Evaluate model on test dataset.
    
    Args:
        model: Trained model
        dataloader: Test DataLoader
        device: Device to run evaluation on
        class_names: List of class names
        
    Returns:
        Dict: Evaluation results
    """
    model.eval()
    
    total_samples = 0
    correct_predictions = 0
    class_correct = {class_name: 0 for class_name in class_names}
    class_total = {class_name: 0 for class_name in class_names}
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for batch, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            
            # Calculate accuracy
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_name = class_names[label]
                class_total[class_name] += 1
                if predictions[i] == labels[i]:
                    class_correct[class_name] += 1
    
    # Calculate overall accuracy
    overall_accuracy = correct_predictions / total_samples
    
    # Calculate per-class accuracy
    class_accuracies = {}
    for class_name in class_names:
        if class_total[class_name] > 0:
            class_accuracies[class_name] = class_correct[class_name] / class_total[class_name]
        else:
            class_accuracies[class_name] = 0.0
    
    results = {
        "overall_accuracy": overall_accuracy,
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "class_accuracies": class_accuracies
    }
    
    return results


def load_model(model_path: str, num_classes: int, device: torch.device) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        num_classes: Number of classes
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    # Create model
    model = create_vit_model(num_classes=num_classes).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Previous test accuracy: {checkpoint.get('test_acc', 'unknown'):.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Test Vision Transformer on Food-101")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
    
    args = parser.parse_args()
    
    # Check if model file exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Download and prepare data if needed
    print("Preparing Food-101 dataset...")
    dataset_path = download_food101(args.data_dir)
    
    # Create data loaders
    print("Creating data loaders...")
    _, test_dataloader, class_names = create_food101_dataloaders(
        dataset_path,
        batch_size=args.batch_size,
        img_size=224,
        num_workers=args.num_workers
    )
    
    print(f"Test samples: {len(test_dataloader.dataset)}")
    print(f"Number of classes: {len(class_names)}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = load_model(str(model_path), len(class_names), device)
    
    # Evaluate model
    results = evaluate_model(model, test_dataloader, device, class_names)
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)")
    print(f"Correct Predictions: {results['correct_predictions']} / {results['total_samples']}")
    
    # Top 10 best performing classes
    sorted_classes = sorted(results['class_accuracies'].items(), key=lambda x: x[1], reverse=True)
    print(f"\n🏆 Top 10 Best Performing Classes:")
    for i, (class_name, accuracy) in enumerate(sorted_classes[:10], 1):
        print(f"  {i:2d}. {class_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Bottom 10 performing classes
    print(f"\n😞 Bottom 10 Performing Classes:")
    for i, (class_name, accuracy) in enumerate(sorted_classes[-10:], 1):
        print(f"  {i:2d}. {class_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Average class accuracy
    avg_class_accuracy = sum(results['class_accuracies'].values()) / len(results['class_accuracies'])
    print(f"\nAverage Class Accuracy: {avg_class_accuracy:.4f} ({avg_class_accuracy*100:.2f}%)")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
