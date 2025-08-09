"""
Training script for Vision Transformer on Food-101 dataset.

This script trains a Vision Transformer model from scratch on the complete Food-101 dataset.
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Dict, List
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.vit_model import create_vit_model
from model.data_utils import download_food101, create_food101_dataloaders


def train_step(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> float:
    """
    Train step for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training DataLoader
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    train_loss = 0.0
    
    for batch, (X, y) in enumerate(dataloader):
        # Send data to device
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        y_pred = model(X)
        
        # Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
    
    # Calculate average loss
    train_loss = train_loss / len(dataloader)
    return train_loss


def test_step(model: nn.Module,
              dataloader: DataLoader,
              loss_fn: nn.Module,
              device: torch.device) -> tuple[float, float]:
    """
    Test step for evaluation.
    
    Args:
        model: Model to evaluate
        dataloader: Test DataLoader
        loss_fn: Loss function
        device: Device to evaluate on
        
    Returns:
        tuple: Test loss and accuracy
    """
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # Send data to device
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            test_pred_logits = model(X)
            
            # Calculate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
    
    # Calculate average loss and accuracy
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    
    return test_loss, test_acc


def train_model(model: nn.Module,
                train_dataloader: DataLoader,
                test_dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: nn.Module,
                epochs: int,
                device: torch.device,
                save_dir: str = "models") -> Dict[str, List[float]]:
    """
    Train the model for multiple epochs.
    
    Args:
        model: Model to train
        train_dataloader: Training DataLoader
        test_dataloader: Test DataLoader
        optimizer: Optimizer
        loss_fn: Loss function
        epochs: Number of epochs to train
        device: Device to train on
        save_dir: Directory to save model checkpoints
        
    Returns:
        Dict: Training history with losses and accuracies
    """
    
    # Create results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    best_test_acc = 0.0
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        try:
            # Train step
            train_loss = train_step(model, train_dataloader, loss_fn, optimizer, device)
            
            # Test step
            test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
            
            # Calculate train accuracy (optional, can be expensive)
            model.eval()
            train_acc = 0.0
            with torch.no_grad():
                for batch, (X, y) in enumerate(train_dataloader):
                    X, y = X.to(device), y.to(device)
                    train_pred_logits = model(X)
                    train_pred_labels = train_pred_logits.argmax(dim=1)
                    train_acc += ((train_pred_labels == y).sum().item() / len(train_pred_labels))
                    if batch >= 10:  # Only calculate on first few batches to save time
                        break
            train_acc = train_acc / min(11, len(train_dataloader))
            
            # Print progress
            print(f"Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
            
            # Update results
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model_path = save_path / "best_vit_food101.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_acc': test_acc,
                    'test_loss': test_loss,
                }, best_model_path)
                print(f"New best model saved with test accuracy: {test_acc:.4f}")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n❌ CUDA out of memory error at epoch {epoch+1}")
                print("💡 Try reducing batch size or model size")
                print("Recommended batch sizes by GPU memory:")
                print("  - 4GB GPU: batch_size=4-8")
                print("  - 8GB GPU: batch_size=8-16") 
                print("  - 12GB GPU: batch_size=16-24")
                print("  - 24GB GPU: batch_size=32-48")
                
                # Clear cache and exit gracefully
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise e
            else:
                raise e
    
    # Save final model
    final_model_path = save_path / "final_vit_food101.pth"
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'test_loss': test_loss,
        'training_results': results
    }, final_model_path)
    
    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"Models saved in: {save_path}")
    
    return results


def plot_training_curves(results: Dict[str, List[float]], save_path: str = None):
    """
    Plot training curves.
    
    Args:
        results: Training results dictionary
        save_path: Path to save the plot
    """
    epochs = range(1, len(results["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    ax1.plot(epochs, results["train_loss"], label="Train Loss", color="blue")
    ax1.plot(epochs, results["test_loss"], label="Test Loss", color="red")
    ax1.set_title("Training and Test Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, results["test_acc"], label="Test Accuracy", color="green")
    ax2.set_title("Test Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train Vision Transformer on Food-101")
    
    # Model parameters
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--embedding_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--mlp_size", type=int, default=3072, help="MLP hidden size")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers (0 for Windows)")
    
    # Save parameters
    parser.add_argument("--save_dir", type=str, default="models", help="Model save directory")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Print GPU memory info if using CUDA
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Recommended batch sizes:")
        print(f"  - 8GB GPU: batch_size=8-16")
        print(f"  - 12GB GPU: batch_size=16-24") 
        print(f"  - 24GB GPU: batch_size=32-48")
        print(f"Current batch size: {args.batch_size}")
        if args.batch_size > 32:
            print("⚠️  Large batch size detected. Reduce if you encounter OOM errors.")
    
    # Download and prepare data if needed
    print("Preparing Food-101 dataset...")
    dataset_path = download_food101(args.data_dir)
    
    # Create data loaders
    print("Creating data loaders...")
    train_dataloader, test_dataloader, class_names = create_food101_dataloaders(
        dataset_path,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating Vision Transformer model...")
    model = create_vit_model(
        num_classes=len(class_names),
        img_size=args.img_size,
        patch_size=args.patch_size,
        embedding_dim=args.embedding_dim,
        num_transformer_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_size=args.mlp_size
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer and loss function
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Train model
    print(f"Starting training for {args.epochs} epochs...")
    print("=" * 70)
    
    results = train_model(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        device=device,
        save_dir=args.save_dir
    )
    
    # Plot training curves
    plot_curves_path = Path(args.save_dir) / "training_curves.png"
    plot_training_curves(results, save_path=plot_curves_path)
    
    print("Training completed successfully! 🎉")


if __name__ == "__main__":
    main()
