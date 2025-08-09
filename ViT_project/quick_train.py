"""
Quick training script for testing the Vision Transformer setup.

This script runs a quick training test with smaller configurations to verify everything works.
"""
import torch
import torch.nn as nn
from pathlib import Path
from model.vit_model import create_vit_model
from model.data_utils import download_food101, create_food101_dataloaders


def quick_train_test():
    """Quick training test to verify setup works."""
    print("=" * 60)
    print("🚀 Quick Training Test for Vision Transformer")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Download and prepare data
    print("\n📁 Preparing Food-101 dataset...")
    try:
        dataset_path = download_food101("data")
        print("✅ Dataset ready!")
    except Exception as e:
        print(f"❌ Dataset preparation failed: {e}")
        return
    
    # Create small data loaders for testing
    print("\n🔄 Creating data loaders...")
    try:
        train_dataloader, test_dataloader, class_names = create_food101_dataloaders(
            dataset_path,
            batch_size=8,  # Small batch size for testing
            img_size=224,
            num_workers=0  # 0 for Windows compatibility
        )
        print(f"✅ Data loaders created!")
        print(f"   - Classes: {len(class_names)}")
        print(f"   - Train batches: {len(train_dataloader)}")
        print(f"   - Test batches: {len(test_dataloader)}")
    except Exception as e:
        print(f"❌ Data loader creation failed: {e}")
        return
    
    # Create smaller model for testing
    print("\n🧠 Creating Vision Transformer model...")
    try:
        model = create_vit_model(
            num_classes=len(class_names),
            img_size=224,
            patch_size=16,
            embedding_dim=384,  # Smaller than default 768
            num_transformer_layers=6,  # Smaller than default 12
            num_heads=6,  # Smaller than default 12
            mlp_size=1536  # Smaller than default 3072
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created!")
        print(f"   - Parameters: {total_params:,}")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return
    
    # Test forward pass
    print("\n🔍 Testing forward pass...")
    try:
        # Get a sample batch
        sample_images, sample_labels = next(iter(train_dataloader))
        sample_images = sample_images.to(device)
        sample_labels = sample_labels.to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(sample_images)
        
        print(f"✅ Forward pass successful!")
        print(f"   - Input shape: {sample_images.shape}")
        print(f"   - Output shape: {outputs.shape}")
        print(f"   - Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        if "out of memory" in str(e):
            print("💡 Try reducing batch_size or model dimensions")
        return
    
    # Test training step
    print("\n🏋️ Testing training step...")
    try:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        
        # One training step
        outputs = model(sample_images)
        loss = loss_fn(outputs, sample_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"✅ Training step successful!")
        print(f"   - Loss: {loss.item():.4f}")
        
        # Test predictions
        model.eval()
        with torch.no_grad():
            pred_logits = model(sample_images)
            pred_probs = torch.softmax(pred_logits, dim=1)
            pred_labels = pred_logits.argmax(dim=1)
            
        accuracy = (pred_labels == sample_labels).float().mean().item()
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - Max probability: {pred_probs.max().item():.4f}")
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        if "out of memory" in str(e):
            print("💡 Try reducing batch_size or model dimensions")
        return
    
    # Test saving model
    print("\n💾 Testing model saving...")
    try:
        save_dir = Path("models")
        save_dir.mkdir(exist_ok=True)
        
        save_path = save_dir / "quick_test_vit.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'accuracy': accuracy
        }, save_path)
        
        print(f"✅ Model saved to: {save_path}")
        
        # Test loading
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Model saving/loading failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! Your setup is ready for training.")
    print("💡 To start full training, run: python train.py")
    print("💡 For smaller experiments, try: python train.py --epochs 5 --batch_size 8")
    print("=" * 60)


if __name__ == "__main__":
    quick_train_test()
