# Vision Transformer (ViT) for Food Recognition

A complete implementation of Vision Transformer (ViT) for food recognition using the Food-101 dataset.

## 📁 Project Structure

```
ViT_project/
├── model/
│   ├── __init__.py
│   ├── vit_model.py          # Vision Transformer implementation
│   └── data_utils.py         # Data loading and preprocessing
├── train.py                  # Main training script
├── quick_train.py           # Quick test training script
├── test_model.py            # Model evaluation script
├── inference.py             # Inference utilities
├── download_data.py         # Dataset download script
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Setup Environment

Make sure you have PyTorch installed:
```bash
pip install torch torchvision tqdm pillow requests
```

### 2. Download Dataset

```bash
python download_data.py
```

This will download the Food-101 dataset (~5GB) to the `data/` directory.

### 3. Quick Test

Run a quick training test to verify everything works:
```bash
python quick_train.py
```

### 4. Full Training

Start full training:
```bash
python train.py
```

For custom training:
```bash
python train.py --epochs 10 --batch_size 16 --learning_rate 1e-4
```

## 📊 Model Architecture

Our Vision Transformer implementation includes:

- **Patch Embedding**: Converts images into sequences of patches
- **Multi-head Self-Attention**: Core transformer attention mechanism  
- **MLP Blocks**: Feed-forward networks with GELU activation
- **Transformer Encoder**: Stack of attention + MLP blocks
- **Classification Head**: Final layer for food classification

### Model Configurations

**Default ViT-Base:**
- Image size: 224x224
- Patch size: 16x16  
- Embedding dimension: 768
- Transformer layers: 12
- Attention heads: 12
- MLP size: 3072
- Parameters: ~86M

**Quick Test (Smaller):**
- Embedding dimension: 384
- Transformer layers: 6
- Attention heads: 6
- MLP size: 1536
- Parameters: ~22M

## 🏋️ Training

### Command Line Arguments

```bash
python train.py [OPTIONS]

Model parameters:
  --img_size 224              Input image size
  --patch_size 16             Patch size  
  --embedding_dim 768         Embedding dimension
  --num_layers 12             Number of transformer layers
  --num_heads 12              Number of attention heads
  --mlp_size 3072             MLP hidden size

Training parameters:
  --epochs 50                 Number of training epochs
  --batch_size 32             Batch size
  --learning_rate 1e-4        Learning rate
  --weight_decay 1e-4         Weight decay

Data parameters:
  --data_dir data             Data directory
  --num_workers 0             Number of data loader workers (0 for Windows)
  --save_dir models           Model save directory
```

### GPU Memory Requirements

**Recommended batch sizes by GPU memory:**
- 4GB GPU: batch_size=4-8
- 8GB GPU: batch_size=8-16
- 12GB GPU: batch_size=16-24  
- 24GB GPU: batch_size=32-48

### Training Features

- **Automatic checkpointing**: Saves best model during training
- **Loss visualization**: Automatically plots training curves
- **Memory error handling**: Graceful handling of CUDA OOM errors
- **Progress tracking**: Real-time training progress with tqdm

## 🧪 Testing & Evaluation

### Evaluate Trained Model

```bash
python test_model.py --model_path models/best_vit_food101.pth
```

### Make Predictions

```bash
python inference.py --model_path models/best_vit_food101.pth --image_path path/to/image.jpg
```

## 📈 Expected Results

With proper training, you should expect:

- **Training accuracy**: 85-95%
- **Test accuracy**: 70-85% 
- **Training time**: 
  - Quick test: ~5 minutes
  - Full training (50 epochs): 6-12 hours (depending on GPU)

## 🐛 Troubleshooting

### Common Issues

**CUDA out of memory:**
```bash
# Reduce batch size
python train.py --batch_size 8

# Use smaller model
python train.py --embedding_dim 384 --num_layers 6 --num_heads 6
```

**Multiprocessing errors on Windows:**
```bash
# Set num_workers to 0
python train.py --num_workers 0
```

**Dataset download fails:**
```bash
# Try downloading manually and extract to data/food-101/
```

## 🔧 Customization

### Using Your Own Dataset

1. Modify `data_utils.py` to load your dataset
2. Update `create_vit_model()` with your number of classes
3. Adjust image size and transforms as needed

### Model Architecture Changes

Edit `vit_model.py` to:
- Change patch sizes
- Modify attention mechanisms
- Add regularization techniques
- Experiment with different embeddings

## 📚 References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [PyTorch Vision Transformer Tutorial](https://pytorch.org/vision/stable/models.html#vision-transformer)

## 📄 License

This project is for educational purposes. Please check the Food-101 dataset license for commercial use.

---

**Happy Training! 🎉**
