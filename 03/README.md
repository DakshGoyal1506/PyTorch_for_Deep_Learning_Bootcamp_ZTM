# 03 - PyTorch Computer Vision

## Overview
This section introduces computer vision with PyTorch, focusing on Convolutional Neural Networks (CNNs) for image classification. It covers working with image datasets, building CNN architectures, and understanding the fundamentals of computer vision in deep learning.

## Learning Objectives
- Understand computer vision libraries in PyTorch (`torchvision`)
- Work with standard computer vision datasets (MNIST, FashionMNIST)
- Build and train Convolutional Neural Networks (CNNs)
- Learn image preprocessing and data augmentation
- Implement the TinyVGG architecture
- Understand GPU acceleration for computer vision

## Key Concepts Covered

### 1. Computer Vision Libraries
- **TorchVision**: PyTorch's computer vision library
- **Datasets**: Pre-built datasets (MNIST, FashionMNIST, CIFAR-10)
- **Transforms**: Image preprocessing and augmentation
- **Models**: Pre-trained model architectures

### 2. Image Data Handling
- Loading and exploring image datasets
- Understanding image tensor shapes (channels, height, width)
- Data normalization and preprocessing
- Creating DataLoaders for batch processing
- Visualizing image data with matplotlib

### 3. Convolutional Neural Networks (CNNs)
- **Convolutional Layers** (`nn.Conv2d`):
  - Kernels/filters and feature maps
  - Stride and padding parameters
  - Feature extraction from images
- **Pooling Layers** (`nn.MaxPool2d`):
  - Dimensionality reduction
  - Translation invariance
  - Feature map compression
- **Activation Functions** (`nn.ReLU`):
  - Non-linearity introduction
  - Feature enhancement

### 4. CNN Architecture Patterns
- **TinyVGG Architecture** (based on CNN Explainer):
  - Two convolutional blocks
  - Feature extraction + classification
  - Practical implementation example
- Layer sequence patterns
- Feature map progression through the network

### 5. Training on GPU
- Device-agnostic code patterns
- Moving models and data to GPU
- Performance improvements with CUDA
- GPU memory management

### 6. Model Evaluation
- Training and testing loops for image data
- Accuracy metrics for image classification
- Loss tracking and visualization
- Making predictions on individual images

## Files in this Section
- `03.ipynb` - Main computer vision notebook
- `03_pytorch_computer_vision_exercises.ipynb` - Exercises with MNIST and FashionMNIST
- `03_pytorch_computer_vision_video.ipynb` - Video walkthrough version
- `helper_functions.py` - Visualization and utility functions
- `data/` - Downloaded datasets directory
- `models/` - Saved model checkpoints

## TinyVGG Architecture Implementation
```python
class TinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, output_shape)
        )
```

## Datasets Worked With
- **MNIST**: Handwritten digits (0-9)
  - 28x28 grayscale images
  - 60,000 training samples
  - 10,000 test samples
- **FashionMNIST**: Fashion items classification
  - 10 classes of clothing items
  - Same dimensions as MNIST
  - More challenging classification task

## Skills Developed
- CNN architecture design and implementation
- Image data preprocessing and augmentation
- GPU-accelerated training
- Computer vision model evaluation
- Feature map visualization and interpretation
- Working with PyTorch's torchvision library

## Computer Vision Concepts
- **Convolution Operation**: Feature detection through learned filters
- **Pooling**: Spatial dimensionality reduction
- **Feature Maps**: Intermediate representations
- **Receptive Fields**: Input regions affecting neurons
- **Translation Invariance**: Position-independent feature detection

## Performance Improvements
- GPU acceleration for faster training
- Efficient data loading with DataLoaders
- Batch processing for optimization
- Memory-efficient training loops

## Prerequisites
- Completion of Sections 00-02
- Understanding of basic neural networks
- Familiarity with image data concepts

## Next Steps
This computer vision foundation leads to:
- Custom Datasets (Section 04) - Working with your own images
- Transfer Learning (Section 06) - Using pre-trained models
- Advanced architectures and techniques
- Real-world computer vision applications

## Key Takeaways
Computer vision is one of the most successful applications of deep learning. This section provides the foundation for understanding how CNNs process images, extract features, and make predictions. The TinyVGG architecture serves as an excellent starting point for understanding more complex computer vision models used in industry.
