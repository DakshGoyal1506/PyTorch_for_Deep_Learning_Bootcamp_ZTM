# 00 - PyTorch Fundamentals

## Overview
This section covers the fundamental concepts of PyTorch, focusing on tensor operations and basic neural network building blocks. It serves as the foundation for all subsequent deep learning concepts in the course.

## Learning Objectives
- Understand PyTorch tensors and their properties
- Learn tensor operations and manipulations
- Explore tensor shapes, dimensions, and indexing
- Introduction to PyTorch's automatic differentiation system
- Basic tensor mathematics for neural networks

## Key Concepts Covered

### 1. Tensor Basics
- **Scalars**: 0-dimensional tensors (`torch.tensor(7)`)
- **Vectors**: 1-dimensional tensors (`torch.tensor([7, 7])`)
- **Matrices**: 2-dimensional tensors
- **Tensors**: n-dimensional arrays

### 2. Tensor Properties
- `.ndim` - Number of dimensions
- `.shape` - Shape of tensor
- `.dtype` - Data type
- `.device` - Device location (CPU/GPU)

### 3. Tensor Operations
- Creation methods (`torch.zeros()`, `torch.ones()`, `torch.rand()`)
- Mathematical operations (addition, multiplication, matrix multiplication)
- Reshaping and manipulation
- Indexing and slicing

### 4. GPU Acceleration
- Moving tensors between CPU and GPU
- Device-agnostic code patterns
- Performance considerations

## Files in this Section
- `00.ipynb` - Main notebook with PyTorch fundamentals
- `00_pytorch_fundamentals_exercises.ipynb` - Practice exercises and solutions

## Skills Developed
- Tensor creation and manipulation
- Understanding PyTorch's computational graph
- Device management (CPU/GPU)
- Foundation for building neural networks
- PyTorch debugging and troubleshooting

## Prerequisites
- Basic Python programming
- Understanding of linear algebra concepts
- Familiarity with NumPy (helpful but not required)

## Next Steps
After completing this section, you'll be ready to move on to:
- PyTorch Workflow (Section 01)
- Building your first neural networks
- Understanding the machine learning pipeline

## Key Takeaways
This foundational section establishes the core tensor operations that underpin all deep learning workflows in PyTorch. Understanding these concepts is crucial for building, training, and deploying neural networks effectively.
