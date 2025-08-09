# 02 - PyTorch Neural Network Classification

## Overview
This section focuses on neural network classification using PyTorch, applying the workflow from Section 01 to solve classification problems. It introduces key concepts for multi-class classification and performance evaluation.

## Learning Objectives
- Build neural networks for classification tasks
- Understand classification-specific loss functions and metrics
- Implement multi-class classification workflows
- Visualize classification boundaries and performance
- Work with probability outputs and class predictions

## Key Concepts Covered

### 1. Classification Fundamentals
- Binary vs multi-class classification
- One-hot encoding and class labels
- Probability distributions and softmax activation
- Classification decision boundaries

### 2. Model Architecture for Classification
- Output layer design for classification
- Activation functions (`nn.ReLU`, `nn.Sigmoid`, `nn.Softmax`)
- Hidden layer sizing considerations
- Architecture patterns for different classification tasks

### 3. Loss Functions for Classification
- **Cross-Entropy Loss** (`nn.CrossEntropyLoss`)
- Why cross-entropy is preferred for classification
- Loss function behavior and interpretation
- Comparing different loss functions

### 4. Classification Metrics
- **Accuracy**: Correct predictions / Total predictions
- Precision, Recall, and F1-Score concepts
- Confusion matrices for detailed analysis
- Metric calculation and interpretation

### 5. Model Evaluation
- Train vs validation accuracy tracking
- Overfitting detection in classification
- Learning curves and performance visualization
- Model comparison techniques

### 6. Advanced Techniques
- Handling class imbalance
- Regularization for classification
- Decision boundary visualization
- Prediction confidence analysis

## Files in this Section
- `02.ipynb` - Main classification notebook
- `02_pytorch_classification_exercises.ipynb` - Classification exercises
- `helper_functions.py` - Utility functions for visualization and metrics

## Key Code Patterns

### Classification Model Example
```python
class ClassificationModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features)
        )
    
    def forward(self, x):
        return self.linear_layer_stack(x)
```

### Accuracy Function
```python
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
```

## Skills Developed
- Multi-class classification implementation
- Performance metrics calculation and interpretation
- Decision boundary visualization
- Classification model debugging
- Hyperparameter tuning for classification
- Model evaluation best practices

## Practical Applications
- Image classification (digit recognition, object classification)
- Text classification (sentiment analysis, topic classification)
- Medical diagnosis (disease classification)
- Customer segmentation
- Fraud detection

## Helper Functions Used
- `plot_decision_boundary()` - Visualize model decision boundaries
- `accuracy_fn()` - Calculate classification accuracy
- `plot_predictions()` - Visualize predictions vs ground truth

## Prerequisites
- Completion of Sections 00-01
- Understanding of basic probability concepts
- Familiarity with classification problems

## Next Steps
This classification foundation prepares you for:
- Computer Vision (Section 03) - Image classification
- Working with real datasets (Section 04)
- Transfer learning for classification (Section 06)
- Advanced classification architectures

## Key Takeaways
Classification is one of the most common machine learning tasks. This section provides the foundation for understanding how neural networks learn to categorize data, evaluate their performance, and visualize their decision-making process. The concepts learned here are directly applicable to computer vision and other advanced topics.
