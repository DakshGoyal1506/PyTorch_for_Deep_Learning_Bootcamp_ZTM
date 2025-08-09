"""
Inference utilities for Vision Transformer model.

This module contains functions for making predictions on new images.
"""
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict
import json

from model.vit_model import create_vit_model
from model.data_utils import create_food101_transforms


class ViTPredictor:
    """Vision Transformer predictor class."""
    
    def __init__(self, model_path: str, class_names: List[str], device: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model
            class_names: List of class names
            device: Device to run inference on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Get transforms
        _, self.transform = create_food101_transforms(img_size=224)
        
        print(f"ViT Predictor initialized on {self.device}")
        print(f"Model loaded from: {model_path}")
        print(f"Number of classes: {len(class_names)}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained model."""
        model = create_vit_model(num_classes=len(self.class_names))
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict_image(self, image_path: str, top_k: int = 5) -> Dict:
        """
        Predict the class of a single image.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            Dict: Prediction results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        # Format results
        results = {
            'image_path': image_path,
            'predictions': []
        }
        
        for i in range(top_k):
            prob = top_probs[0][i].item()
            idx = top_indices[0][i].item()
            class_name = self.class_names[idx]
            
            results['predictions'].append({
                'class': class_name,
                'probability': prob,
                'confidence': f"{prob*100:.2f}%"
            })
        
        return results
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Predict classes for a batch of images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List[Dict]: Prediction results for each image
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path)
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results


def create_predictor_from_checkpoint(model_path: str, 
                                   class_names_path: str = None) -> ViTPredictor:
    """
    Create a predictor from a model checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        class_names_path: Path to class names file (optional)
        
    Returns:
        ViTPredictor instance
    """
    # Load class names
    if class_names_path and Path(class_names_path).exists():
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        # Default Food-101 class names (you might want to save these during training)
        from model.data_utils import download_food101, get_food101_class_names
        dataset_path = download_food101("data")
        class_names = get_food101_class_names(dataset_path)
    
    return ViTPredictor(model_path, class_names)


def demo_inference(model_path: str, image_path: str):
    """
    Demo function for inference.
    
    Args:
        model_path: Path to trained model
        image_path: Path to test image
    """
    print("=" * 60)
    print("🔍 Vision Transformer Inference Demo")
    print("=" * 60)
    
    # Create predictor
    predictor = create_predictor_from_checkpoint(model_path)
    
    # Make prediction
    print(f"\nPredicting image: {image_path}")
    results = predictor.predict_image(image_path, top_k=5)
    
    print(f"\n📊 Top 5 Predictions:")
    for i, pred in enumerate(results['predictions'], 1):
        print(f"  {i}. {pred['class']}: {pred['confidence']}")
    
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ViT Inference Demo")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to test image")
    parser.add_argument("--class_names", type=str, help="Path to class names file")
    
    args = parser.parse_args()
    
    demo_inference(args.model_path, args.image_path)
