# 04 - PyTorch Custom Datasets

## Overview
This section teaches how to work with custom datasets in PyTorch, moving beyond built-in datasets to real-world data scenarios. It covers data loading, preprocessing, and creating custom Dataset classes for your own image data.

## Learning Objectives
- Load and work with custom image datasets
- Create custom Dataset classes for specific data formats
- Implement data transforms for custom data
- Understand dataset organization and structure
- Build foundations for modular PyTorch code
- Work with real-world data scenarios

## Key Concepts Covered

### 1. Custom Dataset Creation
- **Dataset Structure**: Organizing image data in folders
- **Class-based Organization**: Folder structure for classification
- **Custom Dataset Class**: Inheriting from `torch.utils.data.Dataset`
- **Data Indexing**: Implementing `__getitem__` and `__len__` methods

### 2. Data Loading and Preprocessing
- **Image Loading**: Using PIL and torchvision for image I/O
- **Path Handling**: Working with file paths and directory structures
- **Data Transforms**: Custom transform pipelines for your data
- **Label Creation**: Converting folder names to class labels

### 3. Real-World Data Challenges
- **Data Quality**: Handling inconsistent image sizes and formats
- **Missing Data**: Dealing with corrupted or missing files
- **Memory Management**: Efficient data loading for large datasets
- **Performance Optimization**: Speeding up data loading

### 4. Dataset Visualization
- **Data Exploration**: Understanding your custom dataset
- **Class Distribution**: Analyzing dataset balance
- **Sample Visualization**: Displaying random samples
- **Transform Verification**: Checking preprocessing results

### 5. Integration with Training Pipeline
- **DataLoader Integration**: Using custom datasets with DataLoaders
- **Batch Processing**: Handling custom data in batches
- **Training Loop Compatibility**: Ensuring seamless integration
- **Validation Strategies**: Splitting custom data effectively

## Files in this Section
- `04.ipynb` - Main custom datasets notebook
- `04_pytorch_custom_datasets_exercises.ipynb` - Practice exercises
- `04_pytorch_custom_datasets_exercise_solutions.ipynb` - Exercise solutions
- `04_pytorch_custom_datasets_video.ipynb` - Video walkthrough version
- `pizza_dad.jpeg` - Sample image file
- `trail.py` - Experimental/trial code
- `data/` - Custom dataset directory

## Custom Dataset Class Pattern
```python
class CustomImageDataset(Dataset):
    def __init__(self, targ_dir, transform=None):
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)
    
    def load_image(self, index):
        image_path = self.paths[index]
        return Image.open(image_path)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx
```

## Key Utility Functions
- `find_classes()` - Discover classes from directory structure
- `walk_through_dir()` - Explore dataset organization
- `display_random_images()` - Visualize dataset samples

## Data Organization Best Practices
```
dataset/
├── train/
│   ├── class_1/
│   │   ├── image_1.jpg
│   │   ├── image_2.jpg
│   │   └── ...
│   ├── class_2/
│   │   ├── image_1.jpg
│   │   └── ...
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

## Skills Developed
- Custom Dataset class implementation
- Real-world data handling and preprocessing
- Dataset organization and management
- Integration with PyTorch training pipelines
- Data quality assessment and validation
- Performance optimization for data loading

## Common Dataset Scenarios
- **Image Classification**: Custom image categories
- **Medical Imaging**: Specialized medical datasets
- **Satellite Imagery**: Geospatial data processing
- **Industrial Inspection**: Quality control datasets
- **Art and Style**: Creative AI applications

## Performance Considerations
- **Lazy Loading**: Loading data only when needed
- **Caching Strategies**: Balancing memory and speed
- **Parallel Processing**: Multi-worker data loading
- **Transform Optimization**: Efficient preprocessing pipelines

## Prerequisites
- Completion of Sections 00-03
- Understanding of Python file handling
- Familiarity with image data formats
- Basic knowledge of dataset organization

## Next Steps
This custom dataset foundation prepares you for:
- Going Modular (Section 05) - Organizing code into modules
- Transfer Learning (Section 06) - Using pre-trained models on custom data
- Real-world project development
- Production-ready data pipelines

## Key Takeaways
Working with custom datasets is essential for real-world machine learning applications. This section bridges the gap between educational datasets and practical data science challenges, providing the tools and knowledge needed to work with any image dataset. The patterns learned here are fundamental for building production-ready computer vision systems.
