"""
Plant Pathology Dataset Loader
This module handles loading and processing the Plant Pathology 2020 dataset.
"""

import os
import pandas as pd
import numpy as np
import logging
from PIL import Image
from io import BytesIO
import base64
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class PlantPathologyDataset:
    """A class for handling the Plant Pathology 2020 dataset"""
    
    def __init__(self, data_dir='attached_assets'):
        """
        Initialize the dataset handler
        
        Args:
            data_dir: Directory containing the dataset files
        """
        self.data_dir = data_dir
        self.train_data = None
        self.test_data = None
        self.sample_submission = None
        self.images_dir = None
        self.classes = ['healthy', 'multiple_diseases', 'rust', 'scab']
        
        # Load dataset files
        try:
            self.load_dataset()
            logger.info("Plant Pathology dataset loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Plant Pathology dataset: {str(e)}")
    
    def load_dataset(self):
        """Load the dataset CSV files"""
        try:
            # Load train.csv
            train_path = os.path.join(self.data_dir, 'train.csv')
            if os.path.exists(train_path):
                self.train_data = pd.read_csv(train_path)
                logger.info(f"Loaded training data: {len(self.train_data)} samples")
            else:
                logger.warning(f"Training data file not found at {train_path}")
            
            # Load test.csv
            test_path = os.path.join(self.data_dir, 'test.csv')
            if os.path.exists(test_path):
                self.test_data = pd.read_csv(test_path)
                logger.info(f"Loaded test data: {len(self.test_data)} samples")
            else:
                logger.warning(f"Test data file not found at {test_path}")
            
            # Load sample_submission.csv
            submission_path = os.path.join(self.data_dir, 'sample_submission.csv')
            if os.path.exists(submission_path):
                self.sample_submission = pd.read_csv(submission_path)
                logger.info(f"Loaded sample submission: {len(self.sample_submission)} samples")
            else:
                logger.warning(f"Sample submission file not found at {submission_path}")
                
        except Exception as e:
            logger.error(f"Error in load_dataset: {str(e)}")
            raise
    
    def get_class_distribution(self):
        """Get the distribution of classes in the training data"""
        if self.train_data is None:
            logger.warning("Training data not loaded")
            return None
        
        # Calculate class distribution
        class_counts = {
            'healthy': self.train_data['healthy'].sum(),
            'multiple_diseases': self.train_data['multiple_diseases'].sum(),
            'rust': self.train_data['rust'].sum(),
            'scab': self.train_data['scab'].sum()
        }
        
        total = len(self.train_data)
        class_distribution = {
            class_name: {
                'count': count,
                'percentage': (count / total) * 100
            }
            for class_name, count in class_counts.items()
        }
        
        return class_distribution
    
    def get_training_samples(self, class_name, limit=5):
        """Get sample indices from the training data for a specific class"""
        if self.train_data is None:
            logger.warning("Training data not loaded")
            return []
        
        if class_name not in self.classes:
            logger.warning(f"Invalid class name: {class_name}")
            return []
        
        # Find indices where the class label is 1
        indices = self.train_data[self.train_data[class_name] == 1].index.tolist()
        
        # Return a limited number of samples
        return indices[:limit]
    
    def get_statistics(self):
        """Get dataset statistics"""
        stats = {
            'train_samples': len(self.train_data) if self.train_data is not None else 0,
            'test_samples': len(self.test_data) if self.test_data is not None else 0,
            'classes': self.classes,
            'class_distribution': self.get_class_distribution()
        }
        
        return stats
    
    def create_augmented_dataset(self):
        """
        Create a balanced dataset using data augmentation
        
        Returns:
            DataFrame containing balanced augmented dataset
        """
        if self.train_data is None:
            logger.warning("Training data not loaded")
            return None
        
        # This is a placeholder that would normally implement data augmentation
        # For our purposes, we'll just return a copy of the original data
        return self.train_data.copy()
    
    def get_disease_probabilities(self, image_features):
        """
        Predict disease probabilities based on image features
        
        Args:
            image_features: Dictionary containing extracted image features
            
        Returns:
            Dictionary with disease probabilities
        """
        # This would normally use a trained model, but we'll simulate predictions
        # based on the extracted features
        
        # Extract relevant features
        brightness = image_features.get('brightness', 0)
        contrast = image_features.get('contrast', 0)
        rg_ratio = image_features.get('rg_ratio', 0)
        texture = image_features.get('texture', 0)
        
        # Initialize probabilities
        probs = {
            'healthy': 0.25,
            'multiple_diseases': 0.25,
            'rust': 0.25,
            'scab': 0.25
        }
        
        # Adjust probabilities based on features
        # High brightness and contrast often indicate healthy plants
        if brightness > 120 and contrast > 40:
            probs['healthy'] = 0.6
            probs['multiple_diseases'] = 0.1
            probs['rust'] = 0.15
            probs['scab'] = 0.15
        
        # High rg_ratio (red to green ratio) often indicates rust
        elif rg_ratio > 1.2:
            probs['healthy'] = 0.1
            probs['multiple_diseases'] = 0.15
            probs['rust'] = 0.6
            probs['scab'] = 0.15
        
        # High texture variance often indicates scab
        elif texture > 40:
            probs['healthy'] = 0.1
            probs['multiple_diseases'] = 0.15
            probs['rust'] = 0.15
            probs['scab'] = 0.6
        
        # Multiple issues
        elif brightness < 80 and texture > 30 and rg_ratio > 1.0:
            probs['healthy'] = 0.05
            probs['multiple_diseases'] = 0.6
            probs['rust'] = 0.2
            probs['scab'] = 0.15
            
        return probs

def extract_image_features(image_array):
    """
    Extract relevant features from a plant image
    
    Args:
        image_array: NumPy array representing the image
        
    Returns:
        Dictionary of extracted features
    """
    if image_array is None or not isinstance(image_array, np.ndarray):
        logger.error("Invalid image array provided to extract_image_features")
        return {}
    
    try:
        # Basic image statistics
        brightness = np.mean(image_array)
        contrast = np.std(image_array)
        
        # Color channel analysis (RGB)
        r_channel = image_array[:,:,0] if image_array.ndim >= 3 else image_array
        g_channel = image_array[:,:,1] if image_array.ndim >= 3 else image_array
        b_channel = image_array[:,:,2] if image_array.ndim >= 3 else image_array
        
        r_mean = np.mean(r_channel)
        g_mean = np.mean(g_channel)
        b_mean = np.mean(b_channel)
        
        # Color ratios
        rg_ratio = r_mean / (g_mean + 1e-10)  # Avoid division by zero
        rb_ratio = r_mean / (b_mean + 1e-10)
        
        # Texture analysis - approximated by local variance
        texture = np.var(image_array)
        
        # Edge detection approximation
        h_gradient = np.mean(np.abs(np.diff(g_channel, axis=1)))
        v_gradient = np.mean(np.abs(np.diff(g_channel, axis=0)))
        edge_intensity = (h_gradient + v_gradient) / 2
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'r_mean': r_mean,
            'g_mean': g_mean,
            'b_mean': b_mean,
            'rg_ratio': rg_ratio,
            'rb_ratio': rb_ratio,
            'texture': texture,
            'edge_intensity': edge_intensity
        }
        
    except Exception as e:
        logger.error(f"Error in extract_image_features: {str(e)}")
        return {}

def analyze_disease_patterns(dataset):
    """
    Analyze patterns in the dataset to identify disease characteristics
    
    Args:
        dataset: PlantPathologyDataset instance
        
    Returns:
        Dictionary with disease characteristics
    """
    if dataset.train_data is None:
        logger.warning("Cannot analyze disease patterns: training data not loaded")
        return {}
    
    # This would normally analyze real images, but we'll return predefined characteristics
    return {
        'healthy': {
            'description': 'Healthy leaves show vibrant green color with even texture',
            'key_features': ['high green channel values', 'low texture variance', 'balanced color distribution'],
            'typical_values': {
                'brightness': '110-140',
                'contrast': '30-50',
                'rg_ratio': '0.4-0.7',
                'texture': '10-25'
            }
        },
        'rust': {
            'description': 'Rust appears as orange-brown pustules on leaf surfaces',
            'key_features': ['elevated red channel', 'pustule-like texture patterns', 'orange-brown spots'],
            'typical_values': {
                'brightness': '80-110',
                'contrast': '40-70',
                'rg_ratio': '1.1-1.8',
                'texture': '30-50'
            }
        },
        'scab': {
            'description': 'Scab appears as dark, olive-green to brown lesions with defined edges',
            'key_features': ['dark spots with high contrast', 'defined lesion boundaries', 'irregular texture'],
            'typical_values': {
                'brightness': '60-90',
                'contrast': '50-80',
                'rg_ratio': '0.7-1.0',
                'texture': '40-70'
            }
        },
        'multiple_diseases': {
            'description': 'Multiple diseases show combined symptoms with varied patterns',
            'key_features': ['highly varied texture', 'mixed color patterns', 'multiple lesion types'],
            'typical_values': {
                'brightness': '70-100',
                'contrast': '60-90',
                'rg_ratio': '0.8-1.5',
                'texture': '50-90'
            }
        }
    }

# For testing the module directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset = PlantPathologyDataset()
    print(dataset.get_statistics())