"""
Data loading and generation utilities for neural networks
"""

import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List

class DataGenerator:
    """Generate synthetic datasets for testing neural networks"""
    
    @staticmethod
    def generate_moons(n_samples: int = 1000, noise: float = 0.2, 
                      random_state: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate moon-shaped dataset"""
        return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    @staticmethod
    def generate_circles(n_samples: int = 1000, noise: float = 0.1,
                        factor: float = 0.5, random_state: Optional[int] = 42) -> Tuple:
        """Generate concentric circles dataset"""
        return make_circles(n_samples=n_samples, noise=noise, 
                          factor=factor, random_state=random_state)
    
    @staticmethod
    def generate_blobs(n_samples: int = 1000, centers: int = 2, 
                      random_state: Optional[int] = 42) -> Tuple:
        """Generate Gaussian blobs dataset"""
        return make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                                 n_informative=2, n_clusters_per_class=1,
                                 random_state=random_state)
    
    @staticmethod
    def generate_xor(n_samples: int = 1000, noise: float = 0.1) -> Tuple:
        """Generate XOR dataset"""
        np.random.seed(42)
        X = np.random.randn(n_samples, 2) * noise
        y = np.zeros(n_samples)
        
        # Create XOR pattern
        for i in range(n_samples):
            if (X[i, 0] > 0 and X[i, 1] > 0) or (X[i, 0] < 0 and X[i, 1] < 0):
                y[i] = 1
            else:
                y[i] = 0
                
        return X, y
    
    @staticmethod
    def generate_spiral(n_samples: int = 1000, noise: float = 0.5) -> Tuple:
        """Generate spiral dataset"""
        np.random.seed(42)
        n = n_samples // 2
        theta = np.sqrt(np.random.rand(n)) * 2 * np.pi
        
        # First spiral
        r = np.linspace(0, 5, n)
        x1 = r * np.cos(theta) + np.random.randn(n) * noise
        y1 = r * np.sin(theta) + np.random.randn(n) * noise
        
        # Second spiral
        x2 = -r * np.cos(theta) + np.random.randn(n) * noise
        y2 = -r * np.sin(theta) + np.random.randn(n) * noise
        
        X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
        y = np.hstack([np.zeros(n), np.ones(n)])
        
        return X, y

def generate_complex_dataset(n_samples: int = 1000, dataset_type: str = 'moons',
                           test_size: float = 0.2, random_state: Optional[int] = 42) -> Tuple:
    """Generate complex dataset with train/test split"""
    
    if dataset_type == 'moons':
        X, y = DataGenerator.generate_moons(n_samples, random_state=random_state)
    elif dataset_type == 'circles':
        X, y = DataGenerator.generate_circles(n_samples, random_state=random_state)
    elif dataset_type == 'blobs':
        X, y = DataGenerator.generate_blobs(n_samples, random_state=random_state)
    elif dataset_type == 'xor':
        X, y = DataGenerator.generate_xor(n_samples)
    elif dataset_type == 'spiral':
        X, y = DataGenerator.generate_spiral(n_samples)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def normalize_data(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize data to zero mean and unit variance"""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero
    
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    return X_train_norm, X_test_norm

def load_sample_datasets() -> List[Tuple]:
    """Load multiple sample datasets for benchmarking"""
    datasets = []
    
    for dataset_type in ['moons', 'circles', 'blobs', 'xor', 'spiral']:
        try:
            data = generate_complex_dataset(1000, dataset_type)
            datasets.append((dataset_type, data))
        except ValueError as e:
            print(f"Warning: Could not generate {dataset_type}: {e}")
    
    return datasets
