"""
NeuroVision - An intelligent neural network library with real-time visualization
"""

__version__ = "1.0.0"
__author__ = "NeuroVision Team"
__email__ = "neurovision@example.com"

from .core.neural_network import NeuralNetwork
from .visualization.plotter import NeuralNetworkVisualizer
from .utils.data_loader import generate_complex_dataset
from .utils.metrics import benchmark_performance

__all__ = [
    'NeuralNetwork',
    'NeuralNetworkVisualizer', 
    'generate_complex_dataset',
    'benchmark_performance'
]
