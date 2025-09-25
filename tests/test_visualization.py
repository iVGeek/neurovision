"""
Unit tests for visualization module
"""

import numpy as np
import matplotlib.pyplot as plt
from neurovision.visualization.plotter import NeuralNetworkVisualizer
from neurovision.core.neural_network import NeuralNetwork

class TestVisualization:
    """Test cases for visualization functions"""
    
    def test_plotter_initialization(self):
        """Test visualizer initialization"""
        visualizer = NeuralNetworkVisualizer()
        assert visualizer.style == 'default'
        
        visualizer_dark = NeuralNetworkVisualizer(style='dark')
        assert visualizer_dark.style == 'dark'
    
    def test_dashboard_creation(self):
        """Test dashboard creation"""
        # Create a simple network and dummy history
        nn = NeuralNetwork([2, 4, 1])
        X = np.random.randn(100, 2)
        y = np.random.randint(0, 2, 100)
        
        history = {
            'loss': [0.5, 0.4, 0.3],
            'accuracy': [0.6, 0.7, 0.8],
            'epoch_times': [0.1, 0.1, 0.1]
        }
        
        visualizer = NeuralNetworkVisualizer()
        fig = visualizer.create_dashboard(nn, X, y, history)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
