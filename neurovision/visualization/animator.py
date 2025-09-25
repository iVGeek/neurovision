"""
Live training visualization with animation
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import Dict, Optional

class LiveTrainingVisualizer:
    """Real-time animation of neural network training"""
    
    def __init__(self, network, X: np.ndarray, y: np.ndarray, 
                 update_interval: int = 10):
        self.network = network
        self.X = X
        self.y = y
        self.update_interval = update_interval
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Live Training Visualization')
        
        # Initialize plots
        self.loss_line, = self.axes[0,0].plot([], [], 'b-', label='Training Loss')
        self.acc_line, = self.axes[0,1].plot([], [], 'g-', label='Accuracy')
        self.contour = None
        self.scatter = None
        
        self.epochs = []
        self.losses = []
        self.accuracies = []
        
        self._setup_plots()
    
    def _setup_plots(self):
        """Setup the initial plots"""
        # Loss plot
        self.axes[0,0].set_xlabel('Epoch')
        self.axes[0,0].set_ylabel('Loss')
        self.axes[0,0].set_title('Training Loss')
        self.axes[0,0].legend()
        self.axes[0,0].grid(True)
        
        # Accuracy plot
        self.axes[0,1].set_xlabel('Epoch')
        self.axes[0,1].set_ylabel('Accuracy')
        self.axes[0,1].set_title('Accuracy')
        self.axes[0,1].legend()
        self.axes[0,1].grid(True)
        self.axes[0,1].set_ylim(0, 1)
        
        # Decision boundary plot
        self.axes[1,0].set_xlabel('Feature 1')
        self.axes[1,0].set_ylabel('Feature 2')
        self.axes[1,0].set_title('Decision Boundary')
        
        # Weight distribution plot
        self.axes[1,1].set_xlabel('Layer')
        self.axes[1,1].set_ylabel('Weight Value')
        self.axes[1,1].set_title('Weight Distribution')
        
        plt.tight_layout()
    
    def update(self, epoch: int, loss: float, accuracy: float):
        """Update the plots with new data"""
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        
        # Update loss and accuracy plots
        self.loss_line.set_data(self.epochs, self.losses)
        self.acc_line.set_data(self.epochs, self.accuracies)
        
        # Update decision boundary
        self._update_decision_boundary()
        
        # Update weight distribution
        self._update_weight_distribution()
        
        # Adjust limits
        self.axes[0,0].relim()
        self.axes[0,0].autoscale_view()
        self.axes[0,1].relim()
        self.axes[0,1].autoscale_view()
        
        self.fig.canvas.draw()
    
    def _update_decision_boundary(self):
        """Update the decision boundary plot"""
        self.axes[1,0].clear()
        
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                           np.linspace(y_min, y_max, 100))
        
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.network.forward(grid)
        Z = Z.reshape(xx.shape)
        
        self.contour = self.axes[1,0].contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
        self.scatter = self.axes[1,0].scatter(self.X[:, 0], self.X[:, 1], c=self.y, 
                                              cmap='RdYlBu', edgecolors='black', s=30)
        self.axes[1,0].set_xlim(x_min, x_max)
        self.axes[1,0].set_ylim(y_min, y_max)
        self.axes[1,0].set_title('Decision Boundary')
    
    def _update_weight_distribution(self):
        """Update the weight distribution plot"""
        self.axes[1,1].clear()
        
        weights = []
        labels = []
        
        for l in range(1, len(self.network.layers)):
            layer_weights = self.network.parameters[f'W{l}'].flatten()
            weights.extend(layer_weights)
            labels.extend([f'Layer {l}'] * len(layer_weights))
        
        # Box plot of weights by layer
        import pandas as pd
        df = pd.DataFrame({'Weights': weights, 'Layer': labels})
        df.boxplot(column='Weights', by='Layer', ax=self.axes[1,1])
        
        self.axes[1,1].set_title('Weight Distribution by Layer')
        self.axes[1,1].set_ylabel('Weight Value')
    
    def start(self):
        """Start the animation"""
        self.animation = FuncAnimation(self.fig, self.update, interval=self.update_interval)
        plt.show()
