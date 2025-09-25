"""
Advanced visualization for neural networks
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, Optional

class NeuralNetworkVisualizer:
    """Comprehensive visualization suite for neural networks"""
    
    def __init__(self, style: str = 'default'):
        self.style = style
        self._set_style()
    
    def _set_style(self):
        """Set matplotlib style"""
        if self.style == 'dark':
            plt.style.use('dark_background')
            self.colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
        else:
            plt.style.use('seaborn-v0_8')
            self.colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    def create_dashboard(self, network, X: np.ndarray, y: np.ndarray, 
                        history: Dict, figsize: tuple = (16, 12)):
        """Create comprehensive training dashboard"""
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig)
        
        # Plot 1: Training history
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_training_history(ax1, history)
        
        # Plot 2: Decision boundary
        ax2 = fig.add_subplot(gs[0, 1:])
        self._plot_decision_boundary(ax2, network, X, y)
        
        # Plot 3: Accuracy progression
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_accuracy(ax3, history)
        
        # Plot 4: Training time
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_training_time(ax4, history)
        
        # Plot 5: Layer weights distribution
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_weight_distribution(ax5, network)
        
        # Plot 6: Confusion matrix (if validation data)
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_confusion_matrix(ax6, network, X, y)
        
        # Plot 7: Learning rate analysis
        ax7 = fig.add_subplot(gs[2, 1:])
        self._plot_gradient_analysis(ax7, history)
        
        plt.tight_layout()
        return fig
    
    def _plot_training_history(self, ax, history: Dict):
        """Plot training and validation loss"""
        epochs = range(len(history['loss']))
        
        ax.plot(epochs, history['loss'], label='Training Loss', 
                color=self.colors[0], linewidth=2)
        
        if 'val_loss' in history and history['val_loss']:
            ax.plot(epochs, history['val_loss'], label='Validation Loss',
                   color=self.colors[1], linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_decision_boundary(self, ax, network, X: np.ndarray, y: np.ndarray, 
                               resolution: int = 100):
        """Plot decision boundary with data points"""
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                           np.linspace(y_min, y_max, resolution))
        
        # Predict on grid (forward expects (n_samples, n_features))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = network.forward(grid)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        contour = ax.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                           edgecolors='black', s=30, alpha=0.8)
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Decision Boundary')
        plt.colorbar(contour, ax=ax)
    
    def _plot_accuracy(self, ax, history: Dict):
        """Plot accuracy progression"""
        epochs = range(len(history['accuracy']))
        
        ax.plot(epochs, history['accuracy'], label='Training Accuracy',
               color=self.colors[2], linewidth=2)
        
        if 'val_accuracy' in history and history['val_accuracy']:
            ax.plot(epochs, history['val_accuracy'], label='Validation Accuracy',
                   color=self.colors[3], linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_training_time(self, ax, history: Dict):
        """Plot training time per epoch"""
        if 'epoch_times' in history:
            epochs = range(len(history['epoch_times']))
            ax.plot(epochs, history['epoch_times'], color=self.colors[4], linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Time per Epoch')
            ax.grid(True, alpha=0.3)
    
    def _plot_weight_distribution(self, ax, network):
        """Plot distribution of weights across layers"""
        weights = []
        labels = []
        
        for l in range(1, len(network.layers)):
            layer_weights = network.parameters[f'W{l}'].flatten()
            weights.extend(layer_weights)
            labels.extend([f'Layer {l}'] * len(layer_weights))
        
        # Sample for performance if too many weights
        if len(weights) > 10000:
            indices = np.random.choice(len(weights), 10000, replace=False)
            weights = np.array(weights)[indices]
            labels = np.array(labels)[indices]
        
        import pandas as pd
        df = pd.DataFrame({'Weights': weights, 'Layer': labels})
        
        sns.violinplot(data=df, x='Layer', y='Weights', ax=ax, palette=self.colors)
        ax.set_title('Weight Distribution by Layer')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_confusion_matrix(self, ax, network, X: np.ndarray, y: np.ndarray):
        """Plot confusion matrix"""
        predictions = network.predict(X).flatten()
        y_true = y.flatten()
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, predictions)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Pred 0', 'Pred 1'], 
                   yticklabels=['True 0', 'True 1'])
        ax.set_title('Confusion Matrix')
    
    def _plot_gradient_analysis(self, ax, history: Dict):
        """Plot gradient analysis if available"""
        if 'gradient_norms' in history and history['gradient_norms']:
            epochs = range(len(history['gradient_norms']))
            ax.plot(epochs, history['gradient_norms'], color=self.colors[0], linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Norm Progression')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
