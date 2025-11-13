"""
Basic demonstration of NeuroVision
"""

import numpy as np
import matplotlib.pyplot as plt
from ..core.neural_network import NeuralNetwork
from ..visualization.plotter import NeuralNetworkVisualizer
from ..utils.data_loader import generate_complex_dataset

def basic_demo():
    """Basic demonstration of neural network training"""
    print("NeuroVision Basic Demo")
    print("=" * 50)
    
    # Generate dataset
    X_train, X_test, y_train, y_test = generate_complex_dataset(n_samples=1000, dataset_type='moons')
    
    # Create and train neural network
    nn = NeuralNetwork([2, 16, 8, 1], learning_rate=0.1, regularization=0.001)
    
    print("Training neural network...")
    history = nn.train(X_train, y_train.reshape(-1, 1), epochs=200, batch_size=64, verbose=True)
    
    # Visualize results
    visualizer = NeuralNetworkVisualizer()
    fig = visualizer.create_dashboard(nn, X_test, y_test, history)
    plt.show()
    
    # Print final metrics
    from ..utils.metrics import ModelEvaluator
    metrics = ModelEvaluator.calculate_metrics(y_test, nn.predict(X_test).flatten(), nn.forward(X_test).flatten())
    print(f"\nFinal Metrics:")
    for metric, value in metrics.items():
        if hasattr(value, 'shape'):
            continue
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    basic_demo()



###cide 