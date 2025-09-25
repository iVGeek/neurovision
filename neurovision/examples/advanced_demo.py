"""
Advanced demonstration with live visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from ..core.neural_network import NeuralNetwork
from ..visualization.animator import LiveTrainingVisualizer
from ..utils.data_loader import generate_complex_dataset

def advanced_demo():
    """Advanced demonstration with live training visualization"""
    print("NeuroVision Advanced Demo")
    print("=" * 50)
    
    # Generate complex dataset
    X_train, X_test, y_train, y_test = generate_complex_dataset(n_samples=1000, dataset_type='circles')
    
    # Create neural network
    nn = NeuralNetwork([2, 32, 16, 8, 1], learning_rate=0.1, optimizer='adam', regularization=0.001)
    
    # Create live visualizer
    visualizer = LiveTrainingVisualizer(nn, X_train, y_train, update_interval=200)
    
    print("Starting live training visualization...")
    # We'll train in small increments and update the visualizer manually
    epochs = 200
    batch_size = 64
    
    for epoch in range(epochs):
        # Train one epoch
        _ = nn.train(X_train, y_train.reshape(-1, 1), epochs=1, batch_size=batch_size, verbose=False)
        
        # Update visualization every 5 epochs
        if epoch % 5 == 0:
            loss = nn.history['loss'][-1] if nn.history['loss'] else 0
            accuracy = nn.history['accuracy'][-1] if nn.history['accuracy'] else 0
            visualizer.update(epoch, loss, accuracy)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
    
    plt.show()

if __name__ == "__main__":
    advanced_demo()
