"""
Unit tests for neural network implementation
"""

import numpy as np
import pytest
from neurovision.core.neural_network import NeuralNetwork
from neurovision.core.activations import ActivationFunction

class TestNeuralNetwork:
    """Test cases for NeuralNetwork class"""
    
    def test_initialization(self):
        """Test neural network initialization"""
        nn = NeuralNetwork([2, 4, 1])
        assert nn.layers == [2, 4, 1]
        assert 'W1' in nn.parameters
        assert 'b1' in nn.parameters
        assert 'W2' in nn.parameters
        assert 'b2' in nn.parameters
    
    def test_forward_pass(self):
        """Test forward propagation"""
        nn = NeuralNetwork([2, 4, 1])
        X = np.random.randn(10, 2)
        output = nn.forward(X)
        assert output.shape == (1, 10)
        assert np.all(output >= 0) and np.all(output <= 1)  # Sigmoid output
    
    def test_training(self):
        """Test training process"""
        # Simple XOR problem
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0]).reshape(-1, 1)
        
        nn = NeuralNetwork([2, 4, 1], learning_rate=0.1)
        history = nn.train(X, y, epochs=10, verbose=False)
        
        # Check that loss recorded
        assert 'loss' in history
        assert len(history['loss']) >= 1
    
    def test_prediction(self):
        """Test prediction method"""
        nn = NeuralNetwork([2, 4, 1])
        X = np.random.randn(5, 2)
        predictions = nn.predict(X)
        assert predictions.shape == (1, 5)
        assert np.array_equal(predictions, predictions.astype(bool))  # Binary output

class TestActivationFunctions:
    """Test cases for activation functions"""
    
    def test_relu(self):
        """Test ReLU activation"""
        af = ActivationFunction('relu')
        Z = np.array([-1, 0, 1])
        A = af.forward(Z)
        assert np.array_equal(A, np.array([0, 0, 1]))
    
    def test_sigmoid(self):
        """Test sigmoid activation"""
        af = ActivationFunction('sigmoid')
        Z = np.array([0])
        A = af.forward(Z)
        assert np.isclose(A, 0.5)
    
    def test_tanh(self):
        """Test tanh activation"""
        af = ActivationFunction('tanh')
        Z = np.array([0])
        A = af.forward(Z)
        assert np.isclose(A, 0.0)
