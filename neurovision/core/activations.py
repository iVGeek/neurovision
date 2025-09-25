"""
Activation functions with numerical stability
"""

import numpy as np

class ActivationFunction:
    """Factory class for activation functions"""
    
    def __init__(self, activation: str):
        self.activation = activation
    
    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass of activation function"""
        if self.activation == 'relu':
            return self._relu(Z)
        elif self.activation == 'sigmoid':
            return self._sigmoid(Z)
        elif self.activation == 'tanh':
            return self._tanh(Z)
        elif self.activation == 'leaky_relu':
            return self._leaky_relu(Z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
    
    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Backward pass of activation function"""
        if self.activation == 'relu':
            return self._relu_backward(dA, Z)
        elif self.activation == 'sigmoid':
            return self._sigmoid_backward(dA, Z)
        elif self.activation == 'tanh':
            return self._tanh_backward(dA, Z)
        elif self.activation == 'leaky_relu':
            return self._leaky_relu_backward(dA, Z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
    
    def _relu(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)
    
    def _relu_backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
    
    def _sigmoid(self, Z: np.ndarray) -> np.ndarray:
        Z = np.clip(Z, -50, 50)  # Prevent overflow
        return 1 / (1 + np.exp(-Z))
    
    def _sigmoid_backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        s = self._sigmoid(Z)
        return dA * s * (1 - s)
    
    def _tanh(self, Z: np.ndarray) -> np.ndarray:
        return np.tanh(Z)
    
    def _tanh_backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return dA * (1 - np.tanh(Z) ** 2)
    
    def _leaky_relu(self, Z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(Z > 0, Z, Z * alpha)
    
    def _leaky_relu_backward(self, dA: np.ndarray, Z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        dZ = np.where(Z > 0, dA, dA * alpha)
        return dZ
