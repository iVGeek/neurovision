"""
Core neural network implementation with smart optimization
"""

import numpy as np
import time
from typing import List, Optional, Dict, Tuple
from .activations import ActivationFunction

class NeuralNetwork:
    """Intelligent neural network with advanced optimization techniques"""
    
    def __init__(self, layers: List[int], learning_rate: float = 0.01, 
                 activation: str = 'relu', regularization: float = 0.001,
                 optimizer: str = 'adam'):
        self.layers = layers
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.optimizer = optimizer
        self.activation_fn = ActivationFunction(activation)
        self.output_activation = 'sigmoid'  # For binary classification
        
        # Training history
        self.history = {
            'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [],
            'epoch_times': [], 'gradient_norms': []
        }
        
        # Initialize network parameters
        self.parameters = {}

        # Prepare optimizer state containers so parameter initialization can populate them
        self.m = {}
        self.v = {}
        # Adam hyperparameters (kept even if optimizer != 'adam' for simplicity)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0  # Timestep

        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Smart parameter initialization using Xavier/Glorot method"""
        for l in range(1, len(self.layers)):
            # Xavier initialization
            scale = np.sqrt(2.0 / self.layers[l-1])
            
            self.parameters[f'W{l}'] = np.random.randn(
                self.layers[l], self.layers[l-1]) * scale
            self.parameters[f'b{l}'] = np.zeros((self.layers[l], 1))
            
            # Initialize optimizer states
            if self.optimizer == 'adam':
                self.m[f'W{l}'] = np.zeros_like(self.parameters[f'W{l}'])
                self.m[f'b{l}'] = np.zeros_like(self.parameters[f'b{l}'])
                self.v[f'W{l}'] = np.zeros_like(self.parameters[f'W{l}'])
                self.v[f'b{l}'] = np.zeros_like(self.parameters[f'b{l}'])
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward propagation through the network"""
        A = X.T if X.shape[0] > 1 else X.reshape(-1, 1)
        self.cache = {'A0': A}
        
        # Hidden layers
        for l in range(1, len(self.layers) - 1):
            Z = np.dot(self.parameters[f'W{l}'], A) + self.parameters[f'b{l}']
            A = self.activation_fn.forward(Z)
            self.cache[f'Z{l}'] = Z
            self.cache[f'A{l}'] = A
        
        # Output layer
        l = len(self.layers) - 1
        Z = np.dot(self.parameters[f'W{l}'], A) + self.parameters[f'b{l}']
        A = ActivationFunction(self.output_activation).forward(Z)
        self.cache[f'Z{l}'] = Z
        self.cache[f'A{l}'] = A
        
        return A
    
    def compute_cost(self, AL: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """Compute cost with regularization"""
        m = Y.shape[1]
        
        # Cross-entropy cost
        AL = np.clip(AL, 1e-15, 1 - 1e-15)
        cross_entropy = -np.mean(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        
        # L2 regularization
        l2_cost = 0
        for l in range(1, len(self.layers)):
            l2_cost += np.sum(np.square(self.parameters[f'W{l}']))
        
        l2_cost = (self.regularization / (2 * m)) * l2_cost
        total_cost = cross_entropy + l2_cost
        
        return total_cost, cross_entropy
    
    def backward(self, X: np.ndarray, Y: np.ndarray):
        """Backward propagation with gradient computation"""
        m = X.shape[0]
        AL = self.cache[f'A{len(self.layers)-1}']
        
        # Initialize backpropagation
        dAL = - (np.divide(Y.T, AL) - np.divide(1 - Y.T, 1 - AL))
        
        grads = {}
        L = len(self.layers) - 1
        
        # Output layer gradient
        dZ = ActivationFunction(self.output_activation).backward(
            dAL, self.cache[f'Z{L}'])
        
        for l in reversed(range(1, len(self.layers))):
            A_prev = self.cache[f'A{l-1}'] if l > 1 else self.cache['A0']
            
            grads[f'dW{l}'] = (1/m) * np.dot(dZ, A_prev.T) + \
                             (self.regularization/m) * self.parameters[f'W{l}']
            grads[f'db{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            
            if l > 1:
                dA_prev = np.dot(self.parameters[f'W{l}'].T, dZ)
                dZ = self.activation_fn.backward(dA_prev, self.cache[f'Z{l-1}'])
        
        return grads
    
    def update_parameters(self, grads: Dict):
        """Update parameters using selected optimizer"""
        if self.optimizer == 'adam':
            self._adam_update(grads)
        else:  # SGD
            self._sgd_update(grads)
    
    def _sgd_update(self, grads: Dict):
        """Stochastic Gradient Descent update"""
        for l in range(1, len(self.layers)):
            self.parameters[f'W{l}'] -= self.learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * grads[f'db{l}']
    
    def _adam_update(self, grads: Dict):
        """Adam optimizer update"""
        self.t += 1
        for l in range(1, len(self.layers)):
            # Update moments
            self.m[f'W{l}'] = self.beta1 * self.m[f'W{l}'] + (1 - self.beta1) * grads[f'dW{l}']
            self.m[f'b{l}'] = self.beta1 * self.m[f'b{l}'] + (1 - self.beta1) * grads[f'db{l}']
            
            self.v[f'W{l}'] = self.beta2 * self.v[f'W{l}'] + (1 - self.beta2) * np.square(grads[f'dW{l}'])
            self.v[f'b{l}'] = self.beta2 * self.v[f'b{l}'] + (1 - self.beta2) * np.square(grads[f'db{l}'])
            
            # Bias correction
            m_hat_w = self.m[f'W{l}'] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m[f'b{l}'] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v[f'W{l}'] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v[f'b{l}'] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            self.parameters[f'W{l}'] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            self.parameters[f'b{l}'] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        AL = self.forward(X, training=False)
        return (AL > 0.5).astype(int)
    
    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> Dict:
        """Evaluate model performance"""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == Y.T)
        AL = self.forward(X, training=False)
        loss, _ = self.compute_cost(AL, Y.T)
        
        return {'accuracy': accuracy, 'loss': loss}
    
    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int, 
              batch_size: Optional[int] = None, validation_data: Optional[Tuple] = None,
              verbose: bool = True, early_stopping: bool = True, patience: int = 50) -> Dict:
        """Train the neural network with advanced features"""
        if batch_size is None:
            batch_size = X.shape[0]
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Mini-batch training
            indices = np.random.permutation(X.shape[0])
            X_shuffled, Y_shuffled = X[indices], Y[indices]
            
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                Y_batch = Y_shuffled[i:i + batch_size]
                
                # Forward and backward pass
                AL = self.forward(X_batch)
                grads = self.backward(X_batch, Y_batch.reshape(-1, 1))
                self.update_parameters(grads)
            
            # Training metrics
            AL = self.forward(X, training=False)
            train_loss, _ = self.compute_cost(AL, Y.T)
            train_accuracy = np.mean(self.predict(X) == Y.T)
            
            self.history['loss'].append(train_loss)
            self.history['accuracy'].append(train_accuracy)
            self.history['epoch_times'].append(time.time() - epoch_start)
            
            # Validation metrics
            if validation_data is not None:
                X_val, Y_val = validation_data
                val_metrics = self.evaluate(X_val, Y_val.reshape(-1, 1))
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_accuracy'].append(val_metrics['accuracy'])
                
                # Early stopping
                if early_stopping:
                    if val_metrics['loss'] < best_loss:
                        best_loss = val_metrics['loss']
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break
            
            if verbose and epoch % 100 == 0:
                val_info = f", Val Loss: {self.history['val_loss'][-1]:.4f}" if validation_data else ""
                print(f"Epoch {epoch}: Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}{val_info}")
        
        return self.history
