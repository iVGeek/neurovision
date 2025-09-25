"""
Performance metrics and benchmarking utilities
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ModelEvaluator:
    """Comprehensive model evaluation metrics"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict:
        """Calculate comprehensive classification metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        metrics['true_negative'], metrics['false_positive'], metrics['false_negative'], metrics['true_positive'] = cm.ravel()
        
        # Additional metrics
        metrics['specificity'] = metrics['true_negative'] / (metrics['true_negative'] + metrics['false_positive']) if (metrics['true_negative'] + metrics['false_positive']) > 0 else 0
        
        return metrics
    
    @staticmethod
    def calculate_learning_curves(history: Dict) -> Dict:
        """Analyze learning curves for convergence"""
        curves = {}
        
        if 'loss' in history and history['loss']:
            curves['final_loss'] = history['loss'][-1]
            curves['best_loss'] = min(history['loss'])
            curves['loss_convergence_epoch'] = ModelEvaluator._find_convergence_epoch(history['loss'])
            curves['loss_improvement'] = history['loss'][0] - history['loss'][-1]
        
        if 'accuracy' in history and history['accuracy']:
            curves['final_accuracy'] = history['accuracy'][-1]
            curves['best_accuracy'] = max(history['accuracy'])
            curves['accuracy_convergence_epoch'] = ModelEvaluator._find_convergence_epoch(history['accuracy'], increasing=True)
        
        return curves
    
    @staticmethod
    def _find_convergence_epoch(values: List[float], window: int = 10, threshold: float = 0.001, 
                               increasing: bool = False) -> int:
        """Find when the values converged"""
        if len(values) < window * 2:
            return len(values) - 1
        
        for i in range(len(values) - window):
            recent_values = values[i:i + window]
            earlier_values = values[max(0, i - window):i]
            
            if not earlier_values:
                continue
            
            recent_mean = np.mean(recent_values)
            earlier_mean = np.mean(earlier_values)
            
            if increasing:
                improvement = recent_mean - earlier_mean
            else:
                improvement = earlier_mean - recent_mean
            
            if improvement < threshold:
                return i + window
        
        return len(values) - 1

def benchmark_performance(architectures: List[List[int]], datasets: List[Tuple], 
                         epochs: int = 500, learning_rate: float = 0.01,
                         repetitions: int = 3) -> Dict:
    """Comprehensive benchmarking of different architectures"""
    
    results = {}
    
    for arch_idx, architecture in enumerate(architectures):
        arch_name = f"Arch_{arch_idx}_{'x'.join(map(str, architecture))}"
        results[arch_name] = {}
        
        for dataset_name, (X_train, X_test, y_train, y_test) in datasets:
            dataset_results = []
            
            for rep in range(repetitions):
                from ..core.neural_network import NeuralNetwork
                
                # Create and train network
                nn = NeuralNetwork(architecture, learning_rate=learning_rate)
                
                start_time = time.time()
                history = nn.train(X_train, y_train.reshape(-1, 1), epochs=epochs, 
                                  verbose=False, early_stopping=True, patience=50)
                training_time = time.time() - start_time
                
                # Evaluate
                y_pred = nn.predict(X_test).flatten()
                y_prob = nn.forward(X_test, training=False).flatten()
                
                metrics = ModelEvaluator.calculate_metrics(y_test, y_pred, y_prob)
                learning_curves = ModelEvaluator.calculate_learning_curves(history)
                
                dataset_results.append({
                    'metrics': metrics,
                    'learning_curves': learning_curves,
                    'training_time': training_time,
                    'final_epochs': len(history['loss']),
                    'history': history
                })
            
            # Aggregate results
            results[arch_name][dataset_name] = {
                'mean_accuracy': np.mean([r['metrics']['accuracy'] for r in dataset_results]),
                'std_accuracy': np.std([r['metrics']['accuracy'] for r in dataset_results]),
                'mean_training_time': np.mean([r['training_time'] for r in dataset_results]),
                'mean_final_loss': np.mean([r['learning_curves']['final_loss'] for r in dataset_results]),
                'all_runs': dataset_results
            }
    
    return results

def generate_benchmark_report(results: Dict) -> str:
    """Generate a comprehensive benchmark report"""
    report = "NeuroVision Benchmark Report\n"
    report += "=" * 50 + "\n\n"
    
    for arch_name, arch_results in results.items():
        report += f"Architecture: {arch_name}\n"
        report += "-" * 30 + "\n"
        
        for dataset_name, dataset_result in arch_results.items():
            report += f"Dataset: {dataset_name}\n"
            report += f"  Accuracy: {dataset_result['mean_accuracy']:.4f} (±{dataset_result['std_accuracy']:.4f})\n"
            report += f"  Training Time: {dataset_result['mean_training_time']:.2f}s\n"
            report += f"  Final Loss: {dataset_result['mean_final_loss']:.4f}\n\n"
    
    return report
