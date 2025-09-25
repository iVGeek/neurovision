"""
Benchmark different architectures
"""

import matplotlib.pyplot as plt
from ..utils.data_loader import generate_complex_dataset, load_sample_datasets
from ..utils.metrics import benchmark_performance, generate_benchmark_report

def benchmark_demo():
    """Benchmark different neural network architectures"""
    print("NeuroVision Benchmark Demo")
    print("=" * 50)
    
    # Generate dataset
    dataset = generate_complex_dataset(n_samples=2000, dataset_type='moons')
    datasets = [('moons', dataset)]
    
    # Define architectures to test
    architectures = [
        [2, 4, 1],           # Simple
        [2, 8, 4, 1],        # Medium
        [2, 16, 8, 4, 1],    # Complex
        [2, 32, 16, 8, 4, 1] # Very complex
    ]
    
    # Run benchmark
    results = benchmark_performance(architectures, datasets, epochs=100, repetitions=1)
    
    # Print results
    print("\nBenchmark Results:")
    for i in range(len(architectures)):
        arch_result = results[f'Arch_{i}_{'x'.join(map(str, architectures[i]))}']
        for ds_name, ds_res in arch_result.items():
            print(f"Architecture {architectures[i]} on {ds_name}: {ds_res['mean_accuracy']:.4f}")

if __name__ == "__main__":
    benchmark_demo()
