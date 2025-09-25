from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else "NeuroVision"

setup(
    name="neurovision",
    version="1.0.0",
    description="An intelligent neural network library with real-time visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neurovision",
    author="NeuroVision Team",
    author_email="neurovision@example.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="neural-network, machine-learning, deep-learning, visualization",
    package_dir={"": "neurovision"},
    packages=find_packages(where="neurovision"),
    python_requires=">=3.8, <4",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neurovision-demo=neurovision.examples.basic_demo:basic_demo",
            "neurovision-benchmark=neurovision.examples.benchmark_demo:benchmark_demo",
        ],
    },
)
