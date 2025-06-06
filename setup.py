#!/usr/bin/env python3
"""
FlowField FSE Native Training Setup
==================================

Setup script for FlowField continuous field computation framework.
"""

from setuptools import setup, find_packages
import os

# Read version from environment or default
version = os.environ.get('FLOWFIELD_VERSION', '1.1.0') # Updated version

long_description = "FlowField: True continuous field computation for Float-Native State Elements (FSE) - Corrected Implementation"
if os.path.exists('README.md'):
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name="flowfield-fse-corrected", # Updated name
    version=version,
    author="Pirassena Sabaratnam", # Adapted
    author_email="auralithco@gmail.com",
    description="FlowField: Corrected Continuous Field Computation for FSE Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="your_repository_url_here", # Replace with your repo URL
    
    # find_packages() can be used if you structure your code into a package
    # For now, listing individual modules if they are top-level
    py_modules=[
        'flowfield_core_optimized',
        'flowfield_components',
        'flowfield_fluxa_model',
        'flowfield_training_ultra_optimized',
        'flowfield_async_data_loader',
        'flowfield_advanced_cuda_kernels',
        'metrics_fse'
    ],
    # If you create a 'flowfield' directory with an __init__.py and put your files there:
    # packages=find_packages(where=".", include=['flowfield*']),
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License", # Assuming MIT, update if different
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    python_requires=">=3.9",
    
    install_requires=[
        "cupy-cuda11x>=12.0.0",
        "numpy>=1.24.0,<1.27.0",
        "opencv-python>=4.8.0,<5.0.0", # If using CV for data loading
        "google-cloud-storage>=2.10.0,<3.0.0", # If using GCS
    ],
    
    extras_require={
        "dev": [
            "pytest",
            "flake8",
            "mypy",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "flowfield-train=flowfield_training_ultra_optimized:main_training_loop",
        ],
    },
    
    keywords=[
        "machine-learning",
        "neural-networks", 
        "continuous-fields",
        "gpu-computing",
        "fse",
        "flowfield",
        "cupy"
    ],
    
    project_urls={ # Optional: Update these URLs
        "Bug Reports": "your_repo_url_here/issues",
        "Source": "your_repo_url_here",
    },
    
    include_package_data=True,
    zip_safe=False,
)
