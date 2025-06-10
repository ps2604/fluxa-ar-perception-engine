#!/usr/bin/env python3
from setuptools import setup, find_packages
import os

setup(
    name="fluxa-ar-perception-engine",
    version="1.1.0",
    author="Pirassena Sabaratnam",
    author_email="your-personal-email@example.com",
    description="FLUXA: Multi-task AR Perception Engine using FSE Flow Fields",
    long_description=open('README.md').read() if os.path.exists('README.md') else "",
    long_description_content_type="text/markdown",
    url="https://github.com/ps2604/fluxa-ar-perception-engine",
    
    packages=find_packages(),
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    python_requires=">=3.9",
    
    install_requires=[
        "cupy-cuda11x>=12.0.0",
        "numpy>=1.24.0,<1.27.0",
        "opencv-python>=4.8.0,<5.0.0",
        "google-cloud-storage>=2.10.0,<3.0.0",
    ],
    
    entry_points={
        "console_scripts": [
            "fluxa-train=fluxa.flowfield_training_ultra_optimized:ultra_optimized_training_loop",
        ],
    },
)
