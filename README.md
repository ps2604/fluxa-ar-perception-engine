# FSE Native FLUXA — Flow Field Vision Engine
**Author: Pirassena Sabaratnam**

## Overview
A multi-task perception engine for real-time AR, built on a custom **flow-field** architecture. This replaces standard CNN feature extraction with continuous field dynamics — activations evolve as flow fields rather than being computed as static feature maps.

This is the native implementation of the Field Signal Engine (FSE) concept, built from scratch without TensorFlow or PyTorch CNN layers.

## Architecture

### Flow Field Engine
Custom vectorized compute engine using `im2col` GEMM operations, optimized for low-latency inference on NVIDIA hardware via CuPy kernels. All operations work on continuous field representations with associated evolution rates.

### Multi-Task Perception Heads
Four task heads decode the internal flow fields into perception outputs:
- **Semantic Segmentation**: Real-time subject masking
- **Keypoint Detection**: 17 skeletal joint positions
- **Surface Normal Estimation**: Per-pixel 3D orientation using physics-grounded gradient analysis
- **Environment Lighting**: 9-channel scene illumination estimation

### Global Context Module
Generates synchronization signals across the four task heads, ensuring that geometry, lighting, and mask predictions remain mutually consistent. Operates on the shared flow field representation.

## Evaluation
- **Dataset**: COCO2017 (human subjects)
- **Physics-grounded metrics**: Spatial-gradient coherence, unit-length normal validation
- `fsetest3.png`: Sample multi-modal output (keypoints, segmentation, normals, lighting)

## Setup
```bash
pip install -r requirements.txt
pip install -e .
fluxa-train --enable_multi_gpu
```

## License
Apache License 2.0 — see [LICENSE](LICENSE).

---
*Developed May 2025 — Auralith Inc.*
