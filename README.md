# FSE Native FLUXA FF (Flow Field)
**Author: Pirassena Sabaratnam**

## Overview
**FSE Native FLUXA FF** is a multi-task perception engine designed for real-time Augmented Reality (AR) applications. Built on the **Field Signal Engine (FSE)** architecture, it represents a shift from discrete tensor processing to **continuous neural field dynamics**. The model is responsible for extracting high-fidelity scene data—including segmentation, keypoints, surface normals, and environment lighting—providing a unified spatial context for AR synthesis.

## Core Philosophy
The core objective is to achieve **spatio-temporal coherence** in multi-task vision by leveraging physics-informed neural technology. By treating activations as continuous Flow Fields rather than isolated features, the architecture ensures that complex scene properties evolve in a fluid, synchronized manner. This approach prioritizes the stability and physical alignment required for seamless real-time augmentation.

## Architecture & Factual Functionality

### 1. Vectorized Flow Field Engine
The model utilizes a custom `FSEField` structure that integrates data with dynamic field properties (e.g., evolution rates). Operations are executed via a vectorized `im2col` GEMM engine optimized for low-latency inference on NVIDIA hardware.

### 2. Multi-Task Perception Heads (FLIT)
The **Field Layer Integration (FLIT)** heads are responsible for translating internal continuous fields into task-specific outputs:
- **Semantic Segmentation**: Real-time mask generation for subjects.
- **Keypoint Detection**: Identifying 17 skeletal joints for pose tracking.
- **Surface Normal Estimation**: Extracting 3D geometry using physics-grounded gradient analysis.
- **Environment Lighting**: Inferring scene illumination (9-channel representation) from deep bottleneck fields.

### 3. SYNTHA Orchestrator
The **SYNTHA** module acts as the system's global context manager. It generates context signals that synchronize the various perception heads, ensuring that the inferred geometry, lighting, and masks remain mutually consistent across time steps.

## Future Roadmap: PRISM Integration
FLUXA serves as the primary perception layer for a broader AR pipeline involving **FSENativePRISMFF**. In this setup:
1. **FLUXA** extracts the physical context (segmentation, keypoints, normals, lighting).
2. **PRISM** receives these refined Flow Fields to understand the subject's 3D and environmental constraints.
3. **Synthesis**: PRISM uses this context—guided by an image or prompt—to execute real-time visual augmentations or subject transformations.

## Performance & Optimization
- **NVIDIA Hardware Optimization**: Custom CuPy-based kernels and fused operations.
- **Memory Management**: Unified memory pooling to eliminate fragmentation during real-time inference.
- **Physics-Grounded Metrics**: Evaluation includes spatial-gradient coherence and unit-length normal validation to ensure physical realism.

## Evaluation & Test Outputs
The model was trained on the **COCO2017** dataset for high-fidelity human subject perception and validated through testing on real-world subjects. 
- `fsetest3.png`: Demonstrates the integrated multi-modal outputs (Keypoints, Segmentation, Normals, and Lighting) on a human subject.

## Setup & Usage
1. Install requirements: `pip install -r requirements.txt`
2. Install the package in editable mode: `pip install -e .`
3. Configure `job_config.yaml` for your environment (GCP Vertex AI support included).
4. Run training via the console command: `fluxa-train --enable_multi_gpu`

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---
*Developed in 2025 as part of the Auralith Inc. Research.*
