# VEGA Inverted Residual Engine (IRE) Accelerator

Hardware-software co-design for task-aware object detection. 
Developed by Team Aeternum for the DVCon India 2026 Design Contest.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Quantized-EE4C2C)
![Target](https://img.shields.io/badge/Target-Genesys--2%20FPGA-orange)
![Processor](https://img.shields.io/badge/Processor-CDAC%20VEGA-brightgreen)

## Overview

Standard object detection models evaluate all objects in a scene, which is computationally inefficient for targeted tasks. This framework performs task-driven object detection, prioritizing objects based on a semantic text prompt and gating irrelevant features at the hardware level. 

This repository contains the software-hardware co-design targeting the CDAC VEGA Processor and the Digilent Genesys-2 (Kintex-7) FPGA.

## Architecture

### 1. Vision-Language Pipeline
* **Vision Backbone:** INT8 Quantized MobileNetV2.
* **Semantic Encoding:** A static text-embedding lookup on the VEGA processor maps one of 14 specific tasks into a binary task mask.
* **Fusion:** Cross-modal spatial attention fuses vision features with the task mask to predict bounding boxes.

### 2. Hardware Subsystem (IRE)
The Inverted Residual Engine (IRE) is an FPGA datapath optimized for depthwise separable convolutions.
* **On-Chip Caching:** Dual-Port BRAMs and Shift Register LUTs function as line buffers to minimize external DDR memory access.
* **Channel Gating Unit (CGU):** The VEGA processor sends the binary task mask to the FPGA. For irrelevant channels, the CGU drives the DSP48E1 `Clock Enable (CE)` low and zeroes the data buses. This eliminates dynamic switching activity for that channel, reducing power and latency.

## Repository Contents

This repository currently holds the PyTorch simulation and Quantization Aware Training (QAT) environment to validate INT8 precision before RTL implementation.

* `TaskAwareEdgeDetector`: MobileNetV2 + Embedding fusion network.
* `QNNPACK`: Simulates 8-bit integer operations for the DSP slices.
* `COCOTasksDataset`: Custom loader translating COCO JSON annotations into normalized coordinates [cx, cy, w, h].
* `Evaluation`: Calculates IoU and visualizes predicted vs. ground-truth bounding boxes.

## Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/team-aeternum/vega-ire-accelerator.git](https://github.com/team-aeternum/vega-ire-accelerator.git)
   cd vega-ire-accelerator
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision matplotlib pillow requests tqdm
   ```

3. Download the dataset:
   Running the evaluation script will automatically download and format the required COCO 2014 Training images (13GB) and task annotations into the `./dataset/` directory.

## Usage

The evaluation script supports two modes defined by the `TEST_MODE` variable:

### Mode 1: COCO Dataset Evaluation
Selects a random image and task from the dataset, runs inference, and calculates IoU.
```python
TEST_MODE = 'random'
```

### Mode 2: Custom Image Inference
Runs inference on a local image using a specified task ID (1-14).
```python
TEST_MODE = 'custom' 
MY_IMAGE_PATH = "./test_image.jpg" 
MY_TASK_ID = 10 

# Optional target box for IoU calculation [cx, cy, w, h]
MY_TARGET_BOX = [0.52, 0.5, 0.2, 0.55] 
```
