# DeepFish Project Documentation

## Overview
The DeepFish project leverages advanced machine learning techniques for marine conservation efforts by employing state-of-the-art technologies in object detection and image segmentation. This document provides in-depth information about the methodologies used in the project, along with a discussion of their relevance in marine conservation.

## Technologies Used

### 1. U-Net Segmentation
U-Net is a convolutional neural network architecture designed for biomedical image segmentation. It excels in scenarios with limited training data, making it ideal for marine biological imagery where available data can be scarce. By using a contracting path to capture context and a symmetric expanding path to enable precise localization, U-Net effectively delineates boundaries within images.

### 2. YOLOv8 Detection
You Only Look Once version 8 (YOLOv8) is a real-time object detection system. Unlike traditional detection methods, YOLO processes images in a single evaluation, thus significantly improving speed without compromising accuracy. This capability is crucial for identifying and classifying marine species in dynamic environments.

### 3. Watershed Tracking
This approach segments images based on the topology of the image, effectively distinguishing individual objects in complex scenes. In the context of marine life, watershed tracking allows for the precise monitoring of species, providing critical data for conservation efforts.

### 4. Grad-CAM Interpretability
Gradient-weighted Class Activation Mapping (Grad-CAM) provides insights into the decision-making process of convolutional neural networks. By visualizing which parts of the image impact the model's predictions, researchers can ensure that the model is learning relevant features and can explain its findings in a scientifically rigorous manner.

## 2026 Advances in Machine Learning
The integration of these techniques demonstrates a significant advancement in machine learning applications for environmental monitoring, enhancing our ability to analyze underwater ecosystems more effectively than ever before.

## Marine Conservation Context
With increasing threats to marine ecosystems from overfishing, pollution, and climate change, the DeepFish project aims to utilize these cutting-edge technologies to gather and analyze data that inform conservation strategies. By effectively tracking and assessing biodiversity, we can develop more informed approaches to safeguarding our oceans.

## File Descriptions
- **data/**: Contains datasets used for training and evaluation.
- **src/**: Source code for model training and evaluation.
- **results/**: Output from model predictions and evaluations.

## Workflow Diagram
A detailed workflow diagram illustrating the data processing pipeline is included in the repository under `docs/workflow_diagram.png`.

## Metrics
Performance metrics for our models include precision, recall, F1-score, and mean Average Precision (mAP), which are essential in understanding the efficiency and effectiveness of our models in real-world applications.

## Installation Instructions
To set up the DeepFish project, please follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/NoahHaag/Computer-Vision.git
   cd Computer-Vision
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python main.py
   ```

By following the above instructions, you will have the necessary environment to run the DeepFish project successfully.