***

# DeepFish: Computer Vision & Tracking Pipeline 🐟

This repository contains the core architectural components of **DeepFish**, an end-to-end computer vision project designed for detecting, segmenting, and tracking marine life in underwater video feeds. 

These 5 files encapsulate the primary machine learning pipelines, model interpretability, and the full-stack web application used for inference and data tagging.

## Core Files Overview

### 1. [Segmentation.py](cci:7://file:///c:/Users/Noah/OneDrive/Desktop/DeepFish2/Segmentation/Segmentation.py:0:0-0:0)
**The Core Architecture & Training Script**
This file handles the training of the primary semantic segmentation model. 
* **Architecture:** Implements a custom U-Net architecture utilizing a pre-trained `MobileNetV2` as the encoder for efficient feature extraction.
* **Custom Losses:** Integrates advanced loss functions (Dice Loss and Combo Loss) to handle class imbalances, which are common in underwater imagery.
* **Data Pipelines:** Utilizes `tf.data` for heavily optimized, high-performance image loading and batching.

### 2. [realtime_video_pipeline.py](cci:7://file:///c:/Users/Noah/OneDrive/Desktop/DeepFish2/Segmentation/realtime_video_pipeline.py:0:0-0:0)
**Real-Time Inference & Object Tracking**
An end-to-end processing script that runs inference on raw video files.
* **Instance Tracking:** Features a custom [CentroidTracker](cci:2://file:///c:/Users/Noah/OneDrive/Desktop/DeepFish2/Segmentation/realtime_video_pipeline.py:92:0-165:27) to maintain consistent IDs of fish across frames, even through occlusions.
* **Instance Separation:** Applies the Watershed algorithm to separate overlapping semantic masks into distinct instances.
* **Test Time Augmentation (TTA):** Uses TTA (flipping images horizontally and vertically during inference) to robustly increase prediction accuracy on difficult frames.

### 3. [science_demo.py](cci:7://file:///c:/Users/Noah/OneDrive/Desktop/DeepFish2/Segmentation/science_demo.py:0:0-0:0)
**Model Interpretability & Evaluation**
A demonstration script proving *why* the model makes its decisions.
* **Grad-CAM Visualizations:** Extracts intermediate layer feature maps to generate heatmap overlays, highlighting exactly which pixels the model focuses on to identify a fish.
* **Metric Evaluation:** Calculates and visualizes Intersection over Union (IoU) comparing predictions against ground-truth masks to objectively measure performance.

### 4. [app.py](cci:7://file:///c:/Users/Noah/OneDrive/Desktop/DeepFish2/TaggingApp/app.py:0:0-0:0)
**Full-Stack Tagging & Inference API**
A robust Flask-based web application that serves the models and acts as the user interface for dataset building.
* **API Endpoints:** Serves both the YOLOv8 bounding box models and the U-Net segmentation models for real-time predictions.
* **Data Collection Framework:** Allows users to easily tag unknown species and correct bounding boxes, automatically formatting the data to retrain future models.
* **Concurrency:** Implements thread locking to safely handle heavy model loads in a web-server environment.

### 5. [train_poc.py](cci:7://file:///c:/Users/Noah/OneDrive/Desktop/DeepFish2/TaggingApp/training/train_poc.py:0:0-0:0)
**YOLOv8 Fine-Tuning**
A lightweight wrapper script used to quickly bootstrap and fine-tune Ultralytics YOLOv8 models. As new data is collected and tagged via [app.py](cci:7://file:///c:/Users/Noah/OneDrive/Desktop/DeepFish2/TaggingApp/app.py:0:0-0:0), this script rapidly validates improvements in object detection performance.

## Technology Stack
* **Deep Learning:** TensorFlow / Keras, Ultralytics YOLOv8
* **Computer Vision:** OpenCV (cv2), SciPy
* **Backend:** Flask, Python
* **Data Processing:** NumPy, Pandas
