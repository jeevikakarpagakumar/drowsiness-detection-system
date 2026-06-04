# Driver Drowsiness Detection System using MobileNetV2

A real-time driver drowsiness detection system that leverages **MobileNetV2** and computer vision techniques to classify a driver's state as **Alert** or **Drowsy** using facial images. The system is optimized for deployment on lightweight embedded platforms such as Raspberry Pi, making it suitable for intelligent transportation and road safety applications.

---

## Overview

Driver fatigue is one of the leading causes of road accidents worldwide. Traditional monitoring methods often require additional sensors, making them expensive and inconvenient. This project provides a camera-based solution that automatically detects driver drowsiness from facial features using deep learning.

The proposed system:

* Detects driver drowsiness in real time.
* Uses only a camera as input.
* Eliminates the need for wearable devices or physiological sensors.
* Runs efficiently on embedded devices.
* Provides high accuracy while maintaining low computational cost.

---

## Features

* Multi-source dataset integration
* Automated face detection and cropping
* End-to-end preprocessing pipeline
* MobileNetV2 transfer learning architecture
* Real-time video inference
* TensorFlow Lite deployment support
* Lightweight and embedded-device friendly
* Robust under varying lighting conditions

---

## Dataset Sources

The model was trained using a combination of publicly available and custom datasets.

| Dataset                         | Description                         |
| ------------------------------- | ----------------------------------- |
| Driver Drowsiness Dataset (DDD) | Kaggle Dataset                      |
| NTHU DDD Dataset                | National Tsing Hua University CVLab |
| Yawn Dataset                    | Facial yawning image dataset        |
| Custom Dataset                  | Team-collected images               |

---

## Project Pipeline

```text
Dataset Collection
        ↓
Data Preprocessing
        ↓
Face Detection
(Dlib + Haar Cascade Fallback)
        ↓
Face Cropping & Resizing
        ↓
RGB Conversion & Normalization
        ↓
Dataset Splitting
        ↓
MobileNetV2 Training
        ↓
Performance Evaluation
        ↓
Real-Time Deployment
```

---

## Data Preprocessing

### Dataset Assembly

* Combines multiple datasets into a unified structure.
* Automatically identifies alert and drowsy classes.
* Eliminates manual organization.

### Face Detection

* Primary detector: Dlib HOG Face Detector
* Fallback detector: Haar Cascade Classifier
* Automatic face cropping with padding
* Image resizing to **160 × 160**

### Dataset Split

| Dataset Portion | Percentage |
| --------------- | ---------- |
| Training        | 80%        |
| Validation      | 10%        |
| Testing         | 10%        |

---

## Model Architecture

The system uses **MobileNetV2**, a lightweight CNN pre-trained on ImageNet.

### Why MobileNetV2?

* Low memory footprint
* Fast inference
* High accuracy
* Suitable for Raspberry Pi deployment

### Training Strategy

#### Phase 1: Feature Extraction

* Freeze MobileNetV2 backbone
* Train classification layers
* Faster convergence

#### Phase 2: Fine Tuning

* Unfreeze last 30 layers
* Train with lower learning rate
* Improves model generalization

---

## Training Configuration

| Parameter                 | Value                |
| ------------------------- | -------------------- |
| Input Size                | 160 × 160            |
| Batch Size                | 64                   |
| Optimizer                 | Adam                 |
| Loss Function             | Binary Cross Entropy |
| Metrics                   | Accuracy, AUC        |
| Initial Learning Rate     | 1e-4                 |
| Fine-Tuning Learning Rate | 3e-5                 |

---

## Results

### Classification Performance

| Class  | Precision | Recall | F1-Score |
| ------ | --------- | ------ | -------- |
| Alert  | 95.90%    | 98.93% | 97.40%   |
| Drowsy | 99.06%    | 96.36% | 97.69%   |

### Overall Performance

| Metric            | Score  |
| ----------------- | ------ |
| Accuracy          | 97.55% |
| Macro F1 Score    | 97.54% |
| Weighted F1 Score | 97.55% |

---

## Deployment

The trained model is converted into **TensorFlow Lite FP16** format.

| Model          | Size   |
| -------------- | ------ |
| Original Model | ~90 MB |
| TFLite FP16    | ~14 MB |

Benefits:

* Reduced storage requirements
* Faster inference speed
* Lower memory consumption
* Raspberry Pi compatibility

---

## Tech Stack

* Python
* TensorFlow
* Keras
* MobileNetV2
* OpenCV
* Dlib
* Haar Cascade Classifier
* NumPy
* TensorFlow Lite

---



## Future Work

* Eye Aspect Ratio (EAR) integration
* Yawn frequency analysis
* Audio alert generation
* Driver behavior analytics dashboard
* ESP32-CAM deployment
* Multi-modal fatigue detection


---

## References

1. Multi-Attention Fusion Drowsiness Detection Model (MAF), arXiv, 2023.
2. Real-Time Driver Drowsiness Detection Using Transformer Architectures, Scientific Reports, 2024.
3. Optimized Driver Fatigue Detection Using Multimodal Neural Networks, Scientific Reports, 2025.
4. Drowsiness Detection in Real-Time via Convolutional Neural Networks, JEAS, 2024.
5. Driver Drowsiness Dataset (DDD).
6. NTHU Driver Drowsiness Detection Dataset.
7. Yawn Dataset.

---

If you find this project useful, consider giving it a star and contributing to future improvements.
