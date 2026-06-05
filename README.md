# AI-Powered Driver Drowsiness Detection System using MobileNetV2

An **Artificial Intelligence (AI) and Computer Vision-based Driver Drowsiness Detection System** that uses **Deep Learning** to classify a driver's state as **Alert** or **Drowsy** in real time. The system leverages **MobileNetV2**, a pre-trained Convolutional Neural Network (CNN), and Transfer Learning techniques to achieve high accuracy while maintaining low computational requirements for deployment on edge devices such as Raspberry Pi.

---

## AI Components Used

This project incorporates multiple Artificial Intelligence techniques:

* **Deep Learning** using MobileNetV2 CNN
* **Transfer Learning** from ImageNet pre-trained weights
* **Computer Vision** for facial image analysis
* **Image Classification** for Alert vs Drowsy prediction
* **Data Augmentation** for improved model generalization
* **Edge AI Deployment** using TensorFlow Lite FP16 optimization

Unlike traditional rule-based systems, the model learns facial patterns associated with drowsiness directly from thousands of training images, enabling intelligent and adaptive predictions.

---

## Problem Statement

Driver fatigue is one of the major causes of road accidents worldwide. Existing monitoring systems often rely on wearable sensors or specialized hardware, making them expensive and inconvenient.

This project aims to develop an **AI-powered real-time drowsiness detection system** that automatically analyzes a driver's facial features through a camera feed and predicts whether the driver is alert or drowsy. The solution is designed to be lightweight, accurate, and deployable on embedded platforms.

---

## Key AI Features

* Deep Learning-based Driver State Classification

* Transfer Learning using MobileNetV2

* Automated Face Detection and Feature Extraction

* Real-Time AI Inference

* TensorFlow Lite Edge AI Deployment

* Multi-Dataset Training for Better Generalization

* High Accuracy (97.55%)

---

## Why This Is an AI Project

This project belongs to the following AI domains:

* Artificial Intelligence (AI)
* Machine Learning (ML)
* Deep Learning (DL)
* Computer Vision (CV)
* Edge AI

The system learns visual representations of drowsiness from data rather than relying on manually programmed rules, making it a true AI-powered solution.


---

## Proposed Overview

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

## Dataset Links

The model was trained using multiple publicly available datasets along with a custom dataset.

| Dataset | Description | Link |
|----------|-------------|------|
| Driver Drowsiness Dataset (DDD) | Driver drowsiness image dataset | [Dataset Link](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd?utm_source=chatgpt.com) |
| NTHU Driver Drowsiness Detection Dataset | National Tsing Hua University CVLab dataset | [Dataset Link](https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/?utm_source=chatgpt.com) |
| Yawn Dataset | Facial yawning image dataset | [Dataset Link](https://www.kaggle.com/datasets/davidvazquezcic/yawn-dataset?utm_source=chatgpt.com) |
| Custom Dataset | Additional custom-collected images used for training | [Google Drive](https://drive.google.com/drive/folders/1TJlU94BjFfWoZj9BMjLEUDucXtG1XIwj?usp=share_link) |

### Dataset Distribution

The final training dataset was created by combining:
- Driver Drowsiness Dataset (DDD)
- NTHU Driver Drowsiness Detection Dataset
- Yawn Dataset
- Custom Dataset

The datasets were automatically merged, preprocessed, face-cropped, resized, and split into training, validation, and testing sets through the project's automated data pipeline.

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
