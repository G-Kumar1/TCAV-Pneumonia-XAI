# TCAV-Pneumonia-XAI
# Evaluating TCAV-Based Explainable AI for Pneumonia Classification Using ResNet50

This repository contains the implementation of a research study investigating
concept-based and spatial explanations for deep learning-based pneumonia
classification using pediatric chest X-ray images.

## Overview

Deep learning models can achieve high classification performance while
remaining difficult to interpret. This project investigates whether
concept-based explanations using Testing with Concept Activation Vectors
(TCAV) can provide complementary information to spatial explanations
generated using Grad-CAM.

The framework combines:

- ResNet50-based pneumonia classification
- Transfer learning and fine-tuning
- Grad-CAM visualization
- TCAV-based concept analysis
- Statistical validation of concept associations

## Research Objectives

The study investigates:

1. How TCAV-based explanations compare with Grad-CAM explanations.
2. Whether TCAV can provide meaningful concept-level explanations.
3. Which predefined radiological concepts contribute to model predictions.

## Dataset

The experiments use a pediatric chest X-ray dataset containing:

- Normal: 4,273 images
- Pneumonia: 4,273 images
- Total: 8,546 images

The dataset was divided into training, validation, and test subsets.

> Dataset files are not included in this repository.

## Methodology

### 1. Image Preprocessing

- Resize to 224 × 224
- CLAHE preprocessing
- Image normalization
- Training-time augmentation

### 2. Classification Model

An ImageNet-pretrained ResNet50 architecture was used.

The training process consisted of:

1. Feature extraction with the convolutional backbone frozen.
2. Fine-tuning of the deeper convolutional layers using a lower learning rate.

The classification head consisted of:

- Global Average Pooling
- Dense layer (256 units, ReLU)
- Dropout (0.5)
- Sigmoid output

### 3. Grad-CAM

Grad-CAM was used to visualize spatial regions that contributed
to the model's predictions.

### 4. TCAV

TCAV was used to investigate the influence of predefined radiological
concepts:

- Lung Opacity
- Consolidation
- Pleural Effusion

Random concept baselines were used for statistical comparison.

## Results

| Metric | Fine-tuned ResNet50 |
|--------|---------------------|
| Accuracy | 98% |
| ROC-AUC | 0.998 |

### TCAV Results

| Concept | Mean TCAV Score |
|---------|-----------------:|
| Consolidation | 0.9956 |
| Lung Opacity | 0.9819 |
| Pleural Effusion | 0.8913 |

Statistical significance was evaluated using two-sided Welch's t-tests
with a significance threshold of p < 0.01.

## Explainability

Grad-CAM provides a spatial explanation by highlighting image regions
associated with a prediction.

TCAV provides a concept-level explanation by evaluating the directional
sensitivity of the model to predefined concepts.

The two approaches therefore provide complementary forms of
interpretability.

## Research Contribution

This work investigates the use of concept-based explainability alongside
spatial visualization for pneumonia classification and evaluates the
association of predefined radiological concepts with model predictions.

## Limitations

- Three predefined concepts were investigated.
- Experiments were conducted using a single publicly available dataset.
- A single model architecture was evaluated.
- Expert radiologist validation was not performed.

## Future Work

Future research could investigate:

- Additional radiological concepts
- Multiple datasets
- Alternative deep learning architectures
- Expert evaluation of explanations
- Additional concept-based XAI approaches
- Multimodal clinical information

## Technologies

Python  
TensorFlow / Keras  
ResNet50  
Grad-CAM  
TCAV  
OpenCV  
NumPy  
Pandas  
Scikit-learn  
Matplotlib
