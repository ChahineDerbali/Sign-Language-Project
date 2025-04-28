# ASL Alphabet Image Classification

This project builds a deep learning model to classify images of the American Sign Language (ASL) alphabet using TensorFlow and Keras. It covers data preprocessing, augmentation, model training, and evaluation.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Improvements](#improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The goal is to recognize ASL alphabet signs from images. The workflow includes:
- Organizing and preprocessing the dataset
- Applying data augmentation
- Training a Convolutional Neural Network (CNN)
- Evaluating model performance

---

## Dataset

- **Training Data:** Images of ASL signs in class folders
- **Testing Data:** Images for evaluating model performance

Preprocessing steps:
- Normalizing pixel values
- Rearranging test images into class folders

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ASL-Alphabet-Classification.git
   cd ASL-Alphabet-Classification