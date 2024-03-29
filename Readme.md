# Handwritten Digit Recognition System

This repository contains code for a handwritten digit recognition system using deep learning models in PyTorch.

## Overview
The system consists of the following components:
- Models: CNN and FCNN implemented in PyTorch
- Training scripts: `train_fcnn.py`, `train_cnn.py`
- Testing script: `test.py`
- Utility functions for data loading and preprocessing
- Main script: `main.py` for training the models

## Usage
1. Ensure you have all dependencies installed by running `pip install -r requirements.txt`.
2. Dataset will automatically be downloaded and extracted when you run the training script.
3. Run the deployment script (`deploy.py`) to load a trained model and predict digits from a test image.

## Deployment Script
The `deploy.py` script performs the following steps:
1. Loads a saved model (FCNN or CNN) if available, else trains a new model using `main()`.
2. Preprocesses the input image using `preprocess_image()`.
3. Makes predictions for each digit in the preprocessed image using the loaded model.
4. Displays the final predictions for each digit.

## Dependencies
- Python 3
- PyTorch
- OpenCV
- NumPy
- Matplotlib

## How to Run
1. Update the `model_path` and `image_path` variables in the script.
2. Run the script. Example: `python deploy.py`

For any issues or inquiries, feel free to contact [Aashish Kumar](mailto:aashish@iiitmanipur.ac.in).