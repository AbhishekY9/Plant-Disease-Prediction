
# Plant Disease Prediction

This project is a web application built using FastAPI that predicts potato diseases from uploaded images. The backend uses a Convolutional Neural Network (CNN) model trained on 2152 images to classify potato leaves into three categories: Early blight, Late blight, and Healthy.
## Table of Contents

Overview

Features

Model Training

Setup Instructions

Usage

Screenshots
## Overview

The purpose of this application is to help farmers and agricultural professionals identify common diseases in potato plants through image classification. The app accepts an image of a potato leaf, processes it through a trained CNN model, and returns the predicted class along with the confidence level.
## Features

Image Upload: Drag and drop or choose an image file for disease prediction.

Real-Time Prediction: Get instant feedback on the health of the potato plant.

User-Friendly Interface: Easy-to-use interface with a visually appealing background.


## Model Training

The underlying model is a Convolutional Neural Network (CNN) trained on a dataset of 2152 images of potato leaves. The model is capable of distinguishing between Early blight, Late blight, and Healthy leaves. The model was trained using the Keras library with TensorFlow as the backend.

## Dataset

Number of Images: 2152

Classes: Early blight, Late blight, Healthy

Architecture: CNN with multiple convolutional layers, pooling layers, and fully connected layers.
# Setup Instructions
# Setup Instructions
## Prerequisites

Python 3.10+

FastAPI

TensorFlow

Uvicorn

PIL 
## Installation


1. Clone the repository:

       git clone: https://github.com/AbhishekY9/Plant-Disease-Prediction.git

       cd Plant-Disease-Prediction

2. Install the required Python packages:

       pip install -r requirements.txt

3. Place the trained model in the saved_models/1/ directory.

4. Start the FastAPI server:

       uvicorn main:app --reload

5. Open your browser and navigate to http://localhost:8000.


## Usage

Upload Image: Click on the "Choose File" button to upload an image of a potato leaf.

Prediction: Once the image is uploaded, click the "Predict" button to get the disease classification.

Result: The predicted class and confidence level will be displayed below the uploaded image.
## Screenshots

<img src="https://github.com/AbhishekY9/Plant-Disease-Prediction/blob/main/fig/Screenshot_1.png" alt="App Screenshot" width="650">
<img src="https://github.com/AbhishekY9/Plant-Disease-Prediction/blob/main/fig/Screenshot_2.png" alt="App Screenshot" width="650">
<img src="https://github.com/AbhishekY9/Plant-Disease-Prediction/blob/main/fig/Screenshot_3.png" alt="App Screenshot" width="650">

