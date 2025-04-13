Alzheimer's Disease Prediction using Neural Networks

Overview

This project aims to develop a Neural Network (NN) model to predict Alzheimer's disease based on patient data, including biomarkers, medical history, and cognitive evaluations. The dataset used contains 2,149 patients with 35 features, where the last column represents the diagnosis.

Dataset

Source: Alzheimer's Disease Dataset

File: alzheimers_disease_data.csv

Features:

Numerical: Biomarker levels, age, cognitive test scores, etc.

Categorical: Gender, smoking status, education level, etc.

Target Variable: Diagnosis (Alzheimer or Not)

Implementation Steps

1. Data Preprocessing

Handling missing values.

Encoding categorical variables using One-Hot Encoding & Label Encoding.

Normalizing numerical features using StandardScaler (Z-score standardization).

Splitting data into training (80%) and testing (20%) sets.

Implementing 5-Fold Cross Validation using StratifiedKFold.

2. Neural Network Architecture

Input Layer: 34 features.

Hidden Layers: One hidden layer (experimentation with different neuron counts I/2, 2I/3, I, 2I).

Activation Functions: ReLU, Tanh, SiLU.

Output Layer: One neuron with Sigmoid activation.

Loss Function: Binary Cross-Entropy.

Optimizer: Adam (with adjustable learning rate and momentum).

Evaluation Metrics: Accuracy, Cross-Entropy Loss, Mean Squared Error (MSE).

3. Training and Evaluation

Experimenting with different learning rates and momentum values.

Implementing regularization techniques (L1/L2) to prevent overfitting.

Applying early stopping to optimize training time.

Testing the effect of adding multiple hidden layers for performance improvement.
