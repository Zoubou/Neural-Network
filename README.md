# 📘 Neural Network – Alzheimer’s Prediction  
Machine Learning · Feature Selection with Genetic Algorithms · Keras · Python

This repository contains two main components:

1. **Part 1 – Neural Network (NN) model** for binary classification (Alzheimer’s vs. Non-Alzheimer’s)  
2. **Part 2 – Genetic Algorithm (GA)** for feature selection, optimizing the input set for the NN from Part 1  

The project is structured for **Google Colab**, using datasets stored on Google Drive.

---

# How to Run the Project

You can run the project **either on Google Colab (recommended)** or **locally**.

---

# OPTION 1 — Run on Google Colab (Recommended)

Both `.py` files are written for **Google Colab**, including:

- `from google.colab import drive`
- `drive.mount('/content/drive')`
- Dataset paths like  
  `/content/drive/MyDrive/...`

### Steps

1️⃣ Upload the folder to Google Drive  
2️⃣ Open Google Colab  
3️⃣ Upload the scripts  
4️⃣ Mount Google Drive  
5️⃣ Run all cells  

---

# OPTION 2 — Run Locally (VSCode / PyCharm)

Install requirements:

pip install numpy pandas scikit-learn keras tensorflow joblib matplotlib

Replace Google Drive paths with local file paths.

Run:

python Part1_NN.py  
python Part2_GA.py  

---

# Part 1 — Neural Network (Part1_NN.py)

- Loads dataset  
- Scaling, preprocessing  
- Keras Sequential Model  
- SGD optimizer  
- Accuracy, loss, MSE  
- Saves results and model  

---

# Part 2 — Genetic Algorithm (Part2_GA.py)

- Reduces 34 input features  
- Chromosome = feature mask  
- Fitness = NN accuracy  
- Selection, crossover, mutation  
- Produces best feature subset  
- Plots fitness curves  

---

# ⚙️ Requirements

pip install numpy pandas scikit-learn tensorflow keras joblib matplotlib

---

# 📈 Outputs

- Neural network metrics  
- Accuracy graphs  
- GA evolution curves  
- Best feature subset  

