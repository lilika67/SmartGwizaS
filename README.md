#  SmartGwiza - Maize Yield Prediction System

> **Empowering Rwanda’s agriculture with AI-driven insights.**  
> SmartGwiza is an **AI-powered web platform** that predicts maize yield to support farmers, agricultural experts, and policymakers in **making data-informed decisions**.

---

##  Mission

SmartGwiza aims to **assist agricultural stakeholders** in RRwanda by providing:
- Accurate **historical maize yield data**
- Reliable **future maize yield predictions**
- Tools for **risk management** and **policy planning**

---

##  Overview

SmartGwiza integrates machine learning and modern web technologies to predict **maize yield (hg/ha)** in Rwanda using historical data from **1990 to 2023**.

###  System Architecture
1. **Frontend:** Interactive web app (deployed on Vercel)  
2. **Backend:** FastAPI REST API (deployed on Render)  
3. **Machine Learning Engine:** Predictive modeling using PyTorch  

---

##  Dataset

| Feature | Description | Range |
|----------|--------------|--------|
| **Year** | Crop year | 1990–2023 *(gap in 2003)* |
| **hg/ha_yield** | Maize yield (hg/ha) | 10,252–22,845 |
| **pesticides_tonnes** | Pesticides used (tonnes) | 97–2,500 |
| **avg_temp** | Average annual temperature (°C) | 19.22–20.29 |


---

## 🔍 Features

- **Data Analysis:**  
  Visual exploration with histograms, scatter plots, and correlation matrices.
- **Model Training:**  
  Implements and compares three models:
  - Custom Neural Network (`MaizeYieldNN`)
  - Linear Regression
  - Polynomial Regression (Degree 2)
- **Prediction Tool:**  
  Validated inputs and robust yield predictions.
- **FastAPI Integration:**  
  Real-time predictions via `/predict` endpoint on Swagger UI.

---

## 🧠 Model Performance

| Model | R² Score | MSE | MAE |
|--------|-----------|----------|----------|
| **Neural Network (MaizeYieldNN)** | **0.931** | **335,306** | **315** |
| Linear Regression | 0.931 | 338,282 | 395 |
| Polynomial Regression (Degree 2) | 0.705 | 1,434,811 | 967 |

>  **Best Model:** Neural Network (saved as `maize_yield_model.pth`)

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/lilika67/SmartGwizaS.git
cd SmartGwizaS

2. Install Dependencies

Ensure you have Python 3.8+ and pip installed, then run:

pip install -r requirements.txt

requirements.txt includes:
pandas
numpy
matplotlib
seaborn
scikit-learn
torch
fastapi
uvicorn

3. (Optional) Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

 Usage
-Run the Jupyter Notebook

-Open and execute all cells in:

SmartGwizaS.ipynb


This loads the data, performs EDA, trains models, and evaluates performance.

 Input Validation
Input	Valid Range
Year	1990–2035
pesticides_tonnes	97–3,000
avg_temp	18.72–20.79°C

Running FastAPI
Run Locally
uvicorn main:app --reload


Then open http://localhost:8000/docs
 to:

 Deployment
Component	Technology	Platform
Machine Learning	PyTorch, scikit-learn	Jupyter Notebook
Backend	FastAPI	Render
Frontend	React / Next.js	Vercel
Frontend Repository

 SmartGwiza Frontend on GitHub

 Live Frontend (User App)

Visit the deployed SmartGwiza web app here:
🔗 Frontend Link (Add your Vercel URL)

🧱 Model Architecture

MaizeYieldNN Neural Network

3 input nodes (Year, pesticides_tonnes, avg_temp)

2 hidden layers (10 neurons each, ReLU activation)

1 output node (Predicted maize yield)

Trained for 5000 epochs using Adam optimizer and MSE loss

🪄 Saved Model

The best model (maize_yield_model.pth) is automatically saved after training and used for predictions.
Ensure this file is present in your working directory when running the FastAPI app.
