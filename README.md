#  SmartGwiza - Maize Yield Prediction System for Rwanda

## Mission
**SmartGwiza** is an AI-powered web platform designed to assist farmers, agricultural experts, and policymakers by providing accurate historical crop yield data and future predictions.  
This initial solution focuses on predicting **maize yield (hg/ha)** in **Rwanda**, supporting agricultural risk management and informed decision-making.


## Overview
This repository contains a machine learning project to predict maize yield in Rwanda using historical data from **1990 to 2023**.  

SmartGwiza is built with a **three-part architecture**:
1. **Frontend** – User interface for interaction.
2. **Backend (FastAPI)** – API services and endpoints.
3. **Machine Learning component** – Predictive modeling and data analysis.

The dataset includes features such as:
- `Year`
- `pesticides_tonnes`
- `avg_temp`

These features support agricultural planning and policy development.

### Implemented Models
1. **Custom Neural Network (`MaizeYieldNN`)**
2. **Linear Regression**
3. **Polynomial Regression (Degree 2)**

>  The Neural Network achieved the best performance:
- R²: **0.931**
- MSE: **335,306**
- MAE: **315**

The model was saved as **`maize_yield_model.pth`** to be used in making predictions.

##  Dataset
The dataset is a CSV file with **33 rows**, containing:

| Feature | Description | Range |
|----------|--------------|--------|
| `Year` | Year of record | 1990–2023 (gap in 2003) |
| `hg/ha_yield` | Maize yield (hg/ha) | 10,252–22,845 |
| `pesticides_tonnes` | Pesticide use (tonnes) | 97–2,500 |
| `avg_temp` | Average temperature (°C) | 19.22–20.29 |

## Features
- **Data Analysis**  
  Exploratory Data Analysis (EDA) using visualizations such as histograms, scatter plots, and correlation matrices.
  
- **Model Training**  
  Implements and compares three predictive models using PyTorch and NumPy.

- **Prediction Tool**  
  Includes a `predict_yield` function with input validation and range checking.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/lilika67/SmartGwizaS.git
cd SmartGwizaS
```

### 2. Install Dependencies
Ensure Python **3.8+** and `pip` are installed, then run:
```bash
pip install -r requirements.txt
```

Your `requirements.txt` should include:
```
setuptools>=69.0.0
wheel
numpy<2.0
scikit-learn==1.6.1
torch
torchvision  
torchaudio
fastapi
uvicorn[standard]
pydantic
requests
```

### 3. Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```


## Usage

### Run the Jupyter Notebook
Open **`SmartGwizaS.ipynb`** in Jupyter Notebook or Google Colab and run all cells to:
- Load and explore data  
- Train models  
- Evaluate performance  


### Input Validation
Inputs are validated and clipped to safe ranges:
| Input | Range |
|--------|--------|
| `Year` | 1990–2035 |
| `pesticides_tonnes` | 97–3,000 |
| `avg_temp` | 18.72–20.79°C |



##  FastAPI Integration

### Run Locally
To start the FastAPI backend, run:
```bash
uvicorn main:app --reload
```

Then open the interactive **Swagger UI** at:  
 [Swagger link](http://localhost:8070/docs)


###  API Endpoints

#### **GET /predict**
Predict maize yield based on input parameters.

**Parameters:**
- `year` *(int)*: Year (1990–2028)
- `pesticides_tonnes` *(float)*: 97–3,000
- `avg_temp` *(float)*: 18.72–20.79°C


##  Model Details

###  Neural Network (`MaizeYieldNN`)
- **Architecture:** 3 input nodes, 2 hidden layers (10 neurons each, ReLU), 1 output node  
- **Training:** 200 epochs, Adam optimizer, MSE loss  
- **Performance:**  
  - R²: 0.931  
  - MSE: 335,306  
  - MAE: 315  

###  Linear Regression
- **Performance:**  
  - R²: 0.931  
  - MSE: 338,282  
  - MAE: 395  

###  Polynomial Regression (Degree 2)
- **Performance:**  
  - R²: 0.705  
  - MSE: 1,434,811  
  - MAE: 967  


##  Saved Model
The best performing model (`maize_yield_model.pth`) is saved and automatically loaded during predictions.  
Ensure this file is available in your project directory before running the API.


## Deployment

- **Machine Learning:** Developed and trained using Jupyter Notebook.  
- **Backend:** Implemented with FastAPI and deployed on **Render**.  
- **Frontend:** User-friendly web app deployed on **Vercel**.

**Frontend Repository:**  
 [SmartGwiza Frontend](https://github.com/lilika67/SmartGwiza-system.git)


##  Frontend Web App
You can test the frontend interface here:  
 **Frontend Link ([FrontendLink](https://smartgwizasystem.vercel.app/))**  
The web app allows users to input year, pesticide usage, and average temperature to generate yield predictions.

##  Video Demo

To see the SmartGwiza  demo you can  open[Video demo](https://drive.google.com/drive/folders/1UXNqDMK6_AYMFhj8wttsu_a0zuzZb28D?usp=drive_link)


## Summary
SmartGwiza integrates **data science**, **AI**, and **web technologies** to provide actionable insights for Rwanda’s agriculture.  
By combining historical trends and predictive models, it empowers decision makers to plan effectively for the future of maize production.

## Technologies:

** PyTorch • FastAPI • NumPy • Scikit-learn • Vercel • Render  

## Related screenshot of UI

<img width="2880" height="1404" alt="image" src="https://github.com/user-attachments/assets/7a6cfbe8-1e4d-4b50-bfce-8a425b8361e2" />

<img width="2864" height="1394" alt="image" src="https://github.com/user-attachments/assets/296fb94c-9cca-4beb-b5de-2305071f4d0f" />

<img width="2638" height="1432" alt="image" src="https://github.com/user-attachments/assets/0c0dab2a-b754-4c5f-acd6-b06ba7cb7b7a" />



