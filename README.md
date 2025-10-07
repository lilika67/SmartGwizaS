Maize Yield Prediction Project

## Mission

SmartGwiza is an AI-powered webpage which will be helping farmers and other people in the field of agriculture get accurate information about history of crop yield for making decisions related to agricultural risk management and future predictions. For this initial solution we focused on the prediction of maize yield in Rwanda.

Overview

This repository contains a machine learning project designed to predict maize yield (hg/ha) in Rwanda called SmartGwiza based on historical data from 1990 to 2023. The dataset includes features such as Year, pesticides_tonnes, and avg_temp, with the goal of supporting agricultural planning and policy decisions. We implemented three models: a custom Neural Network (MaizeYieldNN), Linear Regression, and Polynomial Regression (Degree 2). The Neural Network, with an R² score of 0.931, MSE of 335,306, and MAE of 315, has been identified as the best-performing model and was saved for predictions.

Dataset

The dataset is a CSV with 33 rows, including:

Year: 1990–2023 (gap in 2003)
hg/ha_yield: 10,252–22,845 hg/ha
pesticides_tonnes: 97–2,500 tonnes
avg_temp: 19.22–20.29°C

Features

Data Analysis: Exploratory Data Analysis (EDA) with visualizations (histograms, correlation matrix and scatter plots) to understand yield patterns.
Model Training: Implements and compares three predictive models using PyTorch (Neural Network) and NumPy (Linear and Polynomial Regression).
Prediction Tool: Provides a function to predict yields for new inputs, with input validation to handle out-of-range values.


Installation
To run this project locally, follow these steps:

Clone the Repository:
  git clone https://github.com/lilika67/SmartGwizaS.git
cd SmartGwizaS

Install Dependencies:Ensure you have Python 3.8+ and pip installed. Then run:
pip install -r requirements.txt

The requirements.txt file should include:
pandas
numpy
matplotlib
seaborn
scikit-learn
torch


Set Up Environment:

Use a virtual environment for isolation

Usage

Run the Jupyter Notebook:

Open SmartGwizaS.ipynb in Jupyter Notebook or Google Colab.
Execute all cells sequentially to load data, perform EDA, train models, and evaluate performance.


Make Predictions:

Use the predict_yield function to forecast yields.

### 3️. Making Predictions and retrain via swagger local  url 
http://localhost:8000/docs


##  How to run the FASTapi on production
To run this fastApi you can use the swagger docs through the link[ Swagger UI](https://ezanai.onrender.com/docs)


Inputs are validated and clipped to ranges: Year (1990–2028), pesticides_tonnes (97–3000), avg_temp (18.72–20.79°C).

Saved Model:

The best model (maize_yield_model.pth) is saved after training and loaded for predictions. 


Model Details

Neural Network (MaizeYieldNN):
Architecture: 3 input nodes, two hidden layers (10 nodes each) with ReLU, 1 output node.
Training: 200 epochs, Adam optimizer, MSE loss.
Performance: R² 0.931, MSE 335,306, MAE 315.


Linear Regression: Simple least squares fit, R² 0.931, MSE 338,282, MAE 395.
Polynomial Regression (Degree 2): Includes quadratic and interaction terms, R² 0.705, MSE 1,434,811, MAE 967.


##  Deployment 
To deploy the EzanAi backend we used render and vercel for frontend

## Frontend repository

Link: https://github.com/lilika67/cropYieldPredictor_fn.git

## How to run this app on frontend
To efficiently use EzanAI app we also created a user friendly web app which will be helping people to test all functionalities.
to test it use [ Frontend link](https://crop-yield-predictor-fn.vercel.app/)



