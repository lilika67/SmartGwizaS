SmartGwiza - Maize Yield Prediction System
Mission
SmartGwiza is an AI-powered web platform designed to assist farmers, agricultural experts, and policymakers by providing accurate historical crop yield data and future predictions. This initial solution focuses on predicting maize yield (hg/ha) in Rwanda, supporting agricultural risk management and informed decision-making.
Overview
This repository contains a machine learning project to predict maize yield in Rwanda using historical data from 1990 to 2023. SmartGwiza is built with a three-part architecture: a frontend for user interaction, a backend using FastAPI for API services, and a machine learning component for predictive modeling. The dataset includes features such as Year, pesticides_tonnes, and avg_temp, aimed at enhancing agricultural planning and policy development. The project implements three models: a custom Neural Network (MaizeYieldNN), Linear Regression, and Polynomial Regression (Degree 2). The Neural Network, with an R² score of 0.931, MSE of 335,306, and MAE of 315, has been identified as the best-performing model and is saved for predictions.
Dataset
The dataset is a CSV file with 33 rows, including:

Year: 1990–2023 (gap in 2003)
hg/ha_yield: 10,252–22,845 hg/ha
pesticides_tonnes: 97–2,500 tonnes
avg_temp: 19.22–20.29°C

The data is embedded in the Jupyter notebook for reproducibility but can be updated with an external file if needed.
Features

Data Analysis: Exploratory Data Analysis (EDA) with visualizations (histograms, correlation matrix, and scatter plots) to uncover yield patterns.
Model Training: Implements and compares three predictive models using PyTorch (Neural Network) and NumPy (Linear and Polynomial Regression).
Prediction Tool: Offers a predict_yield function with input validation to handle out-of-range values, ensuring robust predictions.

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
fastapi
uvicorn


Set Up Environment (Optional):

Use a virtual environment for isolation:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate





Usage

Run the Jupyter Notebook:

Open SmartGwizaS.ipynb in Jupyter Notebook or Google Colab.
Execute all cells sequentially to load data, perform EDA, train models, and evaluate performance.


Inputs Validation:

Inputs are validated and clipped to ranges: Year (1990–2028), pesticides_tonnes (97–3,000), avg_temp (18.72–20.79°C). Warnings are issued for out-of-range values.


Making Predictions and Retraining via Swagger Local URL:

Access the FastAPI Swagger UI locally at: http://localhost:8000/docs to make predictions or retrain the model.
To run the FastAPI locally, use:uvicorn main:app --reload

Ensure a main.py file with FastAPI setup is present (e.g., integrating the predict_yield function).



How to Run the FastAPI in Production

Deploy the FastAPI backend and access the Swagger UI at: Swagger UI.

Deployment

Machine Learning: The predictive models, including the saved maize_yield_model.pth, are developed and trained using Jupyter notebooks.
Backend: Implemented with FastAPI and deployed on Render.
Frontend: Developed as a user-friendly web app and deployed on Vercel.

Frontend Repository

Link: https://github.com/lilika67/SmartGwiza-system.git

How to Run the Frontend App

Test the user-friendly web app at: Frontend Link. This app enables easy access to all functionalities for predicting maize yields.

Model Details

Neural Network (MaizeYieldNN):
Architecture: 3 input nodes, two hidden layers (10 nodes each) with ReLU, 1 output node.
Training: 200 epochs, Adam optimizer, MSE loss.
Performance: R² 0.931, MSE 335,306, MAE 315.


Linear Regression: Simple least squares fit, R² 0.931, MSE 338,282, MAE 395.
Polynomial Regression (Degree 2): Includes quadratic and interaction terms, R² 0.705, MSE 1,434,811, MAE 967.

Saved Model
The best model (maize_yield_model.pth) is saved after training and loaded for predictions. Ensure the file is in the working directory.
Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make changes and commit (git commit -m "Description of changes").
Push to the branch (git push origin feature-branch).
Open a Pull Request with a clear description of your changes.

Please follow PEP 8 style guidelines and include tests or documentation updates.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Thanks to xAI for providing Grok AI assistance in developing this project.
Inspired by agricultural studies and data from Rwanda.
