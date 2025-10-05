from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import numpy as np
import pickle
from typing import Optional
import uvicorn


# Define the model architecture (must match your training)
class MaizeYieldNN(nn.Module):
    def __init__(self):
        super(MaizeYieldNN, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Request model
class PredictionRequest(BaseModel):
    year: int = Field(..., ge=1985, le=2035, description="Year (1985-2035)")
    pesticides_tonnes: float = Field(
        ..., ge=0, le=3000, description="Pesticides in tonnes"
    )
    avg_temp: float = Field(
        ..., ge=19, le=21, description="Average temperature in Celsius"
    )

    class Config:
        schema_extra = {
            "example": {"year": 2024, "pesticides_tonnes": 2600, "avg_temp": 19.6}
        }


# Response model
class PredictionResponse(BaseModel):
    predicted_yield: float
    unit: str = "hg/ha"
    


# Initialize FastAPI
app = FastAPI(
    title="SmartGwiza Maize Yield Prediction API",
    description="API for predicting maize yield based on year, pesticides, and temperature",
    version="1.0.0",
)

# Global variables for model and scalers
model = None
scaler_X = None
scaler_y = None


@app.on_event("startup")
async def load_model():
    """Load model and scalers on startup"""
    global model, scaler_X, scaler_y

    try:
        # Load the trained model
        model = MaizeYieldNN()
        model.load_state_dict(
            torch.load("maize_yield_model.pth", map_location=torch.device("cpu"))
        )
        model.eval()

        from sklearn.preprocessing import StandardScaler


        with open("scaler_X.pkl", "rb") as f:
            scaler_X = pickle.load(f)
        with open("scaler_y.pkl", "rb") as f:
            scaler_y = pickle.load(f)

        print("Model and scalers loaded successfully!")

    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def validate_and_clip_inputs(year: int, pesticides: float, temp: float):
    """Validate inputs and return clipped values with warnings"""
    year_range = (1990, 2023)
    pesticides_range = (97, 2500)
    temp_range = (19.22, 20.29)

    warnings = []

    # Clip with buffers
    year_clipped = max(min(year, year_range[1] + 5), year_range[0])
    pesticides_clipped = max(
        min(pesticides, pesticides_range[1] * 1.2), pesticides_range[0]
    )
    temp_clipped = max(min(temp, temp_range[1] + 0.5), temp_range[0] - 0.5)

    # Generate warnings
    if temp < temp_range[0] - 0.5 or temp > temp_range[1] + 0.5:
        warnings.append(
            f"Temperature {temp}°C outside training range ({temp_range[0]}-{temp_range[1]}°C)"
        )

    if year < year_range[0] or year > year_range[1] + 5:
        warnings.append(
            f"Year {year} outside training range ({year_range[0]}-{year_range[1]+5})"
        )

    if pesticides < pesticides_range[0] or pesticides > pesticides_range[1] * 1.2:
        warnings.append(
            f"Pesticides {pesticides} tonnes outside training range ({pesticides_range[0]}-{pesticides_range[1]*1.2})"
        )

    warning_text = "; ".join(warnings) if warnings else None

    return year_clipped, pesticides_clipped, temp_clipped, warning_text


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Maize Yield Prediction API",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/health": "GET - Check API health",
            "/docs": "GET - Interactive API documentation",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or scaler_X is None or scaler_y is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict maize yield based on input parameters

    - **year**: Year for prediction (1985-2035)
    - **pesticides_tonnes**: Amount of pesticides used in tonnes
    - **avg_temp**: Average temperature in Celsius
    """
    if model is None or scaler_X is None or scaler_y is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Validate and clip inputs
        year_clipped, pesticides_clipped, temp_clipped, warning = (
            validate_and_clip_inputs(
                request.year, request.pesticides_tonnes, request.avg_temp
            )
        )

        # Prepare input
        input_data = np.array([[year_clipped, pesticides_clipped, temp_clipped]])
        input_scaled = scaler_X.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            pred_scaled = model(input_tensor)
            pred = scaler_y.inverse_transform(pred_scaled.numpy())

        predicted_yield = max(float(pred[0][0]), 0)

        return PredictionResponse(
            predicted_yield=round(predicted_yield, 2), warning=warning
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
