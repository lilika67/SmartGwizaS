from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import numpy as np
import pickle
from typing import Optional
import uvicorn
import os
import requests

# ==== CONFIGURATION ====
MODEL_FILE = "maize_yield_model.pth"
SCALER_X_FILE = "scaler_X.pkl"
SCALER_Y_FILE = "scaler_y.pkl"

#  Google Drive direct download links
MODEL_URL = "https://drive.google.com/uc?id=1TLNhNJTnxIfwU8vyn_TSfhU1H-UnA947"
SCALER_X_URL = "https://drive.google.com/uc?id=1iMsDS21VzSc3u0Gv3Pe2l00oRhOZIQMB"
SCALER_Y_URL = "https://drive.google.com/uc?id=1sXkrxP9dQ76gOpxFjJcruLp53ORGUH94"


def download_file(url, destination):
    """Download file from URL if not already present."""
    if not os.path.exists(destination):
        print(f"Downloading {destination} from {url} ...")
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(
                f"Failed to download {destination} (status {response.status_code})"
            )
        with open(destination, "wb") as f:
            f.write(response.content)
        print(f"{destination} downloaded successfully.")
    else:
        print(f"{destination} already exists. Skipping download.")


# ==== MODEL ARCHITECTURE ====
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


# ==== REQUEST & RESPONSE MODELS ====
class PredictionRequest(BaseModel):
    year: int = Field(..., ge=1985, le=2035, description="Year (1985–2035)")
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


class PredictionResponse(BaseModel):
    predicted_yield: float
    unit: str = "hg/ha"
    warning: Optional[str] = None


# ==== INITIALIZATION ====
app = FastAPI(
    title="SmartGwiza Maize Yield Prediction API",
    description="API for predicting maize yield based on year, pesticides, and temperature",
    version="1.0.0",
)

model = None
scaler_X = None
scaler_y = None


@app.on_event("startup")
async def load_model():
    """Load model and scalers on startup."""
    global model, scaler_X, scaler_y

    try:
        # Download model and scaler files if missing
        download_file(MODEL_URL, MODEL_FILE)
        download_file(SCALER_X_URL, SCALER_X_FILE)
        download_file(SCALER_Y_URL, SCALER_Y_FILE)

        # Load model
        model = MaizeYieldNN()
        model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device("cpu")))
        model.eval()

        # Load scalers
        with open(SCALER_X_FILE, "rb") as f:
            scaler_X = pickle.load(f)
        with open(SCALER_Y_FILE, "rb") as f:
            scaler_y = pickle.load(f)

        # Verify everything is loaded
        if model is None or scaler_X is None or scaler_y is None:
            raise ValueError("Model or scalers failed to load properly.")

        print(" Model and scalers loaded successfully!")

    except Exception as e:
        print(f" Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")


def validate_and_clip_inputs(year: int, pesticides: float, temp: float):
    """Validate inputs and return clipped values with warnings."""
    year_range = (1990, 2023)
    pesticides_range = (97, 2500)
    temp_range = (19.22, 20.29)

    warnings = []

    # Clip values
    year_clipped = max(min(year, year_range[1] + 5), year_range[0])
    pesticides_clipped = max(
        min(pesticides, pesticides_range[1] * 1.2), pesticides_range[0]
    )
    temp_clipped = max(min(temp, temp_range[1] + 0.5), temp_range[0] - 0.5)

    # Warnings
    if temp < temp_range[0] - 0.5 or temp > temp_range[1] + 0.5:
        warnings.append(
            f"Temperature {temp}°C outside training range ({temp_range[0]}–{temp_range[1]}°C)"
        )
    if year < year_range[0] or year > year_range[1] + 5:
        warnings.append(
            f"Year {year} outside training range ({year_range[0]}–{year_range[1] + 5})"
        )
    if pesticides < pesticides_range[0] or pesticides > pesticides_range[1] * 1.2:
        warnings.append(
            f"Pesticides {pesticides} tonnes outside training range ({pesticides_range[0]}–{pesticides_range[1] * 1.2})"
        )

    return (
        year_clipped,
        pesticides_clipped,
        temp_clipped,
        "; ".join(warnings) if warnings else None,
    )


@app.get("/")
async def root():
    return {
        "message": " SmartGwiza Maize Yield Prediction API",
        "endpoints": {
            "/predict": "POST → Make a prediction",
            "/health": "GET → Check API health",
            "/docs": "GET → Interactive API documentation",
        },
    }


@app.get("/health")
async def health_check():
    if model is None or scaler_X is None or scaler_y is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None or scaler_X is None or scaler_y is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        year, pesticides, temp, warning = validate_and_clip_inputs(
            request.year, request.pesticides_tonnes, request.avg_temp
        )

        input_data = np.array([[year, pesticides, temp]])
        input_scaled = scaler_X.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

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
    uvicorn.run(app, host="0.0.0.0", port=8080)
