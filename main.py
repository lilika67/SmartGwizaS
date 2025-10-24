from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from auth import get_current_user, get_db
import torch
import torch.nn as nn
import numpy as np
import pickle
from typing import Optional, List
import uvicorn
from dotenv import load_dotenv
import os
import requests
import logging
from auth import app as auth_app
from pymongo import MongoClient
from datetime import datetime

# ==== CONFIGURATION ====
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MODEL_FILE = "/Users/liliane/Documents/SmartGwizaS/maize_yield_model.pth"
SCALER_X_FILE = "/Users/liliane/Documents/SmartGwizaS/scaler_X.pkl"
SCALER_Y_FILE = "/Users/liliane/Documents/SmartGwizaS/scaler_y.pkl"

MODEL_URL = (
    "https://drive.google.com/uc?export=download&id=1TLNhNJTnxIfwU8vyn_TSfhU1H-UnA947"
)
SCALER_X_URL = (
    "https://drive.google.com/uc?export=download&id=1iMsDS21VzSc3u0Gv3Pe2l00oRhOZIQMB"
)
SCALER_Y_URL = (
    "https://drive.google.com/uc?export=download&id=1sXkrxP9dQ76gOpxFjJcruLp53ORGUH94"
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url, destination):
    """Download file from URL if not already present."""
    if not os.path.exists(destination):
        logger.info(f"Downloading {destination} from {url}...")
        try:
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to download {destination} (status {response.status_code})",
                )
            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info(f"{destination} downloaded successfully.")
        except Exception as e:
            logger.error(f"Download error for {destination}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")
    else:
        logger.info(f"{destination} already exists. Skipping download.")


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
    token: str = Field(..., description="JWT token from /auth/login")

    class Config:
        schema_extra = {
            "example": {
                "year": 2024,
                "pesticides_tonnes": 2600,
                "avg_temp": 19.6,
                "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            }
        }


class PredictionResponse(BaseModel):
    predicted_yield: float
    unit: str = "hg/ha"
    phone_number: str
    warning: Optional[str] = None


class UserPrediction(BaseModel):
    phone_number: str
    year: int
    pesticides_tonnes: float
    avg_temp: float
    predicted_yield: float
    unit: str = "hg/ha"
    warning: Optional[str] = None
    timestamp: datetime


class ActualYieldSubmission(BaseModel):
    crop_type: str = Field(..., description="Crop type: Maize or Beans")
    harvest_date: str = Field(
        ..., description="Harvest date (month/year, e.g., 10/2025)"
    )
    actual_yield: float = Field(..., ge=0, description="Actual yield (tons/ha)")
    farm_size: float = Field(..., ge=0, description="Farm size (ha)")
    location: str = Field(..., description="Location (e.g., Rwanda district)")
    avg_temp: float = Field(
        ..., ge=15, le=35, description="Average temperature in Celsius"
    )
    notes: Optional[str] = Field(None, description="Additional notes")
    token: str = Field(..., description="JWT token from /auth/login")

    @validator("crop_type")
    def validate_crop_type(cls, value):
        if value not in ["Maize", "Beans"]:
            raise ValueError("Crop type must be 'Maize' or 'Beans'")
        return value

    @validator("harvest_date")
    def validate_harvest_date(cls, value):
        try:
            datetime.strptime(value, "%m/%Y")
        except ValueError:
            raise ValueError("Harvest date must be in MM/YYYY format")
        return value

    class Config:
        schema_extra = {
            "example": {
                "crop_type": "Maize",
                "harvest_date": "10/2025",
                "actual_yield": 5.2,
                "farm_size": 2.5,
                "location": "Kigali",
                "avg_temp": 19.6,
                "notes": "Good harvest due to favorable weather",
                "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            }
        }


class ActualYieldResponse(BaseModel):
    message: str = "Submission successful"
    phone_number: str


class ActualYieldRecord(BaseModel):
    crop_type: str
    harvest_year: int
    actual_yield: float
    avg_temp: float
    phone_number: str
    timestamp: datetime


# ==== INITIALIZATION ====
app = FastAPI(
    title="SmartGwiza Maize Yield Prediction API",
    description="API for predicting maize yield, submitting actual yield, retrieving yields, and user authentication",
    version="1.0.0",
)

# Mount the authentication app
app.include_router(auth_app.router, prefix="/auth", tags=["auth"])

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
scaler_X = None
scaler_y = None


@app.on_event("startup")
async def load_model():
    """Load model and scalers on startup."""
    global model, scaler_X, scaler_y

    try:
        download_file(MODEL_URL, MODEL_FILE)
        download_file(SCALER_X_URL, SCALER_X_FILE)
        download_file(SCALER_Y_URL, SCALER_Y_FILE)

        model = MaizeYieldNN()
        model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device("cpu")))
        model.eval()

        with open(SCALER_X_FILE, "rb") as f:
            scaler_X = pickle.load(f)
        with open(SCALER_Y_FILE, "rb") as f:
            scaler_y = pickle.load(f)

        if model is None or scaler_X is None or scaler_y is None:
            raise ValueError("Model or scalers failed to load properly.")

        logger.info("Model and scalers loaded successfully!")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")


def validate_and_clip_inputs(year: int, pesticides: float, temp: float):
    """Validate inputs and return clipped values with warnings."""
    year_range = (1990, 2023)
    pesticides_range = (97, 2500)
    temp_range = (19.22, 20.29)

    warnings = []

    year_clipped = max(min(year, year_range[1] + 5), year_range[0])
    pesticides_clipped = max(
        min(pesticides, pesticides_range[1] * 1.2), pesticides_range[0]
    )
    temp_clipped = max(min(temp, temp_range[1] + 0.5), temp_range[0] - 0.5)

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


# ==== ENDPOINTS ====
@app.get("/")
async def root():
    return {
        "message": "SmartGwiza Maize Yield Prediction API",
        "endpoints": {
            "/predict": "POST → Make a prediction (requires token in body)",
            "/predictions": "GET → Get predictions (farmer: own predictions; admin: all predictions)",
            "/submit-actual-yield": "POST → Submit actual yield data (requires token in body)",
            "/actual-yields": "GET → Get actual yield submissions (farmer: own; admin: all)",
            "/health": "GET → Check API health",
            "/auth/signup": "POST → Create a new user account",
            "/auth/login": "POST → Authenticate a user",
            "/auth/health": "GET → Check authentication service health",
            "/docs": "GET → Interactive API documentation",
        },
    }


@app.get("/health")
async def health_check():
    if model is None or scaler_X is None or scaler_y is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest, user=Depends(get_current_user), db=Depends(get_db)
):
    """Predict maize yield for authenticated users and save to MongoDB."""
    if model is None or scaler_X is None or scaler_y is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        logger.info(f"Prediction request by user: {user['phone_number']}")
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

        # Save prediction to MongoDB
        prediction_data = {
            "phone_number": user["phone_number"],
            "year": request.year,
            "pesticides_tonnes": request.pesticides_tonnes,
            "avg_temp": request.avg_temp,
            "predicted_yield": round(predicted_yield, 2),
            "unit": "hg/ha",
            "warning": warning,
            "timestamp": datetime.utcnow(),
        }
        db["predictions"].insert_one(prediction_data)
        logger.info(f"Prediction saved for user: {user['phone_number']}")

        return PredictionResponse(
            predicted_yield=round(predicted_yield, 2),
            phone_number=user["phone_number"],
            warning=warning,
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/predictions", response_model=List[UserPrediction])
async def get_predictions(user=Depends(get_current_user), db=Depends(get_db)):
    """Get predictions for the authenticated user (farmer: own; admin: all)."""
    try:
        logger.info(
            f"Fetching predictions for user: {user['phone_number']}, role: {user['role']}"
        )
        if user["role"] == "admin":
            predictions = list(db["predictions"].find({}, {"_id": 0}))
        else:
            predictions = list(
                db["predictions"].find(
                    {"phone_number": user["phone_number"]}, {"_id": 0}
                )
            )
        return predictions
    except Exception as e:
        logger.error(f"Error fetching predictions: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching predictions: {str(e)}"
        )


@app.post("/submit-actual-yield", response_model=ActualYieldResponse)
async def submit_actual_yield(
    request: ActualYieldSubmission, user=Depends(get_current_user), db=Depends(get_db)
):
    """Submit actual yield data for authenticated users and save to MongoDB."""
    try:
        logger.info(f"Actual yield submission by user: {user['phone_number']}")
        actual_yield_data = {
            "phone_number": user["phone_number"],
            "crop_type": request.crop_type,
            "harvest_date": request.harvest_date,
            "actual_yield": request.actual_yield,
            "farm_size": request.farm_size,
            "location": request.location,
            "avg_temp": request.avg_temp,
            "notes": request.notes,
            "timestamp": datetime.utcnow(),
        }
        db["actual_yields"].insert_one(actual_yield_data)
        logger.info(f"Actual yield saved for user: {user['phone_number']}")
        return ActualYieldResponse(
            message="Submission successful", phone_number=user["phone_number"]
        )
    except Exception as e:
        logger.error(f"Submission error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Submission error: {str(e)}")


@app.get("/actual-yields", response_model=List[ActualYieldRecord])
async def get_actual_yields(user=Depends(get_current_user), db=Depends(get_db)):
    """Get actual yield submissions (farmer: own; admin: all)."""
    try:
        logger.info(
            f"Fetching actual yields for user: {user['phone_number']}, role: {user['role']}"
        )
        if user["role"] == "admin":
            yields = list(db["actual_yields"].find({}, {"_id": 0}))
        else:
            yields = list(
                db["actual_yields"].find(
                    {"phone_number": user["phone_number"]}, {"_id": 0}
                )
            )

        # Extract year from harvest_date
        for yield_data in yields:
            try:
                yield_data["harvest_year"] = int(
                    yield_data["harvest_date"].split("/")[1]
                )
            except (ValueError, IndexError):
                yield_data["harvest_year"] = None  # Handle invalid formats gracefully
            # Remove fields not in ActualYieldRecord
            for field in ["farm_size", "location", "notes", "harvest_date"]:
                yield_data.pop(field, None)

        return yields
    except Exception as e:
        logger.error(f"Error fetching actual yields: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching actual yields: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
