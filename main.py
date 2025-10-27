import os
import pandas as pd
import numpy as np
import pickle
import requests
import re
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from pymongo import MongoClient
from passlib.context import CryptContext
from passlib.exc import PasswordSizeError
from jose import JWTError, jwt
from typing import Optional
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ==== CONFIGURATION ====
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Authentication Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = "smartgwiza"
COLLECTION_NAME = "users"
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Prediction Model Configuration
MODEL_PATH = "/Users/liliane/Documents/SmartGwizaS/smartgwiza_model.pkl"
DATA_PATH = "/Users/liliane/Documents/SmartGwizaS/rwanda_cropyield.csv"
MODEL_URL = (
    "https://drive.google.com/uc?export=download&id=1addazLYCNel0Q74WU-y1KWWTSd-dGx0E"
)
DATA_URL = (
    "https://drive.google.com/uc?export=download&id=1pqUAlKTTebLdz2dhYqlRh_DbqpLzQKVZ"
)

# Global variables for ML model
regression_model = None
scaler = None
features_list = ["year", "pesticides_kg_per_ha", "avg_temp"]
model_name = "Not loaded"
model_artifacts = None
last_retrain_time = None
last_retrain_status = "Not retrained yet"


# ==== FASTAPI APP SETUP ====
app = FastAPI(
    title=" SmartGwiza API",
    version="2.0.0",
    
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==== PYDANTIC MODELS ====


# Authentication Models
class UserSignupRequest(BaseModel):
    fullname: str = Field(..., min_length=2, max_length=100)
    phone_number: str = Field(..., description="Rwandan phone number")
    password: str = Field(..., min_length=8)
    role: str = Field(..., description="User role: 'farmer' or 'admin'")

    @validator("password")
    def check_password_length(cls, password):
        if len(password.encode("utf-8")) > 72:
            raise ValueError("Password cannot be longer than 72 bytes")
        return password

    @validator("role")
    def validate_role(cls, role):
        if role not in ["farmer", "admin"]:
            raise ValueError("Role must be 'farmer' or 'admin'")
        return role

    class Config:
        schema_extra = {
            "example": {
                "fullname": "John Doe",
                "phone_number": "+250781234567",
                "password": "secure123",
                "role": "farmer",
            }
        }


class UserSignupResponse(BaseModel):
    fullname: str
    phone_number: str
    role: str
    message: str = "User created successfully"


class UserLoginRequest(BaseModel):
    phone_number: str = Field(..., description="Rwandan phone number")
    password: str = Field(..., description="User password")

    class Config:
        schema_extra = {
            "example": {"phone_number": "+250781234567", "password": "secure123"}
        }


class UserLoginResponse(BaseModel):
    fullname: str
    phone_number: str
    role: str
    access_token: str
    token_type: str = "bearer"
    message: str = "Login successful"


class UserProfileResponse(BaseModel):
    fullname: str
    phone_number: str
    role: str
    message: str = "Profile retrieved successfully"


# Farmer Yield Data Models
class YieldDataSubmission(BaseModel):
    year: int = Field(..., ge=1990, le=2030, description="Harvest year")
    area_hectares: float = Field(
        ..., ge=0.1, le=10000, description="Farm area in hectares"
    )
    pesticides_kg: float = Field(
        ..., ge=0, le=100000, description="Total pesticides used (kg)"
    )
    avg_temperature: float = Field(
        ..., ge=15, le=30, description="Average temperature (°C)"
    )
    actual_yield_kg: float = Field(
        ..., ge=0, le=1000000, description="Total harvest (kg)"
    )
    location: Optional[str] = Field(None, description="District/Sector (optional)")
    notes: Optional[str] = Field(
        None, max_length=500, description="Additional notes (optional)"
    )

    class Config:
        schema_extra = {
            "example": {
                "year": 2024,
                "area_hectares": 3.5,
                "pesticides_kg": 12,
                "avg_temperature": 19.5,
                "actual_yield_kg": 8500,
                "location": "Bugesera District",
                "notes": "Used hybrid seeds, good rainfall",
            }
        }


class YieldDataResponse(BaseModel):
    message: str
    data_id: str
    submitted_by: str
    submission_date: str


class FarmerDataStats(BaseModel):
    total_submissions: int
    your_submissions: int
    date_range: str
    average_yield_kg_per_ha: float


# Prediction Models
class PredictionRequest(BaseModel):
    year: int = Field(..., ge=1985, le=2035, description="Year for prediction")
    area: float = Field(..., ge=0.1, le=10000, description="Area in hectares")
    pesticides: float = Field(
        ..., ge=0, le=1000000, description="Total pesticides in kg"
    )
    avg_temp: float = Field(..., ge=15, le=25, description="Average temperature in °C")

    class Config:
        schema_extra = {
            "example": {"year": 2024, "area": 50, "pesticides": 150, "avg_temp": 19.5}
        }


class PredictionResponse(BaseModel):
    predicted_yield_kg_per_ha: float
    total_production_tonnes: float

    class Config:
        schema_extra = {
            "example": {
                "predicted_yield_kg_per_ha": 2847.5,
                "total_production_tonnes": 142.375,
            }
        }


class RetrainStatus(BaseModel):
    last_retrain_time: str | None
    status: str


# ==== DATABASE & AUTHENTICATION FUNCTIONS ====


def get_db():
    """Initialize MongoDB client."""
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    try:
        yield db
    finally:
        client.close()


def validate_rwandan_phone(phone: str) -> str:
    """Validate and normalize Rwandan phone number."""
    phone = re.sub(r"\s+", "", phone)
    if phone.startswith("+250"):
        phone = phone
    elif phone.startswith("250"):
        phone = "+" + phone
    elif phone.startswith("0"):
        phone = "+250" + phone[1:]
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid Rwandan phone number",
        )
    if not re.match(r"^\+250(78|79|72|73)\d{7}$", phone):
        raise HTTPException(
            status_code=400,
            detail="Invalid Rwandan phone number format",
        )
    return phone


def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def create_user(db, fullname: str, phone_number: str, password: str, role: str):
    """Create a new user in MongoDB."""
    logger.info(f"Creating user with phone_number: {phone_number}, role: {role}")
    try:
        password_hash = hash_password(password)
        user = {
            "fullname": fullname,
            "phone_number": phone_number,
            "password_hash": password_hash,
            "role": role,
        }
        result = db[COLLECTION_NAME].insert_one(user)
        if not result.inserted_id:
            raise HTTPException(status_code=500, detail="Failed to create user")
        logger.info(f"User created successfully: {phone_number}")
    except PasswordSizeError:
        raise HTTPException(status_code=400, detail="Password too long")
    except Exception as e:
        if "E11000" in str(e):
            raise HTTPException(
                status_code=400, detail="Phone number already registered"
            )
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")


def authenticate_user(db, phone_number: str, password: str):
    """Authenticate user by phone number and password."""
    logger.info(f"Authenticating user: {phone_number}")
    user = db[COLLECTION_NAME].find_one({"phone_number": phone_number})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid phone number or password")
    if not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid phone number or password")
    user["role"] = user.get("role", "farmer")
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db=Depends(get_db),
):
    """Verify JWT token and return the current user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        phone_number: str = payload.get("sub")
        role: str = payload.get("role")
        if phone_number is None or role is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db[COLLECTION_NAME].find_one({"phone_number": phone_number})
    if user is None:
        raise credentials_exception
    user["role"] = user.get("role", "farmer")
    return user


# ==== ML MODEL FUNCTIONS ====


def download_file(url, destination):
    if not os.path.exists(destination):
        logger.info(f"Downloading {destination}...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(destination, "wb") as f:
                f.write(response.content)
            logger.info(f"Downloaded: {destination}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


def initialize_default_model():
    logger.info("Initializing default model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    scaler = StandardScaler()

    dummy_data = np.array([[2020, 3.0, 19.5], [2021, 3.5, 19.8]])
    scaler.fit(dummy_data)

    dummy_y = np.array([25000, 26000])
    model.fit(scaler.transform(dummy_data), dummy_y)

    return {
        "regression_model": model,
        "regression_model_name": "RandomForestRegressor (Default)",
        "scaler": scaler,
        "features": ["year", "pesticides_kg_per_ha", "avg_temp"],
        "metrics": {"test_r2": 0.0},
        "metadata": {
            "training_date": datetime.utcnow().isoformat(),
            "version": "2.0",
        },
    }


def retrain_model(data_path=DATA_PATH):
    global last_retrain_time, last_retrain_status, regression_model, scaler, model_artifacts, model_name
    try:
        if not os.path.exists(data_path):
            raise ValueError(f"Dataset not found at {data_path}")

        data = pd.read_csv(data_path)
        required_columns = ["year", "pesticides_tonnes", "avg_temp", "hg/ha_yield"]
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns")

        # Validate data
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"'{col}' must be numeric")

        if data.isnull().any().any():
            raise ValueError("Dataset contains missing values")
        if data["year"].duplicated().any():
            raise ValueError("Dataset contains duplicate years")
        if len(data) < 10:
            raise ValueError("Dataset too small (< 10 rows)")

        # Feature engineering
        AVERAGE_AREA_HA = 1000
        data["pesticides_kg_per_ha"] = (
            data["pesticides_tonnes"] * 1000
        ) / AVERAGE_AREA_HA
        data["yield_kg_per_ha"] = data["hg/ha_yield"] * 0.1

        X = data[["year", "pesticides_kg_per_ha", "avg_temp"]].values
        y = data["hg/ha_yield"].values

        # Temporal split
        train_size = len(data) - 5
        if train_size < 5:
            raise ValueError("Not enough data for training")

        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        # Scale and train
        new_scaler = StandardScaler()
        X_train_scaled = new_scaler.fit_transform(X_train)
        X_test_scaled = new_scaler.transform(X_test)

        model_retrain = RandomForestRegressor(n_estimators=100, random_state=42)
        model_retrain.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model_retrain.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_pred)

        # Update artifacts
        model_artifacts = {
            "regression_model": model_retrain,
            "regression_model_name": "RandomForestRegressor",
            "scaler": new_scaler,
            "features": ["year", "pesticides_kg_per_ha", "avg_temp"],
            "metrics": {"test_r2": test_r2},
            "metadata": {
                "training_date": datetime.utcnow().isoformat(),
                "dataset_period": f"{data['year'].min()}–{data['year'].max()}",
                "version": "2.0",
            },
        }

        # Save model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model_artifacts, f)

        # Update globals
        regression_model = model_retrain
        scaler = new_scaler
        model_name = "RandomForestRegressor"
        last_retrain_time = datetime.utcnow().isoformat()
        last_retrain_status = f"Success: Test R² = {test_r2:.4f}"

        logger.info(f"Model retrained successfully")
        return {"message": "Model retrained successfully", "test_r2": test_r2}

    except Exception as e:
        last_retrain_status = f"Failed: {str(e)}"
        logger.error(f"Retraining error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


def categorize_yield(yield_kg_per_ha):
    if yield_kg_per_ha < 2000:
        return "Low Yield"
    elif yield_kg_per_ha <= 3500:
        return "Medium Yield"
    else:
        return "High Yield"


# ==== STARTUP EVENT ====
@app.on_event("startup")
async def startup_event():
    global regression_model, scaler, model_artifacts, features_list, model_name, last_retrain_status

  

    try:
        download_file(MODEL_URL, MODEL_PATH)

        with open(MODEL_PATH, "rb") as f:
            model_artifacts = pickle.load(f)

        regression_model = model_artifacts["regression_model"]
        scaler = model_artifacts.get("scaler")
        features_list = model_artifacts.get(
            "features", ["year", "pesticides_kg_per_ha", "avg_temp"]
        )
        model_name = model_artifacts.get("regression_model_name", "Unknown")

        logger.info(f" Model loaded: {model_name}")

        if scaler is None or not hasattr(scaler, "mean_"):
            logger.warning("Scaler not fitted, attempting retrain...")
            try:
                download_file(DATA_URL, DATA_PATH)
                result = retrain_model(DATA_PATH)
                logger.info(f"✓ Retrained: {result}")
            except Exception as e:
                logger.error(f"Retrain failed: {str(e)}")
                model_artifacts = initialize_default_model()
                regression_model = model_artifacts["regression_model"]
                scaler = model_artifacts["scaler"]
                features_list = model_artifacts["features"]
                model_name = model_artifacts["regression_model_name"]
        else:
            logger.info(" Scaler properly fitted")
            last_retrain_status = "Model loaded from pickle"

    except Exception as e:
        logger.error(f"Model load failed: {str(e)}")
        model_artifacts = initialize_default_model()
        regression_model = model_artifacts["regression_model"]
        scaler = model_artifacts["scaler"]
        features_list = model_artifacts["features"]
        model_name = model_artifacts["regression_model_name"]

    
# ==== AUTHENTICATION ENDPOINTS ====


@app.post(
    "/signup",
    response_model=UserSignupResponse,
    tags=["authentication"],
    summary=" Create new user account",
)
async def signup(request: UserSignupRequest, db=Depends(get_db)):
    """Create a new user account with phone number validation."""
    normalized_phone = validate_rwandan_phone(request.phone_number)
    create_user(db, request.fullname, normalized_phone, request.password, request.role)
    return UserSignupResponse(
        fullname=request.fullname,
        phone_number=normalized_phone,
        role=request.role,
    )


@app.post(
    "/login",
    response_model=UserLoginResponse,
    tags=["authentication"],
    summary=" Login and get access token",
)
async def login(request: UserLoginRequest, db=Depends(get_db)):
    """Authenticate and receive JWT token."""
    normalized_phone = validate_rwandan_phone(request.phone_number)
    user = authenticate_user(db, normalized_phone, request.password)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": normalized_phone, "role": user["role"]},
        expires_delta=access_token_expires,
    )
    return UserLoginResponse(
        fullname=user["fullname"],
        phone_number=normalized_phone,
        role=user["role"],
        access_token=access_token,
    )


@app.get(
    "/profile",
    response_model=UserProfileResponse,
    tags=["authentication"],
    summary=" Get current user profile (Protected)",
)
async def get_profile(current_user: dict = Depends(get_current_user)):
    """
    Get authenticated user's profile.

     **Requires authentication** - Click Authorize button and enter your token.
    """
    return UserProfileResponse(
        fullname=current_user["fullname"],
        phone_number=current_user["phone_number"],
        role=current_user["role"],
    )


# ==== FARMER DATA COLLECTION ENDPOINTS ====


@app.post(
    "/farmer/submit-yield",
    response_model=YieldDataResponse,
    tags=["farmer-data"],
    summary=" Submit your actual yield data (Protected)",
)
async def submit_yield_data(
    data: YieldDataSubmission,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Submit actual harvest data from your farm.

     **Requires authentication** - Farmers can submit their real yield data.

    This data can be downloaded by admins and used to retrain the ML model.
    """
    try:
        # Calculate derived metrics
        pesticides_kg_per_ha = data.pesticides_kg / data.area_hectares
        yield_kg_per_ha = data.actual_yield_kg / data.area_hectares
        yield_hg_per_ha = yield_kg_per_ha * 10  # Convert to hg/ha for consistency

        # Create submission document
        submission = {
            "farmer_id": current_user["phone_number"],
            "farmer_name": current_user["fullname"],
            "year": data.year,
            "area_hectares": data.area_hectares,
            "pesticides_kg": data.pesticides_kg,
            "pesticides_kg_per_ha": pesticides_kg_per_ha,
            "avg_temperature": data.avg_temperature,
            "actual_yield_kg": data.actual_yield_kg,
            "yield_kg_per_ha": yield_kg_per_ha,
            "yield_hg_per_ha": yield_hg_per_ha,
            "location": data.location,
            "notes": data.notes,
            "submission_date": datetime.utcnow().isoformat(),
        }

        # Insert into database
        result = db["farmer_yield_data"].insert_one(submission)

        logger.info(
            f"Yield data submitted by {current_user['fullname']}: {yield_kg_per_ha:.2f} kg/ha"
        )

        return YieldDataResponse(
            message="Yield data submitted successfully",
            data_id=str(result.inserted_id),
            submitted_by=current_user["fullname"],
            submission_date=submission["submission_date"],
        )

    except Exception as e:
        logger.error(f"Yield submission error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit data: {str(e)}")


@app.get(
    "/farmer/my-submissions",
    tags=["farmer-data"],
    summary=" View your submitted data (Protected)",
)
async def get_my_submissions(
    current_user: dict = Depends(get_current_user), db=Depends(get_db)
):
    """
    View all yield data you've submitted.

     **Requires authentication**
    """
    try:
        submissions = list(
            db["farmer_yield_data"]
            .find({"farmer_id": current_user["phone_number"]}, {"_id": 0})
            .sort("submission_date", -1)
        )

        return {
            "total_submissions": len(submissions),
            "farmer_name": current_user["fullname"],
            "submissions": submissions,
        }

    except Exception as e:
        logger.error(f"Error fetching submissions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}")


@app.get(
    "/farmer/stats",
    response_model=FarmerDataStats,
    tags=["farmer-data"],
    summary=" Get farmer data statistics (Protected)",
)
async def get_farmer_stats(
    current_user: dict = Depends(get_current_user), db=Depends(get_db)
):
    """
    Get statistics about submitted farmer data.

     **Requires authentication**
    """
    try:
        # Total submissions
        total_submissions = db["farmer_yield_data"].count_documents({})

        # User's submissions
        user_submissions = db["farmer_yield_data"].count_documents(
            {"farmer_id": current_user["phone_number"]}
        )

        # Calculate average yield
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "avg_yield": {"$avg": "$yield_kg_per_ha"},
                    "min_year": {"$min": "$year"},
                    "max_year": {"$max": "$year"},
                }
            }
        ]

        stats = list(db["farmer_yield_data"].aggregate(pipeline))

        if stats:
            avg_yield = stats[0].get("avg_yield", 0)
            min_year = stats[0].get("min_year", "N/A")
            max_year = stats[0].get("max_year", "N/A")
            date_range = f"{min_year}-{max_year}" if min_year != "N/A" else "N/A"
        else:
            avg_yield = 0
            date_range = "N/A"

        return FarmerDataStats(
            total_submissions=total_submissions,
            your_submissions=user_submissions,
            date_range=date_range,
            average_yield_kg_per_ha=round(avg_yield, 2),
        )

    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")


@app.get(
    "/admin/download-farmer-data",
    tags=["admin"],
    summary=" Download all farmer data as CSV (Admin only)",
)
async def download_farmer_data(
    current_user: dict = Depends(get_current_user), db=Depends(get_db)
):
    """
    Download all submitted farmer data as CSV for model retraining.

     **Admin only** - Returns CSV file with all farmer submissions.
    """
    try:
        # Check admin privileges
        if current_user["role"] != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
            )

        # Fetch all submissions
        submissions = list(db["farmer_yield_data"].find({}, {"_id": 0}))

        if not submissions:
            raise HTTPException(
                status_code=404, detail="No farmer data available for download"
            )

        # Convert to DataFrame
        df = pd.DataFrame(submissions)

        # Create training-ready format
        training_df = pd.DataFrame(
            {
                "year": df["year"],
                "pesticides_tonnes": df["pesticides_kg"] / 1000,  # Convert to tonnes
                "avg_temp": df["avg_temperature"],
                "hg/ha_yield": df["yield_hg_per_ha"],
            }
        )

        # Group by year and aggregate (in case multiple farmers submit for same year)
        training_df = training_df.groupby("year", as_index=False).agg(
            {"pesticides_tonnes": "mean", "avg_temp": "mean", "hg/ha_yield": "mean"}
        )

        # Sort by year
        training_df = training_df.sort_values("year")

        # Save to CSV
        csv_path = "/Users/liliane/Documents/SmartGwizaS/farmer_data_export.csv"
        training_df.to_csv(csv_path, index=False)

        logger.info(
            f"Admin {current_user['fullname']} downloaded farmer data: {len(training_df)} rows"
        )

        from fastapi.responses import FileResponse

        return FileResponse(
            path=csv_path,
            filename=f"farmer_yield_data_{datetime.utcnow().strftime('%Y%m%d')}.csv",
            media_type="text/csv",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@app.get(
    "/admin/farmer-submissions",
    tags=["admin"],
    summary=" View all farmer submissions (Admin only)",
)
async def get_all_farmer_submissions(
    current_user: dict = Depends(get_current_user), db=Depends(get_db)
):
    """
    View all farmer yield submissions.

     **Admin only**
    """
    try:
        # Check admin privileges
        if current_user["role"] != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
            )

        # Fetch all submissions
        submissions = list(
            db["farmer_yield_data"].find({}, {"_id": 0}).sort("submission_date", -1)
        )

        return {
            "total_submissions": len(submissions),
            "submissions": submissions,
            "message": "All farmer submissions retrieved successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching submissions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}")


# ==== PREDICTION ENDPOINTS ====


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary=" Predict maize yield",
)
async def predict_yield(request: PredictionRequest):
    """
    Predict maize yield based on agricultural inputs.

    **Inputs:**
    - Year (1985-2035)
    - Area in hectares
    - Total pesticides in kg
    - Average temperature in °C

    **Returns:** Predicted yield and category (Low/Medium/High)
    """
    try:
        if scaler is None or not hasattr(scaler, "mean_"):
            raise HTTPException(
                status_code=503, detail="Model not ready. Please retrain via /retrain"
            )

        if regression_model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        pesticides_kg_per_ha = request.pesticides / request.area
        input_data = np.array([[request.year, pesticides_kg_per_ha, request.avg_temp]])

        input_scaled = scaler.transform(input_data)
        pred_hg_per_ha = regression_model.predict(input_scaled)[0]
        yield_kg_per_ha = max(pred_hg_per_ha * 0.1, 0)
        total_production_kg = yield_kg_per_ha * request.area
        total_production_tonnes = total_production_kg / 1000

        logger.info(f" Prediction: {yield_kg_per_ha:.2f} kg/ha")

        return {
            "predicted_yield_kg_per_ha": float(yield_kg_per_ha),
            "total_production_tonnes": float(total_production_tonnes),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ==== ADMIN ENDPOINTS ====


@app.post(
    "/upload",
    tags=["admin"],
    summary=" Upload new training data",
)
async def upload_file(file: UploadFile = File(...)):
    """Upload CSV file and retrain model."""
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files allowed")

        contents = await file.read()
        temp_path = "/Users/liliane/Documents/SmartGwizaS/temp_upload.csv"
        with open(temp_path, "wb") as f:
            f.write(contents)

        data = pd.read_csv(temp_path)
        required_columns = ["year", "pesticides_tonnes", "avg_temp", "hg/ha_yield"]

        if not all(col in data.columns for col in required_columns):
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Invalid CSV columns")

        if data.isnull().any().any():
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="CSV contains missing values")

        if len(data) < 10:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="CSV too small (< 10 rows)")

        os.replace(temp_path, DATA_PATH)
        logger.info(f" CSV uploaded")

        result = retrain_model(DATA_PATH)
        return {
            "message": "File uploaded and model retrained",
            "retrain_result": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@app.post(
    "/retrain",
    tags=["admin"],
    summary=" Retrain model",
)
async def retrain():
    """Retrain the ML model with existing data."""
    result = retrain_model()
    return result


@app.get(
    "/retrain_status",
    response_model=RetrainStatus,
    tags=["admin"],
    summary=" Get retrain status",
)
async def get_retrain_status():
    """Get the status of the last retraining operation."""
    return {"last_retrain_time": last_retrain_time, "status": last_retrain_status}


# ==== HEALTH ENDPOINTS ====


@app.get(
    "/",
    tags=["health"],
    summary=" API Home",
)
async def root():
    """API information and available endpoints."""
    return {
        "message": "SmartGwiza API - Maize Yield Prediction & User Management",
        "version": "2.0.0",
        "endpoints": {
            "authentication": ["/signup", "/login", "/profile"],
            "prediction": ["/predict"],
            "admin": ["/upload", "/retrain", "/retrain_status"],
            "health": ["/health", "/health/auth", "/health/model"],
        },
    }


@app.get(
    "/health",
    tags=["health"],
    summary=" Overall health check",
)
async def health_check():
    """Check overall system health."""
    model_ready = (
        regression_model is not None and scaler is not None and hasattr(scaler, "mean_")
    )

    db_connected = False
    try:
        client = MongoClient(MONGO_URI)
        client.server_info()
        client.close()
        db_connected = True
    except Exception as e:
        logger.error(f"DB check failed: {str(e)}")

    overall_status = "healthy" if (model_ready and db_connected) else "degraded"

    return {
        "status": overall_status,
        "model_ready": model_ready,
        "database_connected": db_connected,
        "model_name": model_name,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get(
    "/health/auth",
    tags=["health"],
    summary=" Authentication service health",
)
async def auth_health_check():
    """Check authentication service health."""
    try:
        client = MongoClient(MONGO_URI)
        client.server_info()
        client.close()
        return {"status": "healthy", "database_connected": True}
    except Exception as e:
        logger.error(f"Auth health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")


@app.get(
    "/health/model",
    tags=["health"],
    summary=" ML model health",
)
async def model_health_check():
    """Check ML model health."""
    scaler_fitted = scaler is not None and hasattr(scaler, "mean_")
    model_loaded = regression_model is not None

    return {
        "status": "healthy" if (model_loaded and scaler_fitted) else "degraded",
        "model_loaded": model_loaded,
        "scaler_fitted": scaler_fitted,
        "model_name": model_name,
        "last_retrain_status": last_retrain_status,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
