import logging
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, Field, validator
from pymongo import MongoClient
from typing import Optional
import re
from passlib.context import CryptContext
from passlib.exc import PasswordSizeError
from dotenv import load_dotenv
import os
from jose import JWTError, jwt
from datetime import datetime, timedelta

# ==== CONFIGURATION ====
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = "smartgwiza"
COLLECTION_NAME = "users"
SECRET_KEY = os.getenv("SECRET_KEY")  
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==== REQUEST & RESPONSE MODELS ====
class UserSignupRequest(BaseModel):
    fullname: str = Field(
        ..., min_length=2, max_length=100, description="User's full name"
    )
    phone_number: str = Field(
        ..., description="Rwandan phone number (e.g., +25078... or 078...)"
    )
    password: str = Field(
        ..., min_length=8, description="Password (8–72 characters, maximum 72 bytes)"
    )

    @validator("password")
    def check_password_length(cls, password):
        if len(password.encode("utf-8")) > 72:
            raise ValueError(
                "Password cannot be longer than 72 bytes. Please use a shorter password."
            )
        return password

    class Config:
        schema_extra = {
            "example": {
                "fullname": "John Doe",
                "phone_number": "+250781234567",
                "password": "secure123",
            }
        }


class UserSignupResponse(BaseModel):
    fullname: str
    phone_number: str
    message: str = "User created successfully"

    class Config:
        schema_extra = {
            "example": {
                "fullname": "John Doe",
                "phone_number": "+250781234567",
                "message": "User created successfully",
            }
        }


class UserLoginRequest(BaseModel):
    phone_number: str = Field(
        ..., description="Rwandan phone number (e.g., +25078... or 078...)"
    )
    password: str = Field(..., description="User password")

    class Config:
        schema_extra = {
            "example": {
                "phone_number": "+250781234567",
                "password": "secure123",
            }
        }


class UserLoginResponse(BaseModel):
    fullname: str
    phone_number: str
    access_token: str
    token_type: str = "bearer"
    message: str = "Login successful"

    class Config:
        schema_extra = {
            "example": {
                "fullname": "John Doe",
                "phone_number": "+250781234567",
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "message": "Login successful",
            }
        }


class ErrorResponse(BaseModel):
    detail: str

    class Config:
        schema_extra = {"example": {"detail": "Invalid phone number or password"}}


# ==== DATABASE SETUP ====
def get_db():
    """Initialize MongoDB client."""
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    try:
        yield db
    finally:
        client.close()


# ==== VALIDATION ====
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
            detail="Invalid Rwandan phone number. Must start with +250, 250, or 0 followed by 78/79/72/73 and 7 digits.",
        )

    if not re.match(r"^\+250(78|79|72|73)\d{7}$", phone):
        raise HTTPException(
            status_code=400,
            detail="Invalid Rwandan phone number format. Must be +250 followed by 78/79/72/73 and 7 digits.",
        )

    return phone


# ==== AUTHENTICATION LOGIC ====
def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def create_user(db, fullname: str, phone_number: str, password: str):
    """Create a new user in MongoDB."""
    logger.info(f"Creating user with phone_number: {phone_number}")
    try:
        password_hash = hash_password(password)
        user = {
            "fullname": fullname,
            "phone_number": phone_number,
            "password_hash": password_hash,
        }
        result = db[COLLECTION_NAME].insert_one(user)
        if not result.inserted_id:
            raise HTTPException(status_code=500, detail="Failed to create user")
        logger.info(f"User created successfully: {phone_number}")
    except PasswordSizeError:
        logger.error(f"Password too long for user: {phone_number}")
        raise HTTPException(
            status_code=400,
            detail="Password cannot be longer than 72 bytes. Please use a shorter password.",
        )
    except Exception as e:
        logger.error(f"Error creating user {phone_number}: {str(e)}")
        if "E11000" in str(e):  # MongoDB duplicate key error
            raise HTTPException(
                status_code=400, detail="Phone number already registered"
            )
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")


def authenticate_user(db, phone_number: str, password: str):
    """Authenticate user by phone number and password."""
    logger.info(f"Authenticating user with phone_number: {phone_number}")
    user = db[COLLECTION_NAME].find_one({"phone_number": phone_number})
    if not user:
        logger.warning(f"User not found: {phone_number}")
        raise HTTPException(status_code=401, detail="Invalid phone number or password")
    if not verify_password(password, user["password_hash"]):
        logger.warning(f"Invalid password for user: {phone_number}")
        raise HTTPException(status_code=401, detail="Invalid phone number or password")
    logger.info(f"User authenticated successfully: {phone_number}")
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


async def get_current_user(token: str, db=Depends(get_db)):
    """Verify JWT token and return the current user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        phone_number: str = payload.get("sub")
        if phone_number is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db[COLLECTION_NAME].find_one({"phone_number": phone_number})
    if user is None:
        raise credentials_exception
    return user


# ==== FASTAPI SETUP ====
app = FastAPI(
    title="SmartGwiza Authentication API",
    description="API for user signup and login with Rwandan phone number validation",
    version="1.0.0",
    openapi_tags=[{"name": "auth", "description": "User authentication endpoints"}],
)


# ==== ENDPOINTS ====
@app.post(
    "/signup",
    response_model=UserSignupResponse,
    responses={
        200: {"model": UserSignupResponse, "description": "Successful signup"},
        400: {
            "model": ErrorResponse,
            "description": "Invalid input or phone number already registered",
        },
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    tags=["auth"],
)
async def signup(request: UserSignupRequest, db=Depends(get_db)):
    """Create a new user account."""
    try:
        normalized_phone = validate_rwandan_phone(request.phone_number)
        create_user(db, request.fullname, normalized_phone, request.password)
        return UserSignupResponse(
            fullname=request.fullname,
            phone_number=normalized_phone,
            message="User created successfully",
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")


@app.post(
    "/login",
    response_model=UserLoginResponse,
    responses={
        200: {"model": UserLoginResponse, "description": "Successful login"},
        401: {"model": ErrorResponse, "description": "Invalid credentials"},
        400: {"model": ErrorResponse, "description": "Invalid phone number format"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    tags=["auth"],
)
async def login(request: UserLoginRequest, db=Depends(get_db)):
    """Authenticate a user and return a JWT token."""
    try:
        normalized_phone = validate_rwandan_phone(request.phone_number)
        user = authenticate_user(db, normalized_phone, request.password)
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": normalized_phone}, expires_delta=access_token_expires
        )
        return UserLoginResponse(
            fullname=user["fullname"],
            phone_number=normalized_phone,
            access_token=access_token,
            token_type="bearer",
            message="Login successful",
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during login: {str(e)}")


@app.get(
    "/health",
    responses={
        200: {"description": "Service is healthy"},
        503: {"model": ErrorResponse, "description": "Database connection error"},
    },
    tags=["auth"],
)
async def health_check():
    """Check if authentication service is running."""
    try:
        client = MongoClient(MONGO_URI)
        client.server_info()  
        client.close()
        return {"status": "healthy", "database_connected": True}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503, detail=f"Database connection error: {str(e)}"
        )
