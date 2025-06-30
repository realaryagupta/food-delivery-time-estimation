from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import json
import os
import pandas as pd
from typing import Literal, Dict, Any
import traceback

# Initialize FastAPI app
app = FastAPI(
    title="Food Delivery Time Prediction API",
    description="Predicts food delivery time based on various factors using a Stacking Regressor model."
)

# Configure CORS to allow communication from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development, allow all. Change this to your frontend URL in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Model and Metadata ---
# IMPORTANT: Adjusting paths for the new folder structure.
# os.path.dirname(__file__) gives the directory of main.py (e.g., /path/to/backend)
# os.path.join(..., "..") moves up to the parent directory (e.g., /path/to/food_time_prediction_using_mlops)
# Then, we join 'models' to that root.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "stacking_regressor_model.joblib")
METADATA_PATH = os.path.join(ROOT_DIR, "models", "stacking_regressor_metadata.json")

# --- Debugging prints for model/metadata loading paths ---
print(f"Attempting to load model from: {MODEL_PATH}")
print(f"Attempting to load metadata from: {METADATA_PATH}")
# --- End Debugging prints ---


model = None
metadata = None

try:
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    print("Model and metadata loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: Model or metadata file not found. Please ensure paths are correct: {e}")
    raise RuntimeError(f"Required model file not found: {e}")
except Exception as e:
    print(f"Error loading model or metadata: {e}")
    raise RuntimeError(f"Failed to load model or metadata: {e}")

# --- Dynamic Pydantic Model based on metadata ---
# Extract allowed values for categorical features from metadata for Pydantic validation
day_of_week_options = metadata['ordinal_mappings'].get('day_of_week', ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'])
distance_type_options = metadata['ordinal_mappings'].get('distance_type', ['short', 'medium', 'long', 'very_long'])
weather_options = metadata['ordinal_mappings'].get('weather', ['sunny', 'cloudy', 'windy', 'fog', 'stormy', 'sandstorms'])
order_type_options = metadata['ordinal_mappings'].get('order_type', ['snack', 'drinks', 'buffet', 'meal'])
city_category_options = metadata['ordinal_mappings'].get('city_category', ['urban', 'metropolitian', 'semi-urban'])
order_time_of_day_options = metadata['ordinal_mappings'].get('order_time_of_day', ['morning', 'afternoon', 'evening', 'night'])


class PredictionInput(BaseModel):
    # Numerical features
    ratings: float = Field(..., ge=1.0, le=5.0, description="Restaurant ratings from 1.0 to 5.0")
    distance: float = Field(..., ge=0.1, le=50.0, description="Delivery distance in kilometers")

    # Nominal features (using Literal for 0 or 1)
    is_festival: Literal[0, 1] = Field(..., description="Is it a festival day (0 for No, 1 for Yes)")
    is_weekend: Literal[0, 1] = Field(..., description="Is it a weekend (0 for No, 1 for Yes)")

    # Ordinal features (using Literal for strict validation against allowed options)
    # Ensure these are lowercase in the Pydantic model to match the .lower() conversion from frontend
    day_of_week: Literal[*[opt.lower() for opt in day_of_week_options]] = Field(..., description="Day of the week")
    distance_type: Literal[*[opt.lower() for opt in distance_type_options]] = Field(..., description="Type of distance (short, medium, long, very_long)")
    weather: Literal[*[opt.lower() for opt in weather_options]] = Field(..., description="Weather conditions")
    order_type: Literal[*[opt.lower() for opt in order_type_options]] = Field(..., description="Type of order (snack, meal, buffet, drinks)")
    city_category: Literal[*[opt.lower() for opt in city_category_options]] = Field(..., description="Category of the city")
    order_time_of_day: Literal[*[opt.lower() for opt in order_time_of_day_options]] = Field(..., description="Time of day for the order")


def prepare_features_for_model(input_data: Dict[str, Any], metadata: Dict[str, Any]) -> pd.DataFrame:
    """
    Prepares the input DataFrame with raw features, ensuring correct dtypes
    and column order as expected by the ColumnTransformer within the model.
    """
    # Get all raw feature names from metadata, in the order they were categorized in model_building.py
    numerical_features = metadata['feature_categories']['numerical']
    nominal_features = metadata['feature_categories']['nominal']
    ordinal_features = metadata['feature_categories']['ordinal']

    all_raw_features_ordered = numerical_features + nominal_features + ordinal_features

    # Create a Pandas Series from the input dictionary to maintain original keys
    # and then convert to DataFrame for a single row.
    input_series = pd.Series(input_data)
    df = pd.DataFrame([input_series])

    # Ensure all expected raw columns are present and in the correct order
    df = df[all_raw_features_ordered]

    # Explicitly convert dtypes to match what ColumnTransformer typically expects for raw input
    # Numerical columns should be float
    for col in numerical_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Nominal (binary) columns should be integer (0 or 1)
    for col in nominal_features:
        if col in ['is_festival', 'is_weekend']:
            df[col] = df[col].astype(int)
        else: # Other nominals should be object/string for OHE
            df[col] = df[col].astype(str)

    # Ordinal columns should be object (string)
    for col in ordinal_features:
        df[col] = df[col].astype(str)

    return df


@app.get("/")
async def read_root():
    """
    Root endpoint to confirm the API is running.
    """
    return {"message": "Welcome to the Food Delivery Time Prediction API! Visit /docs for API documentation."}


@app.post("/predict")
async def predict_delivery_time(data: PredictionInput):
    """
    Predicts the food delivery time based on input features.
    """
    if model is None or metadata is None:
        raise HTTPException(status_code=500, detail="Model or metadata not loaded. Server issue.")

    try:
        input_dict = data.model_dump()
        print(f"\n--- Received raw input from frontend (Pydantic validated): {input_dict}") # Debugging

        # Prepare the input DataFrame with raw features, exact order, and dtypes
        input_df = prepare_features_for_model(input_dict, metadata)
        print(f"--- DataFrame passed to model (RAW, pre-internal-transformer):") # Debugging
        print(input_df.to_string()) # Print full DataFrame for single row
        print(f"--- DataFrame dtypes: {input_df.dtypes.to_dict()}") # Debugging dtypes

        # Make prediction
        predicted_time = model.predict(input_df)[0]

        predicted_time = max(5.0, round(float(predicted_time), 2)) # Ensure positive and reasonable

        return {"predicted_delivery_time_minutes": predicted_time}

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing or invalid feature in input: {e}. Please ensure all required fields are provided and correctly named.")
    except Exception as e:
        traceback.print_exc() # Print full Python traceback to Uvicorn console
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal server error. Error: {type(e).__name__}: {e}. Check server logs for full traceback.")

