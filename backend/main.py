# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import json
import os
import sys
import pandas as pd
from typing import Literal, Dict, Any, List
import traceback

# --- Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, ".."))
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, 'notebooks')
sys.path.insert(0, NOTEBOOKS_DIR)

# Define paths for model, metadata, and data relative to the project root
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "stacking_regressor_model.joblib")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "stacking_regressor_metadata.json")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "clean_train.csv")


# Initialize FastAPI app
app = FastAPI(
    title="Food Delivery Time Prediction API",
    description="Predicts food delivery time based on various factors using a Stacking Regressor model and provides data analysis charts."
)

# Configure CORS to allow communication from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development, allow all. Change this to your frontend URL in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Load Model, Metadata, and Data for Analysis ---
model = None
metadata = None
data = None # To store your DataFrame for analysis

# --- Debugging prints for loading paths ---
print(f"Project root: {PROJECT_ROOT}")
print(f"Notebooks directory added to sys.path: {NOTEBOOKS_DIR}")
print(f"Attempting to load model from: {MODEL_PATH}")
print(f"Attempting to load metadata from: {METADATA_PATH}")
print(f"Attempting to load data from: {DATA_PATH}")
# --- End Debugging prints ---

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

try:
    data = pd.read_csv(DATA_PATH)
    print("Data loaded successfully for analysis!")
except FileNotFoundError as e:
    print(f"Error: Data file not found at {DATA_PATH}. Analysis endpoints may not function. {e}")
    data = None # Explicitly set to None if not found
except Exception as e:
    print(f"Error loading data for analysis: {e}")
    data = None # Explicitly set to None if error


# --- Import NEW modular analysis functions ---
try:
    from analysis import (
        create_kde_plot,
        create_boxplot,
        create_histogram_with_kde,
        create_numerical_categorical_barplot,
        create_numerical_categorical_boxplot,
        create_numerical_categorical_violin_plot,
        create_numerical_categorical_stripplot,
        create_categorical_countplot,
        create_multivariate_barplot,
        create_multivariate_boxplot,
        create_multivariate_violin_plot,
        create_multivariate_stripplot,
        create_probplot
    )
    print("Modular analysis functions from analysis.py loaded successfully!")
except ImportError as e:
    print(f"Error importing modular analysis functions from analysis.py: {e}")
    # Print sys.path to help debug if the issue persists
    print(f"Current sys.path: {sys.path}")
    raise RuntimeError(f"Failed to load analysis functions: {e}")


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

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio # To convert plotly figures to JSON

@app.get("/analysis/charts")
async def get_analysis_charts():
    """
    Generates and returns Plotly charts as JSON for the analysis section.
    Each key in the returned dictionary will map to a single Plotly chart JSON string.
    """
    if data is None:
        raise HTTPException(status_code=500, detail="Analysis data not loaded. Cannot generate charts.")

    charts = {}
    try:
        # Chart 1: Numerical Analysis - Distribution of 'time'
        # Now we have three separate plots for this, return them as a list under one key
        numerical_time_plots = []
        try:
            fig = create_kde_plot(data, 'time', cat_col='traffic')
            numerical_time_plots.append(pio.to_json(fig))
        except Exception as e:
            print(f"Error generating KDE plot for time: {e}")
            numerical_time_plots.append(None) # Append None if generation fails for one plot

        try:
            fig = create_boxplot(data, 'time', cat_col='traffic')
            numerical_time_plots.append(pio.to_json(fig))
        except Exception as e:
            print(f"Error generating Boxplot for time: {e}")
            numerical_time_plots.append(None)

        try:
            fig = create_histogram_with_kde(data, 'time', cat_col='traffic', bins=20)
            numerical_time_plots.append(pio.to_json(fig))
        except Exception as e:
            print(f"Error generating Histogram with KDE for time: {e}")
            numerical_time_plots.append(None)

        charts['time_distribution_plots'] = numerical_time_plots

        # Chart 2: Numerical-Categorical Analysis - Delivery Time by Day of Week
        # Returning multiple plots for numerical-categorical analysis under one key
        numerical_categorical_day_time_plots = []
        try:
            fig = create_numerical_categorical_barplot(data, "day_of_week", "time")
            numerical_categorical_day_time_plots.append(pio.to_json(fig))
        except Exception as e:
            print(f"Error generating numerical-categorical barplot: {e}")
            numerical_categorical_day_time_plots.append(None)

        try:
            fig = create_numerical_categorical_boxplot(data, "day_of_week", "time")
            numerical_categorical_day_time_plots.append(pio.to_json(fig))
        except Exception as e:
            print(f"Error generating numerical-categorical boxplot: {e}")
            numerical_categorical_day_time_plots.append(None)

        try:
            fig = create_numerical_categorical_violin_plot(data, "day_of_week", "time")
            numerical_categorical_day_time_plots.append(pio.to_json(fig))
        except Exception as e:
            print(f"Error generating numerical-categorical violin plot: {e}")
            numerical_categorical_day_time_plots.append(None)

        try:
            fig = create_numerical_categorical_stripplot(data, "day_of_week", "time")
            numerical_categorical_day_time_plots.append(pio.to_json(fig))
        except Exception as e:
            print(f"Error generating numerical-categorical stripplot: {e}")
            numerical_categorical_day_time_plots.append(None)

        charts['time_by_day_of_week_plots'] = numerical_categorical_day_time_plots


        # Chart 3: Categorical Analysis - Traffic Conditions Distribution
        try:
            fig = create_categorical_countplot(data, 'traffic')
            charts['traffic_conditions_chart'] = pio.to_json(fig)
        except Exception as e:
            print(f"Error generating traffic_conditions_chart: {e}")
            charts['traffic_conditions_chart'] = None

        # Chart 4: Multivariate Analysis - Delivery Time by Day and Order Type
        # Returning multiple plots for multivariate analysis under one key
        multivariate_plots = []
        try:
            fig = create_multivariate_barplot(data, "time", "day_of_week", "order_type")
            multivariate_plots.append(pio.to_json(fig))
        except Exception as e:
            print(f"Error generating multivariate barplot: {e}")
            multivariate_plots.append(None)

        try:
            fig = create_multivariate_boxplot(data, "time", "day_of_week", "order_type")
            multivariate_plots.append(pio.to_json(fig))
        except Exception as e:
            print(f"Error generating multivariate boxplot: {e}")
            multivariate_plots.append(None)

        try:
            fig = create_multivariate_violin_plot(data, "time", "day_of_week", "order_type")
            multivariate_plots.append(pio.to_json(fig))
        except Exception as e:
            print(f"Error generating multivariate violin plot: {e}")
            multivariate_plots.append(None)

        try:
            fig = create_multivariate_stripplot(data, "time", "day_of_week", "order_type")
            multivariate_plots.append(pio.to_json(fig))
        except Exception as e:
            print(f"Error generating multivariate stripplot: {e}")
            multivariate_plots.append(None)

        charts['multivariate_day_order_type_plots'] = multivariate_plots

        # Chart 5: Probability Plot for Delivery Time
        try:
            fig = create_probplot(data['time'], title_text='Probability Plot for Delivery Time')
            charts['time_probplot'] = pio.to_json(fig)
        except Exception as e:
            print(f"Error generating time_probplot: {e}")
            charts['time_probplot'] = None

        return charts

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate analysis charts: {type(e).__name__}: {e}")