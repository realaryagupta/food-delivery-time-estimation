# -*- coding: utf-8 -*-
"""
Comprehensive Data Cleaning Script for Food Delivery Dataset
Author: Data Science Team
Date: 2025-06-14
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from datetime import date
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

# Load the dataset and perform initial preprocessing
def load_data(file_path: str) -> pd.DataFrame:
    print("Loading dataset...")
    data = pd.read_csv(file_path)
    
    # Drop the row with all null values
    if 45593 in data.index:
        data = data.drop([45593], axis=0)
    
    print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns")
    return data

# Standardize column names
def rename_columns(data: pd.DataFrame) -> pd.DataFrame:
    print("Renaming columns...")
    
    # Convert all column names to lowercase
    data.columns = data.columns.str.lower()
    
    # Rename columns for consistency
    column_mapping = {
        "delivery_person_id": "person_id",
        "delivery_person_age": "age",
        "delivery_person_ratings": "ratings",
        "delivery_location_latitude": "delivery_latitude",
        "delivery_location_longitude": "delivery_longitude",
        "time_order_picked": "order_picked",
        "weatherconditions": "weather",
        "road_traffic_density": "traffic",
        "type_of_order": "order_type",
        "time_taken(min)": "time",
        "city": "city_category",
        "festival": "is_festival",
        "type_of_vehicle": "vehicle_type",
        "time_orderd": "order_time",
    }
    
    data = data.rename(columns=column_mapping)
    print("Column renaming completed")
    return data

# Clean and convert data types for all columns
def clean_data_types(data: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning data types...")
    
    # Replace string 'NaN' with actual np.nan
    data = data.replace(["NaN", "NaN ", " NaN"], np.nan)
    
    # ID Feature
    data['id'] = data['id'].replace("NaN", np.nan)
    
    # Extract city from person_id
    data['city'] = data['person_id'].str.split('RES').str.get(0)
    data['person_id'] = data['person_id'].replace("NaN", np.nan)
    
    # Age
    data['age'] = pd.to_numeric(data['age'], errors='coerce').round()
    
    # Ratings
    data['ratings'] = pd.to_numeric(data['ratings'], errors='coerce')
    
    # Location coordinates
    location_cols = ['restaurant_latitude', 'restaurant_longitude', 
                    'delivery_latitude', 'delivery_longitude']
    for col in location_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Date and time columns
    data['order_date'] = pd.to_datetime(data['order_date'], errors='coerce')
    data['order_time'] = pd.to_datetime(data['order_time'], errors='coerce')
    data['order_picked'] = pd.to_datetime(data['order_picked'], errors='coerce')
    
    # Weather
    data['weather'] = (data['weather']
                      .str.lower()
                      .str.replace("conditions ", "", regex=False)
                      .str.replace("weather ", "", regex=False)
                      .str.strip())
    
    # Traffic
    data['traffic'] = data['traffic'].str.lower().str.strip()
    
    # Vehicle condition
    data['vehicle_condition'] = pd.to_numeric(data['vehicle_condition'], errors='coerce').astype('Int64')
    
    # Order type
    data['order_type'] = data['order_type'].str.lower().str.strip()
    
    # Vehicle type
    data['vehicle_type'] = data['vehicle_type'].str.lower().str.strip()
    
    # Multiple deliveries
    data['multiple_deliveries'] = pd.to_numeric(data['multiple_deliveries'], errors='coerce')
    
    # Festival
    data['is_festival'] = data['is_festival'].str.lower().str.strip()
    
    # City category
    data['city_category'] = data['city_category'].str.lower().str.strip()
    
    # Time (remove (min) and convert to numeric)
    data['time'] = data['time'].astype(str).str.replace(r"\(min\)", "", regex=True)
    data['time'] = pd.to_numeric(data['time'], errors='coerce')
    
    print("Data type cleaning completed")
    return data

# Remove records with invalid or suspicious data
def remove_invalid_records(data: pd.DataFrame) -> pd.DataFrame:
    print("Removing invalid records...")
    initial_shape = data.shape[0]
    
    # Remove records with age < 18 (illegal working age)
    data = data[data['age'] >= 18]
    print(f"Removed {initial_shape - data.shape[0]} records with age < 18")
    
    # Remove records with ratings > 5 (impossible rating)
    invalid_ratings = data.shape[0]
    data = data[data['ratings'] <= 5]
    print(f"Removed {invalid_ratings - data.shape[0]} records with ratings > 5")
    
    print(f"Total records removed: {initial_shape - data.shape[0]}")
    return data

# Clean latitude and longitude coordinates for Indian geography
def clean_coordinates(data: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning geographical coordinates...")
    
    # India's geographical bounds
    lower_bound_lat = 6.44
    lower_bound_long = 68.70
    
    location_cols = ['restaurant_latitude', 'restaurant_longitude', 
                    'delivery_latitude', 'delivery_longitude']
    
    # Replace invalid coordinates with NaN
    for col in location_cols:
        if "latitude" in col:
            invalid_count = (data[col] < lower_bound_lat).sum()
            data[col] = np.where(data[col] < lower_bound_lat, np.nan, data[col])
            print(f"Set {invalid_count} invalid {col} values to NaN")
        elif "longitude" in col:
            invalid_count = (data[col] < lower_bound_long).sum()
            data[col] = np.where(data[col] < lower_bound_long, np.nan, data[col])
            print(f"Set {invalid_count} invalid {col} values to NaN")
    
    return data

# Extract datetime features from date and time columns
def create_datetime_features(data: pd.DataFrame) -> pd.DataFrame:
    print("Creating datetime features...")
    
    # Extract date features
    data['day'] = data['order_date'].dt.day
    data['month'] = data['order_date'].dt.month
    data['year'] = data['order_date'].dt.year
    data['day_of_week'] = data['order_date'].dt.day_name()
    data['is_weekend'] = data['order_date'].dt.day_name().isin(["Saturday", "Sunday"]).astype(int)
    
    # Extract hour from order time
    data['order_time_hour'] = data['order_time'].dt.hour
    
    # Create time of day categories
    data['order_time_of_day'] = pd.cut(
        data['order_time_hour'],
        bins=[0, 6, 12, 17, 20, 24],
        right=True,
        labels=["after_midnight", "morning", "afternoon", "evening", "night"]
    )
    
    # Calculate pickup time in minutes
    valid_times_mask = data[['order_time', 'order_picked']].notna().all(axis=1)
    data.loc[valid_times_mask, 'pickup_time_minutes'] = (
        (data.loc[valid_times_mask, 'order_picked'] - 
         data.loc[valid_times_mask, 'order_time']).dt.total_seconds() / 60
    )
    
    print("Datetime features created")
    return data

# Calculate haversine distance and distance categories
def calculate_distance_features(data: pd.DataFrame) -> pd.DataFrame:
    print("Calculating distance features...")
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate haversine distance between two points"""
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in kilometers
        distance = 6371 * c
        return distance
    
    # Calculate distance
    data['distance'] = haversine_distance(
        data['restaurant_latitude'], data['restaurant_longitude'],
        data['delivery_latitude'], data['delivery_longitude']
    )
    
    # Create distance categories
    data['distance_type'] = pd.cut(
        data['distance'],
        bins=[0, 5, 10, 15, 25, np.inf],
        right=False,
        labels=["short", "medium", "long", "very_long", "extreme"]
    )
    
    print("Distance features calculated")
    return data

# Drop columns that are not needed for analysis
def drop_unnecessary_columns(data: pd.DataFrame) -> pd.DataFrame:
    print("Dropping unnecessary columns...")
    
    columns_to_drop = ['id', 'person_id', 'order_time', 'order_picked', 'order_date']
    existing_columns = [col for col in columns_to_drop if col in data.columns]
    
    if existing_columns:
        data = data.drop(columns=existing_columns)
        print(f"Dropped columns: {existing_columns}")
    
    return data


# Generate and display data summary
def generate_data_summary(data: pd.DataFrame) -> None:
    print("\n" + "="*50)
    print("DATA CLEANING SUMMARY")
    print("="*50)
    
    print(f"Final dataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Total missing values: {data.isna().sum().sum()}")
    print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\nMissing values by column:")
    missing_data = data.isna().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        for col, count in missing_data.items():
            percentage = (count / len(data)) * 100
            print(f"  {col}: {count} ({percentage:.2f}%)")
    else:
        print("  No missing values found!")
    
    print(f"\nData types:")
    print(data.dtypes.value_counts())


# Main function to execute the complete data cleaning pipeline
def main(input_path: str, output_path: str = None) -> pd.DataFrame:
    print("="*60)
    print("STARTING DATA CLEANING PIPELINE")
    print("="*60)
    print(f"Processing file: {input_path}")
    print(f"Script run on: {date.today()}")
    
    # Execute cleaning pipeline
    data = load_data(input_path)
    data = rename_columns(data)
    data = clean_data_types(data)
    data = remove_invalid_records(data)
    data = clean_coordinates(data)
    data = create_datetime_features(data)
    data = calculate_distance_features(data)
    data = drop_unnecessary_columns(data)
    
    # Generate summary
    generate_data_summary(data)
    
    # Save cleaned data if output path provided
    if output_path:
        print(f"\nSaving cleaned data to: {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
        print("Data saved successfully!")
    
    print("\n" + "="*60)
    print("DATA CLEANING PIPELINE COMPLETED")
    print("="*60)
    
    return data

# calling the function
if __name__ == "__main__":
    base_dir = Path(__file__).parent 
    INPUT_FILE = base_dir.parent / "data/raw/train.csv"
    OUTPUT_FILE = base_dir.parent / "data/interim/clean_train.csv"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    cleaned_data = main(INPUT_FILE, OUTPUT_FILE)