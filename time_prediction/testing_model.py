import joblib
import pandas as pd
import json
from pathlib import Path

# Load the trained model and metadata
model_path = Path('../models/stacking_regressor_model.joblib')
metadata_path = Path('../models/stacking_regressor_metadata.json')

model = joblib.load(model_path)
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Function to get user input and make prediction
def predict_delivery_time():
    print("Please provide the following delivery details:")
    
    # Get numerical inputs
    ratings = float(input("Restaurant ratings (1-5): "))
    distance = float(input("Delivery distance (km): "))
    
    # Get nominal categorical inputs
    is_festival = input("Is it a festival day? (yes/no): ").lower()
    is_weekend = input("Is it weekend? (yes/no): ").lower()
    day_of_week = input("Day of week (e.g., Monday): ").lower()
    
    # Get ordinal categorical inputs
    print("\nAvailable options for categorical features:")
    print(f"Distance type: {metadata['ordinal_mappings']['distance_type']}")
    distance_type = input("Distance type: ").lower()
    
    print(f"\nWeather: {metadata['ordinal_mappings']['weather']}")
    weather = input("Weather: ").lower()
    
    print(f"\nOrder type: {metadata['ordinal_mappings']['order_type']}")
    order_type = input("Order type: ").lower()
    
    print(f"\nCity category: {metadata['ordinal_mappings']['city_category']}")
    city_category = input("City category: ").lower()
    
    print(f"\nOrder time of day: {metadata['ordinal_mappings']['order_time_of_day']}")
    order_time_of_day = input("Order time of day: ").lower()
    
    # Create input DataFrame
    input_data = {
        'ratings': [ratings],
        'distance': [distance],
        'is_festival': [1 if is_festival == 'yes' else 0],
        'is_weekend': [1 if is_weekend == 'yes' else 0],
        'day_of_week': [day_of_week],
        'distance_type': [distance_type],
        'weather': [weather],
        'order_type': [order_type],
        'city_category': [city_category],
        'order_time_of_day': [order_time_of_day]
    }
    
    df = pd.DataFrame(input_data)
    
    # Make prediction
    predicted_time = model.predict(df)[0]
    print(f"\nPredicted delivery time: {predicted_time:.2f} minutes")

# Run the prediction function
if __name__ == "__main__":
    predict_delivery_time()