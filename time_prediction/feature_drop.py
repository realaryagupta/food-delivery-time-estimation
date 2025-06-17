import pandas as pd
from pathlib import Path

def preprocess_data(input_path: str, output_path: str) -> None:
    # Define columns to drop
    COLUMNS_TO_DROP = [
        'age',
        'restaurant_latitude',
        'restaurant_longitude',
        'delivery_latitude',
        'delivery_longitude',
        'traffic',
        'vehicle_condition',
        'vehicle_type',
        'multiple_deliveries',
        'city',
        'day',
        'month',
        'year',
        'order_time_hour',
        'pickup_time_minutes'
    ]

    # Load data with error handling
    try:
        data = pd.read_csv(input_path)
        print("Data loaded successfully.\n")
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found at {input_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")
    
    # Display initial data info
    print("Initial Data Overview:")
    print(data.info())
    print("\n")
    
    # Drop unnecessary columns
    data_clean = data.drop(columns=COLUMNS_TO_DROP, errors='ignore')
    
    # Verify columns were dropped
    dropped_cols = set(COLUMNS_TO_DROP) & set(data.columns)
    kept_cols = set(COLUMNS_TO_DROP) - dropped_cols
    
    if kept_cols:
        print(f"Warning: Columns not found in dataset: {kept_cols}")
    
    print("\nAfter dropping columns:")
    print(data_clean.info())
    
    # Save processed data
    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        data_clean.to_csv(output_path, index=False)
        print(f"\nProcessed data saved successfully to {output_path}")
    except Exception as e:
        raise Exception(f"Error saving processed data: {str(e)}")

if __name__ == "__main__":
    INTERIM_DATA_PATH = '../data/interim/clean_train_final_version.csv'
    PROCESSED_DATA_PATH = '../data/processed/final_data.csv'
    
    # Run preprocessing
    preprocess_data(INTERIM_DATA_PATH, PROCESSED_DATA_PATH)