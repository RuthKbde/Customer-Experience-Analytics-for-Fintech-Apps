import pandas as pd
import numpy as np
import os # Import the os module

def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses the raw review data.

    Args:
        df (pd.DataFrame): The raw DataFrame containing review data.

    Returns:
        pd.DataFrame: The cleaned and processed DataFrame.
    """
    print("Starting data preprocessing...")

    # --- NEW: Rename columns to match expected format for the project ---
    # This addresses the 'review_text' vs 'review' and 'bank_name' vs 'bank' mismatch
    df = df.rename(columns={
        'review_text': 'review',
        'bank_name': 'bank'
    })
    print("- Columns renamed to 'review' and 'bank' as per project requirements.")

    # 1. Ensure required columns exist and select them
    required_columns = ['review', 'rating', 'date', 'bank', 'source']
    # Check if all required columns are now present after renaming
    if not all(col in df.columns for col in required_columns):
        # This warning should ideally not appear after the rename
        print(f"Warning: After renaming, not all required columns found. Expected: {required_columns}, Found: {df.columns.tolist()}")
        df_cleaned = df[df.columns.intersection(required_columns)].copy()
        # Add missing columns as NaN if any are still missing (shouldn't be if raw data is complete)
        for col in required_columns:
            if col not in df_cleaned.columns:
                df_cleaned[col] = np.nan
    else:
        df_cleaned = df[required_columns].copy()


    # 2. Handle Missing Values
    # Drop rows where 'review' text is missing as it's critical for analysis
    initial_rows = len(df_cleaned)
    df_cleaned.dropna(subset=['review'], inplace=True)
    rows_after_review_na = len(df_cleaned)
    print(f"- Dropped {initial_rows - rows_after_review_na} rows with missing 'review' text.")

    # Convert 'rating' to numeric, coerce errors to NaN, then drop if still NaN
    df_cleaned['rating'] = pd.to_numeric(df_cleaned['rating'], errors='coerce')
    df_cleaned.dropna(subset=['rating'], inplace=True)
    rows_after_rating_na = len(df_cleaned)
    print(f"- Dropped {rows_after_review_na - rows_after_rating_na} rows with missing or invalid 'rating'.")

    # 3. Normalize Dates to YYYY-MM-DD format
    # Convert to datetime objects, coercing errors to NaT (Not a Time)
    df_cleaned['date'] = pd.to_datetime(df_cleaned['date'], errors='coerce')
    # Drop rows where date conversion failed (if any)
    df_cleaned.dropna(subset=['date'], inplace=True)
    rows_after_date_na = len(df_cleaned)
    print(f"- Dropped {rows_after_rating_na - rows_after_date_na} rows with invalid 'date' format.")
    # Format back to string 'YYYY-MM-DD'
    df_cleaned['date'] = df_cleaned['date'].dt.strftime('%Y-%m-%d')
    print("- Dates normalized to YYYY-MM-DD format.")

    # 4. Remove Duplicates
    # Define a subset of columns to identify duplicates (review content, rating, date, bank)
    initial_unique_reviews = len(df_cleaned)
    df_cleaned.drop_duplicates(subset=['review', 'rating', 'date', 'bank'], inplace=True)
    duplicates_removed = initial_unique_reviews - len(df_cleaned)
    print(f"- Removed {duplicates_removed} duplicate reviews.")

    print("Data preprocessing complete.")
    return df_cleaned

if __name__ == "__main__":
    # UPDATED: Use os.path.join to correctly reference file in the 'data' folder
    raw_filename = os.path.join('data', "raw_reviews.csv")
    cleaned_filename = os.path.join('data', "cleaned_reviews.csv") # Also update cleaned_filename to be in data folder

    try:
        # Ensure the 'data' directory exists for output
        os.makedirs(os.path.dirname(cleaned_filename), exist_ok=True)

        # Load the raw data
        raw_reviews_df = pd.read_csv(raw_filename)
        print(f"Loaded {len(raw_reviews_df)} raw reviews from {raw_filename}")

        # Preprocess the data
        cleaned_reviews_df = preprocess_reviews(raw_reviews_df)

        # Save the cleaned data
        cleaned_reviews_df.to_csv(cleaned_filename, index=False)
        print(f"\n✅ Cleaned data successfully saved to {cleaned_filename}")

        # --- KPI Verification ---
        total_reviews_cleaned = len(cleaned_reviews_df)
        # Calculate missing data percentage for the final cleaned dataset
        missing_values_count = cleaned_reviews_df[['review', 'rating', 'date', 'bank', 'source']].isnull().sum().sum()
        total_data_points = total_reviews_cleaned * len(['review', 'rating', 'date', 'bank', 'source'])
        
        missing_data_percentage = (missing_values_count / total_data_points) * 100 if total_data_points > 0 else 0

        print(f"\n--- KPI Check for Preprocessing ---")
        print(f"Total reviews after cleaning: {total_reviews_cleaned}")
        print(f"Percentage of missing data in cleaned dataset: {missing_data_percentage:.2f}%")

        # Check against project KPIs
        if total_reviews_cleaned >= 1200:
            print("✅ KPI: 1,200+ reviews collected - Met!")
        else:
            print(f"❌ KPI: 1,200+ reviews collected - Not Met. Current: {total_reviews_cleaned}. Aim for 1200+.")

        if missing_data_percentage < 5:
            print("✅ KPI: Less than 5% missing data - Met!")
        else:
            print(f"❌ KPI: Less than 5% missing data - Not Met. Current: {missing_data_percentage:.2f}%")

        print("✅ KPI: A clean CSV dataset is produced - Check `cleaned_reviews.csv`.")

    except FileNotFoundError:
        print(f"Error: The file '{raw_filename}' was not found.")
        print(f"Please ensure '{os.path.basename(raw_filename)}' exists in your '{os.path.dirname(raw_filename)}' folder.")
    except Exception as e:
        print(f"An unexpected error occurred during preprocessing: {e}")