import pandas as pd
import sqlite3
import os

# --- Configuration ---
DATABASE_FILE = os.path.join('data', 'customer_reviews.db') # SQLite database file will be in the 'data' folder
INPUT_CSV_FILE = os.path.join('data', 'reviews_with_themes.csv') # Path to your input CSV file
TABLE_NAME = "CUSTOMER_REVIEWS" # Name of the table in your SQLite database

# --- Define Table Schema (SQL CREATE TABLE statement for SQLite) ---
# ADDED: DATE TEXT column back into table creation
CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    REVIEW_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    REVIEW_TEXT TEXT,
    SENTIMENT_LABEL TEXT,
    SENTIMENT_SCORE REAL,
    IDENTIFIED_THEMES TEXT,
    REVIEW_DATE TEXT -- ADDED: New column for review date
)
"""

def store_data_to_sqlite():
    conn = None
    try:
        # Ensure the 'data' directory exists
        os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)

        # Load data from CSV
        df = pd.read_csv(INPUT_CSV_FILE)
        print(f"Loaded {len(df)} reviews from {INPUT_CSV_FILE}")

        # Establish connection to SQLite Database (creates the .db file if it doesn't exist)
        print(f"Attempting to connect to SQLite Database file: {DATABASE_FILE}...")
        conn = sqlite3.connect(DATABASE_FILE)
        print("Successfully connected to SQLite Database!")

        cursor = conn.cursor()

        # Create table if it doesn't exist
        print(f"Creating table '{TABLE_NAME}' if it doesn't exist...")
        cursor.execute(CREATE_TABLE_SQL)
        print(f"Table '{TABLE_NAME}' ready.")

        # Prepare data for insertion
        # ADDED: 'date' column back to the data_to_insert tuple
        data_to_insert = [
            (
                row['review_text'] if pd.notna(row['review_text']) else None,
                row['sentiment_label'] if pd.notna(row['sentiment_label']) else None,
                row['sentiment_score'] if pd.notna(row['sentiment_score']) else None,
                row['identified_theme(s)'] if pd.notna(row['identified_theme(s)']) else None,
                row['date'] if pd.notna(row['date']) else None # ADDED: date value
            )
            for index, row in df.iterrows()
        ]
        
        # Use executemany for efficient insertion
        # ADDED: 'REVIEW_DATE' column to the INSERT statement values and corresponding '?'
        sql_insert = f"INSERT INTO {TABLE_NAME} (REVIEW_TEXT, SENTIMENT_LABEL, SENTIMENT_SCORE, IDENTIFIED_THEMES, REVIEW_DATE) VALUES (?, ?, ?, ?, ?)"
        
        print(f"Inserting {len(data_to_insert)} records into '{TABLE_NAME}'...")
        cursor.executemany(sql_insert, data_to_insert)
        conn.commit() # Commit changes to save them permanently
        print(f"✅ Successfully inserted {cursor.rowcount} records into '{TABLE_NAME}'.")

        # --- KPI Check for Database Storage ---
        cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        row_count_db = cursor.fetchone()[0]
        print(f"\n--- KPI Check for Database Storage ---")
        print(f"Total records in '{TABLE_NAME}' database table: {row_count_db}")

        if row_count_db >= len(df):
            print("✅ KPI: All cleaned and analyzed reviews stored in database - Met!")
        else:
            print("❌ KPI: All cleaned and analyzed reviews stored in database - Not Met.")

    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_CSV_FILE}' not found. Please ensure 'reviews_with_themes.csv' exists in your 'data' folder.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    store_data_to_sqlite()