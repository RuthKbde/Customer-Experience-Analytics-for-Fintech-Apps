import pandas as pd
from textblob import TextBlob # Make sure to install this library
import os # Import the os module

def analyze_sentiment(review_text: str) -> tuple[float, str]:
    """
    Analyzes the sentiment of a given text using TextBlob.
    """
    if pd.isna(review_text) or not isinstance(review_text, str):
        return 0.0, 'neutral' # Default for missing or non-string reviews

    analysis = TextBlob(review_text)
    polarity = analysis.sentiment.polarity # -1.0 (negative) to 1.0 (positive)

    if polarity > 0.1: # Threshold for positive
        label = 'positive'
    elif polarity < -0.1: # Threshold for negative
        label = 'negative'
    else:
        label = 'neutral'
    return polarity, label

if __name__ == "__main__":
    # UPDATED: Use os.path.join to correctly reference files in the 'data' folder
    cleaned_filename = os.path.join('data', "cleaned_reviews.csv")
    sentiment_output_filename = os.path.join('data', "reviews_with_sentiment.csv")

    try:
        # Ensure the 'data' directory exists for output
        os.makedirs(os.path.dirname(sentiment_output_filename), exist_ok=True)

        # Load the cleaned data
        df = pd.read_csv(cleaned_filename)
        print(f"Loaded {len(df)} cleaned reviews from {cleaned_filename}")

        # Add sentiment columns
        print("Performing sentiment analysis...")
        df[['sentiment_score', 'sentiment_label']] = df['review'].apply(
            lambda x: pd.Series(analyze_sentiment(x))
        )
        print("Sentiment analysis complete.")

        # Save the DataFrame with new sentiment columns
        df.to_csv(sentiment_output_filename, index=False)
        print(f"✅ Reviews with sentiment saved to {sentiment_output_filename}")

        # --- KPI Verification ---
        total_reviews = len(df)
        # Check how many reviews had sentiment scores calculated (should be almost all if no NaNs)
        reviews_with_sentiment = df['sentiment_label'].count()
        sentiment_coverage_percentage = (reviews_with_sentiment / total_reviews) * 100 if total_reviews > 0 else 0

        print(f"\n--- KPI Check for Sentiment Analysis ---")
        print(f"Total reviews processed for sentiment: {total_reviews}")
        print(f"Reviews with sentiment scores calculated: {reviews_with_sentiment}")
        print(f"Sentiment score coverage: {sentiment_coverage_percentage:.2f}%")

        if sentiment_coverage_percentage >= 90:
            print("✅ KPI: Sentiment scores calculated for 90%+ of reviews - Met!")
        else:
            print(f"❌ KPI: Sentiment scores calculated for 90%+ of reviews - Not Met. Current: {sentiment_coverage_percentage:.2f}%")

        # Basic aggregation (will be more detailed in later steps)
        print("\n--- Sentiment Distribution by Bank ---")
        sentiment_by_bank = df.groupby(['bank', 'sentiment_label']).size().unstack(fill_value=0)
        print(sentiment_by_bank)
        
        print("\n--- Average Rating and Sentiment Score by Bank ---")
        avg_stats_by_bank = df.groupby('bank').agg(
            Average_Rating=('rating', 'mean'),
            Average_Sentiment_Score=('sentiment_score', 'mean')
        )
        print(avg_stats_by_bank)

    except FileNotFoundError:
        print(f"Error: The file '{cleaned_filename}' was not found.")
        print(f"Please ensure '{os.path.basename(cleaned_filename)}' exists in your '{os.path.dirname(cleaned_filename)}' folder after running preprocess_data.py.")
    except Exception as e:
        print(f"An unexpected error occurred during sentiment analysis: {e}")