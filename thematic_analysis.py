import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load spaCy English model
# You might need to download it first: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm' (first time only)...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def clean_text_for_nlp(text: str) -> str:
    """
    Cleans and preprocesses text for NLP tasks: lowercasing,
    removing non-alphanumeric, tokenization, stop word removal, lemmatization.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove URLs, mentions, hashtags, and special characters (keeping only letters and numbers)
    text = re.sub(r'http\S+|www\S+|https\S+|@\w+|#\w+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove punctuation and special characters
    text = text.lower() # Lowercase the text

    doc = nlp(text)
    # Lemmatize, remove stop words, and remove single-character tokens
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space and len(token.lemma_) > 1]
    
    return " ".join(tokens)

def get_top_tfidf_keywords(df: pd.DataFrame, text_column: str, group_column: str, top_n: int = 10) -> dict:
    """
    Calculates TF-IDF scores for text within groups and returns top keywords per group.
    """
    tfidf_results = {}
    
    for group_name, group_df in df.groupby(group_column):
        # Filter out empty strings from processed text
        texts = [text for text in group_df[text_column] if text]
        
        if not texts:
            print(f"No valid texts to analyze for {group_name}.")
            continue

        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2)) # Consider unigrams and bigrams
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate average TF-IDF score for each feature
        avg_tfidf_scores = tfidf_matrix.mean(axis=0).A1
        
        # Get top N keywords
        top_indices = avg_tfidf_scores.argsort()[-top_n:][::-1]
        top_keywords = [(feature_names[i], avg_tfidf_scores[i]) for i in top_indices]
        
        tfidf_results[group_name] = top_keywords
        
    return tfidf_results

def assign_themes(row: pd.Series) -> str:
    """
    Manually assigns themes based on review text content.
    This function will be updated based on keywords identified by TF-IDF.
    For now, it's a placeholder.
    """
    review = str(row['review_text']).lower() # Use review_text for matching
    
    # Example Theme Categories (you will refine these based on TF-IDF keywords)
    # A review can potentially belong to multiple themes.
    themes = []

    # Theme 1: Account Access & Login Issues
    if 'login' in review or 'access' in review or 'password' in review or 'otp' in review or 'user name' in review:
        themes.append('Account Access & Login Issues')
    
    # Theme 2: Transaction Performance & Speed
    if 'transfer' in review or 'transaction' in review or 'deposit' in review or 'withdraw' in review or 'send money' in review or 'slow' in review or 'fast' in review:
        themes.append('Transaction Performance & Speed')
        
    # Theme 3: App Stability & Bugs
    if 'crash' in review or 'bug' in review or 'freeze' in review or 'error' in review or 'update' in review or 'issue' in review:
        themes.append('App Stability & Bugs')

    # Theme 4: User Interface & Experience
    if 'ui' in review or 'interface' in review or 'design' in review or 'easy' in review or 'confusing' in review or 'user friendly' in review:
        themes.append('User Interface & Experience')

    # Theme 5: Customer Service
    if 'customer service' in review or 'support' in review or 'help' in review or 'response' in review:
        themes.append('Customer Service')

    # If no specific theme is found, categorize as 'General Feedback' or 'Other'
    if not themes:
        return 'General Feedback'
    
    return "; ".join(themes) # Join multiple themes with a semicolon


if __name__ == "__main__":
    sentiment_data_filename = "reviews_with_sentiment.csv"
    themed_output_filename = "reviews_with_themes.csv"

    try:
        # Load the data with sentiment scores
        df = pd.read_csv(sentiment_data_filename)
        print(f"Loaded {len(df)} reviews with sentiment from {sentiment_data_filename}")

        # --- Prepare data for Thematic Analysis ---
        # 1. Add review_id (if not already present)
        df['review_id'] = df.index # Using index as review_id

        # 2. Rename 'review' to 'review_text' for consistency with project output requirements
        df = df.rename(columns={'review': 'review_text'})
        print("Renamed 'review' column to 'review_text' and created 'review_id'.")

        # 3. Clean text for NLP (lemmatization, stop word removal, etc.)
        print("Cleaning text for thematic analysis (this may take a moment)...")
        df['processed_review_text'] = df['review_text'].apply(clean_text_for_nlp)
        print("Text cleaning complete.")

        # 4. Extract Top Keywords using TF-IDF for each bank
        print("\nExtracting top keywords per bank using TF-IDF...")
        top_keywords_per_bank = get_top_tfidf_keywords(df, 'processed_review_text', 'bank', top_n=15)

        for bank, keywords in top_keywords_per_bank.items():
            print(f"\n--- Top Keywords for {bank} ---")
            for keyword, score in keywords:
                print(f"- {keyword} (TF-IDF: {score:.4f})")
        
        print("\n**Manual Theme Grouping Step:**")
        print("Review the keywords above for each bank. Based on these, define 3-5 overarching themes.")
        print("Examples: 'Account Access', 'Transaction Performance', 'User Interface', 'Customer Service', 'App Stability'.")
        print("You will then need to refine the `assign_themes` function based on these insights.")
        print("The current `assign_themes` function has placeholder logic that you should refine.")


        # 5. Assign Themes based on keywords and content
        print("\nAssigning initial themes to reviews (based on placeholder logic)...")
        df['identified_theme(s)'] = df.apply(assign_themes, axis=1)
        print("Theme assignment complete.")

        # --- Final Output Preparation ---
        # Select and reorder columns for the final output CSV as per project requirements
        final_output_df = df[[
            'review_id', 'review_text', 'sentiment_label', 'sentiment_score', 'identified_theme(s)'
        ]].copy()

        # Save the DataFrame with themes
        final_output_df.to_csv(themed_output_filename, index=False)
        print(f"\n✅ Reviews with themes saved to {themed_output_filename}")

        # --- KPI Check for Thematic Analysis ---
        print(f"\n--- KPI Check for Thematic Analysis ---")
        # Check if themes are identified (at least one theme for most reviews)
        themes_identified_count = final_output_df[final_output_df['identified_theme(s)'] != 'General Feedback'].shape[0]
        themes_coverage_percentage = (themes_identified_count / len(final_output_df)) * 100

        print(f"Reviews with specific themes identified: {themes_identified_count} out of {len(final_output_df)} ({themes_coverage_percentage:.2f}%)")

        # Check for 3+ themes (this is a qualitative check based on your `assign_themes` logic)
        # We can't automate checking if *your* themes are 3+ per bank, but we can count unique themes.
        all_unique_themes = set()
        for themes_str in final_output_df['identified_theme(s)'].unique():
            if isinstance(themes_str, str):
                for theme in themes_str.split('; '):
                    if theme != 'General Feedback': # Exclude generic placeholder
                        all_unique_themes.add(theme.strip())
        
        print(f"Total unique themes identified across all reviews (excluding 'General Feedback'): {len(all_unique_themes)}")
        if len(all_unique_themes) >= 3: # General check, needs refinement per bank
            print("✅ KPI: 3+ themes identified (overall, needs per-bank verification) - Met!")
        else:
            print("❌ KPI: 3+ themes identified (overall) - Not Met. Please refine 'assign_themes' and keywords.")
        
        print("✅ KPI: The analysis code is organised into a modular pipeline - Check `thematic_analysis.py` structure.")

    except FileNotFoundError:
        print(f"Error: The file '{sentiment_data_filename}' was not found.")
        print("Please ensure 'reviews_with_sentiment.csv' exists after running sentiment_analysis.py.")
    except Exception as e:
        print(f"An unexpected error occurred during thematic analysis: {e}")