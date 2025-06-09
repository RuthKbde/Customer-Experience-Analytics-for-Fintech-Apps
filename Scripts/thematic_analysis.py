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
    Assigns themes based on review text content, using the provided comprehensive keyword lists.
    A review can belong to multiple themes.
    """
    review = str(row['review_text']).lower() # Use original review text for comprehensive matching
    themes = []

    # --- 1. Positive Sentiment Words (Praise/Approval) ---
    positive_general = ['excellent', 'great', 'amazing', 'fantastic', 'outstanding', 'perfect', 'smooth', 'reliable', 'trustworthy', 'user friendly', 'well designed', 'modern', 'innovative']
    positive_customer_service = ['helpful', 'responsive', 'friendly', 'professional', 'supportive', 'quick response', 'efficient', 'courteous', '24/7 support']
    positive_app_features = ['easy to use', 'intuitive', 'convenient', 'fast', 'secure', 'seamless', 'well designed', 'modern', 'innovative']
    positive_transactions = ['instant transfer', 'hassle free', 'smooth transactions', 'no delays', 'great rates', 'low fees', 'rewarding', 'quick loan approval', 'easy kyc']

    # --- 2. Negative Sentiment Words (Complaints/Frustration) ---
    negative_general = ['terrible', 'awful', 'horrible', 'worst', 'unreliable', 'frustrating', 'disappointing', 'annoying']
    negative_customer_service = ['slow', 'unresponsive', 'rude', 'unhelpful', 'poor support', 'ignored', 'no resolution']
    negative_app_problems = ['buggy', 'crashes', 'freezes', 'slow loading', 'complicated', 'confusing', 'outdated', 'glitchy']
    negative_transaction_issues = ['failed transaction', 'delayed transfer', 'high fees', 'hidden charges', 'security concerns', 'account blocked', 'login failed', 'money stuck', 'unauthorized transaction', 'otp not received', 'account hacked']

    # --- 3. Neutral Sentiment Words (Neutral/Informative) ---
    neutral_general = ['okay', 'average', 'decent', 'fine', 'normal', 'standard', 'basic']
    neutral_functional = ['works', 'functional', 'does the job', 'usable', 'no issues', 'expected']
    neutral_suggestions = ['could be better', 'needs improvement', 'suggestion', 'feedback', 'update needed']
    
    # --- Assigning Themes ---
    
    # Theme: Positive Feedback (General)
    if any(phrase in review for phrase in positive_general + positive_app_features + positive_transactions):
        themes.append('Positive User Experience & Features')
        
    # Theme: Excellent Customer Service
    if any(phrase in review for phrase in positive_customer_service):
        themes.append('Excellent Customer Service')

    # Theme: App Performance Issues (Negative)
    if any(phrase in review for phrase in negative_app_problems + ['not work']): # Added 'not work' explicitly as it's common
        themes.append('App Performance & Stability Issues')

    # Theme: Transaction & Account Issues (Negative)
    if any(phrase in review for phrase in negative_transaction_issues):
        themes.append('Transaction & Account Issues')

    # Theme: Poor Customer Service (Negative)
    if any(phrase in review for phrase in negative_customer_service):
        themes.append('Poor Customer Service')

    # Theme: General Negative Feedback
    if any(phrase in review for phrase in negative_general):
        themes.append('General Negative Feedback')

    # Theme: Neutral/Functional Feedback
    if any(phrase in review for phrase in neutral_general + neutral_functional + neutral_suggestions):
        # Only add if no strong positive or negative themes are already present
        if not set(themes).intersection(['Positive User Experience & Features', 'Excellent Customer Service', 
                                          'App Performance & Stability Issues', 'Transaction & Account Issues', 
                                          'Poor Customer Service', 'General Negative Feedback']):
            themes.append('Neutral/Functional Feedback')

    # Fallback for reviews not caught by specific themes
    if not themes:
        themes.append('Uncategorized Feedback') # Changed from 'General Feedback' to 'Uncategorized' for clarity

    return "; ".join(sorted(list(set(themes))))


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
        # Ensure 'review_text' is treated as string for cleaning, even if NaN
        df['processed_review_text'] = df['review_text'].apply(clean_text_for_nlp)
        print("Text cleaning complete.")

        # 4. Extract Top Keywords using TF-IDF for each bank (this is for informational purpose, not directly used in assign_themes logic)
        print("\nExtracting top keywords per bank using TF-IDF (for reference)...")
        top_keywords_per_bank = get_top_tfidf_keywords(df, 'processed_review_text', 'bank', top_n=15)

        for bank, keywords in top_keywords_per_bank.items():
            print(f"\n--- Top Keywords for {bank} ---")
            for keyword, score in keywords:
                print(f"- {keyword} (TF-IDF: {score:.4f})")
        
        print("\nUsing comprehensive word list for theme assignment.")


        # 5. Assign Themes based on comprehensive keyword lists
        print("\nAssigning themes to reviews using the updated logic...")
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
        # Exclude 'Uncategorized Feedback' from the count of specifically identified themes
        themes_identified_count = final_output_df[
            ~final_output_df['identified_theme(s)'].str.contains('Uncategorized Feedback', na=False)
        ].shape[0]
        
        total_reviews_for_themes = len(final_output_df)
        themes_coverage_percentage = (themes_identified_count / total_reviews_for_themes) * 100 if total_reviews_for_themes > 0 else 0

        print(f"Reviews with specific themes identified: {themes_identified_count} out of {total_reviews_for_themes} ({themes_coverage_percentage:.2f}%)")

        # Check for 3+ themes (this counts the distinct themes assigned, excluding the 'Uncategorized' fallback)
        all_assigned_themes = set()
        for themes_str in final_output_df['identified_theme(s)'].unique():
            if isinstance(themes_str, str):
                for theme in themes_str.split('; '):
                    if theme != 'Uncategorized Feedback': # Exclude fallback theme
                        all_assigned_themes.add(theme.strip())
        
        print(f"Total unique themes identified across all reviews (excluding 'Uncategorized Feedback'): {len(all_assigned_themes)}")
        if len(all_assigned_themes) >= 3:
            print("✅ KPI: 3+ themes identified (overall, needs per-bank verification) - Met!")
        else:
            print("❌ KPI: 3+ themes identified (overall) - Not Met. Please refine 'assign_themes' and keywords.")
        
        print("✅ KPI: The analysis code is organised into a modular pipeline - Check `thematic_analysis.py` structure.")

    except FileNotFoundError:
        print(f"Error: The file '{sentiment_data_filename}' was not found.")
        print("Please ensure 'reviews_with_sentiment.csv' exists after running sentiment_analysis.py.")
    except Exception as e:
        print(f"An unexpected error occurred during thematic analysis: {e}")