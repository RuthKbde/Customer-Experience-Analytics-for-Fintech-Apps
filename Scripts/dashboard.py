import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import sqlite3
import os

# --- Configuration ---
DATABASE_FILE = os.path.join('data', 'customer_reviews.db')
APP_TITLE = "Fintech App Customer Experience Dashboard"

# --- Data Loading Function ---
def load_data_from_db():
    """Loads review data from the SQLite database."""
    conn = None
    df = pd.DataFrame() # Initialize empty DataFrame
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        query = "SELECT review_text, sentiment_label, sentiment_score, identified_themes FROM CUSTOMER_REVIEWS"
        df = pd.read_sql_query(query, conn)
        print(f"Loaded {len(df)} records from the database.")

        # NEW: Normalize column names to lowercase for robust access
        df.columns = df.columns.str.lower()

        # Now, check if 'identified_themes' (lowercase) exists and rename it
        if 'identified_themes' in df.columns:
            df.rename(columns={'identified_themes': 'identified_theme(s)'}, inplace=True)
        else:
            # If for some reason 'identified_themes' still isn't found,
            # create an empty column to prevent further errors and print a warning.
            print("Warning: 'identified_themes' column not found in loaded DataFrame after lowercasing. Theme charts might be empty.")
            df['identified_theme(s)'] = '' # Create an empty column to avoid key errors later

    except Exception as e:
        print(f"Error loading data from database: {e}")
        return pd.DataFrame(columns=['review_text', 'sentiment_label', 'sentiment_score', 'identified_theme(s)'])
    finally:
        if conn:
            conn.close()
    return df

# Load data initially
df_reviews = load_data_from_db()

# Ensure 'identified_theme(s)' is a string, then split into a list of themes
if 'identified_theme(s)' in df_reviews.columns:
    df_reviews['identified_theme(s)'] = df_reviews['identified_theme(s)'].fillna('')
    df_reviews['themes_list'] = df_reviews['identified_theme(s)'].apply(lambda x: [theme.strip() for theme in x.split(';') if theme.strip()])
else:
    df_reviews['themes_list'] = [[] for _ in range(len(df_reviews))]
    # This warning should ideally not show now if the above fix works
    print("Warning: 'identified_theme(s)' column not found in DataFrame for themes_list creation. Theme charts may be empty.")


# Initialize Dash app
app = dash.Dash(__name__, title=APP_TITLE)
server = app.server

# --- Dashboard Layout ---
app.layout = html.Div(children=[
    html.H1(children=APP_TITLE, style={'textAlign': 'center', 'color': '#0A3B5F'}),

    html.Div(children=[
        html.Div([
            html.Label("Select Sentiment:"),
            dcc.Dropdown(
                id='sentiment-dropdown',
                options=[
                    {'label': 'All Sentiments', 'value': 'All Sentiments'},
                    {'label': 'Positive', 'value': 'positive'},
                    {'label': 'Negative', 'value': 'negative'},
                    {'label': 'Neutral', 'value': 'neutral'}
                ],
                value='All Sentiments',
                clearable=False
            )
        ], style={'width': '98%', 'display': 'inline-block', 'padding': '10px'})
    ]),

    html.Div([
        html.Div([
            dcc.Graph(id='sentiment-distribution-chart')
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            dcc.Graph(id='top-themes-chart')
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '10px'})
    ], style={'display': 'flex', 'flex-wrap': 'wrap'}),

    html.Div([
        html.H3("Sentiment Score Over Time", style={'textAlign': 'center'}),
        dcc.Graph(id='sentiment-over-time-chart')
    ], style={'padding': '10px'})

])

# --- Callbacks ---

@app.callback(
    [Output('sentiment-distribution-chart', 'figure'),
     Output('top-themes-chart', 'figure'),
     Output('sentiment-over-time-chart', 'figure')],
    [Input('sentiment-dropdown', 'value')]
)
def update_charts(selected_sentiment):
    filtered_df = df_reviews.copy()

    # Filter by Sentiment
    if selected_sentiment != 'All Sentiments':
        filtered_df = filtered_df[filtered_df['sentiment_label'] == selected_sentiment]
    
    # Handle empty DataFrame after filtering
    if filtered_df.empty:
        empty_fig = px.bar(title="No Data Available for Selection")
        empty_fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, annotations=[dict(text="No data to display", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5, font_size=20)])
        return empty_fig, empty_fig, empty_fig

    # --- 1. Sentiment Distribution Chart ---
    sentiment_counts = filtered_df['sentiment_label'].value_counts(normalize=True).reset_index()
    sentiment_counts.columns = ['Sentiment', 'Percentage']
    sentiment_counts['Percentage'] = sentiment_counts['Percentage'] * 100
    sentiment_fig = px.pie(
        sentiment_counts,
        values='Percentage',
        names='Sentiment',
        title='Sentiment Distribution',
        color='Sentiment',
        color_discrete_map={
            'positive': 'green',
            'neutral': 'blue',
            'negative': 'red'
        }
    )
    sentiment_fig.update_traces(textinfo='percent+label', pull=[0.05 if s == 'negative' else 0 for s in sentiment_counts['Sentiment']])


    # --- 2. Top Themes Chart ---
    all_themes = [theme for sublist in filtered_df['themes_list'] for theme in sublist if theme.strip() != '' and theme.strip() != 'Uncategorized Feedback']
    theme_counts = pd.Series(all_themes).value_counts().reset_index()
    theme_counts.columns = ['Theme', 'Count']
    top_n_themes = 10
    theme_counts = theme_counts.head(top_n_themes)

    themes_fig = px.bar(
        theme_counts,
        x='Count',
        y='Theme',
        orientation='h',
        title=f'Top {top_n_themes} Themes',
        labels={'Count': 'Number of Reviews', 'Theme': 'Identified Theme'},
        color='Count',
        color_continuous_scale=px.colors.sequential.Teal
    )
    themes_fig.update_layout(yaxis={'categoryorder':'total ascending'})


    # --- 3. Sentiment Over Time Chart ---
    if 'date' in filtered_df.columns:
        filtered_df['date'] = pd.to_datetime(filtered_df['date'])
        sentiment_over_time = filtered_df.groupby('date')['sentiment_score'].mean().reset_index()
        sentiment_over_time_fig = px.line(
            sentiment_over_time,
            x='date',
            y='sentiment_score',
            title='Average Sentiment Score Over Time',
            labels={'date': 'Date', 'sentiment_score': 'Average Sentiment Score'}
        )
        sentiment_over_time_fig.update_layout(hovermode="x unified")
    else:
        sentiment_over_time_fig = px.line(title="Date column not available for time series analysis")


    return sentiment_fig, themes_fig, sentiment_over_time_fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)