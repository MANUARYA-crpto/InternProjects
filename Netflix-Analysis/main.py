import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from wordcloud import WordCloud

# --- CONFIGURATION ---
DATA_PATH = 'data/netflix_titles.csv'
OUTPUT_DIR = 'output'

def create_folders():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def load_and_clean_data():
    print("--- STEP 1: LOAD & CLEAN DATA ---")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH}. Please put the CSV in 'data/' folder.")
    
    # [cite_start]PDF Source: Page 3 [cite: 4913]
    data = pd.read_csv(DATA_PATH)
    print(f"Original Shape: {data.shape}")
    
    # [cite_start]1. Drop Duplicates [cite: 4920]
    data.drop_duplicates(inplace=True)
    
    # [cite_start]2. Handle Missing Values [cite: 4922-4925]
    # The PDF suggests dropping rows with missing director, cast, or country
    # We will be slightly more lenient to keep more data for other plots, 
    # but dropping rows with missing 'date_added' is crucial for time plots.
    data.dropna(subset=['date_added', 'rating'], inplace=True)
    
    # [cite_start]3. Convert Date [cite: 4927]
    # .strip() handles potential leading/trailing spaces in the date string
    data['date_added'] = pd.to_datetime(data['date_added'].str.strip())
    
    # [cite_start]4. Extract Date Features [cite: 4957-4958]
    data['year_added'] = data['date_added'].dt.year
    data['month_added'] = data['date_added'].dt.month
    
    print(f"Cleaned Shape: {data.shape}")
    return data

def plot_content_distribution(data):
    print("--- CHART 1: Content Type Distribution ---")
    # [cite_start]PDF Source: Page 4 & 11 [cite: 4931-4940, 5019-5021]
    
    type_counts = data['type'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar Plot
    sns.barplot(x=type_counts.index, y=type_counts.values, palette='Set2', ax=axes[0])
    axes[0].set_title('Distribution of Content by Type')
    axes[0].set_ylabel('Count')
    
    # Pie Chart
    axes[1].pie(type_counts, labels=type_counts.index, autopct='%.0f%%', colors=sns.color_palette('Set2'))
    axes[1].set_title('Total Content on Netflix')
    
    plt.savefig(f"{OUTPUT_DIR}/01_content_distribution.png")
    plt.close()

def plot_ratings(data):
    print("--- CHART 2: Ratings Analysis ---")
    # [cite_start]PDF Source: Page 14-15 [cite: 5122-5127]
    
    plt.figure(figsize=(12, 6))
    order = data['rating'].value_counts().index
    sns.countplot(x='rating', data=data, order=order, palette='viridis')
    plt.title('Rating Frequency on Netflix')
    plt.xlabel('Rating Types')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.savefig(f"{OUTPUT_DIR}/02_ratings_distribution.png")
    plt.close()

def plot_top_countries(data):
    print("--- CHART 3: Top 10 Countries ---")
    # [cite_start]PDF Source: Page 18 [cite: 5186-5193]
    
    # We focus on the primary country listed
    data['first_country'] = data['country'].dropna().apply(lambda x: x.split(',')[0])
    top_countries = data['first_country'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_countries.index, y=top_countries.values, palette='mako')
    plt.title('Top 10 Countries with Most Content')
    plt.xlabel('Country')
    plt.ylabel('Number of Titles')
    plt.xticks(rotation=45)
    
    plt.savefig(f"{OUTPUT_DIR}/03_top_countries.png")
    plt.close()

def plot_release_trends(data):
    print("--- CHART 4: Release Trends (Monthly & Yearly) ---")
    # [cite_start]PDF Source: Page 20-21 [cite: 5224-5265]
    
    # Yearly Trend
    yearly_movies = data[data['type'] == 'Movie']['year_added'].value_counts().sort_index()
    yearly_shows = data[data['type'] == 'TV Show']['year_added'].value_counts().sort_index()
    
    plt.figure(figsize=(14, 6))
    plt.plot(yearly_movies.index, yearly_movies.values, label='Movies', linewidth=2)
    plt.plot(yearly_shows.index, yearly_shows.values, label='TV Shows', linewidth=2)
    plt.title('Yearly Releases of Movies and TV Shows')
    plt.xlabel('Year')
    plt.ylabel('Number of Releases')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(2010, 2021) # Focus on recent years
    
    plt.savefig(f"{OUTPUT_DIR}/04_yearly_trend.png")
    plt.close()

def plot_genres(data):
    print("--- CHART 5: Top Genres ---")
    # [cite_start]PDF Source: Page 23 [cite: 5289-5296]
    
    # Movies Genres
    plt.figure(figsize=(14, 8))
    top_genres = data[data['type'] == 'Movie']['listed_in'].value_counts().head(10)
    
    sns.barplot(x=top_genres.index, y=top_genres.values, palette='Reds_r')
    plt.title('Top 10 Popular Movie Genres')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=60, ha='right')
    plt.tight_layout()
    
    plt.savefig(f"{OUTPUT_DIR}/05_top_genres_movies.png")
    plt.close()

def plot_directors(data):
    print("--- CHART 6: Top Directors ---")
    # [cite_start]PDF Source: Page 27 [cite: 5346-5348]
    
    # Filter out missing directors
    directors = data['director'].dropna().value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=directors.index, y=directors.values, palette='Blues_r')
    plt.title('Top 10 Directors by Content Volume')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Titles')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_top_directors.png")
    plt.close()

def generate_wordcloud(data):
    print("--- CHART 7: Word Cloud ---")
    # [cite_start]PDF Source: Page 6 [cite: 4982-4990]
    
    text = " ".join(title for title in data.title)
    
    # Create wordcloud
    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in Netflix Titles')
    
    plt.savefig(f"{OUTPUT_DIR}/07_title_wordcloud.png")
    plt.close()

if __name__ == "__main__":
    create_folders()
    
    df = load_and_clean_data()
    
    plot_content_distribution(df)
    plot_ratings(df)
    plot_top_countries(df)
    plot_release_trends(df)
    plot_genres(df)
    plot_directors(df)
    generate_wordcloud(df)
    
    print("\nNETFLIX PROJECT COMPLETE. CHECK 'output/' FOLDER.")