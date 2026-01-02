import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIGURATION ---
DATA_PATH = 'data/TCS_stock_data.csv'
OUTPUT_DIR = 'output'
MODEL_DIR = 'models'

def create_folders():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

def load_data():
    print("--- STEP 1: LOAD DATA ---")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH}. Please download the dataset.")
    
    # PDF Source: Page 4 [cite: 3783-3785]
    data = pd.read_csv(DATA_PATH)
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by='Date', inplace=True)
    print(f"Data Loaded. Shape: {data.shape}")
    return data

def preprocess_data(data):
    print("--- STEP 2: PREPROCESSING ---")
    # PDF Source: Page 4 [cite: 3791-3800]
    
    # Check for nulls
    nulls = data.isnull().sum().sum()
    print(f"Null values found: {nulls}")
    
    # Convert columns to numeric if needed and fill missing values
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data.fillna(method='ffill', inplace=True)
    print("Preprocessing complete.")
    return data

def perform_eda(data):
    print("--- STEP 3: EXPLORATORY DATA ANALYSIS (EDA) ---")
    # PDF Source: Page 5-6
    
    # 1. Plot Close Price [cite: 3806-3811]
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Close'], color='blue', label='Close Price')
    plt.title('TCS Stock Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/01_close_price.png")
    plt.close()
    
    # 2. Moving Averages [cite: 3814-3815]
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Close'], label='Close Price', alpha=0.5)
    plt.plot(data['Date'], data['MA50'], label='50-Day MA', color='orange')
    plt.plot(data['Date'], data['MA200'], label='200-Day MA', color='red')
    plt.title('TCS Stock Price with Moving Averages')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/02_moving_averages.png")
    plt.close()
    
    print(f"EDA charts saved to {OUTPUT_DIR}/")
    return data

def feature_engineering(data):
    print("--- STEP 4: FEATURE ENGINEERING ---")
    # PDF Source: Page 6 [cite: 3833-3839]
    
    # Extract Date Features
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Day_of_Week'] = data['Date'].dt.dayofweek
    
    # Lag Features (Previous Day's Close) [cite: 3839]
    data['Prev_Close'] = data['Close'].shift(1)
    
    # Drop rows with NaN created by shifting/rolling
    data.dropna(inplace=True)
    
    print("Features created: Year, Month, Day, Day_of_Week, Prev_Close")
    return data

def train_model(data):
    print("--- STEP 5: MODEL TRAINING (Linear Regression) ---")
    # PDF Source: Page 7 [cite: 3845-3852]
    
    # Select features
    features = ['Open', 'High', 'Low', 'Volume', 'Prev_Close', 'Day_of_Week', 'Month']
    target = 'Close'
    
    X = data[features]
    y = data[target]
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model trained successfully.")
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    print("--- STEP 6: EVALUATION ---")
    # PDF Source: Page 7 [cite: 3854-3857]
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-Squared Score: {r2:.4f}")
    
    # Visualization: Actual vs Predicted [cite: 3861-3866]
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # Diagonal line
    plt.xlabel('Actual Close Price')
    plt.ylabel('Predicted Close Price')
    plt.title('Actual vs Predicted Close Price')
    plt.savefig(f"{OUTPUT_DIR}/03_actual_vs_predicted.png")
    plt.close()
    
    return y_pred

def save_model(model):
    # PDF Source: Page 8 [cite: 3871-3872]
    filename = f"{MODEL_DIR}/TCS_Stock_Predictor.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")

if __name__ == "__main__":
    create_folders()
    
    # Pipeline execution
    df = load_data()
    df = preprocess_data(df)
    df = perform_eda(df)
    df = feature_engineering(df)
    
    model, X_test, y_test = train_model(df)
    evaluate_model(model, X_test, y_test)
    save_model(model)
    
    print("\nPROJECT COMPLETE.")