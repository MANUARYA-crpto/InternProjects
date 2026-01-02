import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import os
import glob
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.seasonal import seasonal_decompose

# --- CONFIGURATION ---
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
CUTOFF_DATE = '2014-09-15 00:00:00' # [cite: 1218]
WINDOW_SIZE = 24 # [cite: 1266]

def create_folders():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def load_and_aggregate_data():
    print("--- STEP 1: LOADING & AGGREGATING DATA ---")
    # PDF Source: Page 10-11
    files = glob.glob(os.path.join(DATA_DIR, "uber-raw-data-*.csv"))
    
    if not files:
        raise FileNotFoundError("No CSV files found in 'data/'. Please download the 2014 Uber datasets.")
    
    print(f"Found {len(files)} files. Loading...")
    
    dataframes = [pd.read_csv(f) for f in files]
    uber_data = pd.concat(dataframes, ignore_index=True)
    
    # --- FIX: Clean Column Names ---
    # This removes hidden spaces (e.g. "Date/Time " becomes "Date/Time")
    uber_data.columns = [c.strip() for c in uber_data.columns]
    
    # Check if the column exists now
    if 'Date/Time' not in uber_data.columns:
        print(f"ERROR: Expected 'Date/Time' column but found: {list(uber_data.columns)}")
        # Try to find a partial match just in case
        possible_cols = [c for c in uber_data.columns if 'date' in c.lower()]
        if possible_cols:
            print(f"Renaming '{possible_cols[0]}' to 'Date/Time'...")
            uber_data.rename(columns={possible_cols[0]: 'Date/Time'}, inplace=True)
    
    # preprocessing: convert to datetime
    print("Converting timestamps (this takes a moment)...")
    uber_data['Date/Time'] = pd.to_datetime(uber_data['Date/Time'], format='%m/%d/%Y %H:%M:%S')
    
    # Aggregate to Hourly Counts
    print("Aggregating to hourly counts...")
    hourly_counts = uber_data.set_index('Date/Time').resample('h')['Base'].count()
    hourly_df = pd.DataFrame({'Count': hourly_counts})
    
    return hourly_df
def perform_eda(df):
    print("--- STEP 2: EDA & DECOMPOSITION ---")
    # PDF Source: Page 13-14
    
    # 1. Plot Trend
    plt.figure(figsize=(15, 6))
    plt.plot(df['Count'], color='darkslateblue', linewidth=1)
    plt.title("Hourly Uber Trips (Apr-Sep 2014)")
    plt.savefig(f"{OUTPUT_DIR}/01_hourly_trips.png")
    plt.close()
    
    # 2. Seasonal Decompose [cite: 1152]
    # We use a period of 24 (Daily seasonality)
    result = seasonal_decompose(df['Count'], model='add', period=24)
    
    plt.figure(figsize=(15, 10))
    result.plot()
    plt.savefig(f"{OUTPUT_DIR}/02_seasonal_decompose.png")
    plt.close()
    print(f"EDA plots saved to {OUTPUT_DIR}/")

def create_lagged_features(data, window_size):
    # PDF Source: Page 16 [cite: 1082-1087]
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def prepare_train_test(df):
    print("--- STEP 3: TRAIN/TEST SPLIT ---")
    # PDF Source: Page 15-16
    
    train_data = df.loc[:CUTOFF_DATE]
    test_data = df.loc[CUTOFF_DATE:]
    
    # Prepare features using Windowing [cite: 1269]
    X_train, y_train = create_lagged_features(train_data['Count'].values, WINDOW_SIZE)
    
    # For testing, we need the last 'window_size' points from train to predict the first test point
    concatenated = np.concatenate([train_data['Count'].values[-WINDOW_SIZE:], test_data['Count'].values])
    X_test, y_test = create_lagged_features(concatenated, WINDOW_SIZE)
    
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    return X_train, y_train, X_test, y_test, test_data.index

def train_models(X_train, y_train, X_test):
    print("--- STEP 4: MODEL TRAINING ---")
    # Using BEST PARAMETERS explicitly identified in the PDF to save time
    
    # 1. XGBoost [cite: 1305]
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        colsample_bytree=1.0, learning_rate=0.1, max_depth=6, 
        n_estimators=300, subsample=0.6, random_state=12345
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    
    # 2. Random Forest [cite: 1375]
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(
        max_depth=30, max_features=None, min_samples_leaf=2, 
        min_samples_split=5, n_estimators=100, random_state=12345, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # 3. Gradient Boosting (GBRT) [cite: 1427]
    print("Training Gradient Boosting...")
    gbr_model = GradientBoostingRegressor(
        learning_rate=0.1, max_depth=5, max_features='sqrt', 
        min_samples_leaf=1, min_samples_split=5, n_estimators=300, random_state=12345
    )
    gbr_model.fit(X_train, y_train)
    gbr_pred = gbr_model.predict(X_test)
    
    return xgb_pred, rf_pred, gbr_pred

def ensemble_predictions(xgb_p, rf_p, gbr_p):
    print("--- STEP 5: ENSEMBLE ---")
    # PDF Source: Page 24 [cite: 1500-1503]
    # Weights calculated based on inverse MAPE
    weights = [0.368, 0.322, 0.310]
    
    ensemble_pred = (weights[0] * xgb_p) + (weights[1] * rf_p) + (weights[2] * gbr_p)
    return ensemble_pred

def evaluate_and_plot(y_test, xgb_p, rf_p, gbr_p, ens_p, test_index):
    print("--- STEP 6: EVALUATION ---")
    
    # Calculate MAPE [cite: 1335]
    xgb_mape = mean_absolute_percentage_error(y_test, xgb_p)
    rf_mape = mean_absolute_percentage_error(y_test, rf_p)
    gbr_mape = mean_absolute_percentage_error(y_test, gbr_p)
    ens_mape = mean_absolute_percentage_error(y_test, ens_p)
    
    print(f"XGBoost MAPE: {xgb_mape:.2%}")   # Expected: ~8.37%
    print(f"Random Forest MAPE: {rf_mape:.2%}") # Expected: ~9.61%
    print(f"GBRT MAPE: {gbr_mape:.2%}")      # Expected: ~10.02%
    print(f"Ensemble MAPE: {ens_mape:.2%}")  # Expected: ~8.60%
    
    # Visualization [cite: 1451]
    plt.figure(figsize=(18, 8))
    plt.plot(test_index, y_test, label='Actual Test Data', color='gray', alpha=0.6)
    plt.plot(test_index, xgb_p, label='XGBoost', linestyle='--', alpha=0.7)
    plt.plot(test_index, ens_p, label='Ensemble', color='purple', linewidth=2)
    
    plt.title('Uber Trips Forecast: Actual vs Ensemble')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Trips')
    plt.savefig(f"{OUTPUT_DIR}/03_forecast_comparison.png")
    plt.close()
    print(f"Forecast plot saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    create_folders()
    
    # 1. Load
    df = load_and_aggregate_data()
    
    # 2. EDA
    perform_eda(df)
    
    # 3. Prep
    X_train, y_train, X_test, y_test, test_index = prepare_train_test(df)
    
    # 4. Train
    xgb_pred, rf_pred, gbr_pred = train_models(X_train, y_train, X_test)
    
    # 5. Ensemble
    ens_pred = ensemble_predictions(xgb_pred, rf_pred, gbr_pred)
    
    # 6. Evaluate
    evaluate_and_plot(y_test, xgb_pred, rf_pred, gbr_pred, ens_pred, test_index)
    
    print("\nPROJECT COMPLETE.")