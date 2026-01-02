import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# --- CONFIGURATION ---
# Ensure you have 'train.csv' and 'test.csv' in a folder named 'data'
TRAIN_PATH = 'data/raw/train.csv'  
TEST_PATH = 'data/raw/test.csv'
OUTPUT_FOLDER = 'project_outputs'

def create_folders():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created folder '{OUTPUT_FOLDER}' to save plots.")

def load_data():
    print("\n--- STEP 1: LOADING DATA [PDF Source: 31] ---")
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Missing {TRAIN_PATH}. Please create a 'data' folder and put the CSVs there.")
    
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    
    # --- FIX: CLEAN COLUMN NAMES ---
    # This removes spaces (e.g. "profile pic" -> "profile_pic") to match the code
    df_train.columns = [c.strip().replace(" ", "_") for c in df_train.columns]
    df_test.columns = [c.strip().replace(" ", "_") for c in df_test.columns]
    
    # Also fix specific tricky columns often found in this specific dataset
    # Sometimes "nums/length username" comes in as "nums/length_username" or other variations
    rename_map = {
        'nums/length_username': 'nums_length_username',
        'fullname_words': 'fullname_words', 
        'nums/length_fullname': 'nums_length_fullname',
        'name==username': 'name_equals_username',
        'description_length': 'description_length', 
        'external_URL': 'external_url',
        '#posts': 'posts',
        '#followers': 'followers',
        '#follows': 'follows'
    }
    
    # Apply renaming only if the column exists
    df_train.rename(columns=rename_map, inplace=True)
    df_test.rename(columns=rename_map, inplace=True)

    print(f"Train Shape: {df_train.shape}")
    print(f"Test Shape: {df_test.shape}")
    print(f"Columns: {list(df_train.columns)}") # Print columns to verify
    return df_train, df_test

def perform_eda(df):
    print("\n--- STEP 2: EDA & VISUALIZATION [PDF Source: 48] ---")
    
    # 1. Target Distribution [PDF Source: 60]
    plt.figure(figsize=(6, 4))
    sns.countplot(x='fake', data=df)
    plt.title("Distribution of Fake vs Genuine Accounts")
    plt.savefig(f"{OUTPUT_FOLDER}/01_target_distribution.png")
    plt.close()
    
    # 2. Correlation Matrix [PDF Source: 66-69]
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    plt.savefig(f"{OUTPUT_FOLDER}/02_correlation_matrix.png")
    plt.close()
    
    # 3. Profile Pic vs Fake [PDF Source: 76]
    plt.figure(figsize=(6, 4))
    sns.barplot(x='fake', y='profile_pic', data=df)
    plt.title("Profile Picture Presence")
    plt.savefig(f"{OUTPUT_FOLDER}/03_profile_pic_comparison.png")
    plt.close()
    
    print(f"EDA Plots saved to '{OUTPUT_FOLDER}/'")

def preprocess_data(df_train, df_test):
    print("\n--- STEP 3: PREPROCESSING [PDF Source: 92] ---")
    
    # Drop target and unnecessary columns
    # Dropping '#followers_bins' if created during experimentation
    cols_to_drop = ['fake']
    
    X_train = df_train.drop(cols_to_drop, axis=1)
    y_train = df_train['fake']
    
    X_test = df_test.drop(cols_to_drop, axis=1)
    y_test = df_test['fake']
    
    # Scaling Features [PDF Source: 99-100]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Return as DataFrames to keep column names for feature importance
    X_train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_final, y_train, X_test_final, y_test

def train_eval_models(X_train, y_train, X_test, y_test):
    print("\n--- STEP 4 & 5: MODEL BUILDING & EVALUATION [PDF Source: 105] ---")
    
    # --- MODEL A: DECISION TREE [PDF Source: 554] ---
    print("\nTraining Decision Tree...")
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_pred):.4f}")
    
    # --- MODEL B: RANDOM FOREST [PDF Source: 116] ---
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    # Evaluation Metrics [PDF Source: 134-136]
    acc = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, rf_pred))
    
    # Confusion Matrix Plot [PDF Source: 141]
    ConfusionMatrixDisplay.from_predictions(y_test, rf_pred, display_labels=['Genuine', 'Fake'], cmap='Blues')
    plt.title("Confusion Matrix (Random Forest)")
    plt.savefig(f"{OUTPUT_FOLDER}/04_confusion_matrix.png")
    plt.close()
    
    return rf, X_train.columns

def feature_importance(model, feature_names):
    print("\n--- STEP 6: INTERPRETATION [PDF Source: 120] ---")
    
    # Calculate importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plotting [PDF Source: 122-125]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    sns.barplot(x=importances[indices], y=feature_names[indices], palette='viridis')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/05_feature_importance.png")
    plt.show()
    print(f"Feature Importance plot saved to '{OUTPUT_FOLDER}/'")

if __name__ == "__main__":
    create_folders()
    train_df, test_df = load_data()
    perform_eda(train_df)
    X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)
    trained_model, feature_names = train_eval_models(X_train, y_train, X_test, y_test)
    feature_importance(trained_model, feature_names)
    print("\nPROJECT EXECUTION COMPLETE. CHECK THE OUTPUT FOLDER.")