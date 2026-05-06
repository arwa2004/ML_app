import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor

# Configuration
DATA_PATH = r'C:\Users\User\Downloads\healthcare_dataset_cleaned.csv'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def train_all_three():
    print(f"🚀 Chargement des données...")
    df = pd.read_csv(DATA_PATH)
    
    # 🛠️ Feature Engineering global
    if 'Stay_Duration' not in df.columns:
        date_col = 'Date of Admission' if 'Date of Admission' in df.columns else 'Admission Date'
        df[date_col] = pd.to_datetime(df[date_col])
        df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
        df['Stay_Duration'] = (df['Discharge Date'] - df[date_col]).dt.days
    df['Health_Risk_Score'] = (df['Age'] * df['Stay_Duration']) / 100

    categorical_cols = ['Gender', 'Blood Type', 'Medical Condition', 'Admission Type']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        joblib.dump(le, os.path.join(MODEL_DIR, f'le_{col.lower().replace(" ", "_")}.joblib'))
    
    features_list = ['Age', 'Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Stay_Duration', 'Health_Risk_Score']
    X = df[features_list]

    # 1. 💉 DIAGNOSTIC (XGBoost) - Ilef
    le_target = LabelEncoder()
    df['target_diag'] = le_target.fit_transform(df['Test Results'])
    joblib.dump(le_target, os.path.join(MODEL_DIR, 'le_target.joblib'))
    xgb = XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.1, random_state=42)
    xgb.fit(X, df['target_diag'])
    joblib.dump(xgb, os.path.join(MODEL_DIR, 'xgboost_diag.joblib'))

    # 2. 💰 BILLING (Random Forest) - Jasser
    billing_col = 'Billing Amount' if 'Billing Amount' in df.columns else 'billing_amount'
    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    rf_reg.fit(X, df[billing_col])
    joblib.dump(rf_reg, os.path.join(MODEL_DIR, 'rf_cost_regressor.joblib'))

    # 3. 📊 SEGMENTATION (PCA + KMeans) - Arwa
    print("📊 Entraînement de la Segmentation (Arwa)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2) # Pour l'affichage 2D
    X_pca = pca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    joblib.dump(pca, os.path.join(MODEL_DIR, 'pca_model.joblib'))
    joblib.dump(kmeans, os.path.join(MODEL_DIR, 'kmeans_model.joblib'))

    print(f"✅ Mission accomplie ! Les 3 piliers du notebook sont prêts.")

if __name__ == "__main__":
    train_all_three()
