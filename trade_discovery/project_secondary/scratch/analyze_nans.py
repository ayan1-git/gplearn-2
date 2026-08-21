import os
import sys
import pandas as pd
import logging

# Ensure we can import from the project root (src/ lives in project_core/)
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "project_core")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.feature_engineering import calculate_features
import src.config as cfg

logging.basicConfig(level=logging.INFO)

if os.path.exists(cfg.DATAPATH):
    df_raw = pd.read_csv(cfg.DATAPATH, parse_dates=['datetime'], index_col='datetime')
    df_raw.sort_index(inplace=True)
    df_raw.columns = [col.lower() for col in df_raw.columns]
    
    df_features = calculate_features(df_raw)
    
    print("\nNaN count per column:")
    print(df_features.isna().sum())
    
    print(f"\nTotal rows: {len(df_features)}")
    print(f"Rows after dropna(): {len(df_features.dropna())}")
else:
    print(f"Data path {cfg.DATAPATH} not found.")
