import os
import sys
import pandas as pd
import logging

# Ensure we can import from the project root
sys.path.append(os.getcwd())

from main_pipeline import load_and_prepare_data
import src.config as cfg

logging.basicConfig(level=logging.INFO)

try:
    if os.path.exists(cfg.DATAPATH):
        print(f"Testing load_and_prepare_data with {cfg.DATAPATH}...")
        df_raw, df_features, y_targets = load_and_prepare_data(cfg.DATAPATH)
        print("Success!")
        print(f"df_raw shape: {df_raw.shape}")
        print(f"df_features shape: {df_features.shape}")
        print(f"y_targets shape: {y_targets.shape}")
        print(f"Columns: {df_features.columns.tolist()}")
    else:
        print(f"Data path {cfg.DATAPATH} not found. Skipping test.")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
