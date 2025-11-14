# check_feature_selection.py
import pandas as pd
import numpy as np

print("🔍 CHECKING FEATURE SELECTION RESULTS")
print("=" * 60)

try:
    # Load clean data to see what was kept
    clean_df = pd.read_csv('clean_student_data.csv')
    original_df = pd.read_csv('data.csv')
    
    print(f"Original dataset: {original_df.shape[1]} columns")
    print(f"After feature selection: {clean_df.shape[1]} columns")
    print(f"Features removed: {original_df.shape[1] - clean_df.shape[1]}")
    
    print("\n📋 FEATURES IN CLEAN DATA (KEPT):")
    for i, col in enumerate(clean_df.columns, 1):
        print(f"{i:2d}. {col}")
        
except Exception as e:
    print(f"Error: {e}")