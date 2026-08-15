import os
import pandas as pd
from src.components.feature_engineering import FeatureEngineering

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    data_raw_dir = os.path.join(project_root, "data", "raw", "train")
    data_processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(data_processed_dir, exist_ok=True)
    
    print("Loading raw data...")
    provider_df = pd.read_csv(os.path.join(data_raw_dir, "PROVIDERS.csv"))
    beneficiary_df = pd.read_csv(os.path.join(data_raw_dir, "BENEFICIARY.csv"))
    inpatient_df = pd.read_csv(os.path.join(data_raw_dir, "INPATIENT.csv"))
    outpatient_df = pd.read_csv(os.path.join(data_raw_dir, "OUTPATIENT.csv"))
    
    print("Initializing FeatureEngineering...")
    fe = FeatureEngineering()
    
    print("Building provider features (this may take a minute)...")
    provider_features = fe.build_provider_features(
        provider_df=provider_df,
        beneficiary_df=beneficiary_df,
        inpatient_df=inpatient_df,
        outpatient_df=outpatient_df
    )
    
    output_path = os.path.join(data_processed_dir, "provider_features.csv")
    provider_features.to_csv(output_path, index=False)
    print(f"Successfully saved {output_path}")
