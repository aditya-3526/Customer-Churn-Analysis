import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_data(filepath):
    """Load the dataset."""
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """Clean missing values and correct data types."""
    # Handle edge case: TotalCharges contains blank strings for newly onboarded customers. Convert to numeric to facilitate analysis.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill NaN with 0 for new customers (tenure = 0)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    return df

def create_features(df):
    """Create new derived features."""
    # Segment customers into tenure cohorts for lifecycle analysis
    def tenure_group(tenure):
        if tenure <= 12:
            return '0-1 Year'
        elif tenure <= 24:
            return '1-2 Years'
        elif tenure <= 36:
            return '2-3 Years'
        elif tenure <= 48:
            return '3-4 Years'
        elif tenure <= 60:
            return '4-5 Years'
        else:
            return '5+ Years'
            
    df['TenureGroup'] = df['tenure'].apply(tenure_group)
    
    # Spending segments (High vs Low based on median MonthlyCharges)
    median_charge = df['MonthlyCharges'].median()
    df['SpendingSegment'] = df['MonthlyCharges'].apply(lambda x: 'High' if x > median_charge else 'Low')
    
    # Create binary indicator for robust long-term commitment (1-year or 2-year contracts)
    df['IsLongTerm'] = df['Contract'].apply(lambda x: 1 if x in ['One year', 'Two year'] else 0)
    
    return df

def preprocess_data(df):
    """Encode categoricals and scale numerical features."""
    # Drop customerID as it's not a useful feature for modeling
    df = df.drop(columns=['customerID'], errors='ignore')
    
    # Separate numeric and categorical
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Exclude the target variable 'Churn' from the independent features encoding pipeline
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'Churn' in categorical_cols:
        categorical_cols.remove('Churn')
        
    # Scale numeric
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Encode categorical (One-Hot Encoding for multi-class, Label Encoding for binary)
    # Binary columns
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'SpendingSegment']
    
    le = LabelEncoder()
    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
            
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    # Multi-class columns (One-Hot)
    multi_cols = [col for col in categorical_cols if col not in binary_cols]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    
    return df

def get_preprocessed_pipeline(filepath):
    """Run the entire preprocessing pipeline."""
    df_raw = load_data(filepath)
    df_clean = clean_data(df_raw)
    df_features = create_features(df_clean)
    
    # Keep df_features for EDA (unscaled, unencoded)
    # Create df_processed for modeling (scaled, encoded)
    df_processed = preprocess_data(df_features.copy())
    
    return df_features, df_processed

if __name__ == "__main__":
    # For testing the script directly
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df_features, df_processed = get_preprocessed_pipeline(filepath)
    print(f"Features shape: {df_features.shape}")
    print(f"Processed shape: {df_processed.shape}")
    print("Preprocessing completed successfully.")
