import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from data_preprocessing import get_preprocessed_pipeline

# Plot styling
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def train_and_evaluate(df_processed):
    """Train models and evaluate them."""
    # Target and Features
    X = df_processed.drop(columns=['Churn'])
    y = df_processed['Churn']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Logistic Regression (Primary model chosen for high interpretability and straightforward coefficient extraction for business drivers)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    
    # Decision Tree (Secondary model for non-linear pattern validation)
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    dt_probs = dt_model.predict_proba(X_test)[:, 1]
    
    # Evaluate models on key business performance metrics
    results = {
        'Logistic Regression': {
            'Accuracy': accuracy_score(y_test, lr_preds),
            'Precision': precision_score(y_test, lr_preds),
            'Recall': recall_score(y_test, lr_preds),
            'ROC-AUC': roc_auc_score(y_test, lr_probs)
        },
        'Decision Tree': {
            'Accuracy': accuracy_score(y_test, dt_preds),
            'Precision': precision_score(y_test, dt_preds),
            'Recall': recall_score(y_test, dt_preds),
            'ROC-AUC': roc_auc_score(y_test, dt_probs)
        }
    }
    
    print("\n--- Model Evaluation Results ---")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
    return lr_model, dt_model, X.columns

def extract_feature_importance(lr_model, feature_names, output_dir):
    """Extract and plot feature importance from Logistic Regression coefficients."""
    coefficients = lr_model.coef_[0]
    
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Absolute_Importance': np.abs(coefficients)
    })
    
    # Isolate the top 15 features to prioritize operational focus
    feat_imp = feat_imp.sort_values(by='Absolute_Importance', ascending=False).head(15)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    # We use a divergent palette based on coefficient value (positive vs negative impact)
    sns.barplot(data=feat_imp, x='Coefficient', y='Feature', hue=np.sign(feat_imp.Coefficient), legend=False, palette={1:"#DD8452", -1:"#4C72B0"})
    
    plt.title('Top 15 Churn Drivers (Logistic Regression)', fontsize=16, fontweight='bold')
    plt.xlabel('Impact on Churn (Coefficient > 0 Increases Churn)', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.axvline(x=0, color='black', linestyle='--')
    plt.tight_layout()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(os.path.join(output_dir, '06_feature_importance.png'), dpi=300)
    plt.close()
    
    print("\n--- Top Churn Drivers Identified ---")
    print(feat_imp[['Feature', 'Coefficient']].head(10))

if __name__ == "__main__":
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = os.path.join(base_dir, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
        
    output_dir = os.path.join(base_dir, 'visuals')
    
    print("Loading data for modeling...")
    _, df_processed = get_preprocessed_pipeline(filepath)
    
    lr_model, dt_model, feature_names = train_and_evaluate(df_processed)
    extract_feature_importance(lr_model, feature_names, output_dir)
