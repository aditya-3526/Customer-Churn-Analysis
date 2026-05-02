import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from data_preprocessing import get_preprocessed_pipeline

# Configure plot style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def calculate_revenue_at_risk(df_features, df_processed):
    """
    Train a Logistic Regression model to predict churn probabilities for all customers,
    and calculate revenue at risk.
    """
    # Baseline Model Evaluation: Score the entire customer base to quantify total revenue at risk
    X = df_processed.drop(columns=['Churn'])
    y = df_processed['Churn']
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X, y)
    
    # Predict Probabilities
    churn_probs = lr_model.predict_proba(X)[:, 1]
    
    # Combine predictions with original business features
    df_impact = df_features.copy()
    df_impact['Churn_Probability'] = churn_probs
    df_impact['Predicted_Churn'] = (churn_probs > 0.5).astype(int)
    
    # Calculate ARPU and Total Revenue
    arpu = df_impact['MonthlyCharges'].mean()
    total_revenue = df_impact['MonthlyCharges'].sum()
    
    # Revenue at Risk (Aggregate of Monthly Charges for customers flagged as high flight risk)
    revenue_at_risk = df_impact[df_impact['Predicted_Churn'] == 1]['MonthlyCharges'].sum()
    
    print("\n--- Revenue Impact Analysis ---")
    print(f"Total Customers: {len(df_impact)}")
    print(f"Average Revenue Per User (ARPU): ${arpu:.2f} / month")
    print(f"Total Monthly Revenue: ${total_revenue:,.2f} / month")
    print(f"Monthly Revenue at Risk: ${revenue_at_risk:,.2f} / month")
    print(f"% of Revenue at Risk: {(revenue_at_risk / total_revenue) * 100:.1f}%\n")
    
    return lr_model, df_impact, X, revenue_at_risk

def simulate_retention_strategy(lr_model, df_impact, X_processed, original_revenue_at_risk, output_dir):
    """
    Business Simulation: Evaluate the ROI of a targeted retention campaign 
    offering a 10% discount for at-risk month-to-month customers to upgrade to an annual contract.
    """
    print("--- Retention Strategy Simulation ---")
    print("Strategy: Convert high-risk (prob > 0.5) Month-to-Month customers to 'One Year' contracts with a 10% discount.")
    
    # Identify high-value operational targets: customers likely to churn who are not locked into long-term contracts
    target_mask = (df_impact['Predicted_Churn'] == 1) & (df_impact['Contract'] == 'Month-to-month')
    target_customers = df_impact[target_mask].index
    
    print(f"Targeted Customers for Strategy: {len(target_customers)}")
    
    # Create modified dataset for simulation
    X_simulated = X_processed.copy()
    
    # Update Contract feature for targeted users to simulate model risk drop
    if 'Contract_One year' in X_simulated.columns:
        X_simulated.loc[target_customers, 'Contract_One year'] = 1
    if 'Contract_Two year' in X_simulated.columns:
         X_simulated.loc[target_customers, 'Contract_Two year'] = 0
         
    # Apply 10% discount to MonthlyCharges
    new_probs = lr_model.predict_proba(X_simulated)[:, 1]
    df_impact['New_Churn_Probability'] = new_probs
    df_impact['New_Predicted_Churn'] = (new_probs > 0.5).astype(int)
    
    # Calculate revised revenue at risk, accounting for the discounted rate for strategy adopters
    # Assume a conservative 30% campaign acceptance rate based on industry benchmarks
    acceptance_rate = 0.30
    np.random.seed(42)
    accepted = np.random.rand(len(target_customers)) < acceptance_rate
    accepted_indices = target_customers[accepted]

    print(f"Assumed Acceptance Rate: {acceptance_rate*100}% ({len(accepted_indices)} customers)")
    
    # Base new monthly charges: 10% discount for those who accepted
    df_impact['Simulated_MonthlyCharges'] = df_impact['MonthlyCharges']
    df_impact.loc[accepted_indices, 'Simulated_MonthlyCharges'] *= 0.90
    
    # Update Churn Prediction based on them accepting the contract
    df_impact['Final_Predicted_Churn'] = df_impact['Predicted_Churn']
    df_impact.loc[accepted_indices, 'Final_Predicted_Churn'] = df_impact.loc[accepted_indices, 'New_Predicted_Churn']
    
    new_revenue_at_risk = df_impact[df_impact['Final_Predicted_Churn'] == 1]['Simulated_MonthlyCharges'].sum()
    revenue_saved = original_revenue_at_risk - new_revenue_at_risk
    
    print(f"New Monthly Revenue at Risk: ${new_revenue_at_risk:,.2f}")
    print(f"Estimated Monthly Revenue Saved: ${revenue_saved:,.2f}")
    
    # Visualize strategy impact
    labels = ['Original Risk', 'Simulated Risk\n(After Strategy)']
    values = [original_revenue_at_risk, new_revenue_at_risk]
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=labels, y=values, palette=['#DD8452', '#4C72B0'], hue=labels, legend=False)
    plt.title('Revenue at Risk: Before vs After Retention Strategy', fontsize=16, fontweight='bold')
    plt.ylabel('Monthly Revenue at Risk ($)', fontsize=14)
    
    for i, v in enumerate(values):
        plt.text(i, v + (max(values)*0.02), f'${v:,.0f}', ha='center', fontweight='bold', fontsize=12)
        
    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, '07_strategy_simulation.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = os.path.join(base_dir, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
        
    output_dir = os.path.join(base_dir, 'visuals')
    
    print("Loading data for retention strategy...")
    df_features, df_processed = get_preprocessed_pipeline(filepath)
    
    lr_model, df_impact, X_processed, rev_at_risk = calculate_revenue_at_risk(df_features, df_processed)
    simulate_retention_strategy(lr_model, df_impact, X_processed, rev_at_risk, output_dir)
