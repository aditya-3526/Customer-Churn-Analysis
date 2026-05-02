import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import get_preprocessed_pipeline

# Set enterprise-grade visualization aesthetic for executive presentations
plt.style.use('ggplot')
sns.set_theme(style="whitegrid", palette="muted")

def plot_churn_distribution(df, output_dir):
    """Plot overall churn distribution."""
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x='Churn', palette=['#4C72B0', '#DD8452'])
    plt.title('Overall Churn Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Churn', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    # Superimpose percentage values to highlight the magnitude of the churn problem
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=12)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_churn_distribution.png'), dpi=300)
    plt.close()

def plot_churn_by_contract(df, output_dir):
    """Plot churn by contract type."""
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='Contract', hue='Churn', palette=['#4C72B0', '#DD8452'])
    plt.title('Churn by Contract Type', fontsize=16, fontweight='bold')
    plt.xlabel('Contract Type', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    # Add annotations
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2, height), 
                        ha='center', va='bottom', fontsize=10)
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_churn_by_contract.png'), dpi=300)
    plt.close()

def plot_churn_by_tenure(df, output_dir):
    """Plot churn by tenure group."""
    plt.figure(figsize=(12, 6))
    order = ['0-1 Year', '1-2 Years', '2-3 Years', '3-4 Years', '4-5 Years', '5+ Years']
    ax = sns.countplot(data=df, x='TenureGroup', hue='Churn', order=order, palette=['#4C72B0', '#DD8452'])
    plt.title('Churn by Tenure Group', fontsize=16, fontweight='bold')
    plt.xlabel('Tenure Group', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_churn_by_tenure.png'), dpi=300)
    plt.close()
    
def plot_monthly_charges_distribution(df, output_dir):
    """Plot monthly charges distribution for churn vs retained."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, palette=['#4C72B0', '#DD8452'], common_norm=False)
    plt.title('Monthly Charges Distribution by Churn Status', fontsize=16, fontweight='bold')
    plt.xlabel('Monthly Charges ($)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_monthly_charges_dist.png'), dpi=300)
    plt.close()

def plot_churn_by_segment(df, output_dir):
    """Plot churn by created segments (SpendingSegment)."""
    plt.figure(figsize=(10, 6))
    # Spending Segment is 'High' or 'Low'
    ax = sns.countplot(data=df, x='SpendingSegment', hue='Churn', palette=['#4C72B0', '#DD8452'])
    plt.title('Churn by Spending Segment', fontsize=16, fontweight='bold')
    plt.xlabel('Spending Segment (Relative to Median)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_churn_by_spending_segment.png'), dpi=300)
    plt.close()

def generate_eda_visuals(filepath, output_dir):
    """Execute full exploratory data analysis and generate business-ready visuals."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df_features, _ = get_preprocessed_pipeline(filepath)
    
    print("Generating EDA Visualizations...")
    plot_churn_distribution(df_features, output_dir)
    plot_churn_by_contract(df_features, output_dir)
    plot_churn_by_tenure(df_features, output_dir)
    plot_monthly_charges_distribution(df_features, output_dir)
    plot_churn_by_segment(df_features, output_dir)
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    import sys
    # For testing the script directly
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Optional override from cmd line
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = os.path.join(base_dir, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    output_dir = os.path.join(base_dir, 'visuals')
    generate_eda_visuals(filepath, output_dir)
