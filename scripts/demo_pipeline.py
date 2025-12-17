#!/usr/bin/env python3
"""
Demo script for the Credit Card Churn Prediction Pipeline

This script demonstrates how to use the advanced ML pipeline components
for credit card churn prediction with business cost analysis.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path.cwd() / 'src'))

from models.churn_predictor import ChurnPredictor
from business.clv_calculator import CLVCalculator
from utils.data_preprocessing import DataPreprocessor
from evaluation.metrics import CustomMetrics


def create_sample_data():
    """Create sample data for demonstration purposes"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic credit card data
    data = {
        'customer_age': np.random.randint(25, 75, n_samples),
        'credit_limit': np.random.uniform(1000, 50000, n_samples),
        'total_revolv_bal': np.random.uniform(0, 20000, n_samples),
        'Total_Trans_Amt': np.random.uniform(500, 50000, n_samples),
        'total_trans_ct': np.random.randint(10, 200, n_samples),
        'months_on_book': np.random.randint(1, 60, n_samples),
        'months_inactive_12_mon': np.random.randint(0, 6, n_samples),
        'contacts_count_12_mon': np.random.randint(0, 5, n_samples),
        'num_products': np.random.randint(1, 5, n_samples),
        'Card_Category': np.random.choice(['Blue', 'Silver', 'Gold', 'Platinum'], n_samples),
        'avg_utilization_ratio': np.random.uniform(0, 1, n_samples),
        'avg_open_to_buy': np.random.uniform(0, 30000, n_samples),
        'Total_Revolving_Bal': np.random.uniform(0, 20000, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create synthetic churn target (16% churn rate)
    # Higher churn probability for high utilization, inactive customers
    churn_prob = (
        0.1 + 
        0.3 * (df['avg_utilization_ratio'] > 0.8) +
        0.2 * (df['months_inactive_12_mon'] > 2) +
        0.1 * (df['contacts_count_12_mon'] > 2) +
        np.random.normal(0, 0.1, n_samples)
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    
    df['is_churned'] = np.random.binomial(1, churn_prob, n_samples)
    
    return df


def demo_pipeline():
    """Demonstrate the complete ML pipeline"""
    print("Credit Card Churn Prediction Pipeline Demo")
    print("=" * 50)
    
    # Create sample data
    print("\n1. Creating sample data...")
    df = create_sample_data()
    print(f"   Dataset shape: {df.shape}")
    print(f"   Churn rate: {df['is_churned'].mean():.1%}")
    
    # Initialize components
    print("\n2. Initializing pipeline components...")
    preprocessor = DataPreprocessor(random_state=42)
    churn_predictor = ChurnPredictor(random_state=42)
    clv_calculator = CLVCalculator()
    metrics_calculator = CustomMetrics()
    
    # Data preprocessing
    print("\n3. Applying data preprocessing...")
    df_processed = preprocessor.fit_preprocessing_pipeline(df, target_column='is_churned')
    print(f"   Processed shape: {df_processed.shape}")
    
    # Prepare features and target
    X = df_processed.drop(columns=['is_churned'])
    y = df_processed['is_churned']
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Train models
    print("\n4. Training advanced ML models...")
    training_results = churn_predictor.fit(
        X_train, y_train,
        sampling_method='smote',
        optimize_threshold=True,
        cv_folds=3  # Reduced for demo
    )
    print(f"   Best model: {training_results['best_model']}")
    print(f"   Optimal threshold: {training_results['optimal_threshold']:.3f}")
    
    # Make predictions
    print("\n5. Making predictions...")
    y_pred, y_pred_proba = churn_predictor.predict(X_test)
    
    # Evaluate model
    print("\n6. Evaluating model performance...")
    evaluation_results = churn_predictor.evaluate_model(X_test, y_test)
    print(f"   F2 Score: {evaluation_results['f2']:.3f}")
    print(f"   Recall: {evaluation_results['recall']:.3f}")
    print(f"   Precision: {evaluation_results['precision']:.3f}")
    print(f"   AUC: {evaluation_results['auc']:.3f}")
    
    # Business cost analysis
    print("\n7. Performing business cost analysis...")
    # Use original test data for CLV calculation (before preprocessing)
    _, X_test_original, _, _ = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    clv_values = clv_calculator.calculate_clv(X_test_original)
    expected_savings = clv_calculator.calculate_expected_savings(y_pred_proba, clv_values)
    roi = clv_calculator.calculate_roi(expected_savings)
    
    print(f"   Average CLV: ${clv_values.mean():.2f}")
    print(f"   Total CLV at risk: ${clv_values[y_pred_proba >= 0.5].sum():.2f}")
    print(f"   Expected savings: ${expected_savings[y_pred_proba >= 0.5].sum():.2f}")
    print(f"   ROI: {roi:.1f}%")
    
    # Feature importance
    print("\n8. Analyzing feature importance...")
    feature_importance = churn_predictor.get_feature_importance()
    print("   Top 5 features:")
    for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
        print(f"     {i+1}. {row['feature']}: {row['importance']:.3f}")
    
    print("\nPipeline demo completed successfully!")
    print("\nKey Insights:")
    print(f"  - Model achieves {evaluation_results['f2']:.3f} F2 score")
    print(f"  - Business ROI of {roi:.1f}% with intervention strategy")
    print(f"  - Top feature: {feature_importance.iloc[0]['feature']}")
    print(f"  - Recommended threshold: {churn_predictor.best_threshold:.3f}")


if __name__ == "__main__":
    demo_pipeline()
