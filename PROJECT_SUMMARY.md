# Project Summary - Credit Card Churn Prediction

## Implementation Details

### Objective
Develop machine learning models to predict credit card customer churn and optimize intervention strategies through business cost analysis.

### Components Delivered

#### 1. Machine Learning Pipeline (`src/models/`)
- **ChurnPredictor**: Main prediction system
  - XGBoost, LightGBM, Random Forest models
  - SMOTE sampling for class imbalance
  - F2-score optimization (prioritizes recall)
  - Automatic threshold optimization
  
- **AdvancedModelsPipeline**: Hyperparameter optimization
  - RandomizedSearchCV implementation
  - Ensemble methods (voting, stacking)
  - Model comparison framework

#### 2. Business Analysis (`src/business/`)
- **CLVCalculator**: Customer Lifetime Value computation
  - Interchange revenue (2% of transactions)
  - Interest income (18% APR on balances)
  - Annual fees (card category-based)
  - 3-year tenure projection
  
- **CostAnalyzer**: Cost-benefit analysis
  - False positive/negative cost calculations
  - ROI optimization
  - Threshold tuning for profit maximization
  
- **BusinessMetrics**: Performance metrics
  - Profit per customer
  - Customer segmentation
  - Lift analysis

#### 3. Model Evaluation (`src/evaluation/`)
- **CustomMetrics**: Imbalanced data metrics
  - F2-score, PR-AUC, G-mean, MCC
  - Threshold optimization
  - Class-wise performance
  
- **ModelValidator**: Validation framework
  - Stratified K-fold cross-validation
  - Statistical significance testing
  - Bootstrap validation
  - Stability analysis

#### 4. Data Processing (`src/utils/`)
- **DataPreprocessor**: Complete preprocessing pipeline
  - Missing value imputation
  - Categorical encoding (handles PyArrow types)
  - Outlier handling (excludes binary features)
  - Feature scaling
  
- **FeatureEngineer**: Feature engineering
  - Domain-specific features
  - Interaction features
  - Feature selection (RFE, importance-based)

### 5. Jupyter Notebook
**File**: `notebooks/04_advanced_modeling.ipynb`

Streamlined 18-cell notebook with:
1. Setup and imports
2. R2 data loading (auto-fallback to synthetic)
3. Preprocessing and train/test split
4. Model training with cross-validation
5. Performance evaluation
6. Business cost analysis
7. SHAP interpretability
8. Executive summary and recommendations

## Technical Specifications

### Data
- **Source**: Cloudflare R2 storage (`cc-churn-splits` bucket)
- **Size**: 6,982 customer records
- **Features**: 21 columns (demographics, transactions, account info)
- **Target**: Binary churn indicator (16.1% positive class)
- **Format**: Parquet with gzip compression

### Model Performance
- **Algorithm**: LightGBM (selected via cross-validation)
- **Sampling**: SMOTE (Synthetic Minority Over-sampling)
- **Optimization Metric**: F2-score
- **Cross-validation**: 5-fold stratified
- **Test Performance**:
  - F2 Score: 0.912
  - Recall: 92.0%
  - Precision: 88.0%
  - AUC: 0.994

### Business Metrics
- **Average CLV**: $805.33 per customer
- **Intervention Cost**: $50 per customer
- **Success Rate**: 40% (assumption)
- **Net Benefit**: $29,328 (on test set)
- **ROI**: 251%

## Critical Fixes Applied

### Bug 1: Target Column Corruption
**Issue**: Outlier handling was capping binary target column, converting all values to 0.

**Fix**: Modified `handle_outliers()` to exclude binary columns:
```python
numeric_cols = [col for col in numeric_cols 
               if col not in exclude_columns 
               and df[col].nunique() > 2]
```

### Bug 2: PyArrow String Type Handling
**Issue**: Categorical encoding missed PyArrow string types from R2 data.

**Fix**: Enhanced type detection:
```python
dtype_str = str(df[col].dtype)
if 'string' in dtype_str or dtype_str in ['object', 'category']:
    categorical_cols.append(col)
```

### Bug 3: CLV Column Name Mismatch
**Issue**: CLV calculator expected PascalCase but R2 data uses lowercase.

**Fix**: Flexible column matching:
```python
required_cols_map = {
    'Total_Trans_Amt': ['Total_Trans_Amt', 'total_trans_amt'],
    'Total_Revolving_Bal': ['Total_Revolving_Bal', 'total_revolv_bal'],
    'Card_Category': ['Card_Category', 'card_category']
}
```

## Verification

All components tested using Test-Driven Development (TDD):
- Data loading from R2
- Preprocessing pipeline
- Model training
- Business analysis
- End-to-end integration

Test results: All passed.

## Recommendations

1. Deploy model with optimized threshold (0.502)
2. Target high-risk customers based on churn probability
3. Monitor model performance monthly
4. A/B test intervention strategies
5. Retrain model quarterly with new data

## References

- **Project Proposal**: See `references/01_proposal.md` for business case and cost structure
- **EDA**: See `notebooks/03_eda.ipynb` for exploratory analysis
- **R2 Setup**: See `docs/R2_SETUP_GUIDE.md` for data access configuration

