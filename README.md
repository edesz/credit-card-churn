# Credit Card Churn Prediction

Machine learning system for predicting credit card customer churn with business cost analysis.

## Overview

This project implements an advanced ML pipeline to identify customers at risk of churning, enabling proactive intervention strategies. The solution includes tree-based models, business cost optimization, and comprehensive evaluation frameworks for imbalanced classification.

See the project scope in `references/01_proposal.md`.

## Project Structure

```
credit-card-churn/
├── src/                          # Source code
│   ├── models/                   # ML models
│   ├── business/                 # Business cost analysis
│   ├── evaluation/               # Model evaluation
│   └── utils/                    # Data processing
├── notebooks/                    # Jupyter notebooks
│   └── 04_advanced_modeling.ipynb
├── scripts/                      # Utility scripts
├── docs/                         # Documentation
├── data/                         # Data storage
├── references/                   # Project documentation
└── requirements.txt              # Dependencies
```

## Installation

### Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### Setup
```bash
# Clone repository
git clone https://github.com/edesz/credit-card-churn.git
cd credit-card-churn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure R2 access
cp docs/env_template.txt .env
# Edit .env with your credentials
```

## Usage

### Run the ML Pipeline
```bash
jupyter lab notebooks/04_advanced_modeling.ipynb
```

The notebook includes:
- Data loading from R2 or synthetic data
- Preprocessing and feature engineering
- Model training (XGBoost, LightGBM, Random Forest)
- Business cost analysis
- Model interpretability

### Test R2 Connection
```bash
python scripts/test_r2_connection.py
```

### Run Demo
```bash
python scripts/demo_pipeline.py
```

## Key Features

### Advanced Models
- XGBoost with SMOTE sampling
- LightGBM with class weights
- Random Forest ensemble
- F2-score optimization (prioritizes recall)

### Business Analysis
- Customer Lifetime Value (CLV) calculation
- Cost-benefit analysis
- ROI optimization
- Threshold tuning for profit maximization

### Evaluation
- Custom metrics for imbalanced data
- Stratified cross-validation
- Statistical significance testing
- SHAP interpretability

## Model Performance

Tested on 6,982 real customer records:

| Metric | Value |
|--------|-------|
| F2 Score | 0.912 |
| Recall | 92.0% |
| Precision | 88.0% |
| AUC | 0.994 |

## Business Impact

| Metric | Value |
|--------|-------|
| Average CLV | $805 |
| Net Benefit | $29,328 |
| ROI | 251% |

## Documentation

- **Project Proposal**: `references/01_proposal.md`
- **R2 Setup Guide**: `docs/R2_SETUP_GUIDE.md`
- **Environment Template**: `docs/env_template.txt`

## Team

- Inderpreet: Data Engineering & EDA
- Elstan: Data Infrastructure & Validation
- Ilkham: Machine Learning & Business Analysis

## License

MIT License - see LICENSE file for details.
