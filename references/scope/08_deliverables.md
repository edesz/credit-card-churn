---
authors:
  - edesz
date: 2025-04-29
---

# Project Deliverables

## Objective

This section lists the contents to be provided to the client.

## Deliverables

### Analysis

All source code used in this project will be [provided in a Github repository](https://github.com/edesz/credit-card-churn).

### Data Preparation

The three processed data splits will be provided in a separate file in the team's private Cloudflare R2 storage bucket

1. Train, Validation, Test data splits
   - `train_data.parquet.gzip`
   - `validation_data.parquet.gzip`
   - `test_data.parquet.gzip`

### ML Model Development

The following will be provided in a private Cloudflare R2 storage bucket

1. (Data Files with Churn Predictions) Validation and test data splits used in model development with the following columns
   - ML model predictions (in the `y_pred` and `y_pred_proba` columns)
   - model name of best model (in the `model_name` column)
   - best decision threshold (in the `best_decision_threshold` column)

   Below are the two example file names to be used for a `LogisticRegression` model
   - `validation_predictions__logisticregression__YYmmdd_HHMMSS.parquet.gzip`
   - `train_predictions__logisticregression__YYmmdd_HHMMSS.parquet.gzip`
2. (Best Trained ML Model Object) trained ML model object in `.joblib` format ([link](https://joblib.readthedocs.io/en/stable/generated/joblib.dump.html)) for model trained on
   - train+validation data
     - example: `logisticregression__train_val__YYmmdd_HHMMSS.joblib`
   - all data (train+validation+test)
     - example: `logisticregression__all__YYmmdd_HHMMSS.joblib`

### Business Requirements

The recommended at-risk customers will be provided in the team's private Cloudflare R2 storage bucket

1. using [ROI and under two budget scenarios](./07_reporting_metrics/#roi-usage)
   - `YYYY-mm-dd/at_risk_customers_with_business_metrics_roi__<model-name>__<YYYYmmdd>_<HHMMSS>.parquet.gzip`
2. using [net savings and combinations of binned risk level and customer value tiers](./07_reporting_metrics.md#net-savings-usage)
   - `YYYY-mm-dd/at_risk_customers_with_business_metrics_savings__<model-name>__<YYYYmmdd>_<HHMMSS>.parquet.gzip`
   - summary for each bin showing the following
     - attributes of bin (e.g. low risk and medium value tier, medium risk and high value tier, etc.)
     - estimated net savings per customer
     - number of customers
     - recommended targeting strategy
