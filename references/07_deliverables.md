# 7. Project Deliverables

## Objective

This section lists the contents to be provided to the client.

## Deliverables

### Analysis

The following will be provided in a Github repository

1. Python notebooks as discussed in the project `README.md` on github repository ([link](https://github.com/edesz/credit-card-churn/blob/main/README.md#analysis))

### Data Preparation

The following will be provided in a private Cloudflare R2 storage bucket

1. (Data Files) Train, Validation, Test data splits
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

The following will be provided in a private Cloudflare R2 storage bucket

1. (File with At-Risk Customers) File with at-risk customers and with identification of at-risk customers that are predicted by the model to maximize Return on Investment (ROI) if they are targeted.
