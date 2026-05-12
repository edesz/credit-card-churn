---
authors:
  - edesz
date: 2025-05-09
---

# Analysis

*In this section, we will specify the analysis that should be performed in this project in order to inform the actions that will achieve the defined goal.*

## Analysis

### What is the Type of Analysis - Description, Detection, Prediction, or Behavior Change?

We will be using Machine Learning (ML) to predict whether customers with the provided characteristics (features) in the data during the preceding 12-16 months canceled their credit card services at the bank in the future.

So, this project will involve predictive analysis.

### What is the Purpose of Analyis?

This analysis will be used to identify cutomers who are at risk of canceling their credit card services at the bank in the future.

### What is the Action Informed by this Analysis?

We need to answer the key question: *Which customers should we prioritize for proactive interventions?* This means we want to identify the customers who are at risk of churning in the future.

Because we want to intervene before a customer churns, we would predict the likelihood (probability, or risk) of churn for every customer. We would use these predictions to rank customers based on the risk (probability) of future churn and prioritize them for intervention. So, this would be a Machine Learning (ML) prediction task.

These customers would then be characterized to understand their attributes.

In this way, the **Actions** from above can be better informed using data science and would ultimately increase revenue for the client by allowing targeted interventions rather than broad, less effective strategies.

```{important} Constraints from Working with Historical Data
We are working with historical customer data for which the true future outcome is known. So, the client will use this analysis to retrospectively evaluate the effectiveness of targeting the at-risk customers in an attempt to get them to revert their decision to cancel their credit card services at the bank.

If this level of performance is deemed acceptable then future iterations will involve proactive targeting of customers for whom the future outcome is not known.
```

### How can we Validate this Analysis? Can we Validate using Existing, Historical Data?

(data-prep-fwd-looking-target)=
We assume the data has been prepared to support predicting *future* credit card churn

1. Compute the customer attributes using a trailing ~12-16 month time window for each customer
2. Determine the customer outcome (target or label) using a **certain period in the future** subsequent to the time window during which customer attributes were extracted. If the cutomer churned during this future time window, then the `Attrition_Flag` (`"Attrited_Customer"`) is set to 1, else it is set to 0 (`"Existing_Customer"`).

We will need to choose a validation approach that reflects the real scenario in which the developed ML model would be deployed. For a predictive analysis, model development would involve two phases with historical customer data in which the customer outcome is known: validation and evaluation. This would be followed by ML inference in which the best ML model is deployed and used to make predictions of the outcome for live customers for which the outcome is not known.

(model-validation)=
Below is the real scenario for the *validation phase*

1. use customer data that was collected during trailing 12-16 month window
2. split this data into training data and test data, using an 80:20 random split
   - the training data split would be used to run ML experiments to find the best ML model, etc.
   - the test data would be used to evaluate the performance of the best ML model and other inputs that were varied during each experiment run
3. experiments would be run by dividing the training data (the 80% split) into five *folds*
   - each fold has its own training and validation splits
   - train the ML model on the training split of each fold and predict and score the validation split of that same fold
   - get the average score across all five folds

Using this workflow, we will aim to achieve the following

1. determine the best data pre-procecessing techniques in order to prepare the data to be passed to a ML model
2. select the best (most predictive) features
3. get the best ML model

The best experiment is the one which gives the highest average score across all five folds for this combination of three inputs.

Next, we would proceed to the *evaluation phase*. Here, the performance of the best combination of data pre-processing, features and ML model is evaluated on the test data - the 20% split that has not been used so far. We can do this since the data, even though it is historical, has changed between these two phases. The customers in the validation split of each of the five *folds* from the validation phase are different from the customers in the 20% test split.

(model-evaluation)=
We will use the test data split to evaluate the following

1. Check if the model outperforms a random model using interpretable metrics like Uplift and Gain, which will allow us to check if the model can outperform a random model
2. Check for concept drift or generalization by comparing the change in ML evaluation metrics on the unseen data during validation versus during evaluation.
3. Check for bias or variance which indicate underfitting (high bias) or overfitting (high variance)
4. Check for prediction drift using statistical hypothesis tests of the predicted probabilities of the test data
5. Check for data drift using statistical hypothesis tests of the features in the test data

If all these test pass then the model's predictions are reliable and we can use the model's predictions during the inference phase as a basis for making recommendations to the client for how to best target the at-risk customers.

(model-inference)=
Finally, since we are working with historical data, the *inference* phase will involve predicting the probability of customer churn and outcome for all available customers (~10,100 customers). These will be used in the following two ways

1. We will estimate business metrics - net savings and return on investment (ROI) - from targeting the top *N* customers. Since this is historical data, we will use the true outcomes to estimate the error in the predicted business metrics from targeting these top *N* customers. The details of estimating and using the reporting business metrics are discussed in the [next step](./07_reporting_metrics.md).
2. We will use the predicted probabilities to make recommendations for the cohort (at-risk customers) to target and trageting strategies.

(model-interpretability)=
We will use SHAP (Shapely Additive Explanations) to identify the primary ML features that most heavily influence whether a customer is likely to stay or churn. [SHAP is a game theory approach used to explain the output of ML models](https://arxiv.org/abs/1705.07874). It assigns an importance value (i.e. a SHAP value) to each feature for a specific prediction. It breaks down predictions into the sum of feature contributions, providing both local, instance-level explanations and global, model-level insights. Here, we will focus on global insights at the model level. Our targeting recommendations will not be on a per-customer basis. So, by viewing the aggregated impact of features, we can ensure the ML model prioritizes variables that make sense from a banking perspective. This prevents the model from relying on correlated patterns that might exist per customer but that are not valid across the general customer base.

### Validation Methodology and Metrics

#### Business Constraints

In a customer churn use-case, the following are important considerations

1. False Negative (missed churner)
   - a false negative occurs when the model fails to identify a customer who actually churns and who we should have reached out to with targeted offers. For the client, losing a customer who actually in the future churned means losing future revenue. This is the most important type of error for the current use-case. This makes it essential to correctly identify as many potential churners as possible, even if some non-churners are flagged (false positive).
2. False Positive (wrongly flagged churner)
   - as per the above, this is less costly. We contact tha true non-churner unnecessarily, and maybe offer a discount.
3. Precision
   - out of those customers predicted to churn, how many were true churners?
   - interpretation for churn: When we flag someone as high-risk, how often are we right?
4. Recall
   - out of all true churners, how many did we correctly predict?
   - interpretation for churn: How many churners are we capturing?

#### Candidate Machine Learning Metrics

There are two possible choices for primary scoring metric to choose from

1. Recall (catching churners) is more important than Precision (avoiding unnecessary offers). The **F2-score** explicitly encodes this by valuing Recall more. It is a natural metric for churn problems where Recall matters more than Precision.
2. **PR AUC** is the area under the precision-recall curve ([1](https://arize.com/blog/what-is-pr-auc/), [2](https://www.deepchecks.com/glossary/pr-auc/)). It focuses on the model's ability to identify the positive class without erroneously categorizing negative instances as positive.

#### Constraints During Imbalanced ML Model Training

When ML models are trained on imbalanced data, they are biased towards the majority class (no churn). Unfortunately, this leads to large errors for the minority class (churn) and, as discussed above, this is of more interest here. The default classification decision threshold is set to 0.5 and is not optimal for imbalanced data such as churn. So, [this decision threshold needs to be tuned for imbalanced classification problems](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00160). For this reason, decision threshold tuning is important and will be performed on the training data. The best threshold will be used during evaluation on the test data.

(choice-of-metrics)=
#### Final Choice of Machine Learning Metrics

In order to satisfy the two requirements above (decision threshold tuning and catching true churners), a separate ML scoring metric should be used during different stages of ML model development

1. Phase 1: Model Validation (Model Selection)
   - **For model selection and (optional) hyperparameter tuning, PR AUC should be used as the primary metric** since it is insensitive to the decision threshold and it evaluates model performance across all decision thresholds.
2. Phase 2: Model Validation (Optimization of Decision Threshold for the best model from Phase 1)
   - **During optimization of the decision threshold for a single (best) ML model, F2-score should be used as the primary metric and Recall should be monitored as the secondary metric.** Thresholded metrics such as F2-score or Recall are impacted by the decision cut-off threshold. Therefore, looking at these metrics for a single decision threshold can be misleading. It is better to look at these metrics across different decision thresholds, such as during threshold optimization. During this phase, Recall and PR AUC should also be monitored, but as secondary metrics only.
3. Phase 3: Model Evaluation
   - **During evaluation of a single (best) ML model, the F2-score should be used as the primary metric and Recall should be monitored as the secondary metric.** For the current churn use-case, the most important type of model error that should be punished is false negatives. F2-score captures this requirement and it is easy to explain to non-technical stakeholders: *This score balances catching churners (Recall) versus avoiding false positives (Precision), but it weights catching churners more heavily*.

### Ethical Issues with Performing Analysis

Since

1. this data will not be joined with external data that could reveal the customer's identity or race
2. the provided data does not itself contain any identifiable attributes

there are no ethcial issues with performing this analysis.
