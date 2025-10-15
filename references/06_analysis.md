# 6. Analysis

## Overview

This section will specify the analysis to be performed in this project in order to inform the actions that will achieve the defined goal.

## Analysis

### Type of Analysis

Predictive

### Purpose of Analyis

Identify at-risk cutomers.

### Action Informed by this Analysis

We need to answer the key question: *Which customers should we prioritize for proactive interventions?* This means we want to identify the customers who are at risk of churning in the future.

Because we want to intervene before a customer churns, we would predict the likelihood (probability) of churn for every customer. We would use these predictions to rank customers based on the risk (probability) of future churn and prioritize them for intervention. So, this would be a Machine Learning (ML) prediction task.

These customers would then be characterized to understand their attributes.

In this way, the **Actions** from above can be better informed using data science and would ultimately increase revenue by allowing targeted interventions rather than broad, less effective strategies.

The client will be starting with a one-time iteration of this ML-assisted intervention, by contacting the at-risk customers identified from the sample data. Since the customers in the sample data have already churned, this would be a retrospective intervention. Future iterations will be proactive.

### Validation Method

We assume the data has been prepared to support predicting future churn

1. Compute the customer attributes using a trailing ~12-16 month time window for each customer
2. Determine the target (outcome or label) using a **certain period in the future** subsequent to the customer attributes extraction time window. If the cutomer churned during this target time window, then the `Attrition_Flag` (`"Attrited_Customer"`) is set to 1, else it is set to 0 (`"Existing_Customer"`).

A trained ML model would be validated by splitting the data into training data and test data, using an 80:20 random split. The choice of ML model and [classification, or decision, threshold](https://scikit-learn.org/stable/auto_examples/model_selection/plot_tuned_decision_threshold.html#post-hoc-tuning-the-cut-off-point-of-decision-function) would be determined using the training data and the best model and threshold would be evaluated using the test data.

### Validation Methodology and Metrics

#### Business Constraints

In this business case (customer churn) the following are important considerations

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

There are two possible choices for primary scoring metric that the development team will have to choose from

1. Recall (catching churners) is more important than Precision (avoiding unnecessary offers). The **F2-score** explicitly encodes this by valuing Recall more. It is a natural metric for churn problems where Recall matters more than Precision.
2. **PR AUC** is the area under the precision-recall curve ([1](https://arize.com/blog/what-is-pr-auc/), [2](https://www.deepchecks.com/glossary/pr-auc/)). It focuses on the model's ability to identify the positive class without erroneously categorizing negative instances as positive.

#### Constraints During Imbalanced ML Model Training

When ML models are trained on imbalanced data, they are biased towards the majority class (no churn). Unfortunately, this leads to large errors for the minority class (churn) and, as discussed above, this is of more interest here. The default classification decision threshold is set to 0.5 and is not optimal for imbalanced data such as churn. So, [this decision threshold needs to be tuned for imbalanced classification problems](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00160). For this reason, decision threshold tuning is important and will be performed on the training data. The best threshold will be used during evaluation on the test data.

#### Final Choice of Machine Learning Metrics

In order to satisfy the two requirements above (decision threshold tuning and catching true churners), a separate ML scoring metric should be used during different stages of ML model development

1. Phase 1: Model Validation (Model Selection)
   - **For model selection and (optional) hyperparameter tuning, PR AUC should be used as the primary metric** since it is insensitive to the decision threshold and it evaluates model performance across all decision thresholds.
2. Phase 2: Model Validation (Optimization of Decision Threshold for the best model from Phase 1)
   - **During optimization of the decision threshold for a single (best) ML model, F2-score should be used as the primary metric and Recall should be monitored as the secondary metric.** Thresholded metrics such as F2-score or Recall are impacted by the decision cut-off threshold. Therefore, looking at these metrics for a single decision threshold can be misleading. During this phase, Recall and PR AUC should also be monitored, but as secondary metrics only.
3. Phase 3: Model Evaluation
   - **During evaluation of a single (best) ML model, the F2-score should be used as the primary metric and Recall should be monitored as the secondary metric.** For the current churn use-case, the most important type of model error that should be punished is false negatives. F2-score captures this requirement and it is easy to explain to non-technical stakeholders: *This score balances catching churners (Recall) versus avoiding false positives (Precision), but it weights catching churners more heavily*.

### Ethical Issues with Performing Analysis

Since

1. this data will not be joined with external data that could reveal the customer's identity or race
2. the provided data does not itself contain any identifiable attributes

there are no ethcial issues with performing this analysis.
