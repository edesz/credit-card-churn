# Project Scope

## Background

The manager of the credit card division at McMaster Bank is worried by more and more customers leaving the bank's credit card services. The manager has provided a random (representative) sample of customer data and asked this team to use this data to predict customer churn for every customer in this dataset. With this prediction, the management team in the credit card division can go to the at-risk customers to provide them better services and reverse their decision to churn.

## Problem Understanding

### Client

The predicted at-risk customers will be delivered to the credit card division manager. So, the business client for this project is the manager of the credit card division.

### Impact of Churn

Before quantifying the cost of customer churn we will identify the components of cost. One common way to calculate the impact (cost) of customer churn is to callculate the customer lifetime value per churned customer and add the replacement cost.

### Customer Acquisition Cost (CaC)

The replacement cost related to customer churn refers to the Customer Acquisition Cost (CAC), which is the total expenses incurred by the client to acquire a new credit card customer to replace one that has churned. This cost includes marketing and sales expenses, such as advertising, salaries for sales and marketing teams, the cost of tools and services needed to attract and onboard a new customer, etc. For credit card providers, this is approximately [167 USD per customer](https://firstpagesage.com/seo-blog/average-customer-acquisition-cost-cac-in-banking/). Here, we will assume this is $200 CAD per customer (`cac = 200`).

### Customer Lifetime Value (CLV)

CLV per customer is the average lifetime value retained if churned customers were convinced to stay. The bank earns fees from each customer that uses its credit card services. Each of these fees contributes to CLV. Since we don't have access to profit per customer, we need to proxy revenue streams for a typical credit card division within a bank using available columns

1. Revenue from Transaction activity (interchange fees)
   - fees charged based on number of credit card transactions performed
   - these are captured using `Total_Trans_Amt` and `Total_Trans_Ct`
   - banks typically earn a fee of 1-3% of transaction volume, so we'll assume 2% (`r = 0.02`) and the interchange revenue is calculated as *Interchange Revenue* = `Total_Trans_Amt` X `r`
2. Interest income from revolving balance
   - captured by `Total_Revolving_Bal`
   - if a customer carries a balance then the bank earns interest of [approximately 15-20% in Canada](https://www.consolidatedcreditcanada.ca/credit-card-debt/what-is-apr/) (we'll assume 18%, `apr = 0.18`), so this revenue from interest is calculated as *Interest Revenue* = `Total_Revolving_Bal` X `apr`
3. Fees from Credit Card Exposure
   - this refers to a scenario in which a credit card customer receives a higher discount as their tenure (how long they've held the credit card) increases, through a multiplier on [rewards points earned per total spent](https://marionthemap.com/credit-card-points-category-multipliers/)
   - approximated by `Credit_Limit` and `Avg_Utilization_Ratio`
   - income from these annual fees can be estimated from the `Card_category` and `Credit_Limit`
   - we'll assume the following nominal fee structure
     - Blue = $0 Fees
     - Silver = $50
     - Gold = $100
     - Platinum = $200

     and this will be called *Fee Revenue*

These three sources of revenue can be combined into an estimated total annual revenue per customer and then multiplied by expected tenure in order to get CLV, as follows

*Annual Revenue* = *Interchange Revenue* + *Interest Revenue* + *Fee Revenue*

Finally, we'll asume an expected remaining tenure per customer of 3 years (`T = 3`) and a discount (loyalty) factor of 0.9 (`d = 0.9`). This means the customer's loyalty discount is 100%-90% = 10% per year. We'll use `T` and `d` to define a multiplier as follows

*multiplier* = (1 - `d`^`T`) / (1 - `d`)

With these terms, the CLV per customer becomes

*CLV* = *Annual Revenue* X `multiplier`

### Costs

In the random sample of representative customer data over the preceding 12 months at McMaster Bank, credit card churn is observed in 16% of customers. Using the assumed estimated revenue and costs defined above, the impact f credit card customer churn to the client is a loss of approximately

1. 508 dollars of CLV (CLV) per customer
2. 10% of overall CLV

When the cost to acquire a new customer is taken into account (CaC, assumed to be $200 per customer) this adds up to a loss of 708 dollars per customer, or 1,151,828 dollars overall, due to churn. If customer churn continues at the same rate, then this cost will be incurred each year.

### Current Approach

Currently, there is no way to determine which customers are most likely to churn in the future. So, the client's intervention is reactive - the credit card product management team (branch managers, financial planners, etc.) only contacts the churned customers after they leave. The problem with this approach is that it incurs costs since it does not have a 100% success rate. In an attempt to get these customers to reverse their decision, the client has to offer deep discounts such as waving credit card fees, increasing the credit card limit, offering no-limit transactions, etc. all of which incur costs instead of increasing revenue.

At-risk customers are those showing signs of likely canceling their credit card services in the future, indicating a high probability of future churn. If these customers could be identified before they churn, then the credit card division's management team can proactively implement customized data-driven strategies or interventions to retain these at-risk customers before they cancel their credit card services in the future.

## Goal

Reduce the number of customers who churn in the future by identifying and characterizing which customers are at high risk of churning in the future (at-risk customers).

## Actions Being Informed by Project Goal

Identifying these at-risk customers allows the client to know who to target. Characterizing them allows the client to know how to target them. This will allow the client to target data-driven interventions at the at-risk customers by implementing proactive strategies to retain them.

So, the action that is taken by the client based on the project goal is reaching out to customers to prevent them from churning in the future, thereby improving customer loyalty. These actions and associated intervention resources should be as efficient as possible since they are constrained by a limited budget available to the client.

## Analysis

We need to answer the key question: *Which customers should we prioritize for proactive interventions?* This means we want to identify the customers who are at risk of churning in the future.

Because we want to intervene before a customer churns, we would predict the likelihood (probability) of churn for every customer. We would then use these predictions to rank customers based on the risk (probability) of future churn and prioritize them for intervention. So, this would be a Machine Learning (ML) prediction task.

These customers would then be characterized to understand their attributes.

In this way, the **Actions** from above can be better informed using data science and would ultimately increase revenue by allowing targeted interventions rather than broad, less effective strategies.

### Current Iteration (Retrospective)

The client will be starting with a one-time iteration of this ML-assisted intervention, by contacting the at-risk customers identified from the sample data. Since the customers in the sample data have already churned, this would be a retrospective intervention.

### Future Iterations (Proactive)

Future iterations of this project would use a larger sample of customers or all customers, and it would be proactive.

## Validation

We assume the client can only intervene on a limited number of customers due to budgetary constraints mentioned above. With this in mind, we can optimize the analysis to identify the `N` at-risk customers that maximize the Return on Investment (ROI).

To do this, we assume that

1. McMaster Bank's intervention cost to prevent a customer from churning is a nominal $50 (`c = 50`) and this cost includes discounts, call center time, retention offers, etc.
2. the success rate (rate of converting churners) is 40% (`s = 0.40`)

There are two types of savings

1. predicted savings
   - before outcome is known
   - ML predictions are used to proactively go after at-risk customers
   - savings are calculated based on what would happen in the future (customer churns or does not churn in the future)
2. true savings
   - after outcome is known
   - true outcomes are known and are used
   - savings are calculated based on what has happened in the past (customer churned or did not churn)

The calculation of each type of savings is different.

### Predicted Savings (ex-ante, before outcome is known)

The predicted savings are useful after the ML model has predicted which customers will churn but the true outcomes (churn or no churn) are not known.

If the probability of churn in the future is predicted and stored in a column `y_pred_proba` then we can use the CLV defined above to estimate the expected savings per customer using

*Predicted Savings* = `y_pred_proba` X `s` X `CLV` - `c`

When we rank customers in descending order or `y_pred_proba`, this savings estimate will weight the high-value customers more heavily.

### True Savings (ex-post, after outcome is known)

The true savings are useful after the ML model has predicted customer churn or no churn, and the true outcomes are also known.

After the customer has churned or not churned following client intervention, the true outcome (`y_true`) and predicted outcome (`y_pred`) are known. So, prediction probability is not required here. The savings can be calculated as follows

1. Predicted outcome is churn
   - client intervenes
     - true outcome = churn
       - client targets a customer who would indeed have churned (savings earned!)
       - *True Savings* = `s` X `CLV` - `c`
     - true outcome = no churn
       - client targets a customer who would not have churned (wasted cost!!!)
       - *True Savings* = - `c`
2. Predicted outcome is no churn
   - client does not intervene
     - true outcome = no churn
       - client does not target customer and they did not churn
       - *True Savings* = 0
     - true outcome = churn
       - client does not target customer who churned
       - *True Savings* = 0 (missed opportunity, but no cost incurred)

### Calculation of ROI

We can calculate ROI using

*ROI* = *Savings* / *c*

where `c` is the intervention cost (above, we assumed `c = 50`).

ROI can be calculated using

1. *Predicted Savings*
   - this gives predicted ROI
2. *True savings*
   - this gives true ROI

## Choice of Machine Learning Scoring Metrics

### Business Constraints

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

### Candidate Metrics

There are two possible choices for primary scoring metric that the development team will have to choose from

1. Recall (catching churners) is more important than Precision (avoiding unnecessary offers). The **F2-score** explicitly encodes this by valuing Recall more. It is a natural metric for churn problems where Recall matters more than Precision.
2. **PR AUC** is the area under the precision-recall curve ([1](https://arize.com/blog/what-is-pr-auc/), [2](https://www.deepchecks.com/glossary/pr-auc/)). It focuses on the model's ability to identify the positive class without erroneously categorizing negative instances as positive.

### Constraints During Imbalanced ML Model Training

When ML models are trained on imbalanced data, they are biased towards the majority class (no churn). Unfortunately, this leads to large errors for the minority class (churn) and, as discussed above, this is of more interest here. The default classification decision threshold is set to 0.5 and is not optimal for imbalanced data such as churn. So, [this decision threshold needs to be tuned for imbalanced classification problems](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00160).

### Final Choice of Metrics

In order to satisfy the two requirements above (decision threshold tuning and catching true churners), a separate ML scoring metric should be used during different stages of ML model development

1. Phase 1: Model Validation (Model Selection)
   - **For model selection and (optional) hyperparameter tuning, PR AUC should be used as the primary metric** since it is insensitive to the decision threshold and it evaluates model performance across all decision thresholds.
2. Phase 2: Model Validation (Optimization of Decision Threshold for the best model from Phase 1)
   - **During optimization of the decision threshold for a single (best) ML model, F2-score should be used as the primary metric and Recall should be monitored as the secondary metric.** Thresholded metrics such as F2-score or Recall are impacted by the decision cut-off threshold. Therefore, looking at these metrics for a single decision threshold can be misleading. During this phase, Recall and PR AUC should also be monitored, but as secondary metrics only.
3. Phase 3: Model Evaluation
   - **During evaluation of a single (best) ML model, the F2-score should be used as the primary metric and Recall should be monitored as the secondary metric.** For the current churn use-case, the most important type of model error that should be punished is false negatives. F2-score captures this requirement and it is easy to explain to non-technical stakeholders: *This score balances catching churners (Recall) versus avoiding false positives (Precision), but it weights catching churners more heavily*.

## Schedule

Below is the proposed schedule to complete the analysis required for this project

October 16 (Thursday): Presentation is due
October 13 (Monday) - October 15 (Wednesday): All team members work on Presentation slides for their assigned tasks
October 7 (Tuesday) - October 12 (Sunday): Complete assigned tasks and push to Github repo using Pull Requests
- I, S.: EDA, Starter ML development
- I. Y.: ML Model development using metrics discussed above
- E. D.: Calculate Business costs for best ML model

## Project Deliverables

The following contents will be provided to the client.

### Analysis

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

1. (optional) Dashboard to visualize at-risk customers predicted by model, with business metrics
