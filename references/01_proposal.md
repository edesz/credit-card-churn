# Project Scope

## Problem Understanding

## Calculation of Costs

Before quantifying the cost of customer churn to the client we will identify the components of cost. One common way to calculate the impact of customer churn is to callculate the customer lifetime value per churned and add the replacement cost. The replacement cost related to customer churn customers refers to the Customer Acquisition Cost (CAC), which is the total expenses incurred by the client to acquire a new credit card customer to replace one that has churned. This cost includes marketing and sales expenses, such as advertising, salaries for sales and marketing teams, the cost of tools and services needed to attract and onboard a new customer, etc. For Credit Card Providers, this is approximately [167 USD per customer](https://firstpagesage.com/seo-blog/average-customer-acquisition-cost-cac-in-banking/). Here, we will assume this is $200 CAD per customer (`cac = 200`).

### Customer Lifetime Value (CLV)

CLV per customer is the average lifetime value retained if churned customers were convinced to stay.

The bank earns fees from each customer that uses its credit card services. Each of these fees contributes to CLV. Since we don't have access to profit per customer, we need to proxy revenue streams for a typical credit card division within a bank using available columns

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

These three sources of revenue can be combined into an estimated annual revenue per customer and then multiply expected tenure to get CLV as follows

*Annual Revenue* = *Interchange Revenue* + *Interest Revenue* + *Fee Revenue*

Finally, we'll asume an expected remaining tenure of 3 years (`T = 3`) and a discount (loyalty) factor of 0.9 (`d = 0.9`). This means the customer's loyalty discount is 100%-90% = 10% per year. We'll use `T` and `d` to define a multiplier as follows

*multiplier* = (1 - `d`^`T`) / (1 - `d`)

With these terms, the CLV per customer becomes

*CLV* = *Annual Revenue* X `multiplier`

### Costs

In a random sample of representative customer data over the preceding 12 months at McMaster Bank, credit card churn is observed in 16% of customers. Using the assumed estimated revenue and costs defined above, the impact to the client (i.e. to the bank's credit card division manager) of credit card customer churn is a loss of approximately

1. 508 dollars of CLV (CLV) per customer
2. 10% of overall CLV

When the cost to acquire a new customer is taken into account (CaC, assumed to be $50 per customer) this adds up to a loss of 708 dollars per customer, or 1,151,828 dollars overall, due to churn.

### Current Approach

Currently, there is no way to determine which customers are most likely to churn in the future. So, the client's intervention is reactive - the credit card product management team (branch managers, financial planners, etc.) only contacts the churned customers after they leave. The problems with this approach are costs incurred and that it does not have a 100% success rate. In an attempt to get these customers to reverse their decision, the client has to offer deep discounts such as waving credit card fees, increasing the credit card limit, offering no-limit transactions, etc. all of which incur costs instead of increasing revenue.

At-risk customers are those showing signs of likely canceling their credit card services in the future, indicating a high probability of churn in the future. If these customers could be identified before they churn, then the credit card division's management team can proactively implement customized strategies or interventions to retain these at-risk customers before they cancel their credit card services in the future.

## Goal

Reduce the number of customers who churn in the future by

1. identifying which customers are at high risk of churning in the future (at-risk customers)
2. targeting interventions at the at-risk customers by implementing proactive strategies to retain them.

## Actions Being Informed

The action taken by the client is reaching out to customers to prevent them from churning (retention measures), thereby improving customer loyalty. These actions and associated intervention resources are constrained by a limited budget.

## Analysis

We need to answer the key question: *Which customers should we prioritize for proactive interventions?* This means we want to identify the customers who are at risk of churning (closing their credit card services) in the future.

Because we want to intervene before a customer churns, we would predict the likelihood of churn for every customer. We would then use these predictions to rank customers based on the risk of future churn and prioritize them for intervention. So, this would be a prediction task.

In this way, the **Actions** from above can be better informed using data science and would ultimately increase revenue by allowing targeted interventions rather than broad, less effective strategies.

## Validation

We assume the client can only intervene on a limited number of customers due to budgetary constraints mentioned above. With this in mind, we can optimize the analysis to identify the `N` at-risk customers that maximize the return on investment (ROI).

We assume that

1. McMaster Bank's intervention cost to prevent a customer from churning is a nominal $50 (`c = 50`) and this cost covers includes discounts, call center time, retention offers, etc
2. the success rate (rate of converting churners) is 40% (`s = 0.40`)

If the probability of churn in the future is predicted and stored in a column `y_pred_proba` then we can use the CLV defined above to estimate the expected savings per customer using

*Expected Savings* = `y_pred_proba` X `s` X `CLV` - `c`

When we rank customers in descending order or `y_pred_proba`, this savings estimate will weight the high-value customers more heavily.

Finally, we can calculate ROI using

*ROI* = to be done

## Choice of Machine Learning Scoring Metrics

### Business Constraints

In this business case (customer churn) the following are important considerations

1. False Negative (missed churner)
   - a false negative occurs when the model fails to identify a customer who actually churns and who we should have reached out to with targeted offers. For the client, losing a customer who actually in the future churned means losing future revenue. This makes it essential to correctly identify as many potential churners as possible, even if some non-churners are flagged (false positive).
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
2. **PR AUC** is the area under the precision-recall curve ([1](https://arize.com/blog/what-is-pr-auc/), [2](https://www.deepchecks.com/glossary/pr-auc/)). It provides more informative results for this dataset due to the class imbalance in churned customers. It focuses on the model's ability to identify the positive class without erroneously categorizing negative instances as positive. PR AUC is calculated as the area under the Precision-Recall curve.

### ML Constraints

When ML models are trained on imbalanced data are biased towards the majority class (no churn). Unfortunately, this leads to a large errors for the minority class (churn) and which is of more interest here. The default classification decision threshold is set to 0.5 and is not optimal for imbalanced data such as churn. So, [this decision threshold needs to be tuned for imbalanced classification problems](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00160).

### Final Choice of Metrics

In order to satisfy the two requirements above (decision threshold tuning and catching true churners), different metrics should be used during different stages of ML model development

1. During optimization of the decision threshold, PR AUC should be used as the primary metric since it is sensitive to the decision threshold.
2. For model selection and hyperparameter tuning, the F2-Score should be used as the primary metric due to its importance for the business use-case as dicsussed above. During this phase, Recall and PR AUC should also be monitored as secondary metrics.
