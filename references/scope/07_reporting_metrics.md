---
authors:
  - edesz
date: 2025-04-29
---

# Business Metrics

*In this section, we will estimate the business metrics that quantify the performance of the best churn ML model and discuss how they will be used in reporting to the client.*

First we will estimate the Net Savings the client can realize if it is possible to successfully revert a customer who has already canceled their credit card services. Net savings will be estimated using

```{math}
:label: net-savings
Predicted Savings = Gross Savings - Cost
```

Next, we will estimate the ROI as the ratio of net savings relative to intervention costs. ROI will be estimated using

```{math}
:label: roi
ROI_N = Net Savings_N / Total Intervention Cost_N
```

:::{important}
Savings is calculated per customer. However, ROI is calculated after identifying the top *N* at-risk customers.
:::

Here, we will first focus on how these two metrics can be estimated from the ML model's predictions and then on how they will be used in reporting. Both metrics will estimate ML model performance before the true outcome of targeting (client intervention) is known. However, since we have historical customer data in which the outcome is known, their errors will be estimated using the true outcome.

```{important} Campaign Outcome is Out of Scope
Calculating the metrics after running a targeting campaign is beyond the scope of this project.
```

## Net Savings

(constraints)=
### Constraints

We assume that

1. The Bank's intervention cost to prevent a customer from churning is a nominal $50 (`c = 50`) and this cost includes discounts, call center time, retention offers, etc.
2. The success rate is the rate of converting customers to be at risk of churning and assumed to be 40% (`s = 0.40`).

`s` will be used to estimate gross savings and `c` will be used when estimating net savings.

### Types of Net Savings

There are two types of savings to be calculated

1. Predicted savings
   - Here, the ML model's predictions are used to retrospectively target at-risk customers. Savings are estimated based on what would be gained by successfully targeting the customers *predicted* to be at risk of churning.
2. True savings
   - Here, savings are estimated based on what would be gained by targeting the customes that did actually churn. Here, we are using the true outcome. We can use this since we are working with historical data, so the customers who did churn are known which means the outcomes are also known.

If the predictive model developed here is expanded to *proactively* predicting at risk customers from a new group of live (not historical) customers, for whom the outcome is not known, then only the predicted savings is useful. However, if the new group of customers shares the same characteristics as the group in the historical data provided to us then the *predicted* savings is our best estimate of the return that the client can expect from targeting the at-risk customers.

For this reason, it would be of interest to the client if we provide an estimate of net savings. Since this is historical data, the true savings is not useful on its own but it can be used to estimate the error in the reported savings as the percent difference between the two types of savings.

### Estimating Predicted Savings

:::{attention}
Predicted savings answers the question *How much did the the model think it would save before outcomes are known?*
:::

As mentioned above, the predicted savings uses customers *predicted* to be at risk of canceling their credit card services at the bank using the ML model.

If the probability of churn is predicted per customer by the churn model and appended to the data in a column named `y_pred_proba` then we can use the [estimated CLV](./02_costs.md#clv-simplified) to update the [estimated net savings formula](#net-savings) to be

```{math}
:label: predicted-savings
Predicted Savings_i = y_pred_proba \times s \times CLV_i - c
```

This is the savings that can be realized over the tenure of a single customer. This calculation is repeated for all customers.

### Estimating True Savings

:::{attention}
True savings answers the question *How much did the model actually save after the true outcomes were known?*
:::

The true savings can be calculated for historical customer data using the following four scenarios that evaluate the net benefit of an intervention based on the models' predicted outcome and the actual (true) outcome

1. Predicted outcome is churn
   - If the true outcome is also churn and the client intervenes, then savings are earned when the client correctly targets an at-risk customer. However, the intervention is not guaranteed to succeed. With an [assumed success rate](#constraints) of `s`, the customer is retained. Therefore, the expected net savings becomes
     ```{math}
     :label: true-savings-churn-correct
     True Savings_i = s \times CLV_i - c
     ```
   - If the client intervenes but the true outcome is no churn, then the targeting cost is wasted because the customer would not have churned anyway. In this case, there is no potential benefit from intervention. There is nothing to *save*, so the benefit term disappears entirely and the savings simply becomes
     ```{math}
     :label: true-savings-churn-incorrect
     True Savings_i = - c
     ```
2. Predicted outcome is no churn
   - If the customer did not churn and the client did not intervene, then there are no savings to be realized. Since no intervention occurred, there is no cost incurred (c = 0) and no opportunity to generate additional savings. The success rate `s` is irrelevant here because no intervention took place, so it is never used. In this scenario, the true savings becomes
     ```{math}
     :label: true-savings-nochurn-correct
     True Savings_i = 0
     ```
   - If the customer did churn but the client did not intervene, this represents a missed opportunity to retain the customer and realize their CLV. However, we only measure realized gains and losses from *actions taken*. Since no intervention was made, no cost was incurred and no savings were realized. Again, true savings becomes
     ```{math}
     :label: true-savings-nochurn-incorrect
     True Savings_i = 0
     ```

     From a business perspective, this scenario represents a loss since the client lost a customer. However, from a model evaluation and intervention accounting perspective, it is not counted as negative savings, because the client didn't intervene (didn't spend money). There is no *incremental* financial impact from the client's decision. In other words, true savings measures the value of actions taken, not the value of missed opportunities.

Similar to the predicted savings above, this logic is repeated for all customers.

An important distinction is that in the predicted savings we use the model's predicted probabilities, which are stored in the `y_pred_proba` column, to identify the customers *predicted* to be at risk of churning. We use this because we don't have the true outcomes so we don't know who will actually churn. It represents a probabilistic expectation based on the model's predicted probability of churn.

By comparison, in the true savings, we use the true outcome (e.g. `y_pred`) since we have this information in our historical data. At this point, uncertainty about churn is resolved since we know if the customer churned (`y_pred` = 1) or did not churn (`y_pred` = 0). The probability is now 1.0 (certainty), so we use the true outcome. For this reason, we no longer need `y_pred_proba` when estimating true savings.

(net-savings-usage)=
### Use Estimated Net Savings to Evaluate Performance of Targeting Recommendations

The [predicted net savings](#predicted-savings) will be reported to the client and it has two assumed terms that are fixed and two terms which change (`y_pred_proba` and CLV) based on the ML model's predictions or customer data. So, one way of reporting this to the client is using a two-dimensional heatmap of estimated *predicted* savings per customer as the targeting risk (e.g. low, medium, high based on the predicted probability) and customer value tier (e.g. silver, gold, platinum based on the CLV) are changed. Since both probability (risk) and CLV (value tier) values are continuous variables, they would have to be binned (e.g. using quantiles) and the *predicted* savings per customer in each bin would be shown on the heatmap. Finally, for each value shown on the heatmap, we will provde a recommended targeting strategy that the client's team can use when contacting the clients in that bin.

This will function as a lookup table that allows the client to

1. evaluate the effectiveness of targeting at-risk customers in one or multiple combinations of risk and value tier, depending on available budget
2. review our team's recommendations for targeting strategy per combination of risk and value

## Return on Investment (ROI)

In the above approach, net savings is estimated per customer. ROI takes a different approach and is calculated as a ratio. The ROI is the ratio of total net savings across *N* customers to the total cost of targeting those *N* customers. For readability purposes, we multiply this by 100 to get a percentage.

(roi-usage)=
### Use Estimated ROI to Evaluate Performance of Targeting Recommendations

The following workflow can be used to estimate the ROI using the ML model's predictions

1. Sort the model's predicted probabilities `y_pred_proba`, which is the risk of customer churn, in descending order, so the highest-risk customers are at the top
2. Select the top *N* at-risk customers to be targeted
3. Compute the total intervention cost for the top *N* customers using the [assumed intervention cost per customer](#constraints)
   ```{math}
   :label: total-intervention-cost
   \text{Total Intervention Cost}_N = N \times \text{Intervention Cost per Customer}
   ```
4. Estimate the net savings from targeting the top-*N* at-risk customers by taking the cumulative sum of the estimated savings
   ```{math}
   :label: net-benefit
   {Net Savings}_N = \sum_{n=1}^{N} Estimated Savings
   ```
5. Calculate the ROI using
   ```{math}
   :label: roi
   {ROI}_N = \frac{Net Savings_N}{Total Intervention Cost_N}
   ```
6. (optional) The ROI can be reported as a percentage by multiplying it with 100
   ```{math}
   :label: roi-pct
   {ROI_percent}_N = {ROI}_N X 100
   ```

This workflow will be performed using the true savings to get the true ROI and separately using the predicted savings to get the predicted ROI realized after targeting the top *N* customers.

If the new group of customers is similar in characteristics and size to the group in the data used to develop the ML churn model then the predicted ROI represents our best estimate of return the client can expect from targeting the top *N* at-risk customers that the predictive ML model has identified. Similar to net savings, the true ROI will be used to calculate the error in the predicted ROI. The predicted ROI and its error will then be reported to the client.

One way to report this is through a simple line chart showing the estimated ROI versus the number of at-risk customers to be targeted. The client can decide on the number of customers (*N*) to target and read the predicted ROI off the chart.

The client can use this ROI value, or range of ROI values depending on the error bounds, to evaluate the effectiveness of targeting the top *N* recommended at-risk customers identified using the ML churn model.

(reporting-budget-scenarios)=
## Budget Constraints

If the client can only intervene on a limited number *N* of predicted at-risk customers due to budget constraints then we can optimize the analysis to identify the top *N* at-risk customers and estimate the net savings or ROI from targeting this group.

This approach can be used if

1. *N* is less than the total number of at-risk customers
   - in this scenario, the client would realize the estimated metric (ROI or savings) from targeting the top *N* customers while keeping costs within the budget
2. *N* is equal to the total number of at-risk customers
   - in this scenario, there is no budget limit, so the client would realize the estimated ROI if all customers predicted to be at risk of churning are targeted

Both metrics will be estimated for each budget scenario when reporting to the client.

## Customer Attributes

(limitations-of-profiling-using-raw-customer-attributes)=
An alternate approach to recommending a targeting strategy is to directly use customers's behavioural and demographic data only (i.e. the ML features) to profile customers predicted to be at risk of churning and those who are not predicted to do so and make recommendations.

An alternate approach to recommending a targeting strategy is to directly use customers's behavioural and demographic characteristics (i.e. the original ML features) in order to profile customers predicted to be at risk of churning and identify common patterns associated with attrition. This type of profiling is valuable for understanding customer behaviour and generating qualitative business insights. For example, it can reveal that customers with declining transaction counts, lower utilization ratios, or shorter account tenure are more likely to churn. These insights can help the client's management and customer-success teams better understand the drivers of attrition in the credit card division at the bank.

However, feature-based customer profiling alone does not determine whether intervening on a customer is economically justified. A churn prediction model may identify a customer as high risk, but the financial value of retaining that customer may still be too low to justify the cost of intervention. In practice, the client must balance retention effectiveness against operational cost, intervention scalability, and expected financial return. This is where the CLV-based framework becomes substantially more valuable from a business analytics perspective.

The net savings and ROI framework discussed above extends beyond pure ML prediction by incorporating estimated customer lifetime value (CLV), intervention cost, retention success probability, and expected savings into the decision-making process. [CLV itself is estimated from](./02_costs.md#clv-simplified) behavioural customer attributes that capture long-term economic value to the client, such as transaction activity, revolving balance behaviour, and account engagement. By combining our ML model's predicted churn probability with customer value, the framework allows the business to prioritize customers not only by likelihood of churn, but also by expected financial impact.

We suggest this produces a more operationally realistic retention strategy. Instead of treating all predicted churners equally, the framework allocates retention resources proportionally to expected economic return. High-value customers with strong expected ROI can justify expensive, high-touch interventions such as executive outreach or premium rewards, while lower-value customers may receive automated or low-cost campaigns. Customers whose expected savings remain negative after accounting for intervention costs can be excluded entirely, even if they are predicted to churn.

For this reason, behavioural and demographic profiling based solely on ML features will still be used to better understand customer segments and churn drivers and then create recommendations. However, the only recommendations reported to the client will be based on business-oriented metrics such as expected net savings and ROI, since these metrics directly support financially actionable retention decisions.
