# Summary of Scoping Assumptions

## Costs

1. the Customer Acquisition Cost (CaC), or replacement cost, is [approximated to be 167 USD per customer](./02_costs.md#customer-acquisition-cost)
2. the Customer Lifetime Value (CLV) can be [estimated using annual credit card revenue and a multiplier](./02_costs.md#customer-lifetime-value)
3. there are [three main sources of annual credit-card revenue](revenue-sources)
   - revenue from [credit card transactions](./02_costs.md#interchange-rev)
   - revenue from carrying a [revolving balance](./02_costs.md#interest-rev)
   - revenue from fees due to a client's credit card exposure, which we approximated using the client's category of credit card (blue, silver, gold or platinum)
4. since clients usually have credit card services with a bank for multiple years, [a CLV multiplier](./02_costs.md#clv-multiplier) term  [was used](./02_costs.md/#clv-multiplier) to correct annual credit card revenue earned by accounting for the current value of expeced future revenue

## Actions

### Ethical Issues

1. the bank complies with industry constraints and policies when targeting at-risk customers ([link](./04_actions.md#ethics))

### Useful Info About Action to be Performed

Below are the assumptions about the [actions to be performed](./04_actions.md#useful-info)

1. targeting will be performed using the proactive outputs of a predictive model that can only be evaluated after the outcome (churn or no churn) of the action is known
2. any required approvals before acting are internal to the bank
3. The length of time before this action bears fruit is variable. Some customers will revert (from churning) immediately, while others will take time (days or weeks) before reverting.

## Data

1. the client can [*retrospectively* target all customers from this random sample](./04_actions.md#retrospective-targeting) that are identified to be at risk of churning.

## Analysis

1. the data has been prepared in such a way that the customer outcome occurs during the period that is occurs after the period during which the customer characteristics were extracted

## Reporting

When estimating net savings, we [assume](./07_reporting_metrics.md#constraints)

1. The intervention cost to try to get a customer to revert their decision to cancel their credit card services at the bank is a nominal $50 (`c = 50`). This cost covers discounts, call center time, retention offers, etc.
2. The rate of success at converting customers from being at risk of churning to not canceling their services is 40% (`s = 0.40`).
