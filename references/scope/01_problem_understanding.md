---
authors:
  - edesz
date: 2025-10-04
---

# Problem Understanding

*This section discusses the business problem, identifies the client and assess the impact of the problem.*

(business-problem-description)=
## Business Problem

A full-service bank provides a comprehensive suite of financial products to its customers, including credit cards. The manager has provided our team, the data science team, with a random but representative sample of customer attributes for approximately 10,000 customers over the preceding ~12-16 months at the Bank. The manager of the bank's credit card division is worried by more and more customers leaving the bank's credit card services. In this sample, credit card churn (churn rate) is observed in 16% of customers.

The client asked for our team to predict churn from this sample of customers. The client would then target the at risk customers with retention offers in an attempt to get them to revert their decision to cancel their credit card services at the bank.

## Client

The business client for this project is the manager of the credit card division.

(impact-of-churn)=
## Impact of Churn

### Churn Rate

As per [Canadian banking laws](https://www.globallegalinsights.com/practice-areas/banking-and-finance-laws-and-regulations/canada), churn rate is an internal business metric used to measure customer retention. It is not reported for mandatory public or compliance-related reasons, as this data is proprietary. So, Canadian banks do not release specific credit card churn rates, making a precise number unavailable. We will compare this churn rate to that at banks in the US.

In the U.S., the overall churn rate at financial institutions is ~19% ([1](https://customergauge.com/blog/average-churn-rate-by-industry), [2](https://thefinancialbrand.com/news/bank-onboarding/the-churn-challenge-four-big-ideas-for-banks-and-credit-unions-looking-to-drive-down-attrition-182528)). At U.S. banks, a [credit card churn rate of 25% is at the start of the high range (20%-30%)](https://uxpressia.com/blog/how-to-approach-customer-churn-measurement-in-banking).

(lost-clv)=
### Lost CLV

On its own, the churn rate of 16% at client's bank is not a concern. However, since the client has observed churn to be increasing, this could be a concern in the near future. Using the provided data, we estimated the impact of churn at the client's bank, as a cost in dollars. The two terms involved in this approximation are discussed in [the next step](./02_costs.md). Before starting any data analysis, these were subsequently used to [make this approximation of the impact of churn](../../notebooks/02_scoping.ipynb).

(cac)=
We estimated the impact of credit card customer churn to the client to be approximately 508 dollars of Customer Lifetime Value (CLV). If the cost to acquire a new customer (CaC), which we assumed to be $200 per customer ([1](https://www.reviewtrackers.com/blog/bank-customer-retention), [2](https://firstpagesage.com/seo-blog/average-customer-acquisition-cost-cac-in-banking/)), is taken into account then this adds up to a [loss of 708 dollars per customer, or a total of 1,151,828 dollars](../../notebooks/02_scoping.ipynb#lost-clv-calc) due to churn. If customer churn continues at the same rate, then this cost will be incurred each year. If churn is growing, as the client has observed, then this annual cost will also grow.

(churn-metrics)=
### Metrics to Quantify Impact

There are two metrics for which the bank is in danger of exceeding industry standards

1. The current churn rate at the Bank is ~16%, which is just below the threshold for the financial sector of ~19%.
2. In the U.S., the average CLV per credit card customer across all financial services is approximately 808 dollars (for existing customers). Using the approach to estimating CLV discussed above, we found that the [CLV of remaining (non-churned) customers is 877 dollars at the client's bank](../../notebooks/02_scoping.ipynb#remaining-clv-calc), which is just above this threshold.

The client's observation that more and more customers are churning suggests that churn rate is growing which would cause it to exceed industry standards in the near future. When this happens both metrics are above the rest of the industry and the bank loses revenue and incurs increased CaC relative to its competitors. So, the impact of churn at the bank is that it is currently relatively close to falling behind its peers in terms of these two metrics.

## Benefits of Solving this Problem

If this problem of increasing churn can be solved, then it has two main benefits

1. it would stop or reduce the observed trend of increasing rate of credit card customer churn
2. if actioned proactively, then it would improve customer satisfaction for those customers at risk of canceling their services at the bank

## Reason for Prioritizing Problem Now

If this problem is made a priority now then the client can prevent the current 16% customer churn from (a) exceeding the industry threshold of 19% and (b) entering the high range, which starts at 25%, both of which would place the bank at a competitive disadvantage against its peers if they occurred.

## Implemented Approaches and Outcome

Currently, there is no way to determine which customers are most likely to churn in the future. So, the client's interventions to-date have been reactive. The credit card product management team (branch manager, financial planners, etc.) only contacts customers after they cancel their credit card services in an attempt to get them to revert their decision.

The problem with this reactive approach is that it incurs costs since it has a low success rate. In an attempt to get these customers to reverse their decision, the client has to offer deep discounts such as

1. waving credit card fees on *future* credit card transactions
2. increasing the credit card limit *going forward* for all customers (including for unqualified customers, who are those with a poor credit history)
3. etc.

all of which incur costs instead of increasing revenue.

## Other Groups Involved

If this project is determined to be a success, deployment will require the involvement of

1. data science team
2. IT department
3. targeting group
   - manager of credit card division
   - call center
   - financial planners
   - bank branch manager
   - marketing team
   - etc.
