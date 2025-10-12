# 1. Problem

## Objective

This section discusses the business problem, identifies the client and assess the impact.

## Problem Understanding

### Business Problem

McMaster Bank is a Full-Service Financial Institution. It provides a range of financial services on premises to its customers, including issuane of credit cards. The manager has provided a random (but representative) sample of customer attributes for approximately 10,000 customers over the preceding ~12-16 months at McMaster Bank In this sample, credit card churn (churn rate) is observed in 16% of customers.

In addition, the manager of the credit card division at the bank is worried by more and more customers leaving the bank's credit card services.

As per [Canadian banking laws](https://www.globallegalinsights.com/practice-areas/banking-and-finance-laws-and-regulations/canada), churn rate is

1. an internal business metric used to measure customer retention
2. not reported for mandatory public or compliance-related reasons, as this data is proprietary

So, Canadian banks do not release specific credit card churn rates, making a precise number unavailable. We will compare this churn rate to that at banks in the US.

In the US, the overall churn rate at financial institutions is ~19% ([1](https://customergauge.com/blog/average-churn-rate-by-industry), [2](https://thefinancialbrand.com/news/bank-onboarding/the-churn-challenge-four-big-ideas-for-banks-and-credit-unions-looking-to-drive-down-attrition-182528)). At US banks, a [credit card churn rate of 25% is at the start of the high range (20%-30%)](https://uxpressia.com/blog/how-to-approach-customer-churn-measurement-in-banking). Considering this to be a reasonable first-order estimate of churn at banks in in Canada, a churn rate of 16% and trending up at McMaster Bank is a concern.

### Client

The predicted at-risk customers will be delivered to the credit card division manager. So, the business client for this project is the manager of the credit card division.

### Impact of Churn

Using the assumed estimated revenue and costs defined in `02_costs.md`, the impact of credit card customer churn to the client is a loss of approximately

1. 508 dollars of CLV (CLV) per customer
2. 10% of overall CLV

In the US, the average CLV per credit card customer across all financial services is ~$808 USD (~$1,100 CAD). Considering that US banks are bigger than in Canada and credit card limits are relatively higher in the US ([1](https://www.finlywealth.com/blog/credit-cards/what-credit-card-limit-should-i-have), [2](https://www.bankrate.com/credit-cards/news/what-is-the-average-credit-limit-for-americans), [3](https://www.chase.com/personal/credit-cards/education/basics/reducing-credit-limit)), this CLV is a reasonable estimate for CLV at McMaster Bank in Canada.

When the cost to acquire a new customer is taken into account (CaC, assumed to be $200 per customer) this adds up to a loss of 708 dollars per customer, or 1,151,828 dollars overall, due to churn.

If customer churn continues at the same rate, then this cost will be incurred each year. If churn grows, as has been observed, then this annual cost will also grow.

### Benefits of Solving this Problem

1. The observed trend of increasing rate of credit card customer churn can be halted or reduced
2. Reduced churn by focused targeting to improve at-risk customer satisfaction
   - if at-risk customers could be identified before they churn, then the client can proactively implement customized data-driven strategies or interventions to provide them with better services in order to prevent them from churning **during some period in the future**

### Reason for Prioritizing Problem Now

The rate of increasing customer churn is assumed to take the churn rate into the high range for the industry (per US standards).

### Implemented Approaches and Outcome

Currently, there is no way to determine which customers are most likely to churn in the future. So, the client's interventions to-date have been reactive. The credit card product management team (branch manager, financial planners, etc.) only contacts the churned customers after they leave in an attempt to get them to revert their decision.

The problem with this reactive approach is that it incurs costs since it has a low success rate. In an attempt to get these customers to reverse their decision, the client has to offer deep discounts such as

1. waving credit card fees on *future* credit card transactions
2. increasing the credit card limit *going forward* (including for for unqualified customers, who are those with a poor credit history)
3. etc.

all of which incur costs instead of increasing revenue.

### Other Groups Involved

If study is determined to be a success, deployment will require the involvement of

1. data science team
2. IT department
3. targeting
   - manager of credit card division
   - call center
   - financial planners
   - bank branch manager
