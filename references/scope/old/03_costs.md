---
authors:
  - edesz
date: 2025-10-04
---

# Costs

In order to estimate the quantitative impact (cost) of customer churn, we will first identify the components of cost. A common way to make this estimate is to calculate the customer lifetime value (CLV) per churned customer and add the replacement cost (Customer Acquisition Cost, or CaC).

In the next two sections, we will estimate these two components.

## Replacement Cost or Customer Acquisition Cost (CaC)

The replacement cost related to customer churn refers to the Customer Acquisition Cost (CaC). This is the total expenses incurred by the client to acquire a new credit card customer to replace one that has churned. This cost includes marketing and sales expenses, such as advertising, salaries for sales and marketing teams, the cost of tools and services needed to attract and onboard a new customer, etc. For credit card providers, this is approximately [167 USD per customer](https://firstpagesage.com/seo-blog/average-customer-acquisition-cost-cac-in-banking/).

Here, we will assume CaC is $200 USD per customer (`cac = 200`) for the client's bank.

## Customer Lifetime Value (CLV)

The two components to estimate the CLV are total annual revenue and a tenure multiplier.

The total revenue is the sum of all sources of credit card revenue earned by the bank per year.

A CLV multiplier represents the present value of $1 of annual revenue but received over `T` years, discounted over time.

In the next two sub-sections, we'll first estimate the total revenue earned by the bank and then approximate the multiplier. These two terms will then be combined to estimate the CLV for credit card customers at the client's bank.

### Sources of Annual Revenue

CLV per customer is the average lifetime value retained if churned customers were convinced to stay. The bank earns fees from each customer that uses its credit card services. Each of these fees contributes to CLV. Since we don't have access to profit per customer, we need a proxy of revenue streams for a typical credit card division within a bank using the available columns in the client's provided churn data.

Below are the three proxies we will use for sources of annual revenue earned by the bank from providing credit card services

1. Revenue from Transaction activity (interchange fees)
   - These are fees charged based on number of credit card transactions performed. These are captured using the `Total_Trans_Amt` and `Total_Trans_Ct` columns in the provided customer churn data. Banks typically earn a fee of 1-3% of transaction volume. So, here, we'll assume this to be 2% (`r = 0.02`) for the client's bank. Usign this, the interchange revenue is then estimated as
     ```{math}
     :label: interchange-rev
     Interchange Revenue = Total_Trans_Amt X r
     ```
2. Interest income from revolving balance
   - These are the fees captured by the `Total_Revolving_Bal` column in the provided customer churn data. If a customer carries a credit card balance then the bank earns interest of approximately [15-20% in Canada](https://www.consolidatedcreditcanada.ca/credit-card-debt/what-is-apr/) and [20-30% in the US](https://www.lendingtree.com/credit-cards/study/average-credit-card-interest-rate-in-america/) on this balance. Here, we will assume this to be 18% (`apr = 0.18`). So, the revenue earned by the client's bank from credit card interest can be estimated using
     ```{math}
     :label: interest-rev
     Interest Revenue = Total_Revolving_Bal X apr
     ```
3. Fees from Credit Card Exposure
   - This refers to a scenario in which a customer receives a higher discount on credit card transactions, certain products or services purchased from retailers, etc. as their tenure with the bank increases, through a multiplier on [rewards points earned per total spent](https://marionthemap.com/credit-card-points-category-multipliers/). Below are some examples of [discounts offered to credit card customers](https://www.investopedia.com/ask/answers/110614/what-are-some-examples-common-credit-card-reward-program-benefits.asp)
     - [cash back](https://www.nerdwallet.com/ca/p/best/credit-cards/best-cash-back-credit-cards)
       - this is the percentage of money back a customer gets on purchases made using a credit card (usually 1-6%)
     - [statement credits](https://www.consumerreports.org/money/credit-cards/credit-card-benefits-you-might-not-know-about-a6135335136/)
       - discounts applied to a credit card account, such as annual credits for streaming services, dining, travel, or pre-check services
     - retail and partner discounts
       - these are exclusive discounts, special sales, or enhanced reward rates (e.g. 5X points instead of 1X) that are paid for with a credit card at specific retailers, restaurants, or via bank-specific shopping portals
     - travel perks such as frequent flyer programs
       - these are discounts on travel-related expenses paid for using a credit card, such as complimentary airport lounge access, hotel room upgrades, or discounts on rental cars
     - interest free periods (or 0% APR)
       - this is a form of discount on financing, allowing customers to pay for large purchases over time without incurring any credit card interest charges for an introductory period

     Using the data available to us, we can approximate these annual fees using the `Credit_Limit` and `Card_Category` columns. We'll assume the following nominal fee structure

     - Blue = $0 Fees
     - Silver = $50
     - Gold = $100
     - Platinum = $200

     We will refer to this source of revenue as *Fee Revenue*.

These three sources of revenue can be combined into a total estimated annual revenue per customer using

```{math}
:label: total-rev
Annual Revenue = Interchange Revenue + Interest Revenue + Fee Revenue
```

### CLV Multiplier

A CLV multiplier is a factor used to convert a customer's annual revenue earned into an estimate of their total value over time. Annual revenue alone only gives how much a customer generates in a single year, but customers typically stay with a business for multiple years. The multiplier accounts for this period of time by aggregating expected future revenue into a single present-day value. Without it, the CLV would systematically underestimate the customer's true value because it would ignore the fact that the annual revenue is earned repeatedly over multiple years the the customer has credit card services with the bank.

The multiplier is needed because of two key ideas.

The first factor is the expected tenure of the customer. If a customer is likely to remain active for several years, then their value is the sum of annual revenue across all those years, not just one.

The second factor is the time value of money. This captures the idea that a dollar earned today is worth more than a dollar earned in the future due to opportunity cost, risk, and inflation. Opportunity cost means that if the bank received money today, then the bank could reinvest these earnings back into their business operations, treasury, or as capital to generate more loans and earn returns over time. Now, if instead the bank receives that money in the future, then they have missed out on those potential earnings. As a result, money received later is effectively worth less than money received today. Risk reflects the uncertainty that future cash flows may not actually happen. A customer might cancel their credit card earlier than expected or reduce spending on the card, so future revenue is less certain. finally, inflation reduces purchasing power over time, so the same dollar in the future will buy less than it does today. These factors explain why future cash flows must be adjusted downward when estimating their value today.

For these reasons, we need to use a multiplier that effectively adds up discounted future revenues into a single factor that can be applied to the annual revenue we [calculated earlier](#total-rev).

We will make the following assumptions to estimate the multiplier

1. a loyalty discount factor per year of 0.9 (`d = 0.9`)
   - this means the customer's loyalty discount is 100%-90% = 10% per year
2. an expected remaining tenure per customer of 3 years (`T = 3`)

### Calculation of CLV

We'll now combine `T`, `d` and the estimated annual revenue to estimate the CLV per customer

```{math}
:label: clv
CLV_i = \sum_{t=1}^{T} (\text{Annual Revenue}_i \times d^t)
```

The annual revenue is constant so it can be factored out

```{math}
:label: clv-refactored
CLV_i = \text{Annual Revenue}_i \times \sum_{t=1}^{T} d^t
```

This can be simplified to

```{math}
:label: clv-simplified
CLV_i \approx \text{Annual Revenue}_i \times \frac{1 - d^T}{1 - d}
```

This means the multiplier can be extracted as

```{math}
:label: multiplier
Multiplier \approx \frac{1 - d^T}{1 - d}
```

Using this CLV, we can estimate the impact of credit card churn at the client's bank.
