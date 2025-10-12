# 2. Costs

In order to help estimate the quantitative impact (cost) of customer churn to a bank, we will identify the components of cost.

One common way to calculate the impact (cost) of customer churn is to calculate the customer lifetime value per churned customer and add the replacement cost. This approach will be used here.

## Replacement Cost or Customer Acquisition Cost (CaC)

The replacement cost related to customer churn refers to the Customer Acquisition Cost (CAC), which is the total expenses incurred by the client to acquire a new credit card customer to replace one that has churned. This cost includes marketing and sales expenses, such as advertising, salaries for sales and marketing teams, the cost of tools and services needed to attract and onboard a new customer, etc. For credit card providers, this is approximately [167 USD per customer](https://firstpagesage.com/seo-blog/average-customer-acquisition-cost-cac-in-banking/). Here, we will assume CaC is $200 CAD per customer (`cac = 200`).

## Customer Lifetime Value (CLV)

### Sources of Annual Revenue

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

### Loyalty Discount and Tenure

We will assume the following

1. a discount (loyalty) factor of 0.9 (`d = 0.9`). This means the customer's loyalty discount is 100%-90% = 10% per year
2. an expected remaining tenure per customer of 3 years (`T = 3`)

We'll use `T` and `d` to define a multiplier as follows

*multiplier* = (1 - `d`^`T`) / (1 - `d`)

### Calculation of CLV

With these terms, the CLV per customer becomes

*CLV* = *Annual Revenue* X `multiplier`

### Net Savings

Next, we assume that

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

The calculation of each type of savings realized by targeting the predicted at-risk churners is different.

#### Predicted Savings (ex-ante, before outcome is known)

The predicted savings are useful after the ML model has predicted which customers will churn but the true outcomes (churn or no churn) are not known.

If the probability of churn in the future is predicted and stored in a column `y_pred_proba` then we can use the CLV defined above to estimate the expected savings per customer using

*Predicted Savings* = `y_pred_proba` X `s` X `CLV` - `c`

When we rank customers in descending order or `y_pred_proba`, this savings estimate will weight the high-value customers more heavily.

#### True Savings (ex-post, after outcome is known)

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

If the client can only intervene on a limited number of customers **during some period in the future** due to budgetary constraints mentioned above. With this in mind, we can optimize the analysis to identify the `N` at-risk customers that maximize the Return on Investment (ROI).

We can calculate ROI using

*ROI* = *Savings* / *c*

where `c` is the intervention cost (above, we assumed `c = 50`).

ROI can be calculated using

1. *Predicted Savings*
   - this gives predicted ROI
2. *True savings*
   - this gives true ROI
