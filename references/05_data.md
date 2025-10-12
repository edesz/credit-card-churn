# 5. Data Sources

## Overview

This section covers

1. Available internal and external data sources
2. Additional information about available data that would be relevant to pursuing goal

## Data

### Internal or External Datasets

Credit customer churn outcomes.

### Attriutes in the Data Source

The dataset contains customer attributes from the past ~12-16 months and their current status (churned or did not churn).

The following types of attributes are present

1. Identifier
   - `CLIENTNUM`
2. Demographic
   - `Age`
   - `Gender`
   - `Education_Level`
   - `Dependent_count`
   - `Total_Relationship_Count`
   - `Marital_Status`
   - `Income_Category`
2. Customer Cccount
   - `Card_Category`
   - `Credit_Limit`
   - `Total_Revolving_Bal`
   - `Avg_Open_To_Buy`
   - `CreditScore`
   - `Contacts_Count_12_mon`
   - `Months_Inactive_12_mon`
3. Transactional attributes (customer credit card behaviour) over the last ~12-16 months
   - `Total_Trans_Amt`
   - `Total_Trans_Ct`
   - `Total_Amt_Chng_Q4_Q1`
   - `Total_Ct_Chng_Q4_Q1`
   - `Avg_Utilization_Ratio`
4. Outcomes (indicator of churn)
   - `Attrition_Flag`

### Level of Granularity of the Data

The data is at the customer level.

### Period Covered by the Data Source

The transactional attributes in the data source covers the last 12-16 months.

Across the industry, customer credit card behavior is seasonal, with the following seasonal patterns ([1](https://www.rbc.com/en/thought-leadership/economics/featured-insights/rbc-consumer-spending-tracker/), [2](https://www.consumerfinance.gov/about-us/blog/aggregate-credit-card-borrowing-exhibits-end-year-seasonal-patterns-which-vary-across-different-sets-consumers/), [3](https://libertystreeteconomics.newyorkfed.org/2021/11/credit-card-trends-begin-to-normalize-after-pandemic-paydown/), [4](https://www.experian.com/blogs/insights/holiday-consumer-credit-card-spending/#:~:text=Tis%27%20the%20season%20for%20hefty%20consumer%20credit%20card%20spending%20%2D%20Experian%20Insights.))

1. fourth quarter's holiday season
2. decrease in the first quarter as consumers pay down debt
3. back-to-school surge in late summer
4. general increase in spending during warmer months

These patterns are relevant in Canada, where the client's institution (McMaster Bank) is located.

### Frequency of Subsequent Data Collection or Updates

None are planned.

### Presence of Unique Identifiers in Data to link to Other Data Sources

There is a reliable and unique identifier (`CLIENTNUM`) that can be used to link this data to other data sources.

### Internal Data Owner

Manager of credit card division.

### Storage Medium

The data is stored in a single `.xlsx` file.

### Ethical Issues Associated with Using this Data Source

1. Consent required
   - no
2. security protocols
   - data should not be made public
   - data should be deleted after project completion
3. bias from data collection
   - none that are provided to the data team

### Additional Useful Information Associated with this Data Source

This dataset is a random but *representative* sample of all the credit card customers at McMaster Bank.

### Additional Helpful Data that would be Relevant to this Problem

1. Due to the seasonality of credit card customer transactions, it would be good to verify that this data covers at least 12 months of cutomer behaviour.
