---
authors:
  - edesz
date: 2025-10-04
---

# Data Sources

This section covers

1. Available internal and external data sources
2. Additional information about available data that would be relevant to pursuing the project's goal

## Data

(internal-external-data)=
### What are the Available Internal or External Datasets?

The internal dataset available is the random sample of approximately 10,000 credit card customers with known churn outcomes.

No external data is avialable that can be linked to the provided internal data.

### What Attriutes are provided in the Data Source?

The dataset contains customer attributes from the past ~12-16 months and their current status (churned or did not churn).

```{note}
The outcome is forward-looking. This means the it corresponds to a period that is ahead of the period (preceding 12-16 months) during which the customers' characteristics were extracted.
```

```{important} Available Data has Known Outcomes Only and is Not Actionable
Crucially, this means the available sample of data is *historical* data, since the outcome (churn or no churn) is already known. This means the identified cohort and profiling recommendations cannot be proactively acted upon by the client since the outcome is already known.

(retrospective-targeting)=
We will **assume** that the client can *retrospectively* target all customers from this random sample that are identified to be at risk of churning.
```

The following types of attributes are present in the provided data

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

### What is the Level of Granularity of the Data?

The data is provided at the customer level.

### What is the Period Covered by the Data Source?

The transactional attributes in the data source covers the last 12-16 months.

Across the industry, customer credit card behavior is seasonal, with the following seasonal patterns ([1](https://www.rbc.com/en/thought-leadership/economics/featured-insights/rbc-consumer-spending-tracker/), [2](https://www.consumerfinance.gov/about-us/blog/aggregate-credit-card-borrowing-exhibits-end-year-seasonal-patterns-which-vary-across-different-sets-consumers/), [3](https://libertystreeteconomics.newyorkfed.org/2021/11/credit-card-trends-begin-to-normalize-after-pandemic-paydown/), [4](https://www.experian.com/blogs/insights/holiday-consumer-credit-card-spending/#:~:text=Tis%27%20the%20season%20for%20hefty%20consumer%20credit%20card%20spending%20%2D%20Experian%20Insights.))

1. fourth quarter's holiday season
2. decrease in the first quarter as consumers pay down debt
3. back-to-school surge in late summer
4. general increase in spending during warmer months

These patterns are relevant in the U.S., where the client's bank is located.

### What is the Frequency of Subsequent Data Collection or Updates?

As mentioned in the [above assumption](#retrospective-targeting), we will only be retrospectively targeting customers identified to be at risk of churning from the entire sample of customer data provided to us. For this reason, no updates to the data are planned.

### What Unique Identifiers are present in the Data to link it to Other Data Sources?

There is a reliable and unique identifier (`CLIENTNUM`) in the data that can be used to link this data to other data sources.

Note that, as mentioned [above](#internal-external-data), this dataset will not nbe linked to additional external or internal datasets. So, although this identifier is present in the data, it will not be used for this project.

### Who is the Internal Data Owner?

As mentioned in the [business problem](./02_problem_understanding.md#business-problem-description), the data is provided by and owned by the manager of credit card division at the bank.

### What is the Storage Medium for the Data?

The data is [provided as a single `.xlsx` file](./02_problem_understanding.md#business-problem-description).

### Ethical Issues Associated with Using this Data Source

1. Consent required
   - no
2. security protocols
   - data should not be made public
   - data should be deleted after project completion
3. bias from data collection
   - none that are provided to the data team

### Any Additional Useful Information Associated with this Data Source?

This dataset is a [random but *representative* sample of all the credit card customers](./02_problem_understanding.md#business-problem-description) at the bank.

### Any Additional Helpful Data that would be Relevant to this Problem?

Due to the seasonality of credit card customer transactions, it would be good to verify from the client that this data covers at least 12 months of cutomer behaviour. Even without direct datetime attributes, a full year of data for churn modeling captures these seasonal trends, customer lifecycles, and annual behavioral shifts. It allows for building a more stable model that captures high-risk behavior patterns throughout the year, such as holiday spikes or end-of-contract behavior, which improves the accuracy of and reliability of churn predictions.

In summary, benefits of training a churn model that covers at least one full year of customer behaviour data include the following

1. Smoothes out monthly anomalies (e.g. February's shorter length versus 31-day months) which makes the model more robust and reliable. This reduces variance.
2. A longer timeframe assists in better recognizing the nuances in high-value customer behaviour. This improves accuracy on high-value customers.
3. This provides a dataset for training that prevents overfitting to temporary trends. This will improve long-term prediction accuracy and ensures it does not become stale quickly when deployed.
4. In terms of deployment for proactive targeting with more customer data, the model can differentiate between long-term behavior patterns and short-term anomalies. This gives more reliable future predictions when it is deployed to production.
