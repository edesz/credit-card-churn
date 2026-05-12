---
authors:
  - edesz
date: 2025-04-24
---

(implementation-phases)=
# Phases of Implementation

Machine Learning (ML) was used to identify customers at risk of canceling their credit card services. We used these predictions to

1. estimate business metrics from successfully getting these customers to revert their decision to churn with an assumed 40% success rate
2. make targeting recommendations based on one business metric
3. understand the most important predictors that a customer would churn or be retained based on their historical behavioural attribu tes

More details, constraints and high-level assumptions are discussed later during [project scoping](../scope/02_problem_understanding.md).

We divided this task into five phases, or sections.

(scoping)=
## Scoping

We started by scoping the project to quantify the magnitude of the business problem facing the client. We did this with two Key Performance Indicators (KPIs) used in the finance industry to measure credit card customer churn. After determining that the problem facing the client was important and actionable using the provided data, we performed a predictive analysis to address the client's [business problem of identifying customes at risk of churning](./00_overview.md#business-problem).

To ensure all team members were working with a single source of truth, the raw customer data `.xlsx` file was stored in a private [Cloudflare R2 bucket](https://www.cloudflare.com/en-ca/developer-platform/products/r2/). All processed datasets and analysis artifacts were stored in the same bucket. [Access keys](https://developers.cloudflare.com/r2/api/tokens/) were generated in order to allow team members programmatic access to the data.

(analysis)=
## Analysis

During the analysis phase, we prepared the raw data for use in a ML development workflow. We first split the raw data into training, validation and test splits. We then performed exploratory data analysis (EDA) on the combined training and validation data split. During EDA, we created charts to explore the behavioural and demographic attributes of the bank's customers in the raw data sample. We concluded this phase by developing a hypothesis about the attributes that would likely be predictors of credit customer churn at the bank.

(ml-development)=
## ML Development

Next, we performed two streams of ML modeling.

(stream-1)=
### Stream 1

In the first stream, we performed cost-sensitive ML experiments using subsets of the existing feature set across the same combination of training and validation data used in EDA. During this process, we identified the best combination of ML model, ML features and feature pre-processing. We were able to validate the hypothesis developed during EDA about the choice of features that would be predictors of churn.

Next, we used a ML monitoring framework to measure concept drift and data drift during model evaluation on unseen data. i.e. the test data split, which was not seen during model validation.

The best model was then trained on all available data and, per the business use-case, used to make inference predictions of *all* available customers data in the random sample we were provided. These predictions were used to identify the at-risk customers.

From this cohort, we estimated Return on Investiment (ROI) and (net) savings for two scenarios. In scenario 1, we assumed the client's budget allowed for a total targeting of at most 20,000 dollars. For scenario 2, we assumed that a budget of approximately 80,000 dollars was available to target all identified at-risk customers.

(stream-2)=
### Stream 2

In the second stream, we developed an end-to-end ML pipeline. We demonstrated the performance of this model on the unseen (test split) data. All steps from data retrieval to ML inference were orchestrated in this pipeline. Unlike the first stream, additional features were engineered from the raw customer data. This gave an improvement in the model evaluation scores on unseen data compared to those found in the first stream.

Using the unseen data, at-risk customers were identified and model interpretability was performed. Finally, under the same conditions as scenario 2 from above, the ROI was estimated for the unseen (test split) data.

(data-validation)=
## Data Validation

In order to use the ML workflow developed here with new (unseen) customer data, the new customers should have the same characteristics as those in the sample we used in this project. So, next, during the data validation phase, we developed a data model using the [Pandera](https://pypi.org/project/pandera/) Python library to validate the customer data using two complimentary types of data validation tests.

### Example Based Tests

First, we performed [example-based tests](https://pandera.readthedocs.io/en/stable/dataframe_models.html). A data schema was defined with with specific rules (e.g. column datatypes, constraints, etc.) and used to validate the data. These tests were focused on whether a specific input (attributes of a single customer) demonstrated the

1. characteristics
2. datatype

that we expected based on the sample data we used in ML model development in the analysis phase. These were rigid characteristics, often based on the bounds of data that were observed in the available customer data.

### Hypothesis Tests

We also used [hypothesis testing](https://pandera.readthedocs.io/en/stable/hypothesis.html) against a single column or combinations of columns. Statistical checks were defined directly in the data validation schema. These were not rigid checks of the data. Instead, their purpose was to verify that the data follows certain statistical distributions or relationships rather than just checking individual values.

## Unit Testing

During the analysis phase, we developed several custom Python modules to perform the analysis. So, in the unit testing phase, we used the [Pytest](https://pypi.org/project/pytest/) Python library to write unit tests for these custom modules. The [Coverage](https://pypi.org/project/coverage/) package was used to measure code coverage in tests. Tests are stored in the `tests` folder.

```{note} Tested Modules
:class: dropdown
The following custom modules in `src` were tested

1. `cc_churn`
   - 11 of 12 modules
2. `r2`
   - 1 of 1 module
3. `utils`
   - 2 of 4 modules
4. `business`
   - 0 of 3 modules
5. `evaluation`
   - 0 of 2 modules
```

For brevity, unit tests are not discussed in this project documentation. They are available in the [`tests` folder in the project's repository](https://github.com/edesz/credit-card-churn/tree/main/tests).
