---
authors:
  - edesz
date: 2025-04-24
---

(implementation-phases)=
# Phases of Implementation

Machine Learning (ML) was used to identify customers at risk of canceling their credit card services. We used these predictions to

1. create a profile of such customers, with targeting recommendations
2. estimate business metrics from successfully getting these customers to revert their decision to churn with an assumed 40% success rate

More details, constraints and high-level assumptions are discussed later during [project scoping](../scope/02_problem_understanding.md).

We divided this task into five phases, or sections.

(scoping)=
## Scoping

We started by scoping the project to quantify the magnitude of the problem. We did this with two Key Performance Indicators (KPIs) used in the finance industry to measure customer churn. After determining that the problem facing the client was important and actionable using the provided data, we proceeded to perform the analysis required to address the client's [business problem](./00_overview.md#business-problem).

To ensure all team members were working with a single source of truth, the raw data `.xlsx` file was stored in a private [Cloudflare R2 bucket](cloudflare). All processed datasets and analysis artifacts were stored in the same bucket. [Access keys](r2-access-keys) were generated in order to allow team members programmatic access to the R2 bucket.

(analysis)=
## Analysis

During the analysis phase, we prepared the raw data for use in a ML development workflow. We first split the raw data into training, validation and test splits. We then performed exploratory data analysis (EDA) on the combined training and validation data split. During EDA, we generated charts to explore the behavioural and demographic attributes of the bank's customers in the raw data sample. We concluded this phase by developing a hypothesis about the predictors of credit customer churn at the bank.

(ml-development)=
## ML Development

We performed two streams of ML modeling.

(stream-1)=
### Stream 1

In the first stream, we performed cost-sensitive ML experiments using subsets of the existing feature set across the same combination of training and validation data used in EDA. During this process, we identified the best combination of ML model, ML features and feature pre-processing. We were able to validate the hypothesis developed during EDA. Next, we estimated drift and overfitting during model evaluation on unseen data. i.e. the test data split, which was not seen during model validation.

The best model was then trained on all available data and, per the business use-case, used to make inference predictions of *all* available customers data in the random sample we were provided. These predictions were used to identify the at-risk customers.

From this cohort, we estimated ROI and (net) savings for two scenarios. In scenario 1, we assumed the client's budget allowed for targeting at most 400 at-risk customers. For scenario 2, we assumed that sufficient budget was available to target all identified at-risk customers.

(stream-2)=
### Stream 2

In the second stream, we developed an end-to-end ML pipeline. We demonstrated the performance of this model on the unseen (test split) data. All steps from data retrieval to ML inference were orchestrated in this pipeline. Unlike the first stream, additional features were engineered from the raw customer data. This and this gave an improvement in the model evaluation scores on unseen data compared to those found in the first stream.

Using the unseen data, at-risk customers were identified and model interpretability was performed. Finally, under the same conditions as scenario 2 from above, the ROI was estimated for the unseen (test split) data.

(data-validation)=
## Data Validation

In order to use the ML workflow developed here with new customer data, new (unseen) customers should have the same characteristics as those in the sample we used in this project. So, during the data validation phase, we developed a data model using the [Pandera](pandera) Python package to validate the customer data using two complimentary types of data validation tests.

### Example Based Tests

First, we performed [example-based tests](ex-tests). A data schema was defined with with specific rules (e.g. column datatypes, constraints, etc.) and used to validate the data. These tests were focused on whether a specific input (customer data) has characteristics we expect based on the sample and datatype of data we used during ML development in the analysis phase. These wree rigid characteristics, often based on the bounds of data that were observed in the available customer data.

### Hypothesis Tests

We also used [hypothesis testing](hypoth-tests) against a single column or combinations of columns. Statistical checks were defined directly in the data validation schema. These were not rigid checks of the data. Instead, their purpose was to verify that the data follows certain statistical distributions or relationships rather than just checking individual values.

## Unit Testing

During the analysis phase, we developed several custom Python modules to perform the analysis. So, in the unit testing phase, we used the Python package [Pytest](pytest) to write unit tests for these custom modules. The [Coverage](coverage-py) package was used to measure code coverage in tests. Tests are stored in the `tests` folder.

```{note} Tested Modules
:class: dropdown
The following custom modules in `src` were tested

1. `cc_churn`
   - 10 of 11 modules
2. `r2`
   - 1 of 1 module
3. `utils`
   - 1 of 3 modules
4. `business`
   - 0 of 3 modules
5. `evaluation`
   - 0 of 2 modules
```

For brevity, unit tests are not discussed in this project documentation. They are available in the [project's `tests` folder on Github](tests).

[cloudflare]: https://www.cloudflare.com/en-ca/developer-platform/products/r2/
[r2-access-keys]: https://developers.cloudflare.com/r2/api/tokens/
[pandera]: https://pypi.org/project/pandera/
[ex-tests]: https://pandera.readthedocs.io/en/stable/dataframe_models.html
[hypoth-tests]: https://pandera.readthedocs.io/en/stable/hypothesis.html
[tests]: https://github.com/edesz/credit-card-churn/tree/main/tests
[pytest]: https://pypi.org/project/pytest/
[coverage-py]: https://pypi.org/project/coverage/
