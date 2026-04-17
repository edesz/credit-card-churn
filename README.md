# Use ML to Find Credit Card Customers At Risk of Churning

## About

Use ML to identify credit card customers at a bank who are at risk of churning, for retrospective targeting.

See the project scope [here](https://github.com/edesz/credit-card-churn/blob/main/references/01_proposal.md).

## Getting Started

### Pre-Requisites

1. [Python](https://www.python.org/)
2. [Pixi](https://pixi.prefix.dev/latest/) for package management
3. [Customer churn data](https://www.kaggle.com/datasets/imanemag/bankchurnerscsv) is stored as `BankChurners.xlsx` in private R2 bucket, accessible to contributors
4. `.env` file should be placed one level above root directory, to support programmatic access to data files using the [`boto3` Python package](https://pypi.org/project/boto3/). See R2 `boto3` documentation for [example usage](https://developers.cloudflare.com/r2/examples/aws/boto3/).

### Installation and Development

1. Create virtual environment for literate programming using Jupyterlab and start server
   ```bash
   pixi run jlab
   ```

   This command runs a Jupyterlab server and launches the web user interface in the default browser. The site will be available at http://localhost:8888.

### Available Commands

Run

```bash
pixi task list
```

to see all available tasks

```bash
Tasks that can run on this machine:
-----------------------------------
jlab, pmill, test
Task   Description
jlab   Launch Jupyterlab
pmill  Perform ML Validation
test   Run unit tests with pytest
```

## Sections

See footnotes for individual team-member contributions to the different sections of this project. Footnotes link to high-level overview of work done in the [Team Member Contributions](#team-member-contributions) section below.

### Analysis

Below are the notebooks for performing data analysis, ML development and estimation of business metrics from ML inference

1. Preprocess and Split Raw Data
   - create train, validation and test splits
   - `01_split_data.ipynb`<sup>[2](#myfootnote2)</sup> ([link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/01_split_data.ipynb), [view](https://nbviewer.org/github/edesz/credit-card-churn/blob/main/notebooks/01_split_data.ipynb))
2. Perform Quantitative Analysis for Use in Project Scoping
   - estimate quantitative impact of credit card churn to quantify problem and scope out the entire project
   - `02_scoping.ipynb`<sup>[2](#myfootnote2)</sup> ([link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/02_scoping.ipynb), [view](https://nbviewer.org/github/edesz/credit-card-churn/blob/main/notebooks/02_scoping.ipynb))
3. EDA
   - perform exploratory data analysis (EDA) on processed data split for combined training and validation data
   - `03_eda.ipynb`<sup>[1](#myfootnote1)</sup> ([link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/04_eda.ipynb), [view](https://nbviewer.org/github/edesz/credit-card-churn/blob/main/notebooks/03_eda.ipynb))
   - `BankChurners-2.ipynb`<sup>[1](#myfootnote1)</sup> ([link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/BankChurners-2.ipynb), [view](https://nbviewer.org/github/edesz/credit-card-churn/blob/main/notebooks/BankChurners-2.ipynb))
4. EDA for ML
   - perform EDA for data transformation before performing Machine Learning (ML) development
   - `04_eda_for_ml.ipynb`<sup>[2](#myfootnote2)</sup> ([link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/04_eda_for_ml.ipynb), [view](https://nbviewer.org/github/edesz/credit-card-churn/blob/main/notebooks/04_eda_for_ml.ipynb))
5. ML Validation - Cost-Sensitive Learning
   - run ML experiments using linear models and tree-based models with the Metaflow framework
   - `05_run_validation_experiments.ipynb`<sup>[2](#myfootnote2)</sup> ([link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/05_run_validation_experiments.ipynb), [view](https://nbviewer.org/github/edesz/credit-card-churn/blob/main/notebooks/05_run_validation_experiments.ipynb))
6. Get best combination of ML model, features and feature pre-processing from results of validation experiments
   - `06_get_best_validation_experiment.ipynb`<sup>[2](#myfootnote2)</sup> ([link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/06_get_best_validation_experiment.ipynb), [view](https://nbviewer.org/github/edesz/credit-card-churn/blob/main/notebooks/06_get_best_validation_experiment.ipynb))
7. ML Evaluation
   - evaluate best ML model performance on unseen (test split) data
   - `07_evaluate.ipynb`<sup>[2](#myfootnote2)</sup> ([link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/07_evaluate.ipynb), [view](https://nbviewer.org/github/edesz/credit-card-churn/blob/main/notebooks/07_evaluate.ipynb))
8. ML Inference
   - perform inference on all available customer data
   - `08_inference.ipynb`<sup>[2](#myfootnote2)</sup> ([link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/08_inference.ipynb), [view](https://nbviewer.org/github/edesz/credit-card-churn/blob/main/notebooks/08_inference.ipynb))
9. Estimate Business Metrics on inference predictions to determine cohort of predicted to be at risk of canceling credit card services and to be targeted, using
   - ROI
     - `09_estimate_cohort_size_using_roi.ipynb`<sup>[2](#myfootnote2)</sup> ([link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/09_estimate_cohort_size_using_roi.ipynb), [view](https://nbviewer.org/github/edesz/credit-card-churn/blob/main/notebooks/09_estimate_cohort_size_using_roi.ipynb))
   - (Net) Savings
     - `10_estimate_cohort_size_using_savings.ipynb`<sup>[2](#myfootnote2)</sup> ([link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/10_estimate_cohort_size_using_savings.ipynb), [view](https://nbviewer.org/github/edesz/credit-card-churn/blob/main/notebooks/10_estimate_cohort_size_using_savings.ipynb))
10. Inference Customer Profiling
    - create profiles for customers predicted to be at risk of canceling credit card services
    - `11_get_at_risk_customer_profiles.ipynb`<sup>[2](#myfootnote2)</sup> ([link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/11_get_at_risk_customer_profiles.ipynb), [view](https://nbviewer.org/github/edesz/credit-card-churn/blob/main/notebooks/11_get_at_risk_customer_profiles.ipynb))
11. Interpret Best Machine Learning Model
    - use SHAP to explain best ML model
    - `12_interpret_model_predictions.ipynb`<sup>[2](#myfootnote2)</sup> ([link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/12_interpret_model_predictions.ipynb), [view](https://nbviewer.org/github/edesz/credit-card-churn/blob/main/notebooks/12_interpret_model_predictions.ipynb))
12. Validate Raw Customer Data
    - use `pandera` to develop data schema to validate raw credit card customer data
    - `13_validate_data.ipynb`<sup>[2](#myfootnote2)</sup> ([link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/13_validate_data.ipynb), [view](https://nbviewer.org/github/edesz/credit-card-churn/blob/main/notebooks/13_validate_data.ipynb))
13. Production ML Pipeline
    - comprehensive ML pipeline with XGBoost, LightGBM, Random Forest
    - SHAP interpretability, CLV calculator, ROI optimization
    - production-ready infrastructure (src/models, src/business, src/evaluation, src/utils)
    - `14_production_ml_pipeline.ipynb`<sup>[3](#myfootnote3)</sup> ([link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/14_production_ml_pipeline.ipynb), [view](https://nbviewer.org/github/edesz/credit-card-churn/blob/main/notebooks/14_production_ml_pipeline.ipynb))

### Data Validation

A data model was developed<sup>[2](#myfootnote2)</sup> using [Pandera](https://pypi.org/project/pandera/) for the raw credit card customer data in the `data_models` folder.

Two types of data tests were performed

1. Example-based Tests
   - a data schema was defined with with specific rules (e.g. column datatypes, constraints, etc.) and used to validate the data
2. Property-based Tests
   - statistical checks were defined in directly in the data validation schema and used to validate the relevant columns of the data

### Unit Tests

For custom modules used to perform the analysis, unit tests were implemented<sup>[2](#myfootnote2)</sup> using the [Pytest framework](https://pypi.org/project/pytest/). [Coverage](https://pypi.org/project/coverage/) was used to measure code coverage in tests. Tests are stored in the `tests` folder.

The following modules in `src` were tested

1. `cc_churn`
   - all 10 modules
2. `r2`
   - 1 of 1 module
3. `utils`
   - 1 of 3 modules

### Team Member Contributions

<a name="myfootnote1">1.</a> [sinderpreet31](https://github.com/sinderpreet31)
  - exploratory data analysis
  - preliminary linear model ML development

<a name="myfootnote2">2.</a> [edesz](https://github.com/edesz)
  - [project scoping](https://github.com/edesz/credit-card-churn/tree/main/references) and data management on private team Cloudflare R2 bucket
  - Python package management in [`pyproject.toml`](https://github.com/edesz/credit-card-churn/blob/main/pyproject.toml) for interactive model development
  - interactive cost-sensitive ML model development, including model interpretability
  - estimation of two business metrics from ML model inference, for reporting
  - Github repository management ([actions](https://github.com/edesz/credit-card-churn/blob/main/.github/workflows/main.yml), [`README.md`](https://github.com/edesz/credit-card-churn/blob/main/README.md), [Issues](https://github.com/edesz/credit-card-churn/issues?q=is%3Aissue%20state%3Aclosed))

<a name="myfootnote3">3. </a> [IlkhamFY](https://github.com/IlkhamFY)
- end-to-end scripted ML model development, including SHAP interpretability
- development of custom modules
  - [`src/business`](https://github.com/edesz/credit-card-churn/tree/main/src/business)
  - [`src/evaluation`](https://github.com/edesz/credit-card-churn/tree/main/src/evaluation)
  - `src/utils`
    - [`data_preprocessing`](https://github.com/edesz/credit-card-churn/blob/main/src/utils/data_preprocessing.py)
    - [`feature_engineering`](https://github.com/edesz/credit-card-churn/blob/main/src/utils/feature_engineering.py)
  - [scripts](https://github.com/edesz/credit-card-churn/tree/main/scripts)
  - [documentation](https://github.com/edesz/credit-card-churn/tree/main/docs)
  - [project summary](https://github.com/edesz/credit-card-churn/blob/main/PROJECT_SUMMARY.md)

## Contributing

Below is the [shared repository workflow](https://uoftcoders.github.io/studyGroup/lessons/git/collaboration/lesson/) to commit analysis code to this repository

1. [Create a fork of the project's repository in your personal Github account](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo#forking-a-repository) ([video](https://youtu.be/a_FLqX3vGR4?si=VRZRA6w4F4SLRMev&t=189))
   - creating a fork is necessary since collaborators do not have push access to main (upstream) repo
2. [Clone your forked repo repo locally](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project#making-a-pull-request) ([video](https://youtu.be/a_FLqX3vGR4?si=3Xanq8QLjp4khNfN&t=243))
   ```bash
   # clone the repository (if not already done)
   git clone https://github.com/edesz/credit-card-churn
   ```
3. Connect to original repository into your local cloned repo ([video](https://youtu.be/a_FLqX3vGR4?si=Wy59AZvvOGe6UpU1&t=272))
   ```bash
   git remote add upstream https://github.com/edesz/credit-card-churn.git
   ```
4. Pull latest changes (recommended before making changes) ([video](https://youtu.be/a_FLqX3vGR4?si=AtzQaRX_p1wyjayE&t=372))
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```
5. Make changes
   - add your analysis
6. [Push changes to your branch](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project#making-and-pushing-changes) ([video](https://youtu.be/a_FLqX3vGR4?si=5MG4CdrBxEDmoNF5&t=478))
   ```
   # verify your changes are being tracked by git
   git status
   # make changes and stage changes
   git add .
   # commit changes
   git commit -m "added advanced ml workflow notebook"
   # push changes to repository on Github
   git push
   ```
7. [Create pull request](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project#making-a-pull-request) ([video](https://youtu.be/a_FLqX3vGR4?si=SuRP9MSCJbBTMu5J&t=492))

## License

[MIT License](LICENSE.md) - Copyright 2026 [edesz](https://github.com/edesz)

## Questions or Issues?

1. GitHub Issues: https://github.com/edesz/credit-card-churn/issues

## 📦 Project Structure

```bash
├── data
│   ├── processed                     <- The final, canonical data sets for modeling.
│   └── raw                           <- The original, immutable data dump.
├── data_models
│   ├── __init__.py
│   └── staging                       <- Validation schema for raw customers data.
│       ├── base_schema.py
│       ├── behavioural_checks.py
│       ├── __init__.py
│       └── schema.py
├── docs
│   ├── env_template.txt              <- example of .env file with R2 credentials.
│   └── R2_SETUP_GUIDE.md             <- Guide to use Python to connect to team resources on R2 bucket.
├── executed-notebooks                <- Executed notebooks.
├── LICENSE                           <- Open-source license.
├── notebooks                         <- Jupyter notebooks with data analysis.
├── papermill_runner.py               <- Programmatic execution of notebooks.
├── PROJECT_SUMMARY.md
├── pyproject.toml
├── pytest.ini                        <- Pytest configuration file.
├── README.md                         <- The top-level README for developers using this project.
├── references                        <- Explanatory materials for project scoping.
├── requirements.txt                  <- Python packages needed for scripted analysis.
├── ruff.toml                         <- Configuration for code linting and formatting.
├── scripts
│   ├── demo_pipeline.py
│   └── test_r2_connection.py
└── src                               <- Source code for use in this project.
│   ├── __init__.py                   <- Makes src a Python module
│   │
│   ├── business
│   │   ├── clv_calculator.py         <- Calculate CLV for pipeline ML notebook.
│   │   ├── cost_analysis.py          <- Implement cost analysis for pipeline ML notebook.
│   │   ├── __init__.py
│   │   └── metrics.py                <- Implement business metrics for cost-sensitive ML.
│   ├── cc_churn
│   │   ├── costs_savings.py          <- Implement functions to estimate savings.
│   │   ├── costs.py                  <- Implement functions to estimate ROI.
│   │   ├── evaluation.py             <- Implement code to evaluate predictions.
│   │   ├── explanation.py            <- Implement code to explain best model.
│   │   ├── mflow_utils.py            <- Implement Metaflow custom helper functions.
│   │   ├── __init__.py
│   │   ├── scoring.py                <- Code to define custom ML scoring metrics.
│   │   ├── transformers.py           <- Source code to define custom scikit-learn transformers.
│   │   ├── tuning.py                 <- Source code to tune decision threshold.
│   │   ├── visualization.py          <- Code to make plots with matplotlib.
│   │   └── visualization.py          <- Code to make plots with altair.
│   ├── evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py                <- Code to define metrics for pipeline ML notebook.
│   │   └── model_validator.py        <- Code to implement cross-validation for pipeline ML notebook.
│   ├── r2
│   │   ├── __init__.py
│   │   └── io_utils.py               <- Source code to read and write to private R2 bucket.
│   └── utils
│       ├── data_preprocessing.py     <- Code to preprocess data for pipeline ML notebook.
│       ├── feature_engineering.py    <- Code to engineer features for pipeline ML notebook.
│       └── __init__.py
└── tests                             <- Unit test definitions, using pytest.
```
