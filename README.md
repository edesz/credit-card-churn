# Use ML to Find Credit Card Customers At Risk of Churning

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/edesz/credit-card-churn/.github%2Fworkflows%2Fmain.yml?style=for-the-badge&label=CI)
 ![Static Badge](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge&link=https%3A%2F%2Fchoosealicense.com%2Flicenses%2Fmit%2F)
 ![Static Badge](https://img.shields.io/badge/Open%20Source-Yes-yellow?style=for-the-badge&link=https%3A%2F%2Fopensource.com%2F)

## About

Use ML to identify credit card customers at a bank who are at risk of churning, for retrospective targeting.

Read the project scope [here](./references/scope/01_problem_understanding.md).

## Getting Started

### Pre-Requisites

1. [Python](https://www.python.org/)
2. [Pixi](https://pixi.prefix.dev/latest/) for package management
3. [Customer churn data](https://www.kaggle.com/datasets/imanemag/bankchurnerscsv) is stored as `BankChurners.xlsx` in private Cloudflare R2 or AWS S3 bucket, accessible to project contributors
4. `.env` file should be placed one level above root directory, to support programmatic access to data files using the [`boto3` Python package](https://pypi.org/project/boto3/). See R2 `boto3` documentation for [example usage](https://developers.cloudflare.com/r2/examples/aws/boto3/).

### Installation and Development

1. Create virtual environment for literate programming using Jupyterlab and start the server
   ```bash
   make jlab
   ```

   This command runs a Jupyterlab server and launches the web user interface in the default browser. The site will be available at http://localhost:8888.

### Available Commands

Run

```bash
make help
```

to see all available `make` rules

```bash
Available rules:

dval                Run data validation with Pandera 
jlab                Run Jupyterlab 
mlval               Run ML Validation experiments with Metaflow 
nbs                 Run notebooks 
pixi-upgrade        Upgrade conda package versions with pxi 
pixi2conda          Export pixi environment config to conda .yml
tests               Run unit tests with Pytest
```

## Sections

There are five phases to this project

1. `make jlab` (Project Scoping)
   - Creates a project scope using *all* provided customer data to determine the importance of the business problem, by estimating churn loss incurred using two KPIs
2. `make jlab` (Analysis, including ML and Data Validation)
   - Splits the data into train, validation and test splits
   - Explores the combined train+validation split of the customer data
   - Develops a parameterized ML experiment flow using Metaflow for use during validation, using the combined train+validation split
   - Compares all ML experiments to determine the best one
   - Evaluates the best model+feature's performance on unseen data (test split)
   - On all available data (combined train+validation+test split)
      - makes inference predictions
      - estimates business metrics
      - interprets model predictions
   - Develops end-to-end pipeline to run entire workflow
3. `make mlval` (dedicated ML validation)
   - During validation, approximately 25 ML experiments are run using Metaflow to determine the best combination of ML model, features and feature pre-processing
4. `make dval` (dedicated data validation)
   - In preparation for inferernce, the best ML model to predict customer churn was trained on all available data. At the same time, a data model was developed using Pandera in order to validate inference and historical data before making predictions on it.
5. `make tests` (Unit Testing)
   - Custom Python modules that were developed during the validation, evaluation and inference phases were tested using the Pytest framework

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

[MIT License](LICENSE.md) - Copyright 2025 [edesz](https://github.com/edesz)

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
├── executed-notebooks                <- Executed Jupyter notebooks.
├── LICENSE                           <- Open-source license.
├── Makefile                          <- Makefile with convenience commands like `make jlab` or `make tests`.
├── CODE_OF_CONDUCT.md                <- Define expected participant behaviour.
├── PROJECT_SUMMARY.md
├── README.md                         <- The top-level README for developers using this project.
├── notebooks                         <- Jupyter notebooks with data analysis.
├── papermill_runner.py               <- Programmatic execution of notebooks.
├── pyproject.toml                    <- Python configuration file for managing Python environments.
├── pytest.ini                        <- Pytest configuration file for running unit tests.
├── references                        <- Documentation for project scoping.
├── environment.yml                   <- Conda packages needed for interactive analysis with MyBinder.
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
|   |   ├── eda.py                    <- Implement functions to transform data for use in EDA.
│   │   ├── evaluation.py             <- Implement code to evaluate predictions.
│   │   ├── explanation.py            <- Implement code to explain best model.
│   │   ├── __init__.py
│   │   ├── mflow_utils.py            <- Implement Metaflow custom helper functions.
│   │   ├── mf_monitor_.py            <- Monitor ML model performance using tests for concept and data drift.
│   │   ├── scoring.py                <- Code to define custom ML scoring metrics.
│   │   ├── transformers.py           <- Source code to define custom scikit-learn transformers.
│   │   ├── tuning.py                 <- Source code to tune decision threshold.
│   │   ├── visualization.py          <- Code to make plots with matplotlib.
│   │   └── viz_altair.py             <- Code to make plots with altair.
│   ├── evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py                <- Code to define metrics for pipeline ML notebook.
│   │   └── model_validator.py        <- Code to implement cross-validation for pipeline ML notebook.
│   ├── r2
│   │   ├── __init__.py
│   │   └── io_utils.py               <- Source code to read and write to private R2 bucket.
│   └── utils
│       ├── data_preprocessing.py     <- Code to preprocess data for pipeline ML notebook.
│       ├── df_utils.py               <- Code to display or summarize data stoerd in a DataFrame.
│       ├── display_utils.py          <- Code to display contents of a Python module using Pygments.
│       ├── feature_engineering.py    <- Code to engineer features for pipeline ML notebook.
│       └── __init__.py
└── tests                             <- Unit test definitions, using pytest.
```
