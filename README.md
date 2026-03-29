# credit-card-churn

## About

Use ML to identify credit card customers at risk of churning for proactive targeting.

See the project scope [here](https://github.com/edesz/credit-card-churn/blob/main/references/01_proposal.md).

## Analysis

1. Split Raw Data (`01_split_data.ipynb`<sup>[1](#myfootnote1)</sup>, [link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/01_split_data.ipynb))
   - create train, validation and test splits
2. Perform Quantitative Analysis for Project Scoping (`02_scoping.ipynb`<sup>[2](#myfootnote2)</sup>, [link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/02_scoping.ipynb))
   - estimate quantitative impact of credit card churn
3. Data Drift, Quality and Stats Tests (`03_data_checks.ipynb`<sup>[2](#myfootnote2)</sup>, [link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/03_data_checks.ipynb))
   - run three types of tests on unseen data relative to ML model training data
4. EDA (`04_eda.ipynb`<sup>[1](#myfootnote1)</sup>, [link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/04_eda.ipynb))
   - exploratory data analysis
5. Machine Learning - Cost-Sensitive Learning
   - linear models (`05_model_development__cost_sensitive_learning__linear_models.ipynb`<sup>[2](#myfootnote2)</sup>, [link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/05_model_development__cost_sensitive_learning__linear_models.ipynb))
   - tree-based models (`06_model_development__cost_sensitive_learning__tree_based_models.ipynb`<sup>[2](#myfootnote2)</sup>, [link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/06_model_development__cost_sensitive_learning__tree_based_models.ipynb))
6. Business Metrics Analysis
   - at-risk customers (`07_get_at_risk_and_max_roi_customers_from_business_metrics.ipynb`<sup>[2](#myfootnote2)</sup>, [link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/07_get_at_risk_and_max_roi_customers_from_business_metrics.ipynb))
   - all customers ROI (`08_all_customers_get_at_risk_and_max_roi_customers_from_business_metrics.ipynb`<sup>[2](#myfootnote2)</sup>, [link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/08_all_customers_get_at_risk_and_max_roi_customers_from_business_metrics.ipynb))
7. Production ML Pipeline (`09_production_ml_pipeline.ipynb`<sup>[3](#myfootnote3)</sup>, [link](https://github.com/edesz/credit-card-churn/blob/main/notebooks/09_production_ml_pipeline.ipynb))
   - comprehensive ML pipeline with XGBoost, LightGBM, Random Forest
   - SHAP interpretability, CLV calculator, ROI optimization
   - production-ready infrastructure (src/models, src/business, src/evaluation, src/utils)

Team Contributions: <a name="myfootnote1">1</a> - [sinderpreet31](https://github.com/sinderpreet31), <a name="myfootnote2">2</a> - [edesz](https://github.com/edesz), <a name="myfootnote3">3</a> - [IlkhamFY](https://github.com/IlkhamFY)

## Contributing

Below is the [shared repository workflow](https://uoftcoders.github.io/studyGroup/lessons/git/collaboration/lesson/) to commit analysis code to this repository

1. [Create a fork of the project's repository in your personal account](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo#forking-a-repository) ([video](https://youtu.be/a_FLqX3vGR4?si=VRZRA6w4F4SLRMev&t=189))
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

## 📦 Project Structure

```bash
├── data
│   ├── processed                    <- The final, canonical data sets for modeling.
│   └── raw                          <- The original, immutable data dump.
├── docs
│   ├── env_template.txt              <- example of .env file contents with credentials.
│   └── R2_SETUP_GUIDE.md             <- Guide to use Python to connect to team resources on R2 bucket.
├── executed-notebooks                <- Executed notebooks.
├── LICENSE                           <- Open-source license.
├── notebooks                         <- Jupyter notebooks with data analysis.
├── PROJECT_SUMMARY.md
├── pyproject.toml
├── README.md                         <- The top-level README for developers using this project.
├── references                        <- Explanatory materials for project scoping.
├── requirements.txt
├── ruff.toml                         <- Configuration for code linting and formatting.
├── scripts
│   ├── demo_pipeline.py
│   └── test_r2_connection.py
└── src                               <- Source code for use in this project.
    ├── __init__.py                   <- Makes src a Python module
    │
    ├── business
    │   ├── clv_calculator.py         <- Calculate CLV for pipeline ML notebook.
    │   ├── cost_analysis.py          <- Implement cost analysis for pipeline ML notebook.
    │   ├── __init__.py
    │   └── metrics.py                <- Implement business metrics for cost-sensitive ML.
    ├── cc_churn
    │   ├── costs.py                  <- Implement functions to calculate savings and cost.
    │   ├── evaluation.py             <- Implement code to evaluate predictions.
    │   ├── __init__.py
    │   ├── scoring.py                <- Code to define custom ML scoring metrics.
    │   ├── tuning.py                 <- Source code to tune decision threshold.
    │   └── visualization.py          <- Code to make plots.
    ├── evaluation
    │   ├── __init__.py
    │   ├── metrics.py                <- Code to define metrics for pipeline ML notebook.
    │   └── model_validator.py        <- Code to implement cross-validation for pipeline ML notebook.
    ├── r2
    │   ├── __init__.py
    │   └── io_utils.py               <- Source code to read and write to private R2 bucket.
    └── utils
        ├── data_preprocessing.py     <- Code to preprocess data for pipeline ML notebook.
        ├── feature_engineering.py    <- Code to engineer features for pipeline ML notebook.
        └── __init__.py
```
