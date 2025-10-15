# credit-card-churn

Welcome to **credit-card-churn**!

## Description

Use ML to identify customers at risk of churning for proactive targeting.

See the project scope [here](https://github.com/edesz/credit-card-churn/blob/main/references/01_proposal.md).

## Analysis

1. Split Raw Data (`01_split_data.ipynb`)
   - create train, validation and test splits
   - team member: I. S.
2. Perform Quantitative Analysis for Project Scoping (`02_scoping.ipynb`)
   - estimate quantitative impact of credit card churn
   - team member: E. D.
3. EDA (`03_eda.ipynb`)
   - exploratory data analysis
   - team member: I. S.
4. Machine Learning
   - starter notebook (`04_model_development_logisticregression.ipynb`)
     - team member: I. S.
   - advanced notebook (`05_*.ipynb`)
     - team member: I. Y.
5. Business Metrics (`06_*.ipynb`)
   - calculating business metrics and identifying customers to target using ML predictions
     - team member: E. D.

## Contributing (for project collaborators)

Below is the [shared repository workflow that can be followed](https://uoftcoders.github.io/studyGroup/lessons/git/collaboration/lesson/) to commit analysis code to Github

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
   - add your notebook
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

This example is for adding an advanced ML notebook (named `05_advanced_ml_development.ipynb`). For updating any other notebook, follow a similar workflow.

## 📦 Project Structure

To be done.
