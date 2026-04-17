#!/usr/bin/env python3


"""Define final schema combining structural and behavioural checks."""

from .base_schema import BankChurnersBase
from .behavioural_checks import BehaviouralChecks


class CreditCardCustomerSchema(BankChurnersBase, BehaviouralChecks):
    """
    Schema for validating the BankChurners dataset.

    This model validates customer credit card data for churn prediction,
    ensures integrity of financial ratios, detects population drift between
    churned and safe customers, coercing types where necessary and enforces:
    - Column presence and datatypes
    - Categorical membership constraints
    - Logical numerical ranges and non-negativity
    - Cross-field consistency rules
    - Statistical outlier detection
    - Behavioral anomaly detection
    - Property-based statistical hypotheses

    Attributes:
        CLIENTNUM (Series[int]): Unique customer identifier.
        Attrition_Flag (Series[str]): Target variable; must be 'Existing
            Customer' or 'Attrited Customer'.
        Gender (Series[pd.CategoricalDtype]): Customer gender ('M', 'F').
        Marital_Status (Series[pd.CategoricalDtype]): Marital status
            categories including 'Unknown'.
        Card_Category (Series[pd.CategoricalDtype]): Type of card held.
        Dependent_count (Series[int]): Number of dependents (0-5).
        Education_Level (Series[pd.CategoricalDtype]): Highest level
            of education completed.
        Income_Category (Series[pd.CategoricalDtype]): Annual income
            bracket.
        Customer_Age (Series[int]): Age of the customer (18-100).
        Months_on_book (Series[int]): Length of relationship (0-600).
        Total_Relationship_Count (Series[int]): Number of products held.
        Months_Inactive_12_mon (Series[int]): Months without usage (0-12).
        Contacts_Count_12_mon (Series[int]): Bank contacts in 12 months.
        Credit_Limit (Series[float]): Total credit limit (> 0).
        Total_Revolving_Bal (Series[float]): Unpaid balance at month end.
        Avg_Open_To_Buy (Series[float]): Available credit.
        Total_Amt_Chng_Q4_Q1 (Series[float]): Spending ratio change.
        Total_Trans_Amt (Series[float]): Total 12-month transaction value.
        Total_Trans_Ct (Series[int]): Total 12-month transaction count.
        Total_Ct_Chng_Q4_Q1 (Series[float]): Transaction count ratio change.
        Avg_Utilization_Ratio (Series[float]): Percentage of credit used (0-1).

    Returns:
        pd.DataFrame: A validated and type-coerced DataFrame.

    Raises:
        pandera.errors.SchemaError: If validation fails due to invalid values,
            missing columns, or constraint violations.
        pandera.errors.SchemaErrors: If multiple validation errors are found
            when lazy validation is enabled.
    """

    pass
