#!/usr/bin/env python3


"""Structural definition of the raw dataset."""

import numpy as np
import pandas as pd
import pandera as pa
from scipy import stats


class BehaviouralChecks:
    """Reusable checks for bank customer behavior."""

    # cross-field validation
    @pa.dataframe_check
    def check_credit_balance_consistency(cls, df: pd.DataFrame) -> pd.Series:
        """
        Validate that credit components sum correctly.

        Ensures:
            Credit_Limit ≈ Total_Revolving_Bal + Avg_Open_To_Buy

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.Series[bool]: Boolean Series indicating valid rows.
        """
        col_1 = "Total_Revolving_Bal (Balance unpaid at month end)"
        col_2 = (
            "Avg_Open_To_Buy (Difference between the credit limit and the "
            "balance)"
        )
        lhs = df["Credit_Limit"]
        rhs = df[col_1].add(df[col_2])
        return (lhs == rhs).all()

    @pa.dataframe_check
    def check_utilization_ratio(cls, df: pd.DataFrame) -> pd.Series:
        """
        Validate utilization ratio consistency.

        Ensures:
            Avg_Utilization_Ratio ≈ Total_Revolving_Bal / Credit_Limit

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.Series[bool]: Boolean Series indicating valid rows.
        """
        numerator_c = "Total_Revolving_Bal (Balance unpaid at month end)"
        denominator_c = "Credit_Limit"
        true_c = "Avg_Utilization_Ratio (Credit usage/Total Credit available)"
        expected = df[numerator_c].div(df[denominator_c]).rename("expected")
        true = df[true_c].rename("true")
        num_decimals = 3
        return (
            pd.concat([true, expected], axis=1)
            .assign(
                check=lambda df: (
                    df["true"]
                    .round(num_decimals)
                    .equals(df["expected"].round(num_decimals))
                )
            )["check"]
            .all()
        )

    @pa.dataframe_check
    def check_zscore_outliers(cls, df: pd.DataFrame) -> pd.Series:
        """
        Detect statistical outliers using z-score bounds.

        Flags rows where any selected numerical column exceeds
        a z-score threshold.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.Series[bool]: Boolean Series indicating valid rows.
        """
        cols = [
            "Customer_Age",
            "Months_on_book (Length of relationship with bansk[months])",
            "Total_Relationship_Count (How many products with customer) ",
            "Months_Inactive_12_mon (Card not used)",
            "Contacts_Count_12_mon (Number of contacts in 12 months)",
            "Credit_Limit",
            "Total_Revolving_Bal (Balance unpaid at month end)",
            (
                "Avg_Open_To_Buy (Difference between the credit limit and "
                "the balance)"
            ),
            "Total_Trans_Ct (Transaction Count 12 months)",
        ]

        z_thresh = 4.0

        sub_df = df[cols]

        # Compute z-scores safely
        mean = sub_df.mean()
        std = sub_df.std(ddof=0).replace(0, np.nan)

        z_scores = (sub_df - mean) / std

        # Identify rows where ANY column exceeds threshold
        outlier_mask = (np.abs(z_scores) > z_thresh).any(axis=1)

        # Valid rows = NOT outliers
        return ~outlier_mask

    @pa.dataframe_check
    def check_behavioral_anomalies(cls, df: pd.DataFrame) -> pd.Series:
        """
        Detect behavioral anomalies based on credit usage patterns.

        Rule:
            - High utilization (>= 0.8)
            - Low transaction count (<= 10)
            - Low total transaction amount (<= 1000)

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.Series[bool]: Boolean Series indicating valid rows.
        """
        c1 = "Avg_Utilization_Ratio (Credit usage/Total Credit available)"
        c2 = "Total_Trans_Ct (Transaction Count 12 months)"
        c3 = "Total_Amt_Chng_Q4_Q1(Ratio Q4/Q1)"
        high_util = df[c1] >= 0.8
        low_txn_count = df[c2] <= 10
        low_txn_amt = df[c3] <= 1000

        anomaly_mask = high_util & low_txn_count & low_txn_amt

        # Valid rows = NOT anomalous
        return ~anomaly_mask

    @pa.check("Attrition_Flag", element_wise=False)
    def check_value_counts(cls, s: pd.Series) -> bool:
        """
        Validate class distribution for the target variable.

        Ensures that the proportion of "Existing Customer" observations
        meets a minimum threshold, reflecting the expected class imbalance
        in the dataset.

        Args:
            s (pd.Series): Target column containing attrition labels.

        Returns:
            bool: True if the proportion of "Existing Customer" is greater
            than or equal to 0.83, otherwise False.
        """
        vc = s.value_counts(normalize=True)
        return vc.get("Existing Customer", 0) >= 0.83

    @pa.check(
        "Total_Revolving_Bal (Balance unpaid at month end)",
        groupby="Attrition_Flag",
        name="revolving_bal_t_test",
    )
    def check_balance_stats(cls, grouped_data):
        """
        Validate statistical difference in revolving balances.

        Performs an independent t-test to ensure that the distribution of
        revolving balances differs significantly between 'Existing' and
        'Attrited' customers, confirming population divergence.

        Args:
            grouped_data (dict): Mapping of 'Attrition_Flag' categories to
                'Total_Revolving_Bal' Series.

        Returns:
            bool: True if p-value < 0.05, indicating significant difference.
        """
        # Access the groups directly from the mapping
        existing = grouped_data.get("Existing Customer")
        attrited = grouped_data.get("Attrited Customer")

        # Guard clause: ensure both groups exist in the data batch
        if existing is None or attrited is None:
            return False

        # Perform the t-test
        t_stat, p_value = stats.ttest_ind(existing, attrited)

        # Check if the difference is statistically significant (p < 0.05)
        return p_value < 0.05

    @pa.check(
        "Total_Trans_Ct (Transaction Count 12 months)",
        groupby="Attrition_Flag",
        name="test_transaction_count_drift",
    )
    def check_trans_count(cls, grouped_data):
        """
        Validate transaction count distribution across customer groups.

        Hypothesizes that existing customers maintain a significantly
        higher transaction count than those who have attrited. Uses a
        Welch's t-test to detect mean drift.

        Args:
            grouped_data (dict): Mapping of 'Attrition_Flag' categories to
                'Total_Trans_Ct' Series.

        Returns:
            bool: True if p-value < 0.01, indicating high significance.
        """
        existing = grouped_data.get("Existing Customer")
        attrited = grouped_data.get("Attrited Customer")
        # Use a t-test to check if the means are different
        _, p_val = stats.ttest_ind(existing, attrited, equal_var=False)
        # Expecting a very high significance ($p < 0.01$)
        return p_val < 0.01

    @pa.check(
        "Total_Trans_Amt ( Total Transactions Value 12 months)",
        groupby="Attrition_Flag",
        name="test_transaction_amount_drift",
    )
    def check_trans_amount(cls, grouped_data):
        """
        Validate transaction amount distribution drift.

        Hypothesizes that the average total transaction amount for existing
        customers is significantly greater than for attrited customers,
        reflecting higher engagement.

        Args:
            grouped_data (dict): Mapping of 'Attrition_Flag' categories to
                'Total_Trans_Amt' Series.

        Returns:
            bool: True if p-value < 0.01, indicating the spending patterns
                between groups remain distinct.
        """
        existing = grouped_data.get("Existing Customer")
        attrited = grouped_data.get("Attrited Customer")
        _, p_val = stats.ttest_ind(existing, attrited, equal_var=False)
        return p_val < 0.01

    @pa.check(
        "Contacts_Count_12_mon (Number of contacts in 12 months)",
        groupby="Attrition_Flag",
        name="test_contact_frequency",
    )
    def check_contact_count(cls, grouped_data):
        """
        Validate contact frequency behavioral property.

        Tests the hypothesis that attrited customers have a higher mean
        contact count (likely due to service queries or complaints) than
        existing customers in the 12 months prior to churn.

        Args:
            grouped_data (dict): Mapping of 'Attrition_Flag' categories to
                'Contacts_Count_12_mon' Series.

        Returns:
            bool: True if mean(Attrited) > mean(Existing).
        """
        existing = grouped_data.get("Existing Customer")
        attrited = grouped_data.get("Attrited Customer")
        # We expect mean(attrited) > mean(existing)
        return attrited.mean() > existing.mean()

    @pa.check(
        "Avg_Utilization_Ratio (Credit usage/Total Credit available)",
        groupby="Attrition_Flag",
        name="test_utilization_drop",
    )
    def check_utilization(cls, grouped_data):
        """
        Validate utilization ratio divergence.

        Ensures that existing customers show higher credit utilization
        on average compared to attrited customers, who typically stop
        using credit before churning.

        Args:
            grouped_data (dict): Mapping of 'Attrition_Flag' categories to
                'Avg_Utilization_Ratio' Series.

        Returns:
            bool: True if p-value < 0.01 and mean(Existing) > mean(Attrited).
        """
        existing = grouped_data.get("Existing Customer")
        attrited = grouped_data.get("Attrited Customer")
        _, p_val = stats.ttest_ind(existing, attrited, equal_var=False)
        return p_val < 0.01 and existing.mean() > attrited.mean()

    @pa.check(
        "Months_Inactive_12_mon (Card not used)",
        groupby="Attrition_Flag",
        name="test_inactivity_drift",
    )
    def check_inactivity(cls, grouped_data):
        """
        Validate months of inactivity distribution.

        Hypothesizes that customers who have attrited show a statistically
        higher mean number of inactive months compared to loyal customers.

        Args:
            grouped_data (dict): Mapping of 'Attrition_Flag' categories to
                'Months_Inactive_12_mon' Series.

        Returns:
            bool: True if mean(Attrited) > mean(Existing).
        """
        existing = grouped_data.get("Existing Customer")
        attrited = grouped_data.get("Attrited Customer")
        return attrited.mean() > existing.mean()
