#!/usr/bin/env python3


"""Structural definition of the raw dataset."""

import pandas as pd
import pandera as pa
from pandera import Field
from pandera.typing import Series


class BankChurnersBase(pa.DataFrameModel):
    """Structural definition of the BankChurners dataset."""

    # Identifier
    CLIENTNUM: Series[int] = pa.Field(unique=True)

    # Target
    Attrition_Flag: Series[str] = Field(
        isin=["Existing Customer", "Attrited Customer"]
    )

    # Categoricals
    Gender: Series[pd.CategoricalDtype] = Field(isin=["M", "F"])

    Marital_Status: Series[pd.CategoricalDtype] = Field(
        isin=["Married", "Single", "Divorced", "Unknown"]
    )

    Card_Category: Series[pd.CategoricalDtype] = Field(
        isin=["Blue", "Silver", "Gold", "Platinum"]
    )

    Dependent_count: Series[int] = Field(ge=0, le=5)

    # Ordinals
    Education_Level: Series[pd.CategoricalDtype] = Field(
        isin=[
            "Uneducated",
            "High School",
            "College",
            "Graduate",
            "Post-Graduate",
            "Doctorate",
            "Unknown",
        ]
    )

    Income_Category: Series[pd.CategoricalDtype] = Field(
        isin=[
            "Less than $40K",
            "$40K - $60K",
            "$60K - $80K",
            "$80K - $120K",
            "$120K +",
            "Unknown",
        ]
    )

    # Numerical columns
    Customer_Age: Series[int] = Field(ge=18, le=100)

    Months_on_book: Series[int] = Field(
        alias="Months_on_book (Length of relationship with bansk[months])",
        ge=0,
        le=600,
    )

    Total_Relationship_Count: Series[int] = Field(
        alias="Total_Relationship_Count (How many products with customer) ",
        ge=1,
        le=10,
    )

    Months_Inactive_12_mon: Series[int] = Field(
        alias="Months_Inactive_12_mon (Card not used)", ge=0, le=12
    )

    Contacts_Count_12_mon: Series[int] = Field(
        alias="Contacts_Count_12_mon (Number of contacts in 12 months)",
        ge=0,
        le=12,
    )

    Credit_Limit: Series[float] = Field(gt=0)

    Total_Revolving_Bal: Series[float] = Field(
        alias="Total_Revolving_Bal (Balance unpaid at month end)", ge=0
    )

    Avg_Open_To_Buy: Series[float] = Field(
        alias=(
            "Avg_Open_To_Buy (Difference between the credit limit and the "
            "balance)"
        ),
        ge=0,
    )

    Total_Amt_Chng_Q4_Q1: Series[float] = Field(
        alias="Total_Amt_Chng_Q4_Q1(Ratio Q4/Q1)", ge=0
    )

    Total_Trans_Amt: Series[float] = Field(
        alias="Total_Trans_Amt ( Total Transactions Value 12 months)", ge=0
    )

    Total_Trans_Ct: Series[int] = Field(
        alias="Total_Trans_Ct (Transaction Count 12 months)", ge=0
    )

    Total_Ct_Chng_Q4_Q1: Series[float] = Field(
        alias="Total_Ct_Chng_Q4_Q1 (Change in the transaction amount Q4/Q1)",
        ge=0,
    )

    Avg_Utilization_Ratio: Series[float] = Field(
        alias="Avg_Utilization_Ratio (Credit usage/Total Credit available)",
        ge=0,
        le=1,
    )

    class Config:
        """Configuration for the schema."""

        strict = False
        coerce = True
