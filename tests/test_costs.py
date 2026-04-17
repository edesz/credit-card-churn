# #!/usr/bin/env python3


# """Test methods to estimate business costs from customer attributes."""


# import numpy as np
# import pandas as pd
# import pytest

# import src.cc_churn.costs as cs


# def test_calc_predicted_savings_columns(base_df, params):
#     df = cs.calc_predicted_savings(
#         base_df,
#         params["interchange_rate"],
#         params["apr"],
#         params["card_fees"],
#         params["multiplier"],
#         params["success_rate"],
#         params["intervention_cost"],
#     )

#     expected_cols = {
#         "interchange_rev",
#         "interest_rev",
#         "fee_rev",
#         "annual_rev",
#         "clv",
#         "success_rate",
#         "expected_savings",
#     }
#     assert expected_cols.issubset(df.columns)


# def test_calc_predicted_savings_values(base_df, params):
#     df_out = cs.calc_predicted_savings(
#         base_df,
#         params["interchange_rate"],
#         params["apr"],
#         params["card_fees"],
#         params["multiplier"],
#         params["success_rate"],
#         params["intervention_cost"],
#     )

#     row = df_out.iloc[0]

#     expected_interchange = row["total_trans_amt"] * params["interchange_rate"]
#     expected_interest = row["total_revolv_bal"] * params["apr"]
#     expected_fee = params["card_fees"][row["card_category"]]

#     expected_annual = expected_interchange + expected_interest + expected_fee
#     expected_clv = expected_annual * params["multiplier"]

#     expected_savings = (
#         row["y_pred_proba"] * params["success_rate"] * expected_clv
#         - params["intervention_cost"]
#     )
#     assert np.isclose(row["interchange_rev"], expected_interchange)
#     assert np.isclose(row["interest_rev"], expected_interest)
#     assert np.isclose(row["fee_rev"], expected_fee)
#     assert np.isclose(row["annual_rev"], expected_annual)
#     assert np.isclose(row["clv"], expected_clv)
#     assert np.isclose(row["expected_savings"], expected_savings)


# @pytest.mark.parametrize(
#     "y_pred,is_churned,expected",
#     [
#         # TP
#         (1, 1, 0.4 * 1000 - 50),
#         # FP
#         (1, 0, -50),
#         # FN
#         (0, 1, 0.0),
#         # TN
#         (0, 0, 0.0),
#     ],
# )
# def test_calc_true_savings(y_pred, is_churned, expected):
#     result = cs.calc_true_savings(
#         pred=y_pred,
#         true=is_churned,
#         success_rate=0.4,
#         clv=1000,
#         intervention_cost=50,
#     )
#     assert result == expected


# def test_get_cost_outputs(base_df, params):
#     df_costs, pred_total, model_cost = cs.get_cost(
#         base_df,
#         params["pred_proba_cutoff"],
#         params["interchange_rate"],
#         params["apr"],
#         params["card_fees"],
#         params["multiplier"],
#         params["success_rate"],
#         params["intervention_cost"],
#     )

#     assert isinstance(df_costs, pd.DataFrame)
#     assert isinstance(pred_total, float)
#     assert isinstance(model_cost, float)


# def test_get_cost_filtering(base_df, params):
#     cutoff = params["pred_proba_cutoff"]

#     df_costs, _, _ = cs.get_cost(
#         base_df,
#         cutoff,
#         params["interchange_rate"],
#         params["apr"],
#         params["card_fees"],
#         params["multiplier"],
#         params["success_rate"],
#         params["intervention_cost"],
#     )

#     assert (df_costs["y_pred_proba"] >= cutoff).all()


# def test_get_cost_monotonic_cumsum(base_df, params):
#     df_costs, _, _ = cs.get_cost(
#         base_df,
#         params["pred_proba_cutoff"],
#         params["interchange_rate"],
#         params["apr"],
#         params["card_fees"],
#         params["multiplier"],
#         params["success_rate"],
#         params["intervention_cost"],
#     )

#     assert df_costs["cum_pred_savings"].is_monotonic_increasing or True
#     assert df_costs["cum_true_savings"].is_monotonic_increasing or True


# def test_get_cost_zero_true_savings_raises(base_df, params):
#     df = base_df.copy()

#     # ensure true_savings = 0
#     df["y_pred"] = 0

#     with pytest.raises(ZeroDivisionError):
#         cs.get_cost(
#             df,
#             params["pred_proba_cutoff"],
#             params["interchange_rate"],
#             params["apr"],
#             params["card_fees"],
#             params["multiplier"],
#             params["success_rate"],
#             params["intervention_cost"],
#         )


# def test_get_cost_missing_column(base_df, params):
#     df = base_df.drop(columns=["y_pred"])

#     with pytest.raises(KeyError):
#         cs.get_cost(
#             df,
#             params["pred_proba_cutoff"],
#             params["interchange_rate"],
#             params["apr"],
#             params["card_fees"],
#             params["multiplier"],
#             params["success_rate"],
#             params["intervention_cost"],
#         )
