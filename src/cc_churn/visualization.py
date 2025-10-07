#!/usr/bin/env python3


"""Define helper functions for visualising model metrics."""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_roi_curves(
    n,
    true_savings,
    predicted_savings,
    roi_percent,
    roi_percent_pred,
    optimal_n_roi_dict,
    ptitle,
    legend_loc,
    xlabel,
    fig_size=(12, 8),
):
    """."""
    _, ax = plt.subplots(figsize=fig_size)
    ax.plot(
        n,
        predicted_savings,
        label="Expected Savings (Predicted)",
        linewidth=1,
        color="orange",
        alpha=0.5,
    )
    ax.plot(
        n,
        true_savings,
        label="Expected Savings (True)",
        linewidth=1,
        color="purple",
        alpha=0.5,
    )
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Expected Savings ($)", fontsize=14)
    for k, v in optimal_n_roi_dict.items():
        ax.axvline(v["x"], color=v["colour"], linestyle="--", label=k)

    ax2 = ax.twinx()
    ax2.plot(
        n,
        roi_percent,
        color="green",
        label="ROI True (%)",
        linewidth=2,
    )
    ax2.plot(
        n,
        roi_percent_pred,
        color="magenta",
        label="ROI Predicted (%)",
        linewidth=2,
    )
    ax2.set_ylabel("ROI (%)", fontsize=14)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc=legend_loc, frameon=False)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_title(ptitle, loc="left", fontsize=14)
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)
    ax2.grid(True, alpha=0.4)
    ax2.tick_params(axis="y", right=False)
    ax.spines[["right", "top"]].set_visible(False)
    ax2.spines[["right", "top"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)
    ax2.tick_params(axis="both", labelsize=12)


def plot_lift_curves(
    n,
    true_savings,
    predicted_savings,
    random_savings,
    lift,
    lift_pred,
    optimal_n_lift_dict,
    ptitle,
    xlabel,
    fig_size=(12, 8),
):
    """."""
    _, ax = plt.subplots(figsize=fig_size)
    ax.plot(
        n,
        random_savings,
        label="Random Targeting (Expected Savings)",
        linestyle="--",
        linewidth=2,
    )
    ax.plot(
        n,
        predicted_savings,
        label="Expected Savings (Predicted)",
        linestyle="-",
        linewidth=1,
        color="orange",
        alpha=0.4,
    )
    ax.plot(
        n,
        true_savings,
        label="Expected Savings (True)",
        linewidth=1,
        color="purple",
        alpha=0.4,
    )
    ax.plot(
        n,
        lift_pred,
        label="Incremental Lift Curve vs Random (Predicted)",
        linewidth=2,
        color="magenta",
    )
    ax.plot(
        n,
        lift,
        label="Incremental Lift Curve vs Random (True)",
        linewidth=2,
        color="green",
    )
    for k, v in optimal_n_lift_dict.items():
        ax.axvline(v["x"], color=v["colour"], linestyle="--", label=k)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Expected Savings, Lift", fontsize=14)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_title(ptitle, loc="left", fontsize=14)
    ax.set_xlim(xmin=0)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, alpha=0.4)
    ax.spines[["right", "top"]].set_visible(False)
