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
    ylabel,
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
    ax.set_ylabel(ylabel, fontsize=14)
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


def plot_class_imbalance_proba_distribution(
    df_class_imbalance,
    df_probabilities,
    ptitle1,
    title1_xloc,
    ptitle2,
    vline_label,
    decision_threshold,
    subfigure_width_ratios=[1.15, 3],
    fig_size=(12, 4),
):
    """."""
    _, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        gridspec_kw={"width_ratios": subfigure_width_ratios},
        figsize=fig_size,
    )

    # class imbalance in true labels
    ax1 = df_class_imbalance.plot.bar(
        grid=False,
        color={"True": "darkgreen", "Predicted": "Red"},
        zorder=2,
        width=0.7,
        edgecolor="white",
        linewidth=3,
        ax=ax1,
    )
    ax1.legend(frameon=False, handletextpad=0.2)
    ax1.bar_label(ax1.containers[0], fmt="%.3f")
    ax1.set_title(ptitle1, x=title1_xloc, loc="left", fontsize=11)
    ax1.set_xlabel("Churn Outcome", fontsize=14)
    ax1.set_ylabel("Fraction of Customers", fontsize=14)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.15))
    ax1.tick_params(axis="both", which="both", labelsize=10, labelcolor="grey")
    ax1.tick_params(axis="y", left=False)
    ax1.tick_params(axis="x", labelrotation=0)
    ax1.grid(True, axis="y", alpha=0.4)
    ax1.spines[["left", "right", "top"]].set_visible(False)

    # distribution of predicted probabilities
    ax2 = df_probabilities.plot.hist(
        bins=50,
        grid=False,
        color="#86bf91",
        zorder=2,
        rwidth=0.9,
        ax=ax2,
        label="",
    )
    ax2.axvline(
        x=decision_threshold * 100,
        color="black",
        linestyle="--",
        linewidth=2,
        zorder=3,
        label=vline_label,
    )
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax2.set_title(ptitle2, loc="left", fontsize=12.5)
    ax2.legend(
        frameon=False,
        handletextpad=0.2,
        handlelength=0,
        loc="upper right",
        bbox_to_anchor=(0.85, 0.65),
    )
    ax2.set_xlim(0)
    ax2.tick_params(axis="both", which="both", labelsize=14, labelcolor="grey")
    ax2.tick_params(axis="y", left=False)
    ax2.set_ylabel("Number of Customers", fontsize=14)
    ax2.set_xlabel("Prediction Probability (%)", fontsize=14)
    ax2.grid(True, axis="x", alpha=0.4)
    ax2.spines[["left", "right", "top"]].set_visible(False)
