#!/usr/bin/env python3


"""Define helper functions for creating visualizations."""

from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import shap


def plot_roi_curves(
    n: pd.Series,
    roi_percent: pd.Series,
    roi_percent_pred: pd.Series,
    ptitle: str,
    legend_loc,
    xlabel: str,
    ylabel: str,
    grid_line_opacity: float = 0.4,
    ticklabel_font_size: int = 14,
    axis_label_font_size: int = 14,
    title_font_size: int = 14,
    axis_font_color: str = "black",
    line_colors: Dict[str, str] = {
        "Predicted": "darkgreen",
        "True": "lightgrey",
    },
    fig_size: Tuple[int] = (12, 8),
) -> None:
    """Plots curves comparing predicted vs true return on investment (ROI).

    This function visualizes ROI as a function of
    the number of customers targeted. It compares cumulative ROI based on
    predicted savings versus true realized savings.

    Args:
        n: Series representing number of customers targeted.
        roi_percent: Series of true ROI percentages.
        roi_percent_pred: Series of predicted ROI percentages.
        ptitle: Plot title.
        legend_loc: Location of legend in matplotlib format.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        grid_line_opacity: Transparency for grid lines.
        ticklabel_font_size: Font size for tick labels.
        axis_label_font_size: Font size for axis labels.
        title_font_size: Font size for title.
        axis_font_color: Color for axis labels and ticks.
        line_colors: Dictionary specifying line colors.
        fig_size: Tuple defining figure size.

    Returns:
        None: Displays the plot.

    Examples:
        >>> plot_roi_curves(n, roi_true, roi_pred, "ROI", "best",
        ...                 "Customers", "ROI (%)")
    """
    _, ax2 = plt.subplots(figsize=fig_size)
    ax2.plot(
        n,
        roi_percent,
        color=line_colors["True"],
        label="True",
        linewidth=2,
    )
    ax2.plot(
        n,
        roi_percent_pred,
        color=line_colors["Predicted"],
        label="Predicted",
        linewidth=2,
    )

    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        lines2,
        labels2,
        loc=legend_loc,
        frameon=False,
        fontsize=ticklabel_font_size,
    )
    ax2.set_title(ptitle, loc="left", fontsize=title_font_size)
    ax2.grid(True, alpha=grid_line_opacity)
    ax2.tick_params(
        axis="both",
        left=False,
        bottom=False,
        labelsize=ticklabel_font_size,
        labelcolor=axis_font_color,
    )
    ax2.spines[["left", "right", "top", "bottom"]].set_visible(False)
    ax2.set_xlabel(xlabel, fontsize=axis_label_font_size, color=axis_font_color)
    ax2.set_ylabel(ylabel, fontsize=axis_label_font_size, color=axis_font_color)
    ax2.tick_params(
        axis="both", labelsize=ticklabel_font_size, color=axis_font_color
    )


def plot_class_imbalance_proba_distribution(
    df_class_imbalance: pd.DataFrame,
    df_probabilities: pd.DataFrame,
    ptitle1: str,
    title1_xloc: float,
    ptitle2: str,
    vline_label: str,
    decision_threshold: float,
    subfigure_width_ratios: List[Union[float, int]] = [1.15, 3],
    fig_size: Tuple[int] = (12, 4),
) -> None:
    """Plots class imbalance and predicted probability distribution.

    This function creates a two-panel figure:
        1. Bar chart showing class distribution.
        2. Histogram of predicted probabilities with threshold marker.

    Args:
        df_class_imbalance: DataFrame with class distribution.
        df_probabilities: Series/DataFrame of predicted probabilities.
        ptitle1: Title for class imbalance plot.
        title1_xloc: Horizontal alignment for first title.
        ptitle2: Title for probability distribution plot.
        vline_label: Label for decision threshold line.
        decision_threshold: Threshold value for classification.
        subfigure_width_ratios: Width ratio of subplots.
        fig_size: Tuple defining figure size.

    Returns:
        None: Displays the plot.

    Examples:
        >>> plot_class_imbalance_proba_distribution(
        ...     df_counts, df_probs, "Class Balance", 0.1,
        ...     "Probability Dist", "Threshold", 0.5
        ... )
    """
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


def plot_shap_summary_plots(
    shap_explainer_values: shap._explanation.Explanation,
    y_axis_annot_offset: float = 0.3,
    axis_label_fontsize: int = 14,
    y_axis_annot_fontsize: int = 14,
    wspace: float = 0.5,
    fpath: str = "../reports/figures/myfile.png",
    fig_size: tuple = (15, 12),
) -> None:
    """Plots SHAP summary visualizations for model interpretation.

    This function generates two SHAP plots:
        1. Bar plot of mean absolute SHAP values (feature importance).
        2. Beeswarm plot showing feature impact distribution.

    Args:
        shap_explainer_values: SHAP Explanation object.
        y_axis_annot_offset: Offset for annotation labels on y-axis.
        axis_label_fontsize: Font size for axis labels.
        y_axis_annot_fontsize: Font size for annotation text.
        wspace: Width space between subplots.
        fpath: The destination file path for sving the figure.
        fig_size: Tuple defining figure size.

    Returns:
        None: Displays the plots.

    Examples:
        >>> plot_shap_summary_plots(shap_values)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=fig_size)
    plt.subplots_adjust(wspace=wspace)

    shap.plots.bar(shap_explainer_values, ax=ax2, show=False)
    ax2.spines["left"].set_visible(False)
    ax2.set_xlabel("SHAP value", fontsize=axis_label_fontsize)
    ax2.tick_params(axis="both", labelsize=axis_label_fontsize)
    ax2.tick_params(axis="y", pad=125)
    for tick in ax2.get_yticklabels():
        tick.set_horizontalalignment("center")
    # ax1.invert_xaxis()
    # _ = ax2.set_yticklabels([])
    for text in ax2.texts:
        x, y = text.get_position()
        text.set_position((x + y_axis_annot_offset, y))
        text.set_size(y_axis_annot_fontsize)

    shap.plots.beeswarm(
        shap_explainer_values,
        ax=ax1,
        plot_size=None,
        show=False,
        color_bar=False,
    )
    ax1.tick_params(axis="both", labelsize=axis_label_fontsize)
    ax1.set_xlabel("Mean Absolute SHAP value", fontsize=axis_label_fontsize)
    _ = ax1.set_yticklabels([])

    if fpath:
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
