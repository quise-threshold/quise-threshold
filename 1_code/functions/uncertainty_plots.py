import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


def plot_classification_uncertainty(
    df,
    nominal_col="NOMINAL",
    var_rep_prefix="Index_Rep",
    n_replicates=80,
    fig_size=(12, 10),
    reference_line=0.75,
    scale_to_100=True,
    add_designation_regions=True,
    save_stages=False,
    save_prefix="plot_stage",
    plot_dir="../2_plots"
):
    """
    Plot classification uncertainty across variance replicate index realizations.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the nominal index and variance replicate indices.

    nominal_col : str
        Name of the nominal index column.

    var_rep_prefix : str
        Prefix used for replicate-index columns.

    n_replicates : int
        Number of variance replicate indices.

    fig_size : tuple
        Matplotlib figure size.

    reference_line : float
        Policy threshold on the original 0-1 index scale.

    scale_to_100 : bool
        Display index values on a 0-100 scale.

    add_designation_regions : bool
        Add classification uncertainty zones where they can be calculated.

    save_stages : bool
        Save intermediate versions of the plot.

    save_prefix : str
        Prefix for intermediate plot filenames.

    plot_dir : str
        Directory for saved intermediate plots.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """

    # ------------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------------

    light_blue = "#3498db"
    dark_blue = "#2980b9"
    line_blue = "#8c6d5c"
    dark_grey = "#2c3e50"
    raw_data_color = "#95a5a6"

    stable_color = "#ecf0f1"
    confident_selected_color = "#7F7F7F"
    uncertainty_color = "#f1c40f"

    fp_99_color = "#e67e22"
    fp_90_color = "#d35400"
    fn_90_color = "#8e44ad"
    fn_99_color = "#c39bd3"

    # ------------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------------

    df_moe = df.copy()

    rep_cols = [
        f"{var_rep_prefix}{i}"
        for i in range(1, n_replicates + 1)
        if f"{var_rep_prefix}{i}" in df_moe.columns
    ]

    print(f"Found {len(rep_cols)} replicate columns")

    if len(rep_cols) == 0:
        raise ValueError(
            f"No replicate columns found with prefix '{var_rep_prefix}'."
        )

    if len(rep_cols) != n_replicates:
        print(
            f"Warning: expected {n_replicates} replicate columns "
            f"but found {len(rep_cols)}."
        )

    # Median and raw replicate range
    df_moe["median_rep"] = df_moe[rep_cols].median(axis=1)
    df_moe["min"] = df_moe[rep_cols].min(axis=1)
    df_moe["max"] = df_moe[rep_cols].max(axis=1)

    # ------------------------------------------------------------------
    # Census variance replicate formula
    # ------------------------------------------------------------------

    squared_differences = np.zeros(len(df_moe))

    for rep_col in rep_cols:
        squared_differences += (
            df_moe[rep_col] - df_moe[nominal_col]
        ) ** 2

    variance = (4 / n_replicates) * squared_differences

    # Margins of error
    df_moe["MOE_65"] = 0.93 * np.sqrt(variance)
    df_moe["MOE_90"] = 1.645 * np.sqrt(variance)
    df_moe["MOE_99"] = 2.576 * np.sqrt(variance)

    # Confidence intervals
    for level in [65, 90, 99]:
        df_moe[f"lower_{level}"] = (
            df_moe[nominal_col] - df_moe[f"MOE_{level}"]
        )
        df_moe[f"upper_{level}"] = (
            df_moe[nominal_col] + df_moe[f"MOE_{level}"]
        )

    required_columns = [
        nominal_col,
        "median_rep",
        "min",
        "max",
        "lower_65",
        "upper_65",
        "lower_90",
        "upper_90",
        "lower_99",
        "upper_99"
    ]

    df_clean = df_moe.dropna(subset=required_columns).copy()

    # ------------------------------------------------------------------
    # Scale from 0-1 to 0-100 for display
    # ------------------------------------------------------------------

    scale_factor = 100 if scale_to_100 else 1

    if scale_to_100:

        scale_columns = [
            nominal_col,
            "median_rep",
            "min",
            "max",
            "MOE_65",
            "MOE_90",
            "MOE_99",
            "lower_65",
            "upper_65",
            "lower_90",
            "upper_90",
            "lower_99",
            "upper_99"
        ] + rep_cols

        for col in scale_columns:
            if col in df_clean.columns:
                df_clean[col] *= scale_factor

    df_sorted = df_clean.sort_values(nominal_col)

    # ------------------------------------------------------------------
    # Control-point summaries used for PCHIP curves
    # ------------------------------------------------------------------

    control_percentiles = [0, 5, 25, 50, 75, 95, 100]

    control_rows = []

    for p in control_percentiles:

        # Preserve fixed endpoints
        if p == 0:
            control_rows.append(
                {
                    "x": 0,
                    "median": 0,
                    "lower_65": 0,
                    "upper_65": 0,
                    "lower_90": 0,
                    "upper_90": 0
                }
            )
            continue

        if p == 100:
            control_rows.append(
                {
                    "x": 100,
                    "median": 100,
                    "lower_65": 100,
                    "upper_65": 100,
                    "lower_90": 100,
                    "upper_90": 100
                }
            )
            continue

        # Original windows used in the manuscript workflow
        if p in [5, 95]:
            window_size = 0.001
        elif p == 50:
            window_size = 10
        else:
            window_size = 0.02

        window_data = df_sorted[
            (df_sorted[nominal_col] >= p - window_size)
            & (df_sorted[nominal_col] <= p + window_size)
        ]

        # Small demo datasets may not contain observations
        # in every narrow percentile window.
        if window_data.empty:
            print(
                f"Skipping {p}th percentile control point: "
                "no observations in window."
            )
            continue

        control_rows.append(
            {
                "x": p,
                "median": window_data["median_rep"].median(),
                "lower_65": window_data["lower_65"].median(),
                "upper_65": window_data["upper_65"].median(),
                "lower_90": window_data["lower_90"].median(),
                "upper_90": window_data["upper_90"].median()
            }
        )

    control_df = pd.DataFrame(control_rows).sort_values("x")

    if len(control_df) < 2:
        raise ValueError(
            "Insufficient control points to construct interpolation curves."
        )

    # ------------------------------------------------------------------
    # PCHIP approximation
    # ------------------------------------------------------------------

    x_smooth = np.linspace(0, 100, 500)

    def interpolate_control(column):
        interpolator = PchipInterpolator(
            control_df["x"],
            control_df[column]
        )

        return np.clip(
            interpolator(x_smooth),
            0,
            100
        )

    median_smooth = interpolate_control("median")
    lower_65_smooth = interpolate_control("lower_65")
    upper_65_smooth = interpolate_control("upper_65")
    lower_90_smooth = interpolate_control("lower_90")
    upper_90_smooth = interpolate_control("upper_90")

    # ------------------------------------------------------------------
    # Create plot
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=fig_size,
        facecolor="white"
    )

    ax.set_facecolor("white")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.set_xlabel(
        "Estimate Based Index Score",
        fontsize=12
    )

    ax.set_ylabel(
        "Variance Replicate Index Score",
        fontsize=12
    )

    ax.grid(
        True,
        linestyle="-",
        alpha=0.2,
        color="lightgray",
        zorder=0
    )

    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 10))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Raw replicate ranges
    # ------------------------------------------------------------------

    for _, row in df_sorted.iterrows():

        x = row[nominal_col]

        ax.plot(
            [x, x],
            [row["min"], row["max"]],
            color=raw_data_color,
            alpha=0.6,
            linewidth=1,
            zorder=1
        )

    if save_stages:

        os.makedirs(plot_dir, exist_ok=True)

        fig.savefig(
            os.path.join(
                plot_dir,
                f"{save_prefix}_1_raw_data.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

    # ------------------------------------------------------------------
    # Model approximation
    # ------------------------------------------------------------------

    ax.fill_between(
        x_smooth,
        lower_90_smooth,
        upper_90_smooth,
        color=dark_blue,
        alpha=0.4,
        zorder=2,
        label="90% prediction interval"
    )

    ax.fill_between(
        x_smooth,
        lower_65_smooth,
        upper_65_smooth,
        color=light_blue,
        alpha=0.5,
        zorder=3,
        label="65% prediction interval"
    )

    ax.plot(
        x_smooth,
        median_smooth,
        color=line_blue,
        linewidth=2,
        label="Median of replicates",
        zorder=5
    )

    # ------------------------------------------------------------------
    # Percentile error bars
    # ------------------------------------------------------------------

    for p in [5, 25, 50, 75, 95]:

        row = control_df[control_df["x"] == p]

        if row.empty:
            continue

        row = row.iloc[0]

        y_point = row["median"]
        y_lower = row["lower_90"]
        y_upper = row["upper_90"]

        delta = y_upper - y_lower

        ax.errorbar(
            p,
            y_point,
            yerr=[
                [y_point - y_lower],
                [y_upper - y_point]
            ],
            fmt="o",
            color=dark_grey,
            capsize=6,
            capthick=2,
            markersize=0,
            elinewidth=2,
            zorder=10
        )

        ax.annotate(
            f"{p}th\npercentile\nΔ = {delta:.0f}",
            xy=(p, y_point),
            xytext=(p, y_upper + 2),
            ha="center",
            va="bottom",
            color=dark_grey,
            fontsize=10,
            fontweight="600",
            zorder=11
        )

    # ------------------------------------------------------------------
    # Classification zones
    # ------------------------------------------------------------------

    threshold = reference_line * scale_factor
    x_boundary = threshold

    crossing_values = None

    if add_designation_regions:

        bins = np.arange(0, 101, 1)

        temp_df = df_sorted.copy()

        temp_df["bin"] = pd.cut(
            temp_df[nominal_col],
            bins=bins,
            labels=bins[:-1],
            include_lowest=True
        )

        bin_df = (
            temp_df
            .groupby("bin", observed=False)
            .agg(
                n=("GEOID", "count"),
                prob_65_up=(
                    "upper_65",
                    lambda x: (x > threshold).mean()
                ),
                prob_90_up=(
                    "upper_90",
                    lambda x: (x > threshold).mean()
                ),
                prob_99_up=(
                    "upper_99",
                    lambda x: (x > threshold).mean()
                ),
                prob_65_lo=(
                    "lower_65",
                    lambda x: (x < threshold).mean()
                ),
                prob_90_lo=(
                    "lower_90",
                    lambda x: (x < threshold).mean()
                ),
                prob_99_lo=(
                    "lower_99",
                    lambda x: (x < threshold).mean()
                )
            )
            .reset_index()
        )

        bin_df["bin"] = bin_df["bin"].astype(float)

        def crossing(colname, side):

            temp = bin_df[
                bin_df[colname] >= 0.5
            ]

            if temp.empty:
                return None

            if side == "left":
                return temp["bin"].min()

            return temp["bin"].max()

        crossing_values = {
            "upper_99": crossing("prob_99_up", "left"),
            "upper_90": crossing("prob_90_up", "left"),
            "upper_65": crossing("prob_65_up", "left"),
            "lower_65": crossing("prob_65_lo", "right"),
            "lower_90": crossing("prob_90_lo", "right"),
            "lower_99": crossing("prob_99_lo", "right")
        }

        # Only draw all classification regions when all boundaries exist.
        if all(
            value is not None
            for value in crossing_values.values()
        ):

            u99 = crossing_values["upper_99"]
            u90 = crossing_values["upper_90"]
            u65 = crossing_values["upper_65"]

            l65 = crossing_values["lower_65"]
            l90 = crossing_values["lower_90"]
            l99 = crossing_values["lower_99"]

            ax.axvspan(
                0,
                u99,
                color=stable_color,
                alpha=0.8,
                zorder=0
            )

            ax.axvspan(
                u99,
                u90,
                color=fn_99_color,
                alpha=0.6,
                zorder=0
            )

            ax.axvspan(
                u90,
                u65,
                color=fn_90_color,
                alpha=0.6,
                zorder=0
            )

            ax.axvspan(
                u65,
                x_boundary,
                color=uncertainty_color,
                alpha=0.7,
                zorder=0
            )

            ax.axvspan(
                x_boundary,
                l65,
                color=uncertainty_color,
                alpha=0.7,
                zorder=0
            )

            ax.axvspan(
                l65,
                l90,
                color=fp_90_color,
                alpha=0.6,
                zorder=0
            )

            ax.axvspan(
                l90,
                l99,
                color=fp_99_color,
                alpha=0.6,
                zorder=0
            )

            ax.axvspan(
                l99,
                100,
                color=confident_selected_color,
                alpha=0.4,
                zorder=1
            )

        else:

            print(
                "Classification regions were not drawn because "
                "the dataset does not contain sufficient observations "
                "to estimate all classification boundaries."
            )

    # ------------------------------------------------------------------
    # Threshold lines
    # ------------------------------------------------------------------

    ax.axvline(
        x=threshold,
        color="red",
        linestyle=":",
        linewidth=3,
        alpha=0.9,
        zorder=6
    )

    ax.axhline(
        y=threshold,
        color="red",
        linestyle=":",
        linewidth=3,
        alpha=0.9,
        zorder=6
    )

    # ------------------------------------------------------------------
    # Prediction interval intersections with policy threshold
    # ------------------------------------------------------------------

    def edge_x_for_y(y_val, lower, upper, x_vals):

        x_hits = [
            x
            for x, lo, hi
            in zip(x_vals, lower, upper)
            if lo <= y_val <= hi
        ]

        if not x_hits:
            return None, None

        return min(x_hits), max(x_hits)

    x65_left, x65_right = edge_x_for_y(
        threshold,
        lower_65_smooth,
        upper_65_smooth,
        x_smooth
    )

    x90_left, x90_right = edge_x_for_y(
        threshold,
        lower_90_smooth,
        upper_90_smooth,
        x_smooth
    )

    for x in [
        x65_left,
        x65_right,
        x90_left,
        x90_right
    ]:

        if x is not None:

            ax.plot(
                x,
                threshold,
                marker="o",
                color="red",
                markersize=6,
                zorder=12
            )

    # ------------------------------------------------------------------
    # Policy action bracket
    # ------------------------------------------------------------------

    x0 = threshold
    x1 = 100
    y_frac = -0.05

    bracket = FancyArrowPatch(
        (x0, y_frac),
        (x1, y_frac),
        arrowstyle="]-[",
        mutation_scale=3,
        lw=2,
        color="red",
        transform=ax.get_xaxis_transform(),
        clip_on=False
    )

    ax.add_patch(bracket)

    ax.text(
        (x0 + x1) / 2,
        y_frac - 0.02,
        "Policy Action",
        ha="center",
        va="top",
        fontsize=10,
        color="red",
        fontweight="bold",
        transform=ax.get_xaxis_transform(),
        clip_on=False
    )

    # ------------------------------------------------------------------
    # Legends
    # ------------------------------------------------------------------

    handles_model = [
        Line2D(
            [],
            [],
            color="none",
            label=r"$\bf{Model\ Approximation}$"
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            color=light_blue,
            alpha=0.5,
            label="65% prediction interval"
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            color=dark_blue,
            alpha=0.4,
            label="90% prediction interval"
        ),
        Line2D(
            [0],
            [0],
            color=line_blue,
            linewidth=3,
            label="Median of replicates"
        )
    ]

    fig.legend(
        handles_model,
        [h.get_label() for h in handles_model],
        loc="upper left",
        bbox_to_anchor=(0.04, 0),
        frameon=False,
        fontsize=10
    )

    # Only show classification-zone legend when zones were actually drawn
    if (
        crossing_values is not None
        and all(
            value is not None
            for value in crossing_values.values()
        )
    ):

        handles_class = [
            Line2D(
                [],
                [],
                color="none",
                label=r"$\bf{Classification\ Zones}$"
            ),
            Rectangle(
                (0, 0),
                1,
                1,
                color=stable_color,
                alpha=0.8,
                label="Confidently Excluded"
            ),
            Rectangle(
                (0, 0),
                1,
                1,
                color=confident_selected_color,
                alpha=0.8,
                label="Confidently Selected"
            ),
            Rectangle(
                (0, 0),
                1,
                1,
                color=uncertainty_color,
                alpha=0.7,
                label="Uncertainty zone (65% CI)"
            ),
            Line2D(
                [0],
                [0],
                color="red",
                linestyle=":",
                linewidth=2,
                label="Threshold (75)"
            )
        ]

        handles_uncertainty = [
            Line2D(
                [],
                [],
                color="none",
                label=r"$\bf{Threshold\ Uncertainty}$"
            ),
            Rectangle(
                (0, 0),
                1,
                1,
                color=fp_99_color,
                alpha=0.6,
                label="99% CI false positive"
            ),
            Rectangle(
                (0, 0),
                1,
                1,
                color=fp_90_color,
                alpha=0.6,
                label="90% CI false positive"
            ),
            Rectangle(
                (0, 0),
                1,
                1,
                color=fn_90_color,
                alpha=0.6,
                label="90% CI false negative"
            ),
            Rectangle(
                (0, 0),
                1,
                1,
                color=fn_99_color,
                alpha=0.6,
                label="99% CI false negative"
            )
        ]

        fig.legend(
            handles_class,
            [h.get_label() for h in handles_class],
            loc="upper left",
            bbox_to_anchor=(0.39, 0),
            frameon=False,
            fontsize=10
        )

        fig.legend(
            handles_uncertainty,
            [h.get_label() for h in handles_uncertainty],
            loc="upper left",
            bbox_to_anchor=(0.70, 0),
            frameon=False,
            fontsize=10
        )

    plt.tight_layout()

    if save_stages:

        os.makedirs(plot_dir, exist_ok=True)

        fig.savefig(
            os.path.join(
                plot_dir,
                f"{save_prefix}_complete_plot.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

    return fig, ax