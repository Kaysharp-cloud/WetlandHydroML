"""
Climate Analysis

"""

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# PROJECT PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Allows the script to be stored either in the project root or in a subfolder
# such as Scripts/.
if (SCRIPT_DIR / "Data").exists():
    PROJECT_ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "Data").exists():
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = DATA_DIR / "Output_data"
FIGURE_DIR = PROJECT_ROOT / "Figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# IMPORTS
# =============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# DF_ANNUAL_WS_GRP.TO_CSV(OUT_DIR / "DF_DAYMET_ANNUAL_WS_GRP.CSV", INDEX=FALSE)
# =============================================================================
# =============================================================================
DAYMET_PATH = r"..\Data\Raw_data\SEF_DAYMET_All_var.csv"
LOCATION_PATH = r"..\Data\Raw_data\Well_id_location.csv"

OUT_DIR = Path(r"..\Figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WATERSHED_COLORS = {
    "WS77": "blue",
    "WS78": "#F58518",
    "WS80": "red",
}


# =============================================================================
def save_figure(fig, filename_base, out_dir=OUT_DIR, dpi=300):
    """Save a figure as PNG and EPS."""
    png_path = out_dir / f"{filename_base}.png"
    eps_path = out_dir / f"{filename_base}.eps"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(eps_path, dpi=dpi, bbox_inches="tight", format="eps")

    print(f"Saved: {png_path}")
    print(f"Saved: {eps_path}")


def prepare_daymet_data(daymet_path, location_path, start_year=2004):
    """Load Daymet and well-location data, then aggregate annually and monthly by watershed."""
    df_daymet = pd.read_csv(daymet_path)
    df_loc = pd.read_csv(location_path)

    df_loc = df_loc.drop(columns=["Type_", "POINT_X", "POINT_Y"], errors="ignore")
    df_loc = df_loc.rename(columns={"Station_Na": "ID"})
    df_loc["Watershed"] = df_loc["Watershed"].replace({"TC": "WS78"})

    df_daymet["date"] = pd.to_datetime(df_daymet["date"])
    df_daymet["Year"] = df_daymet["date"].dt.year
    df_daymet["Month"] = df_daymet["date"].dt.month

    df_daymet = df_daymet[df_daymet["Year"] >= start_year].copy()

    df_annual = (
        df_daymet.groupby(["Year", "ID"], as_index=False)
        .agg(prcp=("prcp", "sum"), tmax=("tmax", "mean"), tmin=("tmin", "mean"))
    )

    df_monthly = (
        df_daymet.groupby(["Year", "Month", "ID"], as_index=False)
        .agg(prcp=("prcp", "sum"), tmax=("tmax", "mean"), tmin=("tmin", "mean"))
    )

    df_monthly_longterm = (
        df_monthly.groupby(["Month", "ID"], as_index=False)
        .agg(prcp=("prcp", "mean"), tmax=("tmax", "mean"), tmin=("tmin", "mean"))
    )

    df_annual_ws = df_annual.merge(df_loc, on="ID", how="left")
    df_monthly_ws = df_monthly_longterm.merge(df_loc, on="ID", how="left")

    df_annual_ws_grp = (
        df_annual_ws.groupby(["Year", "Watershed"], as_index=False)[["prcp", "tmax", "tmin"]]
        .mean()
    )

    df_monthly_ws_grp = (
        df_monthly_ws.groupby(["Month", "Watershed"], as_index=False)[["prcp", "tmax", "tmin"]]
        .mean()
    )

    # df_annual_ws_grp.to_csv(OUT_DIR / "df_daymet_annual_ws_grp.csv", index=False)
    # df_monthly_ws_grp.to_csv(OUT_DIR / "df_daymet_monthly_ws_grp.csv", index=False)

    return df_annual_ws_grp, df_monthly_ws_grp


def plot_grouped_bars(
    ax,
    df,
    x_col,
    y_col,
    ylabel,
    ylim=None,
    watershed_order=("WS77", "WS78", "WS80"),
    colors=WATERSHED_COLORS,
    rotate_xticks=0,
    label_fontsize=16,
    tick_fontsize=14,
    legend_fontsize=12,
    show_legend=True,
):
    """Draw grouped bars on a provided axis."""
    pivot_df = (
        df.pivot_table(index=x_col, columns="Watershed", values=y_col, aggfunc="mean")
        .reindex(columns=watershed_order)
        .sort_index()
    )

    color_list = [colors[w] for w in watershed_order if w in pivot_df.columns]

    pivot_df.plot(kind="bar", ax=ax, color=color_list, width=0.8)

    ax.set_xlabel(x_col, fontsize=label_fontsize, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight="bold")

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.tick_params(axis="x", rotation=rotate_xticks, labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        legend = ax.legend(
            title="Watershed",
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            ncols=3,
            frameon=False,
        )
        legend.get_title().set_fontweight("bold")
        for text in legend.get_texts():
            text.set_fontweight("bold")
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    return ax


def create_individual_figure(df, x_col, y_col, ylabel, ylim, filename, rotate_xticks=0):
    """Create and save one grouped-bar figure."""
    fig, ax = plt.subplots(figsize=(14, 6))

    plot_grouped_bars(
        ax=ax,
        df=df,
        x_col=x_col,
        y_col=y_col,
        ylabel=ylabel,
        ylim=ylim,
        rotate_xticks=rotate_xticks,
        label_fontsize=26,
        tick_fontsize=22,
        legend_fontsize=20,
        show_legend=True,
    )

    fig.tight_layout()
    save_figure(fig, filename)
    plt.close(fig)


def create_climate_panel(df_monthly, df_annual):
    """Create the 3 x 2 climate panel directly from grouped-bar plots."""
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))

    panel_specs = [
        ("prcp", "Precipitation (mm)", (50, 250), df_monthly, "Month", 0),
        ("prcp", "Precipitation (mm)", (600, 2200), df_annual, "Year", 90),
        ("tmax", "Maximum\n temperature (°C)", (10, 35), df_monthly, "Month", 0),
        ("tmax", "Maximum\n temperature (°C)", (23, 26.5), df_annual, "Year", 90),
        ("tmin", "Minimum\n temperature (°C)", (0, 25), df_monthly, "Month", 0),
        ("tmin", "Minimum\n temperature (°C)", (11.5, 14), df_annual, "Year", 90),
    ]

    panel_labels = ["a)", "b)", "c)", "d)", "e)", "f)"]

    for ax, spec, label in zip(axes.flatten(), panel_specs, panel_labels):
        y_col, ylabel, ylim, df, x_col, rotation = spec

        plot_grouped_bars(
            ax=ax,
            df=df,
            x_col=x_col,
            y_col=y_col,
            ylabel=ylabel,
            ylim=ylim,
            rotate_xticks=rotation,
            label_fontsize=18,
            tick_fontsize=14,
            legend_fontsize=12,
            show_legend=False,
        )

        ax.text(
            -0.02,
            1.08,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=18,
            fontweight="bold",
        )

    axes[0, 0].set_title("Monthly", fontsize=20, fontweight="bold")
    axes[0, 1].set_title("Time series", fontsize=20, fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Watershed",
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=16,
        title_fontsize=16,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(fig, "SC_panel_prcp_tmax_tmin")
    plt.close(fig)


# =============================================================================
df_daymet_annual_ws_grp, df_daymet_monthly_ws_grp = prepare_daymet_data(
    DAYMET_PATH,
    LOCATION_PATH,
    start_year=2004,
)


# Direct 3 x 2 panel figure
create_climate_panel(
    df_monthly=df_daymet_monthly_ws_grp,
    df_annual=df_daymet_annual_ws_grp,
)
