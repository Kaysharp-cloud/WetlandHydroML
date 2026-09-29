"""
Analysis of Reconstructed Water Table Depth (WTD)

Post-processing workflow for reconstructed daily WTD at the Santee
Experimental Forest. The script performs:

- data loading and site metadata integration
- annual observed/predicted trend analysis
- daily and annual modified Mann-Kendall trend analysis
- observed-versus-predicted performance plots
- wetland-hydrology classification
- 14-consecutive-day wetland persistence analysis
- wetland persistence comparison by decade
- WTD time-series visualization
- summary-table preparation


"""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk

import data_loader as dl
import WTD_functions as wf

warnings.filterwarnings("ignore")


# =============================================================================
# PROJECT PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Supports placing this script either in the project root or in a subfolder
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
# LOAD RECONSTRUCTED WTD DATA
# =============================================================================

recon_output_path = OUTPUT_DIR / "df_reconstructed_wtd.csv"
legacy_output_path = OUTPUT_DIR / "df_final_output.csv"

if recon_output_path.exists():
    df_output = pd.read_csv(recon_output_path)
elif legacy_output_path.exists():
    df_output = pd.read_csv(legacy_output_path)
else:
    raise FileNotFoundError(
        "Neither reconstructed WTD file was found. Checked:\n"
        f"{recon_output_path}\n{legacy_output_path}"
    )

if "WTD_Predicted" not in df_output.columns and "reconstructed_WTD" in df_output.columns:
    df_output["WTD_Predicted"] = df_output["reconstructed_WTD"]

if "WTD" not in df_output.columns and "observed_WTD" in df_output.columns:
    df_output["WTD"] = df_output["observed_WTD"]

if "date" in df_output.columns:
    df_output["date"] = pd.to_datetime(df_output["date"], errors="coerce")

if "Year" not in df_output.columns and "date" in df_output.columns:
    df_output["Year"] = df_output["date"].dt.year

if "Month" not in df_output.columns and "date" in df_output.columns:
    df_output["Month"] = df_output["date"].dt.month



# =============================================================================
# PREPARE SITE METADATA AND MERGE WITH RECONSTRUCTED WTD
# =============================================================================


df_loc = dl.df_sef_info.copy()
df_loc.rename(columns={"Station_Na":"ID","Watershed":"Location"}, inplace=True)
df_loc = df_loc[["ID","Location","Name"]]
df_loc.Location = df_loc.Location.str.replace("TC","WS78")

df_loc.ID = df_loc.ID.str.replace("well","Well", regex = False)

df = df_output.merge(df_loc, on="ID", how="left")



# =============================================================================
# PREPARE OBSERVED-DATE ANNUAL WTD DATA
# =============================================================================


df_filtered_wtd = df.dropna(subset="WTD")

df_filtered_wtd_annual = df_filtered_wtd.groupby(["ID","Location","Year"])[["WTD","WTD_Predicted"]].mean().reset_index()



# =============================================================================
# MODIFIED MANN-KENDALL TREND ANALYSIS ON COMMON OBSERVED DATES
# =============================================================================


def run_modified_mk_on_series(series):
    """
    Run modified Mann-Kendall test on a 1D numeric series.

    Parameters
    ----------
    series : array-like
        Numeric time series

    Returns
    -------
    dict
        Dictionary of modified MK results
    """
    x = pd.Series(series).dropna().astype(float).values

    if len(x) < 3:
        return {
            "trend": np.nan,
            "h": np.nan,
            "p": np.nan,
            "z": np.nan,
            "Tau": np.nan,
            "s": np.nan,
            "var_s": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "n": len(x)
        }

    result = mk.hamed_rao_modification_test(x)

    return {
        "trend": result.trend,
        "h": result.h,
        "p": result.p,
        "z": result.z,
        "Tau": result.Tau,
        "s": result.s,
        "var_s": result.var_s,
        "slope": result.slope,
        "intercept": result.intercept,
        "n": len(x)
    }


def run_trend_for_one_id_one_variable(df_id, value_col, year_col="Year"):
    """
    Sort one ID by year and run modified MK trend test for one variable.

    Parameters
    ----------
    df_id : pd.DataFrame
        Sub-dataframe for one ID
    value_col : str
        Column to analyze, e.g. 'WTD' or 'WTD_Predicted'
    year_col : str
        Year column name

    Returns
    -------
    dict
        Trend result dictionary
    """
    df_id_sorted = df_id.sort_values(year_col).copy()
    result_dict = run_modified_mk_on_series(df_id_sorted[value_col])

    result_dict["start_year"] = df_id_sorted[year_col].min() if len(df_id_sorted) else np.nan
    result_dict["end_year"] = df_id_sorted[year_col].max() if len(df_id_sorted) else np.nan

    return result_dict


def modified_trend_analysis_by_id(
    df,
    id_col="ID",
    location_col="Location",
    year_col="Year",
    value_cols=("WTD", "WTD_Predicted")
):
    """
    Run modified Mann-Kendall trend analysis for each ID and each variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing ID, Year, WTD, and WTD_Predicted
    id_col : str
        ID column name
    location_col : str
        Location column name
    year_col : str
        Year column name
    value_cols : tuple/list
        Variables to test

    Returns
    -------
    pd.DataFrame
        Trend results for all IDs and all requested variables
    """
    results = []

    for one_id, df_id in df.groupby(id_col):
        df_id = df_id.sort_values(year_col).copy()

        if location_col in df_id.columns:
            loc_vals = df_id[location_col].dropna().unique()
            location = loc_vals[0] if len(loc_vals) > 0 else np.nan
        else:
            location = np.nan

        for value_col in value_cols:
            if value_col not in df_id.columns:
                continue

            result_dict = run_trend_for_one_id_one_variable(
                df_id=df_id,
                value_col=value_col,
                year_col=year_col
            )

            result_dict[id_col] = one_id
            result_dict[location_col] = location
            result_dict["variable"] = value_col

            results.append(result_dict)

    results_df = pd.DataFrame(results)

    desired_order = [
        id_col, location_col, "variable",
        "start_year", "end_year", "n",
        "trend", "h", "p", "z", "Tau", "s", "var_s", "slope", "intercept"
    ]

    existing_cols = [c for c in desired_order if c in results_df.columns]
    other_cols = [c for c in results_df.columns if c not in existing_cols]

    results_df = results_df[existing_cols + other_cols].copy()

    return results_df

df_trend_results = modified_trend_analysis_by_id(
    df=df_filtered_wtd_annual,
    id_col="ID",
    location_col="Location",
    year_col="Year",
    value_cols=("WTD", "WTD_Predicted")
)

cols_2dp = ['h', 'p', 'z', 'Tau', 's', 'var_s', 'slope', 'intercept']

df_trend_results[cols_2dp] = df_trend_results[cols_2dp].round(2)



# =============================================================================
# DAILY MODIFIED MANN-KENDALL TREND ANALYSIS
# =============================================================================


def run_modified_mk_series(series):
    """
    Run modified Mann-Kendall test on a 1D numeric series.
    """
    x = pd.Series(series).dropna().astype(float).values

    if len(x) < 3:
        return {
            "trend": np.nan,
            "h": np.nan,
            "p": np.nan,
            "z": np.nan,
            "Tau": np.nan,
            "s": np.nan,
            "var_s": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "n": len(x)
        }

    result = mk.hamed_rao_modification_test(x)

    return {
        "trend": result.trend,
        "h": result.h,
        "p": result.p,
        "z": result.z,
        "Tau": result.Tau,
        "s": result.s,
        "var_s": result.var_s,
        "slope": result.slope,
        "intercept": result.intercept,
        "n": len(x)
    }


def run_daily_predicted_trend_one_id(
    df_id,
    value_col="WTD_Predicted",
    date_col="date"
):
    """
    Sort one ID by date and run modified MK on WTD_Predicted.
    """
    df_id = df_id.copy()
    df_id[date_col] = pd.to_datetime(df_id[date_col], errors="coerce")
    df_id = df_id.sort_values(date_col)

    result = run_modified_mk_series(df_id[value_col])

    result["start_date"] = df_id[date_col].min()
    result["end_date"] = df_id[date_col].max()

    return result


def modified_daily_trend_predicted_by_id(
    df,
    id_col="ID",
    location_col="Location",
    name_col="Name",
    date_col="date",
    value_col="WTD_Predicted",
    round_cols=True
):
    """
    Run modified Mann-Kendall daily trend analysis for WTD_Predicted by ID.

    Returns
    -------
    pd.DataFrame
        Trend results for all IDs
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    results = []

    for one_id, df_id in df.groupby(id_col):
        df_id = df_id.sort_values(date_col).copy()

        if location_col in df_id.columns:
            loc_vals = df_id[location_col].dropna().unique()
            location = loc_vals[0] if len(loc_vals) > 0 else np.nan
        else:
            location = np.nan

        if name_col in df_id.columns:
            name_vals = df_id[name_col].dropna().unique()
            name = name_vals[0] if len(name_vals) > 0 else np.nan
        else:
            name = np.nan

        result = run_daily_predicted_trend_one_id(
            df_id=df_id,
            value_col=value_col,
            date_col=date_col
        )

        result[id_col] = one_id
        result[location_col] = location
        result[name_col] = name
        result["variable"] = value_col

        results.append(result)

    results_df = pd.DataFrame(results)

    desired_order = [
        id_col, location_col, name_col, "variable",
        "start_date", "end_date", "n",
        "trend", "h", "p", "z", "Tau", "s", "var_s", "slope", "intercept"
    ]

    existing_cols = [c for c in desired_order if c in results_df.columns]
    other_cols = [c for c in results_df.columns if c not in existing_cols]

    results_df = results_df[existing_cols + other_cols].copy()

    if round_cols:
        cols_2dp = ['h', 'p', 'z', 'Tau', 's', 'var_s', 'slope', 'intercept']
        cols_2dp_existing = [col for col in cols_2dp if col in results_df.columns]
        results_df[cols_2dp_existing] = results_df[cols_2dp_existing].round(2)

    return results_df

df_daily_trend_pred = modified_daily_trend_predicted_by_id(
    df=df,
    id_col="ID",
    location_col="Location",
    name_col="Name",
    date_col="date",
    value_col="WTD_Predicted",
    round_cols=True
)

df_daily_trend_pred.to_csv(OUTPUT_DIR / "df_daily_trend.csv", index=False)



# =============================================================================
# ANNUAL TREND ANALYSIS OF RECONSTRUCTED WTD
# =============================================================================




def daily_to_annual_by_id(
    df,
    id_col="ID",
    location_col="Location",
    name_col="Name",
    date_col="date",
    value_col="WTD_Predicted",
    agg_func="mean"
):
    """
    Convert daily data to annual values for each ID.

    Parameters
    ----------
    df : pd.DataFrame
        Daily dataframe
    agg_func : str
        Aggregation for annual series, e.g. 'mean', 'median', 'max', 'min'

    Returns
    -------
    pd.DataFrame
        Annual dataframe
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["Year"] = df[date_col].dt.year

    group_cols = [id_col, "Year"]
    keep_cols = [c for c in [location_col, name_col] if c in df.columns]

    agg_dict = {value_col: agg_func}
    for c in keep_cols:
        agg_dict[c] = "first"

    df_annual = (
        df.groupby(group_cols, as_index=False)
          .agg(agg_dict)
    )

    ordered_cols = [id_col] + keep_cols + ["Year", value_col]
    df_annual = df_annual[ordered_cols].copy()

    return df_annual


def run_annual_predicted_trend_one_id(
    df_id,
    year_col="Year",
    value_col="WTD_Predicted"
):
    """
    Sort one ID by year and run modified MK on annual WTD_Predicted.
    """
    df_id = df_id.copy()
    df_id = df_id.sort_values(year_col)

    result = run_modified_mk_series(df_id[value_col])

    result["start_year"] = df_id[year_col].min()
    result["end_year"] = df_id[year_col].max()

    return result


def modified_annual_trend_predicted_by_id(
    df_annual,
    id_col="ID",
    location_col="Location",
    name_col="Name",
    year_col="Year",
    value_col="WTD_Predicted",
    round_cols=True
):
    """
    Run modified Mann-Kendall annual trend analysis for WTD_Predicted by ID.

    Returns
    -------
    pd.DataFrame
        Annual trend results for all IDs
    """
    results = []

    for one_id, df_id in df_annual.groupby(id_col):
        df_id = df_id.sort_values(year_col).copy()

        if location_col in df_id.columns:
            loc_vals = df_id[location_col].dropna().unique()
            location = loc_vals[0] if len(loc_vals) > 0 else np.nan
        else:
            location = np.nan

        if name_col in df_id.columns:
            name_vals = df_id[name_col].dropna().unique()
            name = name_vals[0] if len(name_vals) > 0 else np.nan
        else:
            name = np.nan

        result = run_annual_predicted_trend_one_id(
            df_id=df_id,
            year_col=year_col,
            value_col=value_col
        )

        result[id_col] = one_id
        result[location_col] = location
        result[name_col] = name
        result["variable"] = value_col

        results.append(result)

    results_df = pd.DataFrame(results)

    desired_order = [
        id_col, location_col, name_col, "variable",
        "start_year", "end_year", "n",
        "trend", "h", "p", "z", "Tau", "s", "var_s", "slope", "intercept"
    ]

    existing_cols = [c for c in desired_order if c in results_df.columns]
    other_cols = [c for c in results_df.columns if c not in existing_cols]

    results_df = results_df[existing_cols + other_cols].copy()

    if round_cols:
        cols_2dp = ['h', 'p', 'z', 'Tau', 's', 'var_s', 'slope', 'intercept']
        cols_2dp_existing = [col for col in cols_2dp if col in results_df.columns]
        results_df[cols_2dp_existing] = results_df[cols_2dp_existing].round(2)

    return results_df

# 1. Convert daily predicted WTD to annual mean predicted WTD
df_annual_pred = daily_to_annual_by_id(
    df=df,
    id_col="ID",
    location_col="Location",
    name_col="Name",
    date_col="date",
    value_col="WTD_Predicted",
    agg_func="mean"
)


# 2. Run modified MK trend on the annual series
df_annual_trend_pred = modified_annual_trend_predicted_by_id(
    df_annual=df_annual_pred,
    id_col="ID",
    location_col="Location",
    name_col="Name",
    year_col="Year",
    value_col="WTD_Predicted",
    round_cols=True
)



# =============================================================================
# OBSERVED VERSUS PREDICTED 1:1 PLOTS
# =============================================================================


def calc_metrics(obs, pred):
    obs = np.asarray(obs).ravel()
    pred = np.asarray(pred).ravel()

    mask = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[mask]
    pred = pred[mask]

    if len(obs) == 0:
        return {
            "RMSE": np.nan,
            "MAE": np.nan,
            "KGE": np.nan,
            "NSE": np.nan,
            "n": 0
        }

    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    mae = np.mean(np.abs(pred - obs))

    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    nse = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    std_obs = np.std(obs)
    std_pred = np.std(pred)
    mean_obs = np.mean(obs)
    mean_pred = np.mean(pred)

    if len(obs) < 2 or std_obs == 0 or mean_obs == 0:
        kge = np.nan
    else:
        r = np.corrcoef(obs, pred)[0, 1]
        alpha = std_pred / std_obs
        beta = mean_pred / mean_obs
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "KGE": kge,
        "NSE": nse,
        "n": len(obs)
    }


def plot_one_to_one_single_id(
    df_id,
    id_value,
    obs_col="WTD",
    pred_col="WTD_Predicted",
    outdir=FIGURE_DIR,
    dpi=600,
    fontsize=18,
    alpha=0.7
):
    df_id = df_id.copy()
    df_id = df_id[[obs_col, pred_col]].dropna()

    metrics = calc_metrics(df_id[obs_col], df_id[pred_col])

    obs = df_id[obs_col].values
    pred = df_id[pred_col].values

    if len(obs) == 0:
        print(f"Skipping {id_value}: no valid data.")
        return

    xy_min = min(np.min(obs), np.min(pred))
    xy_max = max(np.max(obs), np.max(pred))

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(obs, pred, s=22, alpha=alpha)

    # 1:1 line
    ax.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--", linewidth=2)

    ax.set_xlabel("Observed WTD", fontsize=fontsize, fontweight="bold")
    ax.set_ylabel("Predicted WTD", fontsize=fontsize, fontweight="bold")
    ax.set_title(f"1:1 Plot for {id_value}", fontsize=fontsize, fontweight="bold")

    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

    ax.tick_params(axis="both", labelsize=fontsize - 2, width=1.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    text_str = (
        f"n = {metrics['n']}\n"
        f"RMSE = {metrics['RMSE']:.2f}\n"
        f"MAE = {metrics['MAE']:.2f}\n"
        f"KGE = {metrics['KGE']:.2f}\n"
        f"NSE = {metrics['NSE']:.2f}"
    )

    ax.text(
        0.05, 0.95, text_str,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fontsize - 3,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )

    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    Path(outdir).mkdir(parents=True, exist_ok=True)
    safe_id = str(id_value).replace("/", "-").replace("\\", "-").replace(" ", "_")
    outfile = Path(outdir) / f"one_to_one_{safe_id}.png"
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    # plt.show()
    plt.close(fig)


def plot_one_to_one_by_id(
    df,
    id_col="ID",
    obs_col="WTD",
    pred_col="WTD_Predicted",
    outdir=FIGURE_DIR,
    dpi=600,
    fontsize=18,
    alpha=0.7
):
    df = df.copy()

    for one_id in df[id_col].dropna().unique():
        df_id = df[df[id_col] == one_id].copy()

        plot_one_to_one_single_id(
            df_id=df_id,
            id_value=one_id,
            obs_col=obs_col,
            pred_col=pred_col,
            outdir=outdir,
            dpi=dpi,
            fontsize=fontsize,
            alpha=alpha
        )

    print(f"All figures saved in: {outdir}")

plot_one_to_one_by_id(
    df=df,
    id_col="ID",
    obs_col="WTD",
    pred_col="WTD_Predicted",
    outdir="../Figures"
)



# =============================================================================
# MULTI-PANEL OBSERVED VERSUS PREDICTED 1:1 PLOT
# =============================================================================


def plot_one_to_one_by_id_panel(
    df1,
    id_col="ID",
    obs_col="WTD",
    pred_col="WTD_Predicted",
    outdir=FIGURE_DIR,
    filename="one_to_one_panel.png",
    ncols=3,
    dpi=600,
    fontsize=16,
    alpha=0.7,
    panel_width=6,
    panel_height=6
):
    """
    Create a single multi-panel 1:1 plot for all IDs and save it.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing ID, observed WTD, and predicted WTD
    id_col : str
        ID column name
    obs_col : str
        Observed column name
    pred_col : str
        Predicted column name
    outdir : str
        Output folder
    filename : str
        Saved figure name
    ncols : int
        Number of subplot columns
    dpi : int
        Figure dpi
    fontsize : int
        Base font size
    alpha : float
        Scatter transparency
    panel_width : int or float
        Width of each subplot
    panel_height : int or float
        Height of each subplot
    """

    df = df1.copy()
    df.sort_values(["Location","ID"], inplace = True)
    ids = df[id_col].dropna().unique()
    n_ids = len(ids)
    nrows = math.ceil(n_ids / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(panel_width * ncols, panel_height * nrows)
    )

    if n_ids == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(-1)

    for i, one_id in enumerate(ids):
        ax = axes[i]
        df_id = df[df[id_col] == one_id].copy()
        loc = df_id.Location.unique()[0]
        df_id = df_id[[obs_col, pred_col]].dropna()

        metrics = calc_metrics(df_id[obs_col], df_id[pred_col])

        obs = df_id[obs_col].values
        pred = df_id[pred_col].values

        if len(obs) == 0:
            ax.set_visible(False)
            continue

        xy_min = min(np.min(obs), np.min(pred))
        xy_max = max(np.max(obs), np.max(pred))

        ax.scatter(obs, pred, s=18, alpha=alpha, color = "blue")
        ax.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--", linewidth=1.8, color= "red")

        ax.set_xlim(xy_min, xy_max)
        ax.set_ylim(xy_min, xy_max)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(f"{one_id} ({loc})", fontsize=fontsize, fontweight="bold")
        ax.set_xlabel("Observed WTD", fontsize=fontsize - 1, fontweight="bold")
        ax.set_ylabel("Predicted WTD", fontsize=fontsize - 1, fontweight="bold")

        ax.tick_params(axis="both", labelsize=fontsize - 2, width=1.2)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        text_str = (
            f"n={metrics['n']}\n"
            f"RMSE={metrics['RMSE']:.2f}\n"
            f"MAE={metrics['MAE']:.2f}\n"
            f"KGE={metrics['KGE']:.2f}\n"
            f"NSE={metrics['NSE']:.2f}"
        )

        ax.text(
            0.05, 0.95, text_str,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=fontsize - 3,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8)
        )

    # Hide unused axes
    for j in range(n_ids, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()

    Path(outdir).mkdir(parents=True, exist_ok=True)
    outfile = Path(outdir) / filename
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    #plt.show()
    plt.close(fig)

    print(f"Saved panel figure to: {outfile}")

plot_one_to_one_by_id_panel(
    df1=df,
    id_col="ID",
    obs_col="WTD",
    pred_col="WTD_Predicted",
    outdir=FIGURE_DIR,
    filename="daily_wtd_one_to_one_panel.png",
    ncols=3
)



# =============================================================================
# WTD TIME-SERIES PERFORMANCE METRICS AND PLOTS
# =============================================================================


def _calc_metrics(obs, pred):
    """Return RMSE, MAE, NSE, KGE computed on 1D arrays."""
    obs = np.asarray(obs).ravel()
    pred = np.asarray(pred).ravel()
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]
    if obs.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    mae = np.mean(np.abs(pred - obs))
    denom = np.sum((obs - obs.mean()) ** 2)
    nse = 1.0 - np.sum((obs - pred) ** 2) / denom if denom > 0 else np.nan

    std_o, std_p = np.std(obs), np.std(pred)
    if std_o == 0 or std_p == 0 or obs.size < 2:
        r = np.nan
    else:
        r = np.corrcoef(obs, pred)[0, 1]
    alpha = (std_p / std_o) if std_o != 0 else np.nan
    beta = (np.mean(pred) / np.mean(obs)) if np.mean(obs) != 0 else np.nan
    if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta):
        kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    else:
        kge = np.nan

    return rmse, mae, nse, kge

def plot_wtd_timeseries_by_id(
    df_results: pd.DataFrame,
    df_check: pd.DataFrame,
    outdir: str = "./Figures",
    fontsize: int = 28,
    dpi: int = 600,
    transparent: bool = True,
    legend_loc=None,  # set to None for "best", or pass e.g. "upper left"
    ylim=None,        # <— NEW: pass [ymin, ymax] or None for automatic
    ref_line_y=-30,                   # NEW: horizontal reference line value
    ref_line_color="#4f4f4f",         # NEW: elegant dark gray-blue
    ref_line_style="--",              # NEW
    ref_line_width=2.2                # NEW
):
    """
    Plots Predicted vs Observed WTD time series for each ID.
    - Transparent PNGs
    - Thicker axis lines
    - Fanciful color palette
    - Bold tick labels and adjustable font size
    - Optional fixed y-limits via `ylim=[ymin, ymax]`
    """
    df_r = df_results.copy()
    df_c = df_check.copy()
    df_r["Date"] = pd.to_datetime(df_r["Date"], errors="coerce")
    df_c["Date"] = pd.to_datetime(df_c["Date"], errors="coerce")

    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Fanciful, modern color palette
    colors = {
        "pred": "#ff6f61",   # coral red
        "obs": "blue",       # deep teal
    }

    def _bold_ticks(ax, size):
        ax.tick_params(axis="both", which="both", labelsize=size, width=2)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")

    for ids in df_r.ID.dropna().unique():
        df_each = df_r[df_r.ID == ids].copy()
        df_each_original = df_c[df_c.ID == ids].copy()

        # Align on common dates
        overlap = (
            df_each[["Date", "WTD_Predicted"]]
            .merge(df_each_original[["Date", "WTD"]], on="Date", how="inner")
            .dropna(subset=["WTD_Predicted", "WTD"])
            .sort_values("Date")
        )

        rmse, mae, nse, kge = _calc_metrics(overlap["WTD"], overlap["WTD_Predicted"])
        loc_vals = df_each["Location"].dropna().unique()
        loc = loc_vals[0] if len(loc_vals) else "Unknown"

        fig, ax = plt.subplots(figsize=(20, 12))
        ax.plot(
            df_each["Date"], df_each["WTD_Predicted"],
            color=colors["pred"], linewidth=2.5, alpha=0.8, label="Predicted WTD"
        )
        ax.scatter(
            df_each_original["Date"], df_each_original["WTD"],
            color=colors["obs"], s=10, alpha=0.8, label="Observed WTD"
        )
        ax.axhline(
            y=ref_line_y,
            color=ref_line_color,
            linestyle=ref_line_style,
            linewidth=ref_line_width,
            alpha=0.9,
            label=f"WL reference"
        )
        # Title and labels
        ax.set_xlabel("Date", fontsize=fontsize, fontweight="bold")
        ax.set_ylabel("Water table depth (cm)", fontsize=fontsize, fontweight="bold")
        ax.set_title(
            f"WTD — {ids} ({loc}) | RMSE={rmse:.3f}, MAE={mae:.3f}, NSE={nse:.3f}, KGE={kge:.3f}",
            fontsize=fontsize, fontweight="bold", pad=15
        )

        # Optional fixed y-limits
        if isinstance(ylim, (list, tuple)) and len(ylim) == 2:
            ax.set_ylim(ylim[0], ylim[1])

        # Style: thicker axis lines and no top/right spines
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Bold ticks
        _bold_ticks(ax, fontsize)

        # Legend (uses specified loc or falls back to "best")
        leg = ax.legend(frameon=False, fontsize=fontsize, ncol=3,
                        loc=("best" if legend_loc is None else legend_loc))
        for txt in leg.get_texts():
            txt.set_fontweight("bold")

        fig.tight_layout()

        # Save transparent PNG
        safe_id = str(ids).replace("/", "-").replace("\\", "-").replace(" ", "_")
        safe_loc = str(loc).replace("/", "-").replace("\\", "-").replace(" ", "_")
        outfile = Path(outdir) / f"WTD_timeseries_{safe_id}_{safe_loc}.png"
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight", transparent=transparent)
        # plt.show()
        plt.close(fig)

    print(f"✅ All transparent figures saved to: {outdir}")



# =============================================================================
# ORIGINAL WETLAND-HYDROLOGY CLASSIFICATION
# =============================================================================


df.date = pd.to_datetime(df.date)
df["Date"] = df.date
df["Month"] = df.date.dt.month
df_new = wf.apply_wetland_flag(df)
df_new.date = pd.to_datetime(df_new.date)
df_new["Date"] = df_new.date
df_new["Month"] = df_new.date.dt.month
df_summary_ = wf.classify_wetland_years(df)
df_counts = wf.count_wetland_years(df_summary_)

df_results = df_new.copy()
df_results_D = df_results[df_results.ID=='Well D']
df_check = df_results.copy()
df_check_D = df_check[df_check.ID=='Well D']

# force a specific corner
#wf.plot_wtd_timeseries_by_id(df_results_D, df_check_D, legend_loc="upper left",ylim=[-290, 45])
plot_wtd_timeseries_by_id(df_results_D, df_check_D, legend_loc="upper left",ylim=[-290, 45])



# =============================================================================
# 14-CONSECUTIVE-DAY WETLAND-HYDROLOGY CLASSIFICATION
# =============================================================================


def has_14_consecutive_days(cond_series, min_days=14):
    """
    Check whether True occurs for at least min_days consecutively.
    """

    cond_series = cond_series.fillna(False).astype(bool)

    run_id = (cond_series != cond_series.shift()).cumsum()

    max_run = (
        cond_series.groupby(run_id)
        .sum()
        .max()
    )

    return max_run >= min_days


def is_wetland_year_14days(df_year):
    """
    Determine if a given ID-Year qualifies as a wetland year.

    Condition:
    WTD_Predicted is within 30 cm of the surface for at least
    14 consecutive days during the growing season, April to October.
    """

    df_year = df_year.copy()

    df_year["Date"] = pd.to_datetime(df_year["Date"])
    df_year = df_year.sort_values("Date")

    year = df_year["Year"].iloc[0]

    # Growing season calendar: April 1 to October 31
    gs_start = pd.Timestamp(year=year, month=4, day=1)
    gs_end = pd.Timestamp(year=year, month=10, day=31)
    full_dates = pd.date_range(gs_start, gs_end, freq="D")

    # Keep only growing season
    df_gs = df_year[(df_year["Date"] >= gs_start) & (df_year["Date"] <= gs_end)]

    if df_gs.empty:
        return False

    # Reindex to daily calendar so missing days do not falsely connect runs
    df_gs = (
        df_gs.set_index("Date")
        .reindex(full_dates)
        .rename_axis("Date")
        .reset_index()
    )

    cond = df_gs["WTD_Predicted"].abs() <= 30

    return has_14_consecutive_days(cond, min_days=14)


def apply_wetland_flag_14days(df):
    """
    Apply 14-consecutive-day wetland-year classification to each ID-Year.
    Returns original df with a new column: wetland_year_14days.
    """

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    wetland_map = (
        df.groupby(["ID", "Year"], group_keys=False)
        .apply(is_wetland_year_14days)
        .reset_index(name="wetland_year_14days")
    )

    df = df.merge(wetland_map, on=["ID", "Year"], how="left")

    return df


def classify_wetland_years_14days(df):
    """
    Returns a summary dataframe with:
    ID, Year, wetland_year_14days, avg_WTD_predicted_gs

    Wetland condition:
    WTD_Predicted is within 30 cm of the surface for at least
    14 consecutive days during April to October.
    """

    results = []

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    for (id_, year), group in df.groupby(["ID", "Year"]):

        group = group.sort_values("Date")

        gs_start = pd.Timestamp(year=year, month=4, day=1)
        gs_end = pd.Timestamp(year=year, month=10, day=31)
        full_dates = pd.date_range(gs_start, gs_end, freq="D")

        df_gs = group[(group["Date"] >= gs_start) & (group["Date"] <= gs_end)]

        if df_gs.empty:
            wetland_flag = False
            avg_wtd = np.nan
            max_consecutive_days = 0

        else:
            avg_wtd = df_gs["WTD_Predicted"].mean()

            df_gs_daily = (
                df_gs.set_index("Date")
                .reindex(full_dates)
                .rename_axis("Date")
                .reset_index()
            )

            cond = df_gs_daily["WTD_Predicted"].abs() <= 30
            cond = cond.fillna(False).astype(bool)

            run_id = (cond != cond.shift()).cumsum()
            run_lengths = cond.groupby(run_id).sum()

            max_consecutive_days = run_lengths.max()
            wetland_flag = max_consecutive_days >= 14

        results.append({
            "ID": id_,
            "Year": year,
            "wetland_year_14days": wetland_flag,
            "max_consecutive_wetland_days": max_consecutive_days,
            "avg_WTD_predicted_gs": avg_wtd
        })

    return pd.DataFrame(results)


def count_wetland_years_14days(df_summary):
    """
    Count number of wetland years per ID using the 14-consecutive-day rule.
    """

    result = (
        df_summary.groupby("ID")["wetland_year_14days"]
        .sum()
        .reset_index(name="num_wetland_years_14days")
    )

    return result



df = df.copy()

df["date"] = pd.to_datetime(df["date"])
df["Date"] = df["date"]
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

df_new_14days = apply_wetland_flag_14days(df)

df_summary_14days = classify_wetland_years_14days(df)

df_counts_14days = count_wetland_years_14days(df_summary_14days)



# =============================================================================
# COMPARE WETLAND PERSISTENCE BETWEEN DECADES
# =============================================================================


df_summary_first_decade = df_summary_14days[df_summary_14days["Year"]<=2013]
df_summary_second_decade = df_summary_14days[df_summary_14days["Year"]>2013]
df_counts_first_decade_14days = count_wetland_years_14days(df_summary_first_decade)
df_counts__second_decade_14days = count_wetland_years_14days(df_summary_second_decade)



# =============================================================================
# PREPARE WETLAND PERSISTENCE TABLE
# =============================================================================


df_table5 = (
    df_counts_first_decade_14days.rename(
        columns={"num_wetland_years_14days": "first_decade_wetland_years"}
    )
    .merge(
        df_counts__second_decade_14days.rename(
            columns={"num_wetland_years_14days": "second_decade_wetland_years"}
        ),
        on="ID",
        how="outer",
    )
    .merge(df_loc[["ID", "Location"]], on="ID", how="left")
)

df_table5["first_decade_class"] = np.where(df_table5["first_decade_wetland_years"] >= 5, "wetland", "non-wetland")
df_table5["second_decade_class"] = np.where(df_table5["second_decade_wetland_years"] >= 5, "wetland", "non-wetland")

id_order = [
    "Well 12",
    "Well 2",
    "Well 92",
    "Well 93",
    "Well J",
    "Well K",
    "Goldsboro",
    "Lenoir",
    "Lynchburg",
    "Rains",
    "Wahee",
    "Well D",
    "Well H",
]

df_table5["ID"] = pd.Categorical(df_table5["ID"], categories=id_order, ordered=True)

df_table5 = df_table5[
    [
        "ID",
        "Location",
        "first_decade_wetland_years",
        "first_decade_class",
        "second_decade_wetland_years",
        "second_decade_class",
    ]
].sort_values("ID").reset_index(drop=True)


df_table5.to_csv(OUTPUT_DIR / "df_table5_wetland_years.csv", index=False)



# =============================================================================
# GENERATE INDIVIDUAL WTD TIME-SERIES FIGURES
# =============================================================================


df_results = df_new.copy()


# force a specific corner
#wf.plot_wtd_timeseries_by_id(df_results_D, df_check_D, legend_loc="upper left",ylim=[-290, 45])
for ids in df_results.ID.unique():
    df_each = df_results[df_results.ID== ids]
    plot_wtd_timeseries_by_id(df_each, df_each, legend_loc="upper left",ylim=[-290, 45])



# =============================================================================
# GENERATE MULTI-PANEL WTD TIME-SERIES FIGURE
# =============================================================================


def _calc_metrics(obs, pred):
    """Return RMSE, MAE, NSE, KGE computed on 1D arrays."""
    obs = np.asarray(obs).ravel()
    pred = np.asarray(pred).ravel()
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]

    if obs.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    mae = np.mean(np.abs(pred - obs))
    denom = np.sum((obs - obs.mean()) ** 2)
    nse = 1.0 - np.sum((obs - pred) ** 2) / denom if denom > 0 else np.nan

    std_o, std_p = np.std(obs), np.std(pred)
    if std_o == 0 or std_p == 0 or obs.size < 2:
        r = np.nan
    else:
        r = np.corrcoef(obs, pred)[0, 1]

    alpha = (std_p / std_o) if std_o != 0 else np.nan
    beta = (np.mean(pred) / np.mean(obs)) if np.mean(obs) != 0 else np.nan

    if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta):
        kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    else:
        kge = np.nan

    return rmse, mae, nse, kge


def plot_wtd_timeseries_grid(
    df_results: pd.DataFrame,
    df_check: pd.DataFrame,
    outdir: str = "./Figures",
    filename: str = "WTD_timeseries_grid.png",
    ncols: int = 3,
    fontsize: int = 18,
    dpi: int = 600,
    transparent: bool = True,
    legend_loc: str = "upper left",
    ylim=None,
    ref_line_y: float = -30,
    ref_line_color: str = "#4f4f4f",
    ref_line_style: str = "--",
    ref_line_width: float = 2.2,
    panel_width: float = 8,
    panel_height: float = 5
):
    """
    Create and save a multi-panel grid of Predicted vs Observed WTD time series for each ID.
    """

    df_r = df_results.copy()
    df_c = df_check.copy()

    df_r["Date"] = pd.to_datetime(df_r["Date"], errors="coerce")
    df_c["Date"] = pd.to_datetime(df_c["Date"], errors="coerce")

    Path(outdir).mkdir(parents=True, exist_ok=True)

    colors = {
        "pred": "#ff6f61",
        "obs": "blue",
    }

    def _bold_ticks(ax, size):
        ax.tick_params(axis="both", which="both", labelsize=size, width=2)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")

    ids_all = df_r["ID"].dropna().unique()
    n_ids = len(ids_all)
    nrows = math.ceil(n_ids / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        squeeze=False
    )
    axes = axes.flatten()

    for i, ids in enumerate(ids_all):
        ax = axes[i]

        df_each = df_r[df_r["ID"] == ids].copy()
        df_each_original = df_c[df_c["ID"] == ids].copy()

        overlap = (
            df_each[["Date", "WTD_Predicted"]]
            .merge(df_each_original[["Date", "WTD"]], on="Date", how="inner")
            .dropna(subset=["WTD_Predicted", "WTD"])
            .sort_values("Date")
        )

        rmse, mae, nse, kge = _calc_metrics(overlap["WTD"], overlap["WTD_Predicted"])

        if "Location" in df_each.columns:
            loc_vals = df_each["Location"].dropna().unique()
            loc = loc_vals[0] if len(loc_vals) else "Unknown"
        else:
            loc = "Unknown"

        ax.plot(
            df_each["Date"], df_each["WTD_Predicted"],
            color=colors["pred"], linewidth=2.0, alpha=0.8, label="Predicted WTD"
        )
        ax.scatter(
            df_each_original["Date"], df_each_original["WTD"],
            color=colors["obs"], s=10, alpha=0.8, label="Observed WTD"
        )
        ax.axhline(
            y=ref_line_y,
            color=ref_line_color,
            linestyle=ref_line_style,
            linewidth=ref_line_width,
            alpha=0.9
            #label="WL reference"
        )

        ax.set_xlabel("Date", fontsize=fontsize, fontweight="bold")
        ax.set_ylabel("Water table depth (cm)", fontsize=fontsize, fontweight="bold")
        ax.set_title(
            f"{ids} ({loc})\nRMSE={rmse:.2f}, MAE={mae:.2f}, NSE={nse:.2f}, KGE={kge:.2f}",
            fontsize=fontsize,
            fontweight="bold",
            pad=12
        )

        if isinstance(ylim, (list, tuple)) and len(ylim) == 2:
            ax.set_ylim(ylim[0], ylim[1])

        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        _bold_ticks(ax, fontsize - 2)

        leg = ax.legend(
            frameon=False,
            fontsize=fontsize - 3,
            ncol=3,
            loc=legend_loc
        )
        for txt in leg.get_texts():
            txt.set_fontweight("bold")

    for j in range(n_ids, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()

    outfile = Path(outdir) / filename
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", transparent=transparent)
    # plt.show()
    plt.close(fig)

    print(f"✅ Grid figure saved to: {outfile}")

df_results = df_new.copy()
df_check = df_new.copy()

plot_wtd_timeseries_grid(
    df_results=df_results,
    df_check=df_check,
    outdir=FIGURE_DIR,
    filename="WTD_timeseries_grid.png",
    ncols=3,
    legend_loc="lower left",
    ylim=[-290, 45]
)



# =============================================================================
# LOAD SAVED SUMMARY OUTPUTS
# This part will only work if you saved outputs from upper parts of this script
# Therefore only uncomment it if that criteria is fulfiled
# =============================================================================

# df_wetland = pd.read_csv(OUTPUT_DIR / "df_wetland.csv")
# df_trend = pd.read_csv(OUTPUT_DIR / "df_trend_results.csv")
# df_daily_trend = pd.read_csv(OUTPUT_DIR / "df_daily_trend.csv")



# =============================================================================
# PREPARE FINAL TREND SUMMARY TABLES
# =============================================================================


# df_trend_needed = df_trend[["ID","Location","variable","trend"]]
# df_trend_pivot = df_trend_needed.pivot(
#     index=["ID", "Location"],
#     columns="variable",
#     values="trend"
# ).reset_index()
# df_trend_pivot.sort_values(["Location","ID"], inplace=True)
# df_trend_pivot.to_csv(OUTPUT_DIR / "df_trend_pivot.csv", index=False)
# df_daily_trend.sort_values(["Location","ID"], inplace = True)
# df_daily_trend_filter = df_daily_trend[["ID","Location","trend"]]
# df_daily_trend_filter.rename(columns={"trend":"daily trend"}, inplace = True)
# df_annual_trend_pred_filter = df_annual_trend_pred[["ID","Location","trend","slope"]]
# df_annual_trend_pred_filter.rename(columns={"trend":"annual trend","slope":"annual slope"}, inplace = True)
# df_daily_annual = df_daily_trend_filter.merge(df_annual_trend_pred_filter, on= ["ID","Location"], how ="left")
# df_daily_annual_updated = df_daily_annual.merge(df_wetland, on="ID", how="left")
