"""
Core WTD Reconstruction: Classical Machine Learning and Deep Learning

Core workflow
-------------
1. Prepare features.
2. Build cumulative hydroclimatic predictors.
3. Evaluate correlation and multicollinearity.
4. Create the reconstruction train/test split by monitoring well.
5. Train and evaluate LR, RF, and XGBoost.
6. Train and evaluate FNN and LSTM.
7. Generate model-performance and KS summary tables.
8. Generate reconstruction diagnostic figures.


"""

import math
import os
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

import data_loader as dl
import WTD_functions as wf

warnings.filterwarnings("ignore")


# =============================================================================
# REPRODUCIBILITY AND OUTPUT DIRECTORIES
# =============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "Data").exists():
    PROJECT_ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "Data").exists():
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

REVIEWER_OUTPUT_DIR = PROJECT_ROOT / "reviewer_outputs"
REVIEWER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# WORKFLOW BLOCK FROM NOTEBOOK CELL 0
# =============================================================================

### Classical machine learning

warnings.filterwarnings("ignore")

# =============================================================================
# LOAD INPUT AND WATER-TABLE DATA
# =============================================================================

df_input_agu= dl.df_input_agu.copy()
df_sef = dl.df_sef.copy()
df_wtd_daily = df_sef.groupby(["Location","ID","date"]).mean().reset_index()
df_wtd_daily.dropna(inplace= True)
df_wtd_daily.rename(columns={"WTD(cm)":"WTD"}, inplace = True)

# =============================================================================
# ADD FIRST-WTD INFORMATION
# =============================================================================

def add_first_wtd_columns(
    df,
    id_col="ID",
    date_col="date",
    wtd_col="WTD",
    first_date_col="date_first_WTD",
    first_wtd_col="first_WTD",
    days_col="days_since_first_WTD",
):
    df_use = df.copy()
    df_use[date_col] = pd.to_datetime(df_use[date_col])
    df_use = df_use.sort_values([id_col, date_col]).reset_index(drop=True)

    first_rows = (
        df_use.sort_values([id_col, date_col])
        .groupby(id_col, as_index=False)
        .first()[[id_col, date_col, wtd_col]]
        .rename(columns={date_col: first_date_col, wtd_col: first_wtd_col})
    )

    out = df_use.merge(first_rows, on=id_col, how="left")
    out[days_col] = (out[date_col] - out[first_date_col]).dt.days
    return out


def merge_inputs_with_first_wtd_cumulative(
    df_input,
    df_wtd_first,
    id_col="ID",
    date_col="date",
    first_date_col="date_first_WTD",
    cumulative_cols=("prcp", "Evap_mm"),
    exclude_first_date=True,
    exclude_current_date=True,
    keep_all_current_input_cols=True,
):
    df_input_use = df_input.copy()
    df_wtd_use = df_wtd_first.copy()

    df_input_use[date_col] = pd.to_datetime(df_input_use[date_col])
    df_wtd_use[date_col] = pd.to_datetime(df_wtd_use[date_col])
    df_wtd_use[first_date_col] = pd.to_datetime(df_wtd_use[first_date_col])

    df_input_use = df_input_use.sort_values([id_col, date_col]).reset_index(drop=True)
    df_wtd_use = df_wtd_use.sort_values([id_col, date_col]).reset_index(drop=True)

    cumulative_cols = list(cumulative_cols)
    missing_cols = [col for col in cumulative_cols if col not in df_input_use.columns]
    if missing_cols:
        raise ValueError(f"These cumulative columns are missing from df_input: {missing_cols}")

    for col in cumulative_cols:
        df_input_use[f"cumsum_{col}"] = df_input_use.groupby(id_col)[col].cumsum()

    current_lookup = df_input_use[[id_col, date_col] + cumulative_cols + [f"cumsum_{c}" for c in cumulative_cols]].copy()
    rename_map = {col: f"current_day_{col}" for col in cumulative_cols}
    rename_map.update({f"cumsum_{col}": f"current_cumsum_{col}" for col in cumulative_cols})
    current_lookup = current_lookup.rename(columns=rename_map)

    df_final = df_wtd_use.merge(current_lookup, on=[id_col, date_col], how="left")

    for col in cumulative_cols:
        df_final[f"{col}_cumulative_since_first_WTD"] = df_final[f"current_cumsum_{col}"]
        if exclude_first_date:
            df_final.loc[df_final[date_col] == df_final[first_date_col], f"{col}_cumulative_since_first_WTD"] = np.nan
        if exclude_current_date:
            df_final[f"{col}_cumulative_since_first_WTD"] = df_final[f"{col}_cumulative_since_first_WTD"] - df_final[f"current_day_{col}"]

    if not keep_all_current_input_cols:
        drop_cols = [c for c in current_lookup.columns if c not in [id_col, date_col]]
        df_final = df_final.drop(columns=drop_cols, errors="ignore")

    return df_final


daily_lag_col = ["prcp", "tmax", "tmin", "srad", "Evap_mm"]
df_input_agu["Evap_mm"] = df_input_agu["Evap"] * 86400

df_input_updated = dl.lagged_function(df_input_agu, daily_lag_col, length=7)
df_input_updated["tavg"] = (df_input_updated["tmax"] + df_input_updated["tmin"]) / 2
df_input_updated["trange"] = df_input_updated["tmax"] - df_input_updated["tmin"]

needed_cols_updated_input = [
    "ID",
    "date",
    "prcp",
    "srad",
    "tavg",
    "trange",
    "Evap_mm",
    "GWS",
    "SM_S",
    "SM_RZ",
    "DOY",
    "prcp_1",
    "prcp_2",
    "prcp_3",
    "Evap_mm_1",
    "Evap_mm_2",
    "Evap_mm_3",
]

df_input_filter = df_input_updated[needed_cols_updated_input].copy()
df_wtd_daily_first = add_first_wtd_columns(df=df_wtd_daily, id_col="ID", date_col="date", wtd_col="WTD")

df_model_first_wtd = merge_inputs_with_first_wtd_cumulative(
    df_input=df_input_filter,
    df_wtd_first=df_wtd_daily_first,
    id_col="ID",
    date_col="date",
    first_date_col="date_first_WTD",
    cumulative_cols=("prcp", "Evap_mm"),
    exclude_first_date=True,
    exclude_current_date=True,
    keep_all_current_input_cols=True,
)

df_model_first_wtd.drop(columns="date_first_WTD", inplace=True)
df_model_first_wtd["date"] = pd.to_datetime(df_model_first_wtd["date"])
df_model_first_wtd = df_model_first_wtd.sort_values(["ID", "date"]).reset_index(drop=True)

print("df_input_updated shape:", df_input_updated.shape)
print("df_input_filter shape:", df_input_filter.shape)
print("df_model_first_wtd shape:", df_model_first_wtd.shape)

# =============================================================================
# MERGE FIRST-WTD FEATURES WITH CUMULATIVE HYDROCLIMATIC INPUTS
# =============================================================================

def merge_inputs_with_first_wtd_cumulative(
    df_input,
    df_wtd_first,
    id_col="ID",
    date_col="date",
    first_date_col="date_first_WTD",
    cumulative_cols=("prcp", "Evap_mm"),
    exclude_first_date=True,
    exclude_current_date=True,
    keep_all_current_input_cols=True
):
    """
    Merge daily input variables with WTD dataframe and add cumulative forcing
    since the first WTD observation.

    Default interval:
        (date_first_WTD, current_date)

    This means:
        exclude first WTD date
        exclude current WTD date

    Example:
        first WTD date = 2006-06-21
        current date   = 2006-06-25

        cumulative sum uses:
        2006-06-22, 2006-06-23, 2006-06-24

    Parameters
    ----------
    df_input : DataFrame
        Complete daily input dataframe. Must contain ID, date, prcp, Evap_mm, etc.

    df_wtd_first : DataFrame
        WTD dataframe already containing date_first_WTD, first_WTD,
        and days_since_first_WTD.

    cumulative_cols : tuple/list
        Columns to cumulatively sum from df_input.

    keep_all_current_input_cols : bool
        If True, current-day input variables are merged into the final dataframe.

    Returns
    -------
    df_final : DataFrame
        WTD dataframe with current-day input variables and cumulative features.
    """

    df_input_use = df_input.copy()
    df_wtd_use = df_wtd_first.copy()

    df_input_use[date_col] = pd.to_datetime(df_input_use[date_col])
    df_wtd_use[date_col] = pd.to_datetime(df_wtd_use[date_col])
    df_wtd_use[first_date_col] = pd.to_datetime(df_wtd_use[first_date_col])

    df_input_use = df_input_use.sort_values([id_col, date_col]).reset_index(drop=True)
    df_wtd_use = df_wtd_use.sort_values([id_col, date_col]).reset_index(drop=True)

    cumulative_cols = list(cumulative_cols)

    missing_cols = [col for col in cumulative_cols if col not in df_input_use.columns]
    if missing_cols:
        raise ValueError(f"These cumulative columns are missing from df_input: {missing_cols}")

    # ------------------------------------------------------------

    # ------------------------------------------------------------
    for col in cumulative_cols:
        df_input_use[f"cumsum_{col}"] = (
            df_input_use
            .groupby(id_col)[col]
            .cumsum()
        )

    cumsum_cols = [f"cumsum_{col}" for col in cumulative_cols]

    # ------------------------------------------------------------

    # ------------------------------------------------------------
    current_lookup = df_input_use[[id_col, date_col] + cumsum_cols].copy()

    current_lookup = current_lookup.rename(
        columns={
            f"cumsum_{col}": f"current_cumsum_{col}"
            for col in cumulative_cols
        }
    )

    df_final = df_wtd_use.merge(
        current_lookup,
        on=[id_col, date_col],
        how="left"
    )

    # ------------------------------------------------------------

    # ------------------------------------------------------------
    first_lookup = df_input_use[[id_col, date_col] + cumsum_cols].copy()

    first_lookup = first_lookup.rename(
        columns={
            date_col: first_date_col,
            **{
                f"cumsum_{col}": f"first_cumsum_{col}"
                for col in cumulative_cols
            }
        }
    )

    df_final = df_final.merge(
        first_lookup,
        on=[id_col, first_date_col],
        how="left"
    )

    # ------------------------------------------------------------

    # ------------------------------------------------------------
    current_values_lookup = df_input_use[[id_col, date_col] + cumulative_cols].copy()

    current_values_lookup = current_values_lookup.rename(
        columns={
            col: f"current_day_{col}"
            for col in cumulative_cols
        }
    )

    df_final = df_final.merge(
        current_values_lookup,
        on=[id_col, date_col],
        how="left"
    )

    # ------------------------------------------------------------

    #    Default excludes first date, so this is usually not needed.
    # ------------------------------------------------------------
    first_values_lookup = df_input_use[[id_col, date_col] + cumulative_cols].copy()

    first_values_lookup = first_values_lookup.rename(
        columns={
            date_col: first_date_col,
            **{
                col: f"first_day_{col}"
                for col in cumulative_cols
            }
        }
    )

    df_final = df_final.merge(
        first_values_lookup,
        on=[id_col, first_date_col],
        how="left"
    )

    # ------------------------------------------------------------

    # ------------------------------------------------------------
    for col in cumulative_cols:
        # This gives (first_date, current_date]
        total = df_final[f"current_cumsum_{col}"] - df_final[f"first_cumsum_{col}"]

        # If current date should be excluded, convert to (first_date, current_date)
        if exclude_current_date:
            total = total - df_final[f"current_day_{col}"]

        # If first date should be included, add first-day value back
        if not exclude_first_date:
            total = total + df_final[f"first_day_{col}"]

        df_final[f"{col}_cumulative_since_first_WTD"] = total

    # ------------------------------------------------------------

    # ------------------------------------------------------------
    if keep_all_current_input_cols:
        input_cols_to_merge = [
            col for col in df_input_use.columns
            if not col.startswith("cumsum_")
        ]

        df_current_inputs = df_input_use[input_cols_to_merge].copy()

        df_final = df_final.merge(
            df_current_inputs,
            on=[id_col, date_col],
            how="left",
            suffixes=("", "_current")
        )

    # ------------------------------------------------------------

    # ------------------------------------------------------------
    helper_cols = []

    for col in cumulative_cols:
        helper_cols.extend([
            f"current_cumsum_{col}",
            f"first_cumsum_{col}",
            f"current_day_{col}",
            f"first_day_{col}"
        ])

    helper_cols = [col for col in helper_cols if col in df_final.columns]

    df_final = df_final.drop(columns=helper_cols)

    # ------------------------------------------------------------

    # ------------------------------------------------------------
    important_cols = [
        col for col in [
            "Location",
            id_col,
            date_col,
            "WTD",
            first_date_col,
            "first_WTD",
            "days_since_first_WTD"
        ]
        if col in df_final.columns
    ]

    engineered_cols = [
        f"{col}_cumulative_since_first_WTD"
        for col in cumulative_cols
    ]

    remaining_cols = [
        col for col in df_final.columns
        if col not in important_cols + engineered_cols
    ]

    df_final = df_final[
        important_cols + engineered_cols + remaining_cols
    ].copy()

    df_final = df_final.sort_values([id_col, date_col]).reset_index(drop=True)

    return df_final

# =============================================================================
# BUILD FIRST-WTD MODEL DATAFRAME
# =============================================================================

df_model_first_wtd = merge_inputs_with_first_wtd_cumulative(
    df_input=df_input_filter,
    df_wtd_first=df_wtd_daily_first,
    id_col="ID",
    date_col="date",
    first_date_col="date_first_WTD",
    cumulative_cols=("prcp", "Evap_mm"),
    exclude_first_date=True,
    exclude_current_date=True,
    keep_all_current_input_cols=True
)

print(df_model_first_wtd.shape)

# =============================================================================
# FILTER WTD RANGE
# =============================================================================


df_model_first_wtd.drop(columns= "date_first_WTD", inplace =  True)

# =============================================================================
# ADD LAST-WTD INFORMATION
# =============================================================================

def add_last_wtd_columns(
    df,
    id_col="ID",
    date_col="date",
    wtd_col="WTD",
    last_date_col="date_last_WTD",
    last_wtd_col="last_WTD",
    days_col="days_until_last_WTD"
):
    df_out = df.copy()

    df_out[date_col] = pd.to_datetime(df_out[date_col])
    df_out = df_out.sort_values([id_col, date_col]).reset_index(drop=True)

    # Last WTD date for each ID
    df_out[last_date_col] = (
        df_out.groupby(id_col)[date_col]
        .transform("last")
    )

    # Last WTD value for each ID
    df_out[last_wtd_col] = (
        df_out.groupby(id_col)[wtd_col]
        .transform("last")
    )

    # Days until last WTD observation
    df_out[days_col] = (
        df_out[last_date_col] - df_out[date_col]
    ).dt.days

    return df_out

# =============================================================================
# BUILD LAST-WTD DATAFRAME
# =============================================================================

df_wtd_daily_last = add_last_wtd_columns(
    df=df_wtd_daily,
    id_col="ID",
    date_col="date",
    wtd_col="WTD"
)

# =============================================================================
# MERGE LAST-WTD FEATURES WITH CUMULATIVE HYDROCLIMATIC INPUTS
# =============================================================================

def merge_inputs_with_last_wtd_cumulative(
    df_input,
    df_wtd_last,
    id_col="ID",
    date_col="date",
    last_date_col="date_last_WTD",
    cumulative_cols=("prcp", "Evap_mm"),
    keep_all_current_input_cols=True
):
    # Merge daily input variables with WTD dataframe and add backward cumulative
    # forcing until the last WTD observation.
    # Default interval is (current_date, date_last_WTD), excluding both ends.

    df_input_use = df_input.copy()
    df_wtd_use = df_wtd_last.copy()

    df_input_use[date_col] = pd.to_datetime(df_input_use[date_col])
    df_wtd_use[date_col] = pd.to_datetime(df_wtd_use[date_col])
    df_wtd_use[last_date_col] = pd.to_datetime(df_wtd_use[last_date_col])

    df_input_use = df_input_use.sort_values([id_col, date_col]).reset_index(drop=True)
    df_wtd_use = df_wtd_use.sort_values([id_col, date_col]).reset_index(drop=True)

    cumulative_cols = list(cumulative_cols)

    missing_cols = [col for col in cumulative_cols if col not in df_input_use.columns]
    if missing_cols:
        raise ValueError(f"These cumulative columns are missing from df_input: {missing_cols}")

    # ------------------------------------------------------------

    # ------------------------------------------------------------
    for col in cumulative_cols:
        df_input_use[f"backward_cumsum_inclusive_{col}"] = (
            df_input_use
            .groupby(id_col)[col]
            .transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
        )

    back_cols = [f"backward_cumsum_inclusive_{col}" for col in cumulative_cols]

    # ------------------------------------------------------------

    # ------------------------------------------------------------
    current_lookup = df_input_use[[id_col, date_col] + back_cols + cumulative_cols].copy()
    rename_map = {}
    for col in cumulative_cols:
        rename_map[f"backward_cumsum_inclusive_{col}"] = f"current_backward_cumsum_inclusive_{col}"
        rename_map[col] = f"current_day_{col}"
    current_lookup = current_lookup.rename(columns=rename_map)

    df_final = df_wtd_use.merge(
        current_lookup,
        on=[id_col, date_col],
        how="left"
    )

    # ------------------------------------------------------------

    # ------------------------------------------------------------
    last_lookup = df_input_use[[id_col, date_col] + cumulative_cols].copy()
    last_lookup = last_lookup.rename(
        columns={
            date_col: last_date_col,
            **{
                col: f"last_day_{col}"
                for col in cumulative_cols
            }
        }
    )

    df_final = df_final.merge(
        last_lookup,
        on=[id_col, last_date_col],
        how="left"
    )

    # ------------------------------------------------------------

    #    to one day before the last WTD date
    # ------------------------------------------------------------
    for col in cumulative_cols:
        total = (
            df_final[f"current_backward_cumsum_inclusive_{col}"]
            - df_final[f"current_day_{col}"]
            - df_final[f"last_day_{col}"]
        )

        total = np.where(
            df_final[date_col] >= df_final[last_date_col],
            0,
            total
        )

        df_final[f"{col}_cumulative_until_last_WTD"] = total

    # ------------------------------------------------------------

    # ------------------------------------------------------------
    if keep_all_current_input_cols:
        input_cols_to_merge = [
            col for col in df_input_use.columns
            if not col.startswith("backward_cumsum_inclusive_")
        ]
        df_current_inputs = df_input_use[input_cols_to_merge].copy()

        df_final = df_final.merge(
            df_current_inputs,
            on=[id_col, date_col],
            how="left",
            suffixes=("", "_current")
        )

    # ------------------------------------------------------------

    # ------------------------------------------------------------
    helper_cols = []
    for col in cumulative_cols:
        helper_cols.extend([
            f"current_backward_cumsum_inclusive_{col}",
            f"current_day_{col}",
            f"last_day_{col}"
        ])

    helper_cols = [col for col in helper_cols if col in df_final.columns]
    df_final = df_final.drop(columns=helper_cols)

    # ------------------------------------------------------------

    # ------------------------------------------------------------
    important_cols = [
        col for col in [
            "Location",
            id_col,
            date_col,
            "WTD",
            last_date_col,
            "last_WTD",
            "days_until_last_WTD"
        ]
        if col in df_final.columns
    ]

    engineered_cols = [
        f"{col}_cumulative_until_last_WTD"
        for col in cumulative_cols
    ]

    remaining_cols = [
        col for col in df_final.columns
        if col not in important_cols + engineered_cols
    ]

    df_final = df_final[important_cols + engineered_cols + remaining_cols].copy()
    df_final = df_final.sort_values([id_col, date_col]).reset_index(drop=True)

    return df_final

# =============================================================================
# BUILD LAST-WTD MODEL DATAFRAME
# =============================================================================

df_model_last_wtd = merge_inputs_with_last_wtd_cumulative(
    df_input=df_input_filter,
    df_wtd_last=df_wtd_daily_last,
    id_col="ID",
    date_col="date",
    last_date_col="date_last_WTD",
    cumulative_cols=("prcp", "Evap_mm"),
    keep_all_current_input_cols=True
)

print(df_model_last_wtd.shape)

# =============================================================================
# COMBINE FIRST- AND LAST-WTD DATA
# =============================================================================

df_model_last_wtd_ready = df_model_last_wtd.copy()
if "date_last_WTD" in df_model_last_wtd_ready.columns:
    df_model_last_wtd_ready = df_model_last_wtd_ready.drop(columns=["date_last_WTD"])

last_feature_cols = [
    col for col in [
        "ID",
        "date",
        "last_WTD",
        "days_until_last_WTD",
        "prcp_cumulative_until_last_WTD",
        "Evap_mm_cumulative_until_last_WTD",
    ]
    if col in df_model_last_wtd_ready.columns
]

df_model_first_last_wtd_raw = df_model_first_wtd.merge(
    df_model_last_wtd_ready[last_feature_cols],
    on=["ID", "date"],
    how="inner"
)

print("Combined dataframe shape:", df_model_first_last_wtd_raw.shape)

# =============================================================================
# ESTIMATE CUMULATIVE EFFECTIVE PRECIPITATION
# =============================================================================

df_model_first_last_wtd_raw["Est_cumulative_effective_prcp"] = df_model_first_last_wtd_raw['prcp_cumulative_since_first_WTD']- df_model_first_last_wtd_raw['Evap_mm_cumulative_since_first_WTD']
df_model_first_last_wtd = df_model_first_last_wtd_raw.drop(columns=['last_WTD', 'days_until_last_WTD',
       'prcp_cumulative_until_last_WTD', 'Evap_mm_cumulative_until_last_WTD', 'days_since_first_WTD','prcp_cumulative_since_first_WTD','Evap_mm_cumulative_since_first_WTD'])

# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

# Reconstruction correlation heatmap

recon_numeric_df = df_model_first_last_wtd.select_dtypes(include=["int", "float"]).copy()
recon_corr = recon_numeric_df.corr()
recon_mask = np.triu(np.ones_like(recon_corr, dtype=bool))

fig, ax = plt.subplots(figsize=(25, 15))
hm = sns.heatmap(
    recon_corr,
    mask=recon_mask,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    square=False,
    ax=ax,
    annot_kws={"size": 14, "weight": "bold"},
    cbar_kws={"label": "Pearson correlation"},
)

tick_fontsize = 16
label_fontsize = 18
cbar_tick_fontsize = 16
cbar_label_fontsize = 18

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=tick_fontsize, fontweight="bold")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_fontsize, fontweight="bold")
ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize, fontweight="bold")
ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize, fontweight="bold")

cbar = hm.collections[0].colorbar
cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
for tick in cbar.ax.get_yticklabels():
    tick.set_fontweight("bold")
cbar.set_label("Pearson correlation", fontsize=cbar_label_fontsize, fontweight="bold")

plt.tight_layout()
Path("reviewer_outputs").mkdir(parents=True, exist_ok=True)
recon_corr_fig_path = Path("reviewer_outputs") / "reconstruction_numeric_correlation_heatmap.png"
plt.savefig(recon_corr_fig_path, dpi=600, bbox_inches="tight")
# plt.show()
print(f"Saved reconstruction correlation heatmap to: {recon_corr_fig_path}")

# =============================================================================
# VARIANCE INFLATION FACTOR ANALYSIS
# =============================================================================

def calculate_vif(df_features):
    """
    Calculate VIF for all features in a dataframe.
    """
    X = df_features.copy()

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    # Replace inf with nan
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop rows with missing values
    X = X.dropna()

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns

    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]

    vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

    return vif_data


def stepwise_vif_selection(df, target_col="WTD", threshold=6.1, verbose=True):
    """
    Iteratively removes the feature with the highest VIF until all VIF values
    are below the selected threshold.

    Parameters
    ----------
    df : pandas DataFrame
        Input dataframe containing target and predictor variables.

    target_col : str
        Target column to exclude from VIF calculation.

    threshold : float
        VIF threshold. A threshold of 6.1 is used in this workflow.

    verbose : bool
        If True, prints each removed feature.

    Returns
    -------
    selected_features : list
        Final list of retained features.

    removed_features : list
        List of removed features with their VIF values.

    final_vif : pandas DataFrame
        Final VIF table.
    """

    # Candidate predictors only
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].copy()

    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])

    # Remove constant columns
    nunique = X.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()

    if len(constant_cols) > 0:
        if verbose:
            print("Removing constant columns:", constant_cols)
        X = X.drop(columns=constant_cols)

    # Replace inf with NaN and drop missing rows
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    removed_features = []

    while True:
        vif_table = calculate_vif(X)

        max_vif = vif_table["VIF"].max()
        max_feature = vif_table.loc[vif_table["VIF"].idxmax(), "feature"]

        if verbose:
            print("\nCurrent highest VIF:")
            print(vif_table.head(10))

        if max_vif <= threshold:
            break

        if verbose:
            print(f"Removing '{max_feature}' with VIF = {max_vif:.2f}")

        removed_features.append({
            "removed_feature": max_feature,
            "VIF": max_vif
        })

        X = X.drop(columns=[max_feature])

        # Stop if only one feature is left
        if X.shape[1] <= 1:
            break

    selected_features = X.columns.tolist()
    removed_features = pd.DataFrame(removed_features)
    final_vif = calculate_vif(X)

    return selected_features, removed_features, final_vif

# =============================================================================
# RUN STEPWISE VIF SELECTION
# =============================================================================
filtered_cols = recon_numeric_df.columns
selected_features, removed_features, final_vif = stepwise_vif_selection(
    df=df_model_first_last_wtd[filtered_cols],
    target_col="WTD",
    threshold=6.1,
    verbose=True
)

print("Selected features:")
print(selected_features)

print("\nRemoved features:")
print(removed_features)

print("\nFinal VIF:")
print(final_vif)

# =============================================================================
# CLASSICAL MACHINE-LEARNING RECONSTRUCTION WORKFLOW
# =============================================================================

# Classical reconstruction workflow

SEED = 42
EDGE_FRACTION = 0.05
MIDDLE_RANDOM_FRACTION = 0.10
MIN_RECORDS_PER_ID = 5
VALIDATION_SIZE = 0.10
OUTPUT_DIR = REVIEWER_OUTPUT_DIR


def split_edge_random_by_id(
    df,
    target_col="WTD",
    id_col="ID",
    date_col="date",
    edge_fraction=0.05,
    middle_random_fraction=0.10,
    min_records_per_id=5,
    random_state=42,
):
    df_use = df.copy()
    df_use[date_col] = pd.to_datetime(df_use[date_col])
    df_use = df_use.sort_values([id_col, date_col]).reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    train_parts = []
    test_parts = []
    skipped_ids = []

    for well_id, g in df_use.groupby(id_col, sort=False):
        g = g.sort_values(date_col).reset_index(drop=True)
        g = g.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        n = len(g)

        if n < min_records_per_id:
            skipped_ids.append(well_id)
            continue

        edge_n = max(1, int(np.floor(n * edge_fraction)))
        if n - 2 * edge_n < 3:
            edge_n = max(1, (n - 3) // 2)

        head = g.iloc[:edge_n].copy()
        middle = g.iloc[edge_n:n - edge_n].copy()
        tail = g.iloc[n - edge_n:].copy()

        if len(middle) < 1:
            skipped_ids.append(well_id)
            continue

        n_random_test = max(1, int(np.floor(n * middle_random_fraction)))
        n_random_test = min(n_random_test, len(middle))

        middle_pos = np.arange(len(middle))
        rng.shuffle(middle_pos)
        middle_test_pos = np.sort(middle_pos[:n_random_test])
        middle_train_pos = np.sort(middle_pos[n_random_test:])

        middle_test = middle.iloc[middle_test_pos].copy()
        middle_train = middle.iloc[middle_train_pos].copy()

        head["split_group"] = "start_5%"
        middle_test["split_group"] = "middle_random_10%"
        tail["split_group"] = "end_5%"
        middle_train["split_group"] = "train_middle"

        train_parts.append(middle_train)
        test_parts.extend([head, middle_test, tail])

    if len(train_parts) == 0 or len(test_parts) == 0:
        raise ValueError("No train/test splits were produced. Check the dataframe and split settings.")

    train_df = pd.concat(train_parts, ignore_index=True).sort_values([id_col, date_col]).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sort_values([id_col, date_col]).reset_index(drop=True)
    return train_df, test_df, skipped_ids


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def nrmse_std(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    denom = np.nanstd(y_true, ddof=1)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return rmse(y_true, y_pred) / denom


def nse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((y_true - y_pred) ** 2) / denom


def kge(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2:
        return np.nan
    mean_obs = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    std_obs = np.std(y_true, ddof=1)
    std_pred = np.std(y_pred, ddof=1)
    if std_obs == 0 or np.isnan(std_obs) or mean_obs == 0 or np.isnan(mean_obs):
        return np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = std_pred / std_obs
    beta = mean_pred / mean_obs
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def evaluate_model(y_true, y_pred):
    return {
        "NSE": nse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "NRMSE": nrmse_std(y_true, y_pred),
        "KGE": kge(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def fit_yj(y_train):
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    y_train_yj = transformer.fit_transform(np.asarray(y_train, dtype=float).reshape(-1, 1)).ravel()
    return transformer, y_train_yj


def transform_yj(transformer, y):
    return transformer.transform(np.asarray(y, dtype=float).reshape(-1, 1)).ravel()


def inverse_yj(transformer, y):
    return transformer.inverse_transform(np.asarray(y, dtype=float).reshape(-1, 1)).ravel()


def ks_test_observed_vs_predicted(df, actual_col="actual_WTD", prediction_cols=None, model_name_map=None, group_cols=None):
    df_use = df.copy()
    group_cols = group_cols or []
    model_name_map = model_name_map or {}
    if prediction_cols is None:
        prediction_cols = [c for c in df_use.columns if c.startswith("predicted_WTD")]

    rows = []
    grouped = [((), df_use)] if len(group_cols) == 0 else df_use.groupby(group_cols, dropna=False)

    for group_key, g in grouped:
        if len(group_cols) == 0:
            group_info = {"group": "overall"}
        else:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_info = dict(zip(group_cols, group_key))

        observed = g[actual_col].replace([np.inf, -np.inf], np.nan).dropna().values

        for pred_col in prediction_cols:
            predicted = g[pred_col].replace([np.inf, -np.inf], np.nan).dropna().values
            model_name = model_name_map.get(pred_col, pred_col.replace("predicted_WTD_", ""))
            if len(observed) < 2 or len(predicted) < 2:
                ks_statistic = np.nan
                p_value = np.nan
            else:
                ks_result = ks_2samp(observed, predicted, alternative="two-sided", mode="auto")
                ks_statistic = ks_result.statistic
                p_value = ks_result.pvalue
            rows.append({
                **group_info,
                "Model": model_name,
                "prediction_col": pred_col,
                "N_observed": len(observed),
                "N_predicted": len(predicted),
                "KS_statistic": ks_statistic,
                "KS_p_value": p_value,
            })
    return pd.DataFrame(rows)


def prepare_classical_dataframe(df, target_col="WTD", id_col="ID", date_col="date"):
    df_use = df.copy()
    df_use[date_col] = pd.to_datetime(df_use[date_col])
    df_use = df_use.sort_values([id_col, date_col]).reset_index(drop=True)

    feature_cols = [c for c in selected_features if c in df_use.columns and c not in {target_col, id_col, date_col, "split_group", "row_id"}]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_use[c])]
    if len(feature_cols) == 0:
        raise ValueError("No usable numeric predictor columns were found for the classical workflow.")

    keep_cols = [c for c in [id_col, date_col, target_col, "Location"] if c in df_use.columns] + feature_cols
    df_use = df_use[keep_cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col] + feature_cols).reset_index(drop=True)
    return df_use, feature_cols


def fit_predict_classical_model(model, X_train, y_train, X_test, scale_features=False):
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train_fit = scaler.fit_transform(X_train)
        X_test_fit = scaler.transform(X_test)
    else:
        X_train_fit = X_train
        X_test_fit = X_test

    model.fit(X_train_fit, y_train)
    return model.predict(X_test_fit), scaler


def run_classical_reconstruction_suite(df, target_col="WTD", id_col="ID", date_col="date", random_state=42):
    df_clean, feature_cols = prepare_classical_dataframe(df, target_col=target_col, id_col=id_col, date_col=date_col)
    train_df, test_df, skipped_ids = split_edge_random_by_id(
        df_clean,
        target_col=target_col,
        id_col=id_col,
        date_col=date_col,
        edge_fraction=EDGE_FRACTION,
        middle_random_fraction=MIDDLE_RANDOM_FRACTION,
        min_records_per_id=MIN_RECORDS_PER_ID,
        random_state=random_state,
    )

    def make_model_frame(frame):
        keep_cols = [c for c in [id_col, date_col, target_col, "Location", "split_group"] if c in frame.columns]
        return frame[keep_cols + feature_cols].copy()

    train_model_df = make_model_frame(train_df)
    test_model_df = make_model_frame(test_df)

    def tune_random_forest(X_train, y_train, random_state=random_state, n_trials=5):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=random_state,
        )

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 100),
                "max_depth": trial.suggest_int("max_depth", 2, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_float("max_features", 0.3, 1.0),
                "random_state": random_state,
                "n_jobs": -1,
            }
            model = RandomForestRegressor(**params)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            return rmse(inverse_yj(y_transformer, y_val), inverse_yj(y_transformer, pred))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return RandomForestRegressor(**study.best_params, random_state=random_state, n_jobs=-1), study.best_params

    def tune_xgboost(X_train, y_train, random_state=random_state, n_trials=5):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=random_state,
        )

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 100),
                "max_depth": trial.suggest_int("max_depth", 2, 30),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.300, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": random_state,
                "n_jobs": -1,
                "objective": "reg:squarederror",
            }
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            return rmse(inverse_yj(y_transformer, y_val), inverse_yj(y_transformer, pred))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params.copy()
        best_params.update({"random_state": random_state, "n_jobs": -1, "objective": "reg:squarederror"})
        return XGBRegressor(**best_params), study.best_params

    y_train = train_model_df[target_col].to_numpy(dtype=float)
    y_test = test_model_df[target_col].to_numpy(dtype=float)
    X_train = train_model_df[feature_cols].to_numpy(dtype=float)
    X_test = test_model_df[feature_cols].to_numpy(dtype=float)

    y_transformer, y_train_yj = fit_yj(y_train)

    rf_model, rf_best_params = tune_random_forest(X_train, y_train_yj, random_state=random_state, n_trials=30)
    xgb_model, xgb_best_params = tune_xgboost(X_train, y_train_yj, random_state=random_state, n_trials=30)

    model_specs = {
        "LR": (LinearRegression(), True, {}),
        "RF": (rf_model, False, rf_best_params),
        "XGB": (xgb_model, False, xgb_best_params),
    }

    combined_results = []
    for model_name, (model, scale_features, best_params) in model_specs.items():
        pred_yj, _ = fit_predict_classical_model(model, X_train, y_train_yj, X_test, scale_features=scale_features)
        pred = inverse_yj(y_transformer, pred_yj)

        result_df = test_model_df[[c for c in [id_col, date_col, "Location", "split_group"] if c in test_model_df.columns]].copy()
        result_df["actual_WTD"] = y_test
        result_df["predicted_WTD"] = pred
        result_df["Model"] = model_name
        result_df["best_params"] = [best_params] * len(result_df)
        combined_results.append(result_df)

    combined_results = pd.concat(combined_results, ignore_index=True)

    def summarize_by_model(results_df):
        rows = []
        for model_name, g in results_df.groupby("Model", sort=True):
            metrics = evaluate_model(g["actual_WTD"], g["predicted_WTD"])
            rows.append({"Model": model_name, "N": len(g), **metrics})
        return pd.DataFrame(rows)

    def summarize_by_well(results_df):
        rows = []
        for (model_name, well_id), g in results_df.groupby(["Model", id_col], sort=True):
            metrics = evaluate_model(g["actual_WTD"], g["predicted_WTD"])
            row = {"Model": model_name, id_col: well_id, "N": len(g), **metrics}
            if "Location" in g.columns and g["Location"].notna().any():
                row["Location"] = g["Location"].dropna().iloc[0]
            rows.append(row)
        return pd.DataFrame(rows)

    def summarize_by_split(results_df):
        rows = []
        for (model_name, split_group), g in results_df.groupby(["Model", "split_group"], sort=True):
            metrics = evaluate_model(g["actual_WTD"], g["predicted_WTD"])
            rows.append({"Model": model_name, "split_group": split_group, "N": len(g), **metrics})
        return pd.DataFrame(rows)

    table_2_df = summarize_by_model(combined_results)
    table_s2_df = summarize_by_well(combined_results)
    table_s3_df = summarize_by_split(combined_results)

    prediction_cols = ["predicted_WTD"]
    ks_overall_df = ks_test_observed_vs_predicted(combined_results, actual_col="actual_WTD", prediction_cols=prediction_cols)
    ks_by_well_df = ks_test_observed_vs_predicted(combined_results, actual_col="actual_WTD", prediction_cols=prediction_cols, group_cols=["Model", id_col])
    ks_by_split_df = ks_test_observed_vs_predicted(combined_results, actual_col="actual_WTD", prediction_cols=prediction_cols, group_cols=["Model", "split_group"])

    return {
        "df_clean": df_clean,
        "feature_cols": feature_cols,
        "train_df": train_df,
        "test_df": test_df,
        "skipped_ids": skipped_ids,
        "combined_results": combined_results,
        "table_2_df": table_2_df,
        "table_s2_df": table_s2_df,
        "table_s3_df": table_s3_df,
        "ks_overall_df": ks_overall_df,
        "ks_by_well_df": ks_by_well_df,
        "ks_by_split_df": ks_by_split_df,
    }

# =============================================================================
# TRAIN AND EVALUATE CLASSICAL MODELS
# =============================================================================

selected_features = ['first_WTD', 'prcp', 'srad', 'tavg',
       'trange', 'Evap_mm', 'GWS', 'SM_S', 'SM_RZ', 'DOY', 'prcp_1', 'prcp_2',
       'prcp_3', 'Evap_mm_1', 'Evap_mm_2', 'Evap_mm_3',
       'Est_cumulative_effective_prcp']
classical_suite_all = run_classical_reconstruction_suite(
    df_model_first_last_wtd,
    target_col="WTD",
    id_col="ID",
    date_col="date",
    random_state=SEED,
)

# =============================================================================
# SAVE CLASSICAL TRAINING AND TEST DATA
# =============================================================================

classical_suite_all["test_df"].to_csv(REVIEWER_OUTPUT_DIR / "test_dataframe.csv", index=False)
classical_suite_all["train_df"].to_csv(REVIEWER_OUTPUT_DIR / "train_dataframe.csv", index=False)
classical_suite_all["df_clean"].to_csv(REVIEWER_OUTPUT_DIR / "full_dataframe.csv", index=False)

# =============================================================================
# OPTIONAL RANDOM-FOREST OPTUNA SUMMARY
# =============================================================================

# OPTIONAL DIAGNOSTIC BLOCK DISABLED
# The main Random Forest model is already trained/evaluated in the
# classical reconstruction suite. Uncomment this block only if the
# RF-only Optuna parameter summary is needed.
#
# # Random Forest Optuna summary
#
# rf_results = classical_suite_all["combined_results"].copy()
# if "Model" in rf_results.columns:
#     rf_results = rf_results[rf_results["Model"].astype(str).str.contains("RF", case=False, na=False)].copy()
#
# if "best_params" not in rf_results.columns:
#     raise ValueError("No best_params column was found in classical_suite_all['combined_results'].")
#
# rf_best_params = rf_results["best_params"].dropna().iloc[0]
# rf_best_params_df = pd.DataFrame([rf_best_params])
#
# print("Random Forest best parameters from Optuna:")
#
# rf_summary_df = classical_suite_all["table_2_df"].copy()
# if "Model" in rf_summary_df.columns:
#     rf_summary_df = rf_summary_df[rf_summary_df["Model"].astype(str).str.contains("RF", case=False, na=False)].copy()

# =============================================================================
# DEEP-LEARNING MODEL DEFINITIONS
# =============================================================================

def set_dl_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_dl_seed(SEED)
DL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_tabular_dataframe(df, target_col="WTD", id_col="ID", date_col="date"):
    df_use = df.copy()
    df_use[date_col] = pd.to_datetime(df_use[date_col])
    df_use = df_use.sort_values([id_col, date_col]).reset_index(drop=True)
    feature_cols = [
        col for col in df_use.columns
        if col not in {id_col, date_col, target_col, "split_group", "row_id"}
        and pd.api.types.is_numeric_dtype(df_use[col])
    ]
    df_use = df_use.dropna(subset=[target_col] + feature_cols).reset_index(drop=True)
    return df_use, feature_cols


def fit_yj(y_train):
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    y_train_yj = transformer.fit_transform(np.asarray(y_train).reshape(-1, 1)).ravel()
    return transformer, y_train_yj


class FNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.15):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_fnn_model(model, train_loader, val_loader, epochs=120, patience=20, learning_rate=1e-3, weight_decay=1e-5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = None
    best_val = float("inf")
    wait = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(DL_DEVICE)
            yb = yb.to(DL_DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DL_DEVICE)
                yb = yb.to(DL_DEVICE)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss = float(np.mean(val_losses)) if val_losses else np.nan
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predict_fnn(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DL_DEVICE)
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def summarize_by_split(results_df, actual_col="actual_WTD", predicted_col="predicted_WTD", id_col="ID", group_col="split_group"):
    per_well_rows = []
    for (well_id, grp), dfg in results_df.groupby([id_col, group_col], dropna=False):
        per_well_rows.append({
            id_col: well_id,
            group_col: grp,
            "N": len(dfg),
            **evaluate_model(dfg[actual_col], dfg[predicted_col]),
        })
    per_well_df = pd.DataFrame(per_well_rows)

    overall_rows = []
    for grp, dfg in results_df.groupby(group_col, dropna=False):
        overall_rows.append({
            group_col: grp,
            "N": len(dfg),
            **evaluate_model(dfg[actual_col], dfg[predicted_col]),
        })
    overall_df = pd.DataFrame(overall_rows)
    return per_well_df, overall_df


def run_yj_fnn_workflow(
    df,
    split_fn,
    split_kwargs=None,
    target_col="WTD",
    id_col="ID",
    date_col="date",
    experiment_name="Yeo-Johnson FNN",
    n_trials=30,
    validation_size=0.2,
    random_state=SEED,
    show_plots=False,
    ncols=3,
    split_group_col="split_group",
    return_split_summaries=True,
):
    split_kwargs = split_kwargs or {}
    df_clean, feature_cols = prepare_tabular_dataframe(df, target_col=target_col, id_col=id_col, date_col=date_col)
    df_train_raw, df_test_raw, skipped_ids = split_fn(df_clean, id_col=id_col, date_col=date_col, **split_kwargs)
    meta_cols = [id_col, date_col] + ([split_group_col] if split_group_col in df_test_raw.columns else [])
    X_train = df_train_raw[feature_cols].to_numpy(dtype=np.float32)
    X_test = df_test_raw[feature_cols].to_numpy(dtype=np.float32)
    y_train = df_train_raw[target_col].to_numpy(dtype=np.float32)
    y_test = df_test_raw[target_col].to_numpy(dtype=np.float32)

    scaler_x = StandardScaler().fit(X_train)
    X_train_s = scaler_x.transform(X_train).astype(np.float32)
    X_test_s = scaler_x.transform(X_test).astype(np.float32)
    transformer, y_train_yj = fit_yj(y_train)

    X_tr, X_val, y_tr, y_val = train_test_split(X_train_s, y_train_yj, test_size=validation_size, random_state=random_state)
    batch_size = min(64, max(1, len(X_tr)))
    model = FNNRegressor(input_dim=X_train_s.shape[1]).to(DL_DEVICE)
    model, history_df = train_fnn_model(
        model,
        DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.float32)), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.float32)), batch_size=batch_size, shuffle=False),
    )

    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_s, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)),
        batch_size=min(64, max(1, len(X_test_s))),
        shuffle=False,
    )
    y_pred = transformer.inverse_transform(predict_fnn(model, test_loader).reshape(-1, 1)).ravel()

    results_df = df_test_raw[meta_cols].copy()
    results_df["actual_WTD"] = y_test
    results_df["predicted_WTD"] = y_pred
    results_df["Model"] = experiment_name
    evaluation_df = pd.DataFrame([{"Model": experiment_name, **evaluate_model(y_test, y_pred)}])

    per_well_df = None
    overall_df = None
    if return_split_summaries and split_group_col in results_df.columns:
        per_well_df, overall_df = summarize_by_split(results_df, id_col=id_col, group_col=split_group_col)
        per_well_df.insert(0, "Model", experiment_name)
        overall_df.insert(0, "Model", experiment_name)

    return {
        "results_df": results_df.sort_values([id_col, date_col]).reset_index(drop=True),
        "evaluation_df": evaluation_df,
        "model": model,
        "transformer": transformer,
        "scaler_x": scaler_x,
        "history": history_df,
        "skipped_ids": skipped_ids,
        "feature_names": feature_cols,
        "per_well_split_df": per_well_df,
        "overall_split_df": overall_df,
    }


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build_sequence_frame(df, feature_cols, target_col="WTD", id_col="ID", date_col="date", seq_len=1):
    rows = []
    seqs = []
    df_use = df.copy().sort_values([id_col, date_col]).reset_index(drop=True)
    meta_cols = [id_col, date_col]
    if "split_group" in df_use.columns:
        meta_cols.append("split_group")
    for well_id, g in df_use.groupby(id_col, sort=False):
        g = g.sort_values(date_col).reset_index(drop=True)
        X = g[feature_cols].to_numpy(dtype=np.float32)
        y = g[target_col].to_numpy(dtype=np.float32)
        for i in range(len(g)):
            start = max(0, i - seq_len + 1)
            seq = X[start:i+1]
            if len(seq) < seq_len:
                pad = np.repeat(seq[:1], seq_len - len(seq), axis=0)
                seq = np.vstack([pad, seq])
            seqs.append(seq)
            row = {id_col: well_id, date_col: g.loc[i, date_col], target_col: y[i]}
            if "split_group" in meta_cols:
                row["split_group"] = g.loc[i, "split_group"]
            rows.append(row)
    return np.asarray(seqs, dtype=np.float32), pd.DataFrame(rows)


def train_lstm_model(model, train_loader, val_loader, epochs=40, patience=8, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = None
    best_val = float("inf")
    wait = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(DL_DEVICE)
            yb = yb.to(DL_DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DL_DEVICE)
                yb = yb.to(DL_DEVICE)
                val_losses.append(criterion(model(xb), yb).item())

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss = float(np.mean(val_losses)) if val_losses else np.nan
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predict_lstm(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DL_DEVICE)
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def run_lstm_workflow(
    df,
    split_fn,
    split_kwargs=None,
    target_col="WTD",
    id_col="ID",
    date_col="date",
    experiment_name="LSTM",
    seq_len=1,
    hidden_size=32,
    num_layers=1,
    dropout=0.1,
    batch_size=32,
    epochs=40,
    patience=8,
    learning_rate=1e-3,
    validation_size=0.2,
    random_state=SEED,
    show_plots=False,
    verbose=False,
    ncols=3,
    split_group_col="split_group",
    return_split_summaries=True,
):
    split_kwargs = split_kwargs or {}
    df_clean, feature_cols = prepare_tabular_dataframe(df, target_col=target_col, id_col=id_col, date_col=date_col)
    df_train_raw, df_test_raw, skipped_ids = split_fn(df_clean, id_col=id_col, date_col=date_col, **split_kwargs)
    meta_cols = [id_col, date_col] + ([split_group_col] if split_group_col in df_test_raw.columns else [])

    train_seq, train_meta = build_sequence_frame(df_train_raw, feature_cols, target_col=target_col, id_col=id_col, date_col=date_col, seq_len=seq_len)
    test_seq, test_meta = build_sequence_frame(df_test_raw, feature_cols, target_col=target_col, id_col=id_col, date_col=date_col, seq_len=seq_len)

    y_train = train_meta[target_col].to_numpy(dtype=np.float32)
    y_test = test_meta[target_col].to_numpy(dtype=np.float32)
    transformer, y_train_yj = fit_yj(y_train)

    X_tr, X_val, y_tr, y_val = train_test_split(train_seq, y_train_yj, test_size=validation_size, random_state=random_state)
    model = LSTMRegressor(input_size=len(feature_cols), hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(DL_DEVICE)
    model, history_df = train_lstm_model(
        model,
        DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)), batch_size=min(batch_size, max(1, len(X_tr))), shuffle=True),
        DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=min(batch_size, max(1, len(X_val))), shuffle=False),
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
    )

    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_seq, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)),
        batch_size=min(batch_size, max(1, len(test_seq))),
        shuffle=False,
    )
    y_pred = transformer.inverse_transform(predict_lstm(model, test_loader).reshape(-1, 1)).ravel()
    results_df = test_meta[meta_cols].copy()
    results_df["actual_WTD"] = y_test
    results_df["predicted_WTD"] = y_pred
    results_df["Model"] = experiment_name
    evaluation_df = pd.DataFrame([{"Model": experiment_name, **evaluate_model(y_test, y_pred)}])

    per_well_df = None
    overall_df = None
    if return_split_summaries and split_group_col in results_df.columns:
        per_well_df, overall_df = summarize_by_split(results_df, id_col=id_col, group_col=split_group_col)
        per_well_df.insert(0, "Model", experiment_name)
        overall_df.insert(0, "Model", experiment_name)

    return {
        "results_df": results_df.sort_values([id_col, date_col]).reset_index(drop=True),
        "evaluation_df": evaluation_df,
        "model": model,
        "transformer": transformer,
        "history": history_df,
        "skipped_ids": skipped_ids,
        "feature_names": feature_cols,
        "per_well_split_df": per_well_df,
        "overall_split_df": overall_df,
    }

# =============================================================================
# XGBOOST OBSERVED-VERSUS-PREDICTED RECONSTRUCTION FIGURE
# =============================================================================

# Reconstruction XGBoost 1:1 panel plot for held-out test data

def rmse_local(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(mean_squared_error(y_true, y_pred))


def kge_local(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2:
        return np.nan
    mean_obs = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    std_obs = np.std(y_true, ddof=1)
    std_pred = np.std(y_pred, ddof=1)
    if std_obs == 0 or np.isnan(std_obs) or mean_obs == 0 or np.isnan(mean_obs):
        return np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = std_pred / std_obs
    beta = mean_pred / mean_obs
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"n": 0, "RMSE": np.nan, "MAE": np.nan, "KGE": np.nan, "NSE": np.nan}
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    return {
        "n": len(y_true),
        "RMSE": rmse_local(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "KGE": kge_local(y_true, y_pred),
        "NSE": np.nan if denom == 0 else 1 - np.sum((y_true - y_pred) ** 2) / denom,
    }


def plot_one_to_one_by_id_panel(
    df1,
    id_col="ID",
    obs_col="actual_WTD",
    pred_col="predicted_WTD",
    location_col="Location",
    outdir=REVIEWER_OUTPUT_DIR,
    filename="xgboost_reconstruction_test_one_to_one_panel.png",
    ncols=3,
    dpi=600,
    fontsize=16,
    alpha=0.7,
    panel_width=6,
    panel_height=6,
):
    df = df1.copy()
    sort_cols = [id_col]
    if "date" in df.columns:
        sort_cols.append("date")
    elif "Date" in df.columns:
        sort_cols.append("Date")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    ids = df[id_col].dropna().unique()
    n_ids = len(ids)
    nrows = math.ceil(n_ids / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(panel_width * ncols, panel_height * nrows))
    if n_ids == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(-1)

    for i, one_id in enumerate(ids):
        ax = axes[i]
        df_id = df[df[id_col] == one_id].copy()
        if location_col in df_id.columns and df_id[location_col].notna().any():
            loc = str(df_id[location_col].dropna().iloc[0])
            title = f"{one_id} ({loc})"
        else:
            title = str(one_id)

        df_id = df_id[[obs_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(df_id) == 0:
            ax.set_visible(False)
            continue

        metrics = calc_metrics(df_id[obs_col], df_id[pred_col])
        obs = df_id[obs_col].values
        pred = df_id[pred_col].values
        xy_min = min(np.min(obs), np.min(pred))
        xy_max = max(np.max(obs), np.max(pred))

        ax.scatter(obs, pred, s=18, alpha=alpha, color="blue")
        ax.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--", linewidth=1.8, color="red")
        ax.set_xlim(xy_min, xy_max)
        ax.set_ylim(xy_min, xy_max)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(title, fontsize=fontsize, fontweight="bold")
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
            0.05,
            0.95,
            text_str,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=fontsize - 3,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
        )

    for j in range(n_ids, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    outfile = Path(outdir) / filename
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    #plt.show()
    plt.close(fig)
    print(f"Saved panel figure to: {outfile}")
    return outfile


xgb_results_df = classical_suite_all["combined_results"].copy()
if "Model" in xgb_results_df.columns:
    xgb_results_df = xgb_results_df[xgb_results_df["Model"].astype(str).str.contains("XGB", case=False, na=False)].copy()

if "predicted_WTD" not in xgb_results_df.columns:
    candidate_pred_cols = [c for c in xgb_results_df.columns if c.lower().startswith("predicted")]
    if len(candidate_pred_cols) == 1:
        xgb_results_df = xgb_results_df.rename(columns={candidate_pred_cols[0]: "predicted_WTD"})
    elif "predicted_WTD_XGB" in xgb_results_df.columns:
        xgb_results_df = xgb_results_df.rename(columns={"predicted_WTD_XGB": "predicted_WTD"})

xgboost_reconstruction_panel_path = plot_one_to_one_by_id_panel(
    df1=xgb_results_df,
    id_col="ID",
    obs_col="actual_WTD",
    pred_col="predicted_WTD",
    location_col="Location",
    outdir=REVIEWER_OUTPUT_DIR,
    filename="xgboost_reconstruction_test_one_to_one_panel.png",
    ncols=3,
)

print(f"Reconstruction XGBoost 1:1 panel saved to: {xgboost_reconstruction_panel_path}")

# =============================================================================
# TRAIN AND EVALUATE FEEDFORWARD NEURAL NETWORK
# =============================================================================

fnn_output = run_yj_fnn_workflow(
    df_model_first_last_wtd,
    split_fn=split_edge_random_by_id,
    split_kwargs={
        "edge_fraction": EDGE_FRACTION,
        "middle_random_fraction": MIDDLE_RANDOM_FRACTION,
        "min_records_per_id": MIN_RECORDS_PER_ID,
        "random_state": SEED,
    },
    target_col="WTD",
    id_col="ID",
    date_col="date",
    experiment_name="Yeo-Johnson FNN",
    validation_size=VALIDATION_SIZE,
    random_state=SEED,
    show_plots= True,
    ncols=3,
    split_group_col="split_group",
    return_split_summaries=True,
)

# =============================================================================
# TRAIN AND EVALUATE LSTM
# =============================================================================

lstm_output = run_lstm_workflow(
    df_model_first_last_wtd,
    split_fn=split_edge_random_by_id,
    split_kwargs={
        "edge_fraction": EDGE_FRACTION,
        "middle_random_fraction": MIDDLE_RANDOM_FRACTION,
        "min_records_per_id": MIN_RECORDS_PER_ID,
        "random_state": SEED,
    },
    target_col="WTD",
    id_col="ID",
    date_col="date",
    experiment_name="LSTM",
    seq_len=7,
    hidden_size=128,
    num_layers=1,
    dropout=0.10,
    batch_size=32,
    epochs=1000,
    patience=20,
    learning_rate=1e-3,
    validation_size=VALIDATION_SIZE,
    random_state=SEED,
    show_plots=True,
    verbose=False,
    ncols=3,
    split_group_col="split_group",
    return_split_summaries=True,
)

# =============================================================================
# COMBINE DEEP-LEARNING RESULTS
# =============================================================================

deep_results = pd.concat(
    [
        fnn_output["results_df"].copy(),
        lstm_output["results_df"].copy(),
    ],
    ignore_index=True,
)

deep_table_2_df = pd.concat(
    [
        fnn_output["evaluation_df"].assign(Model="FNN"),
        lstm_output["evaluation_df"].assign(Model="LSTM"),
    ],
    ignore_index=True,
)

deep_table_s2_df = pd.concat(
    [
        fnn_output["per_well_split_df"].assign(Model="FNN"),
        lstm_output["per_well_split_df"].assign(Model="LSTM"),
    ],
    ignore_index=True,
)

deep_table_s3_df = pd.concat(
    [
        fnn_output["overall_split_df"].assign(Model="FNN"),
        lstm_output["overall_split_df"].assign(Model="LSTM"),
    ],
    ignore_index=True,
)

deep_ks_overall_df = ks_test_observed_vs_predicted(deep_results, group_cols=["Model"]).sort_values("Model").reset_index(drop=True)
deep_ks_by_well_df = ks_test_observed_vs_predicted(deep_results, group_cols=["Model", "ID"]).sort_values(["Model", "ID"]).reset_index(drop=True)
deep_ks_by_split_df = ks_test_observed_vs_predicted(deep_results, group_cols=["Model", "split_group"]).sort_values(["Model", "split_group"]).reset_index(drop=True)


deep_table_2_df.to_csv(OUTPUT_DIR / "Table_2_deep_models.csv", index=False)
deep_table_s2_df.to_csv(OUTPUT_DIR / "Table_S2_deep_models_by_well.csv", index=False)
deep_table_s3_df.to_csv(OUTPUT_DIR / "Table_S3_deep_models_by_split.csv", index=False)

# =============================================================================
# KOLMOGOROV-SMIRNOV SUMMARY WITHOUT RETRAINING
# =============================================================================

# Reconstruction KS summary without retraining

reconstruction_pred_df = pd.concat(
    [
        classical_suite_all["combined_results"][[c for c in ["Model", "actual_WTD", "predicted_WTD", "split_group"] if c in classical_suite_all["combined_results"].columns]].copy(),
        deep_results[[c for c in ["Model", "actual_WTD", "predicted_WTD", "split_group"] if c in deep_results.columns]].copy(),
    ],
    ignore_index=True,
)

reconstruction_pred_df = reconstruction_pred_df.replace([np.inf, -np.inf], np.nan)
reconstruction_pred_df = reconstruction_pred_df.dropna(subset=["Model", "actual_WTD", "predicted_WTD"]).copy()


def ks_summary_for_grouped_predictions(df, group_cols, actual_col="actual_WTD", pred_col="predicted_WTD"):
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if len(g) < 2:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        ks_stat, ks_p = ks_2samp(g[actual_col].to_numpy(dtype=float), g[pred_col].to_numpy(dtype=float))
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update({"N": len(g), "KS": ks_stat, "KS_p_value": ks_p})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)

reconstruction_ks_df = ks_summary_for_grouped_predictions(reconstruction_pred_df, ["Model"])
print("Reconstruction KS by model:")

if "split_group" in reconstruction_pred_df.columns:
    reconstruction_ks_split_df = ks_summary_for_grouped_predictions(reconstruction_pred_df, ["Model", "split_group"])
    print("Reconstruction KS by model and split:")
else:
    reconstruction_ks_split_df = None

reconstruction_ks_df.to_csv(OUTPUT_DIR / "reconstruction_ks_by_model.csv", index=False)
if reconstruction_ks_split_df is not None:
    reconstruction_ks_split_df.to_csv(OUTPUT_DIR / "reconstruction_ks_by_model_and_split.csv", index=False)

# =============================================================================
# COMBINE CLASSICAL AND DEEP-LEARNING TABLES
# =============================================================================

classical_suite = classical_suite_all.copy()
combined_table_2_df = pd.concat([classical_suite["table_2_df"], deep_table_2_df], ignore_index=True)
combined_table_s2_df = pd.concat([classical_suite["table_s2_df"], deep_table_s2_df], ignore_index=True)
combined_table_s3_df = pd.concat([classical_suite["table_s3_df"], deep_table_s3_df], ignore_index=True)
combined_ks_overall_df = pd.concat([classical_suite["ks_overall_df"], deep_ks_overall_df], ignore_index=True)
combined_ks_by_well_df = pd.concat([classical_suite["ks_by_well_df"], deep_ks_by_well_df], ignore_index=True)
combined_ks_by_split_df = pd.concat([classical_suite["ks_by_split_df"], deep_ks_by_split_df], ignore_index=True)


combined_table_2_df.to_csv(OUTPUT_DIR / "Table_2_combined_models.csv", index=False)
combined_table_s2_df.to_csv(OUTPUT_DIR / "Table_S2_combined_models_by_well.csv", index=False)
combined_table_s3_df.to_csv(OUTPUT_DIR / "Table_S3_combined_models_by_split.csv", index=False)
combined_ks_overall_df.to_csv(OUTPUT_DIR / "KS_overall_combined_models.csv", index=False)
combined_ks_by_well_df.to_csv(OUTPUT_DIR / "KS_by_well_combined_models.csv", index=False)
combined_ks_by_split_df.to_csv(OUTPUT_DIR / "KS_by_split_combined_models.csv", index=False)
