import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
def plot_correlation(
    df,
    target_name="WTD",
    saveas="daily_feature_correlation.png",
    fontsize=12,
):

    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.loc[:, ~numeric_df.columns.duplicated()].copy()

    if target_name not in numeric_df.columns:
        raise ValueError(f"'{target_name}' is not a numeric column in the dataframe.")

    corr_mat = numeric_df.corr()

    corr_series = corr_mat[target_name].drop(labels=[target_name], errors="ignore")
    corr_series = corr_series.dropna()

    corr_df = corr_series.reset_index()
    corr_df.columns = ["Feature", "Correlation"]
    corr_df = corr_df.sort_values(
        "Correlation",
        key=lambda s: s.abs(),
        ascending=False
    )

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # ---- ONLY LEFT & BOTTOM SPINES ----
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # -----------------------------------

    colors = ["red" if x < 0 else "blue" for x in corr_df["Correlation"]]
    plt.barh(
        corr_df["Feature"],
        corr_df["Correlation"],
        color=colors,
        alpha=0.7,
    )

    plt.xlabel("Correlation Coefficient", fontsize=fontsize, fontweight="bold")
    plt.xticks(fontsize=fontsize, fontweight="bold")
    plt.yticks(fontsize=fontsize, fontweight="bold")
    plt.axvline(x=0, color="black", linestyle="-", alpha=0.7)

    plt.grid(False)

    # Legend with transparent background
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="blue", alpha=0.7, label="Positive Correlation"),
        Patch(facecolor="red",  alpha=0.7, label="Negative Correlation"),
    ]
    leg = plt.legend(handles=legend_elements, fontsize=fontsize - 2)
    leg.get_frame().set_alpha(0)
    leg.get_frame().set_facecolor("none")
    leg.get_frame().set_edgecolor("none")

    for t in leg.get_texts():
        t.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(saveas, dpi=600, transparent=True)
    plt.show()


def is_wetland_year(df_year):
    """
    Determine if a given year's data qualifies as a wetland year.
    Condition: |WTD_Predicted| <= 30 for at least 5% of days between April–October.

    Input:
        df_year : DataFrame for a single ID and Year.
    Returns:
        True or False
    """

    # Filter April–October (Months 4 to 10)
    df_gs = df_year[(df_year['Month'] >= 4) & (df_year['Month'] <= 10)]

    if df_gs.empty:
        return False  # no data

    # Condition
    cond = df_gs['WTD_Predicted'].abs() <= 30

    pct = cond.mean() * 100  # percentage

    return pct >= 5  # threshold


def apply_wetland_flag(df):
    """
    Apply wetland-year classification to each ID-year.
    Returns the original df with a new column 'wetland_year'.
    """

    df = df.copy()

    # Compute the wetland flag for each ID-Year
    wetland_map = (
        df.groupby(['ID', 'Year'])
        .apply(is_wetland_year)
        .reset_index(name='wetland_year')
    )

    # Merge back onto original data
    df = df.merge(wetland_map, on=['ID', 'Year'], how='left')

    return df


def classify_wetland_years(df):
    """
    Returns a summary dataframe with:
        ID, Year, wetland_year, avg_WTD_predicted_gs
    Wetland condition:
        |WTD_Predicted| <= 30 for at least 5% of days between April–October.
    """

    results = []

    # Loop over each ID-Year group
    for (id_, year), group in df.groupby(['ID', 'Year']):

        # Select April–October
        df_gs = group[(group['Month'] >= 4) & (group['Month'] <= 10)]

        if df_gs.empty:
            wetland_flag = False
            avg_wtd = np.nan
        else:
            # Compute wetland condition
            cond = df_gs['WTD_Predicted'].abs() <= 30
            pct = cond.mean() * 100

            wetland_flag = pct >= 5

            # Growing-season mean
            avg_wtd = df_gs['WTD_Predicted'].mean()

        results.append({
            'ID': id_,
            'Year': year,
            'wetland_year': wetland_flag,
            'avg_WTD_predicted_gs': avg_wtd
        })

    return pd.DataFrame(results)


def count_wetland_years(df_summary):
    """
    Input:
        df_summary = output of classify_wetland_years(df)
                      containing columns:
                      'ID', 'Year', 'wetland_year', 'avg_WTD_predicted_gs'
    Output:
        DataFrame with:
        'ID', 'num_wetland_years'
    """

    result = (
        df_summary.groupby('ID')['wetland_year']
        .sum()  # True counts as 1
        .reset_index(name='num_wetland_years')
    )

    return result


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
    outdir: str = r".\AGU2025",
    fontsize: int = 24,
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

        fig, ax = plt.subplots(figsize=(20, 8))
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
        plt.show()
        plt.close(fig)

    print(f"✅ All transparent figures saved to: {outdir}")

# updated

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor  # pip install xgboost if needed

def data_split (df, target_col = "WTD"):

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Separate numeric and categorical
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns

    X_numeric = X[numeric_cols].fillna(X[numeric_cols].median())
    X_categorical = X[categorical_cols].fillna("MISSING")
    X_categorical = pd.get_dummies(X_categorical, drop_first=True)

    # Concatenate processed features
    X_processed = pd.concat([X_numeric, X_categorical], axis=1)

    # Train / test split (used for BOTH models)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Base model (some reasonable defaults)
def xgb_hypertune(df):
    X_train, X_test, y_train, y_test = data_split (df, target_col = "WTD")

    xgb_base = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",        # use "gpu_hist" if you have a GPU
        random_state=42
    )

    # Search space (you can adjust ranges later)
    param_distributions = {
        "n_estimators":       [200, 300, 400, 500, 600, 800],
        "learning_rate":      [0.01, 0.03, 0.05, 0.07, 0.1],
        "max_depth":          [3, 4, 5, 6, 8, 10],
        "min_child_weight":   [1, 3, 5, 7],
        "subsample":          [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":   [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma":              [0, 0.1, 0.2, 0.3, 0.5],
        "reg_alpha":          [0, 0.001, 0.01, 0.1, 1.0],
        "reg_lambda":         [0.1, 0.5, 1.0, 2.0, 5.0],
    }

    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=40,                         # number of random combos to try
        scoring="neg_root_mean_squared_error",
        cv=3,                              # 3-fold CV
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )

    # Fit search on your training data
    random_search.fit(X_train, y_train)

    print("Best params:", random_search.best_params_)
    print("Best CV RMSE:", -random_search.best_score_)
    return random_search

