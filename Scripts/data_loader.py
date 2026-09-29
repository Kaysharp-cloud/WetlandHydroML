import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df_wtd_ws77 = pd.read_csv(r"..\Data\Raw_data\Well levels_ws77.csv",usecols=['Location', 'Instr ID', 'Date',  'Depth cm b'])
df_wtd_ws78 = pd.read_csv(r"..\Data\Raw_data\Well levels_ws78.csv", usecols = ['Location', 'Instr_ID', 'Date_', 'Depth_cm_b'])
df_wtd_ws80 = pd.read_csv(r"..\Data\Raw_data\Well levels_80.csv", usecols = ['Instr_ID', 'Date_Time', 'Water Table Depth BGS (cm)'])


df_wtd_ws80.columns = ["ID","DATE","WTD(cm)"]
df_wtd_ws80["Location"] = "WS80"

df_wtd_ws77.columns = ["Location","ID","DATE","WTD(cm)"]

df_wtd_ws78.columns = ["Location","ID","DATE","WTD(cm)"]
df_wtd_ws77["DATE"] = pd.to_datetime(df_wtd_ws77["DATE"])
df_wtd_ws78["DATE"] = pd.to_datetime(df_wtd_ws78["DATE"])
df_wtd_ws80["DATE"] = pd.to_datetime(df_wtd_ws80["DATE"])
df_sef = pd.concat([df_wtd_ws77, df_wtd_ws78, df_wtd_ws80], axis=0)

# Keep only the date part (no time)
df_sef["DATE"] = df_sef["DATE"].dt.normalize()   # or .dt.date if you want Python date objects

df_sef_info =  pd.read_csv(r"..\Data\Raw_data\Well_id_location.csv")
df_sef_daymet = pd.read_csv(r"..\Data\Raw_data\SEF_DAYMET_All_var.csv")
df_sef_gldas = pd.read_csv(r"..\Data\Raw_data\SEF_GLDAS_ALL_STATIONS_2003_2024.csv")
# r"C:\Users\adeba\OneDrive\Documents\Datascience\Full_WTD_project\data_intermediate\data_GEE\SEF_DAYMET_All_var.csv"
df_sef_input = df_sef_daymet.merge(df_sef_gldas, on=["ID","date","Lat","Lon"], how="inner")

# needed_cols = ['date', 'prcp', 'tmax', 'tmin', 'ID', 'Lat', 'Lon', 'ESoil_tavg'
#        , 'Evap_tavg', 'GWS_tavg',  'Qg_tavg','Swnet_tavg',
#         'Qle_tavg', 'Qs_tavg', 'Qsb_tavg',  'SWE_tavg','Lwnet_tavg',
#        'SnowDepth_tavg', 'SnowT_tavg', 'SoilMoist_P_tavg', 'SoilMoist_RZ_tavg',
#        'SoilMoist_S_tavg', 'Swnet_tavg', 'TWS_tavg']

needed_cols = ['date','ID', 'Lat', 'Lon', 'prcp', "dayl","srad","vp",'tmax', 'tmin',  'ESoil_tavg'
       , 'Evap_tavg', 'GWS_tavg',  'Qg_tavg',
         'Qs_tavg', 'Qsb_tavg',
         'SoilMoist_RZ_tavg',
       'SoilMoist_S_tavg']
df_sef_needed_input = df_sef_input[needed_cols]
def rename_climate_columns(df):
    """
    Rename climate dataframe columns from long GLDAS-style names
    to short, consistent names.

    Rules:
    - Keep base columns unchanged: ['date','prcp','tmax','tmin','ID','Lat','Lon']
    - Remove suffix '_tavg'
    - Convert SoilMoist_X → SM_X
    """

    base_cols = ['date', 'prcp', 'tmax', 'tmin', 'ID', 'Lat', 'Lon']
    rename_map = {}

    for col in df.columns:
        # Skip base columns
        if col in base_cols:
            continue

        new_col = col

        # Remove "_tavg"
        if new_col.endswith("_tavg"):
            new_col = new_col.replace("_tavg", "")

        # Convert SoilMoist_* → SM_*
        if new_col.startswith("SoilMoist_"):
            new_col = new_col.replace("SoilMoist_", "SM_")

        rename_map[col] = new_col

    return df.rename(columns=rename_map)
df_sef_needed_input = rename_climate_columns(df_sef_needed_input)
df_sef_needed_input.rename(columns={"ESoil":"Evap_soil"}, inplace= True)
df_sef_needed_input.columns
df_sef_needed_input.date = pd.to_datetime(df_sef_needed_input.date)
df_sef_needed_input["DOY"] = df_sef_needed_input.date.dt.dayofyear
# Parse mixed date/time strings
df_sef["DATE"] = pd.to_datetime(df_sef["DATE"].astype(str).str.strip(), errors="coerce")

# Keep only the date part (no time)
df_sef["DATE"] = df_sef["DATE"].dt.normalize()   # or .dt.date if you want Python date objects
df_input_agu = df_sef_needed_input.copy()
df_input_agu["ID"] = df_input_agu["ID"].str.replace(
    "well",   
    "Well",
    )
df_input_agu["ID"] = df_input_agu["ID"].str.replace(r"\s+", " ", regex=True).str.strip()
df_sef["ID"]       = df_sef["ID"].str.replace(r"\s+", " ", regex=True).str.strip()
df_sef.rename(columns={"DATE": "date"}, inplace=True)
df_needed_data = df_sef.merge(df_input_agu, on=["ID","date"], how="inner")

df_sef.rename(columns={"DATE": "date"}, inplace=True)
df_needed_data = df_sef.merge(df_input_agu, on=["ID","date"], how="inner")
df_wtd_input_nolag = df_needed_data.drop(columns=["Location","date","Lat","Lon"])
df_wtd_input_nolag.rename(columns={"WTD(cm)": "WTD"}, inplace=True)
df_wtd_input_nolag.dropna(inplace = True)

daily_lag_col = [ 'prcp', 'tmax', 'tmin',"srad","vp","dayl" ]
def lagged_function(df, lag_columns, length= 8):
    lag_numbers=range(1,length)
    #lag_columns= ["WTD"]
    df_features_with_lag = df.copy()

    for col in lag_columns:
        for a in lag_numbers:
            df_features_with_lag[f"{col}_{a}"]= df_features_with_lag[col].shift(a)
    return df_features_with_lag
df_input_updated = lagged_function(df_input_agu, daily_lag_col, length= 7)
# df_input_updated.interpolate(inplace= True)
df_wtd_input_lag = df_sef.merge(df_input_updated, on= ["date","ID"], how="inner")
df_wtd_input_lag.dropna(inplace= True)
df_wtd_input_lag_date = df_wtd_input_lag.copy()
df_wtd_input_lag.rename(columns={"WTD(cm)": "WTD"}, inplace=True)
df_wtd_input_lag = df_wtd_input_lag.drop(columns=["Location","date","Lat","Lon"])
