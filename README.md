# 🌿 WetlandHydroML

**Machine-learning reconstruction of daily water table depth and assessment of wetland hydrology in forested wetland systems**

WetlandHydroML is a reproducible Python workflow for reconstructing continuous daily water table depth (WTD) from sparse groundwater observations and evaluating wetland hydrology. The repository integrates hydroclimatic data processing, classical machine learning, deep learning, climate analysis, reconstructed-WTD analysis, and wetland-hydrology assessment.


---

## 🔬 Project workflow

```text
Raw groundwater + hydroclimatic data
                │
                ▼
        Data preparation
                │
                ▼
   Feature engineering and screening
                │
                ▼
 ┌──────────────┴───────────────┐
 │                              │
 ▼                              ▼
Classical ML                Deep learning
LR / RF / XGBoost           FNN / LSTM
 │                              │
 └──────────────┬───────────────┘
                ▼
      Daily WTD reconstruction
                │
                ▼
      Model performance analysis
                │
                ▼
       Climate/WTD trend analysis
                │
                ▼
   Wetland-hydrology assessment
```

---

## 📁 Repository structure

```text
WetlandHydroML/
│
├── 📂 Data/
│   ├── 📂 Raw_data/
│   │   └── Original/input datasets
│   │
│   └── 📂 Output_data/
│       └── Processed data, reconstructed WTD, and analysis outputs
│
├── 📂 Scripts/
│   ├── analyze_climate.py
│   ├── analyze_reconstructed_wtd.py
│   ├── data_loader.py
│   ├── WTD_functions.py
│   └── wtd_reconstruction_ml_dl.py
│
└── 📄 README.md
```

---

## 💧 Data organization

The repository separates original data from generated products:

```text
Data/
├── Raw_data/
└── Output_data/
```

`Raw_data` contains the source datasets required by the workflow.

`Output_data` stores processed datasets and outputs generated during reconstruction and subsequent analyses, including reconstructed WTD and summary products.


---

## 🚀 Recommended scripts execution order

Run the scripts from the project environment in the following order:

```text
1. Scripts/data_loader.py
          ↓
2. Scripts/wtd_reconstruction_ml_dl.py
          ↓
3. Scripts/analyze_reconstructed_wtd.py
     
```

`data_loader.py` and `WTD_functions.py` primarily function as supporting modules and are imported by the analysis scripts. Depending on the configuration of the final workflow, `data_loader.py` may not need to be executed independently before running the modeling script.
analyze_climate.py is not part of the sequential WTD reconstruction workflow. It is a supplementary script used to generate the climate-analysis figures presented in the manuscript and can be run independently when those figures are needed.
For example:

```bash
python Scripts/wtd_reconstruction_ml_dl.py
python Scripts/analyze_reconstructed_wtd.py
python Scripts/analyze_climate.py
```

---

## 🛠️ Python environment

The project uses common scientific-computing, machine-learning, statistical, and deep-learning libraries, including:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
optuna
torch
scipy
statsmodels
pymannkendall
```


---

