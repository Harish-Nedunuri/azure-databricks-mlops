# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Import dependencies and pre-process data data
# MAGIC 
# MAGIC The dataset will be sourced from the repo, however it is not recommended to upload any dataset into a version control repository
# COMMAND ----------

import os
from pprint import pprint
import joblib

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

import numpy as np
import pandas as pd

from alibi_detect.cd import TabularDrift
from alibi_detect.od.isolationforest import IForest
from alibi_detect.utils.saving import load_detector, save_detector

from hyperopt import STATUS_OK, fmin, hp, tpe

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Read dataset

df_railtrack = pd.read_csv("../data/Track_degradation_data_approved_public.csv")
df_railtrack.head()


# Define target column
columns_target = ["Track_degradation"]

# Define categorical feature columns
columns_categorical = ["tilting_wagon", "track_fouling", "track_misalignment",
                       "rail_material_inhomogenous"]

# Define numeric feature columns
columns_numeric = ["Track_age_yrs", "Load_cycles_daily", "Load_amplitude_avg_N", "Balast_thickness_inmm", "track_confining_pressure_kPa", 
                   "curve_radius_mm", "fracture_strength_level","fracture_strength_value","humidty_percent" ,"corrosion_index_iso","speed_mph","Accumulated_tonnage_kgpmeter","rail_fall_in_mm","track_quality_index","inspection_interval_yrs"]

# Change data types of features
df_railtrack[columns_target] = df_railtrack[columns_target].replace(
    {"True": 1, "False": 0}).astype("str")
df_railtrack[columns_categorical] = df_railtrack[columns_categorical].astype(
    "str")
df_railtrack[columns_numeric] = df_railtrack[columns_numeric].astype("float")

# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    df_railtrack[columns_categorical + columns_numeric], df_railtrack[columns_target], test_size=0.20, random_state=2022)


