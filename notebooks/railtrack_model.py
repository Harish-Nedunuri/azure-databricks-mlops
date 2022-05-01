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


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Build classifier
# MAGIC 
# MAGIC A machine learning model will be built to predict the liklihood of Railway attrition.

# COMMAND ----------
#

# Define classifer pipeline
def make_classifer_pipeline(params):
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder())]
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, columns_numeric),
            ("categorical", categorical_transformer, columns_categorical)
        ]
    )

    classifer_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(**params, n_jobs=-1))
    ])

    return classifer_pipeline

# COMMAND ----------


# Define objective function
def hyperparameter_tuning(params):
    mlflow.sklearn.autolog(silent=True)

    with mlflow.start_run(nested=True):
        # Train and model
        estimator = make_classifer_pipeline(params)
        estimator = estimator.fit(X_train, y_train.values.ravel())
        y_predict_proba = estimator.predict_proba(X_test)
        auc_score = roc_auc_score(y_test, y_predict_proba[:, 1])

        # Log artifacts
        signature = infer_signature(X_train, y_predict_proba[:, 1])
        mlflow.sklearn.log_model(estimator, "model", signature=signature)
        mlflow.log_metric("testing_auc", auc_score)

        return {"loss": -auc_score, "status": STATUS_OK}

# COMMAND ----------


# Define search space
search_space = {
    "n_estimators": hp.choice("n_estimators", range(100, 1000)),
    "max_depth": hp.choice("max_depth", range(1, 20)),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
}

# Start model training run
with mlflow.start_run(run_name="Railway-attrition-classifier") as run:
    # Hyperparameter tuning
    best_params = fmin(
        fn=hyperparameter_tuning,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
    )

    # End run
    mlflow.end_run()