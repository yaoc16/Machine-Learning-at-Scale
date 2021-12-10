# Databricks notebook source
# Set up data access:
blob_container = "w261team8rocks"
storage_account = "dataguru"
secret_scope = "w261-team8"
secret_key = "cloudblob"
blob_url = f"wasbs://w261team8rocks@dataguru.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# SAS Token
spark.conf.set(
  f"fs.azure.sas.w261team8rocks.dataguru.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

"""
Gridsearch with time-series cross-validation.
Author: Toby Petty
Date: 2021-11-21
"""

import datetime
import os
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql.functions import to_timestamp, when
import random
from sklearn.model_selection import ParameterGrid
import string
import time


blob_url = f"wasbs://w261team8rocks@dataguru.blob.core.windows.net"


class SparkGridSearchTimeSeriesCV:
    """Custom class for time-series cross-validation with GridSearch."""

    def __init__(self, model_class, train, n_cv: int,
                 numeric_features: list, categorical_features: list,
                 one_hot_encode_features: list = None,
                 grid_params: dict = None, static_params: dict = None,
                 target: str = "DEP_DEL15", class_weights: bool = True,
                 scale_features: bool = False, handle_invalid: str = "error"):
        """Time-series cross-validated, parameter gridsearch.

        Args:
            model_class: Spark estimator class.
            train: full training dataframe.
            n_cv: number of cross-validation splits. Each combination
                of parameters will train n_cv-1 models.
            numeric_features: list of numerical feature column names.
            categorical_features: list of categorical feature column names.
            one_hot_encode_features: subset of `categorical_features` which need
                to be one-hot-encoded.
            target: target feature column name.
            grid_params: parameters to grid search.
            static_params: additional model parameters which will be the
                same in every instance of the estimator.
            class_weights: whether or not to re-weight classes.
            scale_features: whether to scale numeric features to mu=0, sigma=1.
            handle_invalid: arg passed to Spark's feature classes. Can be
                either `error` or `keep`.
        """
        self.model_class = model_class
        self.n_cv = n_cv
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        if one_hot_encode_features is None:
            one_hot_encode_features = list()
        self.one_hot_encode_features = one_hot_encode_features
        assert set(self.one_hot_encode_features).issubset(set(self.categorical_features))
        if grid_params is None:
            grid_params = dict()
        self.grid_params = grid_params
        self.param_grid = ParameterGrid(self.grid_params)
        if static_params is None:
            static_params = dict()
        self.static_params = static_params
        self.target = target
        self.class_weights = class_weights
        if self.class_weights:
            self.static_params["weightCol"] = "weight"
        self.scale_features = scale_features
        self.hi = handle_invalid

        # Add the target col and date col for cross-validation:
        train = train.withColumn("target", train[target])
        train = train.withColumn("CV_DATE", to_timestamp(train.FL_DATE, "yyyy-MM-dd"))
        self.train = train

        # Calculate date cut-offs for cross-validation:
        dates_in_train = self.train.select("CV_DATE").distinct().collect()
        dates = sorted([row["CV_DATE"] for row in dates_in_train])
        n_dates = len(dates)
        split_size = n_dates // n_cv
        cv_splits = list()
        train_start, train_end = 0, split_size
        test_start, test_end = split_size, split_size * 2
        for i in range(n_cv-1):
            split_train = dates[train_start: train_end]
            split_test = dates[test_start: test_end]
            cv_splits.append((split_train, split_test))
            train_end += split_size
            test_start += split_size
            test_end += split_size
        self.cv_splits = cv_splits

        # Dict to store all results:
        self.results = dict()

    def run(self):
        # Generate unique random name for aggregating intermediate results:
        self.run_uuid = "".join(random.choices(string.ascii_uppercase, k=6))  # NOQA.

        for cv, (train_split, test_split) in enumerate(self.cv_splits, start=1):

            print(f"CV iteration {cv}:")

            self.results[cv] = dict()

            # Get the subset of training data:
            train_min_date, train_max_date = min(train_split), max(train_split)
            train_subset = self.train.filter(
                (self.train["CV_DATE"] >= train_min_date) &
                (self.train["CV_DATE"] <= train_max_date)
            ).cache()
            print(f"  Train: {train_min_date.strftime('%b %d %Y')} - {train_max_date.strftime('%b %d %Y')}")
            self.results[cv]["train_min_date"] = train_min_date
            self.results[cv]["train_max_date"] = train_max_date

            # Get the subset of test data:
            test_min_date, test_max_date = min(test_split), max(test_split)
            test_subset = self.train.filter(
                (self.train["CV_DATE"] >= test_min_date) &
                (self.train["CV_DATE"] <= test_max_date)
            ).cache()
            print(f"  Test:  {test_min_date.strftime('%b %d %Y')} - {test_max_date.strftime('%b %d %Y')}")
            self.results[cv]["test_min_date"] = test_min_date
            self.results[cv]["test_max_date"] = test_max_date

            # Add weight column:
            if self.class_weights:
                class_weights = train_subset.groupBy(self.target).count().collect()
                not_delayed = [r["count"] for r in class_weights if r[self.target] == 0][0]
                delayed = [r["count"] for r in class_weights if r[self.target] == 1][0]
                ratio = not_delayed / delayed
                train_subset = train_subset.withColumn("weight",
                                                       when(train_subset[self.target] > 0, ratio).otherwise(1))

            # Iterate through the parameter search space:
            self.results[cv]["models"] = dict()
            for i, parameters in enumerate(list(self.param_grid)):

                t1 = time.time()

                self.results[cv]["models"][i] = dict()
                self.results[cv]["models"][i]["parameters"] = parameters
                print(f"    Model {i+1}: {parameters}")

                # Create the pipeline:
                cat_features_ohe = list(set(self.categorical_features) & set(self.one_hot_encode_features))
                cat_features = list(set(self.categorical_features) - set(self.one_hot_encode_features))

                pipeline_stages, final_features = list(), list()

                # One-Hot-Encoded categorical features:
                if len(cat_features_ohe):
                    ix_output_cols = [f"{c}_ix" for c in cat_features_ohe]
                    ohe_ix = StringIndexer(inputCols=cat_features_ohe, outputCols=ix_output_cols, handleInvalid=self.hi)
                    ohe_output_cols = [f"{c}_ohe" for c in cat_features_ohe]
                    ohe = OneHotEncoder(inputCols=ix_output_cols, outputCols=ohe_output_cols, handleInvalid=self.hi)
                    pipeline_stages += [ohe_ix, ohe]
                    final_features += ohe_output_cols

                # Other categorical features, not One-Hot-Encoded:
                if len(cat_features):
                    cat_output_cols = [f"{c}_ix" for c in cat_features]
                    cat_ix = StringIndexer(inputCols=cat_features, outputCols=cat_output_cols, handleInvalid=self.hi)
                    pipeline_stages.append(cat_ix)
                    final_features += cat_output_cols

                # Numeric features:
                if len(self.numeric_features):
                    num_va = VectorAssembler(inputCols=self.numeric_features, outputCol="num_features",
                                             handleInvalid=self.hi)
                    pipeline_stages.append(num_va)
                    if self.scale_features:  # Apply standard scaling:
                        scaler = StandardScaler(inputCol="num_features", outputCol="scaled_num_features",
                                                withMean=True, withStd=True)
                        pipeline_stages.append(scaler)
                        final_features.append("scaled_num_features")
                    else:
                        final_features.append("num_features")

                # Create final features:
                self.results[cv]["models"][i]["final_features"] = final_features
                feature_assembler = VectorAssembler(inputCols=final_features, outputCol="features")
                pipeline_stages.append(feature_assembler)

                # Create the model instance:
                model = self.model_class(labelCol="target", featuresCol="features", **parameters, **self.static_params)
                self.results[cv]["models"][i]["model"] = model

                # Create the full pipeline:
                pipeline = Pipeline(stages=pipeline_stages + [model]).fit(train_subset)

                # Make the test set predictions:
                test_pred = pipeline.transform(test_subset)

                # Compute the confusion matrix:
                spark_cm = test_pred.groupBy("prediction").pivot("target").count().collect()
                pd_cm = pd.DataFrame([r.asDict() for r in spark_cm]).set_index("prediction")
                pd_cm.columns.name = "target"
                self.results[cv]["models"][i]["conf_matrix"] = pd_cm

                # Calculate time taken:
                self.results[cv]["models"][i]["training_time"] = time.time() - t1

                # Save the intermediate results in case the program crashes later:
                self.save_results(verbose=False)

            train_subset.unpersist()
            test_subset.unpersist()

    @property
    def results_df(self):
        """Combine all results into easy to read pandas.DataFrame."""
        rows = list()
        for k1, v1 in self.results.items():
            for k2, v2 in v1["models"].items():
                conf_matrix = v2["conf_matrix"].reindex([0, 1]).fillna(0).astype(int)
                row = {
                    "n_cv": self.n_cv,
                    "cv_iter": k1,
                    "model_num": k2,
                    "training_time": v2["training_time"],
                    "train_min_date": v1["train_min_date"],
                    "train_max_date": v1["train_max_date"],
                    "test_min_date": v1["test_min_date"],
                    "test_max_date": v1["test_max_date"],
                    "parameters": v2["parameters"],
                    "static_params": self.static_params,
                    "numeric_features": self.numeric_features,
                    "categorical_features": self.categorical_features,
                    "one_hot_encode_features": self.one_hot_encode_features,
                    "tp": conf_matrix.loc[1][1],
                    "fp": conf_matrix.loc[1][0],
                    "tn": conf_matrix.loc[0][0],
                    "fn": conf_matrix.loc[0][1],
                }
                rows.append(row)
        df = pd.DataFrame(rows)
        df["model"] = str(self.model_class)
        df["target"] = self.target
        df["timestamp"] = datetime.datetime.now()  # Timestamp for aggregating results.
        df["class_weights"] = self.class_weights
        df["scale_features"] = self.scale_features
        df["handle_invalid"] = self.hi
        column_order = ["model"] + [c for c in df.columns if c != "model"]
        return df[column_order]

    def save_results(self, verbose: bool = True):
        name = "".join(list(filter(lambda s: s not in "<>'", str(self.model_class)))).replace(".", "_")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
        filename = f"{self.run_uuid} - {name} - {now}"
        df = self.results_df

        # Save CSV:
        filepath = os.path.join("/dbfs/team8_results", f"{filename}.csv")
        df.to_csv(filepath, encoding="utf-8", index=False)
        if verbose:
            print(f"Results CSV saved to: {filepath}")

        # Save to blob storage:
        filepath = f"{blob_url}/{filename}"
        spark_df = spark.createDataFrame(df.astype(str))  # NOQA.
        spark_df.write.mode("overwrite").parquet(filepath)
        if verbose:
            print(f"Results saved to blob storage: {filepath}")


# COMMAND ----------

# Define the training dataset:
data = spark.read.parquet(f"{blob_url}/ML_train_filled")
data = data.dropna(subset="DEP_DEL15")
train = data.filter(data["YEAR"] < 2019).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ### GBTClassifier

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier

# Define the feature set:

# Outcome variable - 1/0 binary variable:
target = "DEP_DEL15"

# Features to use in the model:
numeric_features = [
    "WND_dir_ORIGIN",  # Wind direction at origin.
    "WND_dir_DEST",  # Wind direction at destination.
    "WND_spd_ORIGIN",  # Wind speed at origin.
    "WND_spd_DEST",  # Wind speed at destination.
    "VIS_dim_ORIGIN",  # Visibility at origin.
    "VIS_dim_DEST",  # Visibility at destination.
    "TMP_air_ORIGIN",  # Air temperature at origin.
    "TMP_air_DEST",  # Air temperature at destination.
    "DEW_point_temp_ORIGIN",  # Dew point temperature at origin.
    "DEW_point_temp_DEST",  # Dew point temperature at destination.
    "SLP_pressure_ORIGIN",  # Sea-level pressure at origin.
    "SLP_pressure_DEST",  # Sea-level pressure at destination.
    "CRS_ELAPSED_TIME",  # Scheduled flight time.
    "DISTANCE",  # Scheduled flight distance.
    "avg_delay_ORIGIN",  # Rolling average delay at origin.
    "avg_delay_DEST",  # Rolling average delay at destination.
]

categorical_features = [
    "MONTH",  # Month of year flight departed.
    "DAY_OF_WEEK",  # Day of week flight departed.
    "OP_CARRIER",  # Airline.
    "ORIGIN_STATE_ABR",  # Origin state.
    "DEST_STATE_ABR",  # Destination state.
    "WND_type_ORIGIN",  # Classification of wind at origin.
    "WND_type_DEST",  # Classification of wind at destination.
    "VIS_var_ORIGIN",  # Clasification of visibility at origin.
    "VIS_var_DEST",  # Clasification of visibility at destination.
    "prev_fl_del",  # Was the aircraft's previous flight delayed?
    "poten_for_del",  # Indicator feature based on whether aircraft is at the airport.
    "holiday",  # Was departure date a holiday?
    "local_departure_hour",  # Hour of flight departure.
]

one_hot_encode_features = None


from pyspark.ml.classification import GBTClassifier

# Parameters to gridsearch:
grid_params = {
    "maxIter": [25, 75],  # Number of gradient boosting iterations.
    "minInstancesPerNode": [1, 5],  # Minimum number of instances each leaf node must have.
    "maxDepth": [1, 2, 4],  # Maximum depth of each tree (max allowed is 30).
}

# Parameters to use in every model version:
static_params = {
    "maxBins": 100,  # Must be more than 52 - number of categories in the state features.
    "stepSize": 0.1,  # Learning rate.
}

gscv = SparkGridSearchTimeSeriesCV(
    model_class = GBTClassifier,  # The estimator class.
    train = train,  # The full training dataset.
    n_cv = 4,  # The number of cross-validation splits (will train n_cv-1 models).
    numeric_features = numeric_features,  # Column names of numeric features to use.
    categorical_features = categorical_features,  # Column names of categorical features to use.
    one_hot_encode_features = one_hot_encode_features,  # Column names of categorical features to also one-hot-encode.
    grid_params = grid_params,  # Dict of parameter space to be searched.
    static_params = static_params,  # Additional static parameters to use in every version of the model.
    target = target,  # Name of column being predicted.
    class_weights = True,  # Whether to reweight class imbalance.
    scale_features = False,  # Whether to scale numeric features to mu=0, sigma=1.
    handle_invalid = "keep",  # What to do with Spark feature class errors.
)

# Train the models:
gscv.run()

# Save the full results to CSV:
gscv.save_results()

# COMMAND ----------

# MAGIC %md
# MAGIC ### RandomForestClassifier

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

# Features to use in the model:
numeric_features = ['WND_dir_ORIGIN', 'WND_dir_DEST', # Wind direction (Weather)
                    'WND_spd_ORIGIN', 'WND_spd_DEST', # Wind speed (Weather)
                    'VIS_dim_ORIGIN', 'VIS_dim_DEST', # Visibility distance (Weather)
                    'TMP_air_ORIGIN', 'TMP_air_DEST', # Air temperature (Weather)
                    'DEW_point_temp_ORIGIN', 'DEW_point_temp_DEST', # Dew point temperature (Weather)
                    'SLP_pressure_ORIGIN', 'SLP_pressure_DEST', # Sea level pressure (Weather)
                    'CRS_ELAPSED_TIME', # Scheduled airtime (Flight)
                    'avg_delay_ORIGIN', 'avg_delay_DEST', # Average delay (minutes) 2-6 hours (New feature)
                    'DISTANCE' # Distance between origin and destination
]
categorical_features = ['WND_type_ORIGIN', 'WND_type_DEST', # Wind type (Weather)
                        'VIS_var_ORIGIN', 'VIS_var_DEST', # Visibility variability (Weather)
                        'OP_CARRIER', # Airplane carrier (Flight)
                        'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', # Day/time columns (Flight)
                        'ORIGIN_STATE_ABR', 'DEST_STATE_ABR', # State abbreviation (Flight)
                        'prev_fl_del', 'poten_for_del', # Prior flight delay & potential for delay (New features)
                        'holiday', # Holiday indicator (New feature)
                        'local_departure_hour' # Departure hour local (Flight)
                        #'OD_pair' # Origin-destination Airport pairs (Flight)
]

# Parameters to gridsearch:
grid_params = {
    'numTrees': [20, 40],
    'maxDepth': [3, 5, 7]
}

# Parameters to use in every model version:
static_params = {
    "weightCol": "weight",
    "maxBins":100
}
                        
# Create the instance:
gscv = SparkGridSearchTimeSeriesCV(
    model_class = RandomForestClassifier,  # The estimator class.
    train = train,  # The full training dataset.
    n_cv = 4,  # The number of cross-validation splits (will train n_cv-1 models).
    numeric_features = numeric_features,  # Column names of categorical features to use.
    categorical_features = categorical_features,  # Column names of numeric features to use.
    grid_params = grid_params,  # Dict of parameter space to be searched.
    static_params = static_params,  # Additional static parameters to use in every version of the model.
    target = "DEP_DEL15",  # Name of column being predicted.
    class_weights = True,  # Whether to reweight class imbalance.
    scale_features = False,  # Whether to scale numeric features to mu=0, sigma=1.
    handle_invalid = "keep",  # What to do with Spark feature class errors.
)

# Train the models:
gscv.run()

# Save the results to CSV:
gscv.save_results()

# COMMAND ----------

# MAGIC %md
# MAGIC ### LogisticRegression

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# Features to use in the model:
numeric_features = ['WND_dir_ORIGIN', 'WND_dir_DEST', # Wind direction (Weather)
                    'WND_spd_ORIGIN', 'WND_spd_DEST', # Wind speed (Weather)
                    'VIS_dim_ORIGIN', 'VIS_dim_DEST', # Visibility distance (Weather)
                    'TMP_air_ORIGIN', 'TMP_air_DEST', # Air temperature (Weather)
                    'DEW_point_temp_ORIGIN', 'DEW_point_temp_DEST', # Dew point temperature (Weather)
                    'SLP_pressure_ORIGIN', 'SLP_pressure_DEST', # Sea level pressure (Weather)
                    'CRS_ELAPSED_TIME', # Scheduled airtime (Flight)
                    'avg_delay_ORIGIN', 'avg_delay_DEST', # Average delay (minutes) 2-6 hours (New feature)
                    'DISTANCE' # Distance between origin and destination
]
categorical_features = ['WND_type_ORIGIN', 'WND_type_DEST', # Wind type (Weather)
                        'VIS_var_ORIGIN', 'VIS_var_DEST', # Visibility variability (Weather)
                        'OP_CARRIER', # Airplane carrier (Flight)
                        'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', # Day/time columns (Flight)
                        'ORIGIN_STATE_ABR', 'DEST_STATE_ABR', # State abbreviation (Flight)
                        'prev_fl_del', 'poten_for_del', # Prior flight delay & potential for delay (New features)
                        'holiday', # Holiday indicator (New feature)
                        'local_departure_hour' # Departure hour local (Flight)
                        #'OD_pair' # Origin-destination Airport pairs (Flight)
]

# Parameters to gridsearch:
grid_params = {
    'regParam': [0, 0.01], # 0 means no regularization
    'elasticNetParam': [0, 0.001, 0.01, 0.1, 0.5, 1], # Elastic net contains both L1 and L2 regularization. Elastic net parameter α set to 1 = Lasso model; If α is set to 0 = ridge regression model. L1 removes coefficient (goes to zero). L1 regularization technique is called Lasso Regression (α set to 1) and model which uses L2 is called Ridge Regression (α set to 0).
}

# Parameters to use in every model version:
static_params = {
    "weightCol": "weight", # turn on class_weights
    "maxIter": 10
}
                        
# Create the instance:
gscv = SparkGridSearchTimeSeriesCV(
    model_class = LogisticRegression,  # The estimator class.
    train = train,  # The full training dataset.
    n_cv = 4,  # The number of cross-validation splits (will train n_cv-1 models).
    numeric_features = numeric_features,  # Column names of categorical features to use.
    categorical_features = categorical_features,  # Column names of numeric features to use.
    one_hot_encode_features = categorical_features, # Logistic Regression - turn all categorical features to one hot encode features
    grid_params = grid_params,  # Dict of parameter space to be searched.
    static_params = static_params,  # Additional static parameters to use in every version of the model.
    target = "DEP_DEL15",  # Name of column being predicted.
    class_weights = True,  # Whether to reweight class imbalance.
    scale_features = True,  # Whether to scale numeric features to mu=0, sigma=1.
)

# Train the models:
gscv.run()

# Save the results to CSV:
gscv.save_results()
