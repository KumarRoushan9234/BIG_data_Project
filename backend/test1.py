from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import unix_timestamp, col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier,
    DecisionTreeClassifier, MultilayerPerceptronClassifier, LinearSVC
)
import pandas as pd
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Credit Card Fraud Detection API is running"}

# Initialize Spark Session
try:
    spark = SparkSession.builder.appName("FastApi_spark").getOrCreate()
    print("Spark session initialized successfully.")
except Exception as e:
    print(f"Spark initialization failed: {e}")

# Load Data
try:
    if not os.path.exists("fraudTrain.csv") or not os.path.exists("fraudTest.csv"):
        print("CSV file not found in current directory.")
    else:
        fraudTrain = spark.read.csv("fraudTrain.csv", header=True, inferSchema=True)
        fraudTest = spark.read.csv("fraudTest.csv", header=True, inferSchema=True)
        print("CSV files loaded successfully.")
except Exception as e:
    print(f"Error loading CSV files: {e}")

# Data Preprocessing
try:
    drop_cols = ['_c0', 'cc_num', 'first', 'last', 'gender', 'street', 'city', 'state',
                 'zip', 'job', 'dob', 'trans_num', 'merchant']
    fraudTrain = fraudTrain.drop(*drop_cols)
    fraudTest = fraudTest.drop(*drop_cols)

    fraudTrain = fraudTrain.withColumn("trans_date_ts", unix_timestamp("trans_date_trans_time")).drop("trans_date_trans_time")
    fraudTest = fraudTest.withColumn("trans_date_ts", unix_timestamp("trans_date_trans_time")).drop("trans_date_trans_time")

    fraudTrain = fraudTrain.filter(fraudTrain["category"].isNotNull()).fillna({"category": "unknown"})
    fraudTest = fraudTest.filter(fraudTest["category"].isNotNull()).fillna({"category": "unknown"})

    fraudTrain = fraudTrain.withColumn("category", col("category").cast("string"))
    fraudTest = fraudTest.withColumn("category", col("category").cast("string"))

    indexer = StringIndexer(inputCol="category", outputCol="category_index", handleInvalid="skip")
    indexer_model = indexer.fit(fraudTrain)

    fraudTrain = indexer_model.transform(fraudTrain)
    fraudTest = indexer_model.transform(fraudTest)

    features = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long', 'trans_date_ts', 'category_index']
    assembler = VectorAssembler(inputCols=features, outputCol="features")

    fraudTrain = assembler.transform(fraudTrain).select("features", "is_fraud")
    fraudTest = assembler.transform(fraudTest).select("features", "is_fraud")
    print("Data preprocessing completed successfully.")
except Exception as e:
    print(f"Data preprocessing failed: {e}")

# Train Spark Models
try:
    rf_model = RandomForestClassifier(labelCol="is_fraud", featuresCol="features", numTrees=50).fit(fraudTrain)
    print("Random Forest model trained successfully.")

    gbt_model = GBTClassifier(labelCol="is_fraud", featuresCol="features", maxIter=20).fit(fraudTrain)
    print("GBT model trained successfully.")

    dt_model = DecisionTreeClassifier(labelCol="is_fraud", featuresCol="features").fit(fraudTrain)
    print("Decision Tree model trained successfully.")

    mlp_model = MultilayerPerceptronClassifier(labelCol="is_fraud", featuresCol="features",
                                               layers=[9, 10, 5, 2], blockSize=128, seed=1234, maxIter=100).fit(fraudTrain)
    print("Multilayer Perceptron model trained successfully.")

    svm_model = LinearSVC(labelCol="is_fraud", featuresCol="features").fit(fraudTrain)
    print("SVM model trained successfully.")

    lr_model = LogisticRegression(labelCol="is_fraud", featuresCol="features").fit(fraudTrain)
    print("Logistic Regression model trained successfully.")
except Exception as e:
    print(f"Training Spark models failed: {e}")

# Train XGBoost Model
try:
    train_pd = fraudTrain.select("features", "is_fraud").toPandas()
    test_pd = fraudTest.select("features", "is_fraud").toPandas()

    X_train = np.array([row.toArray() for row in train_pd['features']])
    y_train = train_pd['is_fraud']
    X_test = np.array([row.toArray() for row in test_pd['features']])
    y_test = test_pd['is_fraud']

    xgb_model = xgb.XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    print("XGBoost model trained successfully.")
except Exception as e:
    print(f"XGBoost training failed: {e}")

# User input schema
class UserInput(BaseModel):
    amt: float
    lat: float
    long: float
    city_pop: int
    unix_time: int
    merch_lat: float
    merch_long: float
    trans_date_ts: int
    category: str

# Helper functions
def convert_input_to_df(user_input: UserInput):
    try:
        input_data = [Row(**user_input.dict())]
        input_df = spark.createDataFrame(input_data)
        input_df = indexer_model.transform(input_df)
        input_df = assembler.transform(input_df)
        return input_df
    except Exception as e:
        raise ValueError(f"Input conversion failed: {e}")

import time
def make_spark_prediction(model, input_df, model_name):
    try:
        start_time = time.time()
        # gbt_pred=gbt_model.transform(input_df).select("prediction", "probability")
        prediction = model.transform(input_df).select("prediction").first()
        end_time = time.time()
        duration = end_time - start_time
        print(f"{model_name} Prediction: {prediction['prediction']} (Time taken: {duration:.4f} seconds)")
        return prediction, duration
    except Exception as e:
        raise RuntimeError(f"Prediction failed for model {model_name}: {e}")

@app.post("/predict")
async def predict(user_input: UserInput):
    try:
        input_df = convert_input_to_df(user_input)

        rf_pred, rf_time = make_spark_prediction(rf_model, input_df, "Random Forest")
        print(f"Random Forest Prediction: {rf_pred['prediction']} (Time taken: {rf_time:.4f} seconds)")
        gbt_pred, gbt_time = make_spark_prediction(gbt_model, input_df, "GBT")
        print(f"GBT Prediction: {gbt_pred['prediction']} (Time taken: {gbt_time:.4f} seconds)")

        dt_pred, dt_time = make_spark_prediction(dt_model, input_df, "Decision Tree")
        print(f"Decision Tree Prediction: {dt_pred['prediction']} (Time taken: {dt_time:.4f} seconds)")
        
        mlp_pred, mlp_time = make_spark_prediction(mlp_model, input_df, "MLP")
        print(f"MLP Prediction: {mlp_pred['prediction']} (Time taken: {mlp_time:.4f} seconds)")

        svm_pred, svm_time = make_spark_prediction(svm_model, input_df, "SVM")
        print(f"SVM Prediction: {svm_pred['prediction']} (Time taken: {svm_time:.4f} seconds)")

        lr_pred, lr_time = make_spark_prediction(lr_model, input_df, "Logistic Regression")
        print(f"Logistic Regression Prediction: {lr_pred['prediction']} (Time taken: {lr_time:.4f} seconds)")
        
        user_features = np.array([row.features.toArray() for row in input_df.select("features").collect()])
        start_xgb = time.time()
        xgb_pred = xgb_model.predict(user_features)
        end_xgb = time.time()
        xgb_time = end_xgb - start_xgb
        print(f"XGBoost Prediction: {xgb_pred[0]} (Time taken: {xgb_time:.4f} seconds)")

        preds = [
            int(rf_pred['prediction']),
            int(gbt_pred['prediction']),
            int(dt_pred['prediction']),
            int(mlp_pred['prediction']),
            int(svm_pred['prediction']),
            # int(lr_pred['prediction']),
            int(xgb_pred[0])
        ]
        final_vote = round(sum(preds) / len(preds))
        print(f"Final Ensemble Prediction (Majority Vote): {final_vote}")

        return {
          "lr_prediction": int(lr_pred['prediction']),
          "rf_prediction": int(rf_pred['prediction']),
          "gbt_prediction": int(gbt_pred['prediction']),
          "dt_prediction": int(dt_pred['prediction']),
          "mlp_prediction": int(mlp_pred['prediction']),
          "svm_prediction": int(svm_pred['prediction']),
          "xgb_prediction": int(xgb_pred[0]),
          "ensemble_vote": final_vote,
          "prediction_times": {
              "lr": f"{lr_time:.4f} sec",
              "rf": f"{rf_time:.4f} sec",
              "gbt": f"{gbt_time:.4f} sec",
              "dt": f"{dt_time:.4f} sec",
              "mlp": f"{mlp_time:.4f} sec",
              "svm": f"{svm_time:.4f} sec",
              "xgb": f"{xgb_time:.4f} sec"
          }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

