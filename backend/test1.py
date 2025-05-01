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

        def get_spark_prob(model, input_df, model_name):
            start = time.time()
            output = model.transform(input_df).select("probability", "prediction").first()
            end = time.time()
            prob = output["probability"][1]  # Probability for class 1
            pred = int(output["prediction"])
            duration = end - start
            print(f"{model_name} - Prediction: {pred}, Prob: {prob:.4f}, Time: {duration:.4f}s")
            return pred, prob, duration

        rf_pred, rf_prob, rf_time = get_spark_prob(rf_model, input_df, "Random Forest")
        gbt_pred, gbt_prob, gbt_time = get_spark_prob(gbt_model, input_df, "GBT")
        dt_pred, dt_prob, dt_time = get_spark_prob(dt_model, input_df, "Decision Tree")
        mlp_pred, mlp_prob, mlp_time = get_spark_prob(mlp_model, input_df, "MLP")
        svm_pred, svm_prob, svm_time = get_spark_prob(svm_model, input_df, "SVM")
        lr_pred, lr_prob, lr_time = get_spark_prob(lr_model, input_df, "Logistic Regression")

        user_features = np.array([row.features.toArray() for row in input_df.select("features").collect()])
        start_xgb = time.time()
        xgb_prob = float(xgb_model.predict_proba(user_features)[0][1])
        xgb_pred = int(xgb_model.predict(user_features)[0])
        end_xgb = time.time()
        xgb_time = end_xgb - start_xgb
        print(f"XGBoost - Prediction: {xgb_pred}, Prob: {xgb_prob:.4f}, Time: {xgb_time:.4f}s")

        # Majority vote ensemble (excluding LR)
        majority_preds = [rf_pred, gbt_pred, dt_pred, mlp_pred, svm_pred, xgb_pred]
        ensemble_vote = round(sum(majority_preds) / len(majority_preds))

        # Probability averaging ensemble (excluding LR)
        prob_list = [rf_prob, gbt_prob, dt_prob, mlp_prob, svm_prob, xgb_prob]
        avg_prob = sum(prob_list) / len(prob_list)
        ensemble_prob_pred = 1 if avg_prob >= 0.5 else 0

        return {
            "lr_prediction": lr_pred,
            "lr_probability": round(lr_prob, 4),

            "rf_prediction": rf_pred,
            "rf_probability": round(rf_prob, 4),

            "gbt_prediction": gbt_pred,
            "gbt_probability": round(gbt_prob, 4),

            "dt_prediction": dt_pred,
            "dt_probability": round(dt_prob, 4),

            "mlp_prediction": mlp_pred,
            "mlp_probability": round(mlp_prob, 4),

            "svm_prediction": svm_pred,
            "svm_probability": round(svm_prob, 4),

            "xgb_prediction": xgb_pred,
            "xgb_probability": round(xgb_prob, 4),

            "ensemble_vote_majority": ensemble_vote,
            "ensemble_probability_avg": round(avg_prob, 4),
            "ensemble_probability_prediction": ensemble_prob_pred,

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
