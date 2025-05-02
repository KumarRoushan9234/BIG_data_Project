import streamlit as st
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import unix_timestamp, col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import (RandomForestClassifier, GBTClassifier,
                                       DecisionTreeClassifier, MultilayerPerceptronClassifier,
                                       LinearSVC)
import xgboost as xgb
import numpy as np
import pandas as pd

# -------------------- Cached Spark & Model Setup --------------------
@st.cache_resource(show_spinner=True)
def setup_spark_and_train():
    print("🚀 Starting Spark session...")
    spark = SparkSession.builder.appName("CreditCardFraudDetectionApp").getOrCreate()

    print("📥 Loading datasets...")
    fraudTrain = spark.read.csv("fraudTrain.csv", header=True, inferSchema=True)
    fraudTest = spark.read.csv("fraudTest.csv", header=True, inferSchema=True)
    print("✅ Datasets loaded.")

    drop_cols = ['_c0', 'cc_num', 'first', 'last', 'gender', 'street', 'city', 'state',
                 'zip', 'job', 'dob', 'trans_num', 'merchant']
    fraudTrain = fraudTrain.drop(*drop_cols)
    fraudTest = fraudTest.drop(*drop_cols)
    print("🧹 Dropped unnecessary columns.")

    fraudTrain = fraudTrain.withColumn("trans_date_ts", unix_timestamp("trans_date_trans_time")).drop("trans_date_trans_time")
    fraudTest = fraudTest.withColumn("trans_date_ts", unix_timestamp("trans_date_trans_time")).drop("trans_date_trans_time")
    print("⏱️ Converted transaction timestamp.")

    fraudTrain = fraudTrain.filter(fraudTrain["category"].isNotNull()).fillna({"category": "unknown"})
    fraudTest = fraudTest.filter(fraudTest["category"].isNotNull()).fillna({"category": "unknown"})
    print("🧽 Cleaned missing values.")

    indexer = StringIndexer(inputCol="category", outputCol="category_index", handleInvalid="skip")
    indexer_model = indexer.fit(fraudTrain)
    fraudTrain = indexer_model.transform(fraudTrain)
    fraudTest = indexer_model.transform(fraudTest)
    print("🔠 Encoded categorical feature: 'category'")

    assembler = VectorAssembler(inputCols=[
        'amt', 'lat', 'long', 'city_pop', 'unix_time',
        'merch_lat', 'merch_long', 'trans_date_ts', 'category_index'
    ], outputCol="features")
    fraudTrain = assembler.transform(fraudTrain).select("features", "is_fraud")
    fraudTest = assembler.transform(fraudTest).select("features", "is_fraud")
    print("📦 Assembled features.")

    print("🧠 Training ML models...")
    
    rf_model = RandomForestClassifier(labelCol="is_fraud", featuresCol="features", numTrees=50).fit(fraudTrain)
    print("Random Forest model trained.")

    gbt_model = GBTClassifier(labelCol="is_fraud", featuresCol="features", maxIter=20).fit(fraudTrain)
    print("GBT model trained.")

    dt_model = DecisionTreeClassifier(labelCol="is_fraud", featuresCol="features").fit(fraudTrain)
    print("Decision Tree model trained.")

    mlp_model = MultilayerPerceptronClassifier(labelCol="is_fraud", featuresCol="features",
                                               layers=[9, 10, 5, 2], maxIter=100).fit(fraudTrain)
    print("mlp_model trained.")

    svm_model = LinearSVC(labelCol="is_fraud", featuresCol="features").fit(fraudTrain)
    print("SVM model trained.")

    print("✅ Spark ML models trained.")

    train_pd = fraudTrain.select("features", "is_fraud").toPandas()
    X_train = np.array([row.toArray() for row in train_pd['features']])
    y_train = train_pd['is_fraud']
    xgb_model = xgb.XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train)
    print("✅ XGBoost model trained.")

    return spark, indexer_model, assembler, rf_model, gbt_model, dt_model, mlp_model, svm_model, xgb_model

# Load and train models
spark, indexer_model, assembler, rf_model, gbt_model, dt_model, mlp_model, svm_model, xgb_model = setup_spark_and_train()
print("🚦 All models and preprocessing ready.")

# -------------------- Streamlit UI --------------------
st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details to predict the probability of fraud:")

with st.form("fraud_form"):
    user_input = {
        'amt': st.number_input("Transaction Amount", value=100.0),
        'lat': st.number_input("Latitude", value=37.7749),
        'long': st.number_input("Longitude", value=-122.4194),
        'city_pop': st.number_input("City Population", value=50000),
        'unix_time': st.number_input("Unix Time", value=1325376018),
        'merch_lat': st.number_input("Merchant Latitude", value=37.0),
        'merch_long': st.number_input("Merchant Longitude", value=-122.0),
        'trans_date_ts': st.number_input("Transaction Timestamp", value=1577836800),
        'category': st.text_input("Category", value='misc_pos')
    }
    predict_btn = st.form_submit_button("Predict Fraud")

if predict_btn:
    print("📥 Received user input.")
    input_df = spark.createDataFrame([Row(**user_input)])
    input_df = indexer_model.transform(input_df)
    input_df = assembler.transform(input_df)
    print("🛠️ Preprocessing complete for input data.")

    rf_pred = rf_model.transform(input_df).select("prediction", "probability").first()
    gbt_pred = gbt_model.transform(input_df).select("prediction", "probability").first()
    dt_pred = dt_model.transform(input_df).select("prediction", "probability").first()
    mlp_pred = mlp_model.transform(input_df).select("prediction", "probability").first()
    svm_pred = svm_model.transform(input_df).select("prediction", "rawPrediction").first()
    xgb_features = np.array([row.features.toArray() for row in input_df.select("features").collect()])
    xgb_pred = int(xgb_model.predict(xgb_features)[0])
    xgb_prob = float(xgb_model.predict_proba(xgb_features)[0][1])
    print("✅ Predictions complete.")

    st.subheader("📊 Model Predictions")
    st.write(f"Random Forest: Fraud={int(rf_pred['prediction'])}, Probability={1 - rf_pred['probability'][0]:.4f}")
    st.write(f"GBT: Fraud={int(gbt_pred['prediction'])}, Probability={1 - gbt_pred['probability'][0]:.4f}")
    st.write(f"Decision Tree: Fraud={int(dt_pred['prediction'])}, Probability={1 - dt_pred['probability'][0]:.4f}")
    st.write(f"MLP: Fraud={int(mlp_pred['prediction'])}, Probability={1 - mlp_pred['probability'][0]:.4f}")
    st.write(f"SVM: Fraud={int(svm_pred['prediction'])}, Raw Score={svm_pred['rawPrediction']}")
    st.write(f"XGBoost: Fraud={xgb_pred}, Probability={xgb_prob:.4f}")

    preds = [rf_pred['prediction'], gbt_pred['prediction'], dt_pred['prediction'],
             mlp_pred['prediction'], svm_pred['prediction'], xgb_pred]
    avg_pred = round(sum(preds) / len(preds))

    st.subheader("🧮 Ensemble Result")
    st.success(f"Prediction: {'🚨 Fraud' if avg_pred == 1 else '✅ Not Fraud'}")
    print("📢 Ensemble prediction displayed.")

# -------------------- LLM Prediction Section --------------------
st.subheader("🧠 LLM-Based Prediction (Experimental)")

llm_notes = st.text_area("Describe the transaction (location, time, amount, etc.):", "")
if st.button("Ask LLM"):
    if llm_notes.strip() == "":
        st.warning("Please enter some transaction details.")
    else:
        st.info("LLM integration placeholder. Here you'd send this prompt to an LLM.")
        st.code(f"Prompt sent: 'Is this transaction fraudulent? {llm_notes}'")
        print("🗨️ Sent transaction description to LLM (placeholder).")
