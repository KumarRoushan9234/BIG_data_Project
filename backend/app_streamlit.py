import streamlit as st
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import unix_timestamp
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import (RandomForestClassifier, GBTClassifier,
                                       DecisionTreeClassifier, MultilayerPerceptronClassifier,
                                       LinearSVC)
import xgboost as xgb
import numpy as np
import pandas as pd

from app_llm import chain  # make sure chain is accessible from this module

# -------------------- Cached Spark & Model Setup --------------------
@st.cache_resource(show_spinner=True)
def setup_spark_and_train():
    spark = SparkSession.builder.appName("CreditCardFraudDetectionApp").getOrCreate()

    fraudTrain = spark.read.csv("fraudTrain.csv", header=True, inferSchema=True)
    fraudTest = spark.read.csv("fraudTest.csv", header=True, inferSchema=True)

    drop_cols = ['_c0', 'cc_num', 'first', 'last', 'gender', 'street', 'city', 'state',
                 'zip', 'job', 'dob', 'trans_num', 'merchant']
    fraudTrain = fraudTrain.drop(*drop_cols)
    fraudTest = fraudTest.drop(*drop_cols)

    fraudTrain = fraudTrain.withColumn("trans_date_ts", unix_timestamp("trans_date_trans_time")).drop("trans_date_trans_time")
    fraudTest = fraudTest.withColumn("trans_date_ts", unix_timestamp("trans_date_trans_time")).drop("trans_date_trans_time")

    fraudTrain = fraudTrain.filter(fraudTrain["category"].isNotNull()).fillna({"category": "unknown"})
    fraudTest = fraudTest.filter(fraudTest["category"].isNotNull()).fillna({"category": "unknown"})

    indexer = StringIndexer(inputCol="category", outputCol="category_index", handleInvalid="skip")
    indexer_model = indexer.fit(fraudTrain)
    fraudTrain = indexer_model.transform(fraudTrain)
    fraudTest = indexer_model.transform(fraudTest)

    assembler = VectorAssembler(inputCols=[
        'amt', 'lat', 'long', 'city_pop', 'unix_time',
        'merch_lat', 'merch_long', 'trans_date_ts', 'category_index'
    ], outputCol="features")
    fraudTrain = assembler.transform(fraudTrain).select("features", "is_fraud")
    fraudTest = assembler.transform(fraudTest).select("features", "is_fraud")

    rf_model = RandomForestClassifier(labelCol="is_fraud", featuresCol="features", numTrees=50).fit(fraudTrain)
    gbt_model = GBTClassifier(labelCol="is_fraud", featuresCol="features", maxIter=20).fit(fraudTrain)
    dt_model = DecisionTreeClassifier(labelCol="is_fraud", featuresCol="features").fit(fraudTrain)
    mlp_model = MultilayerPerceptronClassifier(labelCol="is_fraud", featuresCol="features",
                                               layers=[9, 10, 5, 2], maxIter=100).fit(fraudTrain)
    svm_model = LinearSVC(labelCol="is_fraud", featuresCol="features").fit(fraudTrain)

    train_pd = fraudTrain.select("features", "is_fraud").toPandas()
    X_train = np.array([row.toArray() for row in train_pd['features']])
    y_train = train_pd['is_fraud']
    xgb_model = xgb.XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train)

    return spark, indexer_model, assembler, rf_model, gbt_model, dt_model, mlp_model, svm_model, xgb_model

# Load and train models
spark, indexer_model, assembler, rf_model, gbt_model, dt_model, mlp_model, svm_model, xgb_model = setup_spark_and_train()

# -------------------- Streamlit UI --------------------
st.title("Credit Card Fraud Detection App")
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
        'category': st.selectbox("Category", [
            'entertainment', 'food_dining', 'gas_transport', 'grocery_net', 'grocery_pos',
            'health_fitness', 'home', 'kids_pets', 'misc_net', 'misc_pos', 'personal_care',
            'shopping_net', 'shopping_pos', 'travel'
        ])
    }
    predict_btn = st.form_submit_button("Predict Fraud")

if predict_btn:
    input_df = spark.createDataFrame([Row(**user_input)])
    input_df = indexer_model.transform(input_df)
    input_df = assembler.transform(input_df)

    rf_pred = rf_model.transform(input_df).select("prediction", "probability").first()
    gbt_pred = gbt_model.transform(input_df).select("prediction", "probability").first()
    dt_pred = dt_model.transform(input_df).select("prediction", "probability").first()
    mlp_pred = mlp_model.transform(input_df).select("prediction", "probability").first()
    svm_pred = svm_model.transform(input_df).select("prediction", "rawPrediction").first()
    xgb_features = np.array([row.features.toArray() for row in input_df.select("features").collect()])
    xgb_pred = int(xgb_model.predict(xgb_features)[0])
    xgb_prob = float(xgb_model.predict_proba(xgb_features)[0][1])

    st.subheader("Model Predictions")
    st.write(f"Random Forest: Fraud={int(rf_pred['prediction'])}, Probability={1 - rf_pred['probability'][0]:.4f}")
    st.write(f"GBT: Fraud={int(gbt_pred['prediction'])}, Probability={1 - gbt_pred['probability'][0]:.4f}")
    st.write(f"Decision Tree: Fraud={int(dt_pred['prediction'])}, Probability={1 - dt_pred['probability'][0]:.4f}")
    st.write(f"MLP: Fraud={int(mlp_pred['prediction'])}, Probability={1 - mlp_pred['probability'][0]:.4f}")
    st.write(f"SVM: Fraud={int(svm_pred['prediction'])}, Raw Score={svm_pred['rawPrediction']}")
    st.write(f"XGBoost: Fraud={xgb_pred}, Probability={xgb_prob:.4f}")

    preds = [rf_pred['prediction'], gbt_pred['prediction'], dt_pred['prediction'],
             mlp_pred['prediction'], svm_pred['prediction'], xgb_pred]
    avg_pred = round(sum(preds) / len(preds))

    st.subheader("Ensemble Result")
    st.success(f"Prediction: {'Fraud' if avg_pred == 1 else 'Not Fraud'}")

    st.markdown("---")

    # ---------- LLM-Based Prediction ----------
    st.subheader("LLM-Based Prediction (Experimental)")
    description = (
        f"Transaction of ${user_input['amt']} under category '{user_input['category']}'. "
        f"Location: ({user_input['lat']}, {user_input['long']}), merchant location: "
        f"({user_input['merch_lat']}, {user_input['merch_long']}). "
        f"City population: {user_input['city_pop']}. "
        f"Unix time: {user_input['unix_time']}, timestamp: {user_input['trans_date_ts']}."
    )

    try:
        llm_result = chain.invoke({"input": description})
        st.json(llm_result)
        st.success(f"Prediction: {'Fraud' if llm_result['fraud'] == 1 else 'Not Fraud'}")
        st.write(f"Confidence Score: {llm_result['probability']:.2f}")
        st.write("Reason:", llm_result["reasons"])
    except Exception as e:
        st.error(f"Error in LLM prediction: {e}")
