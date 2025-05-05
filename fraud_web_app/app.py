# app.py

import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel
import streamlit as st

# ─── Session‑state flags ───────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "form"
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ─── Lazy init Spark & model ──────────────────────────────────────────
spark = None
model = None
def init_spark_and_model():
    global spark, model
    if spark is None:
        spark = (
            SparkSession
            .builder
            .master("local[*]")
            .appName("SparkSherlock")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
    if model is None:
        model = PipelineModel.load("gs://fraud_detection_dataset_1/rf_100pct_model")

# ─── FORM PAGE ─────────────────────────────────────────────────────────
if st.session_state.page == "form":
    st.title("🔍 Spark Sherlock")

    amount         = st.number_input("Amount",            min_value=0.0, format="%.2f")
    oldbalanceOrg  = st.number_input("Sender's Old Balance",  min_value=0.0, format="%.2f")
    newbalanceOrig = st.number_input("Sender's New Balance",  min_value=0.0, format="%.2f")
    oldbalanceDest = st.number_input("Receiver's Old Balance",min_value=0.0, format="%.2f")
    newbalanceDest = st.number_input("Receiver's New Balance",min_value=0.0, format="%.2f")
    tx_type        = st.selectbox("Transaction Type",
                         ["PAYMENT","TRANSFER","CASH_OUT","DEBIT","CASH_IN"])

    if st.button("Predict"):
        # initialize Spark & model
        init_spark_and_model()

        # build one‑row DF + feature engineering
        df = (
            spark.createDataFrame(
                [(amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, tx_type)],
                ["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","type"]
            )
            .withColumn("deltaOrig",           col("oldbalanceOrg")  - col("newbalanceOrig"))
            .withColumn("deltaDest",           col("newbalanceDest") - col("oldbalanceDest"))
            .withColumn("amount_to_orig_ratio", col("amount")/(col("oldbalanceOrg")+1))
            .withColumn("amount_to_dest_ratio", col("amount")/(col("oldbalanceDest")+1))
        )

        # run the model
        st.session_state.prediction = int(model.transform(df).select("prediction").first()[0])
        st.session_state.page = "result"

# ─── RESULT PAGE ───────────────────────────────────────────────────────
else:
    st.title("📝 Prediction Result")

    if st.session_state.prediction == 1:
        st.markdown(
            "<p style='font-size:36px; color:red; font-weight:bold;'>"
            "⚠️ This transaction is predicted to be FRAUDULENT.</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<p style='font-size:36px; color:blue; font-weight:bold;'>"
            "✅ This transaction appears legitimate.</p>",
            unsafe_allow_html=True
        )

    if st.button("← Back"):
        st.session_state.page = "form"

