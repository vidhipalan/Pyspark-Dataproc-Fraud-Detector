import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

# ─── 0) If running in a notebook, stop any active session ─────────────
try:
    from pyspark.sql import SparkSession as _SS
    if _SS.getActiveSession():
        _SS.getActiveSession().stop()
except:
    pass

# ─── 1) Start Spark in local mode ─────────────────────────────────────
spark = (
    SparkSession
    .builder
    .master("local[*]")
    .appName("RetrainFraudRFFullPipeline")
    .getOrCreate()
)

# ─── 2) Load raw data ─────────────────────────────────────────────────
raw = spark.read.csv(
    "gs://fraud_detection_dataset_1/Synthetic_Financial_datasets_log.csv",
    header=True,
    inferSchema=True
)

# ─── 3) Initial feature engineering ───────────────────────────────────
df = (raw
      .drop("nameOrig", "nameDest", "isFlaggedFraud", "step")
      .withColumn("deltaOrig",           col("oldbalanceOrg")  - col("newbalanceOrig"))
      .withColumn("deltaDest",           col("newbalanceDest") - col("oldbalanceDest"))
      .withColumn("amount_to_orig_ratio", col("amount")/(col("oldbalanceOrg")+1))
      .withColumn("amount_to_dest_ratio", col("amount")/(col("oldbalanceDest")+1))
     )

# ─── 4) Sample & weight ────────────────────────────────────────────────
full  = df.stat.sampleBy("isFraud", {0: 1.0, 1: 1.0}, seed=2025)
train = full.stat.sampleBy("isFraud", {0: 0.8, 1: 0.8}, seed=2025)

total  = train.count()
frauds = train.filter(col("isFraud") == 1).count()
w      = 1.0 / (frauds / total)

train = train.withColumn(
    "weight",
    when(col("isFraud") == 1, w).otherwise(1.0)
)

# ─── 5) Define full pipeline ──────────────────────────────────────────
string_indexer = StringIndexer(
    inputCol="type", outputCol="type_idx", handleInvalid="keep"
)
onehot_encoder = OneHotEncoder(
    inputCols=["type_idx"], outputCols=["type_vec"]
)
assembler      = VectorAssembler(
    inputCols=[
        'amount','oldbalanceOrg','newbalanceOrig',
        'oldbalanceDest','newbalanceDest',
        'deltaOrig','deltaDest',
        'amount_to_orig_ratio','amount_to_dest_ratio',
        'type_vec'
    ],
    outputCol="rawFeatures"
)
scaler         = StandardScaler(
    inputCol="rawFeatures", outputCol="features",
    withStd=True, withMean=False
)
rf             = RandomForestClassifier(
    labelCol="isFraud",
    featuresCol="features",
    weightCol="weight",
    numTrees=100,
    maxDepth=8,
    seed=42
)

pipeline = Pipeline(stages=[
    string_indexer,
    onehot_encoder,
    assembler,
    scaler,
    rf
])

# ─── 6) Fit & time ────────────────────────────────────────────────────
start = time.time()
model = pipeline.fit(train)
print(f"🔨 TRAIN TIME: {time.time() - start:.2f} s")

# ─── 7) Save to GCS ───────────────────────────────────────────────────
model_path = "gs://fraud_detection_dataset_1/rf_100pct_model"
model.write().overwrite().save(model_path)
print(f"✅ Full pipeline saved to: {model_path}")

# ─── 8) Done ──────────────────────────────────────────────────────────
spark.stop()

