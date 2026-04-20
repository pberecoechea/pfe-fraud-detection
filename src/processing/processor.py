# Fichier processor.py
# Permet de récupérer depuis Kafka l'ensemble des transactions
#
# Pablo BERECOECHEA
# 24/03/26

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import json
import redis

fields = [
    StructField("client_id", StringType()), # L'ID qu'on a injecté dans le producer
    StructField("Time", DoubleType()),
    StructField("Amount", DoubleType()),
    StructField("Class", IntegerType())
]

# On ajoute les colonnes V1 à V28 dynamiquement
for i in range(1, 29):
    fields.append(StructField(f"V{i}", DoubleType()))

schema = StructType(fields)

def write_to_redis(df, batch_id):
    r = redis.Redis(host='redis_cache', port=6379, db=0)
    records = df.collect()
    for row in records:
        key = f"client:{row['client_id']}"
        feature_data = {
            "amount": row['Amount'],
            "time": row['Time'],
            "is_fraud_known": row['Class'],
            "v1": row['V1'], # On peut en mettre quelques-unes pour l'exemple
            "v2": row['V2']
        }
        r.set(key, json.dumps(feature_data))
        
    print(f":inbox_tray: Batch {batch_id} : {len(records)} clients mis à jour dans le Feature Store.")

# Initialisation session spark
spark = SparkSession.builder\
    .appName("FraudDetectionProcessor")\
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1,com.redislabs:spark-redis_2.12:3.1.0")\
    .config("spark.redis.host", "redis_cache")\
    .config("spark.redis.port", "6379")\
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

raw_df = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "redpanda:29092").option("subscribe", "transactions").option("startingOffsets", "latest").load()

# Transformation binaire kafka en colonnes lisibles
transactions_df = raw_df.selectExpr("CAST(value AS STRING)").select(from_json(col("value"), schema).alias("data")).select("data.*")


query = transactions_df.writeStream \
    .foreachBatch(write_to_redis) \
    .start()

# high_value_df = transactions_df.filter(col("Amount") > 200000)
query = transactions_df.writeStream.outputMode("append").format("console").start()

print("Spark Processor démarré ! En attente de données venant de Kafka...")
query.awaitTermination()