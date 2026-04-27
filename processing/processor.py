# Fichier processor.py
# Permet de récupérer depuis Kafka l'ensemble des transactions
#
# Pablo BERECOECHEA
# 24/03/26

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, BooleanType
import json
import redis

fields = [
    StructField("trans_date_trans_time", StringType()),
    StructField("cc_num", StringType()),  
    StructField("merchant", StringType()),
    StructField("category", StringType()),
    StructField("amt", DoubleType()),

    StructField("first", StringType()),
    StructField("last", StringType()),
    StructField("gender", StringType()),
    StructField("street", StringType()),
    StructField("city", StringType()),

    StructField("state", StringType()),
    StructField("zip", StringType()),
    StructField("lat", DoubleType()),
    StructField("long", DoubleType()),
    StructField("city_pop", IntegerType()),

    StructField("job", StringType()),
    StructField("dob", StringType()),
    StructField("trans_num", StringType()),
    StructField("unix_time", StringType()),
    StructField("merch_lat", DoubleType()),

    StructField("merch_long", DoubleType()),
    StructField("is_fraud", IntegerType()) 
]

# On ajoute les colonnes V1 à V28 dynamiquement

schema = StructType(fields)

def write_to_redis(df, batch_id):

    def process_partition(partition):
        r = redis.Redis(host='redis_cache', port=6379, db=0)

        for row in partition:
            row_dict = row.asDict()

            transaction_key = f"transaction:{row_dict['trans_num']}"

            transaction_data = {
                "cc_num": row_dict["cc_num"],
                "amount": row_dict["amt"],
                "merchant": row_dict["merchant"],
                "category": row_dict["category"],
                "timestamp": row_dict["trans_date_trans_time"],
                "unix_time": row_dict["unix_time"],
                "is_fraud": row_dict["is_fraud"],
                "lat": row_dict["lat"],
                "long": row_dict["long"],
                "merch_lat": row_dict["merch_lat"],
                "merch_long": row_dict["merch_long"]
            }

            r.hset(transaction_key, mapping=transaction_data)

            client_key = f"client:{row_dict['cc_num']}"

            client_data = {
                "first": row_dict["first"],
                "last": row_dict["last"],
                "gender": row_dict["gender"],
                "street": row_dict["street"],
                "city": row_dict["city"],
                "state": row_dict["state"],
                "zip": row_dict["zip"],
                "lat": row_dict["lat"],
                "long": row_dict["long"],
                "job": row_dict["job"],
                "dob": row_dict["dob"]
            }

            r.hset(client_key, mapping=client_data)

    df.foreachPartition(process_partition)

    print(f"Batch {batch_id} traité.")

# Initialisation session spark
spark = SparkSession.builder\
    .appName("FraudDetectionProcessor")\
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1,com.redislabs:spark-redis_2.12:3.1.0")\
    .config("spark.redis.host", "redis_cache")\
    .config("spark.redis.port", "6379")\
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

raw_df = (
    spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", "redpanda:29092")
        .option("subscribe", "transactions")
        .option("startingOffsets", "latest")
        .load()
)

transactions_df = (
    raw_df.selectExpr("CAST(value AS STRING)")
        .select(from_json(col("value"), schema).alias("data"))
        .select("data.*")
)

query_redis = (
    transactions_df.writeStream
        .foreachBatch(write_to_redis)
        .option("checkpointLocation", "/tmp/checkpoints/redis")
        .start()
)

query_console = (
    transactions_df.writeStream
        .outputMode("append")
        .format("console")
        .start()
)

print("🚀 Spark Processor démarré ! En attente de données depuis Kafka...")

query_redis.awaitTermination()
query_console.awaitTermination()