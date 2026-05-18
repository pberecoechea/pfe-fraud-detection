# Fichier processor.py
# Permet de récupérer depuis Kafka l'ensemble des transactions
#
# Pablo BERECOECHEA
# 24/03/26

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
)
import math
from datetime import datetime, timezone
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
    StructField("is_fraud", IntegerType()),
]

# On ajoute les colonnes V1 à V28 dynamiquement

schema = StructType(fields)


def _welford_update(r_conn, stats_key: str, new_value: float) -> tuple:
    """
    Mise à jour incrémentale de la moyenne et de l'écart-type (algorithme de Welford).
    Stocke count, amt_mean, M2 dans un hash Redis.
    Retourne (new_mean, new_std).
    """
    raw = r_conn.hgetall(stats_key)
    count = int(raw.get("count", 0))
    mean = float(raw.get("amt_mean", 0.0))
    M2 = float(raw.get("M2", 0.0))

    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2

    std = math.sqrt(M2 / count) if count > 1 else 0.0

    r_conn.hset(
        stats_key, mapping={"count": count, "amt_mean": mean, "M2": M2, "amt_std": std}
    )
    return mean, std


def write_to_redis(df, batch_id):
    def process_partition(partition):
        r = redis.Redis(host="redis_cache", port=6379, db=0, decode_responses=True)

        for row in partition:
            row_dict = row.asDict()
            cc_num = row_dict["cc_num"]
            merchant = row_dict["merchant"]
            amt = float(row_dict["amt"])
            timestamp_str = row_dict["trans_date_trans_time"]

            # --- Transaction ---
            transaction_key = f"transaction:{row_dict['trans_num']}"
            transaction_data = {
                "cc_num": cc_num,
                "amt": amt,
                "merchant": merchant,
                "category": row_dict["category"],
                "trans_date_trans_time": timestamp_str,
                "unix_time": row_dict["unix_time"],
                "is_fraud": row_dict["is_fraud"],
                "lat": row_dict["lat"],
                "long": row_dict["long"],
                "merch_lat": row_dict["merch_lat"],
                "merch_long": row_dict["merch_long"],
                "gender": row_dict["gender"],
                "dob": row_dict["dob"],
                "city_pop": row_dict["city_pop"],
            }
            r.hset(transaction_key, mapping=transaction_data)

            # --- Client ---
            client_key = f"client:{cc_num}"
            r.hset(
                client_key,
                mapping={
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
                    "dob": row_dict["dob"],
                },
            )

            # --- Vélocité : temps depuis la dernière transaction ---
            last_tx_key = f"last_tx_time:{cc_num}"
            r.set(last_tx_key, timestamp_str)

            # --- Vélocité : nombre de transactions du jour ---
            try:
                tx_date = timestamp_str[:10]  # "YYYY-MM-DD"
            except Exception:
                tx_date = str(datetime.now(timezone.utc).date())
            count_key = f"tx_count_day:{cc_num}:{tx_date}"
            r.incr(count_key)
            r.expire(count_key, 172800)  # TTL 48h

            # --- Stats par carte (Welford) ---
            card_stats_key = f"card_stats:{cc_num}"
            _welford_update(r, card_stats_key, amt)

            # --- Stats par marchand (moyenne glissante) ---
            merchant_stats_key = f"merchant_stats:{merchant}"
            _welford_update(r, merchant_stats_key, amt)

            # --- Sorted sets pour l'API ---
            unix_time = float(row_dict.get("unix_time") or 0)
            r.zadd("transactions:recent", {row_dict["trans_num"]: unix_time})
            r.zremrangebyrank("transactions:recent", 0, -501)

            r.zadd("client:recent", {cc_num: unix_time})
            r.zremrangebyrank("client:recent", 0, -1001)

    df.foreachPartition(process_partition)

    print(f"Batch {batch_id} traité.")


# Initialisation session spark
spark = (
    SparkSession.builder.appName("FraudDetectionProcessor")
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1,com.redislabs:spark-redis_2.12:3.1.0",
    )
    .config("spark.redis.host", "redis_cache")
    .config("spark.redis.port", "6379")
    .getOrCreate()
)

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
    transactions_df.writeStream.foreachBatch(write_to_redis)
    .option("checkpointLocation", "/tmp/checkpoints/redis")
    .start()
)

query_console = (
    transactions_df.writeStream.outputMode("append").format("console").start()
)

print("Spark Processor démarré ! En attente de données depuis Kafka...")

query_redis.awaitTermination()
query_console.awaitTermination()
