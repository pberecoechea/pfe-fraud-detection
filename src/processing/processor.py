# Fichier processor.py
# Permet de récupérer depuis Kafka l'ensemble des transactions
#
# Pablo BERECOECHEA
# 24/03/26

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

schema = StructType([
    StructField("Time", IntegerType()),
    
    # Pour le futur, composantes principales
    ## 1-5
    #StructField("V1", DoubleType()),
    #StructField("V2", DoubleType()),
    #StructField("V3", DoubleType()),
    #StructField("V4", DoubleType()),
    #StructField("V5", DoubleType()),

    # 6-10
    #StructField("V6", DoubleType()),
    #StructField("V7", DoubleType()),
    #StructField("V8", DoubleType()),
    #StructField("V9", DoubleType()),
    #StructField("V10", DoubleType()),

    # 11-15
    #StructField("V11", DoubleType()),
    #StructField("V12", DoubleType()),
    #StructField("V13", DoubleType()),
    #StructField("V14", DoubleType()),
    #StructField("V15", DoubleType()),

    # 16-20
    #StructField("V16", DoubleType()),
    #StructField("V17", DoubleType()),
    #StructField("V18", DoubleType()),
    #StructField("V19", DoubleType()),
    #StructField("V20", DoubleType()),

    # 21-25
    #StructField("V21", DoubleType()),
    #StructField("V22", DoubleType()),
    #StructField("V23", DoubleType()),
    #StructField("V24", DoubleType()),
    #StructField("V25", DoubleType()),

    # 26-28
    #StructField("V26", DoubleType()),
    #StructField("V27", DoubleType()),
    #StructField("V28", DoubleType()),
    
    StructField("Class", IntegerType()),
    StructField("Amount", DoubleType()),
])

# Initialisation session spark
spark = SparkSession.builder.appName("FraudDetectionProcessor").config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1").getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

raw_df = spark.readStream .format("kafka").option("kafka.bootstrap.servers", "localhost:9092").option("subscribe", "transactions").option("startingOffsets", "latest").load()

# Transformation binaire kafka en colonnes lisibles
transactions_df = raw_df.selectExpr("CAST(value AS STRING)").select(from_json(col("value"), schema).alias("data")).select("data.*")

# high_value_df = transactions_df.filter(col("Amount") > 200000)
query = transactions_df.writeStream.outputMode("append").format("console").start()

print("Spark Processor démarré ! En attente de données venant de Kafka...")
query.awaitTermination()