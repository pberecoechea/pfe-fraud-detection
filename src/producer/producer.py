# Fichier producer.py
# Récupère les transactions depuis le fichier CSV et les envoi à Kafka
#
# Pablo BERECOECHEA
# 24/03/26

import pandas as pd
from kafka import KafkaProducer
import json
import random
import time
import os

# Paramètres
TOPIC_NAME = 'transactions'
KAFKA_SERVER = 'redpanda:9092'
CSV_PATH = os.path.join(os.path.dirname(__file__), '../../data/ccfraud_dataset1.csv')

# Initialisation
producer = KafkaProducer(
    bootstrap_servers=[KAFKA_SERVER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def stream_csv():
    client_ids = [f"C{i:04d}" for i in range(1,1001)]

    if not os.path.exists(CSV_PATH):
        print(f"Erreur : Le fichier CSV est introuvable à l'adresse : {CSV_PATH}")
        return

    print(f"Lecture du CSV et envoi vers le topic '{TOPIC_NAME}'")
    
    for chunk in pd.read_csv(CSV_PATH, chunksize=1):
        record = chunk.to_dict(orient='records')[0]
        record['client_id'] = random.choice(client_ids)
        producer.send(TOPIC_NAME, value=record)
        print(f"Envoyé: Fake_Cli : {record['client_id']} | Step {record['Time']} | Type: {record['Class']} | Montant: {record['Amount']} €")
        time.sleep(0.1)

if __name__ == "__main__":
    try:
        stream_csv()
    except KeyboardInterrupt:
        print("\n# == Arrêt des transaction == #")
    finally:
        producer.close()