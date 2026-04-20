FROM apache/spark-py:latest

USER root
WORKDIR /app

COPY src/processing/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/processing/processor.py ./src/processing/

CMD ["/opt/spark/bin/spark-submit", "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.0,org.apache.commons:commons-pool2:2.12.0 ", "src/processing/processor.py"]