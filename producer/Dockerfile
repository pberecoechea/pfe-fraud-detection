FROM python:3.10.20-alpine3.22

WORKDIR /app

COPY producer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY producer/ ./producer/
COPY data/ ./data/

CMD ["python", "producer/producer.py"]