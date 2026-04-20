FROM python:3.10.20-alpine3.22

WORKDIR /app

COPY src/producer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/producer/ ./src/producer/
COPY data/ ./data/

CMD ["python", "src/producer/producer.py"]