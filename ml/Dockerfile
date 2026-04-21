FROM python:3.11-slim

WORKDIR /app

COPY ml/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml/import_dataset.py .

CMD ["python", "import_dataset.py"]
