FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY data/sample_batch.csv ./data/sample_batch.csv

ENTRYPOINT ["python", "src/inference.py"]
CMD ["--input", "data/sample_batch.csv", "--output", "outputs/predictions.csv"]
