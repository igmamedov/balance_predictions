FROM python:3.11

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose MLflow port
EXPOSE 5001

# Create directory for MLflow data
RUN mkdir -p /app/mlruns

# Command to run MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5001", "--backend-store-uri", "/app/mlruns", "--default-artifact-root", "s3://balance-predictions/mlflow", "--serve-artifacts"] 