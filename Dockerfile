# Python 3.12 slim base
FROM python:3.12-slim

# Avoid .pyc, set UTF-8
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (ca-certificates, minimal build utils)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY main.py .

# Cloud Run will inject $PORT; listen on 0.0.0.0:$PORT
ENV HOST=0.0.0.0
# keep your existing content, just change CMD:
ENV PORT=8080
CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
    
