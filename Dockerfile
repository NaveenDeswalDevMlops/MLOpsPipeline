FROM python:3.10-slim

WORKDIR /workspace

# Install system dependencies
RUN apt-get -o Acquire::ForceIPv4=true update && \
    apt-get install -y build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*
#RUN apt-get update && apt-get install -y build-essential libgomp1 make && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=180 --retries=5 -r requirements.txt

COPY . .

# Expose API port
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]