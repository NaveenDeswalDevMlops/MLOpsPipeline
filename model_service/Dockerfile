FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get -o Acquire::ForceIPv4=true update && \
    apt-get install -y build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --timeout=180 --retries=5 -r requirements.txt

COPY model_service ./model_service
COPY models ./models

EXPOSE 8001

CMD ["uvicorn", "model_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
