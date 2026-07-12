FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get -o Acquire::ForceIPv4=true update && \
    apt-get install -y build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*
#RUN apt-get update && apt-get install -y build-essential libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt dashboard/requirements.txt ./
RUN pip install --no-cache-dir --timeout=180 --retries=5 -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.enableCORS", "false"]
