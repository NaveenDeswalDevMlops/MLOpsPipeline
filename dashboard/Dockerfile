FROM python:3.10-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Disable Docker proxy inside the build
ENV HTTP_PROXY=""
ENV HTTPS_PROXY=""
ENV http_proxy=""
ENV https_proxy=""
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install only required OS package
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files first for Docker layer caching
COPY requirements.txt dashboard/requirements.txt ./

# Upgrade packaging tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install \
    --no-cache-dir \
    --timeout=180 \
    --retries=5 \
    -r requirements.txt

# Copy application source
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.enableCORS", "false"]