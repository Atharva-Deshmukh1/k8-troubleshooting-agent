FROM python:3.12-slim

# Install kubectl
RUN apt-get update && apt-get install -y curl ca-certificates && \
    curl -LO "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    chmod +x kubectl && mv kubectl /usr/local/bin/kubectl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY agent/ ./agent/
COPY pod_agent.py .

# .env is NOT copied — pass secrets via env vars at runtime
# GEMINI_API_KEY, K8S_NAMESPACE, GEMINI_MODEL, LOG_TAIL_LINES

ENTRYPOINT ["python3", "pod_agent.py"]
CMD ["--chat"]