FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir huggingface_hub

COPY app.py .
COPY config.py .
COPY network.py .
COPY dataset.py .
COPY utils ./utils
COPY results ./results

RUN mkdir -p /app/models

RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='AyushiiKumarii/thyroidnet', filename='thyroidnet_best.pth', local_dir='/app/models'); hf_hub_download(repo_id='AyushiiKumarii/thyroidnet', filename='support_bank.pt', local_dir='/app/models')"

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none"]