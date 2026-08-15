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
    pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY config.py .
COPY network.py .
COPY dataset.py .
COPY utils ./utils
COPY results ./results

RUN mkdir -p models && \
    curl -L --fail --retry 5 \
    -o models/thyroidnet_best.pth \
    https://huggingface.co/AyushiiKumarii/thyroidnet/resolve/main/thyroidnet_best.pth?download=true && \
    curl -L --fail --retry 5 \
    -o models/support_bank.pt \
    https://huggingface.co/AyushiiKumarii/thyroidnet/resolve/main/support_bank.pt?download=true

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none"]

