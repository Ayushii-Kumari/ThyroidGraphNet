FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY config.py .
COPY dataset.py .
COPY network.py .
COPY train.py .
COPY evaluate.py .
COPY test.py .

COPY utils ./utils
COPY results ./results

RUN mkdir -p models && \
    python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/AyushiiKumarii/thyroidnet/resolve/main/thyroidnet_best.pth', 'models/thyroidnet_best.pth')" && \
    python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/AyushiiKumarii/thyroidnet/resolve/main/support_bank.pt', 'models/support_bank.pt')"

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]