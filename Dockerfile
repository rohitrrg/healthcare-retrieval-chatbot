FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf \
    TRANSFORMERS_CACHE=/hf \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=8501 \
    MODEL_PATH=/app/Mistral-7B-Instruct-v0.3

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app
RUN pip install -r requirements.txt

# --- Copy your app code ---
COPY src/*.py /app/
COPY /data /app/data

RUN mkdir -p /hf /tmp/offload && chmod -R 777 /hf /tmp/offload

# --- Expose and run ---
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]