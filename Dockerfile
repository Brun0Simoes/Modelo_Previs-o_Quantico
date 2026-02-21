FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r /workspace/requirements.txt && \
    python -m pip check

COPY . /workspace

EXPOSE 8000

CMD ["uvicorn", "previsao_tempo_quantico.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
