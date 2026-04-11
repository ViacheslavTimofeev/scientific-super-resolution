FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app ./app
COPY configs ./configs
COPY outputs ./outputs
COPY scripts ./scripts
COPY src ./src

FROM base AS api

EXPOSE 8000

CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS demo

EXPOSE 7860

CMD ["python", "app/demo.py", "--host", "0.0.0.0", "--port", "7860", "--api-url", "http://api:8000"]
