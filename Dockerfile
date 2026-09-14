FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    CHROMA_PATH=/app/storage/chroma

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /app/storage/chroma /home/app/.cache/huggingface \
    && chown -R app:app /app/storage /home/app/.cache

COPY requirements.txt constraints-docker.txt ./
# Embeddings need CPU PyTorch only. Ollama runs separately on the host.
RUN python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt -c constraints-docker.txt \
    && python -m pip check \
    && python -m pip freeze > /app/installed-requirements.txt

COPY *.py ./
COPY data/tickets.csv data/evaluation_cases.json ./data/

USER app
EXPOSE 8000

# Readiness includes dependencies. Docker reports unhealthy; it does not restart
# a container solely because this check fails.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health/ready', timeout=8).close()"

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
