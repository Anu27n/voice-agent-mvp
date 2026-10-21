FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    DB_PATH=/app/data/voice_agent.db \
    LOG_FORMAT=json

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml README.md ./
RUN mkdir -p voice_agent/providers \
    && touch voice_agent/__init__.py voice_agent/providers/__init__.py \
    && pip install . \
    && pip uninstall -y voice-agent-mvp \
    && rm -rf voice_agent

COPY voice_agent ./voice_agent
COPY web ./web

RUN useradd --create-home --uid 10001 agent && mkdir -p /app/data && chown agent /app/data
USER agent

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/healthz', timeout=4)"

CMD ["python", "-m", "voice_agent"]
