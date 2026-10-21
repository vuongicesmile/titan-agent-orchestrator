FROM python:3.12.11-slim-bookworm

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir . \
    && groupadd --gid 10001 titan \
    && useradd --uid 10001 --gid titan --no-create-home titan \
    && mkdir -p /app/data \
    && chown titan:titan /app/data

USER titan

EXPOSE 8010

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health', timeout=2)"]

CMD ["python", "-m", "uvicorn", "titan_agent_orchestrator.main:app", "--host", "0.0.0.0", "--port", "8010"]
