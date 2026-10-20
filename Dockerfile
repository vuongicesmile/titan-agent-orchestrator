FROM python:3.12.11-slim-bookworm

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 8010

CMD ["fastapi", "run", "src/titan_agent_orchestrator/main.py", "--host", "0.0.0.0", "--port", "8010"]
