# AI Ops Copilot - one image for every Python service (simulator, API, worker, dashboard).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    DATA_DIR=/app/data \
    MODEL_DIR=/app/models \
    REPORTS_DIR=/app/reports

WORKDIR /app

# CPU-only PyTorch keeps the image ~1.5 GB smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY dashboard ./dashboard
COPY knowledge_base ./knowledge_base
COPY reports ./reports

# Run as an unprivileged user; volumes mounted on these folders inherit the ownership.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data/live /app/models \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8001 8501
CMD ["python", "-m", "aiops_copilot.simulator"]
