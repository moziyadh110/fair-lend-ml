# CreditShield real-time API. Build from the project root (the API imports
# explainability/ too):
#   docker build -f serving/Dockerfile -t creditshield-api .
# Needs a trained artifacts/champion_model.joblib (models/train_champion.py) —
# it's gitignored, so the model is baked in at build time, not pulled from git.

FROM python:3.13-slim AS builder
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY serving/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

FROM python:3.13-slim
# libgomp: OpenMP runtime CatBoost needs at import time; slim doesn't ship it.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY --chown=app:app serving/ serving/
COPY --chown=app:app explainability/ explainability/
COPY --chown=app:app artifacts/champion_model.joblib artifacts/champion_model.joblib

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["python", "-m", "uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
