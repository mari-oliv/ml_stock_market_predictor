FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATABASE_URL="sqlite:///data/predictions.db" \
    LOG_LEVEL="INFO" \
    PYTHONPATH="/app" \
    KERAS_BACKEND=tensorflow \
    MODEL_ARTIFACT_DIR=/app/artifacts \
    MODEL_ARTIFACT_PATH=/app/artifacts/best_lstm_artifact.pkl

COPY pyproject.toml README.md ./
COPY src ./src
# Os artefatos do modelo ficam versionados em src/artifacts
COPY src/artifacts ./artifacts
COPY notebooks ./notebooks
COPY data ./data

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . && \
    pip install --no-cache-dir yfinance uvicorn scikit-learn nbclient nbformat ipykernel jupyter_client

RUN python -m ipykernel install --name python3 --display-name "Python 3" --sys-prefix

RUN mkdir -p /app/data

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]