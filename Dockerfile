FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATABASE_URL="sqlite:////app/data/predictions.db" \
    LOG_LEVEL="INFO" \
    PYTHONPATH="/app/src" \
    KERAS_BACKEND=tensorflow \
    MODEL_ARTIFACT_DIR=/app/src/artifacts \
    MODEL_ARTIFACT_PATH=/app/src/artifacts/best_lstm_artifact.pkl

COPY pyproject.toml README.md ./
COPY src ./LSTM/src
# COPY data ./LSTM/data   # remova esta linha

# garante que o diretório existe (opcional, o volume já cria em runtime)
RUN mkdir -p /app/data

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . && \
    pip install --no-cache-dir yfinance uvicorn scikit-learn nbclient nbformat ipykernel jupyter_client

RUN python -m ipykernel install --name python3 --display-name "Python 3" --sys-prefix

EXPOSE 10000
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "10000"]
