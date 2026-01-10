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

# Se pyproject.toml e README.md continuam na raiz do repo:
COPY pyproject.toml README.md ./

# Código agora está em LSTM/src → copiar para /app/src, que bate com PYTHONPATH
COPY LSTM/src ./src

# Copia o notebook de treino para o caminho correto no container
COPY LSTM/notebooks ./LSTM/notebooks

# NÃO copiar data pro container (útil só em treino, evita erro se não estiver no Git)
# COPY LSTM/data ./data

RUN mkdir -p /app/data

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . && \
    pip install --no-cache-dir yfinance uvicorn scikit-learn nbclient nbformat ipykernel jupyter_client

RUN python -m ipykernel install --name python3 --display-name "Python 3" --sys-prefix

EXPOSE 10000
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "10000"]
