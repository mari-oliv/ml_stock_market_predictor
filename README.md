# ml-stock-market-predictor

Serviço FastAPI para predição de preços de fechamento da bolsa de valores de uma empresa escolhida.

## Estrutura do projeto

```
ml-stock-market-predictor/
   artifacts/
   data/
   notebooks/
   src/
      api/
      core/
      inference/
      shared/
      training/
      utils/
   tests/
   pyproject.toml
   Dockerfile
```

## Rotas da API

As rotas estão definidas em [src/api/routes.py](src/api/routes.py) e carregadas em [src/api/main.py](src/api/main.py).

Base URL (local): http://127.0.0.1:8000

### GET /

Health-check.

Resposta (exemplo):

```json
{"status":"api is working"}
```

### POST /predict

Gera a próxima predição de preço. Por padrão (`data_source="auto"`), tenta **yfinance** e, se falhar, faz fallback para **CSV**.

Body (JSON):

- `symbol` (obrigatório): ticker (ex.: `PETR4.SA`, `AAPL`)
- `data_source` (opcional): `auto` | `yfinance` | `csv` (default: `auto`)
- `csv_path` (opcional): caminho do CSV quando usar `csv` (ou como fallback do `auto`)
- `csv_target_col` (opcional): coluna alvo do CSV (se omitida, tenta inferir)
- `artifact_path` (opcional): caminho do artefato do modelo

Exemplo (mínimo):

```bash
curl --location "http://127.0.0.1:8000/predict" \
   --header "Content-Type: application/json" \
   --data '{"symbol":"PETR4.SA"}'
```

Exemplo (auto + fallback CSV explícito):

```bash
curl --location "http://127.0.0.1:8000/predict" \
   --header "Content-Type: application/json" \
   --data '{"symbol":"PETR4.SA","data_source":"auto","csv_path":"data/finance_data.csv","csv_target_col":"Close"}'
```

Resposta (exemplo):

```json
{"value":29.93,"symbol":"PETR4.SA","date":"2026-01-09T03:27:57.060199+00:00"}
```

### POST /train

Dispara o treino em background executando o notebook padrão: `notebooks/notebook.ipynb` (no container ele fica em `/app/notebooks/notebook.ipynb`).

Variáveis úteis para o treino:

- `TRAIN_KERNEL_NAME` (default: `python3`)
- `TRAIN_KERNEL_STARTUP_TIMEOUT_S` (default: `900` = 15min) — aumenta o tempo para o kernel iniciar quando o notebook demora.

Otimização do treino (Docker):

- `TRAIN_GRID_MODE`: `full` (default) | `medium` | `fast`
   - `fast` reduz drasticamente a varredura (grid) e acelera bastante.
- `TRAIN_MAX_ROWS`: limita a quantidade de linhas usadas no treino (0 = sem limite).
- `TRAIN_EARLY_STOPPING_PATIENCE`: reduz/aumenta a paciência do early stopping (default: 10).

Exemplo:

```bash
curl -X POST "http://127.0.0.1:8000/train"

Exemplo (Docker run mais rápido):

```bash
docker run --platform linux/amd64 -d --name ml-stock-market-predictor \
   -p 8000:8000 \
   -v "$PWD/data:/app/data" \
   -e DATABASE_URL="sqlite:///data/predictions.db" \
   -e TRAIN_KERNEL_STARTUP_TIMEOUT_S=900 \
   -e TRAIN_GRID_MODE=fast \
   -e TRAIN_MAX_ROWS=2000 \
   ml-stock-market-predictor:latest
```
```

Resposta (exemplo):

```json
{
   "status": "started",
   "started_at": "2026-01-09T03:42:50.759785+00:00",
   "notebook": "/app/notebooks/notebook.ipynb",
   "message": "Treino iniciado, aguarde uns minutos antes de verificar o status."
}
```

### GET /check_train

Retorna status do treino + métricas de desempenho.

Campos principais:

- `train_status`: `idle` | `running` | `succeeded` | `failed`
- `elapsed_s`: tempo decorrido (segundos) enquanto está `running`
- `duration_s`: tempo total (segundos) quando finaliza
- `metrics`: tempos por etapa (`find_notebook_s`, `prepare_paths_s`, `execute_s`), `engine`, `notebook_in`, `notebook_out`

Exemplo:

```bash
curl "http://127.0.0.1:8000/check_train"
```

## Executar com Docker (macOS / Linux / Windows)

O container sobe a API via Uvicorn (porta `8000`). O Dockerfile está em [Dockerfile](Dockerfile).

### Opção recomendada (um comando, qualquer SO)

Use o Docker Compose (arquivo [compose.yaml](compose.yaml)). Isso funciona igual em macOS/Linux/Windows.

```bash
cd "ml-stock-market-predictor"
docker compose up --build
```

Para rodar em outra plataforma (ex.: Apple Silicon usando amd64 via emulação), você pode setar:

```bash
DOCKER_PLATFORM=linux/amd64 docker compose up --build
```

### macOS (Apple Silicon: M1/M2/M3)

Use `--platform linux/amd64` para evitar warnings e manter compatibilidade com a imagem base do TensorFlow.

```bash
cd "ml-stock-market-predictor"

docker build --platform linux/amd64 -t ml-stock-market-predictor:latest .

docker rm -f ml-stock-market-predictor 2>/dev/null || true
docker run --platform linux/amd64 -d --name ml-stock-market-predictor \
   -p 8000:8000 \
   -v "$PWD/data:/app/data" \
   -e DATABASE_URL="sqlite:///data/predictions.db" \
   ml-stock-market-predictor:latest

docker logs -f --tail=50 ml-stock-market-predictor
```

### macOS (Intel) e Linux (x86_64)

```bash
cd "ml-stock-market-predictor"

docker build -t ml-stock-market-predictor:latest .

docker rm -f ml-stock-market-predictor 2>/dev/null || true
docker run -d --name ml-stock-market-predictor \
   -p 8000:8000 \
   -v "$PWD/data:/app/data" \
   -e DATABASE_URL="sqlite:///data/predictions.db" \
   ml-stock-market-predictor:latest

docker logs -f --tail=50 ml-stock-market-predictor
```

### Windows (PowerShell)

```powershell
cd "ml-stock-market-predictor"

docker build -t ml-stock-market-predictor:latest .

docker rm -f ml-stock-market-predictor 2>$null
docker run -d --name ml-stock-market-predictor `
   -p 8000:8000 `
   -v "${PWD}\data:/app/data" `
   -e DATABASE_URL="sqlite:///data/predictions.db" `
   ml-stock-market-predictor:latest

docker logs -f --tail 50 ml-stock-market-predictor
```

### Parar/remover o container

```bash
docker rm -f ml-stock-market-predictor
```

## Testes

```bash
pytest
```