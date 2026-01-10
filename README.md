# ml-stock-market-predictor

Serviço FastAPI para predição de preços de fechamento da bolsa de valores usando modelo LSTM, com fluxo de **treino via notebook** executado em background.

## Estrutura do projeto

```txt
ml-stock-market-predictor/
│
├── LSTM/
│   ├── notebooks/
│   │   └── notebook.ipynb          # notebook principal de treino
│   └── src/
│       ├── api/                    # rotas FastAPI (/predict, /train, /check_train)
│       ├── artifacts/              # artefatos de modelo treinado (weights, scaler, etc.)
│       ├── core/                   # lógica de domínio / camadas de negócio
│       ├── inference/              # pipeline de inferência (predict_next_price, etc.)
│       ├── shared/                 # persistência de predições, métricas, utilitários comuns
│       ├── training/               # helpers de treino (data loader, preparação, etc.)
│       ├── utils/                  # utilidades (logging, datas, timer, etc.)
│       └── agent/                  # lógica do agente (ex: automação, execução periódica)
│
├── data/                           # dados de entrada (CSV, base local)
├── Dockerfile
├── compose.yaml
└── README.md
```

**Notas:**
- O notebook de treino fica em `LSTM/notebooks/notebook.ipynb`.
- O código Python está todo em `LSTM/src/`, incluindo a pasta `agent/` para a lógica do agente.

---

## Rotas da API

As rotas estão definidas em [`LSTM/src/api/routes.py`](LSTM/src/api/routes.py) e carregadas em [`LSTM/src/api/main.py`](LSTM/src/api/main.py).

Base URL (local): http://127.0.0.1:8000

### GET /

Health-check.

Resposta (exemplo):

```json
{"status":"api is working"}
```

---

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
{
   "value": 29.93,
   "symbol": "PETR4.SA",
   "date": "2026-01-09T03:27:57.060199+00:00",
   "data_source_requested": "auto",
   "data_source_used": "yfinance"
}
```

---

### POST /train

Dispara o **treino em background** executando o notebook padrão:

- Host: `LSTM/notebooks/notebook.ipynb`  
- Container: `/app/LSTM/notebooks/notebook.ipynb`

A rota **apenas inicia** o fluxo em uma thread separada; o acompanhamento é feito via `GET /check_train`.

Também é possível sobrescrever o notebook via:

- query param `notebook` em `/train` (caminho absoluto ou relativo no container)
- variável de ambiente `TRAIN_NOTEBOOK` ou `TRAIN_NOTEBOOK_PATH`

Variáveis úteis para o treino:

- `TRAIN_KERNEL_NAME` (default: `python3`)
- `TRAIN_KERNEL_STARTUP_TIMEOUT_S` (default: `900` = 15min) — tempo máximo para o kernel do notebook iniciar.

Otimização do treino (Docker):

- `TRAIN_GRID_MODE`: `full` (default) | `medium` | `fast`
  - `fast` reduz a varredura (grid) e acelera bastante.
- `TRAIN_MAX_ROWS`: limita a quantidade de linhas usadas no treino (0 = sem limite).
- `TRAIN_EARLY_STOPPING_PATIENCE`: ajusta a paciência do early stopping (default: 10).

Exemplo:

```bash
curl -X POST "http://127.0.0.1:8000/train"
```

Resposta (exemplo):

```json
{
   "status": "started",
   "started_at": "2026-01-09T03:42:50.759785+00:00",
   "notebook": "/app/LSTM/notebooks/notebook.ipynb",
   "message": "Treino do modelo iniciado em background; consulte /check_train para acompanhar o status."
}
```

---

### GET /check_train

Retorna status do treino + métricas de desempenho.

Campos principais:

- `train_status`: `idle` | `running` | `succeeded` | `failed`
- `phase`: fase atual do fluxo de treino  
  (`idle`, `starting`, `find_notebook`, `prepare_paths`, `execute`, `finalizing`, `done`)
- `prereqs`: informações sobre kernel e dependências  
  (`kernel_name`, `kernel_available`, `missing_modules`, `ok`, etc.)
- `execute_engine`: engine usada para execução do notebook (ex.: `nbclient`)
- `execute_started_at` / `nbclient_started_at`: timestamps de início da execução
- `elapsed_s`: tempo decorrido (segundos) enquanto está `running`
- `duration_s`: tempo total (segundos) quando finaliza
- `metrics`: tempos por etapa (`find_notebook_s`, `prepare_paths_s`, `execute_s`),  
  `engine`, `notebook_in`, `notebook_out`
- `artifact_found`: se um artefato (modelo treinado) foi encontrado para o símbolo
- `artifact_path`: caminho absoluto do artefato encontrado

Query params úteis:

- `symbol` (opcional): ticker para o qual deseja verificar se há artefato disponível (ex.: `PETR4.SA`)

Exemplo:

```bash
curl "http://127.0.0.1:8000/check_train"
```

Exemplo (com símbolo para verificação de artefato):

```bash
curl "http://127.0.0.1:8000/check_train?symbol=PETR4.SA"
```

---

## Execução com Docker (macOS / Linux / Windows)

O container sobe a API via Uvicorn (porta `8000`). O Dockerfile está em [`Dockerfile`](Dockerfile).

### Opção recomendada (qualquer SO)

Use o Docker Compose (arquivo [`compose.yaml`](compose.yaml)):

```bash
docker compose up --build
```

Para rodar em outra plataforma (ex.: Apple Silicon usando amd64 via emulação):

```bash
DOCKER_PLATFORM=linux/amd64 docker compose up --build
```

### macOS (Apple Silicon: M1/M2/M3)

```bash
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

---

## Testes

```bash
pytest
```