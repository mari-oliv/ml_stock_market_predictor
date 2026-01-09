"""Rotinas de inferência para predição do próximo preço.

Contexto:
    Este módulo resolve o caminho do artefato treinado (pickle via
    ``joblib``), reconstrói o modelo Keras a partir de configurações e
    pesos armazenados e executa a predição com base em uma janela
    recente de valores, podendo carregar séries históricas tanto de
    CSV quanto de ``yfinance``.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import tensorflow as tf


tf = None


def _ensure_tf() -> None:
    """Garante o carregamento tardio do TensorFlow para uso em inferência.

    Contexto:
        Evita que a importação do projeto falhe em ambientes sem
        TensorFlow instalado, desde que as rotinas de inferência não
        sejam executadas. O módulo é carregado apenas quando
        necessário, via import dinâmico.

    Returns:
        None.

    Raises:
        RuntimeError: Se TensorFlow não estiver disponível no ambiente
            no momento em que a inferência é requerida.
    """
    global tf
    if tf is not None:
        return
    try:
        import tensorflow as _tf

        tf = _tf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow não está instalado neste ambiente. "
            "Instale tensorflow (ou use o Docker) para executar /predict com o modelo."
        ) from exc

logger = logging.getLogger("app.inference")


def _project_root() -> str:
    """Obtém o caminho absoluto da raiz do projeto.

    Contexto:
        Parte da localização deste módulo de inferência para subir na
        árvore de diretórios até a raiz do repositório, permitindo
        resolver caminhos relativos de dados e artefatos.

    Returns:
        Caminho absoluto da raiz do projeto como string.
    """

    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def _default_fallback_csv_path() -> str:
    """Retorna o caminho padrão do CSV de fallback na pasta ``data``.

    Contexto:
        Usado quando nenhuma origem CSV é explicitamente informada,
        assumindo o arquivo padrão ``data/finance_data.csv`` sob a
        raiz do projeto.

    Returns:
        Caminho absoluto para o CSV de fallback.
    """

    return os.path.join(_project_root(), "data", "finance_data.csv")


def _infer_csv_target_col(df: pd.DataFrame, symbol: Optional[str] = None) -> str:
    """Infere a coluna alvo de um CSV com base em símbolo e heurísticas.

    Contexto:
        Tenta primeiro uma coluna cujo nome seja igual ao ``symbol``,
        depois nomes típicos de preço de fechamento (``Close``,
        ``Adj Close``, etc.) e, em último caso, seleciona a primeira
        coluna numérica disponível.

    Args:
        df: DataFrame carregado a partir do CSV bruto.
        symbol: Símbolo (ticker) para priorizar coluna com mesmo nome.

    Returns:
        Nome da coluna alvo escolhida.

    Raises:
        ValueError: Se não for possível inferir uma coluna alvo válida.
    """
    if symbol and symbol in df.columns:
        return symbol

    preferred = ["Close", "close", "Adj Close", "adj_close", "AdjClose", "adjclose"]
    for c in preferred:
        if c in df.columns:
            return c

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return numeric_cols[0]

    raise ValueError(
        "Não foi possível inferir a coluna alvo no CSV (ex.: 'Close' ou o symbol). "
        "Informe csv_target_col."
    )


def _resolve_artifact_path(artifact_path: Optional[str], symbol: Optional[str] = None) -> str:
    """Resolve o caminho do artefato (.pkl) a partir de parâmetro, env vars e fallback.

    Ordem de resolução:
    1) `artifact_path` (parâmetro explícito)
    2) `MODEL_ARTIFACT_PATH` (arquivo direto)
    3) procura por `best_lstm_artifact.pkl` em diretórios candidatos
    4) fallback para o .pkl mais recente em diretórios candidatos

    Args:
        artifact_path: Caminho opcional do artefato.
        symbol: Símbolo (não é usado diretamente na busca, mantido por compatibilidade).

    Returns:
        Caminho absoluto do artefato.

    Raises:
        FileNotFoundError: Se nenhum artefato for encontrado.
    """
    FIXED_NAME = "best_lstm_artifact.pkl"

    def is_file(p: Optional[str]) -> Optional[str]:
        """Normaliza caminho e verifica se referencia um arquivo existente.

        Contexto:
            Facilita a validação de caminhos vindos de parâmetro ou
            variáveis de ambiente, retornando ``None`` quando o arquivo
            não existe.

        Args:
            p: Caminho potencial para o artefato.

        Returns:
            Caminho absoluto validado ou ``None`` se o arquivo não
            estiver presente.
        """
        if not p:
            return None
        p = os.path.abspath(os.path.expanduser(p))
        return p if os.path.isfile(p) else None

    p = is_file(artifact_path)
    if p:
        logger.info(f"artifact:resolved via param path={p}")
        return p

    p = is_file(os.getenv("MODEL_ARTIFACT_PATH"))
    if p:
        logger.info(f"artifact:resolved via env MODEL_ARTIFACT_PATH path={p}")
        return p

    dir_candidates: list[str] = []
    env_dir = os.getenv("MODEL_ARTIFACT_DIR")
    if env_dir:
        dir_candidates.append(os.path.abspath(os.path.expanduser(env_dir)))

    root = _project_root()
    here = os.path.dirname(__file__)
    dir_candidates.extend(
        [
            os.path.join(root, "src", "artifacts"),
            os.path.join(root, "artifacts"),
            os.path.abspath(os.path.join(here, "../artifacts")),
            os.path.abspath(os.path.join(here, "../../artifacts")),
        ]
    )

    logger.info(f"artifact:search dirs={dir_candidates}")

    for d in dir_candidates:
        candidate = os.path.abspath(os.path.join(d, FIXED_NAME))
        if os.path.isfile(candidate):
            logger.info(f"artifact:resolved fixed path={candidate}")
            return candidate

    for d in dir_candidates:
        if not os.path.isdir(d):
            continue
        matches = glob.glob(os.path.join(d, "*.pkl"))
        if matches:
            best = max(matches, key=lambda x: os.path.getmtime(x))
            logger.info(f"artifact:resolved fallback dir={d} selected={best}")
            return best

    raise FileNotFoundError(
        "Artefato não encontrado. Esperado best_lstm_artifact.pkl em src/artifacts ou artifacts, "
        "ou defina MODEL_ARTIFACT_PATH/MODEL_ARTIFACT_DIR."
    )


def _sanitize_keras_config(obj: Any) -> Any:
    """Normaliza estruturas de configuração para compatibilidade TF/Keras.

    Contexto:
        Aplica correções em campos conhecidos de configs de modelo e
        camadas para reduzir quebras entre versões diferentes de
        TensorFlow/Keras.

    Ajustes aplicados:
        * Converte ``DTypePolicy`` em string (por exemplo, ``"float32"``).
        * Remove as chaves ``build_config`` e ``build_input_shape``.
        * Em ``InputLayer``, mapeia ``batch_shape`` para
          ``batch_input_shape`` e remove ``batch_shape``.

    Args:
        obj: Estrutura arbitrária (``dict``, ``list`` ou valor
            escalar) representando a configuração do modelo ou layers.

    Returns:
        Estrutura sanitizada, preservando o formato geral de entrada.
    """
    if isinstance(obj, dict):
        if obj.get("class_name") == "InputLayer" and isinstance(obj.get("config"), dict):
            il_cfg = obj["config"]
            if "batch_shape" in il_cfg and "batch_input_shape" not in il_cfg:
                il_cfg["batch_input_shape"] = il_cfg["batch_shape"]
            il_cfg.pop("batch_shape", None)

        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k in ("build_config", "build_input_shape"):
                continue

            if k == "dtype" and isinstance(v, dict) and v.get("class_name") == "DTypePolicy":
                out[k] = (v.get("config") or {}).get("name", "float32")
                continue

            out[k] = _sanitize_keras_config(v)
        return out

    if isinstance(obj, list):
        return [_sanitize_keras_config(x) for x in obj]

    return obj


def _infer_input_shape_from_cfg(cfg: Any) -> Optional[tuple]:
    """Tenta inferir o ``input_shape`` a partir da configuração do modelo.

    Contexto:
        Extrai shapes de entrada a partir de campos como
        ``build_input_shape``, ``batch_input_shape`` ou ``batch_shape``,
        usados posteriormente para invocar ``build`` manualmente.

    Args:
        cfg: Configuração do modelo, preferencialmente um ``dict``.

    Returns:
        Tupla representando o shape de entrada inferido, ou ``None``
        quando não for possível deduzir.
    """
    if not isinstance(cfg, dict):
        return None

    bis = cfg.get("build_input_shape")
    if isinstance(bis, (list, tuple)) and len(bis) >= 2:
        return tuple(bis)

    layers = cfg.get("layers")
    if isinstance(layers, list) and layers:
        lc = (layers[0] or {}).get("config") or {}
        for key in ("batch_input_shape", "batch_shape"):
            shp = lc.get(key)
            if isinstance(shp, (list, tuple)) and len(shp) >= 2:
                return tuple(shp)

    return None


def _deserialize_model_from_artifact_cfg(raw_cfg: Any) -> tf.keras.Model:
    """Desserializa um modelo Keras a partir de múltiplos formatos de config.

    Contexto:
        Permite reconstituir o modelo armazenado no artefato mesmo
        quando a configuração foi salva em formatos diferentes
        (JSON de ``model.to_json()``, ``Sequential.get_config()``,
        lista de camadas, etc.).

    Args:
        raw_cfg: Configuração bruta armazenada no artefato.

    Returns:
        Instância de :class:`tf.keras.Model` reconstruída.

    Raises:
        ValueError: Se o formato de configuração não for suportado.
    """
    _ensure_tf()
    cfg = raw_cfg
    if isinstance(cfg, str):
        cfg = json.loads(cfg)

    cfg = _sanitize_keras_config(cfg)

    if isinstance(cfg, dict) and "class_name" in cfg and "config" in cfg:
        return tf.keras.models.model_from_json(json.dumps(cfg))

    if isinstance(cfg, dict) and "layers" in cfg:
        try:
            return tf.keras.Sequential.from_config(cfg)
        except Exception as e:
            logger.warning(
                f"artifact:Sequential.from_config failed, falling back to manual build: {e}"
            )

        seq = tf.keras.Sequential(name=cfg.get("name", "sequential"))
        layers_cfg = cfg.get("layers") or []
        if not isinstance(layers_cfg, list) or not layers_cfg:
            raise ValueError("Config inválido: esperado lista não-vazia em cfg['layers'].")

        for layer_cfg in layers_cfg:
            layer = tf.keras.layers.deserialize(layer_cfg)
            seq.add(layer)

        input_shape = _infer_input_shape_from_cfg(cfg)
        if input_shape is not None:
            try:
                seq.build(input_shape)
            except Exception as e:
                logger.debug(f"artifact:manual seq.build skipped: {e}")

        return seq

    if isinstance(cfg, list):
        return tf.keras.Sequential.from_config(cfg)

    raise ValueError(
        f"Formato de config não suportado para desserialização: type={type(cfg)} "
        f"keys={list(cfg.keys()) if isinstance(cfg, dict) else None}"
    )


def load_artifact_pkl(path: str) -> Dict[str, Any]:
    """Carrega e valida um artefato ``.pkl`` e reconstrói o modelo Keras.

    Contexto:
        Lê o arquivo serializado via ``joblib``, reconstrói o modelo,
        aplica pesos, recompila conforme configuração armazenada e
        expõe componentes principais em um dicionário.

    Args:
        path: Caminho para o arquivo de artefato ``.pkl``.

    Returns:
        Dicionário contendo, no mínimo, as chaves ``model``, ``scaler``,
        ``window_size``, ``feature_names``, ``metadata`` e
        ``best_params``.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o conteúdo do artefato estiver inválido ou
            incompleto.
        RuntimeError: Se falhar a desserialização do modelo ou a
            aplicação de pesos.
    """
    _ensure_tf()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Artifact não encontrado: {path}")

    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise ValueError(
            "Artifact inválido: esperado dict com chaves como model_json, model_weights, scaler, window_size."
        )

    cfg = artifact.get("model_json")
    weights = artifact.get("model_weights")
    if cfg is None or weights is None:
        raise ValueError("Artifact incompleto: faltam 'model_json' e/ou 'model_weights'.")

    try:
        model = _deserialize_model_from_artifact_cfg(cfg)
    except Exception as e:
        raise RuntimeError(f"Failed to deserialize model config: {e}")

    try:
        if not model.built:
            window_size = int(artifact.get("window_size", 30))
            try:
                model.build((None, window_size, 1))
            except Exception:
                pass
        model.set_weights(weights)
    except Exception as e:
        raise RuntimeError(f"Falha ao setar pesos: {e}")

    compile_cfg = artifact.get("compile", {"loss": "mse", "optimizer": "adam"})
    try:
        opt = compile_cfg.get("optimizer", "adam")
        loss = compile_cfg.get("loss", "mse")
        model.compile(optimizer=opt, loss=loss)
    except Exception as e:
        logger.debug(f"compile fallback: {e}")
        model.compile(optimizer="adam", loss="mse")

    scaler = artifact.get("scaler")
    window_size = int(artifact.get("window_size", 30))
    feature_names = artifact.get("feature_names", None)

    return {
        "model": model,
        "scaler": scaler,
        "window_size": window_size,
        "feature_names": feature_names,
        "metadata": artifact.get("metadata", {}),
        "best_params": artifact.get("best_params", {}),
    }


def predict_next_from_recent(
    artifact: Dict[str, Any],
    recent_values: Union[pd.Series, np.ndarray],
) -> float:
    """Prediz o próximo valor real a partir de uma janela recente.

    Contexto:
        Usa o scaler e o modelo contidos no artefato para normalizar a
        última janela de valores, realizar a inferência e retornar o
        próximo valor na escala original.

    Args:
        artifact: Artefato carregado por :func:`load_artifact_pkl`.
        recent_values: Série ou array 1D com os valores históricos
            mais recentes; deve conter pelo menos ``window_size``
            observações.

    Returns:
        Próximo valor previsto em escala real (``float``).

    Raises:
        ValueError: Se ``recent_values`` não tiver amostras suficientes.
    """
    _ensure_tf()
    scaler = artifact["scaler"]
    window_size = artifact["window_size"]
    model: tf.keras.Model = artifact["model"]

    if isinstance(recent_values, pd.Series):
        recent = recent_values.values.astype(float)
    else:
        recent = np.asarray(recent_values, dtype=float)

    if len(recent) < window_size:
        raise ValueError(f"Precisa de pelo menos {window_size} valores recentes.")

    last_window = recent[-window_size:].reshape(-1, 1)
    last_window_scaled = scaler.transform(last_window).reshape(1, window_size, 1)

    pred_scaled = model.predict(last_window_scaled, verbose=0).reshape(-1, 1)
    pred_real = scaler.inverse_transform(pred_scaled).reshape(-1)[0]
    return float(pred_real)


def _load_series_from_csv(
    csv_path: str,
    date_col: Optional[str],
    target_col: Optional[str],
    symbol: Optional[str] = None,
) -> pd.Series:
    """Carrega uma série temporal a partir de CSV, inferindo colunas quando necessário.

    Contexto:
        Constrói uma série indexada por datas, usando as colunas de
        data e alvo informadas ou inferidas por heurísticas, para
        posterior uso em inferência ou treinamento.

    Args:
        csv_path: Caminho do arquivo CSV.
        date_col: Nome da coluna de data; se ``None``, tenta inferir.
        target_col: Nome da coluna alvo; se ``None``, tenta inferir.
        symbol: Símbolo usado como pista para inferência da coluna alvo.

    Returns:
        Série Pandas com índice datetime e valores ``float``.

    Raises:
        ValueError: Se não for possível inferir ou encontrar colunas
            obrigatórias de data/alvo.
    """
    df = pd.read_csv(csv_path)

    if date_col is None:
        for c in df.columns:
            if str(c).lower() in ("date", "data", "dt"):
                date_col = c
                break
    if date_col is None:
        raise ValueError("Não foi possível inferir a coluna de data. Passe 'date_col'.")

    if target_col is None:
        target_col = _infer_csv_target_col(df, symbol=symbol)

    if target_col not in df.columns:
        raise ValueError(
            f"Coluna alvo '{target_col}' não encontrada no CSV. Colunas={list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, target_col]).sort_values(date_col)

    s = df[target_col].astype(float)
    s.index = df[date_col].values
    s.name = target_col
    return s


def _load_series_from_yf(symbol: str) -> pd.Series:
    """Carrega preços históricos via ``yfinance`` e devolve a série de fechamento.

    Contexto:
        Faz o download de cerca de 6 meses de dados, trata multi-index,
        garante a presença de colunas ``Close`` e ``Date`` e retorna
        uma série univariada pronta para uso em inferência.

    Args:
        symbol: Ticker no formato aceito pelo ``yfinance`` (por exemplo,
            ``"PETR4.SA"``).

    Returns:
        Série de preços de fechamento indexada por data.

    Raises:
        ValueError: Se o retorno do ``yfinance`` não contiver as
            colunas esperadas.
    """
    import yfinance as yf

    df = yf.download(symbol, period="6mo", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", symbol) in df.columns:
            close = df[("Close", symbol)]
        else:
            close_cols = [c for c in df.columns if c[0] == "Close"]
            if not close_cols:
                raise ValueError(f"yfinance: não encontrou coluna Close. colunas={df.columns}")
            close = df[close_cols[0]]
        df = close.to_frame(name="Close")
    else:
        if "Close" not in df.columns:
            raise ValueError(f"yfinance: não encontrou coluna Close. colunas={list(df.columns)}")
        df = df[["Close"]].copy()

    df = df.reset_index()
    if "Date" not in df.columns:
        raise ValueError(
            f"yfinance: não encontrou coluna Date após reset_index. colunas={list(df.columns)}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")

    s = df["Close"].astype(float)
    s.index = df["Date"].values
    s.name = symbol
    return s


def predict_next_price(
    symbol: str,
    *,
    artifact_path: Optional[str] = None,
    data_source: str = "yfinance",
    csv_path: Optional[str] = None,
    date_col: Optional[str] = None,
    csv_target_col: Optional[str] = None,
) -> Tuple[float, str]:
    """Prediz o próximo preço para um símbolo usando artefato salvo.

    Contexto:
        Resolve o artefato de modelo, carrega a série histórica a
        partir de ``yfinance`` ou CSV e delega a inferência para
        :func:`predict_next_from_recent`, retornando também qual fonte
        de dados foi efetivamente utilizada.

    Args:
        symbol: Ticker do ativo a ser previsto.
        artifact_path: Caminho opcional para o artefato ``.pkl``;
            quando ``None``, é resolvido por heurísticas internas.
        data_source: Origem preferencial dos dados ("yfinance",
            "csv" ou "auto").
        csv_path: Caminho opcional para um CSV local; usado quando
            ``data_source == "csv"`` ou como fallback.
        date_col: Nome da coluna de data no CSV, quando aplicável.
        csv_target_col: Nome da coluna alvo no CSV, quando aplicável.

    Returns:
        Tupla ``(valor_previsto, fonte_utilizada)``, onde
        ``fonte_utilizada`` é "yfinance" ou "csv".
    """
    path = _resolve_artifact_path(artifact_path, symbol=symbol)
    logger.info(f"predict:artifact path={path}")
    art = load_artifact_pkl(path)

    series: pd.Series
    data_source_used: Optional[str] = None

    if data_source == "csv":
        final_csv_path = csv_path or _default_fallback_csv_path()
        series = _load_series_from_csv(final_csv_path, date_col, csv_target_col, symbol=symbol)
        data_source_used = "csv"
    else:
        try:
            series = _load_series_from_yf(symbol)
            data_source_used = "yfinance"
        except Exception as e:
            fallback_csv = csv_path or _default_fallback_csv_path()
            logger.warning(
                f"predict:data_source yfinance failed ({e}); falling back to csv={fallback_csv}"
            )
            series = _load_series_from_csv(fallback_csv, date_col, csv_target_col, symbol=symbol)
            data_source_used = "csv"

    recent = series.iloc[-art["window_size"] :]
    value = predict_next_from_recent(art, recent)
    logger.info(
        "predict:completed symbol=%s data_source_requested=%s data_source_used=%s value=%.4f",
        symbol,
        data_source,
        data_source_used,
        value,
    )
    return value, (data_source_used or data_source)