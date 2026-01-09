"""Utilitários simples para persistência e carregamento de modelos via joblib."""

from joblib import dump, load


def save_model(model, filename: str) -> None:
    """Serializa e salva um modelo/objeto em disco usando joblib.

    Args:
        model: Objeto a ser persistido (ex.: modelo scikit-learn, dict, etc.).
        filename: Caminho do arquivo de saída.
    """
    dump(model, filename)


def load_model(filename: str):
    """Carrega e retorna um modelo/objeto previamente salvo com joblib.

    Args:
        filename: Caminho do arquivo a ser carregado.

    Returns:
        Objeto desserializado.
    """
    return load(filename)