"""Utilitários simples para persistência e carregamento de modelos via joblib.

Contexto:
    Centraliza operações básicas de serialização e desserialização de
    objetos (modelos, dicionários, etc.) usando ``joblib``, para uso
    em diferentes partes da aplicação.
"""

from joblib import dump, load


def save_model(model, filename: str) -> None:
    """Serializa e salva um modelo ou objeto arbitrário em disco.

    Contexto:
        Encapsula a chamada a :func:`joblib.dump`, permitindo que
        componentes da aplicação persistam artefatos em um único
        ponto de escrita.

    Args:
        model: Objeto a ser persistido (por exemplo, modelo
            scikit-learn, dicionário de parâmetros, etc.).
        filename: Caminho completo do arquivo de saída.

    Returns:
        None. O efeito colateral é a criação/atualização do arquivo
        no sistema de arquivos.
    """

    dump(model, filename)


def load_model(filename: str):
    """Carrega e retorna um modelo ou objeto previamente salvo com joblib.

    Contexto:
        Encapsula a chamada a :func:`joblib.load` para recuperar
        artefatos persistidos, mantendo a lógica de I/O centralizada.

    Args:
        filename: Caminho do arquivo a ser carregado.

    Returns:
        Objeto desserializado exatamente como foi salvo por
        :func:`save_model` ou uso direto de ``joblib.dump``.
    """

    return load(filename)