import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(df: pd.DataFrame):
    """Preprocessa um DataFrame com transformações numéricas e categóricas.

    Contexto:
        Detecta automaticamente colunas numéricas e categóricas para
        aplicar ``StandardScaler`` e ``OneHotEncoder``, respectivamente,
        retornando a matriz transformada pronta para modelagem.

    Args:
        df: DataFrame de entrada contendo variáveis numéricas e/ou
            categóricas.

    Returns:
        Matriz resultante do :class:`sklearn.compose.ColumnTransformer`,
        tipicamente ``numpy.ndarray`` ou matriz *sparse*.
    """
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor.fit_transform(df)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica etapas de engenharia de features em um DataFrame.

    Contexto:
        Exemplo simples de criação de atributo derivado a partir da
        coluna ``existing_feature``. Pode ser estendido com regras de
        negócio adicionais.

    Args:
        df: DataFrame de entrada contendo, no mínimo, a coluna
            ``existing_feature``.

    Returns:
        Novo DataFrame com a coluna adicional ``new_feature``.

    Raises:
        KeyError: Se a coluna ``existing_feature`` não existir no
            DataFrame de entrada.
    """
    df = df.copy()
    df["new_feature"] = df["existing_feature"] * 2
    return df


def create_pipeline() -> Pipeline:
    """Cria um pipeline de pré-processamento e engenharia de features.

    Contexto:
        Encapsula as etapas de transformação em um único objeto
        compatível com a API de pipelines do scikit-learn, combinando
        pré-processamento e engenharia de features.

    Returns:
        Instância de :class:`sklearn.pipeline.Pipeline` contendo as
        etapas ``preprocess_data`` e ``feature_engineering``.
    """
    return Pipeline(
        steps=[
            ("preprocessor", preprocess_data),
            ("feature_engineering", feature_engineering),
        ]
    )