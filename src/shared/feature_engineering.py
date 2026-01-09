import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(df: pd.DataFrame):
    """Preprocessa um DataFrame aplicando transformações numéricas e categóricas.

    Seleciona automaticamente:
    - colunas numéricas (`int64`, `float64`) para `StandardScaler`
    - colunas categóricas (`object`) para `OneHotEncoder`

    Args:
        df: DataFrame de entrada.

    Returns:
        Matriz transformada resultante do `ColumnTransformer` (tipicamente numpy array ou sparse).
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

    Args:
        df: DataFrame de entrada.

    Returns:
        DataFrame com features adicionais.

    Raises:
        KeyError: Se as colunas esperadas não existirem no DataFrame.
    """
    df = df.copy()
    df["new_feature"] = df["existing_feature"] * 2
    return df


def create_pipeline() -> Pipeline:
    """Cria um pipeline de pré-processamento e engenharia de features.

    Returns:
        Pipeline do scikit-learn contendo as etapas de preprocessamento e feature engineering.
    """
    return Pipeline(
        steps=[
            ("preprocessor", preprocess_data),
            ("feature_engineering", feature_engineering),
        ]
    )