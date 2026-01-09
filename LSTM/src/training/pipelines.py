"""Pipeline de treino do modelo.

Contexto:
    Este módulo define uma orquestração simples para carregar dados a partir
    de uma fonte configurada, aplicar pré-processamento/feature engineering,
    delegar o treino a um ``Trainer`` e, ao final, persistir o modelo
    resultante em um caminho configurado.
"""

from src.shared.data_access import load_data, save_data
from src.shared.feature_engineering import preprocess_data
from src.training.trainer import Trainer


class TrainingPipeline:
    """Orquestra o fluxo de treino: carregar dados, preprocessar, treinar e salvar.

    Contexto:
        Esta classe encapsula o fluxo de alto nível de treino, servindo como
        um ponto único para coordenar acesso a dados, pré-processamento e
        uso de um ``Trainer`` especializado.
    """

    def __init__(self, config):
        """Inicializa a pipeline com configuração e um ``Trainer``.

        Args:
            config: Dicionário ou objeto de configuração que contenha, no
                mínimo, as chaves ``"data_source"`` (origem dos dados) e
                ``"model_output_path"`` (caminho onde o modelo será salvo).

        Returns:
            None
        """
        self.config = config
        self.trainer = Trainer(config)

    def run(self):
        """Executa o fluxo completo de treino e persistência do modelo.

        Contexto:
            Este método é a "entrada principal" da pipeline. Ele carrega os
            dados a partir da fonte configurada, aplica pré-processamento com
            ``preprocess_data``, chama o ``Trainer`` para treinar o modelo e
            persiste o artefato treinado na saída configurada.

        Returns:
            None
        """
        data = load_data(self.config["data_source"])
        processed_data = preprocess_data(data)
        model = self.trainer.train(processed_data)
        save_data(model, self.config["model_output_path"])


def main():
    """Ponto de entrada para execução manual da pipeline de treino.

    Contexto:
        Função utilitária para executar a pipeline de treino de forma
        direta (por exemplo, via linha de comando), usando uma configuração
        mínima de exemplo.

    Returns:
        None
    """
    config = {
        "data_source": "path/to/data",
        "model_output_path": "path/to/save/model",
    }
    pipeline = TrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()