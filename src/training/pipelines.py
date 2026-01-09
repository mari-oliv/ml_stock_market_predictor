"""Pipeline de treino do modelo.

Define uma orquestração simples para carregar dados, pré-processar, treinar e
persistir o modelo treinado.
"""

from src.shared.data_access import load_data, save_data
from src.shared.feature_engineering import preprocess_data
from src.training.trainer import Trainer


class TrainingPipeline:
    """Orquestra o fluxo de treino: carregar dados, preprocessar, treinar e salvar."""

    def __init__(self, config):
        """Inicializa a pipeline com configuração e um `Trainer`.

        Args:
            config: Dicionário/objeto de configuração com chaves como `data_source`
                e `model_output_path`.
        """
        self.config = config
        self.trainer = Trainer(config)

    def run(self):
        """Executa o fluxo completo de treino e persistência do modelo."""
        data = load_data(self.config["data_source"])
        processed_data = preprocess_data(data)
        model = self.trainer.train(processed_data)
        save_data(model, self.config["model_output_path"])


def main():
    """Ponto de entrada para execução manual da pipeline de treino."""
    config = {
        "data_source": "path/to/data",
        "model_output_path": "path/to/save/model",
    }
    pipeline = TrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()