"""Treinador genérico para modelos com interface estilo PyTorch.

Contexto:
    Este módulo fornece uma implementação simples de um *trainer* genérico,
    compatível com modelos e data loaders no estilo PyTorch. Ele expõe
    operações para configurar componentes (modelo, data loader, otimizador e
    função de perda), executar o loop de treino por múltiplas épocas e avaliar
    em um loader de validação.
"""

from typing import Any, Optional

try:
    import torch
except Exception:
    torch = None


class Trainer:
    """Encapsula rotinas simples de treino e avaliação.

    Contexto:
        Esta classe assume que o modelo, o otimizador e a função de perda
        seguem a interface básica do PyTorch (métodos como ``zero_grad()``,
        ``step()`` e suporte a ``backward()``), e que os data loaders
        produzem *batches* com chaves ``"input"`` e ``"target"``.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        data_loader: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        loss_fn: Optional[Any] = None,
    ):
        """Inicializa o ``Trainer`` com dependências opcionais.

        Contexto:
            Permite injetar desde o início todos os componentes necessários
            para treino e avaliação, mas eles também podem ser configurados ou
            atualizados depois via :meth:`setup`.

        Args:
            model: Modelo a ser treinado/avaliado.
            data_loader: Iterável de *batches* para treino.
            optimizer: Otimizador (por exemplo, ``torch.optim.*``) com métodos
                ``zero_grad()`` e ``step()``.
            loss_fn: Função de perda chamável que recebe ``(outputs, target)``.

        Returns:
            None
        """
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def setup(self, model: Any, data_loader: Any, optimizer: Any, loss_fn: Any) -> None:
        """Configura ou atualiza os componentes do ``Trainer``.

        Contexto:
            Pode ser usado tanto para a configuração inicial quanto para
            reconfigurar o trainer com novos componentes (por exemplo, em um
            ciclo de experimentação).

        Args:
            model: Modelo a ser usado para treino/avaliação.
            data_loader: Loader/iterável de *batches* para treino.
            optimizer: Otimizador associado ao modelo.
            loss_fn: Função de perda.

        Returns:
            None
        """
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def _check_ready(self, require_torch: bool = False) -> None:
        """Valida se o ``Trainer`` está pronto para uso.

        Contexto:
            Garante que modelo, data loader, otimizador e função de perda
            estejam configurados antes de iniciar treino ou avaliação. Quando
            ``require_torch=True``, também verifica se a biblioteca PyTorch
            está disponível.

        Args:
            require_torch: Se ``True``, exige que ``torch`` esteja disponível.

        Raises:
            RuntimeError: Se algum componente obrigatório não estiver configurado.
            ImportError: Se ``require_torch=True`` e o PyTorch não estiver instalado.
        """
        if (
            self.model is None
            or self.data_loader is None
            or self.optimizer is None
            or self.loss_fn is None
        ):
            raise RuntimeError(
                "Trainer não está configurado. Chame setup(model, data_loader, optimizer, loss_fn)."
            )
        if require_torch and torch is None:
            raise ImportError("PyTorch não está instalado. Instale 'torch' para usar evaluate().")

    def train(self, epochs: int = 1, data_loader: Optional[Any] = None) -> None:
        """Executa o loop de treino por um número de épocas.

        Contexto:
            Realiza o loop clássico de treino: para cada *batch* no
            ``data_loader``, zera gradientes, faz *forward*, calcula a perda,
            executa ``backward()`` e aplica ``optimizer.step()``.

        Args:
            epochs: Número de épocas de treino.
            data_loader: Loader opcional para sobrescrever ``self.data_loader``.

        Raises:
            RuntimeError: Se o ``data_loader`` estiver ausente ou o ``Trainer``
                não estiver devidamente configurado.

        Returns:
            None
        """
        dl = data_loader or self.data_loader
        if dl is None:
            raise RuntimeError(
                "DataLoader ausente. Informe em train(..., data_loader=...) ou chame setup()."
            )
        self._check_ready()

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dl:
                self.optimizer.zero_grad()
                outputs = self.model(batch["input"])
                loss = self.loss_fn(outputs, batch["target"])
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.item())
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dl):.6f}")

    def evaluate(self, validation_loader: Any) -> float:
        """Avalia o modelo em um loader de validação e retorna a perda média.

        Contexto:
            Executa um *forward* sem gradientes (``torch.no_grad()``) sobre o
            loader de validação e acumula a perda média, útil para monitorar o
            desempenho fora do conjunto de treino.

        Args:
            validation_loader: Loader/iterável de *batches* de validação.

        Returns:
            float: Perda média no conjunto de validação.

        Raises:
            RuntimeError: Se o ``Trainer`` não estiver configurado.
            ImportError: Se o PyTorch não estiver instalado.
        """
        self._check_ready(require_torch=True)
        total_loss = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                outputs = self.model(batch["input"])
                loss = self.loss_fn(outputs, batch["target"])
                total_loss += float(loss.item())
        return total_loss / len(validation_loader)