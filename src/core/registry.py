"""Infraestrutura de registro simples de componentes da aplicação.

Contexto:
    Fornece um *service locator* minimalista para armazenar e recuperar
    instâncias de serviços ou objetos compartilhados por nome, sem
    acoplamento direto entre módulos.
"""

from typing import Any, Dict


class Registry:
    """Registro de componentes por nome.

    Contexto:
        Utilizado como ponto central de cadastro e resolução de
        dependências simples na aplicação, permitindo que diferentes
        partes do código compartilhem instâncias sem conhecerem suas
        implementações concretas.
    """

    def __init__(self) -> None:
        """Inicializa o registro vazio de componentes.

        Contexto:
            Cria o dicionário interno que mapeia nomes de componentes
            para suas instâncias.

        Returns:
            None.
        """

        self._components: Dict[str, Any] = {}

    def register(self, name: str, component: Any) -> None:
        """Registra um componente por nome.

        Contexto:
            Permite associar um identificador textual a uma instância
            concreta (por exemplo, um cliente de banco, serviço de
            previsão, etc.), para posterior recuperação.

        Args:
            name: Identificador único do componente no registro.
            component: Instância/objeto a ser registrado.

        Returns:
            None.

        Raises:
            ValueError: Se já existir um componente registrado com o
                mesmo nome.
        """

        if name in self._components:
            raise ValueError(f"Component '{name}' is already registered.")
        self._components[name] = component

    def get(self, name: str) -> Any:
        """Recupera um componente registrado.

        Contexto:
            Fornece acesso à instância previamente registrada sob um
            determinado nome, permitindo desacoplamento entre consumidor
            e implementação concreta.

        Args:
            name: Nome do componente desejado.

        Returns:
            Instância do componente registrada sob o nome informado.

        Raises:
            KeyError: Se o componente não estiver registrado.
        """

        component = self._components.get(name)
        if component is None:
            raise KeyError(f"Component '{name}' not found.")
        return component

    def list_components(self) -> Dict[str, Any]:
        """Lista todos os componentes atualmente registrados.

        Contexto:
            Útil para inspeção, depuração ou exportação do estado do
            registro sem permitir modificação direta do dicionário
            interno.

        Returns:
            Cópia raso (`shallow copy`) do dicionário de componentes
            registrados, mapeando nomes para instâncias.
        """

        return self._components.copy()