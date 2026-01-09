from typing import Any, Dict


class Registry:
    """Registro simples de componentes (service locator) por nome."""

    def __init__(self) -> None:
        """Inicializa o registro vazio de componentes."""
        self._components: Dict[str, Any] = {}

    def register(self, name: str, component: Any) -> None:
        """Registra um componente por nome.

        Args:
            name: Identificador único do componente.
            component: Instância/objeto a ser registrado.

        Raises:
            ValueError: Se já existir um componente registrado com o mesmo nome.
        """
        if name in self._components:
            raise ValueError(f"Component '{name}' is already registered.")
        self._components[name] = component

    def get(self, name: str) -> Any:
        """Recupera um componente registrado.

        Args:
            name: Nome do componente.

        Returns:
            O componente registrado.

        Raises:
            KeyError: Se o componente não estiver registrado.
        """
        component = self._components.get(name)
        if component is None:
            raise KeyError(f"Component '{name}' not found.")
        return component

    def list_components(self) -> Dict[str, Any]:
        """Retorna uma cópia do dicionário de componentes registrados."""
        return self._components.copy()