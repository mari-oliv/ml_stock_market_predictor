import json
import os

import nbformat as nbf
from nbclient import NotebookClient


def run_notebook_last_cell(
    notebook_path: str,
    params: dict,
    output_path: str,
    timeout: int = 600,
) -> float:
    """Executa um notebook injetando parâmetros e retorna o valor produzido no output JSON.

    O notebook é carregado, recebe uma célula inicial com as variáveis de `params`,
    e é executado via `nbclient` usando o kernel definido em `NB_KERNEL` (ou um padrão).
    Após a execução, este método lê `output_path` (JSON) e retorna um float.

    Args:
        notebook_path: Caminho do notebook `.ipynb` a ser executado.
        params: Dicionário de parâmetros a serem injetados como variáveis Python.
        output_path: Caminho do arquivo JSON esperado como saída (criado pelo notebook).
        timeout: Timeout (em segundos) para execução do notebook.

    Returns:
        Valor numérico lido do JSON. Aceita JSON sendo número direto ou dict com chave "value".

    Raises:
        FileNotFoundError: Se `notebook_path` não existir.
        RuntimeError: Se o notebook não gerar o arquivo esperado em `output_path`.
        ValueError: Se o conteúdo do JSON não puder ser convertido para float.
        json.JSONDecodeError: Se o arquivo de saída não for JSON válido.
    """
    if not os.path.exists(notebook_path):
        raise FileNotFoundError(f"Notebook não encontrado: {notebook_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    nb = nbf.read(notebook_path, as_version=4)

    param_lines = [f"{k} = {repr(v)}" for k, v in params.items()]
    nb.cells.insert(0, nbf.v4.new_code_cell("\n".join(param_lines)))

    kernel_name = os.getenv("NB_KERNEL", "ml-unified-service")
    nb.metadata.setdefault("kernelspec", {})
    nb.metadata.kernelspec.update(
        {
            "name": kernel_name,
            "display_name": f"Python ({kernel_name})",
            "language": "python",
        }
    )

    client = NotebookClient(nb, timeout=timeout, kernel_name=kernel_name)
    client.execute()

    if not os.path.exists(output_path):
        raise RuntimeError(f"A última célula não gerou o arquivo esperado: {output_path}")

    with open(output_path, "r") as f:
        data = json.load(f)

    return float(data) if isinstance(data, (int, float)) else float(data["value"])