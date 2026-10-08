from pathlib import Path

import pandas as pd


def carregar_dataset(caminho: str | Path) -> pd.DataFrame:
    """
    Carrega um dataset CSV e retorna um DataFrame.

    Args:
        caminho: Caminho local para o arquivo CSV.

    Returns:
        DataFrame com os dados carregados.
    """
    arquivo = Path(caminho).expanduser()

    if not arquivo.is_file():
        raise FileNotFoundError(
            f"Dataset não encontrado: {arquivo}"
        )

    return pd.read_csv(arquivo)