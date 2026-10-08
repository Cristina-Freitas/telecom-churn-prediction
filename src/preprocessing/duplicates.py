import pandas as pd


def remover_duplicatas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas completamente duplicadas de um DataFrame,
    preservando o objeto original.
    """
    return df.drop_duplicates().copy()