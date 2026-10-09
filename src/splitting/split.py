# ============================================================
# FASE 4 — DIVISÃO ESTRATIFICADA DOS DADOS
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split


def dividir_dados(
    df: pd.DataFrame,
    coluna_alvo: str = "cancelou",
    proporcao_teste: float = 0.20,
    random_state: int = 42
):
    """
    Separa as variáveis preditoras (X) da variável-alvo (y)
    e divide os dados em treinamento e teste.

    Utiliza estratificação para preservar aproximadamente
    a proporção das classes da variável-alvo.
    """

    # Verifica se a variável-alvo existe
    if coluna_alvo not in df.columns:
        raise ValueError(
            f"Coluna-alvo '{coluna_alvo}' não encontrada."
        )

    # Separa preditoras e variável-alvo
    X = df.drop(columns=[coluna_alvo, "id_cliente"], errors="ignore")
    y = df[coluna_alvo].copy()

    # Divide os dados mantendo a proporção das classes
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=proporcao_teste,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test