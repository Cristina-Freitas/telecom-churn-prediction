# ============================================================
# FASE 4 — BALANCEAMENTO POR RANDOM UNDER SAMPLING
# ============================================================

import pandas as pd


def balancear_treinamento(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Balanceia as classes exclusivamente no conjunto de treinamento,
    reduzindo aleatoriamente a classe majoritária.

    Preserva os dados originais e mantém X e y alinhados.
    """

    if not X_train.index.equals(y_train.index):
        raise ValueError("Os índices de X_train e y_train devem coincidir.")

    contagem = y_train.value_counts()

    if len(contagem) != 2:
        raise ValueError("O balanceamento requer exatamente duas classes.")

    quantidade_minima = contagem.min()

    # Seleciona a mesma quantidade de registros de cada classe
    indices_selecionados = []

    for classe in contagem.index:
        indices_classe = y_train[y_train == classe].sample(
            n=quantidade_minima,
            random_state=random_state
        ).index

        indices_selecionados.extend(indices_classe)

    # Embaralha os índices selecionados
    indices_selecionados = (
        pd.Series(indices_selecionados)
        .sample(frac=1, random_state=random_state)
        .tolist()
    )

    X_balanceado = X_train.loc[indices_selecionados].copy()
    y_balanceado = y_train.loc[indices_selecionados].copy()

    return X_balanceado, y_balanceado