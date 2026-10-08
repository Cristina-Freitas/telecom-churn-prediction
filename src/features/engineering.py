# ============================================================
# FEATURE ENGINEERING — GASTO MÉDIO MENSAL
# ============================================================

import numpy as np
import pandas as pd


def criar_gasto_medio_mensal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a variável gasto_medio_mensal a partir da divisão
    do valor total pelos meses de contrato.

    Preserva o DataFrame original e evita divisões por zero.
    Valores ausentes permanecem como NaN para tratamento
    posterior.
    """
    resultado = df.copy()

    # Evita divisão por zero
    meses = resultado["meses_contrato"].replace(0, np.nan)

    # Calcula o novo atributo
    resultado["gasto_medio_mensal"] = (
        resultado["valor_total"] / meses
    )

    return resultado