# ============================================================
# IMPUTAÇÃO DE VALORES AUSENTES
# ============================================================

from sklearn.impute import SimpleImputer

COLUNAS_IMPUTACAO = [
    "meses_contrato",
    "valor_mensal",
    "valor_total",
]


def criar_imputador():
    """
    Cria um imputador baseado na mediana.

    O ajuste (fit) deve ser realizado exclusivamente
    com os dados de treinamento.
    """
    return SimpleImputer(strategy="median")