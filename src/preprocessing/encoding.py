# ============================================================
# CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS
# ============================================================

from sklearn.preprocessing import OneHotEncoder


def criar_codificador():
    """
    Cria um codificador One-Hot para variáveis categóricas.

    O ajuste (fit) deve ocorrer exclusivamente no treinamento.

    Categorias desconhecidas no teste são ignoradas,
    evitando erros durante a transformação.
    """
    return OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )