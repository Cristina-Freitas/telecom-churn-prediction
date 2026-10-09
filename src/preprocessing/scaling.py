from sklearn.preprocessing import StandardScaler


def criar_escalonador():
    """
    Cria um escalonador para padronizar variáveis contínuas.

    O ajuste (fit) deve ocorrer exclusivamente no treinamento.
    O conjunto de teste deve utilizar apenas transform().
    """
    return StandardScaler()