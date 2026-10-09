# ============================================================
# TREINAMENTO E AVALIAÇÃO DA ÁRVORE DE DECISÃO
# ============================================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def avaliar_arvore(
    X_train,
    y_train,
    X_test,
    y_test,
    profundidades=(3, 5, None),
    random_state=42
):
    """
    Treina e avalia Árvores de Decisão com diferentes
    valores de max_depth.

    Retorna as acurácias de treinamento e teste
    para cada configuração.
    """
    resultados = []

    for profundidade in profundidades:
        modelo = DecisionTreeClassifier(
            max_depth=profundidade,
            random_state=random_state
        )

        modelo.fit(X_train, y_train)

        acuracia_treino = accuracy_score(
            y_train,
            modelo.predict(X_train)
        )

        acuracia_teste = accuracy_score(
            y_test,
            modelo.predict(X_test)
        )

        resultados.append({
            "max_depth": profundidade,
            "acuracia_treino": acuracia_treino,
            "acuracia_teste": acuracia_teste,
        })

    return resultados