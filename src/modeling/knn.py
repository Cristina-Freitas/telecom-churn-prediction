# ============================================================
# TREINAMENTO E AVALIAÇÃO DO KNN
# ============================================================

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def avaliar_knn(X_train, y_train, X_test, y_test, valores_k=(3, 5, 7)):
    """
    Treina e avalia o KNN com diferentes valores de n_neighbors.

    Retorna as acurácias de treinamento e teste
    para cada configuração.
    """
    resultados = []

    for k in valores_k:
        modelo = KNeighborsClassifier(n_neighbors=k)

        modelo.fit(X_train, y_train)

        acuracia_treino = accuracy_score(
            y_train, modelo.predict(X_train)
        )

        acuracia_teste = accuracy_score(
            y_test, modelo.predict(X_test)
        )

        resultados.append({
            "n_neighbors": k,
            "acuracia_treino": acuracia_treino,
            "acuracia_teste": acuracia_teste,
        })

    return resultados