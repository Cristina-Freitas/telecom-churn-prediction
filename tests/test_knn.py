# ============================================================
# TESTES DO TREINAMENTO E AVALIAÇÃO DO KNN
# ============================================================

import numpy as np
import pytest

from src.modeling.knn import avaliar_knn


@pytest.fixture
def dados_classificacao():
    """Cria dados sintéticos para testar o KNN."""
    X_train = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.2],
        [1.0, 1.0],
        [1.1, 1.1],
        [1.2, 1.2],
    ])

    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_test = np.array([
        [0.15, 0.15],
        [1.15, 1.15],
    ])

    y_test = np.array([0, 1])

    return X_train, y_train, X_test, y_test


def test_tres_configuracoes_knn(dados_classificacao):
    X_train, y_train, X_test, y_test = dados_classificacao

    resultados = avaliar_knn(
        X_train, y_train, X_test, y_test,
        valores_k=(3, 5, 6)
    )

    assert len(resultados) == 3


def test_valores_k_retornados(dados_classificacao):
    X_train, y_train, X_test, y_test = dados_classificacao

    resultados = avaliar_knn(
        X_train, y_train, X_test, y_test,
        valores_k=(3, 5)
    )

    assert [r["n_neighbors"] for r in resultados] == [3, 5]


def test_acuracias_entre_zero_e_um(dados_classificacao):
    X_train, y_train, X_test, y_test = dados_classificacao

    resultados = avaliar_knn(
        X_train, y_train, X_test, y_test,
        valores_k=(3, 5)
    )

    for resultado in resultados:
        assert 0 <= resultado["acuracia_treino"] <= 1
        assert 0 <= resultado["acuracia_teste"] <= 1


def test_acuracia_perfeita_em_dados_separaveis(dados_classificacao):
    X_train, y_train, X_test, y_test = dados_classificacao

    resultados = avaliar_knn(
        X_train, y_train, X_test, y_test,
        valores_k=(3,)
    )

    assert resultados[0]["acuracia_treino"] == 1.0
    assert resultados[0]["acuracia_teste"] == 1.0


def test_resultados_reprodutiveis(dados_classificacao):
    X_train, y_train, X_test, y_test = dados_classificacao

    primeira_execucao = avaliar_knn(
        X_train, y_train, X_test, y_test,
        valores_k=(3, 5)
    )

    segunda_execucao = avaliar_knn(
        X_train, y_train, X_test, y_test,
        valores_k=(3, 5)
    )

    assert primeira_execucao == segunda_execucao