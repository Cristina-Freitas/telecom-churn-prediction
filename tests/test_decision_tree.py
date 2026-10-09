# ============================================================
# TESTES DO TREINAMENTO E AVALIAÇÃO DA ÁRVORE DE DECISÃO
# ============================================================

import numpy as np
import pytest

from src.modeling.decision_tree import avaliar_arvore


@pytest.fixture
def dados_classificacao():
    """Cria dados sintéticos para testar a Árvore de Decisão."""
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


def test_tres_configuracoes_arvore(dados_classificacao):
    X_train, y_train, X_test, y_test = dados_classificacao

    resultados = avaliar_arvore(
        X_train, y_train, X_test, y_test,
        profundidades=(3, 5, None)
    )

    assert len(resultados) == 3


def test_profundidades_retornadas(dados_classificacao):
    X_train, y_train, X_test, y_test = dados_classificacao

    resultados = avaliar_arvore(
        X_train, y_train, X_test, y_test,
        profundidades=(3, 5, None)
    )

    assert [r["max_depth"] for r in resultados] == [3, 5, None]


def test_acuracias_entre_zero_e_um(dados_classificacao):
    X_train, y_train, X_test, y_test = dados_classificacao

    resultados = avaliar_arvore(
        X_train, y_train, X_test, y_test,
        profundidades=(3, 5, None)
    )

    for resultado in resultados:
        assert 0 <= resultado["acuracia_treino"] <= 1
        assert 0 <= resultado["acuracia_teste"] <= 1


def test_acuracia_perfeita_em_dados_separaveis(dados_classificacao):
    X_train, y_train, X_test, y_test = dados_classificacao

    resultados = avaliar_arvore(
        X_train, y_train, X_test, y_test,
        profundidades=(3,)
    )

    assert resultados[0]["acuracia_treino"] == 1.0
    assert resultados[0]["acuracia_teste"] == 1.0


def test_resultados_reprodutiveis(dados_classificacao):
    X_train, y_train, X_test, y_test = dados_classificacao

    primeira_execucao = avaliar_arvore(
        X_train, y_train, X_test, y_test,
        profundidades=(3, 5, None),
        random_state=42
    )

    segunda_execucao = avaliar_arvore(
        X_train, y_train, X_test, y_test,
        profundidades=(3, 5, None),
        random_state=42
    )

    assert primeira_execucao == segunda_execucao