# ============================================================
# TESTES DA CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.preprocessing.encoding import criar_codificador


def test_criar_codificador():
    codificador = criar_codificador()

    assert isinstance(codificador, OneHotEncoder)
    assert codificador.handle_unknown == "ignore"
    assert codificador.sparse_output is False


def test_quantidade_colunas_geradas():
    treino = pd.DataFrame({
        "genero": ["Female", "Male", "Female"],
        "contrato": ["Mensal", "Anual", "Mensal"],
    })

    codificador = criar_codificador()
    resultado = codificador.fit_transform(treino)

    assert resultado.shape == (3, 4)


def test_categorias_desconhecidas_no_teste():
    treino = pd.DataFrame({
        "contrato": ["Mensal", "Anual"]
    })

    teste = pd.DataFrame({
        "contrato": ["Trimestral"]
    })

    codificador = criar_codificador()
    codificador.fit(treino)

    resultado = codificador.transform(teste)

    assert resultado.shape == (1, 2)
    assert np.array_equal(resultado, [[0.0, 0.0]])


def test_codificador_nao_altera_dados_originais():
    treino = pd.DataFrame({
        "genero": ["Female", "Male"]
    })

    original = treino.copy(deep=True)

    codificador = criar_codificador()
    codificador.fit_transform(treino)

    pd.testing.assert_frame_equal(treino, original)


def test_mesmas_colunas_no_treino_e_teste():
    treino = pd.DataFrame({
        "genero": ["Female", "Male"],
        "contrato": ["Mensal", "Anual"],
    })

    teste = pd.DataFrame({
        "genero": ["Male"],
        "contrato": ["Mensal"],
    })

    codificador = criar_codificador()

    resultado_treino = codificador.fit_transform(treino)
    resultado_teste = codificador.transform(teste)

    assert resultado_treino.shape[1] == resultado_teste.shape[1]