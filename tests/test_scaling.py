import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocessing.scaling import criar_escalonador


def test_criar_escalonador():
    escalonador = criar_escalonador()
    assert isinstance(escalonador, StandardScaler)


def test_media_zero_no_treinamento():
    treino = pd.DataFrame({
        "valor_mensal": [20.0, 40.0, 60.0, 80.0]
    })

    escalonador = criar_escalonador()
    resultado = escalonador.fit_transform(treino)

    assert np.allclose(resultado.mean(axis=0), 0.0)


def test_desvio_padrao_um_no_treinamento():
    treino = pd.DataFrame({
        "valor_mensal": [20.0, 40.0, 60.0, 80.0]
    })

    escalonador = criar_escalonador()
    resultado = escalonador.fit_transform(treino)

    assert np.allclose(resultado.std(axis=0), 1.0)


def test_transformacao_do_teste_usa_parametros_do_treino():
    treino = pd.DataFrame({
        "valor_mensal": [10.0, 20.0, 30.0]
    })
    teste = pd.DataFrame({
        "valor_mensal": [40.0]
    })

    escalonador = criar_escalonador()
    escalonador.fit_transform(treino)

    media_treino = escalonador.mean_.copy()
    resultado_teste = escalonador.transform(teste)

    esperado = (40.0 - media_treino[0]) / np.sqrt(
        escalonador.var_[0]
    )

    assert np.isclose(resultado_teste[0, 0], esperado)
    assert np.array_equal(escalonador.mean_, media_treino)


def test_escalonamento_nao_altera_dados_originais():
    treino = pd.DataFrame({
        "valor_mensal": [10.0, 20.0, 30.0]
    })
    original = treino.copy(deep=True)

    escalonador = criar_escalonador()
    escalonador.fit_transform(treino)

    pd.testing.assert_frame_equal(treino, original)