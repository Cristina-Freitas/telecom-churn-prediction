# ============================================================
# TESTES AUTOMATIZADOS DA IMPUTAÇÃO
# ============================================================

import numpy as np
import pandas as pd

from src.preprocessing.imputation import criar_imputador


def test_imputacao_pela_mediana():
    dados = pd.DataFrame({
        "valor_mensal": [10.0, 20.0, 100.0, np.nan]
    })

    imputador = criar_imputador()
    resultado = imputador.fit_transform(dados)

    assert resultado[3, 0] == 20.0


def test_imputacao_remove_valores_ausentes():
    dados = pd.DataFrame({
        "meses_contrato": [12.0, np.nan, 36.0],
        "valor_total": [100.0, 200.0, np.nan]
    })

    imputador = criar_imputador()
    resultado = imputador.fit_transform(dados)

    assert not np.isnan(resultado).any()


def test_imputador_utiliza_mediana_do_treino():
    treino = pd.DataFrame({
        "valor_mensal": [10.0, 20.0, 30.0]
    })

    teste = pd.DataFrame({
        "valor_mensal": [np.nan, 1000.0]
    })

    imputador = criar_imputador()
    imputador.fit(treino)

    resultado = imputador.transform(teste)

    assert resultado[0, 0] == 20.0
    assert resultado[1, 0] == 1000.0