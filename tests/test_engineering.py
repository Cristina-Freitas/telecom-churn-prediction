# ============================================================
# TESTES — FEATURE ENGINEERING
# ============================================================

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import criar_gasto_medio_mensal


def test_calculo_gasto_medio_mensal():
    """Verifica se o novo atributo é calculado corretamente."""

    dados = pd.DataFrame({
        "valor_total": [1200.0, 3600.0],
        "meses_contrato": [12.0, 24.0]
    })

    resultado = criar_gasto_medio_mensal(dados)

    assert resultado["gasto_medio_mensal"].tolist() == [100.0, 150.0]


def test_divisao_por_zero():
    """Verifica se contratos com zero meses não geram infinito."""

    dados = pd.DataFrame({
        "valor_total": [500.0],
        "meses_contrato": [0.0]
    })

    resultado = criar_gasto_medio_mensal(dados)

    assert pd.isna(resultado.loc[0, "gasto_medio_mensal"])


def test_valores_ausentes():
    """Verifica o comportamento diante de valores ausentes."""

    dados = pd.DataFrame({
        "valor_total": [np.nan, 1200.0],
        "meses_contrato": [12.0, np.nan]
    })

    resultado = criar_gasto_medio_mensal(dados)

    assert resultado["gasto_medio_mensal"].isna().all()


def test_preservacao_dataframe_original():
    """Verifica se a função não modifica os dados de entrada."""

    dados = pd.DataFrame({
        "valor_total": [1200.0],
        "meses_contrato": [12.0]
    })

    original = dados.copy(deep=True)

    criar_gasto_medio_mensal(dados)

    pd.testing.assert_frame_equal(dados, original)
    assert "gasto_medio_mensal" not in dados.columns


@pytest.mark.parametrize(
    "valor_total, meses_contrato, esperado",
    [
        (600.0, 6.0, 100.0),
        (1500.0, 10.0, 150.0),
        (0.0, 12.0, 0.0),
    ]
)
def test_diferentes_calculos(valor_total, meses_contrato, esperado):
    """Valida diferentes combinações de valores."""

    dados = pd.DataFrame({
        "valor_total": [valor_total],
        "meses_contrato": [meses_contrato]
    })

    resultado = criar_gasto_medio_mensal(dados)

    assert resultado.loc[0, "gasto_medio_mensal"] == pytest.approx(esperado)