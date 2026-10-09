# ============================================================
# TESTES — BALANCEAMENTO DOS DADOS
# ============================================================

import pandas as pd
import pytest

from src.splitting.balancing import balancear_treinamento


@pytest.fixture
def dados_exemplo():
    """Cria dados fictícios com classes desbalanceadas."""

    X = pd.DataFrame({
        "meses_contrato": range(100),
        "tipo_contrato": ["Mensal"] * 100
    })

    y = pd.Series([0] * 75 + [1] * 25, name="cancelou")

    return X, y


def test_classes_balanceadas(dados_exemplo):
    """Verifica se as duas classes ficam com a mesma quantidade."""

    X, y = dados_exemplo

    X_bal, y_bal = balancear_treinamento(X, y)

    assert y_bal.value_counts().to_dict() == {0: 25, 1: 25}
    assert len(X_bal) == 50


def test_alinhamento_indices(dados_exemplo):
    """Verifica se X e y permanecem alinhados."""

    X, y = dados_exemplo

    X_bal, y_bal = balancear_treinamento(X, y)

    assert X_bal.index.equals(y_bal.index)


def test_preservacao_dados_originais(dados_exemplo):
    """Verifica se os dados de entrada não são modificados."""

    X, y = dados_exemplo

    X_original = X.copy(deep=True)
    y_original = y.copy(deep=True)

    balancear_treinamento(X, y)

    pd.testing.assert_frame_equal(X, X_original)
    pd.testing.assert_series_equal(y, y_original)


def test_reprodutibilidade(dados_exemplo):
    """Verifica se a mesma semente gera a mesma amostra."""

    X, y = dados_exemplo

    X1, y1 = balancear_treinamento(X, y, random_state=42)
    X2, y2 = balancear_treinamento(X, y, random_state=42)

    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)


def test_indices_desalinhados(dados_exemplo):
    """Verifica se índices incompatíveis geram erro."""

    X, y = dados_exemplo
    y.index = range(100, 200)

    with pytest.raises(ValueError, match="índices"):
        balancear_treinamento(X, y)


def test_classe_unica():
    """Verifica se uma única classe gera erro."""

    X = pd.DataFrame({"valor": [10, 20, 30]})
    y = pd.Series([0, 0, 0])

    with pytest.raises(ValueError, match="duas classes"):
        balancear_treinamento(X, y)


def test_dados_categoricos_e_ausentes():
    """Verifica a compatibilidade com categorias e valores ausentes."""

    X = pd.DataFrame({
        "valor_mensal": [50.0, None, 70.0, 80.0],
        "tipo_contrato": ["Mensal", "Anual", "Mensal", "Anual"]
    })