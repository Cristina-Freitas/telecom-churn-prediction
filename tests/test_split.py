# ============================================================
# TESTES — DIVISÃO ESTRATIFICADA DOS DADOS
# ============================================================

import pandas as pd
import pytest

from src.splitting.split import dividir_dados


@pytest.fixture
def dados_exemplo():
    """Cria um dataset fictício com classes desbalanceadas."""
    return pd.DataFrame({
        "id_cliente": range(100),
        "meses_contrato": range(100),
        "valor_mensal": [50.0] * 100,
        "cancelou": [0] * 75 + [1] * 25
    })


def test_proporcao_treino_teste(dados_exemplo):
    """Verifica a divisão de 80% para treino e 20% para teste."""

    X_train, X_test, y_train, y_test = dividir_dados(dados_exemplo)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_estratificacao(dados_exemplo):
    """Verifica se a proporção das classes é preservada."""

    _, _, y_train, y_test = dividir_dados(dados_exemplo)

    assert y_train.mean() == pytest.approx(0.25)
    assert y_test.mean() == pytest.approx(0.25)


def test_separacao_variavel_alvo(dados_exemplo):
    """Verifica se a variável-alvo não está nas preditoras."""

    X_train, X_test, _, _ = dividir_dados(dados_exemplo)

    assert "cancelou" not in X_train.columns
    assert "cancelou" not in X_test.columns


def test_exclusao_identificador(dados_exemplo):
    """Verifica se o identificador do cliente foi removido."""

    X_train, X_test, _, _ = dividir_dados(dados_exemplo)

    assert "id_cliente" not in X_train.columns
    assert "id_cliente" not in X_test.columns


def test_ausencia_sobreposicao(dados_exemplo):
    """Verifica se nenhum registro aparece em treino e teste."""

    X_train, X_test, _, _ = dividir_dados(dados_exemplo)

    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_reprodutibilidade(dados_exemplo):
    """Verifica se a mesma semente produz a mesma divisão."""

    primeira = dividir_dados(dados_exemplo, random_state=42)
    segunda = dividir_dados(dados_exemplo, random_state=42)

    for resultado_1, resultado_2 in zip(primeira, segunda):
        pd.testing.assert_series_equal(
            resultado_1,
            resultado_2
        ) if isinstance(resultado_1, pd.Series) else (
            pd.testing.assert_frame_equal(resultado_1, resultado_2)
        )


def test_coluna_alvo_inexistente(dados_exemplo):
    """Verifica o tratamento de uma coluna-alvo inexistente."""

    with pytest.raises(ValueError, match="não encontrada"):
        dividir_dados(dados_exemplo, coluna_alvo="churn")