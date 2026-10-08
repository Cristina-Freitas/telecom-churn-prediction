# ============================================================
# TESTES DOS ARQUIVOS DE SAÍDA
# ============================================================

from pathlib import Path


RAIZ_PROJETO = Path(__file__).resolve().parents[1]


def test_boxplots_foram_salvos():
    caminho = (
        RAIZ_PROJETO
        / "outputs"
        / "figures"
        / "05_boxplots_variaveis_explicativas.png"
    )

    assert caminho.is_file(), "O arquivo do boxplot não foi encontrado."
    assert caminho.stat().st_size > 0, "O arquivo do boxplot está vazio."