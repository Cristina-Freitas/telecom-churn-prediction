import pandas as pd

from src.preprocessing.duplicates import remover_duplicatas


def test_remove_linhas_duplicadas():
    df = pd.DataFrame({
        "id_cliente": ["001", "002", "002"],
        "cancelou": [0, 1, 1]
    })

    resultado = remover_duplicatas(df)

    assert len(resultado) == 2
    assert resultado["id_cliente"].tolist() == ["001", "002"]


def test_preserva_dataframe_original():
    df = pd.DataFrame({
        "id_cliente": ["001", "001"],
        "cancelou": [0, 0]
    })

    remover_duplicatas(df)

    assert len(df) == 2


def test_preserva_registros_distintos():
    df = pd.DataFrame({
        "id_cliente": ["001", "001"],
        "cancelou": [0, 1]
    })

    resultado = remover_duplicatas(df)

    assert len(resultado) == 2