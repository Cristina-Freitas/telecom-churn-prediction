# Telecom Churn Prediction

## 1. Objetivo e organização

Prever o cancelamento de clientes de telecomunicações (*churn*) por meio de modelos supervisionados. O projeto utiliza notebooks para experimentos, funções reutilizáveis em `src/`, testes em `tests/`, dados em `data/` e gráficos em `outputs/figures/`. A execução foi realizada localmente no VS Code com ambiente virtual Python.

## 2. Decisões técnicas por etapa

| Fase | Decisão e justificativa |
|:---|:---|
| **1 — Análise exploratória** | Inspeção da distribuição de churn, variáveis numéricas, correlações e qualidade dos dados; gráficos exportados em PNG. |
| **2 — Preparação** | Remoção de duplicatas, investigação de ausências e tratamento de atributos; prevenção de vazamento de dados entre treino e teste. |
| **3 — Engenharia de atributos** | Construção de `gasto_medio_mensal` e preparação das variáveis explicativas. |
| **4 — Divisão e balanceamento** | Divisão estratificada 80/20; *undersampling* aplicado somente ao treino (1.495 registros por classe). |
| **5 — Escalonamento** | Imputação pela mediana ajustada no treino; `StandardScaler` nas variáveis contínuas para KNN. Árvore sem padronização. |
| **6 — Modelagem e ajuste** | One-Hot Encoding com categorias desconhecidas ignoradas; KNN com K=3, 5 e 7; Árvore com profundidades 3, 5 e sem limite; comparação treino/teste para identificar *overfitting*. |
| **7 — Avaliação final** | Comparação das melhores configurações no teste e recomendação do modelo de maior acurácia. |

## 3. Resultados e decisão

**Base:** 7.043 registros após remoção de duplicatas. **Treino:** 5.634 registros. **Teste:** 1.409 registros. O balanceamento foi aplicado somente ao treinamento.

| Modelo | Configuração | Acurácia no teste |
|:---|:---:|---:|
| KNN | `K=5` | 70,76% |
| **Árvore de Decisão** | **`max_depth=5`** | **74,80%** |

**Veredito:** recomenda-se a **Árvore de Decisão (`max_depth=5`)**, com vantagem de **4,04 pontos percentuais** sobre o KNN. A Árvore sem limite de profundidade apresentou forte *overfitting*: 99,77% no treino e 67,99% no teste.

## 4. Validação e execução

Foram aprovados **48 testes automatizados com `pytest` até a Fase 6**.

Para reproduzir os experimentos, instale as dependências de `requirements.txt`, abra `notebooks/01_analise_exploratoria.ipynb` e execute as células em sequência.

```bash
python -m pytest tests/ -v
```

## 5. Limitações e melhorias

O conjunto de teste foi consultado na seleção dos hiperparâmetros da Fase 6; assim, a acurácia final **não representa uma validação totalmente independente**.

Melhorias futuras: validação cruzada com teste reservado, avaliação de precisão, *recall* e F1-score, além de uma interface de inferência. **Não há API ou interface de produção implementada.**
