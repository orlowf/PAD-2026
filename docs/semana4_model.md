# Semana 4 — Model the Data

**Projeto FMF escolhido:** Predição de Churn — *Predicting Customer Churn Project: A Machine Learning Approach to Binary Classification*
**Etapa do AGEMC:** M — Model the Data
**Dataset:** `data/CustomerChurnData.csv`
**Notebook:** `notebooks/projeto_churn_agemc.ipynb` (seção *M — Model the Data*)

---

## Originalidade do projeto

O Projeto FMF de referência vai direto para um classificador. A **contribuição original**
do nosso grupo está nesta etapa: uma **modelagem estatisticamente fundamentada**. Antes de
treinar qualquer modelo, **testamos formalmente as hipóteses H1–H4** levantadas na
exploração. Os testes (significância + tamanho de efeito) **guiam a seleção de variáveis** e
fornecem a **interpretação** do modelo final. Assim, a estatística inferencial e o modelo
preditivo se reforçam, em vez de andarem separados.

## Fluxo da etapa M

1. Separação treino/teste do *df* bruto **antes de tudo** — anti-vazamento (M.0)
2. Testes de hipótese **somente no treino** (M.1)
3. Seleção de variáveis **informada pelos testes** e validada por CV; codificação no treino (M.2)
4. Modelo *baseline* (M.3)
5. Validação cruzada com tratamento de desbalanceamento (M.4)
6. Avaliação no teste *held-out* com limiar escolhido por F1 (M.5)
7. Interpretação (M.6)
8. Conclusões (M.7)

## M.0 — Separação treino/teste (anti-vazamento)

Para evitar **vazamento de informação**, o *df* **bruto** é dividido **antes de qualquer
análise**: 75% treino / 25% teste, **estratificado** (mantém ~14,5% de churn nos dois lados —
treino com 2.499 linhas, teste com 834). A partir daí, **todos os testes (M.1) e a seleção
(M.2) usam apenas o treino (`df_tr`)**; o teste (`df_te`) só é tocado na avaliação final
(M.5). Nenhuma transformação muta o `df` global.

## M.1 — Testes de hipótese

Para cada hipótese foi escolhido o teste adequado ao **tipo de dado**, com α = 0,05.
**Todos rodam somente no treino.**

| Hipótese | Variável | Teste | Estatística | p-valor | Tamanho de efeito | Decisão |
|---|---|---|---:|---:|---:|---|
| **H1** | customer service calls (≥4) | qui-quadrado | χ² = 266,40 | ≈ 6,9 × 10⁻⁶⁰ | Cramér's V = 0,327 | **Rejeita H0** |
| **H2** | total day minutes | Mann-Whitney U | U = 488.538 | ≈ 5,5 × 10⁻¹⁶ | r_pb = 0,193 | **Rejeita H0** |
| **H3** | international plan | qui-quadrado | χ² = 162,47 | ≈ 3,3 × 10⁻³⁷ | Cramér's V = 0,255 | **Rejeita H0** |
| **H4** | state | qui-quadrado | χ² = 74,04 | ≈ 1,5 × 10⁻² | Cramér's V = 0,172 | Rejeita H0 (sinal fraco) |
| extra | voice mail plan | qui-quadrado | χ² = 31,58 | ≈ 1,9 × 10⁻⁸ | Cramér's V = 0,112 | **Rejeita H0** (protetor) |

Notas metodológicas:

- usamos **Mann-Whitney** em H2 por ser não paramétrico (não assumimos normalidade dos minutos);
- a **correlação ponto-bisserial** mede o tamanho de efeito entre a variável binária `churn`
  e variáveis numéricas;
- **Cramér's V** mede a força da associação em tabelas de contingência (0 = nenhuma);
- **H4** é apenas **confirmatório** aqui (a análise detalhada do estado está na etapa E).

**Leitura:** H1, H2 e H3 são fortemente significativas. O maior efeito é o das **chamadas
ao atendimento ≥ 4** (V ≈ 0,33), seguido do **plano internacional** (V ≈ 0,26). H4 (estado)
sai significativa, mas com **efeito pequeno e células esparsas** (50 graus de liberdade),
confirmando que é um sinal fraco. O voice mail aparece como fator **protetor**.

### Ranking de tamanho de efeito (ponto-bisserial) com churn — no treino

| Variável | r ponto-bisserial |
|---|---:|
| customer service calls | 0,224 |
| total day minutes | 0,193 |
| number vmail messages | −0,100 |
| total eve minutes | 0,089 |
| total intl calls | −0,072 |
| total intl minutes | 0,069 |
| total day calls / total night minutes / account length / total eve calls / total night calls / area code | ≈ 0 (sem sinal, p > 0,05) |

## M.2 — Seleção de variáveis informada pelos testes

A seleção **deriva diretamente dos testes de M.1** (calculados no treino):

- removida `phone number` (identificador, sem valor preditivo, minimização de dados);
- removidas as colunas de cobrança (`total * charge`), redundantes com os minutos (r ≈ 1,0);
- **mantidas as numéricas com sinal** (|r| ≥ 0,05 e p < 0,05): `customer service calls`,
  `total day minutes`, `number vmail messages`, `total eve minutes`, `total intl calls`,
  `total intl minutes` (6 de 11 candidatas); **descartadas** as sem sinal (`total day calls`,
  `total night minutes`, `account length`, `total eve calls`, `total night calls`);
- `international plan`, `voice mail plan` e **`area code`** (nominal) com *one-hot*
  (`OneHotEncoder(handle_unknown="ignore")` dentro do `ColumnTransformer`, ajustado no treino);
- **`state`:** comparado **manter vs. dropar** por validação cruzada (RF de referência, k=5):

  | Conjunto | F1 | PR-AUC |
  |---|---:|---:|
  | completo (sem state) | 0,780 ± 0,029 | 0,853 ± 0,043 |
  | enxuto (sem state) | 0,774 ± 0,046 | 0,836 ± 0,051 |
  | enxuto + state | 0,755 ± 0,063 | 0,838 ± 0,050 |

  A seleção **enxuta** não é pior que a completa além da variabilidade da CV (mantida, por
  parcimônia); os *dummies* de `state` **não melhoram o F1** (na verdade pioram) com ganho
  desprezível de PR-AUC, então **`state` é DROPADO** — coerente com o sinal fraco de H4.

Dimensão final de `X` (encoding ajustado no treino): **13 colunas** (6 numéricas + 7 *dummies*
de `international plan`, `voice mail plan` e `area code`), aplicadas a 2.499 linhas de treino.

## M.3 — Modelo baseline

- **Baseline** (classe majoritária): acurácia 0,855, mas **recall de churn = 0** — não
  identifica nenhum cliente em risco. É o piso que a modelagem precisa superar. (O split já
  foi feito em M.0.)

## M.4 — Validação cruzada

Três modelos, com desbalanceamento tratado de duas formas, comparados por
**`StratifiedKFold` (k=5)**. *Encoding*, *scaling* (logística) e *SMOTE* ficam **dentro do
pipeline avaliado por fold** (`imblearn.Pipeline`) — nunca antes da CV, evitando vazamento.

1. **Regressão Logística** com `class_weight="balanced"` (interpretável);
2. **Random Forest** com `class_weight="balanced"`;
3. **Random Forest + SMOTE** (reamostragem sintética só no treino de cada fold).

Métricas de validação cruzada (**média ± desvio**):

| Modelo | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| Regressão Logística (balanced) | **0,743 ± 0,056** | 0,483 ± 0,018 | 0,822 ± 0,019 | 0,451 ± 0,016 |
| Random Forest (balanced) | 0,683 ± 0,077 | 0,774 ± 0,046 | **0,909 ± 0,034** | **0,836 ± 0,051** |
| Random Forest + SMOTE | 0,738 ± 0,056 | **0,786 ± 0,030** | 0,907 ± 0,031 | 0,815 ± 0,057 |

## M.5 — Avaliação no teste e escolha do limiar

Base desbalanceada → a acurácia não basta. O limiar **não** é fixado em 0,5: para cada modelo
escolhemos o que **maximiza o F1**, estimado por validação cruzada no treino
(`cross_val_predict`, evitando otimismo). Limiares escolhidos: Logística 0,60; RF balanceada
0,39; RF + SMOTE 0,49.

Métricas no conjunto de teste *held-out* (cada modelo no seu limiar):

| Modelo | Limiar | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (classe majoritária) | — | 0,00 | — | — | 0,50 | 0,14 |
| Regressão Logística (balanced) | 0,60 | 0,59 | 0,37 | 0,46 | 0,81 | 0,42 |
| Random Forest (balanced) | 0,39 | **0,76** | 0,75 | **0,76** | **0,90** | **0,82** |
| Random Forest + SMOTE | 0,49 | 0,69 | **0,76** | 0,72 | 0,89 | 0,79 |

**Leitura:**

- a **Logística** tem recall alto na validação cruzada (~0,74): boa para *triagem*, mas com muitos alarmes falsos;
- a **Random Forest balanceada** lidera no teste em ranqueamento (ROC-AUC 0,90, PR-AUC 0,82) **e** em F1 (0,76) no limiar escolhido;
- a **Random Forest + SMOTE** fica logo atrás, com bom equilíbrio.

A escolha do limiar materializa o custo de um falso negativo (perder cliente) vs. um falso
positivo (acionar quem não cancelaria): limiares menores aumentam recall e falsos positivos.

## M.6 — Interpretação

### Odds ratios (regressão logística, variáveis padronizadas, no treino)

Pseudo R² de McFadden = **0,20**. Variáveis nomeadas em inglês, conforme o dataset.

| Fator | Odds ratio | p-valor | Leitura |
|---|---:|---:|---|
| customer service calls | 2,02 | ≈ 3,3 × 10⁻³² | +102% na chance de churn por desvio-padrão (**H1**) |
| total day minutes | 1,89 | ≈ 8,1 × 10⁻²² | +89% na chance (**H2**) |
| international plan | 1,81 | ≈ 6,9 × 10⁻³⁴ | +81% na chance (**H3**) |
| total eve minutes | 1,40 | ≈ 2,5 × 10⁻⁷ | +40% na chance |
| voice mail plan | 0,65 | ≈ 1,2 × 10⁻⁸ | **protetor** (reduz a chance) |

### Importância de variáveis (Random Forest)

Lideram `total day minutes`, `customer service calls` e `international plan` — **os mesmos
fatores apontados pelos testes de hipótese**. Essa coerência entre estatística inferencial e
modelo preditivo é a materialização da originalidade da etapa M.

## M.7 — Conclusões da modelagem

- **H1, H2 e H3 confirmadas** por testes formais no treino (p ≪ 0,05); **H4 (estado)** é sinal
  fraco — daí a decisão de **dropar** os *dummies* de `state`, tomada por validação cruzada.
- A **seleção de variáveis é informada pelos testes** (M.1 → M.2) e **validada por CV**:
  descartar as variáveis sem sinal e as cobranças redundantes não prejudica as métricas e
  reduz `X` de dezenas de colunas para apenas **13**.
- O **baseline** é inútil para o objetivo (recall 0): tratar o desbalanceamento é essencial.
- Na CV e no teste *held-out*, a **Random Forest balanceada** entrega o melhor ranqueamento
  (ROC-AUC/PR-AUC) **e** o melhor F1 no limiar escolhido; a **RF + SMOTE** fica logo atrás; a
  **Logística** é a mais interpretável e a de maior recall. O **limiar** foi escolhido por F1
  (via CV), não fixado em 0,5.
- Os fatores de risco (uso diurno, contato com atendimento, plano internacional) e o fator
  protetor (voice mail) são consistentes entre testes, odds ratios e importância.
- **Uso prático:** numa campanha de retenção, priorizar clientes com maior probabilidade
  prevista, ajustando o limiar conforme o orçamento da ação.

Esses resultados alimentam diretamente a etapa **C — Communicate**.
