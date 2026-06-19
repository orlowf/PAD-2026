# Semana 3 — Explore the Data

**Projeto FMF escolhido:** Predição de Churn — *Predicting Customer Churn Project: A Machine Learning Approach to Binary Classification*
**Etapa do AGEMC:** E — Explore the Data
**Dataset:** `data/CustomerChurnData.csv`

---

## Objetivo da exploração

Nesta etapa, o objetivo é entender melhor os dados antes da modelagem. Como o projeto original seguirá o mesmo **Ask**, **Get** e **Explore** do Projeto FMF, esta exploração serve para identificar padrões iniciais, problemas de qualidade, variáveis relevantes e cuidados que deverão orientar a etapa **M — Model the Data**, onde será introduzida a originalidade do nosso trabalho.

As perguntas principais da exploração são:

1. Os dados possuem valores ausentes, duplicados ou inconsistências evidentes?
2. A variável-alvo `churn` está balanceada?
3. Quais variáveis parecem mais associadas ao cancelamento?
4. Existem colunas redundantes ou pouco úteis para a modelagem?

## Estrutura geral dos dados

| Item | Resultado |
|---|---:|
| Linhas | 3.333 |
| Colunas | 21 |
| Valores ausentes | 0 |
| Linhas duplicadas | 0 |
| Variável-alvo | `churn` |

O dataset possui uma linha por cliente e combina informações de perfil da conta, planos contratados, uso de chamadas, cobranças e o desfecho final: se o cliente cancelou ou não.

## Distribuição da variável-alvo

| Churn | Clientes | Percentual |
|---|---:|---:|
| False | 2.850 | 85,51% |
| True | 483 | 14,49% |

A base é **desbalanceada**: a maior parte dos clientes não cancelou. Isso é importante porque uma métrica simples como acurácia pode ser enganosa. Um modelo que sempre previsse "não churn" já teria uma acurácia alta, mas seria ruim para o objetivo do projeto. Na etapa de modelagem, será necessário observar métricas como *recall*, *precision*, *F1-score*, matriz de confusão e possivelmente ROC-AUC.

## Primeiros padrões observados

### Plano internacional

| Plano internacional | Clientes | Taxa de churn |
|---|---:|---:|
| No | 3.010 | 11,50% |
| Yes | 323 | 42,41% |

Clientes com plano internacional apresentam uma taxa de churn muito superior. Esse é um dos sinais mais fortes encontrados na exploração inicial.

### Plano de voice mail

| Plano de voice mail | Clientes | Taxa de churn |
|---|---:|---:|
| No | 2.411 | 16,72% |
| Yes | 922 | 8,68% |

Clientes com plano de voice mail parecem cancelar menos. Isso pode indicar maior engajamento com os serviços contratados ou um perfil de cliente mais estável.

### Chamadas ao atendimento

| Chamadas ao atendimento | Clientes | Taxa de churn |
|---:|---:|---:|
| 0 | 697 | 13,20% |
| 1 | 1.181 | 10,33% |
| 2 | 759 | 11,46% |
| 3 | 429 | 10,26% |
| 4 | 166 | 45,78% |
| 5 | 66 | 60,61% |
| 6 | 22 | 63,64% |

A partir de 4 chamadas ao atendimento, a taxa de churn cresce bastante. Isso reforça a hipótese de que contato frequente com o suporte pode ser um sinal de insatisfação.

## Comparação de médias entre churn e não churn

| Variável | Média sem churn | Média com churn | Leitura inicial |
|---|---:|---:|---|
| `total day minutes` | 175,18 | 206,91 | Clientes que cancelam usam mais minutos durante o dia |
| `total eve minutes` | 199,04 | 212,41 | Diferença moderada |
| `total night minutes` | 200,13 | 205,23 | Diferença pequena |
| `total intl minutes` | 10,16 | 10,70 | Diferença pequena |
| `customer service calls` | 1,45 | 2,23 | Clientes que cancelam acionam mais o atendimento |
| `number vmail messages` | 8,60 | 5,12 | Clientes que cancelam têm menos mensagens de voice mail |

As diferenças mais relevantes aparecem em `total day minutes`, `customer service calls` e `number vmail messages`.

## Correlações iniciais com churn

As maiores correlações lineares observadas com a variável `churn` foram:

| Variável | Correlação com churn |
|---|---:|
| `customer service calls` | 0,209 |
| `total day minutes` | 0,205 |
| `total day charge` | 0,205 |
| `total eve minutes` | 0,093 |
| `number vmail messages` | -0,090 |
| `total intl minutes` | 0,068 |

Essas correlações não devem ser interpretadas como causalidade, mas ajudam a indicar variáveis importantes para investigar e modelar.

## Localização (estado) — teste da hipótese H4

A hipótese **H4** afirma que a localização (`state`) tem pouco poder preditivo isoladamente. Como `state` é categórica com muitas categorias, a correlação linear usada acima não é adequada. Para avaliá-la, observamos a taxa de churn por estado, o tamanho da amostra em cada grupo e aplicamos o **teste qui-quadrado de independência**.

### Taxa de churn por estado

| | Estado | Clientes | Taxa de churn |
|---|---|---:|---:|
| Maior | NJ | 68 | 26,47% |
| Maior | CA | 34 | 26,47% |
| Maior | TX | 72 | 25,00% |
| Maior | MD | 70 | 24,29% |
| Maior | SC | 60 | 23,33% |
| Menor | IA | 44 | 6,82% |
| Menor | VA | 77 | 6,49% |
| Menor | AZ | 64 | 6,25% |
| Menor | AK | 52 | 5,77% |
| Menor | HI | 53 | 5,66% |

A taxa varia de cerca de 5,7% a 26,5% entre os extremos, contra uma média geral de 14,49%.

### Tamanho da amostra por estado

| Indicador | Valor |
|---|---:|
| Estados | 51 |
| Mínimo de clientes por estado | 34 |
| Mediana de clientes por estado | 65 |
| Máximo de clientes por estado | 106 |
| Média de churners por estado | 9,47 |

Cada estado tem poucos registros (mediana de 65 clientes e cerca de 9 churners por estado). Isso torna a taxa de churn de qualquer estado isolado **instável**: uma diferença de poucos clientes muda bastante o percentual.

### Teste qui-quadrado de independência

| Indicador | Valor |
|---|---:|
| Qui-quadrado (χ²) | 83,04 |
| Graus de liberdade | 50 |
| p-valor | 0,0023 |
| Cramér's V | 0,158 |
| Células esperadas < 5 | 1 de 102 |

O p-valor (0,0023) é menor que 0,05, então a distribuição de churn **varia de forma estatisticamente significativa** entre estados — não é apenas ruído. O teste é válido, pois apenas 1 das 102 células tem frequência esperada abaixo de 5. Porém, o tamanho do efeito é **pequeno** (Cramér's V ≈ 0,16).

A leitura é que existe **algum** sinal associado à localização, mas ele é fraco e instável quando usado isoladamente, além de envolver alta cardinalidade (51 categorias) com risco de overfitting na modelagem.

## Colunas redundantes e pouco úteis

As colunas de cobrança (`total day charge`, `total eve charge`, `total night charge`, `total intl charge`) são praticamente combinações diretas dos minutos correspondentes. Por exemplo:

| Par de variáveis | Correlação |
|---|---:|
| `total day minutes` x `total day charge` | 1,000000 |
| `total eve minutes` x `total eve charge` | 1,000000 |
| `total night minutes` x `total night charge` | 0,999999 |
| `total intl minutes` x `total intl charge` | 0,999993 |

Para evitar redundância na modelagem, será razoável escolher entre minutos ou cobranças, em vez de usar os dois grupos ao mesmo tempo.

Além disso:

- `phone number` é um identificador e deve ser removido por não ter valor preditivo real;
- `area code` apresentou correlação muito baixa com churn;
- `state` pode conter algum sinal, mas precisa ser tratado com cuidado porque cada estado tem uma quantidade limitada de registros.

## Anomalias e cuidados

Não foram encontrados valores ausentes nem duplicados. Também não apareceram inconsistências evidentes nos valores mínimos e máximos das principais variáveis numéricas. O principal cuidado identificado é o **desbalanceamento da variável-alvo**.

Outro cuidado é não interpretar padrões de forma causal. Por exemplo, clientes com mais chamadas ao atendimento cancelam mais, mas isso não prova que as chamadas causam o churn; elas provavelmente são um sinal de problemas anteriores ou insatisfação.

## Hipóteses refinadas para a modelagem

Com base na exploração, as hipóteses iniciais ficam mais fortes:

- **H1 confirmada como promissora:** clientes com muitas chamadas ao atendimento têm maior propensão ao churn.
- **H2 confirmada como promissora:** clientes com maior uso diurno apresentam maior taxa de churn.
- **H3 confirmada como promissora:** clientes com plano internacional apresentam churn bem acima da média.
- **H4 refinada (testada):** o teste qui-quadrado mostra que a localização tem associação estatisticamente significativa com o churn (p = 0,0023), mas o efeito é pequeno (Cramér's V ≈ 0,16) e a amostra por estado é reduzida. Logo, a localização não é puro ruído, porém é um preditor fraco e instável isoladamente — menos central do que uso, plano internacional e atendimento.

## Conclusão da etapa Explore

A exploração indica que o problema é adequado para classificação binária, mas exige cuidado com desbalanceamento e seleção de variáveis. As variáveis mais importantes para levar à modelagem parecem ser `international plan`, `customer service calls`, `total day minutes`, `voice mail plan` e `number vmail messages`.

A originalidade do projeto será introduzida na etapa **M — Model the Data**. Portanto, esta etapa **E** mantém o comportamento esperado do Projeto FMF: compreender os dados, validar sua qualidade e preparar as decisões para a modelagem.
