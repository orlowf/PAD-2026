# PAD-2026 — Predição de Churn

Repositório da disciplina de **Pensamento Analítico de Dados** para desenvolvimento de um projeto seguindo o processo **AGEMC**:

- **A — Ask an interesting question**
- **G — Get the data**
- **E — Explore the data**
- **M — Model the data**
- **C — Communicate and visualize the results**

## Tema do projeto

O projeto trabalha com **predição de churn** em uma empresa de telecomunicações. Churn é o cancelamento do serviço por parte de um cliente. O objetivo é usar dados de perfil, planos contratados, uso do serviço e chamadas ao atendimento para estimar se um cliente tende a cancelar ou permanecer.

## Projeto FMF escolhido

O Projeto FMF de referência é:

**Predicting Customer Churn Project: A Machine Learning Approach to Binary Classification**

Referência: <https://medium.com/@gracemwendemicheni/predicting-customer-churn-project-df39da063221>

Esse projeto já apresenta um fluxo completo de ciência de dados: pergunta, dados, exploração, modelagem e comunicação dos resultados.

## Projeto original

O projeto original deste grupo mantém a mesma base conceitual do Projeto FMF nas três primeiras etapas:

- **Ask:** mesma pergunta científica sobre previsão de churn.
- **Get:** mesmo dataset de churn da SyriaTel.
- **Explore:** mesma investigação inicial dos dados, padrões e cuidados.

A **originalidade será definida e aplicada na etapa M — Model the Data**. Como a comunicação dos resultados depende da modelagem, a etapa **C — Communicate** também será ajustada de acordo com a originalidade escolhida.

Status atual da originalidade: **a definir**.

## Pergunta científica

É possível prever, a partir do comportamento de uso e do perfil de assinatura de um cliente de telecomunicações, se ele irá cancelar o serviço antes que o cancelamento aconteça?

## Dados

O dataset utilizado é o **Churn in Telecom's Dataset**, associado ao caso SyriaTel.

- Fonte primária: <https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset>
- Fonte usada no Projeto FMF: <https://github.com/GraceMwende/Customer-Churn>
- Arquivo local: `data/CustomerChurnData.csv`

Resumo verificado dos dados:

- 3.333 clientes
- 21 colunas
- 0 valores ausentes
- 0 linhas duplicadas
- variável-alvo: `churn`
- distribuição do alvo: 85,51% sem churn e 14,49% com churn

## Estrutura do repositório

```text
PAD-2026/
├── data/
│   └── CustomerChurnData.csv
├── docs/
│   ├── semana1_ask.md
│   ├── semana2_get.md
│   └── semana3_explore.md
├── notebooks/
│   └── projeto_churn_agemc.ipynb
├── src/
│   └── .gitkeep
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Entregas por etapa

| Semana | Etapa | Arquivo | Status |
|---|---|---|---|
| 1 | Ask | `docs/semana1_ask.md` | Concluído |
| 2 | Get | `docs/semana2_get.md` | Concluído |
| 3 | Explore | `docs/semana3_explore.md` | Concluído |
| Projeto completo | AGEMC parcial | `notebooks/projeto_churn_agemc.ipynb` | Em desenvolvimento |
| Próxima etapa | Model | a definir | Pendente |
| Resultado final | Communicate | a definir | Pendente |

## Principais achados da exploração

A análise exploratória inicial indicou que:

- a base é desbalanceada, pois apenas 14,49% dos clientes possuem `churn = True`;
- clientes com plano internacional apresentam taxa de churn maior;
- clientes com 4 ou mais chamadas ao atendimento têm aumento forte na taxa de churn;
- clientes que cancelam usam, em média, mais minutos diurnos;
- colunas de cobrança são quase perfeitamente correlacionadas com as colunas de minutos e podem ser redundantes na modelagem;
- `phone number` é identificador e deve ser removido antes da modelagem.

Esses pontos preparam a etapa de modelagem, onde será inserida a originalidade do projeto.
