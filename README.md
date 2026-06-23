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

A **originalidade foi definida e aplicada na etapa M — Model the Data** (e reflete na etapa **C — Communicate**, que depende da modelagem).

### Originalidade do projeto (etapa M)

Enquanto o Projeto FMF de referência vai direto para um classificador, a contribuição original do grupo é uma **modelagem estatisticamente fundamentada**: antes de treinar qualquer modelo, **testamos formalmente as hipóteses H1–H4** levantadas na exploração, usando para cada uma o teste adequado ao tipo de dado (qui-quadrado, Mann-Whitney, correlação ponto-bisserial), com significância **e** tamanho de efeito (Cramér's V, r ponto-bisserial). Os resultados desses testes:

1. **guiam a seleção de variáveis** — só entram no modelo as variáveis com sinal estatístico; saem as redundantes (cobranças, correlação ≈ 1,0 com os minutos) e as sem sinal (`state`, `area code`);
2. **fornecem a interpretação** do modelo final — os mesmos fatores apontados pelos testes (uso diurno, chamadas ao atendimento, plano internacional) lideram a importância da Random Forest e os *odds ratios* da regressão logística.

Assim, a estatística inferencial e o modelo preditivo se reforçam, em vez de andarem separados. Todo o processo segue **rigor anti-vazamento**: o split treino/teste é feito antes de qualquer análise; os testes e a seleção usam apenas o treino; e *encoding*, *scaling* e balanceamento ficam dentro do *pipeline* avaliado por validação cruzada.

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
│   ├── semana3_explore.md
│   └── semana4_model.md
├── notebooks/
│   └── projeto_churn_agemc.ipynb
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
| 4 | Model | `docs/semana4_model.md` | Concluído |
| Projeto completo | AGEMC (A–C) | `notebooks/projeto_churn_agemc.ipynb` | Concluído |
| Semana 5 | Communicate | `docs/semana5_communicate.md` + seção C do notebook | Concluído |

## Principais achados da exploração

A análise exploratória inicial indicou que:

- a base é desbalanceada, pois apenas 14,49% dos clientes possuem `churn = True`;
- clientes com plano internacional apresentam taxa de churn maior;
- clientes com 4 ou mais chamadas ao atendimento têm aumento forte na taxa de churn;
- clientes que cancelam usam, em média, mais minutos diurnos;
- a localização (estado) tem associação estatisticamente significativa com o churn (qui-quadrado, p = 0,0023), mas com efeito pequeno (Cramér's V ≈ 0,16) e amostra reduzida por estado, sendo um preditor fraco e instável isoladamente;
- colunas de cobrança são quase perfeitamente correlacionadas com as colunas de minutos e podem ser redundantes na modelagem;
- `phone number` é identificador e deve ser removido antes da modelagem.

Esses pontos prepararam a etapa de modelagem, onde foi inserida a originalidade do projeto.

## Resultados da modelagem

A etapa **M** confirmou e quantificou os achados da exploração e produziu o modelo preditivo:

- **Hipóteses testadas no treino:** H1 (chamadas ao atendimento ≥ 4), H2 (minutos diurnos) e H3 (plano internacional) são **fortemente significativas** (p ≪ 0,05); o maior efeito é o das chamadas ao atendimento (Cramér's V ≈ 0,33), seguido do plano internacional (≈ 0,26). **H4 (estado)** tem sinal **fraco** e foi descartado por validação cruzada; o plano de *voice mail* aparece como fator **protetor**.
- **Seleção enxuta:** a matriz de entrada caiu de dezenas de colunas para **8 *features*** (6 numéricas com sinal + 2 categóricas), sem perda de desempenho.
- **Modelo escolhido:** a **Random Forest balanceada** lidera o conjunto de teste em todas as métricas — recall **0,81**, F1 **0,78**, ROC-AUC **0,90**, PR-AUC **0,80** — com o limiar de decisão escolhido por F1 via validação cruzada (não fixado em 0,5).
- **Baseline** (prever sempre "não churn") tem acurácia 0,855 mas **recall 0** — inútil para retenção, o que confirma a necessidade de tratar o desbalanceamento.
- **Interpretação:** uso diurno, contato com atendimento e plano internacional elevam o churn; *voice mail* reduz. O resultado é coerente entre testes de hipótese, *odds ratios* e importância por permutação.

Detalhes completos em [`docs/semana4_model.md`](docs/semana4_model.md) e no notebook.

## Como rodar o projeto

Crie e ative o ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Abra o notebook:

```bash
jupyter notebook notebooks/projeto_churn_agemc.ipynb
```

Ou, se preferir JupyterLab:

```bash
jupyter lab
```

## O que deve ficar no GitHub

Devem ser versionados:

- documentos das semanas em `docs/`;
- notebook principal em `notebooks/`;
- dataset utilizado em `data/`;
- `README.md`;
- `requirements.txt`;
- `.gitignore`;
- `LICENSE`.

Não devem ser versionados:

- `.venv/`;
- `.cache/`;
- `.DS_Store`;
- checkpoints de notebook;
- arquivos temporários;
- modelos treinados e artefatos pesados futuros, como `.pkl` e `.joblib`.

A pasta `src/` poderá ser criada futuramente se o projeto passar a ter código reutilizável fora do notebook.

## Próximos passos

1. ✔ Originalidade da etapa **M** definida (modelagem estatisticamente fundamentada) e aplicada no notebook.
2. ✔ Etapa **C — Communicate** concluída: `docs/semana5_communicate.md` e seção C do notebook com narrativa acessível, painel de fatores (C.2), desempenho dos modelos (C.3), curva de ganho para uso prático (C.4) e comparação com o Projeto FMF (C.5).
3. ✔ Comparação metodológica com o Projeto FMF documentada em `docs/semana5_communicate.md` e na célula C.5 do notebook.
