# Semana 2 — Get the Data

**Dataset:** Churn in Telecom's Dataset (SyriaTel)
**Fonte primária:** Kaggle — https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset
**Fonte usada no projeto FMF:** repositório da autora — https://github.com/GraceMwende/Customer-Churn (`CustomerChurnData.csv`)
**Arquivo no repositório:** `data/CustomerChurnData.csv`

---

## Visão geral (verificada via Pandas)

| Item | Valor |
|---|---|
| Linhas (clientes) | 3.333 |
| Colunas (atributos) | 21 |
| Valores ausentes | 0 |
| Linhas duplicadas | 0 |
| Variável-alvo | `churn` (booleana) |
| Distribuição do alvo | 85,51% False (retidos) / **14,49% True (churn)** |

⚠️ **Dataset desbalanceado:** a classe positiva (churn) é minoritária (~1 em cada 7 clientes). Isso será tratado na fase de Modelagem (ex.: SMOTE, métricas além de acurácia).

## Como os dados foram amostrados?

O dataset é um conjunto público clássico (originalmente distribuído pela BigML), contendo registros de clientes de uma operadora de telecomunicações ("SyriaTel"). Cada linha representa **um cliente único**, com atributos agregados de uso (minutos, chamadas e cobranças por período do dia), planos contratados e o desfecho final (cancelou ou não). Trata-se de um **snapshot retrospectivo** — os desfechos já são conhecidos, o que viabiliza aprendizado supervisionado.

## Quais dados são relevantes?

**Atributos de uso (numéricos):** `total day/eve/night/intl minutes`, `total day/eve/night/intl calls`, `account length`, `number vmail messages`, `customer service calls`

**Atributos de plano (categóricos):** `international plan`, `voice mail plan`, `state`

**Atributos redundantes (candidatos a exclusão na Exploração):** as colunas `total * charge` têm correlação perfeita com as respectivas colunas de minutos (charge = minutos × tarifa fixa) — manter ambas gera multicolinearidade.

**Atributos irrelevantes:** `phone number` (identificador único, sem valor preditivo) e `area code` (não discrimina churn).

## Há questões de privacidade?

- O dataset contém a coluna `phone number`, que em contexto real seria **dado pessoal identificável (PII)**. Como o conjunto é público, anonimizado na origem e usado para fins educacionais, não há violação — mas a coluna será **descartada** também por boa prática de privacidade (princípio da minimização de dados, alinhado à LGPD).
- Não há nomes, documentos, endereços ou dados sensíveis (saúde, biometria etc.).
- Em um cenário de produção real, dados de uso de telefonia exigiriam consentimento e tratamento conforme LGPD/GDPR.

## Reprodutibilidade — como obter os dados

```bash
# Opção 1: Kaggle (requer conta)
# https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset

# Opção 2: direto do repositório do projeto FMF de referência
curl -L "https://raw.githubusercontent.com/GraceMwende/Customer-Churn/main/CustomerChurnData.csv" -o data/CustomerChurnData.csv
```

```python
import pandas as pd
df = pd.read_csv("data/CustomerChurnData.csv")
assert df.shape == (3333, 21)
```
