# Semana 1 — Ask an Interesting Question

**Projeto FMF escolhido:** Predição de Churn — *Predicting Customer Churn Project: A Machine Learning Approach to Binary Classification* (Grace Mwende, Medium, 2025)
**Tipo de problema:** Classificação binária (Aplicação Típica nº 4 — Predição de "Churn")

---

## A Pergunta Interessante

> **É possível prever, a partir do comportamento de uso e do perfil de assinatura de um cliente de telecomunicações, se ele irá cancelar o serviço (churn) — antes que o cancelamento aconteça?**

## Qual é o objetivo científico?

Identificar **padrões de comportamento** que diferenciam clientes que cancelam (churners) de clientes que permanecem, e construir um **modelo preditivo** capaz de generalizar esses padrões para clientes novos. Em termos formais: aprender uma função *f(características do cliente) → probabilidade de churn*.

## O que faríamos se tivéssemos todos os dados?

Se tivéssemos o histórico completo de todos os clientes da operadora (uso de minutos, planos contratados, chamadas ao atendimento, e o desfecho — cancelou ou não), poderíamos:

1. Comparar estatisticamente o comportamento dos dois grupos (churn × não-churn);
2. Identificar os fatores de maior peso na decisão de cancelar;
3. Estimar, para cada cliente ativo, o risco individual de cancelamento.

## O que queremos prever ou estimar?

Queremos **prever** a variável binária `churn` (True = cliente cancelou / False = cliente permaneceu) para clientes cujo desfecho ainda não é conhecido, usando como entrada 20 atributos de perfil e uso.

## Por que a pergunta é interessante (relevância de negócio)?

No setor de telecomunicações, **reter um cliente custa significativamente menos do que adquirir um novo**. Taxas altas de churn geram perda de receita recorrente e aumento do custo de aquisição (CAC). Um modelo que sinalize clientes em risco permite ações de retenção **proativas** e direcionadas (ofertas personalizadas, melhoria de atendimento), em vez de reativas — quando o cliente já decidiu sair.

## Hipóteses iniciais (a verificar na fase de Exploração)

- **H1:** Clientes que ligam muitas vezes para o atendimento (customer service calls) têm maior propensão ao churn (insatisfação);
- **H2:** Clientes com alto uso diurno (total day minutes) churnam mais — contas mais caras geram insatisfação;
- **H3:** Clientes com plano internacional têm taxa de churn acima da média;
- **H4:** A localização (estado) tem pouco poder preditivo isoladamente.
