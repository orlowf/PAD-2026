# Semana 5 — Communicate and Visualize the Results

**Projeto FMF escolhido:** Predição de Churn — *Predicting Customer Churn Project: A Machine Learning Approach to Binary Classification*
**Etapa do AGEMC:** C — Communicate and Visualize the Results
**Dataset:** `data/CustomerChurnData.csv`
**Notebook:** `notebooks/projeto_churn_agemc.ipynb` (seção *C — Communicate*)

---

## Objetivo da etapa

A etapa **C** fecha o ciclo AGEMC. Seu papel é transformar os resultados técnicos produzidos nas etapas anteriores — hipóteses, escolhas de modelagem, métricas e interpretação — em uma comunicação clara, visual e acessível. O objetivo é que qualquer pessoa, com ou sem formação técnica, consiga entender o que o projeto encontrou, por que os resultados são confiáveis e o que fazer com eles.

Nesta etapa não há novos cálculos: toda a matemática foi feita em M. O que muda é a perspectiva — saímos do "como foi construído" e vamos para o "o que isso significa e como usar".

---

## Público-alvo e princípios de comunicação

A comunicação foi estruturada em dois níveis:

- **Público técnico** (cientistas de dados, professores, colegas do grupo): encontra nos notebooks as decisões metodológicas, os testes formais, as métricas detalhadas e a discussão de trade-offs entre modelos.
- **Público não técnico** (gestores, equipes de negócio, qualquer pessoa curiosa): encontra neste documento e na seção C do notebook uma narrativa em linguagem acessível, com visualizações autoexplicativas e recomendações práticas.

**Princípios adotados:**

1. **Honestidade sobre incertezas**: o modelo erra. Isso é dito explicitamente, com os números.
2. **Contexto antes de métrica**: cada número vem acompanhado do que ele significa na prática.
3. **Visualizações orientadas à decisão**: cada gráfico responde a uma pergunta, não apenas "mostra os dados".
4. **Narrativa com começo, meio e fim**: da pergunta ao modelo ao uso prático.

---

## A história em linguagem acessível

### O problema

A SyriaTel, uma operadora de telecomunicações, enfrenta um desafio comum no setor: clientes que cancelam o serviço — o chamado *churn*. O cancelamento é caro: adquirir um novo cliente custa muito mais do que reter um existente. Mas a retenção reativa — ligar para o cliente após ele pedir cancelamento — costuma chegar tarde demais.

A pergunta que motivou este projeto foi:

> **É possível identificar, com antecedência, quais clientes têm maior risco de cancelar — antes que o cancelamento aconteça?**

### Os dados

Trabalhamos com um dataset de **3.333 clientes**, cada um descrito por 21 características: quanto tempo liga por período do dia, se tem plano internacional, quantas vezes acionou o atendimento, entre outras. Dos 3.333 clientes, **483 cancelaram** — cerca de 1 em cada 7.

Esse desequilíbrio (85,5% de clientes fiéis contra 14,5% que cancelaram) foi o primeiro cuidado técnico do projeto: um modelo que simplesmente dissesse "nenhum cliente vai cancelar" teria 85,5% de acurácia — e seria completamente inútil para o problema real.

### O que descobrimos

Antes de construir qualquer modelo, testamos formalmente quatro hipóteses levantadas na exploração dos dados. Os resultados foram claros:

**H1 — Chamadas ao atendimento:** clientes que ligaram 4 ou mais vezes para o atendimento cancelam em taxas altíssimas — mais de 45% a 60%, contra cerca de 10% dos que ligaram 0 a 3 vezes. O teste estatístico confirmou que essa diferença não é acaso (p ≈ 6,9 × 10⁻⁶⁰).

**H2 — Uso diurno:** clientes que cancelaram usam, em média, 18% mais minutos em ligações diurnas (206 min vs. 175 min). Ligações diurnas são as mais caras — clientes com contas mais altas ficam mais insatisfeitos.

**H3 — Plano internacional:** clientes com plano internacional cancelam em taxa de **42%**, contra 11% dos que não têm esse plano. É o segundo fator mais forte.

**H4 — Localização:** existe associação entre o estado do cliente e o churn, mas o efeito é pequeno e instável — não é um bom preditor sozinho.

**Fator protetor:** clientes com plano de *voice mail* cancelam menos (8,7% vs. 16,7%). Esse plano parece indicar maior engajamento com os serviços.

### A originalidade do projeto

O Projeto FMF de referência vai direto para o modelo de machine learning. Nosso projeto fez diferente: **usamos os testes de hipótese para decidir quais variáveis entrariam no modelo**. Variáveis sem sinal estatístico foram descartadas; as com sinal forte foram mantidas. O modelo aprendeu exatamente o que os testes indicaram — e quando perguntamos ao modelo quais variáveis mais influenciaram suas previsões, as respostas foram as mesmas que os testes apontaram.

Isso não é coincidência. É coerência metodológica: a estatística inferencial e o modelo preditivo contam a mesma história.

### O modelo

Treinamos três abordagens e avaliamos todas em dados que nunca viram durante o treinamento:

| Modelo | Recall | Precisão | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Baseline (nunca alerta) | 0% | — | — | 0,50 |
| Regressão Logística | 60% | 37% | 0,46 | 0,81 |
| **Random Forest balanceada** | **81%** | **75%** | **0,78** | **0,90** |
| Random Forest + SMOTENC | 74% | 72% | 0,73 | 0,88 |

A **Random Forest balanceada** é o modelo escolhido. Ela captura **81% dos clientes que cancelariam** (recall), com uma precisão de **75%** — de cada 4 alertas emitidos, pelo menos 3 eram cancelamentos reais.

Em termos práticos, no conjunto de teste com 834 clientes:
- 121 clientes cancelaram no período.
- O modelo identificou **98 deles** antes do cancelamento.
- Emitiu **130 alertas** no total: 98 corretos e 32 incorretos.

### O que fazer com isso

O modelo não é uma bola de cristal. Ele erra, e isso precisa ser levado em conta. O que ele oferece é uma **ordenação por risco**: dado um grupo de clientes ativos, quais têm maior probabilidade de cancelar?

Com essa ordenação, a equipe de retenção pode focar esforços nos clientes de maior risco, sem precisar acionar todos. Ordenando os clientes do mais arriscado para o menos arriscado:

- Acionando os **20% de maior risco**: captura aproximadamente **65% de todos os churners**, com foco muito mais eficiente do que uma abordagem aleatória.
- Acionando os **30% de maior risco**: a cobertura sobe para aproximadamente **80% dos churners**.

O ponto de corte ideal depende do custo de cada ação de retenção e do custo de perder um cliente — decisão que cabe ao negócio, não ao modelo.

---

## Comparação com o Projeto FMF de referência

| Dimensão | Projeto FMF (referência) | Nosso projeto |
|---|---|---|
| **Pergunta científica** | Prever churn em telecomunicações | Mesma |
| **Dataset** | SyriaTel (3.333 clientes, 21 colunas) | Mesmo |
| **Exploração** | Análise descritiva e visual | Mesma análise, com teste qui-quadrado para H4 |
| **Testes de hipótese formais** | Ausentes | H1–H4: qui-quadrado, Mann-Whitney U, ponto-bisserial (no treino) |
| **Seleção de variáveis** | Não documentada formalmente | Guiada pelos testes (M.1 → M.2), validada por validação cruzada |
| **Anti-vazamento de informação** | Não explicitado | Split antes de qualquer análise; encoding e SMOTENC dentro do pipeline |
| **Limiar de decisão** | Padrão (0,5) | Escolhido por F1 via validação cruzada no treino |
| **Interpretação** | Feature importance por impureza | Odds ratios (statsmodels) + importância por permutação (mais robusta) |
| **Coerência interna** | Modelo separado da análise | Testes, seleção e modelo se reforçam mutuamente |

A contribuição do nosso grupo não é apenas técnica — é metodológica. Ao conectar os testes de hipótese à seleção de variáveis e ao modelo final, cada decisão do pipeline tem justificativa estatística. O resultado é mais confiável, mais interpretável e mais honesto sobre o que os dados suportam.

---

## Visualizações produzidas e o que cada uma comunica

As visualizações da seção C do notebook foram projetadas para contar a história do projeto, não apenas para exibir resultados.

### Painel dos fatores de risco (C.2)

Quatro gráficos em grade 2×2, cada um respondendo a uma pergunta:

- **Plano internacional:** clientes com esse plano cancelam 3,7× mais do que os sem ele. Um sinal imediato e forte.
- **Chamadas ao atendimento:** taxa de churn quase estável até 3 chamadas (~10%), depois dispara a partir de 4 chamadas (acima de 45%). O limiar de 4 chamadas é um gatilho de alerta claro.
- **Minutos diurnos:** o boxplot mostra que a mediana dos clientes que cancelaram é visivelmente mais alta — e as médias (marcadas com losango) confirmam uma diferença de ~18% no uso.
- **Voice mail:** clientes sem esse plano cancelam o dobro dos que têm — mostrando seu efeito protetor.

### Desempenho dos modelos (C.3)

Dois painéis lado a lado:

- **Comparação de métricas:** barras agrupadas mostrando recall, precisão, F1, ROC-AUC e PR-AUC dos três modelos no teste. Permite ver de imediato que a Random Forest balanceada lidera em todas as dimensões relevantes, e que o baseline (linha tracejada) é o piso que todos os modelos superam.
- **Matriz de confusão da RF:** a melhor forma de comunicar o que o modelo faz em termos absolutos — mostra exatamente quantos clientes foram corretamente identificados como churn (Verdadeiros Positivos), quantos foram perdidos (Falsos Negativos) e quantos alarmes foram falsos (Falsos Positivos).

### Uso prático do modelo (C.4)

Dois painéis:

- **Curva de ganho:** mostra que, ordenando os clientes pelo risco previsto e acionando os primeiros 20%, capturamos muito mais churners do que o esperado por acaso. É o gráfico mais diretamente ligado à tomada de decisão de negócio.
- **Distribuição de probabilidades:** histograma separando as probabilidades previstas para clientes fiéis e churners. Mostra que o modelo separa bem os dois grupos — a maioria dos churners recebe probabilidades altas, e a maioria dos fiéis recebe probabilidades baixas — e onde o limiar de decisão foi fixado.

---

## Recomendações de negócio

Com base nos resultados do projeto, as seguintes ações são recomendadas:

### 1. Monitorar chamadas ao atendimento em tempo real

O número de chamadas ao atendimento é o preditor de maior efeito (Cramér's V ≈ 0,33 no treino). Clientes que atingem 4 chamadas devem ser imediatamente sinalizados para a equipe de retenção — independentemente do modelo de ML, esse gatilho simples já identifica um grupo de alto risco.

### 2. Atenção especial aos clientes com plano internacional

Com 42% de taxa de churn, esse segmento precisa de ação específica: revisão da percepção de valor do plano, comparação com concorrentes, ou melhoria na proposta de valor. Apenas 9,7% dos clientes têm plano internacional, então o impacto de ações específicas para esse grupo é gerenciável.

### 3. Usar o modelo para priorizar a campanha de retenção

O modelo permite ordenar toda a base de clientes ativos por probabilidade de churn. A equipe de retenção deve trabalhar do topo dessa lista para baixo, até o limite do orçamento disponível:
- **Recursos limitados (≤ 20% da base):** foque no topo — você capturará ~65% dos churners com alta eficiência.
- **Recursos moderados (≤ 30% da base):** cobertura de ~80% dos churners.

O limiar de 0,35 (que maximiza F1) pode ser ajustado para 0,26 (maximiza F2, mais pró-recall) se o custo de perder um cliente for muito maior que o custo de uma ação de retenção desnecessária.

### 4. Investigar o papel do voice mail como retentor

O plano de voice mail reduz em ~50% a probabilidade de churn (odds ratio = 0,65). Isso pode indicar que incentivar a adoção desse plano — especialmente entre clientes sem ele que já mostraram outros sinais de risco — pode ser uma ação de baixo custo com impacto real na retenção.

### 5. Recalibrar o modelo periodicamente

Datasets de telecom evoluem: tarifas mudam, concorrentes entram, o perfil dos clientes muda. Recomenda-se retreinar e reavaliar o modelo a cada 6 a 12 meses, ou sempre que mudanças de negócio relevantes ocorrerem.

---

## Conclusão

Este projeto respondeu à pergunta original com clareza: **sim, é possível prever o churn de clientes de telecomunicações antes que ele aconteça**, com qualidade suficiente para suportar ações de retenção reais.

Os três fatores mais importantes são, em ordem de força de efeito:
1. **Chamadas ao atendimento ≥ 4** — o sinal mais forte e mais acionável.
2. **Plano internacional** — taxa de churn 3,7× maior.
3. **Alto uso diurno** — correlacionado com contas mais caras e maior insatisfação.

O plano de voice mail é o único fator protetor identificado com significância estatística.

A Random Forest balanceada, com limiar escolhido por F1 via validação cruzada, entrega:
- **Recall de 81%:** 4 em 5 churners são identificados antes de cancelar.
- **Precisão de 75%:** de cada alerta, 3 em 4 são cancelamentos reais.
- **ROC-AUC de 0,90:** o modelo ranqueia muito bem os clientes por risco.

A contribuição metodológica do projeto — conectar testes de hipótese, seleção de variáveis e modelo preditivo em um único pipeline coerente — resultou em um modelo com menos variáveis (8 features vs. dezenas possíveis), mais interpretável e com resultados que se sustentam tanto pela estatística inferencial quanto pelo desempenho preditivo.
