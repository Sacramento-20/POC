# 1. Ajuste da função de recompensa — eu começaria por aqui

Na sua Seção 8, o problema identificado é muito específico: o agente consegue reduzir as interrupções, mas paga por isso fazendo muitos handovers. A função atual é:

$$
r_t=SNR-\alpha L_{prop}-\beta I_{HO}-\delta I_{disconnect}
$$

com \(\beta=15\) e \(\delta=50\). O próprio relatório sugere aumentar \(\beta\). 

### Artigo 1 — Deep Reinforcement Learning Based Multi-Objective Handover for LEO Satellite Networks

[Deep Reinforcement Learning Based Multi-Objective Handover for LEO Satellite Networks — IEEE](https://doi.org/10.1109/VTC2025-SPRING65109.2025.11174574?utm_source=chatgpt.com)

Esse é provavelmente **um dos artigos mais importantes para a sua próxima etapa**.

O trabalho formula o handover como um problema multiobjetivo, buscando simultaneamente:

* reduzir frequência de handover;
* balancear carga dos satélites;
* maximizar throughput;
* manter estabilidade do serviço.

Os autores desenvolvem uma **nova função de recompensa** justamente para combinar esses objetivos. ([DOI][1])

### Por que ele é particularmente útil para você?

Porque você pode fazer um experimento praticamente direto:

**Seu trabalho atual:**

$$
R = SNR-\alpha L-\beta HO-\delta Disconnect
$$

**Nova versão:**

$$
R =
w_1 Q_{signal}
+w_2 Q_{latency}
+w_3 Q_{availability}
-w_4 I_{HO}
-w_5 Load
$$

e estudar sistematicamente diferentes valores dos pesos.

Isso transforma "aumentar β" em uma investigação científica muito mais interessante:

> **Qual é o trade-off ótimo entre continuidade de conexão, qualidade do enlace e número de handovers?**

### POC: Primeira etapa da avaliação do trabalho no cronograma
### H: Avaliar os resultados com base em novas métricas para a função de recompensa

---

# 2. Arquitetura com memória temporal — LSTM

Esse é, na minha opinião, o **próximo passo mais interessante cientificamente**.

Seu relatório identifica uma limitação importante:

> a MLP atual processa cada time step independentemente.

E justamente por isso sugere LSTM/Transformer para capturar a periodicidade orbital de aproximadamente 96 minutos. 

### Artigo 3 — Joint Traffic Prediction and Handover Design for LEO Satellite Networks with LSTM and Attention-Enhanced Rainbow DQN

[Joint Traffic Prediction and Handover Design for LEO Satellite Networks with LSTM and Attention-Enhanced Rainbow DQN](https://www.mdpi.com/2079-9292/14/15/3040?utm_source=chatgpt.com)

**Esse é um dos artigos que eu mais recomendo para você.**

O trabalho utiliza:

* LSTM;
* histórico temporal;
* previsão;
* handover como MDP;
* atenção;
* Rainbow DQN;
* qualidade de comunicação;
* frequência de switching;
* carga do satélite.

Os autores usam o LSTM para prever informações futuras e depois utilizam essa informação na política de handover. ([MDPI][3])

### Como adaptar ao seu projeto

Hoje:

```text
state(t) → MLP → action(t)
```

Você poderia implementar:

```text
state(t-10)
state(t-9)
...
state(t)
      ↓
    LSTM
      ↓
hidden state
      ↓
policy
      ↓
satellite
```

Por exemplo, usar uma janela de 10 ou 20 passos:

$$
S_t =
[s_{t-19},...,s_{t-1},s_t]
$$

Isso permitiria ao agente observar **a trajetória**, e não apenas o estado atual.

### POC: Segunda etapa da avaliação do trabalho no cronograma
### H: Aplicar LSTM no arquivo resultante das tragetorias, dado o timestamp gerado pelo baseline e recriar uma cenário que tem como base para a ação atual quais são as possibilidades no futuro (um dos objetivos é diminuir o handover) 

---

# 3. Transformer / atenção temporal

Eu colocaria o Transformer como **segunda alternativa ao LSTM**, e não como primeira implementação.

### Artigo 4 — Attention Is All You Need

[Attention Is All You Need — Vaswani et al.](https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com)

É o trabalho original do Transformer. A arquitetura substitui recorrência por mecanismos de self-attention, permitindo modelar relações entre diferentes posições da sequência. ([arXiv][5])

Para seu problema, a ideia seria:

```text
s(t-20) ─┐
s(t-19)  │
...      ├── Transformer Encoder → Policy → Satellite
s(t-1)   │
s(t)   ──┘
```

A vantagem conceitual é interessante: o agente poderia aprender quais momentos do histórico orbital são mais relevantes para a decisão atual.

**Mas eu não começaria por aqui.**

LSTM é uma extensão experimental mais natural da sua MLP atual e será mais fácil demonstrar que a melhoria veio da **memória temporal**, e não simplesmente de uma arquitetura muito mais complexa.

**Prioridade: ⭐⭐⭐⭐**

### POC: Terceira etapa da avaliação do trabalho no cronograma
### H: Aplicar a metodologia utilizando transformers e comparar com os resultados obtidos através do LSTM

---

# 4. Trocar REINFORCE por PPO

Esse é outro próximo passo que considero **muito forte para o seu trabalho**.

Seu relatório já identifica PPO, DQN e SAC como candidatos a algoritmos mais eficientes que REINFORCE. 

### Artigo 5 — Proximal Policy Optimization Algorithms

[Proximal Policy Optimization Algorithms — Schulman et al.](https://arxiv.org/abs/1707.06347?utm_source=chatgpt.com)

É o artigo original do PPO.

A principal motivação é melhorar a estabilidade e eficiência dos métodos policy-gradient, permitindo múltiplas atualizações sobre os dados coletados e utilizando uma função objetivo limitada/clipped. ([arXiv][6])

Isso é particularmente adequado ao seu caso porque você já possui uma política:

$$
\pi_\theta(a|s)
$$

e atualmente faz:

$$
L=-G_t\log\pi_\theta(a_t|s_t)
$$

Com PPO você poderia comparar diretamente:

| Método        | Arquitetura | Algoritmo        |
| ------------- | ----------- | ---------------- |
| Baseline      | MLP         | REINFORCE        |
| Experimento 1 | MLP         | PPO              |
| Experimento 2 | LSTM        | PPO              |
| Experimento 3 | LSTM        | PPO + load       |
| Baseline      | —           | distância mínima |

Isso produziria uma evolução experimental muito mais convincente.

**Prioridade: ⭐⭐⭐⭐⭐**

### POC: Quarta etapa da avaliação do trabalho no cronograma
### H: Implementar o algoritmo PPO para politica de recompensa e com parar com a abordagem original REINFORCE

---

# 5. Mega-constelações

A sua Seção 8 propõe sair de:

> 625 satélites

para:

> 4.000–12.000 satélites.

### Artigo 11 — SaTE: Low-Latency Traffic Engineering for Satellite Networks

[SaTE — Low-Latency Traffic Engineering for Satellite Networks](https://doi.org/10.1145/3718958.3750524?utm_source=chatgpt.com)

Esse trabalho considera uma constelação Starlink com **4.236 satélites** e explora uma representação baseada em grafos para engenharia de tráfego.
É excelente para justificar a mudança de escala.

Além disso, ele mostra que a estrutura topológica e a distribuição geográfica do tráfego podem ser exploradas para acelerar decisões de engenharia de tráfego.

---

# 11. Validação estatística — não deixe isso para o final

Sua Seção 8 corretamente propõe repetir os experimentos com múltiplas seeds e calcular intervalos de confiança. 

### Artigo 13 — Deep Reinforcement Learning That Matters

[Deep Reinforcement Learning That Matters — AAAI](https://doi.org/10.1609/aaai.v32i1.11694?utm_source=chatgpt.com)

Esse artigo é praticamente obrigatório para justificar essa parte metodológica.

Os autores mostram que resultados de Deep RL podem apresentar variação significativa devido ao não determinismo e às próprias características estocásticas do treinamento, recomendando práticas mais rigorosas de avaliação e reporte. ([AAAI Publicações][14])

Portanto, em vez de:

```text
seed = 42
resultado = 47.1%
```

faça:

```text
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

treinar
↓
avaliar
↓
média
↓
desvio padrão
↓
IC 95%
```

Isso melhora **muito** a qualidade científica do trabalho.

---

# Experimentos

### Experimento 0 — seu trabalho atual

```text
Distance heuristic
       ↓
REINFORCE + MLP
       ↓
4 satellites
       ↓
32 features
```

Isso é seu baseline.
Utilizar baseline para aplicar a outros experimentos

### Experimento 1 — Reward shaping

```text
REINFORCE + MLP
       +
nova função de recompensa
```

Variar:

$$
\beta \in \{5,10,15,25,50,75\}
$$

e medir:

* handovers;
* disconnects;
* RTT;
* UDP delivery;
* TCP FCT;
* throughput.

**Analisar resultados entre as duas funções**

---

### Experimento 2 — Memória temporal

Depois:

```text
REINFORCE + MLP
        ↓
REINFORCE + LSTM
```

Comparar:

* 20 min;
* 2 h;
* eventualmente 4 h.

A pergunta científica seria:

> **A memória temporal permite ao agente explorar a periodicidade orbital e melhorar o desempenho antes de completar um ciclo orbital?**

Essa pergunta está diretamente conectada à hipótese central do seu trabalho.

---

### Experimento 3 — PPO

Depois:

```text
MLP + REINFORCE
MLP + PPO
LSTM + REINFORCE
LSTM + PPO
```

Aí você tem um **estudo ablation muito bom**.

---

### Experimento 4 — robustez estatística

Para cada configuração:

$$
N=10
$$

seeds independentes.

Reportar:

$$
\mu \pm \sigma
$$

e IC 95%.

Isso evita que uma eventual melhoria seja simplesmente consequência de uma seed favorável.

---

## Em termos de contribuição científica

com duas hipóteses:

**H1 — Reward shaping:** uma função de recompensa multiobjetivo reduz o excesso de handovers sem sacrificar significativamente a disponibilidade.

**H2 — Temporalidade:** uma política LSTM melhora a capacidade de antecipação em relação à MLP, especialmente antes/com ao longo do primeiro ciclo orbital.


E aí o **PPO** seria a evolução algorítmica, enquanto **LSTM** seria a extensão de estado/modelagem.