# Treinamento de Aprendizado por Reforço para Handover LEO (RL-Train)

Este diretório contém uma implementação completa em Python puro (sem dependências externas pesadas além de `numpy`, `astropy`, `ephem` e `networkx`) para treinar um agente de Aprendizado por Reforço (RL) para otimização de handovers em constelações de satélites LEO no framework **Hypatia**.

O algoritmo utilizado é o **REINFORCE (Policy Gradient)** com otimizador Adam, implementado diretamente em NumPy.

---

## 🛠️ Estrutura de Arquivos

*   `satellite_env.py`: O ambiente de simulação geométrica leve que calcula distâncias, SNR, visibilidade e tempo de perda de sinal (`time_to_loss`) para os satélites em cada timestamp.
*   `policy_gradient_agent.py`: A rede neural de política (MLP de 2 camadas) e o algoritmo de backpropagation em NumPy.
*   `train.py`: Script principal para treinar o agente e gerar os arquivos de rotas (`fstate_*.txt`) e banda (`gsl_if_bandwidth_*.txt`) otimizados para o simulador ns-3.

---

## 🧠 Modelagem de Aprendizado por Reforço (RL)

### 1. Estado (Observação)
Para cada Ground Station (estação terrestre), o agente observa os **4 satélites visíveis mais próximos** ($K=4$). Para cada um deles, o vetor de observação de tamanho 16 contém:
1.  **Elevação normalizada** (ângulo de elevação / 90°).
2.  **Tempo restante de visibilidade** (`time_to_loss` normalizado até 300 segundos).
3.  **Qualidade do canal (SNR)** normalizada.
4.  **Estado da conexão atual** (1 se a estação já está conectada a este satélite, 0 caso contrário).

### 2. Ações
O espaço de ação é discreto de tamanho 4. O agente escolhe um número de `0` a `3` correspondente ao índice do satélite candidato (do mais próximo ao 4º mais próximo).

### 3. Recompensa
A recompensa é calculada a cada passo de tempo com base em:
*   `+ SNR`: Premiar conexões com sinal forte (maior throughput).
*   `- 0.1 * Latência`: Penalizar conexões com satélites muito distantes.
*   `- 15.0` (Penalidade de Handover): Aplicada caso o agente decida trocar de satélite, evitando o efeito ping-pong.
*   `- 50.0` (Penalidade de Desconexão): Aplicada caso a estação fique sem nenhum satélite visível associado.

---

## 🚀 Como Rodar o Treinamento

1.  Garanta que você já rodou o `run_simulation_executable.py` na pasta `my_simulation` para gerar os dados brutos da constelação (TLEs, ISLs, ground stations).
2.  Execute o script de treinamento:

```bash
python rl_train/train.py --episodes 30 --lr 0.01 --hidden_dim 32
```

### Parâmetros do Script:
*   `--data_dir`: Caminho para a pasta da constelação gerada (padrão: `my_simulation/gen_data/Test_Rede_500_sat_500_gs_algorithm_paired_many_only_over_isls`).
*   `--output_dir`: Pasta onde as rotas otimizadas pelo RL serão salvas (padrão: `dynamic_state_60000ms_for_1800s_rl` dentro da pasta de dados).
*   `--episodes`: Número de episódios de treino (padrão: 30, convergência rápida devido ao batch de 100 GS em paralelo).
*   `--lr`: Taxa de aprendizado (learning rate).
*   `--hidden_dim`: Neurônios na camada oculta da rede neural.

---

## 🔗 Validando os Resultados no ns-3

Depois que o treinamento concluir e gerar a pasta de estado dinâmico otimizada (`dynamic_state_60000ms_for_1800s_rl`):

1.  Abra o arquivo `ns3_experiments/step_1_prepare_run.py` e altere a seguinte variável:
    ```python
    DYNAMIC_STATE_NAME = "dynamic_state_60000ms_for_1800s_rl"
    ```
2.  Execute o fluxo padrão de simulação:
    ```bash
    python ns3_experiments/step_1_prepare_run.py
    ./ns3_experiments/step_2_run_ns3.sh
    python ns3_experiments/step_4_handover_network_metrics.py
    ```
3.  Compare o arquivo `handover_network_metrics_summary.json` gerado pelo RL contra o gerado pela heurística padrão para verificar a redução na taxa de handovers e perdas de pacotes!
