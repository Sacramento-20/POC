# Protocolo de Experimentos de Handover (rede + mobilidade)

Este guia define um protocolo reproduzivel para validar politicas de handover com metricas de rede reais no pipeline atual.

## Objetivo

Comparar politicas de handover considerando:
- continuidade de servico (interrupcoes e perda)
- desempenho de transporte (throughput/FCT/RTT)
- saude da rede (utilizacao/filas/hotspots)

## O que mudou no setup

O script `step_1_prepare_run.py` agora:
- gera `config_ns3.properties` com schedulers habilitados
- gera `tcp_flow_schedule.csv`
- gera `udp_burst_schedule.csv`
- configura `pingmesh_endpoint_pairs`
- grava `runs/.last_prepared_run` para o runner usar automaticamente

## Guia de parametrizacao

Use este resumo como referencia pratica ao ajustar os experimentos.

### Parametros de tempo e granularidade

- `SIMULATION_END_TIME_NS`: duração total da simulacao. Use o perfil rapido para validar fim a fim e aumente para a rodada principal do artigo.
- `TIME_STEP_MS`: granularidade do estado dinamico. Menor valor aumenta fidelidade e custo; maior valor reduz custo, mas pode esconder transicoes.
- `DYNAMIC_STATE_UPDATE_INTERVAL_NS`: geralmente igual a `TIME_STEP_MS`. Reduza se quiser capturar handovers mais finos.
- `ISL_UTILIZATION_TRACKING_INTERVAL_NS`: igual ao `TIME_STEP_MS` por padrao. Reduzir aumenta o volume de logs e o overhead.

### Parametros de carga de rede

- `NUM_TCP_FLOWS`: numero de fluxos TCP agendados. Para o teste rapido, o valor e baixo; para o resultado principal, aumente para dezenas ou mais.
- `TCP_FLOW_SIZE_BYTES`: tamanho de cada fluxo TCP. Aumentar deixa o teste mais pesado e melhora a chance de observar FCT e stalls.
- `TCP_FLOW_SPACING_NS`: separacao entre fluxos TCP. Reduzir aumenta concorrencia.
- `NUM_UDP_BURSTS`: numero de bursts UDP. Para o artigo, rode uma versao maior do que a de validacao rapida.
- `UDP_BURST_RATE_MEGABIT_PER_S`: taxa alvo do burst; use para estressar congestionamento.
- `UDP_BURST_DURATION_NS`: duracao do burst; aumentar prolonga a pressao na rede.
- `UDP_BURST_SPACING_NS`: separacao entre bursts; reduzir aumenta sobreposicao e disputa por recursos.
- `PINGMESH_INTERVAL_NS`: intervalos menores geram RTT mais fino, mas aumentam custo de controle.
- `PINGMESH_MAX_PAIRS`: numero de pares de pingmesh. Reduza para acelerar a simulacao quando nao precisar de tantos pontos.

### Parametros de logging

- `MAX_LOGGED_TCP_FLOW_IDS`: quantos fluxos detalhados salvar em cwnd/RTT/progress.
- `MAX_LOGGED_UDP_BURST_IDS`: quantos bursts detalhados salvar.
- `ENABLE_TCP_FLOW_SCHEDULER`, `ENABLE_UDP_BURST_SCHEDULER`, `ENABLE_PINGMESH_SCHEDULER`: mantenha `true` quando quiser metricas reais de rede.

### O que usar em cada fase

- Perfil rapido: use os valores atuais para verificar que o pipeline fecha de ponta a ponta em poucos minutos.
- Rodada principal: aumente `SIMULATION_END_TIME_NS`, `NUM_TCP_FLOWS`, `NUM_UDP_BURSTS` e, se necessario, reduza `PINGMESH_INTERVAL_NS` apenas se o custo permanecer aceitavel.
- Analise final do artigo: rode multiplas sementes com a mesma configuracao principal e compare medianas, p95/p99 e intervalos de confianca.

## Fluxo recomendado

1. Preparar run

```bash
cd /home/ns3/hypatia/my_simulation/ns3_experiments
python3 step_1_prepare_run.py
```

2. Rodar simulacao ns-3

```bash
./step_2_run_ns3.sh
```

Opcional: forcar um run especifico

```bash
./step_2_run_ns3.sh visibility_dynamic_state_300000ms_for_7200s
```

3. Extrair features

```bash
python3 step_3_extract_features.py
```

4. Consolidar metricas de handover + rede

```bash
python3 step_4_handover_network_metrics.py
```

## Perfil rapido (recomendado para iteracao)

O `step_1_prepare_run.py` esta configurado para um perfil rapido por padrao:

- `simulation_end_time_ns = 900s`
- `NUM_TCP_FLOWS = 8`
- `NUM_UDP_BURSTS = 8`
- `UDP_BURST_DURATION_NS = 10s`
- `pingmesh_interval_ns = 5s`

Esse perfil reduz bastante o tempo de simulacao e ainda gera sinais suficientes para validar handover com metricas de rede.

Para o resultado principal do artigo, mantenha a mesma logica de rotas e schedulers, mas aumente a carga: mais fluxos TCP, mais bursts UDP, simulacao mais longa e, se necessario, mais sementes.

Comandos:

```bash
cd /home/ns3/hypatia/my_simulation/ns3_experiments
python3 step_1_prepare_run.py
./step_2_run_ns3.sh
python3 step_3_extract_features.py
python3 step_4_handover_network_metrics.py
```

## Logs esperados em logs_ns3

- `isl_utilization.csv`
- `tcp_flows.csv` e `tcp_flows.txt`
- `udp_bursts_outgoing.csv` e `udp_bursts_incoming.csv`
- `pingmesh.csv`
- `timing_results.csv`

## Saidas esperadas em analysis

- `features_*.csv` (step 3)
- `handover_events.csv` (step 4)
- `handover_network_metrics_summary.json` (step 4)

Se algum deles nao aparecer, confira no `config_ns3.properties` se os schedulers estao habilitados.

## Metricas para o artigo

### 1) Continuidade do handover
- Handover interruption time (por evento)
- Handover failure rate
- Packet loss em janela temporal do handover
- RTT spike pre/post handover

### 2) Transporte fim a fim
- Throughput medio e p95
- Flow Completion Time (FCT) por classe de fluxo
- CDF de RTT e cauda p99
- Fairness entre GS (Jain)

### 3) Saude da rede
- Utilizacao de ISL por tempo
- Persistencia de hotspots (>80%)
- Churn de rota (mudancas de hops)

## Desenho experimental minimo

- Politicas: geometrica, geometrica+histerese, network-aware
- Carga: baixa, media, alta
- Sementes: >= 5 por cenario
- Granularidade temporal: fina/media/grossa (ajuste `TIME_STEP_MS`)

## Validade dos resultados

Use metricas de log do ns-3 como fonte principal para conclusoes de rede.
As colunas aproximadas em `step_3_extract_features.py` (ex.: utilizacao/fila estimadas) devem ser auxiliares, nao evidencia primaria.
