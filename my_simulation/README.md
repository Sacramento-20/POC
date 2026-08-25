# Como rodar 

## 1 Passo
- Incluir dados de ground_stations e tles em input_data/

- No arquivo run_simulation_executable.py modificar campos obrigatorios para cada tipo de experimento
    - BASE_NAME
    - DURATION_S
    - TIME_STEP_MS
    - NUM_GROUND_STATIONS
    - NUM_ORBITS > 3
    - NUM_SATS_PER_ORBIT > 3
    - NUM_THREADS

- Rodar comando python run_simulation_executable.py para gerar todo cenario dinamico da orbita

- O dynamic_state ficara localizado no path gen_data/nome_da_simulacao

- Rodar simulacao no ns3, ir para diretorio ns3_experiments

- No arquivo step_1_prepare_run.py modificar campos obrigatoriamente para cada tipo de experimento:
    - DIR_NAME -> nome_da_simulacao
    - DYNAMIC_STATE_NAME -> nome da pasta do dynamic state no gen_data/nome_da_simulacao

- Modificar CONFIG_TEXT para gerar propriedades da simulacao
    Campos obrigatorios
    - simulation_end_time_ns -> DURATION_S * 100
    - dynamic_state_update_interval_ns -> TIME_STEP_MS * 100
    - isl_utilization_tracking_interval_ns -> TIME_STEP_MS * 100
    Campos opcionais
    - SIMULATION_SEED

- Rodar comando python step_1_prepare_run.py para gerar propriedades da simulacao e copia das pastas com dados gerados

- Modificar campos no script step_2_run_ns3
    - TARGET_DIR -> modificar campo para o mesmo nome do RUN_NAME no step_1_prepare_run.py
    - tirar o numero de orbitas e satelites do arquivo tle.txt para rodar o script 30 100

- Modificar campos no step_3_extract_features.py e rodar
    - RUN_NAME
    - DYNAMIC_STATE

- Correr pro abraco!s