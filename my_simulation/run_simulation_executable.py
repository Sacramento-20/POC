#!/usr/bin/env python3
"""Versão executável do exemplo mínimo usando arquivos já preparados.

O script procura primeiro por arquivos em `my_simulation/input_data/` e os
copie para `gen_data/<constellation>/` quando fizer sentido. Se um arquivo
de entrada já existir, ele é reutilizado em vez de ser regenerado.
"""

from __future__ import annotations

import math
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Adiciona o caminho para importar satgenpy
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "satgenpy"))

try:
    import satgen
except ImportError as e:
    sys.exit(1)

# ============================================================================
# CONFIGURAÇÕES DA ORBITA
# ============================================================================
INPUT_DIR = os.path.join(SCRIPT_DIR, "input_data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "gen_data")

EARTH_RADIUS_M = 6378135.0

ECCENTRICITY = 0.0000001
ARG_OF_PERIGEE_DEGREE = 0.0
PHASE_DIFF = True

ALTITUDE_M = 550000
INCLINATION_DEGREE = 53.0
MEAN_MOTION_REV_PER_DAY = 15.0

SATELLITE_CONE_RADIUS_M = ALTITUDE_M / math.tan(math.radians(30.0))
MAX_GSL_LENGTH_M = math.sqrt(SATELLITE_CONE_RADIUS_M**2 + ALTITUDE_M**2) # distancia do enlace que pode ajudar
MAX_ISL_LENGTH_M = 14_000_000.0 # margem acima do diametro da orbita para os ISLs desse shell
#MAX_ISL_LENGTH_M = 2 * (EARTH_RADIUS_M + ALTITUDE_M) # distancia do enlace que pode ajudar


# ============================================================================
# PARAMETROS DE SIMULAÇÃO
# ============================================================================
""" Simulações Validas """
INDEX = 0
ORBITS = [25,25,10,10]
SATS = [25,200,100,100]
GS = [100,100,200,300]
SIMULACAO = "simulacao_10_minutes_50s"


BASE_NAME = SIMULACAO   # NOME DO EXPERIMENTO
DURATION_S = 1200       # 20 minutos
TIME_STEP_MS = 5000    # 5 segundos

NUM_GROUND_STATIONS = GS[INDEX]
NUM_ORBITS = ORBITS[INDEX]
NUM_SATS_PER_ORBIT = SATS[INDEX]
TOTAL_SATS = NUM_ORBITS * NUM_SATS_PER_ORBIT
NUM_THREADS = 6

algorithms = [
    "algorithm_free_one_only_gs_relays",
    "algorithm_free_one_only_over_isls",
    "algorithm_free_gs_one_sat_many_only_over_isls",
    "algorithm_paired_many_only_over_isls",
]

ALGORITHM = algorithms[3]

# ============================================================================
# FIM DOS PARAMETROS
# ============================================================================

def get_gsl_interface_config(dynamic_state_algorithm: str) -> tuple[int, int, float, float]:
    if dynamic_state_algorithm == "algorithm_free_one_only_over_isls":
        return 1, 1, 1.0, 1.0
    if dynamic_state_algorithm == "algorithm_free_one_only_gs_relays":
        return 1, 1, 1.0, 1.0
    if dynamic_state_algorithm == "algorithm_free_gs_one_sat_many_only_over_isls":
        return NUM_GROUND_STATIONS, 1, float(NUM_GROUND_STATIONS), 1.0
    if dynamic_state_algorithm == "algorithm_paired_many_only_over_isls":
        return NUM_GROUND_STATIONS, 1, 1.0, 1.0
    raise ValueError("Unknown dynamic state algorithm: " + dynamic_state_algorithm)

CONSTELLATION_NAME = BASE_NAME
CONSTELLATION_DIR = os.path.join(OUTPUT_DIR, CONSTELLATION_NAME)
# ============================================================================
# HELPERS
# ============================================================================


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def copy_if_exists(src: str, dst: str) -> bool:
    if os.path.exists(src):
        ensure_parent_dir(dst)
        shutil.copyfile(src, dst)
        return True
    return False


def read_isls(path: str) -> list[tuple[int, int]]:
    isls: list[tuple[int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                isls.append((int(parts[0]), int(parts[1])))
    return isls


def main() -> int:
    print("\n" + "=" * 80)
    print("HYPATIA: Simulação de Constelação de Satélites")
    print("=" * 80)

    print("\n[1/8] Preparando diretórios...")
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(CONSTELLATION_DIR, exist_ok=True)
    print(f"      ✓ {CONSTELLATION_DIR}/")

    input_ground_basic = os.path.join(INPUT_DIR, "ground_stations.txt")
    input_tles = os.path.join(INPUT_DIR, "tles.txt")
    
    # Arquivos de saida que serao criados em output_dir/
    output_ground_extended = os.path.join(CONSTELLATION_DIR, "ground_stations.txt")
    output_tles = os.path.join(CONSTELLATION_DIR, "tles.txt")
    output_isls = os.path.join(CONSTELLATION_DIR, "isls.txt")
    output_gsl = os.path.join(CONSTELLATION_DIR, "gsl_interfaces_info.txt")
    output_description = os.path.join(CONSTELLATION_DIR, "description.txt")
    

    print("\n[2/8] Estendendo Ground Stations...")
    # gid,name,latitude_degrees_str,longitude_degrees_str,elevation_m_float,

    if os.path.exists(input_ground_basic):
        satgen.extend_ground_stations(input_ground_basic, output_ground_extended)
        print(f"      ✓ {output_ground_extended} (gerado a partir de ground_stations.basic.txt)")
    else:
        print(f"      ✗ Arquivo não encontrado: {input_ground_basic}")
        return 1

    
    print("\n[3/8] Gerando TLEs (Two-Line Element Set)...")
    if copy_if_exists(input_tles, output_tles):
        print(f"      ✓ {output_tles} (copiado de input_data/tles.txt)")
    
    print("\n[4/8] Gerando ISLs (Inter-Satellite Links)...")
    isls = satgen.generate_plus_grid_isls(
        output_isls,        # output_filename_isls
        NUM_ORBITS,         # n_orbits
        NUM_SATS_PER_ORBIT, # n_sats_per_orbit
        isl_shift=0,
    )
    print(f"      ✓ {output_isls} (gerado)")
    print(f"      → {len(isls)} ISLs")
    
    print("\n[5/8] Gerando GSL Interfaces Info...")
    num_gsl_interfaces_per_satellite, num_gsl_interfaces_per_ground_station, \
        agg_max_bandwidth_satellite, agg_max_bandwidth_ground_station = get_gsl_interface_config(ALGORITHM)
    satgen.generate_simple_gsl_interfaces_info(
        output_gsl, # filename_gsl_interfaces_info
        TOTAL_SATS, # number_of_satellites
        NUM_GROUND_STATIONS,          # number_of_ground_stations
        num_gsl_interfaces_per_satellite,
        num_gsl_interfaces_per_ground_station,
        agg_max_bandwidth_satellite,
        agg_max_bandwidth_ground_station,
    )
    print(f"      ✓ {output_gsl} (gerado)")
    
    
    print("\n[6/8] Gerando Description...")
    satgen.generate_description(output_description, MAX_GSL_LENGTH_M, MAX_ISL_LENGTH_M)
    print(f"      ✓ {output_description} (gerado)")
    print(f"      → Max GSL: {MAX_GSL_LENGTH_M/1000:.1f} km") # distancia do 
    print(f"      → Max ISL: {MAX_ISL_LENGTH_M/1000:.1f} km") # 

    print("\n[7/8] Gerando Dynamic State...")
    print(f"      → Simulando {DURATION_S}s em steps de {TIME_STEP_MS}ms")
    print(f"      → Paralelizado em {NUM_THREADS} threads")

    satgen.help_dynamic_state(
        OUTPUT_DIR,
        NUM_THREADS,
        CONSTELLATION_NAME,
        TIME_STEP_MS,
        DURATION_S,
        MAX_GSL_LENGTH_M,
        MAX_ISL_LENGTH_M,
        ALGORITHM,
        print_logs=True,
    )

    dynamic_state_dir = os.path.join(
        CONSTELLATION_DIR, f"dynamic_state_{TIME_STEP_MS}ms_for_{DURATION_S}s"
    )
    print(f"      ✓ {dynamic_state_dir}/")

    print("\n[8/8] Verificando arquivos gerados...")
    files_expected = [
        output_ground_extended,
        output_tles,
        output_isls,
        output_gsl,
        output_description,
    ]

    for file_path in files_expected:
        if os.path.exists(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            print(f"      ✓ {file_path} ({size_kb:.1f} KB)")
        else:
            print(f"      ✗ {file_path} NÃO ENCONTRADO")

    num_timesteps = int(DURATION_S * 1000 / TIME_STEP_MS) + 1
    fstate_files = [
        os.path.join(dynamic_state_dir, f"fstate_{t * TIME_STEP_MS * 1000000}.txt")
        for t in range(num_timesteps)
    ]
    fstate_count = sum(1 for file_path in fstate_files if os.path.exists(file_path))
    print(f"      ✓ {dynamic_state_dir}/ contém {fstate_count} arquivos fstate")

    gsl_files = [
        os.path.join(dynamic_state_dir, f"gsl_if_bandwidth_{t * TIME_STEP_MS * 1000000}.txt")
        for t in range(num_timesteps)
    ]
    gsl_count = sum(1 for file_path in gsl_files if os.path.exists(file_path))
    print(f"      ✓ {dynamic_state_dir}/ contém {gsl_count} arquivos gsl_if_bandwidth")

    print("\n" + "=" * 80)
    print("✓ SIMULAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 80)


    print(f"\nDados gerados em: {CONSTELLATION_DIR}/")
    print(f"\nArquivos principais:")
    print(f"  • ground_stations.txt          - Ground stations com coordenadas")
    print(f"  • tles.txt                     - Órbitas dos {TOTAL_SATS} satélites")
    print(f"  • isls.txt                     - {len(isls)} inter-satellite links")
    print(f"  • gsl_interfaces_info.txt      - Interfaces GSL")
    print(f"  • description.txt              - Parâmetros da rede")
    print(f"  • dynamic_state_*.txt/         - Estado dinâmico ({fstate_count} timesteps)")
    print(f"      ├─ fstate_*.txt            - Forwarding state (roteamento)")
    print(f"      └─ gsl_if_bandwidth_*.txt  - Bandwidth das interfaces GSL")

    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
