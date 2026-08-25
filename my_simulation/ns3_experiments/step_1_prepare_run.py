#!/usr/bin/env python3
"""Prepare an ns-3 run folder that links to pre-generated satellite-network data."""

from __future__ import annotations

import csv
import glob
import os
import random
import shutil
from dotenv import load_dotenv

load_dotenv('config.env')
# ============================================================================
# DIRETÓRIOS COM DADOS GERADOS
# ============================================================================

SIMULACAO = os.getenv("SIMULACAO")

DYNAMIC_STATE_NAME = os.getenv("STATE")

SEED = int(os.getenv("SEED"))

TIME_STEP_S = int(os.getenv("TIME_STEP_S"))
TIME_STEP_MS = TIME_STEP_S * 1000
DURATION_S = int(os.getenv("DURATION_S"))

DIR_NAME = SIMULACAO
SIMULATION_END_TIME_NS = DURATION_S * 1000000000 # 20 MINUTOS
DYNAMIC_STATE_UPDATE_INTERVAL_NS = TIME_STEP_MS * 1000000  # Normalmente igual ao TIME_STEP_MS; reduza para capturar mais trocas
ISL_UTILIZATION_TRACKING_INTERVAL_NS = TIME_STEP_MS * 1000000  # Granularidade da utilizacao dos ISLs; reduzir aumenta overhead
SIMULATION_SEED = SEED

# ============================================================================
# CARGA DE REDE PARA AVALIACAO DE HANDOVER
# ============================================================================
ENABLE_TCP_FLOW_SCHEDULER = True  # Habilita fluxos TCP fim a fim para medir FCT/throughput/completude
ENABLE_UDP_BURST_SCHEDULER = True  # Habilita bursts UDP para medir entrega e sensibilidade a congestionamento
ENABLE_PINGMESH_SCHEDULER = True  # Habilita RTT continuo entre pares de GS

NUM_TCP_FLOWS = int(os.getenv("NUM_TCP_FLOWS"))  # MUDAR: para rodada principal, aumente para dezenas ou centenas de fluxos
TCP_FLOW_SIZE_BYTES = os.getenv("TCP_FLOW_SIZE_BYTES")  # Tamanho de cada fluxo TCP; aumente para carga mais pesada
TCP_FLOW_SPACING_NS = os.getenv("TCP_FLOW_SPACING_NS")  # Separacao entre inicios dos fluxos; reduza para maior concorrencia

NUM_UDP_BURSTS = os.getenv("NUM_UDP_BURSTS")  # MUDAR: para rodada principal, aumente o numero de bursts
UDP_BURST_RATE_MEGABIT_PER_S = float(os.getenv("UDP_BURST_RATE_MEGABIT_PER_S"))  # Taxa alvo do burst; ajuste para testar saturacao
UDP_BURST_DURATION_NS = os.getenv("UDP_BURST_DURATION_NS")  # Duração de cada burst; aumentar estressa mais a rede
UDP_BURST_SPACING_NS = os.getenv("UDP_BURST_SPACING_NS")  # Separacao entre bursts; reduzir aumenta superposicao

PINGMESH_INTERVAL_NS = os.getenv("PINGMESH_INTERVAL_NS")  # Intervalo de ping; menor = RTT mais fino, maior = menos overhead
PINGMESH_MAX_PAIRS = int(os.getenv("PINGMESH_MAX_PAIRS"))  # MUDAR: Limite de pares para pingmesh; reduza para aliviar custo de controle

MAX_LOGGED_TCP_FLOW_IDS = int(os.getenv("MAX_LOGGED_TCP_FLOW_IDS"))  # Quantos fluxos terão logs detalhados de cwnd/RTT/progress
MAX_LOGGED_UDP_BURST_IDS = int(os.getenv("MAX_LOGGED_UDP_BURST_IDS"))  # Quantos bursts terão logs detalhados de sent/received timestamps

TCP_FLOW_SCHEDULE_FILENAME = "tcp_flow_schedule.csv"  # Arquivo de agenda gerado em runs/<run_name>/
UDP_BURST_SCHEDULE_FILENAME = "udp_burst_schedule.csv"  # Arquivo de agenda gerado em runs/<run_name>/


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = SCRIPT_DIR
REPO_ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, "..", ".."))

SOURCE_SATELLITE_NETWORK_DIR = os.path.join(REPO_ROOT, "my_simulation", "gen_data", DIR_NAME)
SOURCE_DYNAMIC_STATE_DIR = os.path.join(SOURCE_SATELLITE_NETWORK_DIR, DYNAMIC_STATE_NAME)


RUN_NAME = f"{DIR_NAME}_{DYNAMIC_STATE_NAME}"
RUNS_DIR = os.path.join(EXPERIMENT_DIR, "runs")
RUN_DIR = os.path.join(RUNS_DIR, RUN_NAME)
LAST_PREPARED_RUN_FILE = os.path.join(RUNS_DIR, ".last_prepared_run")

# olhar o que tem que mudar aqui 
# o que mudar baseado no tempo de execucao: simulation_end_time_ns, dynamic_state_update_interval_ns, isl_utilization_tracking_interval_ns
def read_num_satellites(tles_path: str) -> int:
    with open(tles_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    parts = first_line.split()
    if len(parts) != 2:
        raise ValueError(f"Invalid TLE header in {tles_path}: {first_line}")
    num_orbits = int(parts[0])
    sats_per_orbit = int(parts[1])
    return num_orbits * sats_per_orbit


def read_ground_station_ids(ground_stations_path: str) -> list[int]:
    gs_ids: list[int] = []
    with open(ground_stations_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            gs_id = int(line.split(",", 1)[0])
            gs_ids.append(gs_id)
    if not gs_ids:
        raise ValueError(f"No ground stations found in {ground_stations_path}")
    return gs_ids


def format_set(values: list[int]) -> str:
    if not values:
        return "set()"
    return "set(" + ",".join(str(v) for v in values) + ")"


def format_endpoint_pairs(pairs: list[tuple[int, int]]) -> str:
    if not pairs:
        return "set()"
    return "set(" + ",".join(f"{src}->{dst}" for src, dst in pairs) + ")"


def load_first_fstate_path(dynamic_state_dir: str) -> str | None:
    paths = glob.glob(os.path.join(dynamic_state_dir, "fstate_*.txt"))
    if not paths:
        return None
    paths.sort(key=lambda p: int(os.path.basename(p).split(".", 1)[0].rsplit("_", 1)[1]))
    return paths[0]


def build_reachable_gs_pairs(
    fstate_path: str | None,
    num_sats: int,
    gs_node_ids: list[int],
) -> list[tuple[int, int]]:
    if fstate_path is None:
        return []

    gs_set = set(gs_node_ids)
    connected_sat_per_gs: dict[int, int] = {}
    sat_reachable_gs: dict[int, set[int]] = {}

    with open(fstate_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 5:
                continue

            current = int(parts[0])
            dest = int(parts[1])
            next_hop = int(parts[2])

            if current in gs_set and 0 <= next_hop < num_sats:
                connected_sat_per_gs.setdefault(current, next_hop)

            if 0 <= current < num_sats and dest in gs_set and next_hop >= 0:
                sat_reachable_gs.setdefault(current, set()).add(dest)

    pairs: list[tuple[int, int]] = []
    for src_gs in gs_node_ids:
        src_sat = connected_sat_per_gs.get(src_gs)
        if src_sat is None:
            continue
        for dst_gs in sorted(sat_reachable_gs.get(src_sat, set())):
            if dst_gs != src_gs:
                pairs.append((src_gs, dst_gs))

    return pairs


def generate_tcp_flows(
    gs_node_ids: list[int],
    candidate_pairs: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int, int, int, int]]:
    flows: list[tuple[int, int, int, int, int]] = []
    num_nodes = len(gs_node_ids)
    if num_nodes < 2:
        return flows

    for flow_id in range(NUM_TCP_FLOWS):
        if candidate_pairs:
            src, dst = candidate_pairs[(flow_id + rng.randint(0, len(candidate_pairs) - 1)) % len(candidate_pairs)]
        else:
            src_idx = flow_id % num_nodes
            dst_idx = (flow_id * 7 + rng.randint(1, max(1, num_nodes - 1))) % num_nodes
            if dst_idx == src_idx:
                dst_idx = (dst_idx + 1) % num_nodes
            src = gs_node_ids[src_idx]
            dst = gs_node_ids[dst_idx]
        start_ns = flow_id * TCP_FLOW_SPACING_NS
        flows.append((flow_id, src, dst, TCP_FLOW_SIZE_BYTES, start_ns))
    return flows


def generate_udp_bursts(
    gs_node_ids: list[int],
    candidate_pairs: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int, int, float, int, int]]:
    bursts: list[tuple[int, int, int, float, int, int]] = []
    num_nodes = len(gs_node_ids)
    if num_nodes < 2:
        return bursts

    for burst_id in range(NUM_UDP_BURSTS):
        if candidate_pairs:
            src, dst = candidate_pairs[(burst_id * 3 + rng.randint(0, len(candidate_pairs) - 1)) % len(candidate_pairs)]
        else:
            src_idx = (burst_id * 5 + 1) % num_nodes
            dst_idx = (burst_id * 11 + rng.randint(1, max(1, num_nodes - 1))) % num_nodes
            if dst_idx == src_idx:
                dst_idx = (dst_idx + 1) % num_nodes
            src = gs_node_ids[src_idx]
            dst = gs_node_ids[dst_idx]
        start_ns = burst_id * UDP_BURST_SPACING_NS
        bursts.append((burst_id, src, dst, UDP_BURST_RATE_MEGABIT_PER_S, start_ns, UDP_BURST_DURATION_NS))
    return bursts


def generate_pingmesh_pairs(gs_node_ids: list[int]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    num_nodes = len(gs_node_ids)
    if num_nodes < 2:
        return pairs

    limit = min(PINGMESH_MAX_PAIRS, num_nodes)
    for i in range(limit):
        src = gs_node_ids[i]
        dst = gs_node_ids[(i + 1) % num_nodes]
        pairs.append((src, dst))
    return pairs


def write_tcp_schedule(path: str, flows: list[tuple[int, int, int, int, int]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for flow_id, src, dst, size_bytes, start_ns in flows:
            writer.writerow([flow_id, src, dst, size_bytes, start_ns, "", "handover_eval"]) 


def write_udp_schedule(path: str, bursts: list[tuple[int, int, int, float, int, int]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for burst_id, src, dst, rate_mbps, start_ns, duration_ns in bursts:
            writer.writerow([burst_id, src, dst, rate_mbps, start_ns, duration_ns, "", "handover_eval"]) 


def build_config_text(pingmesh_endpoint_pairs: str, logged_tcp_ids: list[int], logged_udp_ids: list[int]) -> str:
    # ============================================================================
    # PROPERTIES DE CONFIGURACAO PARA NS-3
    # ============================================================================
    return f"""simulation_end_time_ns={SIMULATION_END_TIME_NS}
dynamic_state_update_interval_ns={DYNAMIC_STATE_UPDATE_INTERVAL_NS}
isl_utilization_tracking_interval_ns={ISL_UTILIZATION_TRACKING_INTERVAL_NS}
isl_data_rate_megabit_per_s=10.0
gsl_data_rate_megabit_per_s=10.0
isl_max_queue_size_pkts=100
gsl_max_queue_size_pkts=100
enable_isl_utilization_tracking=true
simulation_seed={SIMULATION_SEED}
satellite_network_dir=satellite_network_state
satellite_network_routes_dir={DYNAMIC_STATE_NAME}
enable_tcp_flow_scheduler={str(ENABLE_TCP_FLOW_SCHEDULER).lower()}
tcp_flow_schedule_filename={TCP_FLOW_SCHEDULE_FILENAME}
tcp_flow_enable_logging_for_tcp_flow_ids={format_set(logged_tcp_ids)}
enable_udp_burst_scheduler={str(ENABLE_UDP_BURST_SCHEDULER).lower()}
udp_burst_schedule_filename={UDP_BURST_SCHEDULE_FILENAME}
udp_burst_enable_logging_for_udp_burst_ids={format_set(logged_udp_ids)}
enable_pingmesh_scheduler={str(ENABLE_PINGMESH_SCHEDULER).lower()}
pingmesh_interval_ns={PINGMESH_INTERVAL_NS}
pingmesh_endpoint_pairs={pingmesh_endpoint_pairs}
tcp_socket_type=TcpNewReno
"""

def reset_path(path: str) -> None:
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def link_or_copy(source: str, target: str) -> None:
    reset_path(target)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    relative_source = os.path.relpath(source, os.path.dirname(target))
    try:
        os.symlink(relative_source, target)
    except OSError:
        if os.path.isdir(source):
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def main() -> int:
    if not os.path.isdir(SOURCE_SATELLITE_NETWORK_DIR):
        raise SystemExit(f"Missing source data directory: {SOURCE_SATELLITE_NETWORK_DIR}")
    if not os.path.isdir(SOURCE_DYNAMIC_STATE_DIR):
        raise SystemExit(f"Missing source dynamic-state directory: {SOURCE_DYNAMIC_STATE_DIR}")

    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)
    link_or_copy(SOURCE_SATELLITE_NETWORK_DIR, os.path.join(RUN_DIR, "satellite_network_state"))
    link_or_copy(SOURCE_DYNAMIC_STATE_DIR, os.path.join(RUN_DIR, DYNAMIC_STATE_NAME))

    tles_path = os.path.join(RUN_DIR, "satellite_network_state", "tles.txt")
    ground_stations_path = os.path.join(RUN_DIR, "satellite_network_state", "ground_stations.txt")
    num_sats = read_num_satellites(tles_path)
    gs_ids = read_ground_station_ids(ground_stations_path)
    gs_node_ids = [num_sats + gs_id for gs_id in gs_ids]
    rng = random.Random(SIMULATION_SEED)

    first_fstate_path = load_first_fstate_path(SOURCE_DYNAMIC_STATE_DIR)
    candidate_pairs = build_reachable_gs_pairs(first_fstate_path, num_sats, gs_node_ids)

    tcp_flows = generate_tcp_flows(gs_node_ids, candidate_pairs, rng) if ENABLE_TCP_FLOW_SCHEDULER else []
    udp_bursts = generate_udp_bursts(gs_node_ids, candidate_pairs, rng) if ENABLE_UDP_BURST_SCHEDULER else []
    pingmesh_pairs = generate_pingmesh_pairs(gs_node_ids) if ENABLE_PINGMESH_SCHEDULER else []

    if ENABLE_TCP_FLOW_SCHEDULER:
        write_tcp_schedule(os.path.join(RUN_DIR, TCP_FLOW_SCHEDULE_FILENAME), tcp_flows)
    if ENABLE_UDP_BURST_SCHEDULER:
        write_udp_schedule(os.path.join(RUN_DIR, UDP_BURST_SCHEDULE_FILENAME), udp_bursts)

    logged_tcp_ids = [entry[0] for entry in tcp_flows[:MAX_LOGGED_TCP_FLOW_IDS]]
    logged_udp_ids = [entry[0] for entry in udp_bursts[:MAX_LOGGED_UDP_BURST_IDS]]
    config_text = build_config_text(
        format_endpoint_pairs(pingmesh_pairs),
        logged_tcp_ids,
        logged_udp_ids,
    )

    with open(os.path.join(RUN_DIR, "config_ns3.properties"), "w", encoding="utf-8") as f:
        f.write(config_text)

    with open(LAST_PREPARED_RUN_FILE, "w", encoding="utf-8") as f:
        f.write(RUN_NAME + "\n")

    os.makedirs(os.path.join(RUN_DIR, "logs_ns3"), exist_ok=True)

    print(f"Prepared run folder: {RUN_DIR}")
    print("Linked/created:")
    print("  - satellite_network_state")
    print(f"  - {DYNAMIC_STATE_NAME}")
    print("  - config_ns3.properties")
    if ENABLE_TCP_FLOW_SCHEDULER:
        print(f"  - {TCP_FLOW_SCHEDULE_FILENAME} ({len(tcp_flows)} flows)")
    if ENABLE_UDP_BURST_SCHEDULER:
        print(f"  - {UDP_BURST_SCHEDULE_FILENAME} ({len(udp_bursts)} bursts)")
    if ENABLE_PINGMESH_SCHEDULER:
        print(f"  - pingmesh pairs ({len(pingmesh_pairs)} pares direcionados)")
    if candidate_pairs:
        print(f"  - reachable GS pairs from first fstate: {len(candidate_pairs)}")
    else:
        print("  - reachable GS pairs from first fstate: 0 (fallback para pares pseudo-aleatorios)")
    print("  - logs_ns3/")
    print(f"  - marker: {LAST_PREPARED_RUN_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
