#!/usr/bin/env python3
"""Extract ML-ready features from simulation outputs for RL training."""

from __future__ import annotations

import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict

import ephem
import numpy as np
from astropy import units as u

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "satgenpy"))

from satgen.distance_tools import distance_m_ground_station_to_satellite
from satgen.ground_stations import read_ground_stations_extended
from satgen.tles import read_tles
from dotenv import load_dotenv

load_dotenv('config.env')
# ============================================================================
# CONFIGURAÇÃO: ALTERAR CONFORME NECESSÁRIO
# ============================================================================

SIMULACAO = os.getenv("SIMULACAO")  # 'simulacao_20_minutes_5s_' 

STATE = os.getenv("STATE")  # 'dynamic_state_5000ms_for_1200s'


RUN_NAME = SIMULACAO + STATE  # NOME DO EXPERIMENTO
DYNAMIC_STATE = STATE  # NOME DO DYNAMIC STATE
DATA_NAME = SIMULACAO

RUN_DIR = os.path.join(SCRIPT_DIR, "runs", RUN_NAME)
DYNAMIC_STATE_DIR = os.path.join(RUN_DIR, DYNAMIC_STATE)
SATELLITE_NETWORK_DIR = os.path.join(RUN_DIR, "satellite_network_state")
ANALYSIS_DIR = os.path.join(RUN_DIR, "analysis")
LOGS_DIR = os.path.join(RUN_DIR, "logs_ns3")
ISL_UTILIZATION_CSV = os.path.join(LOGS_DIR, "isl_utilization.csv")

# Parâmetros de propagação de sinal
FREQ_GHZ = 28.0  # Frequência em GHz (Ka band típica para satélites)
TX_POWER_DBM = 20.0  # Potência transmitida em dBm
TX_GAIN_DBI = 10.0  # Ganho transmissor em dBi
RX_GAIN_DBI = 10.0  # Ganho receptor em dBi

# ============================================================================
# CONSTANTES
# ============================================================================
SPEED_OF_LIGHT = 299792458.0  # m/s
EARTH_RADIUS_M = 6378135.0


def parse_description(path: str) -> dict[str, float]:
    """Parse description.txt file."""
    result: dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            result[key] = float(value)
    return result


def parse_time_ns(filename: str) -> int:
    """Extract nanoseconds timestamp from filename."""
    match = re.search(r"_(\d+)\.txt$", os.path.basename(filename))
    if not match:
        raise ValueError(f"Invalid filename: {filename}")
    return int(match.group(1))


def load_times(dynamic_state_dir: str) -> list[int]:
    """Load all unique timestamps from fstate files."""
    files = glob.glob(os.path.join(dynamic_state_dir, "fstate_*.txt"))
    return sorted(set(parse_time_ns(path) for path in files))


def calculate_elevation_and_azimuth(
    gs_pos: dict[str, float],
    sat_pos: tuple[float, float, float],
) -> tuple[float, float]:
    """Calculate elevation and azimuth angles.
    
    Returns: (elevation_deg, azimuth_deg)
    """
    gs_x, gs_y, gs_z = float(gs_pos["x"]), float(gs_pos["y"]), float(gs_pos["z"])
    sat_x, sat_y, sat_z = sat_pos

    # Vector from GS to satellite
    dx, dy, dz = sat_x - gs_x, sat_y - gs_y, sat_z - gs_z
    distance = math.sqrt(dx**2 + dy**2 + dz**2)

    # Elevation angle: angle above horizon
    # Using dot product with radial vector (GS to Earth center)
    radial_x, radial_y, radial_z = gs_x, gs_y, gs_z
    radial_dist = math.sqrt(radial_x**2 + radial_y**2 + radial_z**2)

    dot_product = dx * radial_x + dy * radial_y + dz * radial_z
    cos_zenith = dot_product / (distance * radial_dist)
    elevation_rad = math.pi / 2 - math.acos(np.clip(cos_zenith, -1, 1))
    elevation_deg = math.degrees(elevation_rad)

    # Azimuth: angle in horizontal plane
    # Project to local horizon plane and calculate bearing
    # Simplified: use lat/lon conversion
    gs_lat = math.radians(float(gs_pos["latitude_degrees"]))
    gs_lon = math.radians(float(gs_pos["longitude_degrees"]))

    # Local horizon coordinate system
    north_x = -math.sin(gs_lat) * math.cos(gs_lon)
    north_y = -math.sin(gs_lat) * math.sin(gs_lon)
    north_z = math.cos(gs_lat)

    east_x = -math.sin(gs_lon)
    east_y = math.cos(gs_lon)
    east_z = 0

    north_component = dx * north_x + dy * north_y + dz * north_z
    east_component = dx * east_x + dy * east_y + dz * east_z

    azimuth_rad = math.atan2(east_component, north_component)
    azimuth_deg = math.degrees(azimuth_rad)
    if azimuth_deg < 0:
        azimuth_deg += 360

    return elevation_deg, azimuth_deg


def calculate_link_geometry(
    ground_station: dict[str, float],
    satellite,
    epoch_str: str,
    date_str: str,
) -> tuple[float, float, float, float, float, float]:
    """Compute distance, elevation, azimuth, radial velocity and satellite sub-lat/lon.

    Returns: (distance_m, elevation_deg, azimuth_deg, distance_rate_m_per_s,
              sat_lat_deg, sat_lon_deg)
    """
    observer = ephem.Observer()
    observer.epoch = epoch_str
    observer.date = date_str
    observer.lat = str(ground_station["latitude_degrees_str"])
    observer.lon = str(ground_station["longitude_degrees_str"])
    observer.elevation = ground_station["elevation_m_float"]

    satellite.compute(observer)

    distance_m = float(satellite.range)
    elevation_deg = math.degrees(float(satellite.alt))
    azimuth_deg = math.degrees(float(satellite.az))
    distance_rate_m_per_s = float(getattr(satellite, "range_velocity", 0.0))

    # Satellite sublatitude / sublongitude (geodetic nadir point)
    sat_lat_deg = None
    sat_lon_deg = None
    try:
        sat_lat_deg = math.degrees(float(getattr(satellite, "sublat", 0.0)))
        sat_lon_deg = math.degrees(float(getattr(satellite, "sublong", 0.0)))
    except Exception:
        # Fallback: set to 0.0 if unavailable
        sat_lat_deg = 0.0
        sat_lon_deg = 0.0

    return (
        distance_m,
        elevation_deg,
        azimuth_deg,
        distance_rate_m_per_s,
        sat_lat_deg,
        sat_lon_deg,
    )


def infer_connected_satellite(
    fstate: dict[tuple[int, int], tuple[int, int, int]],
    gs_node_id: int,
    num_sats: int,
) -> int | None:
    """Infer currently selected serving satellite for a GS from fstate.

    For GS source entries, next hop is expected to be the chosen satellite.
    """
    for (current_node, _dest_node), (next_hop, _if_out, _if_in) in fstate.items():
        if current_node == gs_node_id and 0 <= next_hop < num_sats:
            return next_hop
    return None


def compute_orbit_phase_percent(satellite) -> float:
    """Estimate orbital phase percent from mean anomaly when available."""
    angle_obj = getattr(satellite, "M", None)
    if angle_obj is None:
        angle_obj = getattr(satellite, "_M", None)
    if angle_obj is None:
        return 0.0
    phase_rad = float(angle_obj) % (2.0 * math.pi)
    return (phase_rad / (2.0 * math.pi)) * 100.0


def calculate_path_loss_friis(distance_m: float, freq_ghz: float = FREQ_GHZ) -> float:
    """Calculate path loss using Friis equation (in dB).
    
    PL(dB) = 20*log10(4π*d*f/c)
    """
    freq_hz = freq_ghz * 1e9
    numerator = 4 * math.pi * distance_m * freq_hz
    path_loss = 20 * math.log10(numerator / SPEED_OF_LIGHT)
    return path_loss


def calculate_snr(path_loss_db: float) -> float:
    """Calculate SNR from path loss (simplified model).
    
    SNR(dB) = TX_Power + TX_Gain + RX_Gain - PathLoss
    """
    snr = TX_POWER_DBM + TX_GAIN_DBI + RX_GAIN_DBI - path_loss_db
    return snr


def calculate_time_to_loss(
    distance_m: float,
    distance_rate_m_per_s: float,
    max_gsl_length_m: float,
) -> float:
    """Calculate time until GS goes out of range.
    
    Returns: time in seconds, or -1 if already out of range or moving away
    """
    if distance_m > max_gsl_length_m:
        return -1.0
    
    # ======================================================================================
    # change it to higher value to avoid inf number (something like 900s) check the ratio
    # ======================================================================================
    if distance_rate_m_per_s <= 0:  # Moving away or stationary 
        return float('inf')

    # Time = (max_distance - current_distance) / rate_of_change
    time_to_loss = (max_gsl_length_m - distance_m) / distance_rate_m_per_s
    return max(0.0, time_to_loss)


def parse_fstate(fstate_path: str) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Parse forwarding state file.
    
    Returns: {(current_node, dest_node): (next_hop, if_out, if_in)}
    """
    fstate = {}
    if not os.path.exists(fstate_path):
        return fstate

    with open(fstate_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 5:
                continue
            current = int(parts[0])
            dest = int(parts[1])
            next_hop = int(parts[2])
            if_out = int(parts[3])
            if_in = int(parts[4])
            fstate[(current, dest)] = (next_hop, if_out, if_in)
    return fstate


def parse_bandwidth_state(bw_path: str) -> dict[tuple[int, int], float]:
    """Parse bandwidth state file.
    
    Returns: {(node_id, interface_id): bandwidth_fraction}
    """
    bw_state = {}
    if not os.path.exists(bw_path):
        return bw_state

    with open(bw_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            node_id = int(parts[0])
            if_id = int(parts[1])
            bandwidth = float(parts[2])
            bw_state[(node_id, if_id)] = bandwidth
    return bw_state


def parse_isl_utilization(path: str) -> list[tuple[int, int, int, int, float]]:
    """Parse ns-3 ISL utilization rows.

    Expected format per row: from_node,to_node,interval_start_ns,interval_end_ns,utilization_fraction
    """
    records: list[tuple[int, int, int, int, float]] = []
    if not os.path.exists(path):
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 5:
                continue
            src = int(parts[0])
            dst = int(parts[1])
            start_ns = int(parts[2])
            end_ns = int(parts[3])
            util = float(parts[4])
            records.append((src, dst, start_ns, end_ns, util))
    return records


def build_sat_isl_utilization_at_time(
    time_ns: int,
    num_sats: int,
    isl_records: list[tuple[int, int, int, int, float]],
) -> tuple[dict[int, float], dict[int, float]]:
    """Aggregate ISL utilization per satellite at a given time.

    Returns:
    - mean utilization per satellite over incident ISLs active at time_ns
    - max utilization per satellite over incident ISLs active at time_ns
    """
    sum_util: dict[int, float] = defaultdict(float)
    count_util: dict[int, int] = defaultdict(int)
    max_util: dict[int, float] = defaultdict(float)

    for src, dst, start_ns, end_ns, util in isl_records:
        if not (start_ns <= time_ns < end_ns):
            continue

        if 0 <= src < num_sats:
            sum_util[src] += util
            count_util[src] += 1
            if util > max_util[src]:
                max_util[src] = util

        if 0 <= dst < num_sats:
            sum_util[dst] += util
            count_util[dst] += 1
            if util > max_util[dst]:
                max_util[dst] = util

    mean_util: dict[int, float] = {}
    for sid in range(num_sats):
        if count_util[sid] > 0:
            mean_util[sid] = sum_util[sid] / float(count_util[sid])
        else:
            mean_util[sid] = 0.0
        if sid not in max_util:
            max_util[sid] = 0.0

    return mean_util, max_util


def count_hops_to_destination(
    fstate: dict[tuple[int, int], tuple[int, int, int]],
    current_node: int,
    dest_node: int,
    max_hops: int = 100,
) -> int:
    """Count hops from current node to destination by following fstate."""
    hops = 0
    visited = set()

    while current_node != dest_node and hops < max_hops:
        if current_node in visited:
            return -1  # Cycle detected

        visited.add(current_node)
        key = (current_node, dest_node)

        if key not in fstate:
            return -1  # No path found

        next_hop, _, _ = fstate[key]
        current_node = next_hop
        hops += 1

    return hops if current_node == dest_node else -1


def main() -> int:
    """Main extraction function."""
    # Load configuration
    description_path = os.path.join(SATELLITE_NETWORK_DIR, "description.txt")
    ground_stations_path = os.path.join(SATELLITE_NETWORK_DIR, "ground_stations.txt")
    tles_path = os.path.join(SATELLITE_NETWORK_DIR, "tles.txt")

    if not os.path.isdir(RUN_DIR):
        raise SystemExit(f"Missing run folder: {RUN_DIR}")

    description = parse_description(description_path)
    max_gsl_length_m = description["max_gsl_length_m"]

    ground_stations = read_ground_stations_extended(ground_stations_path)
    tles_data = read_tles(tles_path)
    satellites = tles_data["satellites"]
    epoch = tles_data["epoch"]
    times_ns = load_times(DYNAMIC_STATE_DIR)

    if not times_ns:
        raise SystemExit(f"No fstate files found in {DYNAMIC_STATE_DIR}")

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # Track connection history for stability scores
    connection_history: dict[tuple[int, int], list[int]] = defaultdict(list)
    connection_start_time: dict[tuple[int, int], int] = {}

    # Output file
    features_csv = os.path.join(ANALYSIS_DIR, f"features_{DATA_NAME}.csv")
    num_sats = len(satellites)
    _num_gss = len(ground_stations)
    # NS-3 ISL metrics removed by user request; parser kept for compatibility if needed

    with open(features_csv, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)

        # Header
        header = [
            "time_ns",
            "gs_id",
            "sat_id",
            # Geométricos
            "distance_m",
            "elevation_angle_deg",
            "azimuth_angle_deg",
            "distance_rate_m_per_s",
            # Qualidade de Sinal
            "time_to_loss_s",
            "signal_margin_percent",
            "estimated_snr_db",
            "path_loss_db",
            # Capacidade
            "bandwidth_available_mbps",
            "utilization_percent",
            "num_active_connections",
            "estimated_queue_latency_ms",
            # Topologia
            "hops_to_destination",
            "num_satellite_alternatives",
            "connection_stability_score",
            # Temporal
            "orbit_phase_percent",
            "connection_age_s",
            # Geographical positions
            "gs_latitude_deg",
            "gs_longitude_deg",
            "sat_latitude_deg",
            "sat_longitude_deg",
            "visible",
        ]
        writer.writerow(header)

        # Process each timestep
        prev_distances: dict[tuple[int, int], float] = {}

        for idx, time_ns in enumerate(times_ns):
            time_s = time_ns / 1e9  # Convert to seconds
            time_str = str(epoch + time_ns * u.ns)

            # Load dynamic state for this timestep
            fstate_path = os.path.join(DYNAMIC_STATE_DIR, f"fstate_{time_ns}.txt")
            bw_path = os.path.join(DYNAMIC_STATE_DIR, f"gsl_if_bandwidth_{time_ns}.txt")
            fstate = parse_fstate(fstate_path)
            bw_state = parse_bandwidth_state(bw_path)
            # NS-3 aggregation removed from per-row features

            # Count visible satellites and active connections per GS
            visible_per_gs: dict[int, list[int]] = defaultdict(list)
            connected_sat_per_gs: dict[int, int] = {}

            for gs in ground_stations:
                gid = int(gs["gid"])
                for sid, satellite in enumerate(satellites):
                    distance_m = distance_m_ground_station_to_satellite(
                        gs, satellite, str(epoch), time_str
                    )
                    visible = distance_m <= max_gsl_length_m
                    if visible:
                        visible_per_gs[gid].append(sid)

                connected_sid = infer_connected_satellite(fstate, num_sats + gid, num_sats)
                if connected_sid is not None:
                    connected_sat_per_gs[gid] = connected_sid

            active_connections_per_sat: dict[int, int] = defaultdict(int)
            for connected_sid in connected_sat_per_gs.values():
                active_connections_per_sat[connected_sid] += 1

            # Extract routing information
            for gs in ground_stations:
                gid = int(gs["gid"])
                gs_node_id = num_sats + gid

                for sid in visible_per_gs[gid]:
                    distance_m = distance_m_ground_station_to_satellite(
                        gs, satellites[sid], str(epoch), time_str
                    )
                    visible = distance_m <= max_gsl_length_m

                    if not visible:
                        continue

                    # Geometric features
                    try:
                        (
                            distance_m,
                            elevation_deg,
                            azimuth_deg,
                            distance_rate,
                            sat_lat_deg,
                            sat_lon_deg,
                        ) = calculate_link_geometry(gs, satellites[sid], str(epoch), time_str)
                    except Exception:
                        elevation_deg, azimuth_deg = 0.0, 0.0
                        distance_rate = 0.0
                        sat_lat_deg = 0.0
                        sat_lon_deg = 0.0

                    # Distance rate fallback (derivative)
                    pair = (gid, sid)
                    if abs(distance_rate) < 1e-12 and pair in prev_distances:
                        prev_dist = prev_distances[pair]
                        dist_diff = distance_m - prev_dist
                        if idx > 0:
                            time_diff = (times_ns[idx] - times_ns[idx - 1]) / 1e9
                            distance_rate = dist_diff / time_diff if time_diff > 0 else 0.0
                        else:
                            distance_rate = 0.0
                    prev_distances[pair] = distance_m

                    # Signal quality
                    path_loss_db = calculate_path_loss_friis(distance_m)
                    snr_db = calculate_snr(path_loss_db)
                    signal_margin_percent = 100 * (1 - distance_m / max_gsl_length_m)
                    time_to_loss = calculate_time_to_loss(
                        distance_m, distance_rate, max_gsl_length_m
                    )

                    # Bandwidth (from gsl_if_bandwidth, assuming interface 0 for GS)
                    bandwidth_key = (gs_node_id, 0)
                    bw_fraction = bw_state.get(bandwidth_key, 1.0)
                    bandwidth_mbps = 10.0 * bw_fraction  # 10 Mbps is the configured rate

                    is_connected = connected_sat_per_gs.get(gid) == sid
                    if is_connected:
                        if pair not in connection_start_time:
                            connection_start_time[pair] = time_ns
                        connection_history[pair].append(time_ns)

                    # Utilization (placeholder: assume equal distribution)
                    if active_connections_per_sat[sid] > 0:
                        utilization_percent = 100.0 / (
                            active_connections_per_sat[sid] + 1
                        )  # +1 for fairness
                    else:
                        utilization_percent = 0.0

                    # Queue latency (simplified: proxy from utilization)
                    queue_latency_ms = utilization_percent * 10  # Rough estimate

                    # NS-3 ISL metrics removed from output

                    # Topology
                    num_alternatives = len(visible_per_gs[gid])
                    hops = count_hops_to_destination(fstate, sid, gs_node_id)
                    if hops < 0:
                        hops = 0

                    # Connection stability
                    if pair in connection_history and len(connection_history[pair]) > 1:
                        # Simple metric: how many consecutive timesteps connected
                        stability_score = min(
                            1.0, len(connection_history[pair]) / max(1, idx)
                        )
                    else:
                        stability_score = 0.0

                    # Temporal
                    connection_age = (
                        (time_ns - connection_start_time[pair]) / 1e9
                        if pair in connection_start_time
                        else 0.0
                    )

                    orbit_phase = compute_orbit_phase_percent(satellites[sid])

                    # Ground station lat/lon (safe coercion)
                    try:
                        gs_lat_deg = float(gs.get("latitude_degrees", gs.get("latitude_degrees_str", 0.0)))
                    except Exception:
                        gs_lat_deg = 0.0
                    try:
                        gs_lon_deg = float(gs.get("longitude_degrees", gs.get("longitude_degrees_str", 0.0)))
                    except Exception:
                        gs_lon_deg = 0.0

                    # Write row
                    writer.writerow([
                        time_ns,
                        gid,
                        sid,
                        distance_m,
                        elevation_deg,
                        azimuth_deg,
                        distance_rate,
                        time_to_loss,
                        signal_margin_percent,
                        snr_db,
                        path_loss_db,
                        bandwidth_mbps,
                        utilization_percent,
                        active_connections_per_sat.get(sid, 0),
                        queue_latency_ms,
                        hops,
                        num_alternatives,
                        stability_score,
                        orbit_phase,
                        connection_age,
                        gs_lat_deg,
                        gs_lon_deg,
                        sat_lat_deg,
                        sat_lon_deg,
                        int(visible),
                    ])
    print(f"✓ Extracted features to: {features_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
