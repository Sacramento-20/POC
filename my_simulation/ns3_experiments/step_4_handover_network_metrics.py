#!/usr/bin/env python3
"""Compute handover/network paper metrics from a prepared ns-3 run folder."""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass


INDEX = 0

SIMULACAO = 'simulacao_20_minutes_5s_'

STATE = ['dynamic_state_5000ms_for_1200s', 
         'dynamic_state_5000ms_for_1200s_rl']


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_NAME = SIMULACAO + STATE[1]
DYNAMIC_STATE = STATE[1]
PING_WINDOW_NS = 5_000_000_000

RUN_DIR = os.path.join(SCRIPT_DIR, "runs", RUN_NAME)
DYNAMIC_STATE_DIR = os.path.join(RUN_DIR, DYNAMIC_STATE)
SATELLITE_NETWORK_DIR = os.path.join(RUN_DIR, "satellite_network_state")
LOGS_DIR = os.path.join(RUN_DIR, "logs_ns3")
ANALYSIS_DIR = os.path.join(RUN_DIR, "analysis")


@dataclass
class HandoverEvent:
    gs_id: int
    time_ns: int
    old_sat: int
    new_sat: int


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    values_sorted = sorted(values)
    rank = (p / 100.0) * (len(values_sorted) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(values_sorted[lo])
    frac = rank - lo
    return float(values_sorted[lo] * (1.0 - frac) + values_sorted[hi] * frac)


def read_num_satellites(tles_path: str) -> int:
    with open(tles_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    parts = first_line.split()
    if len(parts) != 2:
        raise ValueError(f"Invalid TLE header: {first_line}")
    return int(parts[0]) * int(parts[1])


def parse_time_ns(path: str) -> int:
    base = os.path.basename(path)
    left = base.split(".", 1)[0]
    return int(left.rsplit("_", 1)[1])


def load_fstate_paths(dynamic_state_dir: str) -> list[tuple[int, str]]:
    paths = glob.glob(os.path.join(dynamic_state_dir, "fstate_*.txt"))
    entries = [(parse_time_ns(p), p) for p in paths]
    entries.sort(key=lambda x: x[0])
    return entries


def parse_connected_sat_per_gs(fstate_path: str, num_sats: int) -> dict[int, int]:
    connected: dict[int, int] = {}
    with open(fstate_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 5:
                continue
            current = int(parts[0])
            next_hop = int(parts[2])
            if current < num_sats:
                continue
            gs_id = current - num_sats
            sat = next_hop if 0 <= next_hop < num_sats else -1
            if gs_id not in connected:
                connected[gs_id] = sat
    return connected


def detect_handover_events(entries: list[tuple[int, str]], num_sats: int) -> tuple[list[HandoverEvent], dict[int, int], list[float]]:
    events: list[HandoverEvent] = []
    handovers_per_gs: dict[int, int] = defaultdict(int)
    interruptions_s: list[float] = []

    previous_sat: dict[int, int] = {}
    last_disconnect_time: dict[int, int] = {}

    for time_ns, fstate_path in entries:
        now_map = parse_connected_sat_per_gs(fstate_path, num_sats)
        all_gs = set(previous_sat.keys()) | set(now_map.keys())

        for gs_id in all_gs:
            old_sat = previous_sat.get(gs_id, -1)
            new_sat = now_map.get(gs_id, -1)
            if old_sat == new_sat:
                previous_sat[gs_id] = new_sat
                continue

            events.append(HandoverEvent(gs_id=gs_id, time_ns=time_ns, old_sat=old_sat, new_sat=new_sat))
            if old_sat >= 0 and new_sat >= 0:
                handovers_per_gs[gs_id] += 1

            if old_sat >= 0 and new_sat == -1:
                last_disconnect_time[gs_id] = time_ns
            elif old_sat == -1 and new_sat >= 0 and gs_id in last_disconnect_time:
                dt_s = (time_ns - last_disconnect_time[gs_id]) / 1e9
                interruptions_s.append(max(0.0, dt_s))
                del last_disconnect_time[gs_id]

            previous_sat[gs_id] = new_sat

    return events, dict(handovers_per_gs), interruptions_s


def parse_pingmesh(path: str) -> list[tuple[int, int, int, int, int, int, int, int, int, bool]]:
    rows: list[tuple[int, int, int, int, int, int, int, int, int, bool]] = []
    if not os.path.exists(path):
        return rows

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 10:
                continue
            try:
                from_id = int(row[0])
                to_id = int(row[1])
                i = int(row[2])
                send_ts = int(row[3])
                reply_ts = int(row[4])
                recv_ts = int(row[5])
                lat_there = int(row[6])
                lat_back = int(row[7])
                rtt_ns = int(row[8])
                ok = row[9].strip().upper() == "YES"
            except ValueError:
                continue
            rows.append((from_id, to_id, i, send_ts, reply_ts, recv_ts, lat_there, lat_back, rtt_ns, ok))
    return rows


def parse_tcp_flows(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        result = []
        for row in reader:
            if len(row) < 10:
                continue
            result.append(
                {
                    "id": row[0],
                    "from": row[1],
                    "to": row[2],
                    "size_byte": row[3],
                    "start_time_ns": row[4],
                    "end_time_ns": row[5],
                    "duration_ns": row[6],
                    "amount_sent_byte": row[7],
                    "finished": row[8],
                    "metadata": row[9],
                }
            )
        return result


def parse_udp_bursts(path: str) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    if not os.path.exists(path):
        return result

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 12:
                continue
            try:
                bid = int(row[0])
                packets = float(row[8])
                data_payload = float(row[10])
            except ValueError:
                continue
            result[bid] = {"packets": packets, "payload": data_payload}
    return result


def parse_isl_utilization(path: str) -> list[tuple[int, int, int, int, float]]:
    rows: list[tuple[int, int, int, int, float]] = []
    if not os.path.exists(path):
        return rows

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 5:
                continue
            try:
                src = int(row[0])
                dst = int(row[1])
                start_ns = int(row[2])
                end_ns = int(row[3])
                util = float(row[4])
            except ValueError:
                continue
            rows.append((src, dst, start_ns, end_ns, util))
    return rows


def summarize_ping(
    ping_rows: list[tuple[int, int, int, int, int, int, int, int, int, bool]],
    event_times_ns: list[int],
    num_sats: int,
) -> dict[str, float]:
    if not ping_rows:
        return {
            "total_samples": 0,
            "loss_rate": 0.0,
            "rtt_median_ms": 0.0,
            "rtt_p95_ms": 0.0,
            "loss_rate_around_handover": 0.0,
            "rtt_p95_ms_around_handover": 0.0,
        }

    event_times_ns_sorted = sorted(event_times_ns)

    def near_handover(send_ts: int, from_node: int, to_node: int) -> bool:
        if not event_times_ns_sorted:
            return False
        if from_node < num_sats or to_node < num_sats:
            return False
        for t in event_times_ns_sorted:
            if abs(send_ts - t) <= PING_WINDOW_NS:
                return True
        return False

    ok_rows = [r for r in ping_rows if r[9]]
    rtt_ms = [r[8] / 1e6 for r in ok_rows if r[8] >= 0]

    around = [r for r in ping_rows if near_handover(r[3], r[0], r[1])]
    around_ok = [r for r in around if r[9]]
    around_rtt_ms = [r[8] / 1e6 for r in around_ok if r[8] >= 0]

    loss_rate = 1.0 - (len(ok_rows) / float(len(ping_rows)))
    loss_rate_around = 0.0
    if around:
        loss_rate_around = 1.0 - (len(around_ok) / float(len(around)))

    return {
        "total_samples": len(ping_rows),
        "loss_rate": loss_rate,
        "rtt_median_ms": statistics.median(rtt_ms) if rtt_ms else 0.0,
        "rtt_p95_ms": percentile(rtt_ms, 95.0) if rtt_ms else 0.0,
        "loss_rate_around_handover": loss_rate_around,
        "rtt_p95_ms_around_handover": percentile(around_rtt_ms, 95.0) if around_rtt_ms else 0.0,
    }


def summarize_tcp(flows: list[dict[str, str]]) -> dict[str, float]:
    if not flows:
        return {
            "total_flows": 0,
            "completion_rate": 0.0,
            "throughput_mbps_mean": 0.0,
            "throughput_mbps_p95": 0.0,
            "fct_s_median": 0.0,
            "fct_s_p95": 0.0,
        }

    completed = [f for f in flows if f["finished"] == "YES"]
    throughputs = []
    fcts_s = []
    for flow in completed:
        try:
            duration_ns = float(flow["duration_ns"])
            sent_b = float(flow["amount_sent_byte"])
        except ValueError:
            continue
        if duration_ns <= 0:
            continue
        throughputs.append((sent_b * 8.0) / (duration_ns / 1e9) / 1e6)
        fcts_s.append(duration_ns / 1e9)

    return {
        "total_flows": len(flows),
        "completion_rate": len(completed) / float(len(flows)),
        "throughput_mbps_mean": statistics.mean(throughputs) if throughputs else 0.0,
        "throughput_mbps_p95": percentile(throughputs, 95.0) if throughputs else 0.0,
        "fct_s_median": statistics.median(fcts_s) if fcts_s else 0.0,
        "fct_s_p95": percentile(fcts_s, 95.0) if fcts_s else 0.0,
    }


def summarize_udp(outgoing: dict[int, dict[str, float]], incoming: dict[int, dict[str, float]]) -> dict[str, float]:
    if not outgoing:
        return {
            "total_bursts": 0,
            "delivery_ratio_packets": 0.0,
            "delivery_ratio_payload_bytes": 0.0,
        }

    sent_packets = 0.0
    recv_packets = 0.0
    sent_payload = 0.0
    recv_payload = 0.0

    for bid, sent in outgoing.items():
        recv = incoming.get(bid, {"packets": 0.0, "payload": 0.0})
        sent_packets += sent["packets"]
        recv_packets += recv["packets"]
        sent_payload += sent["payload"]
        recv_payload += recv["payload"]

    return {
        "total_bursts": len(outgoing),
        "delivery_ratio_packets": (recv_packets / sent_packets) if sent_packets > 0 else 0.0,
        "delivery_ratio_payload_bytes": (recv_payload / sent_payload) if sent_payload > 0 else 0.0,
    }


def summarize_isl(isl_rows: list[tuple[int, int, int, int, float]]) -> dict[str, float]:
    if not isl_rows:
        return {
            "samples": 0,
            "util_mean": 0.0,
            "util_p95": 0.0,
            "hotspot_fraction_over_0_8": 0.0,
        }

    values: list[float] = []
    weighted_sum = 0.0
    weighted_time = 0.0
    hotspot_time = 0.0

    for _src, _dst, start_ns, end_ns, util in isl_rows:
        dt = max(0, end_ns - start_ns)
        if dt <= 0:
            continue
        values.append(util)
        weighted_sum += util * dt
        weighted_time += dt
        if util > 0.8:
            hotspot_time += dt

    mean_util = (weighted_sum / weighted_time) if weighted_time > 0 else 0.0
    hotspot_fraction = (hotspot_time / weighted_time) if weighted_time > 0 else 0.0

    return {
        "samples": len(values),
        "util_mean": mean_util,
        "util_p95": percentile(values, 95.0) if values else 0.0,
        "hotspot_fraction_over_0_8": hotspot_fraction,
    }


def main() -> int:
    if not os.path.isdir(RUN_DIR):
        raise SystemExit(f"Missing run dir: {RUN_DIR}")
    if not os.path.isdir(DYNAMIC_STATE_DIR):
        raise SystemExit(f"Missing dynamic state dir: {DYNAMIC_STATE_DIR}")

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    tles_path = os.path.join(SATELLITE_NETWORK_DIR, "tles.txt")
    num_sats = read_num_satellites(tles_path)

    entries = load_fstate_paths(DYNAMIC_STATE_DIR)
    if not entries:
        raise SystemExit(f"No fstate files in {DYNAMIC_STATE_DIR}")

    events, handovers_per_gs, interruptions_s = detect_handover_events(entries, num_sats)
    time_span_s = (entries[-1][0] - entries[0][0]) / 1e9 if len(entries) > 1 else 0.0

    ping_rows = parse_pingmesh(os.path.join(LOGS_DIR, "pingmesh.csv"))
    tcp_rows = parse_tcp_flows(os.path.join(LOGS_DIR, "tcp_flows.csv"))
    udp_out = parse_udp_bursts(os.path.join(LOGS_DIR, "udp_bursts_outgoing.csv"))
    udp_in = parse_udp_bursts(os.path.join(LOGS_DIR, "udp_bursts_incoming.csv"))
    isl_rows = parse_isl_utilization(os.path.join(LOGS_DIR, "isl_utilization.csv"))

    ping_summary = summarize_ping(ping_rows, [e.time_ns for e in events], num_sats)
    tcp_summary = summarize_tcp(tcp_rows)
    udp_summary = summarize_udp(udp_out, udp_in)
    isl_summary = summarize_isl(isl_rows)

    handover_total = sum(handovers_per_gs.values())
    avg_handover_rate_per_hour = 0.0
    if time_span_s > 0 and handovers_per_gs:
        per_gs_rates = [count / (time_span_s / 3600.0) for count in handovers_per_gs.values()]
        avg_handover_rate_per_hour = statistics.mean(per_gs_rates)

    interruption_median_s = statistics.median(interruptions_s) if interruptions_s else 0.0
    interruption_p95_s = percentile(interruptions_s, 95.0) if interruptions_s else 0.0

    summary = {
        "run_name": RUN_NAME,
        "dynamic_state": DYNAMIC_STATE,
        "time_span_s": time_span_s,
        "handover": {
            "handover_events_total": handover_total,
            "num_gs_with_handover": len(handovers_per_gs),
            "avg_handover_rate_per_hour": avg_handover_rate_per_hour,
            "interruptions_count": len(interruptions_s),
            "interruption_median_s": interruption_median_s,
            "interruption_p95_s": interruption_p95_s,
        },
        "pingmesh": ping_summary,
        "tcp_flows": tcp_summary,
        "udp_bursts": udp_summary,
        "isl": isl_summary,
    }

    summary_json = os.path.join(ANALYSIS_DIR, f"handover_network_metrics_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    events_csv = os.path.join(ANALYSIS_DIR, "handover_events.csv")
    with open(events_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gs_id", "time_ns", "old_sat", "new_sat"]) 
        for e in events:
            writer.writerow([e.gs_id, e.time_ns, e.old_sat, e.new_sat])

    print(f"Saved: {summary_json}")
    print(f"Saved: {events_csv}")
    print("Handover + network summary:")
    print(f"  handover_events_total: {summary['handover']['handover_events_total']}")
    print(f"  interruption_p95_s: {summary['handover']['interruption_p95_s']:.3f}")
    print(f"  ping_loss_rate: {summary['pingmesh']['loss_rate']:.4f}")
    print(f"  tcp_completion_rate: {summary['tcp_flows']['completion_rate']:.4f}")
    print(f"  udp_delivery_ratio_packets: {summary['udp_bursts']['delivery_ratio_packets']:.4f}")
    print(f"  isl_util_mean: {summary['isl']['util_mean']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
