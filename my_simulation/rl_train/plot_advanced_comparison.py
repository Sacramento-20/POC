import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "ns3_experiments", "runs"))


# Paths to the raw log directories for both runs
HEURISTIC_LOGS = os.path.join(RUNS_DIR, "simulacao_20_minutes_5s_dynamic_state_5000ms_for_1200s", "logs_ns3")
RL_LOGS = os.path.join(RUNS_DIR, "simulacao_20_minutes_5s_dynamic_state_5000ms_for_1200s_rl", "logs_ns3")
HEURISTIC_ANALYSIS = os.path.join(RUNS_DIR, "simulacao_20_minutes_5s_dynamic_state_5000ms_for_1200s", "analysis")
RL_ANALYSIS = os.path.join(RUNS_DIR, "simulacao_20_minutes_5s_dynamic_state_5000ms_for_1200s_rl", "analysis")

def parse_rtts(ping_csv_path):
    """Loads RTT samples in ms for successful pings."""
    if not os.path.exists(ping_csv_path):
        return []
    df = pd.read_csv(ping_csv_path, header=None)
    # Column 8 is RTT in ns, column 9 is 'YES'/'NO' for success
    successful_pings = df[df[9].str.strip().str.upper() == 'YES']
    rtts_ms = successful_pings[8].astype(float) / 1e6
    return rtts_ms.tolist()

def parse_tcp_metrics(tcp_csv_path):
    """Loads FCT (seconds) and Throughput (Mbps) for finished flows."""
    if not os.path.exists(tcp_csv_path):
        return [], []
    df = pd.read_csv(tcp_csv_path, header=None)
    # col 6 is duration_ns, col 7 is amount_sent_byte, col 8 is finished ('YES'/'NO')
    finished_flows = df[df[8].str.strip().str.upper() == 'YES']
    
    durations_s = finished_flows[6].astype(float) / 1e9
    sent_bytes = finished_flows[7].astype(float)
    
    fct = durations_s.tolist()
    # Throughput (Mbps) = (bytes * 8) / (duration_s) / 1e6
    throughputs = ((sent_bytes * 8.0) / durations_s / 1e6).tolist()
    
    return fct, throughputs

def parse_handover_times_s(events_csv_path):
    """Loads handover timestamps converted to seconds."""
    if not os.path.exists(events_csv_path):
        return []
    df = pd.read_csv(events_csv_path)
    # columns: gs_id, time_ns, old_sat, new_sat
    times_s = df['time_ns'].astype(float) / 1e9
    return times_s.tolist()

def plot_cdf(data_h, data_rl, xlabel, title, save_path):
    """Plots a Cumulative Distribution Function (CDF) comparing Heuristic vs RL."""
    if len(data_h) == 0 or len(data_rl) == 0:
        print(f"Skipping CDF {title} due to missing data.")
        return

    plt.figure(figsize=(8, 5))
    
    # Heuristic CDF
    sorted_h = np.sort(data_h)
    y_h = np.arange(1, len(sorted_h) + 1) / len(sorted_h)
    plt.plot(sorted_h, y_h, label='Heuristic (Closest Sat)', color='#377eb8', linewidth=2)
    
    # RL CDF
    sorted_rl = np.sort(data_rl)
    y_rl = np.arange(1, len(sorted_rl) + 1) / len(sorted_rl)
    plt.plot(sorted_rl, y_rl, label='RL Agent (Trained Policy)', color='#4daf4a', linewidth=2)
    
    plt.xlabel(xlabel)
    plt.ylabel('CDF (Cumulative Probability)')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved CDF plot to: {save_path}")

def plot_throughput_boxplot(tp_h, tp_rl, save_path):
    """Plots side-by-side boxplots of TCP throughput."""
    if len(tp_h) == 0 or len(tp_rl) == 0:
        print("Skipping Throughput Boxplot due to missing data.")
        return

    plt.figure(figsize=(7, 5))
    plt.boxplot([tp_h, tp_rl], labels=['Heuristic', 'RL Agent'], patch_artist=True,
                boxprops=dict(facecolor='#e0e0e0', color='#333333'),
                medianprops=dict(color='red', linewidth=1.5))
    
    plt.ylabel('Throughput (Mbps)')
    plt.title('TCP Throughput Distribution Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Throughput boxplot to: {save_path}")

def plot_handover_density(times_h, times_rl, duration_s, save_path):
    """Plots handover events occurrence density over time (binned by 60s windows)."""
    if len(times_h) == 0 and len(times_rl) == 0:
        print("Skipping Handover Density due to missing data.")
        return
        
    bin_width = 60 # 1 minute bins
    bins = np.arange(0, duration_s + bin_width, bin_width)
    
    counts_h, _ = np.histogram(times_h, bins=bins)
    counts_rl, _ = np.histogram(times_rl, bins=bins)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2.0 / 60.0 # to minutes
    
    plt.figure(figsize=(10, 5))
    plt.plot(bin_centers, counts_h, marker='o', label='Heuristic (Closest Sat)', color='#377eb8', linewidth=2)
    plt.plot(bin_centers, counts_rl, marker='s', label='RL Agent (Trained Policy)', color='#4daf4a', linewidth=2)
    
    plt.xlabel('Simulation Time (Minutes)')
    plt.ylabel('Handovers per Minute')
    plt.title('Handover Frequency/Storms Over Time')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Handover density plot to: {save_path}")

def main():
    print("Loading raw network simulation traces...")
    
    # 1. Load Ping RTT
    rtts_h = parse_rtts(os.path.join(HEURISTIC_LOGS, "pingmesh.csv"))
    rtts_rl = parse_rtts(os.path.join(RL_LOGS, "pingmesh.csv"))
    
    # 2. Load TCP flows FCT & Throughput
    fct_h, tp_h = parse_tcp_metrics(os.path.join(HEURISTIC_LOGS, "tcp_flows.csv"))
    fct_rl, tp_rl = parse_tcp_metrics(os.path.join(RL_LOGS, "tcp_flows.csv"))
    
    # 3. Load Handover Event timings
    ho_times_h = parse_handover_times_s(os.path.join(HEURISTIC_ANALYSIS, "handover_events.csv"))
    ho_times_rl = parse_handover_times_s(os.path.join(RL_ANALYSIS, "handover_events.csv"))
    
    plots_dir = os.path.join(SCRIPT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate CDF of Latency RTT
    plot_cdf(
        rtts_h, rtts_rl, 
        xlabel="RTT Latency (ms)", 
        title="CDF of Ping Latency (RTT)", 
        save_path=os.path.join(plots_dir, "rtt_cdf_comparison.png")
    )
    
    # Generate CDF of Flow Completion Time (FCT)
    plot_cdf(
        fct_h, fct_rl, 
        xlabel="Flow Completion Time (s)", 
        title="CDF of TCP Flow Completion Time (FCT)", 
        save_path=os.path.join(plots_dir, "fct_cdf_comparison.png")
    )
    
    # Generate Throughput Boxplot
    plot_throughput_boxplot(
        tp_h, tp_rl, 
        save_path=os.path.join(plots_dir, "tcp_throughput_boxplot.png")
    )
    
    # Generate Handover Storms Over Time
    plot_handover_density(
        ho_times_h, ho_times_rl, 
        duration_s=1800.0, 
        save_path=os.path.join(plots_dir, "handover_density_over_time.png")
    )
    
    print("\nAdvanced comparison plots generated successfully inside 'rl_train/plots/' directory!")

if __name__ == "__main__":
    main()
