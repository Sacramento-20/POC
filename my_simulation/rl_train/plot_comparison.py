import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "ns3_experiments", "runs"))



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_NAME = 'simulacao_20_minutes_5s_dynamic_state_5000ms_for_1200s'
RUN_NAME_RL = 'simulacao_20_minutes_5s_dynamic_state_5000ms_for_1200s_rl'

# Paths to the analysis directories for both runs
HEURISTIC_DIR = os.path.join(RUNS_DIR, RUN_NAME, "analysis")
RL_DIR = os.path.join(RUNS_DIR, RUN_NAME_RL, "analysis")

def load_json_summary(path):
    if not os.path.exists(path):
        print(f"Warning: Summary file not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)

def load_handover_events(path):
    if not os.path.exists(path):
        print(f"Warning: Handover events file not found: {path}")
        return None
    return pd.read_csv(path)

def plot_handover_and_disconnects(h_data, rl_data, save_path):
    """Generates a bar chart comparing total handovers and disconnection events."""
    if not h_data or not rl_data:
        return
        
    h_ho = h_data["handover"]["handover_events_total"]
    rl_ho = rl_data["handover"]["handover_events_total"]
    
    h_disc = h_data["handover"]["interruptions_count"]
    rl_disc = rl_data["handover"]["interruptions_count"]
    
    labels = ['Total Handovers', 'Total Disconnections']
    heuristic_vals = [h_ho, h_disc]
    rl_vals = [rl_ho, rl_disc]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, heuristic_vals, width, label='Heuristic (Closest Sat)', color='#377eb8')
    rects2 = ax.bar(x + width/2, rl_vals, width, label='RL Agent (Trained Policy)', color='#4daf4a')
    
    ax.set_ylabel('Event Count')
    ax.set_title('Network Stability Comparison: Heuristic vs RL')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved bar plot to: {save_path}")

def plot_latency_comparison(h_data, rl_data, save_path):
    """Generates a bar chart comparing latency metrics (median, p95, and around handover)."""
    if not h_data or not rl_data:
        return
        
    metrics = ['RTT Median', 'RTT P95', 'RTT P95 (Around Handover)']
    h_vals = [
        h_data["pingmesh"]["rtt_median_ms"],
        h_data["pingmesh"]["rtt_p95_ms"],
        h_data["pingmesh"]["rtt_p95_ms_around_handover"]
    ]
    rl_vals = [
        rl_data["pingmesh"]["rtt_median_ms"],
        rl_data["pingmesh"]["rtt_p95_ms"],
        rl_data["pingmesh"]["rtt_p95_ms_around_handover"]
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, h_vals, width, label='Heuristic (Closest Sat)', color='#377eb8')
    rects2 = ax.bar(x + width/2, rl_vals, width, label='RL Agent (Trained Policy)', color='#4daf4a')
    
    ax.set_ylabel('Latency (ms)')
    ax.set_title('RTT Latency Comparison: Heuristic vs RL')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}ms',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved latency plot to: {save_path}")

def main():
    print("Reading simulation metrics summaries...")
    h_summary = load_json_summary(os.path.join(HEURISTIC_DIR, "handover_network_metrics_summary.json"))
    rl_summary = load_json_summary(os.path.join(RL_DIR, "handover_network_metrics_summary.json"))
    
    if not h_summary or not rl_summary:
        print("\nError: Could not load summary files. Make sure you have run the step_4 script for both simulations:")
        print(f"  Heuristic Summary Path: {os.path.join(HEURISTIC_DIR, 'handover_network_metrics_summary.json')}")
        print(f"  RL Summary Path: {os.path.join(RL_DIR, 'handover_network_metrics_summary.json')}")
        return
        
    os.makedirs(os.path.join(SCRIPT_DIR, "plots"), exist_ok=True)
    
    # Generate stability comparison plot
    plot_handover_and_disconnects(
        h_summary, 
        rl_summary, 
        os.path.join(SCRIPT_DIR, "plots", "stability_comparison.png")
    )
    
    # Generate latency comparison plot
    plot_latency_comparison(
        h_summary, 
        rl_summary, 
        os.path.join(SCRIPT_DIR, "plots", "latency_comparison.png")
    )
    
    print("\nComparison plots generated successfully inside 'rl_train/plots/' directory!")

if __name__ == "__main__":
    main()
