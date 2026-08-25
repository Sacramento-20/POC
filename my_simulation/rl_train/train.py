import os
import sys
import argparse
import numpy as np
import networkx as nx
from astropy import units as u

# Add satgenpy path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "satgenpy"))

from satellite_env import SatelliteHandoverEnv
from policy_gradient_agent import PolicyGradientAgent
from satgen.dynamic_state.fstate_calculation import calculate_fstate_shortest_path_without_gs_relaying
from satgen.distance_tools import distance_m_between_satellites

def train_agent(env: SatelliteHandoverEnv, agent: PolicyGradientAgent, num_episodes: int = 50):
    print("\n" + "=" * 80)
    print("STARTING RL AGENT TRAINING (REINFORCE - Policy Gradient)")
    print("=" * 80)
    
    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            actions = {}
            for gs_id in range(len(env.ground_stations)):
                actions[gs_id] = agent.select_action(gs_id, obs[gs_id])
                
            next_obs, rewards, done, _ = env.step(actions)
            
            for gs_id, r in rewards.items():
                agent.store_reward(gs_id, r)
                episode_reward += r
                
            obs = next_obs
            step_count += 1
            
        # Update policy weights at the end of the episode
        avg_reward = agent.update()
        
        # Calculate metric values
        total_transitions = step_count * len(env.ground_stations)
        ep_avg_reward = episode_reward / total_transitions
        print(f"Episode {ep:02d}/{num_episodes:02d} | Step count: {step_count} | Average Step Reward: {ep_avg_reward:.4f}")
        
    print("\nTraining completed successfully!")

def generate_rl_dynamic_state(env: SatelliteHandoverEnv, agent: PolicyGradientAgent, output_dir: str):
    """
    Runs inference using the trained RL agent and generates the optimized
    routing tables and GSL interface bandwidth states.
    """
    print("\n" + "=" * 80)
    print(f"GENERATING OPTIMIZED ROUTES TO: {output_dir}")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    obs = env.reset()
    done = False
    step = 0
    prev_output = None
    
    num_satellites = len(env.satellites)
    num_ground_stations = len(env.ground_stations)
    
    # Pre-build satellite ISL topology to write bandwidth files correctly
    num_isls_per_sat = [0] * num_satellites
    sat_neighbor_to_if = {}
    for (a, b) in env.isls:
        sat_neighbor_to_if[(a, b)] = num_isls_per_sat[a]
        sat_neighbor_to_if[(b, a)] = num_isls_per_sat[b]
        num_isls_per_sat[a] += 1
        num_isls_per_sat[b] += 1
        
    # GID to satellite GSL interface index (matching standard Hypatia representation)
    gid_to_sat_gsl_if_idx = list(range(num_ground_stations))
    
    while not done:
        current_time = env._get_time_at_step(step)
        time_since_epoch_ns = step * env.time_step_ns
        
        print(f"Step {step+1}/{env.max_steps} | Generating for t = {time_since_epoch_ns / 1e9:.2f} s")
        
        # 1. Deterministic action selection (argmax of probabilities with action masking)
        actions = {}
        for gs_id in range(num_ground_stations):
            obs_gs = obs[gs_id]
            mask_gs = np.zeros(agent.action_dim, dtype=np.float32)
            for i in range(agent.action_dim):
                if obs_gs[i * 8] > 0:
                    mask_gs[i] = 1.0
            if np.sum(mask_gs) == 0:
                mask_gs[0] = 1.0
                
            _, _, probs = agent.forward(obs_gs, mask_gs)
            actions[gs_id] = int(np.argmax(probs[0]))
            
        # 2. Step the env to apply actions and get candidates
        next_obs, _, done, _ = env.step(actions)
        
        # 3. Build sat net graph for this step (weights are distances)
        sat_net_graph_without_gs = nx.Graph()
        for i in range(num_satellites):
            sat_net_graph_without_gs.add_node(i)
        for (a, b) in env.isls:
            dist = distance_m_between_satellites(env.satellites[a], env.satellites[b], str(env.epoch), str(current_time))
            sat_net_graph_without_gs.add_edge(a, b, weight=dist)
            
        # 4. Map the selected satellites to ground_station_satellites_in_range_select_one_at_most
        satellite_gsl_ifs_paired = [[] for _ in range(num_satellites)]
        ground_station_satellites_in_range_select_one_at_most = []
        
        for gid in range(num_ground_stations):
            chosen_sid = env.current_satellites[gid]
            if chosen_sid == -1:
                ground_station_satellites_in_range_select_one_at_most.append([])
            else:
                # Find distance to chosen satellite
                candidates = env._get_candidates(gid)
                dist_m = next((c["distance_m"] for c in candidates if c["sid"] == chosen_sid), env.max_gsl_length_m)
                ground_station_satellites_in_range_select_one_at_most.append([(dist_m, chosen_sid)])
                satellite_gsl_ifs_paired[chosen_sid].append(gid)
                
        # 5. Determine the GSL interface bandwidth state
        gsl_if_bandwidth_state = {}
        for sid in range(num_satellites):
            satellite_frequency_chosen = len(satellite_gsl_ifs_paired[sid])
            for gsl_if_idx in range(num_ground_stations):
                if gsl_if_idx in satellite_gsl_ifs_paired[sid]:
                    gsl_if_bandwidth_state[(sid, num_isls_per_sat[sid] + gsl_if_idx)] = (
                        1.0 / float(satellite_frequency_chosen)
                    )
                else:
                    gsl_if_bandwidth_state[(sid, num_isls_per_sat[sid] + gsl_if_idx)] = 1.0
                    
        for gid in range(num_ground_stations):
            if len(ground_station_satellites_in_range_select_one_at_most[gid]) == 1:
                paired_satellite_id = ground_station_satellites_in_range_select_one_at_most[gid][0][1]
                satellite_frequency_chosen = len(satellite_gsl_ifs_paired[paired_satellite_id])
                gsl_if_bandwidth_state[(num_satellites + gid, 0)] = 1.0 / float(satellite_frequency_chosen)
            else:
                gsl_if_bandwidth_state[(num_satellites + gid, 0)] = 1.0
                
        # Write GSL interface bandwidth state
        prev_gsl_if_bandwidth_state = prev_output["gsl_if_bandwidth_state"] if prev_output else None
        bandwidth_file = os.path.join(output_dir, f"gsl_if_bandwidth_{time_since_epoch_ns}.txt")
        with open(bandwidth_file, "w+") as f_out:
            for (node_id, if_id) in gsl_if_bandwidth_state:
                if (
                    prev_gsl_if_bandwidth_state is None
                    or prev_gsl_if_bandwidth_state[(node_id, if_id)] != gsl_if_bandwidth_state[(node_id, if_id)]
                ):
                    f_out.write(f"{node_id},{if_id},{gsl_if_bandwidth_state[(node_id, if_id)]:.6f}\n")
                    
        # 6. Calculate forwarding state and write fstate file
        prev_fstate = prev_output["fstate"] if prev_output else None
        fstate = calculate_fstate_shortest_path_without_gs_relaying(
            output_dir,
            time_since_epoch_ns,
            num_satellites,
            num_ground_stations,
            sat_net_graph_without_gs,
            num_isls_per_sat,
            gid_to_sat_gsl_if_idx,
            ground_station_satellites_in_range_select_one_at_most,
            sat_neighbor_to_if,
            prev_fstate,
            False # enable_verbose_logs
        )
        
        # Save output state for delta comparison
        prev_output = {
            "fstate": fstate,
            "gsl_if_bandwidth_state": gsl_if_bandwidth_state
        }
        
        obs = next_obs
        step += 1
        
    print("\nDynamic State files generated successfully!")

def main():
    parser = argparse.ArgumentParser(description="Train RL Handover Agent on Hypatia Constellation Layout.")
    parser.add_argument("--data_dir", type=str, 
                        default=os.path.join(REPO_ROOT, "my_simulation", "gen_data", "simulacao_20_minutes_5s"),
                        help="Path to the generated constellation directory.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Path to write the optimized RL dynamic state. Defaults to a subfolder inside data_dir.")
    parser.add_argument("--episodes", type=int, default=30, help="Number of training episodes.") # Mudar Previous: 40 
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.") # Mudar Previous: 0.01
    parser.add_argument("--hidden_dim", type=int, default=32, help="Policy MLP hidden dimension.") # Mudar?
    parser.add_argument("--eval_duration_s", type=int, default=1200, help="Duration of evaluation simulation in seconds.")
    parser.add_argument("--load_weights", type=str, default=None, help="Path to pre-trained policy weights (e.g. policy_weights.npz). If provided, skips training.")
    parser.add_argument("--save_weights", type=str, default=None, help="Path to save the newly trained policy weights (e.g. new_policy_weights.npz).")

    args = parser.parse_args()
    
    if not os.path.isdir(args.data_dir):
        print(f"Error: Constellation data directory does not exist: {args.data_dir}")
        sys.exit(1)
        
    # Set default output path if not provided
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, f"dynamic_state_5000ms_for_{args.eval_duration_s}s_rl_3")
        
    # Read simulation parameters from input directory
    # For speed of training, we will simulate a 3600s window,
    # but when generating the final fstates we can run for the full evaluation duration.
    train_env = SatelliteHandoverEnv(args.data_dir, time_step_ms=5000, duration_s=1000, K=4)
    eval_env = SatelliteHandoverEnv(args.data_dir, time_step_ms=5000, duration_s=args.eval_duration_s, K=4)
    
    agent = PolicyGradientAgent(
        state_dim=train_env.state_dim,
        action_dim=train_env.action_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=0.95
    )
    
    default_weights_path = os.path.join(SCRIPT_DIR, "policy_weights.npz")
    weights_save_path = args.save_weights if args.save_weights else default_weights_path
    
    # 1. Load weights or train
    if args.load_weights:
        print(f"Loading weights from: {args.load_weights}")
        agent.load_weights(args.load_weights)
    elif os.path.exists(default_weights_path) and args.episodes == 0:
        print(f"Episodes is 0. Loading default weights from: {default_weights_path}")
        agent.load_weights(default_weights_path)
    else:
        # Train the RL agent
        train_agent(train_env, agent, num_episodes=args.episodes)
        # Save weights
        agent.save_weights(weights_save_path)
        print(f"Policy weights saved to {weights_save_path}")
    
    # 2. Generate the optimized routing tables using the trained agent
    generate_rl_dynamic_state(eval_env, agent, args.output_dir)
    
    print("\n" + "=" * 80)
    print("SUCCESS: RL OPTIMIZED DYNAMIC STATE GENERATED!")
    print("You can now modify step_1_prepare_run.py to use this directory:")
    print(f"  DYNAMIC_STATE_NAME = \"dynamic_state_5000ms_for_{args.eval_duration_s}s_rl\"")
    print("=" * 80)

if __name__ == "__main__":
    main()
