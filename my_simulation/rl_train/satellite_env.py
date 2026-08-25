import os
import sys
import math
import numpy as np
from astropy import units as u
from astropy.time import Time
import ephem

# Add satgenpy path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "satgenpy"))

from satgen.ground_stations import read_ground_stations_extended
from satgen.tles import read_tles
from satgen.distance_tools import distance_m_ground_station_to_satellite

# Helper to read ISL topology
def read_isls(path: str) -> list[tuple[int, int]]:
    isls: list[tuple[int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                isls.append((int(parts[0]), int(parts[1])))
    return isls

# Helper to parse description
def parse_description(path: str) -> dict[str, float]:
    result: dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            result[key] = float(value)
    return result

class SatelliteHandoverEnv:
    """
    A lightweight, fast Python environment that simulates satellite geometry
    and link qualities over time to train Reinforcement Learning agents for handover.
    """
    def __init__(self, data_dir: str, time_step_ms: int = 1000, duration_s: int = 100, K: int = 4):
        self.data_dir = data_dir
        self.time_step_ms = time_step_ms
        self.duration_s = duration_s
        self.K = K
        self.time_step_ns = time_step_ms * 1_000_000
        
        # Load topology files
        self.ground_stations = read_ground_stations_extended(os.path.join(data_dir, "ground_stations.txt"))
        self.tles = read_tles(os.path.join(data_dir, "tles.txt"))
        self.satellites = self.tles["satellites"]
        self.epoch = self.tles["epoch"]
        self.isls = read_isls(os.path.join(data_dir, "isls.txt"))
        
        desc = parse_description(os.path.join(data_dir, "description.txt"))
        self.max_gsl_length_m = desc.get("max_gsl_length_m", 900000.0)
        self.max_isl_length_m = desc.get("max_isl_length_m", 14000000.0)
        
        # Signal propagation constants
        self.freq_ghz = 28.0  # Ka band typical
        self.tx_power_dbm = 20.0
        self.tx_gain_dbi = 10.0
        self.rx_gain_dbi = 10.0
        self.speed_of_light = 299792458.0
        
        # Environment penalties
        self.handover_penalty = 5.0  # Penalty in SNR equivalent
        self.disconnect_penalty = 80.0 # Penalty for no active connection
        
        # State tracking
        self.current_step = 0
        self.max_steps = int(duration_s * 1000 / time_step_ms)
        self.current_satellites = [-1] * len(self.ground_stations) # Active satellite per GS
        
        # For mapping observations
        self.state_dim = self.K * 8
        self.action_dim = self.K

    def reset(self) -> dict[int, np.ndarray]:
        """Resets the environment to t=0."""
        self.current_step = 0
        self.current_satellites = [-1] * len(self.ground_stations)
        
        # Initially connect to the closest satellite
        obs = self._get_observations()
        for gs_id in range(len(self.ground_stations)):
            candidates = self._get_candidates(gs_id)
            if candidates:
                self.current_satellites[gs_id] = candidates[0]["sid"]
                
        # Get updated observations with active connections set
        return self._get_observations()

    def _get_time_at_step(self, step: int) -> Time:
        """Returns the astropy Time at the given step index."""
        time_since_epoch_ns = step * self.time_step_ns
        return self.epoch + time_since_epoch_ns * u.ns

    def _get_candidates(self, gs_id: int) -> list[dict]:
        """Calculates visible satellites and returns their attributes sorted by distance."""
        gs = self.ground_stations[gs_id]
        current_time = self._get_time_at_step(self.current_step)
        next_time = current_time + 0.1 * u.s  # small offset to calculate velocity
        
        step_s = self.time_step_ms / 1000.0
        time_plus_1 = current_time + step_s * u.s
        time_plus_2 = current_time + 2.0 * step_s * u.s
        
        epoch_str = str(self.epoch)
        current_time_str = str(current_time)
        next_time_str = str(next_time)
        
        # Set observer for elevation/azimuth calculations
        observer = ephem.Observer()
        observer.epoch = epoch_str
        observer.date = current_time_str
        observer.lat = str(gs["latitude_degrees_str"])
        observer.lon = str(gs["longitude_degrees_str"])
        observer.elevation = gs["elevation_m_float"]
        
        observer_next = ephem.Observer()
        observer_next.epoch = epoch_str
        observer_next.date = next_time_str
        observer_next.lat = str(gs["latitude_degrees_str"])
        observer_next.lon = str(gs["longitude_degrees_str"])
        observer_next.elevation = gs["elevation_m_float"]
        
        observer_p1 = ephem.Observer()
        observer_p1.epoch = epoch_str
        observer_p1.date = str(time_plus_1)
        observer_p1.lat = str(gs["latitude_degrees_str"])
        observer_p1.lon = str(gs["longitude_degrees_str"])
        observer_p1.elevation = gs["elevation_m_float"]
        
        observer_p2 = ephem.Observer()
        observer_p2.epoch = epoch_str
        observer_p2.date = str(time_plus_2)
        observer_p2.lat = str(gs["latitude_degrees_str"])
        observer_p2.lon = str(gs["longitude_degrees_str"])
        observer_p2.elevation = gs["elevation_m_float"]
        
        candidates = []
        for sid, sat in enumerate(self.satellites):
            # Distance at t
            sat.compute(observer)
            dist_m = sat.range
            elevation_deg = math.degrees(float(sat.alt))
            
            # Check visibility limit
            if dist_m <= self.max_gsl_length_m and elevation_deg >= 10.0:
                # Distance at t + dt to get rate of change
                sat.compute(observer_next)
                dist_next_m = sat.range
                
                velocity_m_per_s = (dist_next_m - dist_m) / 0.1
                
                # Calculate time to loss
                if velocity_m_per_s <= 0:
                    time_to_loss = 3600.0  # Large value representing approaching or stationary
                else:
                    time_to_loss = (self.max_gsl_length_m - dist_m) / velocity_m_per_s
                
                # SNR calculation (Friis propagation)
                path_loss = 20 * math.log10(4 * math.pi * dist_m * (self.freq_ghz * 1e9) / self.speed_of_light)
                snr = self.tx_power_dbm + self.tx_gain_dbi + self.rx_gain_dbi - path_loss
                
                # Lookahead 1 step
                sat.compute(observer_p1)
                dist_p1_m = sat.range
                elev_p1_deg = math.degrees(float(sat.alt))
                visible_p1 = 1.0 if (dist_p1_m <= self.max_gsl_length_m and elev_p1_deg >= 10.0) else 0.0
                if visible_p1:
                    path_loss_p1 = 20 * math.log10(4 * math.pi * dist_p1_m * (self.freq_ghz * 1e9) / self.speed_of_light)
                    snr_p1 = self.tx_power_dbm + self.tx_gain_dbi + self.rx_gain_dbi - path_loss_p1
                else:
                    snr_p1 = 0.0
                    
                # Lookahead 2 steps
                sat.compute(observer_p2)
                dist_p2_m = sat.range
                elev_p2_deg = math.degrees(float(sat.alt))
                visible_p2 = 1.0 if (dist_p2_m <= self.max_gsl_length_m and elev_p2_deg >= 10.0) else 0.0
                if visible_p2:
                    path_loss_p2 = 20 * math.log10(4 * math.pi * dist_p2_m * (self.freq_ghz * 1e9) / self.speed_of_light)
                    snr_p2 = self.tx_power_dbm + self.tx_gain_dbi + self.rx_gain_dbi - path_loss_p2
                else:
                    snr_p2 = 0.0
                
                candidates.append({
                    "sid": sid,
                    "distance_m": dist_m,
                    "elevation_deg": elevation_deg,
                    "velocity_m_per_s": velocity_m_per_s,
                    "time_to_loss": max(0.0, time_to_loss),
                    "snr": snr,
                    "visible_p1": visible_p1,
                    "snr_p1": snr_p1,
                    "visible_p2": visible_p2,
                    "snr_p2": snr_p2
                })
                
        # Sort candidates by distance (closest first)
        candidates.sort(key=lambda x: x["distance_m"])
        return candidates

    def _get_observations(self) -> dict[int, np.ndarray]:
        """Generates observations for all ground stations."""
        obs_dict = {}
        for gs_id in range(len(self.ground_stations)):
            candidates = self._get_candidates(gs_id)
            current_active = self.current_satellites[gs_id]
            
            # Construct observation array of size self.K * 8
            obs = np.zeros(self.state_dim, dtype=np.float32)
            
            for i in range(self.K):
                if i < len(candidates):
                    cand = candidates[i]
                    # Normalized features
                    elevation_norm = cand["elevation_deg"] / 90.0
                    time_to_loss_norm = min(300.0, cand["time_to_loss"]) / 300.0
                    snr_norm = min(40.0, max(0.0, cand["snr"])) / 40.0
                    is_connected = 1.0 if cand["sid"] == current_active else 0.0
                    visible_p1 = cand["visible_p1"]
                    snr_p1_norm = min(40.0, max(0.0, cand["snr_p1"])) / 40.0
                    visible_p2 = cand["visible_p2"]
                    snr_p2_norm = min(40.0, max(0.0, cand["snr_p2"])) / 40.0
                    
                    idx = i * 8
                    obs[idx] = elevation_norm
                    obs[idx + 1] = time_to_loss_norm
                    obs[idx + 2] = snr_norm
                    obs[idx + 3] = is_connected
                    obs[idx + 4] = visible_p1
                    obs[idx + 5] = snr_p1_norm
                    obs[idx + 6] = visible_p2
                    obs[idx + 7] = snr_p2_norm
                    
            obs_dict[gs_id] = obs
        return obs_dict

    def step(self, actions: dict[int, int]) -> tuple[dict[int, np.ndarray], dict[int, float], bool, dict]:
        """
        Advances the simulation by 1 step.
        actions: dict mapping gs_id to choice index (0 to K-1)
        """
        self.current_step += 1
        rewards = {}
        
        # Calculate rewards for this step before updating time index (evaluating choices made in the current step)
        for gs_id in range(len(self.ground_stations)):
            action = actions.get(gs_id, 0)
            candidates = self._get_candidates(gs_id)
            prev_connected = self.current_satellites[gs_id]
            
            # Default: disconnected
            selected_sid = -1
            reward = -self.disconnect_penalty
            
            if action < len(candidates):
                chosen_cand = candidates[action]
                selected_sid = chosen_cand["sid"]
                
                # Check link parameters
                snr = chosen_cand["snr"]
                dist_m = chosen_cand["distance_m"]
                latency_ms = (2.0 * dist_m / self.speed_of_light) * 1000.0
                
                # Base Reward is SNR minus latency penalty
                reward = snr - 0.1 * latency_ms
                
                # Apply Handover penalty if we switched satellite
                if prev_connected != -1 and selected_sid != prev_connected:
                    reward -= self.handover_penalty
                    
                self.current_satellites[gs_id] = selected_sid
            else:
                # Invalid choice or index out of range of visible satellites
                # Connect to the closest visible satellite with a penalty, if any
                if candidates:
                    selected_sid = candidates[0]["sid"]
                    snr = candidates[0]["snr"]
                    reward = snr - self.handover_penalty - 5.0 # extra penalty for invalid action selection
                    self.current_satellites[gs_id] = selected_sid
                else:
                    self.current_satellites[gs_id] = -1
                    reward = -self.disconnect_penalty
                    
            rewards[gs_id] = reward
            
        terminated = self.current_step >= self.max_steps
        
        # Get next observations
        next_obs = self._get_observations()
        
        return next_obs, rewards, terminated, {}
