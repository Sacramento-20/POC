import numpy as np

class PolicyGradientAgent:
    """
    A pure NumPy implementation of the REINFORCE (Policy Gradient) algorithm.
    It uses a simple 2-layer Multi-Layer Perceptron (MLP) with ReLU activations.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 32, lr: float = 0.01, gamma: float = 0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        
        # Xavier initialization of weights
        self.W1 = np.random.randn(state_dim, hidden_dim) / np.sqrt(state_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, action_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros((1, action_dim))
        
        # Adam Optimizer parameters
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0
        
        # Trajectory storage
        self.reset_trajectory()

    def reset_trajectory(self):
        # Maps gs_id to lists
        self.states = {}
        self.actions = {}
        self.probs = {}
        self.rewards = {}

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, state: np.ndarray, mask: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes forward pass of the policy network with optional action masking.
        state: array of shape (obs_dim,) or (batch, obs_dim)
        Returns: logits, hidden_activation, probabilities
        """
        # Ensure 2D shape (1, obs_dim) if single sample
        if state.ndim == 1:
            state = state[np.newaxis, :]
            
        z1 = np.dot(state, self.W1) + self.b1
        a1 = np.maximum(0, z1) # ReLU
        z2 = np.dot(a1, self.W2) + self.b2
        
        # Apply action masking by setting invalid actions' logits to a very large negative value
        if mask is not None:
            if mask.ndim == 1:
                mask = mask[np.newaxis, :]
            z2 = np.where(mask > 0, z2, -1e9)
            
        probs = self._softmax(z2)
        return z2, a1, probs

    def select_action(self, gs_id: int, state: np.ndarray) -> int:
        """Selects an action by sampling from the policy network's output distribution with masking."""
        # Dynamically compute action mask from state:
        # Candidate i is visible if its elevation (feature index i * 8) is > 0
        mask = np.zeros(self.action_dim, dtype=np.float32)
        for i in range(self.action_dim):
            if state[i * 8] > 0:
                mask[i] = 1.0
        if np.sum(mask) == 0:
            mask[0] = 1.0
            
        _, _, probs = self.forward(state, mask)
        probs = probs[0] # Take first element as batch is 1
        
        # Sample action based on probabilities
        action = np.random.choice(self.action_dim, p=probs)
        
        # Save transitions
        self.states.setdefault(gs_id, []).append(state)
        self.actions.setdefault(gs_id, []).append(action)
        self.probs.setdefault(gs_id, []).append(probs)
        
        return action

    def store_reward(self, gs_id: int, reward: float):
        self.rewards.setdefault(gs_id, []).append(reward)

    def update(self) -> float:
        """
        Performs policy gradient update using REINFORCE with global normalization of returns.
        """
        self.t += 1
        
        # Collect trajectories and compute discounted returns
        trajectories = {}
        all_returns = []
        
        for gs_id in self.states.keys():
            states = np.vstack(self.states[gs_id]) # (T, state_dim)
            actions = np.array(self.actions[gs_id]) # (T,)
            probs = np.vstack(self.probs[gs_id]) # (T, action_dim)
            rewards = np.array(self.rewards[gs_id]) # (T,)
            
            T = len(rewards)
            returns = np.zeros_like(rewards, dtype=np.float32)
            discounted_sum = 0
            for k in reversed(range(T)):
                discounted_sum = rewards[k] + self.gamma * discounted_sum
                returns[k] = discounted_sum
                
            trajectories[gs_id] = {
                "states": states,
                "actions": actions,
                "probs": probs,
                "rewards": rewards,
                "returns": returns
            }
            all_returns.extend(returns)
            
        # Global normalization of returns to compare ground stations fairly
        all_returns = np.array(all_returns)
        global_mean = np.mean(all_returns)
        global_std = np.std(all_returns)
        
        # Accumulate gradients across all ground stations
        g_W1 = np.zeros_like(self.W1)
        g_b1 = np.zeros_like(self.b1)
        g_W2 = np.zeros_like(self.W2)
        g_b2 = np.zeros_like(self.b2)
        
        total_transitions = 0
        all_rewards = []
        
        for gs_id, traj in trajectories.items():
            states = traj["states"]
            actions = traj["actions"]
            probs = traj["probs"]
            rewards = traj["rewards"]
            returns = traj["returns"]
            
            T = len(rewards)
            total_transitions += T
            all_rewards.extend(rewards)
            
            # Normalize returns globally
            if global_std > 1e-5:
                norm_returns = (returns - global_mean) / global_std
            else:
                norm_returns = returns - global_mean
            
            # Compute gradients for each transition
            for k in range(T):
                x = states[k:k+1] # (1, state_dim)
                y = np.zeros((1, self.action_dim))
                y[0, actions[k]] = 1.0 # One-hot action
                
                # Re-compute mask from observation slice to apply during backprop
                mask = np.zeros((1, self.action_dim), dtype=np.float32)
                for i in range(self.action_dim):
                    if x[0, i * 8] > 0:
                        mask[0, i] = 1.0
                if np.sum(mask) == 0:
                    mask[0, 0] = 1.0
                
                # Forward values needed for backprop (applying the mask!)
                z2, a1, pr = self.forward(x, mask)
                
                # Gradient of loss with respect to logits (z2)
                # d_loss/d_z2 = (pr - y) * G_t
                d_z2 = (pr - y) * norm_returns[k]
                
                # Backprop to layers
                d_W2 = np.dot(a1.T, d_z2)
                d_b2 = d_z2
                
                d_a1 = np.dot(d_z2, self.W2.T)
                d_z1 = d_a1 * (a1 > 0).astype(np.float32) # ReLU derivative
                
                d_W1 = np.dot(x.T, d_z1)
                d_b1 = d_z1
                
                # Accumulate gradients
                g_W1 += d_W1
                g_b1 += d_b1
                g_W2 += d_W2
                g_b2 += d_b2
                
        # 3. Apply Adam optimizer updates
        if total_transitions > 0:
            # Average gradients
            g_W1 /= total_transitions
            g_b1 /= total_transitions
            g_W2 /= total_transitions
            g_b2 /= total_transitions
            
            # Update W1, b1
            self.m_W1 = self.beta1 * self.m_W1 + (1 - self.beta1) * g_W1
            self.v_W1 = self.beta2 * self.v_W1 + (1 - self.beta2) * (g_W1 ** 2)
            m_hat = self.m_W1 / (1 - self.beta1 ** self.t)
            v_hat = self.v_W1 / (1 - self.beta2 ** self.t)
            self.W1 -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
            self.m_b1 = self.beta1 * self.m_b1 + (1 - self.beta1) * g_b1
            self.v_b1 = self.beta2 * self.v_b1 + (1 - self.beta2) * (g_b1 ** 2)
            m_hat_b = self.m_b1 / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_b1 / (1 - self.beta2 ** self.t)
            self.b1 -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)
            
            # Update W2, b2
            self.m_W2 = self.beta1 * self.m_W2 + (1 - self.beta1) * g_W2
            self.v_W2 = self.beta2 * self.v_W2 + (1 - self.beta2) * (g_W2 ** 2)
            m_hat2 = self.m_W2 / (1 - self.beta1 ** self.t)
            v_hat2 = self.v_W2 / (1 - self.beta2 ** self.t)
            self.W2 -= self.lr * m_hat2 / (np.sqrt(v_hat2) + self.eps)
            
            self.m_b2 = self.beta1 * self.m_b2 + (1 - self.beta1) * g_b2
            self.v_b2 = self.beta2 * self.v_b2 + (1 - self.beta2) * (g_b2 ** 2)
            m_hat2_b = self.m_b2 / (1 - self.beta1 ** self.t)
            v_hat2_b = self.v_b2 / (1 - self.beta2 ** self.t)
            self.b2 -= self.lr * m_hat2_b / (np.sqrt(v_hat2_b) + self.eps)
            
        self.reset_trajectory()
        return np.mean(all_rewards) if all_rewards else 0.0

    def save_weights(self, path: str):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load_weights(self, path: str):
        data = np.load(path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
