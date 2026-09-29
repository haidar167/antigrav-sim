import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

class AntiGravEnv(gym.Env):
    """
    1D Anti-Gravity Levitation Environment.
    
    A point-mass is controlled by vertical thrust to maintain a target altitude.
    
    State:
        - z: Vertical position
        - vz: Vertical velocity
        - error_to_target: z - z_target
        
    Actions:
        - u: Continuous thrust in [0, u_max]
        
    Reward:
        Penalizes distance to target, velocity, thrust usage, and acceleration.
        Bonus for stability.
    """
    
    def __init__(self, z_target: float = 1.0, wind_strength: float = 0.0):
        super().__init__()
        
        self.mass = 1.0
        self.gravity = 9.81
        self.dt = 0.02
        self.u_max = 20.0
        self.max_steps = 500
        
        self.z_target = z_target
        self.wind_strength = wind_strength
        
        # State: [z, vz, error_to_target]
        self.observation_space = spaces.Box(
            low=np.array([-10.0, -10.0, -20.0], dtype=np.float32),
            high=np.array([10.0, 10.0, 20.0], dtype=np.float32)
        )
        
        # Action: [u]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([self.u_max], dtype=np.float32)
        )
        
        self.state = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.step_count = 0
        
        # Initial state: on the ground with zero velocity
        z = 0.01 # Slightly above 0 to avoid immediate crash
        vz = 0.0
        error = z - self.z_target
        
        self.state = np.array([z, vz, error], dtype=np.float32)
        
        return self.state, {}
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        
        z, vz, _ = self.state
        u = np.clip(action[0], 0.0, self.u_max)
        
        # Wind disturbance
        wind_force = self.np_random.uniform(-self.wind_strength, self.wind_strength)
        
        # Dynamics
        az = (u + wind_force) / self.mass - self.gravity
        
        # Euler integration
        vz_new = vz + az * self.dt
        z_new = z + vz * self.dt
        
        error = z_new - self.z_target
        
        self.state = np.array([z_new, vz_new, error], dtype=np.float32)
        
        # Reward
        stable_bonus = 0.5 if abs(error) < 0.1 and abs(vz_new) < 0.1 else 0.0
        reward = -1.0 * (error ** 2) - 0.1 * (vz_new ** 2) - 0.01 * (u ** 2) - 0.05 * abs(az) + stable_bonus
        
        # Termination
        terminated = bool(z_new <= 0.0)
        truncated = bool(self.step_count >= self.max_steps)
        
        return self.state, float(reward), terminated, truncated, {}
