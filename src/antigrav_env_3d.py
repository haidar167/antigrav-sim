import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

class AntiGravEnv3D(gym.Env):
    """
    3D Anti-Gravity Levitation Environment.
    
    A point-mass is controlled by 3D thrust to maintain a target 3D position.
    
    State:
        - [x, y, z]: Position
        - [vx, vy, vz]: Velocity
        - [error_x, error_y, error_z]: Position error relative to target
        
    Actions:
        - [ux, uy, uz]: Continuous thrust in [0, u_max] for each axis
        
    Reward:
        Penalizes 3D distance to target, velocity magnitude, thrust usage, and acceleration.
        Bonus for stability.
    """
    
    def __init__(self, target_pos: Optional[List[float]] = None, wind_strength: float = 0.0):
        super().__init__()
        
        if target_pos is None:
            target_pos = [0.0, 0.0, 1.0]
            
        self.mass = 1.0
        self.gravity = 9.81
        self.dt = 0.02
        self.u_max = 20.0
        self.max_steps = 500
        
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self.wind_strength = wind_strength
        
        # State: 9D (pos, vel, error)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32
        )
        
        # Action: 3D (thrust per axis)
        self.action_space = spaces.Box(
            low=0.0,
            high=self.u_max,
            shape=(3,),
            dtype=np.float32
        )
        
        self.state = np.zeros(9, dtype=np.float32)
        self.step_count = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.step_count = 0
        
        pos = np.array([0.0, 0.0, 0.01], dtype=np.float32)
        vel = np.zeros(3, dtype=np.float32)
        error = pos - self.target_pos
        
        self.state = np.concatenate([pos, vel, error]).astype(np.float32)
        
        return self.state, {}
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        
        pos = self.state[0:3]
        vel = self.state[3:6]
        
        u = np.clip(action, 0.0, self.u_max)
        
        # Wind disturbance
        wind_force = self.np_random.uniform(-self.wind_strength, self.wind_strength, size=(3,))
        
        # Dynamics
        a = (u + wind_force) / self.mass
        a[2] -= self.gravity
        
        # Euler integration
        vel_new = vel + a * self.dt
        pos_new = pos + vel * self.dt
        
        error_new = pos_new - self.target_pos
        
        self.state = np.concatenate([pos_new, vel_new, error_new]).astype(np.float32)
        
        # Reward
        pos_error_sq = np.sum(error_new ** 2)
        vel_sq = np.sum(vel_new ** 2)
        u_sq = np.sum(u ** 2)
        a_mag = np.linalg.norm(a)
        
        error_dist = np.linalg.norm(error_new)
        vel_mag = np.linalg.norm(vel_new)
        
        stable_bonus = 0.5 if error_dist < 0.1 and vel_mag < 0.1 else 0.0
        
        reward = -1.0 * pos_error_sq - 0.1 * vel_sq - 0.01 * u_sq - 0.05 * a_mag + stable_bonus
        
        # Termination
        terminated = bool(pos_new[2] <= 0.0)
        truncated = bool(self.step_count >= self.max_steps)
        
        return self.state, float(reward), terminated, truncated, {}
