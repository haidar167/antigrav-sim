"""AntiGravSim: Custom Gymnasium Environment for Anti-Gravity Simulation.

Simulates a point-mass levitator under Earth gravity (g = 9.81 m/s^2).
The reinforcement learning agent learns to command continuous thrust to cancel
gravity locally, hovering stably at a target altitude and rejecting disturbances.
"""
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class AntiGravEnv(gym.Env):
    """1D Anti-Gravity Levitation Environment.

    Observation Space:
        - z (float): Current vertical altitude [m]
        - vz (float): Current vertical velocity [m/s]
        - error_to_target (float): Distance to target altitude (z - z_target) [m]

    Action Space:
        - action (float in [-1.0, 1.0]): Normalized continuous thrust command.
          Mapped linearly to thrust u in [0, u_max]. When action=0, thrust ~ 10N
          (close to gravity cancellation for mass=1.0 kg).

    Physics:
        - Mass: 1.0 kg
        - Gravity: 9.81 m/s^2 (downward)
        - Max Thrust: 20.0 N
        - Timestep dt: 0.02 s (50 Hz control loop)
        - Max steps per episode: 500 (10 seconds)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        z_target: float = 1.0,
        wind_strength: float = 0.0,
        random_start: bool = True,
        max_steps: int = 500,
    ) -> None:
        super().__init__()

        self.mass: float = 1.0
        self.gravity: float = 9.81
        self.dt: float = 0.02
        self.u_max: float = 20.0
        self.max_steps: int = max_steps

        self.z_target: float = float(z_target)
        self.wind_strength: float = float(wind_strength)
        self.random_start: bool = random_start

        # State: [z, vz, error_to_target]
        self.observation_space = spaces.Box(
            low=np.array([-5.0, -15.0, -10.0], dtype=np.float32),
            high=np.array([10.0, 15.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Action: [-1.0, 1.0] -> mapped to [0, u_max]
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.state: np.ndarray = np.zeros(3, dtype=np.float32)
        self.step_count: int = 0
        self.last_thrust: float = 0.0
        self.last_az: float = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0

        # Determine initial altitude
        if options and "initial_z" in options:
            z = float(options["initial_z"])
            vz = float(options.get("initial_vz", 0.0))
        elif self.random_start:
            # Perturb around target height for robust exploration
            z = float(self.np_random.uniform(0.3, 1.7))
            vz = float(self.np_random.uniform(-0.2, 0.2))
        else:
            z = self.z_target
            vz = 0.0

        error = z - self.z_target
        self.state = np.array([z, vz, error], dtype=np.float32)
        self.last_thrust = self.mass * self.gravity
        self.last_az = 0.0

        return self.state, {"z": z, "vz": vz, "target": self.z_target}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        z, vz, _ = self.state

        # Map action [-1, 1] to continuous thrust [0, u_max]
        clipped_action = float(np.clip(action[0], -1.0, 1.0))
        u = (clipped_action + 1.0) * 0.5 * self.u_max
        self.last_thrust = u

        # Apply optional wind disturbance force
        wind = (
            float(self.np_random.uniform(-self.wind_strength, self.wind_strength))
            if self.wind_strength > 0.0
            else 0.0
        )

        # Net vertical acceleration: F_net / m
        az = (u + wind) / self.mass - self.gravity
        self.last_az = az

        # Euler numerical integration
        vz_new = vz + az * self.dt
        z_new = z + vz_new * self.dt

        # Physical floor collision constraint (inelastic ground contact)
        if z_new < 0.0:
            z_new = 0.0
            vz_new = max(0.0, vz_new)

        error_new = z_new - self.z_target
        self.state = np.array([z_new, vz_new, error_new], dtype=np.float32)

        # --- Reward Formulation ---
        # 1. Quadratic altitude error penalty
        r_pos = -2.0 * (error_new ** 2)
        # 2. Velocity damping penalty
        r_vel = -0.2 * (vz_new ** 2)
        # 3. Thrust energy efficiency penalty
        r_energy = -0.001 * (u ** 2)
        # 4. Net acceleration penalty (encourages zero net force = true anti-gravity)
        r_accel = -0.05 * abs(az)
        # 5. Stability bonus when precisely hovering near target
        r_hover = (
            2.0 if (abs(error_new) < 0.08 and abs(vz_new) < 0.1) else 0.0
        )

        reward = float(r_pos + r_vel + r_energy + r_accel + r_hover)

        # Early termination if runaway altitude occurs; truncation at max_steps
        terminated = bool(z_new > 6.0)
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "z": z_new,
            "vz": vz_new,
            "thrust": u,
            "az": az,
            "wind": wind,
            "step": self.step_count,
            "target": self.z_target,
        }

        return self.state, reward, terminated, truncated, info
