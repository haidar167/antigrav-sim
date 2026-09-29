"""AntiGravSim: Altitude and Trajectory Visualization.

Generates publication-ready plots of altitude convergence, target tracking,
and vertical velocity profiles.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

# Allow importing antigrav_env from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from antigrav_env import AntiGravEnv


def main():
    parser = argparse.ArgumentParser(description="Plot Altitude and Velocity Profiles")
    parser.add_argument("--model-path", type=str, default="./logs/best_model.zip", help="Path to trained PPO model")
    parser.add_argument("--output", type=str, default="./outputs/altitude_plot.png", help="Path to save output plot")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
    parser.add_argument("--wind", type=float, default=0.0, help="Wind disturbance strength")

    args = parser.parse_args()

    model_path = args.model_path
    if not os.path.exists(model_path):
        alt_path = "./logs/best_model/best_model.zip"
        if os.path.exists(alt_path):
            model_path = alt_path

    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'. Run train.py first.")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"Loading policy from: {model_path}")
    model = PPO.load(model_path)

    # Initialize at a distinct non-target height to demonstrate active climb & hover
    env = AntiGravEnv(wind_strength=args.wind, random_start=False)
    obs, info = env.reset(options={"initial_z": 0.2, "initial_vz": 0.0})

    times = []
    z_history = []
    vz_history = []

    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        times.append(step * env.dt)
        z_history.append(info["z"])
        vz_history.append(info["vz"])

        if terminated or truncated:
            break

    # Setup Dark Theme Plot
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("#0e1117")
    ax1.set_facecolor("#161b22")
    ax2.set_facecolor("#161b22")

    # Top Plot: Altitude Trajectory
    ax1.plot(times, z_history, color="#00e5ff", linewidth=2.4, label="Object Altitude z(t)")
    ax1.axhline(env.z_target, color="#ffea00", linestyle="--", linewidth=1.8, label=f"Target ({env.z_target:.1f} m)")
    ax1.fill_between(
        times,
        env.z_target - 0.05,
        env.z_target + 0.05,
        color="#00e5ff",
        alpha=0.15,
        label="Tolerance Band (±0.05 m)",
    )
    ax1.set_ylabel("Altitude [m]", fontsize=12, fontweight="bold", color="#e6edf3")
    ax1.set_title("AntiGravSim: Autonomous Altitude Control & Stable Levitation", fontsize=14, fontweight="bold", pad=12, color="#ffffff")
    ax1.grid(True, linestyle="--", alpha=0.25, color="#8b949e")
    ax1.legend(loc="upper right", framealpha=0.85, facecolor="#1f242c")

    # Bottom Plot: Vertical Velocity
    ax2.plot(times, vz_history, color="#00ff88", linewidth=1.8, label="Vertical Velocity vz(t)")
    ax2.axhline(0.0, color="#ffffff", linestyle=":", alpha=0.6, linewidth=1.2, label="Zero Velocity")
    ax2.set_xlabel("Time [seconds]", fontsize=12, fontweight="bold", color="#e6edf3")
    ax2.set_ylabel("Velocity [m/s]", fontsize=12, fontweight="bold", color="#e6edf3")
    ax2.grid(True, linestyle="--", alpha=0.25, color="#8b949e")
    ax2.legend(loc="upper right", framealpha=0.85, facecolor="#1f242c")

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"Successfully generated altitude plot: {args.output}")
    env.close()


if __name__ == "__main__":
    main()
