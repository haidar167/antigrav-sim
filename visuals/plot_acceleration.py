"""AntiGravSim: Net Acceleration and Force Analysis.

Plots thrust command vs. Earth gravity baseline (mg = 9.81 N)
and net vertical acceleration, proving that effective local gravity is canceled to zero.
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
    parser = argparse.ArgumentParser(description="Plot Net Acceleration and Gravity Cancellation")
    parser.add_argument("--model-path", type=str, default="./logs/best_model.zip", help="Path to trained PPO model")
    parser.add_argument("--output", type=str, default="./outputs/acceleration_plot.png", help="Path to save output plot")
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

    env = AntiGravEnv(wind_strength=args.wind, random_start=False)
    obs, info = env.reset(options={"initial_z": 0.2, "initial_vz": 0.0})

    times = []
    thrust_history = []
    az_history = []

    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        times.append(step * env.dt)
        thrust_history.append(info["thrust"])
        az_history.append(info["az"])

        if terminated or truncated:
            break

    times = np.array(times)
    thrust_history = np.array(thrust_history)
    az_history = np.array(az_history)

    # Hover phase (after initial climb, steps > 100)
    hover_slice = slice(100, len(az_history))
    mean_abs_az = np.mean(np.abs(az_history[hover_slice]))
    grav_cancel_pct = max(0.0, (1.0 - mean_abs_az / 9.81)) * 100.0

    # Setup Dark Theme Plot
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    fig.patch.set_facecolor("#0e1117")
    ax1.set_facecolor("#161b22")
    ax2.set_facecolor("#161b22")

    # Top Plot: Thrust Command vs Gravity Baseline
    mg = env.mass * env.gravity
    ax1.plot(times, thrust_history, color="#ff9100", linewidth=2.0, label="Agent Continuous Thrust Command [N]")
    ax1.axhline(mg, color="#ff3d71", linestyle="--", linewidth=1.8, label=f"Earth Gravity Equilibrium (mg = {mg:.2f} N)")
    ax1.set_ylabel("Force [N]", fontsize=12, fontweight="bold", color="#e6edf3")
    ax1.set_title("AntiGravSim: Force Equilibrium & Gravity Counteraction", fontsize=14, fontweight="bold", pad=12, color="#ffffff")
    ax1.grid(True, linestyle="--", alpha=0.25, color="#8b949e")
    ax1.legend(loc="upper right", framealpha=0.85, facecolor="#1f242c")

    # Bottom Plot: Net Acceleration
    ax2.plot(times, az_history, color="#d500f9", linewidth=2.0, label="Net Vertical Acceleration az = u/m - g")
    ax2.axhline(0.0, color="#00ff88", linestyle="-", linewidth=2.0, label="True Anti-Gravity Line (az = 0 m/s^2)")
    ax2.fill_between(times, -0.3, 0.3, color="#00ff88", alpha=0.15, label="Local Zero-G Envelope (±0.3 m/s^2)")
    ax2.set_xlabel("Time [seconds]", fontsize=12, fontweight="bold", color="#e6edf3")
    ax2.set_ylabel("Net Accel [m/s^2]", fontsize=12, fontweight="bold", color="#e6edf3")
    ax2.grid(True, linestyle="--", alpha=0.25, color="#8b949e")
    ax2.legend(loc="upper right", framealpha=0.85, facecolor="#1f242c")

    # Efficiency badge
    ax2.text(
        0.02,
        0.15,
        f"Hover Gravity Canceled: {grav_cancel_pct:.1f}%\nMean Hover |az|: {mean_abs_az:.3f} m/s^2",
        transform=ax2.transAxes,
        fontsize=11,
        fontweight="bold",
        color="#00ff88",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d1117", edgecolor="#00ff88", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"Successfully generated acceleration plot: {args.output}")
    print(f"Hover Gravity Cancellation: {grav_cancel_pct:.1f}% (Mean |az| = {mean_abs_az:.3f} m/s^2)")
    env.close()


if __name__ == "__main__":
    main()
