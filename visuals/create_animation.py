"""AntiGravSim: Dynamic Telemetry Animation Generator.

Renders an animated physics simulation of the object taking off,
stabilizing, and executing local gravity cancellation with a live telemetry HUD.
Generates outputs/demo_video.gif and outputs/demo_video.mp4.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import imageio
from stable_baselines3 import PPO

# Allow importing antigrav_env from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from antigrav_env import AntiGravEnv


def main():
    parser = argparse.ArgumentParser(description="Create AntiGrav Telemetry Animation")
    parser.add_argument("--model-path", type=str, default="./logs/best_model.zip", help="Path to trained PPO model")
    parser.add_argument("--output-gif", type=str, default="./outputs/demo_video.gif", help="Output GIF path")
    parser.add_argument("--output-mp4", type=str, default="./outputs/demo_video.mp4", help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=30, help="Playback frame rate")
    parser.add_argument("--steps", type=int, default=350, help="Number of simulation steps")

    args = parser.parse_args()

    model_path = args.model_path
    if not os.path.exists(model_path):
        alt_path = "./logs/best_model/best_model.zip"
        if os.path.exists(alt_path):
            model_path = alt_path

    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'. Run train.py first.")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_gif)), exist_ok=True)

    print(f"Loading policy from: {model_path}")
    model = PPO.load(model_path)

    env = AntiGravEnv(random_start=False)
    # Start on launch pad (z = 0.1 m)
    obs, info = env.reset(options={"initial_z": 0.1, "initial_vz": 0.0})

    trajectory = []
    print("Recording physics trajectory...")
    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        trajectory.append({
            "step": step,
            "time": step * env.dt,
            "z": info["z"],
            "vz": info["vz"],
            "thrust": info["thrust"],
            "az": info["az"],
        })

        if terminated or truncated:
            break

    print(f"Collected {len(trajectory)} frames. Rendering telemetry visualization...")

    frames = []
    # Sample every 2nd step if trajectory is long to keep GIF lightweight and responsive
    step_skip = 2 if len(trajectory) > 200 else 1

    for idx in range(0, len(trajectory), step_skip):
        item = trajectory[idx]
        z = item["z"]
        vz = item["vz"]
        thrust = item["thrust"]
        az = item["az"]
        t = item["time"]

        fig, ax = plt.subplots(figsize=(6, 8), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")

        # 1. Ground & Environment Boundaries
        ax.axhspan(-0.5, 0.0, color="#1c2128", alpha=0.9, zorder=1)
        ax.axhline(0.0, color="#484f58", linewidth=2.5, zorder=2)

        # 2. Target Altitude Line
        ax.axhline(env.z_target, color="#ffea00", linestyle="--", linewidth=1.5, alpha=0.8, zorder=2)
        ax.text(
            1.8,
            env.z_target + 0.04,
            f"Target: {env.z_target:.1f} m",
            color="#ffea00",
            fontsize=9,
            fontweight="bold",
            ha="right",
            alpha=0.9,
        )

        # Tolerance zone
        ax.axhspan(env.z_target - 0.05, env.z_target + 0.05, color="#00e5ff", alpha=0.08, zorder=1)

        # 3. Dynamic Levitating Body
        thrust_ratio = np.clip(thrust / env.u_max, 0.0, 1.0)
        # Sphere color transitions from electric cyan to intense orange under maximum thrust
        body_color = plt.cm.coolwarm(thrust_ratio)
        body = Circle((0.0, z), radius=0.18, facecolor=body_color, edgecolor="#ffffff", linewidth=2.0, zorder=5)
        ax.add_patch(body)

        # 4. Animated Thruster Plume
        if thrust > 0.5:
            flame_len = 0.15 + (thrust_ratio * 0.55)
            # Outer glow
            ax.plot([0.0, 0.0], [z - 0.18, z - 0.18 - flame_len], color="#ff9100", linewidth=5, alpha=0.7, zorder=4)
            # Inner white-hot core
            ax.plot([0.0, 0.0], [z - 0.18, z - 0.18 - (flame_len * 0.6)], color="#ffffff", linewidth=2.5, alpha=0.9, zorder=4)

        # 5. Anti-Gravity Status
        is_hovering = abs(z - env.z_target) < 0.06 and abs(vz) < 0.15
        status_text = "ANTIGRAVITY HOVER" if is_hovering else ("ASCENDING" if vz > 0.1 else "STABILIZING")
        status_color = "#00ff88" if is_hovering else "#ffea00"

        # 6. HUD Telemetry Card
        hud_lines = [
            "─── FLIGHT TELEMETRY ───",
            f"Status:   {status_text}",
            f"Time:     {t:5.2f} s",
            f"Altitude: {z:5.2f} m",
            f"Velocity: {vz:5.2f} m/s",
            f"Thrust:   {thrust:5.2f} N",
            f"Net Acc:  {az:5.2f} m/s²",
        ]
        hud_box = "\n".join(hud_lines)

        ax.text(
            0.05,
            0.95,
            hud_box,
            transform=ax.transAxes,
            fontsize=9.5,
            fontfamily="monospace",
            fontweight="bold",
            color="#e6edf3",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="#161b22",
                edgecolor=status_color,
                linewidth=1.5,
                alpha=0.92,
            ),
        )

        # 7. Axes Setup
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-0.3, max(2.0, env.z_target + 0.8))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("AntiGravSim: Neural Gravity Cancellation", fontsize=12, fontweight="bold", color="#ffffff", pad=12)

        # Render frame buffer
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(img)
        plt.close(fig)

        if (len(frames)) % 40 == 0:
            print(f"  Rendered {len(frames)} frames...")

    # Write GIF
    print(f"Compiling GIF animation ({len(frames)} frames at {args.fps} FPS)...")
    duration = 1.0 / args.fps
    imageio.mimsave(args.output_gif, frames, duration=duration, loop=0)
    print(f"Saved: {args.output_gif}")

    # Write MP4 if supported
    try:
        writer = imageio.get_writer(args.output_mp4, fps=args.fps, codec="libx264")
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"Saved: {args.output_mp4}")
    except Exception as e:
        print(f"Note: MP4 export skipped ({e}). GIF is ready at {args.output_gif}.")

    env.close()
    print("Animation generation complete!")


if __name__ == "__main__":
    main()
