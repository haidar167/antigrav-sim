"""AntiGravSim: Quantitative Evaluation Pipeline.

Evaluates a trained policy across multiple episodes and reports
aerospace/control-theory performance metrics including local gravity cancellation.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import csv
import os
import numpy as np
from stable_baselines3 import PPO

from antigrav_env import AntiGravEnv


def main():
    parser = argparse.ArgumentParser(description="Evaluate AntiGravSim Model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./logs/best_model.zip",
        help="Path to trained PPO model archive (.zip)",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--wind", type=float, default=0.0, help="Wind disturbance force magnitude")
    parser.add_argument("--save-data", action="store_true", help="Save per-step trajectories to CSV")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")

    args = parser.parse_args()

    # Fallback to best_model/best_model.zip if logs/best_model.zip is absent
    model_path = args.model_path
    if not os.path.exists(model_path):
        alt_path = "./logs/best_model/best_model.zip"
        if os.path.exists(alt_path):
            model_path = alt_path

    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{args.model_path}'. Please run 'python train.py' first.")
        sys.exit(1)

    print(f"Loading trained policy from: {model_path}")
    model = PPO.load(model_path)

    env = AntiGravEnv(wind_strength=args.wind, random_start=True)

    if args.save_data:
        os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = []

    print("\n" + "=" * 84)
    print("  ANTIGRAVSIM QUANTITATIVE BENCHMARK")
    print("=" * 84)
    header = (
        f"{'Ep':>3} | {'Steps':>5} | {'Alt Err (m)':>11} | {'|az| (m/s^2)':>12} | "
        f"{'Grav Cancel%':>12} | {'Hover%':>8} | {'Reward':>9}"
    )
    print(header)
    print("-" * 84)

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        step = 0
        ep_data = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            ep_data.append({
                "step": step,
                "time": step * env.dt,
                "z": info["z"],
                "vz": info["vz"],
                "thrust": info["thrust"],
                "az": info["az"],
                "reward": reward,
            })

        # Calculate metrics for the episode
        z_arr = np.array([d["z"] for d in ep_data])
        az_arr = np.array([d["az"] for d in ep_data])
        rew_arr = np.array([d["reward"] for d in ep_data])
        thrust_arr = np.array([d["thrust"] for d in ep_data])

        alt_err = float(np.mean(np.abs(z_arr - env.z_target)))
        mean_abs_az = float(np.mean(np.abs(az_arr)))
        grav_cancel_pct = float(max(0.0, (1.0 - mean_abs_az / 9.81)) * 100.0)
        hover_pct = float(np.mean(np.abs(z_arr - env.z_target) < 0.08) * 100.0)
        tot_reward = float(np.sum(rew_arr))
        energy = float(np.sum(thrust_arr ** 2))

        m = {
            "episode": ep + 1,
            "steps": len(ep_data),
            "alt_err": alt_err,
            "mean_abs_az": mean_abs_az,
            "grav_cancel_pct": grav_cancel_pct,
            "hover_pct": hover_pct,
            "reward": tot_reward,
            "energy": energy,
        }
        all_metrics.append(m)

        print(
            f"{m['episode']:3d} | {m['steps']:5d} | {m['alt_err']:11.4f} | "
            f"{m['mean_abs_az']:12.4f} | {m['grav_cancel_pct']:11.1f}% | "
            f"{m['hover_pct']:7.1f}% | {m['reward']:9.1f}"
        )

        if args.save_data:
            csv_path = os.path.join(args.output_dir, f"eval_episode_{ep+1}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=ep_data[0].keys())
                writer.writeheader()
                writer.writerows(ep_data)

    print("-" * 84)

    # Compute overall averages
    avg_alt_err = np.mean([m["alt_err"] for m in all_metrics])
    avg_az = np.mean([m["mean_abs_az"] for m in all_metrics])
    avg_grav_cancel = np.mean([m["grav_cancel_pct"] for m in all_metrics])
    avg_hover = np.mean([m["hover_pct"] for m in all_metrics])
    avg_reward = np.mean([m["reward"] for m in all_metrics])

    print(
        f"{'AVG':>3} | {'-':>5} | {avg_alt_err:11.4f} | "
        f"{avg_az:12.4f} | {avg_grav_cancel:11.1f}% | "
        f"{avg_hover:7.1f}% | {avg_reward:9.1f}"
    )
    print("=" * 84)

    env.close()


if __name__ == "__main__":
    main()
