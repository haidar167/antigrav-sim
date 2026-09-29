"""AntiGravSim: PPO Training Pipeline.

Trains a continuous-action Proximal Policy Optimization (PPO) agent
to master anti-gravity levitation and hover control.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from antigrav_env import AntiGravEnv


def main():
    parser = argparse.ArgumentParser(description="Train AntiGravSim PPO Agent")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of vectorized parallel environments")
    parser.add_argument("--wind", type=float, default=0.0, help="Wind disturbance force magnitude")
    parser.add_argument("--lr", type=float, default=3e-4, help="PPO learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Directory to save checkpoints and logs")

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    tb_log_dir = os.path.join(args.log_dir, "tb")
    best_model_dir = os.path.join(args.log_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)

    print("=" * 60)
    print("  AntiGravSim: Reinforcement Learning Training")
    print("=" * 60)
    print(f"  Timesteps:     {args.timesteps:,}")
    print(f"  Parallel Envs: {args.n_envs}")
    print(f"  Wind Force:    {args.wind} N")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Log Directory: {args.log_dir}")
    print("=" * 60)

    # Factory for vectorized envs
    def make_env():
        return AntiGravEnv(wind_strength=args.wind, random_start=True)

    env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)
    eval_env = make_vec_env(make_env, n_envs=1, seed=args.seed + 99)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=args.log_dir,
        eval_freq=max(5000 // args.n_envs, 100),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        seed=args.seed,
        tensorboard_log=tb_log_dir,
    )

    print("\nStarting PPO optimization loop...")
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)

    final_model_path = os.path.join(args.log_dir, "final_model.zip")
    model.save(final_model_path)

    # Also copy or save directly to logs/best_model.zip for easy access
    best_model_src = os.path.join(best_model_dir, "best_model.zip")
    best_model_dst = os.path.join(args.log_dir, "best_model.zip")
    if os.path.exists(best_model_src):
        model_best = PPO.load(best_model_src)
        model_best.save(best_model_dst)
    else:
        model.save(best_model_dst)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Best Model:  {best_model_dst}")
    print(f"  Final Model: {final_model_path}")
    print(f"  TensorBoard: tensorboard --logdir {tb_log_dir}")
    print("=" * 60)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
