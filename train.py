"""
train.py — PPO training script for SmartGridEnv

Fixes vs. original:
    - check_env() validates the environment before training starts
    - VecNormalize auto-normalizes observations and rewards for stable gradients
    - 500,000 timesteps (was 10,000 — far too few for PPO to learn anything)
    - EvalCallback saves the best model checkpoint automatically
    - Hyperparameters tuned for this problem (n_steps, batch_size, ent_coef)
    - vec_normalize stats saved alongside model (required for correct inference)
    - TensorBoard logging enabled (optional — run: tensorboard --logdir ./tb_logs)
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env

from smart_grid_env import SmartGridEnv

# ── 1. Validate environment ───────────────────────────────────────────────────
print("Checking environment...")
check_env(SmartGridEnv(), warn=True)
print("Environment check passed.\n")

# ── 2. Vectorised training environment (4 parallel workers) ───────────────────
N_ENVS = 4
train_env = make_vec_env(SmartGridEnv, n_envs=N_ENVS)
train_env = VecNormalize(
    train_env,
    norm_obs=True,      # normalizes each obs dimension to ~N(0,1)
    norm_reward=True,   # normalizes reward scale — critical for PPO stability
    clip_obs=10.0,
)

# ── 3. Separate evaluation environment (no reward normalisation) ───────────────
eval_env = make_vec_env(SmartGridEnv, n_envs=1)
eval_env = VecNormalize(
    eval_env,
    norm_obs=True,
    norm_reward=False,  # raw rewards for interpretable eval metrics
    training=False,     # stats are copied from train_env, not updated
    clip_obs=10.0,
)

# ── 4. Define the PPO model ────────────────────────────────────────────────────
model = PPO(
    policy          = "MlpPolicy",
    env             = train_env,
    verbose         = 1,
    tensorboard_log = "./tb_logs",
    # --- Core PPO hyperparameters ---
    n_steps         = 1024,     # steps collected per env per rollout
    batch_size      = 256,      # minibatch size for gradient update
    n_epochs        = 10,       # number of passes over each rollout buffer
    gamma           = 0.99,     # discount factor (long-horizon cost matters)
    gae_lambda      = 0.95,     # GAE smoothing
    clip_range      = 0.2,      # PPO clip parameter
    learning_rate   = 3e-4,     # Adam lr
    ent_coef        = 0.01,     # entropy bonus (encourages exploration early on)
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    # --- Policy network architecture ---
    policy_kwargs   = dict(net_arch=[128, 128]),  # 2-layer MLP, 128 units each
)

# ── 5. Callbacks ───────────────────────────────────────────────────────────────
os.makedirs("./best_model",   exist_ok=True)
os.makedirs("./checkpoints",  exist_ok=True)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path = "./best_model",
    log_path             = "./eval_logs",
    eval_freq            = max(5_000 // N_ENVS, 1),  # evaluate every ~5k env steps
    n_eval_episodes      = 20,      # average over 20 full 24-hour episodes
    deterministic        = True,
    render               = False,
)

checkpoint_callback = CheckpointCallback(
    save_freq  = max(50_000 // N_ENVS, 1),
    save_path  = "./checkpoints",
    name_prefix= "ppo_smart_grid",
)

# ── 6. Train ───────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 500_000
print(f"Training PPO for {TOTAL_TIMESTEPS:,} timesteps across {N_ENVS} parallel envs...")
print("Tip: run `tensorboard --logdir ./tb_logs` to monitor training live.\n")

model.learn(
    total_timesteps = TOTAL_TIMESTEPS,
    callback        = [eval_callback, checkpoint_callback],
    progress_bar    = True,
)

# ── 7. Save final model + normalisation statistics ────────────────────────────
model.save("ppo_smart_grid")
train_env.save("vec_normalize.pkl")   # MUST be saved — needed for inference

print("\nTraining complete!")
print("  Saved: ppo_smart_grid.zip")
print("  Saved: vec_normalize.pkl  (required alongside the model for inference)")
print("  Best checkpoint: ./best_model/best_model.zip")

# ── 8. Quick sanity-check: run one episode with the trained agent ──────────────
print("\n--- Sanity check: one 24-hour episode ---")
from stable_baselines3.common.vec_env import DummyVecEnv

test_env = DummyVecEnv([SmartGridEnv])
test_env = VecNormalize.load("vec_normalize.pkl", test_env)
test_env.training = False
test_env.norm_reward = False

obs = test_env.reset()
total_cost = 0.0
for hour in range(24):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    total_cost += info[0]["cost"]
    action_label = ["Hold", "Charge", "Discharge"][int(action[0])]
    print(
        f"  Hour {hour+1:02d} | Action: {action_label:<10} | "
        f"Battery: {info[0]['battery_soc']:5.1f}% | "
        f"Price: ${info[0]['price']:.3f} | "
        f"Step cost: ${info[0]['cost']:.3f}"
    )

print(f"\nTotal 24-hour cost: ${total_cost:.2f}")
test_env.close()
