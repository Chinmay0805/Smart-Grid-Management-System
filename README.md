# Smart Grid Energy Management System
### Reinforcement Learning (PPO) for Home Battery Optimisation

A complete end-to-end project that trains an AI agent to manage a home battery storage system — deciding every hour whether to **charge from the grid**, **discharge to the house**, or **hold** — in order to minimise electricity costs over a 24-hour day.

---

##  The Core Idea 

Imagine you have:
- A **home battery**
- **Solar panels** on your roof
- An electricity **grid** where prices change every hour (cheap at night, expensive at peak)

The question is: **When should you charge the battery? When should you use it?**

This project trains an AI using **Reinforcement Learning** to figure out the optimal strategy — charge cheaply at night, discharge during expensive peak hours, and use free solar power whenever available.

The AI learns by trial-and-error over 500,000 simulated hours, getting a "reward" every time it saves money and a "penalty" every time it wastes money.

---

##  Project Structure

```
smart-grid-ppo/
│
├── smart_grid_env.py   # The simulated world (environment)
├── train.py            # Trains the AI agent
├── app.py              # Streamlit dashboard (live demo)
│
├── ppo_smart_grid.zip  # Saved model (generated after training)
├── vec_normalize.pkl   # Observation normalisation stats (generated after training)
│
├── best_model/         # Best checkpoint saved during training
├── checkpoints/        # Periodic checkpoints
├── tb_logs/            # TensorBoard training logs
└── eval_logs/          # Evaluation logs
```

---

##  How Each File Works

### 1. `smart_grid_env.py` : The Simulated World

This file defines the **environment** — the world the AI lives and learns in. It follows the standard [Gymnasium](https://gymnasium.farama.org/) interface, which is the universal standard for RL environments.

**What the AI sees every hour (observations):**

| Variable | Range | Meaning |
|---|---|---|
| `battery_soc` | 0–100 % | How full the battery currently is |
| `solar_gen_kw` | 0–10 kW | How much solar power is being generated right now |
| `house_demand_kw` | 0–10 kW | How much electricity the house needs right now |
| `grid_price` | $0.05–1.50/kWh | Current electricity price from the grid |

**What the AI can do (actions):**

| Action | Code | What happens |
|---|---|---|
| Hold | `0` | Do nothing — buy/sell only what the house needs |
| Charge | `1` | Buy electricity from the grid to fill the battery |
| Discharge | `2` | Use battery power to run the house, avoid buying from grid |

**Energy balance logic (the physics):**

Every step, solar power covers house demand first. Only the leftover gap is handled by the battery or grid:

```
net_load = house_demand - solar_generation

If net_load > 0 → house needs more than solar provides → buy from grid or discharge battery
If net_load < 0 → solar surplus → sell back to grid (at 50% feed-in tariff) or charge battery
```

**Reward signal:**

```
reward = -(grid_cost + battery_health_penalty)
```

The agent gets a negative reward (penalty) when it spends money and a positive reward (savings) when it avoids buying expensive electricity. A small health penalty discourages operating the battery at extreme levels (below 10% or above 90% SoC).

**Realistic time-series generation:**

Rather than picking random prices/solar/demand each hour (which would be unrealistic), the environment generates smooth, correlated values:
- Solar follows a **bell curve peaking at noon** and zero at night
- Demand has **morning (8am) and evening (7pm) peaks**
- Prices are **cheapest overnight**, spike during **peak hours (5–9pm)**

---

###  `train.py` 

This file runs the actual training loop. It uses [Stable-Baselines3](https://stable-baselines3.readthedocs.io/), a high-quality RL library built on PyTorch.

**Step-by-step what happens:**

```
1. check_env()           → validates the environment catches any bugs before wasting compute
2. make_vec_env()        → creates 4 parallel environments (4x faster data collection)
3. VecNormalize()        → automatically normalises observations to ~N(0,1) for stable training
4. PPO(...)              → defines the agent with a 2-layer MLP neural network (128 × 128)
5. EvalCallback()        → evaluates the agent every 5,000 steps, saves the best version
6. model.learn(500_000)  → runs 500,000 environment steps of training
7. model.save(...)       → saves the trained model + normalisation statistics
8. Sanity check          → runs one full 24-hour episode and prints results
```

**Why PPO?**

PPO (Proximal Policy Optimisation) is the industry standard for this type of problem. It's stable, sample-efficient, and handles discrete action spaces well. Key hyperparameters used:

| Parameter | Value | Why |
|---|---|---|
| `n_steps` | 1024 | Steps collected per environment per rollout |
| `batch_size` | 256 | Minibatch size for gradient updates |
| `n_epochs` | 10 | Passes over each rollout buffer |
| `ent_coef` | 0.01 | Entropy bonus — encourages exploration early in training |
| `gamma` | 0.99 | Discount factor — agent cares about future costs, not just immediate |
| `net_arch` | [128, 128] | Two hidden layers of 128 neurons each |

**Why VecNormalize?**

Battery SoC is 0–100, but price is 0.05–1.50. These wildly different scales confuse the neural network. `VecNormalize` automatically scales every observation to have mean ≈ 0 and std ≈ 1 during training, which dramatically stabilises gradient descent.

---

### 3. `app.py` — The Streamlit Dashboard

This file is the **live demo interface**. Run it after training to watch the agent make decisions in real time.

**What it does:**

1. Loads the trained PPO model + normalisation stats (`vec_normalize.pkl`)
2. Runs a 24-step (24-hour) episode
3. Updates live metrics every step: battery level, current price, running cost
4. Plots real-time charts: battery/solar/demand over time, hourly step costs
5. Also runs a **rule-based baseline** for comparison (charge when price < $0.20, discharge when price > $0.40)
6. Shows a side-by-side cost comparison of PPO vs baseline

**Three modes (selectable in sidebar):**
- `PPO Agent` — runs the trained AI only
- `Rule-Based Baseline` — runs the simple rule-based agent only
- `Compare Both` — runs both sequentially and shows savings

---

## 🚀 How to Run

### Prerequisites

```bash
pip install stable-baselines3 gymnasium streamlit pandas numpy
```

### Step 1 — Train the agent

```bash
python train.py
```

This takes ~5–10 minutes. You'll see progress printed to the terminal. After training, two files are created:
- `ppo_smart_grid.zip` — the trained neural network
- `vec_normalize.pkl` — the observation normalisation statistics

> **Important:** Both files must be kept together. The model won't work correctly without `vec_normalize.pkl`.

### Step 2 — Run the dashboard

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`. Click **▶ Start 24-Hour Simulation** in the sidebar.

### (Optional) Step 3 — Monitor training with TensorBoard

```bash
tensorboard --logdir ./tb_logs
```

Open `http://localhost:6006` to see live training curves (reward, loss, entropy) during or after training.

---

## 📊 Understanding the Results

After training, a well-trained agent should:

- **Charge at night** (hours 22:00–06:00) when grid price is cheapest (~$0.12/kWh)
- **Use solar** during the day (hours 06:00–18:00) to cover demand for free
- **Discharge during peak hours** (17:00–21:00) when grid price is highest (~$0.65/kWh), avoiding expensive purchases
- **Outperform the rule-based baseline** by at least 15–25% on total daily cost

---

## 🏗️ Architecture Summary

```
Observation (4 values)
        │
        ▼
  PPO Neural Network
  (MLP: 128 → 128 → 3)
        │
        ▼
  Action (0 / 1 / 2)
        │
        ▼
  SmartGridEnv.step()
  ┌─────────────────────┐
  │ 1. Solar covers load │
  │ 2. Apply action      │
  │ 3. Grid interaction  │
  │ 4. Compute reward    │
  │ 5. Advance time      │
  └─────────────────────┘
        │
        ▼
  Next observation + reward
        │
        ▼
  PPO updates policy weights
  (repeat 500,000 times)
```

---

## 🔧 Key Design Decisions

| Decision | Reason |
|---|---|
| Discrete action space (3 actions) | Simple enough for a demo; real systems use continuous control |
| 24-step episodes | One step = one hour, one episode = one day |
| Feed-in tariff at 50% | Models real-world net metering — selling solar back isn't free money |
| 90% charging efficiency | Batteries lose energy during charge/discharge cycles |
| Battery health penalty | Prevents the agent from thrashing the battery at 0% or 100% |
| Correlated time-series | Makes the problem realistic — prices don't teleport randomly |
| VecNormalize | Mandatory for stable PPO training with mixed-scale observations |

---

## 🔭 Possible Extensions

- **Continuous action space** — instead of "charge/hold/discharge", let the agent pick exactly how many kW to charge/discharge (use `Box` action space with SAC or TD3 algorithm)
- **Real price data** — replace the simulated price curve with actual historical electricity prices (e.g. AEMO data for Australia, ENTSO-E for Europe)
- **Multi-day memory** — add a time-of-day encoding to the observation so the agent learns weekly patterns
- **Multiple batteries or appliances** — extend to control EV charging, heat pumps, and grid export simultaneously
- **Model predictive control comparison** — compare the RL agent against a classical MPC solver

---

## 📚 References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PPO Paper — Schulman et al. 2017](https://arxiv.org/abs/1707.06347)
- [Home Battery Optimisation with RL — survey](https://arxiv.org/abs/2301.01913)