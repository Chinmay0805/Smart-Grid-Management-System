"""
app.py — Streamlit dashboard for SmartGridEnv PPO agent

Fixes vs. original:
    - Loads VecNormalize stats (vec_normalize.pkl) alongside the PPO model
      so observations are correctly normalised at inference time
    - int(action.item()) fixes numpy array comparison in action_to_text()
    - Added a rule-based baseline agent for comparison
    - Richer charts: cost-per-hour bar chart + solar/demand/battery area chart
    - Step-level info table logged per episode
    - Graceful error handling throughout
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from smart_grid_env import SmartGridEnv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart Grid AI Control", layout="wide")
st.title("Smart Grid Energy Management System")
st.markdown("### PPO Reinforcement Learning Agent vs. Rule-Based Baseline")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Simulation Settings")
sim_speed   = st.sidebar.slider("Speed (sec / step)", 0.05, 2.0, 0.3)
agent_choice = st.sidebar.radio(
    "Agent to run",
    ["PPO Agent", "Rule-Based Baseline", "Compare Both"],
)
run_btn = st.sidebar.button("▶ Start 24-Hour Simulation", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Rule-based logic:** charge when price < 0.20 and battery < 80%, "
    "discharge when price > 0.40 and battery > 20%, else hold."
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def action_to_label(action: int) -> str:
    return {0: "Hold ⏸", 1: "Charge ⬆", 2: "Discharge ⬇"}.get(action, "?")


def rule_based_action(obs: np.ndarray) -> int:
    """Simple price-threshold rule — useful as a sanity-check baseline."""
    battery, solar, demand, price = obs
    if price < 0.20 and battery < 80.0:
        return 1   # cheap electricity → charge
    if price > 0.40 and battery > 20.0:
        return 2   # expensive electricity → use battery
    return 0       # hold


def load_ppo_model():
    """Load trained PPO model + normalisation stats. Returns (model, vec_env) or None."""
    if not os.path.exists("ppo_smart_grid.zip"):
        return None, None
    try:
        env = DummyVecEnv([SmartGridEnv])
        if os.path.exists("vec_normalize.pkl"):
            env = VecNormalize.load("vec_normalize.pkl", env)
            env.training    = False
            env.norm_reward = False
        model = PPO.load("ppo_smart_grid", env=env)
        return model, env
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None, None


def run_episode(agent: str, model=None, vec_env=None, speed: float = 0.3):
    """
    Run a single 24-step episode and return a DataFrame of step-level data.
    agent: 'ppo' | 'rule'
    """
    raw_env = SmartGridEnv()
    obs_raw, _ = raw_env.reset()

    # PPO uses the normalised vec_env; rule-based uses raw env directly
    if agent == "ppo" and vec_env is not None:
        obs_vec = vec_env.reset()

    records = []
    total_cost = 0.0

    live_battery = st.empty()
    live_price   = st.empty()
    live_cost    = st.empty()
    live_chart   = st.empty()

    for step in range(24):
        # ---- Pick action ----
        if agent == "ppo" and model is not None:
            action_arr, _ = model.predict(obs_vec, deterministic=True)
            action = int(action_arr.item())        # numpy → plain int
            obs_vec, _, _, _ = vec_env.step(action_arr)
            # Step the raw env with the same action to get proper info dict
            obs_raw, reward, terminated, _, info = raw_env.step(action)
        else:
            action = rule_based_action(obs_raw)
            obs_raw, reward, terminated, _, info = raw_env.step(action)

        cost = info["cost"]
        total_cost += cost

        battery = info["battery_soc"]
        solar   = info["solar_kw"]
        demand  = info["demand_kw"]
        price   = info["price"]

        # ---- Live metrics ----
        col1, col2, col3 = live_battery, live_price, live_cost
        live_battery.metric("🔋 Battery SoC",  f"{battery:.1f} %",  action_to_label(action))
        live_price.metric(  "💲 Grid Price",    f"${price:.3f}/kWh")
        live_cost.metric(   "💰 Running Cost",  f"${total_cost:.2f}", delta_color="inverse")

        records.append({
            "Hour":        step + 1,
            "Battery (%)": round(battery, 2),
            "Solar (kW)":  round(solar, 2),
            "Demand (kW)": round(demand, 2),
            "Price ($/kWh)": round(price, 3),
            "Step Cost ($)": round(cost, 3),
            "Action":      action_to_label(action),
        })

        # ---- Live chart (updates every step) ----
        df_so_far = pd.DataFrame(records).set_index("Hour")
        live_chart.line_chart(
            df_so_far[["Battery (%)", "Solar (kW)", "Demand (kW)"]],
            height=250,
        )

        time.sleep(speed)

    raw_env.close()
    return pd.DataFrame(records), total_cost


def show_results(df: pd.DataFrame, total_cost: float, label: str):
    st.success(f"**{label}** — Total 24-hour cost: **${total_cost:.2f}**")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Hourly step cost ($)")
        st.bar_chart(df.set_index("Hour")[["Step Cost ($)"]], height=220)
    with col_b:
        st.subheader("Battery, solar and demand")
        st.line_chart(
            df.set_index("Hour")[["Battery (%)", "Solar (kW)", "Demand (kW)"]],
            height=220,
        )

    with st.expander("📋 Full step-by-step log"):
        st.dataframe(df, use_container_width=True)


# ── Main simulation ───────────────────────────────────────────────────────────
if run_btn:
    model, vec_env = load_ppo_model()
    ppo_available  = model is not None

    if not ppo_available and agent_choice in ("PPO Agent", "Compare Both"):
        st.warning(
            "ppo_smart_grid.zip not found — run `python train.py` first. "
            "Falling back to rule-based agent."
        )

    # ---- PPO only ----
    if agent_choice == "PPO Agent":
        st.markdown("### PPO Agent")
        agent = "ppo" if ppo_available else "rule"
        df, cost = run_episode(agent, model, vec_env, sim_speed)
        show_results(df, cost, "PPO Agent" if ppo_available else "Rule-Based (fallback)")

    # ---- Rule-based only ----
    elif agent_choice == "Rule-Based Baseline":
        st.markdown("###  Rule-Based Baseline")
        df, cost = run_episode("rule", speed=sim_speed)
        show_results(df, cost, "Rule-Based Baseline")

    # ---- Compare both ----
    else:
        tab_ppo, tab_rule = st.tabs([" PPO Agent", " Rule-Based Baseline"])

        with tab_ppo:
            st.markdown("#### PPO Agent — running...")
            agent = "ppo" if ppo_available else "rule"
            df_ppo, cost_ppo = run_episode(agent, model, vec_env, sim_speed)
            show_results(df_ppo, cost_ppo, "PPO Agent" if ppo_available else "Rule-Based (fallback)")

        with tab_rule:
            st.markdown("#### Rule-Based Baseline — running...")
            df_rule, cost_rule = run_episode("rule", speed=sim_speed)
            show_results(df_rule, cost_rule, "Rule-Based Baseline")

        # ---- Side-by-side cost summary ----
        st.markdown("---")
        st.subheader("Cost comparison")
        c1, c2, c3 = st.columns(3)
        c1.metric("PPO total cost",       f"${cost_ppo:.2f}")
        c2.metric("Rule-based total cost", f"${cost_rule:.2f}")
        saving = cost_rule - cost_ppo
        c3.metric("PPO saving",           f"${saving:.2f}",
                  delta=f"{'better' if saving > 0 else 'worse'} than rule-based",
                  delta_color="normal" if saving > 0 else "inverse")

else:
    st.info("Configure settings in the sidebar and click **▶ Start 24-Hour Simulation**.")
    st.markdown("""
    #### How it works
    | Component | Detail |
    |---|---|
    | Environment | 24-step episode (1 step = 1 hour) |
    | Observation | Battery SoC, solar generation, house demand, grid price |
    | Actions | Hold / Charge from grid / Discharge battery |
    | Reward | Negative net grid cost (includes solar sell-back revenue) |
    | Agent | PPO with MLP policy, trained via `stable-baselines3` |
    | Baseline | Simple price-threshold rule for comparison |
    """)
