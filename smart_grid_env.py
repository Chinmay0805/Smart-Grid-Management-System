import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SmartGridEnv(gym.Env):
    """
    Smart Grid Battery Management Environment (Gymnasium-compatible)

    Goal: Minimize daily electricity cost by intelligently charging/discharging
          a home battery, using solar generation, and interacting with the grid.

    Observation (4 values, all normalized to [0, 1]):
        [battery_soc, solar_gen_kw, house_demand_kw, grid_price]

    Action space (Discrete 3):
        0 = Hold   — do nothing beyond covering net load from grid/solar
        1 = Charge — buy from grid to fill battery (10 kW rate)
        2 = Discharge — draw from battery to cover load (10 kW rate)

    Reward:
        Negative net cost per step (agent learns to minimize cost).
        Includes sell-back revenue when solar surplus is fed to grid.
        Includes a small battery health penalty for extreme SoC operation.

    Fixes vs. original:
        - Observation space bounds match actual value ranges
        - Solar/demand energy balance applied in all action branches
        - Grid sell-back (feed-in tariff) modeled in HOLD branch
        - Battery SoC clamped to [0, battery_capacity]
        - Charging efficiency loss (90%) modeled
        - Correlated time-series for price/solar/demand (no more i.i.d. jumps)
        - Battery health penalty for operating near 0% or 100% SoC
        - render() method added
        - Fully compatible with check_env() and VecNormalize
    """

    metadata = {"render_modes": ["human"]}

    # Physical constants
    BATTERY_CAPACITY  = 100.0   # kWh
    CHARGE_RATE       = 10.0    # kW  (max charge/discharge per step)
    CHARGE_EFFICIENCY = 0.90    # 90% round-trip efficiency
    FEED_IN_TARIFF    = 0.50    # sell surplus solar at 50% of grid price
    SOC_PENALTY_COEF  = 0.005   # small penalty for extreme battery levels

    # Observation high limits (battery 0-100 %, solar 0-10 kW,
    # demand 0-10 kW, price 0-1.5 $/kWh to cover double-peak)
    OBS_HIGH = np.array([100.0, 10.0, 10.0, 1.5], dtype=np.float32)
    OBS_LOW  = np.zeros(4, dtype=np.float32)

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # --- Action / Observation Spaces ---
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=self.OBS_LOW,
            high=self.OBS_HIGH,
            shape=(4,),
            dtype=np.float32,
        )

        # Internal state
        self.current_step    = 0
        self.current_battery = self.BATTERY_CAPACITY * 0.5
        self._state          = self._make_initial_state()

        # For correlated time-series generation
        self._price_base   = 0.2   # $/kWh (drifts each step)
        self._demand_base  = 3.0   # kW
        self._solar_base   = 0.0   # kW

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step    = 0
        self.current_battery = self.BATTERY_CAPACITY * 0.5
        self._price_base     = 0.2
        self._demand_base    = 3.0
        self._solar_base     = 0.0
        self._state = self._make_initial_state()
        return self._state.copy(), {}

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"

        battery, solar, demand, price = self._state

        # ---- 1. Compute net load BEFORE battery action ----
        # Positive  → house needs more than solar provides (must buy or discharge)
        # Negative  → solar surplus (can sell back or charge)
        net_load = demand - solar

        grid_cost = 0.0   # positive = paying, negative = earning

        # ---- 2. Execute battery action ----
        if action == 1:  # CHARGE from grid
            # How much can we actually charge?
            headroom = self.BATTERY_CAPACITY - battery
            charge_requested = min(self.CHARGE_RATE, headroom)
            # Efficiency: we buy more from grid than actually stored
            grid_draw = charge_requested / self.CHARGE_EFFICIENCY
            self.current_battery = np.clip(
                battery + charge_requested,
                0.0, self.BATTERY_CAPACITY
            )
            # Also cover net_load from grid
            grid_cost = (max(0.0, net_load) + grid_draw) * price
            # If solar surplus even after load, get sell-back credit
            grid_cost -= max(0.0, -net_load) * price * self.FEED_IN_TARIFF

        elif action == 2:  # DISCHARGE battery to cover load
            # How much battery can supply?
            discharge_requested = min(self.CHARGE_RATE, battery)
            self.current_battery = np.clip(
                battery - discharge_requested,
                0.0, self.BATTERY_CAPACITY
            )
            # Remaining load after battery contribution
            residual_load = net_load - discharge_requested
            if residual_load > 0:
                grid_cost = residual_load * price          # still need some grid
            else:
                grid_cost = residual_load * price * self.FEED_IN_TARIFF  # surplus → sell

        else:  # HOLD — let solar + grid balance the load
            if net_load > 0:
                grid_cost = net_load * price               # buy deficit from grid
            else:
                grid_cost = net_load * price * self.FEED_IN_TARIFF  # sell surplus

        # ---- 3. Battery health penalty (discourages extreme SoC) ----
        soc_frac = self.current_battery / self.BATTERY_CAPACITY
        health_penalty = self.SOC_PENALTY_COEF * (
            max(0.0, soc_frac - 0.9) + max(0.0, 0.1 - soc_frac)
        )

        # ---- 4. Reward ----
        reward = -(grid_cost + health_penalty)

        # ---- 5. Advance time, generate next state ----
        self.current_step += 1
        terminated = self.current_step >= 24
        truncated  = False

        if not terminated:
            self._state = self._generate_next_state()
        else:
            self._state = np.zeros(4, dtype=np.float32)

        info = {
            "cost":        float(grid_cost),
            "battery_soc": float(self.current_battery),
            "solar_kw":    float(solar),
            "demand_kw":   float(demand),
            "price":       float(price),
            "action":      int(action),
        }

        if self.render_mode == "human":
            self.render()

        return self._state.copy(), float(reward), terminated, truncated, info

    def render(self):
        b, s, d, p = self._state
        print(
            f"[Hour {self.current_step:02d}] "
            f"Battery={self.current_battery:.1f}% | "
            f"Solar={s:.2f}kW | Demand={d:.2f}kW | Price=${p:.3f}/kWh"
        )

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_initial_state(self) -> np.ndarray:
        return np.array(
            [self.current_battery, 0.0, 2.5, 0.15],
            dtype=np.float32
        )

    def _generate_next_state(self) -> np.ndarray:
        """
        Correlated time-series generation so successive hours are smooth.
        Each variable drifts toward a time-of-day mean with Gaussian noise.
        """
        hour = self.current_step  # 0–23

        # --- Solar: bell-curve peaking at noon, zero at night ---
        if 6 <= hour <= 18:
            solar_mean = 5.0 * np.exp(-0.5 * ((hour - 12) / 3.5) ** 2)
        else:
            solar_mean = 0.0
        self._solar_base += 0.3 * (solar_mean - self._solar_base)
        next_solar = float(np.clip(
            self._solar_base + self.np_random.normal(0, 0.4),
            0.0, 10.0
        ))

        # --- Demand: morning and evening peaks ---
        demand_mean = (
            2.5
            + 2.0 * np.exp(-0.5 * ((hour - 8)  / 1.5) ** 2)   # morning
            + 3.0 * np.exp(-0.5 * ((hour - 19) / 2.0) ** 2)   # evening
        )
        self._demand_base += 0.3 * (demand_mean - self._demand_base)
        next_demand = float(np.clip(
            self._demand_base + self.np_random.normal(0, 0.5),
            0.0, 10.0
        ))

        # --- Price: cheap at night, expensive at peak (17–21) ---
        if 17 <= hour <= 21:
            price_mean = 0.65
        elif 6 <= hour <= 9:
            price_mean = 0.30
        elif 23 <= hour or hour <= 5:
            price_mean = 0.12
        else:
            price_mean = 0.22
        self._price_base += 0.4 * (price_mean - self._price_base)
        next_price = float(np.clip(
            self._price_base + self.np_random.normal(0, 0.03),
            0.05, 1.5
        ))

        return np.array(
            [self.current_battery, next_solar, next_demand, next_price],
            dtype=np.float32
        )
