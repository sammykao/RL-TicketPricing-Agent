# RL Dynamic Ticket Pricing Environment - Implementation Plan

## Overview

Build a reinforcement learning environment by:
1. **Extracting demand patterns** from 50+ historical NBA events
2. **Building a synthetic demand simulator** to generate 100,000+ training episodes
3. **Creating a Gymnasium environment** for RL agent training

---

## Step 1: Extract Demand Patterns from Historical Data

Use your 50+ historical event datasets to extract:

### 1.1 Demand Curves
- **Query**: Group sales by `time_to_event` bins (e.g., hourly) and `ticket_quality`
- **Extract**: 
  - Average demand rate per hour at different price points
  - Price-demand relationship: `demand = f(price, quality, time)`
  - Demand distribution across quality tiers
- **Output**: Demand curves for Premium, High, Medium, Low quality tiers

### 1.2 Price Elasticity
- **Query**: Analyze how sales volume changes with price within each event
- **Extract**:
  - Elasticity coefficient: `ε = (ΔQ/Q) / (ΔP/P)`
  - Quality-specific elasticity (Premium less elastic, Low more elastic)
  - Time-dependent elasticity (more elastic early, less elastic near event)
- **Output**: Elasticity parameters `β(quality, time)` for demand model

### 1.3 Demand Noise/Volatility
- **Query**: Calculate variance in sales rates across similar price/time conditions
- **Extract**:
  - Stochastic noise distribution: `σ(quality, time)`
  - Demand volatility patterns (higher near event start)
- **Output**: Noise parameters for realistic demand simulation

### 1.4 Seat Quality Effects
- **Query**: Analyze how `ticket_quality` affects demand and pricing
- **Extract**:
  - Quality-tier demand multipliers
  - Price premiums by quality (Premium vs Low price ratios)
  - Quality-specific demand patterns
- **Output**: Quality tier parameters for demand model

### 1.5 Time-to-Event Effects
- **Query**: Analyze demand patterns as `time_to_event` decreases
- **Extract**:
  - Demand surge patterns (last 24h, last 6h before event)
  - Urgency effects (demand increases as event approaches)
  - Early vs late buyer price sensitivity
- **Output**: Time-dependent demand multipliers: `f(time_to_event)`

**Implementation**:
```python
# Query database to extract patterns
demand_patterns = {
    'base_demand': {quality: λ_base},
    'price_elasticity': {quality: β},
    'time_factor': f(time_to_event),
    'noise_std': {quality: σ},
    'quality_multipliers': {quality: multiplier}
}
```

---

## Step 2: Build Synthetic Demand Simulator

Create a simulator that generates 100,000+ realistic event episodes.

### 2.1 Event Generator
- Sample event characteristics from historical distributions:
  - Total inventory (15,000 - 20,000 seats)
  - Quality distribution (match historical: 25% Premium, 35% High, 33% Medium, 7% Low)
  - Event type (regular season, playoffs, rivalry)
  - Day of week, time of day
  - Initial prices per quality tier

### 2.2 Demand Model
Use extracted parameters to simulate realistic demand:

```python
demand(t, price, quality) = (
    base_demand[quality] 
    * exp(-elasticity[quality] * price)
    * time_factor(time_to_event)
    * quality_multiplier[quality]
    + noise(0, σ[quality])
)
```

### 2.3 Episode Generation
For each synthetic event:
- **Initialize**: Inventory, prices, time horizon (e.g., 168 hours = 1 week)
- **Simulate**: For each time step:
  - Calculate demand based on current price, quality, time
  - Sample sales: `sales = min(demand, inventory)`
  - Update inventory
  - Track revenue
- **Output**: Episode trajectory (states, actions, rewards)

### 2.4 What This Gives the RL Agent

By training on 100,000+ synthetic episodes, the RL agent learns:

- **Baseline Strategy**: Understand fundamental pricing principles
- **Dynamic Time-Aware Behaviors**: Adapt pricing as event approaches
- **Correct Price Adjustments**: Learn when to raise/lower prices
- **Sensitivity to Seat Quality**: Different strategies for Premium vs Low tickets
- **Understanding of Too-High/Too-Low Prices**: Learn price boundaries that kill demand or leave money on table

---

## Step 3: Build Gymnasium Environment

Create a standard RL environment interface.

### 3.1 State Space
```python
state = [
    time_remaining,        # Normalized: 0 = event time, 1 = start
    inventory_premium,     # Remaining / initial
    inventory_high,
    inventory_medium,
    inventory_low,
    price_premium,         # Current / max_historical
    price_high,
    price_medium,
    price_low,
    demand_rate,          # Recent sales rate
    revenue_so_far,        # Cumulative / max_possible
]
```

### 3.2 Action Space
**Discrete actions** (start simple):
```python
actions = [
    'decrease_5%',   # -5% price change
    'decrease_10%',  # -10% price change
    'keep_price',    # No change
    'increase_5%',   # +5% price change
    'increase_10%',  # +10% price change
]
```

**Or multi-tier** (more realistic):
```python
actions = {
    'premium': price_change,
    'high': price_change,
    'medium': price_change,
    'low': price_change,
}
```

### 3.3 Reward Function
```python
reward = (
    revenue_gained              # Immediate revenue
    + sellout_bonus             # +1000 if inventory = 0
    - unsold_penalty            # -50 × remaining_inventory
    - volatility_penalty        # -10 × |price_change| if > 20%
)
```

### 3.4 Environment Dynamics
```python
def step(action):
    # 1. Update prices based on action
    # 2. Sample demand from demand model
    # 3. Sales = min(demand, inventory)
    # 4. Update inventory, revenue
    # 5. Calculate reward
    # 6. Check termination (inventory=0 or time=0)
    return next_state, reward, done, info
```

---

## Implementation Structure

```
learning_environment/
├── demand_modeling/
│   ├── __init__.py
│   ├── extract_patterns.py      # Step 1: Query DB, extract patterns
│   └── fit_demand_model.py        # Fit demand curves, elasticity
│
├── synthetic_data/
│   ├── __init__.py
│   ├── event_generator.py         # Generate synthetic events
│   ├── demand_simulator.py       # Simulate demand using fitted model
│   └── episode_generator.py      # Create 100k+ episodes
│
└── env/
    ├── __init__.py
    ├── ticket_pricing_env.py     # Gymnasium environment
    └── reward_calculator.py       # Reward function
```

---

## Implementation Order

### Week 1: Extract Patterns (Step 1)
1. **`extract_patterns.py`**: Query SQLite, aggregate sales by time/quality/price
2. **`fit_demand_model.py`**: Fit demand curves, estimate elasticity, noise
3. **Validate**: Compare fitted model to historical data

### Week 2: Build Simulator (Step 2)
1. **`event_generator.py`**: Sample event characteristics
2. **`demand_simulator.py`**: Implement demand model with extracted parameters
3. **`episode_generator.py`**: Generate 100,000+ episodes
4. **Validate**: Check synthetic episodes match historical patterns

### Week 3: Build Environment (Step 3)
1. **`ticket_pricing_env.py`**: Gymnasium environment class
2. **`reward_calculator.py`**: Reward function
3. **Test**: Run random agent, validate environment works

### Week 4: Integration & Testing
1. Connect environment to synthetic data generator
2. Test with simple RL agents (random, greedy)
3. Benchmark against baselines

---

## Key Design Decisions

### Start Simple, Expand Later
- **Single quality tier** → Multi-tier later
- **Discrete actions** → Continuous later
- **Simple demand model** → Complex later

### Demand Model Formula
```python
# Exponential demand model (simple, interpretable)
demand = λ_base * exp(-β * price) * f(time) + N(0, σ)

# Where:
# λ_base = base demand rate (quality-specific)
# β = price elasticity (quality-specific)
# f(time) = time-dependent multiplier
# σ = demand noise
```

### Episode Configuration
- **Time horizon**: 168 hours (1 week)
- **Time step**: 1 hour
- **Initial inventory**: 10,000-20,000 tickets
- **Quality distribution**: Match historical (25/35/33/7)

---

## Success Criteria

### Demand Model Quality
- R² > 0.7 on test events
- Price elasticity matches economic theory
- Time effects capture urgency patterns

### Synthetic Data Quality
- 100,000+ diverse episodes
- Episodes match historical price/revenue distributions
- Realistic demand patterns

### Environment Quality
- Runs at >1000 steps/second
- Random agent achieves expected baseline
- Reward function encourages good behavior

### RL Agent Performance
- Revenue > 10% above static pricing
- Sellout rate > 80%
- Learns time-aware pricing strategies

---

## Next Steps

1. **Start with Step 1**: Extract demand patterns from your database
2. **Validate patterns**: Ensure they make economic sense
3. **Build simulator**: Generate synthetic episodes
4. **Create environment**: Gymnasium interface
5. **Train RL agent**: Use your favorite RL algorithm (DQN, PPO, etc.)

---

## Resources

- **Data**: Your SQLite database with 50+ events, 92k+ sales
- **Libraries**: `gymnasium`, `numpy`, `pandas`, `scikit-learn`, `scipy`
- **Time**: ~3 weeks for complete implementation

This plan gives you a clear path from historical data → synthetic simulator → RL environment!
