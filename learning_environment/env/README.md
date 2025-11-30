# Ticket Pricing RL Environment

Gymnasium-compatible environment for single-ticket pricing reinforcement learning. Each episode simulates pricing a single ticket from some time before an event until it sells or the event time arrives.

## Overview

The environment wraps a trained demand probability model to simulate realistic ticket pricing dynamics. The agent learns to adjust prices over time to maximize revenue (price - initial_price) while balancing the trade-off between price and sale probability.

## Architecture

```
┌─────────────────────────────────────────┐
│         TicketPricingEnv               │
│  (Gymnasium Environment)               │
├─────────────────────────────────────────┤
│  State: [time_norm, price_mult,        │
│          quality, is_weekend,          │
│          is_playoff]                    │
│                                         │
│  Action: Discrete(7)                   │
│  [-20%, -10%, -5%, 0%, +5%, +10%, +20%]│
│                                         │
│  Reward: price - initial_price (if sold)│
│          else 0                         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Feature Builder                    │
│  Builds 24-dim feature vector           │
│  matching demand model format           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Demand Model                       │
│  Predicts P(sale | features)            │
└─────────────────────────────────────────┘
```

## Files

### `ticket_pricing_env.py`
Main Gymnasium environment class. Implements:
- `reset()`: Initialize new episode with random parameters
- `step(action)`: Execute action, sample sale, compute reward
- `_get_obs()`: Build normalized observation vector
- `_compute_sale_probability()`: Query demand model

### `feature_builder.py`
Constructs 24-dim feature vectors from environment state. Replicates exact feature engineering logic from training to ensure compatibility with the demand model.

**Features** (in order):
1. `time_log` (1 dim)
2. `time_bin_0` through `time_bin_5` (6 dims)
3. `price_log_rel` (1 dim)
4. `price_rel` (1 dim)
5. `quality_Low`, `quality_Medium`, `quality_High`, `quality_Premium` (4 dims)
6. `is_weekend` (1 dim)
7. `is_playoff` (1 dim)
8. `day_Mon` through `day_Sun` (7 dims)
9. `price_time_interaction` (1 dim)
10. `quality_time_interaction` (1 dim)

### `test_env.py`
Test script that validates environment functionality:
- Environment initialization
- Observation space validity
- Episode termination logic
- Reward computation
- Edge case handling

## Usage

### Basic Usage

```python
from env import TicketPricingEnv
from pathlib import Path

# Create environment
env = TicketPricingEnv(
    demand_model_path=Path('models/demand_model_v1.pkl'),
    initial_price_range=(100.0, 500.0),
    quality_range=(0.0, 1.0),
    time_horizon=720.0,  # 30 days
    time_step=6.0,        # 6 hours per step
    price_bounds=(0.3, 3.0),
    random_seed=42
)

# Run one episode
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    
    print(f"Step: price=${info['current_price']:.2f}, "
          f"p_sale={info['p_sale']:.3f}, reward=${reward:.2f}")

print(f"Episode reward: ${total_reward:.2f}")
```

### Custom Episode Parameters

```python
# Fixed episode parameters (for testing)
obs, info = env.reset(
    options={
        'initial_price': 200.0,
        'quality_score': 0.6,
        'event_context': {
            'is_weekend': True,
            'is_playoff': False,
            'day_of_week': 'Sat'
        }
    }
)
```

## Environment Specifications

### Observation Space

`Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)`

Components:
- `time_remaining_norm`: Normalized time remaining (0 = event time, 1 = start)
- `price_multiplier`: Current price / initial price
- `quality_score`: Ticket quality (0-1)
- `is_weekend`: 1 if weekend, 0 otherwise
- `is_playoff`: 1 if playoff game, 0 otherwise

### Action Space

`Discrete(7)` - Percentage price changes:
- Action 0: -20%
- Action 1: -10%
- Action 2: -5%
- Action 3: 0% (keep price)
- Action 4: +5%
- Action 5: +10%
- Action 6: +20%

### Reward Function

```
reward = {
    current_price - initial_price  if ticket sold
    0                              otherwise
}
```

**Interpretation**:
- Positive reward: Sold at price higher than initial (profit)
- Negative reward: Sold at discount (loss, but ticket sold)
- Zero reward: No sale this step

### Episode Termination

- **Terminated**: Ticket sold (`terminated=True`)
- **Truncated**: Time expired without sale (`truncated=True`)

## Episode Dynamics

1. **Initialization** (`reset()`):
   - Sample `initial_price` from `initial_price_range`
   - Sample `quality_score` from `quality_range`
   - Sample `event_context` (weekend, playoff, day-of-week)
   - Set `time_remaining = time_horizon`
   - Set `current_price = initial_price`

2. **Step** (`step(action)`):
   - Apply action: `new_price = current_price * (1 + action_pct)`
   - Clip to bounds: `[initial_price * price_bounds[0], initial_price * price_bounds[1]]`
   - Build feature vector from current state
   - Query demand model: `p_sale = model.predict_proba(features)[0, 1]`
   - Sample sale: `sold ~ Bernoulli(p_sale)`
   - Compute reward: `price - initial_price` if sold, else `0`
   - Advance time: `time_remaining -= time_step`
   - Check termination

3. **Feature Construction**:
   - Map `time_remaining` to time_bin (log-scale)
   - Compute `price_rel = current_price / initial_price`
   - Map `quality_score` to quality tier
   - Extract event context features
   - Compute interaction terms
   - Stack in exact order matching model's `feature_names`

## Configuration Parameters

### `demand_model_path` (required)
Path to saved demand model (.pkl file). Model must be trained using `demand_modeling/train_model.py`.

### `initial_price_range` (default: (100.0, 500.0))
Range for sampling initial ticket price. Each episode samples uniformly from this range.

### `quality_range` (default: (0.0, 1.0))
Range for sampling ticket quality score. Quality affects demand probability.

### `time_horizon` (default: 720.0)
Maximum hours before event. Default is 30 days. Episodes start at this time and count down to 0.

### `time_step` (default: 6.0)
Hours per step. Smaller values = more steps per episode, finer control.

### `price_bounds` (default: (0.3, 3.0))
Price multiplier bounds relative to initial_price. Prevents extreme pricing:
- Minimum: `initial_price * 0.3` (70% discount)
- Maximum: `initial_price * 3.0` (200% markup)

### `random_seed` (default: None)
Random seed for reproducibility. If None, uses system time.

## Testing

Run the test script to validate environment:

```bash
python test_env.py
```

This will:
- Create environment
- Run 10 episodes
- Validate observations
- Check reward computation
- Test edge cases
- Print statistics

Expected output:
- Observations in valid range [0, 1]
- Episodes terminate correctly (sold or expired)
- Rewards computed correctly
- No errors or warnings

## Integration with RL Agents

The environment follows the standard Gymnasium API, so it works with any RL library:

### Stable-Baselines3

```python
from stable_baselines3 import DQN
from env import TicketPricingEnv

env = TicketPricingEnv(...)
model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
```

### Custom Agent

```python
class MyAgent:
    def __init__(self, env):
        self.env = env
        
    def train(self, n_episodes):
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            
            while not done:
                action = self.select_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.update(obs, action, reward, done)
```

## Key Implementation Details

### Feature Construction
The `feature_builder.py` module exactly replicates the feature engineering logic from training. This ensures:
- Features match model's expected format
- Feature order matches `model.feature_names`
- Normalization/scaling is consistent

### Price Normalization
The environment uses `initial_price` as the reference price for normalization. This is an approximation (training used event-specific medians), but works well for simulation.

### Random Episode Generation
Each `reset()` samples new episode parameters. This provides:
- Diverse training scenarios
- Better generalization
- Realistic market conditions

### Error Handling
- Validates actions are in action space
- Prevents stepping after termination
- Handles model prediction failures gracefully
- Clips prices to valid bounds

## Performance Considerations

- **Feature construction**: ~0.1ms per step
- **Model prediction**: ~1ms per step
- **Total step time**: ~1-2ms
- **Episode length**: ~120 steps on average (720h / 6h per step)

The environment is fast enough for RL training (1000+ episodes per minute).

## Limitations

1. **Single ticket**: Each episode is one ticket. Multi-ticket inventory not supported.
2. **Price normalization**: Uses `initial_price` as reference (approximation)
3. **Event context**: Randomly sampled, may not match historical distribution
4. **Deterministic demand model**: No model uncertainty (fixed probabilities)

## Future Enhancements

- Multi-ticket inventory support
- Continuous action space
- Model uncertainty/ensemble predictions
- Historical event context distribution
- Custom reward functions
- Render mode for visualization

