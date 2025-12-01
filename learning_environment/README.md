# Learning Environment for Ticket Pricing RL

This module provides a complete pipeline for training reinforcement learning agents on dynamic ticket pricing. It includes data processing, demand curve modeling, and a Gymnasium-compatible environment for RL training.

## Overview

The learning environment consists of three main components:

1. **Data Generation** (`data_generation/`): Import and process historical ticket sales data into SQLite
2. **Demand Modeling** (`demand_modeling/`): Fit a probability model for `P(sale | price, time, quality, context)`
3. **RL Environment** (`env/`): Gymnasium-compatible environment that simulates ticket pricing dynamics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA GENERATION                          │
│  CSV Files → SQLite DB (53 events, 92k sales)               │
│  - Import sales data                                        │
│  - Compute ticket quality scores                            │
│  - Store in normalized schema                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   DEMAND MODELING                           │
│  SQLite DB → Aggregated Data → Features → Model            │
│  - Extract and bin sales data                              │
│  - Engineer features (24 dimensions)                       │
│  - Fit logistic regression model                           │
│  - Validate and save model                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    RL ENVIRONMENT                           │
│  Demand Model → Gymnasium Environment                       │
│  - Single-ticket pricing MDP                                │
│  - Random episode generation                                │
│  - Reward = price - initial_price                          │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
learning_environment/
├── data_generation/          # Data import and processing
│   ├── data/                 # Raw CSV files (53 events)
│   ├── db/                   # Database schema definitions
│   ├── import_utils/         # CSV processing utilities
│   ├── db.sqlite            # SQLite database (output)
│   └── import_data.py        # Main import script
│
├── demand_modeling/          # Demand curve fitting
│   ├── data_extractor.py     # Extract and aggregate sales
│   ├── feature_engineer.py   # Build feature vectors
│   ├── demand_fitter.py      # Fit logistic regression
│   ├── model_validator.py   # Cross-validation
│   ├── model_serializer.py  # Save/load models
│   └── train_model.py       # Training script
│
├── env/                      # RL Environment
│   ├── feature_builder.py    # Build features from state
│   ├── ticket_pricing_env.py # Gymnasium environment
│   └── test_env.py          # Environment test script
│
└── models/                   # Trained models
    ├── demand_model_v1.pkl   # Saved demand model
    └── demand_model_v1.json   # Model metadata
```

## Workflow

### Step 1: Import Data

Import historical ticket sales from CSV files into SQLite:

```bash
cd data_generation
python import_data.py
```

This creates `db.sqlite` with:
- **53 events** (Boston Celtics games)
- **92,672 ticket sales**
- Computed quality scores (0-1 scale)
- Time-to-event calculations

See `data_generation/IMPORT_README.md` for details.

### Step 2: Train Demand Model

Fit a logistic regression model to predict sale probability:

```bash
cd demand_modeling
python train_model.py --cv
```

This:
- Extracts and aggregates sales data (787 observations)
- Engineers 24-dim feature vectors
- Fits logistic regression model
- Validates with cross-validation
- Saves model to `models/demand_model_v1.pkl`

**Model Performance** (current):
- Test AUC: 0.799
- Calibration Error: 0.086
- 24 features: time, price, quality, event context, interactions

See `demand_modeling/README.md` for details.

### Step 3: Create Environment

Use the trained model to create an RL environment:

```python
from env import TicketPricingEnv
from pathlib import Path

env = TicketPricingEnv(
    demand_model_path=Path('models/demand_model_v1.pkl'),
    initial_price_range=(100.0, 500.0),
    quality_range=(0.0, 1.0),
    time_horizon=720.0,  # 30 days
    time_step=6.0,        # 6 hours per step
    price_bounds=(0.3, 3.0),
    random_seed=42
)

# Reset for new episode
obs, info = env.reset()

# Step through episode
action = env.action_space.sample()  # Random action
obs, reward, terminated, truncated, info = env.step(action)
```

See `env/README.md` for details.

### Step 4: Train RL Agent

Train your RL agent on the environment:

```python
# Example: Random agent
for episode in range(1000):
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
    agent.update(total_reward)
```

## Key Design Decisions

### Time Binning
- **Log-scale bins**: `[0-24h, 24-72h, 72-168h, 168-336h, 336-720h, 720h+]`
- **Rationale**: NBA tickets sell mostly 30+ days before event (45% of sales). No surge pricing pattern observed.

### Price Normalization
- Reference price: median price in "main sales window" (7-30 days) per `(event_id, quality_tier)`
- Normalize: `price_rel = Price / p_ref`
- **Rationale**: Removes event-specific scale effects, enables generalization

### Model Choice
- **Logistic Regression** (Binomial GLM)
- **Rationale**: Interpretable, calibrated probabilities, sufficient for feature space

### Environment Design
- **Single-ticket MDP**: Each episode = one ticket
- **Discrete actions**: 7 percentage changes [-20%, -10%, -5%, 0%, +5%, +10%, +20%]
- **Reward**: `price - initial_price` if sold, else `0`
- **Random episodes**: Each `reset()` samples new initial_price, quality, event_context

## Data Statistics

- **Events**: 53 Boston Celtics games
- **Total Sales**: 92,672 tickets
- **Time Range**: 0.1 to 5,730 hours before event
- **Price Range**: $34.38 to $32,112.76 (avg $412.31)
- **Quality Distribution**:
  - Premium (0.75-1.0): 24.8%
  - High (0.50-0.75): 35.5%
  - Medium (0.25-0.50): 32.7%
  - Low (0.0-0.25): 7.0%

## Testing

Test the environment:

```bash
cd env
python test_env.py
```

This runs 10 episodes and validates:
- Environment initialization
- Observation space validity
- Episode termination
- Reward computation

## Dependencies

All dependencies are in `pyproject.toml`:
- `numpy >= 1.26`
- `pandas >= 2.1`
- `scikit-learn >= 1.4`
- `gymnasium >= 0.29`

## Module Documentation

- **Data Generation**: See `data_generation/IMPORT_README.md`
- **Demand Modeling**: See `demand_modeling/README.md`
- **RL Environment**: See `env/README.md`

## Next Steps

1. Train your RL agent (DQN, PPO, etc.) on the environment
2. Compare against baselines (static pricing, greedy EV optimizer)
3. Evaluate on held-out events
4. Deploy to production with safety constraints

## Notes

- The demand model uses `initial_price` as reference price (approximation)
- Event context (weekend, playoff, day-of-week) is randomly sampled
- Time horizon defaults to 30 days (720 hours)
- Price bounds prevent extreme pricing (0.3x to 3.0x initial price)


