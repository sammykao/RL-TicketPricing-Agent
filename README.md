# Dynamic Ticket Pricing with Reinforcement Learning  
Tufts COMP 138 – Final Project

This project implements a reinforcement learning (RL) agent that dynamically sets ticket prices for live events.  
The agent interacts with a data-driven simulator built from SeatData.io historical ticket sales, learning pricing strategies that maximize revenue and sell out inventory before the event.

The system includes the full RL workflow: data ingestion, environment modeling, agent training, baseline comparison, and visualization.

---

## 📌 1. Overview

Traditional pricing assumes stable demand curves. Real markets fluctuate based on time, remaining inventory, and prior pricing decisions.  
We frame pricing as a **sequential decision-making problem**, where the RL agent:

- observes: time remaining, inventory, last price, demand rate  
- chooses: raise price, lower price, or keep it  
- receives reward: revenue + bonuses for selling out + penalty for unstable pricing  
- learns a policy that adapts to market behavior  

Two agents are implemented:

1. **Tabular Q-Learning** (discretized state space)  
2. **Deep Q-Network (DQN)** (neural function approximation)

Both are evaluated against heuristic baselines.

---

## 📁 2. Project Structure

project/
│
├── data/
│ ├── raw/ # SeatData.io files
│ ├── processed/ # cleaned data
│ └── loader.py
│
├── env/
│ ├── ticket_env.py # Gym-like environment
│ ├── demand_model.py # elasticity-based demand simulator
│ └── utils.py
│
├── agent/
│ ├── q_learning.py
│ ├── dqn.py
│ └── networks.py
│
├── train/
│ ├── train_qlearning.py
│ ├── train_dqn.py
│ └── replay_buffer.py
│
├── eval/
│ ├── baselines.py
│ ├── metrics.py
│ ├── experiment_runner.py
│ └── plots.py
│
└── notebooks/
└── EDA.ipynb


## ⚙️ 3. Installation
pip install -r requirements.txt

Required packages include:
numpy
pandas
matplotlib
scikit-learn
torch
tqdm
seaborn
gymnasium

---

## 🧹 4. Dataset & Pipeline (SeatData.io)

### Source  
SeatData.io historical event ticket sales.

### Steps  
1. Place raw files in `data/raw/`.  
2. Run:
python data/loader.py

This produces cleaned sequences in `data/processed/`, with:

- aligned timestamps  
- inventory levels  
- historical prices  
- demand rates  
- per-interval ticket sales  

These sequences feed the RL environment.

---

## 🎮 5. Environment

Implemented in `env/ticket_env.py`.

### **State**

[time_remaining, inventory_remaining, last_price, demand_rate]

### **Actions**
-Δp, 0, +Δp

(or discrete price levels)

### **Demand Model**
q_t = α_t * exp(-β * price_t) + noise

### **Reward**
revenue + sellout_bonus - price_jump_penalty

Episode ends when inventory hits zero or time runs out.

---

## 🤖 6. Agents

### **Tabular Q-Learning**  
File: `agent/q_learning.py`  
- discretized state bins  
- ε-greedy exploration  
- simple debugging baseline  

### **Deep Q-Network (DQN)**  
File: `agent/dqn.py`  
- two fully connected layers (64 units)  
- replay buffer  
- target network  
- ε-decay for exploration  
- handles continuous state space  

---

## 🏋️ 7. Training Instructions

### Tabular Q-Learning:
python train/train_qlearning.py

### DQN:
python train/train_dqn.py

Logs, rewards, and model weights are saved under `runs/`.

---

## 📊 8. Evaluation & Baselines

Run experiments and performance comparisons:
python eval/experiment_runner.py

### Metrics:
- total revenue  
- sell-through rate  
- price volatility  
- convergence stability  
- regret vs. optimal static price  
- inventory trajectory  

### Baselines:
1. Constant price  
2. Linear price decay  
3. Elasticity-based heuristic  

---

## 📈 9. Plotting & Visualization

Run:
python eval/plots.py

Plots include:

- learning curves  
- revenue distributions  
- price trajectories  
- inventory over time  
- Q-value heatmaps  
- demand model vs. historical sales  

More advanced plots available in `notebooks/`.

---

## 🔄 10. Reproducibility

Set seeds in `config.py`:

Or pass via CLI:
python train/train_dqn.py --seed 42


## 👥 11. Team Responsibilities

- **Sammy:** Data pipeline, environment  
- **Javier:** Q-Learning + DQN implementation  
- **Reed:** Evaluation, experiments, visualizations  

---

## 🎯 12. Expected Outcomes

The RL agents are expected to:

- beat heuristic pricing on revenue  
- consistently sell out inventory  
- adapt policies to demand shifts  
- reduce price volatility compared to naïve strategies  
- generalize across different event types  

---

If you'd like, I can also generate:

- `requirements.txt`  
- a polished version with badges, logos, section icons  
- a mermaid diagram of the full system architecture  
