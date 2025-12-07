import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.dqn_agent import DQNAgent
from env.ticket_pricing_env import TicketPricingEnv

# Get paths relative to learning_environment directory
learning_env_dir = Path(__file__).parent.parent
model_path = learning_env_dir / 'models' / 'demand_model_v1.pkl'
checkpoint_path = learning_env_dir / 'checkpoints' / 'dqn_ticket_pricing.pt'
plots_dir = learning_env_dir / 'plots'
plots_dir.mkdir(exist_ok=True)

# Create environment 
# demand_scale: Lower values = harder (lower sale probability)
# Examples: 0.5 = 50% of original probability, 0.3 = 30% of original, etc.
# max_probability: Caps maximum probability (default 0.95) to prevent overconfident predictions
#                  Even if model predicts 100%, it's capped because:
#                  - Empirical probabilities are aggregate, not individual guarantees
#                  - Real-world sales have inherent uncertainty
env = TicketPricingEnv(
    demand_model_path=model_path,
    initial_price_range=(100.0, 500.0),
    quality_range=(0.0, 1.0),
    time_horizon=720.0,
    time_step=6.0,
    price_bounds=(0.3, 3.0),
    demand_scale=0.5,   # moderate difficulty
    max_probability=0.95,
    random_seed=42,
)

# Create or load the agent
if checkpoint_path.exists():
    print("Loading existing checkpoint...")
    agent = DQNAgent.load(env, checkpoint_path)
else:
    print("Creating new agent...")
    agent = DQNAgent(
        env=env,
        hidden_dim=256,
        gamma=0.99,
        lr=5e-4,
        batch_size=128,
        buffer_size=100_000,
        min_buffer_size=5_000,
        target_update_freq=500,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_rate=0.9992,
    )
# Training parameters
n_episodes = 2000
print_freq = 50  # Print metrics every N episodes
plot_freq = 100  # Update plots every N episodes

eval_freq = 100          # evaluate every 50 training episodes
eval_episodes = []
eval_sell_through = []
eval_avg_markup = []

# Tracking metrics
episode_rewards = []
episode_lengths = []
episode_losses = []
epsilon_values = []
steps_history = []

def evaluate_agent(agent, env, n_eval_episodes=50):
    sell_flags = []
    markups = []

    for _ in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        last_info = info

        while not done:
            action = agent.select_greedy_action(np.asarray(obs, dtype=float))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            last_info = info

        sold = bool(last_info.get("sold", False))
        sell_flags.append(int(sold))

        if sold:
            init_p = last_info["initial_price"]
            final_p = last_info["current_price"]
            markup = (final_p - init_p) / init_p
            markups.append(markup)

    sell_through = float(np.mean(sell_flags)) if sell_flags else 0.0
    avg_markup = float(np.mean(markups)) if markups else 0.0

    return sell_through, avg_markup

# Rolling averages for smoothing
reward_window = deque(maxlen=100)
loss_window = deque(maxlen=1000)

print("=" * 60)
print("Starting DQN Training")
print("=" * 60)
print(f"Total episodes: {n_episodes}")
print(f"Print frequency: every {print_freq} episodes")
print(f"Device: {agent.device}")
print(f"Initial epsilon: {agent.epsilon:.3f}")
print("=" * 60)

# Ensure environment is in a clean state
env.reset()

# Custom training loop with metrics tracking
for episode in range(n_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    episode_losses_list = []
    agent.last_loss = None  # Reset loss tracking for new episode
    
    while not done:
        action = agent.select_action(np.asarray(obs, dtype=float))
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Observe and potentially train
        agent.observe(
            np.asarray(obs, dtype=float),
            action,
            float(reward),
            np.asarray(next_obs, dtype=float),
            bool(terminated),
            bool(truncated),
            info,
        )
        
        # Track loss if training occurred
        if agent.last_loss is not None:
            episode_losses_list.append(agent.last_loss)
            loss_window.append(agent.last_loss)
        
        total_reward += float(reward)
        steps += 1
        obs = next_obs
        done = terminated or truncated
    
    # Record episode metrics
    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    reward_window.append(total_reward)
    epsilon_values.append(agent.epsilon)
    steps_history.append(agent.total_steps)
    
    avg_loss = np.mean(episode_losses_list) if episode_losses_list else None
    if avg_loss is not None:
        episode_losses.append(avg_loss)
    else:
        episode_losses.append(None)

    agent.update_epsilon()
    
    
    # Print metrics periodically
    if (episode + 1) % print_freq == 0 or episode == 0:
        avg_reward = np.mean(list(reward_window)) if reward_window else 0.0
        avg_loss_val = np.mean(list(loss_window)) if loss_window else None
        current_epsilon = agent.epsilon
        
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        print(f"  Reward: {total_reward:7.2%} | Avg (last 100): {avg_reward:7.2%}")
        print(f"  Steps: {steps:4d} | Total steps: {agent.total_steps:6d}")
        print(f"  Epsilon: {current_epsilon:.4f} | Buffer: {len(agent.replay_buffer):5d}/{agent.buffer_size}")
        if avg_loss_val is not None:
            print(f"  Avg Loss: {avg_loss_val:.6f}")
        if info.get('sold'):
            price_pct = ((info.get('current_price', 0) - info.get('initial_price', 0)) / info.get('initial_price', 1)) * 100
            print(f"  ✓ Ticket sold at ${info.get('current_price', 0):.2f} ({price_pct:+.1f}% from initial ${info.get('initial_price', 0):.2f})")
        else:
            print(f"  ✗ Ticket not sold (time expired)")

    if (episode + 1) % eval_freq == 0:
        st, mk = evaluate_agent(agent, env, n_eval_episodes=50)
        eval_episodes.append(episode + 1)
        eval_sell_through.append(st)
        eval_avg_markup.append(mk)
        print(f"  [Eval greedy] sell-through={st:6.2%} | avg markup={mk:6.2%}")

# Save checkpoint
print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
agent.save(checkpoint_path)

# Create plots
print("\nGenerating training plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('DQN Training Metrics', fontsize=16, fontweight='bold')

# Plot 1: Episode Rewards
ax1 = axes[0, 0]
ax1.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
if len(reward_window) > 0:
    # Plot rolling average
    window_size = min(100, len(episode_rewards))
    rolling_avg = [np.mean(episode_rewards[max(0, i-window_size+1):i+1]) 
                   for i in range(len(episode_rewards))]
    ax1.plot(rolling_avg, color='red', linewidth=2, label=f'Rolling Avg ({window_size})')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward ($)')
ax1.set_title('Episode Rewards')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Epsilon Decay
ax2 = axes[0, 1]
ax2.plot(epsilon_values, color='green', linewidth=2)
ax2.set_xlabel('Episode')
ax2.set_ylabel('Epsilon')
ax2.set_title('Epsilon Decay (Exploration Rate)')
ax2.grid(True, alpha=0.3)

# Plot 3: Training Loss
ax3 = axes[1, 0]
valid_losses = [(i, loss) for i, loss in enumerate(episode_losses) if loss is not None]
if valid_losses:
    loss_episodes, losses = zip(*valid_losses)
    ax3.plot(loss_episodes, losses, alpha=0.5, color='orange', label='Episode Avg Loss')
    if len(loss_window) > 0:
        # Plot rolling average of losses
        window_size = min(100, len(valid_losses))
        loss_rolling = []
        for i in range(len(valid_losses)):
            window_losses = [l for _, l in valid_losses[max(0, i-window_size+1):i+1]]
            if window_losses:
                loss_rolling.append(np.mean(window_losses))
        if loss_rolling:
            ax3.plot([e for e, _ in valid_losses[:len(loss_rolling)]], 
                    loss_rolling, color='red', linewidth=2, label=f'Rolling Avg ({window_size})')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
else:
    ax3.text(0.5, 0.5, 'No loss data yet\n(Waiting for buffer to fill)', 
            ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Training Loss')

# Plot 4: Episode Lengths
ax4 = axes[1, 1]
ax4.plot(episode_lengths, alpha=0.5, color='purple', label='Episode Length')
window_size = min(100, len(episode_lengths))
rolling_length = [np.mean(episode_lengths[max(0, i-window_size+1):i+1]) 
                   for i in range(len(episode_lengths))]
ax4.plot(rolling_length, color='red', linewidth=2, label=f'Rolling Avg ({window_size})')
ax4.set_xlabel('Episode')
ax4.set_ylabel('Steps')
ax4.set_title('Episode Lengths')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = plots_dir / 'dqn_training_metrics.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Plots saved to {plot_path}")
plt.close()

# Final evaluation
print("\n" + "=" * 60)
print("Running Final Evaluation...")
print("=" * 60)
eval_metrics = agent.evaluate(n_episodes=100)
print(f"\nEvaluation Results (100 episodes):")
print(f"  Mean Reward: {eval_metrics['mean_reward']:.2%}")
print(f"  Std Reward:  {eval_metrics['std_reward']:.2%}")
print(f"  Min Reward:  {eval_metrics['min_reward']:.2%}")
print(f"  Max Reward:  {eval_metrics['max_reward']:.2%}")

print("\n" + "=" * 60)
print("Training Summary")
print("=" * 60)
print(f"Total episodes: {n_episodes}")
print(f"Total steps: {agent.total_steps:,}")
print(f"Final epsilon: {agent.epsilon:.4f}")
print(f"Replay buffer size: {len(agent.replay_buffer):,}/{agent.buffer_size:,}")
print(f"Final avg reward (last 100): {np.mean(list(reward_window)):.2%}")
print("=" * 60)

fig2, ax = plt.subplots(figsize=(8, 5))
if eval_episodes:
    ax.plot(eval_episodes, eval_sell_through, label="Sell-through", linewidth=2)
    ax.plot(eval_episodes, eval_avg_markup, label="Avg markup (sold only)", linewidth=2)
ax.set_xlabel("Training Episode")
ax.set_ylabel("Metric")
ax.set_title("Greedy Policy Performance Over Training")
ax.legend()
ax.grid(True, alpha=0.3)

perf_plot_path = plots_dir / 'dqn_eval_performance.png'
plt.savefig(perf_plot_path, dpi=150, bbox_inches='tight')
print(f"Performance plot saved to {perf_plot_path}")
plt.close()

