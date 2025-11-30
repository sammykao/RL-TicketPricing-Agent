import numpy as np
import gymnasium as gym
from gymnasium import spaces

from agents.tabular_q_agent import TabularQAgent
from agents.dqn_agent import DQNAgent


class LineWorldEnv(gym.Env):
    """
    1D line with 5 states: positions 0..4.
    Start at 0, goal at 4.
    Actions: 0=left, 1=right.
    Reward: +1 if you reach position 4, else 0.
    Episode length: max 10 steps.
    Observation: normalized position in [0,1].
    """
    metadata = {}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.pos = 0
        self.max_steps = 10
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = 0
        self.steps = 0
        obs = np.array([self.pos / 4.0], dtype=np.float32)
        return obs, {}

    def step(self, action):
        self.steps += 1

        # Apply action
        if action == 0:      # left
            self.pos = max(0, self.pos - 1)
        elif action == 1:    # right
            self.pos = min(4, self.pos + 1)

        # Reward only when at goal
        reward = 1.0 if self.pos == 4 else 0.0

        terminated = self.pos == 4
        truncated = self.steps >= self.max_steps

        obs = np.array([self.pos / 4.0], dtype=np.float32)
        info = {}
        return obs, reward, terminated, truncated, info
 

def test_tabular():
    env = LineWorldEnv()
    agent = TabularQAgent(env, n_bins=5, alpha=0.3, gamma=0.99,
                          epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_episodes=200)

    print("Training TabularQAgent on LineWorld...")
    rewards = agent.train(n_episodes=500)
    eval_metrics = agent.evaluate(n_episodes=100)
    print("Tabular eval:", eval_metrics)


def test_dqn():
    env = LineWorldEnv()
    agent = DQNAgent(env,
                     hidden_dim=64,
                     gamma=0.99,
                     lr=1e-3,
                     batch_size=64,
                     buffer_size=10000,
                     min_buffer_size=500,
                     target_update_freq=500,
                     epsilon_start=1.0,
                     epsilon_end=0.05,
                     epsilon_decay_steps=10000)

    print("Training DQNAgent on LineWorld...")
    agent.train(n_episodes=500)
    eval_metrics = agent.evaluate(n_episodes=100)
    print("DQN eval:", eval_metrics)


if __name__ == "__main__":
    test_tabular()
    test_dqn()
