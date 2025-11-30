# agents/tabular_q_agent.py

from __future__ import annotations
from typing import Tuple, Dict, Any
from collections import defaultdict

import numpy as np
import gymnasium as gym

from .base_agent import BaseAgent


class TabularQAgent(BaseAgent):
    """
    Generic tabular Q-learning agent.

    - Works with any env that has a discrete action_space.
    - Discretizes each observation dimension into bins.
    """

    def __init__(
        self,
        env: gym.Env,
        n_bins: int = 8,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 500,
    ):
        super().__init__(env)
        assert hasattr(env.action_space, "n"), "TabularQAgent requires discrete action space"

        self.n_actions = env.action_space.n
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.episode_idx = 0

        # Q-table: mapping from discrete state key -> np.array(shape=(n_actions,))
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )

        # For simple use we assume each obs dimension roughly in [0, 1].
        # You can override self.obs_low / self.obs_high later if needed.
        obs_space = env.observation_space
        if hasattr(obs_space, "low") and hasattr(obs_space, "high"):
            self.obs_low = np.array(obs_space.low, dtype=float)
            self.obs_high = np.array(obs_space.high, dtype=float)
        else:
            # Fallback, assume 5-dim 0..1
            self.obs_low = np.zeros(5, dtype=float)
            self.obs_high = np.ones(5, dtype=float)

    # discretization utilities

    def discretize_obs(self, obs: np.ndarray) -> Tuple[int, ...]:
        """Map a continuous observation vector to a discrete state key."""
        obs = np.asarray(obs, dtype=float)
        # Clip to bounds
        obs = np.clip(obs, self.obs_low, self.obs_high)

        # Compute bin indices per dimension
        ratios = (obs - self.obs_low) / (self.obs_high - self.obs_low + 1e-8)
        bins = np.floor(ratios * self.n_bins).astype(int)
        bins = np.clip(bins, 0, self.n_bins - 1)

        return tuple(int(b) for b in bins)

    def epsilon(self) -> float:
        """Linearly decaying epsilon."""
        frac = min(1.0, self.episode_idx / max(1, self.epsilon_decay_episodes))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    # BaseAgent interface

    def select_action(self, obs: np.ndarray) -> int:
        state = self.discretize_obs(obs)
        eps = self.epsilon()

        if np.random.rand() < eps:
            # Explore
            return int(self.env.action_space.sample())
        else:
            # Exploit
            q_values = self.q_table[state]
            return int(np.argmax(q_values))

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
            terminated: bool,
            truncated: bool,
            info: Dict[str, Any],
    ) -> None:
        """Standard Q-learning update."""
        state = self.discretize_obs(obs)
        next_state = self.discretize_obs(next_obs)

        q_values = self.q_table[state]
        q_next = self.q_table[next_state]

        target = reward
        if not (terminated or truncated):
            target += self.gamma * float(np.max(q_next))

        # Q-learning update
        q_values[action] += self.alpha * (target - q_values[action])

        # If the episode ended on this transition, advance episode counter
        if terminated or truncated:
            self.episode_idx += 1
