"""Base class for the pricing agents that act on the environment"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List
import numpy as np
import gymnasium as gym


class BaseAgent(ABC):
    def __init__(self, env: gym.Env):
        self.env = env

    # declaring abstract methods and then defined later on in subclasses
    @abstractmethod
    def select_action(self, obs: np.ndarray) -> int:
        """Choose an action given the current observation"""
        ...

    @abstractmethod
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
        # the agent would then learn its own way from the transition
        ...

    def run_episode(self, train: bool = True, render: bool = False) -> Tuple[float, int, Dict[str, Any]]:
        """Run a single episode""" 
        # You can Optionally call observe() for learning
        obs, info = self.env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        last_info: Dict[str, Any] = info

        while not done:
            if render:
                self.env.render()

            action = self.select_action(np.asarray(obs, dtype=float))
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            if train:
                self.observe(
                    np.asarray(obs, dtype=float),
                    action,
                    float(reward),
                    np.asarray(next_obs, dtype=float),
                    bool(terminated),
                    bool(truncated),
                    info,
                )

            total_reward += float(reward)
            steps += 1
            obs = next_obs
            done = terminated or truncated
            last_info = info

        return total_reward, steps, last_info

    def train(self, n_episodes: int) -> List[float]:
        """
        Simple training loop.
        
        Note: Each episode automatically resets the environment via run_episode().
        """
        # Ensure environment is in a clean state before training
        self.env.reset()
        
        rewards: List[float] = []
        for _ in range(n_episodes):
            ep_reward, _, _ = self.run_episode(train=True, render=False)
            rewards.append(ep_reward)
        return rewards

    def evaluate(self, n_episodes: int) -> Dict[str, float]:
        """Run episodes with no learning and report average reward."""
        rewards: List[float] = []
        for _ in range(n_episodes):
            ep_reward, _, _ = self.run_episode(train=False, render=False)
            rewards.append(ep_reward)

        rewards_arr = np.array(rewards, dtype=float)
        return {
            "mean_reward": float(rewards_arr.mean()),
            "std_reward": float(rewards_arr.std()),
            "min_reward": float(rewards_arr.min()),
            "max_reward": float(rewards_arr.max()),
        }
