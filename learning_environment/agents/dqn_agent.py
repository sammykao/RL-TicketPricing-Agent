from __future__ import annotations
from typing import Deque, Tuple, Dict, Any
from collections import deque
from pathlib import Path
import random

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent
from env.ticket_pricing_env import TicketPricingEnv


class QNetwork(nn.Module):
    """Simple MLP for the DQN: obs -> Q-values for each action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(BaseAgent):
    """Deep Q-Network agent with replay buffer, target network and ε-greedy exploration"""

    def __init__(
        self,
        env: gym.Env,
        hidden_dim: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        min_buffer_size: int = 1_000,
        target_update_freq: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        device: str | None = None,
    ):
        super().__init__(env)

        assert hasattr(env.action_space, "n"), "DQNAgent requires discrete action space"
        obs_dim = int(np.prod(env.observation_space.shape))
        n_actions = env.action_space.n

        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.target_update_freq = target_update_freq

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_net = QNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer: (obs, action, reward, next_obs, done)
        self.replay_buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=buffer_size
        )
        
        # Track training metrics
        self.last_loss: float | None = None

    # ε schedule
    def epsilon(self) -> float:
        frac = min(1.0, self.total_steps / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    # BaseAgent interface
    def select_action(self, obs: np.ndarray) -> int:
        eps = self.epsilon()
        if random.random() < eps:
            return int(self.env.action_space.sample())

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        action = int(torch.argmax(q_values, dim=1).item())
        return action

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
        done = bool(terminated or truncated)

        # Store transition
        self.replay_buffer.append((obs.copy(), int(action), float(reward), next_obs.copy(), done))
        self.total_steps += 1

        # Train only if we have enough data
        if len(self.replay_buffer) >= self.min_buffer_size:
            self.train_step()

        # Update target network periodically
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    # training step

    def sample_batch(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs = torch.as_tensor(np.array(obs), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.array(next_obs), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        return obs, actions, rewards, next_obs, dones

    def train_step(self) -> float | None:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.sample_batch()

        # Q(s,a) for actions taken
        q_values = self.q_net(obs)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + γ max_a' Q_target(s',a') * (1 - done)
        with torch.no_grad():
            q_next = self.target_net(next_obs)
            q_next_max = q_next.max(dim=1).values
            targets = rewards + self.gamma * q_next_max * (1.0 - dones)

        loss = self.loss_fn(q_sa, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss_value = loss.item()
        self.last_loss = loss_value
        return loss_value

    def save(self, filepath: Path) -> None:
        """
        Save agent's Q-network weights and training state.
        
        Args:
            filepath: Path to save checkpoint (should end in .pt or .pth)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay_steps': self.epsilon_decay_steps,
            'gamma': self.gamma,
            'lr': self.optimizer.param_groups[0]['lr'],
            'hidden_dim': self.q_net.net[0].out_features,  # Get from network
            'batch_size': self.batch_size,
            'buffer_size': self.buffer_size,
            'min_buffer_size': self.min_buffer_size,
            'target_update_freq': self.target_update_freq,
        }
        
        torch.save(checkpoint, filepath)
        print(f"Agent saved to {filepath}")

    @classmethod
    def load(cls, env: gym.Env, filepath: Path, device: str | None = None) -> 'DQNAgent':
        """
        Load agent from checkpoint.
        
        Args:
            env: Environment instance
            filepath: Path to checkpoint file
            device: Device to load on (default: auto-detect)
        
        Returns:
            Loaded DQNAgent instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Auto-detect device if not provided
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create agent with saved hyperparameters
        agent = cls(
            env=env,
            hidden_dim=checkpoint['hidden_dim'],
            gamma=checkpoint['gamma'],
            lr=checkpoint['lr'],
            batch_size=checkpoint.get('batch_size', 64),
            buffer_size=checkpoint.get('buffer_size', 50_000),
            min_buffer_size=checkpoint.get('min_buffer_size', 1_000),
            target_update_freq=checkpoint.get('target_update_freq', 1000),
            epsilon_start=checkpoint['epsilon_start'],
            epsilon_end=checkpoint['epsilon_end'],
            epsilon_decay_steps=checkpoint['epsilon_decay_steps'],
            device=device,
        )
        
        # Load network weights
        agent.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.total_steps = checkpoint['total_steps']
        
        print(f"Agent loaded from {filepath}")
        print(f"Resuming from step {agent.total_steps}")
        
        return agent
