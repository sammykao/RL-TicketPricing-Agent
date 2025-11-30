"""
Gymnasium-compatible environment for single-ticket pricing RL.

Each episode represents one ticket from some time before event until it sells
or event time arrives.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from demand_modeling.model_serializer import load_model
from .feature_builder import build_features_from_state


def sample_event_context(random_state: np.random.Generator) -> Dict:
    """
    Sample random event context.
    
    Args:
        random_state: NumPy random generator
    
    Returns:
        Dict with 'is_weekend', 'is_playoff', 'day_of_week'
    """
    # Sample day_of_week first, then derive is_weekend from it
    # (matching data_extractor.py logic)
    day_of_week = random_state.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    is_weekend = day_of_week in ['Fri', 'Sat', 'Sun']
    
    is_playoff = random_state.random() < 0.2  # 20% probability
    
    return {
        'is_weekend': is_weekend,
        'is_playoff': is_playoff,
        'day_of_week': day_of_week
    }


def validate_price_bounds(price_bounds: Tuple[float, float]) -> bool:
    """
    Validate price bounds are reasonable.
    
    Args:
        price_bounds: (min_multiplier, max_multiplier)
    
    Returns:
        True if valid
    
    Raises:
        ValueError if invalid
    """
    min_mult, max_mult = price_bounds
    
    if min_mult < 0.1:
        raise ValueError(f"price_bounds[0] must be >= 0.1, got {min_mult}")
    if max_mult > 10.0:
        raise ValueError(f"price_bounds[1] must be <= 10.0, got {max_mult}")
    if min_mult >= max_mult:
        raise ValueError(f"price_bounds[0] must be < price_bounds[1], got {price_bounds}")
    
    return True


class TicketPricingEnv(gym.Env):
    """
    Single-ticket pricing MDP environment.
    
    Episode: One ticket from some time_before_event until sale or event_time.
    Action: Percentage price change (discrete: -20%, -10%, -5%, 0%, +5%, +10%, +20%).
    Reward: price - initial_price if sold, else 0.
    """
    
    metadata = {"render_modes": [], "render_fps": 4}
    
    def __init__(
        self,
        demand_model_path: Path,
        initial_price_range: Tuple[float, float] = (100.0, 500.0),
        quality_range: Tuple[float, float] = (0.0, 1.0),
        time_horizon: float = 720.0,
        time_step: float = 6.0,
        price_bounds: Tuple[float, float] = (0.3, 3.0),
        random_seed: Optional[int] = None
    ):
        """
        Initialize environment.
        
        Args:
            demand_model_path: Path to saved .pkl demand model
            initial_price_range: (min, max) for random initial_price sampling
            quality_range: (min, max) for random quality_score sampling
            time_horizon: Maximum hours before event (default 30 days = 720h)
            time_step: Hours per step (default 6h)
            price_bounds: (min_multiplier, max_multiplier) relative to initial_price
            random_seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Validate price bounds
        validate_price_bounds(price_bounds)
        
        # Load demand model
        self.demand_model = load_model(demand_model_path)
        
        # Store parameters
        self.initial_price_range = initial_price_range
        self.quality_range = quality_range
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.price_bounds = price_bounds
        
        # Initialize random state
        self.random_state = np.random.default_rng(random_seed)
        
        # Action space: Discrete percentage changes
        self.action_space = spaces.Discrete(7)
        self.action_map = np.array([-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20])
        
        # Observation space: [time_remaining_norm, price_multiplier, quality_score, is_weekend, is_playoff]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )
        
        # Episode state (initialized in reset)
        self.initial_price: Optional[float] = None
        self.current_price: Optional[float] = None
        self.quality_score: Optional[float] = None
        self.time_remaining: Optional[float] = None
        self.sold: bool = False
        self.event_context: Optional[Dict] = None
    
    def _get_obs(self) -> np.ndarray:
        """
        Build observation vector.
        
        Returns:
            Normalized observation: [time_remaining_norm, price_multiplier, quality_score, is_weekend, is_playoff]
        """
        time_remaining_norm = self.time_remaining / self.time_horizon
        price_multiplier = self.current_price / self.initial_price
        
        return np.array([
            time_remaining_norm,
            price_multiplier,
            self.quality_score,
            float(self.event_context['is_weekend']),
            float(self.event_context['is_playoff'])
        ], dtype=np.float32)
    
    def _compute_sale_probability(self) -> float:
        """
        Compute P(sale | current state) using demand model.
        
        Returns:
            Probability of sale (0-1)
        """
        # Build feature vector
        features = build_features_from_state(
            time_remaining=self.time_remaining,
            current_price=self.current_price,
            initial_price=self.initial_price,
            quality_score=self.quality_score,
            event_context=self.event_context,
            include_interactions=True
        )
        
        # Query demand model
        # Features must be 2D for predict_proba
        features_2d = features.reshape(1, -1)
        p_sale = self.demand_model.predict_proba(features_2d)[0, 1]
        
        return float(p_sale)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed (if provided, updates internal random state)
            options: Optional dict with episode parameters (for testing/fixed episodes)
        
        Returns:
            observation: Initial observation
            info: Info dict
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.random_state = np.random.default_rng(seed)
        
        # Sample episode parameters (or use provided options)
        if options is not None:
            self.initial_price = options.get('initial_price', 
                self.random_state.uniform(*self.initial_price_range))
            self.quality_score = options.get('quality_score',
                self.random_state.uniform(*self.quality_range))
            self.event_context = options.get('event_context',
                sample_event_context(self.random_state))
        else:
            self.initial_price = self.random_state.uniform(*self.initial_price_range)
            self.quality_score = self.random_state.uniform(*self.quality_range)
            self.event_context = sample_event_context(self.random_state)
        
        # Reset episode state
        self.current_price = self.initial_price
        self.time_remaining = self.time_horizon
        self.sold = False
        
        return self._get_obs(), {}
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Discrete action index (0-6)
        
        Returns:
            observation: Next observation
            reward: Reward (price - initial_price if sold, else 0)
            terminated: Whether episode ended due to sale
            truncated: Whether episode ended due to time expiration
            info: Info dict with episode details
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in {self.action_space}")
        
        # Check if episode already terminated
        if self.sold:
            raise ValueError("Episode already terminated (ticket sold). Call reset() first.")
        if self.time_remaining <= 0:
            raise ValueError("Episode already terminated (time expired). Call reset() first.")
        
        # Apply action: update price
        price_change_pct = self.action_map[action]
        new_price = self.current_price * (1 + price_change_pct)
        
        # Clip to bounds
        min_price = self.initial_price * self.price_bounds[0]
        max_price = self.initial_price * self.price_bounds[1]
        new_price = np.clip(new_price, min_price, max_price)
        self.current_price = new_price
        
        # Compute sale probability
        try:
            p_sale = self._compute_sale_probability()
            # Ensure probability is in valid range
            p_sale = np.clip(p_sale, 0.0, 1.0)
        except Exception as e:
            # Fallback to 0.0 if model prediction fails
            p_sale = 0.0
            print(f"Warning: Demand model prediction failed: {e}. Using p_sale=0.0")
        
        # Sample sale outcome
        sold = self.random_state.random() < p_sale
        
        # Compute reward
        if sold:
            reward = self.current_price - self.initial_price
            self.sold = True
            terminated = True
        else:
            reward = 0.0
            terminated = False
        
        # Advance time
        self.time_remaining = max(0.0, self.time_remaining - self.time_step)
        
        # Check truncation (time ran out without sale)
        truncated = (self.time_remaining <= 0) and not sold
        
        # Ensure time_remaining doesn't go negative
        if truncated:
            self.time_remaining = 0.0
        
        # Info dict
        info = {
            'sold': sold,
            'p_sale': p_sale,
            'current_price': float(self.current_price),
            'initial_price': float(self.initial_price),
            'time_remaining': float(self.time_remaining),
            'price_multiplier': float(self.current_price / self.initial_price)
        }
        
        return self._get_obs(), reward, terminated, truncated, info

