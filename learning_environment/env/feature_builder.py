"""
Feature builder for constructing model features from environment state.

Replicates the exact feature engineering logic from demand_modeling/feature_engineer.py
to ensure feature vectors match the trained model's expectations.
"""

import numpy as np
from typing import Dict


# Time bin centers (matching feature_engineer.py)
TIME_BIN_CENTERS = {
    0: 12.0,    # 0-24h -> 12h
    1: 48.0,    # 24-72h -> 48h
    2: 120.0,   # 72-168h -> 120h
    3: 252.0,   # 168-336h -> 252h
    4: 528.0,   # 336-720h -> 528h
    5: 1080.0   # 720h+ -> 1080h (approximate)
}

QUALITY_TIERS = ['Low', 'Medium', 'High', 'Premium']
DAYS_OF_WEEK = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


def compute_time_bin_log_scale(time_to_event: float, max_hours: float = 720.0) -> int:
    """
    Compute time bin using log-scale (matching data_extractor.py).
    
    Args:
        time_to_event: Hours until event
        max_hours: Maximum time to consider (default 30 days)
    
    Returns:
        Bin index (0-5)
    """
    time_clipped = max(0.0, min(time_to_event, max_hours))
    
    if time_clipped < 24:
        return 0
    elif time_clipped < 72:
        return 1
    elif time_clipped < 168:
        return 2
    elif time_clipped < 336:
        return 3
    elif time_clipped < 720:
        return 4
    else:
        return 5


def quality_score_to_tier(quality_score: float) -> str:
    """Convert quality score (0-1) to tier string."""
    if quality_score >= 0.75:
        return 'Premium'
    elif quality_score >= 0.50:
        return 'High'
    elif quality_score >= 0.25:
        return 'Medium'
    else:
        return 'Low'


def quality_tier_to_one_hot(quality_tier: str) -> np.ndarray:
    """Convert quality tier to one-hot encoding."""
    one_hot = np.zeros(4)
    if quality_tier in QUALITY_TIERS:
        idx = QUALITY_TIERS.index(quality_tier)
        one_hot[idx] = 1.0
    return one_hot


def day_of_week_to_one_hot(day_of_week: str) -> np.ndarray:
    """Convert day of week to one-hot encoding."""
    one_hot = np.zeros(7)
    if day_of_week in DAYS_OF_WEEK:
        idx = DAYS_OF_WEEK.index(day_of_week)
        one_hot[idx] = 1.0
    return one_hot


def time_bin_to_one_hot(time_bin: int) -> np.ndarray:
    """Convert time bin to one-hot encoding."""
    one_hot = np.zeros(6)
    if 0 <= time_bin < 6:
        one_hot[time_bin] = 1.0
    return one_hot


def build_features_from_state(
    time_remaining: float,
    current_price: float,
    initial_price: float,
    quality_score: float,
    event_context: Dict,
    include_interactions: bool = True
) -> np.ndarray:
    """
    Build 24-dim feature vector from environment state.
    
    Feature order (matching model.feature_names):
    1. time_log (1 dim)
    2. time_bin_0 through time_bin_5 (6 dims)
    3. price_log_rel (1 dim)
    4. price_rel (1 dim)
    5. quality_Low, quality_Medium, quality_High, quality_Premium (4 dims)
    6. is_weekend (1 dim)
    7. is_playoff (1 dim)
    8. day_Mon through day_Sun (7 dims)
    9. price_time_interaction (1 dim, if include_interactions)
    10. quality_time_interaction (1 dim, if include_interactions)
    
    Args:
        time_remaining: Hours until event
        current_price: Current ticket price
        initial_price: Initial ticket price (used as reference price)
        quality_score: Quality score (0-1)
        event_context: Dict with 'is_weekend', 'is_playoff', 'day_of_week'
        include_interactions: Whether to include interaction terms
    
    Returns:
        Feature vector (24 dims if interactions, 22 dims otherwise)
    """
    # 1. Time features
    time_bin = compute_time_bin_log_scale(time_remaining)
    time_approx = TIME_BIN_CENTERS[time_bin]
    time_log = np.log1p(time_approx)
    
    time_bin_onehot = time_bin_to_one_hot(time_bin)
    
    # 2. Price features
    price_rel = current_price / initial_price
    price_log_rel = np.log(np.clip(price_rel, 0.1, 10.0))  # Clip to avoid log(0)
    
    # 3. Quality features
    quality_tier = quality_score_to_tier(quality_score)
    quality_onehot = quality_tier_to_one_hot(quality_tier)
    
    # 4. Event context
    is_weekend = float(event_context.get('is_weekend', 0))
    is_playoff = float(event_context.get('is_playoff', 0))
    day_of_week = event_context.get('day_of_week', 'Mon')
    day_onehot = day_of_week_to_one_hot(day_of_week)
    
    # Build feature vector in exact order
    features = [
        time_log,                           # time_log
        *time_bin_onehot,                   # time_bin_0 through time_bin_5
        price_log_rel,                      # price_log_rel
        price_rel,                          # price_rel
        *quality_onehot,                    # quality_Low, Medium, High, Premium
        is_weekend,                         # is_weekend
        is_playoff,                         # is_playoff
        *day_onehot,                        # day_Mon through day_Sun
    ]
    
    # 5. Interaction terms (if enabled)
    if include_interactions:
        # Price × Time interaction
        price_time_interaction = price_log_rel * time_log
        features.append(price_time_interaction)
        
        # Quality × Time interaction (using tier index)
        quality_tier_idx = QUALITY_TIERS.index(quality_tier) if quality_tier in QUALITY_TIERS else 0
        quality_time_interaction = quality_tier_idx * time_log
        features.append(quality_time_interaction)
    
    return np.array(features, dtype=np.float32)

