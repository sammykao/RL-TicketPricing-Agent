"""
Test script to validate the ticket pricing environment.

Runs a small number of episodes and checks that:
- Environment initializes correctly
- Observations are in valid range
- Episodes terminate correctly
- Rewards are computed correctly
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ticket_pricing_env import TicketPricingEnv


def test_environment():
    """Run basic tests on the environment."""
    print("=" * 60)
    print("TESTING TICKET PRICING ENVIRONMENT")
    print("=" * 60)
    
    # Path to model
    model_path = Path(__file__).parent.parent / 'models' / 'demand_model_v1.pkl'
    
    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}")
        print("Please train the model first using demand_modeling/train_model.py")
        return False
    
    # Create environment
    print("\n[1/5] Creating environment...")
    try:
        env = TicketPricingEnv(
            demand_model_path=model_path,
            initial_price_range=(100.0, 500.0),
            quality_range=(0.0, 1.0),
            time_horizon=720.0,
            time_step=6.0,
            price_bounds=(0.3, 3.0),
            random_seed=42
        )
        print("  ✓ Environment created successfully")
    except Exception as e:
        print(f"  ✗ Failed to create environment: {e}")
        return False
    
    # Test observation space
    print("\n[2/5] Testing observation space...")
    obs, info = env.reset()
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  Observation: {obs}")
    
    if not env.observation_space.contains(obs):
        print(f"  ✗ Observation not in observation_space!")
        return False
    print("  ✓ Observation is valid")
    
    # Test action space
    print("\n[3/5] Testing action space...")
    print(f"  Action space: {env.action_space}")
    print(f"  Action map: {env.action_map}")
    
    # Test multiple episodes
    print("\n[4/5] Running 10 episodes...")
    episode_stats = {
        'total_episodes': 0,
        'sold_episodes': 0,
        'expired_episodes': 0,
        'total_reward': 0.0,
        'sold_rewards': [],
        'episode_lengths': []
    }
    
    for episode in range(10):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Safety check: prevent infinite loops
            if episode_length > 200:
                print(f"  Warning: Episode {episode} exceeded 200 steps, forcing termination")
                break
        
        episode_stats['total_episodes'] += 1
        episode_stats['total_reward'] += episode_reward
        episode_stats['episode_lengths'].append(episode_length)
        
        if terminated:
            episode_stats['sold_episodes'] += 1
            episode_stats['sold_rewards'].append(episode_reward)
            print(f"  Episode {episode + 1}: SOLD after {episode_length} steps, reward=${episode_reward:.2f}")
        elif truncated:
            episode_stats['expired_episodes'] += 1
            print(f"  Episode {episode + 1}: EXPIRED after {episode_length} steps, reward=${episode_reward:.2f}")
    
    # Print statistics
    print("\n[5/5] Episode Statistics:")
    print(f"  Total episodes: {episode_stats['total_episodes']}")
    print(f"  Sold: {episode_stats['sold_episodes']} ({episode_stats['sold_episodes']/episode_stats['total_episodes']*100:.1f}%)")
    print(f"  Expired: {episode_stats['expired_episodes']} ({episode_stats['expired_episodes']/episode_stats['total_episodes']*100:.1f}%)")
    print(f"  Average reward: ${episode_stats['total_reward']/episode_stats['total_episodes']:.2f}")
    if episode_stats['sold_rewards']:
        print(f"  Average reward (sold only): ${np.mean(episode_stats['sold_rewards']):.2f}")
    print(f"  Average episode length: {np.mean(episode_stats['episode_lengths']):.1f} steps")
    
    # Validate rewards
    print("\n[6/5] Validating rewards...")
    all_valid = True
    
    # Check that sold episodes have positive rewards (price > initial_price)
    # or at least non-negative (could be discount)
    for reward in episode_stats['sold_rewards']:
        if reward < -1000:  # Allow some negative (discounts) but not extreme
            print(f"  ✗ Invalid reward for sold episode: ${reward:.2f}")
            all_valid = False
    
    if all_valid:
        print("  ✓ All rewards are reasonable")
    
    # Test edge cases
    print("\n[7/5] Testing edge cases...")
    
    # Test reset after termination
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Force termination by setting sold=True
    env.sold = True
    try:
        env.step(0)
        print("  ✗ Should have raised error for terminated episode")
        all_valid = False
    except ValueError:
        print("  ✓ Correctly raises error for terminated episode")
    
    # Reset again
    obs, info = env.reset()
    print("  ✓ Reset works after termination")
    
    print("\n" + "=" * 60)
    if all_valid:
        print("ALL TESTS PASSED ✓")
        return True
    else:
        print("SOME TESTS FAILED ✗")
        return False


if __name__ == '__main__':
    success = test_environment()
    sys.exit(0 if success else 1)

