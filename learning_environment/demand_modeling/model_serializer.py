"""
Save and load demand models.
"""

from pathlib import Path
from typing import Optional
import pickle
import json
from demand_modeling.demand_fitter import DemandModel


def save_model(
    model: DemandModel,
    filepath: Path,
    metadata: Optional[dict] = None
) -> None:
    """
    Save demand model to disk.
    
    Saves:
    - Model object (pickle)
    - Metadata (JSON)
    
    Args:
        model: DemandModel to save
        filepath: Path to save (should end in .pkl)
        metadata: Additional metadata to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    metadata_path = filepath.with_suffix('.json')
    metadata_dict = {
        'model_type': 'LogisticRegression',
        'feature_names': model.feature_names,
        'training_metrics': model.training_metrics,
        'n_features': len(model.feature_names),
        **(metadata or {})
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    
    print(f"Model saved to {filepath}")
    print(f"Metadata saved to {metadata_path}")


def load_model(filepath: Path) -> DemandModel:
    """
    Load demand model from disk.
    
    Args:
        filepath: Path to .pkl file
    
    Returns:
        DemandModel instance
    """
    import sys
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    # Fix for pickle compatibility: map old module name to new one
    # This handles models saved with the old import path 'demand_fitter'
    if 'demand_fitter' not in sys.modules:
        # Import the actual module
        from demand_modeling import demand_fitter
        # Create alias so pickle can find it
        sys.modules['demand_fitter'] = demand_fitter
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    return model


def load_metadata(filepath: Path) -> dict:
    """Load model metadata JSON."""
    metadata_path = Path(filepath).with_suffix('.json')
    
    if not metadata_path.exists():
        return {}
    
    with open(metadata_path, 'r') as f:
        return json.load(f)

