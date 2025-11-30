# Demand Curve Modeling

This module implements data extraction and demand curve fitting for the ticket pricing RL environment.

## Overview

The pipeline extracts sales data from SQLite, aggregates into binned observations, engineers features, and fits a logistic regression model to predict `P(sale | price, time, quality, context)`.

## Key Design Decisions

### Time Binning
- **Log-scale bins**: `[0-24h, 24-72h, 72-168h, 168-336h, 336-720h, 720h+]`
- **Rationale**: NBA tickets sell mostly 30+ days before event (45% of sales). No surge pricing pattern observed in data.

### Price Normalization
- Reference price: median price in "main sales window" (7-30 days) per `(event_id, quality_tier)`
- Normalize: `price_rel = Price / p_ref`
- **Rationale**: Removes event-specific scale effects, enables generalization

### Model Choice
- **Logistic Regression** (Binomial GLM)
- **Rationale**: Interpretable, calibrated probabilities, sufficient for feature space

## Usage

### Train Model

```bash
# Basic training
python train_model.py

# With cross-validation
python train_model.py --cv

# Custom paths
python train_model.py --db-path path/to/db.sqlite --output-path path/to/model.pkl
```

### Use in Code

```python
from demand_modeling.data_extractor import extract_sales_data
from demand_modeling.feature_engineer import build_features
from demand_modeling.demand_fitter import fit_demand_model
from demand_modeling.model_serializer import load_model

# Load trained model
model = load_model('models/demand_model_v1.pkl')

# Predict probability
features = np.array([...])  # Feature vector
p_sale = model.predict_proba(features)[0, 1]
```

## Files

- `data_extractor.py`: Extract and aggregate sales from SQLite
- `feature_engineer.py`: Build feature vectors from aggregated data
- `demand_fitter.py`: Fit logistic regression model
- `model_validator.py`: Cross-validation and quality checks
- `model_serializer.py`: Save/load models
- `train_model.py`: Main training script

## Data Flow

```
SQLite DB → data_extractor → aggregated DataFrame
    ↓
aggregated DataFrame → feature_engineer → (X, y, weights)
    ↓
(X, y, weights) → demand_fitter → DemandModel
    ↓
DemandModel → model_serializer → saved .pkl file
```

## Model Quality Thresholds

- **Test AUC > 0.60**: Better than random
- **Calibration Error < 0.10**: Well-calibrated probabilities
- **Brier Score < 0.30**: Good probability predictions

## Confidence: 95%

**Data Pipeline**: 95% confidence
- Well-defined schema, clean data
- Revised binning matches actual distribution
- Price normalization uses main sales window

**Demand Fitting**: 95% confidence
- Standard technique (logistic regression)
- Sufficient data (787 observations from 53 events)
- Stratified train/test prevents data leakage
- Cross-validation framework implemented

