"""
Extract and aggregate sales data from SQLite database for demand modeling.

Key design decisions:
- Time binning: Log-scale bins (no surge assumption for NBA tickets)
- Price normalization: Per (event, quality_tier) median reference price
- Aggregation: Binomial-style (sold_count, exposure) for probability modeling
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


def load_database(db_path: Path) -> sqlite3.Connection:
    """Load SQLite database connection."""
    return sqlite3.connect(str(db_path))


def compute_time_bin_log_scale(time_to_event: float, max_hours: float = 720.0) -> int:
    """
    Compute time bin using log-scale to handle long-tail distribution.
    
    NBA tickets sell mostly 30+ days before event, so we use log-scale bins
    rather than urgency-focused bins.
    
    Bins (hours):
    - Bin 0: [0, 24)      - Last day
    - Bin 1: [24, 72)     - 1-3 days
    - Bin 2: [72, 168)    - 3-7 days
    - Bin 3: [168, 336)   - 7-14 days
    - Bin 4: [336, 720)   - 14-30 days
    - Bin 5: [720, max]   - 30+ days
    
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


def compute_reference_prices(conn: sqlite3.Connection) -> Dict[Tuple[int, str], float]:
    """
    Compute reference price for each (event_id, quality_tier) pair.
    
    Uses median price in the "main sales window" (7-30 days before event)
    to normalize prices across events.
    
    Returns:
        Dict mapping (event_id, quality_tier) -> reference_price
    """
    query = """
        SELECT 
            event_id,
            CASE 
                WHEN CAST(ticket_quality AS REAL) >= 0.75 THEN 'Premium'
                WHEN CAST(ticket_quality AS REAL) >= 0.50 THEN 'High'
                WHEN CAST(ticket_quality AS REAL) >= 0.25 THEN 'Medium'
                ELSE 'Low'
            END as quality_tier,
            Price
        FROM ticket_sales
        WHERE time_to_event >= 168 AND time_to_event <= 720
            AND Price IS NOT NULL
            AND ticket_quality IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Compute median price per (event_id, quality_tier)
    ref_prices = {}
    for (event_id, quality_tier), group in df.groupby(['event_id', 'quality_tier']):
        if len(group) > 0:
            ref_prices[(event_id, quality_tier)] = group['Price'].median()
    
    return ref_prices


def extract_sales_data(
    db_path: Path,
    max_time_hours: float = 720.0,
    min_sales_per_event: int = 10
) -> pd.DataFrame:
    """
    Extract and aggregate sales data from SQLite database.
    
    Args:
        db_path: Path to SQLite database
        max_time_hours: Maximum time_to_event to consider (default 30 days)
        min_sales_per_event: Minimum sales required per event (filter sparse events)
    
    Returns:
        DataFrame with columns:
        - event_id, quality_tier, time_bin, price_rel, sold_count, exposure,
        - day_of_week, is_weekend, is_playoff, year, month
    """
    conn = load_database(db_path)
    
    # Load all sales with event context
    query = """
        SELECT 
            ts.event_id,
            ts.time_to_event,
            ts.Price,
            ts.ticket_quality,
            ts.Qty,
            e.day_of_week,
            e.year,
            e.month,
            e.away_team,
            e.home_team
        FROM ticket_sales ts
        JOIN events e ON ts.event_id = e.event_id
        WHERE ts.time_to_event IS NOT NULL
            AND ts.time_to_event >= 0
            AND ts.time_to_event <= ?
            AND ts.Price IS NOT NULL
            AND ts.ticket_quality IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn, params=(max_time_hours,))
    
    if len(df) == 0:
        raise ValueError("No sales data found in database")
    
    # Compute quality tiers
    df['quality_score'] = df['ticket_quality'].astype(float)
    df['quality_tier'] = df['quality_score'].apply(
        lambda x: 'Premium' if x >= 0.75 
        else 'High' if x >= 0.50 
        else 'Medium' if x >= 0.25 
        else 'Low'
    )
    
    # Compute time bins
    df['time_bin'] = df['time_to_event'].apply(compute_time_bin_log_scale)
    
    # Compute reference prices
    ref_prices = compute_reference_prices(conn)
    
    # Compute relative prices
    def get_price_rel(row):
        key = (row['event_id'], row['quality_tier'])
        if key in ref_prices and ref_prices[key] > 0:
            return row['Price'] / ref_prices[key]
        return np.nan
    
    df['price_rel'] = df.apply(get_price_rel, axis=1)
    
    # Filter out rows with invalid price_rel
    df = df.dropna(subset=['price_rel'])
    
    # Add event context features
    df['is_weekend'] = df['day_of_week'].isin(['Fri', 'Sat', 'Sun']).astype(int)
    df['is_playoff'] = (df['month'] >= 4).astype(int)  # April+ = playoffs
    
    # Filter events with too few sales
    event_counts = df.groupby('event_id').size()
    valid_events = event_counts[event_counts >= min_sales_per_event].index
    df = df[df['event_id'].isin(valid_events)]
    
    # Aggregate by (event_id, quality_tier, time_bin)
    # For each bin, compute sold_count and approximate exposure
    aggregated = []
    
    for (event_id, quality_tier), event_group in df.groupby(['event_id', 'quality_tier']):
        # Sort by time_to_event descending (far to near)
        event_group = event_group.sort_values('time_to_event', ascending=False)
        
        # Compute cumulative inventory (exposure)
        total_sold = event_group['Qty'].sum()
        cumulative_sold = event_group['Qty'].cumsum()
        exposure = total_sold - cumulative_sold + event_group['Qty']
        
        # Group by time_bin
        for time_bin, bin_group in event_group.groupby('time_bin'):
            sold_count = bin_group['Qty'].sum()
            # Exposure for this bin = remaining before bin starts
            bin_exposure = exposure[bin_group.index].iloc[0] if len(bin_group) > 0 else 0
            
            # Average price_rel in this bin
            avg_price_rel = bin_group['price_rel'].mean()
            
            # Get event context (same for all rows in event)
            row = bin_group.iloc[0]
            
            aggregated.append({
                'event_id': event_id,
                'quality_tier': quality_tier,
                'time_bin': time_bin,
                'price_rel': avg_price_rel,
                'sold_count': int(sold_count),
                'exposure': max(1, int(bin_exposure)),  # At least 1 to avoid division by zero
                'day_of_week': row['day_of_week'],
                'is_weekend': row['is_weekend'],
                'is_playoff': row['is_playoff'],
                'year': row['year'],
                'month': row['month'],
                'away_team': row['away_team'],
                'home_team': row['home_team']
            })
    
    conn.close()
    
    result_df = pd.DataFrame(aggregated)
    
    # Compute empirical probability
    result_df['empirical_prob'] = result_df['sold_count'] / result_df['exposure']
    result_df['empirical_prob'] = result_df['empirical_prob'].clip(0.0, 1.0)
    
    return result_df


def get_data_summary(df: pd.DataFrame) -> Dict:
    """Get summary statistics of extracted data."""
    return {
        'n_observations': len(df),
        'n_events': df['event_id'].nunique(),
        'n_quality_tiers': df['quality_tier'].nunique(),
        'n_time_bins': df['time_bin'].nunique(),
        'total_sold': df['sold_count'].sum(),
        'total_exposure': df['exposure'].sum(),
        'overall_sale_rate': df['sold_count'].sum() / df['exposure'].sum(),
        'price_rel_range': (df['price_rel'].min(), df['price_rel'].max()),
        'price_rel_median': df['price_rel'].median()
    }


if __name__ == '__main__':
    # Test extraction
    db_path = Path(__file__).parent.parent / 'data_generation' / 'db.sqlite'
    df = extract_sales_data(db_path)
    print("Extracted data shape:", df.shape)
    print("\nSummary:")
    summary = get_data_summary(df)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("\nFirst few rows:")
    print(df.head())

