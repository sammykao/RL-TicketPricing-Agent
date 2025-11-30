"""Price-driven ticket quality determination based on market data."""

from typing import List, Dict, Optional, Tuple


def compute_price_percentile(
    price: float,
    all_prices: List[float]
) -> float:
    """
    Compute percentile rank of a price within a list of prices.
    
    Args:
        price: The price to rank
        all_prices: All prices in the event (for percentile calculation)
    
    Returns:
        Percentile rank (0.0 to 1.0)
    """
    if not all_prices or price is None:
        return 0.0
    
    # Filter out None values
    valid_prices = [p for p in all_prices if p is not None]
    if not valid_prices:
        return 0.0
    
    # Handle edge case where all prices are the same
    if len(set(valid_prices)) == 1:
        return 0.5  # Middle percentile if all same
    
    # Compute percentile rank
    percentile = sum(1 for p in valid_prices if p <= price) / len(valid_prices)
    return percentile


def compute_clearance_score(
    time_to_event: Optional[float],
    all_times_to_event: List[Optional[float]]
) -> float:
    """
    Compute clearance score based on when ticket sold relative to event.
    
    Higher score = sold closer to event (higher demand indicator)
    Lower score = sold far from event (lower demand indicator)
    
    Args:
        time_to_event: Hours until event for this sale
        all_times_to_event: All time_to_event values for the event
    
    Returns:
        Normalized clearance score (0.0 to 1.0)
    """
    if time_to_event is None:
        return 0.5  # Default middle score if missing
    
    # Filter out None values and negative (sold after event)
    valid_times = [t for t in all_times_to_event if t is not None and t >= 0]
    if not valid_times:
        return 0.5
    
    # Normalize: tickets sold closer to event (smaller time_to_event) = higher score
    max_time = max(valid_times)
    min_time = min(valid_times)
    
    if max_time == min_time:
        return 0.5  # All sold at same time
    
    # Invert: smaller time_to_event = higher score
    # Normalize to 0-1 range
    normalized = 1.0 - ((time_to_event - min_time) / (max_time - min_time))
    return max(0.0, min(1.0, normalized))  # Clamp to [0, 1]


def compute_price_driven_quality(
    price: Optional[float],
    time_to_event: Optional[float],
    all_prices: List[float],
    all_times_to_event: List[Optional[float]],
    price_weight: float = 0.7,
    clearance_weight: float = 0.3
) -> float:
    """
    Compute price-driven quality score.
    
    Formula: quality = price_weight * price_percentile + clearance_weight * clearance_score
    
    Args:
        price: Ticket price
        time_to_event: Hours until event
        all_prices: All prices in the event (for percentile calculation)
        all_times_to_event: All time_to_event values in the event
        price_weight: Weight for price percentile (default 0.7)
        clearance_weight: Weight for clearance score (default 0.3)
    
    Returns:
        Quality score (0.0 to 1.0)
    """
    if price is None:
        # If no price, use only clearance score (scaled)
        clearance_score = compute_clearance_score(time_to_event, all_times_to_event)
        return clearance_score * clearance_weight
    
    # Compute price percentile
    price_percentile = compute_price_percentile(price, all_prices)
    
    # Compute clearance score
    clearance_score = compute_clearance_score(time_to_event, all_times_to_event)
    
    # Combine with weights
    quality = (price_weight * price_percentile) + (clearance_weight * clearance_score)
    
    return max(0.0, min(1.0, quality))  # Clamp to [0, 1]


def compute_quality_for_event_sales(
    sales: List[Dict],
    price_weight: float = 0.7,
    clearance_weight: float = 0.3
) -> List[float]:
    """
    Compute quality scores for all sales in an event.
    
    This function processes all sales together to compute percentiles
    and clearance scores relative to the event.
    
    Args:
        sales: List of sale dictionaries with 'price' and 'time_to_event' keys
        price_weight: Weight for price percentile (default 0.7)
        clearance_weight: Weight for clearance score (default 0.3)
    
    Returns:
        List of quality scores (0.0 to 1.0) corresponding to each sale
    """
    # Extract all prices and times
    all_prices = [sale.get('price') for sale in sales]
    all_times_to_event = [sale.get('time_to_event') for sale in sales]
    
    # Compute quality for each sale
    qualities = []
    for sale in sales:
        quality = compute_price_driven_quality(
            price=sale.get('price'),
            time_to_event=sale.get('time_to_event'),
            all_prices=all_prices,
            all_times_to_event=all_times_to_event,
            price_weight=price_weight,
            clearance_weight=clearance_weight
        )
        qualities.append(quality)
    
    return qualities


def quality_to_category(quality: float) -> str:
    """
    Convert numeric quality score (0.0-1.0) to category string.
    
    Categories:
    - Premium: 0.75-1.0
    - High: 0.50-0.75
    - Medium: 0.25-0.50
    - Low: 0.0-0.25
    
    Args:
        quality: Quality score (0.0 to 1.0)
    
    Returns:
        Category string
    """
    if quality >= 0.75:
        return "Premium"
    elif quality >= 0.50:
        return "High"
    elif quality >= 0.25:
        return "Medium"
    else:
        return "Low"
