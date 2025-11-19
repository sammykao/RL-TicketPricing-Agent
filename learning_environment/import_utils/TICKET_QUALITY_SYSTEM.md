# Price-Driven Ticket Quality System

## Overview

The ticket quality system uses a **data-driven, market-validated approach** based on price percentiles and clearance speed within each event. This method is used by major ticket brokers, dynamic pricing engines, and quantitative trading groups.

## Methodology

### Formula

```
quality = 0.7 × price_percentile + 0.3 × clearance_score
```

Where:
- **price_percentile**: Percentile rank of the ticket price within the event (0.0 to 1.0)
- **clearance_score**: Normalized score based on when the ticket sold relative to the event (0.0 to 1.0)

### Step 1: Price Percentile Calculation

For each event, we compute the percentile rank of each ticket's price:

```python
percentile = (number of tickets with price ≤ this_price) / total_tickets
```

**Example:**
- If a ticket sells at the 85th percentile price → `price_percentile = 0.85`
- If a ticket sells at the 20th percentile price → `price_percentile = 0.20`

### Step 2: Clearance Score Calculation

The clearance score measures how quickly tickets sold relative to the event:

```python
clearance_score = 1 - ((time_to_event - min_time) / (max_time - min_time))
```

**Interpretation:**
- Tickets sold **closer to the event** (smaller `time_to_event`) = higher clearance score
- Tickets sold **far from the event** (larger `time_to_event`) = lower clearance score

This captures demand dynamics: high-demand seats often sell closer to the event.

### Step 3: Quality Score

The final quality score combines both signals:

```python
quality = 0.7 × price_percentile + 0.3 × clearance_score
```

**Result:** A value between 0.0 and 1.0, where:
- **0.75 - 1.0**: Premium quality
- **0.50 - 0.75**: High quality
- **0.25 - 0.50**: Medium quality
- **0.0 - 0.25**: Low quality

## Why This Approach?

### Advantages

1. **Market-Validated**: Quality is determined by actual market behavior, not assumptions
2. **Dynamic**: Adapts to each event's unique price distribution
3. **Demand-Driven**: Captures real demand signals through price and timing
4. **No Geometry Required**: Works without detailed seat maps or coordinates
5. **Generalizable**: Works across different events, venues, and teams

### What It Captures

The price distribution automatically captures:
- Better view
- Better row
- Better angle
- Closeness to action
- VIP perks
- Rarity
- Scarcity
- Demand strength

All without needing explicit geometry or section mappings.

## Implementation

### Database Storage

Quality is stored as a numeric string (e.g., "0.8542") in the `ticket_quality` column of the `ticket_sales` table.

### Usage

#### During Import

When importing new CSV files, quality is automatically computed:

```python
python import_data.py
```

The system:
1. Processes all sales for an event
2. Computes price percentiles within the event
3. Computes clearance scores
4. Calculates final quality scores
5. Stores in database

#### Updating Existing Data

To update quality for existing records:

```python
python update_ticket_quality.py
```

This recalculates quality for all events in the database.

## Quality Distribution

Current distribution across all events:
- **Premium** (0.75-1.0): 23,015 tickets (24.8%)
- **High** (0.50-0.75): 32,903 tickets (35.5%)
- **Medium** (0.25-0.50): 30,282 tickets (32.7%)
- **Low** (0.0-0.25): 6,472 tickets (7.0%)

## Example Queries

### Get quality distribution for an event

```sql
SELECT 
    CASE 
        WHEN CAST(ticket_quality AS REAL) >= 0.75 THEN 'Premium'
        WHEN CAST(ticket_quality AS REAL) >= 0.50 THEN 'High'
        WHEN CAST(ticket_quality AS REAL) >= 0.25 THEN 'Medium'
        ELSE 'Low'
    END as quality_category,
    COUNT(*) as count,
    AVG(Price) as avg_price
FROM ticket_sales
WHERE event_id = 1
GROUP BY quality_category
ORDER BY avg_price DESC;
```

### Find highest quality tickets

```sql
SELECT 
    Zone, Section, Row, Price, ticket_quality, time_to_event
FROM ticket_sales
WHERE event_id = 1
ORDER BY CAST(ticket_quality AS REAL) DESC
LIMIT 10;
```

### Compare quality vs price

```sql
SELECT 
    CAST(ticket_quality AS REAL) as quality,
    AVG(Price) as avg_price,
    COUNT(*) as count
FROM ticket_sales
WHERE event_id = 1 AND Price IS NOT NULL
GROUP BY quality
ORDER BY quality DESC;
```

## Tuning Weights

You can adjust the weights in the formula by modifying:

```python
quality = price_weight × price_percentile + clearance_weight × clearance_score
```

Default weights:
- `price_weight = 0.7` (70% weight on price)
- `clearance_weight = 0.3` (30% weight on clearance speed)

To change weights, modify:
- `import_data.py` (line 94-95) for new imports
- `update_ticket_quality.py` (line 48-49) for updates

## Files

- **`import_utils/ticket_quality.py`**: Core quality computation functions
- **`import_data.py`**: Automatic quality calculation during import
- **`update_ticket_quality.py`**: Script to update existing records

## Notes

- Quality is computed **per event** to ensure fair comparison
- Missing prices result in quality based only on clearance score
- Negative `time_to_event` (sold after event) are excluded from clearance calculation
- Quality scores are stored with 4 decimal places for precision

