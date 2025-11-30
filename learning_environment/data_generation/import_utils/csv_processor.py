"""Process CSV files and transform data for database insertion."""

import csv
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path


def parse_sale_datetime(date_time_str: str) -> Optional[datetime]:
    """
    Parse sale datetime from CSV format.
    
    Expected format: "MM-DD-YY / HH:MM AM/PM"
    Example: "05-07-24 / 04:48 PM"
    """
    try:
        # Remove quotes if present
        date_time_str = date_time_str.strip('"')
        
        # Split date and time
        parts = date_time_str.split(' / ')
        if len(parts) != 2:
            return None
        
        date_str, time_str = parts
        
        # Parse date (MM-DD-YY)
        date_parts = date_str.split('-')
        if len(date_parts) != 3:
            return None
        
        month, day, year = map(int, date_parts)
        # Convert 2-digit year to 4-digit (assuming 2000s)
        if year < 100:
            year += 2000
        
        # Parse time (HH:MM AM/PM)
        time_match = re.match(r'(\d{1,2}):(\d{2})\s*(AM|PM)', time_str, re.IGNORECASE)
        if not time_match:
            return None
        
        hour, minute = int(time_match.group(1)), int(time_match.group(2))
        am_pm = time_match.group(3).upper()
        
        # Convert to 24-hour format
        if am_pm == 'PM' and hour != 12:
            hour += 12
        elif am_pm == 'AM' and hour == 12:
            hour = 0
        
        return datetime(year, month, day, hour, minute)
    except Exception:
        return None


def calculate_time_to_event(sale_datetime: datetime, event_datetime: datetime) -> Optional[float]:
    """
    Calculate time to event in hours.
    
    Returns positive value if sale is before event, negative if after.
    """
    if sale_datetime is None or event_datetime is None:
        return None
    
    delta = event_datetime - sale_datetime
    return delta.total_seconds() / 3600.0  # Convert to hours


def process_csv_file(
    csv_path: Path,
    event_datetime: datetime
) -> List[Dict[str, any]]:
    """
    Process a CSV file and return list of ticket sale records.
    
    Args:
        csv_path: Path to CSV file
        event_datetime: Datetime of the event
    
    Returns:
        List of dictionaries with ticket sale data
    """
    sales = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Parse sale datetime
                date_time_str = row.get('Date/Time (EDT)', '').strip('"')
                sale_datetime = parse_sale_datetime(date_time_str)
                
                # Calculate time to event
                time_to_event = calculate_time_to_event(sale_datetime, event_datetime) if sale_datetime else None
                
                # Extract other fields
                zone = row.get('Zone', '').strip('"') or None
                section = row.get('Section', '').strip('"') or None
                row_num = row.get('Row', '').strip('"') or None
                
                # Parse quantity
                qty_str = row.get('Qty', '').strip('"')
                try:
                    qty = int(qty_str) if qty_str else None
                except ValueError:
                    qty = None
                
                # Parse price
                price_str = row.get('Price', '').strip('"')
                try:
                    price = float(price_str) if price_str else None
                except ValueError:
                    price = None
                
                # Format datetime as ISO string for storage
                date_time_iso = sale_datetime.isoformat() if sale_datetime else None
                
                # Note: ticket_quality will be computed after all sales are collected
                # to enable percentile-based calculation within the event
                sales.append({
                    'date_time': date_time_iso,
                    'sale_datetime': sale_datetime,
                    'time_to_event': time_to_event,
                    'zone': zone,
                    'section': section,
                    'row': row_num,
                    'qty': qty,
                    'ticket_quality': None,  # Will be computed later
                    'price': price
                })
    
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return []
    
    return sales

