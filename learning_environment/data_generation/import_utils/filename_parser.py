"""Parse CSV filenames to extract event information."""

import re
from typing import Dict, Optional, Tuple
from pathlib import Path


def parse_filename(filename: str) -> Optional[Dict[str, any]]:
    """
    Parse NBA game filename to extract event information.
    
    Expected format: [away]_[home]_[year]_[month]_[day]_[day_of_week]_[time]
    Examples:
    - Cavaliers_Celtics_2024_05_07_Tue_700PM.csv
    - Mavs_Celtics_2024_03_01_730PM.csv (missing day_of_week)
    - Warriors_Celtics_202_03_03_Sun_330PM.csv (typo in year)
    
    Returns dict with event info or None if parsing fails.
    """
    # Remove .csv extension
    base_name = Path(filename).stem
    
    # Pattern 1: Full format with day of week
    # [away]_[home]_[year]_[month]_[day]_[day_of_week]_[time]
    pattern1 = r'^(.+?)_(.+?)_(\d{4})_(\d{1,2})_(\d{1,2})_([A-Za-z]{3})_(\d{1,4}[AP]M)$'
    match = re.match(pattern1, base_name)
    
    if match:
        away, home, year, month, day, day_of_week, time = match.groups()
        return {
            'away_team': away,
            'home_team': home,
            'year': int(year),
            'month': int(month),
            'day': int(day),
            'day_of_week': day_of_week,
            'time': time
        }
    
    # Pattern 2: Missing day of week
    # [away]_[home]_[year]_[month]_[day]_[time]
    pattern2 = r'^(.+?)_(.+?)_(\d{4})_(\d{1,2})_(\d{1,2})_(\d{1,4}[AP]M)$'
    match = re.match(pattern2, base_name)
    
    if match:
        away, home, year, month, day, time = match.groups()
        return {
            'away_team': away,
            'home_team': home,
            'year': int(year),
            'month': int(month),
            'day': int(day),
            'day_of_week': None,
            'time': time
        }
    
    # Pattern 3: Handle typo in year (e.g., 202_ instead of 2024)
    pattern3 = r'^(.+?)_(.+?)_(\d{2,3})_(\d{1,2})_(\d{1,2})_([A-Za-z]{3})_(\d{1,4}[AP]M)$'
    match = re.match(pattern3, base_name)
    
    if match:
        away, home, year_str, month, day, day_of_week, time = match.groups()
        # Try to fix year (e.g., 202 -> 2024, 202_ -> 2024)
        if len(year_str) == 3:
            year = int(year_str + '4')  # Assume 2024 if 202
        else:
            year = int(year_str)
        return {
            'away_team': away,
            'home_team': home,
            'year': year,
            'month': int(month),
            'day': int(day),
            'day_of_week': day_of_week,
            'time': time
        }
    
    return None


def format_event_datetime(year: int, month: int, day: int, time_str: str) -> Tuple[str, str]:
    """
    Format event date and time strings.
    
    Returns (event_date, event_time) as strings.
    event_date format: YYYY-MM-DD
    event_time format: HH:MM (24-hour)
    """
    # Parse time string (e.g., "700PM", "730PM", "100PM", "330PM")
    time_str_upper = time_str.upper()
    is_pm = 'PM' in time_str_upper
    is_am = 'AM' in time_str_upper
    
    # Remove AM/PM from time string
    time_str_clean = time_str_upper.replace('PM', '').replace('AM', '').strip()
    
    # Handle various time formats
    # Examples: "700" (7:00), "730" (7:30), "100" (1:00), "1000" (10:00), "1230" (12:30)
    if len(time_str_clean) == 3:  # e.g., "700" -> 7:00, "100" -> 1:00
        hour = int(time_str_clean[0])
        minute = int(time_str_clean[1:3])
    elif len(time_str_clean) == 4:
        # Could be "1000" (10:00) or "1230" (12:30) or "7300" (7:30)
        # Check if first two digits form a valid hour (00-12)
        first_two = int(time_str_clean[:2])
        if first_two >= 10 and first_two <= 12:
            # Likely HHMM format (e.g., 1000 -> 10:00, 1230 -> 12:30)
            hour = first_two
            minute = int(time_str_clean[2:4])
        else:
            # Likely HMMM format (e.g., 7300 -> 7:30)
            hour = int(time_str_clean[0])
            minute = int(time_str_clean[1:4])
    else:
        # Default parsing - take last 2 digits as minutes
        if len(time_str_clean) >= 2:
            minute = int(time_str_clean[-2:])
            hour = int(time_str_clean[:-2]) if len(time_str_clean) > 2 else 0
        else:
            hour = int(time_str_clean)
            minute = 0
    
    # Convert to 24-hour format
    if is_pm and hour != 12:
        hour += 12
    elif is_am and hour == 12:
        hour = 0
    
    # Format date
    event_date = f"{year:04d}-{month:02d}-{day:02d}"
    event_time = f"{hour:02d}:{minute:02d}"
    
    return event_date, event_time

