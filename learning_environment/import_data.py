"""Main script to import CSV ticket sales data into SQLite database."""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

from db.schema import create_tables, get_or_create_event
from import_utils.filename_parser import parse_filename, format_event_datetime
from import_utils.csv_processor import process_csv_file
from import_utils.ticket_quality import compute_quality_for_event_sales, quality_to_category


def get_venue_from_path(csv_path: Path) -> str:
    """Extract venue name from file path."""
    # Path structure: .../data/VenueName/filename.csv
    parts = csv_path.parts
    if 'data' in parts:
        idx = parts.index('data')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return 'Unknown'


def import_csv_file(
    conn: sqlite3.Connection,
    csv_path: Path,
    venue: Optional[str] = None
) -> int:
    """
    Import a single CSV file into the database.
    
    Returns number of records imported.
    """
    # Parse filename
    event_info = parse_filename(csv_path.name)
    if not event_info:
        print(f"Warning: Could not parse filename: {csv_path.name}")
        return 0
    
    # Format event datetime
    event_date, event_time = format_event_datetime(
        event_info['year'],
        event_info['month'],
        event_info['day'],
        event_info['time']
    )
    
    # Create event datetime for time_to_event calculation
    try:
        hour, minute = map(int, event_time.split(':'))
        event_datetime = datetime(
            event_info['year'],
            event_info['month'],
            event_info['day'],
            hour,
            minute
        )
    except Exception as e:
        print(f"Warning: Could not parse event datetime for {csv_path.name}: {e}")
        return 0
    
    # Get or create event
    if venue is None:
        venue = get_venue_from_path(csv_path)
    
    # Set venue, city, state (TD Garden, Boston, MA)
    event_venue = 'TD Garden'
    event_city = 'Boston'
    event_state = 'MA'
    
    event_id = get_or_create_event(
        conn,
        away_team=event_info['away_team'],
        home_team=event_info['home_team'],
        event_date=event_date,
        event_time=event_time,
        day_of_week=event_info['day_of_week'],
        venue=event_venue,
        city=event_city,
        state=event_state,
        year=event_info['year'],
        month=event_info['month'],
        day=event_info['day']
    )
    
    # Process CSV file
    sales = process_csv_file(csv_path, event_datetime)
    
    # Compute price-driven quality for all sales in this event
    if sales:
        quality_scores = compute_quality_for_event_sales(
            sales,
            price_weight=0.7,
            clearance_weight=0.3
        )
        # Assign quality scores to sales
        for sale, quality_score in zip(sales, quality_scores):
            # Store as numeric string (0.0 to 1.0) for flexibility
            sale['ticket_quality'] = f"{quality_score:.4f}"
    
    # Insert sales records
    cursor = conn.cursor()
    inserted = 0
    
    for sale in sales:
        try:
            cursor.execute("""
                INSERT INTO ticket_sales (
                    event_id, date_time, time_to_event, Zone, Section,
                    Row, Qty, ticket_quality, Price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                sale['date_time'],
                sale['time_to_event'],
                sale['zone'],
                sale['section'],
                sale['row'],
                sale['qty'],
                sale['ticket_quality'],
                sale['price']
            ))
            inserted += 1
        except Exception as e:
            print(f"Error inserting sale record: {e}")
            continue
    
    conn.commit()
    return inserted


def import_all_csvs(db_path: Path, data_dir: Path) -> None:
    """
    Import all CSV files from data directory into database.
    
    Args:
        db_path: Path to SQLite database file
        data_dir: Path to directory containing venue subdirectories with CSV files
    """
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    try:
        # Create tables
        print("Creating database tables...")
        create_tables(conn)
        
        # Find all CSV files
        csv_files = list(data_dir.rglob('*.csv'))
        print(f"Found {len(csv_files)} CSV files to import")
        
        total_imported = 0
        for i, csv_path in enumerate(csv_files, 1):
            print(f"[{i}/{len(csv_files)}] Processing {csv_path.name}...")
            imported = import_csv_file(conn, csv_path)
            total_imported += imported
            print(f"  Imported {imported} records")
        
        print(f"\nImport complete! Total records imported: {total_imported}")
        
        # Print summary
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM events")
        event_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM ticket_sales")
        sale_count = cursor.fetchone()[0]
        
        print(f"\nDatabase summary:")
        print(f"  Events: {event_count}")
        print(f"  Ticket sales: {sale_count}")
        
    finally:
        conn.close()


def main():
    """Main entry point."""
    # Set paths
    script_dir = Path(__file__).parent
    db_path = script_dir / 'db.sqlite'
    data_dir = script_dir / 'data'
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # Import all CSV files
    import_all_csvs(db_path, data_dir)


if __name__ == '__main__':
    main()

