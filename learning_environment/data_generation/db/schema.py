"""Database schema definitions for ticket sales and events."""

import sqlite3
from typing import Optional


def create_tables(conn: sqlite3.Connection) -> None:
    """Create events and ticket_sales tables if they don't exist."""
    cursor = conn.cursor()
    
    # Events table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            away_team TEXT NOT NULL,
            home_team TEXT NOT NULL,
            event_date TEXT NOT NULL,
            event_time TEXT NOT NULL,
            day_of_week TEXT,
            venue TEXT,
            city TEXT,
            state TEXT,
            year INTEGER,
            month INTEGER,
            day INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(away_team, home_team, event_date, event_time)
        )
    """)
    
    # Ticket sales table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticket_sales (
            sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            date_time TEXT NOT NULL,
            time_to_event REAL,
            Zone TEXT,
            Section TEXT,
            Row TEXT,
            Qty INTEGER,
            ticket_quality TEXT,
            Price REAL,
            FOREIGN KEY (event_id) REFERENCES events(event_id)
        )
    """)
    
    # Create indexes separately
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_event_id ON ticket_sales(event_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_date_time ON ticket_sales(date_time)
    """)
    
    conn.commit()


def get_or_create_event(
    conn: sqlite3.Connection,
    away_team: str,
    home_team: str,
    event_date: str,
    event_time: str,
    day_of_week: Optional[str] = None,
    venue: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    day: Optional[int] = None
) -> int:
    """Get existing event_id or create new event and return event_id."""
    cursor = conn.cursor()
    
    # Check if event exists
    cursor.execute("""
        SELECT event_id FROM events
        WHERE away_team = ? AND home_team = ? AND event_date = ? AND event_time = ?
    """, (away_team, home_team, event_date, event_time))
    
    result = cursor.fetchone()
    if result:
        return result[0]
    
    # Create new event
    cursor.execute("""
        INSERT INTO events (
            away_team, home_team, event_date, event_time, day_of_week,
            venue, city, state, year, month, day
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (away_team, home_team, event_date, event_time, day_of_week,
          venue, city, state, year, month, day))
    
    conn.commit()
    return cursor.lastrowid


def insert_ticket_sale(
    conn: sqlite3.Connection,
    event_id: int,
    date_time: str,
    time_to_event: Optional[float],
    zone: Optional[str],
    section: Optional[str],
    row: Optional[str],
    qty: Optional[int],
    ticket_quality: Optional[str],
    price: Optional[float]
) -> None:
    """Insert a single ticket sale record."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ticket_sales (
            event_id, date_time, time_to_event, Zone, Section,
            Row, Qty, ticket_quality, Price
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (event_id, date_time, time_to_event, zone, section,
          row, qty, ticket_quality, price))
    conn.commit()

