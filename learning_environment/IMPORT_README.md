# CSV to SQLite Import Script

This script imports NBA ticket sales CSV files into a SQLite database for use in the RL ticket pricing agent.

## Usage

```bash
python import_data.py
```

The script will:
1. Scan the `data/` directory for all CSV files
2. Parse filenames to extract event information
3. Create/update the SQLite database (`db.sqlite`)
4. Import all ticket sales records

## Database Schema

### Events Table
- `event_id` (PRIMARY KEY)
- `away_team` - Away team name
- `home_team` - Home team name
- `event_date` - Event date (YYYY-MM-DD)
- `event_time` - Event time (HH:MM, 24-hour format)
- `day_of_week` - Day of week (Mon, Tue, etc.)
- `venue` - Venue name (extracted from directory)
- `year`, `month`, `day` - Date components
- `created_at` - Timestamp when record was created

### Ticket Sales Table
- `sale_id` (PRIMARY KEY)
- `event_id` (FOREIGN KEY) - Links to events table
- `date_time` - Sale datetime (ISO format)
- `time_to_event` - Hours until event (positive = before event, negative = after)
- `Zone` - Ticket zone (e.g., "Balcony", "Loge", "Club")
- `Section` - Section identifier
- `Ticket` - Ticket identifier (NULL, not in CSV)
- `Row` - Row identifier
- `Qty` - Quantity of tickets
- `ticket_quality` - Ticket quality (NULL, not in CSV)
- `Price` - Ticket price

## Filename Format

CSV files should follow the naming pattern:
```
[away_team]_[home_team]_[year]_[month]_[day]_[day_of_week]_[time].csv
```

Examples:
- `Cavaliers_Celtics_2024_05_07_Tue_700PM.csv`
- `Mavs_Celtics_2024_03_01_730PM.csv` (day_of_week optional)

## CSV Format

CSV files should have the following columns:
- `Date/Time (EDT)` - Sale datetime
- `Zone` - Ticket zone
- `Section` - Section identifier
- `Row` - Row identifier
- `Qty` - Quantity
- `Price` - Price

## Project Structure

```
learning_environment/
├── db/
│   ├── __init__.py
│   └── schema.py          # Database schema definitions
├── import_utils/
│   ├── __init__.py
│   ├── filename_parser.py # Filename parsing logic
│   └── csv_processor.py   # CSV processing logic
├── import_data.py         # Main import script
├── db.sqlite              # SQLite database (created by script)
└── data/                  # CSV files directory
    ├── TD_Garden/
    └── Madison_Square_Garden/
```

## Notes

- The script handles various time formats (e.g., "700PM", "730PM", "100PM")
- Missing fields (Ticket, ticket_quality) are set to NULL
- Duplicate events are handled (same away/home/date/time = same event_id)
- The script calculates `time_to_event` automatically from sale datetime and event datetime

