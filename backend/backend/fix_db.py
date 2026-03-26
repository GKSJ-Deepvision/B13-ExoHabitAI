import sqlite3
from datetime import datetime

print("Connecting to database.db ...")

conn = sqlite3.connect("database.db")
cur = conn.cursor()

# Create the table if it doesn't exist
cur.execute("""
CREATE TABLE IF NOT EXISTS exoplanets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT DEFAULT 'Unnamed Planet',
    habitability_score REAL NOT NULL,
    planet_mass_earth REAL,
    orbital_period_days REAL,
    orbit_distance_au REAL,
    star_temperature_k REAL,
    star_radius_solar REAL,
    is_habitable INTEGER DEFAULT 0,
    category TEXT DEFAULT 'Unknown',
    created_at TEXT DEFAULT (datetime('now'))
)
""")

conn.commit()

# Quick verification
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='exoplanets';")
if cur.fetchone():
    print("SUCCESS: exoplanets table is now created or already existed.")
else:
    print("FAILED: table still not found — check folder permissions or db path.")

conn.close()
print("Done. You can now restart the server.")