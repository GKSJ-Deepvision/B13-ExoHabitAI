import sqlite3

conn = sqlite3.connect("database.db")
cur = conn.cursor()

# Delete rows where key parameters are NULL
cur.execute("""
    DELETE FROM exoplanets 
    WHERE planet_mass_earth IS NULL 
       OR orbital_period_days IS NULL 
       OR orbit_distance_au IS NULL
""")

deleted = cur.rowcount
conn.commit()
conn.close()

print(f"Deleted {deleted} old/incomplete rows. Kept only rows with full data.")