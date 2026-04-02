import sqlite3

conn = sqlite3.connect("database.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS exoplanets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    score REAL
)
""")

conn.commit()
conn.close()

print("Table ensured successfully")
