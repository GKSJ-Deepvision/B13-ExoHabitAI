import sqlite3

db_file = "database.db"

try:
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    # Check if table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='exoplanets'")
    if not cur.fetchone():
        print("Table 'exoplanets' does not exist — nothing to clear.")
    else:
        # Delete all data
        cur.execute("DELETE FROM exoplanets")
        conn.commit()
        print(f"Successfully deleted all {cur.rowcount} rows from exoplanets table.")
        print("Table structure remains — ready for new entries.")

    conn.close()

except Exception as e:
    print("Error:", str(e))