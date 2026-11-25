import sqlite3
import os

DB_PATH = 'database/dental.db'

# Backup verification
if not os.path.exists('database/dental_backup_before_rename.db'):
    print("WARNING: Backup not found. Run 'copy database\\dental.db database\\dental_backup_before_rename.db' first.")
    print("Continuing anyway...")

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Step 1: Create Django-managed tables for our models if they don't exist
# We use CREATE TABLE ... AS SELECT to copy structure and data

print("=== Starting table rename ===")

# Check if tables exist and their row counts
tables = ['users', 'doctors', 'appointments', 'detection_history']
for t in tables:
    try:
        c.execute(f'SELECT COUNT(*) FROM {t}')
        print(f"  {t}: {c.fetchone()[0]} rows")
    except Exception as e:
        print(f"  {t}: {e}")

# ---- Rename users -> ai_app_user ----
try:
    c.execute("SELECT COUNT(*) FROM ai_app_user")
    print("ai_app_user already exists, skipping rename")
except Exception:
    print("Renaming users -> ai_app_user")
    c.execute("ALTER TABLE users RENAME TO ai_app_user")

# ---- Rename doctors -> ai_app_doctor ----
try:
    c.execute("SELECT COUNT(*) FROM ai_app_doctor")
    print("ai_app_doctor already exists, skipping rename")
except Exception:
    print("Renaming doctors -> ai_app_doctor")
    c.execute("ALTER TABLE doctors RENAME TO ai_app_doctor")

# ---- Rename appointments -> ai_app_appointment ----
try:
    c.execute("SELECT COUNT(*) FROM ai_app_appointment")
    print("ai_app_appointment already exists, skipping rename")
except Exception:
    print("Renaming appointments -> ai_app_appointment")
    c.execute("ALTER TABLE appointments RENAME TO ai_app_appointment")

# ---- Rename detection_history -> ai_app_detectionhistory ----
try:
    c.execute("SELECT COUNT(*) FROM ai_app_detectionhistory")
    print("ai_app_detectionhistory already exists, skipping rename")
except Exception:
    print("Renaming detection_history -> ai_app_detectionhistory")
    c.execute("ALTER TABLE detection_history RENAME TO ai_app_detectionhistory")

conn.commit()
conn.close()

print("=== Rename complete ===")
print("Now run: python manage.py migrate")