import sqlite3

DB = 'database/dental.db'
conn = sqlite3.connect(DB)
c = conn.cursor()

def has_column(table, column):
    try:
        c.execute(f'PRAGMA table_info({table})')
        cols = [row[1] for row in c.fetchall()]
        return column in cols
    except Exception:
        return False

def has_table(table):
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return c.fetchone() is not None

print("=" * 50)
print("ADDING MISSING SCHEMA")
print("=" * 50)

# --- ai_app_doctor: new columns ---
if not has_column('ai_app_doctor', 'city'):
    c.execute('ALTER TABLE ai_app_doctor ADD COLUMN city VARCHAR(100)')
    print("  Added column: ai_app_doctor.city")
else:
    print("  ai_app_doctor.city already exists")

if not has_column('ai_app_doctor', 'clinic_address'):
    c.execute('ALTER TABLE ai_app_doctor ADD COLUMN clinic_address VARCHAR(255)')
    print("  Added column: ai_app_doctor.clinic_address")
else:
    print("  ai_app_doctor.clinic_address already exists")

if not has_column('ai_app_doctor', 'available_days'):
    c.execute('ALTER TABLE ai_app_doctor ADD COLUMN available_days VARCHAR(200)')
    print("  Added column: ai_app_doctor.available_days")

if not has_column('ai_app_doctor', 'available_times'):
    c.execute('ALTER TABLE ai_app_doctor ADD COLUMN available_times VARCHAR(200)')
    print("  Added column: ai_app_doctor.available_times")

# --- ai_app_appointment: created_at ---
if not has_column('ai_app_appointment', 'created_at'):
    c.execute("ALTER TABLE ai_app_appointment ADD COLUMN created_at DATETIME")
    c.execute("UPDATE ai_app_appointment SET created_at = datetime('now') WHERE created_at IS NULL")
    print("  Added column: ai_app_appointment.created_at")

# --- ai_app_passwordresetrequest ---
if not has_table('ai_app_passwordresetrequest'):
    c.execute('''
        CREATE TABLE ai_app_passwordresetrequest (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email VARCHAR(254) NOT NULL,
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            token VARCHAR(64) UNIQUE,
            requested_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            resolved_at DATETIME
        )
    ''')
    print("  Created table: ai_app_passwordresetrequest")

# --- ai_app_review ---
if not has_table('ai_app_review'):
    c.execute('''
        CREATE TABLE ai_app_review (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            doctor_id INTEGER NOT NULL,
            appointment_id INTEGER,
            rating INTEGER NOT NULL,
            comment TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES ai_app_user(id),
            FOREIGN KEY (doctor_id) REFERENCES ai_app_doctor(id),
            FOREIGN KEY (appointment_id) REFERENCES ai_app_appointment(id)
        )
    ''')
    print("  Created table: ai_app_review")

conn.commit()
conn.close()

print()
print("=" * 50)
print("DONE — run 'python verify_db.py' to confirm")
print("=" * 50)