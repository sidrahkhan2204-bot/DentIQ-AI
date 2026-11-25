import sqlite3

conn = sqlite3.connect('database/dental.db')
c = conn.cursor()

for table in ['users', 'doctors', 'appointments', 'detection_history']:
    try:
        c.execute(f'SELECT COUNT(*) FROM {table}')
        count = c.fetchone()[0]
        print(f'{table}: {count}')
    except Exception as e:
        print(f'{table}: ERROR — {e}')

conn.close()