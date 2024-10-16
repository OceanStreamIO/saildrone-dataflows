import sqlite3
from datetime import datetime, timezone


current_time = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')
# Configuration
DB_NAME = f'results_{current_time}.db'


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Create tables if they do not exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_name TEXT,
                        folder TEXT,
                        size INTEGER,
                        location TEXT,
                        processed BOOLEAN,
                        last_modified TEXT)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS folders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        folder_name TEXT,
                        number_of_raw_files INTEGER,
                        total_size INTEGER)''')

    conn.commit()
    return conn


def is_file_processed(file_name, folder):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''SELECT id FROM files WHERE file_name=? AND folder=? AND processed=1''', (file_name, folder))
    result = cursor.fetchone()
    conn.close()
    return result is not None


def insert_file_record(file_name, folder, size, location, last_modified):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO files (file_name, folder, size, location, processed, last_modified)
                      VALUES (?, ?, ?, ?, 0, ?)''', (file_name, folder, size, location, last_modified))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id


def mark_file_processed(file_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''UPDATE files SET processed=1 WHERE id=?''', (file_id,))
    conn.commit()
    conn.close()


def update_folder_info(folder, size):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''SELECT id, number_of_raw_files, total_size FROM folders WHERE folder_name=?''', (folder,))
    record = cursor.fetchone()

    if record:
        folder_id, number_of_raw_files, total_size = record
        number_of_raw_files += 1
        total_size += size
        cursor.execute('''UPDATE folders SET number_of_raw_files=?, total_size=? WHERE id=?''',
                       (number_of_raw_files, total_size, folder_id))
    else:
        cursor.execute('''INSERT INTO folders (folder_name, number_of_raw_files, total_size)
                          VALUES (?, 1, ?)''', (folder, size))

    conn.commit()
    conn.close()