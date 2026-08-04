import sqlite3
import os

class LeadStore:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT,
                lead_type TEXT NOT NULL,
                area_interests TEXT,
                message TEXT,
                level INTEGER NOT NULL,
                price INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'novo',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def create(self, data):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """INSERT INTO leads (name, email, phone, lead_type, area_interests, message, level, price, status) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data.get("name"), 
                data.get("email"), 
                data.get("phone"), 
                data.get("type"), 
                str(data.get("area_interests", [])), 
                data.get("message"), 
                data.get("level"), 
                data.get("price"), 
                data.get("status")
            )
        )
        conn.commit()
        lead_id = cursor.lastrowid
        conn.close()
        return lead_id

    def get(self, lead_id):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM leads WHERE id = ?", (lead_id,))
        row = cursor.fetchone()
        conn.close()
        if row is None:
            return None
        return dict(row)

    def list(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM leads ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def update(self, lead_id, data):
        conn = sqlite3.connect(self.db_path)
        fields = ", ".join(f"{k} = ?" for k in data.keys())
        values = list(data.values()) + [lead_id]
        conn.execute(f"UPDATE leads SET {fields} WHERE id = ?", values)
        conn.commit()
        conn.close()

    def delete(self, lead_id):
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM leads WHERE id = ?", (lead_id,))
        conn.commit()
        conn.close()