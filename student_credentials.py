"""Encrypted, expiring student credentials, scoped to one browser and course."""
import hashlib
import json
import secrets
import time

class StudentCredentials:
    def __init__(self, store):
        self.store = store
        with store._connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS student_credentials (
                token_hash TEXT PRIMARY KEY, course_id TEXT NOT NULL,
                encrypted TEXT NOT NULL, expires_at REAL NOT NULL)''')
            db.execute('CREATE INDEX IF NOT EXISTS student_credentials_expiry ON student_credentials(expires_at)')

    def purge(self):
        with self.store._connect() as db:
            db.execute('DELETE FROM student_credentials WHERE expires_at <= ?', (time.time(),))

    def get(self, token, course_id):
        self.purge()
        if not token:
            return None
        with self.store._connect() as db:
            row = db.execute('SELECT encrypted, expires_at FROM student_credentials WHERE token_hash=? AND course_id=?',
                             (hashlib.sha256(token.encode()).hexdigest(), course_id)).fetchone()
        if not row:
            return None
        data = json.loads(self.store.fernet.decrypt(row['encrypted'].encode()))
        data['expires_at'] = row['expires_at']
        return data

    def forget(self, token, course_id):
        if token:
            with self.store._connect() as db:
                db.execute('DELETE FROM student_credentials WHERE token_hash=? AND course_id=?',
                           (hashlib.sha256(token.encode()).hexdigest(), course_id))

    def save(self, previous, course_id, data, weeks):
        if weeks not in (1, 2, 3, 4):
            raise ValueError('Choose 1, 2, 3, or 4 weeks.')
        self.purge()
        token = secrets.token_urlsafe(32)
        expiry = time.time() + weeks * 7 * 86400
        encrypted = self.store.fernet.encrypt(json.dumps(data).encode()).decode()
        with self.store._connect() as db:
            if previous:
                db.execute('DELETE FROM student_credentials WHERE token_hash=? AND course_id=?',
                           (hashlib.sha256(previous.encode()).hexdigest(), course_id))
            db.execute('INSERT INTO student_credentials VALUES (?, ?, ?, ?)',
                       (hashlib.sha256(token.encode()).hexdigest(), course_id, encrypted, expiry))
        return token, expiry
