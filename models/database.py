import sqlite3
import os
import json
import threading
from datetime import datetime


class DatabaseManager:
    def __init__(self, db_path="../data/papers.db"):
        """Initialize the database connection."""
        self.db_path = db_path
        self.ensure_db_directory()
        self._local = threading.local()
        self._initialize_connection()
        self.create_tables()

    def ensure_db_directory(self):
        """Ensure the directory for the database exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def _initialize_connection(self):
        """Initialize a thread-local connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                self.db_path, check_same_thread=False)

    def _get_connection(self):
        """Get the thread-local connection, creating it if needed."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._initialize_connection()
        return self._local.conn

    def create_tables(self):
        """Create necessary tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create papers table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            abstract TEXT,
            authors TEXT,
            source TEXT,
            file_path TEXT,
            embedding_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        conn.commit()

    def add_paper(self, title, abstract, authors=None, source="internal_upload", file_path=None, embedding_id=None):
        """Add a paper to the database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        authors_json = json.dumps(authors) if authors else None

        cursor.execute('''
        INSERT INTO papers (title, abstract, authors, source, file_path, embedding_id)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (title, abstract, authors_json, source, file_path, embedding_id))

        conn.commit()
        return cursor.lastrowid

    def get_paper(self, paper_id):
        """Get a paper by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM papers WHERE id = ?', (paper_id,))
        paper = cursor.fetchone()

        if paper:
            columns = [desc[0] for desc in cursor.description]
            paper_dict = dict(zip(columns, paper))
            if paper_dict.get('authors'):
                paper_dict['authors'] = json.loads(paper_dict['authors'])
            return paper_dict
        return None

    def search_papers(self, keyword, limit=5):
        """Search papers by keyword in title or abstract."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        SELECT * FROM papers 
        WHERE title LIKE ? OR abstract LIKE ?
        ORDER BY created_at DESC LIMIT ?
        ''', (f'%{keyword}%', f'%{keyword}%', limit))

        papers = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        results = []
        for paper in papers:
            paper_dict = dict(zip(columns, paper))
            if paper_dict.get('authors'):
                paper_dict['authors'] = json.loads(paper_dict['authors'])
            results.append(paper_dict)

        return results

    def get_all_papers(self, limit=10):
        """Get all papers with optional limit."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM papers ORDER BY created_at DESC LIMIT ?', (limit,))

        papers = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        results = []
        for paper in papers:
            paper_dict = dict(zip(columns, paper))
            if paper_dict.get('authors'):
                paper_dict['authors'] = json.loads(paper_dict['authors'])
            results.append(paper_dict)

        return results

    def close(self):
        """Close the database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()
