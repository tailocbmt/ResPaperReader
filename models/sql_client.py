import psycopg2
import json
import logging
from typing import Dict, List
from models.enums import PaperSource, AUTHORS, COLUMNS


class SQLClient:
    def __init__(
            self,
            db_config: Dict):
        """Initialize the database connection."""
        self.db_config = db_config
        self._initialize_connection(
            config=self.db_config
        )
        self._initialize_table()

    def _initialize_connection(
            self,
            config: Dict):
        """Initialize a thread-local connection."""
        self.db_conn = psycopg2.connect(
            database=config["db_name"],
            host=config["db_host"],
            user=config["db_user"],
            password=config.get("db_pass", ""),
            port=config["db_port"]
        )

    def close(self):
        """Close the database connection."""
        self.db_conn.close()

    def _get_connection(self):
        """Get the thread-local connection, creating it if needed."""
        if self.db_conn is None:
            self.initialize_connection(
                config=self.db_config
            )

        return self.db_conn

    def _initialize_table(self):
        """Ensure the directory for the database exists."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create papers table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            title TEXT NOT NULL,
            abstract TEXT,
            authors TEXT,
            source TEXT,
            url_path TEXT,
            embedding_id TEXT,
            full_text TEXT
        );
        ''')

        conn.commit()

    def insert_paper(
            self,
            title: str,
            abstract: str,
            authors: str = None,
            source: str = PaperSource.USER.value,
            file_path: str = None,
            embedding_id: str = None,
            full_text: str = None):
        """Add a paper to the database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        authors = json.dumps(authors) if authors else None

        cursor.execute(
            '''
                INSERT INTO papers (title, abstract, authors, source, url_path, embedding_id, full_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            (title, abstract, authors, source, file_path, embedding_id, full_text)
        )

        conn.commit()
        return cursor.lastrowid

    def get_paper_by_id(
            self,
            paper_id: int):
        """Get a paper by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM papers WHERE id = ?;',
            (paper_id)
        )
        paper = cursor.fetchone()

        if len(paper) > 0:
            paper_dict = dict(zip(COLUMNS, paper))
            paper_dict[AUTHORS] = json.loads(
                paper_dict[AUTHORS]) if AUTHORS in paper_dict.keys() else None

            return paper_dict
        return None

    def get_papers(
            self,
            keyword: str = "",
            limit: int = 10
    ) -> List[Dict]:
        """Search papers by keyword in title, abstract, or full text."""
        conn = self._get_connection()
        cursor = conn.cursor()
        query = 'SELECT * FROM papers'
        if keyword:
            query = f'''
                    {query}
                    WHERE title LIKE ? OR abstract LIKE ? OR full_text LIKE ?        
                '''

        cursor.execute(f'''
            {query}
            ORDER BY created_at DESC LIMIT ?;
            ''', (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%', limit)
        )

        papers = cursor.fetchall()

        results = []
        for paper in papers:
            paper_dict = dict(zip(COLUMNS, paper))
            paper_dict[AUTHORS] = json.loads(
                paper_dict[AUTHORS]) if AUTHORS in paper_dict.keys() else None
            results.append(paper_dict)

        return results

    def delete_paper(self, paper_id):
        """Delete a paper from the database.

        Args:
            paper_id: ID of the paper to delete

        Returns:
            Boolean indicating success
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                'SELECT id FROM papers WHERE id = ?;',
                (paper_id)
            )
            result = cursor.fetchall()

            if len(result) == 0:
                return False, "Paper not found"

            cursor.execute(
                'DELETE FROM papers WHERE id = ?;',
                (paper_id)
            )
            conn.commit()

            return True, paper_id
        except Exception as e:
            logging.error(f"Error deleting paper: {e}")
            return False, str(e)
