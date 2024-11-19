"""Movie database class that handles SQLite operations for movie posters.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import sqlite3
from typing import List, Optional, Tuple

import numpy as np


class MovieDB:
    """Movie database class that handles SQLite operations for movie posters.

    The database contains the following table structure:
    - movie_posters:
        - movie_id (INTEGER): Primary key
        - movie_name (TEXT): Name of the movie
        - poster_data (BLOB): Binary image data of the movie poster
    """

    def __init__(self, db_path: str) -> None:
        """Initialize MovieDB with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None
        self._movie_ids: List[Tuple[int]] = []

    def connect(self) -> None:
        """Establish database connection."""
        self._conn = sqlite3.connect(self.db_path)
        self._cursor = self._conn.cursor()

    def disconnect(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            self._cursor = None

    def get_movie_poster(self, movie_id: int) -> Optional[Tuple[str, bytes]]:
        """Get movie poster data by movie ID.

        Args:
            movie_id: ID of the movie

        Returns:
            Tuple of (movie_name, poster_data) if found, None otherwise
        """
        if not self._conn:
            self.connect()

        if not self._cursor:
            return None

        self._cursor.execute(
            "SELECT movie_name, poster_data FROM movie_posters WHERE movie_id = ?",
            (movie_id,),
        )
        result = self._cursor.fetchone()

        if result:
            return result[0], result[1]
        return None

    def get_movie_description(self, movie_id: int) -> Optional[Tuple[str, str]]:
        """Get movie description by movie ID.

        Args:
            movie_id: ID of the movie

        Returns:
            Tuple of (movie_name, movie_description) if found, None otherwise
        """
        if not self._conn:
            self.connect()

        if not self._cursor:
            return None

        self._cursor.execute(
            "SELECT movie_name, description FROM movie_posters WHERE movie_id = ?",
            (movie_id,),
        )
        result = self._cursor.fetchone()

        if result:
            return result[0], result[1]
        return None


    def get_random_movie(self) -> Optional[Tuple[int, str, bytes]]:
        """Get a random movie from the database.

        Returns:
            Tuple of (movie_id, movie_name, poster_data) if found, None otherwise
        """
        if not self._conn:
            self.connect()

        if not self._cursor:
            return None

        if not self._movie_ids:
            self._cursor.execute("SELECT movie_id FROM movie_posters")
            self._movie_ids = self._cursor.fetchall()

        random_idx = np.random.choice(len(self._movie_ids))
        movie_id = self._movie_ids[random_idx][0]

        result = self.get_movie_poster(movie_id)
        if result:
            return movie_id, result[0], result[1]
        return None


if __name__ == "__main__":
    MAIN_DB_PATH = "/home/limjk/Datasets/MovieLens20M/movie_posters.sqlite"
    main_movie_db = MovieDB(MAIN_DB_PATH)
    main_movie_db.connect()

    # Test by retrieving and displaying a random movie
    main_result = main_movie_db.get_random_movie()
    if main_result:
        main_movie_id, movie_name, poster_data = main_result

        # Convert to image and display
        import io

        from PIL import Image

        image = Image.open(io.BytesIO(poster_data))
        image.show()
        print(f"Displayed poster for movie: {movie_name} (ID: {main_movie_id})")

    main_movie_db.disconnect()
