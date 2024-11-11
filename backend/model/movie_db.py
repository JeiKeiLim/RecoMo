"""

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import numpy as np
import sqlite3
from typing import Optional, Tuple


class MovieDB:
    """Movie database class that handles SQLite operations for movie posters.

    The database contains the following table structure:
    - movie_posters:
        - movie_id (INTEGER): Primary key
        - movie_name (TEXT): Name of the movie
        - poster_data (BLOB): Binary image data of the movie poster
    """

    def __init__(self, db_path: str):
        """Initialize MovieDB with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._conn = None
        self._cursor = None

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

        self._cursor.execute(
            "SELECT movie_name, poster_data FROM movie_posters WHERE movie_id = ?",
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

        self._cursor.execute(
            "SELECT movie_id FROM movie_posters"
        )
        movie_ids = self._cursor.fetchall()
        random_idx = np.random.choice(len(movie_ids))
        movie_id = movie_ids[random_idx][0]

        result = self.get_movie_poster(movie_id)
        if result:
            return movie_id, result[0], result[1]
        return None

        # self._cursor.execute(
        #     "SELECT movie_id, movie_name, poster_data FROM movie_posters ORDER BY RANDOM() LIMIT 1"
        # )
        # result = self._cursor.fetchone()

        # if result:
        #     return result[0], result[1], result[2]
        # return None


if __name__ == "__main__":
    db_path = "/home/limjk/Datasets/MovieLens20M/movie_posters.sqlite"
    movie_db = MovieDB(db_path)
    movie_db.connect()

    # Test by retrieving and displaying a random movie
    result = movie_db.get_random_movie()
    if result:
        movie_id, movie_name, poster_data = result

        # Convert to image and display
        import io

        from PIL import Image

        image = Image.open(io.BytesIO(poster_data))
        image.show()
        print(f"Displayed poster for movie: {movie_name} (ID: {movie_id})")

    movie_db.disconnect()
