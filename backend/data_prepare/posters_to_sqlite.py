"""Convert movie poster images to SQLite database.

This script reads poster images from a directory and stores them in a SQLite database.
The poster filenames are expected to be in the format: {movieId}_{movieName}.jpg
"""

import sqlite3
from pathlib import Path

from tqdm import tqdm


def init_db(db_path: str) -> None:
    """Initialize SQLite database with required table.

    Args:
        db_path: Path to SQLite database file
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create table to store poster images
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS movie_posters (
            movie_id INTEGER PRIMARY KEY,
            movie_name TEXT NOT NULL,
            poster_data BLOB NOT NULL,
            description TEXT
        )
    """
    )

    conn.commit()
    conn.close()


def store_poster(
    db_path: str, movie_id: int, movie_name: str, poster_path: Path, desc_path: Path
) -> bool:
    """Store a single poster image and description in the database.

    Args:
        db_path: Path to SQLite database file
        movie_id: ID of the movie
        movie_name: Name of the movie
        poster_path: Path to the poster image file
        desc_path: Path to the description file

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(poster_path, "rb") as f:
            poster_data = f.read()

        # Try to read description if it exists
        description = None
        if desc_path.exists():
            with open(desc_path, "r", encoding="utf-8") as f:
                description = f.read().strip()

        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        c.execute(
            """INSERT OR REPLACE INTO movie_posters 
               (movie_id, movie_name, poster_data, description) 
               VALUES (?, ?, ?, ?)""",
            (movie_id, movie_name, poster_data, description),
        )

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"Error processing {poster_path}: {e}")
        return False


def process_posters(posters_dir: str, db_path: str) -> None:
    """Process all poster images in directory and store in database.

    Args:
        posters_dir: Directory containing poster images
        db_path: Path to SQLite database file
    """
    # Initialize database
    init_db(db_path)

    # Get list of all jpg files
    poster_files = list(Path(posters_dir).glob("*.jpg"))
    desc_dir = Path(posters_dir) / "descriptions"

    for poster_path in tqdm(poster_files, desc="Storing posters"):
        # Extract movie ID and name from filename
        filename = poster_path.stem  # Get filename without extension
        movie_id = int(filename.split("_")[0])
        movie_name = "_".join(
            filename.split("_")[1:]
        )  # Join remaining parts as movie name

        # Construct description file path
        desc_path = desc_dir / f"{filename}.txt"

        store_poster(db_path, movie_id, movie_name, poster_path, desc_path)


if __name__ == "__main__":
    POSTERS_DIR = "/home/limjk/Datasets/MovieLens20M/posters"
    DB_PATH = "/home/limjk/Datasets/MovieLens20M/movie_posters.sqlite"

    process_posters(POSTERS_DIR, DB_PATH)
    # Test by retrieving and displaying a random poster
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Get random movie_id and name
    c.execute(
        "SELECT movie_id, movie_name, description FROM movie_posters ORDER BY RANDOM() LIMIT 1"
    )
    movie_id, movie_name, description = c.fetchone()

    # Get poster data
    c.execute("SELECT poster_data FROM movie_posters WHERE movie_id = ?", (movie_id,))
    poster_data = c.fetchone()[0]
    conn.close()

    # Convert to image and display
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(poster_data))
    image.show()
    print(f"Displayed poster for movie: {movie_name} (ID: {movie_id})")
    print(f"Description: {description}")