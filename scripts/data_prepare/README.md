# Data Preparation Scripts

This directory contains scripts for preparing movie poster data for the recommendation system.

## Scripts

### grab_movie_poster.py

Downloads movie poster images from TMDB API.

#### Requirements
- TMDB API key (set as environment variable `TMDB_API_KEY`)
- Python packages: requests, pandas, tqdm


### posters_to_sqlite.py

Converts downloaded movie poster images to SQLite database.

#### Database Schema
The script creates a SQLite database with the following table:
- `movie_posters`:
  - `movie_id` (INTEGER): Primary key
  - `movie_name` (TEXT): Name of the movie
  - `poster_data` (BLOB): Binary image data of the movie poster

#### Usage
1. Ensure movie poster images are downloaded using `grab_movie_poster.py`
2. Run the script to create SQLite database:
