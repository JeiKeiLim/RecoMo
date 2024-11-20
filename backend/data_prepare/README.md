# Data Preparation Scripts

This directory contains scripts for preparing movie poster data for the recommendation system.

## Scripts

### grab_movie_poster.py

Downloads movie poster images from TMDB API. This scripts takes about a day to download all the movie posters from the MovieLens20M dataset.

#### Requirements
- TMDB API key (set as environment variable `TMDB_API_KEY`)
    - You can get the API key from [here](https://www.themoviedb.org/documentation/api)
- Python packages: requests, pandas, tqdm

#### Modify the following variables in the script:
```python
...
data = pd.read_csv("~/Datasets/MovieLens20M/movie.csv")  # Path to movie.csv
...
ROOT_DIR = "/home/limjk/Datasets/MovieLens20M/posters"  # Path to save movie posters
```

#### Usage
```shell
TMDB_API_KEY=your_api_key python grab_movie_poster.py
```


### posters_to_sqlite.py

Converts downloaded movie poster images to SQLite database.

#### Database Schema
The script creates a SQLite database with the following table:
- `movie_posters`:
  - `movie_id` (INTEGER): Primary key
  - `movie_name` (TEXT): Name of the movie
  - `poster_data` (BLOB): Binary image data of the movie poster
  - `description` (TEXT): Movie description text

#### Modify the following variables in the script:
```python
POSTERS_DIR = "/home/limjk/Datasets/MovieLens20M/posters"  # Path to movie posters
DB_PATH = "/home/limjk/Datasets/MovieLens20M/movie_posters.sqlite"  # Path to save SQLite database
```

#### Usage
1. Ensure movie poster images are downloaded using `grab_movie_poster.py`
2. Run the script to create SQLite database:
```shell
TMDB_API_KEY=your_api_key python posters_to_sqlite.py
```
