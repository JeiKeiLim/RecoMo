import React, { useEffect, useState } from 'react';
import './App.css';

// Define movie interface
interface Movie {
  id: number;
  name: string;
  poster: string;
  rating?: number;
}

interface RatingRequest {
  movie_id: number;
  rating: number;
}

interface RatingResponse {
  success: boolean;
  movie_id: number;
  rating: number;
}

function App() {
  const [movies, setMovies] = useState<Movie[]>([]);

  const fetchRandomMovie = async () => {
    try {
      const response = await fetch('http://localhost:7777/get_random_movie', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
      });
      
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      
      const movie = await response.json();
      return movie;
    } catch (error) {
      console.error('Error fetching random movie:', error);
      return null;
    }
  };

  const loadInitialMovies = async () => {
    const moviePromises = Array(16).fill(null).map(() => fetchRandomMovie());
    const newMovies = (await Promise.all(moviePromises)).filter((movie): movie is Movie => movie !== null);
    setMovies(newMovies);
  };

  useEffect(() => {
    loadInitialMovies();
  }, []);

  const submitRating = async (movieId: number, rating: number) => {
    try {
      const response = await fetch('http://localhost:7777/submit_rating', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          movie_id: movieId,
          rating: rating
        } as RatingRequest),
      });

      if (!response.ok) {
        throw new Error('Failed to submit rating');
      }

      const result: RatingResponse = await response.json();
      
      if (result.success) {
        setMovies(movies.map(movie => 
          movie.id === movieId ? { ...movie, rating } : movie
        ));
      }
    } catch (error) {
      console.error('Error submitting rating:', error);
      // Optionally add error handling UI
    }
  };

  const handleRating = async (movieId: number, rating: number) => {
    await submitRating(movieId, rating);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Movie Recommender</h1>
      </header>
      <main className="movie-grid">
        {movies.map(movie => (
          <div key={movie.id} className="movie-card">
            <div className="movie-poster">
              <img 
                src={`data:image/jpeg;base64,${movie.poster}`}
                alt={movie.name}
              />
            </div>
            <div className="movie-info">
              <h3>{movie.name}</h3>
              <div className="rating">
                <div className="stars">
                  {[1, 2, 3, 4, 5].map((star) => (
                    <button
                      key={star}
                      onClick={() => handleRating(movie.id, star)}
                      className={`star ${
                        movie.rating && Math.round(movie.rating) >= star ? 'active' : ''
                      }`}
                    >
                      {movie.rating && Math.round(movie.rating) >= star ? '★' : '☆'}
                    </button>
                  ))}
                </div>
                <span className="rating-value">
                  {movie.rating?.toFixed(1) || '0.0'}
                </span>
              </div>
            </div>
          </div>
        ))}
      </main>
    </div>
  );
}

export default App;
