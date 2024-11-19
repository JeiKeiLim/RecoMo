import React, { useEffect, useState } from 'react';
import { Movie, RatingRequest, RatingResponse } from './utils/types';import './App.css';

const API_BASE_URL = 'http://home.limjk.ai:57777';

function App() {
  const [movies, setMovies] = useState<Movie[]>([]);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [currentPage, setCurrentPage] = useState('recommendations');
  const [myRatings, setMyRatings] = useState<{ [key: number]: number }>({});
  const [predictedRatings, setPredictedRatings] = useState<{ [key: number]: number }>({});
  const [ratedMovies, setRatedMovies] = useState<Movie[]>([]);
  const [selectedMovie, setSelectedMovie] = useState<Movie | null>(null);

  const fetchRandomMovie = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/get_random_movie`, {
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

  useEffect(() => {
    // Add a cleanup flag to prevent double loading
    let mounted = true;

    const loadInitialMovies = async () => {
      const moviePromises = Array(16).fill(null).map(() => fetchRandomMovie());
      const newMovies = (await Promise.all(moviePromises)).filter((movie): movie is Movie => movie !== null);
      if (mounted) {
        setMovies(newMovies);
      }
    };

    loadInitialMovies();

    // Cleanup function
    return () => {
      mounted = false;
    };
  }, []); // Keep empty dependency array

  const submitRating = async (movieId: number, rating: number) => {
    try {
      const response = await fetch(`${API_BASE_URL}/submit_rating`, {
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
          movie.id === movieId 
            ? { ...movie, rating, predicted_rating: result.predicted_rating } 
            : movie
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

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const handleMenuSelect = (page: string) => {
    setCurrentPage(page);
    setIsMenuOpen(false);
  };

  const fetchMovie = async (movieId: number): Promise<Movie | null> => {
    try {
      const response = await fetch(`${API_BASE_URL}/get_movie`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ movie_id: movieId }),
      });
      
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      
      const movie = await response.json();
      return movie;
    } catch (error) {
      console.error('Error fetching movie:', error);
      return null;
    }
  };

  const loadRatedMovies = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/get_my_ratings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        throw new Error('Failed to fetch ratings');
      }
      
      const data = await response.json();
      setMyRatings(data.ratings);
      setPredictedRatings(data.predicted_ratings);

      // Fetch movie details for each rated movie
      const movieIds = Object.keys(data.ratings).map(Number);
      const moviePromises = movieIds.map(id => fetchMovie(id));
      const movies = (await Promise.all(moviePromises)).filter((movie): movie is Movie => movie !== null);
      setRatedMovies(movies);
    } catch (error) {
      console.error('Error loading rated movies:', error);
    }
  };

  useEffect(() => {
    if (currentPage === 'myratings') {
      loadRatedMovies();
    }
  }, [currentPage]);

  const handlePosterClick = (movie: Movie) => {
    setSelectedMovie(movie);
  };

  const handleCloseModal = () => {
    setSelectedMovie(null);
  };

  const renderModal = () => {
    if (!selectedMovie) return null;

    return (
      <div className="modal-overlay" onClick={handleCloseModal}>
        <div className="modal-content" onClick={e => e.stopPropagation()}>
          <div className="modal-poster">
            <img 
              src={`data:image/jpeg;base64,${selectedMovie.poster}`}
              alt={selectedMovie.name}
            />
          </div>
          <div className="modal-info">
            <h2>{selectedMovie.name}</h2>
            <p>{selectedMovie.description}</p>
          </div>
          <button className="modal-close" onClick={handleCloseModal}>×</button>
        </div>
      </div>
    );
  };

  const renderMyRatings = () => {
    // Calculate MSE and RMSE
    const ratedMoviesWithPredictions = ratedMovies.filter(
      movie => myRatings[movie.id] && predictedRatings[movie.id]
    );
    
    const mse = ratedMoviesWithPredictions.reduce((sum, movie) => {
      const diff = myRatings[movie.id] - predictedRatings[movie.id];
      return sum + (diff * diff);
    }, 0) / (ratedMoviesWithPredictions.length || 1);

    const rmse = Math.sqrt(mse);

    return (
      <div className="rated-movies-container">
        <h1>My Ratings</h1>
        <div className="metrics">
          <p><strong>MSE:</strong> {mse.toFixed(4)}</p>
          <p><strong>RMSE:</strong> {rmse.toFixed(4)}</p>
        </div>
        <table className="rated-movies-table">
          <thead>
            <tr>
              <th>Poster</th>
              <th>Movie ID</th>
              <th>Movie Name</th>
              <th>My Rating</th>
              <th>Predicted Rating</th>
            </tr>
          </thead>
          <tbody>
            {ratedMovies.map(movie => (
              <tr key={movie.id}>
                <td>
                  <img 
                    src={`data:image/jpeg;base64,${movie.poster}`}
                    alt={movie.name}
                    className="thumbnail"
                  />
                </td>
                <td>{movie.id}</td>
                <td>{movie.name}</td>
                <td>
                  <div className="rating-row">
                    <div className="stars">
                      {[1, 2, 3, 4, 5].map((star) => (
                        <span
                          key={star}
                          className={`star ${
                            myRatings[movie.id] && Math.round(myRatings[movie.id]) >= star ? 'active' : ''
                          }`}
                        >
                          {myRatings[movie.id] && Math.round(myRatings[movie.id]) >= star ? '★' : '☆'}
                        </span>
                      ))}
                    </div>
                    <span className="rating-value">{myRatings[movie.id]?.toFixed(1) || '-'}</span>
                  </div>
                </td>
                <td>
                  <div className="rating-row predicted">
                    <div className="stars">
                      {[1, 2, 3, 4, 5].map((star) => (
                        <span
                          key={star}
                          className={`star ${
                            predictedRatings[movie.id] && Math.round(predictedRatings[movie.id]) >= star ? 'active' : ''
                          }`}
                        >
                          {predictedRatings[movie.id] && Math.round(predictedRatings[movie.id]) >= star ? '★' : '☆'}
                        </span>
                      ))}
                    </div>
                    <span className="rating-value">{predictedRatings[movie.id]?.toFixed(1) || '-'}</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  const renderRecommendations = () => {
    return (
      <div className="recommendations-container">
        <h1 style={{ paddingLeft: 15 }}>Movie Recommender</h1>
        <div className="movie-grid">
          {movies.map(movie => (
            <div className="movie-card">
              <div className="movie-poster" onClick={() => handlePosterClick(movie)}>
                <img 
                  src={`data:image/jpeg;base64,${movie.poster}`}
                  alt={movie.name}
                />
              </div>
              <div className="movie-info">
                <h3>{movie.name}</h3>
                <div className="rating">
                  <div className="rating-row">
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
                    <span className="rating-value">{movie.rating?.toFixed(1) || '0.0'}</span>
                  </div>

                  {movie.rating && movie.predicted_rating && (
                    <div className="rating-row predicted">
                      <div className="stars">
                        {[1, 2, 3, 4, 5].map((star) => (
                          <span
                            key={star}
                            className={`star ${
                              Math.round(movie.predicted_rating!) >= star ? 'active' : ''
                            }`}
                          >
                            {Math.round(movie.predicted_rating!) >= star ? '★' : '☆'}
                          </span>
                        ))}
                      </div>
                      <span className="rating-value">{movie.predicted_rating.toFixed(1)}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderContent = () => {
    if (currentPage === 'myratings') {
      return renderMyRatings();
    }
    else if (currentPage === 'recommendations') {
      return renderRecommendations();
    }

  };

  return (
    <div className="App">
      {/* Add overlay when menu is open */}
      {isMenuOpen && (
        <div className="menu-overlay" onClick={toggleMenu}></div>
      )}
      
      {/* Sliding sidebar menu */}
      <div className={`sidebar-menu ${isMenuOpen ? 'open' : ''}`}>
        <div className="menu-items">
          <button 
            onClick={() => handleMenuSelect('recommendations')}
            className={currentPage === 'recommendations' ? 'active' : ''}
          >
            Recommendations
          </button>
          <button 
            onClick={() => handleMenuSelect('myratings')}
            className={currentPage === 'myratings' ? 'active' : ''}
          >
            My Ratings
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className={`main-content ${isMenuOpen ? 'shifted' : ''}`}>
        <header className="App-header">
          <button className="menu-button" onClick={toggleMenu}>
            ☰
          </button>
        </header>
        <main>
          {renderContent()}
        </main>
      </div>
      {renderModal()}
    </div>
  );
}

export default App;
