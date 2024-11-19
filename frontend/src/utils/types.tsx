// Define movie interface
export interface Movie {
    id: number;
    name: string;
    poster: string;
    description: string;
    rating?: number;
    predicted_rating?: number;
  }
  
export interface RatingRequest {
    movie_id: number;
    rating: number;
}

export interface RatingResponse {
    success: boolean;
    movie_id: number;
    rating: number;
    predicted_rating: number;
    description?: string;
}