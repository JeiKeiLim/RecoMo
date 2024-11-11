"""Main script for your project.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import argparse

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.fastapi_app import FastAPIApp
from model.movie_db import MovieDB


def get_parser() -> argparse.Namespace:
    """Get argument parser.

    Modify this function as your porject needs
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    return parser.parse_args()


def create_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    movie_db = MovieDB("/home/limjk/Datasets/MovieLens20M/movie_posters.sqlite")
    fastapi_app = FastAPIApp(movie_db, rating_path="../res/ratings.json")
    app.include_router(fastapi_app.router)
    return app


app = create_app()


def main_app():
    uvicorn.run("backend_main:app", host="0.0.0.0", port=7777, reload=True)


if __name__ == "__main__":
    args = get_parser()
    main_app()
