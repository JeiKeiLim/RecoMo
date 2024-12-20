"""Backend API main module.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import os
import tempfile

import hydra
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import DictConfig, OmegaConf

from api.fastapi_app import FastAPIApp
from model.movie_db import MovieDB


def save_hydra_config_to_tempfile(config: DictConfig) -> None:
    """Save hydra config to temp file."""
    temp_file_path = os.path.join(tempfile.gettempdir(), "recomo_config.yaml")
    OmegaConf.save(config, temp_file_path)


def load_temp_hydra_config() -> DictConfig:
    """Load hydra config from temp file."""
    temp_file_path = os.path.join(tempfile.gettempdir(), "recomo_config.yaml")
    if os.path.exists(temp_file_path):
        return OmegaConf.load(temp_file_path)  # type: ignore

    return OmegaConf.load(
        os.path.join("..", "res", "configs", "base_config.yaml")
    )  # type: ignore


def create_app() -> FastAPI:
    """Create FastAPI app."""
    config = load_temp_hydra_config()

    fastapi = FastAPI()
    fastapi.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    movie_db = MovieDB(config.dataset.poster_db_path)
    fastapi_application = FastAPIApp(movie_db, config.data.my_rating_path)
    fastapi.include_router(fastapi_application.router)
    return fastapi


app = create_app()


@hydra.main(
    version_base=None,
    config_path=os.path.join("..", "res", "configs"),
    config_name="base_config",
)
def main_app(config: DictConfig) -> None:
    """Run FastAPI backend."""
    save_hydra_config_to_tempfile(config)

    uvicorn.run(
        "backend_main:app",
        host=config.backend.host,
        port=config.backend.port,
        reload=True,
    )


if __name__ == "__main__":
    main_app()  # pylint: disable=no-value-for-parameter
