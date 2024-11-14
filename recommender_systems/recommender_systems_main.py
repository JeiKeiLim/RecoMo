"""Main script for your project.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import os

import hydra
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import DictConfig

from api.fastapi_app import FastAPIApp
from models.model_utils import model_loader


def create_app() -> FastAPI:
    """Create FastAPI app."""
    with hydra.initialize(config_path=os.path.join("..", "res", "configs")):
        config = hydra.compose(config_name="base_config")

    device = torch.device(
        config.recommender_systems.device if torch.cuda.is_available() else "cpu"
    )

    model_name = config.recommender_systems.model
    model_config = config.models[model_name]

    model = model_loader(model_name, model_config, device)

    fastapi = FastAPI()
    fastapi.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    fastapi_app = FastAPIApp(rating_path=config.data.my_rating_path, model=model)
    fastapi.include_router(fastapi_app.router)
    return fastapi


app = create_app()


@hydra.main(
    version_base=None,
    config_path=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "res", "configs")
    ),
    config_name="base_config",
)
def main_app(config: DictConfig) -> None:
    """Run recommender systems API."""
    uvicorn.run(
        "recommender_systems_main:app",
        host=config.recommender_systems.host,
        port=config.recommender_systems.port,
        reload=True,
    )


if __name__ == "__main__":
    main_app()  # pylint: disable=no-value-for-parameter
