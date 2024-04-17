from pydantic import BaseModel
import json
from typing import Any

def load(model: type[BaseModel], path: str) -> Any:
    def load_fn() -> model:
        return model.parse_file(path)

    return load_fn

def save(model: type[BaseModel], path: str):
    def save_fn(instance: model) -> None:
        with open(path, "w") as f:
            return json.dump(instance.dict(), f)

    return save_fn
