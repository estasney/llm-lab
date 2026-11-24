from typing import Annotated, Literal

from pydantic import BaseModel, JsonValue, PositiveInt, StringConstraints

type DatasetName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$", max_length=80)]


class RecipeFile(BaseModel):
    recipe: dict[str, JsonValue]


class RunRequest(BaseModel):
    rows: PositiveInt
    run_name: DatasetName
    execution_type: Literal["full"] = "full"


class JobRequest(BaseModel):
    recipe: dict[str, JsonValue]
    run: RunRequest
