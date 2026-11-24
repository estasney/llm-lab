from typing import Annotated

from llm_lab.datasets.designer import DataDesignerRunner, DatasetDesigner
from llm_lab.datasets.hub import DatasetHub, HuggingFaceHub
from llm_lab.datasets.serving import ModelServer, VllmServer
from llm_lab.dependencies import Depends
from llm_lab.settings import SettingsDep


def get_model_server(settings: SettingsDep) -> ModelServer:
    return VllmServer(settings)


type ModelServerDep = Annotated[ModelServer, Depends(get_model_server)]


def get_dataset_designer(settings: SettingsDep) -> DatasetDesigner:
    return DataDesignerRunner(settings)


type DatasetDesignerDep = Annotated[DatasetDesigner, Depends(get_dataset_designer)]


def get_dataset_hub(settings: SettingsDep) -> DatasetHub:
    return HuggingFaceHub(settings.hf_token)


type DatasetHubDep = Annotated[DatasetHub, Depends(get_dataset_hub)]
