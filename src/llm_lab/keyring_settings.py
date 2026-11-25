from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import keyring
from keyring.errors import NoKeyringError
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource


class SecretName(StrEnum):
    hf_token = "HF_TOKEN"
    vast_api_key = "VAST_API_KEY"


@dataclass(frozen=True)
class KeyringEntry:
    service: str
    username: str


def read_keyring_entry(field: FieldInfo, field_name: str) -> KeyringEntry | None:
    match [item for item in field.metadata if isinstance(item, KeyringEntry)]:
        case []:
            return None
        case [KeyringEntry() as entry]:
            return entry
        case _:
            raise TypeError(f"field {field_name!r} has more than one KeyringEntry annotation")


def read_keyring_password(entry: KeyringEntry) -> str | None:
    try:
        return keyring.get_password(entry.service, entry.username)
    except NoKeyringError:
        return None


class SecretSettingsSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings], read: Callable[[KeyringEntry], str | None]) -> None:
        super().__init__(settings_cls)
        self.read = read

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[str | None, str, bool]:
        entry = read_keyring_entry(field, field_name)
        return (None if entry is None else self.read(entry)), field_name, False

    def __call__(self) -> dict[str, object]:
        values = {name: self.get_field_value(field, name)[0] for name, field in self.settings_cls.model_fields.items()}
        return {name: value for name, value in values.items() if value is not None}
