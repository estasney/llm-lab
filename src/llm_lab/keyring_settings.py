from dataclasses import dataclass

import keyring
from keyring.errors import NoKeyringError
from pydantic.fields import FieldInfo
from pydantic_settings import PydanticBaseSettingsSource


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


class KeyringSettingsSource(PydanticBaseSettingsSource):
    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[str | None, str, bool]:
        entry = read_keyring_entry(field, field_name)
        if entry is None:
            return None, field_name, False
        try:
            return keyring.get_password(entry.service, entry.username), field_name, False
        except NoKeyringError:
            return None, field_name, False

    def __call__(self) -> dict[str, object]:
        values = {name: self.get_field_value(field, name)[0] for name, field in self.settings_cls.model_fields.items()}
        return {name: value for name, value in values.items() if value is not None}
