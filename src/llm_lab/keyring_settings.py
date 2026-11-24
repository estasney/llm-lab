from dataclasses import dataclass

import keyring
from keyring.errors import NoKeyringError
from pydantic.fields import FieldInfo
from pydantic_settings import PydanticBaseSettingsSource


@dataclass(frozen=True)
class KeyringEntry:
    service: str
    username: str


class KeyringSettingsSource(PydanticBaseSettingsSource):
    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[str | None, str, bool]:
        match [item for item in field.metadata if isinstance(item, KeyringEntry)]:
            case []:
                return None, field_name, False
            case [KeyringEntry(service=service, username=username)]:
                try:
                    return keyring.get_password(service, username), field_name, False
                except NoKeyringError:
                    return None, field_name, False
            case _:
                raise TypeError(f"field {field_name!r} has more than one KeyringEntry annotation")

    def __call__(self) -> dict[str, object]:
        values = {name: self.get_field_value(field, name)[0] for name, field in self.settings_cls.model_fields.items()}
        return {name: value for name, value in values.items() if value is not None}
