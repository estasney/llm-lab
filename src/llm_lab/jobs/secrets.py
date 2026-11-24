from pydantic import SecretStr

from llm_lab.keyring_settings import KeyringEntry, read_keyring_entry
from llm_lab.settings import Settings


def find_secret_entry(name: str) -> KeyringEntry:
    field_name = name.lower()
    field = Settings.model_fields.get(field_name)
    entry = None if field is None else read_keyring_entry(field, field_name)
    if entry is None:
        raise LookupError(
            f"Settings has no field {field_name!r} with a KeyringEntry annotation for the secret {name!r}"
        )
    return entry


class SettingsSecretSource:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def read(self, name: str) -> SecretStr:
        entry = find_secret_entry(name)
        match getattr(self.settings, name.lower()):
            case SecretStr() as secret:
                return secret
            case None:
                raise LookupError(
                    f"the secret {name!r} is not set, so set it with `llm-lab secrets set {entry.username}`"
                )
            case value:
                raise TypeError(f"the settings field for the secret {name!r} has the type {type(value).__name__}")
