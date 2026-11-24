from pydantic import SecretStr

from llm_lab.keyring_settings import KeyringEntry, SecretName, read_keyring_entry
from llm_lab.settings import Settings


def find_secret_entry(name: SecretName) -> KeyringEntry:
    entry = read_keyring_entry(Settings.model_fields[name.name], name.name)
    if entry is None:
        raise TypeError(f"the Settings field {name.name!r} has no KeyringEntry annotation")
    return entry


class SettingsSecretSource:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def read(self, name: SecretName) -> SecretStr:
        match getattr(self.settings, name.name):
            case SecretStr() as secret:
                return secret
            case None:
                raise LookupError(f"the secret {name} is not set, so set it with `llm-lab secrets set {name}`")
            case value:
                raise TypeError(f"the Settings field {name.name!r} has the type {type(value).__name__}")
