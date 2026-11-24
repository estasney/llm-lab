from keyring.backend import KeyringBackend
from pydantic import SecretStr


class KeyringSecretSource:
    def __init__(self, backend: KeyringBackend, service_name: str) -> None:
        self.backend = backend
        self.service_name = service_name

    def read(self, name: str) -> SecretStr:
        value = self.backend.get_password(self.service_name, name)
        if value is None:
            raise LookupError(f"the keyring service {self.service_name!r} has no secret with the name {name!r}")
        return SecretStr(value)

    def write(self, name: str, value: SecretStr) -> None:
        self.backend.set_password(self.service_name, name, value.get_secret_value())
