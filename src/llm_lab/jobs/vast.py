import json
import re
import shlex
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, assert_never

import httpx
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from llm_lab.jobs.protocol import Active, Completed, ExplicitOffer, Halted, JobSpec, ResourceRequest
from llm_lab.keyring_settings import KeyringEntry, KeyringSettingsSource


class VastSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VAST_")

    api_key: Annotated[SecretStr, KeyringEntry(service="llm-lab", username="VAST_API_KEY")]
    api_url: str = "https://console.vast.ai/api"
    request_attempts: int = 5
    retry_delay_seconds: float = 2
    offer_limit: int = 10
    log_tail_lines: int = 200

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, KeyringSettingsSource(settings_cls)


class Offer(BaseModel):
    id: int
    dph_total: float


class OfferPage(BaseModel):
    offers: list[Offer]


class CreatedInstance(BaseModel):
    new_contract: int


class InstanceView(BaseModel):
    actual_status: str | None


class InstanceEnvelope(BaseModel):
    instances: InstanceView | None


class InstanceIdentity(BaseModel):
    id: int


class InstancePage(BaseModel):
    instances: list[InstanceIdentity] | None
    next_token: str | None = None


class RequestedLog(BaseModel):
    result_url: str


def label_job(job_name: str) -> str:
    return f"llm-lab-job-{job_name}"


def build_offer_query(request: ResourceRequest, disk_gb: float, limit: int) -> dict[str, object]:
    return {
        "gpu_name": {"eq": request.gpu_name},
        "num_gpus": {"eq": request.gpu_count},
        "disk_space": {"gte": disk_gb},
        "dph_total": {"lte": request.max_dollars_per_hour},
        "verified": {"eq": True},
        "external": {"eq": False},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "allocated_storage": disk_gb,
        "type": "ondemand",
        "order": [["dph_total", "asc"]],
        "limit": limit,
    }


def build_instance_request(spec: JobSpec, secrets: Mapping[str, SecretStr], api_url: str) -> dict[str, object]:
    script = "\n".join(
        [
            "if [ ! -e /llm-lab-job-done ]; then",
            shlex.join(spec.command),
            "exit_code=$?",
            "touch /llm-lab-job-done",
            'echo "llm-lab-job-exit-code=$exit_code"',
            "fi",
            f"sleep {spec.ttl_seconds_after_finished}",
            'curl -fsS -X DELETE -H "Authorization: Bearer $CONTAINER_API_KEY" '
            f'"{api_url}/v0/instances/$CONTAINER_ID/"',
        ]
    )
    return {
        "client_id": "me",
        "image": spec.image,
        "disk": spec.disk_gb,
        "label": label_job(spec.name),
        "runtype": "args",
        "args": ["sh", "-c", script],
        "env": {**spec.environment, **{name: secret.get_secret_value() for name, secret in secrets.items()}},
    }


def read_exit_code(log_text: str) -> int | None:
    reported = re.findall(r"llm-lab-job-exit-code=(\d+)", log_text)
    return int(reported[-1]) if reported else None


def classify(actual_status: str | None, exit_code: int | None) -> Active | Completed | Halted:
    if exit_code is not None:
        return Completed(exit_code)
    match actual_status:
        case "exited" | "offline" | "unknown":
            return Halted(actual_status)
        case _:
            return Active()


class VastInstanceProvider:
    def __init__(self, settings: VastSettings, http: httpx.Client, sleep: Callable[[float], None]) -> None:
        self.settings = settings
        self.http = http
        self.sleep = sleep

    def find_offers(self, spec: JobSpec) -> Sequence[int]:
        match spec.placement:
            case ExplicitOffer(offer_id=offer_id):
                return [offer_id]
            case ResourceRequest() as request:
                response = self.call_api(
                    "POST", "/v0/bundles/", build_offer_query(request, spec.disk_gb, self.settings.offer_limit)
                )
                response.raise_for_status()
                offers = OfferPage.model_validate_json(response.content).offers
                return [offer.id for offer in sorted(offers, key=lambda offer: offer.dph_total)]
            case _:
                assert_never(spec.placement)

    def rent_offer(self, offer_id: int, spec: JobSpec, secrets: Mapping[str, SecretStr]) -> int | None:
        response = self.call_api(
            "PUT", f"/v0/asks/{offer_id}/", build_instance_request(spec, secrets, self.settings.api_url)
        )
        if response.status_code in {404, 410}:
            return None
        response.raise_for_status()
        return CreatedInstance.model_validate_json(response.content).new_contract

    def observe(self, instance_id: int) -> Active | Completed | Halted:
        response = self.call_api("GET", f"/v0/instances/{instance_id}/", None)
        if response.status_code == 404:
            return Halted("missing")
        response.raise_for_status()
        view = InstanceEnvelope.model_validate_json(response.content).instances
        if view is None:
            return Halted("missing")
        match view.actual_status:
            case "running" | "exited" as actual_status:
                requested = self.call_api(
                    "PUT", f"/v0/instances/request_logs/{instance_id}/", {"tail": str(self.settings.log_tail_lines)}
                )
                if actual_status == "exited" and requested.is_client_error:
                    return Halted(actual_status)
                requested.raise_for_status()
                return classify(actual_status, self.download_reported_exit_code(requested))
            case _:
                return classify(view.actual_status, None)

    def find_instances(self, job_name: str) -> Sequence[int]:
        params = {
            "select_filters": json.dumps({"label": {"eq": label_job(job_name)}}),
            "order_by": json.dumps([{"col": "id", "dir": "asc"}]),
            "limit": "25",
        }
        found: list[int] = []
        while True:
            response = self.send_with_retries(
                "GET",
                f"{self.settings.api_url}/v1/instances/",
                {500, 502, 503, 504},
                headers=self.authorize(),
                params=params,
            )
            response.raise_for_status()
            page = InstancePage.model_validate_json(response.content)
            found.extend(instance.id for instance in page.instances or [])
            if page.next_token is None:
                return found
            params = params | {"after_token": page.next_token}

    def destroy(self, instance_id: int) -> None:
        response = self.call_api("DELETE", f"/v0/instances/{instance_id}/", {})
        if response.is_success or response.status_code == 404:
            return
        match self.observe(instance_id):
            case Halted(instance_state="missing"):
                return
            case _:
                response.raise_for_status()

    def download_reported_exit_code(self, requested: httpx.Response) -> int | None:
        log = self.send_with_retries(
            "GET", RequestedLog.model_validate_json(requested.content).result_url, {403, 404, 500, 502, 503, 504}
        )
        if log.status_code in {403, 404}:
            return None
        log.raise_for_status()
        return read_exit_code(log.text)

    def authorize(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.api_key.get_secret_value()}"}

    def call_api(self, method: str, path: str, body: object) -> httpx.Response:
        return self.send_with_retries(
            method, f"{self.settings.api_url}{path}", {500, 502, 503, 504}, headers=self.authorize(), json=body
        )

    def send_with_retries(
        self,
        method: str,
        url: str,
        retried_statuses: set[int],
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        json: object = None,
    ) -> httpx.Response:
        response = self.http.request(method, url, headers=headers, params=params, json=json)
        for attempt in range(1, self.settings.request_attempts):
            if response.status_code != 429 and response.status_code not in retried_statuses:
                break
            self.sleep(self.settings.retry_delay_seconds * attempt)
            response = self.http.request(method, url, headers=headers, params=params, json=json)
        return response
