import base64
import json
import re
import shlex
from collections.abc import Callable, Mapping, Sequence
from typing import assert_never

import httpx
from pydantic import BaseModel, Field, SecretStr

from llm_lab.errors import CommandError
from llm_lab.jobs.access import InstanceAccess, SshEndpoint
from llm_lab.jobs.protocol import (
    Active,
    Completed,
    ExcludedCountries,
    ExplicitOffer,
    Halted,
    IncludedCountries,
    JobSpec,
    ResourceRequest,
)
from llm_lab.keyring_settings import SecretName
from llm_lab.settings import Settings


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


class PortBinding(BaseModel):
    host_port: int = Field(alias="HostPort")


class InstanceAddress(BaseModel):
    public_ipaddr: str
    ports: dict[str, list[PortBinding]]


class AddressEnvelope(BaseModel):
    instances: InstanceAddress


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
    ceilings = {
        "dph_base": request.max_compute_dollars_per_hour,
        "storage_cost": request.max_storage_dollars_per_gb_month,
        "inet_down_cost": request.max_download_dollars_per_gb,
        "inet_up_cost": request.max_upload_dollars_per_gb,
    }
    match request.countries:
        case IncludedCountries(include=codes):
            location = {"geolocation": {"in": list(codes)}}
        case ExcludedCountries(exclude=codes):
            location = {"geolocation": {"notin": list(codes)}}
        case None:
            location = {}
        case _:
            assert_never(request.countries)
    return {
        **{field: {"lte": ceiling} for field, ceiling in ceilings.items() if ceiling is not None},
        **location,
        **({"datacenter": {"eq": True}} if request.datacenter_only else {}),
        "cuda_max_good": {"gte": request.min_cuda},
        "gpu_name": {"eq": request.gpu_name},
        "num_gpus": {"eq": request.gpu_count},
        "disk_space": {"gte": disk_gb},
        "dph_total": {"lte": request.max_dollars_per_hour},
        "verified": {"eq": True},
        "external": {"eq": False},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "direct_port_count": {"gte": 1},
        "allocated_storage": disk_gb,
        "type": "ondemand",
        "order": [["dph_total", "asc"]],
        "limit": limit,
    }


def build_instance_request(
    spec: JobSpec, secrets: Mapping[SecretName, SecretStr], access: InstanceAccess, api_url: str, ssh_port: int
) -> dict[str, object]:
    # Vast.ai overwrites /root/.ssh/authorized_keys after the container starts.
    script = "\n".join(
        [
            "if command -v sshd >/dev/null 2>&1; then",
            "mkdir -p -m 700 /run/sshd /root/.ssh",
            "mkdir -p /etc/profile.d",
            "printf '%s\\n' \"$LLM_LAB_SSH_AUTHORIZED_KEY\" > /run/sshd/llm-lab-authorized-keys",
            "(umask 077 && printf '%s' \"$LLM_LAB_SSH_HOST_KEY\" | base64 -d > /run/sshd/llm-lab-host-key)",
            'printf \'export PATH="%s"\\n\' "$PATH" > /etc/profile.d/llm-lab-path.sh',
            f'"$(command -v sshd)" -p {ssh_port} -h /run/sshd/llm-lab-host-key '
            "-o 'AuthorizedKeysFile=/run/sshd/llm-lab-authorized-keys .ssh/authorized_keys' "
            "-o PasswordAuthentication=no -o KbdInteractiveAuthentication=no -o PermitRootLogin=prohibit-password",
            "fi",
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
        "env": {
            **spec.environment,
            **{name: secret.get_secret_value() for name, secret in secrets.items()},
            "LLM_LAB_SSH_AUTHORIZED_KEY": access.authorized_key,
            "LLM_LAB_SSH_HOST_KEY": base64.b64encode(access.host_private_key.get_secret_value().encode()).decode(),
            f"-p {ssh_port}:{ssh_port}": "1",
        },
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
    def __init__(self, settings: Settings, http: httpx.Client, sleep: Callable[[float], None]) -> None:
        if settings.vast_api_key is None:
            raise CommandError("The Vast.ai API key is not set. Run `llm-lab secrets set VAST_API_KEY`.")
        self.settings = settings
        self.api_key = settings.vast_api_key
        self.http = http
        self.sleep = sleep

    def find_offers(self, spec: JobSpec) -> Sequence[int]:
        match spec.placement:
            case ExplicitOffer(offer_id=offer_id):
                return [offer_id]
            case ResourceRequest() as request:
                response = self.call_api(
                    "POST", "/v0/bundles/", build_offer_query(request, spec.disk_gb, self.settings.vast_offer_limit)
                )
                self.require_success(response)
                offers = OfferPage.model_validate_json(response.content).offers
                return [offer.id for offer in sorted(offers, key=lambda offer: offer.dph_total)]
            case _:
                assert_never(spec.placement)

    def rent_offer(
        self, offer_id: int, spec: JobSpec, secrets: Mapping[SecretName, SecretStr], access: InstanceAccess
    ) -> int | None:
        response = self.call_api(
            "PUT",
            f"/v0/asks/{offer_id}/",
            build_instance_request(spec, secrets, access, self.settings.vast_api_url, self.settings.vast_ssh_port),
        )
        if response.status_code in {404, 410}:
            return None
        self.require_success(response)
        return CreatedInstance.model_validate_json(response.content).new_contract

    def observe(self, instance_id: int) -> Active | Completed | Halted:
        response = self.call_api("GET", f"/v0/instances/{instance_id}/", None)
        if response.status_code == 404:
            return Halted("missing")
        self.require_success(response)
        view = InstanceEnvelope.model_validate_json(response.content).instances
        if view is None:
            return Halted("missing")
        match view.actual_status:
            case "running" | "exited" as actual_status:
                requested = self.call_api(
                    "PUT",
                    f"/v0/instances/request_logs/{instance_id}/",
                    {"tail": str(self.settings.vast_log_tail_lines)},
                )
                if actual_status == "exited" and requested.is_client_error:
                    return Halted(actual_status)
                self.require_success(requested)
                return classify(actual_status, self.download_reported_exit_code(requested))
            case _:
                return classify(view.actual_status, None)

    def find_ssh_endpoint(self, instance_id: int) -> SshEndpoint:
        response = self.call_api("GET", f"/v0/instances/{instance_id}/", None)
        self.require_success(response)
        address = AddressEnvelope.model_validate_json(response.content).instances
        match address.ports.get(f"{self.settings.vast_ssh_port}/tcp", []):
            case [binding, *_]:
                return SshEndpoint(host=address.public_ipaddr, port=binding.host_port)
            case _:
                raise CommandError(
                    f"Instance {instance_id} does not map the SSH port {self.settings.vast_ssh_port} to a public port."
                )

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
                f"{self.settings.vast_api_url}/v1/instances/",
                {500, 502, 503, 504},
                headers=self.authorize(),
                params=params,
            )
            self.require_success(response)
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
                self.require_success(response)

    def download_reported_exit_code(self, requested: httpx.Response) -> int | None:
        log = self.send_with_retries(
            "GET", RequestedLog.model_validate_json(requested.content).result_url, {403, 404, 500, 502, 503, 504}
        )
        if log.status_code in {403, 404}:
            return None
        self.require_success(log)
        return read_exit_code(log.text)

    def authorize(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}

    def call_api(self, method: str, path: str, body: object) -> httpx.Response:
        return self.send_with_retries(
            method, f"{self.settings.vast_api_url}{path}", {500, 502, 503, 504}, headers=self.authorize(), json=body
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
        response = self.send(method, url, headers, params, json)
        for attempt in range(1, self.settings.vast_request_attempts):
            if response.status_code != 429 and response.status_code not in retried_statuses:
                break
            self.sleep(self.settings.vast_retry_delay_seconds * attempt)
            response = self.send(method, url, headers, params, json)
        return response

    def send(
        self, method: str, url: str, headers: dict[str, str] | None, params: dict[str, str] | None, json: object
    ) -> httpx.Response:
        try:
            return self.http.request(method, url, headers=headers, params=params, json=json)
        except httpx.TransportError as error:
            raise CommandError(f"The request to {httpx.URL(url).host} did not complete: {error}") from error

    def require_success(self, response: httpx.Response) -> None:
        if not response.is_success:
            request = response.request
            raise CommandError(
                f"{request.url.host} returned status {response.status_code} for {request.method} {request.url.path}."
            )
