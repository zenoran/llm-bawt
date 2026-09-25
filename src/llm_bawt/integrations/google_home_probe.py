"""Capture-only Google cloud-to-cloud experiment; never dispatches bot work."""
from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr

DEVICE_ID = "loopy-text-probe"
Text = Annotated[StrictStr, Field(min_length=1, max_length=2048)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Device(StrictModel):
    id: Text
    customData: dict | None = None


class Execution(StrictModel):
    command: Text
    params: dict[str, Text] = Field(default_factory=dict, max_length=8)


class Command(StrictModel):
    devices: list[Device] = Field(min_length=1, max_length=5)
    execution: list[Execution] = Field(min_length=1, max_length=5)


class Payload(StrictModel):
    devices: list[Device] | None = Field(default=None, max_length=5)
    commands: list[Command] | None = Field(default=None, max_length=5)


class Intent(StrictModel):
    intent: Literal["action.devices.SYNC", "action.devices.QUERY", "action.devices.EXECUTE", "action.devices.DISCONNECT"]
    payload: Payload = Field(default_factory=Payload)
    # Live Google requests include intent metadata. It is not command input
    # and must not be retained in captures. The HTTP body remains size-bounded.
    context: dict | None = Field(default=None, exclude=True)


class ProbeRequest(StrictModel):
    requestId: Annotated[StrictStr, Field(min_length=1, max_length=256)]
    inputs: list[Intent] = Field(min_length=1, max_length=1)


class GoogleHomeProbe:
    def __init__(self, captures):
        self.captures = captures

    def handle(self, subject: str, request: ProbeRequest) -> dict:
        intent = request.inputs[0]
        # Save only the recognized protocol fields, not headers/customData.
        # No raw token or general request-body logging. The store encrypts this.
        event = request.model_dump(exclude_none=True)
        payload = event["inputs"][0]["payload"]
        for device in payload.get("devices", []):
            device.pop("customData", None)
        for group in payload.get("commands", []):
            for device in group["devices"]:
                device.pop("customData", None)
        key = hashlib.sha256(f"{subject}\0{request.requestId}".encode()).hexdigest()
        self.captures.record(key, json.dumps(event, ensure_ascii=False, sort_keys=True))
        result = {"requestId": request.requestId}
        if intent.intent == "action.devices.SYNC":
            result["payload"] = {
                "agentUserId": subject,
                "devices": [{
                    "id": DEVICE_ID,
                    "type": "action.devices.types.TV",
                    "traits": ["action.devices.traits.AppSelector"],
                    "name": {"name": "BawtHub TV", "nicknames": ["BawtHub", "Bot Hub TV"]},
                    "willReportState": False,
                    "attributes": {"availableApplications": [{
                        "key": "youtube", "names": [{
                            "name_synonym": ["YouTube", "You Tube"], "lang": "en",
                        }],
                    }]},
                    "deviceInfo": {"manufacturer": "BawtHub", "model": "Capture-only probe"},
                }],
            }
        elif intent.intent == "action.devices.QUERY":
            result["payload"] = {"devices": {
                device.id: ({"online": True, "status": "SUCCESS", "currentApplication": "youtube"}
                            if device.id == DEVICE_ID else {"online": False, "errorCode": "deviceNotFound"})
                for device in intent.payload.devices or []
            }}
        elif intent.intent == "action.devices.EXECUTE":
            # Intentionally report NOT executed, even when successfully captured.
            # Google may speak an error; that is not evidence the capture failed.
            result["payload"] = {"commands": [
                {"ids": [device.id], "status": "ERROR", "errorCode": (
                    "functionNotSupported" if device.id == DEVICE_ID else "deviceNotFound"
                )}
                for group in intent.payload.commands or [] for device in group.devices
            ]}
        return result
