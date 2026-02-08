"""Client for Nextcloud Talk provisioner service."""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import httpx

log = logging.getLogger(__name__)

# Provisioner configuration from environment (prefer LLM_BAWT_ prefix)
PROVISIONER_URL = os.getenv('LLM_BAWT_TALK_PROVISIONER_URL') or os.getenv('TALK_PROVISIONER_URL', 'http://localhost:8790')
PROVISIONER_TOKEN = os.getenv('LLM_BAWT_TALK_PROVISIONER_TOKEN') or os.getenv('TALK_PROVISIONER_TOKEN')


@dataclass
class ProvisionResult:
    """Result from provisioning a Nextcloud Talk room and bot."""
    nextcloud_base_url: str
    room_token: str
    room_url: str
    bot_id: int
    bot_name: str
    bot_secret: str
    webhook_url: str

    def __repr__(self):
        """Redact secret in repr."""
        return (
            f"ProvisionResult("
            f"bot_id={self.bot_id}, "
            f"bot_name='{self.bot_name}', "
            f"room_token='{self.room_token}', "
            f"nextcloud_base_url='{self.nextcloud_base_url}', "
            f"bot_secret='***REDACTED***')"
        )


class ProvisionerClient:
    """Client for the Nextcloud Talk provisioner service."""

    def __init__(
        self,
        base_url: str = PROVISIONER_URL,
        token: Optional[str] = PROVISIONER_TOKEN,
        timeout: float = 30.0,
    ):
        if not token:
            raise ValueError(
                "LLM_BAWT_TALK_PROVISIONER_TOKEN (or TALK_PROVISIONER_TOKEN) environment variable is required"
            )

        self.base_url = base_url.rstrip('/')
        self.token = token
        self.timeout = timeout

    def _get_headers(self) -> dict:
        """Get request headers with auth."""
        return {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json',
        }

    async def provision_talk_room_and_bot(
        self,
        room_name: str,
        bot_name: str,
        webhook_url: Optional[str] = None,
        owner_user_id: str = "user",
        room_description: Optional[str] = None,
        bot_description: Optional[str] = None,
    ) -> ProvisionResult:
        """
        Provision a Nextcloud Talk room with a bot.

        Args:
            room_name: Name for the Talk room
            bot_name: Name for the bot
            webhook_url: Webhook URL for the bot (optional - provisioner may have default)
            owner_user_id: Nextcloud user ID who will own the room
            room_description: Optional room description
            bot_description: Optional bot description

        Returns:
            ProvisionResult with room and bot details

        Raises:
            httpx.HTTPError: If provisioning fails
            ValueError: If response is invalid
        """
        url = f"{self.base_url}/provision/nextcloud-talk"

        payload = {
            "roomName": room_name,
            "botName": bot_name,
            "ownerUserId": owner_user_id,
        }

        if webhook_url:
            payload["webhookUrl"] = webhook_url

        if room_description:
            payload["roomDescription"] = room_description
        if bot_description:
            payload["botDescription"] = bot_description

        log.info(f"Provisioning Nextcloud Talk room: {room_name}")
        log.debug(f"Provisioner URL: {url}")
        # DO NOT log payload - it will contain the token in response

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                )
                response.raise_for_status()

                data = response.json()

                # Validate response
                required_fields = [
                    'nextcloudBaseUrl',
                    'roomToken',
                    'roomUrl',
                    'botId',
                    'botName',
                    'botSecret',
                    'webhookUrl',
                ]

                missing = [f for f in required_fields if f not in data]
                if missing:
                    raise ValueError(f"Provisioner response missing fields: {missing}")

                result = ProvisionResult(
                    nextcloud_base_url=data['nextcloudBaseUrl'],
                    room_token=data['roomToken'],
                    room_url=data['roomUrl'],
                    bot_id=data['botId'],
                    bot_name=data['botName'],
                    bot_secret=data['botSecret'],
                    webhook_url=data['webhookUrl'],
                )

                log.info(
                    f"✓ Provisioned: bot_id={result.bot_id}, "
                    f"room_token={result.room_token}"
                )

                return result

            except httpx.HTTPStatusError as e:
                log.error(
                    f"Provisioner returned {e.response.status_code}: "
                    f"{e.response.text}"
                )
                raise
            except httpx.RequestError as e:
                log.error(f"Provisioner request failed: {e}")
                raise

    async def rename_room(self, room_token: str, new_name: str) -> bool:
        """
        Rename a Nextcloud Talk room.

        Args:
            room_token: The room token
            new_name: New name for the room

        Returns:
            True if successful

        Raises:
            httpx.HTTPError: If rename fails
        """
        url = f"{self.base_url}/room/rename"

        payload = {
            "token": room_token,
            "name": new_name,
        }

        log.info(f"Renaming room {room_token} to '{new_name}'")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                )
                response.raise_for_status()

                log.info(f"✓ Renamed room {room_token}")
                return True

            except httpx.HTTPStatusError as e:
                log.error(
                    f"Room rename failed {e.response.status_code}: "
                    f"{e.response.text}"
                )
                raise
            except httpx.RequestError as e:
                log.error(f"Room rename request failed: {e}")
                raise


# Global accessor
def get_provisioner_client() -> ProvisionerClient:
    """Get provisioner client singleton."""
    return ProvisionerClient()
