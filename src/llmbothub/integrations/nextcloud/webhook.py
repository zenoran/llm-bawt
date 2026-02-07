"""Nextcloud Talk webhook handler."""

import hmac
import hashlib
import logging
import secrets
import json
from datetime import datetime
from typing import Optional

import httpx
from fastapi import Request, HTTPException

from .manager import get_nextcloud_manager

log = logging.getLogger(__name__)


def verify_nextcloud_signature(
    signature: str,
    random: str,
    body: bytes,
    secret: str
) -> bool:
    """Verify HMAC-SHA256 signature from Nextcloud Talk."""
    expected = hmac.new(
        secret.encode('utf-8'),
        random.encode('utf-8') + body,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature.lower())


async def send_nextcloud_message(
    nextcloud_url: str,
    conversation_token: str,
    message: str,
    secret: str,
    reply_to: Optional[int] = None,
) -> bool:
    """
    Send a message to Nextcloud Talk.

    Args:
        nextcloud_url: Nextcloud instance URL
        conversation_token: The conversation/room token
        message: The message text to send
        secret: Bot secret for signing
        reply_to: Optional message ID to reply to

    Returns:
        True if successful, False otherwise
    """
    url = f"{nextcloud_url}/ocs/v2.php/apps/spreed/api/v1/bot/{conversation_token}/message"

    # Build payload
    payload = {"message": message}
    if reply_to:
        payload["replyTo"] = reply_to

    body = json.dumps(payload).encode('utf-8')

    # Generate random nonce
    random_string = secrets.token_hex(32)

    # Calculate signature: HMAC-SHA256(random + message_text, secret)
    # NOTE: Sign the MESSAGE TEXT, not the full JSON body!
    signature = hmac.new(
        secret.encode('utf-8'),
        (random_string + message).encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    # Set headers
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'OCS-APIRequest': 'true',
        'X-Nextcloud-Talk-Bot-Random': random_string,
        'X-Nextcloud-Talk-Bot-Signature': signature,
    }

    # Log outgoing message (truncated preview)
    preview = message[:100] + "..." if len(message) > 100 else message
    log.info(f"📤 Sending ({len(message)} chars): {preview!r}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, content=body)

            if response.status_code == 201:
                log.info(f"✓ Message sent to {conversation_token} ({len(message)} chars)")
                return True
            else:
                log.error(f"Failed to send message: {response.status_code} {response.text}")
                log.error(f"Message was ({len(message)} chars): {message[:200]}...")
                return False
    except Exception as e:
        log.error(f"Error sending message: {e}")
        return False


async def handle_nextcloud_webhook(request: Request) -> dict:
    """
    Handle incoming Nextcloud Talk webhooks.
    Routes to appropriate bot based on conversation token.
    """
    # Get signature headers
    signature = request.headers.get('X-Nextcloud-Talk-Signature', '')
    random = request.headers.get('X-Nextcloud-Talk-Random', '')
    backend = request.headers.get('X-Nextcloud-Talk-Backend', '')

    # Get raw body
    body = await request.body()

    # Parse payload
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    conversation_token = payload.get('target', {}).get('id', '')

    # Log the webhook
    log.info("=" * 80)
    log.info(f"Nextcloud Talk webhook received at {datetime.now()}")
    log.info(f"Backend: {backend}")
    log.info(f"Conversation: {conversation_token}")

    # Look up bot configuration
    manager = get_nextcloud_manager()
    bot_config = manager.get_bot_by_conversation(conversation_token)

    if not bot_config:
        log.warning(f"No bot configured for conversation {conversation_token}")
        log.info("=" * 80)
        return {"status": "no_bot_configured", "conversation": conversation_token}

    log.info(f"Routing to bot: {bot_config.llmbothub_bot}")

    # Verify signature with this bot's secret
    if not verify_nextcloud_signature(signature, random, body, bot_config.secret):
        log.error(f"Invalid signature for bot {bot_config.llmbothub_bot}")
        log.info("=" * 80)
        raise HTTPException(status_code=401, detail="Invalid signature")

    log.info("✓ Signature verified")

    # Extract message info
    msg_type = payload.get('type', 'unknown')
    actor_name = payload.get('actor', {}).get('name', 'unknown')
    user_id = payload.get('actor', {}).get('id', 'users/unknown')

    # Clean up user_id (remove "users/" prefix)
    if user_id.startswith('users/'):
        user_id = user_id[6:]

    # Parse the actual message text from the content field
    content = payload.get('object', {}).get('content', '')
    try:
        content_json = json.loads(content) if content else {}
        message = content_json.get('message', '')
    except json.JSONDecodeError:
        message = payload.get('object', {}).get('name', '')

    short_msg = message[:80] + "..." if len(message) > 80 else message
    log.info(f"📨 {actor_name}: \"{short_msg}\"")

    # Process message with appropriate bot
    if message and msg_type == "Create":
        try:
            # Import here to avoid circular imports
            from ...service.api import ChatCompletionRequest, ChatMessage, get_service

            service = get_service()

            # Build chat completion request with the specific bot
            request_obj = ChatCompletionRequest(
                messages=[ChatMessage(role="user", content=message)],
                user=user_id,
                bot_id=bot_config.llmbothub_bot,  # Use the bot from config!
                stream=False,
            )

            # Generate response using the service
            response = await service.chat_completion(request_obj)

            # Extract the assistant's message
            if response.choices and len(response.choices) > 0:
                llm_response = response.choices[0].message.content
                resp_preview = llm_response[:100] + "..." if llm_response and len(llm_response) > 100 else llm_response
                log.info(f"🤖 Response ({len(llm_response) if llm_response else 0} chars): {resp_preview!r}")
                # Send response back to Talk with this bot's secret
                await send_nextcloud_message(
                    nextcloud_url=backend.rstrip('/'),
                    conversation_token=conversation_token,
                    message=llm_response,
                    secret=bot_config.secret,
                )
            else:
                log.error("No response choices returned from LLM")

        except Exception as e:
            log.error(f"Error processing message with bot {bot_config.llmbothub_bot}: {e}", exc_info=True)
            # Send error message back
            await send_nextcloud_message(
                nextcloud_url=backend.rstrip('/'),
                conversation_token=conversation_token,
                message="Sorry, I encountered an error processing your message.",
                secret=bot_config.secret,
            )

    return {
        "status": "received",
        "bot": bot_config.llmbothub_bot,
        "message": message,
        "from": actor_name
    }
