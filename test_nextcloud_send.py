#!/usr/bin/env python3
"""
Test script for sending messages to Nextcloud Talk as a bot.
"""
import os
import requests
import hmac
import hashlib
import secrets
import json
import sys

# Configuration (override with env vars for local use)
NEXTCLOUD_URL = os.getenv("LLM_BAWT_NEXTCLOUD_URL", "https://nextcloud.example.com")
BOT_SECRET = os.getenv("LLM_BAWT_NEXTCLOUD_BOT_SECRET", "your-secret-here")
CONVERSATION_TOKEN = os.getenv("LLM_BAWT_NEXTCLOUD_ROOM_TOKEN", "your-room-token")

def send_message(message: str, reference_id: str = None):
    """Send a message to Nextcloud Talk as a bot."""

    # Bot message endpoint
    url = f"{NEXTCLOUD_URL}/ocs/v2.php/apps/spreed/api/v1/bot/{CONVERSATION_TOKEN}/message"

    # Create payload
    payload = {"message": message}
    if reference_id:
        payload["replyTo"] = reference_id

    body = json.dumps(payload).encode('utf-8')

    # Generate random nonce
    random_string = secrets.token_hex(32)  # 64 character hex string

    # Calculate signature: HMAC-SHA256(random + message_text, secret)
    # NOTE: Sign the MESSAGE TEXT, not the full JSON body!
    signature_input = (random_string + message).encode('utf-8')
    signature = hmac.new(
        BOT_SECRET.encode('utf-8'),
        signature_input,
        hashlib.sha256
    ).hexdigest()

    print(f"Debug - Signature calculation:")
    print(f"  Random: {random_string}")
    print(f"  Message: {message}")
    print(f"  Signature input: {random_string + message}")
    print(f"  Signature: {signature}")

    # Set headers
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'OCS-APIRequest': 'true',
        'X-Nextcloud-Talk-Bot-Random': random_string,
        'X-Nextcloud-Talk-Bot-Signature': signature,
    }

    # Send request
    print(f"Sending to: {url}")
    print(f"Headers: {json.dumps(headers, indent=2)}")
    print(f"Body: {body.decode('utf-8')}")

    response = requests.post(url, headers=headers, data=body)

    print(f"\nResponse status: {response.status_code}")
    print(f"Response body: {response.text}")

    return response


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_nextcloud_send.py <message>")
        print(f"\nCurrent config:")
        print(f"  NEXTCLOUD_URL: {NEXTCLOUD_URL}")
        print(f"  BOT_SECRET: {'*' * len(BOT_SECRET) if BOT_SECRET != 'your-secret-here' else 'NOT SET'}")
        print(f"  CONVERSATION_TOKEN: {CONVERSATION_TOKEN}")
        print("\nSet LLM_BAWT_NEXTCLOUD_URL, LLM_BAWT_NEXTCLOUD_BOT_SECRET, and LLM_BAWT_NEXTCLOUD_ROOM_TOKEN before testing.")
        sys.exit(1)

    message = ' '.join(sys.argv[1:])
    send_message(message)
