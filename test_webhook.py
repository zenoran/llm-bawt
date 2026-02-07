#!/usr/bin/env python3
"""
Simple webhook receiver for testing Nextcloud Talk bot payloads.
Run this to see what Nextcloud sends.
"""
import json
import hmac
import hashlib
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# Set this to the secret from the occ command above
SHARED_SECRET = "your-secret-here"  # TODO: Replace with actual secret from occ command

@app.route('/webhook/nextcloud', methods=['POST'])
def nextcloud_webhook():
    """Receive and log Nextcloud Talk webhooks."""

    # Log headers
    print("\n" + "="*80)
    print(f"[{datetime.now()}] Incoming webhook")
    print("="*80)

    print("\nHeaders:")
    for key, value in request.headers.items():
        print(f"  {key}: {value}")

    # Get signature headers
    signature = request.headers.get('X-Nextcloud-Talk-Signature', '')
    random = request.headers.get('X-Nextcloud-Talk-Random', '')
    backend = request.headers.get('X-Nextcloud-Talk-Backend', '')

    # Get body
    body = request.get_data()

    print("\nBody (raw):")
    print(f"  {body.decode('utf-8')}")

    # Verify signature
    if SHARED_SECRET and SHARED_SECRET != "your-secret-here":
        expected_digest = hmac.new(
            SHARED_SECRET.encode('utf-8'),
            random.encode('utf-8') + body,
            hashlib.sha256
        ).hexdigest()

        signature_valid = hmac.compare_digest(expected_digest, signature.lower())
        print(f"\nSignature validation: {'✓ VALID' if signature_valid else '✗ INVALID'}")
        print(f"  Expected: {expected_digest}")
        print(f"  Received: {signature}")

        if not signature_valid:
            return jsonify({"error": "Invalid signature"}), 401
    else:
        print("\nSignature validation: SKIPPED (no secret configured)")

    # Parse JSON
    try:
        data = json.loads(body)
        print("\nParsed JSON:")
        print(json.dumps(data, indent=2))

        # Extract key info
        msg_type = data.get('type', 'unknown')
        actor_name = data.get('actor', {}).get('name', 'unknown')
        message = data.get('object', {}).get('name', '')

        print(f"\nExtracted:")
        print(f"  Type: {msg_type}")
        print(f"  From: {actor_name}")
        print(f"  Message: {message}")

    except json.JSONDecodeError as e:
        print(f"\nJSON parse error: {e}")

    print("="*80 + "\n")

    # Return 200 OK
    return jsonify({"status": "received"}), 200


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "nextcloud-webhook-test"}), 200


if __name__ == '__main__':
    print("Starting Nextcloud Talk webhook test server...")
    print(f"Remember to update SHARED_SECRET in this file!")
    print(f"Webhook endpoint: http://localhost:5000/webhook/nextcloud")
    print(f"Health check: http://localhost:5000/health")
    print("")
    app.run(host='0.0.0.0', port=5000, debug=True)
