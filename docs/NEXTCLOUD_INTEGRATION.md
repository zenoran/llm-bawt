# Nextcloud Talk Integration

## Overview

Multiple llm-bawt bot personalities can have dedicated Nextcloud Talk rooms. A single webhook endpoint routes messages to the appropriate bot based on conversation token. Room and bot provisioning is handled by an external provisioner service.

## Architecture

```
Nextcloud Talk Room (token: abc123)
    ↓ webhook POST
http://localhost:8642/webhook/nextcloud
    ↓ routes by conversation_token
llm-bawt bot: nova (uses Nova personality)

Nextcloud Talk Room (token: def456)
    ↓ webhook POST
http://localhost:8642/webhook/nextcloud
    ↓ routes by conversation_token
llm-bawt bot: monika (uses Monika personality)
```

Each bot has its own secret, conversation context, and personality. A single webhook endpoint handles all bots, routing by the `conversation_token` in the payload.

## Configuration

### Bot Config (`~/.config/llm-bawt/bots.yaml`)

Each bot can have a `nextcloud:` section:

```yaml
bots:
  nova:
    name: Nova
    description: Full-featured assistant
    system_prompt: |
      You are Nova...
    nextcloud:
      bot_id: 5                        # Nextcloud bot ID
      secret: "abc123..."              # Bot secret for HMAC signing
      conversation_token: "abc123xyz"  # Room token
      enabled: true
```

Nextcloud config is stored only in the user config file, not the repo `bots.yaml`.

### Environment Variables

Add to `~/.config/llm-bawt/.env`:

```bash
# Provisioner service
LLM_BAWT_TALK_PROVISIONER_URL=http://localhost:8790
LLM_BAWT_TALK_PROVISIONER_TOKEN=your-token-here

# Nextcloud server URL
LLM_BAWT_NEXTCLOUD_URL=https://nextcloud.example.com
```

## CLI Commands

```bash
# List configured bots and their Nextcloud status
llm-nextcloud list

# Provision a new bot with a dedicated Talk room
llm-nextcloud provision --bot nova
llm-nextcloud provision --bot monika --room-name "Monika's Room"

# Rename a room
llm-nextcloud rename --bot nova --name "Nova Assistant"

# Remove Nextcloud config for a bot
llm-nextcloud remove --bot nova

# Reload config (notifies running service)
llm-nextcloud reload
```

## API Endpoints

### POST /webhook/nextcloud

Receives webhooks from Nextcloud Talk and routes to the appropriate bot.

**Request headers:**
- `X-Nextcloud-Talk-Signature` - HMAC-SHA256 signature
- `X-Nextcloud-Talk-Random` - Random nonce
- `X-Nextcloud-Talk-Backend` - Nextcloud server URL

**Request body** (Activity Streams 2.0):
```json
{
  "type": "Create",
  "actor": {"id": "users/alice", "name": "Alice"},
  "object": {
    "content": "{\"message\": \"Hello bot\"}"
  },
  "target": {
    "id": "abc123xyz",
    "name": "Nova"
  }
}
```

**Processing:**
1. Extracts `conversation_token` from `target.id`
2. Looks up bot config by conversation token
3. Verifies HMAC-SHA256 signature using the bot's secret
4. Processes message through the matching bot personality
5. Sends response back to Nextcloud Talk

### POST /admin/nextcloud-talk/provision

Provisions a new room and bot via the provisioner service.

**Request:**
```json
{
  "bot_id": "nova",
  "room_name": "Nova",
  "bot_name": "Nova",
  "owner_user_id": "user"
}
```

### POST /admin/nextcloud-talk/reload

Forces config reload from disk. Called automatically by CLI commands when config changes.

## Message Signing

### Inbound (Nextcloud -> llm-bawt)

Signature verification: `HMAC-SHA256(random + body, secret)`

### Outbound (llm-bawt -> Nextcloud)

Signature is computed over `random + messageText` (NOT the full JSON body):

```python
signature = hmac.new(
    secret.encode('utf-8'),
    (random_string + message_text).encode('utf-8'),
    hashlib.sha256
).hexdigest()
```

Endpoint: `POST /ocs/v2.php/apps/spreed/api/v1/bot/{token}/message`

Headers:
- `X-Nextcloud-Talk-Bot-Random: <random>`
- `X-Nextcloud-Talk-Bot-Signature: <signature>`

## Provisioner Service

An external service (default: `http://localhost:8790`) handles Nextcloud server operations.

**Auth:** `Authorization: Bearer <TALK_PROVISIONER_TOKEN>`

### POST /provision/nextcloud-talk

Creates a Talk room and registers a bot with Nextcloud:

```json
{
  "roomName": "Nova",
  "botName": "Nova",
  "webhookUrl": "http://localhost:8642/webhook/nextcloud",
  "ownerUserId": "user"
}
```

Returns room token, bot ID, secret, and room URL.

### POST /room/rename

Renames a Talk room by token.

## Nextcloud Server Setup

### Register a Bot Manually

If not using the provisioner service, register bots directly on the Nextcloud server:

```bash
sudo -u www-data php occ talk:bot:install \
  "llm-bawt Bot" \
  "my-super-secret-key-12345" \
  "http://your-server:8642/webhook/nextcloud"
```

### Useful occ Commands

```bash
sudo -u www-data php occ talk:bot:list              # List all bots
sudo -u www-data php occ talk:bot:state <token>      # List bots in a conversation
sudo -u www-data php occ talk:bot:uninstall <bot-id> # Remove a bot
```

### Add Bot to a Talk Room

1. Open a conversation in Nextcloud Talk
2. Click the "..." menu (top right)
3. Select "Conversation settings"
4. Go to "Bots" section
5. Select and add your bot

## File Locations

```
src/llm_bawt/integrations/nextcloud/
├── __init__.py       # Exports: NextcloudBot, NextcloudBotConfig, NextcloudBotManager
├── config.py         # NextcloudBotConfig, NextcloudBot dataclass
├── manager.py        # NextcloudBotManager singleton (routing, config CRUD)
├── webhook.py        # Webhook handler (signature verification, message processing)
├── provisioner.py    # ProvisionerClient (room/bot provisioning)
└── cli.py            # Click-based CLI commands (llm-nextcloud)
```

Service integration: `src/llm_bawt/service/api.py` (webhook and admin endpoints)

## Troubleshooting

**Command not found: llm-nextcloud**
```bash
./install.sh --local .   # For global (pipx)
uv run llm-nextcloud     # For local venv
```

**Provisioning fails with 401**
- Check `LLM_BAWT_TALK_PROVISIONER_TOKEN` is set and correct

**Bot not responding**
- Check `llm-nextcloud list` shows the bot with correct config
- Verify conversation_token matches the Talk room
- Check service logs: `./start.sh logs`

**Wrong bot personality responding**
- Verify conversation_token in bots.yaml matches the intended room
- Check webhook logs for routing decisions

**Webhooks not arriving**
- Check Nextcloud logs: `tail -f /var/log/nextcloud/nextcloud.log`
- Verify webhook URL is reachable from Nextcloud server
- Check firewall rules
- Verify bot is added to the conversation

**Signature validation fails**
- Ensure the secret matches exactly between Nextcloud and bots.yaml
- Verify UTF-8 encoding
- Check signature is computed as `HMAC-SHA256(random + body, secret)`
