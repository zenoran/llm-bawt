# Inter-Bot Communication

llm-bawt now supports **inter-bot communication** through MCP tools, allowing bots to send messages to each other and receive responses. This enables collaborative workflows, delegation, and multi-bot interactions within the same llm-bawt instance.

## Overview

The inter-bot communication system provides two MCP tools:

1. **`send_message_to_bot`** - Send a message to another bot and get their response
2. **`list_available_bots`** - Discover what bots are available for communication

These tools are exposed via the MCP memory server (port 8001) and can be used by any bot that has access to MCP tools.

## How It Works

```
Bot A → send_message_to_bot → llm-bawt → Bot B → Response → Bot A
```

When a bot uses `send_message_to_bot`:

1. The message is sent through the internal chat completions API
2. The target bot processes it with their personality, model, and memory
3. The response is returned to the sender
4. Each interaction is isolated - no persistent conversation state is maintained

## MCP Tools Reference

### send_message_to_bot

Send a message to another bot and receive their response.

```python
{
  "target_bot_id": "nova",           # Bot slug to send message to
  "message": "What's your specialty?", # Message content
  "sender_bot_id": "mira",           # Optional: sender identification
  "max_tokens": 150,                 # Optional: response length limit
  "temperature": 0.7                 # Optional: response creativity
}
```

**Returns:**
```python
{
  "success": True,
  "content": "I specialize in creative writing and storytelling...",
  "bot_id": "nova",
  "sender": "mira",
  "response_model": "gpt-4"
}
```

**Error Response:**
```python
{
  "success": False,
  "error": "Bot 'nonexistent' not found",
  "content": "",
  "bot_id": "nonexistent",
  "sender": "mira"
}
```

### list_available_bots

Discover available bots for communication.

```python
# No parameters required
```

**Returns:**
```python
[
  {
    "slug": "nova",
    "name": "Nova",
    "bot_type": "chat",
    "description": "Creative writing and storytelling assistant",
    "default_model": "gpt-4",
    "agent_backend": null
  },
  {
    "slug": "mira",
    "name": "Mira",
    "bot_type": "agent",
    "description": "Technical analysis and research",
    "default_model": "claude-3-sonnet",
    "agent_backend": "claude-code"
  }
]
```

## Usage Examples

### Basic Bot-to-Bot Communication

```python
# Bot A wants to ask Bot B a question
response = await send_message_to_bot(
    target_bot_id="researcher",
    message="Can you research the latest developments in quantum computing?",
    sender_bot_id="coordinator"
)

if response["success"]:
    research_results = response["content"]
    # Process the research results...
```

### Discovering Available Bots

```python
# Find out what bots are available
bots = await list_available_bots()

# Filter for specific types
chat_bots = [bot for bot in bots if bot["bot_type"] == "chat"]
agent_bots = [bot for bot in bots if bot["bot_type"] == "agent"]

print(f"Available chat bots: {[b['slug'] for b in chat_bots]}")
print(f"Available agent bots: {[b['slug'] for b in agent_bots]}")
```

### Delegation Pattern

```python
# Coordinator bot delegating tasks
async def delegate_task(task_description, specialist_bot):
    response = await send_message_to_bot(
        target_bot_id=specialist_bot,
        message=f"Please handle this task: {task_description}",
        sender_bot_id="coordinator",
        max_tokens=500
    )
    
    return {
        "specialist": specialist_bot,
        "result": response["content"],
        "success": response["success"]
    }

# Usage
code_review = await delegate_task(
    "Review this Python code for security issues",
    "security_expert"
)

writing_help = await delegate_task(
    "Polish this technical documentation",
    "editor"
)
```

## Memory and Isolation

### Memory Behavior

- **Target bot memory**: The receiving bot can access their own memories (`augment_memory=True`)
- **Memory extraction**: By default, inter-bot messages don't create new memories (`extract_memory=False`)
- **Isolation**: Each bot maintains separate memory namespaces - no cross-contamination

### Conversation State

- Each `send_message_to_bot` call is **stateless**
- No persistent conversation history between bots
- To maintain context across multiple exchanges, bots must explicitly include relevant information in each message

## Security and Best Practices

### Access Control

- All bots can communicate with each other within the same llm-bawt instance
- No authentication between bots - they trust each other
- Consider this when designing bot personalities and capabilities

### Message Design

- **Be explicit**: Include context in each message since there's no conversation history
- **Identify yourself**: Use `sender_bot_id` to help the target bot understand who's asking
- **Set limits**: Use `max_tokens` to control response length for efficiency

### Error Handling

```python
response = await send_message_to_bot(target_bot_id="analyst", message="Analyze this data")

if not response["success"]:
    if "not found" in response["error"]:
        # Handle missing bot
        available_bots = await list_available_bots()
        # Find alternative or notify user
    else:
        # Handle other errors (service unavailable, etc.)
        logger.error(f"Bot communication failed: {response['error']}")
```

## Use Cases

### 1. Specialist Coordination
- A **coordinator bot** routes questions to specialist bots (technical, creative, research)
- Each specialist handles their domain and returns focused responses

### 2. Multi-Step Workflows
- **Research bot** gathers information → **Analysis bot** processes it → **Writer bot** creates summary
- Each step uses the previous bot's output as input

### 3. Quality Assurance
- **Creator bot** generates content → **Reviewer bot** checks quality → **Editor bot** polishes
- Multiple bots collaborate to improve output quality

### 4. Domain Translation
- **Technical bot** explains complex concepts → **Educator bot** simplifies for learning
- Information flows between bots with different communication styles

## Technical Details

### Implementation

- Built on top of existing `ChatCompletionRequest` infrastructure
- Uses internal service calls - no external HTTP requests
- Integrates with bot personality system and memory isolation
- Exposed via MCP tools for universal access

### Performance

- **Fast**: Internal API calls within the same process
- **Efficient**: No network overhead or serialization costs
- **Scalable**: Limited only by bot processing capacity

### Limitations

- **Same instance only**: Bots must be in the same llm-bawt deployment
- **No conversation persistence**: Each exchange is independent
- **No streaming**: Responses are returned complete (not streamed)
- **Memory isolation**: Bots can't access each other's memories directly

## Configuration

No additional configuration required. Inter-bot communication is enabled automatically when:

1. MCP memory server is running (port 8001)
2. Multiple bots are configured
3. Bots have access to MCP tools

The feature uses existing bot configurations and respects all bot settings (model, temperature, tools, etc.).