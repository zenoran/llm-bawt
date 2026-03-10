  Issue: Tool call XML leaking into chat message text for Grok (proto bot)                
                                                                                          
  The proto bot (grok-4-fast) sometimes outputs <tool_call> XML directly in delta.content 
  instead of using native delta.tool_calls when called from the UI. Curiously, direct curl
   tests to the backend show clean separation — delta.tool_calls comes first, then clean
  delta.content with no XML. The difference appears to be context-dependent (conversation
  history, memory augmentation, etc.).

  Root cause hypothesis: We're using xAI's Chat Completions API (/v1/chat/completions),
  which xAI now considers legacy. Their docs show a newer Responses API (/v1/responses) as
   the recommended endpoint, with "native support for tools (search, code execution, MCP)"
   vs Chat Completions' "function calling only." Grok models may be optimized for the
  Responses API flow, causing inconsistent tool calling behavior on the older endpoint.

  What needs planning:
  - Refactor the GrokClient (and possibly OpenAIClient base) to support xAI's Responses
  API (/v1/responses)
  - The Responses API has a different request/response format: stateful conversations via
  previous_response_id, server-side response storage (30 days), different tool calling
  semantics
  - Evaluate whether to keep Chat Completions as fallback for other OpenAI-compatible
  providers (GGUF, other APIs) while Grok uses Responses
  - Relevant xAI docs: https://docs.x.ai/developers/model-capabilities/text/generate-text
  and https://docs.x.ai/developers/model-capabilities/text/comparison
  - Key files: /home/nick/dev/llm-bawt/src/llm_bawt/clients/grok_client.py,
  openai_client.py, /home/nick/dev/llm-bawt/src/llm_bawt/service/chat_streaming.py
