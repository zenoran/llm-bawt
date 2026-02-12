Plan: Scheduler-Based Memory Extraction via Grok
Context
Memory extraction currently fires after every chat turn, using the same local GGUF model that handles chat. This produces garbage-quality facts from 8B/32B quantized models. The system already has a scheduler job that periodically summarizes conversation sessions (HISTORY_SUMMARIZATION).

This change merges extraction into that same scheduler loop: after summaries are created, feed them to Grok (xAI API) for high-quality fact extraction. Per-turn extraction is removed entirely.

The codebase already has: GrokClient, EXTRACTION_MODEL config, XAI_API_KEY config, and the full MemoryExtractionService infrastructure.

Changes
1. New summary-based extraction prompt
File: src/llm_bawt/memory/extraction/prompts.py

Add SUMMARY_EXTRACTION_PROMPT_TEMPLATE and get_summary_extraction_prompt() helper. The prompt takes a conversation summary + session date range and asks for structured JSON facts. Same output format as existing (facts array with content, tags, importance, profile_attribute), but input is a summary instead of raw message pairs.

Key prompt differences from current:

No [USER SAID] / [ASSISTANT SAID] framing — summaries are already third-person
Includes session date range for temporal context
Emphasizes that summaries are pre-filtered so extraction should still be selective
2. Add extract_from_summary() to MemoryExtractionService
File: src/llm_bawt/memory/extraction/service.py

New method extract_from_summary(summary_text, session_start, session_end, summary_id, use_llm=True):

With LLM: builds prompt via get_summary_extraction_prompt(), queries client, parses JSON response using existing _parse_extraction_response(), enriches with _enrich_meaning()
Without LLM: falls back to heuristic extraction on the summary text (existing _extract_with_heuristics())
Returns list[ExtractedFact] — same as current extraction
3. Add session timestamps to summarizer return value
File: src/llm_bawt/memory/summarization.py

In summarize_session() return dict (line 592), add session_start and session_end from the Session object. The extraction step needs these for the prompt and they're already available on the Session dataclass.

4. Add extraction client management to BackgroundService
File: src/llm_bawt/service/background_service.py

Add _extraction_client / _extraction_client_model instance variables and _get_extraction_client() method:

Reads config.EXTRACTION_MODEL to determine model alias
Resolves model definition, validates it's openai or grok type (no local models)
Creates client via existing _get_llm_bawt() infrastructure
Caches for reuse across scheduler runs
Returns (None, None) if unavailable (no config, bad key, etc.)
5. Add _extract_from_summaries() to BackgroundService
File: src/llm_bawt/service/background_service.py

New async method called after summarization completes. For each newly created summary in the results:

Call extraction_service.extract_from_summary() with summary text + session metadata
Filter by MEMORY_EXTRACTION_MIN_IMPORTANCE
Run determine_memory_actions() against existing memories (dedup)
Process ADD/UPDATE/DELETE actions via memory client
Extract profile attributes for ADD/UPDATE actions
Return stats dict (summaries_processed, facts_extracted, facts_stored, extraction_method)
6. Wire extraction into _process_history_summarization()
File: src/llm_bawt/service/background_service.py

After the existing summarizer.summarize_eligible_sessions() call, add:


extraction_results = await self._extract_from_summaries(
    summarization_result.get("results", []),
    bot_id=bot_id,
    user_id=task.user_id,
)
Merge extraction stats into the returned result dict.

7. Remove per-turn extraction triggers
File: src/llm_bawt/core/pipeline.py

Remove _trigger_memory_extraction() call from _stage_post_process() (line 560-561)
Remove _trigger_memory_extraction() method (lines 568-594)
Update record_output to remove extraction_triggered key
File: src/llm_bawt/core/base.py

Remove self._trigger_memory_extraction(prompt, assistant_response) call (line 382-383)
Remove _trigger_memory_extraction() method (lines 541-564)
8. Update config documentation
File: src/llm_bawt/utils/config.py

Update EXTRACTION_MODEL field description to reference Grok and scheduler-based extraction
File: .env.docker

Add comments explaining the new extraction flow and Grok configuration
Fallback Behavior
No EXTRACTION_MODEL configured → extraction step skipped, summarization still runs normally
Grok API key missing/invalid → client creation fails, extraction skipped with warning log
Grok API call fails mid-run → that summary's extraction fails, continues to next summary
Heuristic summaries → still extracted from (Grok can work with any text, doesn't matter how the summary was produced)
Files Modified
File	Change
src/llm_bawt/memory/extraction/prompts.py	Add summary extraction prompt
src/llm_bawt/memory/extraction/service.py	Add extract_from_summary() method
src/llm_bawt/memory/summarization.py	Add session_start/session_end to return dict
src/llm_bawt/service/background_service.py	Add extraction client mgmt + wire into summarization job
src/llm_bawt/core/pipeline.py	Remove per-turn extraction trigger
src/llm_bawt/core/base.py	Remove per-turn extraction trigger
src/llm_bawt/utils/config.py	Update EXTRACTION_MODEL description
.env.docker	Update extraction config docs
Not Changed
HistorySummarizer logic (summarization works as-is)
GrokClient (already works)
Memory storage (postgresql.py add/update/delete)
ExtractedFact / MemoryAction dataclasses
Scheduler job type / init_default_jobs() (same job, just does more)
determine_memory_actions() dedup logic
Verification
Set LLM_BAWT_EXTRACTION_MODEL=grok-2 and LLM_BAWT_XAI_API_KEY in config
Have a multi-turn conversation with facts ("I'm a software engineer in Ohio with 2 dogs")
Wait for summarization interval (or trigger manually via API)
Check logs for extraction output (Extracted N memories from M summaries)
Query memories via llm-memory or API to verify facts stored
Test without EXTRACTION_MODEL set — verify summarization still works, extraction skipped gracefully
Run existing tests: uv run pytest tests/test_scheduler.py