"""Background task processing mixin.

Extracted from background_service.py — handles all _process_* methods
for compaction, embeddings, meaning updates, maintenance, profile
maintenance, history summarization, and memory extraction.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from .logging import get_service_logger
from .tasks import Task, TaskResult, TaskStatus, TaskType

log = get_service_logger(__name__)


class BackgroundTasksMixin:
    """Mixin providing background task processing for BackgroundService."""

    async def process_task(self, task: Task) -> TaskResult:
        """Process a single background task."""
        start_time = time.time()
        task_type_str = task.task_type.value

        log.debug(f"Processing task {task.task_id[:8]} ({task_type_str})")

        try:
            if task.task_type == TaskType.CONTEXT_COMPACTION:
                result = await self._process_compaction(task)
            elif task.task_type == TaskType.EMBEDDING_GENERATION:
                result = await self._process_embeddings(task)
            elif task.task_type == TaskType.MEANING_UPDATE:
                result = await self._process_meaning_update(task)
            elif task.task_type == TaskType.MEMORY_MAINTENANCE:
                result = await self._process_maintenance(task)
            elif task.task_type == TaskType.PROFILE_MAINTENANCE:
                result = await self._process_profile_maintenance(task)
            elif task.task_type == TaskType.HISTORY_SUMMARIZATION:
                result = await self._process_history_summarization(task)
            elif task.task_type == TaskType.MEMORY_EXTRACTION:
                result = await self._process_memory_extraction(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            elapsed_ms = (time.time() - start_time) * 1000
            log.task_completed(task.task_id, task_type_str, elapsed_ms, result)

            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                processing_time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            log.task_failed(task.task_id, task_type_str, str(e), elapsed_ms)
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                processing_time_ms=elapsed_ms,
            )

    async def _process_compaction(self, task: Task) -> dict:
        """Process a context compaction task."""
        # TODO: Implement with summarization model
        return {"compacted": False, "reason": "Not yet implemented"}

    async def _process_embeddings(self, task: Task) -> dict:
        """Process an embedding generation task."""
        # TODO: Implement with embedding model
        return {"embeddings_generated": 0, "reason": "Not yet implemented"}

    async def _process_meaning_update(self, task: Task) -> dict:
        """Process a meaning update task (via MCP tools)."""
        bot_id = task.bot_id
        payload = task.payload
        memory_id = payload.get("memory_id")
        if not memory_id:
            return {"updated": False, "reason": "No memory_id provided"}

        memory_client = self.get_memory_client(bot_id)
        if not memory_client:
            return {"updated": False, "reason": "Memory client unavailable"}

        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None,
            lambda: memory_client.update_memory_meaning(
                memory_id=memory_id,
                intent=payload.get("intent"),
                stakes=payload.get("stakes"),
                emotional_charge=payload.get("emotional_charge"),
                recurrence_keywords=payload.get("recurrence_keywords"),
                updated_tags=payload.get("updated_tags"),
            ),
        )
        return {"updated": bool(success), "memory_id": memory_id}

    async def _process_maintenance(self, task: Task) -> dict:
        """Process a unified memory maintenance task.

        Uses a cached LLM client if available, otherwise runs without LLM.
        Will NOT load a new model to avoid VRAM conflicts.
        """
        bot_id = task.bot_id
        payload = task.payload

        memory_client = self.get_memory_client(bot_id)
        if not memory_client:
            return {"error": "Memory client unavailable"}

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: memory_client.run_maintenance(
                run_consolidation=payload.get("run_consolidation", True),
                run_recurrence_detection=payload.get("run_recurrence_detection", True),
                run_decay_pruning=payload.get("run_decay_pruning", False),
                run_orphan_cleanup=payload.get("run_orphan_cleanup", False),
                dry_run=payload.get("dry_run", False),
            ),
        )
        return result

    async def _process_profile_maintenance(self, task: Task) -> dict:
        """Process profile maintenance - consolidate attributes into summary.

        Uses configured maintenance model resolution and loads that model
        deterministically when needed.
        """
        from ..memory.profile_maintenance import ProfileMaintenanceService
        from ..profiles import ProfileManager

        entity_id = task.payload.get("entity_id", task.user_id)
        entity_type = task.payload.get("entity_type", "user")
        dry_run = task.payload.get("dry_run", False)

        requested_model = (
            task.payload.get("model")
            or getattr(self.config, "MAINTENANCE_MODEL", None)
            or (self.config.PROFILE_MAINTENANCE_MODEL or None)
            or getattr(self.config, "SUMMARIZATION_MODEL", None)
        )
        try:
            model_to_use, _ = self._resolve_request_model(
                requested_model,
                task.bot_id or self._default_bot,
                local_mode=False,
            )
        except Exception as e:
            log.error(f"Failed to resolve model for profile maintenance: {e}")
            return {"error": f"Failed to resolve model: {e}"}

        if not model_to_use:
            err = "No model available for profile maintenance"
            log.error(err)
            return {"error": err}

        model_def = self.config.defined_models.get("models", {}).get(model_to_use, {})
        if model_def.get("type") == "openai" and not (self.config.OPENAI_API_KEY or self.config.XAI_API_KEY):
            err = f"Profile maintenance model '{model_to_use}' requires API key configuration"
            log.error(err)
            return {"error": err}

        if model_to_use not in self._available_models:
            err = f"Model '{model_to_use}' unavailable for profile maintenance"
            log.error(err)
            return {"error": err}

        log.info(f"🔧 Profile maintenance: {entity_type}/{entity_id} (model={model_to_use})")

        # Get or create profile manager
        profile_manager = ProfileManager(self.config)

        # Use isolated background client — never interferes with main chat model
        llm_client, _ = self._get_background_client(model_override=model_to_use)
        if not llm_client:
            return {"error": f"Failed to create background client for profile maintenance"}

        service = ProfileMaintenanceService(profile_manager, llm_client)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._llm_executor,  # Use same executor as extraction
            lambda: service.run(entity_id, entity_type, dry_run)
        )

        return {
            "entity_id": result.entity_id,
            "attributes_before": result.attributes_before,
            "attributes_after": result.attributes_after,
            "categories_updated": result.categories_updated,
            "error": result.error,
        }

    async def _extract_from_summaries(
        self,
        summarization_results: list[dict],
        bot_id: str,
        user_id: str,
    ) -> dict:
        """Extract facts from newly created summaries using an API model.

        Called after ``summarize_eligible_sessions()`` completes.  Only
        processes results where ``created`` is ``True``.

        Args:
            summarization_results: Result dicts from ``HistorySummarizer``
            bot_id: Bot whose memories are being updated
            user_id: User identity for memory storage

        Returns:
            Stats dict with extraction outcome.
        """
        from ..memory.extraction.service import MemoryExtractionService, MemoryAction

        extraction_client, extraction_model = self._get_background_client()
        use_llm = extraction_client is not None

        if not use_llm:
            return {
                "summaries_processed": 0,
                "facts_extracted": 0,
                "facts_stored": 0,
                "extraction_method": "skipped",
            }

        extraction_service = MemoryExtractionService(llm_client=extraction_client)
        memory_client = self.get_memory_client(bot_id, user_id)

        if not memory_client:
            log.warning("No memory client available — skipping extraction")
            return {
                "summaries_processed": 0,
                "facts_extracted": 0,
                "facts_stored": 0,
                "extraction_method": "skipped",
            }

        min_importance = getattr(self.config, "MEMORY_EXTRACTION_MIN_IMPORTANCE", 0.5)
        profile_enabled = getattr(self.config, "MEMORY_PROFILE_ATTRIBUTE_ENABLED", True)

        total_facts = 0
        total_stored = 0
        summaries_processed = 0

        for result in summarization_results:
            if not result.get("created", False):
                continue

            summary_text = result.get("summary_text")
            summary_id = result.get("summary_id")
            session_start = result.get("session_start", 0)
            session_end = result.get("session_end", 0)

            if not summary_text or not summary_id:
                continue

            try:
                facts = extraction_service.extract_from_summary(
                    summary_text=summary_text,
                    session_start=session_start,
                    session_end=session_end,
                    summary_id=summary_id,
                    use_llm=use_llm,
                )

                summaries_processed += 1
                total_facts += len(facts)

                log.info(
                    f"[Extraction] Summary {summary_id[:8]}: "
                    f"{len(facts)} facts extracted from: {summary_text[:200]}"
                )

                if not facts:
                    log.info(f"[Extraction] Summary {summary_id[:8]}: no facts found — skipping")
                    self._mark_summary_extracted(bot_id, summary_id)
                    continue

                # Filter by importance
                pre_filter = len(facts)
                facts = [f for f in facts if f.importance >= min_importance]
                if not facts:
                    log.info(
                        f"[Extraction] Summary {summary_id[:8]}: "
                        f"all {pre_filter} facts below importance threshold ({min_importance})"
                    )
                    self._mark_summary_extracted(bot_id, summary_id)
                    continue

                # Deduplicate against existing memories
                existing_memories = memory_client.list_memories(limit=100, min_importance=0.0)

                if existing_memories:
                    actions = extraction_service.determine_memory_actions(
                        new_facts=facts,
                        existing_memories=existing_memories,
                    )
                else:
                    actions = [MemoryAction(action="ADD", fact=f) for f in facts]

                for action in actions:
                    fact = action.fact
                    if not fact:
                        continue

                    try:
                        tag_str = ",".join(fact.tags)
                        if action.action == "ADD":
                            log.info(
                                f"[Extraction] ADD: '{fact.content}' "
                                f"[{tag_str}] importance={fact.importance:.2f}"
                            )
                            memory_client.add_memory(
                                content=fact.content,
                                tags=fact.tags,
                                importance=fact.importance,
                                source_message_ids=fact.source_message_ids,
                            )
                            total_stored += 1
                        elif action.action == "UPDATE" and action.target_memory_id:
                            log.info(
                                f"[Extraction] UPDATE ({action.target_memory_id[:8]}): "
                                f"'{fact.content}' [{tag_str}] importance={fact.importance:.2f}"
                            )
                            memory_client.update_memory(
                                memory_id=action.target_memory_id,
                                content=fact.content,
                                importance=fact.importance,
                                tags=fact.tags,
                            )
                            total_stored += 1
                        elif action.action == "DELETE" and action.target_memory_id:
                            log.info(
                                f"[Extraction] DELETE: {action.target_memory_id[:8]} "
                                f"reason='{action.reason}'"
                            )
                            memory_client.delete_memory(memory_id=action.target_memory_id)

                        # Profile attribute extraction
                        if action.action in ("ADD", "UPDATE") and profile_enabled:
                            from ..memory_server.extraction import extract_profile_attributes_from_fact
                            extract_profile_attributes_from_fact(
                                fact=fact,
                                user_id=user_id,
                                config=self.config,
                            )
                    except Exception as e:
                        log.warning(f"Failed to process memory action {action.action}: {e}")

                # Mark summary as extracted (crash recovery marker)
                self._mark_summary_extracted(bot_id, summary_id)

            except Exception as e:
                log.error(f"Failed to extract from summary {summary_id[:8]}: {e}")
                continue

        log.info(
            f"[Extraction] Done: {summaries_processed} summaries processed, "
            f"{total_facts} facts extracted, {total_stored} stored"
        )

        return {
            "summaries_processed": summaries_processed,
            "facts_extracted": total_facts,
            "facts_stored": total_stored,
            "extraction_method": extraction_model or "heuristic",
        }

    def _mark_summary_extracted(self, bot_id: str, summary_id: str) -> None:
        """Set ``extracted_at`` in a summary's metadata for crash recovery."""
        try:
            memory_client = self.get_memory_client(bot_id, "system")
            if not memory_client:
                return
            backend = getattr(memory_client, "_backend", None) or getattr(memory_client, "backend", None)
            if not backend or not hasattr(backend, "engine"):
                return
            from sqlalchemy import text as sa_text
            table = f"{bot_id}_messages"
            with backend.engine.connect() as conn:
                conn.execute(
                    sa_text(f"""
                        UPDATE {table}
                        SET summary_metadata = jsonb_set(
                            COALESCE(summary_metadata::jsonb, '{{}}'::jsonb),
                            '{{extracted_at}}',
                            to_jsonb(:ts)
                        )
                        WHERE id = :sid AND role = 'summary'
                    """),
                    {"ts": datetime.now(timezone.utc).isoformat(), "sid": summary_id},
                )
                conn.commit()
        except Exception as e:
            log.debug(f"Failed to mark summary {summary_id[:8]} as extracted: {e}")

    async def _process_history_summarization(self, task: Task) -> dict:
        """Process proactive history summarization and extraction for a bot."""
        from ..memory.summarization import (
            HistorySummarizer,
            format_session_for_summarization,
        )
        from ..models.message import Message
        from ..prompt_registry import PromptResolver

        bot_id = task.bot_id or self._default_bot
        requested_model = (
            task.payload.get("model")
            or getattr(self.config, "MAINTENANCE_MODEL", None)
            or getattr(self.config, "SUMMARIZATION_MODEL", None)
        )
        use_heuristic_fallback = bool(task.payload.get("use_heuristic_fallback", True))
        max_tokens_per_chunk = int(task.payload.get("max_tokens_per_chunk", 4000))

        # Use isolated background client — never interferes with main chat model
        client, model_alias = self._get_background_client(model_override=requested_model)

        prompt_resolver = PromptResolver(self.config)

        def summarize_with_loaded_client(session) -> str | None:
            if not client:
                return None

            conversation_text = format_session_for_summarization(session)
            prompt = prompt_resolver.render(
                key="history.summarization.single",
                variables={"messages": conversation_text},
            )

            # Conservative budget to reduce context overflows on smaller models.
            if len(prompt) // 4 > 6000:
                return None

            try:
                messages = [
                    Message(
                        role="system",
                        content="You are a helpful assistant that summarizes conversations concisely.",
                    ),
                    Message(role="user", content=prompt),
                ]
                response = client.query(
                    messages=messages,
                    max_tokens=320,
                    temperature=0.3,
                    plaintext_output=True,
                    stream=False,
                )
                if not response:
                    return None

                lower = response.lower()
                error_indicators = ("error:", "exception occurred", "exceed context window", "tokens exceed")
                if any(ind in lower for ind in error_indicators):
                    return None
                return response.strip()
            except Exception as e:
                log.error(f"History summarization LLM call failed: {e}")
                return None

        # Create a settings resolver for per-bot summarization tunables
        from ..runtime_settings import RuntimeSettingsResolver
        from ..bots import BotManager
        bot_manager = BotManager(self.config)
        bot_obj = bot_manager.get_bot(bot_id) or bot_manager.get_default_bot()
        resolver = RuntimeSettingsResolver(config=self.config, bot=bot_obj, bot_id=bot_id)

        # Compute effective context budget from the resolved model
        max_context_tokens = 0
        if model_alias:
            ctx_window = int(self.config.get_model_context_window(model_alias) or 0)
            max_output = int(self.config.get_model_max_tokens(model_alias) or 4096)
            if ctx_window > 0:
                max_context_tokens = ctx_window - max_output

        summarizer = HistorySummarizer(
            self.config,
            bot_id=bot_id,
            summarize_fn=summarize_with_loaded_client,
            settings_getter=resolver.resolve,
            max_context_tokens=max_context_tokens,
        )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._llm_executor,
            lambda: summarizer.summarize_eligible_sessions(
                use_heuristic_fallback=use_heuristic_fallback,
                max_tokens_per_chunk=max_tokens_per_chunk,
            ),
        )

        if model_alias:
            result["model"] = model_alias
        result["used_heuristic_fallback"] = use_heuristic_fallback

        # Phase 2: Extract memories from newly created summaries
        # For profile attribute extraction, use the actual human user — not the
        # technical "system" user_id that history summarization jobs carry.
        profile_user_id = task.user_id
        if not profile_user_id or profile_user_id == "system":
            profile_user_id = getattr(self.config, "DEFAULT_USER", None) or "system"
        extraction_results = await self._extract_from_summaries(
            result.get("results", []),
            bot_id=bot_id,
            user_id=profile_user_id,
        )
        result["extraction"] = extraction_results

        # Summary writes change context composition; invalidate cached history views
        # for this bot so next request reloads from DB immediately.
        for (_model, cached_bot_id, _user), llm_bawt in self._llm_bawt_cache.items():
            if cached_bot_id != bot_id:
                continue
            invalidate = getattr(llm_bawt, "invalidate_history_cache", None)
            if callable(invalidate):
                invalidate()

        return result

    async def _process_memory_extraction(self, task: Task) -> dict:
        """Extract memories from unprocessed conversation messages.

        Reads batches of unprocessed messages from the bot's message table,
        runs LLM-based fact extraction, stores resulting memories, and marks
        the source messages as processed.
        """
        from ..memory.extraction.service import MemoryExtractionService, MemoryAction

        bot_id = task.bot_id or self._default_bot
        user_id = task.user_id
        if not user_id or user_id == "system":
            user_id = getattr(self.config, "DEFAULT_USER", None) or "system"
        batch_size = int(task.payload.get("batch_size", 50))

        # Check config gate
        if not getattr(self.config, "MEMORY_EXTRACTION_ENABLED", True):
            return {"skipped": True, "reason": "Memory extraction disabled in config"}

        memory_client = self.get_memory_client(bot_id, user_id)
        if not memory_client:
            return {"error": "Memory client unavailable", "bot_id": bot_id}

        # Get a PostgreSQL backend directly for message table access.
        # MemoryClient may be in MCP server mode where _get_storage() is
        # unavailable, so we create the backend ourselves.
        from ..memory.postgresql import PostgreSQLMemoryBackend

        try:
            backend = PostgreSQLMemoryBackend(self.config, bot_id=bot_id)
        except Exception as e:
            return {"error": f"Failed to create memory backend: {e}"}

        loop = asyncio.get_event_loop()

        unprocessed = await loop.run_in_executor(
            None,
            lambda: backend.get_unprocessed_messages(limit=batch_size),
        )

        if not unprocessed:
            return {
                "bot_id": bot_id,
                "messages_found": 0,
                "facts_extracted": 0,
                "facts_stored": 0,
            }

        log.info(
            "[MemExtract] %s: %d unprocessed messages found",
            bot_id,
            len(unprocessed),
        )

        # Use isolated background client for LLM calls
        extraction_client, extraction_model = self._get_background_client(
            model_override=task.payload.get("model"),
        )
        use_llm = extraction_client is not None

        extraction_service = MemoryExtractionService(llm_client=extraction_client)
        min_importance = getattr(self.config, "MEMORY_EXTRACTION_MIN_IMPORTANCE", 0.5)
        profile_enabled = getattr(self.config, "MEMORY_PROFILE_ATTRIBUTE_ENABLED", True)

        # Build conversation chunks for extraction — group consecutive messages
        conversation = [
            {"role": m["role"], "content": m["content"]}
            for m in unprocessed
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        message_ids = [m["id"] for m in unprocessed]

        total_facts = 0
        total_stored = 0

        if conversation:
            try:
                facts = await loop.run_in_executor(
                    self._llm_executor,
                    lambda: extraction_service.extract_from_conversation(
                        messages=conversation,
                        use_llm=use_llm and extraction_service.llm_client is not None,
                    ),
                )

                # Filter by importance
                facts = [f for f in facts if f.importance >= min_importance]
                total_facts = len(facts)

                if facts:
                    # Deduplicate against existing memories
                    existing_memories = memory_client.list_memories(
                        limit=100, min_importance=0.0,
                    )

                    if existing_memories:
                        actions = extraction_service.determine_memory_actions(
                            new_facts=facts,
                            existing_memories=existing_memories,
                        )
                    else:
                        actions = [MemoryAction(action="ADD", fact=f) for f in facts]

                    for action in actions:
                        fact = action.fact
                        if not fact:
                            continue
                        try:
                            tag_str = ",".join(fact.tags)
                            if action.action == "ADD":
                                log.info(
                                    "[MemExtract] ADD: '%s' [%s] importance=%.2f",
                                    fact.content,
                                    tag_str,
                                    fact.importance,
                                )
                                memory_client.add_memory(
                                    content=fact.content,
                                    tags=fact.tags,
                                    importance=fact.importance,
                                    source_message_ids=message_ids[:5],
                                )
                                total_stored += 1
                            elif action.action == "UPDATE" and action.target_memory_id:
                                log.info(
                                    "[MemExtract] UPDATE (%s): '%s'",
                                    action.target_memory_id[:8],
                                    fact.content,
                                )
                                memory_client.update_memory(
                                    memory_id=action.target_memory_id,
                                    content=fact.content,
                                    importance=fact.importance,
                                    tags=fact.tags,
                                )
                                total_stored += 1
                            elif action.action == "DELETE" and action.target_memory_id:
                                log.info(
                                    "[MemExtract] DELETE: %s reason='%s'",
                                    action.target_memory_id[:8],
                                    action.reason,
                                )
                                memory_client.delete_memory(
                                    memory_id=action.target_memory_id,
                                )

                            # Profile attribute extraction
                            if action.action in ("ADD", "UPDATE") and profile_enabled:
                                from ..memory_server.extraction import (
                                    extract_profile_attributes_from_fact,
                                )
                                extract_profile_attributes_from_fact(
                                    fact=fact,
                                    user_id=user_id,
                                    config=self.config,
                                )
                        except Exception as e:
                            log.warning(
                                "[MemExtract] Failed action %s: %s",
                                action.action,
                                e,
                            )

            except Exception as e:
                log.error("[MemExtract] Extraction failed for %s: %s", bot_id, e)

        # Mark messages as processed regardless of extraction outcome
        # so they aren't re-processed on the next run
        await loop.run_in_executor(
            None,
            lambda: backend.mark_messages_processed(message_ids),
        )

        log.info(
            "[MemExtract] %s: %d messages processed, %d facts extracted, %d stored",
            bot_id,
            len(unprocessed),
            total_facts,
            total_stored,
        )

        return {
            "bot_id": bot_id,
            "messages_found": len(unprocessed),
            "facts_extracted": total_facts,
            "facts_stored": total_stored,
            "extraction_method": extraction_model or "heuristic",
        }
