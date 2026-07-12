"""FastAPI-based background service wiring for llm-bawt."""

import asyncio
from contextlib import asynccontextmanager

from ..utils.config import Config
from .background_service import BackgroundService, _ensure_mcp_server
from .dependencies import get_service, set_service
from .logging import get_service_logger, setup_service_logging
from .schemas import ChatCompletionRequest, ChatMessage

log = get_service_logger(__name__)

# Configuration
DEFAULT_HTTP_PORT = 8642
SERVICE_VERSION = "0.1.0"

def _is_tcp_listening(host: str, port: int) -> bool:
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan handler for startup/shutdown."""
    config = Config()

    # Prefer MCP tool-based memory retrieval for llm-service.
    # This ensures memory retrieval happens via MCP tools and can be logged clearly.
    _ensure_mcp_server(config)

    # Load model definitions from DB and merge into config.
    # DB always takes priority; YAML is seeded to DB on first run if DB is empty.
    from ..runtime_settings import ModelDefinitionStore
    model_store = ModelDefinitionStore(config)
    if model_store.engine is not None:
        # TASK-548: additive, idempotent catalog normalization.  This runs
        # before legacy definitions are loaded so bot endpoint/harness columns
        # and the compatibility projection are always ready for the resolver
        # and API cutover tasks that follow.
        try:
            from ..memory.model_catalog_migration import migrate_model_catalog
            _catalog_result = migrate_model_catalog(model_store.engine)
            log.info("model catalog migration: %s", _catalog_result)
        except Exception as e:
            log.warning("model catalog migration skipped: %s", e)

        yaml_models = config.defined_models.get("models", {})
        if yaml_models and model_store.count() == 0:
            seeded = model_store.seed_from_yaml(yaml_models)
            log.info("Seeded %d model definitions from YAML to DB", seeded)
        db_models = model_store.to_config_dict()
        if db_models:
            config.merge_db_models(db_models)
            log.debug("Loaded %d model definitions from DB", len(db_models))

    # Consolidate legacy agent_backend_config.model onto default_model
    # (idempotent; cheap existence check before doing any work). Runs after
    # model seeding so created catalog entries are immediately visible, and
    # the merged config is refreshed when the migration created entries.
    if model_store.engine is not None:
        try:
            from sqlalchemy import text as _sa_text
            with model_store.engine.connect() as _conn:
                _has_legacy = _conn.execute(_sa_text(
                    "SELECT 1 FROM bot_profiles "
                    "WHERE agent_backend IS NOT NULL "
                    "  AND agent_backend_config ? 'model' "
                    "  AND NOT (agent_backend_config ? 'session_model') LIMIT 1"
                )).fetchone()
            if _has_legacy:
                from types import SimpleNamespace
                from ..memory.migrations import migrate_agent_backend_config_model
                _result = migrate_agent_backend_config_model(
                    SimpleNamespace(engine=model_store.engine)
                )
                log.info("agent_backend_config.model migration: %s", _result)
                db_models = model_store.to_config_dict()
                if db_models:
                    config.merge_db_models(db_models)
        except Exception as e:
            log.warning("agent_backend_config.model migration skipped: %s", e)

    # Ensure media_generations table exists
    try:
        from ..media.db import MediaGenerationStore
        _media_store = MediaGenerationStore(config)
        log.info("media_generations table ready")
    except Exception as e:
        log.warning("Failed to initialise media_generations table: %s", e)

    # TASK-222: ensure media_assets table exists. Separate from
    # media_generations (different concern — that one tracks long-running
    # generation jobs; this one is the canonical registry for normalized
    # chat-upload / tool-output / agent-attachment blobs).
    try:
        from ..media.assets import MediaAssetStore
        _media_asset_store = MediaAssetStore(config)
        log.info("media_assets table ready")
    except Exception as e:
        log.warning("Failed to initialise media_assets table: %s", e)

    # TASK-216: warm the sentence-transformer used by the animation
    # classifier (and the memory subsystem) in the background so the first
    # voice response after a restart doesn't pay a ~6s lazy-load. Fire it
    # off-thread — the model load is CPU-bound and ~6s, and we don't want
    # to block the event loop or delay the rest of startup.
    # NB: a later `import asyncio` inside this lifespan function shadows
    # the module-level import, so use threading directly to avoid the
    # UnboundLocalError trap.
    def _warm_embedding_model() -> None:
        try:
            from ..memory.embeddings import generate_embedding
            generate_embedding("warmup")
            log.info("🔥 embedding model warm")
        except Exception as warm_err:
            log.debug("embedding warmup skipped: %s", warm_err)

    import threading as _threading
    _threading.Thread(target=_warm_embedding_model, daemon=True, name="embedding-warmup").start()

    service = BackgroundService(config)
    set_service(service)
    service.start_worker()

    # Start job scheduler if enabled
    scheduler = None
    if config.SCHEDULER_ENABLED:
        from ..profiles import ProfileManager
        from .scheduler import JobScheduler, create_scheduler_tables, init_default_jobs

        # Get engine from profile manager (reuse existing connection)
        pm = ProfileManager(config)
        create_scheduler_tables(pm.engine)
        init_default_jobs(pm.engine, config)

        scheduler = JobScheduler(
            engine=pm.engine,
            task_processor=service,
            check_interval=config.SCHEDULER_CHECK_INTERVAL_SECONDS,
        )
        await scheduler.start()
        log.info(f"📅 Scheduler started (interval={config.SCHEDULER_CHECK_INTERVAL_SECONDS}s)")

    # Start OpenClaw integration (Redis subscriber or in-process bridge)
    redis_subscriber = None
    history_drain_task = None
    if config.OPENCLAW_WS_ENABLED and config.OPENCLAW_WS_URL:
        try:
            # Build session_key -> bot_id mapping from openclaw bots
            from ..bots import BotManager
            session_to_bot: dict[str, str] = {}
            bot_mgr = BotManager(config)

            for bot in bot_mgr.list_bots():
                if bot.agent_backend != "openclaw":
                    continue
                bc = bot.agent_backend_config or {}
                sk = bc.get("session_key")
                if sk:
                    session_to_bot[sk] = bot.slug

            if session_to_bot:
                log.info("OpenClaw session->bot mapping: %s", session_to_bot)

            if not config.REDIS_URL:
                log.error(
                    "OpenClaw WS is enabled but REDIS_URL is not set. "
                    "The bridge requires Redis for command/event transport."
                )
            else:
                from agent_bridge.subscriber import RedisSubscriber
                from ..agent_backends.agent_bridge import set_agent_subscriber

                redis_subscriber = RedisSubscriber(config.REDIS_URL)
                await redis_subscriber.connect()
                service._redis_subscriber = redis_subscriber

                # Make subscriber available to OpenClawBackend instances
                set_agent_subscriber(redis_subscriber)

                # DEACTIVATED: passive history drain causes duplicates with
                # finalize_response().  The active chat.send path persists via
                # finalize_response() and does not need this consumer.
                # Re-enable when async/cron traffic persistence is redesigned.
                #
                # def _history_sink(bot_id: str, role: str, content: str) -> None:
                #     client = service.get_memory_client(bot_id)
                #     if client:
                #         client.add_message(role=role, content=content)
                #
                # import asyncio
                # history_drain_task = asyncio.create_task(
                #     redis_subscriber.drain_history(_history_sink)
                # )
                # service._history_drain_task = history_drain_task

                # Start background task to clean up stale consumer groups
                async def _cleanup_stale_groups():
                    """Periodically destroy idle ui:* consumer groups."""
                    import asyncio as _asyncio
                    while True:
                        await _asyncio.sleep(300)  # every 5 minutes
                        try:
                            streams = await redis_subscriber.list_unified_streams()
                            total = 0
                            for stream_key in streams:
                                total += await redis_subscriber.cleanup_stale_groups(stream_key)
                            if total:
                                log.info("Cleaned up %d stale consumer groups", total)
                        except Exception:
                            log.debug("Stale group cleanup error", exc_info=True)

                import asyncio
                service._group_cleanup_task = asyncio.create_task(_cleanup_stale_groups())

                # Start persistence consumer for tool events → Postgres.
                # TASK-286: also accumulates assistant text deltas per turn so a
                # COLD reload (and a turn that completes after the client
                # disconnects) recovers the response TEXT, not just tool calls.
                _partial_text: dict[str, str] = {}
                # TASK-360 (P4): accumulate per-turn reasoning ("thinking") the
                # same way, flushed to turn_logs.reasoning so a COLD reload
                # mid-turn recovers already-produced reasoning, not just text.
                _partial_reasoning: dict[str, str] = {}

                def _tool_event_sink(event_data: dict) -> None:
                    """Persist tool_start/tool_end + text_delta + reasoning_delta events to Postgres."""
                    store = service._turn_log_store
                    _type = event_data.get("_type")
                    if _type == "reasoning_delta":
                        turn_id = event_data.get("turn_id")
                        if not turn_id:
                            return
                        delta = event_data.get("delta", "") or ""
                        rbuf = _partial_reasoning.get(turn_id, "") + delta
                        _partial_reasoning[turn_id] = rbuf
                        # Pass the current text buffer too so the write never
                        # clobbers already-flushed partial text (reasoning
                        # usually precedes text, so the buffer is often "").
                        store.update_partial_response(
                            turn_id=turn_id,
                            response_text=_partial_text.get(turn_id, ""),
                            reasoning=rbuf,
                        )
                        return
                    if _type == "text_delta":
                        turn_id = event_data.get("turn_id")
                        if not turn_id:
                            return
                        delta = event_data.get("delta", "") or ""
                        offset = event_data.get("text_offset")
                        buf = _partial_text.get(turn_id, "")
                        # Offset-aware splice: contiguous deltas append; an
                        # overlapping/replayed delta rewrites in place rather
                        # than duplicating; a gap (offset past the buffer) falls
                        # back to append (best effort — finalize fixes the rest).
                        if isinstance(offset, int) and 0 <= offset <= len(buf):
                            buf = buf[:offset] + delta
                        else:
                            buf = buf + delta
                        _partial_text[turn_id] = buf
                        store.update_partial_response(turn_id=turn_id, response_text=buf)
                        return
                    if _type == "turn_complete":
                        tid = event_data.get("turn_id")
                        if tid:
                            _partial_text.pop(tid, None)
                            _partial_reasoning.pop(tid, None)
                        return
                    event_type = event_data.get("event", "")
                    if event_type == "tool_start":
                        store.save_tool_call(
                            turn_id=event_data.get("turn_id"),
                            bot_id=event_data.get("bot_id"),
                            user_id=event_data.get("user_id"),
                            call_id=event_data.get("call_id"),
                            tool_name=event_data.get("tool_name", "unknown"),
                            arguments=event_data.get("arguments"),
                            iteration=event_data.get("iteration", 1),
                            started_at=event_data.get("ts"),
                            text_offset=event_data.get("text_offset"),
                            tool_use_id=event_data.get("tool_use_id"),
                            parent_tool_use_id=event_data.get("parent_tool_use_id"),
                        )
                    elif event_type == "tool_end":
                        call_id = event_data.get("call_id")
                        if call_id:
                            store.update_tool_call_result(
                                call_id=call_id,
                                result=event_data.get("result", ""),
                                ended_at=event_data.get("ts"),
                                is_error=event_data.get("is_error"),
                            )
                        else:
                            # No call_id — save as complete record
                            store.save_tool_call(
                                turn_id=event_data.get("turn_id"),
                                bot_id=event_data.get("bot_id"),
                                user_id=event_data.get("user_id"),
                                call_id=None,
                                tool_name=event_data.get("tool_name", "unknown"),
                                result=event_data.get("result", ""),
                                iteration=event_data.get("iteration", 1),
                                ended_at=event_data.get("ts"),
                                is_error=event_data.get("is_error"),
                                tool_use_id=event_data.get("tool_use_id"),
                                parent_tool_use_id=event_data.get("parent_tool_use_id"),
                            )

                service._tool_persist_task = asyncio.create_task(
                    redis_subscriber.drain_tool_events(_tool_event_sink)
                )

                log.info(
                    "OpenClaw Redis subscriber started (redis=%s, sessions=%s)",
                    config.REDIS_URL,
                    list(session_to_bot.keys()),
                )
        except Exception:
            log.exception("Failed to start OpenClaw integration")
            redis_subscriber = None

    # Log startup with rich formatting
    log.startup(
        version=SERVICE_VERSION,
        host=config.SERVICE_HOST,
        port=config.SERVICE_PORT,
        models=service._available_models,
        default_model=service._default_model,
    )

    log.info(
        "Memory mode: %s (%s)",
        "mcp" if config.MCP_SERVER_URL else "embedded",
        config.MCP_SERVER_URL or "",
    )

    try:
        yield
    finally:
        # DEACTIVATED: history drain task (see above)
        # if history_drain_task:
        #     history_drain_task.cancel()
        #     try:
        #         await history_drain_task
        #     except (asyncio.CancelledError, Exception):
        #         pass
        # Cancel background tasks
        for task_attr in ("_group_cleanup_task", "_tool_persist_task"):
            task = getattr(service, task_attr, None)
            if task:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        if redis_subscriber:
            from ..agent_backends.agent_bridge import set_agent_subscriber
            set_agent_subscriber(None)
            await redis_subscriber.close()
        if scheduler:
            await scheduler.stop()
        await service.shutdown()
        set_service(None)


# Create FastAPI app
try:
    from fastapi import FastAPI

    from .routes import all_routers

    app = FastAPI(
        title="llm-bawt API",
        description="OpenAI-compatible API with integrated memory system",
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )

    for router in all_routers:
        app.include_router(router)

except ImportError:
    # FastAPI not installed - create stub
    app = None
    log.warning("FastAPI not installed. Install with: pip install fastapi uvicorn")


# =============================================================================
# CLI Entry Point
# =============================================================================

def _find_service_pid(port: int) -> int | None:
    """Find the PID of a process listening on the given port."""
    import subprocess
    try:
        # Use lsof to find the process listening on the port
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            # May return multiple PIDs (parent/child), get the first one
            pids = result.stdout.strip().split("\n")
            return int(pids[0])
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass
    return None


def _is_service_running(host: str, port: int) -> bool:
    """Check if the service is already running by attempting to connect."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            # Use 127.0.0.1 for 0.0.0.0 since we can't connect to 0.0.0.0
            check_host = "127.0.0.1" if host == "0.0.0.0" else host
            result = sock.connect_ex((check_host, port))
            return result == 0
    except (OSError, socket.error):
        return False


def _kill_service(port: int) -> bool:
    """Kill the service running on the given port. Returns True if successful."""
    import signal
    import os
    
    pid = _find_service_pid(port)
    if pid is None:
        return False
    
    try:
        # Send SIGTERM for graceful shutdown
        os.kill(pid, signal.SIGTERM)
        
        # Wait briefly for the process to terminate
        import time
        for _ in range(10):  # Wait up to 1 second
            time.sleep(0.1)
            try:
                os.kill(pid, 0)  # Check if process still exists
            except OSError:
                return True  # Process terminated
        
        # If still running, send SIGKILL
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.1)
        return True
    except OSError:
        return False


def main():
    """Entry point for the background service."""
    import argparse
    
    # Load config for defaults
    config = Config()
    
    parser = argparse.ArgumentParser(description="llm-bawt background service")
    parser.add_argument("--host", default=config.SERVICE_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=config.SERVICE_PORT, help="Port to listen on")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show additional detail (payloads, timing)")
    parser.add_argument("--debug", action="store_true", help="Enable low-level DEBUG messages (unformatted)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    parser.add_argument("--restart", action="store_true", help="Kill existing service and start a new one")
    parser.add_argument("--stop", action="store_true", help="Stop the running service and exit")
    args = parser.parse_args()
    
    # Setup logging with the new rich-formatted logger
    # Note: --verbose enables payload logging, --debug enables low-level DEBUG
    setup_service_logging(verbose=args.verbose, debug=args.debug)
    
    # Also update config.VERBOSE so BackgroundService can use it
    config.VERBOSE = args.verbose
    
    # Handle --stop: kill the service and exit
    if args.stop:
        if _is_service_running(args.host, args.port):
            print(f"Stopping service on port {args.port}...")
            if _kill_service(args.port):
                print("Service stopped.")
                return 0
            else:
                print("Failed to stop service.")
                return 1
        else:
            print(f"No service running on port {args.port}.")
            return 0
    
    # Check if service is already running
    if _is_service_running(args.host, args.port):
        if args.restart:
            print(f"Restarting service on port {args.port}...")
            if not _kill_service(args.port):
                print("Warning: Could not kill existing service, attempting to start anyway...")
            # Brief pause to ensure port is released
            import time
            time.sleep(0.5)
        else:
            print(f"Service is already running on port {args.port}.")
            print("Use --restart to restart the service, or --stop to stop it.")
            return 0
    
    if app is None:
        print("Error: FastAPI not installed. Install with: pip install fastapi uvicorn")
        return 1
    
    try:
        import uvicorn
        
        # Configure uvicorn log level
        # When using our rich logging, set uvicorn to warning to reduce noise
        uvicorn_log_level = "debug" if args.debug else "warning"
        
        # Exclude generated/data files from reload watching to prevent feedback loops
        reload_excludes = [
            "__pycache__", "*.pyc", ".git",
            ".logs", ".run", "models",
            "*.log", "*.pid",
        ] if args.reload else None
        
        uvicorn.run(
            "llm_bawt.service.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            reload_excludes=reload_excludes,
            log_level=uvicorn_log_level,
        )
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install uvicorn")
        return 1
