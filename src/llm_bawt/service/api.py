"""FastAPI-based background service wiring for llm-bawt."""

from contextlib import asynccontextmanager

from ..utils.config import Config
from .background_service import BackgroundService, _ensure_memory_mcp_server
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
    _ensure_memory_mcp_server(config)

    # Load model definitions from DB and merge into config.
    # DB always takes priority; YAML is seeded to DB on first run if DB is empty.
    from ..runtime_settings import ModelDefinitionStore
    model_store = ModelDefinitionStore(config)
    if model_store.engine is not None:
        yaml_models = config.defined_models.get("models", {})
        if yaml_models and model_store.count() == 0:
            seeded = model_store.seed_from_yaml(yaml_models)
            log.info("Seeded %d model definitions from YAML to DB", seeded)
        db_models = model_store.to_config_dict()
        if db_models:
            config.merge_db_models(db_models)
            log.debug("Loaded %d model definitions from DB", len(db_models))

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
        "mcp" if getattr(config, "MEMORY_SERVER_URL", None) else "embedded",
        getattr(config, "MEMORY_SERVER_URL", ""),
    )

    try:
        yield
    finally:
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
