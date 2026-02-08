"""
Client for communicating with the background service.

The client gracefully handles the case where the service is unavailable,
allowing the main CLI to work standalone.

Uses httpx for async operations when available, falls back to urllib.
"""

import json
import logging
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tasks import Task, TaskResult, TaskStatus

logger = logging.getLogger(__name__)

# Check for optional httpx
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# Default socket path for Unix domain socket communication
DEFAULT_SOCKET_PATH = Path.home() / ".cache" / "llm-bawt" / "service.sock"
DEFAULT_HTTP_PORT = 8642  # Fallback to HTTP if Unix socket unavailable


@dataclass
class ServiceStatus:
    """Status of the background service."""
    available: bool
    version: str | None = None
    uptime_seconds: float | None = None
    tasks_processed: int = 0
    tasks_pending: int = 0
    models_loaded: list[str] | None = None
    current_model: str | None = None
    available_models: list[str] | None = None

    @property
    def healthy(self) -> bool:
        return self.available


class ServiceClient:
    """
    Client for communicating with the llm_bawt background service.
    
    Supports both Unix domain sockets (preferred for local) and HTTP.
    Gracefully handles service unavailability.
    """
    
    def __init__(
        self,
        socket_path: Path | str | None = None,
        http_url: str | None = None,
        timeout: float = 5.0,
    ):
        self.socket_path = Path(socket_path) if socket_path else DEFAULT_SOCKET_PATH
        self.http_url = http_url or f"http://localhost:{DEFAULT_HTTP_PORT}"
        self.timeout = timeout
        self._availability_timeout = 0.5  # Fast timeout for availability checks
        self._available: bool | None = None
        self._last_check: float = 0
        self._check_interval = 30.0  # Re-check availability every 30 seconds
        self.last_error: str | None = None  # Track last error for better user feedback
    
    def is_available(self, force_check: bool = False) -> bool:
        """Check if the background service is available."""
        now = time.time()
        
        # Use cached result if recent
        if not force_check and self._available is not None:
            if now - self._last_check < self._check_interval:
                return self._available
        
        self._available = self._check_availability()
        self._last_check = now
        return self._available
    
    def _check_availability(self) -> bool:
        """Actually check if service is reachable.
        
        Uses raw TCP socket connect which fails immediately on connection refused,
        avoiding HTTP timeout delays. Also supports HTTP health check for remote hosts.
        """
        # Try Unix socket first (Linux/macOS) - only for local connections
        if self.socket_path.exists() and self._is_local_url():
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.setblocking(False)
                sock.connect(str(self.socket_path))
                sock.close()
                logger.debug(f"Background service available via Unix socket: {self.socket_path}")
                return True
            except (socket.error, OSError, BlockingIOError) as e:
                logger.debug(f"Unix socket unavailable: {e}")
        
        # Parse host from http_url for TCP check
        host, port = self._parse_url_host_port()
        
        # Try raw TCP connect (fast - fails immediately if port not listening)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)  # 500ms timeout for remote hosts
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                logger.debug(f"Background service available at {host}:{port}")
                return True
            else:
                logger.debug(f"TCP {host}:{port} not listening (errno={result})")
        except Exception as e:
            logger.debug(f"TCP check failed for {host}:{port}: {e}")
        
        # For non-local hosts, also try HTTP health endpoint as fallback
        if not self._is_local_url():
            try:
                import urllib.request
                import urllib.error
                req = urllib.request.Request(f"{self.http_url}/health", method="GET")
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    if resp.status == 200:
                        logger.debug(f"Background service available via HTTP health check")
                        return True
            except Exception as e:
                logger.debug(f"HTTP health check failed: {e}")
        
        return False
    
    def _parse_url_host_port(self) -> tuple[str, int]:
        """Parse host and port from http_url."""
        url = self.http_url
        # Remove protocol prefix
        if "://" in url:
            url = url.split("://", 1)[1]
        # Remove path
        if "/" in url:
            url = url.split("/", 1)[0]
        # Parse host:port
        if ":" in url:
            host, port_str = url.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                port = DEFAULT_HTTP_PORT
        else:
            host = url
            port = DEFAULT_HTTP_PORT
        return host, port

    def _format_http_error(self, method: str, url: str, status: int | None = None, body: str | None = None) -> str:
        parts = [f"{method} {url}"]
        if status is not None:
            parts.append(f"HTTP {status}")
        if body:
            parts.append(body)
        return " | ".join(parts)
    
    def _is_local_url(self) -> bool:
        """Check if URL points to localhost."""
        host, _ = self._parse_url_host_port()
        return host in ("localhost", "127.0.0.1", "::1")
    
    def get_status(self, silent: bool = False) -> ServiceStatus:
        """Get detailed status of the background service."""
        if not self.is_available():
            return ServiceStatus(available=False)
        
        try:
            response = self._request("GET", "/status")
            return ServiceStatus(
                available=True,
                version=response.get("version"),
                uptime_seconds=response.get("uptime_seconds"),
                tasks_processed=response.get("tasks_processed", 0),
                tasks_pending=response.get("tasks_pending", 0),
                models_loaded=response.get("models_loaded"),
                current_model=response.get("current_model"),
                available_models=response.get("available_models"),
            )
        except Exception as e:
            if not silent:
                logger.warning(f"Failed to get service status: {e}")
            return ServiceStatus(available=False)
    
    def submit_task(self, task: Task) -> bool:
        """
        Submit a task to the background service.
        
        Returns True if task was accepted, False if service unavailable.
        This is fire-and-forget - use get_task_result for results.
        """
        if not self.is_available():
            logger.debug("Background service unavailable, task not submitted")
            return False
        
        try:
            self._request("POST", "/v1/tasks", data=task.to_dict())
            logger.debug(f"Submitted task {task.task_id} ({task.task_type.value})")
            return True
        except Exception as e:
            logger.warning(f"Failed to submit task: {e}")
            return False
    
    def get_task_result(self, task_id: str, wait: bool = False, timeout: float = 30.0) -> TaskResult | None:
        """
        Get the result of a submitted task.
        
        Args:
            task_id: The task ID to check
            wait: If True, block until result is ready (up to timeout)
            timeout: Maximum seconds to wait if wait=True
        
        Returns:
            TaskResult if available, None if not found or service unavailable
        """
        if not self.is_available():
            return None
        
        try:
            params = {"wait": str(wait).lower(), "timeout": str(timeout)}
            response = self._request("GET", f"/v1/tasks/{task_id}", params=params)
            
            if response:
                return TaskResult(
                    task_id=response["task_id"],
                    status=TaskStatus(response["status"]),
                    result=response.get("result"),
                    error=response.get("error"),
                    processing_time_ms=response.get("processing_time_ms", 0),
                )
        except Exception as e:
            logger.debug(f"Failed to get task result: {e}")
        
        return None
    
    def chat_completion(
        self,
        messages: list[dict],
        model: str | None = None,
        bot_id: str | None = None,
        user_id: str | None = None,
        stream: bool = False,
        **kwargs,
    ) -> dict | None:
        """
        Send a chat completion request to the background service.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model alias to use (uses service default if not specified)
            bot_id: Bot personality to use
            user_id: User ID for memory context
            stream: Whether to stream the response
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            The completion response dict, or None if service unavailable
        """
        if not self.is_available():
            return None
        
        payload = {
            "messages": messages,
            "stream": stream,
            **kwargs,
        }
        if model:
            payload["model"] = model
        if bot_id:
            payload["bot_id"] = bot_id
        if user_id:
            payload["user"] = user_id
        
        try:
            if stream:
                return self._stream_chat_completion(payload)
            else:
                response = self._request("POST", "/v1/chat/completions", data=payload)
                self.last_error = None  # Clear on success
                return response
        except Exception as e:
            if not self.last_error:
                self.last_error = str(e)
            logger.warning(f"Chat completion via service failed: {e}")
            return None
    
    def _stream_chat_completion(self, payload: dict):
        """Stream chat completion from the service.

        Yields content chunks. If the request fails, yields nothing and logs error.
        The first yielded item is a dict with metadata: {"model": "actual_model_used"}.
        Subsequent yields are content strings.
        """
        import urllib.request
        import urllib.error

        url = f"{self.http_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        body = json.dumps(payload).encode()

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                first_chunk = True
                # Use readline() instead of iterating the response directly.
                # `for line in resp` uses BufferedIOBase iteration which reads
                # in 8KB chunks before yielding, defeating streaming.
                while True:
                    raw_line = resp.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode().strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            # Service warning event (e.g. model fallback)
                            if chunk.get("object") == "service.warning":
                                yield {"warnings": chunk.get("warnings", []), "model": chunk.get("model")}
                                continue
                            # On first chunk, yield metadata with actual model used
                            if first_chunk:
                                actual_model = chunk.get("model")
                                if actual_model:
                                    yield {"model": actual_model}
                                first_chunk = False
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode()[:500]  # Read error body, limit size
            except Exception:
                pass
            self.last_error = self._format_http_error("POST", url, e.code, error_body)
            logger.warning(f"Service streaming failed: {self.last_error}")
            return
        except Exception as e:
            self.last_error = self._format_http_error("POST", url, None, str(e))
            logger.warning(f"Service streaming failed: {e}")
            return
    
    def chat_completion_full(
        self,
        messages: list[dict],
        model: str | None = None,
        bot_id: str | None = None,
        user_id: str | None = None,
        **kwargs,
    ) -> str | None:
        """
        Get a complete chat response as a string.
        
        Convenience method that handles streaming internally.
        
        Returns:
            The assistant's response content, or None if unavailable.
        """
        response = self.chat_completion(
            messages=messages,
            model=model,
            bot_id=bot_id,
            user_id=user_id,
            stream=False,
            **kwargs,
        )
        
        if response and "choices" in response:
            return response["choices"][0].get("message", {}).get("content")
        return None

    def _request(
        self,
        method: str,
        path: str,
        data: dict | None = None,
        params: dict | None = None,
    ) -> dict[str, Any]:
        """Make a request to the service."""
        import urllib.request
        import urllib.parse
        import urllib.error
        
        url = f"{self.http_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        
        headers = {"Content-Type": "application/json"}
        body = json.dumps(data).encode() if data else None
        
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode()[:500]
            except Exception:
                pass
            self.last_error = self._format_http_error(method, url, e.code, error_body)
            raise
        except Exception as e:
            self.last_error = self._format_http_error(method, url, None, str(e))
            raise
    
    async def _request_async(
        self,
        method: str,
        path: str,
        data: dict | None = None,
        params: dict | None = None,
    ) -> dict[str, Any]:
        """Make an async request to the service (requires httpx)."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx not available. Install with: pip install llm-bawt[service]")
        
        url = f"{self.http_url}{path}"
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if method == "GET":
                resp = await client.get(url, params=params)
            elif method == "POST":
                resp = await client.post(url, json=data, params=params)
            else:
                raise ValueError(f"Unsupported method: {method}")
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = e.response.text[:500] if e.response is not None else ""
                self.last_error = self._format_http_error(method, url, e.response.status_code if e.response else None, body)
                raise
            except Exception as e:
                self.last_error = self._format_http_error(method, url, None, str(e))
                raise
            return resp.json()
    
    async def submit_task_async(self, task: Task) -> bool:
        """
        Submit a task to the background service asynchronously.
        
        Returns True if task was accepted, False if service unavailable.
        Requires httpx to be installed.
        """
        if not self.is_available():
            logger.debug("Background service unavailable, task not submitted")
            return False
        
        if not HTTPX_AVAILABLE:
            # Fall back to sync method
            return self.submit_task(task)
        
        try:
            await self._request_async("POST", "/v1/tasks", data=task.to_dict())
            logger.debug(f"Submitted task {task.task_id} ({task.task_type.value})")
            return True
        except Exception as e:
            logger.warning(f"Failed to submit task: {e}")
            return False
    
    async def get_task_result_async(
        self, task_id: str, wait: bool = False, timeout: float = 30.0
    ) -> TaskResult | None:
        """
        Get the result of a submitted task asynchronously.
        
        Requires httpx to be installed.
        """
        if not self.is_available():
            return None
        
        if not HTTPX_AVAILABLE:
            # Fall back to sync method
            return self.get_task_result(task_id, wait=wait, timeout=timeout)
        
        try:
            params = {"wait": str(wait).lower(), "timeout": str(timeout)}
            response = await self._request_async("GET", f"/v1/tasks/{task_id}", params=params)
            
            if response:
                return TaskResult(
                    task_id=response["task_id"],
                    status=TaskStatus(response["status"]),
                    result=response.get("result"),
                    error=response.get("error"),
                    processing_time_ms=response.get("processing_time_ms", 0),
                )
        except Exception as e:
            logger.debug(f"Failed to get task result: {e}")
        
        return None

    def search_memories(
        self,
        query: str,
        bot_id: str = "nova",
        n_results: int = 5,
        min_relevance: float = 0.0,
    ) -> list[dict] | None:
        """
        Search memories using the service's semantic search.
        
        The service keeps the embedding model loaded, so this is fast.
        
        Args:
            query: Search query text
            bot_id: Bot to search memories for
            n_results: Maximum number of results
            min_relevance: Minimum relevance score (0.0-1.0)
        
        Returns:
            List of memory dicts, or None if service unavailable
        """
        if not self.is_available():
            return None
        
        try:
            response = self._request("POST", "/v1/memory/search", data={
                "query": query,
                "bot_id": bot_id,
                "n_results": n_results,
                "min_relevance": min_relevance,
            })
            
            if response and "results" in response:
                # Format results for compatibility with memory backend
                return [
                    {
                        "id": r["id"],
                        "document": r["content"],
                        "metadata": {
                            "tags": r.get("tags", ["misc"]),
                            "importance": r.get("importance", 0.5),
                            "source_message_ids": r.get("source_message_ids", []),
                        },
                        "relevance": r.get("relevance", 0.0),
                    }
                    for r in response["results"]
                ]
        except Exception as e:
            logger.warning(f"Memory search via service failed: {e}")
        
        return None

    def generate_embedding(self, text: str) -> list[float] | None:
        """
        Generate an embedding using the service's loaded model.

        Args:
            text: Text to generate embedding for

        Returns:
            Embedding vector, or None if service unavailable
        """
        if not self.is_available():
            return None

        try:
            response = self._request("POST", "/v1/memory/embedding", data={
                "text": text,
            })

            if response and "embedding" in response:
                return response["embedding"]
        except Exception as e:
            logger.warning(f"Embedding generation via service failed: {e}")

        return None

    def list_models(self) -> list[str] | None:
        """
        List available models from the service.

        Returns:
            List of model IDs, or None if service unavailable
        """
        if not self.is_available():
            return None

        try:
            response = self._request("GET", "/v1/models")
            if response and "data" in response:
                return [m.get("id") for m in response["data"] if m.get("id")]
        except Exception as e:
            logger.warning(f"List models via service failed: {e}")

        return None


# Singleton instance for easy access
_service_client: ServiceClient | None = None


def get_service_client() -> ServiceClient:
    """Get the global service client instance."""
    global _service_client
    if _service_client is None:
        _service_client = ServiceClient()
    return _service_client
