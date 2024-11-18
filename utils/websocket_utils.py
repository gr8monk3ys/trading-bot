import asyncio
import functools
import logging
from typing import Any, Callable, Optional
from websockets.exceptions import WebSocketException

logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries: int = 3, initial_delay: float = 1.0, max_delay: float = 60.0):
    """
    Decorator that implements exponential backoff retry logic for async functions.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        initial_delay (float): Initial delay between retries in seconds
        max_delay (float): Maximum delay between retries in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"Failed after {max_retries} attempts. "
                            f"Last error: {str(e)}"
                        )
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, max_delay)
            
            raise last_exception
        return wrapper
    return decorator

async def safe_disconnect(ws_client: Any) -> None:
    """
    Safely disconnect a websocket client with proper error handling.
    
    Args:
        ws_client: The websocket client to disconnect
    """
    if not ws_client:
        return
        
    try:
        if hasattr(ws_client, 'disconnect'):
            await ws_client.disconnect()
        elif hasattr(ws_client, 'close'):
            await ws_client.close()
        else:
            logger.warning(
                f"Unknown websocket client type: {type(ws_client)}. "
                "Unable to disconnect."
            )
    except Exception as e:
        logger.error(f"Error during websocket disconnect: {e}")

class WebSocketConnection:
    """
    A context manager for handling websocket connections with automatic reconnection.
    """
    
    def __init__(
        self,
        url: str,
        on_message: Callable,
        on_error: Optional[Callable] = None,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0
    ):
        self.url = url
        self.on_message = on_message
        self.on_error = on_error or self._default_error_handler
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self._ws = None
        self._running = False
        self._task = None
    
    @staticmethod
    def _default_error_handler(error: Exception) -> None:
        """Default error handler that logs the error."""
        logger.error(f"WebSocket error: {error}")
    
    async def _connect(self) -> None:
        """Establish websocket connection with retry logic."""
        delay = self.initial_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                self._ws = await asyncio.wait_for(
                    asyncio.create_task(
                        asyncio.get_event_loop().create_connection(
                            lambda: self,
                            self.url,
                            ssl=True
                        )
                    ),
                    timeout=30
                )
                logger.info("WebSocket connection established")
                return
                
            except Exception as e:
                if attempt == self.max_retries:
                    logger.error(f"Failed to connect after {self.max_retries} attempts")
                    raise
                
                logger.warning(
                    f"Connection attempt {attempt + 1}/{self.max_retries} "
                    f"failed: {e}. Retrying in {delay:.1f} seconds..."
                )
                
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.max_delay)
    
    async def _handle_messages(self) -> None:
        """Handle incoming websocket messages."""
        while self._running:
            try:
                if not self._ws:
                    await self._connect()
                
                async for message in self._ws:
                    try:
                        await self.on_message(message)
                    except Exception as e:
                        self.on_error(e)
                        
            except WebSocketException as e:
                logger.error(f"WebSocket error: {e}")
                await self._reconnect()
                
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                self.on_error(e)
                await asyncio.sleep(1)
    
    async def _reconnect(self) -> None:
        """Attempt to reconnect to the websocket."""
        if self._ws:
            await safe_disconnect(self._ws)
            self._ws = None
        
        await self._connect()
    
    async def start(self) -> None:
        """Start the websocket connection."""
        if self._running:
            return
            
        self._running = True
        self._task = asyncio.create_task(self._handle_messages())
    
    async def stop(self) -> None:
        """Stop the websocket connection."""
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        if self._ws:
            await safe_disconnect(self._ws)
            self._ws = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

def create_websocket_client(
    url: str,
    on_message: Callable,
    on_error: Optional[Callable] = None,
    **kwargs
) -> WebSocketConnection:
    """
    Create a new websocket client with the specified handlers.
    
    Args:
        url (str): WebSocket URL to connect to
        on_message (callable): Callback for handling incoming messages
        on_error (callable, optional): Callback for handling errors
        **kwargs: Additional arguments to pass to WebSocketConnection
    
    Returns:
        WebSocketConnection: A new websocket client instance
    """
    return WebSocketConnection(
        url=url,
        on_message=on_message,
        on_error=on_error,
        **kwargs
    )
