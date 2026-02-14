"""
Shared PyTorch utility functions for the ML module.

Provides lazy importing of PyTorch and consistent device selection
across all ML components (LSTM predictor, DQN agent, etc.).

Usage:
    from ml.torch_utils import import_torch, get_torch_device

    torch, nn, optim, F = import_torch()
    device = get_torch_device(use_gpu=True)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module-level globals for lazy-loaded PyTorch modules
_torch = None
_nn = None
_optim = None
_F = None


def import_torch():
    """
    Lazy import PyTorch modules.

    Imports torch, torch.nn, torch.optim, and torch.nn.functional,
    caching them as module-level globals so subsequent calls are free.

    Returns:
        Tuple of (torch, nn, optim, F) modules

    Raises:
        ImportError: If PyTorch is not installed
    """
    global _torch, _nn, _optim, _F
    if _torch is None:
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            import torch.optim as optim

            _torch = torch
            _nn = nn
            _optim = optim
            _F = F
            logger.debug("PyTorch imported successfully")
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for ML features. " "Install with: pip install torch"
            ) from e
    return _torch, _nn, _optim, _F


def get_torch_device(use_gpu: bool = False) -> Any:
    """
    Select the best available compute device.

    Fallback order: CUDA -> MPS (Apple Silicon) -> CPU

    Args:
        use_gpu: Whether to attempt GPU acceleration.
                 If False, always returns CPU device.

    Returns:
        torch.device for the selected compute backend
    """
    torch, _, _, _ = import_torch()

    if use_gpu and torch.cuda.is_available():
        logger.info("Using GPU (CUDA)")
        return torch.device("cuda")
    elif use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using GPU (MPS / Apple Silicon)")
        return torch.device("mps")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")
