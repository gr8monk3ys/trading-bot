"""
Machine Learning module for trading strategies.

Contains:
- LSTM-based price prediction
- DQN reinforcement learning agent for trading decisions
- Shared PyTorch utilities

Usage:
    from ml import LSTMPredictor, DQNAgent

    # LSTM prediction
    predictor = LSTMPredictor()
    result = predictor.predict(prices)

    # DQN agent
    agent = DQNAgent(state_size=20)
    action = agent.select_action(state)
"""

# Shared PyTorch utilities
# LSTM predictor (existing)
from ml.lstm_predictor import LSTMPredictor, PredictionResult, TrainingMetrics

# DQN reinforcement learning agent
from ml.rl_agent import (
    DoubleDQNAgent,
    DQNAgent,
    DQNNetwork,
    Experience,
    ReplayBuffer,
    TradingAction,
)
from ml.torch_utils import get_torch_device, import_torch

__all__ = [
    # Utilities
    "import_torch",
    "get_torch_device",
    # LSTM
    "LSTMPredictor",
    "PredictionResult",
    "TrainingMetrics",
    # DQN
    "DQNAgent",
    "DoubleDQNAgent",
    "TradingAction",
    "Experience",
    "ReplayBuffer",
    "DQNNetwork",
]
