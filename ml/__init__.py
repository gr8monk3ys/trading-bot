"""
Machine Learning module for trading strategies.

Contains:
- LSTM-based price prediction
- DQN reinforcement learning agent for trading decisions

Usage:
    from ml import LSTMPredictor, DQNAgent

    # LSTM prediction
    predictor = LSTMPredictor()
    result = predictor.predict(prices)

    # DQN agent
    agent = DQNAgent(state_size=20)
    action = agent.select_action(state)
"""

# LSTM predictor (existing)
from ml.lstm_predictor import LSTMPredictor, PredictionResult, TrainingMetrics

# DQN reinforcement learning agent
from ml.rl_agent import (
    DQNAgent,
    DoubleDQNAgent,
    TradingAction,
    Experience,
    ReplayBuffer,
    DQNNetwork,
    DuelingDQNNetwork,
)

__all__ = [
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
    "DuelingDQNNetwork",
]
