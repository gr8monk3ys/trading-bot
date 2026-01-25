"""
Deep Q-Network (DQN) Reinforcement Learning agent for trading.

This module implements a DQN agent that learns optimal trading actions
(buy, sell, hold) through experience replay and target networks.

Key components:
- DQNAgent: Main agent class with epsilon-greedy exploration
- DQNNetwork: Neural network architecture for Q-value estimation
- ReplayBuffer: Experience storage for training
- TradingAction: Action enumeration (HOLD=0, BUY=1, SELL=2)

Usage:
    agent = DQNAgent(state_size=20, action_size=3)
    state = agent.build_state(prices, position, equity)
    action = agent.select_action(state)
    reward = agent.calculate_reward(action, price_change, position)
    agent.store_experience(state, action, reward, next_state, done)
    loss = agent.train_step()
"""
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import deque
import random
import os

# Lazy imports for PyTorch - only load when needed
_torch = None
_nn = None
_optim = None


def _import_torch():
    """
    Lazy import PyTorch modules.

    Returns:
        Tuple of (torch, nn, optim) modules
    """
    global _torch, _nn, _optim
    if _torch is None:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            _torch = torch
            _nn = nn
            _optim = optim
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for the DQN agent. "
                "Install with: pip install torch"
            ) from e
    return _torch, _nn, _optim


@dataclass
class TradingAction:
    """Trading action constants."""
    HOLD: int = 0
    BUY: int = 1
    SELL: int = 2

    @classmethod
    def to_string(cls, action: int) -> str:
        """Convert action integer to string."""
        return {0: "HOLD", 1: "BUY", 2: "SELL"}.get(action, "UNKNOWN")


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.

    Stores experiences (state, action, reward, next_state, done) tuples
    and allows random sampling for training batches.

    Args:
        capacity: Maximum number of experiences to store
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.buffer.clear()


class DQNNetwork:
    """
    Deep Q-Network architecture.

    Multi-layer perceptron with ReLU activations and dropout.

    Args:
        state_size: Dimension of input state vector
        action_size: Number of output actions (default: 3 for hold/buy/sell)
        hidden_sizes: List of hidden layer sizes
    """

    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        hidden_sizes: List[int] = None
    ):
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        torch, nn, _ = _import_torch()

        layers = []
        prev_size = state_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, action_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)


class DQNAgent:
    """
    Deep Q-Network agent for trading decisions.

    Implements DQN with:
    - Target network for stable training
    - Experience replay buffer
    - Epsilon-greedy exploration
    - Huber loss (SmoothL1Loss)

    Actions:
    - 0: HOLD - Maintain current position
    - 1: BUY - Enter or increase long position
    - 2: SELL - Exit or enter short position

    State features include:
    - Price features (normalized returns)
    - Volume features (relative to average)
    - Volatility metrics
    - Trend indicators
    - Current position information
    - Optional technical indicators (RSI, MACD)

    Args:
        state_size: Dimension of state vector
        action_size: Number of actions (default: 3)
        hidden_sizes: Hidden layer sizes for Q-network
        learning_rate: Optimizer learning rate
        gamma: Discount factor for future rewards
        epsilon_start: Initial exploration rate
        epsilon_end: Minimum exploration rate
        epsilon_decay: Exploration decay rate per step
        batch_size: Training batch size
        buffer_size: Replay buffer capacity
        target_update_freq: Steps between target network updates
        use_gpu: Whether to use GPU acceleration
        model_dir: Directory to save/load models

    Example:
        >>> agent = DQNAgent(state_size=20)
        >>> state = agent.build_state(prices, 0.0, 100000.0)
        >>> action = agent.select_action(state, training=True)
        >>> print(agent.get_action_name(action))
        'BUY'
    """

    def __init__(
        self,
        state_size: int = 20,
        action_size: int = 3,
        hidden_sizes: List[int] = None,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 10000,
        target_update_freq: int = 100,
        use_gpu: bool = False,
        model_dir: str = "models"
    ):
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.model_dir = model_dir
        self.logger = logging.getLogger(__name__)

        torch, nn, optim = _import_torch()

        # Device selection
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.logger.info("DQN Agent using GPU (CUDA)")
        elif use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.logger.info("DQN Agent using GPU (MPS)")
        else:
            self.device = torch.device("cpu")
            self.logger.info("DQN Agent using CPU")

        # Policy network (online network)
        self.policy_net = DQNNetwork(
            state_size, action_size, hidden_sizes
        ).model.to(self.device)

        # Target network (for stable Q-value estimates)
        self.target_net = DQNNetwork(
            state_size, action_size, hidden_sizes
        ).model.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.requires_grad_(False)  # Target network is never trained directly

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss for stability

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training state
        self.steps = 0
        self.episodes = 0
        self.total_reward = 0.0
        self.episode_rewards = deque(maxlen=1000)  # Keep last 1000 episodes

        # Create model directory
        os.makedirs(model_dir, exist_ok=True)

    def build_state(
        self,
        prices: List[Dict[str, Any]],
        position: float,
        account_equity: float,
        indicators: Dict[str, float] = None,
        unrealized_pnl: float = 0.0
    ) -> np.ndarray:
        """
        Build state vector from market data.

        Constructs a normalized state representation suitable for the
        neural network from raw market data.

        Args:
            prices: List of OHLCV bar dictionaries with keys:
                   'open', 'high', 'low', 'close', 'volume'
            position: Current position as fraction (-1 to 1, negative = short)
            account_equity: Current account equity value
            indicators: Optional dict with technical indicators:
                       'rsi', 'macd_hist', 'atr', etc.
            unrealized_pnl: Current unrealized P&L as decimal

        Returns:
            Normalized state vector of shape (state_size,)
        """
        if len(prices) < 10:
            self.logger.warning(
                f"Insufficient price data ({len(prices)} bars), "
                "returning zero state"
            )
            return np.zeros(self.state_size, dtype=np.float32)

        recent_prices = prices[-10:]

        # Extract close prices for calculations
        closes = np.array([p.get("close", p.get("c", 0)) for p in recent_prices])
        opens = np.array([p.get("open", p.get("o", 0)) for p in recent_prices])
        highs = np.array([p.get("high", p.get("h", 0)) for p in recent_prices])
        lows = np.array([p.get("low", p.get("l", 0)) for p in recent_prices])
        volumes = np.array([p.get("volume", p.get("v", 0)) for p in recent_prices])

        # Price features (normalized returns) - 5 features
        returns = np.diff(closes) / np.where(closes[:-1] != 0, closes[:-1], 1)
        recent_returns = returns[-5:] if len(returns) >= 5 else np.pad(
            returns, (5 - len(returns), 0), constant_values=0
        )

        # Volume features (relative to average) - 5 features
        vol_mean = np.mean(volumes) if np.mean(volumes) > 0 else 1.0
        vol_ratio = volumes[-5:] / vol_mean
        vol_ratio = np.clip(vol_ratio, 0, 10)  # Cap extreme values

        # Volatility (standard deviation of returns) - 1 feature
        volatility = np.std(returns) if len(returns) > 1 else 0.0
        volatility = np.clip(volatility, 0, 1)

        # Trend (short-term vs long-term average) - 1 feature
        sma_short = np.mean(closes[-5:])
        sma_long = np.mean(closes)
        trend = (sma_short / sma_long - 1) if sma_long > 0 else 0
        trend = np.clip(trend, -0.5, 0.5)

        # Price range features - 1 feature
        price_range = (closes[-1] - lows[-1]) / (highs[-1] - lows[-1] + 1e-8)
        price_range = np.clip(price_range, 0, 1)

        # Momentum (rate of change) - 1 feature
        if len(closes) >= 5 and closes[-5] > 0:
            momentum = (closes[-1] / closes[-5]) - 1
        else:
            momentum = 0.0
        momentum = np.clip(momentum, -0.5, 0.5)

        # Build state vector
        state = [
            *recent_returns,          # Recent returns (5)
            *vol_ratio,               # Volume ratios (5)
            volatility,               # Volatility (1)
            trend,                    # Trend (1)
            price_range,              # Price range position (1)
            momentum,                 # Momentum (1)
            position,                 # Current position (1)
            unrealized_pnl,           # Unrealized P&L (1)
        ]

        # Add technical indicators if available
        if indicators:
            # RSI (normalized to -0.5 to 0.5)
            rsi = indicators.get("rsi", 50)
            rsi_normalized = (rsi / 100) - 0.5
            state.append(np.clip(rsi_normalized, -0.5, 0.5))

            # MACD histogram (scaled)
            macd_hist = indicators.get("macd_hist", 0)
            macd_normalized = np.clip(macd_hist / 10, -1, 1)
            state.append(macd_normalized)

            # ATR if available (normalized)
            atr = indicators.get("atr", 0)
            if closes[-1] > 0:
                atr_pct = atr / closes[-1]
            else:
                atr_pct = 0
            state.append(np.clip(atr_pct, 0, 0.5))

        # Pad or truncate to state_size
        state = state[:self.state_size]
        while len(state) < self.state_size:
            state.append(0.0)

        return np.array(state, dtype=np.float32)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        During training, explores with probability epsilon.
        During inference, always exploits (greedy).

        Args:
            state: Current state vector
            training: Whether in training mode (enables exploration)

        Returns:
            Action index (0=HOLD, 1=BUY, 2=SELL)
        """
        torch, _, _ = _import_torch()

        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # Exploitation: select action with highest Q-value
        # Set network to eval mode to disable dropout for deterministic inference
        was_training = self.policy_net.training
        self.policy_net.eval()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax().item()

        # Restore training mode if it was set
        if was_training:
            self.policy_net.train()

        return action

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions given a state.

        Useful for debugging and analysis.

        Args:
            state: Current state vector

        Returns:
            Array of Q-values for each action
        """
        torch, _, _ = _import_torch()

        # Set to eval mode for deterministic output
        was_training = self.policy_net.training
        self.policy_net.eval()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            result = q_values.cpu().numpy()[0]

        if was_training:
            self.policy_net.train()

        return result

    def calculate_reward(
        self,
        action: int,
        price_change_pct: float,
        position: float,
        transaction_cost: float = 0.001,
        holding_cost: float = 0.0001
    ) -> float:
        """
        Calculate reward for an action.

        Reward is based on:
        - P&L from position and price movement
        - Transaction costs for trades
        - Small penalty for excessive trading
        - Risk-adjusted scaling

        Args:
            action: Action taken (0=HOLD, 1=BUY, 2=SELL)
            price_change_pct: Percentage price change (e.g., 0.01 for 1%)
            position: Position after action (-1 to 1)
            transaction_cost: Cost per trade as decimal
            holding_cost: Cost per period for holding (e.g., borrow cost)

        Returns:
            Reward value (typically in range -10 to +10)
        """
        # Base reward from position and price movement
        pnl = position * price_change_pct

        # Transaction cost penalty
        if action != TradingAction.HOLD:
            pnl -= transaction_cost

        # Holding cost (borrow fees, opportunity cost)
        if abs(position) > 0:
            pnl -= holding_cost * abs(position)

        # Scale reward to meaningful range
        reward = pnl * 100

        # Penalty for excessive trading (encourages meaningful trades)
        if action != TradingAction.HOLD:
            reward -= 0.01

        # Bonus for profitable trades
        if action == TradingAction.BUY and price_change_pct > 0:
            reward += 0.05
        elif action == TradingAction.SELL and price_change_pct < 0:
            reward += 0.05

        # Penalty for wrong direction
        if action == TradingAction.BUY and price_change_pct < -0.02:
            reward -= 0.1
        elif action == TradingAction.SELL and price_change_pct > 0.02:
            reward -= 0.1

        return float(reward)

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store experience in replay buffer.

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: State after action
            done: Whether episode ended
        """
        exp = Experience(state, action, reward, next_state, done)
        self.replay_buffer.push(exp)
        self.total_reward += reward

    def train_step(self) -> Optional[float]:
        """
        Perform one training step using experience replay.

        Samples a batch from the replay buffer and performs
        gradient descent on the Bellman error.

        Returns:
            Loss value, or None if insufficient samples
        """
        torch, _, _ = _import_torch()

        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(
            np.array([e.state for e in batch])
        ).to(self.device)
        actions = torch.LongTensor(
            [e.action for e in batch]
        ).to(self.device)
        rewards = torch.FloatTensor(
            [e.reward for e in batch]
        ).to(self.device)
        next_states = torch.FloatTensor(
            np.array([e.next_state for e in batch])
        ).to(self.device)
        dones = torch.FloatTensor(
            [1.0 if e.done else 0.0 for e in batch]
        ).to(self.device)

        # Current Q-values: Q(s, a)
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q-values: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss (Huber loss)
        loss = self.criterion(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update step counter
        self.steps += 1

        # Update target network periodically
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.logger.debug(f"Updated target network at step {self.steps}")

        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )

        return loss.item()

    def end_episode(self) -> Dict[str, float]:
        """
        End the current episode and return statistics.

        Returns:
            Dictionary with episode statistics
        """
        self.episodes += 1
        episode_reward = self.total_reward

        self.episode_rewards.append(episode_reward)
        self.total_reward = 0.0

        avg_reward = (
            np.mean(self.episode_rewards[-100:])
            if self.episode_rewards else 0.0
        )

        return {
            "episode": self.episodes,
            "episode_reward": episode_reward,
            "avg_reward_100": avg_reward,
            "epsilon": self.epsilon,
            "steps": self.steps,
            "buffer_size": len(self.replay_buffer)
        }

    def reset_epsilon(self, epsilon: float = None) -> None:
        """
        Reset exploration rate.

        Args:
            epsilon: New epsilon value (default: epsilon_start)
        """
        self.epsilon = epsilon if epsilon is not None else self.epsilon_start

    def get_action_name(self, action: int) -> str:
        """
        Get human-readable action name.

        Args:
            action: Action index

        Returns:
            Action name string
        """
        return TradingAction.to_string(action)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current agent statistics.

        Returns:
            Dictionary with agent stats
        """
        return {
            "steps": self.steps,
            "episodes": self.episodes,
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
            "device": str(self.device),
            "avg_reward_100": (
                np.mean(self.episode_rewards[-100:])
                if self.episode_rewards else 0.0
            )
        }

    def save(self, filename: str = "dqn_agent.pt") -> str:
        """
        Save agent to file.

        Saves model weights, optimizer state, and training progress.

        Args:
            filename: Name of save file

        Returns:
            Full path to saved file
        """
        torch, _, _ = _import_torch()

        path = os.path.join(self.model_dir, filename)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "episodes": self.episodes,
            "episode_rewards": self.episode_rewards,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "hidden_sizes": self.hidden_sizes,
        }, path)

        self.logger.info(f"Agent saved to {path}")
        return path

    def load(self, filename: str = "dqn_agent.pt") -> bool:
        """
        Load agent from file.

        Restores model weights, optimizer state, and training progress.

        Args:
            filename: Name of save file

        Returns:
            True if loaded successfully, False if file not found
        """
        torch, _, _ = _import_torch()

        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            self.logger.warning(f"No saved agent found at {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Verify architecture matches
            if checkpoint.get("state_size") != self.state_size:
                self.logger.error(
                    f"State size mismatch: saved={checkpoint.get('state_size')}, "
                    f"current={self.state_size}"
                )
                return False

            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
            self.steps = checkpoint.get("steps", 0)
            self.episodes = checkpoint.get("episodes", 0)
            # Convert loaded list to deque with maxlen
            loaded_rewards = checkpoint.get("episode_rewards", [])
            self.episode_rewards = deque(loaded_rewards, maxlen=1000)

            self.logger.info(
                f"Agent loaded from {path} "
                f"(episodes={self.episodes}, steps={self.steps})"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error loading agent: {e}")
            return False

    def set_inference_mode(self) -> None:
        """Set network to inference mode (disables dropout)."""
        self.policy_net.eval()

    def set_train_mode(self) -> None:
        """Set network to training mode (enables dropout)."""
        self.policy_net.train()


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN variant that reduces overestimation bias.

    Uses the policy network to select actions and the target network
    to assess them, reducing the upward bias in Q-value estimates.

    Reference: van Hasselt et al., "Deep Reinforcement Learning with
    Double Q-learning", AAAI 2016
    """

    def train_step(self) -> Optional[float]:
        """
        Perform one training step using Double DQN.

        The key difference from vanilla DQN:
        - Action selection: argmax_a Q_policy(s', a)
        - Action assessment: Q_target(s', argmax_a Q_policy(s', a))

        Returns:
            Loss value, or None if insufficient samples
        """
        torch, _, _ = _import_torch()

        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(
            np.array([e.state for e in batch])
        ).to(self.device)
        actions = torch.LongTensor(
            [e.action for e in batch]
        ).to(self.device)
        rewards = torch.FloatTensor(
            [e.reward for e in batch]
        ).to(self.device)
        next_states = torch.FloatTensor(
            np.array([e.next_state for e in batch])
        ).to(self.device)
        dones = torch.FloatTensor(
            [1.0 if e.done else 0.0 for e in batch]
        ).to(self.device)

        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Double DQN: Use policy net to select action, target net for assessment
        with torch.no_grad():
            # Select best action using policy network
            next_actions = self.policy_net(next_states).argmax(1)
            # Assess using target network
            next_q = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1

        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()


class DuelingDQNNetwork:
    """
    Dueling DQN network architecture.

    Separates state-value V(s) and advantage A(s,a) estimation,
    which can improve learning efficiency.

    Reference: Wang et al., "Dueling Network Architectures for
    Deep Reinforcement Learning", ICML 2016
    """

    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        hidden_sizes: List[int] = None
    ):
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        torch, nn, _ = _import_torch()

        # Shared feature layers
        feature_layers = []
        prev_size = state_size
        for hidden_size in hidden_sizes[:-1]:
            feature_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size

        self.features = nn.Sequential(*feature_layers)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], action_size)
        )

        # Combine into single module for state_dict compatibility
        self.model = nn.ModuleDict({
            'features': self.features,
            'value': self.value_stream,
            'advantage': self.advantage_stream
        })

    def forward(self, x):
        """
        Forward pass with dueling architecture.

        Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        """
        features = self.features(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
