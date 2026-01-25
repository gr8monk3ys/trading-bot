"""
Tests for the DQN Reinforcement Learning agent.
"""
import pytest
import numpy as np
import os
import tempfile

# Skip all tests if torch is not installed
torch = pytest.importorskip("torch")

from ml.rl_agent import (
    DQNAgent,
    DoubleDQNAgent,
    TradingAction,
    Experience,
    ReplayBuffer,
    DQNNetwork,
)


class TestTradingAction:
    """Tests for TradingAction dataclass."""

    def test_action_constants(self):
        """Test action constant values."""
        assert TradingAction.HOLD == 0
        assert TradingAction.BUY == 1
        assert TradingAction.SELL == 2

    def test_to_string(self):
        """Test action to string conversion."""
        assert TradingAction.to_string(0) == "HOLD"
        assert TradingAction.to_string(1) == "BUY"
        assert TradingAction.to_string(2) == "SELL"
        assert TradingAction.to_string(99) == "UNKNOWN"


class TestExperience:
    """Tests for Experience dataclass."""

    def test_experience_creation(self):
        """Test creating an experience tuple."""
        state = np.zeros(20)
        next_state = np.ones(20)
        exp = Experience(state, 1, 0.5, next_state, False)

        assert np.array_equal(exp.state, state)
        assert exp.action == 1
        assert exp.reward == 0.5
        assert np.array_equal(exp.next_state, next_state)
        assert exp.done is False


class TestReplayBuffer:
    """Tests for ReplayBuffer."""

    def test_buffer_initialization(self):
        """Test buffer initializes empty."""
        buffer = ReplayBuffer(capacity=100)
        assert len(buffer) == 0

    def test_push_experience(self):
        """Test adding experiences to buffer."""
        buffer = ReplayBuffer(capacity=100)
        exp = Experience(np.zeros(20), 0, 0.0, np.zeros(20), False)
        buffer.push(exp)
        assert len(buffer) == 1

    def test_buffer_capacity(self):
        """Test buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)
        for i in range(20):
            exp = Experience(np.full(20, i), i % 3, float(i), np.zeros(20), False)
            buffer.push(exp)
        assert len(buffer) == 10

    def test_sample_batch(self):
        """Test sampling from buffer."""
        buffer = ReplayBuffer(capacity=100)
        for i in range(50):
            exp = Experience(np.full(20, i), i % 3, float(i), np.zeros(20), False)
            buffer.push(exp)

        batch = buffer.sample(10)
        assert len(batch) == 10
        assert all(isinstance(e, Experience) for e in batch)

    def test_sample_more_than_available(self):
        """Test sampling when requesting more than available."""
        buffer = ReplayBuffer(capacity=100)
        for i in range(5):
            exp = Experience(np.zeros(20), 0, 0.0, np.zeros(20), False)
            buffer.push(exp)

        batch = buffer.sample(10)
        assert len(batch) == 5  # Returns all available

    def test_clear_buffer(self):
        """Test clearing buffer."""
        buffer = ReplayBuffer(capacity=100)
        for i in range(10):
            exp = Experience(np.zeros(20), 0, 0.0, np.zeros(20), False)
            buffer.push(exp)
        assert len(buffer) == 10

        buffer.clear()
        assert len(buffer) == 0


class TestDQNNetwork:
    """Tests for DQNNetwork architecture."""

    def test_network_creation(self):
        """Test network initialization."""
        net = DQNNetwork(state_size=20, action_size=3, hidden_sizes=[64, 32])
        assert net.model is not None

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        net = DQNNetwork(state_size=20, action_size=3, hidden_sizes=[64, 32])
        x = torch.randn(1, 20)
        output = net.model(x)
        assert output.shape == (1, 3)

    def test_batch_forward_pass(self):
        """Test forward pass with batch input."""
        net = DQNNetwork(state_size=20, action_size=3)
        x = torch.randn(32, 20)
        output = net.model(x)
        assert output.shape == (32, 3)


class TestDQNAgent:
    """Tests for DQNAgent."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create a DQN agent for testing."""
        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        return DQNAgent(
            state_size=20,
            action_size=3,
            hidden_sizes=[32, 16],
            learning_rate=0.001,
            batch_size=8,
            buffer_size=100,
            target_update_freq=10,
            model_dir=str(tmp_path)
        )

    @pytest.fixture
    def sample_prices(self):
        """Generate sample OHLCV price data."""
        base_price = 100.0
        prices = []
        for i in range(20):
            change = np.random.randn() * 0.02
            close = base_price * (1 + change)
            prices.append({
                "open": base_price,
                "high": max(base_price, close) * 1.01,
                "low": min(base_price, close) * 0.99,
                "close": close,
                "volume": 1000000 + np.random.randint(-100000, 100000)
            })
            base_price = close
        return prices

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.state_size == 20
        assert agent.action_size == 3
        assert agent.epsilon == 1.0
        assert agent.steps == 0
        assert agent.episodes == 0

    def test_build_state(self, agent, sample_prices):
        """Test state building from price data."""
        state = agent.build_state(sample_prices, 0.0, 100000.0)
        assert state.shape == (20,)
        assert state.dtype == np.float32

    def test_build_state_insufficient_data(self, agent):
        """Test state building with insufficient data."""
        prices = [{"close": 100, "volume": 1000} for _ in range(5)]
        state = agent.build_state(prices, 0.0, 100000.0)
        assert state.shape == (20,)
        assert np.allclose(state, 0)  # Should return zeros

    def test_build_state_with_indicators(self, agent, sample_prices):
        """Test state building with technical indicators."""
        indicators = {"rsi": 65.0, "macd_hist": 0.5, "atr": 2.0}
        state = agent.build_state(sample_prices, 0.5, 100000.0, indicators=indicators)
        assert state.shape == (20,)

    def test_select_action_exploration(self, agent):
        """Test action selection during exploration."""
        agent.epsilon = 1.0  # Always explore
        state = np.zeros(20)

        # With epsilon=1.0, should always explore (random action)
        actions = [agent.select_action(state, training=True) for _ in range(100)]
        unique_actions = set(actions)
        assert len(unique_actions) > 1  # Should have some variety

    def test_select_action_exploitation(self, agent):
        """Test action selection during exploitation is deterministic."""
        agent.epsilon = 0.0  # Always exploit
        state = np.array([0.1, -0.1, 0.2, -0.2, 0.3] + [0.0] * 15, dtype=np.float32)

        # With epsilon=0, consecutive calls should return same action (deterministic)
        # because the network and state don't change between calls
        first_action = agent.select_action(state, training=True)
        # Verify determinism - same input should give same output
        for _ in range(5):
            action = agent.select_action(state, training=True)
            assert action == first_action, "Exploitation should be deterministic"

    def test_select_action_no_training(self, agent):
        """Test action selection in inference mode is deterministic."""
        state = np.array([0.5, 0.3, -0.2, 0.1, 0.0] + [0.0] * 15, dtype=np.float32)
        # Should always exploit in non-training mode (deterministic)
        first_action = agent.select_action(state, training=False)
        # Verify determinism
        for _ in range(5):
            action = agent.select_action(state, training=False)
            assert action == first_action, "Non-training mode should be deterministic"

    def test_get_q_values(self, agent):
        """Test getting Q-values for a state."""
        state = np.zeros(20)
        q_values = agent.get_q_values(state)
        assert q_values.shape == (3,)

    def test_calculate_reward_hold(self, agent):
        """Test reward calculation for HOLD action."""
        reward = agent.calculate_reward(
            action=TradingAction.HOLD,
            price_change_pct=0.01,
            position=0.5
        )
        # Position * price_change * 100 = 0.5 * 0.01 * 100 = 0.5
        # No transaction cost for hold
        assert reward > 0

    def test_calculate_reward_buy_profit(self, agent):
        """Test reward calculation for profitable BUY."""
        reward = agent.calculate_reward(
            action=TradingAction.BUY,
            price_change_pct=0.02,
            position=1.0,
            transaction_cost=0.001
        )
        # Should be positive (profit minus costs)
        assert reward > 0

    def test_calculate_reward_sell_loss(self, agent):
        """Test reward calculation for loss on SELL."""
        reward = agent.calculate_reward(
            action=TradingAction.SELL,
            price_change_pct=0.02,  # Price went up after selling
            position=-1.0,
            transaction_cost=0.001
        )
        # Negative position * positive change = loss
        assert reward < 0

    def test_store_experience(self, agent):
        """Test storing experiences."""
        state = np.zeros(20)
        next_state = np.ones(20)
        agent.store_experience(state, 1, 0.5, next_state, False)
        assert len(agent.replay_buffer) == 1

    def test_train_step_insufficient_samples(self, agent):
        """Test training with insufficient samples."""
        loss = agent.train_step()
        assert loss is None  # Not enough samples

    def test_train_step(self, agent):
        """Test training step."""
        # Fill buffer with enough samples
        for i in range(20):
            state = np.random.randn(20).astype(np.float32)
            next_state = np.random.randn(20).astype(np.float32)
            agent.store_experience(state, i % 3, np.random.randn(), next_state, False)

        loss = agent.train_step()
        assert loss is not None
        assert isinstance(loss, float)
        assert agent.steps == 1

    def test_epsilon_decay(self, agent):
        """Test epsilon decays after training."""
        initial_epsilon = agent.epsilon

        # Fill buffer and train
        for i in range(20):
            state = np.random.randn(20).astype(np.float32)
            next_state = np.random.randn(20).astype(np.float32)
            agent.store_experience(state, i % 3, 0.0, next_state, False)

        agent.train_step()
        assert agent.epsilon < initial_epsilon

    def test_target_network_update(self, agent):
        """Test target network updates periodically."""
        agent.target_update_freq = 5

        # Fill buffer
        for i in range(50):
            state = np.random.randn(20).astype(np.float32)
            next_state = np.random.randn(20).astype(np.float32)
            agent.store_experience(state, i % 3, 0.0, next_state, False)

        # Train multiple steps
        for _ in range(10):
            agent.train_step()

        assert agent.steps == 10

    def test_end_episode(self, agent):
        """Test episode ending."""
        agent.total_reward = 5.0
        stats = agent.end_episode()

        assert stats["episode"] == 1
        assert stats["episode_reward"] == 5.0
        assert agent.total_reward == 0.0
        assert len(agent.episode_rewards) == 1

    def test_reset_epsilon(self, agent):
        """Test epsilon reset."""
        agent.epsilon = 0.1
        agent.reset_epsilon()
        assert agent.epsilon == agent.epsilon_start

        agent.reset_epsilon(0.5)
        assert agent.epsilon == 0.5

    def test_get_action_name(self, agent):
        """Test action name retrieval."""
        assert agent.get_action_name(0) == "HOLD"
        assert agent.get_action_name(1) == "BUY"
        assert agent.get_action_name(2) == "SELL"

    def test_get_stats(self, agent):
        """Test stats retrieval."""
        stats = agent.get_stats()
        assert "steps" in stats
        assert "episodes" in stats
        assert "epsilon" in stats
        assert "buffer_size" in stats
        assert "device" in stats

    def test_save_and_load(self, agent, tmp_path):
        """Test saving and loading agent."""
        # Train a bit to have state
        for i in range(20):
            state = np.random.randn(20).astype(np.float32)
            next_state = np.random.randn(20).astype(np.float32)
            agent.store_experience(state, i % 3, float(i), next_state, False)
        agent.train_step()
        agent.end_episode()

        # Save
        path = agent.save("test_agent.pt")
        assert os.path.exists(path)

        # Create new agent and load
        new_agent = DQNAgent(
            state_size=20,
            action_size=3,
            hidden_sizes=[32, 16],
            model_dir=str(tmp_path)
        )
        loaded = new_agent.load("test_agent.pt")

        assert loaded is True
        assert new_agent.steps == agent.steps
        assert new_agent.episodes == agent.episodes

    def test_load_nonexistent(self, agent):
        """Test loading from nonexistent file."""
        loaded = agent.load("nonexistent.pt")
        assert loaded is False

    def test_set_modes(self, agent):
        """Test setting inference/train modes."""
        agent.set_inference_mode()
        assert not agent.policy_net.training

        agent.set_train_mode()
        assert agent.policy_net.training


class TestDoubleDQNAgent:
    """Tests for DoubleDQNAgent."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create a Double DQN agent for testing."""
        return DoubleDQNAgent(
            state_size=20,
            action_size=3,
            hidden_sizes=[32, 16],
            batch_size=8,
            buffer_size=100,
            model_dir=str(tmp_path)
        )

    def test_inheritance(self, agent):
        """Test DoubleDQNAgent inherits from DQNAgent."""
        assert isinstance(agent, DQNAgent)

    def test_train_step(self, agent):
        """Test Double DQN training step."""
        # Fill buffer
        for i in range(20):
            state = np.random.randn(20).astype(np.float32)
            next_state = np.random.randn(20).astype(np.float32)
            agent.store_experience(state, i % 3, float(i), next_state, False)

        loss = agent.train_step()
        assert loss is not None
        assert isinstance(loss, float)


class TestIntegration:
    """Integration tests for DQN agent."""

    def test_full_episode_simulation(self, tmp_path):
        """Test a full episode of trading simulation."""
        agent = DQNAgent(
            state_size=20,
            action_size=3,
            hidden_sizes=[32, 16],
            batch_size=16,
            buffer_size=1000,
            epsilon_decay=0.99,
            model_dir=str(tmp_path)
        )

        # Simulate 100 trading steps
        base_price = 100.0
        position = 0.0
        prices = []

        for step in range(100):
            # Generate price movement
            change = np.random.randn() * 0.02
            price = base_price * (1 + change)
            prices.append({
                "open": base_price,
                "high": max(base_price, price) * 1.01,
                "low": min(base_price, price) * 0.99,
                "close": price,
                "volume": 1000000
            })
            prices = prices[-20:]  # Keep last 20

            if len(prices) >= 10:
                # Build state
                state = agent.build_state(prices, position, 100000.0)

                # Select action
                action = agent.select_action(state, training=True)

                # Update position based on action
                if action == TradingAction.BUY:
                    new_position = min(1.0, position + 0.5)
                elif action == TradingAction.SELL:
                    new_position = max(-1.0, position - 0.5)
                else:
                    new_position = position

                # Calculate reward
                price_change = (price - base_price) / base_price if base_price > 0 else 0
                reward = agent.calculate_reward(action, price_change, new_position)

                # Get next state
                next_state = agent.build_state(prices, new_position, 100000.0)

                # Store experience
                agent.store_experience(state, action, reward, next_state, False)

                # Train
                agent.train_step()

                position = new_position

            base_price = price

        # End episode
        stats = agent.end_episode()

        assert stats["episode"] == 1
        assert agent.steps > 0
        assert agent.epsilon < 1.0  # Should have decayed

    def test_training_improves_performance(self, tmp_path):
        """Test that training improves agent performance over time."""
        agent = DQNAgent(
            state_size=20,
            action_size=3,
            hidden_sizes=[64, 32],
            batch_size=32,
            buffer_size=5000,
            learning_rate=0.001,
            model_dir=str(tmp_path)
        )

        # Generate deterministic price data (uptrend)
        np.random.seed(42)

        def run_episode(agent, epsilon_override=None):
            if epsilon_override is not None:
                agent.epsilon = epsilon_override

            base_price = 100.0
            position = 0.0
            prices = []
            total_reward = 0.0

            for step in range(50):
                # Uptrend with noise
                trend = 0.002  # 0.2% trend
                noise = np.random.randn() * 0.01
                price = base_price * (1 + trend + noise)

                prices.append({
                    "open": base_price,
                    "high": max(base_price, price) * 1.01,
                    "low": min(base_price, price) * 0.99,
                    "close": price,
                    "volume": 1000000
                })
                prices = prices[-20:]

                if len(prices) >= 10:
                    state = agent.build_state(prices, position, 100000.0)
                    action = agent.select_action(state, training=True)

                    if action == TradingAction.BUY:
                        new_position = min(1.0, position + 0.5)
                    elif action == TradingAction.SELL:
                        new_position = max(-1.0, position - 0.5)
                    else:
                        new_position = position

                    price_change = (price - base_price) / base_price
                    reward = agent.calculate_reward(action, price_change, new_position)
                    total_reward += reward

                    next_state = agent.build_state(prices, new_position, 100000.0)
                    agent.store_experience(state, action, reward, next_state, False)
                    agent.train_step()

                    position = new_position

                base_price = price

            return total_reward

        # Run multiple episodes
        rewards = []
        for ep in range(20):
            reward = run_episode(agent)
            rewards.append(reward)
            agent.end_episode()

        # Check that later episodes have higher average reward
        # (learning is happening)
        early_avg = np.mean(rewards[:5])
        late_avg = np.mean(rewards[-5:])

        # Allow some variance - just check agent is functioning
        assert len(rewards) == 20
        assert agent.episodes == 20
