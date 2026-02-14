from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayBuffer:
    buffer_capacity: int
    state_dim: int
    action_dim: int

    _buffer_idx: int = 0
    _buffer: dict = None

    def __post_init__(self):
        self._buffer = {
            "state": np.zeros(
                (self.buffer_capacity, self.state_dim), dtype=np.float32
            ),
            "next_state": np.zeros(
                (self.buffer_capacity, self.state_dim), dtype=np.float32
            ),
            "action": np.zeros(
                (self.buffer_capacity, self.action_dim), dtype=np.float32
            ),
            "reward": np.zeros((self.buffer_capacity, 1), dtype=np.float32),
            "done": np.zeros((self.buffer_capacity, 1), dtype=bool),
        }

    def store(self, state, next_state, action, reward, done):
        idx = (
            self._buffer_idx % self.buffer_capacity
            if self._buffer_idx != 0
            else 0
        )

        self._buffer["state"][idx] = state
        self._buffer["next_state"][idx] = next_state
        self._buffer["action"][idx] = action
        self._buffer["reward"][idx] = reward
        self._buffer["done"][idx] = done

        self._buffer_idx += 1

    def sample(self, batch_size: int):
        max_idx = min(self._buffer_idx, self.buffer_capacity)
        idx = np.random.choice(max_idx, batch_size, replace=False)

        return (
            self._buffer["state"][idx],
            self._buffer["next_state"][idx],
            self._buffer["action"][idx],
            self._buffer["reward"][idx],
            self._buffer["done"][idx],
        )

    def __len__(self):
        return self._buffer_idx
