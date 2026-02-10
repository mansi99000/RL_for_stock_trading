from dataclasses import dataclass

import numpy as np


@dataclass
class OUNoise:
    action_dim: int
    mu: float = 0
    theta: float = 0.15
    sigma: float = 0.2
    state: np.array = None
    reset_state: np.array = None

    def __post_init__(self):
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = (
            self.reset_state
            if self.reset_state is not None
            else np.ones(self.action_dim) * self.mu
        )

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
