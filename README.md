# Reinforcement Learning for Stock Trading

This project implements and compares multiple reinforcement learning algorithms for automated stock trading. The environment models a multi-stock portfolio where an agent learns to allocate investments across stocks to maximize returns.

## Algorithms Implemented

- **A2C (Advantage Actor-Critic)** — An on-policy actor-critic method that uses the advantage function to reduce variance in policy gradient updates. (`a2c.py`)
- **DDPG (Deep Deterministic Policy Gradient)** — An off-policy algorithm for continuous action spaces, using experience replay and target networks for stable training. (`ddpg/`)
- **PILCO (Probabilistic Inference for Learning Control)** — A model-based RL approach that learns environment dynamics using Gaussian Processes and optimizes a linear policy via trajectory optimization. (`pilco/`)

## Project Structure

```
├── a2c.py              # A2C implementation with Actor-Critic networks
├── ddpg/
│   ├── ddpg.py         # DDPG agent with target networks and soft updates
│   ├── networks.py     # Actor and Critic neural network architectures
│   ├── memory.py       # Experience replay buffer
│   └── utils.py        # Ornstein-Uhlenbeck noise for exploration
├── pilco/
│   └── pilco.py        # PILCO with GP dynamics model and linear policy
├── mdp.py              # Multi-stock trading environment (MDP formulation)
├── helper.py           # Technical indicators: MACD, RSI, CCI, ADX
└── requirements.txt    # Python dependencies
```

## Environment

The `MultiStockEnv` in `mdp.py` models a stock trading environment as a Markov Decision Process:

- **State**: Current balance, shares owned per stock, stock prices, and optional technical indicators (MACD, RSI, CCI, ADX)
- **Action**: Fraction of maximum shares to buy/sell for each stock
- **Reward**: Change in portfolio value after executing a trade
- **Termination**: End of the historical price data or negative balance

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### A2C — Stock Trading

```bash
python a2c.py
```

By default, this trains the A2C agent on a dummy multi-stock environment. You can switch to the CartPole benchmark by editing the `__main__` block.

### DDPG — Pendulum (Continuous Control)

```bash
cd ddpg
python ddpg.py
```

Trains DDPG on the Pendulum-v1 environment as a continuous control benchmark.

### PILCO — Model-Based RL

```bash
cd pilco
python pilco.py
```

Runs PILCO with a synthetic dynamics function, demonstrating GP-based model learning and policy optimization.

## Technical Indicators

The `helper.py` module provides the following indicators that can be added to the state representation:

| Indicator | Description |
|-----------|-------------|
| **MACD**  | Moving Average Convergence Divergence — momentum indicator |
| **RSI**   | Relative Strength Index — measures speed of price changes |
| **CCI**   | Commodity Channel Index — identifies cyclical trends |
| **ADX**   | Average Directional Index — measures trend strength |

Enable them via command-line flags: `--use-macd`, `--use-rsi`, `--use-cci`, `--use-adx`.

## Requirements

- Python 3.9+
- PyTorch
- NumPy, Pandas
- Gymnasium (OpenAI Gym)
- GPy (for PILCO)
- SciPy (for PILCO optimization)
