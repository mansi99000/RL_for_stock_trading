"""
Multi-Stock Trading Environment formulated as a Markov Decision Process (MDP).

The agent manages a portfolio of stocks, deciding how many shares to buy/sell
at each time step. The state includes the current balance, shares owned,
stock prices, and optional technical indicators (MACD, RSI, CCI, ADX).
"""

import argparse
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd

import helper


class MDP(ABC):
    """Abstract base class for Markov Decision Process environments."""

    @abstractmethod
    def initialize(self): ...

    @abstractmethod
    def transition(self, state, action): ...


class MultiStockEnv(MDP):
    def __init__(self, args, data: Dict[str, pd.DataFrame]):
        """
        data: each stock contain high, low, close in a pandas dataframe
        """
        self.args = args
        self.stock_price_history = data
        self.n_stock = len(self.stock_price_history)
        self.n_step = len(
            self.stock_price_history[list(self.stock_price_history.keys())[0]]
        )

        self.stock_prices = None

        self.macd = None  # Moving Average Convergence Divergence
        self.rsi = None  # Relative Strength Index
        self.cci = None  # Commodity Channel Index
        self.adx = None  # Average Directional Index

        self._build()

        self.time_step = None

        self.balance = None

        self.stocks_owned = None

        self.state_dim: int = helper.calculate_state_dim(args, self.n_stock)
        self._reset()

    def _build(self):
        print("Building data...")
        if self.args.use_macd:
            self.macd = np.zeros((self.n_stock, self.n_step))
            for i, data in enumerate(self.stock_price_history.values()):
                self.macd[i, :] = helper.macd(data)

        if self.args.use_rsi:
            self.rsi = np.zeros((self.n_stock, self.n_step))
            for i, data in enumerate(self.stock_price_history.values()):
                self.rsi[i, :] = helper.rsi(data)

        if self.args.use_cci:
            self.cci = np.zeros((self.n_stock, self.n_step))
            for i, data in enumerate(self.stock_price_history.values()):
                self.cci[i, :] = helper.cci(data)

        if self.args.use_adx:
            self.adx = np.zeros((self.n_stock, self.n_step))
            for i, data in enumerate(self.stock_price_history.values()):
                self.adx[i, :] = helper.adx(data)

        self.stock_prices = np.zeros((self.n_stock, self.n_step))
        for i, data in enumerate(self.stock_price_history.values()):
            self.stock_prices[i, :] = data["close"]

    def _reset(self):
        self.time_step = 0
        self.balance = self.args.initial_investment
        self.stock_owned = np.zeros(self.n_stock)

    def _get_state_vector(self):
        state_vector = np.empty(self.state_dim)

        state_vector[0] = self.balance
        state_vector[1 : self.n_stock + 1] = self.stock_owned
        state_vector[self.n_stock + 1 : 2 * self.n_stock + 1] = (
            self.stock_prices[:, self.time_step]
        )

        i = 0
        if self.args.use_macd:
            state_vector[
                (2 + i) * self.n_stock + 1 : (3 + i) * self.n_stock + 1
            ] = self.macd[:, self.time_step]
            i += 1

        if self.args.use_rsi:
            state_vector[
                (2 + i) * self.n_stock + 1 : (3 + i) * self.n_stock + 1
            ] = self.rsi[:, self.time_step]
            i += 1

        if self.args.use_cci:
            state_vector[
                (2 + i) * self.n_stock + 1 : (3 + i) * self.n_stock + 1
            ] = self.cci[:, self.time_step]
            i += 1

        if self.args.use_adx:
            state_vector[
                (2 + i) * self.n_stock + 1 : (3 + i) * self.n_stock + 1
            ] = self.adx[:, self.time_step]

        return state_vector

    def _calculate_value(self, state: np.ndarray):
        return state[0] + state[1 : self.n_stock + 1].dot(
            self.stock_prices[:, self.time_step]
        )

    def _execute_trade(self, action: np.ndarray):
        assert (
            len(action) == self.n_stock
        ), f"{len(action)=} != {self.n_stock=}"

        stock_buy_cost = action.dot(self.stock_prices[:, self.time_step])
        transaction_cost = stock_buy_cost * self.args.fee_percentage

        self.balance -= stock_buy_cost + transaction_cost

        self.stock_owned += action

        self.time_step += 1

    def initialize(self):
        self._reset()
        return self._get_state_vector()

    def transition(self, state, action):
        self._execute_trade(action * self.args.hmax)

        # TODO: handle non-negative constraint
        if self.balance < 0:
            return state, -10000, True  # is this okay !?!

        next_state = self._get_state_vector()

        reward = self._calculate_value(next_state) - self._calculate_value(
            state
        )

        return next_state, reward, self.time_step == self.n_step - 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-macd", action="store_true", default=True)
    parser.add_argument("--use-rsi", action="store_true", default=False)
    parser.add_argument("--use-cci", action="store_true", default=False)
    parser.add_argument("--use-adx", action="store_true", default=False)
    parser.add_argument(
        "--hmax",
        type=int,
        default=100,
        help="Maximum number of shares to trade",
    )
    parser.add_argument(
        "--initial-investment",
        type=int,
        default=20000,
        help="Initial investment amount",
    )
    parser.add_argument(
        "--fee-percentage",
        type=float,
        default=0.001,
        help="Transaction fee percentage",
    )
    return parser.parse_args()
