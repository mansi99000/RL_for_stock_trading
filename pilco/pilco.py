import GPy  # Install with `pip install GPy`
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm


# Define environment dynamics (for example purposes)
def true_dynamics(state, action):
    """Simulate true environment dynamics."""
    next_state = (
        state + action + 0.1 * np.random.randn(*state.shape)
    )  # Add noise
    return next_state


# Define cost function (e.g., minimize distance to target)
def cost_function(state, target):
    """Quadratic cost to penalize deviation from target."""
    return np.sum((state - target) ** 2, axis=-1)


# Generate initial dataset
def generate_data(env, num_samples=10, state_dim=2, action_dim=1):
    """Generate initial random data for GP model training."""
    states = np.random.uniform(-1, 1, (num_samples, state_dim))
    actions = np.random.uniform(-1, 1, (num_samples, action_dim))
    next_states = np.array([env(s, a) for s, a in zip(states, actions)])
    return states, actions, next_states


# Define PILCO dynamics model using GPs
class PILCODynamicsModel:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.models = []

    def train(self, states, actions, next_states):
        inputs = np.hstack((states, actions))
        targets = next_states - states  # Model predicts state differences
        self.models = [
            GPy.models.GPRegression(inputs, targets[:, i : i + 1])
            for i in range(self.state_dim)
        ]
        for model in self.models:
            model.optimize(messages=False)

    def predict(self, state, action):
        """Predict next state mean and variance."""
        input_data = np.hstack((state, action))
        mean, var = zip(*[model.predict(input_data) for model in self.models])
        return np.hstack(mean), np.hstack(var)


# Define PILCO policy
class PILCOPolicy:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.weights = np.random.randn(state_dim, action_dim)
        self.bias = np.random.randn(action_dim)

    def predict(self, state):
        """Linear policy."""
        return state @ self.weights + self.bias

    def optimize(self, dynamics_model, initial_states, target, horizon=10):
        """Optimize policy to minimize expected cost."""

        def objective(params):
            self.weights = params[: self.state_dim * self.action_dim].reshape(
                self.state_dim, self.action_dim
            )
            self.bias = params[self.state_dim * self.action_dim :]

            cost = 0
            for state in initial_states:
                mean_state = np.array(state).reshape(1, -1)  # Ensure 2D shape
                for _ in range(horizon):
                    action = self.predict(mean_state)
                    mean_state, _ = dynamics_model.predict(mean_state, action)
                    cost += cost_function(mean_state, target)
            return cost

        params = np.hstack((self.weights.ravel(), self.bias))
        result = minimize(objective, params, method="L-BFGS-B")
        optimized_params = result.x
        self.weights = optimized_params[
            : self.state_dim * self.action_dim
        ].reshape(self.state_dim, self.action_dim)
        self.bias = optimized_params[self.state_dim * self.action_dim :]


if __name__ == "__main__":
    state_dim = 2
    action_dim = 1
    target = np.array([0.0, 0.0])  # Target state
    env = true_dynamics
    num_iterations = 5

    # Initial data
    states, actions, next_states = generate_data(env)
    dynamics_model = PILCODynamicsModel(state_dim, action_dim)
    policy = PILCOPolicy(state_dim, action_dim)

    for iteration in tqdm(range(num_iterations)):
        # Train GP model
        dynamics_model.train(states, actions, next_states)

        # Optimize policy
        policy.optimize(dynamics_model, states, target)

        # Apply policy and collect new data
        new_states = []
        new_actions = []
        new_next_states = []
        for state in states:
            action = policy.predict(state)
            next_state = env(state, action)
            new_states.append(state)
            new_actions.append(action)
            new_next_states.append(next_state)
        states = np.array(new_states)
        actions = np.array(new_actions)
        next_states = np.array(new_next_states)

    print("Policy optimized!")
