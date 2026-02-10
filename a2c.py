import os

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


def save_model(actor, critic, actor_path, critic_path):
    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    print(f"Saved actor model to {actor_path}")
    print(f"Saved critic model to {critic_path}")


def load_model(actor, critic, actor_path, critic_path):
    actor.load_state_dict(torch.load(actor_path, weights_only=True))
    critic.load_state_dict(torch.load(critic_path, weights_only=True))
    actor.eval()
    critic.eval()
    print(f"Loaded actor model from {actor_path}")
    print(f"Loaded critic model from {critic_path}")


# Actor Network
class Actor(nn.Module):
    def __init__(self, input_size, num_actions, use_softmax=True):
        super(Actor, self).__init__()
        self.use_softmax = use_softmax

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_actions)

        if self.use_softmax:
            self.activation = F.softmax
        else:
            self.activation = F.tanh

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.use_softmax:
            x = self.activation(self.fc2(x), dim=-1)
        else:
            x = self.activation(self.fc2(x))
        return x


# Critic Network
class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def actor_critic(
    actor,
    critic,
    episodes,
    env,
    max_steps=20,
    gamma=0.99,
    lr_actor=1e-3,
    lr_critic=1e-3,
):
    # set up optimizer for both actor and critic
    optimizer_actor = optim.AdamW(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.AdamW(critic.parameters(), lr=lr_critic)

    stats = {"Actor Loss": [], "Critic Loss": [], "Returns": []}

    for episode in range(1, episodes + 1):
        if env.__class__.__name__ == "MultiStockEnv":
            state = env.initialize()
        else:
            state = env.reset()[0]

        ep_return = 0
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            state_tensor = torch.FloatTensor(state)

            # Actor selects action
            action_probs = actor(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()

            # Take action and observe next state and reward
            if env.__class__.__name__ == "MultiStockEnv":
                next_state, reward, done = env.transition(
                    state, action_probs.detach().numpy()
                )
            else:
                next_state, reward, done, _, _ = env.step(action.item())

            # Critic estimates value function
            value = critic(state_tensor)
            next_value = critic(torch.FloatTensor(next_state))

            # Calculate TD target and Advantage
            td_target = reward + gamma * next_value * (1 - done)
            advantage = td_target - value

            # Critic update with MSE loss
            critic_loss = F.mse_loss(value, td_target.detach())
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            # Actor update
            log_prob = dist.log_prob(action)
            actor_loss = -log_prob * advantage.detach()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            # Update state, episode return, and step count
            state = next_state
            ep_return += reward
            step_count += 1

        # Record statistics
        stats["Actor Loss"].append(actor_loss.item())
        stats["Critic Loss"].append(critic_loss.item())
        stats["Returns"].append(ep_return)

        # Print episode statistics
        print(
            f"Episode {episode}: Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Return: {ep_return}, Steps: {step_count}"
        )
    if not env.__class__.__name__ == "MultiStockEnv":
        env.close()

    return stats


def cartpole():
    actor_path = "actor_model.pth"
    critic_path = "critic_model.pth"
    use_pretrained = True

    # Check if pretrained model exists
    if (
        use_pretrained
        and os.path.exists(actor_path)
        and os.path.exists(critic_path)
    ):
        print("Loading pretrained models...")
        actor = Actor(input_size=4, num_actions=2)
        critic = Critic(input_size=4)
        load_model(actor, critic, actor_path, critic_path)
    else:
        print("Training new models...")
        actor = Actor(input_size=4, num_actions=2)
        critic = Critic(input_size=4)
        episodes = 3000  # Number of episodes for training
        stats = actor_critic(actor, critic, episodes)
        save_model(actor, critic, actor_path, critic_path)

    # Test the trained agent in human mode
    env = gym.make("CartPole-v1")
    state = env.reset()[0]
    done = False
    total_reward = 0
    max_steps = 2000  # 2000  # Maximum steps per episode for testing

    print(env.reward_range)

    while not done:
        # env.render()
        state_tensor = torch.FloatTensor(state)
        action_probs = actor(state_tensor)
        action = torch.argmax(action_probs).item()
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        if total_reward >= max_steps:
            break

    print(f"Total reward in human mode: {total_reward}")
    env.close()


def stock_trading():
    import pandas as pd

    from mdp import MultiStockEnv, parse_args

    dummy_stock_data = {
        "AAPL": pd.DataFrame(
            {
                "high": [2, 3, 4, 5, 6],
                "low": [1, 2, 3, 4, 5],
                "close": [2, 3, 4, 5, 6],
            }
        ),
        "GOOGL": pd.DataFrame(
            {
                "high": [4, 5, 6, 7, 8],
                "low": [1, 2, 3, 4, 5],
                "close": [2, 3, 4, 5, 6],
            }
        ),
    }
    env = MultiStockEnv(parse_args(), dummy_stock_data)
    state = env.initialize()

    actor_path = f"{env.__class__.__name__}_actor_model.pth"
    critic_path = f"{env.__class__.__name__}_critic_model.pth"
    use_pretrained = False

    # Check if pretrained model exists
    if (
        use_pretrained
        and os.path.exists(actor_path)
        and os.path.exists(critic_path)
    ):
        print("Loading pretrained models...")
        actor = Actor(
            input_size=env.state_dim,
            num_actions=env.n_stock,
            use_softmax=False,
        )
        critic = Critic(input_size=env.state_dim)
        load_model(actor, critic, actor_path, critic_path)
    else:
        print("Training new models...")
        actor = Actor(
            input_size=env.state_dim,
            num_actions=env.n_stock,
            use_softmax=False,
        )
        critic = Critic(input_size=env.state_dim)
        episodes = 3000  # Number of episodes for training
        _ = actor_critic(actor, critic, episodes, env)
        save_model(actor, critic, actor_path, critic_path)

    # Test the trained agent in human mode
    state = env.initialize()
    done = False
    total_reward = 0
    max_steps = 2000  # 2000  # Maximum steps per episode for testing

    while not done:
        state_tensor = torch.FloatTensor(state)
        action_probs = actor(state_tensor)
        state, reward, done = env.transition(
            state, action_probs.detach().numpy()
        )
        total_reward += reward
        if total_reward >= max_steps:
            break

    print(f"Total reward in evaulate mode: {total_reward}")


if __name__ == "__main__":
    # cartpole()
    stock_trading()
