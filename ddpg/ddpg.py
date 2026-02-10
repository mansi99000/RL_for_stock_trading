"""
Deep Deterministic Policy Gradient (DDPG) agent.

An off-policy actor-critic algorithm for continuous action spaces. Uses
experience replay, target networks with Polyak averaging, and Gaussian
exploration noise for stable training.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from networks import Actor, Critic


class DDPG:
    def __init__(self, state_dim, action_dim, min_action, max_action, device):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=2e-3)

        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_criterion = nn.MSELoss()

        self.min_action = min_action
        self.max_action = max_action

        self.device = device
        self.compile()

    def compile(self):
        if (
            self.device == "cuda"
            and torch.cuda.get_device_capability(0)[0] >= 7.1
        ):
            self.actor = torch.compile(self.actor)
            self.actor_target = torch.compile(self.actor_target)
            self.critic = torch.compile(self.critic)
            self.critic_target = torch.compile(self.critic_target)

        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

    def select_action(self, state, evaluate=False):
        self.actor.eval()
        state = torch.from_numpy(state).unsqueeze(0)
        state = state.to(self.device)
        action = self.actor(state)

        if not evaluate:
            action += torch.randn_like(action) * 0.1

        action = torch.clamp(action, self.min_action, self.max_action)
        self.actor.train()
        return action.detach().cpu().numpy()[0]

    def train(self, replay_buffer, batch_size=64, discount=0.99, tau=0.005):
        if len(replay_buffer) < batch_size:
            return

        # Sample replay buffer
        x, y, u, r, d = replay_buffer.sample(batch_size)
        state = torch.from_numpy(x).to(self.device)
        next_state = torch.from_numpy(y).to(self.device)
        action = torch.from_numpy(u).to(self.device)
        reward = torch.from_numpy(r).to(self.device)
        not_done = torch.from_numpy(1 - d).to(self.device)

        self.actor_target.eval()
        self.critic_target.eval()
        self.critic.eval()

        # Compute the target Q value
        target_Q = self.critic_target(
            next_state, self.actor_target(next_state)
        ).detach()
        target_Q = reward + (not_done * discount * target_Q)
        target_Q = target_Q.squeeze(1)

        # Get current Q estimate
        current_Q = self.critic(state, action)
        current_Q = current_Q.squeeze(1)

        # Compute critic loss
        critic_loss = self.critic_criterion(current_Q, target_Q)

        self.critic.train()
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        self.critic_optimizer.step()

        self.critic.eval()
        self.actor.train()
        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).squeeze(1).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
        self.actor_optimizer.step()

        # Update the target models with polyak averaging
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )  # soft update
        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    def save(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        torch.save(self.actor.state_dict(), os.path.join(log_dir, "actor.pth"))
        torch.save(
            self.actor_target.state_dict(),
            os.path.join(log_dir, "actor_target.pth"),
        )

        torch.save(
            self.critic.state_dict(), os.path.join(log_dir, "critic.pth")
        )
        torch.save(
            self.critic_target.state_dict(),
            os.path.join(log_dir, "critic_target.pth"),
        )

    def load(self, log_dir):
        self.actor.load_state_dict(
            torch.load(
                os.path.join(log_dir, "actor.pth"),
                map_location=torch.device("cpu"),
            )
        )
        self.actor_target.load_state_dict(
            torch.load(
                os.path.join(log_dir, "actor_target.pth"),
                map_location=torch.device("cpu"),
            )
        )

        self.critic.load_state_dict(
            torch.load(
                os.path.join(log_dir, "critic.pth"),
                map_location=torch.device("cpu"),
            )
        )
        self.critic_target.load_state_dict(
            torch.load(
                os.path.join(log_dir, "critic_target.pth"),
                map_location=torch.device("cpu"),
            )
        )


if __name__ == "__main__":
    import os

    import gym
    import numpy as np
    from memory import ReplayBuffer
    from tqdm import trange

    Agent = DDPG

    env = gym.make("Pendulum-v1")
    agent = Agent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        min_action=env.action_space.low[0],
        device="cuda",
    )

    print("state_dim", env.observation_space.shape[0])
    print("action_dim", env.action_space.shape[0])
    print("max_action", env.action_space.high[0])
    print("min_action", env.action_space.low[0])

    n_games = 250

    best_score = env.reward_range[0]
    score_history = []
    buffer = ReplayBuffer(
        100000, env.observation_space.shape[0], env.action_space.shape[0]
    )

    for i in (t := trange(n_games, leave=True)):
        observation = env.reset()[0]
        done = False
        score = 0
        while not done:
            action = agent.select_action(observation)
            observation_, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            buffer.store(observation, observation_, action, reward, done)
            agent.train(buffer)
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save("weights/")

        t.set_description(f"{score=}, {avg_score=}")
