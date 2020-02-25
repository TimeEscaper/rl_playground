from common import AbstractEstimator
import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(ActorNetwork, self).__init__()
        self.linear_input_ = nn.Linear(n_states, n_hidden)
        self.linear_mu_ = nn.Linear(n_hidden, n_actions)
        self.linear_sigma_ = nn.Linear(n_hidden, n_actions)
        self.action_distribution_ = torch.distributions.Normal

    def forward(self, x):
        x = F.relu(self.linear_input_(x))
        mu = 2 * torch.tanh(self.linear_mu_(x))
        sigma = 0.5 * torch.sigmoid(self.linear_sigma_(x)) + 1e-5
        distribution = self.action_distribution_(mu.view(1, ).data, sigma.view(1, ).data)
        return distribution


class CriticNetwork(nn.Module):
    def __init__(self, n_states, n_hidden):
        super(CriticNetwork, self).__init__()
        self.linear_input_ = nn.Linear(n_states, n_hidden)
        self.linear_value_ = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = F.relu(self.linear_input_(x))
        value = self.linear_value_(x)
        return value


class A2CEstimator:
    def __init__(self, n_states, n_actions, n_hidden, scaler=None, gamma=0.5, learning_rate=0.001):
        self.actor_ = ActorNetwork(n_states, n_actions, n_hidden)
        self.critic_ = CriticNetwork(n_states, n_hidden)
        self.actor_optimizer_ = torch.optim.Adam(self.actor_.parameters(), learning_rate)
        self.critic_optimizer_ = torch.optim.Adam(self.critic_.parameters(), learning_rate)
        self.scaler_ = scaler
        self.gamma_ = gamma

    def get_returns_(self, rewards):
        returns = []
        Gt = 0
        pw = 0
        for reward in rewards[::-1]:
            Gt += self.gamma_ ** pw * reward
            pw += 1
            returns.append(Gt)
        returns = returns[::-1]
        returns = torch.tensor(returns, requires_grad=True)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns

    def update(self, rewards, log_probabilities, state_values):
        returns = self.get_returns_(rewards)
        policy_loss = 0
        value_loss = 0
        for log_probability, value, Gt in zip(log_probabilities, state_values, returns):
            advantage = Gt - value.item()
            policy_loss += -log_probability * advantage
            value_loss += F.smooth_l1_loss(value.squeeze(), Gt)
            #loss += policy_loss + value_loss
        self.actor_optimizer_.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor_optimizer_.step()
        self.critic_optimizer_.zero_grad()
        value_loss.backward(retain_graph=True)
        self.critic_optimizer_.step()

    def get_action(self, state):
        #self.actor_.training = False
        #self.critic_.training = False
        if self.scaler_ is not None:
            state = self.scaler_.transform([state])[0]
        distribution = self.actor_(torch.Tensor(state))
        state_value = self.critic_(torch.Tensor(state))
        action = distribution.sample().numpy()
        log_probability = distribution.log_prob(action[0])
        return action, log_probability, state_value


def run_scenario(env, estimator, n_episodes):
    total_rewards = []
    for episode in range(n_episodes):
        log_probabilities = []
        rewards = []
        state_values = []
        total_reward = 0
        state = env.reset()
        while True:
            action, log_probability, state_value = estimator.get_action(state)
            action = action.clip(env.action_space.low[0], env.action_space.high[0])
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            log_probabilities.append(log_probability)
            rewards.append(reward)
            state_values.append(state_value)
            if is_done:
                estimator.update(rewards, log_probabilities, state_values)
                print('Episode: {}, total reward: {}'.format(episode, total_reward))
                break
            state = new_state
        total_rewards.append(total_reward)
    return total_rewards


if __name__ == "__main__":
    env = gym.make('MountainCarContinuous-v0')
    #env = gym.make('Pendulum-v0')
    scaler = StandardScaler()
    scaler.fit(np.array([env.observation_space.sample() for x in range(10000)]))
    estimator = A2CEstimator(env.observation_space.shape[0], 1, 128, scaler)
    total_episodes_rewards = run_scenario(env, estimator, 200)
    plt.plot(total_episodes_rewards)
    plt.title("Total reward over episode")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.show()
