import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from collections import defaultdict


class Agent:
    def __init__(self, env, model, optimizer, cfg):        
        self.env = env
        self.device = cfg.train.device
        self.q_values = model.to(self.device)
        self.lr = cfg.train.optimize.learning_rate
        self.optimizer = optimizer
        self.discount_factor = cfg.agent.discount_factor
        self.epsilon = cfg.agent.start_epsilon
        self.start_epsilon, self.final_epsilon = cfg.agent.start_epsilon, cfg.agent.final_epsilon
        self.n_episodes = cfg.train.n_episodes
        self.epsilon_decay = cfg.agent.epsilon_decay
        self.final_epsilon = cfg.agent.final_epsilon
        self.training_error = []
        
    def get_action(self, obs, test = False) -> int:
        # epsilon-greedy strategy
        obs = torch.tensor(obs).to(self.device)
        actionresult = self.q_values(obs).detach().cpu().numpy()
        
        if test == False:
            if np.random.random() < self.epsilon:
                return self.env.action_space.sample()
            else:
                return int(np.argmax(actionresult))
        
        else:
            return int(np.argmax(actionresult))
        
    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
        writer,
        epoch
    ):

        self.optimizer.zero_grad()
        obs = torch.tensor(obs, requires_grad = True).to(self.device)
        next_obs = torch.tensor(next_obs, requires_grad = True).to(self.device)
        future_q_value = (not terminated) * torch.max(self.q_values(next_obs))
        temporal_difference = torch.tensor(
            reward + self.discount_factor * future_q_value - self.q_values(obs)[action]
        )
        lossftn = nn.MSELoss()
        loss = lossftn(temporal_difference, torch.zeros_like(temporal_difference))
        loss.requires_grad_(True)
        loss.backward()
        self.optimizer.step()
        writer.add_scalar('train_loss', loss, epoch)


    def decay_epsilon(self):
        if self.epsilon_decay == "linear":
            epsilon_decay_rate = (self.start_epsilon - self.final_epsilon) / (self.n_episodes / 2)
            self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay_rate)

        else:
            raise NotImplementedError
