import os
import torch
import torch.nn as nn
import gymnasium as gym
from collections import defaultdict

class Agent:
    def __init__(self, env, model, cfg):        
        self.env = env,
        self.q_values = model.to(device)
        self.optimizer_str = cfg.train.optimize.optimizer
        self.lr = cfg.train.optimize.learning_rate
        
        self.discount_factor = cfg.agent.discount_factor
        self.epsilon = cfg.agent.start_epsilon
        self.epsilon_decay = cfg.agent.epsilon_decay
        self.final_epsilon = cfg.agent.final_epsilon
        self.training_error = []
        
    def optimizer_setting(self):
        if self.optimizer_str == "SGD":
            self.optimizer = torch.optim.SGD(self.q_values.parameters(), self.lr)
        elif self.optimizer_str == "ADAM":
            self.optimizer = torch.optim.Adam(self.q_values.parameters(), self.lr)
        else:
            raise NotImplementedError
        
    def get_action(self, obs, test = False) -> int:
        # epsilon-greedy strategy
        obs = torch.tensor(obs).to(device)
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
    ):

        self.optimizer.zero_grad()
        obs = torch.tensor(obs, requires_grad = True).to(device)
        next_obs = torch.tensor(next_obs, requires_grad = True).to(device)
        future_q_value = (not terminated) * torch.max(self.q_values(next_obs))
        temporal_difference = torch.tensor(
            reward + self.discount_factor * future_q_value - self.q_values(obs)[action]
        )
        lossftn = nn.MSELoss()
        loss = lossftn(temporal_difference, torch.zeros_like(temporal_difference))
        loss.backward()
        optimizer.step()

        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)