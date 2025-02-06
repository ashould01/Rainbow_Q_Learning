import os
import numpy as np
import tqdm
import yaml
import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from train import Agent

config_name = "LunaLander"

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def load_config(config_name):
    with open(os.path.join('configs', config_name + '.yaml'), 'rb') as f:
        config_dict = yaml.safe_load(f)
    cfg = dict2namespace(config_dict)
    return cfg

cfg = load_config(config_name)
mode = 'train'

if cfg.env.environment == "LunaLander-v3":
    env = gym.make(
        env = cfg.env.environment,
        continuous = False,
        gravity = cfg.env.gravity,
        enable_wind = cfg.env.enable_wind,
        render_mode = "rgb_array")
    
else:
    raise NotImplementedError

agent = Agent(
    env=env,
    model = model,
    cfg = cfg
)

if mode == 'train':
    for episode in tqdm(range(cfg.train.n_episodes)):
        obs, info = env.reset()
        done = False

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # update the agent
            agent.update(obs, action, reward, terminated, next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()
        
if mode == 'test':
    for i in range(10):
        obs, info = env.reset()
        env = RecordVideo(env, f"video/Trial{i}")
        done = False

        # play one episode
        while not done:
            action = agent.get_action(state, test = True)
            next_state, reward, done, _, _ = env_test.step(action)
            state = next_state
        agent.decay_epsilon()
    