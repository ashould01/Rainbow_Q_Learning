import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
import logging
from datetime import datetime

from train import Agent
from models.DQN import DQNetwork

os.makedirs('logs/', exist_ok = True)
save_path = f'logs/{datetime.now().strftime("%Y-%m-%d")}'
os.makedirs(save_path, exist_ok = True)

log_path = 'logs'
log_file = os.path.join(log_path, 'log.log')

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s %(levelname)s:%(message)s',
    datefmt = '%m/%d/%Y %I:%M:%S %p',
    handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    )
logging.info('First Logging Learning')

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--environment", required = True, choices = ['LunarLander'], help = "choose environment type")
parser.add_argument("--mode", default = 'train', choices = ['train', 'test'], help = "choose train/test type")
parser.add_argument("-m", "--model", required = True, choices = ['DQN'], help = 'choose model')
parser.add_argument("-o", "--optimizer", required = True, choices = ['SGD', 'ADAM'], help = 'choose optimizer')

args = parser.parse_args()

config_name = args.environment
mode = args.mode

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

if cfg.env.environment == "LunarLander-v3":
    env = gym.make(
        id = cfg.env.environment,
        continuous = False,
        gravity = cfg.env.gravity,
        enable_wind = cfg.env.enable_wind,
        render_mode = "rgb_array"
        )
    # env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length = cfg.train.n_episodes)
    
else:
    raise NotImplementedError # go to utils

input_dim = cfg.env.obs_dim
output_dim = cfg.env.action_dim

if args.model == "DQN":
    model = DQNetwork(input_dim, output_dim) 
else:
    raise NotImplementedError # go to utils

if args.optimizer == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), cfg.train.optimize.learning_rate)
elif args.optimizer == "ADAM":
    optimizer = torch.optim.Adam(model.parameters(), cfg.train.optimize.learning_rate)
else:
    raise NotImplementedError # go to utils

agent = Agent(
    env = env,
    model = model,
    optimizer = optimizer,
    cfg = cfg 
)

if mode == 'train':
    writer = SummaryWriter(log_dir = save_path)
    
    for episode in tqdm(range(cfg.train.n_episodes), desc = "Training"):
        obs, info = env.reset()
        done = False
        
        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # update the agent
            agent.update(obs, action, reward, terminated, next_obs, writer, episode)
            
            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()
    writer.close()
    model_save = torch.save(model.state_dict(), os.path.join(save_path, f"model_weights_{datetime.now().strftime('%H-%M-%S')}.pth"))
    
elif mode == 'test':
    
    # Environment that crashes RecordEpisodeStatistics
    
    if cfg.env.environment == "LunarLander-v3":
    
        model = torch.load('logs/2025-02-07/model_weights_17-18-05.pth', map_location = torch.device('cuda:0')) # argparse
        video_path = os.path.join(save_path, 'videos')
        os.makedirs(video_path, exist_ok = True)

        num_eval_episodes = 5

        env = RecordVideo(env, video_folder = os.path.join(video_path), name_prefix="eval",
                          episode_trigger=lambda x: True)

        for _ in range(num_eval_episodes):
            state, info = env.reset()
            done = False

            while not done:
                action = agent.get_action(state, test = True)
                next_state, reward, done, _, info = env.step(action)
                state = next_state

            env.close()
        
    else:
        raise NotImplementedError
            
else:
    raise ValueError('You take only train and test mode.')