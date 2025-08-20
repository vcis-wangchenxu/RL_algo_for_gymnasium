# """-----------------------------------------
# main.py
# -----------------------------------------"""

# import sys, os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# curr_path = os.path.dirname(os.path.abspath(__file__))
# pare_path = os.path.dirname(curr_path)
# sys.path.append(pare_path)

# import gymnasium as gym
# import minigrid
# import wandb
# import yaml
# import argparse
# from pathlib import Path

# from common.utils import all_seed

# class Main:
#     def __init__(self) -> None:
#         self.general_cfg = None
#         self.sweep_cfg   = None

#     def load_config(self):
#         parser = argparse.ArgumentParser(description="Reinforcement Learning with PyTorch and WandB")
#         parser.add_argument('--config', type=str, default='configs/DQN_Empty-5x5-v0.yaml', 
#                             help='Path to the configuration YAML file')
#         args = parser.parse_args()

#         try:
#             with open(args.config, 'r') as f:
#                 config = yaml.safe_load(f)
#             print("Configuration loaded successfully:")
#         except FileNotFoundError:
#             print(f"Error: Configuration file '{args.config}' not found.")
#             return
#         except Exception as e:
#             print(f"Error loading configuration file: {e}")
#             return
#         self.general_cfg, self.sweep_cfg = config['general_config'], config['sweep_config']
    
#     def create_dirs(self):
#         # curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time
#         task_dir = f"{curr_path}/tasks/{self.general_cfg['mode'].capitalize()}_{self.general_cfg['env_name']}_{self.general_cfg['algo_name']}"
#         self.general_cfg.setdefault('task_dir', task_dir)
#         Path(self.general_cfg['task_dir']).mkdir(parents=True, exist_ok=True)
#         model_dir = f"{task_dir}/models"
#         self.general_cfg.setdefault('model_dir', model_dir)
#         traj_dir = f"{task_dir}/traj"
#         self.general_cfg.setdefault('traj_dir', traj_dir)

#     def envs_config(self, cfg, is_eval=False):
#         ''' configure environment
#         '''
#         env = gym.make(cfg['env_name'], render_mode=cfg['env_render'])
#         env = minigrid.wrappers.RGBImgPartialObsWrapper(env) 

#         if 'state_shape' not in cfg:
#             temp_env = gym.make(cfg['env_name'])
#             temp_env = minigrid.wrappers.RGBImgPartialObsWrapper(temp_env)
#             reset_result = temp_env.reset()

#             state_shape = reset_result[0]['image'].shape
#             action_dim  = temp_env.action_space.n

#             # update to cfg paramters
#             cfg.update({
#                 'state_shape': state_shape,
#                 'action_dim' : action_dim
#                 })
#             temp_env.close()
#         return env

#     def run_sweep_agent(self):
#         wandb.init()
#         wandb.config.update(self.general_cfg)
#         cfg = wandb.config

#         # --- Set random seed ---
#         if 'seed' in cfg:
#             all_seed(cfg['seed'])
#             print(f"Random seed set to {cfg['seed']}")

#         # --- Definie ENV ---
#         train_env = self.envs_config(cfg)
#         eval_env  = self.envs_config(cfg, is_eval=True)

#         # --- Load agent, trainer ---
#         agent_mod = __import__(f"algos.{cfg.algo_name}.agent", fromlist=['Agent'])
#         agent = agent_mod.Agent(cfg)  # create agent
#         trainer_mod = __import__(f"algos.{cfg.algo_name}.trainer", fromlist=['Trainer'])
#         trainer = trainer_mod.Trainer(cfg)  # create trainer

#         # --- training ---
#         agent = trainer.train(agent, train_env, eval_env)

# if __name__ == "__main__":
#     main = Main()
#     main.load_config()    # load YAML file
#     main.create_dirs()    # make model dir

#     sweep_id = wandb.sweep(main.sweep_cfg, project=main.general_cfg['project_name'])
#     count = 10
#     print(f"Start WandB Agent, which will execute {count} experiments...")
#     wandb.agent(sweep_id, function=main.run_sweep_agent, count=count)

import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random
import time
import os
import wandb
import math
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt

# --- 1. MODIFIED CNN Q-Network ---
class CNN_QNetwork(nn.Module):
    def __init__(self, input_shape, output_size):
        super(CNN_QNetwork, self).__init__()
        c, h, w = input_shape

        self.conv1 = nn.Conv2d(c, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)

        conv_out_size = self._get_conv_out(input_shape)

        self.fc1 = nn.Linear(conv_out_size, 256)
        self.fc2 = nn.Linear(256, output_size)
    
    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        return int(np.prod(o.size()))

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --- 2. Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity) -> None:
        self.buffer = deque(maxlen=capacity)
        
    def push(self, transitions):
        self.buffer.append(transitions)
    
    def sample(self, batch_size, sequential: bool=False):
        if sequential:
            rand = random.randint(0, len(self.buffer)-batch_size)
            batch = [self.buffer[i] for i in range(rand, rand+batch_size)]
            return zip(*batch)
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)
    
    def __len__(self):
        return len(self.buffer)

# --- 3. MODIFIED: DQN Agent ---
class DQNAgent:
    def __init__(self, state_shape, action_size, config):
        self.lr = config["lr"]
        self.gamma = config["gamma"]
        self.device = config["device"]
        self.batch_size = config["batch_size"]
        self.action_size = action_size
        
        # e-greedy parameters
        self.sample_count = 0
        self.epsilon_start = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay_steps = config["epsilon_decay"]

        self.q_network = CNN_QNetwork(state_shape, action_size).to(self.device)
        self.target_network = CNN_QNetwork(state_shape, action_size).to(self.device)
        # Copy the weights of the policy network to the target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval() # The target network is used only for inference, not for training
        # gradient
        wandb.watch(self.q_network, log="all", log_freq=100)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(config["buffer_size"])

    def sample_action(self, state, evaluate=False):
        # epsilon = end + (start - end) * exp(-1. * steps_done / decay)
        if not evaluate:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                            math.exp(-1. * self.sample_count / self.epsilon_decay_steps)
            
            self.sample_count += 1
        
        # Exploration vs. Exploitation
        self.q_network.eval()
        if not evaluate and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                action = self.q_network(state).argmax(dim=1).item()

        return action
    
    def update_policy(self):
        """ Execute one training iteration """
        self.q_network.train()

        if len(self.memory) < self.batch_size:
            return 
        
        b_s, b_a, b_r, b_ns, b_d = self.memory.sample(self.batch_size)
        # transition -> torch.tensor
        states  = torch.tensor(np.array(b_s), device=self.device, dtype=torch.float32)
        actions = torch.tensor(np.array(b_a), device=self.device, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(np.array(b_r), device=self.device, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(b_ns), device=self.device, dtype=torch.float32)
        dones   = torch.tensor(np.array(b_d), device=self.device, dtype=torch.float32).unsqueeze(1)

        q_values = self.q_network(states).gather(dim=1, index=actions)
        max_next_q_values = self.target_network(next_states).max(dim=1)[0].detach().unsqueeze(1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = nn.MSELoss()(q_values, q_targets)

        self.optimizer.zero_grad()
        dqn_loss.backward()
        # clip to avoid gradient explosion
        for param in self.q_network.parameters():  
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return dqn_loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, fpath):
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.target_network.state_dict(), f"{fpath}/DQN_checkpoint.pt")

    def load_model(self, fpath):
        self.target_network.load_state_dict(torch.load(f"{fpath}/DQN_checkpoint.pt",
                                                   map_location=self.device,
                                                   weights_only=True))
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            param.data.copy_(target_param.data)
        
# --- 4. TRAIN One Episode --- 
class Trainer:
    def __init__(self, config) -> None:
        self.device = config['device']
        self.train_steps = config['train_steps']
        self.seed = config['seed']
        self.model_path = config['model_dir']
        self.target_update_freq = config['target_update_freq']
        self.eval_freq = config['eval_freq']
        self.eval_eps = config['eval_eps']
        self.total_steps = 0
    
    def train(self, agent, train_env, eval_env):
        # --- WandB SetUp: Define Metries and Custom X-Axes ---
        wandb.define_metric("total_steps")
        wandb.define_metric("train/loss", step_metric="total_steps")
        wandb.define_metric("train/epsilon", step_metric="total_steps")
        wandb.define_metric("train/episode_reward", step_metric="total_steps")
        wandb.define_metric("train/episode_steps", step_metric="total_steps")
        wandb.define_metric("episode_count", step_metric="total_steps")

        wandb.define_metric("val/reward", step_metric="total_steps")
        wandb.define_metric("val/time", step_metric="total_steps")
        # -------------------------------------------------------------------

        start_time = time.time()
        best_eval_reward = -float('inf')
        episode_count = 0
        current_episode_reward = 0
        current_episode_steps = 0

        state, _ = train_env.reset(seed=self.seed)
        state = preprocess_state(state)

        print(f"\nStarting DQN Training for {self.train_steps} steps...")
        while self.total_steps < self.train_steps:
            action = agent.sample_action(state, evaluate=False)
            next_state, reward, terminated, truncated, info = train_env.step(action)
            next_state = preprocess_state(next_state)
            done = terminated or truncated

            # --- Shaping Reward ---
            step_penalty = -0.1
            modified_reward = reward + step_penalty
            if done and reward > 0:
                modified_reward = reward

            transition = (state, 
                          action, 
                          modified_reward, 
                          next_state, 
                          done)
            agent.memory.push(transition)

            # --- Update the state and counter ---
            state = next_state
            current_episode_reward += reward
            current_episode_steps  += 1
            self.total_steps += 1

            # --- Update Q_net and Record the loss---
            loss = agent.update_policy()
            wandb.log({
                "total_steps": self.total_steps,
                "train/loss": loss,
                "train/epsilon": agent.epsilon
            }, step=self.total_steps)

            if done:
                episode_count += 1
                print(f"Step: {self.total_steps}/{self.train_steps} | Ep: {episode_count} finished | "
                      f"EpRew: {current_episode_reward:.2f} | EpSteps: {current_episode_steps}")
                
                wandb.log({
                    "total_steps": self.total_steps,
                    "train/episode_reward": current_episode_reward,
                    "train/episode_steps": current_episode_steps, 
                    "episode_count": episode_count
                }, step=self.total_steps)

                # --- Reset environment and episode counter ---
                state, _ = train_env.reset(seed=self.seed + episode_count)
                state = preprocess_state(state)
                current_episode_reward = 0
                current_episode_steps  = 0

            # --- Update target_net ---
            if self.total_steps % self.target_update_freq == 0:
                agent.update_target_network()

            # --- Check if evaluation is needed ---
            if self.total_steps % self.eval_freq == 0:
                mean_eval_reward, eval_duration = self.evaluate_agent(agent, eval_env)
                print(f"--- Evaluation @ Step {self.total_steps} ---")
                print(f"  Avg Reward: {mean_eval_reward:.3f} (over {self.eval_eps} eps) | Eval Time: {eval_duration:.2f}s")

                wandb.log({
                    "total_steps": self.total_steps,
                    "val/reward": mean_eval_reward,
                    "val/time": eval_duration
                }, step=self.total_steps)

                if mean_eval_reward > best_eval_reward:
                    print(f"  New best eval reward ({best_eval_reward:.2f} -> {mean_eval_reward:.2f}). Saving model...")
                    best_eval_reward = mean_eval_reward
                    agent.save_model(self.model_path)
                    wandb.summary['best_eval_reward'] = best_eval_reward
                    wandb.summary['best_total_steps'] = self.total_steps
                    wandb.summary['best_episode_at_best_step'] = episode_count
            
            if self.total_steps >= self.train_steps:
                print("Reached target train_steps. Finishing training...")
                break

        total_training_time = time.time() - start_time
        print(f"\nTraining finished in {total_training_time:.2f} seconds after {self.total_steps} steps.")
        print(f"Best evaluation reward: {best_eval_reward:.2f} achieved around step {wandb.summary.get('best_total_steps', 'N/A')}")
        
        # --- Load the best model ---
        if os.path.exists(self.model_path):
            print(f"Loading best model from {self.model_path}")
            agent.load_model(self.model_path)
        else:
            print("Warning: Best model file not found. Returning agent with last state.")

        return agent

    def evaluate_agent(self, agent, env):
        total_reward = 0.0
        start_time = time.time()

        for i in range(self.eval_eps):
            ep_reward = 0
            state, _ = env.reset(seed = self.seed + i + 1)
            state = preprocess_state(state)
            done = False
            while not done:
                action = agent.sample_action(state, evaluate=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = preprocess_state(next_state)
                done = terminated or truncated
                state = next_state
                ep_reward += reward
            total_reward += ep_reward
        avg_reward = total_reward / self.eval_eps
        eval_duration = time.time() - start_time

        return avg_reward , eval_duration

# --- UTILS ---
def preprocess_state(state):
    """
    Processes the observation dictionary from the wrapper.
    - Extracts the 'image' array.
    - Normalizes pixel values to [0, 1].
    - Permutes dimensions from (H, W, C) to (C, H, W).
    """
    state = state.transpose((2, 0, 1))
    return state.astype(np.float32) / 255.0

def all_seed(seed=1):
    ''' omnipotent seed for RL, attention the position of seed function, you'd better put it just following the env create function
    Args:
        env (_type_): 
        seed (int, optional): _description_. Defaults to 1.
    '''
    if seed == 0:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # config for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # config for GPU
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed) # config for python scripts
        # config for cudnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

# --- MAIN ---
class MAIN:
    def __init__(self) -> None:
        self.general_cfg = {
            'env_name': 'MiniGrid-Empty-16x16-v0',
            'env_render': 'rgb_array',
            'project_name': 'MiniGrid-DQN',
            'device': 'cuda:1',
            'mode': 'train',
            'seed': 42,
            'buffer_size': int(1e4),
            'train_steps': int(5e4),
            'eval_eps': 50,
            'eval_freq': int(1e3),
            'model_dir': './task/'
        }
        self.sweep_cfg = {
            "method": "bayes",
            "metric":{"goal": "maximize", "name":"val/reward"},
            "parameters":{
                "lr": {"distribution": "uniform", "max": 5e-4, "min": 1e-4},
                "batch_size": {
                    "distribution": "q_log_uniform_values",
                    "max": 128,
                    "min": 32,
                    "q": 8
                },
                "gamma": {"max": 0.99, "min": 0.95},
                "epsilon_start": {"value": 1.0},
                "epsilon_end": {"value": 0.05},
                "epsilon_decay": {"values": [18000, 20000, 22000]},
                "target_update_freq": {"values": [900, 1000, 1100]}
            }
        }

    def envs_config(self, cfg, is_eval=False):
        env = gym.make(cfg['env_name'], render_mode=cfg['env_render'])
        env = ImgObsWrapper(env)
        if 'state_shape' not in cfg:
            state_shape = (env.observation_space.shape[2], env.observation_space.shape[0], env.observation_space.shape[1])
            action_dim = env.action_space.n
            
            # update to cfg paramters
            cfg.update({
                'state_shape': state_shape,
                'action_dim' : action_dim
                })
        return env
    
    def run_sweep_agent(self):
        wandb.init()
        wandb.config.update(self.general_cfg)
        cfg = wandb.config

        # --- Set random seed ---
        all_seed(cfg['seed'])

        # --- Definie ENV ---
        train_env = self.envs_config(cfg)
        eval_env = self.envs_config(cfg, is_eval=True)

        # --- Load agent, trainer ---
        agent = DQNAgent(cfg['state_shape'], cfg['action_dim'], cfg)
        trainer = Trainer(cfg)

        # --- Training ---
        agent = trainer.train(agent, train_env, eval_env)

if __name__ == "__main__":
    main = MAIN()

    sweep_id = wandb.sweep(main.sweep_cfg, project=main.general_cfg['project_name'])
    count = 10
    print(f"Start WandB Agent, which will execute {count} experiments...")
    wandb.agent(sweep_id, function=main.run_sweep_agent, count=count)
    
