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

    def forward(self, obs):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --- 2. Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)
    
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
class DoubleDQNAgent:
    def __init__(self, obs_shape, action_size, config):
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

        self.q_network = CNN_QNetwork(obs_shape, action_size).to(self.device)
        self.t_network = CNN_QNetwork(obs_shape, action_size).to(self.device)
        # Copy the weights of the policy network to the target network
        self.t_network.load_state_dict(self.q_network.state_dict())
        self.t_network.eval() # The target network is used only for inference, not for training
        # gradient
        wandb.watch(self.q_network, log="all", log_freq=100)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(config["buffer_size"])

    def sample_action(self, obs, evaluate=False):
        # epsilon = end + (start - end) * exp(-1. * steps_done / decay)
        if not evaluate:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                            math.exp(-1. * self.sample_count / self.epsilon_decay_steps)
            self.sample_count += 1 # current step / total steps
        
        # Exploration vs. Exploitation
        self.q_network.eval()
        if not evaluate and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            with torch.no_grad():
                obs = torch.tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                action = self.q_network(obs).argmax(dim=1).item()

        return action

    def update_policy(self):
        """ Execute one training iteration """
        self.q_network.train()

        if len(self.memory) < self.batch_size:
            return 
        
        b_s, b_a, b_r, b_ns, b_d = self.memory.sample(self.batch_size)
        # transition -> torch.tensor
        obss  = torch.tensor(np.array(b_s), device=self.device, dtype=torch.float32)
        actions = torch.tensor(np.array(b_a), device=self.device, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(np.array(b_r), device=self.device, dtype=torch.float32).unsqueeze(1)
        next_obss = torch.tensor(np.array(b_ns), device=self.device, dtype=torch.float32)
        dones   = torch.tensor(np.array(b_d), device=self.device, dtype=torch.float32).unsqueeze(1)

        q_values = self.q_network(obss).gather(dim=1, index=actions)
        max_policy_q_acitons = self.q_network(next_obss).max(1)[1].unsqueeze(1)
        next_target_q_values = self.t_network(next_obss).gather(1, index=max_policy_q_acitons).detach()

        # Calculate the loss
        q_targets = rewards + self.gamma * next_target_q_values * (1 - dones)
        double_dqn_loss = nn.MSELoss()(q_values, q_targets)

        # Update the policy network
        self.optimizer.zero_grad()
        double_dqn_loss.backward()
        # clip to avoid gradient explosion
        for param in self.q_network.parameters():  
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return double_dqn_loss.item()

    def update_target_network(self):
        self.t_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, fpath):
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.t_network.state_dict(), f"{fpath}/DoubleDQN_checkpoint.pt")

    def load_model(self, fpath):
        self.t_network.load_state_dict(torch.load(f"{fpath}/DoubleDQN_checkpoint.pt",
                                                   map_location=self.device,
                                                   weights_only=True))
        for target_param, param in zip(self.t_network.parameters(), self.q_network.parameters()):
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
        self.step_penalty = config['step_penalty']
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

        obs, _ = train_env.reset(seed=self.seed)
        obs = preprocess_obs(obs)

        print(f"\nStarting DoubleDQN Training for {self.train_steps} steps...")
        while self.total_steps < self.train_steps:
            action = agent.sample_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, info = train_env.step(action)
            next_obs = preprocess_obs(next_obs)
            done = terminated or truncated

            # --- Shaping Reward ---
            modified_reward = reward + self.step_penalty
            if done and reward > 0:
                modified_reward = reward

            transition = (obs, 
                          action, 
                          modified_reward, 
                          next_obs, 
                          done)
            agent.memory.push(transition)

            # --- Update the state and counter ---
            obs = next_obs
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
                obs, _ = train_env.reset(seed=self.seed + episode_count)
                obs = preprocess_obs(obs)
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
            obs, _ = env.reset(seed = self.seed + i + 1)
            obs = preprocess_obs(obs)
            done = False
            while not done:
                action = agent.sample_action(obs, evaluate=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_obs = preprocess_obs(next_obs)
                done = terminated or truncated
                obs = next_obs
                ep_reward += reward
            total_reward += ep_reward
        avg_reward = total_reward / self.eval_eps
        eval_duration = time.time() - start_time

        return avg_reward , eval_duration   

# --- UTILS ---
def preprocess_obs(obs):
    """
    Processes the observation dictionary from the wrapper.
    - Extracts the 'image' array.
    - Normalizes pixel values to [0, 1].
    - Permutes dimensions from (H, W, C) to (C, H, W).
    """
    obs = obs.transpose((2, 0, 1))
    return obs.astype(np.float32) / 255.0

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
            'project_name': 'MiniGrid-DoubleDQN',
            'device': 'cuda:1',
            'mode': 'train',
            'seed': 42,
            'buffer_size': int(2e4),
            'train_steps': int(6e4),
            'eval_eps': 50,
            'eval_freq': int(1e3),
            'model_dir': './task/'
        }
        self.sweep_cfg = {
            "method": "bayes",
            "metric":{"goal": "maximize", "name":"val/reward"},
            "parameters":{
                "lr": {"distribution": "log_uniform_values", "max": 5e-4, "min": 5e-5},
                "batch_size": {"values": [32, 64, 128]},
                "gamma": {"distribution": "uniform", "max": 0.99, "min": 0.9},
                "epsilon_start": {"value": 1.0},
                "epsilon_end": {"value": 0.05},
                "epsilon_decay": {"values": [5000, 10000, 20000, 30000]},
                "target_update_freq": {"values": [500, 1000, 2000, 5000]},
                "step_penalty": {"distribution": "uniform", "max": -0.01, "min": -0.1}
            }
        }

    def envs_config(self, cfg, is_eval=False):
        env = gym.make(cfg['env_name'], render_mode=cfg['env_render'])
        env = ImgObsWrapper(env)
        if 'obs_shape' not in cfg:
            h, w, c = env.observation_space.shape
            obs_shape = (c, h, w)
            action_dim = env.action_space.n
            
            # update to cfg paramters
            cfg.update({
                'obs_shape': obs_shape,
                'action_dim' : action_dim
                })
        return env
    
    def run_sweep_agent(self):
        wandb.init()

        sweep_id = wandb.run.sweep_id if wandb.run.sweep_id else "single_run"
        run_id = wandb.run.id
        unique_model_dir = os.path.join(self.general_cfg['model_dir'], 
                                        f"sweep_{sweep_id}", 
                                        f"run_{run_id}")
        
        cfg = wandb.config
        temp_cfg = self.general_cfg.copy()
        temp_cfg.update(dict(cfg))
        cfg = temp_cfg
        cfg['model_dir'] = unique_model_dir

        # --- Set random seed ---
        all_seed(cfg['seed'])

        # --- Definie ENV ---
        train_env = self.envs_config(cfg)
        eval_env = self.envs_config(cfg, is_eval=True)

        # --- Load agent, trainer ---
        agent = DoubleDQNAgent(cfg['obs_shape'], cfg['action_dim'], cfg)
        trainer = Trainer(cfg)

        # --- Training ---
        agent = trainer.train(agent, train_env, eval_env)

if __name__ == "__main__":
    main = MAIN()

    sweep_id = wandb.sweep(main.sweep_cfg, project=main.general_cfg['project_name'])
    count = 10
    print(f"Start WandB Agent, which will execute {count} experiments...")
    wandb.agent(sweep_id, function=main.run_sweep_agent, count=count)

