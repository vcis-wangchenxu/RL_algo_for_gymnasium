import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
import random
import time
import os
import wandb
import math
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt

# --- 1. MODIFIED Policy Network ---
class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, output_size):
        super(PolicyNetwork, self).__init__()
        c, h, w = input_shape

        self.conv1 = nn.Conv2d(c, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)

        conv_out_size = self._get_conv_out(input_shape)

        self.fc1 = nn.Linear(conv_out_size, 256)
        self.action_head = nn.Linear(256, output_size)

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
        action_logits = self.action_head(x)
        return action_logits

# --- 2. Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transitions):
        self.buffer.append(transitions)

    def sample(self, batch_size, sequential: bool=False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential:
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)
    
    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

class PGReplay(ReplayBuffer):
    '''replay buffer for policy gradient based methods, each time these methods will sample all transitions
    Args:
        ReplayBuffer (_type_): _description_
    '''
    def __init__(self):
        self.buffer = deque()
    def sample(self):
        ''' sample all the transitions
        '''
        batch = list(self.buffer)
        return zip(*batch)

# --- 3. MODIFIED: REINFORCE Agent ---
class REINFORCEAgent:
    def __init__(self, obs_shape, action_size, config):
        self.lr = config["lr"]
        self.gamma = config["gamma"]
        self.device = config["device"]
        self.action_size = action_size

        self.policy_network = PolicyNetwork(obs_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.memory = PGReplay()

        # gradient
        wandb.watch(self.policy_network, log="all", log_freq=100)

    def sample_action(self, obs, evaluate=False):
        self.policy_network.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            action_logits = self.policy_network(obs_tensor)

            action_dist = Categorical(logits=action_logits)
            if evaluate:
                action = action_logits.argmax(dim=1).item()
            else:
                action = action_dist.sample()
            
            log_prob = action_dist.log_prob(action)

        return action.item(), log_prob
    
    def update_policy(self):
        self.policy_network.train()

        if len(self.memory) == 0:
            return 
        
        all_s, all_a, all_r, all_log_probs, all_d = self.memory.sample()
        
        # Calculate discounted returns (G_t)
        returns = []
        discounted_return = 0   
        for reward in reversed(all_r):
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)
        
        # Convert lists to tensor
        returns = torch.tensor(np.array(returns), device=self.device, dtype=torch.float32) # shape:[N]
        all_log_probs = torch.cat(all_log_probs).to(self.device)                           # shape:[N]

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = -(all_log_probs * returns).mean()

        # Update the actor network
        self.optimizer.zero_grad()
        policy_loss.backward()
        # clip to avoid gradient explosion
        for param in self.policy_network.parameters():  
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Clear the episode memory
        self.memory.clear()

        return policy_loss.item()

    def save_model(self, fpath):
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_network.state_dict(), f"{fpath}/REINFORCE_checkpoint.pt")
    
    def load_model(self, fpath):
        self.policy_network.load_state_dict(torch.load(f"{fpath}/REINFORCE_checkpoint.pt",
                                                   map_location=self.device,
                                                   weights_only=True))
    
# --- 4. MODIFIED: Trainer --- 
class Trainer:
    def __init__(self, config):
        self.device = config['device']
        self.train_steps = config['train_steps']
        self.seed = config['seed']
        self.model_path = config['model_dir']
        self.eval_freq = config['eval_freq']
        self.eval_eps = config['eval_eps']
        self.step_penalty = config['step_penalty']
        self.total_steps = 0

    def train(self, agent, train_env, eval_env):
        # --- WandB SetUp: Define Metrics and Custom X-Axes ---
        wandb.define_metric("total_steps")
        wandb.define_metric("train/loss", step_metric="total_steps")
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

        print(f"\nStarting REINFORCE Training for {self.train_steps} steps...")
        while self.total_steps < self.train_steps:
            action, log_prob = agent.sample_action(obs, evaluate=False)

            next_obs, reward, terminated, truncated, info = train_env.step(action)
            next_obs = preprocess_obs(next_obs)
            done = terminated or truncated

            # shaping reward
            modified_reward = reward + self.step_penalty
            if done and reward > 0:
                modified_reward = reward
            
            transition = (obs, action, modified_reward, log_prob, done)
            agent.memory.push(transition)

            obs = next_obs
            current_episode_reward += reward
            current_episode_steps += 1
            self.total_steps += 1

            if done:
                loss = agent.update_policy()
                episode_count += 1

                print(f"Step: {self.total_steps}/{self.train_steps} | Ep: {episode_count} finished | "
                      f"EpRew: {current_episode_reward:.2f} | EpSteps: {current_episode_steps} | Loss: {loss:.4f}")
                
                wandb.log({
                    "total_steps": self.total_steps,
                    "train/loss": loss,
                    "train/episode_reward": current_episode_reward,
                    "train/episode_steps": current_episode_steps, 
                    "episode_count": episode_count
                }, step=self.total_steps)

                # --- Reset environment and episode counter ---
                obs, _ = train_env.reset(seed=self.seed + episode_count)
                obs = preprocess_obs(obs)
                current_episode_reward = 0
                current_episode_steps  = 0
            
            if self.total_steps % self.eval_freq == 0:
                mean_eval_reward, eval_duration = self.evaluate_agent(agent, eval_env)
                print(f"--- Evaluation @ Step {self.total_steps} ---")
                print(f"   Avg Reward: {mean_eval_reward:.3f} (over {self.eval_eps} eps) | Eval Time: {eval_duration:.2f}s")

                wandb.log({
                    "total_steps": self.total_steps,
                    "val/reward": mean_eval_reward,
                    "val/time": eval_duration
                }, step=self.total_steps)

                if mean_eval_reward > best_eval_reward:
                    print(f"   New best eval reward ({best_eval_reward:.2f} -> {mean_eval_reward:.2f}). Saving model...")
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
                action, _ = agent.sample_action(obs, evaluate=True)
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
    obs = obs.transpose((2, 0, 1))
    return obs.astype(np.float32) / 255.0

def all_seed(seed=1):
    if seed == 0: return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

class MAIN:
    def __init__(self) -> None:
        self.general_cfg = {
            'env_name': 'MiniGrid-Empty-16x16-v0',
            'env_render': 'rgb_array',
            'project_name': 'MiniGrid-REINFORCE', # 修改项目名称
            'device': 'cuda:1',
            'mode': 'train',
            'seed': 42,
            'train_steps': int(8e4),
            'eval_eps': 50,
            'eval_freq': int(1e3),
            'model_dir': './task/'
        }
        # --- 修改超参数搜索空间以适应REINFORCE ---
        self.sweep_cfg = {
            "method": "bayes",
            "metric":{"goal": "maximize", "name":"val/reward"},
            "parameters":{
                "lr": {"distribution": "log_uniform_values", "max": 1e-3, "min": 5e-5},
                "gamma": {"distribution": "uniform", "max": 0.995, "min": 0.95},
                "step_penalty": {"distribution": "uniform", "max": -0.001, "min": -0.05}
            }
        }

    def envs_config(self, cfg, is_eval=False):
        env = gym.make(cfg['env_name'], render_mode=cfg['env_render'])
        env = ImgObsWrapper(env)
        if 'obs_shape' not in cfg:
            h, w, c = env.observation_space.shape
            obs_shape = (c, h, w)
            action_dim = env.action_space.n
            
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

        all_seed(cfg['seed'])

        train_env = self.envs_config(cfg)
        eval_env = self.envs_config(cfg, is_eval=True)
        
        agent = REINFORCEAgent(cfg['obs_shape'], cfg['action_dim'], cfg)
        trainer = Trainer(cfg)

        agent = trainer.train(agent, train_env, eval_env)

if __name__ == "__main__":
    main = MAIN()
    sweep_id = wandb.sweep(main.sweep_cfg, project=main.general_cfg['project_name'])
    count = 50
    print(f"Start WandB Agent, which will execute {count} experiments...")
    wandb.agent(sweep_id, function=main.run_sweep_agent, count=count)
