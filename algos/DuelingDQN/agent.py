"""---------------------------------------------
Define the policy model as an agent update
---------------------------------------------"""

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from pathlib import Path
import numpy as np
import math

from common.models import VAQnet
from common.memories import ReplayBuffer

class Agent:
    def __init__(self, cfg):
        self.lr = cfg["lr"]
        self.gamma = cfg["gamma"]
        self.device = cfg["device"]
        self.batch_size = cfg["batch_size"]
        self.action_dim = cfg["action_dim"]
        self.state_shape = cfg["state_shape"]
    
        # e-greedy parameters
        self.sample_count = 0  # sample count for epsilon decay
        self.epsilon_start = cfg["epsilon_start"]
        self.epsilon_end = cfg["epsilon_end"]
        self.epsilon_decay_steps = cfg["epsilon_decay"]

        # Initialize the model
        h, w, _  = self.state_shape
        self.policy_net = VAQnet(h, w, self.action_dim).to(self.device)
        self.target_net = VAQnet(h, w, self.action_dim).to(self.device)
        # Copy the weights of the policy network to the target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # The target network is used only for inference, not for training
        # gradient
        wandb.watch(self.policy_net, log="all", log_freq=100)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr) 

        self.memory = ReplayBuffer(cfg["buffer_size"])

    def sample_action(self, state, evaluate=False):
        # epsilon = end + (start - end) * exp(-1. * steps_done / decay)
        if not evaluate:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                            math.exp(-1. * self.sample_count / self.epsilon_decay_steps)
            self.sample_count += 1 # current step / total steps
        
        # Exploration vs. Exploitation
        if not evaluate and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            self.set_eval_mode()
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                action = self.policy_net(state).argmax(dim=1).item()
        
        return action
    
    def update_policy(self):
        """ Execute one training iteration """
        self.set_train_mode()

        if len(self.memory) < self.batch_size:
            return 

        b_s, b_a, b_r, b_ns, b_d = self.memory.sample(self.batch_size)
        transition_dict = {
            'states': np.array(b_s),
            'actions': np.array(b_a), 
            'rewards': np.array(b_r),
            'next_states': np.array(b_ns),
            'dones': np.array(b_d)
        }

        # transition -> torch.tensor
        states  = torch.tensor(transition_dict['states'], device=self.device, dtype=torch.float)
        actions = torch.tensor(transition_dict['actions'], device=self.device).unsqueeze(1)
        rewards = torch.tensor(transition_dict['rewards'], device=self.device, dtype=torch.float).unsqueeze(1)
        next_states = torch.tensor(transition_dict['next_states'], device=self.device, dtype=torch.float)
        dones   = torch.tensor(transition_dict['dones'], device=self.device, dtype=torch.float).unsqueeze(1)

        # The Q-value of the current state for executing action actions
        q_values = self.policy_net(states).gather(dim=1, index=actions)
        # The maxQ of the next state next_states
        max_next_q_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)

        # Calculate the loss
        q_targets = rewards + self.gamma * max_next_q_values * (1-dones) # TD误差目标
        dqn_loss  = nn.MSELoss()(q_values, q_targets)

        # Update the policy network
        self.optimizer.zero_grad() 
        dqn_loss.backward() 
        # clip to avoid gradient explosion
        for param in self.policy_net.parameters():  
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return dqn_loss.item()
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, fpath):
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.target_net.state_dict(), f"{fpath}/DuelingDQN_checkpoint.pt")
        # wandb.save(f"{fpath}/DQN_checkpoint.pt") 

    def load_model(self, fpath):
        self.target_net.load_state_dict(torch.load(f"{fpath}/DuelingDQN_checkpoint.pt",
                                                   map_location=self.device,
                                                   weights_only=True))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)

    def set_train_mode(self):
        self.policy_net.train()

    def set_eval_mode(self):
        self.policy_net.eval()

