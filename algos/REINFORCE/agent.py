"""---------------------------------------------
Define the policy model as an agent update
---------------------------------------------"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import wandb
from pathlib import Path
import numpy as np
import math

from common.models import ActorSoftmax_net
from common.memories import PGReplay

class Agent:
    def __init__(self, cfg) -> None:
        self.lr = cfg["lr"]
        self.gamma = cfg["gamma"]
        self.device = cfg["device"]
        self.action_dim = cfg["action_dim"]
        self.state_shape = cfg["state_shape"]

        self.episode_log_probs = []
        self.episode_rewards   = []

        # Initialize the model
        h, w, _ = self.state_shape
        self.policy_net = ActorSoftmax_net(h, w, self.action_dim).to(self.device)
        # gradient
        wandb.watch(self.policy_net, log="all", log_freq=100)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr) 
        self.memory = PGReplay()

    def sample_action(self, state, evaluate=False):
        state= torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)

        if evaluate:
            # --- eval ---
            self.set_eval_mode()
            with torch.no_grad():
                action_probs = self.policy_net(state)
                action = torch.argmax(action_probs, dim=1).item()
            log_prob = None
        else:
            # --- train ---
            self.set_train_mode()
            action_probs = self.policy_net(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action = action.item()

        return action, log_prob

    def update_policy(self):
        self.set_train_mode()

        if len(self.memory) == 0:
            return     # No data to update
        
        all_s, all_a, all_r, all_log_probs, all_d = self.memory.sample()
        
        # Calculate discounted returns (G_t)
        returns = []
        discounted_return = 0
        for reward in reversed(all_r):
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return) # Prepend to keep order
        
        # Convert lists to tensor
        returns = torch.tensor(np.array(returns), device=self.device, dtype=torch.float32) # shape:[N]
        all_log_probs = torch.cat(all_log_probs).to(self.device)                           # shape:[N]

        # Calculate policy loss
        policy_loss = -(all_log_probs * returns).mean() # Use mean instead of sum for better scaling

        # Update the actor network
        self.optimizer.zero_grad()
        policy_loss.backward()
        # clip to avoid gradient explosion
        for param in self.policy_net.parameters():  
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Clear the episode memory
        self.memory.clear()

        return policy_loss.item()
    
    def save_model(self, fpath):
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"{fpath}/REINFORCE_checkpoint.pt")
    
    def load_model(self, fpath):
        self.policy_net.load_state_dict(torch.load(f"{fpath}/REINFORCE_checkpoint.pt",
                                                   map_location=self.device,
                                                   weights_only=True))
    
    def set_train_mode(self):
        self.policy_net.train()
    
    def set_eval_mode(self):
        self.policy_net.eval()






        


