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

from common.models import ActorCriticNetMinigird
from common.memories import PGReplay

class Agent:
    def __init__(self, cfg) -> None:
        self.lr = cfg["lr"]
        self.gamma = cfg["gamma"]
        self.device = cfg["device"]
        self.action_dim = cfg["action_dim"]
        self.state_shape = cfg["state_shape"] # Should be H, W, C

        # Loss coefficients
        self.actor_loss_coef = cfg.get("actor_loss_coef", 1.0)
        self.critic_loss_coef = cfg.get("critic_loss_coef", 0.5)
        self.entropy_coef = cfg.get("entropy_coef", 0.01) # Entropy bonus coefficient

        # Initialize the shared Actor-Critic model
        h, w, _ = self.state_shape # Assuming cfg["state_shape"] = (H, W, C)
        self.ac_net = ActorCriticNetMinigird(h, w, self.action_dim).to(self.device)

        # Gradient tracking
        wandb.watch(self.ac_net, log="all", log_freq=100)

        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=self.lr)

        # Use PGReplay buffer to store data for one episode
        self.memory = PGReplay()

    def sample_action(self, state, evaluate=False):
        """ Samples action from policy network and gets value estimate. """
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)

        # Set mode correctly
        if evaluate:
            self.set_eval_mode()
        else:
            self.set_train_mode()
        
        with torch.no_grad():
            action_probs, state_value = self.ac_net(state)
        dist = Categorical(action_probs)

        if evaluate:
            action = dist.probs.argmax(dim=1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        action = action.item()
        state_value = state_value.item()

        return action, log_prob, state_value

    def update_policy(self, final_next_state, final_done):
        """ Execute one training update using data from the last episode. """
        if len(self.memory) == 0:
            return 
        
        self.set_train_mode()

        states, actions, rewards, dones = self.memory.sample()
        # Convert lists to tensors
        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float32)
        actions = torch.tensor(np.array(actions), device=self.device, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards), device=self.device, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(np.array(dones), device=self.device, dtype=torch.float32).unsqueeze(1)

        num_transitions = len(rewards)
        # --- Calculate the Values, Log Probs, and Entropy under the current policy ---
        action_probs, current_values = self.ac_net(states) # Get pi(a|s) and V(s)
        dist = Categorical(action_probs)
        current_log_probs = dist.log_prob(actions.squeeze(1)).unsqueeze(1)
        entropy = dist.entropy().mean() # Average entropy over the batch

        # --- Calculate the value V(s_N+1) of the last state for bootstrapping ---
        with torch.no_grad():
            if final_done:
                bootstrap_value = torch.tensor(0.0, device=self.device)
            else:
                final_next_state_tensor = torch.tensor(final_next_state, device=self.device, dtype=torch.float32).unsqueeze(0)
                _, bootstrap_value = self.ac_net(final_next_state_tensor)
                bootstrap_value = bootstrap_value.squeeze() # become a scalar tensor

        # --- Compute the return (Returns) G_t ---
        returns = torch.zeros_like(rewards) # Initialization return G_t
        next_return = bootstrap_value
        for t in reversed(range(num_transitions)):
            returns[t] = rewards[t] + self.gamma * next_return * (1 - dones[t])
            next_return = returns[t]

        # --- Calculate the advantage function A_t = G_t - V(s_t) ---
        advantages = returns - current_values

        # --- Calculate Losses ---
        # Actor Loss = -log_prob * Advantage (detached) - entropy_bonus
        actor_loss = -(current_log_probs * advantages.detach()).mean()
        entropy_bonus = self.entropy_coef * entropy
        total_actor_loss = actor_loss - entropy_bonus

        # Critic Loss = MSE( V(s), TD Target (detached) )
        critic_loss = nn.MSELoss()(current_values, returns.detach())

        # Combined Loss
        total_loss = (self.actor_loss_coef * total_actor_loss + 
                      self.critic_loss_coef * critic_loss)
        
        # --- Update the Network ---
        self.optimizer.zero_grad()
        total_loss.backward()

        # Optional: Clip gradients
        # torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), max_norm=1.0)
        for param in self.ac_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()

        # Clear the episodic memory
        self.memory.clear()

        return total_actor_loss.item(), critic_loss.item(), entropy.item()
    
    def save_model(self, fpath):
        """ Saves the actor-critic network's state dictionary. """
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.ac_net.state_dict(), f"{fpath}/ActorCritic_checkpoint.pt")
    
    def load_model(self, fpath):
        """ Loads the actor-critic network's state dictionary. """
        model_file = f"{fpath}/ActorCritic_checkpoint.pt"
        self.ac_net.load_state_dict(torch.load(model_file, 
                                               map_location=self.device,
                                               weights_only=True))
        
    def set_train_mode(self):
        """ Sets the actor-critic network to training mode. """
        self.ac_net.train()

    def set_eval_mode(self):
        """ Sets the actor-critic network to evaluation mode. """
        self.ac_net.eval()

