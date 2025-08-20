"""---------------------------------------------
Define the policy model as an agent update
---------------------------------------------"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

import numpy as np
import wandb
from pathlib import Path

from common.models import ActorNet, CriticNet
from common.memories import PGReplay

class Agent:
    def __init__(self, cfg) -> None:
        self.actor_lr = cfg["actor_lr"]
        self.critic_lr = cfg["critic_lr"]
        self.gamma = cfg["gamma"]
        self.device = cfg["device"]
        self.action_dim = cfg["action_dim"]
        self.state_shape = cfg["state_shape"] # Should be H, W, C

        # PPO specific hyperparameters
        self.gae_lambda = cfg.get("gae_lambda", 0.95) # Lambda for GAE
        self.ppo_clip_param = cfg.get("ppo_clip_param", 0.2) # Epsilon for clipping
        self.ppo_epochs = cfg.get("ppo_epochs", 10) # Number of optimization epochs per update
        self.mini_batch_size = cfg.get("mini_batch_size", 64) # Size of mini-batches for optimization
        self.eps = cfg.get("eps", 1e-8)
        self.grad_clip_norm = cfg.get("grad_clip_norm", 0.5)
        self.normalize_advantage = cfg.get("normalize_advantage", True)

        # Loss coefficients
        self.actor_loss_coef = cfg.get("actor_loss_coef", 1.0) # Often 1.0 for PPO
        self.critic_loss_coef = cfg.get("critic_loss_coef", 0.5)
        self.entropy_coef = cfg.get("entropy_coef", 0.01)

        # Initialize the shared Actor-Critic model
        h, w, _ = self.state_shape
        self.actor = ActorNet(h, w, self.action_dim).to(self.device)
        self.critic = CriticNet(h, w).to(self.device)

        # Gradient tracking
        wandb.watch(self.actor, log="all", log_freq=100)
        wandb.watch(self.critic, log="all", log_freq=100)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)


        # Use PGReplay buffer to store data until update_steps is reached
        self.memory = PGReplay()

    def sample_action(self, state, evaluate=False):
        """ Samples action """
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)

        self.set_eval_mode()
        
        with torch.no_grad():
            action_probs = self.actor(state)
        dist = Categorical(action_probs)

        if evaluate:
            action = dist.probs.argmax(dim=1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()
    
    def update_policy(self):
        """ Execute PPO training update using data collected in memory. """
        if len(self.memory) < self.mini_batch_size:
            return 
        
        self.set_train_mode()

        states_list, actions_list, rewards_list, next_states_list, dones_list, \
        old_log_probs = self.memory.sample()

        # Convert lists to tensors
        states = torch.tensor(np.array(states_list), device=self.device, dtype=torch.float32)
        actions = torch.tensor(np.array(actions_list)).to(self.device).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards_list), device=self.device, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states_list), device=self.device, dtype=torch.float32)
        dones = torch.tensor(np.array(dones_list), device=self.device, dtype=torch.float32).unsqueeze(1)

        old_log_probs = torch.tensor(np.array(old_log_probs), device=self.device, dtype=torch.float32).unsqueeze(1)

        # --- Calculate Advantages using GAE ---
        with torch.no_grad():
            values_next = self.critic(next_states)
            td_target  = rewards + self.gamma * values_next * (1 - dones)
            values_current = self.critic(states)
            td_delta = td_target - values_current

        advantage = self.compute_advantage(td_delta.squeeze(-1))
        advantage = advantage.unsqueeze(-1)

        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + self.eps)

        # --- PPO Update Loop ---
        total_actor_loss_epoch = 0
        total_critic_loss_epoch = 0
        total_entropy_epoch = 0

        num_samples = states.size(0)
        indices = np.arange(num_samples)

        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, num_samples, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, num_samples)
                mini_batch_indices = indices[start_idx:end_idx]

                batch_states = states[mini_batch_indices]
                batch_actions = actions[mini_batch_indices]
                batch_advantage = advantage[mini_batch_indices]
                batch_old_log_probs = old_log_probs[mini_batch_indices]
                batch_td_target = td_target[mini_batch_indices]

                current_action_probs = self.actor(batch_states)
                dist_current_policy = Categorical(probs=current_action_probs)
                current_log_probs = dist_current_policy.log_prob(batch_actions.squeeze(-1)).unsqueeze(-1)

                ratio = torch.exp(current_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantage
                surr2 = torch.clamp(ratio, 1 - self.ppo_clip_param, 1 + self.ppo_clip_param) * batch_advantage
                
                actor_objective_loss = -torch.min(surr1, surr2).mean()
                entropy = dist_current_policy.entropy().mean()
                actor_loss = self.actor_loss_coef * actor_objective_loss - self.entropy_coef * entropy
                
                current_values_critic = self.critic(batch_states)
                critic_loss = self.critic_loss_coef * nn.MSELoss()(current_values_critic, batch_td_target.detach())
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if self.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
                self.critic_optimizer.step()

                total_actor_loss_epoch += actor_objective_loss.item()
                total_critic_loss_epoch += critic_loss.item() / self.critic_loss_coef if self.critic_loss_coef > 0 else critic_loss.item()
                total_entropy_epoch += entropy.item()
            
        # Clear the memory after all epochs are done
        self.memory.clear()
        
        num_update_iterations = (num_samples / self.mini_batch_size) * self.ppo_epochs
        if num_update_iterations == 0: num_update_iterations = 1 # Avoid division by zero if batch too small for any update

        avg_actor_loss = total_actor_loss_epoch / num_update_iterations
        avg_critic_loss = total_critic_loss_epoch / num_update_iterations
        avg_entropy = total_entropy_epoch / num_update_iterations

        return avg_actor_loss, avg_critic_loss, avg_entropy
    
    def compute_advantage(self, td_delta):
        """
        Compute advantages using GAE.
        """
        advantage_list = []
        advantage = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for delta in reversed(td_delta):
            advantage = self.gamma * self.gae_lambda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.stack(advantage_list)
       
    def save_model(self, fpath):
        """ Saves the actor-critic network's state dictionary. """
        Path(fpath).mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, f"{fpath}/PPO_ActorCritic_checkpoint.pt")
    
    def load_model(self, fpath):
        """ Loads the actor-critic network's state dictionary. """
        model_file = f"{fpath}/PPO_ActorCritic_checkpoint.pt"
        checkpoint = torch.load(model_file, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
    def set_train_mode(self):
        """ Sets the actor-critic network to training mode. """
        self.actor.train()
        self.critic.train()

    def set_eval_mode(self):
        """ Sets the actor-critic network to evaluation mode. """
        self.actor.eval()
        self.critic.eval()
