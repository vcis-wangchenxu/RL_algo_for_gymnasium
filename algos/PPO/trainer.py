"""---------------------------------------
Define training and evaluation functions
---------------------------------------"""

from common.utils import preprocess_state
import wandb
import time
import os
import numpy as np
import torch

class Trainer:
    def __init__(self, cfg) -> None:
        self.device = cfg['device']
        self.train_steps = int(cfg.get('train_steps', 1_000_000))
        self.seed = cfg['seed']
        self.model_path = cfg['model_dir']
        self.update_steps = cfg['update_steps']
        self.eval_freq = cfg['eval_freq'] # Evaluate every N *steps*
        self.eval_eps = cfg['eval_eps']
        self.total_steps = 0

    def train(self, agent, train_env, eval_env):
        # --- WandB SetUp: total_steps ---
        # Reuse the same WandB setup, metrics are still relevant
        wandb.define_metric("total_steps")
        wandb.define_metric("train/actor_loss", step_metric="total_steps")
        wandb.define_metric("train/critic_loss", step_metric="total_steps")
        wandb.define_metric("train/entropy", step_metric="total_steps")
        wandb.define_metric("train/episode_reward", step_metric="total_steps")
        wandb.define_metric("train/episode_steps", step_metric="total_steps")
        wandb.define_metric("val/reward", step_metric="total_steps")
        wandb.define_metric("val/time", step_metric="total_steps")
        wandb.define_metric("episode_count", step_metric="total_steps")
        # -------------------------------------------------------------------

        best_eval_reward = -float('inf')
        start_time = time.time()
        episode_count = 0
        current_episode_reward = 0
        current_episode_steps  = 0

        state, _ = train_env.reset(seed=self.seed + episode_count)
        state = preprocess_state(state)

        print(f"\nStarting PPO Training for {self.train_steps} steps...")
        while self.total_steps < self.train_steps:
            action, old_log_prob = agent.sample_action(state, evaluate=False)
            next_state, reward, terminated, truncated, info = train_env.step(action)
            next_state = preprocess_state(next_state)
            done = terminated or truncated

            # --- shaping reward ---
            step_penalty = -0.1
            modified_reward = reward + step_penalty
            if done and reward > 0:
                modified_reward = reward

            transition = (state, 
                          action,
                          modified_reward, 
                          next_state,
                          done, 
                          old_log_prob)
            agent.memory.push(transition)

            # --- Update the state and counter ---
            state = next_state
            current_episode_reward += reward # Log original reward
            current_episode_steps  += 1
            self.total_steps += 1

            # --- PPO Update Trigger ---
            if len(agent.memory) >= self.update_steps:
                # PPO update consumes all data in memory and clears it internally
                update_results = agent.update_policy()
                if update_results: # Check if update happened (memory wasn't too small)
                    actor_loss, critic_loss, entropy = update_results
                    # --- Record the loss ---
                    wandb.log({
                        "total_steps": self.total_steps,
                        "train/actor_loss": actor_loss,
                        "train/critic_loss": critic_loss,
                        "train/entropy": entropy,
                    }, step=self.total_steps)
            
            if done:
                episode_count += 1
                print(f"Step: {self.total_steps}/{self.train_steps} | Ep: {episode_count} finished | "
                      f"EpRew: {current_episode_reward:.2f} | EpSteps: {current_episode_steps}")

                wandb.log({
                    "total_steps": self.total_steps,
                    "train/episode_reward": current_episode_reward,
                    "train/episode_steps": current_episode_steps,
                    "episode_count": episode_count,
                }, step=self.total_steps)

                # --- Reset environment and episode counter ---
                state, _ = train_env.reset(seed=self.seed + episode_count + self.total_steps) # Add total_steps for more seed variation
                state = preprocess_state(state)
                current_episode_reward = 0
                current_episode_steps = 0

            # --- Check if evaluation is needed ---
            # This part remains the same
            if self.total_steps % self.eval_freq == 0:
                mean_eval_reward, eval_duration = self.evaluate_agent(agent, eval_env) # evaluate_agent needs slight adjustment
                print(f"--- Evaluation @ Step {self.total_steps} ---")
                print(f"  Avg Reward: {mean_eval_reward:.3f} (over {self.eval_eps} eps) | Eval Time: {eval_duration:.2f}s")

                wandb.log({
                    "total_steps": self.total_steps,
                    "val/reward": mean_eval_reward,
                    "val/time": eval_duration,
                }, step=self.total_steps)

                if mean_eval_reward > best_eval_reward:
                    print(f"  New best eval reward ({best_eval_reward:.2f} -> {mean_eval_reward:.2f}). Saving model...")
                    best_eval_reward = mean_eval_reward
                    agent.save_model(self.model_path) # Agent saves with PPO name now if implemented
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
             print(f"Warning: No checkpoint file found in {self.model_path}. Returning agent with last state.")

        return agent
    
    # Adjust evaluate_agent slightly to handle tensor inputs/outputs if needed
    def evaluate_agent(self, agent, env):
        total_reward = 0.0
        start_time = time.time()
        agent.set_eval_mode() # Ensure eval mode

        for i in range(self.eval_eps):
            ep_reward = 0
            state, _ = env.reset(seed=self.seed + i + self.total_steps + 1) # Use varied seed
            state = preprocess_state(state)
            done = False
            eval_steps = 0
            while not done:
                # Use evaluate=True flag
                action, _ = agent.sample_action(state, evaluate=True)

                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = preprocess_state(next_state)
                done = terminated or truncated

                state = next_state # Update numpy state for next loop
                ep_reward += reward
                eval_steps += 1
            total_reward += ep_reward

        avg_reward = total_reward / self.eval_eps
        eval_duration = time.time() - start_time
        # agent.set_train_mode() # Set back to train mode after evaluation if needed (PPO agent does this internally before update)
        return avg_reward, eval_duration






