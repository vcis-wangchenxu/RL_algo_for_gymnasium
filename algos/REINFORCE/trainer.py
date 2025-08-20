"""---------------------------------------
Define training and evaluation functions
---------------------------------------"""

from common.utils import preprocess_state
import wandb
import time
import os
import numpy as np

class Trainer:
    def __init__(self, cfg) -> None:
        self.device = cfg['device']
        self.train_episodes = cfg['train_episodes']
        self.seed = cfg['seed']
        self.model_path = cfg['model_dir']
        self.eval_freq = cfg['eval_freq'] # Evaluate every N *episodes*
        self.eval_eps = cfg['eval_eps']

    def train(self, agent, train_env, eval_env):
        # --- WandB SetUp: Define Metrics and Custom X-Axes ---
        # Use episode as the primary x-axis for REINFORCE
        wandb.define_metric("episode")
        wandb.define_metric("train/loss", step_metric="episode")
        wandb.define_metric("train/episode_reward", step_metric="episode")
        wandb.define_metric("train/episode_steps", step_metric="episode")
        wandb.define_metric("val/reward", step_metric="episode") # Log validation against episode
        wandb.define_metric("val/time", step_metric="episode")
        # -------------------------------------------------------------------

        best_eval_reward = -float('inf')
        start_time = time.time()

        print(f"\nStarting REINFORCE Training for {self.train_episodes} episodes...")
        for episode in range(self.train_episodes):
            state, _ = train_env.reset(seed=self.seed + episode)
            state = preprocess_state(state)
            episode_reward = 0
            episode_steps  = 0
            done = False

            # --- Run One Episode ---
            while not done:
                # Sample action and log probability
                action, log_prob = agent.sample_action(state, evaluate=False)

                # Interact with training environment
                next_state, reward, terminated, truncated, info = train_env.step(action)
                next_state = preprocess_state(next_state)
                done = terminated or truncated

                # shaping reward
                step_penalty = -0.1
                modified_reward = reward + step_penalty
                if done and reward > 0:
                    modified_reward = reward
                
                transition = (state, action, modified_reward, log_prob, done)
                agent.memory.push(transition)

                state = next_state
                # Update episode reward and step
                episode_reward += reward
                episode_steps   += 1
            
            # --- Update policy_net after the episode ---
            loss = agent.update_policy()

            # --- Logging ---
            print(f"Episode: {episode+1}/{self.train_episodes} | Steps: {episode_steps} | "
                  f"Reward: {episode_reward:.2f} | Loss: {loss if loss is not None else 'N/A'} ")

            wandb.log({
                "episode": episode + 1,
                "train/loss": loss if loss is not None else 0, # Log 0 if no update occurred
                "train/episode_reward": episode_reward,
                "train/episode_steps": episode_steps,
            })

            # --- Periodic Evaluation ---
            if (episode+1) % self.eval_freq == 0:
                mean_eval_reward, eval_duration = self.evaluate_agent(agent, eval_env)
                print(f"  Avg Reward: {mean_eval_reward:.3f} (over {self.eval_eps} eps) | Eval Time: {eval_duration:.2f}s")

                wandb.log({
                    "episode": episode + 1, # Log against episode
                    "val/reward": mean_eval_reward,
                    "val/time": eval_duration,
                })

                # Save best model based on evaluation reward
                if mean_eval_reward > best_eval_reward:
                    print(f"  New best evaluation reward ({best_eval_reward:.2f} -> {mean_eval_reward:.2f}). Saving model...")
                    best_eval_reward = mean_eval_reward
                    agent.save_model(self.model_path)
                    # Update WandB summary
                    wandb.summary['best_eval_reward'] = best_eval_reward
                    wandb.summary['best_episode'] = episode + 1
                    
            
        total_training_time = time.time() - start_time
        print(f"\nTraining finished in {total_training_time:.2f} seconds.")
        print(f"Best evaluation reward: {best_eval_reward:.2f} achieved at episode {wandb.summary.get('best_episode', 'N/A')}")

        # --- Load the best model ---
        best_model_path = os.path.join(self.model_path, "REINFORCE_checkpoint.pt")
        if os.path.exists(best_model_path):
            print(f"Loading best model from {self.model_path}")
            agent.load_model(self.model_path) # load_model expects directory path
        else:
            print("Warning: Best model file not found. Returning agent with last weights.")

        return agent
    
    def evaluate_agent(self, agent, env):
        """ Evaluates the agent's policy greedily. """
        total_reward = 0.0
        start_time = time.time()

        for i in range(self.eval_eps):
            ep_reward = 0
            state, _ = env.reset(seed=self.seed + i + 1 + self.train_episodes) # Use different seeds for eval
            state = preprocess_state(state)
            done = False
            while not done: 
                # Use evaluate=True flag in sample_action
                action, _ = agent.sample_action(state, evaluate=True) # Don't need log_prob
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = preprocess_state(next_state)
                done = terminated or truncated
                state = next_state
                ep_reward += reward
            total_reward += ep_reward

        avg_reward = total_reward / self.eval_eps
        eval_duration = time.time() - start_time

        return avg_reward, eval_duration