"""---------------------------------------
Define training and evaluation functions
---------------------------------------"""

from common.utils import preprocess_state
import wandb
import time
import os

class Trainer:
    def __init__(self, cfg):
        self.device = cfg['device']
        self.train_steps = int(cfg['train_steps'])
        self.seed = cfg['seed']
        self.model_path = cfg['model_dir']
        self.target_update_freq = cfg['target_update_freq']
        self.eval_freq = cfg['eval_freq']
        self.eval_eps = cfg['eval_eps']

    def train(self, agent, train_env, eval_env):
        # --- WandB SetUp: Define Metries and Custom X-Axes ---
        wandb.define_metric("time_step")
        wandb.define_metric("train/loss", step_metric="time_step")
        wandb.define_metric("val/reward", step_metric="time_step")
        wandb.define_metric("val/time", step_metric="time_step")
        wandb.define_metric("val/epsilon", step_metric="time_step")
        # -------------------------------------------------------------------

        best_ep_reward = -float('inf')
        num_episodes = 0

        print("\nStarting Training...")
        while agent.sample_count < self.train_steps:
            state, _ = train_env.reset(seed=self.seed)
            state = preprocess_state(state)
            episode_reward = 0
            episode_step   = 0
            done = False

            # --- Training Episode Loop ---
            while not done:
                if agent.sample_count >= self.train_steps:
                    break # Exit if max steps reached

                # Select action for training (epsilon-greedy)
                action = agent.sample_action(state)

                # Interact with training environment
                next_state, reward, terminated, truncated, info = train_env.step(action)
                next_state = preprocess_state(next_state)
                done = terminated or truncated

                # shaping reward
                step_penalty = -0.1
                modified_reward = reward + step_penalty
                if done and reward > 0:
                    modified_reward = reward
                
                # Preprocess observation for buffer
                transition = (state, action, modified_reward, next_state, done)
                agent.memory.push(transition)
                
                state = next_state
                # Update episode reward and step
                episode_reward += reward
                episode_step   += 1

                # --- Update policy_net ---
                loss = agent.update_policy()

                # --- Update target_net ---
                if agent.sample_count % self.target_update_freq == 0:
                    agent.update_target_net()
                
                # --- Check if it's time for evaluation
                if agent.sample_count % self.eval_freq == 0:
                    mean_eval_reward, eval_duration = self.evaluate_agent(agent, eval_env)
                    print(f"Step: {agent.sample_count}/{self.train_steps} | Evaluation Reward: {mean_eval_reward:.3f} (Avg over {self.eval_eps} eps) | "
                          f"Ep: {num_episodes} | Loss: {loss:.4f} | Eps: {agent.epsilon:.3f} | Eval Time: {eval_duration:.2f}s")
            
                    if mean_eval_reward > best_ep_reward: # update best reward
                        print(f"Validation accuracy improved ({best_ep_reward:.2f} -> {mean_eval_reward:.2f}). Saving model...")
                        best_ep_reward = mean_eval_reward
                        agent.save_model(self.model_path)

                        wandb.summary['best_ep_reward'] = best_ep_reward
                        wandb.summary['best_step'] = agent.sample_count

                    # --- WandB log ---
                    wandb.log({
                        "time_step": agent.sample_count,
                        "training/loss": loss,
                        "val/reward": mean_eval_reward,
                        "val/time": eval_duration,
                        "val/epsilon": agent.epsilon
                    })  
            
            num_episodes += 1
            print(f"--- training/episode_reward: {episode_reward:.3f}; training/episode_step: {episode_step} ---")
        
        print("\nTraining finished.")
        print(f"Best ep reward: {best_ep_reward:.2f} achieved at step {wandb.summary.get('best_step', 'N/A')}")

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
