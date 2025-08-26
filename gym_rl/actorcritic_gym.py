import random
import gymnasium as gym
import numpy as np
from collections import deque
from typing import Dict, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import swanlab

def set_seeds(seed: int) -> None:
    """ 设置所有随机种子 """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ReplayBuffer:
    """ 定义经验回放池 """
    def __init__(self) -> None:
        self.buffer = deque()
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self) -> Tuple[np.ndarray, Tuple, Tuple, np.ndarray, Tuple]:
        state, action, reward, next_state, done = zip(*self.buffer)
        return np.array(state), action, reward, np.array(next_state), done
    
    def clear(self) -> None:
        return self.buffer.clear()

class ActorNet(nn.Module):
    """ Policy network for the Actor. """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        # Use softmax to get action probabilities
        return F.softmax(self.fc2(x), dim=-1)

class CriticNet(nn.Module):
    """ Value network for the Critic. """
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # Outputs a single state-value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ActorCritic:
    """ Actor-Critic algorithm agent implementation. """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, actor_lr: float, 
                 critic_lr: float, gamma: float, device: torch.device) -> None:
        self.actor = ActorNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = CriticNet(state_dim, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.criterion = nn.MSELoss()

        self.gamma = gamma
        self.count = 0
        self.device = device
    
    @torch.no_grad()
    def take_action(self, state: np.ndarray) -> int:
        """ Samples an action from the policy distribution. """
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        # Get action probabilities from the actor network
        probs = self.actor(state_tensor)
        # Create a categorical distribution and sample an action
        action_dist = Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition: Dict[str, any]) -> Tuple[float, float]:
        """ Updates the Actor and Critic networks. """
        state = torch.tensor(np.array([transition['state']]), dtype=torch.float).to(self.device)
        action = torch.tensor([transition['action']], dtype=torch.int64).unsqueeze(1).to(self.device)
        reward = torch.tensor([transition['reward']], dtype=torch.float).unsqueeze(1).to(self.device)
        next_state = torch.tensor(np.array([transition['next_state']]), dtype=torch.float).to(self.device)
        done = torch.tensor([transition['done']], dtype=torch.float).unsqueeze(1).to(self.device)

        # --- Update Critic ---
        # Calculate TD Target
        td_target = reward + self.gamma * self.critic(next_state) * (1 - done)
        # Calculate TD Error (Advantage)
        td_delta = td_target - self.critic(state)
        # Critic loss is the squared TD error
        critic_loss = self.criterion(self.critic(state), td_target.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actor ---
        # Get log probability of the taken action
        log_probs = torch.log(self.actor(state).gather(1, action))
        # Actor loss is the negative log probability scaled by the TD error (advantage)
        # We use .detach() on td_delta because we don't want to backpropagate through the critic here.
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.count += 1
        return actor_loss.item(), critic_loss.item()

def evaluate(agent: ActorCritic, env: gym.Env, n_episodes: int = 10) -> float:
    """ Evaluates the agent's performance. """
    total_return = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_return += reward
        total_return += episode_return
    return total_return / n_episodes

def train_ActorCritic(agent: ActorCritic, env: gym.Env, config: dict) -> None:
    print("--- Starting Training ---")
    return_list: List[float] = []
    total_steps = 0
    i_episode = 0

    while total_steps < config['total_timesteps']:
        episode_return = 0.0
        state, _ = env.reset()
        done = False
        i_episode += 1

        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_return += reward
            total_steps += 1

            transition_dict = {
                'state': state, 'action': action, 'reward': reward,
                'next_state': next_state, 'done': done
            }
            actor_loss, critic_loss = agent.update(transition_dict)
            swanlab.log({
                "Train/Actor_Loss": actor_loss,
                "Train/Critic_Loss": critic_loss
            }, step=agent.count)

            state = next_state

        return_list.append(episode_return)
        swanlab.log({"Return/by_Episode": episode_return}, step=i_episode)
        swanlab.log({"Return/by_Step": episode_return}, step=total_steps)
        
        if i_episode % config['eval_freq'] == 0:
            eval_reward = evaluate(agent, env, n_episodes=10)
            swanlab.log({"Eval/Average_Return": eval_reward}, step=total_steps)
            print(f"Episode: {i_episode}, Steps: {total_steps}/{config['total_timesteps']}, "
                  f"Eval Avg Return: {eval_reward:.2f}")
    
    env.close()
    print("--- Training Finished ---")

if __name__ == "__main__":
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    run = swanlab.init(
        project="ActorCritic_CartPole",
        name=f"AC_CartPole_{device_str}",
        config={
            "actor_lr": 1e-3,
            "critic_lr": 1e-2,
            "total_timesteps": 60000,
            "hidden_dim": 128,
            "gamma": 0.98,
            "eval_freq": 20,
            "env_name": 'CartPole-v1',
            "seed": 0,
            "device": device_str,
        },
    )

    config = swanlab.config

    set_seeds(config['seed'])
    # replay_buffer = ReplayBuffer()

    env = gym.make(config['env_name'], render_mode='rgb_array', max_episode_steps=200)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ActorCritic(state_dim, config['hidden_dim'], action_dim, config['actor_lr'],
                      config['critic_lr'], config['gamma'], device)
    
    print(f"Using device: {config['device']}")

    train_ActorCritic(agent, env, config)

    swanlab.finish()
        
    