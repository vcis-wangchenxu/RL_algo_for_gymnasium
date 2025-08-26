import random
import gymnasium as gym
import numpy as np
from collections import deque
from typing import Dict, Tuple, List
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
    def __init__(self) -> None:
        self.buffer = deque()
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, log_prob: float) -> None:
        # PPO需要存储log_prob, 所以在push时加入
        self.buffer.append((state, action, reward, next_state, done, log_prob))

    def sample(self):
        state, action, reward, next_state, done, log_prob = zip(*self.buffer)
        return np.array(state), action, reward, np.array(next_state), done, log_prob
    
    def ready(self, rollout_steps: int) -> bool:
        """判断是否可以取 batch"""
        return len(self.buffer) >= rollout_steps

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

class ActorNet(nn.Module):
    """ Actor的策略网络 """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

class CriticNet(nn.Module):
    """ Critic的价值网络 """
    def __init__(self, state_dim: int, hidden_dim: int):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PPO:
    """ PPO 算法的智能体实现 """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, actor_lr: float,
                 critic_lr: float, lmbda: float, epochs: int, eps: float, gamma: float,
                 device: torch.device, batch_size: int):
        self.actor = ActorNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = CriticNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda  # GAE参数
        self.epochs = epochs  # 一条序列数据用来训练的轮数
        self.eps = eps  # PPO-Clip的截断参数
        self.device = device
        self.batch_size = batch_size
        self.count = 0

    @torch.no_grad()
    def take_action(self, state: np.ndarray) -> tuple[int, float]:
        """ 根据策略分布采样一个动作 """
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state_tensor)
        action_dist = Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob.item()

    @torch.no_grad()  # 评估时采用确定性策略
    def act_deterministic(self, state: np.ndarray) -> int:
        s = torch.tensor(np.array([state]), dtype=torch.float, device=self.device)
        probs = self.actor(s)
        return torch.argmax(probs, dim=1).item()

    def update(self, transitions: dict):
        """ 使用收集到的序列数据更新网络 """
        states = torch.tensor(transitions['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transitions['actions'], dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(transitions['rewards'], dtype=torch.float).unsqueeze(1).to(self.device)
        next_states = torch.tensor(transitions['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transitions['dones'], dtype=torch.float).unsqueeze(1).to(self.device)
        old_log_probs = torch.tensor(transitions['log_probs'], dtype=torch.float).unsqueeze(1).to(self.device)

        with torch.no_grad():
            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
            td_delta = td_target - self.critic(states)
            advantages = torch.zeros_like(rewards).to(self.device)
            advantage = 0.0
            for i in reversed(range(len(rewards))):
                advantage = self.gamma * self.lmbda * advantage * (1 - dones[i]) + td_delta[i]
                advantages[i] = advantage
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = states.size(0)
        for _ in range(self.epochs):
            indices = torch.randperm(T, device=self.device)
            for start in range(0, T, self.batch_size):
                mb_idx = indices[start: start + self.batch_size]
                
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx].squeeze()
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_td_target = td_target[mb_idx]

                dist = Categorical(self.actor(mb_states))
                new_log_probs = dist.log_prob(mb_actions).unsqueeze(1)
                entropy = dist.entropy().mean()
                state_values = self.critic(mb_states)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * mb_advantages
            
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(state_values, mb_td_target)
                entropy_loss = -entropy

                total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                self.count += 1
                swanlab.log({
                    "Train/Actor_Loss": actor_loss.item(),
                    "Train/Critic_Loss": critic_loss.item(),
                    "Train/Total_Loss": total_loss.item()
                }, step=self.count)

def evaluate(agent: PPO, env: gym.Env, n_episodes: int = 10) -> float:
    """ 评估智能体的性能 """
    total_return = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            action = agent.act_deterministic(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_return += reward
        total_return += episode_return
    return total_return / n_episodes

def train_PPO(agent: PPO, env: gym.Env, config: dict):
    print("--- 开始训练 ---")
    total_steps = 0
    i_episode = 0
    
    # 在训练主函数中实例化ReplayBuffer
    replay_buffer = ReplayBuffer()

    while total_steps < config['total_timesteps']:
        state, _ = env.reset()
        done = False
        ep_return = 0.0
        i_episode += 1

        while not done:
            action, log_prob = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 使用replay_buffer.push()存储数据
            replay_buffer.push(state, action, reward, next_state, done, log_prob)

            state = next_state
            ep_return += reward
            total_steps += 1
        
            # 当缓冲区到达 rollout_steps 时更新
            if replay_buffer.ready(config['rollout_steps']):
                states, actions, rewards, next_states, dones, log_probs = replay_buffer.sample()
                transitions = {
                    'states': states, 'actions': actions, 'rewards': rewards,
                    'next_states': next_states, 'dones': dones, 'log_probs': log_probs
                }
                # 使用收集到的数据进行更新
                agent.update(transitions)
                replay_buffer.clear()

        swanlab.log({"Return/by_Episode": ep_return}, step=i_episode)
        swanlab.log({"Return/by_Step": ep_return}, step=total_steps)

        if i_episode % config['eval_freq'] == 0:
            eval_reward = evaluate(agent, env, n_episodes=10)
            swanlab.log({"Eval/Average_Return": eval_reward}, step=total_steps)
            print(f"回合: {i_episode}, 步数: {total_steps}/{config['total_timesteps']}, "
                  f"评估平均回报: {eval_reward:.2f}")

    env.close()
    print("--- 训练结束 ---")

if __name__ == "__main__":
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    run = swanlab.init(
        project="PPO_CartPole",
        name=f"PPO_CartPole_Buffer_{device_str}",
        config={
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "total_timesteps": 80000,
            "hidden_dim": 128,
            "gamma": 0.99,
            "lmbda": 0.95,
            "epochs": 10,
            "eps_clip": 0.2,
            "eval_freq": 20,
            "env_name": 'CartPole-v1',
            "seed": 0,
            "device": device_str,
            "algorithm": "PPO",
            "batch_size": 64,
            "rollout_steps": 2048
        },
    )
    config = swanlab.config
    set_seeds(config['seed'])

    env = gym.make(config['env_name'], render_mode='rgb_array', max_episode_steps=200)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPO(state_dim, config['hidden_dim'], action_dim, 
                config['actor_lr'], config['critic_lr'], 
                config['lmbda'], config['epochs'],
                config['eps_clip'], config['gamma'], 
                device, batch_size=config['batch_size'])

    print(f"使用设备: {config['device']}")
    train_PPO(agent, env, config)
    swanlab.finish()