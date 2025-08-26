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
    
    def push(self, reward: float, log_prob: float) -> None:
        self.buffer.append((reward, log_prob))

    def sample(self) -> Tuple[Tuple, Tuple]:
        reward, log_prob = zip(*self.buffer)
        return reward, log_prob
    
    def clear(self) -> int:
        return self.buffer.clear()
    
class PolicyNet(nn.Module):
    """ 策略网络 """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class REINFORCE:
    """ REINFORCE算法的智能体实现 """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, learning_rate: float, 
                 gamma: float, device: torch.device) -> None:
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.device = device

    def take_action(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.policy_net(state_tensor)
        action_dist = Categorical(probs)
        action = action_dist.sample() # 采样动作
        return action.item(), action_dist.log_prob(action)  # 返回动作和其对数概率

    def update(self, transition_dict: Dict[str, any]) -> float:
        reward_tuple = transition_dict['rewards']
        log_probs_tuple = transition_dict['log_probs']

        G = 0             # 从轨迹末尾，反向计算每个时间步的折扣奖励
        policy_loss = []
        returns = []
        for r, log_prob in zip(reward_tuple[::-1], log_probs_tuple[::-1]):
            G = r + self.gamma * G
            returns.insert(0, G)
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()  # 损失函数
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        return policy_loss.item()
    
def evaluate(agent: REINFORCE, env: gym.Env, num_episodes: int) -> List[float]:
    """ 评估策略 """
    rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = agent.take_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
    return sum(rewards) / len(rewards)  # 返回平均奖励

def train_REINFORCE(agent: REINFORCE, env: gym.Env, config: dict, replay_buffer: ReplayBuffer) -> None:
    print("--- 开始训练 ---")
    return_list: List[float] = []
    total_steps = 0
    i_episode = 0

    while total_steps < config['total_timesteps']:
        state, _ = env.reset()
        done = False
        episode_return = 0
        i_episode += 1
        log_probs = []
        rewards = []

        while not done:
            action, log_prob = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            replay_buffer.push(reward, log_prob)
            state = next_state
            episode_return += reward
            total_steps += 1
        
        rewards, log_probs = replay_buffer.sample()
        transition_dict = {
            'rewards': rewards,
            'log_probs': log_probs
        }
        loss = agent.update(transition_dict)  # 更新策略
        replay_buffer.clear()  # 清空回放池

        swanlab.log({'Train/loss': loss}, step=total_steps)
        swanlab.log({"Return/by_Episode": episode_return}, step=i_episode)
        swanlab.log({"Return/by_Step": episode_return}, step=total_steps)

        if i_episode % config['eval_freq'] == 0:
            eval_reward = evaluate(agent, env, num_episodes=5)
            swanlab.log({"Eval/Average_Return": eval_reward}, step=total_steps)
            print(f"Episode: {i_episode}, Steps: {total_steps}/{config['total_timesteps']}, "
                  f"Eval Avg Return: {eval_reward:.2f}")
    
    env.close()
    print("--- 训练结束 ---")

if __name__ == "__main__":
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    run = swanlab.init(
        project= "REINFORCE-Gym",
        experiment_name="REINFORCE Training",
        config={
            "lr": 1e-3,
            "total_timesteps": 60000,
            "gamma": 0.99,
            "hidden_dim": 128,
            "eval_freq": 20,
            "env_name": 'CartPole-v1',
            "seed": 1,
            "device": device_str
        }
    )

    config = run.config
    
    set_seeds(config['seed'])
    replay_buffer = ReplayBuffer()

    env = gym.make(config['env_name'], render_mode='rgb_array', max_episode_steps=200)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCE(state_dim, config['hidden_dim'], action_dim, config['lr'], 
                      config['gamma'], device)

    # 开始训练
    train_REINFORCE(agent, env, config, replay_buffer)

    swanlab.finish()