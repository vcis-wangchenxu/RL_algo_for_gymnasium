""" Dueling DQN 在Pendulum-v1环境上训练 """
import random
import numpy as np
from collections import deque
from typing import Dict, Tuple, List
import math
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

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
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, Tuple, Tuple, np.ndarray, Tuple]:
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    
    def size(self) -> int:
        return len(self.buffer)

class VAnet(nn.Module):
    """ 价值网络 """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(VAnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_A = nn.Linear(hidden_dim, action_dim)
        self.fc_V = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean(dim=1, keepdim=True)
        return Q
    
class DQN:
    """ Dueling DQN算法的智能体实现 """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, learning_rate: float, 
                 gamma: float, target_update: int, device: torch.device, dqn_type: str = 'VanillaDQN') -> None:
        self.action_dim = action_dim
        self.q_net = VAnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net = VAnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()  # 将损失函数定义为类的一个属性
        self.gamma = gamma
        self.target_update = target_update
        self.device = device
        self.dqn_type = dqn_type  # 添加dqn_type属性以区分
        self.learn_step_counter = 0

    @torch.no_grad()
    def take_action(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            return self.q_net(state_tensor).argmax(dim=1).item()
    
    @torch.no_grad()
    def max_q_value(self, state: np.ndarray) -> float:
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        return self.q_net(state_tensor).max(dim=1)[0].item()

    def update(self, transition_dict: Dict[str, any]) -> float:
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).unsqueeze(1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).unsqueeze(1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)   
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).unsqueeze(1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        if self.dqn_type == 'DoubleDQN':
            max_action = self.q_net(next_states).max(1)[1].unsqueeze(1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action).detach()
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].unsqueeze(1).detach()
        q_target = rewards + self.gamma * max_next_q_values * (1 - dones)   

        dqn_loss = self.criterion(q_values, q_target)  # 计算损失

        self.optimizer.zero_grad()
        dqn_loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数

        if self.learn_step_counter % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.learn_step_counter += 1
        return dqn_loss.item()  

def dis_to_con(discrete_action: int, env, action_dim: int) -> float:
    """ 将离散动作转换为连续动作 """
    action_lowbound = env.action_space.low[0] # 获取连续动作空间的下界
    action_upbound = env.action_space.high[0] # 获取连续动作空间的上界
    return action_lowbound + (discrete_action / (action_dim - 1)) * (action_upbound - action_lowbound)

def train_DQN(agent, env, total_timesteps: int, epsilon_start: float, epsilon_end: float, 
              epsilon_decay_steps: int, buffer_size: int, minimal_size: int, batch_size: int, 
              moving_avg_window: int, replaybuffer: ReplayBuffer) -> None:
    """ 训练DQN智能体 """
    return_list: List[float] = []
    max_q_value_list: List[float] = []
    max_q_value = 0.0
    total_steps = 0
    i_episode = 0

    # 主循环基于总步数
    while total_steps < total_timesteps:
        episode_return = 0.0
        state, _ = env.reset()  # 重置环境
        done = False
        i_episode += 1

        while not done:
            # 计算当前步数对应的epsilon
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                math.exp(-1. * total_steps / epsilon_decay_steps)
            
            action = agent.take_action(state, epsilon)
            max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995  # 获取最大Q值
            max_q_value_list.append(max_q_value)
            action_continuous = dis_to_con(action, env, agent.action_dim)  # 将离散动作转换为连续动作
            next_state, reward, terminated, truncated, _ = env.step([action_continuous])
            done = terminated or truncated

            replaybuffer.push(state, action, reward, next_state, done)  # 存储经验
            state = next_state
            episode_return += reward
            total_steps += 1

            if replaybuffer.size() >= minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)  # 采样批次
                transition_dict = {
                    'states': b_s, 'actions': b_a,
                    'rewards': b_r, 'next_states': b_ns, 'dones': b_d
                }
                loss = agent.update(transition_dict)
                writer.add_scalar('Train/Loss', loss, agent.learn_step_counter)

        return_list.append(episode_return)
        writer.add_scalar('Return/by_Episode', episode_return, i_episode)
        writer.add_scalar('Return/by_Step', episode_return, total_steps)
        writer.add_scalar('Train/Epsilon_by_Step', epsilon, total_steps)
        writer.add_scalar('Train/Max_Q_Value', max_q_value, total_steps)

        # 计算并记录移动平均回报
        if len(return_list) >= moving_avg_window:
            # 为了让曲线更平滑，可以每隔一定回合数再记录一次均值
            if i_episode % 10 == 0:
                avg_return = np.mean(return_list[-moving_avg_window:])
                writer.add_scalar('Train/Average_Return', avg_return, total_steps)
                print(f"Episode: {i_episode}, Steps: {total_steps}/{total_timesteps}, "
                      f"Avg Return: {avg_return:.2f}, Epsilon: {epsilon:.3f}, Max Q Value: {max_q_value:.2f}")
    
    env.close()
    writer.close()

if __name__ == "__main__":
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    config = {
        "lr": 1e-3,
        "total_timesteps": 80000, # 使用总步数作为终止条件
        "hidden_dim": 128,
        "gamma": 0.98,
        "epsilon_start": 0.9,
        "epsilon_end": 0.01,
        "epsilon_decay_steps": 10000, # 控制衰减速度的步数
        "target_update": 100, # 目标网络更新频率可以适当增加
        "buffer_size": 10000,
        "minimal_size": 1000, # 初始收集更多经验再开始学习
        "batch_size": 64,
        "moving_avg_window": 20,
        "env_name": 'Pendulum-v1',
        "seed": 1,
        "device": device_str
    }

    set_seeds(config['seed'])

    env = gym.make(config['env_name'], render_mode='rgb_array', max_episode_steps=200)
    state_dim = env.observation_space.shape[0]
    action_dim = 11 # 离散化动作空间为11个离散动作

    agent = DQN(state_dim, config['hidden_dim'], action_dim, config['lr'],
                config['gamma'], config['target_update'], device, dqn_type='DuelingDQN')
    replay_buffer = ReplayBuffer(config['buffer_size'])

    log_path = f"runs/DuelingDQN_{config['env_name']}_seed{config['seed']}"
    writer = SummaryWriter(log_path)
    print(f"Log path: {log_path}")
    print(f"Using device: {config['device']}")
    writer.add_hparams(config, {"Train/Average_Return": 0})

    train_DQN(agent, env, config['total_timesteps'], config['epsilon_start'], config['epsilon_end'],
              config['epsilon_decay_steps'], config['buffer_size'], config['minimal_size'], config['batch_size'],
              config['moving_avg_window'], replay_buffer)
    