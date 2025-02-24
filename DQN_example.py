import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


#这是我用gpt 生成得一个例子 我准备参照 https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae 这个帖子里的模板


# 创建停车环境
class ParkingEnv(gym.Env):
    def __init__(self):
        super(ParkingEnv, self).__init__()
        self.state = np.array([10, 10, 10])  # 每个区域有 10 个车位
        self.action_space = gym.spaces.Discrete(3)  # 0: 短时停车, 1: 长时停车, 2: PDI 检修
        self.observation_space = gym.spaces.Box(low=0, high=10, shape=(3,), dtype=np.int32)

    def reset(self):
        self.state = np.array([10, 10, 10])
        return self.state

    def step(self, action):
        if action == 0:  # 选择短时停车
            reward = 1
        else:
            reward = -1
       
        self.state[action] -= 1  # 车位减少
        done = np.sum(self.state) == 0  # 如果所有车位都满了，终止
        return self.state, reward, done, {}

env = ParkingEnv()


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),  # 输入层 -> 隐藏层
            nn.ReLU(),
            nn.Linear(128, output_dim)  # 隐藏层 -> 输出层
        )

    def forward(self, x):
        return self.fc(x)  # 输出 Q 值

# 初始化 DQN
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DQN(input_dim=3, output_dim=3).to(device)
target_model = DQN(input_dim=3, output_dim=3).to(device)  # 目标网络
target_model.load_state_dict(model.state_dict())  # 初始同步

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 经验回放
replay_buffer = deque(maxlen=10000)
gamma = 0.9
epsilon = 0.1
batch_size = 32
update_target_every = 10

# 训练 DQN
num_episodes = 500

for episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device)
    done = False

    while not done:
        # ε-贪心策略选择动作
        if random.random() < epsilon:
            action = env.action_space.sample()  # 随机探索
        else:
            with torch.no_grad():
                action = torch.argmax(model(state)).item()  # 选择 Q 值最高的动作

        next_state, reward, done, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)

        # 经验存入 replay buffer
        replay_buffer.append((state, action, reward, next_state, done))

        # 训练 DQN
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*batch)

            batch_states = torch.stack(batch_states)
            batch_actions = torch.tensor(batch_actions, device=device)
            batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
            batch_next_states = torch.stack(batch_next_states)
            batch_dones = torch.tensor(batch_dones, dtype=torch.float32, device=device)

            # 计算目标 Q 值
            with torch.no_grad():
                target_q_values = model(batch_next_states)
                max_target_q_values = torch.max(target_q_values, dim=1)[0]
                target_q = batch_rewards + gamma * max_target_q_values * (1 - batch_dones)

            # 计算当前 Q 值
            current_q_values = model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)

            # 计算损失
            loss = loss_fn(current_q_values, target_q)

            # 反向传播更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 更新目标网络
        if episode % update_target_every == 0:
            target_model.load_state_dict(model.state_dict())

        state = next_state


state = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device)
with torch.no_grad():
    action = torch.argmax(model(state)).item()
print(f"推荐的停车区域: {action}")
