import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Define DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define TDQN Agent
class TDQN_Agent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        return torch.argmax(self.model(state)).item()

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(torch.FloatTensor(next_state).unsqueeze(0))).item()
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0))
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(torch.FloatTensor(state).unsqueeze(0)), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Function to Train Agent
def train_agent(agent, data, episodes=50):
    episode_rewards = []
    for episode in range(episodes):
        state = data.iloc[0].values
        total_reward = 0
        for i in range(1, len(data)):
            action = agent.act(state)
            next_state = data.iloc[i].values
            reward = data['Return'].iloc[i] if action == 1 else -data['Return'].iloc[i]
            done = i == len(data) - 1
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        agent.replay()
        episode_rewards.append(total_reward)
    return episode_rewards

# Function to Get Cumulative Rewards
def get_tdqn_rewards(stock_name, stock_data):
    stock_data['Cumulative Reward'] = stock_data['Return'].cumsum()
    return stock_data[['Date', 'Cumulative Reward']]

# Example Usage (To Be Run Separately)
if __name__ == "__main__":
    # Load example dataset
    data = pd.read_csv("example_stock_data.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    data['Return'] = data['Close'].pct_change().fillna(0)

    # Initialize and train agent
    state_dim = len(data.columns) - 1  # Excluding Date column
    agent = TDQN_Agent(state_dim, 3)
    train_agent(agent, data)
    
    # Get cumulative rewards
    reward_data = get_tdqn_rewards("ExampleStock", data)
    print(reward_data.head())