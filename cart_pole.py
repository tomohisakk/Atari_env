from collections import deque
import random
import numpy as np
import gym
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class ReplayBuffer:
	def __init__(self, buffer_size, batch_size):
		self.buffer = deque(maxlen=buffer_size)
		self.batch_size = batch_size

	def add(self, state, action, reward, state_, done):
		data = (state, action, reward, state_, done)
		self.buffer.append(data)

	def __len__(self):
		return len(self.buffer)

	def get_batch(self):
		data = random.sample(self.buffer, self.batch_size)

		states = T.tensor(np.stack([x[0] for x in data]))
		actions = T.tensor(np.array([x[1] for x in data]).astype(np.long))
		rewards = T.tensor(np.array([x[2] for x in data]).astype(np.float))
		states_ = T.tensor(np.stack([x[3] for x in data]))
		dones = T.tensor(np.array([x[4] for x in data]).astype(np.int32))

		return states, actions, rewards, states_, dones


class QNet(nn.Module):
	def __init__(self, action_size):
		super().__init__()
		self.l1 = nn.Linear(4, 128)
		self.l2 = nn.Linear(128, 128)
		self.l3 = nn.Linear(128, action_size)

	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.l3(x)
		return x


class Agent:
	def __init__(self):
		self.lr = 0.0005
		self.gamma = 0.98
		self.buffer_size = 10000
		self.batch_size = 32
		self.action_size = 2
		self.epsilon = 0.1
		
		self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
		self.qnet = QNet(self.action_size)
		self.qnet_target = QNet(self.action_size)
		self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

	def choose_action(self, state):
		if np.random.rand() < self.epsilon:
			return np.random.choice(self.action_size)
		else:
			state = T.tensor(state[np.newaxis, :])
			qs = self.qnet(state)
			return qs.argmax().item()

	def update(self, state, action, reward, state_, done):
		self.replay_buffer.add(state, action, reward, state_, done)
		if len(self.replay_buffer) < self.batch_size:
			return
		
		states, actions, rewards, states_, dones = self.replay_buffer.get_batch()
		qs = self.qnet(states)
		q = qs[np.arange(len(actions)), actions.to(T.long)]

		qs_ = self.qnet_target(states_)
		q_ = qs_.max(1)[0]

		q_.detach()
		target = rewards + (1 - dones) * self.gamma * q_

		loss_fn = nn.MSELoss()
		loss = loss_fn(q, target)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def sync_qnet(self):
		self.qnet_target.load_state_dict(self.qnet.state_dict())


episodes = 200
sync_interval = 20
env = gym.make('CartPole-v0')
agent = Agent()
reward_history = []

for episode in range(episodes):
	done = False
	state = env.reset()
	rewards = 0

	while not done:
		action = agent.choose_action(state)
		state_, reward, done, info = env.step(action)

		agent.update(state, action, reward, state_, done)
		rewards += reward
		state = state_

	if episode % sync_interval == 0:
		agent.sync_qnet()

	reward_history.append(rewards)
	if episode % 10 == 0:
		print('episode ', episode, 'rewards ', rewards)
plt.plot(reward_history)
plt.show()