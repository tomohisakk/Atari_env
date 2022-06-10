import gym
import matplotlib.pyplot as plt
import numpy as np
import numpy as np

class Agent():
	def __init__(self):
		self.lr = 0.001
		self.gamma = 0.9
		self.n_actions = 4
		self.n_states = 16
		self.epsilon = 1.0
		self.eps_min = 0.01
		self.eps_dec = 0.9999995

		self.Q = {}
		self.init_Q()

	def init_Q(self):
		for state in range(self.n_states):
			for action in range(self.n_actions):
				self.Q[(state, action)] = 0.0

	def choose_action(self, state):
		if np.random.random() < self.epsilon:
			action = np.random.choice([i for i in range(self.n_actions)])
		else:
			actions = np.array([self.Q[(state, a)] for a in range(self.n_actions)])
			action = np.argmax(actions)

		return action

	def decrement_epsilon(self):
		self.epsilon = self.epsilon * self.eps_dec  if self.epsilon > self.eps_min else self.eps_min

	def learn(self, state, action, reward, state_):
		actions = np.array([self.Q[(state_, a)] for a in range(self.n_actions)])
		a_max = np.argmax(actions)
		self.Q[(state, action)] += self.lr * (reward + self.gamma * self.Q[(state_, a_max)] - self.Q[(state, action)])
		self.decrement_epsilon()


n_games = 500000
env = gym.make('FrozenLake-v1')
agent = Agent()
scores = []
win_pct_list = []

for i in range(n_games):
	done = False
	observation = env.reset()
	score = 0
	while not done:
		action = agent.choose_action(observation)
		observation_, reward, done, info = env.step(action)
		agent.learn(observation, action, reward, observation_)
		score += reward
		observation = observation_
	scores.append(score)
	if i % 100 == 0:
		win_pct = np.mean(scores[-100:])
		win_pct_list.append(win_pct)
		if i % 1000 == 0:
			print('episode ', i, 'rewards %.2f' % win_pct,
				  'epsilon %.2f' % agent.epsilon)
plt.plot(win_pct_list)
plt.show()