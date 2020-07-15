import gym
import numpy as np


env = gym.make('MountainCar-v0')

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

SHOW_EVERY = 500

DISCRETE_OS_SIZE = [50, 50]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE


epsilon = 0.75
start_epsilon = 1
end_epsilon = 10000
delta_epsilon = epsilon / (end_epsilon - start_epsilon)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_val(state):
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
	if episode % SHOW_EVERY == 0:
		print(f'At episode {episode}')
		render = True
	else:
		render = False

	discrete_state = get_val(env.reset())
	done = False
	while not done:
		if np.random.random() > epsilon:
			action = np.argmax(q_table[discrete_state])
		else:
			action = np.random.randint(0, env.action_space.n)
		new_state, reward, done, _ = env.step(action)
		new_discrete_state = get_val(new_state)

		if render:
			env.render()

		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action, )]
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
			q_table[discrete_state+(action, )] = new_q
		elif new_state[0] >= env.goal_position:
			print(f'Success on episode {episode}')
			q_table[discrete_state+(action,)] = 0
		discrete_state = new_discrete_state

	if end_epsilon >= episode >= start_epsilon:
		epsilon -= delta_epsilon

env.close()