from Simple_PacMan import SimplePacMan

env = SimplePacMan()
n_actions = 20
over=False
goal=False

for i_episode in range(20):

	if over: print('*******MAX ACTION******')
	if goal: print('********GOAL*********')
	
	over=False
	goal=False

	observation, _, _, info = env.reset()
	print(info['curr_action'],observation)
	
	for t in range(n_actions):
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		print(info['curr_action'],observation)
		if done:
			print('*******GHOST*******')
			break
		if t==n_actions-1: over=True
		if observation[2] == 0: goal=True 