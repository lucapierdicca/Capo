from Simple_PacMan import SimplePacMan
import numpy as np
#define the environment
env = SimplePacMan()

#n of actions to perform
n_actions = 100 
over=False
goal=False

#let's try 5 episode (each time we hit a ghost a new episode starts)
for i_episode in range(5):

	if over: print('*******MAX ACTION******')
	if goal: print('********GOAL*********')
	
	over=False
	goal=False

	#initialize the environment
	observation, _, _, info = env.reset()
	print(info['curr_action'],observation)
	
	for t in range(n_actions):
		#basic rendering [3-pacman,2-ghost,1-food,0-nofood]
		env.render()
		
		#random sample an action in A
		action = env.action_space.sample()
		
		#execute the action to get the s'
		observation, reward, done, info = env.step(action)
		print(info['curr_action'],observation)
		
		if done:
			print('*******GHOST*******')
			break
		if t==n_actions-1: over=True
		if not np.count_nonzero(observation[2:]) >= 1: goal= True 



