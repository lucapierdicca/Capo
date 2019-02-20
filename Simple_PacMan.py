import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import random

GRID_SIZE = 5
N_GHOSTS = 4
START_COORD = (GRID_SIZE-1,0)

#========= MDP ========

#FULLY OBSERVABLE

#set of actions A
AID_TO_LABEL = {0:'Down',
                1:'Up',
                2:'Left',
                3:'Right'}

#set of state S
#|S| = GRID_SIZE * GRID_SIZE * 2^(np.power(GRID_SIZE,2)-N_GHOSTS-1)
#lo stato è composto da [coord_x_pacman, cord_y_pacman, Fi]
#coordinata x e coordinata y della posizione di pacman 
#e presenza/assenza di cibo nelle caselle senza ghosts 
#dove coord_x_pacman € {0, ... , GRID_SIZE-1}
#     coord_y_pacman € {0, ... , GRID_SIZE-1}
#                 Fi € {True,False} con i = 1, ... ,np.power(GRID_SIZE,2)-N_GHOSTS-1

#transition model
#P(s'|s,a)
#the transition model entry is implicitly defined to be 1.0 for all adjacent cell w.r.t the current one 


#reward function
#R(s)


class SimplePacMan(gym.Env):
    metadata = {
        'render.modes': ['ansi'],
    }

    def __init__(self):
        #self.seed()
        self.obs_space = spaces.Box(low=np.hstack((np.zeros((2,)),np.zeros((np.power(GRID_SIZE,2)-N_GHOSTS),))), 
                                    high=np.hstack((np.array([GRID_SIZE-1,GRID_SIZE-1]) , np.ones((np.power(GRID_SIZE,2)-N_GHOSTS),))),
                                    dtype=int)
        self.action_space = spaces.Discrete(4)
        self.obs = np.hstack((np.zeros((2,),dtype=np.uint8),np.zeros((np.power(GRID_SIZE,2)-N_GHOSTS),dtype=np.uint8)))

    def seed(self, seed=None):
        return 0

    def reset(self):

        grid = np.ones((GRID_SIZE,GRID_SIZE),dtype=np.uint8)
        
        grid[START_COORD] = 3

        random.seed(2)
        ghosts_coord = []
        for _ in range(N_GHOSTS):
            x = random.randint(0, GRID_SIZE-1)
            y = random.randint(0, GRID_SIZE-1)
            if (x,y) not in ghosts_coord:
                ghosts_coord.append((x,y))

        for g_x,g_y in ghosts_coord:
            grid[g_x,g_y] = 2

        self.ghosts_coord = ghosts_coord
        self.grid = grid

        return self.step(-1)
    
    def step(self, action):
        reward=0
        done=False
        info={'curr_action':''}
        obs = self.obs
        grid = self.grid
        
        # NOOP
        if action==-1:
            obs[0] = START_COORD[0]
            obs[1] = START_COORD[1]

            flat_grid = list(self.grid.ravel())
            flat_grid = [i for i in flat_grid if i!=2]
            flat_grid = [i if i!=3 else 0 for i in flat_grid]
            for i,e in enumerate(flat_grid):
                obs[i+2] = e
            info['curr_action'] = 'Start'

        # DOWN
        elif action==0:
            prev_r = obs[0]
            if obs[0] != GRID_SIZE-1:
                obs[0]+=1
                if (obs[0],obs[1]) in self.ghosts_coord:
                    done=True
                else:
                    self.grid[prev_r,obs[1]] = 0
                    self.grid[obs[0],obs[1]] = 3
                    flat_grid = list(self.grid.ravel())
                    flat_grid = [i for i in flat_grid if i!=2]
                    flat_grid = [i if i!=3 else 0 for i in flat_grid]
                    for i,e in enumerate(flat_grid):
                        obs[i+2] = e                  


            info['curr_action'] = AID_TO_LABEL[0]
        
        # UP
        elif action==1:
            prev_r = obs[0]
            if obs[0] != 0:
                obs[0]-=1
                if (obs[0],obs[1]) in self.ghosts_coord:
                    done=True
                else:
                    self.grid[prev_r,obs[1]] = 0
                    self.grid[obs[0],obs[1]] = 3

                    flat_grid = list(self.grid.ravel())
                    flat_grid = [i for i in flat_grid if i!=2]
                    flat_grid = [i if i!=3 else 0 for i in flat_grid]
                    for i,e in enumerate(flat_grid):
                        obs[i+2] = e
            info['curr_action'] = AID_TO_LABEL[1]
        
        # LEFT
        elif action==2:
            prev_c = obs[1]
            if obs[1] != 0:
                obs[1]-=1
                if (obs[0],obs[1]) in self.ghosts_coord:
                    done=True
                else:
                    self.grid[obs[0],prev_c] = 0
                    self.grid[obs[0],obs[1]] = 3
                    flat_grid = list(self.grid.ravel())
                    flat_grid = [i for i in flat_grid if i!=2]
                    flat_grid = [i if i!=3 else 0 for i in flat_grid]
                    for i,e in enumerate(flat_grid):
                        obs[i+2] = e
            info['curr_action'] = AID_TO_LABEL[2]
        
        # RIGHT
        elif action==3:
            prev_c = obs[1]
            if obs[1] != GRID_SIZE-1:
                obs[1]+=1
                if (obs[0],obs[1]) in self.ghosts_coord:
                    done=True
                else:
                    self.grid[obs[0],prev_c] = 0
                    self.grid[obs[0],obs[1]] = 3
                    flat_grid = list(self.grid.ravel())
                    flat_grid = [i for i in flat_grid if i!=2]
                    flat_grid = [i if i!=3 else 0 for i in flat_grid]
                    for i,e in enumerate(flat_grid):
                        obs[i+2] = e
            info['curr_action'] = AID_TO_LABEL[3]

        self.obs = obs
        self.grid = grid

        return obs,reward,done,info

    def render(self, mode='ansi', close=False):
        print(np.array_str(self.grid))