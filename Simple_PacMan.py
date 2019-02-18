import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import random

GRID_SIZE = 5
N_GHOSTS = 4
START_COORD = (GRID_SIZE-1,0)

#actions
AID_TO_LABEL = {0:'Down',
                1:'Up',
                2:'Left',
                3:'Right'}

class SimplePacMan(gym.Env):
    metadata = {
        'render.modes': ['human', 'ansi'],
    }

    def __init__(self):
        #self.seed()
        self.obs_space = spaces.Box(low=np.array([0,0,0]), high=np.array([GRID_SIZE,GRID_SIZE,np.power(GRID_SIZE,2)-N_GHOSTS-1]),dtype=int)
        self.action_space = spaces.Discrete(4)
        self.obs = np.zeros((3,),dtype=np.uint8)

    def seed(self, seed=None):
        return 0

    def random_color(self):
        return np.array([
            self.np_random.randint(low=0, high=255),
            self.np_random.randint(low=0, high=255),
            self.np_random.randint(low=0, high=255),
            ]).astype('uint8')

    def reset(self):

        grid = np.ones((GRID_SIZE,GRID_SIZE),dtype=np.uint8)
        
        grid[START_COORD] = 3

        random.seed(1)
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
        
        if action==-1:
            obs[0] = START_COORD[0]
            obs[1] = START_COORD[1]
            obs[2] = np.power(GRID_SIZE,2)-N_GHOSTS-1
            info['curr_action'] = 'Start'
        elif action==0:
            prev = obs[0]
            if obs[0] != GRID_SIZE-1:
                obs[0]+=1
                if (obs[0],obs[1]) in self.ghosts_coord:
                    done=True
                else:
                    obs[2]-=1
                    self.grid[prev,obs[1]] = 0
                    self.grid[obs[0],obs[1]] = 3
            info['curr_action'] = AID_TO_LABEL[0]
        elif action==1:
            prev = obs[0]
            if obs[0] != 0:
                obs[0]-=1
                if (obs[0],obs[1]) in self.ghosts_coord:
                    done=True
                else:
                    obs[2]-=1
                    self.grid[prev,obs[1]] = 0
                    self.grid[obs[0],obs[1]] = 3
            info['curr_action'] = AID_TO_LABEL[1]
        elif action==2:
            prev = obs[1]
            if obs[1] != 0:
                obs[1]-=1
                if (obs[0],obs[1]) in self.ghosts_coord:
                    done=True
                else:
                    obs[2]-=1
                    self.grid[obs[0],prev] = 0
                    self.grid[obs[0],obs[1]] = 3
            info['curr_action'] = AID_TO_LABEL[2]
        elif action==3:
            prev = obs[1]
            if obs[1] != GRID_SIZE-1:
                obs[1]+=1
                if (obs[0],obs[1]) in self.ghosts_coord:
                    done=True
                else:
                    obs[2]-=1
                    self.grid[obs[0],prev] = 0
                    self.grid[obs[0],obs[1]] = 3
            info['curr_action'] = AID_TO_LABEL[3]

        self.obs = obs
        self.grid = grid

        return obs,reward,done,info

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if mode == 'rgb_array':
            return self.last_obs

        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(self.last_obs)
            return self.viewer.isopen

        else:
            assert 0, "Render mode '%s' is not supported" % mod