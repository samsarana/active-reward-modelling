"""
https://github.com/awjuliani/DeepRL-Agents/blob/master/gridworld.py
"""

import math, itertools, gym
from PIL import Image
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from gym.utils import seeding
from time import sleep

class gameOb():
    def __init__(self,coordinates,size,intensity,channel,reward,name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name
        
class UpsampledGridworldEnv(gym.Env): # sam subclassed gameEnv as gym.Env
    def __init__(self, partial=False, size=5, random_resets=True, terminate_ep_if_done=False, n_goals=1, n_lavas=1): # sam added defaults
        """
        partial: does agent get partial observations or full state?
        size: length and width of square grid.
        random_resets: if True, upon each call to env.reset(),
                       `gameOb`s will be placed in new random positions
                       Else, they will be placed in the same,
                       fixed positions.
        """
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial
        self.random_resets = random_resets
        self.n_goals = n_goals
        self.n_lavas = n_lavas
        # Sam added the following three lines
        self.action_space = spaces.Discrete(4) # Discrete(n) is a discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`
        # self.observation_space = spaces.Box(low=0, high=255, shape=(84,84,3)) # Box represents the Cartesian product of n closed intervals
        self.observation_space = spaces.Box(low=0, high=255, shape=(75,)) # Box represents the Cartesian product of n closed intervals
        # 255 comes from scipy.misc.imresize
        self.determined_locations = {
            'hero': (0,0),
            'goal': (0,4),
            'goal2': (2,3),
            'goal3': (4,2),
            'goal4': (4,0),
            'fire': (1,2),
            'fire2': (3,1)
        }
        self.terminate_ep_if_done = terminate_ep_if_done
        if not random_resets:
            assert terminate_ep_if_done,\
                '''If you want to reset objects deterministically,
                the game becomes very easy if episodes don't terminate
                when agent reaches goal/lava'''
        self.reset()
        # plt.imshow(self.state, interpolation="nearest")
        
    def reset(self):
        self.objects = []
        hero = gameOb(self.newPosition('hero'),1,1,2,None,'hero')
        self.objects.append(hero)
        bug = gameOb(self.newPosition('goal'),1,1,1,1,'goal')
        self.objects.append(bug)
        hole = gameOb(self.newPosition('fire'),1,1,0,-1,'fire')
        self.objects.append(hole)
        if self.n_goals >= 2:
            bug2 = gameOb(self.newPosition('goal2'),1,1,1,1,'goal2')
            self.objects.append(bug2)
        if self.n_goals >= 3:
            bug3 = gameOb(self.newPosition('goal3'),1,1,1,1,'goal3')
            self.objects.append(bug3)
        if self.n_goals >= 4:
            bug4 = gameOb(self.newPosition('goal4'),1,1,1,1,'goal4')
            self.objects.append(bug4)
        if self.n_lavas >= 2:
            hole2 = gameOb(self.newPosition('fire2'),1,1,0,-1,'fire2')
            self.objects.append(hole2)
        if self.n_goals > 4 or self.n_lavas > 2:
            raise NotImplementedError("Using more than 4 goals or 2 lavas is undefined.")
        state = self.renderEnv()
        self.state = state
        self.done = False
        return self._get_obs()

    def moveChar(self,direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        penalize = 0. # penalty for taking a step
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY-2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX-2:
            hero.x += 1     
        if hero.x == heroX and hero.y == heroY:
            penalize = 0.0
        self.objects[0] = hero
        return penalize
    
    def newPosition(self, ob_name=None):
        iterables = [ range(self.sizeX), range(self.sizeY) ]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x,objectA.y) not in currentPositions:
                currentPositions.append((objectA.x,objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        if self.random_resets:
            loc_index = np.random.choice(range(len(points)),replace=False)
            location = points[loc_index]
        else: # reset `gameOb`s determinstically
            try:
                location = self.determined_locations[ob_name]
            except KeyError:
                raise KeyError("{} is not a valid ob_name!".format(ob_name))
            assert location in points, "You can't place object {} at location {}, it's already occupied!".format(
                ob_name, location)
        return location

    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        done = False
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                # if other.reward == 1:
                if self.terminate_ep_if_done:
                    done = True
                else: # don't terminate; add the goal or lava back in at a new point
                    if 'goal' in other.name:
                        self.objects.append(gameOb(self.newPosition(other.name),
                        1, 1, 1, 1, other.name))
                    else:
                        assert 'fire' in other.name
                        self.objects.append(gameOb(self.newPosition(other.name),
                        1, 1, 0, -1, other.name))
                return other.reward, done
        # hero moved to unoccupied cell
        return 0., done

    def renderEnv(self):
        #a = np.zeros([self.sizeY,self.sizeX,3])
        a = np.ones([self.sizeY+2,self.sizeX+2,3])
        a[1:-1,1:-1,:] = 0
        hero = None
        for item in self.objects:
            a[item.y+1:item.y+item.size+1, item.x+1:item.x+item.size+1, item.channel] = item.intensity
            if item.name == 'hero':
                hero = item
        if self.partial == True:
            a = a[hero.y:hero.y+3,hero.x:hero.x+3,:]
        b = scipy.misc.imresize(a[:,:,0],[84,84,1],interp='nearest')
        # b = np.array(Image.fromarray(a[:,:,0]).resize([84,84,1]))#, resample=PIL.Image.NEAREST)) # maybe include this too if rendering doesn't work -- or revert to SciPy 1.0.0 where scipy.misc.imresize is not deprecated
        c = scipy.misc.imresize(a[:,:,1],[84,84,1],interp='nearest')
        # c = np.array(Image.fromarray(a[:,:,1]).resize([84,84,1]))
        d = scipy.misc.imresize(a[:,:,2],[84,84,1],interp='nearest')
        # d = np.array(Image.fromarray(a[:,:,2]).resize([84,84,1]))
        a = np.stack([b,c,d],axis=2)
        return a

    def step(self,action):
        assert not self.done,\
            '''You are calling 'step()' even though this environment has already returned done = True.
            You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.'''
        penalty = self.moveChar(action)
        reward, done = self.checkGoal()
        state = self.renderEnv()
        self.state = state
        self.done = done
        assert reward != None, "Hmm, reward is None. What's going on?"
        return self._get_obs(), (reward + penalty), done, {} # sam added {} - empty info


    def render(self, mode):
        """Sam added this function.
           Not sure it does quite what gym does...
           In particular, I can't recall if gym
           closes automatically
           Also, if partial=True, this renders
           observation NOT state
        """
        state = self.renderEnv() # self.state would seem to be better code, I'm not 100% sure that self.state is updated when it should be
        plt.imshow(state, interpolation="nearest")
        if mode == 'human':
            plt.show() # show() is a blocking function: code will hang until you x the matplotlib popup window
        elif mode == 'nonblock':
            plt.show(block=False)
            plt.pause(1)
            plt.close()
        else:
            raise NotImplementedError("Haven't specified what rendering in mode {} means".format(mode))

    def _get_obs(self):
        """
        Move channel to dim 0 to (in order to use with pytorch-style cnn)
        """
        return np.moveaxis(self.state, -1, 0)
        # return np.reshape(self.state,[21168]) # 21168 = 84*84*3


class GridworldEnv(UpsampledGridworldEnv):
    def __init__(self, partial=False, size=5, random_resets=True, terminate_ep_if_done=True): # sam added defaults
        """
        partial: does agent get partial observations or full state?
        size: length and width of square grid.
        random_resets: if True, upon each call to env.reset(),
                       `gameOb`s will be placed in new random positions
                       Else, they will be placed in the same,
                       fixed positions.
        """
        super().__init__()
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial
        self.random_resets = random_resets
        self.action_space = spaces.Discrete(4) # Discrete(n) is a discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`
        self.observation_space = spaces.Box(low=0, high=255, shape=(75,)) # Box represents the Cartesian product of n closed intervals
        self.determined_locations = {
            'hero': (0,0),
            'goal': (0,4),
            'goal2': (2,3),
            'goal3': (4,2),
            'goal4': (4,0),
            'fire': (1,2),
            'fire2': (3,1)
        }
        self.terminate_ep_if_done = terminate_ep_if_done
        if not random_resets:
            assert terminate_ep_if_done,\
                '''If you want to reset objects deterministically,
                the game becomes very easy if episodes don't terminate
                when agent reaches goal/lava'''
        self.reset()

        
    def reset(self):
        self.objects = []
        hero = gameOb(self.newPosition('hero'),1,1,2,None,'hero')
        self.objects.append(hero)
        bug = gameOb(self.newPosition('goal'),1,1,1,1,'goal')
        self.objects.append(bug)
        hole = gameOb(self.newPosition('fire'),1,1,0,-1,'fire')
        self.objects.append(hole)
        bug2 = gameOb(self.newPosition('goal2'),1,1,1,1,'goal2')
        self.objects.append(bug2)
        hole2 = gameOb(self.newPosition('fire2'),1,1,0,-1,'fire2')
        self.objects.append(hole2)
        bug3 = gameOb(self.newPosition('goal3'),1,1,1,1,'goal3')
        self.objects.append(bug3)
        bug4 = gameOb(self.newPosition('goal4'),1,1,1,1,'goal4')
        self.objects.append(bug4)
        state = self.renderEnv()
        self.state = state
        self.done = False
        return self._get_obs()
    

    def renderEnv(self):
        a = np.zeros([self.sizeY,self.sizeX,3])
        # a = np.ones([self.sizeY+2,self.sizeX+2,3])
        # a[1:-1,1:-1,:] = 0
        hero = None
        for item in self.objects:
            a[item.y:item.y+item.size, item.x:item.x+item.size, item.channel] = item.intensity
            if item.name == 'hero':
                hero = item
        if self.partial == True:
            a = a[hero.y:hero.y+3,hero.x:hero.x+3,:]
        b = scipy.misc.imresize(a[:,:,0],[5,5,1],interp='nearest')
        # b = np.array(Image.fromareray(a[:,:,0], mode='RGB').resize((5,5), resample=Image.NEAREST))
        c = scipy.misc.imresize(a[:,:,1],[5,5,1],interp='nearest')
        d = scipy.misc.imresize(a[:,:,2],[5,5,1],interp='nearest')
        a = np.stack([b,c,d],axis=2)
        return a


    def _get_obs(self):
        """
        Move channel to dim 0 and flatten
        From Zac's paper:
        "The agent receives the observation as an array of
        RGB pixel values flattened across the channel dimension."
        This is what is implemented here
        """
        return np.moveaxis(self.state, -1, 0).reshape(-1)