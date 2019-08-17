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
        
class GridworldEnv(gym.Env): # sam subclassed gameEnv as gym.Env
    def __init__(self, partial=False, size=5): # sam added defaults
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial
        self.reset()
        plt.imshow(self.state, interpolation="nearest")
        # Sam added the following two lines
        self.action_space = spaces.Discrete(4) # Discrete(n) is a discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`
        # self.observation_space = spaces.Box(low=0, high=255, shape=(84,84,3)) # Box represents the Cartesian product of n closed intervals
        self.observation_space = spaces.Box(low=0, high=255, shape=(21168,)) # Box represents the Cartesian product of n closed intervals
        # not certain that high is 255 but pretty sure... this comes from scipy.misc.imresize
        
    def reset(self):
        self.objects = []
        hero = gameOb(self.newPosition(),1,1,2,None,'hero')
        self.objects.append(hero)
        bug = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(bug)
        hole = gameOb(self.newPosition(),1,1,0,-1,'fire')
        self.objects.append(hole)
        bug2 = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(bug2)
        hole2 = gameOb(self.newPosition(),1,1,0,-1,'fire')
        self.objects.append(hole2)
        bug3 = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(bug3)
        bug4 = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(bug4)
        state = self.renderEnv()
        self.state = state
        return self._get_obs()

    def moveChar(self,direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        penalize = 0.
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
    
    def newPosition(self):
        iterables = [ range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x,objectA.y) not in currentPositions:
                currentPositions.append((objectA.x,objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)),replace=False)
        return points[location]

    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        ended = False
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(gameOb(self.newPosition(),1,1,1,1,'goal'))
                else: 
                    self.objects.append(gameOb(self.newPosition(),1,1,0,-1,'fire'))
                return other.reward,False
        if ended == False:
            return 0.0,False

    def renderEnv(self):
        #a = np.zeros([self.sizeY,self.sizeX,3])
        a = np.ones([self.sizeY+2,self.sizeX+2,3])
        a[1:-1,1:-1,:] = 0
        hero = None
        for item in self.objects:
            a[item.y+1:item.y+item.size+1,item.x+1:item.x+item.size+1,item.channel] = item.intensity
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
        penalty = self.moveChar(action)
        reward, done = self.checkGoal()
        state = self.renderEnv()
        self.state = state
        assert reward != None, "Hmm, reward is None. What's going on?"
        return self._get_obs(), (reward + penalty), done, {} # sam added {} - empty info


    def render(self, mode='human'):
        """Sam added this function.
           Not sure it does quite what gym does...
           In particular, I can't recall if gym
           closes automatically
           Also, if partial=True, this renders
           observation NOT state
        """
        if mode == 'human':
            state = self.renderEnv() # self.state would seem to be better code, I'm not 100% sure that self.state is updated when it should be
            plt.imshow(state, interpolation="nearest")
            # plt.ion() # this wasn't working for me
            plt.show() # show() is a blocking function: code will hang until you x the matplotlib popup window
            # sleep(1)
            # plt.close()
        else:
            raise NotImplementedError("Haven't specified what rendering not in human mode means")

    def _get_obs(self):
        """
        Returns same as processState function used in https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb (see In[5]...)
        """
        return np.reshape(self.state,[21168]) # 21168 = 84*84*3