import numpy as np
from copy import deepcopy

# simple gaussian thompson sampling bandit
class bandit():
    def __init__(self, nArms, trials = []):
        self.nArms = nArms
        self.trials = [deepcopy(trials) for _ in range(nArms)]
        self.lastAction = 0
    
    def decision(self,window_length=10000):
        res = []
        for i in range(self.nArms):
            mu = np.nan_to_num(np.mean(self.trials[i][-window_length:]),nan=0)
            std = np.nan_to_num(np.std(self.trials[i][-window_length:]),nan=1)
            sample = np.random.randn()*std+mu
            res.append(sample)
        self.lastAction = np.argmax(res)
        return self.lastAction
    def getReward(self, reward):
        self.trials[self.lastAction].append(reward)
    def getMeans(self,window_length=10000):
        return [np.mean(_[-window_length:]) for _ in self.trials]