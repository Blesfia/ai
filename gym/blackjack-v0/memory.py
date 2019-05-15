import random
import numpy as np
import tensorflow as tf
class Memory:
    def __init__(self):
        self.table = {}

    def add(self, state, action):
        if state not in self.table:
            self.table[state] = []
        self.table[state].append(action)

    def getTrainData(self):
        x = np.array(list(self.table.keys()))
        y = [
            ([0, 1] if np.mean(self.table[state]) >= 0.5 else [1, 0]) for state in self.table.keys()
            ]
        return x, np.array(y)

    def print(self):
        keys = list(self.table.keys())
        keys.sort()
        for state in keys:
            print(str(state[0]) + '\t' + str(state[1]) + '\t' + str(state[2]) + '\t=>\t', 0 if np.mean(self.table[state]) < 0.5 else 1)