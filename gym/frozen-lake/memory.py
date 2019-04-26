import random
import numpy as np

class Memory:
    def __init__(self):
        self.table = np.zeros((16, 4))
        # self.table.fill(0)

    def get(self, position):
        return self.table[position]

    def getMaxProb(self, position):
        return max(self.table[position])

    def add(self, position, action, value):
        self.table[position][action] = max(value, self.table[position][action])

    def print(self):
        for index in range(16):
            print(index, self.table[index])