import random
import numpy as np

class Memory:
    def __init__(self):
        increment = 1/14
        self.table = np.zeros((64, 4))
        # self.table.fill(0)

    def get(self, position):
        return self.table[position]

    def getMaxProb(self, position):
        return max(self.table[position])

    def add(self, position, action, value):
        self.table[position][action] = max(value, self.table[position][action])

    def print(self):
        for index in range(64):
            print(index, self.table[index])
    def printArrows(self):
        for index in range(8):
            content = ""
            for row in self.table[index*8:(index+1)*8]:
                max_action = np.argmax(row)
                content += ' < ' if max_action == 0 else (' v ' if max_action == 1 else (' > ' if max_action == 2 else ' ^ '))
            print(content)