import numpy as np
class Memory:
    def __init__(self):
        self.plays = []
        self.actions = []

    def addGame(self, plays, actions):
        if len(plays) < 18:
            return
        for play in plays[0:-10]:
            self.plays.append(play)
        for action in actions[0:-10]:
            if action == 0:
                self.actions.append([1, 0])
            else:
                self.actions.append([0, 1])

    def getX(self):
        return np.array(self.plays)

    def getY(self):
        return np.array(self.actions)