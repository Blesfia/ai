import numpy as np
import random
import math

class Game:
    def __init__(self, memory, env, model):
        self.memory = memory
        self.env = env
        self.model = model
        self.iterations = 64

    def play(self, random=False, render=False):
        position = self.env.reset()
        for iteration in range(self.iterations):
            # Play
            action = self.choose_action(position, random, render=render)
            new_position, reward, done, _ = self.env.step(action)
            if render:
                self.env.render()
            # Memorize
            reward_position = self.get_reward(position, new_position, iteration)
            if render:
                print('From', position, 'to', new_position, ': ', reward_position)
            if done:
                if reward == 0:
                    self.memory.add(position, action, 0)
                    # print('Game over')
                    return False
                if reward == 1:
                    self.memory.add(position, action, 1)
                    # print("Winner")
                    return True
            else:
                self.memory.add(position, action, reward_position)

            # Finalize
            position = new_position

    def choose_action(self, position, play_random=False, render=False):
        if play_random:
            return random.randint(0, 3)
        return self.model.predict_one(position, render=render)

    def get_reward(self, old_position, new_position, step):
        if old_position == new_position:
            return 0
        max_prob = max(self.get_position_reward(new_position), self.memory.getMaxProb(new_position)) 
        return min(1, max_prob - (((step) / self.iterations)))

    def get_position_reward(self, position):
        increment = 1 / 14
        row = (position // 8)
        col = (position - 8*row)
        return increment * (row + col)
    
    def train(self, verbose=False, epochs=100):
        self.model.train(self.memory.table, verbose, epochs)