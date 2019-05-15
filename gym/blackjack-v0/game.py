import numpy as np
import random
import math

class Game:
    def __init__(self, memory, env, model):
        self.memory = memory
        self.env = env
        self.model = model

    def play(self, random=False, render=False):
        state = self.env.reset()
        state = (state[0], state[1], state[0] - 10 if state[2] else state[0])
        while True:
            # Play
            action = self.choose_action(state, random, render=render)
            new_state, reward, done, _ = self.env.step(action)
            new_state = (new_state[0], new_state[1], new_state[0] - 10 if new_state[2] else new_state[0])

            if render:
                print('From', state, 'to', new_state, ': ', action)

            if done:
                if reward == 0:
                    # Si empato el juego, poss bueno
                    self.memory.add(state, 0)
                    if render:
                        print('Draw!')
                    return 0
                if reward == 1:
                    # Si gano el juego, que bueno que paraste!
                    self.memory.add(state, 0)
                    if render:
                        print('Winner!')
                    return 1
                if reward == -1:
                    if action == 0:
                        # si perdiste el juego por parar, pues debiste seguir jugando!
                        self.memory.add(state, 1)
                    else:
                        # si perdiste el juego por seguir jugando, pues debiste parar!
                        self.memory.add(state, 0)
                    if render:
                        print('Game Over')
                    return -1
            # Si el juego continua, pues que bueno que continuaste
            # self.memory.add(state, 1)

            # Memorize
            #if new_state[0] > state[0] and new_state[2] > state[2]:
            #    self.memory.add(state, 1)
      
            # Finalize
            state = new_state

    def choose_action(self, state, play_random=False, render=False):
        if play_random:
            return random.randint(0, 1)
        return self.model.predict_one(state, render=render)

    def get_reward(self, old_state, new_state, action):
        if old_state == new_state:
            return 0
        max_prob = max(self.get_position_reward(new_state), self.memory.getMaxProb(new_state)) 
        return min(1, max_prob - (((step) / self.iterations)))

    def get_position_reward(self, position):
        if position in [0]:
            return 0
        if position in [1, 4]:
            return 0.1
        if position in [2, 5, 8]:
            return 0.2
        if position in [3, 6, 9, 12]:
            return 0.4
        if position in [7, 10, 13]:
            return 0.6
        if position in [11, 14]:
            return 0.8
        if position in [15]:
            return 1
    
    def train(self, verbose=False, epochs=100):
        x, y = self.memory.getTrainData()
        self.model.train(x, y, verbose, epochs)