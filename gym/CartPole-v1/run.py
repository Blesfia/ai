# Celda de todos
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
import gym
import numpy as np
env = gym.make('CartPole-v1')
from memory import Memory
from tqdm import tqdm
    
from model import Model
model = Model()

memory = Memory()
games = 1000
average_steps = []
# Play random
for game in range(games):
    observation = env.reset()
    plays = [] 
    actions = []
    step = 0
    while True:
        step += 1
        action = env.action_space.sample()
        plays.append(observation)
        actions.append(action)
        next_observation, reward, done, _ = env.step(action)
        
        if done:
            memory.addGame(plays, actions)
            break
    average_steps.append(step)
print("Random average step:", np.mean(average_steps))

# Import model


model.train(memory.getX(), memory.getY())

average_steps = []
games = 100
# Play random
for game in tqdm(range(games)):
    observation = env.reset()
    step = 0
    while True:
        env.render()
        step += 1
        action = model.predict(observation)
        next_observation, reward, done, _ = env.step(action)
        observation = next_observation 
        if done:
            break
    average_steps.append(step)
print("Production average step:", np.mean(average_steps))
env.close()