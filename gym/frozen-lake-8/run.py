# Colors for console
from colorama import init
init()

import gym
env = gym.make('FrozenLake8x8-v0', is_slippery=False)

from model import Model
model = Model()

from memory import Memory
memory = Memory()

from game import Game
game = Game(memory, env, model)

# Get initial dataset
print("Play random")
for i in range(1000):
  game.play(random=True)

print("ML Training")
games = 100
for i in range(games):
  game.play()
  game.train()

print("Playing")
wins = 0
for i in range(games):
  if game.play():
    wins += 1
print("Win rate:", wins/games)

memory.print()
game.train(True)
game.play(False, True)