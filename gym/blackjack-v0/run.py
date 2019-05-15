# Colors for console
from colorama import init
init()

from progressBar import printProgressBar
import gym
env = gym.make('Blackjack-v0')

from model import Model
model = Model()

from memory import Memory
memory = Memory()

from game import Game
game = Game(memory, env, model)

class BatchOfGames:

  def __init__(self, attempts, random, train, game):
    self.wins = 0
    self.loses = 0
    self.draws = 0
    self.attempts = attempts
    self.random = random
    self.train = train
    self.game = game

  def run(self):
    for i in range(0, self.attempts):
      printProgressBar(i, self.attempts, 'Progress: ', 'Complete. (' + str(self.wins) + ' Win, ' + str(self.draws) + ' Draw, ' + str(self.loses) + ' Lose)')
      score = self.game.play(self.random)
      self.addScore(score)
      if self.train and i % 10 == 0:
        self.game.train()
    print()
    print("Finish:", (self.wins* 100)/self.attempts , "% (" + str(self.wins) + ' Wins, ' + str(self.draws) + ' Draws, ' + str(self.loses) + ' Loses)')

  def addScore(self, score):
      if score == 1:
        self.wins += 1
      elif score == 0:
        self.draws += 1
      else:
        self.loses += 1
# Get initial dataset
print("Play random")
BatchOfGames(1000, random=True, train=False, game=game).run()

game.train()

games = 1000
print("Play Training")
BatchOfGames(1000, random=False, train=True, game=game).run()

game.train(True)
print("Play Normal")
BatchOfGames(1000, random=False, train=False, game=game).run()
