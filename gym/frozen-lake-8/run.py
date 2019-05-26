# Colors for console
from colorama import init
init()
import numpy as np
import gym

env = gym.make('FrozenLake8x8-v0', is_slippery=False)

# Data
x = np.identity(64)
y = np.zeros((64, 4))

# Play random
wins = 0
def game():
  actual_state = env.reset()
  done = False
  reward = 0
  while not done:
    action = env.action_space.sample()
    new_state, reward, done, _ = env.step(action)
    if new_state == actual_state:
      y[actual_state][action] = 0
    else: 
      y[actual_state][action] = np.max(y[new_state]) * 0.9
    actual_state = new_state
  if reward == 1:
    y[actual_state] = np.ones(4)
    return True
  return False
for i in range(10000):
  if game():
    wins += 1
print("Memory", y)
print('Random Ratio', wins/10000)
  
# Build model
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, input_shape=(64,)),
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(4, activation='softmax')
])
model.compile(
  loss='mean_squared_error',
  optimizer='adam',
  metrics=['accuracy', 'mean_squared_error'],
)

def predict_action(actual_position):
  return np.argmax(model.predict(np.array([np.identity(64)[actual_position]])))

model.fit(x, np.array(y), verbose=True, epochs=100)

# Play agent
wins = 0
def game(render=False):
  limit = 16
  actual_state = env.reset()
  done = False
  reward = 0
  while not done and limit > 0:
    action = predict_action(actual_state)
    new_state, reward, done, _ = env.step(action)
    actual_state = new_state
    limit -= 1
    if render:
      env.render()
  if reward == 1:
    return True
  return False
games = 100
for i in range(games):
  if game():
    wins += 1

print('Agent Ratio', wins/games)

game(True)
