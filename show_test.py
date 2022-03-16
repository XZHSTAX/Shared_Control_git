import gym
import time

env = gym.make("LunarLander_SC-v2")

observation = env.reset()

for _ in range(2000):
  _,action = env.render()
  print(action)
  # action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  time.sleep(0.007)
  if done:
    env.render()
    time.sleep(0.3)

  if done:
    observation = env.reset()
env.close()