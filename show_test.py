import gym
import time

env = gym.make("LunarLander_SC-v2")
num_done = 0
observation = env.reset()

while 1:
  observation,action = env.render()
  # print(action)
  # action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  time.sleep(0.015)
  if done:
    env.render()
    time.sleep(0.3)

  if done:
    observation = env.reset()
    num_done +=1
  if num_done >=50:
    break
env.close()