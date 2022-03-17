import gym
import time
from pyglet.window import key as pygkey

# init_human_action = lambda: [0, 1] # noop
# 0号位置对应上下，0为下，1为上
# 1号位置对应左右，0为左，1为无，2为右
# human_agent_action = init_human_action()
human_agent_action = 0

LEFT = pygkey.LEFT
RIGHT = pygkey.RIGHT
UP = pygkey.UP
DOWN = pygkey.DOWN


def key_press(key, mod):
    global human_agent_action
    a = int(key)
    if a <= 0:
        return
    if a == LEFT:
        human_agent_action = 3
    elif a == RIGHT:    
        human_agent_action = 1
    elif a == UP:
        human_agent_action = 2
    elif a == DOWN:
        human_agent_action = 0           


def key_release(key, mod):
    global human_agent_action
    
    a = int(key)

    if a == LEFT:
        a = 3
    elif a == RIGHT:    
        a = 1
    elif a == UP:
        a = 2     
    if a <= 0:
        return
    if human_agent_action == a:
        human_agent_action = 0

def human_pilot_policy(obs):
    global human_agent_action
    return human_agent_action


env = gym.make("LunarLander-v2")
env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release


observation = env.reset()
done_num = 0
while 1:
    env.render()
    action = human_pilot_policy(0)
    #   action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    if done:
        env.render()
        time.sleep(0.3)
        done_num +=1
        observation = env.reset()
    if done_num>=20:
            break
env.close()