import gym
import time
import os
from pyglet.window import key as pygkey
import tensorlayer as tl
import tensorflow as tf
import numpy as np
human_agent_action = 0
human_input_flag = 0
LEFT = pygkey.LEFT
RIGHT = pygkey.RIGHT
UP = pygkey.UP
DOWN = pygkey.DOWN

ALG_NAME = 'DQN'
ENV_ID = 'LunarLander_SC-v2'

def key_press(key, mod):
    global human_agent_action
    global human_input_flag
    a = int(key)
    if a <= 0:
        return
    if a == LEFT:
        human_agent_action = 3
        human_input_flag = 1
    elif a == RIGHT:    
        human_agent_action = 1
        human_input_flag = 1
    elif a == UP:
        human_agent_action = 2
        human_input_flag = 1
    elif a == DOWN:
        human_agent_action = 0    
        human_input_flag = 1       


def key_release(key, mod):
    global human_agent_action
    global human_input_flag    
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
        human_input_flag = 0

def human_pilot_policy(obs):
    global human_agent_action
    return human_agent_action

def create_model(input_state_shape):
    input_layer = tl.layers.Input(input_state_shape)
    layer_1 = tl.layers.Dense(n_units=256, act=tf.nn.relu)(input_layer)
    layer_2 = tl.layers.Dense(n_units=256, act=tf.nn.relu)(layer_1)

    output_layer = tl.layers.Dense(n_units=4)(layer_2)
    return tl.models.Model(inputs=input_layer, outputs=output_layer)

model = create_model([None, 8])
model.eval()

# load_model_path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
load_model_path = 'model/DQN_LunarLander_SC-v2_作者环境训练，小改,use-shaping-copilot-no-FuelCost'
if os.path.exists(load_model_path):
    print('Load DQN Network parametets ...')
    tl.files.load_hdf5_to_weights_in_order(os.path.join(load_model_path, 'model.hdf5'),model)
    print('Load weights!')
else: print("No model file find, please train model first...")





env = gym.make(ENV_ID)
env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release



state = env.reset()
state = state[:8]

done_num = 0
success_times = 0
crash_times = 0
test_episodes = 50
human_action_taken_num = 0
total_time = 1
while 1:
    total_time +=1
    env.render()
    human_action = human_pilot_policy(0)
    bot_action_Q = model(np.array([state], dtype=np.float32))[0]
    bot_action = np.argmax(bot_action_Q)

    if human_input_flag == 0:        
        action = bot_action
    if bot_action_Q[human_action] <= bot_action_Q[bot_action]*0.96:
        action = bot_action
    else:
        action = human_action
        human_action_taken_num+=1
    
    next_state, reward, done, info = env.step(action)
    next_state = next_state[:8]
    state = next_state
    if done:
        env.render()
        time.sleep(0.3)
        done_num +=1
        if info['success'] == 1:
            success_times += 1
        if info['crash'] == 1:
            crash_times += 1 
        observation = env.reset()
    if done_num>=test_episodes:
            break
env.close()
print("Success rate = "+str(success_times/test_episodes))
print("Crash rate = "+ str(crash_times/test_episodes))
print("human action taken rate = "+ str(human_action_taken_num/total_time))