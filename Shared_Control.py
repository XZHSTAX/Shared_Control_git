import gym
import time
import os
from pyglet.window import key as pygkey
import tensorlayer as tl
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
human_agent_action = 0
human_input_flag = 0
LEFT = pygkey.LEFT
RIGHT = pygkey.RIGHT
UP = pygkey.UP
DOWN = pygkey.DOWN

ALG_NAME = 'DQN'
ENV_ID = 'LunarLander_SC-v2'

#Done 添加当用户不操作，就让机器接管；注意，是用户不操作，即不按键
key_flag = [0,0,0,0]   # ? 用于储存上下左右是否被按下，按下为1，抬起为0
inverse_control = 0    # ? 0 inverse 1 normal

def key_press(key, mod):
    global human_agent_action
    global human_input_flag
    a = int(key)
    if a <= 0:
        return
    if a == LEFT:
        human_agent_action = 1 if inverse_control else 3
        key_flag[2] = 1
    elif a == RIGHT:    
        human_agent_action = 3 if inverse_control else 1
        key_flag[3] = 1
    elif a == UP:
        human_agent_action = 2
        key_flag[0] = 1
    elif a == DOWN:
        human_agent_action = 0 
        key_flag[1] = 1
             

def key_release(key, mod):
    global human_agent_action
    global human_input_flag    
    a = int(key)

    if a == LEFT:
        a = 1 if inverse_control else 3
        key_flag[2] = 0
    elif a == RIGHT:    
        a = 3 if inverse_control else 1
        key_flag[3] = 0
    elif a == UP:
        a = 2
        key_flag[0] = 0     
    if a <= 0:
        a = 0
        key_flag[1] = 0
    if human_agent_action == a:
        human_agent_action = 0

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

# ! 可调参数--------------------------------------
test_episodes = 10  # ? 测试次数
pilot_name = 'xzh-ez'  # ? 驾驶员名称
alpha = 0.96        # ? 相似系数
load_model_path = 'model/DQN_LunarLander_SC-v2_作者环境训练，小改,use-shaping-copilot-no-FuelCost' # ? 加载模型位置
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
human_action_taken_num = 0
total_time = 1
total_reward = 0

#Done 把reward用rensorboard记录
time_now  = datetime.datetime.now().strftime("%Y-%m-%d~%H-%M-%S")
reward_log_dir = 'logs/test/' + time_now+'_'+pilot_name
reward_summary_writer = tf.summary.create_file_writer(reward_log_dir)


while 1:
    total_time +=1
    env.render()
    human_action = human_pilot_policy(0)
    bot_action_Q = model(np.array([state], dtype=np.float32))[0]
    bot_action = np.argmax(bot_action_Q)

    if not any(key_flag):
        # ? 如果没有按键按下，key_flag全0，any(key_flag)返回false
        # ? 有按键按下，key_flag非全0      any(key_flag)返回True
        action = bot_action
    if bot_action_Q[human_action] <= bot_action_Q[bot_action]*alpha:
        action = bot_action
    else:
        action = human_action
        human_action_taken_num+=1
    
    next_state, reward, done, info = env.step(action)
    next_state = next_state[:8]
    state = next_state
    total_reward += reward
    if done:
        env.render()
        time.sleep(0.3)           # ? 暂停0.3s，看见结果
        done_num +=1              # ? 完成次数+1
        if info['success'] == 1:  # ? 记录成功和坠毁次数
            success_times += 1
        if info['crash'] == 1:
            crash_times += 1 
        observation = env.reset() # ? 重启环境
        key_flag = [0,0,0,0]

        # ? 记录总回报
        with reward_summary_writer.as_default():                   
            tf.summary.scalar('reward', total_reward, step=done_num)
        total_reward = 0
    if done_num>=test_episodes:
            break
env.close()
print("Success rate = "+str(success_times/test_episodes))
print("Crash rate = "+ str(crash_times/test_episodes))
print("human action taken rate = "+ str(human_action_taken_num/total_time))




# 记录数据，存在CSV文件中
df1 = pd.read_csv('TestData_Record.csv')
df2 = pd.DataFrame({'pilot\'s name':pilot_name,'Success rate':success_times/test_episodes,
                    'Crash rate':crash_times/test_episodes,'human action taken rate':human_action_taken_num/total_time,
                    'test_time':test_episodes,'alpha':alpha,'time':time_now,
                    'model_name':load_model_path},index=[0])
dataframe = df1.merge(df2,how="outer")

dataframe.to_csv("TestData_Record.csv",index=False,sep=',',mode='w')