import gym
import time
import os
from pyglet.window import key as pygkey
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import pandas as pd
import datetime

human_agent_action = [0,1]
human_input_flag = 0
LEFT = pygkey.LEFT
RIGHT = pygkey.RIGHT
UP = pygkey.UP
DOWN = pygkey.DOWN

ALG_NAME = 'DQN'
ENV_ID = 'LunarLanderContinuous_SC-v2'


key_flag = [0,0,0,0]   # ? 用于储存上下左右是否被按下，按下为1，抬起为0
inverse_control = 0    # ? 0 inverse 1 normal   i use 0

def key_press(key, mod):
  """
  这是一个按键按下转换函数，当按下 上下左右转化为human_agent_action数组中的数值
  并且激活human_agent_active
  """
  global human_agent_action
  a = int(key)
  if a == LEFT:
      human_agent_action[1] = 0 if inverse_control else 2
  elif a == RIGHT:
      human_agent_action[1] = 2 if inverse_control else 0
  elif a == UP:
      human_agent_action[0] = 1
  elif a == DOWN:
      human_agent_action[0] = 0

def key_release(key, mod):
  """
  这是一个按键抬起转换函数，当抬起 上下左右转化为human_agent_action数组中的数值
  并且关闭human_agent_active
  """
  global human_agent_action
  a = int(key)
  if a == LEFT or a == RIGHT:
      human_agent_action[1] = 1
  elif a == UP or a == DOWN:
      human_agent_action[0] = 0
    
# 转化输入码 return = 上下*3+左右  范围0~5
# down + left    0  下左
# down + none    1  下
# down + right   2  下右
# up   + left    3  上左
# up   + none    4  上
# up   + right   5  上右
def encode_human_action(action):
    return action[0]*3+action[1]

throttle_mag = 0.75
def disc_to_cont(action):
    if type(action) == np.ndarray:
        return action
    # main engine
    if action < 3:
        m = -throttle_mag
    elif action < 6:
        m = throttle_mag
    else:
        raise ValueError
    # steering
    if action % 3 == 0:
        s = -throttle_mag
    elif action % 3 == 1:
        s = 0
    else:
        s = throttle_mag
    return np.array([m, s])



def human_pilot_policy(obs):
  """
  此函数包装了上面的内容，返回人类输入，被转化后的动作码
  """
  global human_agent_action
  return disc_to_cont(encode_human_action(human_agent_action))





# ! 可调参数--------------------------------------
test_episodes = 5  # ? 测试次数
pilot_name = 'zzc'  # ? 驾驶员名称

env = gym.make(ENV_ID)
env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

# outdir = "Video"
# env = gym.wrappers.Monitor(env, directory=outdir, force=True,video_callable=lambda episode_id: True)  
# # video_callable可以设置多少episode记录一次 
# #（video_callable=lambda episode_id: True）[reference](https://github.com/openai/gym/issues/494)
# # force可强制性地覆盖掉之前生成的记录文件

state = env.reset()

done_num = 0
success_times = 0
crash_times = 0
total_time = 1
total_reward = 0

time_now  = datetime.datetime.now().strftime("%Y-%m-%d~%H-%M-%S")
reward_log_dir = 'logs/test_origin/' + time_now+'_'+pilot_name
reward_summary_writer = tf.summary.create_file_writer(reward_log_dir)


while 1:
    total_time +=1
    env.render()
    human_action = human_pilot_policy(0)
    next_state, reward, done, info = env.step(human_action)

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

        # ? 记录总回报
        with reward_summary_writer.as_default():                   
            tf.summary.scalar('reward', total_reward, step=done_num)
        total_reward = 0
    if done_num>=test_episodes:
            break
env.close()
print("Success rate = "+str(success_times/test_episodes))
print("Crash rate = "+ str(crash_times/test_episodes))




# 记录数据，存在CSV文件中
cvspath = 'TestData_Record_OriginVersionContinue.csv'
df1 = pd.read_csv(cvspath)
df2 = pd.DataFrame({'pilot\'s name':pilot_name,'Success rate':success_times/test_episodes,
                    'Crash rate':crash_times/test_episodes,'test_time':test_episodes,'time':time_now,},index=[0])
dataframe = df1.merge(df2,how="outer")

dataframe.to_csv(cvspath,index=False,sep=',',mode='w')