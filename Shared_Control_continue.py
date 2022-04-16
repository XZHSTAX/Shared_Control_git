import gym
import time
import os
from pyglet.window import key as pygkey
import tensorlayer as tl
import tensorflow as tf
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

#Done 添加当用户不操作，就让机器接管；注意，是用户不操作，即不按键
key_flag = [0,0,0,0]   # ? 用于储存上下左右是否被按下，按下为1，抬起为0
inverse_control = 1    # ? 0 inverse 1 normal   i use 0

def key_press(key, mod):
  """
  这是一个按键按下转换函数，当按下 上下左右转化为human_agent_action数组中的数值
  并且激活human_agent_active
  """
  global human_agent_action
  a = int(key)
  if a == LEFT:
      human_agent_action[1] = 0 if inverse_control else 2
      key_flag[2] = 1
  elif a == RIGHT:
      human_agent_action[1] = 2 if inverse_control else 0
      key_flag[3] = 1
  elif a == UP:
      human_agent_action[0] = 1
      key_flag[0] = 1
  elif a == DOWN:
      human_agent_action[0] = 0
      key_flag[1] = 1

def key_release(key, mod):
    """
    这是一个按键抬起转换函数，当抬起 上下左右转化为human_agent_action数组中的数值
    并且关闭human_agent_active
    """
    global human_agent_action
    a = int(key)
    if a == LEFT or a == RIGHT:
        human_agent_action[1] = 1
        if a == LEFT:   key_flag[2] = 0
        if a == RIGHT:  key_flag[3] = 0
    elif a == UP or a == DOWN:
        human_agent_action[0] = 0
        if a == UP:     key_flag[0] = 0
        if a == DOWN:   key_flag[1] = 0


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
  return disc_to_cont(encode_human_action(human_agent_action)),encode_human_action(human_agent_action)

def how_sim(a1,a2):
    X = np.array([[2,1,1,1,0,0],
                  [1,2,1,0,0,0],
                  [1,1,2,0,0,1],
                  [1,0,0,2,1,1],
                  [0,0,0,1,2,1],
                  [0,0,1,1,1,2]])
    return X[a1,a2]    
human_action_taken_num = 0
def human_copilot_action_arbitration(copilot_Q,pilot_action,arbitration_way =1):
    """
    人机交互仲裁，arbitration_way =1最优动作替换,arbitration_way =2 相似动作替换
    输入： copilot_Q：Copilot对当前动作的评估值；pilot_action飞行员输入的动作
    输出： 仲裁后，认为最优的动作
    """
    global human_action_taken_num
    copilot_Qmax = np.max(copilot_Q)                            # ? copilot 认为最佳动作的Q值
    copilot_action = np.argmax(copilot_Q)                        # ? copilot 认为最佳动作

    if arbitration_way==1:
        if copilot_Q[int(pilot_action)] >= copilot_Qmax*0.975:
            human_action_taken_num +=1
            return pilot_action
        else:
            return copilot_action
    elif arbitration_way==2:

        copilot_Qmin = np.min(copilot_Q)                            # ? copilot 认为最差动作的Q值
        action_mask = ((copilot_Q-copilot_Qmin*np.ones(copilot_Q.shape)) >= 0.9*(copilot_Qmax-copilot_Qmin))
        action_sim = how_sim(np.array([0,1,2,3,4,5]),pilot_action)

        action_chooseable =  action_mask*action_sim
        if not any(action_chooseable):                             # ? 如果全是0
            return copilot_action                                  # ? 返回copilot的动作
        elif np.sum(action_chooseable==1)<=1 or np.sum(action_chooseable==2)==1:  # ? 如果就1个1或有2
            human_action_taken_num +=1
            return np.argmax(action_chooseable)                                   # ? 返回最佳动作
        else:
            a = np.nonzero(action_chooseable==1)
            a = np.array(a)[0]
            human_action_taken_num +=1
            return a[np.argmax(np.random.rand(a.shape[0]))]                       # ? 从相同的最佳中随机抽取

def create_model(input_state_shape):
    input_layer = tl.layers.Input(input_state_shape)
    layer_1 = tl.layers.Dense(n_units=256, act=tf.nn.relu)(input_layer)
    layer_2 = tl.layers.Dense(n_units=256, act=tf.nn.relu)(layer_1)

    output_layer = tl.layers.Dense(n_units=6)(layer_2)
    return tl.models.Model(inputs=input_layer, outputs=output_layer)

model = create_model([None, 8])
model.eval()

# load_model_path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))

# ! 可调参数--------------------------------------
test_episodes = 10  # ? 测试次数
pilot_name = 'gsf-test'  # ? 驾驶员名称
alpha = 0.96        # ? 相似系数
load_model_path = 'model/A3_plus2.5' # ? 加载模型位置
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

total_time = 1
total_reward = 0

#Done 把reward用rensorboard记录
time_now  = datetime.datetime.now().strftime("%Y-%m-%d~%H-%M-%S")
reward_log_dir = 'logs/test/' + time_now+'_'+pilot_name
reward_summary_writer = tf.summary.create_file_writer(reward_log_dir)


while 1:
    total_time +=1
    env.render()
    human_action,encode_action = human_pilot_policy(0)
    bot_action_Q = model(np.array([state], dtype=np.float32))[0]
    bot_action = np.argmax(bot_action_Q)


    # if not any(key_flag):
    #     # ? 如果没有按键按下，key_flag全0，any(key_flag)返回false
    #     # ? 有按键按下，key_flag非全0      any(key_flag)返回True
    #     action = disc_to_cont(bot_action)
    # if bot_action_Q[encode_action] <= bot_action_Q[bot_action]*alpha:
    #     action = disc_to_cont(bot_action)
    # else:
    #     action = human_action
    #     human_action_taken_num+=1   
    if not any(key_flag):
        # ? 如果没有按键按下，key_flag全0，any(key_flag)返回false
        # ? 有按键按下，key_flag非全0      any(key_flag)返回True
        action = disc_to_cont(bot_action)
    else:
        action = disc_to_cont(human_copilot_action_arbitration(bot_action_Q.numpy(),encode_action))     
    
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
df1 = pd.read_csv('TestData_RecordContinue.csv')
df2 = pd.DataFrame({'pilot\'s name':pilot_name,'Success rate':success_times/test_episodes,
                    'Crash rate':crash_times/test_episodes,'human action taken rate':human_action_taken_num/total_time,
                    'test_time':test_episodes,'alpha':alpha,'time':time_now,
                    'model_name':load_model_path},index=[0])
dataframe = df1.merge(df2,how="outer")

dataframe.to_csv("TestData_RecordContinue.csv",index=False,sep=',',mode='w')