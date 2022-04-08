import argparse
import os
import random
import tensorboard
from tensorboard.plugins.hparams import api as hp
import numpy as np

import gym
import tensorflow as tf
import tensorlayer as tl
import datetime
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
parser = argparse.ArgumentParser()

parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.00075)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--eps', type=float, default=0.05)

parser.add_argument('--update_episodes', type=int, default=10)
parser.add_argument('--continue_train', type=int, default=1)         # 是否使用上一次训练的模型
parser.add_argument('--test_render', type=int, default=1)


ALG_NAME = 'DQN'
# ENV_ID = 'CartPole-v0'

# ! 常改参数------------------------------------------------------------------------------------------
ENV_ID = 'LunarLanderContinuous_SC-v2'
parser.add_argument('--train', dest='train', default=False)
parser.add_argument('--test', dest='test', default=True)

parser.add_argument('--train_episodes', type=int, default=1000)
parser.add_argument('--test_episodes', type=int, default=100)

parser.add_argument('--run_target', type=str, default='A5_plus3-测试100轮') # ? 会影响log文件的命名
args = parser.parse_args()
save_model_path = os.path.join('model', 'A5_plus3') # ? 模型保存位置
load_model_path = 'model/A5_plus3'                  # ? 模型加载位置
laggy_model_path = 'model/A4_plus3'
use_copilot = 1                                     # ? 0：不使用copilot  1：使用copilot
# ! --------------------------------------------------------------------------------------------------

def tf_gather_new(x,h_index):
    """以h_index为索引，提取x对应行上的元素，x为二位数组,h也为二位数组"""
    if x.shape[0] != h_index.shape[0]:
        print("传入tf_gather_new的参数维度不统一，x有%d行，h_index有%d行"%(x.shape[0],h_index.shape[0]))
    line = np.arange(x.shape[0]).reshape(-1, 1)
    index = np.hstack((line, h_index))
    result = tf.gather_nd(x, index)
    return result

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

def how_sim(a1,a2):
    X = np.array([[2,1,1,1,0,0],
                  [1,2,1,0,0,0],
                  [1,1,2,0,0,1],
                  [1,0,0,2,1,1],
                  [0,0,0,1,2,1],
                  [0,0,1,1,1,2]])
    return X[a1,a2]    



class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size = args.batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        """ 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        """
        return state, action, reward, next_state, done


class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]-1
        # self.action_dim = self.env.action_space.n
        self.action_dim = 6

        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if args.test:
            self.test_log_dir = 'logs/gradient_tape/' + self.current_time+'_'+args.run_target + '/test'
            self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir)
        if args.train:
            self.train_log_dir = 'logs/gradient_tape/' + self.current_time+'_'+args.run_target+ '/train'
            self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
            self.reward_log_dir = 'logs/gradient_tape/' + self.current_time+'_'+args.run_target + '/reward'
            self.reward_summary_writer = tf.summary.create_file_writer(self.reward_log_dir)
        
        self.save_model_path = save_model_path
        self.load_model_path = load_model_path
        self.laggy_model_path = laggy_model_path  
        self.use_copilot     = use_copilot
        self.torlence_alpha = 0.975

        def create_model(input_state_shape):
            input_layer = tl.layers.Input(input_state_shape)
            layer_1 = tl.layers.Dense(n_units=256, act=tf.nn.relu)(input_layer)
            layer_2 = tl.layers.Dense(n_units=256, act=tf.nn.relu)(layer_1)

            output_layer = tl.layers.Dense(n_units=self.action_dim)(layer_2)
            return tl.models.Model(inputs=input_layer, outputs=output_layer)

        self.model = create_model([None, self.state_dim])
        self.target_model = create_model([None, self.state_dim])        
        
        
        self.model.train()
        self.target_model.eval()
        if self.use_copilot:
            self.laggy_model = create_model([None, self.state_dim])
            self.laggy_model.eval()
        
        if os.path.exists(self.load_model_path) and args.continue_train==1:
            self.loadModel()

        self.model_optim = self.target_model_optim = tf.optimizers.Adam(lr=args.lr)

        self.epsilon = args.eps

        self.buffer = ReplayBuffer()
        self.last_laggy_action = 1    # 初始化laggy动作为 无

    def target_update(self):
        """Copy q network to target q network"""
        for weights, target_weights in zip(
                self.model.trainable_weights, self.target_model.trainable_weights):
            target_weights.assign(weights)

    def laggy_pilot(self,state,copilot_rl,lag_prob=0.5):
        """
        laggy的组成：输入：
        state当前状态，copilot_rl左右手控制信号,lag_prob重复上一个动作的概率
        如果laggy network认为左右手控制的更好，那就返回左右手
        """
        Q_value_laggy_action = self.laggy_model(state[np.newaxis, :])[0].numpy() # ? laggy对Q值的评估
        laggy_action         = np.argmax(Q_value_laggy_action)                   # ? laggy认为最佳动作
        Q_value_max_laggy_action = Q_value_laggy_action[laggy_action]            # ? laggy认为最佳动作的Q值
        
        if np.random.random() >= lag_prob:                                       # ? 如果执行新的动作
            if Q_value_laggy_action[copilot_rl] >= 0.975*Q_value_max_laggy_action: # ? 如果左右手更好
                laggy_action = copilot_rl
                self.last_laggy_action = laggy_action
        else:
            laggy_action = self.last_laggy_action
        return laggy_action

    def human_copilot_action_arbitration(self,copilot_Q,pilot_action,arbitration_way =2):
        """
        人机交互仲裁，arbitration_way =1最优动作替换,arbitration_way =2 相似动作替换
        输入： copilot_Q：Copilot对当前动作的评估值；pilot_action飞行员输入的动作
        输出： 仲裁后，认为最优的动作
        """
        copilot_Qmax = np.max(copilot_Q)                            # ? copilot 认为最佳动作的Q值
        copilot_action = np.argmax(copilot_Q)                        # ? copilot 认为最佳动作

        if arbitration_way==1:
            if copilot_Q[int(pilot_action)] >= copilot_Qmax*0.975:
                return pilot_action
            else:
                return copilot_action
        elif arbitration_way==2:

            copilot_Qmin = np.min(copilot_Q)                            # ? copilot 认为最差动作的Q值
            action_mask = ((copilot_Q-copilot_Qmin*np.ones(copilot_Q.shape)) >= self.torlence_alpha*(copilot_Qmax-copilot_Qmin))
            action_sim = how_sim(np.array([0,1,2,3,4,5]),pilot_action)

            action_chooseable =  action_mask*action_sim
            if not any(action_chooseable):                             # ? 如果全是0
                return copilot_action                                  # ? 返回copilot的动作
            elif np.sum(action_chooseable==1)<=1 or np.sum(action_chooseable==2)==1:  # ? 如果就1个1或有2
                return np.argmax(action_chooseable)                                   # ? 返回最佳动作
            else:
                a = np.nonzero(action_chooseable==1)
                a = np.array(a)[0]
                return a[np.argmax(np.random.rand(a.shape[0]))]                       # ? 从相同的最佳中随机抽取
                

    def choose_action(self, state,copilot_rl,epsilon_on = 1):
        if np.random.uniform() < self.epsilon and epsilon_on==1:
            return np.random.choice(self.action_dim)
        else:
            if self.use_copilot==0:
                q_value = self.model(state[np.newaxis, :])[0]
                return np.argmax(q_value)
            else:
                copilot_Q     = self.model(state[np.newaxis, :])[0].numpy()  # ? copilot 对Q值的评估
                pilot_action  = self.laggy_pilot(state,copilot_rl)           # ? pilot 认为最佳动作
                # pilot_action = copilot_rl
                return self.human_copilot_action_arbitration(copilot_Q,pilot_action)


    def replay(self):
        # for i in range(10):
        # sample an experience tuple from the dataset(buffer)
        states, actions, rewards, next_states, done = self.buffer.sample()
        # compute the target value for the sample tuple
        # targets [batch_size, action_dim]
        # Target represents the current fitting level
        # next_q_values [batch_size, action_dim]
        
        next_Q = self.model(next_states)                # 评估网络推测状态s'下所有动作的Q值
        next_a = tf.argmax(next_Q, axis=1).numpy()      # 评估网络推测状态s'下应该采取的动作

        next_target = self.target_model(next_states)        # 目标网络推测状态s'下所有动作的Q值
        next_q_value = tf_gather_new(next_target, next_a.reshape([args.batch_size,1]))   # 目标网络推测状态s'下，采取next_a动作对饮的Q值

        target = rewards + (1 - done) * args.gamma * next_q_value

        # use sgd to update the network weight
        with tf.GradientTape() as tape:
            q_pred = self.model(states)
            q_pred = tf_gather_new(q_pred, actions.reshape([args.batch_size,1]))
            loss = tf.losses.mean_squared_error(target, q_pred)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.model_optim.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss.numpy()


    def test_episode(self, test_episodes):
        success_times = 0
        crash_times = 0
        for episode in range(test_episodes):
            state = self.env.reset().astype(np.float32)
            copilot_rl = int(state[8]) 
            state = state[:8]
            total_reward, done = 0, False
            while not done:
                if args.test_render: self.env.render()             
                # action = self.model(np.array([state], dtype=np.float32))[0]
                # action = np.argmax(action)
                action = self.choose_action(state, copilot_rl,epsilon_on = 0)
                next_state, reward, done, info = self.env.step(disc_to_cont(action))
                
                copilot_rl = int(next_state[8])
                next_state = next_state[:8]
                next_state = next_state.astype(np.float32)

                total_reward += reward
                state = next_state
            self.last_laggy_action = 1     # ! 完成一幕后，要让last_laggy_action恢复默认值
            if args.test_render:
                self.env.render()
                time.sleep(0.3)
            if info['success'] == 1:
                success_times += 1
            if info['crash'] == 1:
                crash_times += 1 
            # 每测试完一次，记录一次
            with self.test_summary_writer.as_default():
                tf.summary.scalar('test-reward', total_reward, step=episode)
            print("Test {} | episode rewards is {}".format(episode, total_reward))
        print("Success rate = "+str(success_times/test_episodes))
        print("Crash rate = "+ str(crash_times/test_episodes))

    def train(self, train_episodes=200):
        i = 0
        if args.train:
            for episode in range(train_episodes):
                total_reward, done = 0, False
                state = self.env.reset().astype(np.float32)
                copilot_rl = int(state[8])
                state = state[:8]
                while not done:
                    action = self.choose_action(state,copilot_rl)
                    next_state, reward, done, _ = self.env.step(disc_to_cont(action))
                    copilot_rl = int(next_state[8])
                    next_state = next_state[:8]
                    next_state = next_state.astype(np.float32)
                    self.buffer.push(state, action, reward, next_state, done)
                    total_reward += reward
                    state = next_state
                    # self.render()
                    if len(self.buffer.buffer) > args.batch_size:
                        loss = self.replay()
                    i = i+1
                    if i>= args.update_episodes:
                        self.target_update()
                        i = 0
                print('EP:{} | R:{}'.format(episode, total_reward))
                self.last_laggy_action = 1                     # ! 完成一幕后，要让last_laggy_action恢复默认值
                with self.reward_summary_writer.as_default():
                    tf.summary.scalar('reward', total_reward, step=episode)
                if len(self.buffer.buffer) > args.batch_size:
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('loss', loss, step=episode)
                if episode % 100 == 0 and episode != 0:
                    self.saveModel()
            self.saveModel()
        if args.test:
            self.loadModel()
            self.test_episode(test_episodes=args.test_episodes)

    def saveModel(self):
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        tl.files.save_weights_to_hdf5(os.path.join(self.save_model_path, 'model.hdf5'), self.model)
        tl.files.save_weights_to_hdf5(os.path.join(self.save_model_path, 'target_model.hdf5'), self.target_model)
        print('Saved weights.')

    def loadModel(self):
        if os.path.exists(self.load_model_path):
            print('Load DQN Network parametets ...')
            tl.files.load_hdf5_to_weights_in_order(os.path.join(self.load_model_path, 'model.hdf5'), self.model)
            tl.files.load_hdf5_to_weights_in_order(os.path.join(self.load_model_path, 'target_model.hdf5'), self.target_model)
            print('Load weights!')
        else: print("No model file find, please train model first...")
        if self.use_copilot:
            if os.path.exists(self.laggy_model_path):
                print('Load Laggy_Pilot parametets ...')
                tl.files.load_hdf5_to_weights_in_order(os.path.join(self.laggy_model_path, 'model.hdf5'), self.laggy_model)
            else: print("No laggy_model file find, please train model first...")


if __name__ == '__main__':
    env = gym.make(ENV_ID)
    agent = Agent(env)
    agent.train(train_episodes=args.train_episodes)
    env.close()