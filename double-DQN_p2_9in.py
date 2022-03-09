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
parser.add_argument('--train', dest='train', default=False)
parser.add_argument('--test', dest='test', default=True)

parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.00075)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--eps', type=float, default=0.05)

parser.add_argument('--train_episodes', type=int, default=2500)
parser.add_argument('--test_episodes', type=int, default=100)
parser.add_argument('--update_episodes', type=int, default=10)
parser.add_argument('--run_target', type=str, default='9in-mode-test')
parser.add_argument('--continue_train', type=int, default=1)         # 是否使用上一次训练的模型
args = parser.parse_args()

ALG_NAME = 'DQN'
# ENV_ID = 'CartPole-v0'
ENV_ID = 'LunarLander_SC-v2'

HP_UPDATE_EPISODE= hp.HParam('update_episode', hp.Discrete([2, 10, 15]))


def tf_gather_new(x,h_index):
    """以h_index为索引，提取x对应行上的元素，x为二位数组,h也为二位数组"""
    if x.shape[0] != h_index.shape[0]:
        print("传入tf_gather_new的参数维度不统一，x有%d行，h_index有%d行"%(x.shape[0],h_index.shape[0]))
    line = np.arange(x.shape[0]).reshape(-1, 1)
    index = np.hstack((line, h_index))
    result = tf.gather_nd(x, index)
    return result

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
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if args.test:
            self.test_log_dir = 'logs/gradient_tape/' + self.current_time+'_'+args.run_target + '/test'
            self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir)
        if args.train:
            self.train_log_dir = 'logs/gradient_tape/' + self.current_time+'_'+args.run_target+ '/train'
            self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
            self.reward_log_dir = 'logs/gradient_tape/' + self.current_time+'_'+args.run_target + '/reward'
            self.reward_summary_writer = tf.summary.create_file_writer(self.reward_log_dir)
        
        self.save_model_path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID,args.run_target]))

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
        if os.path.exists(self.save_model_path) and args.continue_train==1:
            self.loadModel()

        self.model_optim = self.target_model_optim = tf.optimizers.Adam(lr=args.lr)

        self.epsilon = args.eps

        self.buffer = ReplayBuffer()

    def target_update(self):
        """Copy q network to target q network"""
        for weights, target_weights in zip(
                self.model.trainable_weights, self.target_model.trainable_weights):
            target_weights.assign(weights)

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            q_value = self.model(state[np.newaxis, :])[0]
            return np.argmax(q_value)

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
            total_reward, done = 0, False
            while not done:
                self.env.render()
                action = self.model(np.array([state], dtype=np.float32))[0]
                action = np.argmax(action)
                next_state, reward, done, info = self.env.step(action)
                next_state = next_state.astype(np.float32)
                total_reward += reward
                state = next_state
            if info['success'] == 1:
                success_times += 1
            if info['crash'] == 1:
                crash_times += 1 
            self.env.render()
            time.sleep(0.3)
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
                while not done:
                    action = self.choose_action(state)
                    next_state, reward, done, _ = self.env.step(action)
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
                with self.reward_summary_writer.as_default():
                    tf.summary.scalar('reward', total_reward, step=episode)

                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=episode)
                # if episode % 10 == 0:
                #     self.test_episode()
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
        if os.path.exists(self.save_model_path):
            print('Load DQN Network parametets ...')
            tl.files.load_hdf5_to_weights_in_order(os.path.join(self.save_model_path, 'model.hdf5'), self.model)
            tl.files.load_hdf5_to_weights_in_order(os.path.join(self.save_model_path, 'target_model.hdf5'), self.target_model)
            print('Load weights!')
        else: print("No model file find, please train model first...")


if __name__ == '__main__':
    env = gym.make(ENV_ID)
    agent = Agent(env)
    agent.train(train_episodes=args.train_episodes)
    env.close()