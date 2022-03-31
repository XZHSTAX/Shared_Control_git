#!/usr/bin/env python
# coding: utf-8

# [1]:


from __future__ import division
import pickle
import random
import os
import math
import types
import uuid
import time
from copy import copy
from collections import defaultdict, Counter

import numpy as np
import gym
from gym import spaces, wrappers

import dill
import tempfile
import tensorflow as tf
from tensorflow.contrib import rnn
import zipfile

import baselines.common.tf_util as U

from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.simple import ActWrapper

from scipy.special import logsumexp
from scipy.stats import binom_test, ttest_1samp

from pyglet.window import key as pygkey

data_dir = os.path.join('data', 'lunarlander-human')


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

# 用于遮住停机坪位置信息
def mask_helipad(obs, replace=0):
  obs = copy(obs)
  if len(obs.shape) == 1:
    obs[8] = replace
  else:
    obs[:, 8] = replace
  return obs

def traj_mask_helipad(traj):
  return [mask_helipad(obs) for obs in traj]

n_act_dim = 6 # 2 x 3
n_act_true_dim = 2
n_obs_dim = 9


def onehot_encode(i, n=n_act_dim):
    x = np.zeros(n)
    x[i] = 1
    return x

def onehot_decode(x):
    l = np.nonzero(x)[0]
    assert len(l) == 1
    return l[0]

def make_env(using_lander_reward_shaping=False):
  env = gym.make('LunarLanderContinuous-v2')
  env.action_space = spaces.Discrete(n_act_dim)
  env.unwrapped._step_orig = env.unwrapped._step
  def _step(self, action):
      obs, r, done, info = self._step_orig(disc_to_cont(action))
      return obs, r, done, info
  env.unwrapped._step = types.MethodType(_step, env.unwrapped)
  env.unwrapped.using_lander_reward_shaping = using_lander_reward_shaping
  return env

env = make_env(using_lander_reward_shaping=True)
max_ep_len = 1000
make_q_func = lambda: deepq.models.mlp([64, 64])

def run_ep(policy, env, max_ep_len=max_ep_len, render=False, pilot_is_human=False):
  """
  一幕的测试过程，使用policy来选择动作，返回总回报，结果，轨迹，动作序列
  但其中的if语句没看懂，init_human_action()根本找不到
  """
  if pilot_is_human:
    global human_agent_action
    global human_agent_active
    human_agent_action = init_human_action()
    human_agent_active = False
  obs = env.reset()
  done = False
  totalr = 0.
  trajectory = None
  actions = []
  for step_idx in range(max_ep_len+1):
      if done:
          trajectory = info['trajectory']
          break
      action = policy(obs[None, :])
      obs, r, done, info = env.step(action)
      actions.append(action)
      if render:
        env.render()
      totalr += r
  outcome = r if r % 100 == 0 else 0
  return totalr, outcome, trajectory, actions


def noop_pilot_policy(obs):
  return 1


def save_tf_vars(scope, path):
  sess = U.get_session()
  saver = tf.train.Saver([v for v in tf.global_variables() if v.name.startswith(scope + '/')])
  saver.save(sess, save_path=path)


def load_tf_vars(scope, path):
  sess = U.get_session()
  saver = tf.train.Saver([v for v in tf.global_variables() if v.name.startswith(scope + '/')])
  saver.restore(sess, path)


copilot_dqn_learn_kwargs = {
  'lr': 1e-3,
  'exploration_fraction': 0.1,
  'exploration_final_eps': 0.02,
  'target_network_update_freq': 1500,
  'print_freq': 100,
  'num_cpu': 5,
  'gamma': 0.99,
}


def make_co_env(pilot_policy, build_goal_decoder=None, using_lander_reward_shaping=False, **extras):
    """
    创建合作驾驶环境，重写了step方法
    使得使用step时自动使用disc_to_cont过滤，并且在obs的最后，自动加入pilot_policy选择动作的onehot编码
    pilot_policy为策略函数，需要返回选择的动作
    build_goal_decoder为是否使用目标解码器，using_lander_reward_shaping表示是否需要使用shaping回报来训练
    """
    env = gym.make('LunarLanderContinuous-v2')
    env.unwrapped.using_lander_reward_shaping = using_lander_reward_shaping
    env.action_space = spaces.Discrete(n_act_dim) # 把环境的动作空间设为n_act_dim
    env.unwrapped.pilot_policy = pilot_policy
    if build_goal_decoder is None:
      obs_box = env.observation_space
      # 向原有的观测空间加入了6维，下限为0，上限为1;其实就是在原本的观测空间后面加上动作价值的onehot编码
      env.observation_space = spaces.Box(np.concatenate((obs_box.low, np.zeros(n_act_dim))), 
                                         np.concatenate((obs_box.high, np.ones(n_act_dim))))
    
    env.unwrapped._step_orig = env.unwrapped._step
    # 重写step方法
    if build_goal_decoder is None:
      def _step(self, action):
        obs, r, done, info = self._step_orig(disc_to_cont(action))
        obs = np.concatenate((obs, onehot_encode(self.pilot_policy(obs[None, :]))))
        return obs, r, done, info
    else:
      goal_decoder = build_goal_decoder()
      def _step(self, action):
        obs, r, done, info = self._step_orig(disc_to_cont(action))
        self.actions.append(self.pilot_policy(obs[None, :]))
        traj = traj_mask_helipad(combined_rollout(self.trajectory[-1:], self.actions[-1:]))
        goal, self.init_state = goal_decoder(traj, init_state=self.init_state, only_final=True)
        obs = mask_helipad(obs, replace=goal)
        return obs, r, done, info
    env.unwrapped._step = types.MethodType(_step, env.unwrapped)
    
    return env


def co_build_act(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None, using_control_sharing=True):
  """
  人机交互中，根据人输入动作和网络输出，选择动作返回;仅仅做决策的时候需要使用这个函数，
  训练时是不需要的(训练的时候直接用replybuffer就好)，训练时只需要这里网络的参数来预测动作就好
  此外，用到这个函数的地方，实际上用的是下面的打包函数
  输入的observations_ph就是env.reset()所返回的状态。所以维度为[9]
  输入：make_obs_ph用于产生placeholder,q_func为评估Q值的函数,
  num_actions动作数目,scope就是命名,reuse,using_control_sharing是否使用共享控制
  返回一个打包函数 act
  act:
  输入：
  observations_ph: 当前状态
  pilot_action_ph: 驾驶员动作            是placeholder name = pilot_action
  pilot_tol_ph   : 原文的容忍度alpha     是placeholder name = pilot_tol
  输出：
  actions：      : 动作
  """
  with tf.variable_scope(scope, reuse=reuse):
    # 下文的None其实都是1
    observations_ph = U.ensure_tf_input(make_obs_ph("observation"))   # observations_ph的shape是 [None,15s] None 表示可以为任意数，取决于输入
    if using_control_sharing:
      pilot_action_ph = tf.placeholder(tf.int32, (), name='pilot_action')       # 驾驶员选择的动作
      pilot_tol_ph = tf.placeholder(tf.float32, (), name='pilot_tol')           # 这个就是原文的容忍度alpha
    else:
      eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))
      stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
      update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

    q_values = q_func(observations_ph.get(), num_actions, scope="q_func")       # q_values.shape [None,6] 就是输入到网络后，得到的Q值

    batch_size = tf.shape(q_values)[0]                                          # batch_size 就是包的大小 None

    if using_control_sharing:
      q_values -= tf.reduce_min(q_values, axis=1)                                  # 取Q值中的最小值，并取负数；对应原文中Q'(s,a) = Q(s,a)-min Q(s,a')，shape是[None,6]
      # 这里的操作很奇怪，这样减的话，只有当q_values=[1,6]或[6,6]才能不报错
      # 首先，tf.reduce_min(q_values, axis=1)会找出q_values每一行的最小值，因为q_values.shape=[None,6],所以式子shape为[None]
      # 当None>1时，上式的操作会使得q_values每一行都减去最小值，且会错位
      # 如 q_values = [[1,2],[3,4]]  所以tf.reduce_min(q_values, axis=1) = [1,3]
      # q_values - tf.reduce_min(q_values, axis=1) = [[0,-1],[2,1]]      这很明显没有任何意义
      # 所以None只能是1，如q_values = [[1,2]] 所以tf.reduce_min(q_values, axis=1) = [1]
      # q_values - tf.reduce_min(q_values, axis=1) = [[0,2]]
      
      opt_actions = tf.argmax(q_values, axis=1, output_type=tf.int32)              # 取得Q'(s,a)最大值的索引，或者说，这个是q_func预测的最佳动作 a max_a Q'(s,a)
      opt_q_values = tf.reduce_max(q_values, axis=1)                               # 取得Q'(s,a)的最大值，最佳动作的Q值，max Q'(s,a)

      batch_idxes = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])        # shape[None,1] 内容为 0-batch_size，步长为1 内容：[0 0]
      reshaped_batch_size = tf.reshape(batch_size, [1])                            # shape[1] 内容0



      # 下面这三句，就是pilot_action_ph应当是一个索引，在Q'(s,a)中取出索引对应的值
      pi_actions = tf.tile(tf.reshape(pilot_action_ph, [1]), reshaped_batch_size)  # 驾驶员选择的动作
      pi_act_idxes = tf.concat([batch_idxes, tf.reshape(pi_actions, [batch_size, 1])], axis=1) # 把pi_actions和batch_idxes在横向上拼接--->[batch_idxes pi_actions] 其shape为[None,2]
      pi_act_q_values = tf.gather_nd(q_values, pi_act_idxes)                       # 驾驶员选择动作的Q值

      # if necessary, switch steering and keep main
      # 下面这三句，就是 mixed_actions 应当是一个索引，在q_values中取出索引对应的值
      mixed_actions = 3 * (pi_actions // 3) + (opt_actions % 3)                    # 把人类选择的动作和最佳动作混合
      # 看意思是把动作融合，但这种融合方法让我难以理解，相当于
      # if pi_actions < 3: 
      #   pi_actions=0
      # elif pi_actions<6: 
      #   pi_actions = 3
      mixed_act_idxes = tf.concat([batch_idxes, tf.reshape(mixed_actions, [batch_size, 1])], axis=1) #新的索引
      mixed_act_q_values = tf.gather_nd(q_values, mixed_act_idxes)                # 混合动作的Q值
      # where(condition, x=None, y=None, name=None)
      # condition中元素为True的元素替换为x中的元素，为False的元素替换为y中对应元素
      # 如果公式成立， mixed_actions 换为 pi_actions
      # 即如果公式成立，那么变为驾驶员的动作，如果不成立则继续用mix的原值
      #----------------------------------------------------------------------------------------------------------#
      # 从这里开始，tf.where会进行3次，每次不等式的右边都是(1 - pilot_tol_ph) * opt_q_values，对应原文中网络推出的最佳动作
      # 但不等式左边有变化
      # 这里，不等式的左侧就是驾驶员动作的Q值，意为：如果驾驶员动作Q值达标，那么直接用驾驶员动作，如果不达标，直接换mix试试
      # 这里换mix是为后面做准备的，因为驾驶员动作不达标，那么当然应该换其他动作，其他动作先从mixed_actions试
      mixed_actions = tf.where(pi_act_q_values >= (1 - pilot_tol_ph) * opt_q_values, pi_actions, mixed_actions)

      # if necessary, keep steering and switch main
      # 如果驾驶员动作达标，那么这两句就把索引换成驾驶员动作对应索引和动作了
      # 如果驾驶员动作不达标，那么仍然是mix动作的索引
      mixed_act_idxes = tf.concat([batch_idxes, tf.reshape(mixed_actions, [batch_size, 1])], axis=1)
      mixed_act_q_values = tf.gather_nd(q_values, mixed_act_idxes)                # 混合动作的Q值

      #--------------------------------------------------------------#
      
      # 这里又用了一种新的混合方式
      steer_mixed_actions = 3 * (opt_actions // 3) + (pi_actions % 3)            # 相对mixed_actions，这个交换了两个变量的位置
      

      #同理，如果mix达标（此时mix可能是驾驶员动作），那么不变，否则用新的混合动作
      mixed_actions = tf.where(mixed_act_q_values >= (1 - pilot_tol_ph) * opt_q_values, mixed_actions, steer_mixed_actions)

      # if necessary, switch steering and main
      mixed_act_idxes = tf.concat([batch_idxes, tf.reshape(mixed_actions, [batch_size, 1])], axis=1)#不解释了
      mixed_act_q_values = tf.gather_nd(q_values, mixed_act_idxes)
      #同理，如果新mix达标（此时mix可能是驾驶员动作），那么不变，否则用最佳动作
      actions = tf.where(mixed_act_q_values >= (1 - pilot_tol_ph) * opt_q_values, mixed_actions, opt_actions)# 如果不公式成立，mixed_actions换为steer_mixed_actions，不成立则保留

      act = U.function(inputs=[
        observations_ph, pilot_action_ph, pilot_tol_ph
      ],
                       outputs=[actions])
    else:
      deterministic_actions = tf.argmax(q_values, axis=1)

      random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
      chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
      stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

      output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
      update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
      act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                       outputs=[output_actions],
                       givens={update_eps_ph: -1.0, stochastic_ph: True},
                       updates=[update_eps_expr])
    return act

def co_build_train(make_obs_ph, q_func, num_actions, optimizer, grad_norm_clipping=None, gamma=1.0,
    double_q=True, scope="deepq", reuse=None, using_control_sharing=True):
  """
  Double DQN训练函数
  进行了训练，并且
  返回：
  act_f：        函数co_build_act返回的动作函数
  train：        一个打包函数，返回：td_error 就是TD误差
                                   optimize_expr 所计算的梯度apply，返回的是一个计算图，需要run
  update_target：用于更新目标网络
  {'q_values': q_values}：字典，返回一个打包函数，返回评估网络，获得在状态 obs_t_input_get 下，各种动作对应q值

  其中的变量使用很多
  """
  act_f = co_build_act(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse, using_control_sharing=using_control_sharing)

  with tf.variable_scope(scope, reuse=reuse):
      # set up placeholders
      obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))                      # 状态————>评估网络使用
      act_t_ph = tf.placeholder(tf.int32, [None], name="action")                 # 动作————>挑选 评估网络评估obs_t_input产生的q值
      rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")               # 汇报
      obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))                  # 状态————>目标网络使用
      done_mask_ph = tf.placeholder(tf.float32, [None], name="done")             # 去除已经done的q值
      importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")  # 不知干啥的

      obs_t_input_get = obs_t_input.get()
      obs_tp1_input_get = obs_tp1_input.get()

      # q network evaluation
      # 用评估网络，获得在状态 obs_t_input_get 下，各种动作对应q值，与act选择使用同样的q-function
      q_t = q_func(obs_t_input_get, num_actions, scope='q_func', reuse=True)  # reuse parameters from act 
      q_func_vars = U.scope_vars(U.absolute_scope_name('q_func'))

      # target q network evalution
      # 用目标网络，获得在状态 obs_tp1_input_get 下，各种动作对应q值
      q_tp1 = q_func(obs_tp1_input_get, num_actions, scope="target_q_func")                     
      target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))

      # q scores for actions which we know were selected in the given state.
      # 只留下act_t_ph所选定动作对应的Q值，应该是评估网络所得的最佳动作Q值
      q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1) 

      # compute estimate of best possible value starting from state at t + 1
      if double_q:
          q_tp1_using_online_net = q_func(obs_tp1_input_get, num_actions, scope='q_func', reuse=True)# 用评估网络，获得在状态 obs_tp1_input_get 下，各种动作对应q值
          q_tp1_best_using_online_net = tf.arg_max(q_tp1_using_online_net, 1)                        # 获得Q值最大位置索引
          q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)# 只留下q_tp1_best_using_online_net所选定动作对应的Q值，即使用评估网络获得最佳动作对应Q值
      else:
          q_tp1_best = tf.reduce_max(q_tp1, 1)
      q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best          # 去除已经done的q值

      # compute RHS of bellman equation
      q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked     # yi ?

      # compute the error (potentially clipped)
      td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
      errors = U.huber_loss(td_error)
      weighted_error = tf.reduce_mean(importance_weights_ph * errors)

      # compute optimization op (potentially with gradient clipping)
      # 下面这个U.minimize_and_clip就已经计算梯度并且对权重进行更新了
      # loss函数为weighted_error，权重为scope='q_func'的权重，即评估网络的权重
      if grad_norm_clipping is not None:
          optimize_expr = U.minimize_and_clip(optimizer,
                                              weighted_error,
                                              var_list=q_func_vars,
                                              clip_val=grad_norm_clipping)
      else:
          optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

      # update_target_fn will be called periodically to copy Q network to target Q network
      update_target_expr = []
      for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                  sorted(target_q_func_vars, key=lambda v: v.name)):
          update_target_expr.append(var_target.assign(var))
      update_target_expr = tf.group(*update_target_expr)

      # Create callable functions
      train = U.function(
          inputs=[
              obs_t_input,
              act_t_ph,
              rew_t_ph,
              obs_tp1_input,
              done_mask_ph,
              importance_weights_ph
          ],
          outputs=td_error,
          updates=[optimize_expr]
      )
      update_target = U.function([], [], updates=[update_target_expr])

      q_values = U.function([obs_t_input], q_t)

  return act_f, train, update_target, {'q_values': q_values}

def co_dqn_learn(
    env,
    q_func,
    lr=1e-3,
    max_timesteps=100000,
    buffer_size=50000,
    train_freq=1,
    batch_size=32,
    print_freq=1,
    checkpoint_freq=10000,
    learning_starts=1000,
    gamma=1.0,
    target_network_update_freq=500,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    num_cpu=5,
    callback=None,
    scope='deepq',
    pilot_tol=0,
    pilot_is_human=False,
    reuse=False,
    using_supervised_goal_decoder=False):
    """
    相当于是对agent的一个训练综合，其中包含了在进行learning_starts时间步后，开始训练
    训练过程中会有参数的保存，数据的记录等，写这么长也只是在组合前两个函数co_build_act，co_build_train的功能，
    和加了些代码控制与打印保存功能
    """
    global human_agent_action
    global human_agent_active

    # Create all the functions necessary to train the model

    sess = U.get_session()
    if sess is None:
      sess = U.make_session(num_cpu=num_cpu) # 获得一个默认会话
      sess.__enter__()

    # Creates a placeholder for a batch of tensors of a given shape and dtype
    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.shape, name=name)
      
    using_control_sharing = pilot_tol > 0
    
    act, train, update_target, debug = co_build_train(
        scope=scope,
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        reuse=reuse,
        using_control_sharing=using_control_sharing
    )
    
    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    replay_buffer = ReplayBuffer(buffer_size)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_trajectories = []
    episode_actions = []
    episode_rewards = []
    episode_outcomes = []
    saved_mean_reward = None
    obs = env.reset()
    prev_t = 0
    episode_reward = 0
    episode_trajectory = []
    episode_action = []
    
    # noop状态下为false
    if pilot_is_human:
      # global human_agent_action
      # global human_agent_active
      human_agent_action = init_human_action()
      human_agent_active = False
    
    if not using_control_sharing:
      exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)
        
    with tempfile.TemporaryDirectory() as td:  # 生成临时文件管理
        model_saved = False
        model_file = os.path.join(td, 'model')
        for t in range(max_timesteps):
            episode_trajectory.append(obs)
            masked_obs = obs if using_supervised_goal_decoder else mask_helipad(obs) # 被掩盖停机位置的obs

            act_kwargs = {}
            if using_control_sharing:
              act_kwargs['pilot_action'] = env.unwrapped.pilot_policy(obs[None, :n_obs_dim])  # 取出obs的最后一行，给pilot_policy，会返回动作
              act_kwargs['pilot_tol'] = pilot_tol if not pilot_is_human or (pilot_is_human and human_agent_active) else 0
            else:
              act_kwargs['update_eps'] = exploration.value(t)
              
            action = act(masked_obs[None, :], **act_kwargs)[0][0] # 直接把观察到的数据和pilot输入数据、pilot_tol传入
            new_obs, rew, done, info = env.step(action)
            episode_action.append(action)

            if pilot_is_human:
              env.render()

            # Store transition in the replay buffer.
            masked_new_obs = new_obs if using_supervised_goal_decoder else mask_helipad(new_obs)
            replay_buffer.add(masked_obs, action, rew, masked_new_obs, float(done))
            obs = new_obs

            episode_reward += rew

            # 如果完成了，并且进行了超过了learning_starts时间步，则开始学习/训练
            # 如果完成了，并且是人类在驾驶，则关闭human_agent_active，睡眠2s
            if done:
                if t > learning_starts:
                  for _ in range(t - prev_t):
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                    td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)

                obs = env.reset()

                episode_outcomes.append(rew)
                episode_rewards.append(episode_reward)
                episode_trajectories.append(episode_trajectory + [new_obs])
                episode_actions.append(episode_action)
                episode_trajectory = []
                episode_action = []
                episode_reward = 0

                if pilot_is_human:
                  # global human_agent_action
                  # global human_agent_active
                  human_agent_action = init_human_action()
                  human_agent_active = False
                  
                prev_t = t
                    
                if pilot_is_human:
                  time.sleep(2)

            # 更新网络
            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            # 计算数据，计算的不是训练数据，而是驾驶员驾驶的数据
            mean_100ep_reward = round(np.mean(episode_rewards[-10:]), 1)
            mean_100ep_succ = round(np.mean([1 if x==100 else 0 for x in episode_outcomes[-100:]]), 2)
            mean_100ep_crash = round(np.mean([1 if x==-100 else 0 for x in episode_outcomes[-100:]]), 2)
            num_episodes = len(episode_rewards)
            # 记录平均回报，成功率和坠毁率
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("mean 100 episode succ", mean_100ep_succ)
                logger.record_tabular("mean 100 episode crash", mean_100ep_crash)
                logger.dump_tabular()
            # 保存权重
            if checkpoint_freq is not None and t > learning_starts and num_episodes > 100 and t % checkpoint_freq == 0 and (saved_mean_reward is None or mean_100ep_reward > saved_mean_reward):
                if print_freq is not None:
                    print('Saving model due to mean reward increase:')
                    print(saved_mean_reward, mean_100ep_reward)
                U.save_state(model_file)
                model_saved = True
                saved_mean_reward = mean_100ep_reward

        if model_saved:
            U.load_state(model_file)

    reward_data = {
      'rewards': episode_rewards,
      'outcomes': episode_outcomes,
      'trajectories': episode_trajectories,
      'actions': episode_actions
    }
          
    return ActWrapper(act, act_params), reward_data

def make_co_policy(
  env, scope=None, pilot_tol=0, pilot_is_human=False, 
  n_eps=None, copilot_scope=None, 
  copilot_q_func=None, build_goal_decoder=None, 
  reuse=False, **extras):
  """
  env是环境，scope是作用域，
  """
  
  if copilot_scope is not None:
    scope = copilot_scope
  elif scope is None:
    scope = str(uuid.uuid4())
  q_func = copilot_q_func if copilot_scope is not None else make_q_func()
    
  return (scope, q_func), co_dqn_learn(
    env,
    scope=scope,
    q_func=q_func,
    max_timesteps=max_ep_len*n_eps,
    pilot_tol=pilot_tol,
    pilot_is_human=pilot_is_human,
    reuse=reuse,
    using_supervised_goal_decoder=(build_goal_decoder is not None),
    **copilot_dqn_learn_kwargs
  )


def str_of_config(pilot_tol, pilot_type, embedding_type, using_lander_reward_shaping):
  return "{'pilot_type': '%s', 'pilot_tol': %s, 'embedding_type': '%s', 'using_lander_reward_shaping': %s}" % (pilot_type, pilot_tol, embedding_type, str(using_lander_reward_shaping))


init_human_action = lambda: [0, 1] # noop


# 0号位置对应上下，0为下，1为上
# 1号位置对应左右，0为左，1为无，2为右
human_agent_action = init_human_action()
human_agent_active = False

LEFT = pygkey.LEFT
RIGHT = pygkey.RIGHT
UP = pygkey.UP
DOWN = pygkey.DOWN

def key_press(key, mod):
  """
  这是一个按键按下转换函数，当按下 上下左右转化为human_agent_action数组中的数值
  并且激活human_agent_active
  """
  global human_agent_action
  global human_agent_active
  a = int(key)
  if a == LEFT:
      human_agent_action[1] = 0
      human_agent_active = True
  elif a == RIGHT:
      human_agent_action[1] = 2
      human_agent_active = True
  elif a == UP:
      human_agent_action[0] = 1
      human_agent_active = True
  elif a == DOWN:
      human_agent_action[0] = 0
      human_agent_active = True

def key_release(key, mod):
  """
  这是一个按键抬起转换函数，当抬起 上下左右转化为human_agent_action数组中的数值
  并且关闭human_agent_active
  """
  global human_agent_action
  global human_agent_active
  a = int(key)
  if a == LEFT or a == RIGHT:
      human_agent_action[1] = 1
      human_agent_active = False
  elif a == UP or a == DOWN:
      human_agent_action[0] = 0
      human_agent_active = False
    
# 转化输入码 return = 上下*3+左右  范围0~5
# down + left    0  下左
# down + none    1  下
# down + right   2  下右
# up   + left    3  上左
# up   + none    4  上
# up   + right   5  上右
def encode_human_action(action):
    return action[0]*3+action[1]


def human_pilot_policy(obs):
  """
  此函数包装了上面的内容，返回人类输入，被转化后的动作码
  """
  global human_agent_action
  return encode_human_action(human_agent_action)


# load pretrained copilot
copilot_path = os.path.join(data_dir, 'pretrained_noop_copilot')
copilot_scope = ''

# 创建一个无解码器，无shaping回报 的 noop驾驶环境
co_env = make_co_env(
  noop_pilot_policy, 
  build_goal_decoder=None, 
  using_lander_reward_shaping=False
)

(scope, q_func), (raw_copilot_policy, reward_data) = make_co_policy(
  co_env, pilot_tol=1e-3, pilot_is_human=False, n_eps=1,
  copilot_scope=copilot_scope,
  copilot_q_func=make_q_func(),
  reuse=False,
  using_lander_reward_shaping=False,
  pilot_policy=noop_pilot_policy,
  build_goal_decoder=None
)

load_tf_vars(copilot_scope, copilot_path)


# balance and randomize experiment order
for i in range(20):
  x, y = '12', '21'
  if np.random.random() < 0.5:
    x, y = y, x
  print('E%d: %s\nE%d: %s' % (2*i, x, 2*i+1, y))


# E0: 21
# E1: 12
# E2: 12
# E3: 21
# E4: 12
# E5: 21
# E6: 21
# E7: 12
# E8: 12
# E9: 21
# E10: 21
# E11: 12
# E12: 21
# E13: 12
# E14: 12
# E15: 21
# E16: 12
# E17: 21
# E18: 21
# E19: 12
# E20: 21
# E21: 12
# E22: 21
# E23: 12
# E24: 12
# E25: 21
# E26: 21
# E27: 12
# E28: 21
# E29: 12
# E30: 12
# E31: 21
# E32: 21
# E33: 12
# E34: 12
# E35: 21
# E36: 21
# E37: 12
# E38: 21
# E39: 12

# intro solo human pilot
pilot_id = 'spike'
n_intro_eps = 20

# 修改键值对应函数？
env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

intro_rollouts = []
# 对人类控制试验20次，并用intro_rollouts记录试验过程参数
time.sleep(10)
for _ in range(n_intro_eps - len(intro_rollouts)):
  intro_rollouts.append(run_ep(human_pilot_policy, env, render=True))
  time.sleep(2)

env.close()

# 保存上面的intro_rollouts
with open(os.path.join(data_dir, '%s_pilot_intro.pkl' % pilot_id), 'wb') as f:
  pickle.dump({pilot_id: list(zip(*(intro_rollouts)))}, f, pickle.HIGHEST_PROTOCOL)


# evaluate solo human pilot
# 下面这一段和intro solo human pilot几乎没有区别
n_eval_eps = 30

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

eval_rollouts = []

time.sleep(10)
for _ in range(n_eval_eps - len(eval_rollouts)):
  eval_rollouts.append(run_ep(human_pilot_policy, env, render=True))
  time.sleep(2)

env.close()

with open(os.path.join(data_dir, '%s_pilot_eval.pkl' % pilot_id), 'wb') as f:
  pickle.dump({pilot_id: list(zip(*(eval_rollouts)))}, f, pickle.HIGHEST_PROTOCOL)


# evaluate copilot with human pilot while fine-tuning
# 创建一个无解码器，无shaping回报 的 human_pilot_policy驾驶环境
co_env = make_co_env(
  human_pilot_policy,
  build_goal_decoder=None,
  using_lander_reward_shaping=False
)

co_env.render()
co_env.env.unwrapped.viewer.window.on_key_press = key_press
co_env.env.unwrapped.viewer.window.on_key_release = key_release

n_eps = 30
pilot_tol = 0.6

time.sleep(10)
(scope, q_func), (raw_copilot_policy, reward_data) = make_co_policy(
  co_env, pilot_tol=pilot_tol, pilot_is_human=True, n_eps=n_eps,
  copilot_scope=copilot_scope,
  copilot_q_func=make_q_func(),
  reuse=True
)

co_env.close()

config_name = str_of_config(pilot_tol, pilot_id, 'rawaction', False)

reward_logs = {config_name: defaultdict(list)}
for k, v in reward_data.items():
  reward_logs[config_name][k].append(v)

with open(os.path.join(data_dir, '%s_reward_logs.pkl' % pilot_id), 'wb') as f:
  pickle.dump(reward_logs, f, pickle.HIGHEST_PROTOCOL)
