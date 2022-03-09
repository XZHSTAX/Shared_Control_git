说明：

`log`中是tensoboard的数据

`model`是储存的训练模型

`double-DQN.py` 是最原始的Double-DQN

`double-DQN_p1.py`在原始上加了些功能（数据记录等）

`LunarLander_SC.py`是修改了Gym中原有的环境。为了使用，需要

1. 在`envs\py_new2\Lib\site-packages\gym\__init__.py`中加入如下代码：

```python
register(
    id="LunarLander_SC-v2",
    entry_point="gym.envs.box2d:LunarLander_SC",
    max_episode_steps=1000,
    reward_threshold=200,
)
```

2. 在`envs\py_new2\Lib\site-packages\gym\envs\box2d\__init__.py`

中加入：

```python
from gym.envs.box2d.LunarLander_SC import LunarLander_SC
```

3. 把这个`LunarLander_SC.py`文件丢入`envs\py_new2\Lib\site-packages\gym\envs\box2d`

# 问题

现在不降落在旗帜里面也会得分。

# 开发日志

`20220307-220108_DDQN-p1-test`是训练好的DDQN，测试结果

2022.3.9 14：47

已经完成对gym环境的初步改造，完成了随机降落位置，完成后变色，修改回报（使用发动机扣分，船体撞击-100，降落到指定位置+150，只是平稳降落+50）

并且对其进行了测试：

1. 使用原始Double-DQN测试，测试结果在`logs\gradient_tape\20220309-145936_test-origin-model`中

2. 以原始Double-DQN为预训练模型，训练1000次后；进行测试；结果记录在`logs\gradient_tape\new_GymEnv_test_log_no_statueback`和`logs\gradient_tape\new_GymEnv_train_log_no_statueback`.

下一步继续改造gym，返回是否坠毁，返回旗帜位置。

需要进行的测试如下：

在可以得到旗帜位置情况下进行训练，并且测试效果。

2022.3.9 21：23

现在会返回旗帜位置，返回是否坠毁；并且用新的网络进行了训练，但效果不佳，训练过程在`logs\gradient_tape\20220309-183053_9in-mode-train`，测试结果在`20220309-214020_9in-mode-test100`

猜测是因为把shape的得分关闭，导致了飞行器在运行过程中过于不稳，计划把shape重新加入再测试。