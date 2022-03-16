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

**2022.3.9 14：47**

已经完成对gym环境的初步改造，完成了随机降落位置，完成后变色，修改回报（使用发动机扣分，船体撞击-100，降落到指定位置+150，只是平稳降落+50）

并且对其进行了测试：

1. 使用原始Double-DQN测试，测试结果在`logs\gradient_tape\20220309-145936_test-origin-model`中

2. 以原始Double-DQN为预训练模型，训练1000次后；进行测试；结果记录在`logs\gradient_tape\new_GymEnv_test_log_no_statueback`和`logs\gradient_tape\new_GymEnv_train_log_no_statueback`.

下一步继续改造gym，返回是否坠毁，返回旗帜位置。

需要进行的测试如下：

在可以得到旗帜位置情况下进行训练，并且测试效果。

**2022.3.9 21：23**

`double-DQN_p2_9in.py`是针对9输入（旗帜位置）写的，除了增加追汇率和成功率的，打印别的没有更新。

现在会返回旗帜位置，返回是否坠毁；并且用新的网络进行了训练，但效果不佳，训练过程在`logs\gradient_tape\20220309-183053_9in-mode-train`，测试结果在`20220309-214020_9in-mode-test100`

猜测是因为把shape的得分关闭，导致了飞行器在运行过程中过于不稳，计划把shape重新加入再测试。

---



**2022.3.11**

给Gym的界面加了倒计时

加入了 shape得分进行训练，效果不佳，感觉agent对旗帜位置这个信息的利用率不高？

训练过程在`20220310-083328_9in-mode-train-addshaping`，测试结果在`20220310-104534_9in-mode-train-addshaping-test100`

换了一种shape形式，是原文作者给出的shape形式；训练后，结果也很差；训练过程在：`20220310-120653_9in-mode-train-newshaping`，测试过程在`20220310-145120_9in-mode-train-newshaping`

`double-DQN_p2_9in.py`加载模型和储存模型现在可以为两个路径，加入了每100次保存一下模型。

目前，存在一个问题：

返回降落位置后，为什么训练的agent无法完成任务？是网络结构，超参数，还是别的问题？

但无论怎样，不会是DDQN代码的问题，因为之前训练原版月球车都没出问题。现在有个想法是使用迁移学习：

1. 第一种想法，是迁移之前8输入的网络权重前两层，然后在9输入中训练
2. 第二种，是在9输入中训练，但不随机降落位置；训练好后再开启降落位置随机
3. 第三种想法是，使用更大的网络架构。

无法训练成功的问题可以以后再纠，现在必须进行下一步了。

---



2022.3.16

今天修改了Gym环境，加入了可以进行人的控制；但游戏速度过快，只能通过`sleep`模块来减慢速度。下面要加入开关，可以选择性的使用人类输入。具体使用示例为`show_test.py`
