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