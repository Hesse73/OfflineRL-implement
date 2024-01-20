# BCQ implement

## 介绍

代码中包含两个文件 `behavior.py` 和 `dis_bcq.py`:
- `behavior.py`: 使用Tianshou强化学习库运行DQN算法，并收集交互数据。数据和模型将会保存在`./dqn`的目录下
- `dis_bcq.py`: 实现BCQ算法，从收集好的离线数据中训练模型，并测试

需要的库：
```
gymnasium==0.29.1
numpy==1.24.3
torch==2.0.1
tianshou==0.5.1
tqdm==4.65.0
```

## 生成离线数据

运行：
```sh
python behavior.py --generate_data
```

即可训练DQN网络并收集1000条轨迹，分别保存在 `./dqn/policy.ckpt` 和 `./dqn/offline_dataset` 中。

## 运行BCQ算法

在收集好离线数据之后，运行：
```sh
python dis_bcq.py
```

即可训练BCQ网络，在online环境中测试，包括计算收益和展示GUI效果。

最终结果：
Average reward: -139.840000 , Max reward: -84.000000