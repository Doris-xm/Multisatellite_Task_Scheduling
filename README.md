# README

```text
Multisatellite_Task_Scheduling/
│
├── README.md             # 项目说明文件
│
├── requirements.txt      # 项目依赖文件
│
├── config.py             # 配置文件，包含超参数
│
├── utils.py              #
│
├── models/
│   ├── __init__.py
│   ├── resfcn.py         # ResFCN
│   └── ddqn_agent.py       # ddqn智能体定义
│
├── trainers/
│   ├── __init__.py
│   └── ddqn_trainer.py   # 训练器
│
├── simulators/
│   ├── __init__.py
│   └── satellite_simulator.py # 卫星任务调度模拟器
│
├── scripts/
│   ├── evaluate_model.py  # 评估
│   └── run_simulation.sh  # 运行仿真脚本?
│
└── experiments/
    ├── ...
```

根据您提供的论文PDF和分析的Word文档，我们可以梳理出强化学习中主要的几个模块：环境（Environment）、模型（Model）、训练器（Trainer），以及它们各自需要维护的要素。

### 环境（Environment）
需要维护以下：
1. **状态空间（State Space）**：最大资源卫星数 x 最大beam数 x schedule_horizon中的grid数量。其中，1表示被占用，0表示空闲。
2. **动作空间（Action Space）**：输出[0,资源卫星数], 表示对当前任务分配的卫星编号，0表示不分配。
4. **奖励函数（Reward Function）**：
5. **观察（Observation）**：提供当前状态。
7. **模拟（Simulation）**：卫星的剩余能量、卫星是否处于region中，触发事件。

### 模型（Model）
模型是强化学习中的智能体，它需要维护以下要素：
1. **策略（Policy）**：定义了智能体如何根据当前状态选择动作。
2. **价值函数（Value Function）**：估计当前状态或状态-动作对的期望回报。
3. **Q函数（Q Function）**：估计给定状态和动作的期望回报。
4. **神经网络（Neural Network）**：如残差全连接网络（ResFCN），用于特征提取和Q函数的近似。
5. **目标网络（Target Network）**：用于稳定训练过程的另一个参数相同的网络。
6. **经验回放（Experience Replay）**：存储历史经验，用于训练神经网络。

### 训练器（Trainer）
训练器负责智能体的学习和训练过程，它需要维护以下要素：
1. **优化算法（Optimization Algorithm）**：如梯度下降，用于更新模型的参数。
2. **损失函数（Loss Function）**：用于计算预测值和目标值之间的差异。
3. **训练循环（Training Loop）**：控制训练过程，包括数据采样、损失计算和参数更新。
4. **探索策略（Exploration Strategy）**：如ε-greedy，平衡探索和利用。
5. **学习率（Learning Rate）**：控制梯度下降的步长。
6. **折扣因子（Discount Factor）**：定义了未来奖励的当前价值。
7. **训练数据（Training Data）**：包括状态、动作、奖励和下一个状态的集合。
8. **评估（Evaluation）**：定期评估智能体的性能，并与基线算法进行比较。

### 其他考虑因素
- **事件触发机制（Event-Triggered Mechanism）**：处理紧急任务的策略，可能需要修改当前的调度策略。
- **性能比较（Performance Comparison）**：与论文中提到的其他算法进行性能比较，以验证所提方法的有效性。
- **模拟环境（Simulation Environment）**：构建一个模拟环境来模拟卫星任务调度过程，以便在没有实际硬件的情况下测试算法。



### Assumption 或 未处理的情况
1. task执行时长 ＜ schedule_horizon
2. 同一个task satellite生成的task可执行窗口不重叠
3. 不可预定义task数量