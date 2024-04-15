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

