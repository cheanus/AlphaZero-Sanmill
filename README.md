# AlphaZero Sanmill
一个适用于直棋（Sanmill）的AlphaZero训练及测试程序，基于开源项目[alpha-zero-general](https://github.com/suragnair/alpha-zero-general)。测试时提供了难度自平衡的人机对弈功能，以提高游玩体验。

本项目的直棋所选用的特殊规则如下：
- 棋盘上有4条斜线
- 超过100步判定和棋
- 当一方剩3枚棋子时可飞棋
- 允许吃三连中的子

## 优化
相比于原项目，本项目主要做了以下优化：
- 使用多进程加速Self-play和Pitting
- 使用orjson加速训练样本数据的读写
- 更改reward的计算方式，使其重视步数
- 忽略loss中对无效action项的计算
- 在Training的每个epoch后添加Validation步骤
- 使用类似于Vision Transformer的结构作为backbone
- 优化并丰富了终端输出的信息
- 调整了MCTS以适配直棋规则
- 引入基于指数移动平均（EMA）的对弈时难度动态自平衡机制

## 环境
本项目使用了以下第三方库：
- pytorch
- tqdm
- coloredlogs
- orjson

## 训练
运行`main.py`即可开始训练：
```bash
python main.py
```
如有需要，可以修改`main.py`中的参数，主要有：
- `numIters`：一次Iteration中Self-play的次数
- `numEps`：整个项目运行过程中Iteration的次数，应是`num_processes`的整数倍
- `tempThreshold`：在Self-play过程中，当步数大于该值时，会直接选择最大概率的动作
- `updateThreshold`：在Pitting过程中，当胜率大于该值时，会更新最佳模型
- `maxlenOfQueue`：一次Self-play过程中，保存用于后续训练的最大棋局数
- `numMCTSSims`：每次MCTS搜索的次数
- `arenaCompare`：每次Pitting过程中，进行对弈的次数，应是`2*num_processes`的整数倍
- `cpuct`：MCTS搜索中的探索参数
- `checkpoint`：保存模型的路径
- `numItersForTrainExamplesHistory`：保存用于后续训练的最大Iteration数
- `num_processes`：Self-play和Pitting时使用的进程数
- `lr`：学习率

训练过程中会自动保存模型，训练结束后会自动保存最后一次训练的模型。

**训练技巧**：遇到模型不收敛的情况时，可以尝试增大`numMCTSSims`，减小`lr`。

## 测试
运行`pit.py`即可开始测试：
```bash
python pit.py
```
如有需要，可以修改`pit.py`中的参数，主要有：
- `difficulty`：改变AI的难度，范围在-1到1之间，越大越难。
- `numMCTSSims`：每次MCTS搜索的次数
- `cpuct`：MCTS搜索中的探索参数
- `eat_factor`：AI对吃子的重视权重因子，越大越重视

## 相关项目
- [alpha-zero-general](https://github.com/suragnair/alpha-zero-general)  
  通用的AlphaZero训练及测试程序
- [Sanmill](https://github.com/calcitem/Sanmill)  
  安卓平台上一个直棋游戏的实现
