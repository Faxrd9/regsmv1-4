# ReGSM

**Loser, learning.**

这个仓库是一份失败记录：我想做“遇到难题时分裂思考”的模型，但一路实验下来，发现很多看起来合理的设计并没有真正带来稳定收益。

我把它放出来，仅作为存在的记录

## 项目目标

最初的想法很简单：

```text
模型遇到问题
  -> 可以继续思考
  -> 可以分裂成多个 branch
  -> 可以从 branch 里选一条更好的路
  -> 最后输出答案
```

这个想法在直觉上像“分裂思考”。但实验结果并不完全支持它。

当前比较明确的结论是：

```text
多分支 selector/router 不稳定
oracle winner 说明 selector 是明显瓶颈
真正稳定有效的是固定两轮 recurrent refinement
V4 默认已经去掉 runtime router / selector / merger
```

## 模型版本

### V1

文件：

- `regsm/regsm.py`
- `train.py`
- `train_v2.py`
- `train_v3.py`

早期探索版本，主要是围绕 ReGSM 的基础结构、训练流程和符号追踪任务做实验。

### V2

文件：

- `regsm/regsm_v2.py`
- `train_select_v2.py`
- `train_text_v2.py`
- `generate_text_v2.py`

V2 重点尝试 selector / router / branch。  
核心问题是 selector 容易追着短期最优 branch 跑，导致分支没有稳定分工。

后来做了 oracle-winner 训练消融：

```text
训练时直接使用当前 CE 最低的 branch
selector 只模仿 oracle
```

这个实验显示：如果 oracle 能显著强于普通 selector，说明 selector 选错路是核心问题之一。

### V3

文件：

- `regsm/regsm_v3.py`
- `train_merge_v3.py`
- `diagnose_merge_v3.py`

V3 尝试 split-think-merge：

```text
输入
  -> 多个 branch 并行处理
  -> merger 合并
  -> 输出
```

结果是：强制 DISPATCH 的版本比 learned router 更好，说明动态 router/selector 依旧是风险点。  
但继续消融后发现，多分支本身也不是稳定关键。

### V4

文件：

- `regsm/regsm_v4.py`
- `train_primary_v4.py`
- `eval_primary_v4.py`
- `train_text_v4.py`
- `generate_text_v4.py`

V4 是目前最干净的版本。默认结构已经不是多分支选择，而是：

```text
输入
  -> 普通 Transformer base blocks
  -> 固定 branch/recurrent block 第 1 轮
  -> 固定 branch/recurrent block 第 2 轮
  -> 输出
```

V4 默认：

```text
runtime router: no
runtime selector: no
runtime merger: no
max_k: 1
max_recurrent: 2
branch_w: 0.2
```

白话说，V4 现在是一个小型固定迭代模型：  
不是“多条路里选一条”，而是“同一条路反复修正两轮”。

## 当前实验结论

在变量追踪任务上，V4 single-branch 的结果最好也最稳定。

一些记录：

```text
V4 single-branch + branch_w=0.2
seed0 20k eval OOD ~= 0.9992
seed3 20k eval OOD ~= 0.9995

V4 single-branch + branch_w=0
seed0 20k eval OOD ~= 0.9996
seed3 20k eval OOD ~= 0.9922

max_recurrent=1
20k eval OOD ~= 0.8745

max_recurrent=2, max_branch=1
20k eval OOD ~= 0.9965
```

我的理解：

```text
多分支不是关键
selector/router 反而容易拖后腿
固定两轮 refinement 是关键
中间态辅助监督有助于稳定
```

## 中文小说实验

文件：

- `data/v4_chinese_novel.txt`
- `train_text_v4.py`
- `generate_text_v4.py`

我写了一篇很短的中文小说《雾城回声》喂给 V4，让它做字符级 next-char 训练。

结果很一般：

```text
corpus_chars ~= 2752
vocab ~= 582
params ~= 0.721M
best val acc ~= 0.1865
```

它能学到一些局部词块，比如“林澈”“录音带”“雾城”“最暗的灯”，但不会真正写小说。生成结果更像碎片拼接。

这个实验提醒我：

```text
符号追踪任务上强，不代表语言生成强
小语料字符模型不等于真正语言能力
架构想法必须放在清楚的任务边界里看
```
