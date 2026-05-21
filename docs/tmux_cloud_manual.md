# 云上 `tmux` 运行手册（适配本项目）

> 这份手册专门讲：  
> **如何在 Linux 云服务器上，使用 `tmux` 稳定运行本项目的正式训练、评估、汇总与绘图流程。**

适用场景：

- 你准备在云上正式跑 `VMD-CCG-Phys-xLSTM` 主线实验；
- 训练时间较长，不希望因为 SSH 断开导致任务中断；
- 你希望同时看：
  - 训练日志
  - GPU 占用
  - 结果文件输出

---

## 1. 为什么推荐 `tmux`

如果你直接在普通 SSH 终端里跑：

```bash
bash scripts/run_full_training.sh
```

一旦发生下面任一情况，训练就可能中断：

- 本地电脑休眠 / 关机
- SSH 连接断开
- 浏览器窗口关闭
- 云平台 Web Shell 自动断开

而 `tmux` 可以把训练放进一个**不会因为你断开连接就消失的终端会话**里。

你可以理解成：

> `tmux` = 云服务器里的“持久终端房间”

你进入这个房间跑训练，退出 SSH 后，房间还在，训练还在继续。

---

## 2. 本项目推荐的 `tmux` 使用方式

本项目已经提供两个 Linux 云上脚本：

```text
scripts/setup_cloud_env.sh
scripts/run_full_training.sh
```

推荐做法：

1. 先用 `tmux` 创建一个训练会话；
2. 先激活你自己的 conda / venv 环境；
3. 在该会话里执行：
   - `bash scripts/setup_cloud_env.sh`
   - `bash scripts/run_full_training.sh`
4. 再开一个 `tmux` 窗口专门看 GPU；
5. 再开一个窗口专门 `tail -f logs/*.log` 看日志。

---

## 3. 第一次登录云服务器后要做什么

### 3.1 进入项目根目录

假设你的项目目录叫：

```text
~/thesis-work
```

那先执行：

```bash
cd ~/thesis-work
```

### 3.2 检查关键目录

建议先确认：

```bash
ls
ls data
ls scripts
```

你至少应该能看到：

- `configs/`
- `scripts/`
- `src/`
- `data/`
- `pyproject.toml`

---

## 4. 安装 `tmux`

Ubuntu / Debian 常用安装方式：

```bash
sudo apt update
sudo apt install -y tmux
```

检查是否安装成功：

```bash
tmux -V
```

正常会看到类似：

```text
tmux 3.x
```

---

## 5. 创建训练会话

### 5.1 新建会话

推荐会话名：

```bash
tmux new -s thesis_train
```

含义：

- `new`：新建会话
- `-s thesis_train`：会话名字叫 `thesis_train`

进入后，你会看到一个新的命令行界面。  
后续训练命令就在这里面跑。

---

## 6. 在 `tmux` 中初始化环境

进入项目根目录后，先执行：

```bash
conda activate your_env
bash scripts/setup_cloud_env.sh
```

如果你不是 conda，而是 venv，请改成你自己的激活方式；核心要求只有一个：

> 进入脚本前，先确保当前 `python` 就是你想复用的那个环境里的 Python。

这个脚本会做：

1. 检查系统与 Python
2. 检查当前 Python 环境里是否已安装关键依赖
3. 检查 GPU
4. 检查 `torch.cuda.is_available()`
5. 检查数据目录
6. 执行最小 smoke：
   - Step 01 数据 smoke
   - Step 02 LSTM smoke
   - Step 03 VMD smoke

### 6.1 什么时候算通过

如果脚本正常结束，并看到类似：

```text
==> Cloud environment setup finished
```

说明云上环境基本可用。

---

## 7. 在 `tmux` 中运行正式训练

环境初始化通过后，在同一个 `tmux` 会话里执行：

```bash
bash scripts/run_full_training.sh
```

这个脚本会顺序执行：

1. 直接复用当前已激活环境里的 `python`
2. 构建正式 VMD 缓存
3. 训练主线模型：
   - LSTM
   - GRU
   - Transformer
   - Lite-xLSTM
   - CCG-xLSTM
   - VMD-CCG-xLSTM
   - VMD-CCG-Phys-xLSTM
4. 导出 predictions
5. 汇总结果
6. 生成图表

### 7.1 日志会写到哪里

脚本里每一步都会写独立日志到：

```text
logs/
```

例如：

- `logs/train_lstm.log`
- `logs/train_vmd_ccg_phys_xlstm.log`
- `logs/eval_vmd_ccg_phys_xlstm.log`
- `logs/summarize_results.log`
- `logs/plot_results.log`

---

## 8. 如何“退出但不停止训练”

这是 `tmux` 最重要的操作。

按下面两个键：

```text
Ctrl + b
然后按 d
```

这叫：

```text
detach
```

意思是：

- 你退出这个 `tmux` 会话
- 但训练继续在云服务器后台跑

如果你只是想暂时断开 SSH，一定要用这个方式，而不是直接把训练窗口关掉。

---

## 9. 以后怎么重新接回训练界面

重新登录云服务器后，先看当前会话：

```bash
tmux ls
```

你会看到类似：

```text
thesis_train: 1 windows (created ...)
```

然后重新接回去：

```bash
tmux attach -t thesis_train
```

这样你就会回到之前的训练界面。

---

## 10. 推荐的多窗口工作流

正式训练时，不建议只开一个窗口。  
推荐至少开 3 个窗口。

### 窗口 1：主训练窗口

在这里运行：

```bash
bash scripts/run_full_training.sh
```

### 窗口 2：GPU 监控

在 `tmux` 里按：

```text
Ctrl + b
然后按 c
```

新建一个窗口后运行：

```bash
watch -n 1 nvidia-smi
```

作用：

- 每 1 秒刷新一次 GPU 状态
- 可以看显存占用、算力利用率、进程状态

### 窗口 3：日志监控

再新建一个窗口，运行：

```bash
tail -f logs/train_vmd_ccg_phys_xlstm.log
```

或者先看某个阶段日志：

```bash
ls logs
```

再决定要跟哪一个。

---

## 11. 在 `tmux` 里常用的几个快捷键

### 11.1 新建窗口

```text
Ctrl + b
然后按 c
```

### 11.2 切换到下一个窗口

```text
Ctrl + b
然后按 n
```

### 11.3 切换到上一个窗口

```text
Ctrl + b
然后按 p
```

### 11.4 查看窗口列表

```text
Ctrl + b
然后按 w
```

### 11.5 会话分离（最重要）

```text
Ctrl + b
然后按 d
```

---

## 12. 本项目推荐的标准运行流程

这里给你一套最稳的顺序。

### 12.1 登录服务器

```bash
ssh your_user@your_server
```

### 12.2 进入项目目录

```bash
cd ~/thesis-work
```

### 12.3 创建 `tmux` 会话

```bash
tmux new -s thesis_train
```

### 12.4 运行环境初始化

```bash
bash scripts/setup_cloud_env.sh
```

### 12.5 运行正式全流程

```bash
bash scripts/run_full_training.sh
```

### 12.6 临时退出

```text
Ctrl + b
然后按 d
```

### 12.7 以后回来继续查看

```bash
tmux attach -t thesis_train
```

---

## 13. 如何只跑某一部分，而不是全流程

有时你不想一次跑完全部，而是只想跑最终主模型。

### 13.1 只构建 VMD 缓存

```bash
python -m ship_motion.data.vmd --config configs/vmd_ccg_phys_xlstm.yaml
```

### 13.2 只训练最终主模型

```bash
python -m ship_motion.train --config configs/vmd_ccg_phys_xlstm.yaml
```

### 13.3 只导出最终主模型 predictions

```bash
python -m ship_motion.evaluate \
  --config configs/vmd_ccg_phys_xlstm.yaml \
  --checkpoint outputs/vmd_ccg_phys_xlstm_seq128_pred10/best.pt \
  --split all \
  --save-predictions
```

### 13.4 只做汇总和绘图

```bash
python -m ship_motion.summarize_results \
  --ablation-config configs/ablation_list.yaml \
  --run-output-root outputs \
  --summary-output-dir outputs/summary

python -m ship_motion.plot_results \
  --ablation-config configs/ablation_list.yaml \
  --run-output-root outputs \
  --summary-output-dir outputs/summary
```

---

## 14. 如何后台运行并同时保留 `tmux`

通常来说：

- 有 `tmux` 就已经够用了
- 不一定还要 `nohup`

也就是说，你**直接在 `tmux` 里运行训练**就行，不必再多套一层：

```bash
nohup bash scripts/run_full_training.sh &
```

对这个项目来说，推荐：

```text
tmux + 正常前台执行脚本
```

理由：

- 更容易看实时输出
- 更容易中途检查
- 更容易手动重跑某个阶段

---

## 15. 训练时重点关注什么

### 15.1 GPU 是否在工作

看：

```bash
watch -n 1 nvidia-smi
```

重点看：

- GPU Memory Usage
- GPU Utilization

如果一直 0%，说明训练可能没真正用上 GPU。

### 15.2 日志是否持续更新

例如：

```bash
tail -f logs/train_vmd_ccg_phys_xlstm.log
```

如果长时间没有新内容，要检查：

- 程序是否卡住
- 数据路径是否有问题
- 显存是否爆了

### 15.3 输出目录是否在增长

训练完成后应能看到：

```text
outputs/{run_name}/best.pt
outputs/{run_name}/metrics_*.json
outputs/{run_name}/predictions_*.npz
```

Step 07 完成后应能看到：

```text
outputs/summary/
outputs/summary/figures/
```

---

## 16. 常见问题排查

### 16.1 `tmux: command not found`

安装：

```bash
sudo apt update
sudo apt install -y tmux
```

### 16.2 `uv: command not found`

重新执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 16.3 `nvidia-smi` 不存在

说明：

- 云平台没有给你 GPU
- 或驱动没准备好

这时先不要正式训练深度模型，先确认机器类型。

### 16.4 断线后找不到会话

先看：

```bash
tmux ls
```

如果没有任何会话，说明：

- 你之前没有在 `tmux` 里运行
- 或会话已经结束

### 16.5 输出目录为空

先检查：

```bash
ls outputs
ls logs
```

再看日志：

```bash
tail -n 100 logs/train_vmd_ccg_phys_xlstm.log
```

### 16.6 训练太久

建议分阶段跑：

1. 先只跑 `lstm`
2. 再只跑 `vmd_ccg_phys_xlstm`
3. 确认结果正常，再跑全量主线

---

## 17. 推荐的会话命名方式

如果你只跑一个实验，会话名用：

```bash
tmux new -s thesis_train
```

如果你有多个实验任务，建议命名更明确：

```bash
tmux new -s thesis_main
tmux new -s thesis_ablation
tmux new -s thesis_debug
```

这样后面不会混。

---

## 18. 给你的最简版命令清单

如果你只想记最核心的几条：

### 第一次进入云服务器

```bash
cd ~/thesis-work
sudo apt update && sudo apt install -y tmux
tmux new -s thesis_train
bash scripts/setup_cloud_env.sh
bash scripts/run_full_training.sh
```

### 中途退出但不停训练

```text
Ctrl + b
然后按 d
```

### 重新回来接上

```bash
tmux attach -t thesis_train
```

### 看 GPU

```bash
watch -n 1 nvidia-smi
```

### 看日志

```bash
tail -f logs/train_vmd_ccg_phys_xlstm.log
```

---

## 19. 最后建议

对你这个项目，最稳的做法不是“一上来跑全套然后不管”，而是：

1. `tmux` 建会话  
2. 先跑 `setup_cloud_env.sh`  
3. 再跑 `run_full_training.sh`  
4. 开 GPU 监控窗口  
5. 开日志窗口  
6. 定期回来查看  

这样最适合长时间正式实验，也最不容易因为断线白跑。

