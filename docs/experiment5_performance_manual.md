# 实验5手册：模型复杂度与仿真推理性能测试

## 1. 实验目的

实验5不再训练或重新评估预测精度，而是面向后续仿真系统嵌入，测试已有模型在单样本滚动调用场景下的资源开销。

性能测试固定使用：

```text
batch_size = 1
输入形状 = [1, 128, 11]
输出形状 = [1, 10, 5]
```

这对应仿真系统每次拥有一个历史窗口、调用模型预测未来 10 秒状态的场景。

## 2. 比较模型

正式配置 `configs/experiment5_performance.yaml` 只引用已经完成的 checkpoint：

| 类别 | 模型 |
|---|---|
| 公开循环网络基线 | LSTM、GRU |
| 公开注意力基线 | Transformer Encoder |
| xLSTM 主干参考 | Lite-xLSTM |
| 本文主干 | CCG-xLSTM Baseline |
| 本文最终模型 | Joint Final（VMD + CCG + Physics） |

公开基线沿用 `result_1` 的 seed=42 checkpoint。CCG baseline 与 Joint Final 使用 `result4_robust` 的 seed=42 checkpoint；最终 Joint 的 5-seed OOD 统计继续引用已有论文汇总，不在本实验重复计算。

## 3. 输出指标

正式输出目录：

```text
results/<result_name>/outputs/performance/
  model_benchmark.csv
  benchmark_metadata.json
```

`model_benchmark.csv` 的核心字段：

| 分类 | 字段 | 含义 |
|---|---|---|
| 精度参考 | `ood_rmse_reference` | 读取已有 `metrics_ood_test.json`，不会重新评估 |
| 复杂度 | `total_params`、`params_m` | 模型参数量与百万参数量 |
| 权重体积 | `state_dict_tensor_size_mib` | 模型 tensor 原始体积 |
| 部署包体积 | `checkpoint_size_mib` | 实际 `best.pt` 文件大小 |
| 平均响应 | `latency_mean_ms` | 单次前向平均耗时 |
| 实时性 | `latency_p50_ms`、`latency_p95_ms`、`latency_p99_ms` | 延迟分位数，P95 是主要报告指标 |
| 吞吐量 | `throughput_samples_per_sec` | 每秒可完成的单样本预测数 |
| GPU 资源 | `gpu_peak_memory_allocated_mib` | CUDA profile 下的峰值分配显存 |

计时范围只包含已加载模型的 `forward`：不包含 checkpoint 加载、数据读取、标准化、VMD cache 构建、Physics loss 和仿真器接口。

> `Physics` 是训练期损失，因此 CCG-Phys 不会额外增加纯前向推理成本。Joint 相对 CCG 的主要推理增量来自 VMD 多头输出分支，而不是 Physics loss。

## 4. 正式运行

Linux 云服务器默认同时测 CPU 与 CUDA：

```bash
bash scripts/run_result5_performance.sh . result5_performance
```

只测 CPU：

```bash
BENCHMARK_PROFILES=cpu_fp32 \
bash scripts/run_result5_performance.sh . result5_cpu
```

只测 GPU：

```bash
BENCHMARK_PROFILES=cuda_fp32 \
bash scripts/run_result5_performance.sh . result5_cuda
```

只测论文最终 shortlist：

```bash
BENCHMARK_MODELS=ccg_baseline,joint_final \
bash scripts/run_result5_performance.sh . result5_shortlist
```

正式测量先预热 100 次，再记录 1000 次独立前向调用。GPU 在每一次计时前后同步，因此记录的是单请求端到端前向延迟，而非异步提交时间。

运行时脚本会打印阶段 banner、每个 profile 的开始/结束、以及每个模型的加载和完成进度，便于远程 tmux/日志中观察任务是否还在跑。

## 5. Smoke 测试

Linux：

```bash
bash scripts/smoke_result5_performance.sh
```

Windows PowerShell：

```powershell
pwsh -File .\scripts\smoke_result5_performance.ps1
```

Smoke 强制 CPU，仅测试 LSTM 和 Joint，使用 3 次预热与 10 次测量。它不代表正式性能结论，输出仅写入：

```text
outputs/smoke/result5_performance/
```

## 6. 论文表格建议

主表优先选择与实际仿真部署环境一致的一个 profile：

| Model | OOD RMSE | Params (M) | Weight Size (MiB) | P95 Latency (ms) | Throughput (samples/s) |
|---|---:|---:|---:|---:|---:|
| LSTM | existing result |  |  |  |  |
| GRU | existing result |  |  |  |  |
| Transformer | existing result |  |  |  |  |
| Lite-xLSTM | existing result |  |  |  |  |
| CCG-xLSTM | existing result |  |  |  |  |
| Joint Final | existing result |  |  |  |  |

若正式仿真部署在 CPU，应以 `cpu_fp32` 为主表；云服务器 GPU 结果可以放在附录或作为服务端部署参考。最终 Joint 的 robust OOD 应写为 `mean ± std`，其余既有单 seed 基线应在表注中明确标识。
