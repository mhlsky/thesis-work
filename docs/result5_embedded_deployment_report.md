# 面向嵌入部署的实验综合报告

## 1. 报告目标

本报告在已有精度实验完成后，从**嵌入部署**视角重新整理模型选型结论，重点回答三个问题：

1. 当前哪一类模型预测精度最好；
2. 哪一类模型更适合资源受限设备部署；
3. 精度、模型量级、单样本推理延迟之间应如何取舍。

本报告不是重新训练实验，而是**结合已有结果做部署导向总结**。

## 2. 结果来源与口径说明

### 2.1 精度结论来源

- 基础模型对比：`docs/result_analysis_result_1.md`
- Phys / VMD 独立消融：`docs/result3_analysis_report.md`
- 实验四单 seed 收敛：`docs/result4_main_lite_analysis_report.md`
- 论文汇总与 5-seed 统计：`docs/paper/result_summary/data/result4_final_shortlist_robust.csv`

### 2.2 推理性能与模型量级来源

- 性能实验说明：`docs/experiment5_performance_manual.md`
- CPU 单线程主表：`results/result5_performance_cpu/outputs/performance/model_benchmark.csv`
- GPU 参考表：`results/result5_performance/outputs/performance/model_benchmark.csv`
- 计时实现：`src/ship_motion/benchmark_inference.py`

### 2.3 需要明确的假设

1. 本报告把 `result5_performance_cpu` 的 `cpu_fp32` 作为**嵌入式 CPU 主口径**，因为它是单独 CPU 运行，更适合作为边端部署参考。
2. 性能实验的计时范围只包含**已加载模型的 forward**，**不包含**：
   - checkpoint 加载；
   - 数据读取与归一化；
   - VMD cache 构建；
   - Physics loss；
   - 仿真器接口调用。
3. 因此，若目标设备需要**在线实时计算 VMD 分解或额外预处理**，实际端到端延迟会高于本报告中的 forward 延迟。
4. `result4_robust` 的精度结论优先采用 **5-seed mean ± std**；性能实验中的 CCG / Joint 延迟则来自 **seed=42 checkpoint**，二者用途不同。

## 3. 精度主线结论

从完整实验链条看，项目结论已经比较清晰：

- `result_1` 证明了 **CCG-xLSTM baseline** 在 OOD 泛化上优于 LSTM、GRU、Lite-xLSTM 和 Transformer，是本文主干的起点。
- `result3` 证明了 **Phys** 和 **VMD** 都可能带来收益，但作用方式不同：
  - Phys 更偏向物理一致性和轨迹约束；
  - VMD 更偏向直接改善预测误差。
- `result4` 进一步把主线收敛到：
  - **VMD-no mixer** 是有效结构；
  - **Joint（VMD + CCG + Physics）** 已经超过单独 VMD 候选。

最终以 `result4_robust` 的 5-seed 统计作为正式精度口径时，结论如下：

| 模型 | OOD RMSE mean | OOD RMSE std | 相对 baseline 改善 |
|---|---:|---:|---:|
| Joint (`l=0.05, lr=3e-4, r=0.005`) | 0.180640 | 0.000623 | 5.44% |
| Joint (`l=0.05, lr=3e-4, s=0.001, r=0.005`) | 0.180745 | 0.000575 | 5.38% |
| VMD-CCG (`l=0.03, lr=5e-4, no mixer`) | 0.181291 | 0.001076 | 5.09% |
| CCG-Phys (`p/phi smooth, r=0.0075`) | 0.188913 | 0.006159 | 1.10% |
| CCG Baseline | 0.191022 | 0.007513 | 0.00% |

这说明如果**只看最终预测性能**，当前最优选择已经不是 baseline，而是 **Joint Final**。

## 4. 嵌入部署视角的复杂度与推理表现

### 4.1 CPU 单线程（更接近边端逐步调用）

输入统一为 `batch=1, seq_len=128, input_dim=11`，输出为未来 `10` 步 `5` 个状态量。

| 模型 | OOD RMSE 参考值 | 参数量 (M) | checkpoint (MiB) | P95 延迟 (ms) | 吞吐量 (samples/s) |
|---|---:|---:|---:|---:|---:|
| LSTM | 0.206998 | 0.227 | 0.870 | 1.186 | 859.70 |
| GRU | 0.188521 | 0.176 | 0.675 | 4.076 | 249.34 |
| Transformer Encoder | 0.185287 | 0.290 | 1.114 | 1.336 | 755.98 |
| Lite-xLSTM | 0.203533 | 0.288 | 1.104 | 14.368 | 71.58 |
| CCG-xLSTM Baseline | 0.184785 | 0.363 | 1.393 | 28.660 | 35.51 |
| Joint Final | 0.180341 | 0.399 | 1.532 | 28.998 | 35.44 |

### 4.2 GPU 单样本参考（云服务器 CUDA）

| 模型 | P95 延迟 (ms) | 吞吐量 (samples/s) | 峰值显存 (MiB) |
|---|---:|---:|---:|
| LSTM | 0.864 | 1277.33 | 11.09 |
| GRU | 0.423 | 2392.23 | 10.51 |
| Transformer Encoder | 0.634 | 1596.58 | 12.67 |
| Lite-xLSTM | 45.839 | 22.66 | 10.36 |
| CCG-xLSTM Baseline | 77.044 | 14.11 | 10.65 |
| Joint Final | 73.818 | 14.12 | 10.85 |

## 5. 部署导向分析

### 5.1 在本文自有模型内部，Joint 基本“支配” CCG baseline

如果比较 `CCG-xLSTM Baseline` 和 `Joint Final`，可以看到：

- 参数量：`0.363M -> 0.399M`，仅增加约 `9.9%`
- 包体积：`1.393 MiB -> 1.532 MiB`，仅增加约 `10.0%`
- CPU P95：`28.660 ms -> 28.998 ms`，几乎持平
- OOD 误差：`0.184785 -> 0.180341`（seed=42 性能实验口径），Joint 更优
- 从 5-seed robust 口径看，Joint 相比 baseline 的 OOD 均值提升约 `5.4%`

这意味着：**如果已经决定部署本文提出的模型家族，那么没有明显理由继续优先选 CCG baseline，Joint Final 更值得作为默认部署版本。**

### 5.2 Transformer 是当前最强“轻量部署候选”

从嵌入部署而不是论文主模型的角度看，`Transformer Encoder` 很突出：

- OOD RMSE `0.185287`，已经接近 CCG baseline，距离 Joint 也不算大；
- 参数量只有 `0.290M`，checkpoint 只有 `1.114 MiB`；
- CPU P95 仅 `1.336 ms`，比 Joint 快约 `21.7x`；
- GPU P95 仅 `0.634 ms`，比 Joint 快约 `116x`。

因此，如果部署目标强调：

- 强实时；
- 单样本滚动调用；
- 边端算力有限；
- 需要尽量降低控制周期内模型占用；

那么 **Transformer 比 Joint 更像工程上“容易落地”的方案**。

### 5.3 Lite-xLSTM 不适合作为部署候选

`Lite-xLSTM` 的结果比较明确：

- 精度不占优：OOD RMSE `0.203533`
- CPU 不快：P95 `14.368 ms`
- GPU 也不快：P95 `45.839 ms`

也就是说，它同时没有拿到“更准”或“更快”这两类优势，因此**不建议作为正式部署候选**。

### 5.4 对本文提出模型而言，GPU 不一定比 CPU 更适合单样本在线推理

一个非常关键的现象是：

- `Joint Final` 在 CPU 单线程的 P95 是 `28.998 ms`
- 但在本次 GPU 单样本测试中，P95 反而是 `73.818 ms`

`CCG-xLSTM Baseline` 也有同样趋势。

这说明在 **batch=1、逐步滚动预测** 的场景下，本文模型的当前实现对 GPU 并不友好，可能受到以下因素影响：

- 小 batch 下 kernel launch 开销占比高；
- xLSTM / CCG / Joint 结构没有针对 GPU 单样本路径做特别优化；
- 当前测试是 eager forward，不是 TensorRT / TorchScript / ONNX Runtime 优化后的部署图。

因此，对本文最终模型来说：

- **若是当前实现直接部署**，CPU 反而更可能是更合适的在线推理设备；
- **若希望利用 GPU**，更适合做批量推理、服务端集中部署，或先做图优化/算子融合。

## 6. 最终部署建议

### 6.1 如果目标是论文主模型落地

推荐部署：**Joint Final (`l=0.05, lr=3e-4, r=0.005`)**

理由：

- 当前 5-seed robust OOD 均值最低；
- 相比 CCG baseline，推理成本增加很小；
- 代表本文完整方法，论文叙事最一致。

适用场景：

- 嵌入式 Linux 主机、工控机、边缘计算盒子；
- 控制周期允许模型 forward 预算在 `30 ms` 左右；
- 更重视泛化精度而不是极限低时延。

### 6.2 如果目标是更强实时性 / 更低算力占用

推荐部署：**Transformer Encoder**

理由：

- 精度接近本文主干模型；
- CPU / GPU 单样本延迟都明显更低；
- 模型体积小，工程集成成熟度高。

适用场景：

- 需要更紧的控制周期；
- 希望在较弱 CPU 上先稳定落地；
- 先完成系统集成，再迭代到 Joint。

### 6.3 不建议的候选

- `Lite-xLSTM`：精度和延迟都没有形成优势。
- 早期 `VMD-CCG` / `VMD-CCG-Phys` 原型：历史价值主要在实验演化，不适合作为正式部署版本。
- 单独 `CCG-Phys`：物理叙事有价值，但预测性能和部署收益不如 Joint 明确。

## 7. 风险与后续优化方向

当前报告已经足以支撑“部署选型”，但还需要明确几个风险：

1. **forward 延迟不等于端到端系统延迟。**
   如果设备侧要实时做归一化、缓存维护、VMD 分解、通讯和仿真耦合，实际周期会更长。
2. **CPU 结果对运行环境有一定敏感性。**
   `result5_performance` 与 `result5_performance_cpu` 中的 CPU 数值存在差异，说明正式上线前仍应在目标硬件上复测。
3. **当前 GPU 结果不代表优化后上限。**
   若后续导出 ONNX、TensorRT 或做 batch 化推理，GPU 表现可能改变。

建议的后续工程动作：

1. 在目标边端硬件上补做一次**端到端延迟测试**，把预处理、缓存更新和模型 forward 一起计入。
2. 对 `Joint Final` 优先尝试：
   - ONNX Runtime；
   - TorchScript / `torch.compile`；
   - INT8 或 FP16 量化。
3. 若系统控制周期严格小于 `10 ms`，建议先以 `Transformer Encoder` 作为第一版部署模型。

## 8. 一句话结论

从论文精度结论看，**Joint Final 是当前最优正式模型**；但从嵌入部署角度看，**Transformer 是当前性价比最高的轻量候选**。如果设备可以接受约 `30 ms` 的单次 forward 预算，应优先部署 Joint；如果系统更强调强实时和低算力占用，应优先部署 Transformer。
