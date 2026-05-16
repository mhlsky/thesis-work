# 实验代码分步骤实现索引

这些文档按顺序交给 AI 编写代码。不要跳步，也不要一次性实现全部模块。

推荐顺序：

1. `01_data_pipeline.md`：数据读取、滑窗、标准化，返回 `x_exog` 和 `x_state`。
2. `02_training_lstm.md`：Persistence、LSTM、GRU、Transformer Encoder、TCN（可选）、训练和评估框架。
3. `03_vmd_module.md`：VMD 分解、缓存、Dataset 返回 `y_modes`。
4. `04_xlstm_backbone.md`：Lite-xLSTM 与控制/风场条件门控 CCG-xLSTM。
5. `05_vmd_ccg_integration.md`：VMD 多分支预测头、Delta Decoder、State Coupling Mixer，形成 VMD-CCG-xLSTM。
6. `06_physics_final_model.md`：加入物理约束，形成主线最终模型 VMD-CCG-Phys-xLSTM。
7. `07_experiments.md`：主线消融实验、结果汇总、绘图。
8. `08_optional_attention.md`：线性注意力可选增强，主线完成后再做。

核心原则：

- 每一步完成后先验收，再做下一步；
- 每一步都必须提供对应的轻量 smoke-test 脚本或命令，用于在本地低性能机器上验证该步骤的端到端流程；
- smoke test 只验证流程正确性，不追求指标；应使用极小样本、极少 epoch/step、小 batch，并将输出写入 `outputs/`；
- 不要求本地完成正式训练、完整消融或全量评估；这些实验默认迁移到租用算力平台执行；
- VMD 是必做模块，但要作为辅助标签使用，不作为测试输入特征；
- CCG 是场景化创新：外生控制/风场量参与 xLSTM 门控；
- Delta Decoder 和 State Mixer 是轻量结构创新，放在 VMD-CCG-xLSTM 集成步骤；
- 物理约束是主线必做，形成 `VMD-CCG-Phys-xLSTM`；
- 线性注意力是最后的可选增强，不阻塞主线；
- 任何模型最终预测输出都统一为 `[B, pred_len, 5]`；
- 指标必须在反标准化后的真实物理尺度上计算；
- OOD test 只能最终测试，不参与训练和调参。

推荐 smoke 脚本命名：

```text
scripts/smoke_step_01_data.ps1
scripts/smoke_step_02_training.ps1
scripts/smoke_step_03_vmd.ps1
scripts/smoke_step_04_xlstm.ps1
scripts/smoke_step_05_vmd_ccg.ps1
scripts/smoke_step_06_physics.ps1
scripts/smoke_step_07_experiments.ps1
scripts/smoke_step_08_attention.ps1
```

新增步骤应优先提供更明确的分步骤 smoke 脚本，方便在本地逐步验收。

