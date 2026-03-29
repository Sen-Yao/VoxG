# Scripts 目录说明

## 目录结构

- analysis/ - 分析脚本
- diagnostics/ - 诊断脚本
- sweep/ - Sweep 相关
- deprecated/ - 废弃脚本（数据泄露）
- *.sh - 运行脚本

## 核心脚本

| 脚本 | 用途 |
|------|------|
| analysis/run_delta_oneclass.py | Delta Vector 半监督评估（正确版） |
| sweep/sweep_manager.py | WandB Sweep 管理 |
| reproduction_photo.sh | Photo 复现脚本 |

## 废弃脚本

以下脚本存在数据泄露问题，仅供参考：

- deprecated/run_delta_semisupervised.py
- deprecated/delta_vector_analysis.py

---

整理时间: 2026-03-26