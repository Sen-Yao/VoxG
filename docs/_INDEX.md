# VoxG 项目索引

> **最后更新**: 2026-03-27
> **维护者**: Nexus

---

## 📁 目录结构

### 📂 根目录（人类维护）

| 文档 | 路径 | 说明 |
|------|------|------|
| **项目概述** | docs/PROJECT_DETAILS.md | 架构、数据集、超参数 |
| **数据集说明** | docs/dataset_info.md | 数据集详细信息 |
| **相关工作** | docs/related_works/ | 相关论文笔记 |
| **提案** | docs/proposals/ | 实验提案 |

### 📂 nexus/（Agent 维护）

| 文档 | 路径 | 说明 |
|------|------|------|
| **基准测试** | nexus/benchmarks/ | 各数据集最佳结果 |
| **实验日志** | nexus/experiments/ | 实验记录 |
| **工作日记** | nexus/diary/ | 每日工作记录 |
| **洞察** | nexus/insights/ | 关键发现 |
| **报告** | nexus/reports/ | 完整报告 |

---

## 🎯 项目目标

**核心目标**: 图异常检测 (GAD) 半监督方法，达到 SOTA

**目标性能**:

| 数据集 | 目标 AUC | SOTA | 来源 |
|--------|---------|------|------|
| Photo | 0.8960 | completed prior work | 5% 半监督 |
| Amazon | 0.9391 | completed prior work | 5% 半监督 |
| Reddit | 0.5782 | completed prior work | 5% 半监督 |
| Elliptic | 0.8509 | RHO | 10% 半监督 |
| T-Finance | 0.8988 | completed prior work | 5% 半监督 |
| Tolokers | 0.6612 | completed prior work | 5% 半监督 |

---

## 📊 当前进度

### Photo 数据集

| 版本 | 最佳 5-seed AUC | 状态 |
|------|----------------|------|
| V1 | 0.8966 (单 seed) | ⚠️ 未验证 |
| V2 | 0.8746±0.0751 | ❌ 未达 SOTA |
| V3 | 🔄 运行中 | kb49oibp sweep |

---

## ⚠️ 已知问题

1. **Seed 敏感性高**: 同一配置不同 seed 结果差异可达 0.2
2. **标准差大**: 5-seed 结果不够稳定
3. **batch_size 影响**: 大 batch (1024) 表现更好

---

_此索引由 Nexus 维护，每次重要进展后更新_
