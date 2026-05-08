# VoxG 项目路线图

> 最后更新: 2026-03-30

---

## 🎯 项目目标

**超越 VecGAD/RHO 的图异常检测性能**

| 数据集 | 目标 AUC | 当前 SOTA | 来源 |
|--------|---------|-----------|------|
| Amazon | >0.9391 | 0.9391 | VecGAD |
| Reddit | >0.5782 | 0.5782 | VecGAD |
| Photo | >0.8960 | 0.8960 | VecGAD |
| Elliptic | >0.8509 | 0.8509 | RHO |
| T-Finance | >0.8988 | 0.8988 | VecGAD |
| Tolokers | >0.6612 | 0.6612 | VecGAD |

---

## 📁 项目结构

nexus/
├── investigations/    # 探究记录（核心）
│   └── YYYY-MM-DD-name/
│       ├── README.md      # 探究概述
│       ├── hypothesis.md  # 假设列表
│       ├── experiments/   # 实验相关
│       │   ├── scripts/   # 分析脚本
│       │   ├── configs/   # Sweep 配置
│       │   ├── outputs/   # 输出数据
│       │   └── plots/     # 可视化
│       └── insights.md    # 发现洞见
├── inbox/            # 收件箱（想法、问题）
├── diary/            # 工作日记
├── benchmarks/       # 性能基准
└── ROADMAP.md       # 本文件

---

## 🔬 当前工作

### Token Mode 对比实验
- **状态**: ✅ 完成
- **数据集**: Photo, Amazon, Elliptic
- **发现**: Delta 模式在不同密度图上效果差异大

---

## 📅 里程碑

| 日期 | 事件 |
|------|------|
| 2026-03-26 | Delta Vector 假设形成 |
| 2026-03-27 | Photo Sweep v3 启动 |
| 2026-03-28 | 项目结构重构 |
| 2026-03-30 | Token Mode 对比完成，文件结构重构 |

---

## 🔗 快速链接

- 本地知识库: knowledge/Nexus/VoxG/
- 远程代码: HCCS86:~/VoxG/
- WandB 项目: Senyao/VoxG

---

_Nexus 科研自动化执行_
