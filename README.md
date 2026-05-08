# VoxG

> **VoxG 是一个由 Nexus 主导开发和探索的半监督图异常检测（GAD）研究孵化器。**
>
> 它用于承载尚未成熟的 semi-supervised GAD 想法、机制假设、诊断脚本、快速原型和复现实验。VoxG 的定位不是单一方法仓库，而是一个综合探索平台。

## 项目定位

VoxG 当前定义为：

- **综合性探索项目**：用于孵化不同的半监督 GAD 方向，而不是只维护一个固定模型。
- **Agent-driven research incubator**：由 Nexus 负责组织实验、诊断、文档和阶段性整理。
- **半监督 GAD 实验场**：聚焦少标注正常节点条件下的图异常检测，包括表示学习、参考构造、伪异常生成、结构诊断、分数融合等方向。
- **不成熟想法的隔离区**：新想法可以先在 VoxG 中验证；成熟后再拆分为独立项目。

## 与其他项目的关系

VoxG **完全独立于**以下项目：

| 项目 | 关系 |
|---|---|
| `VecGAD` | 用户已经完成的既有工作 / 外部成果，不作为 VoxG incubator 的继续开发内容 |
| `DualRefGAD` | 已拆分为独立活跃项目，不再由 VoxG 承载主开发 |
| `Nexus` knowledge base | 已迁移到 openclawvm 的 Nexus workspace，不再把 VoxG 内部 `nexus/` 作为主知识库 |

DualRefGAD 独立仓库：

```text
git@github.com:Sen-Yao/DualRefGAD.git
```

Nexus 研究记录主位置：

```text
/home/openclawvm/.openclaw/workspace/agents/nexus
```

## 关于 VecGAD

VecGAD 是用户已经完成的既有工作，不属于 VoxG incubator 的当前探索资产。

VoxG 后续不保留 VecGAD 专属代码、复现脚本或文档；如果需要引用 VecGAD，只作为已完成工作的背景或基线来源，而不是 VoxG 的开发主线。

## VoxG 中适合保留的内容

```text
core exploratory code
semi-supervised GAD prototypes
analysis / diagnostic scripts
small reproducibility scripts
configuration templates
method notes and design docs
```

## VoxG 中不应保留为 Git 资产的内容

```text
dataset/
logs/
wandb/
GGAD/
checkpoints
large binary arrays
manual backup snapshots
migrated Nexus research records
credentials
```

这些内容通过 `.gitignore` 排除，或由外部系统管理。

## 工作流原则

1. **探索优先，但要可追踪**：每个有价值方向应有代码、配置和简短说明。
2. **成熟后拆分**：当某个方向形成稳定方法，应迁移为独立仓库，例如 DualRefGAD。
3. **避免污染主线**：临时输出、wandb、本地数据和手动备份不进入 Git。
4. **Nexus 负责整理**：实验过程和阶段性结论由 Nexus 维护到独立知识库。

## 当前整理状态

本仓库正在从历史混合状态中清理：

- `.gitignore` 已更新，排除生成物、大目录和迁移后的知识库。
- `DualRefGAD` 已作为独立项目维护。
- `Nexus` research records 已迁移到 openclawvm。
- VoxG 保留为 semi-supervised GAD incubator。
