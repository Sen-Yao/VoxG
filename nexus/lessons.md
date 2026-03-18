# Lessons Learned

> 记录 Nexus 在科研过程中学到的知识和经验。

---

## 技术发现

### 2026-03-18: VoxG 项目架构理解

1. **核心架构**: GGADFormer = Token Projection + Graph Transformer × 3 + Token Decoder
2. **关键创新**: nagphormer_tokenization() 将图结构编码为 token 序列，支持 mini-batch 训练
3. **损失函数**: BCE + Reconstruction + Ring + Contrastive，权重可调
4. **数据规模**: 支持 8 个数据集，最大 DGraph 有 370 万节点
5. **标注效率**: 仅需 5% 标注率即可达到 SOTA

### 模型参数

- embedding_dim: 256
- GT_num_layers: 3
- GT_num_heads: 2
- GT_dropout: 0.4
- pp_k: 6 (PPR 步数)

---

## 流程优化

### 2026-03-18: 代理工作流程

1. 每次循环先读取 USER_FEEDBACK.md 和 AGENT_STATE.yaml
2. 小步快跑，每次只执行一个步骤
3. 及时更新状态文件和记录发现
4. 遇到问题最多重试 3 次，然后请求用户指导

---

## 失败案例

_暂无_

---

_最后更新：2026-03-18_
