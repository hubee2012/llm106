import torch
from torch import nn
from transformers import AutoTokenizer
import argpars
# ACT2FN 是 HuggingFace 提供的激活函数映射字典，例如 "silu" -> nn.SiLU()
from transformers.activations import ACT2FN
import LlmConfig
# 导入 PyTorch 函数式接口，提供 softmax、one_hot 等操作
import torch.nn.functional as F
# 导入 HuggingFace 的基类：PreTrainedModel（预训练模型基类）、GenerationMixin（生成能力混入）、PretrainedConfig（配置基类）
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig

# 显式设置当前模块所属的包名，确保相对导入和包路径解析正确
__package__ = "ch3"


class FeedForward(nn.Module):
    """
    标准的前馈网络（FFN），采用 SwiGLU 结构。
    计算流程：down_proj( act(gate_proj(x)) * up_proj(x) )
    这是 LLaMA 等现代 LLM 中常用的 FFN 变体。
    """

    def __init__(self, config: LlmConfig, intermediate_size: int = None):
        super().__init__()
        # 如果未显式传入 intermediate_size，则使用配置中的默认值
        intermediate_size = intermediate_size or config.intermediate_size
        # 门控投影：将 hidden_size 映射到 intermediate_size，用于生成激活门控信号
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        # 下投影：将 intermediate_size 映射回 hidden_size，作为最终输出
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        # 上投影：将 hidden_size 映射到 intermediate_size，提供被门控的主分支
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)

        # 根据配置中的激活函数名称（如 "silu"）获取对应的激活函数
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # SwiGLU 前向：
        # 1. gate_proj(x) 通过激活函数（如 SiLU）产生门控
        # 2. 与 up_proj(x) 逐元素相乘
        # 3. 再经过 down_proj 投影回原始维度
        # 在低维（hidden_size）里，很多特征纠缠在一起，难以用简单的激活函数切分。
        # 升到高维（intermediate_size）后，原本纠缠的特征在高维空间中更容易被「分开」，激活函数（ReLU / SiLU / GELU）才能有效地做非线性筛选。
        # 这类似于 核方法（kernel trick） 的思想：把数据映射到高维，线性操作就能表达更复杂的函数。
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
                            #         └────── 门控信号 ──────┘   └── 内容 ──┘

class MOEFeedForward(nn.Module):
    """
    混合专家（Mixture of Experts, MoE）前馈网络。
    包含多个并行的 FeedForward 专家，以及一个路由门控（gate）。
    每个 token 只被路由到 top-k 个专家，实现稀疏激活。
    """

    def __init__(self, config: LlmConfig):
        super().__init__()
        self.config = config
        # 路由门控：将 hidden_size 映射到 num_experts，输出每个专家的得分
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # 创建 num_experts 个独立的专家（每个都是标准 FeedForward），
        # 每个专家的隐藏层维度由 moe_intermediate_size 决定
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])

        # 保存激活函数（本类中未直接使用，但保留以备扩展）
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # x 的形状：(batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.shape

        # 将 (batch, seq) 维度展平，便于按 token 处理
        x_flat = x.view(-1, hidden_dim)  # (batch*seq, hidden_dim)

        # 计算每个 token 对所有专家的路由得分，并做 softmax 归一化
        scores = F.softmax(self.gate(x_flat), dim=-1)  # (N, num_experts)

        # 选取每个 token 得分最高的 top-k 个专家
        # topk_weight: 这些专家的权重；topk_idx: 这些专家的索引
        # sorted=False 表示不要求按权重排序，提升效率
        #topk_idx形状 (N, k)，N = batch*seq,
        topk_weight, topk_idx = torch.topk(
            scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False
        )

        # 如果配置要求对 top-k 权重做归一化，则重新归一化到和为 1
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # 初始化输出张量（与输入同形状），用于累加各专家的加权输出
        y = torch.zeros_like(x_flat)

        # 遍历每个专家，处理被路由到该专家的所有 token
        for i, expert in enumerate(self.experts):
            # mask 形状 (N, k)，N = batch*seq,标记哪些 token 的第几个 top-k 选择是该专家
            mask = (topk_idx == i)
            if mask.any():#只要有一个元素为 True，就返回 True，否则返回 False。
                # 找出至少有一个 top-k 选择指向该专家的 token 索引
                token_idx = mask.any(dim=-1).nonzero().flatten()

                # 取出对应的权重。注意：如果某个 token 的 top-k 中有多个位置指向同一专家（通常不会），
                # 这里 mask[token_idx] 会取出该 token 所有匹配的权重，并展平
                weight = topk_weight[mask].view(-1, 1)

                # 只对该专家计算对应 token 的输出，乘以权重后累加到 y 的对应位置
                # index_add_ 用于按 token_idx 在 0 维上累加
                y.index_add_(
                    0,
                    token_idx,
                    (expert(x_flat[token_idx]) * weight).to(y.dtype)
                )
            elif self.training:
                # 训练时，如果某个专家完全没有被任何 token 选中，
                # 通过加上 0 * 参数和 的方式让该专家的参数参与计算图，
                # 避免 DDP（分布式数据并行）因未使用参数而报错
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        # 计算路由辅助损失（auxiliary loss），用于鼓励专家负载均衡
        if self.training and self.config.router_aux_loss_coef > 0:
            # load: 每个专家被选中的平均频率，形状 (num_experts,)
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            # 辅助损失 = sum(load * 平均路由概率) * num_experts * 系数
            # 该损失鼓励负载分布均匀（避免某些专家被过度使用）
            # 核心思想: 用「专家被选中的硬频率 load」作为权重，去惩罚「专家被选中的软概率 scores」，迫使 gate 把 token 更均匀地分给各专家。
            self.aux_loss = (
                (load * scores.mean(0)).sum()
                * self.config.num_experts
                * self.config.router_aux_loss_coef
            )
        else:
            # 非训练阶段或系数为 0 时，辅助损失设为 0 标量
            self.aux_loss = scores.new_zeros(1).squeeze()

        # 将展平的输出恢复为 (batch_size, seq_len, hidden_dim)
        return y.view(batch_size, seq_len, hidden_dim)