# ==================== 导入必要的库 ====================
import torch
from torch import optim, nn


# ==================== 定义LoRA网络结构 ====================
class LoRA(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 模块

    LoRA通过在原始权重矩阵旁添加一个低秩分解的矩阵来微调模型，
    而不需要修改原始权重。这种方式大大减少了可训练参数的数量。

    原理: W' = W + ΔW, 其中 ΔW = B @ A
    A: (in_features, rank), B: (rank, out_features)
    """

    def __init__(self, in_features, out_features, rank):
        """
        初始化LoRA模块

        参数:
            in_features: 输入特征的维度
            out_features: 输出特征的维度
            rank: 低秩矩阵的秩（通常远小于in_features和out_features）
        """
        super().__init__()
        self.rank = rank  # 存储秩的大小

        # 低秩矩阵A: 将输入从高维映射到低维空间 (in_features -> rank)
        self.A = nn.Linear(in_features, rank, bias=False)

        # 低秩矩阵B: 将低维空间映射回高维空间 (rank -> out_features)
        self.B = nn.Linear(rank, out_features, bias=False)

        # 矩阵A使用高斯初始化 (均值为0，标准差0.02)
        # 这样使得初始的A具有较小的随机值
        self.A.weight.data.normal_(mean=0.0, std=0.02)

        # 矩阵B全0初始化，确保初始时 ΔW = 0
        # 即初始状态：W' = W + 0，模型行为不变
        self.B.weight.data.zero_()

    def forward(self, x):
        """
        前向传播：计算 ΔW @ x = B @ A @ x

        参数:
            x: 输入张量，形状为 [batch_size, in_features]

        返回:
            低秩更新后的输出，形状为 [batch_size, out_features]
        """
        # 先通过A降维，再通过B升维，等价于 ΔW = B @ A
        return self.B(self.A(x))


# ==================== 应用LoRA到模型 ====================
def apply_lora(model, rank=16):
    """
    将LoRA应用到模型的所有满足条件的线性层

    参数:
        model: 要应用LoRA的PyTorch模型
        rank: LoRA的秩，默认为16
    """
    # 遍历模型的所有模块（包括子模块）
    for name, module in model.named_modules():
        # 只对线性层应用LoRA，并且要求输入维度等于输出维度
        # 这样做的原因：通常只对Attention层的Q/K/V/O投影应用LoRA
        # 这些投影通常具有相同的输入输出维度 (hidden_size -> hidden_size)
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            # 创建LoRA模块，并移动到与原始模块相同的设备
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)

            # 将LoRA模块作为属性添加到原始模块中
            setattr(module, "lora", lora)

            # 保存原始的forward方法
            original_forward = module.forward

            # 定义新的forward方法：原始输出 + LoRA输出
            # 使用闭包捕获original_forward和lora，避免循环引用
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                """
                新的前向传播函数：原始线性变换 + LoRA低秩更新

                参数:
                    x: 输入张量
                    layer1: 原始的线性层forward函数
                    layer2: LoRA模块

                返回:
                    原始输出 + LoRA输出
                """
                # W' @ x = W @ x + B @ A @ x
                return layer1(x) + layer2(x)

            # 替换模块的forward方法
            module.forward = forward_with_lora


# ==================== 加载LoRA权重 ====================
def load_lora(model, path):
    """
    从文件加载LoRA权重到模型

    参数:
        model: 已应用LoRA的模型
        path: LoRA权重文件路径 (.pth)
    """
    # 加载状态字典，确保加载到正确的设备
    state_dict = torch.load(path, map_location=model.device)

    # 处理分布式训练中可能出现的"module."前缀
    # 如果键以"module."开头，则移除该前缀
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 如果模块有lora属性（即应用了LoRA）
        if hasattr(module, 'lora'):
            # 提取该模块对应的LoRA权重
            # 例如：如果name是"layer1.attention"，则只取以"layer1.attention.lora."开头的键
            lora_state = {
                k.replace(f'{name}.lora.', ''): v
                for k, v in state_dict.items()
                if f'{name}.lora.' in k
            }
            # 加载LoRA权重
            module.lora.load_state_dict(lora_state)


# ==================== 保存LoRA权重 ====================
def save_lora(model, path):
    """
    仅保存模型的LoRA权重到文件（不保存原始模型权重）

    参数:
        model: 已应用LoRA的模型
        path: 保存路径
    """
    # 如果模型被torch.compile包装，获取原始模型
    raw_model = getattr(model, '_orig_mod', model)

    state_dict = {}  # 用于存储LoRA权重

    # 遍历模型的所有模块
    for name, module in raw_model.named_modules():
        # 如果模块有lora属性
        if hasattr(module, 'lora'):
            # 处理分布式训练中的"module."前缀
            clean_name = name[7:] if name.startswith("module.") else name

            # 提取LoRA的权重（A和B矩阵）
            # 转换为CPU并半精度存储，节省空间
            lora_state = {
                f'{clean_name}.lora.{k}': v.cpu().half()
                for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)

    # 保存LoRA权重到文件
    torch.save(state_dict, path)


# ==================== 合并LoRA权重到原始模型 ====================
def merge_lora(model, lora_path, save_path):
    """
    将LoRA权重合并到原始模型权重，生成完整的模型权重文件

    参数:
        model: 已应用LoRA的模型
        lora_path: LoRA权重文件路径
        save_path: 合并后权重的保存路径
    """
    # 1. 加载LoRA权重到模型
    load_lora(model, lora_path)

    # 获取原始模型（处理torch.compile包装）
    raw_model = getattr(model, '_orig_mod', model)

    # 2. 准备状态字典，只包含非LoRA参数
    # 排除所有包含".lora."的键，只保留原始权重
    state_dict = {
        k: v.cpu().half()
        for k, v in raw_model.state_dict().items()
        if '.lora.' not in k
    }

    # 3. 遍历所有线性层，将LoRA权重合并到原始权重
    for name, module in raw_model.named_modules():
        # 只处理线性层，且不处理LoRA模块本身
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            # 克隆原始权重并转为半精度
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()

            # 如果该模块有LoRA，则合并
            if hasattr(module, 'lora'):
                # 计算LoRA的更新权重: ΔW = B @ A
                # 注意：B和A都是存储在当前模块的lora属性中
                lora_weight = (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()

                # 合并到原始权重: W' = W + ΔW
                state_dict[f'{name}.weight'] += lora_weight

    # 4. 保存合并后的完整权重
    torch.save(state_dict, save_path)