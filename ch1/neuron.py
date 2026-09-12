"""单神经元的前向传播与反向传播示例。

模型:
    z = w · x + b
    y_hat = sigmoid(z)
    L = 0.5 * (y_hat - y)^2

反向传播（链式法则）:
    dL/dy_hat = y_hat - y
    dL/dz     = dL/dy_hat * sigmoid'(z)
              = (y_hat - y) * y_hat * (1 - y_hat)
    dL/dw     = dL/dz * x
    dL/db     = dL/dz
"""

from __future__ import annotations
import matplotlib
matplotlib.use('Agg')  # 禁用GUI后端
import numpy as np
import matplotlib
import warnings
import os
import argparse
from datetime import datetime
import time
import sys
import swanlab

# 抑制matplotlib的字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')

# 强制使用交互式后端 - 在import pyplot之前设置
import matplotlib
# 尝试多个交互式后端
backends_to_try = ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'GTK3Agg', 'WXAgg']
backend_loaded = False

for backend in backends_to_try:
    try:
        matplotlib.use(backend)
        # 测试是否成功
        import matplotlib.pyplot as plt_test
        plt_test.figure()
        plt_test.close()
        print(f"使用交互式后端: {backend}")
        backend_loaded = True
        break
    except Exception as e:
        continue

if not backend_loaded:
    print("警告: 无法加载交互式后端，将使用默认后端")
    # 尝试使用默认后端
    try:
        matplotlib.use('TkAgg')
    except:
        pass

# 现在导入pyplot
import matplotlib.pyplot as plt

# 设置字体，使用系统默认字体避免警告
import matplotlib.font_manager as fm
# 获取系统字体列表
font_list = fm.findSystemFonts()
# 尝试找到可用的中文字体
chinese_fonts = []
for font in font_list:
    try:
        # 尝试加载字体
        prop = fm.FontProperties(fname=font)
        if prop.get_name() and ('Hei' in prop.get_name() or
                               'Song' in prop.get_name() or
                               'Sim' in prop.get_name() or
                               'Microsoft' in prop.get_name() or
                               'PingFang' in prop.get_name()):
            chinese_fonts.append(prop.get_name())
    except:
        pass

# 设置字体列表
if chinese_fonts:
    matplotlib.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans', 'Arial', 'Helvetica']
else:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置日志级别，减少不必要的输出
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid激活函数。
    参数:
        z: 输入数组，可以是标量或numpy数组
    返回:
        sigmoid(z) = 1/(1+e^(-z))，输出范围在(0,1)之间
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative_from_output(y_hat: np.ndarray) -> np.ndarray:
    """
    利用 σ'(z) = σ(z) * (1 - σ(z))，直接用输出计算导数。
    这是sigmoid函数的一个优美性质：其导数可以用函数值本身表示。
    参数:
        y_hat: sigmoid函数的输出值，范围在(0,1)之间
    返回:
        sigmoid函数的导数值
    """
    return y_hat * (1.0 - y_hat)


class Neuron:
    """
    单个神经元类，实现了前向传播、反向传播和参数更新。
    这是神经网络中最基本的单元，包含权重w、偏置b，
    以及前向传播和反向传播的计算逻辑。
    """
    def __init__(self, n_features: int, seed: int = 42) -> None:
        """
        初始化神经元。

        参数:
            n_features: 输入特征的数量
            seed: 随机数种子，用于保证结果的可重复性
        """
        # 创建随机数生成器，指定种子以确保可重复性
        rng = np.random.default_rng(seed)
        # 初始化权重：从正态分布中采样，均值为0，标准差为0.5
        # 大小为(n_features,)表示这是一个一维数组
        self.w = rng.normal(0.0, 0.5, size=(n_features,))
        # 初始化偏置为0
        self.b = 0.0
        # 前向缓存，供反向传播使用
        # 存储输入x，类型可以是None或numpy数组
        self.x: np.ndarray | None = None
        # 存储线性组合z = w·x + b
        self.z: float | None = None
        # 存储激活输出 y_hat = sigmoid(z)
        self.y_hat: float | None = None

    def forward(self, x: np.ndarray) -> float:
        """
        前向传播：计算神经元的输出。
        步骤:
        1. 保存输入x以备反向传播使用
        2. 计算线性组合 z = w·x + b
        3. 应用sigmoid激活函数得到输出 y_hat
        参数:
            x: 输入特征向量
        返回:
            神经元的输出值 y_hat (0到1之间的浮点数)
        """
        # 将输入转换为numpy数组并确保数据类型为float
        self.x = np.asarray(x, dtype=float)
        # 计算线性组合：权重与输入的点击加上偏置
        # float()确保结果是Python浮点数而不是numpy数组
        self.z = float(np.dot(self.w, self.x) + self.b)
        # 应用sigmoid激活函数，将结果转换为Python浮点数
        self.y_hat = float(sigmoid(np.array(self.z)))
        # 返回输出值
        return self.y_hat

    def loss(self, y: float) -> float:
        """
        计算均方误差损失。
        L = 0.5 * (y_hat - y)^2
        参数:
            y: 真实标签（目标值）
        返回:
            损失值
        注意:
            使用assert确保forward()已经被调用，self.y_hat不为None
        """
        # 计算并返回均方误差损失的一半（乘以0.5简化梯度计算）
        return 0.5 * (self.y_hat - y) ** 2

    def backward(self, y: float) -> tuple[np.ndarray, float]:
        """
        反向传播：根据当前前向结果计算梯度。
        使用链式法则计算损失函数对权重和偏置的梯度：
        dL/dw = dL/dy_hat * dy_hat/dz * dz/dw
        dL/db = dL/dy_hat * dy_hat/dz * dz/db
        参数:
            y: 真实标签（目标值）
        返回:
            (dL_dw, dL_db): 损失对权重和偏置的梯度
        """
        # 计算损失对输出的梯度：dL/dy_hat = y_hat - y
        dL_dy = self.y_hat - y
        # 计算损失对线性组合的梯度：dL/dz = dL/dy_hat * sigmoid'(z)
        # sigmoid'(z) = y_hat * (1 - y_hat)
        dL_dz = dL_dy * sigmoid_derivative_from_output(np.array(self.y_hat))
        # 计算损失对权重的梯度：dL/dw = dL/dz * x
        # 这里x是输入向量，结果是一个与w同形状的数组
        dL_dw = dL_dz * self.x
        # 计算损失对偏置的梯度：dL/db = dL/dz
        # 因为dz/db = 1
        dL_db = float(dL_dz)
        # 返回梯度
        return dL_dw, dL_db

    def step(self, dL_dw: np.ndarray, dL_db: float, lr: float = 0.1) -> None:
        """
        使用梯度下降更新参数。
        w_new = w - lr * dL/dw
        b_new = b - lr * dL/db
        参数:
            dL_dw: 损失对权重的梯度
            dL_db: 损失对偏置的梯度
            lr: 学习率，控制更新步长
        """
        # 沿负梯度方向更新权重
        self.w -= lr * dL_dw
        # 沿负梯度方向更新偏置
        self.b -= lr * dL_db



def plot_decision_boundary(ax, neuron, X, y, epoch, title_prefix=""):
    """
    绘制决策边界和数据点。
    这个函数可视化神经元在二维平面上学到的决策边界。
    参数:
        ax: matplotlib的Axes对象
        neuron: 训练好的神经元
        X: 输入数据 (N, 2)
        y: 标签 (N,)
        epoch: 当前epoch数，用于标题
        title_prefix: 标题前缀
    """
    # 创建网格：在数据范围内生成密集的点
    # 在x和y方向各扩展0.5的范围
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02  # 网格步长
    # 生成网格坐标矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 计算网格上每个点的预测值
    # 使用列表推导式遍历所有网格点
    Z = np.array([neuron.forward(np.array([xx[i, j], yy[i, j]]))
                  for i in range(xx.shape[0])
                  for j in range(xx.shape[1])])
    # 重塑为网格形状
    Z = Z.reshape(xx.shape)

    # 绘制决策边界：使用等高线填充
    # levels=[0, 0.5, 1]表示填充两个区域：<0.5和>0.5
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1],
                colors=['#FFE4B5', '#B0E0E6'], alpha=0.4)
    # 绘制决策边界线（概率=0.5的位置）
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5)

    # 绘制数据点：根据标签使用不同颜色
    # c=y使用标签值作为颜色，cmap='RdYlBu'是红-黄-蓝颜色映射
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                         edgecolors='k', s=80, alpha=0.8)

    # 设置轴标签和标题（使用英文，避免中文乱码）
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    if title_prefix:
        ax.set_title(f'{title_prefix}Epoch {epoch}')
    else:
        ax.set_title(f'Epoch {epoch}')
    # 添加网格
    ax.grid(True, alpha=0.3)
    return scatter


def demo_train_with_plot(args=None) -> None:
    """
    训练神经元并绘制训练过程的可视化，同时记录到SwanLab。
    包括：
    1. 损失曲线
    2. 准确率曲线
    3. 参数变化
    4. 初始决策边界
    5. 最终决策边界
    6. 决策边界的演变动画
    """
    # 解析参数
    if args is None:
        # 默认参数
        class DefaultArgs:
            def __init__(self):
                self.epochs = 20  # 改为20次迭代
                self.batch_size = 32
                self.learning_rate = 0.5
                self.swanlab_project = "neuron-training"
                self.swanlab_run_name = None
                self.swanlab_id = None
                self.resume = False
                self.seed = 42
                self.no_swanlab = True  # 默认禁用swanlab（因为需要登录）
                self.no_display = False  # 默认显示图形
        args = DefaultArgs()

    # 生成数据：标签大致由 x0 + x1 > 0 决定
    rng = np.random.default_rng(args.seed if hasattr(args, 'seed') else 0)
    X = rng.normal(size=(100, 2),loc=[0.4,0.5],scale=1)  # 100个样本，2个特征，标准正态分布
    #y = (X[:, 0] + X[:, 1] > 0).astype(float)  # 标签：x0+x1>0为1，否则为0
    # 原始线性可分标签,
    # 第一列X[:, 0]，第二列X[:, 1]
    #决策边界是直线：x₁ + x₂ = 0，即 x₂ = -x₁
    y_true = (X[:, 0] + X[:, 1] > 0.9).astype(float)

    # 加入小的随机波动（标签翻转噪声）
    noise_prob = 0.05  # 10% 的样本标签被翻转
    noise_mask = rng.random(size=100) < noise_prob#10% 的样本标签被标记
    y = y_true.copy()
    y[noise_mask] = 1 - y[noise_mask]  # 翻转标签


    # 创建神经元
    neuron = Neuron(n_features=2, seed=args.seed if hasattr(args, 'seed') else 1)
    lr = args.learning_rate
    epochs = args.epochs

    # 初始化SwanLab（默认禁用，避免需要登录）
    if not args.no_swanlab:
        try:
            # 生成运行名称
            if args.swanlab_run_name is None:
                # 如果没有指定run name，自动生成
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                swanlab_run_name = f"neuron-{epochs}-bsize-{args.batch_size}-lr-{args.learning_rate}-{timestamp}"
            else:
                swanlab_run_name = args.swanlab_run_name

            # 初始化SwanLab
            swanlab.init(
                project=args.swanlab_project,
                experiment_name=swanlab_run_name,
                config={
                    "epochs": epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": lr,
                    "n_samples": len(X),
                    "n_features": 2,
                    "seed": args.seed if hasattr(args, 'seed') else 0,
                    "model": "SingleNeuron",
                    "activation": "sigmoid",
                    "loss_function": "MSE"
                }
            )
            print(f"SwanLab已启用，项目: {args.swanlab_project}, 运行: {swanlab_run_name}")
        except Exception as e:
            print(f"警告: SwanLab初始化失败: {e}")
            print("将继续训练，但不会记录到SwanLab")
            args.no_swanlab = True
    else:
        print("SwanLab已禁用")

    # 用于记录训练过程中的指标
    history = {
        'epoch': [],  # epoch编号
        'loss': [],   # 平均损失
        'acc': [],    # 准确率
        'w': [],      # 权重向量
        'b': []       # 偏置
    }

    print("\n=== 训练过程 ===")
    print(f"Epochs: {epochs}, Batch Size: {args.batch_size}, Learning Rate: {lr}")
    print("=" * 60)

    # 训练循环
    for epoch in range(epochs + 1):
        total_loss = 0.0
        # 每个epoch重新洗牌数据，提高训练的随机性
        indices = np.random.permutation(len(X))
        # 遍历所有样本
        for idx in indices:
            xi = X[idx]
            yi = y[idx]
            # 前向传播
            neuron.forward(xi)
            # 累加损失
            total_loss += neuron.loss(yi)
            # 反向传播计算梯度
            dL_dw, dL_db = neuron.backward(yi)
            # 更新参数
            neuron.step(dL_dw, dL_db, lr=lr)

        # 计算平均损失
        avg_loss = total_loss / len(X)
        # 计算所有样本的预测值
        preds = np.array([neuron.forward(xi) for xi in X])
        # 计算准确率：预测>=0.5视为正类
        acc = np.mean((preds >= 0.5) == y)

        # 记录历史数据
        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        history['acc'].append(acc)
        history['w'].append(neuron.w.copy())
        history['b'].append(neuron.b)

        # 打印每个epoch的训练信息
        print(f"epoch {epoch:3d} | loss={avg_loss:.4f} | acc={acc:.3f} | w={neuron.w} | b={neuron.b:.4f}")

        # 记录到SwanLab（每个epoch都记录）
        if not args.no_swanlab:
            try:
                swanlab.log({
                    "loss": avg_loss,
                    "accuracy": acc,
                    "weight_0": neuron.w[0],
                    "weight_1": neuron.w[1],
                    "bias": neuron.b,
                }, step=epoch)
            except Exception as e:
                pass

    print("=" * 60)
    print("训练完成！")

    # 创建训练过程可视化图形
    print("\n正在生成可视化图表...")

    # 设置图形为交互模式（如果可能）
    try:
        plt.ion()  # 开启交互模式
    except:
        pass

    fig = plt.figure(figsize=(15, 10))

    # 子图1：损失曲线
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(history['epoch'], history['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    # 子图2：准确率曲线
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(history['epoch'], history['acc'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # 子图3：参数变化
    ax3 = plt.subplot(2, 3, 3)
    history_w = np.array(history['w'])
    ax3.plot(history['epoch'], history_w[:, 0], 'r-', label='w0', linewidth=2)
    ax3.plot(history['epoch'], history_w[:, 1], 'b-', label='w1', linewidth=2)
    ax3.plot(history['epoch'], history['b'], 'k-', label='b', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Weight Value')
    ax3.set_title('Parameter Changes')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 子图4：初始决策边界
    ax4 = plt.subplot(2, 3, 4)
    init_neuron = Neuron(n_features=2, seed=args.seed if hasattr(args, 'seed') else 1)
    plot_decision_boundary(ax4, init_neuron, X, y, 0, title_prefix="Initial ")

    # 子图5：最终决策边界
    ax5 = plt.subplot(2, 3, 5)
    plot_decision_boundary(ax5, neuron, X, y, epochs, title_prefix="Final ")

    # 子图6：训练过程中的决策边界动画
    ax6 = plt.subplot(2, 3, 6)
    # 选择所有epoch来展示决策边界的演变
    key_epochs = list(range(0, epochs + 1, max(1, epochs // 10)))  # 最多显示10个点
    if epochs not in key_epochs:
        key_epochs.append(epochs)
    key_epochs = sorted(set(key_epochs))

    print("正在播放决策边界演变动画...")
    # 遍历关键epoch，显示决策边界的演变
    for i, ep in enumerate(key_epochs):
        temp_neuron = Neuron(n_features=2, seed=args.seed if hasattr(args, 'seed') else 1)
        temp_neuron.w = history['w'][ep]
        temp_neuron.b = history['b'][ep]
        temp_neuron.forward(X[0])

        ax6.clear()
        plot_decision_boundary(ax6, temp_neuron, X, y, ep)
        ax6.set_title(f'Decision Boundary Evolution - Epoch {ep}')

        # 刷新图形
        try:
            fig.canvas.draw()
            fig.canvas.flush_events()
        except:
            pass

        # 放慢显示速度，每次暂停0.5秒
        plt.pause(0.5)

    # 保持最后一个状态
    plot_decision_boundary(ax6, neuron, X, y, epochs)
    ax6.set_title(f'Final Decision Boundary - Epoch {epochs}')

    plt.tight_layout()

    # 保存图片
    plt.savefig('training_process.png', dpi=300, bbox_inches='tight')
    print(f"图表已保存到: training_process.png")

    # 记录最终图表到SwanLab
    if not args.no_swanlab:
        try:
            swanlab.log({
                "training_summary": swanlab.Image(fig),
                "training_process_plot": swanlab.Image(fig)
            })
        except Exception as e:
            pass

    # 结束SwanLab运行
    if not args.no_swanlab:
        try:
            swanlab.finish()
        except Exception as e:
            pass


def main():
    """
    主函数：解析命令行参数并运行训练。
    """
    parser = argparse.ArgumentParser(description='单神经元训练示例')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数（默认30）')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='学习率')
    parser.add_argument('--swanlab_project', type=str, default='neuron-training', help='SwanLab项目名称')
    parser.add_argument('--swanlab_run_name', type=str, default=None, help='SwanLab运行名称')
    parser.add_argument('--swanlab_id', type=str, default=None, help='SwanLab运行ID（用于恢复）')
    parser.add_argument('--resume', action='store_true', help='是否恢复之前的运行')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no_swanlab', action='store_true', default=True, help='禁用SwanLab日志（默认启用）')
    parser.add_argument('--enable_swanlab', action='store_true', help='启用SwanLab日志')
    parser.add_argument('--no_display', action='store_true', help='不显示图形，只保存')

    args = parser.parse_args()

    # 如果启用swanlab，则尝试初始化
    if args.enable_swanlab:
        args.no_swanlab = False
        try:
            print("SwanLab已启用，将记录训练日志")
            print("访问 https://swanlab.cn 查看实验")
        except ImportError:
            print("警告: swanlab未安装，将使用普通训练模式")
            print("可以使用 'pip install swanlab' 安装")
            args.no_swanlab = True
    else:
        print("使用默认训练模式（SwanLab已禁用）")

    # 如果设置了不显示，强制使用Agg后端
    if args.no_display:
        try:
            matplotlib.use('Agg')
            print("图形显示已禁用，只保存文件")
        except:
            pass

    # 运行训练
    demo_train_with_plot(args)


if __name__ == "__main__":
    # 程序入口点

    # 取消注释下面的行可以运行单步演示
    # demo_one_step()

    # 使用命令行参数运行
    main()

    # 或者直接运行默认训练（不带命令行参数）
    # demo_train_with_plot()