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

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative_from_output(y_hat: np.ndarray) -> np.ndarray:
    """利用 σ'(z) = σ(z) * (1 - σ(z))，直接用输出计算导数。"""
    return y_hat * (1.0 - y_hat)


class Neuron:
    def __init__(self, n_features: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.5, size=(n_features,))
        self.b = 0.0
        # 前向缓存，供反向使用
        self.x: np.ndarray | None = None
        self.z: float | None = None
        self.y_hat: float | None = None

    def forward(self, x: np.ndarray) -> float:
        self.x = np.asarray(x, dtype=float)
        self.z = float(np.dot(self.w, self.x) + self.b)
        self.y_hat = float(sigmoid(np.array(self.z)))
        return self.y_hat

    def loss(self, y: float) -> float:
        assert self.y_hat is not None
        return 0.5 * (self.y_hat - y) ** 2

    def backward(self, y: float) -> tuple[np.ndarray, float]:
        """根据当前前向结果，计算 dL/dw 与 dL/db。"""
        assert self.x is not None and self.y_hat is not None
        dL_dy = self.y_hat - y
        dL_dz = dL_dy * sigmoid_derivative_from_output(np.array(self.y_hat))
        dL_dw = dL_dz * self.x
        dL_db = float(dL_dz)
        return dL_dw, dL_db

    def step(self, dL_dw: np.ndarray, dL_db: float, lr: float = 0.1) -> None:
        self.w -= lr * dL_dw
        self.b -= lr * dL_db


def numerical_grad(
    neuron: Neuron, x: np.ndarray, y: float, eps: float = 1e-5
) -> tuple[np.ndarray, float]:
    """有限差分数值梯度，用来核对解析反向传播是否正确。"""
    w0 = neuron.w.copy()
    b0 = neuron.b

    dL_dw = np.zeros_like(w0)
    for i in range(len(w0)):
        neuron.w = w0.copy()
        neuron.w[i] = w0[i] + eps
        loss_pos = 0.5 * (neuron.forward(x) - y) ** 2
        neuron.w[i] = w0[i] - eps
        loss_neg = 0.5 * (neuron.forward(x) - y) ** 2
        dL_dw[i] = (loss_pos - loss_neg) / (2 * eps)

    neuron.w = w0.copy()
    neuron.b = b0 + eps
    loss_pos = 0.5 * (neuron.forward(x) - y) ** 2
    neuron.b = b0 - eps
    loss_neg = 0.5 * (neuron.forward(x) - y) ** 2
    dL_db = (loss_pos - loss_neg) / (2 * eps)

    neuron.w = w0
    neuron.b = b0
    neuron.forward(x)
    return dL_dw, float(dL_db)


# def demo_one_step() -> None:
#     x = np.array([1.0, -2.0, 0.5])
#     y = 1.0  # 期望输出
#
#     neuron = Neuron(n_features=3)
#     y_hat = neuron.forward(x)
#     loss = neuron.loss(y)
#     dL_dw, dL_db = neuron.backward(y)
#     num_dw, num_db = numerical_grad(neuron, x, y)
#
#     print("=== 单步前向 / 反向 ===")
#     print(f"x       = {x}")
#     print(f"w       = {neuron.w}")
#     print(f"b       = {neuron.b:.6f}")
#     print(f"z       = {neuron.z:.6f}")
#     print(f"y_hat   = {y_hat:.6f}")
#     print(f"y       = {y}")
#     print(f"loss    = {loss:.6f}")
#     print(f"dL/dw   = {dL_dw}")
#     print(f"dL/db   = {dL_db:.6f}")
#     print(f"num dw  = {num_dw}")
#     print(f"num db  = {num_db:.6f}")
#     print(f"dw 误差 = {np.max(np.abs(dL_dw - num_dw)):.2e}")
#     print(f"db 误差 = {abs(dL_db - num_db):.2e}")


def demo_train() -> None:
    """用一个神经元拟合简单线性可分数据（二分类）。"""
    # 标签大致由 x0 + x1 > 0 决定
    rng = np.random.default_rng(0)    #随机变量生成器
    X = rng.normal(size=(100, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)

    print('---X---')
    print(X)
    print('---y---')
    print(y)

    neuron = Neuron(n_features=2, seed=1)
    lr = 0.5
    epochs = 50

    print("\n=== 训练过程 ===")
    for epoch in range(epochs + 1):
        total_loss = 0.0
        for xi, yi in zip(X, y):
            neuron.forward(xi)
            total_loss += neuron.loss(yi)
            dL_dw, dL_db = neuron.backward(yi)
            neuron.step(dL_dw, dL_db, lr=lr)

        if epoch % 10 == 0:
            preds = np.array([neuron.forward(xi) for xi in X])
            acc = np.mean((preds >= 0.5) == y)
            print(
                f"epoch {epoch:3d} | loss={total_loss / len(X):.4f} | "
                f"acc={acc:.3f} | w={neuron.w} | b={neuron.b:.4f}"
            )


if __name__ == "__main__":
    # demo_one_step()
    demo_train()
