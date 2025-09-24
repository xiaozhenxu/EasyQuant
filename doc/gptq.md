# GPTQ 量化算法

# 1. 算法背景

## 1.1 问题定义

线性层的计算方式可以简单定义为

$$
y=Wx
$$

- `W ∈ R^{d_out × d_in}`：权重矩阵（浮点数）
- `x ∈ R^{d_in}`：输入向量
- `y ∈ R^{d_out}`：输出向量

目标：将权重W量化为整数矩阵Q，并且最小化量化失真。

## 1.2 优化目标推导

量化误差：`𝚫W = W - dequantize(Q)`

输出误差：`𝚫y = 𝚫W · x = (W - dequantize(Q))x`

希望最小化输出误差的期望平方范数

$$
E\left [ \left \| y \right \| ^{2}  \right ] = E\left [ \left \| (W-dequantize(Q))x \right \| ^{2}  \right ] = E\left [ \left \| (W-dequantize(Q))x x^{t} (W-dequantize(Q))^{t}  \right \| ^{2}  \right ] 
$$

令 `H = E[xxᵀ]`（Hessian矩阵/协方差矩阵），则可以等价于最小化：

$$
min \left [ \left \| (W-dequantize(Q))H (W-dequantize(Q))^{t}  \right \| ^{2}  \right ] 
$$

Hessian矩阵 `H` 在这里充当**重要性权重矩阵**：

- **对角线元素** `H[i,i]`：表示第i个输入神经元的重要性
- **非对角线元素** `H[i,j]`：表示输入神经元i和j之间的相关性

# 2. 算法实现

## 2.1 add_batch函数 - Hessian 矩阵估计

该函数用于估计hessian矩阵的近似值，公式推导如下：

<aside>
💡

**数学原理** 对于线性linear层，权重W的Hessian矩阵可以近似为

</aside>

$$
H \approx E[x \cdot x^{t}  ]
$$

<aside>
💡

**具体实现** 采用批迭代的方案不断更新hessian矩阵，需要对输入数据做标准化处理，以及需要衰减旧的hessian矩阵。

</aside>

- 输入预处理

$$
x_{m} = \sqrt{2/(n+m)}x_{m}
$$

- 衰减旧信息并加入新信息

$$
H_{n} = x_{m} \cdot x_{m}^{t} + (n/(n+m))\cdot H_{n-1}
$$

                                            其中n是之前样本总数，m是当前批样本数量

## 2.2 quantize函数 - 最优量化

基于Hessian矩阵的逐层量化，核心是最小化量化误差带来的输出失真。

<aside>
💡

计算步骤

</aside>

- **step 1.  对hessian矩阵预处理，删除dead神经元以及增加阻尼系数**

```python
# 删去dead输入神经元
dead = torch.diag(H) == 0
H[dead, dead] = 1
W[:, dead] = 0

# 添加阻尼项防止数值不稳定
damp = self.percdamp * torch.mean(torch.diag(H))
H[diag, diag] += damp
```

$$
H^{'} = H + \lambda I
$$

- **step 2. (可选)按照hessian矩阵对角线参数(输入神经元重要性程度)，对hessian矩阵和weight进行排序**

```python
if actorder:
	  perm = torch.argsort(torch.diag(H), descending=True)
	  W = W[:, perm]
	  H = H[perm][:, perm]
	  invperm = torch.argsort(perm)
```

- **step 3. Cholesky分解求逆**

```python
H = torch.linalg.cholesky(H)
H = torch.cholesky_inverse(H)
Hinv = torch.linalg.cholesky(H, upper=True)
```

公式：

$$
Hinv = (H^{'} )^{-1} 
$$

- **step 4. 分块量化**

**对当前列的量化所产生的量化误差，通过对后续列的调整进行补偿，从而全局最小化输出失真**。

```python
for i in range(count):
    w = W1[:, i]           # 当前权重列
    d = Hinv1[i, i]        # H逆矩阵对角线元素
    
    q = quantizer.quantize(w)  # 量化当前列
    err = (w - q) / d      # 加权误差
    
    # 更新后续权重，补偿当前列的量化误差
    W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
```

**数学公式**

```python
w_j ← w_j - (w_i - q_i) · Hinv[i,j] / Hinv[i,i]  (对于j > i)
```