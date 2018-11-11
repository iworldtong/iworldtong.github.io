---
layout: post
title: "从矩阵微分的角度推导BP算法（附代码）"
subtitle: '矩阵微分在深度学习中的应用'
author: "iworld"
header-img: img/2018-11-09-matrix-differential-to-BP.jpg
mathjax: true
tags:
  - Deep Learning
---

## 1. 矩阵的微分

### 1.1 函数矩阵的微分形式

为简单起见，记 $\mathbf {e}_i$ 为第 $i$ 个分量为1，其余分量为0的单位列向量。

对于 $\mathbf{X} \in \mathbb{R}^{m\times n}$，有 $m\times n$ 元函数 $f(\mathbf{X})$，定义 $f(\mathbf{X})$ 对 $\mathbf{X}$ 的导数为


$$
\begin{eqnarray*}
\frac{df}{d\mathbf{X}} &=& \left( \frac{\partial f}{\partial \xi_{ij}} \right)_{m\times n} = \begin{bmatrix} \frac{\partial f}{\partial \xi_{11}} & \cdots & \frac{\partial f}{\partial \xi_{1n}} \\ \vdots &&\vdots \\ \frac{\partial f}{\partial \xi_{m1}}&\cdots&\frac{\partial f}{\partial \xi_{mn}} \end{bmatrix}\\
&=& \sum_{i=1}^m \sum_{j=1}^n \frac{\partial f}{\partial \xi_{ij}}\mathbf{e}_i\mathbf{e}_j^T
\end{eqnarray*}
$$


**注意：**上式中 $\mathbf{e}_i$为 $m$ 维列向量，$\mathbf{e}_j$为 $n$ 维列向量。

### 1.2 矩阵函数的微分形式

结合1.1，定义矩阵 $\frac{d\mathbf{F}}{d \mathbf{X}} = \begin{bmatrix} \frac{\partial \mathbf{F}}{\partial \xi_{11}} & \cdots & \frac{\partial \mathbf{F}}{\partial \xi_{1n}} \\ \vdots &&\vdots \\ \frac{\partial \mathbf{F}}{\partial \xi_{m1}}&\cdots&\frac{\partial \mathbf{F}}{\partial \xi_{mn}} \end{bmatrix}$，对 $\mathbf{X}$ 的导数为：


$$
\frac{d\mathbf{F}}{d \mathbf{X}} = \begin{bmatrix} \frac{\partial \mathbf{F}}{\partial \xi_{11}} & \cdots & \frac{\partial \mathbf{F}}{\partial \xi_{1n}} \\ \vdots &&\vdots \\ \frac{\partial \mathbf{F}}{\partial \xi_{m1}}&\cdots&\frac{\partial \mathbf{F}}{\partial \xi_{mn}} \end{bmatrix}_{mr\times ns} \quad \text{其中 } \frac{d\mathbf{F}}{d \xi_{ij}} = \begin{bmatrix} \frac{\partial f_{11}}{\partial \xi_{ij}} & \cdots & \frac{\partial f_{1s}}{\partial \xi_{ij}} \\ \vdots &&\vdots \\ \frac{\partial f_{r1}}{\partial \xi_{ij}}&\cdots&\frac{\partial f_{rs}}{\partial \xi_{ij}} \end{bmatrix}_{r\times s}
$$


### 1.3 一些重要性质

首先给出向量和矩阵求导的链式法则：

>* 若 $\mathbf{y}(\mathbf{x})$ 是 $\mathbf{x}$ 的向量值函数，则
>
>
>$$
>\frac{\partial f(\mathbf{y}(\mathbf{x}))}{\partial \mathbf{x}}=\frac{\partial \mathbf{y}^T(\mathbf{x})}{\partial \mathbf{x}} \frac{\partial f(\mathbf{y})}{\partial \mathbf{y}}
>$$
>
>
>式中 $\frac{\partial \mathbf{y}^T(\mathbf{x})}{\partial \mathbf{x}}$ 为 $n\times n$ 矩阵。 
>
>* 设 $\mathbf{A}$ 为 $m\times n$ 矩阵，且 $y=f(\mathbf{A})$ 和 $g(y)$ 分别是以矩阵 $\mathbf {A}$ 和标量 $y$ 为变元的实值函数，则
>
>$$
>\frac{\partial g(f(\mathbf{A}))}{\partial \mathbf{A}}=\frac{dg(y)}{dy}\frac{\partial f(\mathbf{A})}{\partial \mathbf{A}}
>$$
>

总结一下：


$$
\begin{eqnarray*}
f(\mathbf{t})=f(\mathbf{x(\mathbf{t})}) &\rightarrow& df=d\mathbf{x}^Tdf\\
\mathbf{f}(\mathbf{t})=\mathbf{f}(\mathbf{x(\mathbf{t})}) &\rightarrow& d\mathbf{f}=d\mathbf{f}d\mathbf{x}
\end{eqnarray*}
$$


下面介绍两个在推导 BP 算法时用到的性质：

> **性质1**    $\mathbf{y}=\mathbf{W}\mathbf{x}$，函数 $f(\mathbf{y})$ 是向量 $\mathbf{y}$ 的函数，其中 $\mathbf{W}\in C^{m\times n}$ 和 $\mathbf{x}\in C^n$ 无关,则有
>
>
> $$
> \frac{d\mathbf{y}^T}{d\mathbf{x}}=\mathbf{W}^T,\quad\frac{d(f(\mathbf{y}))}{d\mathbf{W}}=\frac{d(f(\mathbf{y}))}{d\mathbf{y}}\cdot \mathbf{x}^T
> $$
>

**证明：**观察到$\frac{d\mathbf{x}^T}{d\mathbf{x}}=\mathbf{I}$，则有


$$
\begin{eqnarray*}
\frac{d\mathbf{y}^T}{d\mathbf{x}}&=&\frac{d\mathbf{x}^T\mathbf{W}^T}{d\mathbf{x}}=\frac{d\mathbf{x}^T}{d\mathbf{x}}\mathbf{W}^T=\mathbf{W}^T \\
\frac{d(f(\mathbf{y}))}{d\mathbf{W}}&=&\frac{d(f(\mathbf{y}))}{d\mathbf{y}}\frac{d\mathbf{y}}{d\mathbf{W}} \\
&=& \frac{d(f(\mathbf{y}))}{d\mathbf{y}}\cdot \mathbf{x}^T
\end{eqnarray*}
$$




> **性质2**    设 $f(\mathbf{x})$ 是向量 $\mathbf{x}$ 的函数，而 $\mathbf{x}$ 又是 $\mathbf{u}$ 的函数，则有
>
>
> $$
> \frac{df}{d\mathbf{u}}=\frac{d\mathbf{x}^T}{d\mathbf{u}}\cdot\frac{df}{d\mathbf{x}}
> $$
>
>
> 根据1.2中定义，有 $\frac{d\mathbf{x}^T}{d\mathbf{u}}=\left(\frac{d\mathbf{x}^T}{du_1},\frac{d\mathbf{x}^T}{du_2},\ldots, \frac{d\mathbf{x}^T}{du_n}  \right)^T$，其中 $\frac{d\mathbf{x}^T}{du_i}=\left( \frac{dx_1}{du_i}, \frac{dx_2}{du_i},\ldots, \frac{dx_n}{du_i} \right)$。

**证明：**由雅可比矩阵的传递性可知


$$
\frac{\partial \mathbf{f}}{\partial \mathbf{u}}=\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\frac{\partial \mathbf{x}}{\partial \mathbf{u}}
$$


再根据 $f$ 退化成实数时雅克⽐矩阵和函数导数的关系，有


$$
\frac{\partial \mathbf{f}}{\partial \mathbf{x}}=\frac{\partial {f}}{\partial \mathbf{x}^T}, \quad\frac{\partial \mathbf{f}}{\partial \mathbf{u}}=\frac{\partial f}{\partial \mathbf{u}^T}
$$


将上面三式结合，可得到如下链式法则


$$
\begin{eqnarray*}
\frac{\partial f}{\partial \mathbf{u}^T} &=& \frac{\partial {f}}{\partial \mathbf{x}^T}\frac{\partial \mathbf{x}}{\partial \mathbf{u}}\\
&\downarrow& 等号两边同时转置 \\
\frac{df}{d\mathbf{u}} &=& \frac{d\mathbf{x}^T}{d\mathbf{u}}\cdot\frac{df}{d\mathbf{x}}
\end{eqnarray*}
$$




**推广：**类似的，若 $\mathbf{u}$ 是 $\mathbf{v}$ 的向量值函数，则有


$$
\frac{df}{d\mathbf{v}}=\frac{d\mathbf{u}^T}{d\mathbf{v}}\cdot\frac{d\mathbf{x}^T}{d\mathbf{u}}\cdot\frac{df}{d\mathbf{x}}
$$




## 2. BP算法推导的矩阵形式

考虑如下三层前馈神经网络的回归问题：

<div align="center"><img width="200px"  src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1541658478/iblog/matrix-differential-to-BP/matrix-differential-to-BP-1.png"></div>

* 输入层：输入 $\mathbf{x}$ 
* 隐含层：连接权重为 $\mathbf{W}_1$ ，激活函数为 $\varphi (\cdot)$ ，该层输入为 $\mathbf{z}_1=\mathbf{W}_1\mathbf{x}$，输出为 $\mathbf{h}=\varphi(\mathbf{z}_1)$ 
* 输出层：连接权重为 $\mathbf{W}_2$ ，该层输入为 $\mathbf{z}_2=\mathbf{W}_2\mathbf{h}$，输出为 $\mathbf{o}=\mathbf{z}_2$
* 损失函数： $e=\frac{1}{2}\|\|\mathbf{y}-\mathbf{o}\|\|_2$

我们的目标是求解 $\frac{de}{d\mathbf{W}_1},\frac{de}{d\mathbf{W}_2}$。求解过程如下：


$$
\begin{eqnarray*}
\frac{de}{d\mathbf{o}} &=& \mathbf{o}-\mathbf{y} \\
\frac{de}{d\mathbf{W}_2} &=& \frac{de}{d\mathbf{z_2}} \mathbf{h}^T \\
&=& \frac{d\mathbf{o}^T}{d\mathbf{z}_2} \frac{de}{d\mathbf{o}} \mathbf{h}^T \\
&=&  (\mathbf{o}-\mathbf{y})\mathbf{h}^T \\
\frac{de}{d\mathbf{W}_1} &=& \frac{de}{d\mathbf{z_1}} \mathbf{x}^T \\
&=& \frac{d\mathbf{h}^T}{d\mathbf{z}_1} \frac{d\mathbf{z}_2^T}{d\mathbf{h}} \frac{de}{d\mathbf{z}_2} \mathbf{x}^T \\
&=& \text{diag}(\varphi'(z_{1_i})) \mathbf{W_2}^T \frac{de}{d\mathbf{z}_2} \mathbf{x}^T \\
&=& \text{diag}(\varphi'(z_{1_i})) \mathbf{W_2}^T (\mathbf{o}-\mathbf{y}) \mathbf{x}^T
\end{eqnarray*}
$$


通过上面推导，我们可以得到逐层误差之间的关系。定义 $\delta_3=\mathbf{o}-\mathbf{y},\delta_2=\frac{de}{d\mathbf{z}_2},\delta_1=\frac{de}{d\mathbf{z}_1}$，则：


$$
\begin{eqnarray*}
\delta_2 &=& \delta_3 \\
\delta_1 &=& \text{diag}(\varphi'(z_{1_i}))\mathbf{W}_2^T\delta_2
\end{eqnarray*}
$$


得到梯度后，可以利用随机梯度下降（Stochastic gradient descent）更新模型：

1. init $W_1, W_2$
2. for j=1 to $T$ <br>
   ​    　for each sample $( \textbf{x}_i, y_i)$<br>
   ​    　　calc   　$\frac{de}{d\mathbf{W}_1},\frac{de}{d\mathbf{W}_2}$<br>
   ​    　　update　$\mathbf{W}_1 \leftarrow \mathbf{W}_1 - \gamma\frac{de}{d\mathbf{W}_1},\mathbf{W}_2 \leftarrow \mathbf{W}_2 - \gamma\frac{de}{d\mathbf{W}_2}$<br>
   ​    　stop until convergence

## 3. Numpy实现

基于 MNIST 实现手写数字识别。

设置三层全连接网络，输入层神经元数量为784，隐含层神经元数量为500，输出层神经元数量为10。

注意隐含层激活函数为 ReLU，输出层激活函数为 Softmax ，重点关注下计算梯度的代码：

```python
def backward(probs, labels, x, h1, h2):
    n = probs.shape[1]
    e2 = probs - labels
    e1 = np.dot(w2.T, e2)
    e1[h1 <= 0.0] = 0.0
    dw2[:] = np.dot(e2, h1.T) / n
    db2[:] = np.mean(e2, axis=1)[:,np.newaxis]
    dw1[:] = np.dot(e1, x.T) / n
    db1[:] = np.mean(e1, axis=1)[:,np.newaxis]
```

可以看到`e2`对应 $\delta_2$，`e1`对应 $\delta_1$ 。

[完整代码](https://github.com/iworldtong/ML-and-DL-notes/blob/master/DL/mlp/mnist_mlp_np.py)

## 4. 参考资料

* 《矩阵分析与应用》张贤达
* [机器学习中的矩阵/向量求导](https://zhuanlan.zhihu.com/p/25063314) —— 对矩阵求导的介绍很完整











