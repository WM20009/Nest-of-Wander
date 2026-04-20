---
title: Diffusion
date: "2025-10-14"
author: Wander
authorImage: /images/avatar.jpg
category: design
tags:
- Diffusion
description: 学习Diffusion时存一下checkpoints，便于复习。
---

# DDPM（likelihood-based）
参考- [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)
**这里省略了大量的数学推导，仅保留必要的数学表达式，以提高可读性**

书接上回，我们可以认为variable diffusion model就是具有如下三条假设的HVAE:
-  潜在维度恰好等于数据维度
- 每个时间步的潜在编码器结构不是学成的；它是作为线性高斯模型预先定义的。换句话说，它是一个
以前一时间步输出为中心的高斯分布。
- 潜在编码器的高斯参数随时间变化，使得最终时间步T 的潜在分布为标准正态分布
## 编码器
**编码器**显式地建模为:
$$q\left(\boldsymbol{x}_{t} \mid \boldsymbol{x}_{t-1}\right)=\mathcal{N}\left(\boldsymbol{x}_{t} ; \sqrt{\alpha_{t}} \boldsymbol{x}_{t-1},\left(1-\alpha_{t}\right) \mathbf{I}\right)$$
这称为线性高斯模型。

编码器这样规定的目的是：当我们任意给定一个干净图像$x_0$,我们总可以通过设置噪声调度并且在足够长的时间步T（这也是其推理很慢的原因）后使得$x_T$服从标准正态，进而方便我们进行采样。

并且，它有一些十分良好的性质，当我们给定了$\alpha_1,\alpha_2,\dots,\alpha_T$之后，$q(\boldsymbol{x}_t \mid \boldsymbol{x}_0)$和$q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0)$可以直接解析得到，且都是正态分布。

其中，$q(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\sim \mathcal{N}(\sqrt{\bar{\alpha_t}}\boldsymbol{x_0},(1-\bar{\alpha_t})\mathbf{I}),\bar{\alpha_t}=\prod_{i=1}^t\alpha_i$
另一个在下文给出。
## ELBO
我们通过最大似然估计来优化模型参数，具体来说是通过优化似然函数的下界（ELBO）来间接优化似然函数。

表达式为：
$$
\log{p(x_0)}=\int \log{p(x_{0:T})} dx_{1:T}
$$

这表示了任意干净图像$x_0$的似然,对所有从正态分布到$x_0$的轨迹积分。

ELBO为:

$$
\underbrace{\mathbb{E}_{q\left(\boldsymbol{x}_{1} \mid \boldsymbol{x}_{0}\right)}\left[\log p_{\theta}\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{1}\right)\right]}_{\text {reconstruction term }}-\underbrace{D_{\mathrm{KL}}\left(q\left(\boldsymbol{x}_{T} \mid \boldsymbol{x}_{0}\right) \| p\left(\boldsymbol{x}_{T}\right)\right)}_{\text {prior matching term }}-\sum_{t=2}^{T} \underbrace{\mathbb{E}_{q\left(\boldsymbol{x}_{t} \mid \boldsymbol{x}_{0}\right)}\left[D_{\mathrm{KL}}\left(q\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{t}, \boldsymbol{x}_{0}\right) \| p_{\theta}\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{t}\right)\right)\right]}_{\text {denoising matching term }}
$$

1. $\mathbb{E}_{q(\boldsymbol{x}_{1}|\boldsymbol{x}_{0})}\left[\log p_{\theta}(\boldsymbol{x}_{0}|\boldsymbol{x}_{1})\right]$ 可以解释为一个重构项；类似于普通VAE的ELBO中的对应项，这一项可以使用蒙特卡罗估计进行近似和优化。

2. $D_{\mathrm{KL}}(q(\boldsymbol{x}_{T}|\boldsymbol{x}_{0})\parallel p(\boldsymbol{x}_{T}))$ 表示最终加噪输入的分布与标准高斯先验的接近程度。它没有可训练参数，并且在我们的假设下也等于零。

3. $\mathbb{E}_{q(\boldsymbol{x}_{t}|\boldsymbol{x}_{0})}\left[D_{\mathrm{KL}}(q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t},\boldsymbol{x}_{0})\parallel p_{\theta}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}))\right]$ 是一个降噪匹配项。我们学习期望的降噪转移步骤 $p_{\theta}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t})$ 作为对易处理的、真实降噪转移步骤 $q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t},\boldsymbol{x}_{0})$ 的近似。$q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t},\boldsymbol{x}_{0})$ 转移步骤可以作为真实信号，因为它定义了如何在知道最终完全降噪图像 $\boldsymbol{x}_{0}$ 应该是什么

其中，第一项的计算和VAE中的差不多，第二项是没有训练参数且T足够大时为0，计算量主要是第三项。根据线性高斯模型的性质可知，$q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t,\boldsymbol{x}_0)$有解析式，且解析式为：

$$
q\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0\right) = \mathcal{N}\left(\boldsymbol{x}_{t-1}; \underbrace{\frac{\sqrt{\alpha_t}\left(1-\bar{\alpha}_{t-1}\right)\boldsymbol{x}_t + \sqrt{\bar{\alpha}_{t-1}}\left(1-\alpha_t\right)\boldsymbol{x}_0}{1-\bar{\alpha}_t}}_{\mu_q(\boldsymbol{x}_t, \boldsymbol{x}_0)}, \frac{\left(1-\alpha_t\right)\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_t}\mathbf{I}\right)
$$

由于协方差矩阵是常数且我们想要$p_\theta (\boldsymbol{x}_{t-1}\mid \boldsymbol{x}_t)$与其尽可能接近，于是我们也把$p_\theta(\boldsymbol{x}_{t-1}\mid \boldsymbol{x}_t)$建模为正态分布,且协方差矩阵与其相同。

再代入正态分布KL散度的公式，最后的优化目标为：

$$
||\mu_\theta-\mu_q||_2^2
$$

即两者均值的L2距离的平方。
由于$\mu_\theta$为$\boldsymbol{x}_t$和$t$的函数，而并不是$x_0$的函数（这也就是为什么我们不可能直接解析地得到$p_\theta$的原因），为了与$\mu_q$尽量接近，我们将其建模为

$$ 
\mu_{\theta}\left(\boldsymbol{x}_{t}, t\right)=\frac{\sqrt{\alpha_{t}}\left(1-\bar{\alpha}_{t-1}\right)\boldsymbol{x}_{t}+\sqrt{\bar{\alpha}_{t-1}}\left(1-\alpha_{t}\right)\hat{\boldsymbol{x}}_{\theta}\left(\boldsymbol{x}_{t}, t\right)}{1-\bar{\alpha}_{t}} 
$$

于是最后的的优化目标变成了

$$
\left\|\hat{\boldsymbol{x}}_{\theta}\left(\boldsymbol{x}_{t}, t\right)-\boldsymbol{x}_{0}\right\|_{2}^{2}
$$

因此，优化一个VDM归结为学习一个神经网络，从任意噪声化的版本中预测原始真实值图像。（为什么这个结论看起来是如此平凡🤣）此外，通过在所有噪声级别上最小化我们推导出的ELBO目标的第三项可以近似为在所有时间步上最小化期望（这是蒙特卡洛采样积分法，不得不说要不是有这种采样法，推导的这么一长串东西都完全没法算啊）：

$$
\underset{\boldsymbol{\theta}}{\arg\min}\,\mathbb{E}_{t \sim U\{2,T\}}\left[\mathbb{E}_{q\left(\boldsymbol{x}_t \mid \boldsymbol{x}_0\right)}\left[D_{\mathrm{KL}}\left(q\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0\right) \parallel p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t\right)\right)\right]\right]
$$

然后可以使用随机样本在时间步上进行优化。

## 另外两种目标函数

而事实上这个目标函数还有另外两种等效的形式

### 第二种（$\epsilon$-prediction）
首先，我们可以利用重参数化技巧。在推导 $q(\boldsymbol{x}_{t}|\boldsymbol{x}_{0})$ 的形式时，我们可以重新排列方程来证明：

$$
\boldsymbol{x}_{0} = \frac{\boldsymbol{x}_{t} - \sqrt{1-\bar{\alpha}_{t}}\boldsymbol{\epsilon}_{0}}{\sqrt{\bar{\alpha}_{t}}}
$$
于是$\mu_q(\boldsymbol{x}_t,\boldsymbol{x}_0)$可重新推导为：
$$
\mu_q(\boldsymbol{x}_t,\boldsymbol{x}_0)=\frac{1}{\sqrt{\alpha_{t}}}\boldsymbol{x}_{t} - \frac{1-\alpha_{t}}{\sqrt{1-\bar{\alpha}_{t}}\sqrt{\alpha_{t}}}\boldsymbol{\epsilon}_{0}
$$

因此，我们可以将我们的近似降噪转移均值 $\mu_{\theta}(\boldsymbol{x}_{t},t)$ 设置为：

$$
\mu_{\theta}(\boldsymbol{x}_{t},t) = \frac{1}{\sqrt{\alpha_{t}}}\boldsymbol{x}_{t} - \frac{1-\alpha_{t}}{\sqrt{1-\bar{\alpha}_{t}}\sqrt{\alpha_{t}}}\hat{\boldsymbol{\epsilon}}_{\theta}(\boldsymbol{x}_{t},t)
$$
并且相应的最优化问题变为：
$$
\underset{\boldsymbol{\theta}}{\arg\min}\,\frac{1}{2\sigma_{q}^{2}(t)}\frac{\left(1-\alpha_{t}\right)^{2}}{\left(1-\bar{\alpha}_{t}\right)\alpha_{t}}\left[\left\|\boldsymbol{\epsilon}_{0}-\hat{\boldsymbol{\epsilon}}_{\theta}\left(\boldsymbol{x}_{t}, t\right)\right\|_{2}^{2}\right]
$$
在这里，$\hat{\bm{\epsilon}}_{\theta}(\bm{x}_{t}, t)$是一个神经网络，它学习预测决定$\bm{x}_{t}$的源噪声$\bm{\epsilon}_{0} \sim \mathcal{N}(\bm{\epsilon}; \mathbf{0}, \mathbf{I})$，从$\bm{x}_{0}$。因此，我们已经证明，通过预测原始图像$\bm{x}_{0}$来学习VDM等同于学习预测噪声；然而，实证研究表明，预测噪声取得了更好的性能。

这里也可以这样理解，任意时间步的图像都可以由原始图像一步加噪得到，我们只要用网络去预测加的那个噪声也可以达到同样的效果。并且也许预测噪声相当于resnet预测残差的想法，可能更容易拟合（？）故而效果更好。

而这实际上就是DDPM的做法。
![ddpm](/images/blog/diffusion/image.png)
### 第三种(v-prediction)

v-prediction 通常在**方差保持 (Variance Preserving, VP)** 的设定下最为直观，即假定前向加噪过程满足 $\alpha_t^2 + \sigma_t^2 = 1$。
既然两个数的平方和为 1，我们在数学上自然可以用三角函数来参数化它们。我们定义一个“噪声角” $\phi_t \in [0, \pi/2]$：
*   **信号系数**: $\alpha_t = \cos\phi_t$
*   **噪声系数**: $\sigma_t = \sin\phi_t$

当 $t=0$ 时，$\phi_t \approx 0$（纯数据）；当 $t \to T$ 时，$\phi_t \approx \pi/2$（纯噪声）。
此时，前向加噪公式可以写成两个正交基（数据 $x$ 和 独立噪声 $\epsilon$）的线性组合：
$$z_t = \cos\phi_t \cdot x + \sin\phi_t \cdot \epsilon$$

**定义 v (Velocity, 速度向量)**

既然 $z_t$ 是随着角度 $\phi_t$ 变化在圆弧上移动的点，它的**运动速度 (Velocity)** 也就是对角度 $\phi_t$ 求导：
$$v_t = \frac{\partial z_t}{\partial \phi_t} = \frac{\partial}{\partial \phi_t}(\cos\phi_t \cdot x + \sin\phi_t \cdot \epsilon)$$
$$v_t = -\sin\phi_t \cdot x + \cos\phi_t \cdot \epsilon$$

替换回 $\alpha_t$ 和 $\sigma_t$，这就得到了著名的 v-prediction 目标公式：
$$v_t = \alpha_t \epsilon - \sigma_t x$$
*(注意：$z_t$ 和 $v_t$ 在几何上永远是正交的，即 $z_t \cdot v_t = 0$)*

写成矩阵的形式：
$$ \begin{bmatrix} z_t \\ v_t \end{bmatrix} = \begin{bmatrix} \cos\phi_t & \sin\phi_t \\ -\sin\phi_t & \cos\phi_t \end{bmatrix} \begin{bmatrix} x \\ \epsilon \end{bmatrix} $$

你会发现，中间那个矩阵**就是一个标准的二维旋转矩阵 $R(\phi_t)$**！它代表把 $(x, \epsilon)$ 坐标系旋转了 $-\phi_t$ 角度。

**而正交旋转矩阵的逆，就是它的转置。**
所以，想要反解 $x$ 和 $\epsilon$，根本不需要像图片里那样做恶心的代入消元，直接两边左乘转置矩阵即可：

$$ \begin{bmatrix} x \\ \epsilon \end{bmatrix} = \begin{bmatrix} \cos\phi_t & -\sin\phi_t \\ \sin\phi_t & \cos\phi_t \end{bmatrix} \begin{bmatrix} z_t \\ v_t \end{bmatrix} $$

直接展开，得到反解结果（对应图片中复杂的推导结论）：
$$x = \cos\phi_t \cdot z_t - \sin\phi_t \cdot v_t$$
$$\epsilon = \sin\phi_t \cdot z_t + \cos\phi_t \cdot v_t$$

接下来我们推导$q(z_s|z_t,x),0<s<t<T$

我们将刚才用神经网络预测出的 $\hat{v}_t$ 得到的反解 $\hat{x}$ 和 $\hat{\epsilon}$ 代入：
$$z_s = \cos\phi_s (\cos\phi_t \cdot z_t - \sin\phi_t \cdot \hat{v}_t) + \sin\phi_s (\sin\phi_t \cdot z_t + \cos\phi_t \cdot \hat{v}_t)$$

接下来，把 $z_t$ 和 $\hat{v}_t$ 的项分别合并（这正是初中数学的三角函数展开逆运算）：
$$z_s = (\cos\phi_s \cos\phi_t + \sin\phi_s \sin\phi_t) \cdot z_t + (\sin\phi_s \cos\phi_t - \cos\phi_s \sin\phi_t) \cdot \hat{v}_t$$

利用两角差的余弦和正弦公式，瞬间得到最终极致简约的结果：
$$z_s = \cos(\phi_s - \phi_t) \cdot z_t + \sin(\phi_s - \phi_t) \cdot \hat{v}_t$$

于是我们的预测目标就变成了v.

通过上面的推导，你会发现 v-prediction 将扩散模型从“一维的加减噪声”变成了一个“**在超球面上旋转**”的问题（也就是图1那个四分之一圆弧的意思）。

*   **前向加噪**：就是在 $x$ 和 $\epsilon$ 构成的平面上，将初始向量 $x$ 以速度 $v$ 向 $\epsilon$ 的方向**旋转** $\phi_t$ 的角度，得到 $z_t$。
*   **网络预测**：模型不直接猜起点 $x$ 或终点 $\epsilon$，而是猜当前点 $z_t$ 所在位置的**切线运动方向（速度 $v_t$）**。


# score-based diffusion
[Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/)
这是作者本人的博客，写得清晰易懂，并不需要有任何SDE基础，只要有基础概率论基础和懂一点点随机过程的概念即可。

事实上整个diffusion理论都被统一在了该SDE的框架之内，精妙绝伦。

## 条件生成
参考[这个视频](https://www.youtube.com/watch?v=iv-5mZ_9CPY)，现在主流的方法是classifier-free guidance，大概的想法就是：
在推理（生成）时，我们不需要任何外部分类器。对于同一个输入噪声，我们让这个统一的模型同时进行两次预测：
- 无条件预测：$ε_{uncond} = model(x_t, ∅)$
- 条件预测：$ε_{cond} = model(x_t, y)$
  
然后，我们计算两者的方向差，并将这个差值放大：
$ε_{final} = ε_{uncond} + s * (ε_{cond} - ε_{uncond})$

![training](/images/blog/diffusion/image2.png)
![sampling](/images/blog/diffusion/image3.png)
# DDIM
[参考苏神博客](https://kexue.fm/archives/9181)
DDIM是DDPM在的推广，DDPM通过定义$p(x_t|x_{t-1})$和引入马尔科夫性完整定义了前向过程的联合分布。但是我们观察到其实并不一定需要这样。因为如果我们承认拟合p(x_{t-1}|x_t,x_0)的训练目标是正确的（直接承认它是正确的，而不要从最大似然估计中推导ELBO），那么根据贝叶斯公式$p(x_{t-1}|x_t,x_0)=\frac{p(x_{t}|x_{t-1},x_0)p(x_{t-1}|x_0)}{p(x_t|x_0)}$其实我们只需要知道$p(x_t|x_0),t=1,2,\dots,T$以及$p(x_{t-1}|x_t,x_0)=p(x_{t-1}|x_t)$即可，那么我们依然保持DDPM对边缘分布的定义，对于状态转移，我们改变因果顺序，通过先定义满足边缘分布的$p(x_{t-1}|x_t,x_0)$再把它定义出来，这样就实现了前向过程的推广。
当我们做出上述改动之后，$q(x_{t-1}|x_t,x_0)$不再是原来的表达式，我们可以通过推导得到其**可以**具备以下形式：

$$
q(x_{t-1} \mid x_t, x_0) \sim \mathcal{N} \left(
x_{t-1};\ 
\frac{\sqrt{\beta_{t-1}^2 - \sigma_t^2}}{\bar{\beta}_t} x_t + 
\left( \bar{\alpha}_{t-1} - \bar{\alpha}_t \cdot \frac{\sqrt{\beta_{t-1}^2 - \sigma_t^2}}{\bar{\beta}_t} \right) x_0,\ 
\sigma_t^2 \mathbf{I}
\right)
$$
其中$\sigma_t$可以自己设置，取某个值的时候会退化为DDPM，取0时,会变成确定性采样

进行上述改动后，事实上DDIM的训练算法和DDPM完全相同，区别只在采样。

而DDIM对于采样的加速取决于另一个观察：**DDPM的训练结果实质上包含了它的任意子序列参数的训练结果。** 因为DDPM训练的就是从任意时间步的噪声向量中预测原图，任意子序列的训练目标显然包含其中。

所以实际上DDPM也可以进行跳步采样，而DDIM在此基础上更进一步，通过改变$\sigma_t$能得到更好的采样效果。

根据实验结果，应该是$\sigma_t$越小。效果越好。

下面给出$\sigma_t$取某一个值的时候的采样算法。
### 1. 选择时间步
- 从集合 {1, ..., T} 中选择一个包含 N 个时间步的子序列  
  

$$
  S = \{s_1, s_2, ..., s_N\}, \quad s_1 = 1, \; s_N = T, \; s_i < s_{i+1}
$$

  
- 记作  
  

$$
S = \{t_0, t_1, ..., t_N\}, \quad t_N = T, \; t_0 = 0
$$


### 2. 获取噪声预测模型
训练一个模型 $\epsilon_\theta(x_t, t)$ 来预测噪声 $\epsilon$，满足：


$$
x_t = \sqrt{\alpha_t} \, x_0 + \sqrt{1 - \alpha_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

### 3. 采样过程
- 初始化：  
$$
  x_T \sim \mathcal{N}(0, I)
$$

- 对于 $i = N, ..., 1$：  
  1. **预测噪声**  
    $$
     \epsilon = \epsilon_\theta(x_{t_i}, t_i)
    $$

  
  2. **估计 $x_0$**  
    $$
     \hat{x}_0 = \frac{x_{t_i} - \sqrt{1 - \alpha_{t_i}} \, \epsilon_\theta(x_{t_i}, t_i)}{\sqrt{\alpha_{t_i}}}
    $$

  
  3. **计算 $\sigma_t$ 和均值 $\mu_t$**  
   （注意这里的$\alpha_t$并不是训练时所指定的那些）
    $$
    \sigma_t = \eta \cdot \sqrt{\frac{1 - \alpha_{t_{i-1}}}{1 - \alpha_{t_i}}} \cdot \sqrt{1 - \frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}}
    $$
    $$
     \mu_t = \sqrt{\alpha_{t_{i-1}}} \, \hat{x}_0 + \sqrt{1 - \alpha_{t_{i-1}} - \sigma_t^2} \, \epsilon_\theta(x_{t_i}, t_i)
    $$

  
  4. **更新采样**  
    $$
    x_{t_{i-1}} \sim \mathcal{N}(\mu_t, \sigma_t^2)
    $$
  （当设定 $\eta = 0$ 时，$\sigma_t$ = 0）

### 4. 输出
最终输出 $x_0$，即为生成的样本。
## 说明
- $\alpha_t$：根据预定义的时间步 $t$ 计算的超参数，通常是线性或余弦调度函数。  
- $\epsilon_\theta$：训练好的噪声预测模型。  
- $\hat{x}_0$：对原始样本的估计值。  
- $\sigma_t$：控制采样过程中的随机性。  
- $\mu_t$：生成下一个时间步的均值。

# DiT
用transformer架构替代了Unet，作为diffusion使用的架构。
![DiT](/images/blog/diffusion/image1.png)
这张图已然说明一切.


# Denoising Score Matching

## 1. 核心问题背景

在扩散模型框架下，分布对齐等价于在各个噪声水平 $t$ 下对齐 Score：

$$
\nabla_{x_t} \log p_{\text{fake}}(x_t, t) \longleftrightarrow \nabla_{x_t} \log p_{\text{real}}(x_t, t)
$$

然而，边缘分布 $p(x_t)$ 是通过对整个数据集进行积分得到的：

$$
p(x_t) = \int p(x_t|x_0)p(x_0) dx_0
$$

直接对该积分求对数梯度在数学上是不可解的（Intractable）。为此,引入了 **Denoising Score Matching (DSM)** 的等价性结论。

---

## 2. 必备数学工具

在推导开始前，我们需要掌握两个核心数学技巧：

### 2.1 莱布尼茨积分规则 (Leibniz Rule)
在满足平滑性条件（如高斯分布）时，梯度算子可以与积分符号交换：

$$
\nabla_{x_t} \int p(x_t | x_0) p(x_0) dx_0 = \int \nabla_{x_t} p(x_t | x_0) p(x_0) dx_0
$$

### 2.2 对数导数技巧 (Log-Derivative Trick)
根据复合函数链式法则，对于任何正值函数 $f(x)$：

$$
\nabla \log f(x) = \frac{\nabla f(x)}{f(x)} \implies \nabla f(x) = f(x) \nabla \log f(x)
$$

该技巧常用于将梯度的积分转化为**期望**的形式。

---

## 3. 关键引理：边缘分数是条件分数的期望

首先，我们要证明边缘分数本质上是所有可能原图 $x_0$ 对应的条件分数的加权平均。

**证明：**
根据边缘分布定义，对其求梯度：

$$
\nabla_{x_t} p(x_t) = \int \nabla_{x_t} p(x_t|x_0) p(x_0) dx_0
$$

应用**对数导数技巧**：

$$
\nabla_{x_t} p(x_t) = \int [p(x_t|x_0) \nabla_{x_t} \log p(x_t|x_0)] p(x_0) dx_0
$$

两边同时除以 $p(x_t)$：

$$
\frac{\nabla_{x_t} p(x_t)}{p(x_t)} = \int \nabla_{x_t} \log p(x_t|x_0) \frac{p(x_t|x_0) p(x_0)}{p(x_t)} dx_0
$$

根据贝叶斯定理 $p(x_0|x_t) = \frac{p(x_t|x_0)p(x_0)}{p(x_t)}$，上式简化为：

$$
\nabla_{x_t} \log p(x_t) = \int \nabla_{x_t} \log p(x_t|x_0) p(x_0|x_t) dx_0 = \mathbb{E}_{p(x_0|x_t)} [\nabla_{x_t} \log p(x_t|x_0)]
$$

**结论：边缘分数等于条件分数的后验期望。**

---

## 4. Score Matching 目标的等价性证明

我们要对比两个目标函数：
1. **Score Matching (SM):** 直接匹配边缘分数。
2. 
   $$
   J_{SM}(\theta) = \mathbb{E}_{p(x_t)} \left[ \frac{1}{2} \| s_\theta(x_t) - \nabla_{x_t} \log p(x_t) \|^2 \right]
   $$

3. **Denoising Score Matching (DSM):** 匹配条件分数（DMD 实际采用的）。
4. 
   $$
   J_{DSM}(\theta) = \mathbb{E}_{p(x_0, x_t)} \left[ \frac{1}{2} \| s_\theta(x_t) - \nabla_{x_t} \log p(x_t|x_0) \|^2 \right]
   $$

### 推导过程：
利用**全期望公式** $\mathbb{E}_{p(x_0, x_t)}[\cdot] = \mathbb{E}_{p(x_t)}[\mathbb{E}_{p(x_0|x_t)}[\cdot]]$，将 $J_{DSM}$ 展开：

$$
J_{DSM}(\theta) = \mathbb{E}_{p(x_t)} \mathbb{E}_{p(x_0|x_t)} \left[ \frac{1}{2} \| s_\theta(x_t) \|^2 - s_\theta(x_t) \cdot \nabla_{x_t} \log p(x_t|x_0) + \frac{1}{2} \| \nabla_{x_t} \log p(x_t|x_0) \|^2 \right]
$$

将内部期望 $\mathbb{E}_{p(x_0|x_t)}$ 逐项作用：
*   **第一项**：$\frac{1}{2} \| s_\theta(x_t) \|^2$ 与 $x_0$ 无关，保持不变。
*   **第二项**：$- s_\theta(x_t) \cdot \mathbb{E}_{p(x_0|x_t)} [\nabla_{x_t} \log p(x_t|x_0)]$。根据第 3 节的引理，这等于 $- s_\theta(x_t) \cdot \nabla_{x_t} \log p(x_t)$。
*   **第三项**：与参数 $\theta$ 无关，在优化时可视为常数 $C$。

整合后得到：

$$
J_{DSM}(\theta) = \mathbb{E}_{p(x_t)} \left[ \frac{1}{2} \| s_\theta(x_t) \|^2 - s_\theta(x_t) \cdot \nabla_{x_t} \log p(x_t) \right] + C
$$

对比 $J_{SM}$ 的展开式：

$$
J_{SM}(\theta) = \mathbb{E}_{p(x_t)} \left[ \frac{1}{2} \| s_\theta(x_t) \|^2 - s_\theta(x_t) \cdot \nabla_{x_t} \log p(x_t) + \text{const} \right]
$$

### 结论：

$$
J_{DSM}(\theta) = J_{SM}(\theta) + \text{Const}
$$

这意味着两者的梯度 $\nabla_\theta$ 是完全一致的。**优化易于计算的条件分数目标，在统计期望上完全等价于优化复杂的边缘分布目标。**

---

## 5. 总结与直观理解

1.  **蒙特卡洛积分的视角**：直接计算边缘分布需要对全局进行积分。通过转化为条件分数目标，我们实际上是在进行蒙特卡洛采样——每次采样一个样本对 $(x_0, x_t)$，计算其梯度。随着训练 Batch 的累积，这些样本梯度的平均值会精确指向全局分布对齐的方向。
2.  **DMD 的工程实现**：在 DMD 中，由于 $p(x_t|x_0)$ 通常是标准的高斯加噪过程，其分数 $\nabla_{x_t} \log p(x_t|x_0)$ 有解析解（即 $-\frac{\epsilon}{\sigma_t}$），这使得大规模训练变得简单且高效。

通过这一数学证明，我们不仅确认了 DMD 损失函数的合理性，也深刻理解了扩散模型为何能通过“去噪”这一简单任务，最终学会生成复杂的真实分布。

# Latent Diffusion Model
至此，我们从VAE来又回到了VAE，但是此VAE又不复当年。

为了减少计算量，将diffusion过程从像素空间搬到一个潜空间中。这里的潜空间选用vae的潜空间。

注意，这里的vae与原始的vae有一些不同。

原始的VAE没有那么多各种各样的重建损失，还被KL压缩得太狠，损失掉信息，并且明明潜空间不是高斯还非要按高斯来采样，自然不行。

LDM中的VAE更加重视重建，对潜空间本身没有太多要求，只约束方差不要太大，更像是一种AE（当然并非是一一对应的那种）。

之前直接把潜空间当高斯分布来采样当然不行，而现在我们有了diffusion以后就能很从容地从高斯出发采样潜空间了。

# Noise Schedule

## 一、 离散马尔科夫转移到连续 SDE 的等价性证明

在 Score-based 视角中，连续时间 $t \in [0, 1]$ 是通过将离散步骤 $N \to \infty$ 取极限得到的。连续形式的 SDE 一般定义为：
$$dx = f(x, t)dt + g(t)dw$$
其中 $f(x, t)$ 是漂移系数（Drift coefficient），$g(t)$ 是扩散系数（Diffusion coefficient），$dw$ 是标准的布朗运动（Wiener process）。

我们可以通过 **Euler-Maruyama 离散化** 结合 **泰勒展开** 来证明离散马尔科夫链与连续 SDE 的等价性。设定时间步长 $\Delta t = 1/N$。

### 1. Variance Preserving (VP SDE) ——  DDPM
**离散马尔科夫转移（DDPM）**:
$x_i = \sqrt{1 - \beta_i} x_{i-1} + \sqrt{\beta_i} z_i, \quad z_i \sim \mathcal{N}(0, I)$

**推导过程**:
令离散的噪声参数 $\beta_i$ 与连续时间的连续函数 $\beta(t)$ 关联，即 $\beta_i = \beta(t_i)\Delta t$。
当 $\Delta t \to 0$ 时，对 $\sqrt{1 - \beta_i}$ 进行一阶泰勒展开：
$\sqrt{1 - \beta_i} = \sqrt{1 - \beta(t_i)\Delta t} \approx 1 - \frac{1}{2}\beta(t_i)\Delta t$

代入原式计算增量 $x_i - x_{i-1}$：
$x_i - x_{i-1} \approx \left(1 - \frac{1}{2}\beta(t_i)\Delta t\right)x_{i-1} - x_{i-1} + \sqrt{\beta(t_i)\Delta t} z_i$
$x_i - x_{i-1} \approx -\frac{1}{2}\beta(t_i)x_{i-1}\Delta t + \sqrt{\beta(t_i)}\sqrt{\Delta t} z_i$

在极限 $\Delta t \to 0$ 下，增量 $x_i - x_{i-1} \to dx$，$\Delta t \to dt$，且随机项 $\sqrt{\Delta t} z_i \to dw$（布朗运动的增量特性）。
**连续 SDE**:
$$dx = -\frac{1}{2}\beta(t)x dt + \sqrt{\beta(t)} dw$$
**Q:什么是VP？（方差保持）**

**A:在加噪的每一步中，整个图像数据的边缘分布的方差（Magnitude / Scale）被保持住了。**

让我们用最简单的数学来看。扩散模型的前向过程公式是：
$$z_t = \alpha_t x + \sigma_t \epsilon$$

在把图像 $x$ 喂给扩散模型之前，我们通常会做一步极其重要的数据预处理：**把图像像素值归一化到 $[-1, 1]$ 之间**。
这意味着，在宏观统计上，我们可以假设初始真实数据 $x$ 的均值为 0，**方差为 1**，即 $Var(x) = 1$。
同时，我们注入的纯噪声 $\epsilon \sim \mathcal{N}(0, I)$，它的**方差也是 1**，即 $Var(\epsilon) = 1$。

现在，我们来计算一下 $t$ 时刻，混合了信号和噪声的图像 $z_t$ 的方差：
根据概率论基本公式（因为真实图像 $x$ 和采样的噪声 $\epsilon$ 是相互独立的）：
$$Var(z_t) = Var(\alpha_t x + \sigma_t \epsilon)$$
$$Var(z_t) = \alpha_t^2 Var(x) + \sigma_t^2 Var(\epsilon)$$
$$Var(z_t) = \alpha_t^2 \cdot 1 + \sigma_t^2 \cdot 1 = \alpha_t^2 + \sigma_t^2$$

**“方差保持”的本质就在这里：**
我们希望无论加噪到了哪一步（无论是 $t=1$ 还是 $t=1000$），输入给 U-Net 的图像 $z_t$ 的整体方差**始终维持在 1 不变**。
为了让 $Var(z_t) \equiv 1$，我们就**必须强行设计一个规则（约束条件）**：
$$\alpha_t^2 + \sigma_t^2 = 1$$
这就是 VP（方差保持）这个名字的全部来源！

**Q: 为什么要强行保持方差为 1？**
**A:神经网络极其讨厌方差剧烈变化的输入。**

*   如果不管方差，随着不断加噪声，图片的数值范围会变得越来越大。
*   U-Net 在 $t=1$ 时看到的是方差为 1 的数据，在 $t=1000$ 时可能看到的是方差为 100 的数据。这种巨大的尺度差异会导致网络权重的梯度极其不稳定（容易梯度爆炸或消失），模型根本收敛不了。
*   **VP 的意义在于**：它保证了 U-Net 在任何时刻接收到的输入张量（Tensor），其数值尺度（Scale）永远是稳定的 $\mathcal{N}(0, I)$。这极大地降低了神经网络的拟合难度。

---


### 2. Variance Exploding (VE SDE)
**离散马尔科夫转移（SMLD）**:
$x_i = x_{i-1} + \sqrt{\sigma_i^2 - \sigma_{i-1}^2} z_i, \quad z_i \sim \mathcal{N}(0, I)$

**推导过程**:
在这里，数据没有衰减项（$x_{i-1}$ 的系数为 1），仅有加噪项。
令方差的增量 $\sigma_i^2 - \sigma_{i-1}^2 \approx \frac{d[\sigma^2(t)]}{dt} \Delta t$。
代入原式计算增量 $x_i - x_{i-1}$：
$x_i - x_{i-1} = \sqrt{\frac{d[\sigma^2(t)]}{dt} \Delta t} z_i = \sqrt{\frac{d[\sigma^2(t)]}{dt}} \sqrt{\Delta t} z_i$

同样取极限 $\Delta t \to 0$：
**连续 SDE**:
$$dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}} dw$$

---

#### 3. sub-VP SDE 
Yang Song 在论文中提出了一种方差比 VP 更小的变体。其马尔科夫转移并非从历史模型中继承，而是直接为了在连续时间中约束方差而设计的。
**连续 SDE**:
$$dx = -\frac{1}{2}\beta(t)x dt + \sqrt{\beta(t)(1 - e^{-2\int_0^t \beta(s)ds})} dw$$

---

### 二、 三种调度的核心机制与优势

为了理解它们“好在哪”，我们需要看它们随着时间 $t$ 变化时，**边缘分布 $q(x_t | x_0)$** 的方差表现。

| 调度方案 | 边缘分布方差 $\text{Var}[x_t \vert x_0]$ | 优势与核心亮点 |
|:---|:---|:---|
| **VE SDE** | $\sigma^2(t)I$ | **方差爆炸，保持尺度**：模型不改变原始数据的均值尺度（只加噪不收缩）。在处理高维空间（如高分辨率图像、3D点云）时，数学形式极其简单，非常适合基于 Score Matching 的网络去学习不同尺度下的分数。 |
| **VP SDE** | $(1 - e^{-\int_0^t \beta(s)ds})I$ | **方差守恒，极限稳定**：由于 $-\frac{1}{2}\beta(t)x$ 这个负漂移项的存在，数据会被不断拉向原点。当 $t \to \infty$ 时，其方差始终被死死限制在 $1$（即标准正态分布）。这使得训练极其稳定，天然契合变分推断（VAE）和似然估计的数学框架。 |
| **sub-VP SDE** | $(1 - e^{-\int_0^t \beta(s)ds})^2 I$ | **方差更小，更高似然**：这是论文作者通过数学推导出的特例。它的方差在任何时刻都严格小于 VP SDE。**好在哪**：在精确估计似然度（Likelihood）和计算 BPD（Bits Per Dimension）跑分时，由于随机游走的方差被进一步压扁，前向和反向过程的轨迹更加确定，通常能得到目前这三种里最好的理论似然度得分。 |

---

### 三、 究竟满足怎样的标准才算一个“好”的噪声调度？

评价一个噪声调度方案好坏，本质上是在评价它**构建数据与噪声之间桥梁（信噪比演化轨迹）的质量**。在数学和工程上，一个完美的调度需要满足以下四大标准：

1. **信噪比 (SNR) 边界的极致覆盖**
   * **初始状态 ($t=0$)**：SNR 必须趋近于 $\infty$。模型能看见纯净的数据，没有任何噪声干扰。如果 $t=0$ 时仍有微弱噪声（这在某些早期的线性调度中会发生），会导致模型生成的图片在最后一刻依然模糊。
   * **终结状态 ($t=1$)**：SNR 必须趋近于 $0$（即所谓的 Zero-SNR 缺陷修复）。数据必须彻底被磨灭为完全的先验分布 $\mathcal{N}(0, I)$。如果信息没擦干净，模型在反向生成时就容易产生颜色偏移（例如生成图片平均亮度偏暗）。
2. **中间状态的信息流失平滑度 (Smooth Information Destruction)**
   * 在加噪的中间过程，信号不能消失得太快，也不能太慢。如果破坏太快（像线性的 VP 调度在后期），模型绝大多数时间都在对着纯噪声“瞎猜”；如果像 Cosine Schedule 一样让信噪比线性下降，网络在每一个时间步都能获得稳定且有效的分数匹配（Score Matching）梯度。
3. **网络学习的难易度 (Optimal Target Variance)**
   * 好的调度应当使得目标函数的方差在整个时间步上尽可能平稳。如 Karras 在 EDM 中指出的，如果 $\sigma_t$ 选取不当，某些时间步的 Loss 权重会极大，导致网络只顾着拟合某一种特定程度的噪声，而忽略了其他细节。
4. **易于计算和求解 ODE (Tractability)**
   * 它必须能推导出解析的 $q(x_t | x_0)$（即必须能写成高斯分布），否则训练时无法通过“一步加噪”进行重参数化采样，这直接决定了扩散模型能否进行大规模并行训练。

**从宏观（物理和流形）的角度来看，我们真正关心的确实是边缘分布 $q(x_t)$。** 因为正是 $q(x_t)$ 描述了整个真实数据分布 $p_{data}(x)$ 是如何一步步演化成纯噪声分布 $\mathcal{N}(0, I)$ 的。

但在**数学推导、模型训练和调度设计**的微观视角中，我们绝口不提（或者说尽量避开） $q(x_t)$，而是死死盯住条件分布（也就是转移核） $q(x_t|x_0)$ 及其方差。

为什么会产生这种反差？核心原因可以归结为以下三点：

### 1. 现实的无奈：$q(x_t)$ 在数学上是“不可计算”的 (Intractable)

根据全概率公式，$q(x_t)$ 是由初始数据分布和转移概率积分得来的：
$$q(x_t) = \int q(x_t|x_0) p_{data}(x_0) dx_0$$

这里的致命问题是：**我们根本不知道真实的数据分布 $p_{data}(x_0)$ 是什么表达式！** 
自然界中所有猫的图片、所有高分辨率的 3D 点云，它们的分布解析式是未知的（这也是我们为什么要训练生成模型的原因）。因为 $p_{data}(x_0)$ 未知且极其复杂，上述积分根本无法求解。

如果 $q(x_t)$ 算不出来，我们就无法知道在任意时刻 $t$ 的**分数 (Score)** $\nabla_{x_t} \log q(x_t)$。既然不知道真实的分数，神经网络 $s_\theta(x_t, t)$ 就失去了学习的“目标标签”（Ground Truth）。

### 2. 破局的关键：Denoising Score Matching (去噪分数匹配) 的魔法

2011年，Pascal Vincent 提出了去噪分数匹配（后来被 Yang Song 完美融合到 SDE 框架中）。这是一个堪称“魔法”的数学等价定理：

**“拟合未知的边缘分布分数 $\nabla_{x_t} \log q(x_t)$，在数学期望上等价于拟合已知的条件分布分数 $\nabla_{x_t} \log q(x_t|x_0)$。”**

由于 $q(x_t|x_0)$ 通常被我们设计为一个简单的高斯分布 $\mathcal{N}(\mu(x_0, t), \sigma^2(t)I)$，它的分数极其简单，就是纯粹的噪声：
$$\nabla_{x_t} \log q(x_t|x_0) = -\frac{x_t - \mu(x_0, t)}{\sigma^2(t)} = -\frac{\epsilon}{\sigma(t)}$$

**结论：** 我们之所以只盯着 $q(x_t|x_0)$，是因为**它是我们在整个系统里唯一能写出解析式、唯一能精确计算、并直接作为神经网络 Loss 目标的东西**。

### 3. 为什么要特别关注 $q(x_t|x_0)$ 的“方差”？

如果你仔细看上面那个分数的解析式 $\nabla_{x_t} \log q(x_t|x_0) = -\frac{\epsilon}{\sigma(t)}$，你会发现分母正是 $q(x_t|x_0)$ 的标准差 $\sigma(t)$。

我们极其关注方差 $\sigma^2(t)$（也就是噪声调度），原因如下：

*   **方差定义了信噪比 (SNR)：** $x_t$ 是由原始数据（信号）和注入的方差（噪声）组合而成的。方差的变化曲线 $\sigma^2(t)$ 直接决定了在时刻 $t$，图像里还剩多少“原图”，加了多少“雪花”。
*   **方差控制了神经网络的学习难度：** 
    *   如果方差增长太快（例如一上来就加满噪声），网络在 $t=0.1$ 时就完全看不见原图了，只能瞎猜，导致 Loss 爆炸或梯度消失。
    *   如果方差增长太慢，网络大部分时间都在做极其简单的微小去噪任务，不仅浪费算力，而且到了最后时刻 $T$，图像还没变成纯正态分布。
*   **保证 $q(x_T)$ 成为标准高斯的先决条件：** 我们虽然算不出 $q(x_t)$，但我们**必须保证**终局的 $q(x_T) \approx \mathcal{N}(0, I)$。怎样才能保证？只有当 $q(x_T|x_0)$ 的方差大到足以彻底“淹没”任意 $x_0$ 的特征时，积分出来的 $q(x_T)$ 才会是一个完美的纯噪声分布。



这是一个非常务实的问题！既然咱们不谈纯理论，那我们就直接从**工业界（如 OpenAI, NVIDIA, Stability AI）的实际工程经验**出发，来看看在真实的代码和模型中，究竟哪种更好，以及大家都在用什么“具体参数和配方”。

先给出直接的结论：**没有绝对的谁比谁好，但 VP 赢得了“古典时代”，EDM 统一了标准，而现在的“当红炸子鸡”是 Flow Matching（本质上是一种全新的调度）。**

*   **早期的赢家是 VP (方差保持)**：因为 VP 在加噪过程中强行把方差缩放到 1，这让神经网络的输入数值范围始终稳定在 $[-1, 1]$ 附近，**极大地避免了梯度爆炸和数值溢出**。早期的 Stable Diffusion (1.4/1.5) 和 Midjourney 都是基于 VP 的。
*   **VE (方差爆炸) 的工程痛点**：VE 的方差会一直涨到几百甚至上千，这在写代码时对神经网络的 LayerNorm 和权重初始化极度不友好。
*   **后来的大一统 (EDM)**：NVIDIA 的 Karras 证明，只要你在网络输入前做一次**“缩放预处理 (Pre-conditioning)”**，VE 和 VP 效果一模一样。

下面是目前工业界**最流行、最成熟的 4 种具体噪声调度设置方案**（可以直接抄进代码里的那种）：


### 1. Cosine Schedule (余弦调度) —— OpenAI 的经典配方
**适用场景**：在像素空间（Pixel-space）直接生成图像，或者做 3D/4D 视觉的基础重建。
**为什么好**：最早的线性调度（Linear）在最后几个时间步加噪太猛，导致图像信息瞬间丢失，模型最后全在瞎猜。Cosine 调度让加噪过程变成一条平滑的“S型曲线”，中间慢、两头快，极大地保留了图像的中频细节。

**具体设置方案 (公式与参数)**：
*   具体函数：$\bar{\alpha_t}=f(t) = \cos^2\left( \frac{t + s}{1 + s} \cdot \frac{\pi}{2} \right) (t \in [0,1])$
*   定义噪声方差 $\beta_t =1- \frac{\bar{\alpha_t}}{\bar{\alpha_{t-1}}}$
*   **核心参数**：偏移量 $s = 0.008$。加这个 $s$ 是为了防止在 $t=0$（刚开始加噪）时噪声太小导致网络难以优化。
![cos scheduler](/images/blog/diffusion/image4.png)
### 2. Zero-SNR & Enforce Zero (零信噪比调度) —— SD 2.0+ 的补丁
**适用场景**：需要生成极暗、极亮图像，或者做高保真图像编辑（Inpainting）。
**为什么好**：人们发现标准 VP 调度在最后一步 $T$ 时，信噪比并没有降到绝对的 0（原图信息还剩了一点点隐蔽的低频信号）。这导致 Stable Diffusion 1.5 永远画不出“纯黑的夜空”或“纯白的雪地”，平均亮度总是灰蒙蒙的。

**具体设置方案 (Lin et al., 2023)**：
*   在代码里强制修改最后一步：原本 $\bar{\alpha}_T$ 可能是一个很小的数（如 $10^{-4}$），现在**强制设置 $\bar{\alpha}_T = 0$**。
*   同时对前面的时间步进行线性缩放（Rescale），确保曲线平滑。配合 `v-prediction` 目标函数一起使用，直接解决了图像发灰的问题。
```python
# Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas
```



### 3. EDM Schedule (Karras 调度) —— 追求极致质量的工业标准
**适用场景**：高分辨率图像生成，当前许多最强二次元模型（如 NovelAI 及其变体）底层的采样策略。
**为什么好**：Karras 抛弃了传统的 $t$，直接把噪声标准差 $\sigma$ 作为调度的核心。他发现网络在特定大小的噪声下学得最好，于是提出在训练时，直接按照对数正态分布去采样 $\sigma$。

**具体设置方案 (来自 EDM 论文的硬核参数)**：
*   训练时采用对数正态分布抽样噪声大小：$\ln(\sigma) \sim \mathcal{N}(P_{mean}, P_{std}^2)$
*   **最佳超参数**：$P_{mean} = -1.2$, $P_{std} = 1.2$。
*   采样时（Inference）使用等比数列加上微小的多项式扭曲：
    $\sigma_i = \left( \sigma_{max}^{\frac{1}{\rho}} + \frac{i}{N-1}(\sigma_{min}^{\frac{1}{\rho}} - \sigma_{max}^{\frac{1}{\rho}}) \right)^\rho$
*   **最佳超参数**：$\sigma_{min} = 0.002$, $\sigma_{max} = 80$, $\rho = 7$。


### 4. Rectified Flow / Flow Matching (流匹配) —— 当下最火的绝对主流
**适用场景**：Stable Diffusion 3, Flux.1, Sora 以及最新的 3D/视频生成大模型。
**为什么好**：不玩 SDE 的弯弯绕绕了，直接用常微分方程 (ODE) 走**直线**。从纯噪声到真实图像，两点之间直线最短。这让采样速度发生质变（过去需要 50 步，现在 4-8 步就能出极高质量的图），而且极其容易扩展到高分辨率。

**具体设置方案 (Shifted Flow)**：
*   加噪极其简单，就是一个线性组合：$x_t = t \cdot \text{noise} + (1 - t) \cdot \text{data}$ （$t \in [0, 1]$）
*   **高分辨率的杀手锏 (Time Shifting)**：SD3 和 Flux 发现，分辨率越高（比如 1024x1024），图像本身含有的信息量极大，同样的噪声加进去显得“微不足道”。因此他们引入了 Shift 参数 $m$：
    $t' = \frac{m \cdot t}{1 + (m - 1) \cdot t}$
*   **具体参数**：对于 256x256 图像 $m \approx 1$；对于 1024x1024 图像，**$m$ 通常设置为 $3.0$ 甚至更大**（把重点计算资源集中在噪声更大的早期阶段）。