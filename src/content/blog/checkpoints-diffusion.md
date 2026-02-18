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

# DDPM
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

## 另外两种视角

而事实上这个目标函数还有另外两种等效的形式

### 第二种
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
### 第三种
一个基于score-function的理解，可以看这个[视频](https://www.youtube.com/watch?v=lUljxdkolK8)理解一下大概的思想。

接着可以看[Yang Song的blog](https://yang-song.net/blog/2021/score/),没有什么比这个写得更直观且完整了。

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


# Distribution Matching Distillation

在单步扩散模型蒸馏（One-step Diffusion Distribution Matching Distillation, DMD）等论文中，损失函数的核心思想是让 Student 模型生成的分布 $p_{\text{fake}}$ 匹配 Teacher 模型的真实分布 $p_{\text{real}}$。

在数学实现时，一个关键的转化是将难以直接计算的**边缘分布分数（Marginal Score）** $\nabla \log p(x_t)$ 转化为**条件分布分数（Conditional Score）** $\nabla \log p(x_t | x_0)$。本文将从数学推导出发，详细证明这一转化的合理性及其背后的**去噪分数匹配（Denoising Score Matching, DSM）**理论。

## 1. 核心问题背景

在扩散模型框架下，分布对齐等价于在各个噪声水平 $t$ 下对齐 Score：

$$
\nabla_{x_t} \log p_{\text{fake}}(x_t, t) \longleftrightarrow \nabla_{x_t} \log p_{\text{real}}(x_t, t)
$$

然而，边缘分布 $p(x_t)$ 是通过对整个数据集进行积分得到的：

$$
p(x_t) = \int p(x_t|x_0)p(x_0) dx_0
$$

直接对该积分求对数梯度在数学上是不可解的（Intractable）。为此，DMD 引入了 **Denoising Score Matching (DSM)** 的等价性结论。

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