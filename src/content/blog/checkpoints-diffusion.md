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
# VDM
**这里省略了大量的数学推导，仅保留必要的数学表达式，以提高可读性**

书接上回，我们可以认为variable diffusion model就是具有如下三条假设的HVAE:
-  潜在维度恰好等于数据维度
- 每个时间步的潜在编码器结构不是学成的；它是作为线性高斯模型预先定义的。换句话说，它是一个
以前一时间步输出为中心的高斯分布。
- 潜在编码器的高斯参数随时间变化，使得最终时间步T 的潜在分布为标准正态分布
## 编码器
**编码器**显式地建模为:
$$q\left(\boldsymbol{x}_{t} \mid \boldsymbol{x}_{t-1}\right)=\mathcal{N}\left(\boldsymbol{x}_{t} ; \sqrt{\alpha_{t}} \boldsymbol{x}_{t-1},\left(1-\alpha_{t}\right) \mathbf{I}\right)$$
这成为线性高斯模型。
它有一些十分良好的性质，当我们给定了$\alpha_1,\alpha_2,\dots,\alpha_T$之后，$q(\boldsymbol{x}_t \mid \boldsymbol{x}_0)$和$q(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0)$可以直接解析得到，且都是正态分布。

其中，$q(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\sim \mathcal{N}(\sqrt{\bar{\alpha_t}}\boldsymbol{x_0},(1-\bar{\alpha_t})\mathbf{I}),\bar{\alpha_t}=\prod_{i=1}^t\alpha_i$
另一下在下文给出。
## ELBO
$ELBO$为:

$$
\underbrace{\mathbb{E}_{q\left(\boldsymbol{x}_{1} \mid \boldsymbol{x}_{0}\right)}\left[\log p_{\theta}\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{1}\right)\right]}_{\text {reconstruction term }}-\underbrace{D_{\mathrm{KL}}\left(q\left(\boldsymbol{x}_{T} \mid \boldsymbol{x}_{0}\right) \| p\left(\boldsymbol{x}_{T}\right)\right)}_{\text {prior matching term }}-\sum_{t=2}^{T} \underbrace{\mathbb{E}_{q\left(\boldsymbol{x}_{t} \mid \boldsymbol{x}_{0}\right)}\left[D_{\mathrm{KL}}\left(q\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{t}, \boldsymbol{x}_{0}\right) \| p_{\theta}\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{t}\right)\right)\right]}_{\text {denoising matching term }}
$$

1. $\mathbb{E}_{q(\boldsymbol{x}_{1}|\boldsymbol{x}_{0})}\left[\log p_{\theta}(\boldsymbol{x}_{0}|\boldsymbol{x}_{1})\right]$ 可以解释为一个重构项；类似于普通VAE的ELBO中的对应项，该术语可以使用蒙特卡罗估计进行近似和优化。

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
### 第三种

## 条件生成

# 参考资料
[Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)（所有内容均是基于该教程的压缩和总结）