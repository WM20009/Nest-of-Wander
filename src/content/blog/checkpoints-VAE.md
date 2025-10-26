---
title: VAE
date: "2025-10-12"
author: Wander
authorImage: /images/avatar.jpg
category: design
tags:
- VAE
description: 学习VAE时存一下checkpoints，便于复习。
---
> 对于许多模态，我们可以将我们观察到的数据视为由相关的不可见的潜在变量表示或生成的，我们可以用随机变量 z 来表示。表达这个想法的最佳直觉是通过柏拉图的Allegory of the Cave.
> 在寓言中，一群人一生都被锁在一个洞穴里，只能看到火光前经过的不可见三维物体投射在他们面前墙上的二维阴影。对这些人来说，他们所观察到的一切实际上是由他们永远无法看到的高维抽象概念决定的。

## Vanilla VAE
- 如果你不求甚解，Start here:
  - VAE是建模$P_{data}(x)$的生成模型。
  - 训练时VAE就是一个encoder-z-decoder架构，推理时直接采样z然后生成图片。
  - 隐变量z是多维标准高斯分布，不随训练而发生改变。
  - encoder生成给定样本x后,z的后验高斯分布的$\mu$和$\sigma$。**注意:生成后验分布只是为了更容易采样去估计重构项，并不是要重塑z的分布，z依然是标准高斯分布**
  - decoder就是接收一个z，生成给定z时的真实数据的概率分布。**注：这里如果x展平后各维度都是连续的，就建模为各维度独立的高斯分布，然后decoder输出$\mu$和$\sigma$；如果是离散的就待用分类分布或者两点分布，输出每一维的概率。**
    - 我们也许会问，为什么decoder不是直接把隐变量映射到一张确定的图片上去？这个用洞穴语言也能解释：我们把高维向量压缩到低维，那很有可能会把高维空间中不同的向量压缩到同一个低维向量，所以这并不是一个一一映射，就当然不可以预测确定的图片。就像很多不同的物体映在墙上的影子也许是一样的。
  - 训练过程：目标是最大化ELBO,分为重构项（后验分布下，样本x的对数似然）和正则项（后验分布和先验分布的reverse-KL散度），正则项有解析解；重构项通过重参数化和采样进行估计，具体来说是从标准正态分布中采样一个$\epsilon$,然后$z=\mu+\sigma*\epsilon$,输入decoder计算似然。
- 如果你刨根问底，From scratch:
  - 首先给出[参考链接](https://zhuanlan.zhihu.com/p/711402258)
  - 然后参考原文😂

## Hierarchical VAE
正如上面所说，观测数据可能是潜变量的投影，而潜变量可能又是更抽象的变量的投影，于是我们自然引出了分层的VAE，即HVAE。

而在一般的具有 $T$ 层级的 HVAE 中，每个潜在变量都可以依赖于所有之前的潜在变量，在本文中我们关注一种特殊的案例，我们称之为马尔可夫 HVAE(MHVAE）。在 MHVAE 中，生成过程是一个马尔可夫链；也就是一个具有 $T$ 个层次潜在变量的马尔可夫链分层变分自编码器。生成过程被建模为一个马尔可夫链，其中每个潜在 $z_t$ 仅从之前的潜在 $z_{t+1}$ 生成。就是说，每一层向下的转移都是马尔可夫性的，其中解码每个潜在变量 $z_t$ 仅依赖于前一个潜在变量 $z_{t+1}$ 。
<div style="width: 60%; margin: 0 auto;">
  <img src="/images/blog/checkpoints-VAE/image.png">
</div>
直观上和视觉上，这可以看作是将 VAEs 逐层堆叠在一起.

**为什么要分层？** 我目前认为如果只有一层隐变量，被信息被压缩得太过厉害，我们很难训练一个能很好恢复的解码器。而分成很多层也许能允许我们更细粒度地从隐变量分布恢复出真实分布

### 一些自己瞎搞的数学推导
---
#### 各维独立的多维分布的KL散度等于各维KL散度的和
**Proof.**
$$
\begin{align*}
  KL(q(z)||p(z))&=\int_z q(z)\log{\frac{q(z)}{p(z)}}dz\\
                &=\int_z \prod_{i=1}^Jq(z_i)\sum_{i=1}^J\log{\frac{q(z_i)}{p(z_i)}}dz\\
                &=\int_{z_1}q(z_1)dz_1\int_{z_2}q(z_2)dz_2\dots\int_{z_n}q(z_n)\sum_{i=1}^J\log{\frac{q(z_i)}{p(z_i)}}dz_n\\
                &=\int_{z_1}q(z_1)dz_1\int_{z_2}q(z_2)dz_2\dots\int_{z_n}q(z_n)(\log{\frac{q(z_n)}{p(z_n)}}+\sum_{i=1}^{n-1}\log{\frac{q(z_i)}{p(z_i)}})dz_n\\
                &=\int_{z_1}q(z_1)dz_1\int_{z_2}q(z_2)dz_2\dots[\int_{z_{n}}q(z_n)\log{\frac{q(z_n)}{p(z_n)}}dz_n+\sum_{i=1}^{n-1}\log{\frac{q(z_i)}{p(z_i)}}]\\
                &=\int_{z_1}q(z_1)dz_1\int_{z_2}q(z_2)dz_2\dots\int_{z_{n-1}}q(z_{n-1})(KL(q(z_n)||p(z_n))+\sum_{i=1}^{n-1}\log{\frac{q(z_i)}{p(z_i)}})dz_{n-1})\\
                &\dots\\
                &=\int_{z_1}q(z_1)(\sum_{i=2}^{n}KL(q(z_i)||p(z_i))+\log{\frac{q(z_1)}{p(z_1)}})dz_1\\
                &=\sum_{i=1}^{n}KL(q(z_i)||p(z_i))
\end{align*}
$$
其中,$J$为随机向量$z$的维度，第二个等号是由于各维度相互独立，第四个等号是由于,当p(x)为概率密度函数时$\int_x p(x) dx=1$

原文中这个结论可以由积分的线性性质直接得到，但我不是很懂，故而自己用笨办法推导了一下。
#### 为什么最大化观测数据的似然就是最大化ELBO
