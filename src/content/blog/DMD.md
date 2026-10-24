---
title: DMD (Distribution Matching Distillation)
date: "2026-4-23"
author: Wander
authorImage: /images/avatar.jpg
category: design
tags:
- Diffusion
- Distillation
description: 从头到尾补全 DMD 的完整逻辑链条与数学推导。
---

# DMD 从头到尾推导

本文尝试把 **DMD (Distribution Matching Distillation)** 原论文里省略掉的大部分中间步骤都补上。目标不是复述论文，而是回答下面几个最关键的问题：

1. DMD 到底在优化什么目标？
2. 为什么这个目标的梯度会变成 `real score - fake score`？
3. 为什么 diffusion denoiser 的输出可以拿来近似这个 score？
4. 为什么一定要有 fake score，不然会发生什么？
5. 为什么原始 DMD 还需要额外的 regression loss？

为了尽量把主线讲清楚，本文默认先忽略 condition `c`。如果考虑文本条件，只需要把所有分布都替换成条件分布 `p(x|c)` 即可，推导结构完全一样。

## 1. 问题背景：一步生成器到底在学什么

我们有一个一步生成器

$$ x = G_\theta(z), \qquad z \sim p(z) = \mathcal N(0, I). $$

它把噪声 `z` 映射到图像 `x`，于是自然诱导出一个生成分布

$$ p_\theta(x). $$

我们真正想要的，是让这个分布尽量接近真实数据分布

$$ p_{\text{real}}(x). $$

一个最自然的目标是最小化 reverse KL：

$$ \mathcal L_{\mathrm{KL}}(\theta) = D_{\mathrm{KL}}(p_\theta \| p_{\mathrm{real}}) = \int p_\theta(x)\log \frac{p_\theta(x)}{p_{\mathrm{real}}(x)}\,dx. $$

这里先强调一件事：**DMD 并不试图显式计算这个 KL 的数值，而是只关心它对生成器参数的梯度长什么样。**

这和扩散蒸馏、SDS、VSD 这一类方法的共同思路非常像：真正重要的不是你能不能算出一个漂亮的标量 loss，而是你能不能得到一个正确的更新方向。

## 2. 第一步：先别对参数求导，先对“分布怎么移动”求导

论文里最跳的一步，就是直接从 KL 写到了 score difference。要补全这一段，最干净的办法不是上来就对 `\theta` 硬求导，而是先看：

> 如果我把生成分布里的每一个样本点都沿着某个速度场轻轻推一下，KL 会怎么变化？

这本质上是一个分布上的一阶变分问题。

### 2.1 用速度场描述无穷小分布变化

设当前生成分布是 `p(x)`，这里临时省略 `\theta` 下标。考虑一个无穷小流动：

$$ T_\varepsilon(x) = x + \varepsilon v(x), $$

其中 `v(x)` 是一个速度场，`\varepsilon` 很小。把 `p` 经过这个变换后得到的新分布记作 `p_\varepsilon`。

标准连续性方程告诉我们：

$$ \frac{\partial p_\varepsilon(x)}{\partial \varepsilon}\Big|_{\varepsilon=0} = -\nabla_x \cdot (p(x) v(x)). $$

它的含义非常直观：如果某一区域的流入大于流出，密度就会上升；反过来就下降。

### 2.2 对 KL 做一阶变分

定义泛函

$$ \mathcal F(p) = D_{\mathrm{KL}}(p \| p_{\mathrm{real}}) = \int p(x)\left(\log p(x) - \log p_{\mathrm{real}}(x)\right) dx. $$

对 `\varepsilon` 求导：

$$ \frac{d}{d\varepsilon}\mathcal F(p_\varepsilon)\Big|_{\varepsilon=0} = \int \frac{\partial p_\varepsilon(x)}{\partial \varepsilon}\Big|_{\varepsilon=0} \left(1 + \log p(x) - \log p_{\mathrm{real}}(x)\right)\,dx. $$

把连续性方程代入：

$$ = -\int \nabla_x \cdot (p(x) v(x)) \left(1 + \log p(x) - \log p_{\mathrm{real}}(x)\right)\,dx. $$

现在对散度项做分部积分。边界项在常规假设下为 0，于是得到：

$$ = \int p(x) v(x) \cdot \nabla_x \left(1 + \log p(x) - \log p_{\mathrm{real}}(x)\right)\,dx. $$

因为常数 `1` 的梯度为 0，所以

$$ \frac{d}{d\varepsilon}\mathcal F(p_\varepsilon)\Big|_{\varepsilon=0} = \int p(x) v(x)\cdot \left( \nabla_x \log p(x) - \nabla_x \log p_{\mathrm{real}}(x) \right)\,dx. $$

定义两个 score：

$$ s_{\mathrm{fake}}(x) = \nabla_x \log p(x), \qquad s_{\mathrm{real}}(x) = \nabla_x \log p_{\mathrm{real}}(x), $$

于是

$$ \frac{d}{d\varepsilon}\mathcal F(p_\varepsilon)\Big|_{\varepsilon=0} = \mathbb E_{x \sim p} \left[ v(x)\cdot \left(s_{\mathrm{fake}}(x) - s_{\mathrm{real}}(x)\right) \right]. $$

这一步非常关键。它告诉我们：

> 对 KL 来说，若把样本 `x` 当成可移动的粒子，那么使 KL 最快下降的方向就是 $s_{\mathrm{real}}(x) - s_{\mathrm{fake}}(x)$。

也就是说，真实分布的 score 想把样本往真实高密度区域拉，而 fake 分布自己的 score 会把样本从当前 fake 分布已经堆积过头的区域里推开。

这就是 DMD 里那句最核心的话的理论来源。

## 3. 第二步：把“分布上的速度场”翻译回“参数梯度”

上面得到的是“如果样本点以速度场 `v(x)` 移动，KL 怎么变”。但生成器并不是直接优化样本坐标，而是优化参数 `\theta`。

对生成器

$$ x = G_\theta(z), $$

如果参数发生一个无穷小变化 `\delta \theta`，那么样本位置变化为

$$ \delta x = \frac{\partial G_\theta(z)}{\partial \theta}\delta \theta. $$

因此，速度场实际上是

$$ v(x) = \frac{\partial G_\theta(z)}{\partial \theta}\delta \theta $$

在样本空间中的体现。把它代入上一节的变分公式，就得到

$$ \delta \mathcal L_{\mathrm{KL}} = \mathbb E_{z} \left[ \left( s_{\mathrm{fake}}(x) - s_{\mathrm{real}}(x) \right)^T \frac{\partial G_\theta(z)}{\partial \theta}\delta \theta \right]. $$

于是可以直接读出参数梯度

$$ \nabla_\theta \mathcal L_{\mathrm{KL}} = \mathbb E_z \left[ \left(\frac{\partial G_\theta(z)}{\partial \theta}\right)^T \left( s_{\mathrm{fake}}(x) - s_{\mathrm{real}}(x) \right) \right], \qquad x = G_\theta(z). $$

等价地，把负号提出去：

$$ \nabla_\theta \mathcal L_{\mathrm{KL}} = -\mathbb E_z \left[ \left(\frac{\partial G_\theta(z)}{\partial \theta}\right)^T \left( s_{\mathrm{real}}(x) - s_{\mathrm{fake}}(x) \right) \right]. $$

所以对生成器做梯度下降时，真正推着样本走的向量场就是

$$ s_{\mathrm{real}}(x) - s_{\mathrm{fake}}(x). $$

到这里为止，DMD 的主线其实已经出来了：

1. 目标是最小化 `D_{\mathrm{KL}}(p_\theta \| p_{\mathrm{real}})`。
2. 这个目标的变分梯度对应的样本更新方向是 `s_real - s_fake`。
3. 如果我们能估计出这两个 score，就能训练一步生成器。

但新的问题立刻来了：**原始图像空间里的这两个 score 很难直接算。**

## 4. 第三步：为什么不能直接在原始图像空间算 score

在高维图像空间里，`p_\theta` 和 `p_{\mathrm{real}}` 的支撑集通常非常薄、非常复杂，而且经常几乎不重叠。直接在原空间用

$$ \nabla_x \log p_{\mathrm{real}}(x) $$

会碰到两个问题：

1. 理论上很脆弱。某些区域里真实分布密度极低，score 很不稳定。
2. 实践上也不方便。扩散模型真正擅长估计的并不是原始数据分布的 score，而是**加噪后分布**的 score。

所以 DMD 和 Score-SDE、VSD 采用了同一个关键操作：

> 先把 real/fake 分布都用高斯核模糊一遍，再在加噪空间里做 score matching。

## 5. 第四步：构造 perturbed distribution

定义标准扩散式加噪：

$$ x_t = \alpha_t x + \sigma_t \epsilon, \qquad \epsilon \sim \mathcal N(0, I). $$

如果干净样本 `x` 来自真实分布 `p_{\mathrm{real}}`，那么 `x_t` 的分布记作

$$ p_{\mathrm{real}, t}(x_t). $$

如果干净样本 `x` 来自生成器分布 `p_\theta`，那么对应的加噪分布记作

$$ p_{\mathrm{fake}, t}(x_t). $$

更形式化地说，这两个分布都是把原分布和高斯核做卷积得到的：

$$ p_{\ast, t}(x_t) = \int p_\ast(x)\,q_t(x_t|x)\,dx, \qquad q_t(x_t|x) = \mathcal N(\alpha_t x, \sigma_t^2 I). $$

现在 DMD 不再直接最小化原始 KL，而是最小化一个 time-averaged perturbed KL：

$$ \mathcal L_{\mathrm{DM}}(\theta) = \mathbb E_{t \sim \rho(t)} \left[ \lambda(t) D_{\mathrm{KL}} \left( p_{\mathrm{fake}, t}\,\|\,p_{\mathrm{real}, t} \right) \right]. $$

为什么这么做是合理的？

1. 加噪后分布更平滑，score 更容易存在且更容易估计。
2. 预训练 diffusion model 恰好就学的是这族 perturbed score。
3. 不同噪声水平对应不同尺度的几何信息，高噪声看全局结构，低噪声看局部细节。

## 6. 第五步：在加噪空间里重新推导梯度

我们现在对

$$ D_{\mathrm{KL}}(p_{\mathrm{fake}, t}\|p_{\mathrm{real}, t}) $$

做和前面一模一样的变分分析。

对于固定的 `t`，在加噪空间里，最陡下降方向是

$$ s_{\mathrm{real}, t}(x_t) - s_{\mathrm{fake}, t}(x_t), $$

其中

$$ s_{\mathrm{real}, t}(x_t) = \nabla_{x_t}\log p_{\mathrm{real}, t}(x_t), \qquad s_{\mathrm{fake}, t}(x_t) = \nabla_{x_t}\log p_{\mathrm{fake}, t}(x_t). $$

接下来要解释一个关键问题：noisy space 里的更新方向，为什么会在回到干净样本 `x` 时多出一个 `\alpha_t`？这里如果直接写

$$ \nabla_x \log p_{\ast, t}(x_t) = \alpha_t s_{\ast, t}(x_t) $$

其实是不严谨的，因为 `x` 和 `x_t` 并不是单值函数关系；在随机变量层面，`x_t = \alpha_t x + \sigma_t \epsilon` 只说明了它们通过加噪核耦合在一起，并不能直接拿来对“随机变量本身”使用链式法则。

更稳妥的做法，是继续沿用前面“粒子运动”的视角。设干净样本 `x` 在生成空间里发生一个无穷小位移

$$ x \mapsto x + \delta x. $$

那么对同一条噪声路径，也就是固定同一个噪声样本 `\epsilon`，对应的 noisy sample

$$ x_t = \alpha_t x + \sigma_t \epsilon $$

会同步变成

$$ x_t \mapsto x_t + \delta x_t, \qquad \delta x_t = \alpha_t \delta x. $$

这一步说的不是“`x_t` 是 `x` 的全局函数”，而只是说：**沿着同一条采样路径，当干净粒子发生一个无穷小位移时，它在 noisy space 里的对应位移会被缩放为 `\alpha_t` 倍。**

因此，在固定 `t` 的 perturbed KL 变分里，如果干净空间的速度场是 `v(x)`，那么沿同一条噪声路径传到 noisy space 的速度就是

$$ \delta x_t / \delta \varepsilon = \alpha_t v(x). $$

把它代回 noisy space 的一阶变分公式，便得到

$$ \frac{d}{d\varepsilon} D_{\mathrm{KL}}(p_{\mathrm{fake}, t}^{(\varepsilon)} \| p_{\mathrm{real}, t})\Big|_{\varepsilon=0} = \mathbb E_{x \sim p_{\mathrm{fake}},\, \epsilon} \left[ \alpha_t v(x)\cdot \left( s_{\mathrm{fake}, t}(x_t)-s_{\mathrm{real}, t}(x_t) \right) \right]. $$

所以，真正成立的结论不是“对随机变量 `x_t` 对 `x` 求导”，而是：**同一条噪声路径上的无穷小位移满足 `\delta x_t = \alpha_t \delta x`，于是把 noisy-space 的梯度拉回干净样本时，会额外多出一个 `\alpha_t`。**

因此固定 `t` 时，生成器样本 `x` 在图像空间里对应的更新方向变成

$$ \alpha_t \left( s_{\mathrm{real}, t}(x_t) - s_{\mathrm{fake}, t}(x_t) \right). $$

进一步回传到参数空间：

$$ \nabla_\theta \mathcal L_{\mathrm{DM}} = -\mathbb E_{t,z,\epsilon} \left[ \lambda(t) \left(\frac{\partial G_\theta(z)}{\partial \theta}\right)^T \alpha_t \left( s_{\mathrm{real}, t}(x_t) - s_{\mathrm{fake}, t}(x_t) \right) \right], $$

其中

$$ x = G_\theta(z), \qquad x_t = \alpha_t x + \sigma_t \epsilon. $$

到这里，DMD 的训练目标已经被彻底还原成一句话：

> 在不同噪声尺度下，把生成样本往 `perturbed real score` 指向的方向推，同时减去 `perturbed fake score`。

接下来的核心问题就是：**这两个 perturbed score 怎么从 diffusion model 里读出来？**

## 7. 第六步：score identity 到底是怎么来的

这一步是论文里另一处最容易让人卡住的地方。很多人第一次看到下面这个式子时都会皱眉：

$$ s_t(x_t) = -\frac{x_t - \alpha_t \mu(x_t, t)}{\sigma_t^2}. $$

因为表面上看，`p_t(x_t)` 是个极复杂的边缘分布，根本不是高斯，为什么它的 score 却长得像高斯的 score？

答案是：**这个式子不是说边缘分布本身是高斯，而是说它的 score 可以写成“条件高斯 score 的后验平均”。**

下面把这个结论从头推出来。

### 7.1 从边缘分布定义开始

任意原始分布 `p(x)` 经过加噪后，边缘分布为

$$ p_t(x_t) = \int p(x)\,q_t(x_t|x)\,dx, $$

其中

$$ q_t(x_t|x) = \mathcal N(\alpha_t x, \sigma_t^2 I). $$

对 `x_t` 求梯度：

$$ \nabla_{x_t} p_t(x_t) = \int p(x)\,\nabla_{x_t} q_t(x_t|x)\,dx. $$

利用

$$ \nabla q = q \nabla \log q, $$

得到

$$ \nabla_{x_t} p_t(x_t) = \int p(x)\,q_t(x_t|x)\, \nabla_{x_t}\log q_t(x_t|x)\,dx. $$

两边除以 `p_t(x_t)`：

$$ \nabla_{x_t}\log p_t(x_t) = \int \frac{p(x)q_t(x_t|x)}{p_t(x_t)} \nabla_{x_t}\log q_t(x_t|x)\,dx. $$

注意

$$ \frac{p(x)q_t(x_t|x)}{p_t(x_t)} = p(x|x_t), $$

于是

$$ \nabla_{x_t}\log p_t(x_t) = \mathbb E_{p(x|x_t)} \left[ \nabla_{x_t}\log q_t(x_t|x) \right]. $$

这就是所谓的 **score identity**：

> 边缘分布的 score，等于条件分布 score 的后验期望。

### 7.2 把高斯核代进去

由于

$$ q_t(x_t|x) = \mathcal N(\alpha_t x, \sigma_t^2 I), $$

所以它的对数梯度是

$$ \nabla_{x_t}\log q_t(x_t|x) = -\frac{x_t - \alpha_t x}{\sigma_t^2}. $$

代回 score identity：

$$ \nabla_{x_t}\log p_t(x_t) = \mathbb E_{p(x|x_t)} \left[ -\frac{x_t - \alpha_t x}{\sigma_t^2} \right]. $$

因为 `x_t` 和 `\sigma_t` 对条件期望来说都是常数，可以直接提出去：

$$ \nabla_{x_t}\log p_t(x_t) = -\frac{x_t - \alpha_t \mathbb E[x|x_t]}{\sigma_t^2}. $$

这就是完整形式：

$$ s_t(x_t) = \nabla_{x_t}\log p_t(x_t) = -\frac{x_t - \alpha_t \mathbb E[x|x_t]}{\sigma_t^2}. $$

所以真正控制 score 的并不是“边缘分布是高斯”，而是 **高斯加噪核 + 后验均值**。

## 8. 第七步：为什么 denoiser 预测的是后验均值

上节公式里出现了

$$ \mathbb E[x|x_t]. $$

而扩散模型里常见的 `x_0`-prediction denoiser 恰好就和这个对象对应。下面把这一步也写完整。

考虑一个函数 `\mu_\phi(x_t, t)`，用均方误差训练：

$$ \mathcal L_{\mathrm{denoise}}(\phi) = \mathbb E_{x \sim p(x),\, t,\, \epsilon} \left[ \|\mu_\phi(x_t, t) - x\|_2^2 \right], \qquad x_t = \alpha_t x + \sigma_t \epsilon. $$

把期望按 `x_t` 分组展开：

$$ \mathcal L_{\mathrm{denoise}}(\phi) = \int p_t(x_t) \left( \int p(x|x_t) \|\mu_\phi(x_t, t) - x\|_2^2 dx \right) dx_t. $$

对固定的 `x_t` 和 `t`，我们要最小化的是

$$ f(\mu) = \int p(x|x_t)\|\mu - x\|_2^2 dx. $$

对 `\mu` 求导：

$$ \frac{\partial f}{\partial \mu} = 2\int p(x|x_t)(\mu - x)\,dx = 2\mu \int p(x|x_t)\,dx - 2\int x\,p(x|x_t)\,dx. $$

由于

$$ \int p(x|x_t)\,dx = 1, $$

令导数为 0 就得到最优解

$$ \mu^\star(x_t, t) = \int x\,p(x|x_t)\,dx = \mathbb E[x|x_t]. $$

这说明：

> 如果一个 denoiser 用 `x_0` 回归形式的 `L_2` loss 训练到最优，它学到的就是后验均值。

于是上节的 score 公式可以写成

$$ s_t(x_t) \approx -\frac{x_t - \alpha_t \mu_\phi(x_t, t)}{\sigma_t^2}. $$

这就把“denoiser 输出”和“score”连上了。

## 9. 第八步：real score 和 fake score 如何分别得到

现在把上面的通用公式分别应用到真实分布和生成分布。

### 9.1 real score

预训练 diffusion teacher 或 base model 在真实数据上训练，因此它的 `x_0`-prediction denoiser 记作

$$ \mu_{\mathrm{base}}(x_t, t). $$

于是

$$ s_{\mathrm{real}, t}(x_t) \approx -\frac{x_t - \alpha_t \mu_{\mathrm{base}}(x_t, t)}{\sigma_t^2}. $$

### 9.2 fake score

如果我们想估计生成器当前分布 `p_\theta` 的 perturbed score，就需要一个在“当前生成数据”上训练的 denoiser，记作

$$ \mu_{\mathrm{fake}}(x_t, t). $$

它通过对生成器样本做标准去噪回归来训练：

$$ \mathcal L_{\mathrm{fake}}(\phi) = \mathbb E_{z,t,\epsilon} \left[ \left\| \mu_{\mathrm{fake}}(x_t, t) - x \right\|_2^2 \right], $$

其中

$$ x = \operatorname{sg}[G_\theta(z)], \qquad x_t = \alpha_t x + \sigma_t \epsilon. $$

这里 `sg` 表示 stop-gradient，意思是：

1. 更新 fake score network 时，把当前生成图像当成一个固定数据集。
2. 这条去噪损失只训练 fake score model，不反向修改生成器。

于是同理有

$$ s_{\mathrm{fake}, t}(x_t) \approx -\frac{x_t - \alpha_t \mu_{\mathrm{fake}}(x_t, t)}{\sigma_t^2}. $$

### 9.3 两个 score 的差

现在直接相减：

$$ s_{\mathrm{real}, t}(x_t) - s_{\mathrm{fake}, t}(x_t) = \left( -\frac{x_t - \alpha_t \mu_{\mathrm{base}}}{\sigma_t^2} \right) - \left( -\frac{x_t - \alpha_t \mu_{\mathrm{fake}}}{\sigma_t^2} \right). $$

整理得

$$ s_{\mathrm{real}, t}(x_t) - s_{\mathrm{fake}, t}(x_t) = \frac{\alpha_t}{\sigma_t^2} \left( \mu_{\mathrm{base}}(x_t, t) - \mu_{\mathrm{fake}}(x_t, t) \right). $$

这一步非常漂亮，因为 `x_t` 项自动抵消了。于是我们根本不用显式恢复绝对 score，只要比较两个 denoiser 的输出差即可。

## 10. 第九步：把 score difference 代回生成器梯度

第 6 节里我们已经得到 perturbed KL 对生成器的梯度形式：

$$ \nabla_\theta \mathcal L_{\mathrm{DM}} = -\mathbb E_{t,z,\epsilon} \left[ \lambda(t) \left(\frac{\partial G_\theta(z)}{\partial \theta}\right)^T \alpha_t \left( s_{\mathrm{real}, t}(x_t) - s_{\mathrm{fake}, t}(x_t) \right) \right]. $$

再把上一节的结果代进去：

$$ \nabla_\theta \mathcal L_{\mathrm{DM}} \approx -\mathbb E_{t,z,\epsilon} \left[ \lambda(t) \left(\frac{\partial G_\theta(z)}{\partial \theta}\right)^T \alpha_t \cdot \frac{\alpha_t}{\sigma_t^2} \left( \mu_{\mathrm{base}}(x_t, t) - \mu_{\mathrm{fake}}(x_t, t) \right) \right]. $$

即

$$ \nabla_\theta \mathcal L_{\mathrm{DM}} \approx -\mathbb E_{t,z,\epsilon} \left[ \lambda(t)\frac{\alpha_t^2}{\sigma_t^2} \left(\frac{\partial G_\theta(z)}{\partial \theta}\right)^T \left( \mu_{\mathrm{base}}(x_t, t) - \mu_{\mathrm{fake}}(x_t, t) \right) \right]. $$

所以从实现角度看，DMD 的 distribution matching 部分就是：

1. 采样 `z`，得到 `x = G_\theta(z)`。
2. 再采样 `t, \epsilon`，构造 `x_t = \alpha_t x + \sigma_t \epsilon`。
3. 用 frozen base model 得到 `\mu_{\mathrm{base}}(x_t, t)`。
4. 用在线 fake model 得到 `\mu_{\mathrm{fake}}(x_t, t)`。
5. 取它们的差，乘以系数 `\alpha_t^2 / \sigma_t^2`，通过链式法则反传给生成器。

这就是 DMD 的核心训练闭环。

## 11. 第十步：为什么一定要减去 fake score

这一步如果只看公式，很多人会觉得像“经验 trick”。其实不是，它是从 reverse KL 的一阶变分里强行长出来的。

### 11.1 如果只保留 real score 会发生什么

假设你用的更新方向只有

$$ s_{\mathrm{real}, t}(x_t), $$

也就是总想把 fake 样本往真实高密度区域推。那会发生什么？

答案是：所有样本都会试图往“当前看起来最容易增大真实密度”的区域集中。

从粒子视角看，这相当于所有粒子只受外力吸引，不受内部排斥。结果往往是：

1. 样本向少数高密度模式聚集。
2. 不同 `z` 生成的样本逐渐变得相似。
3. 模式覆盖越来越差，最终出现 mode collapse。

### 11.2 fake score 的作用是什么

`s_{\mathrm{fake}, t}(x_t)` 本质上刻画了“当前 fake 分布自己已经堆积成什么形状”。

当某一区域 fake 样本已经很多时，那里的 `\log p_{\mathrm{fake}, t}(x_t)` 较大，它的梯度就会提供一种“别再往这挤了”的修正。

所以

$$ s_{\mathrm{real}, t}(x_t) - s_{\mathrm{fake}, t}(x_t) $$

不是单纯的“朝真实分布前进”，而是更精确地说：

> 朝着能让 `\log p_{\mathrm{real}, t} - \log p_{\mathrm{fake}, t}` 上升的方向前进。

也就是朝着能减小

$$ D_{\mathrm{KL}}(p_{\mathrm{fake}, t}\|p_{\mathrm{real}, t}) $$

的一阶最优方向前进。

### 11.3 从目标函数角度看 fake score 不可省略

因为

$$ \nabla_{x_t} \log \frac{p_{\mathrm{fake}, t}(x_t)}{p_{\mathrm{real}, t}(x_t)} = s_{\mathrm{fake}, t}(x_t) - s_{\mathrm{real}, t}(x_t), $$

所以梯度下降方向自然是

$$ s_{\mathrm{real}, t}(x_t) - s_{\mathrm{fake}, t}(x_t). $$

如果把 fake score 去掉，那你优化的就已经不是原始 reverse KL 的梯度流了，而是另一种更偏启发式的“朝 teacher 靠拢”更新。这也是 SDS 容易 mode collapse 的核心原因之一。

## 12. 第十一步：为什么有了分布匹配还要再加 regression loss

如果 DMD 的理论闭环已经这么漂亮，为什么论文里还要加一个 regression loss？

原因很现实：**理论上正确的梯度，在有限模型容量、有限 batch、fake score 估计不准、one-step 生成器表达能力有限的情况下，不一定足够稳定。**

### 12.1 distribution matching 项的现实问题

即便理论目标是对的，训练时仍然存在几个难点：

1. fake score model 需要在线追着生成器分布跑，估计误差会比较大。
2. reverse KL 本身就偏向 mode-seeking，不天然鼓励覆盖所有模式。
3. 一步生成器的映射容量有限，纯靠 score difference 有时会把样本迅速推向局部更容易的模式。

所以单独使用 distribution matching，虽然方向对，但训练可能不够稳。

### 12.2 regression 项在做什么

原始 DMD 额外加入一个回归项，把 student 生成结果拉向 teacher 的输出：

$$ \mathcal L_{\mathrm{reg}} = \mathbb E_{z} \left[ d\left(G_\theta(z), x_{\mathrm{teacher}}(z)\right) \right]. $$

这里 `d(\cdot, \cdot)` 常取像素或感知空间距离，`x_{\mathrm{teacher}}(z)` 表示同一 latent 或噪声输入下 teacher 生成出的样本。

这个项的作用更像一个锚点：

1. 它告诉 student 至少要保住 teacher 给出的粗结构。
2. 它减缓只靠 distribution matching 时的剧烈漂移。
3. 它在 fake score 还没学稳时，给 student 一条更平滑的训练信号。

于是总目标可以写成

$$ \mathcal L_{\mathrm{total}} = \mathcal L_{\mathrm{DM}} + \gamma \mathcal L_{\mathrm{reg}}. $$

直观上可以理解为：

1. `\mathcal L_{\mathrm{DM}}` 负责“让分布对”。
2. `\mathcal L_{\mathrm{reg}}` 负责“让训练稳”。

## 13. 第十二步：DMD 的完整训练流程

把上面所有部件连起来，DMD 的训练可以写成一个交替优化过程。

### 13.1 更新 fake score model

固定生成器 `G_\theta`，采样

$$ z \sim \mathcal N(0, I), \qquad x = \operatorname{sg}[G_\theta(z)]. $$

再采样

$$ t \sim \rho(t), \qquad \epsilon \sim \mathcal N(0, I), $$

构造

$$ x_t = \alpha_t x + \sigma_t \epsilon. $$

然后最小化

$$ \mathcal L_{\mathrm{fake}}(\phi) = \mathbb E \left[ \|\mu_{\mathrm{fake}, \phi}(x_t, t) - x\|_2^2 \right]. $$

这个步骤的目标是：让 fake model 近似当前 `p_\theta` 对应的 perturbed posterior mean，从而给出 `s_{\mathrm{fake}, t}`。

### 13.2 更新生成器

重新采样 `z, t, \epsilon`，令

$$ x = G_\theta(z), \qquad x_t = \alpha_t x + \sigma_t \epsilon. $$

计算

$$ \mu_{\mathrm{base}}(x_t, t), \qquad \mu_{\mathrm{fake}}(x_t, t), $$

从而构造 distribution matching 梯度方向

$$ \Delta x \propto \frac{\alpha_t^2}{\sigma_t^2} \left( \mu_{\mathrm{base}}(x_t, t) - \mu_{\mathrm{fake}}(x_t, t) \right). $$

再把它链式反传给 `\theta`，同时叠加 regression 项梯度。

## 14. 第十三步：把 DMD 的逻辑链条压缩成一句完整的话

到这里，DMD 的完整推导链已经闭合了。把它压缩成一句最本质的话就是：

> 我们想最小化一步生成器分布和真实分布之间的 reverse KL；这个 KL 的一阶变分给出了样本空间中的最优下降方向 `s_{\mathrm{real}} - s_{\mathrm{fake}}`；由于原始空间中的 score 难以处理，我们转而在高斯加噪后的 perturbed distribution 上做同样的事情；而 perturbed score 又可以通过 diffusion denoiser 预测的后验均值来表达，于是最终得到了 DMD 的训练规则。

如果把这条链拆开，就是：

1. 生成器诱导分布 `p_\theta(x)`。
2. 目标是最小化 `D_{\mathrm{KL}}(p_\theta \| p_{\mathrm{real}})`。
3. 这个目标的变分梯度对应样本更新方向 `s_{\mathrm{real}} - s_{\mathrm{fake}}`。
4. 在原始空间难做，所以转到加噪空间。
5. 加噪空间里的目标变成 `D_{\mathrm{KL}}(p_{\mathrm{fake}, t}\|p_{\mathrm{real}, t})`。
6. 它的梯度方向是 `s_{\mathrm{real}, t} - s_{\mathrm{fake}, t}`。
7. score identity 说明 perturbed score 等于高斯条件 score 的后验平均。
8. `L_2` 去噪器学到的是后验均值。
9. 所以 denoiser 可以给出 perturbed score。
10. base denoiser 给 real score，online fake denoiser 给 fake score。
11. 两者相减并回传给生成器，就得到了 DMD。
12. 额外回归项负责增强训练稳定性。

## 15. 第十四步：几个最容易混淆的点

### 15.1 DMD 不是在直接最小化像素级蒸馏误差

如果只有 regression，那么 student 学到的更多是 teacher 的点对点输出。DMD 真正新增的部分是一个**分布级**的修正方向，它关心的是整族样本怎么移动，才能让 fake distribution 更接近 real distribution。

### 15.2 DMD 不是只靠 teacher score

只有 teacher score 更像“吸引力”，而没有 fake score 的“排斥力”。真正对应 reverse KL 梯度的是二者之差，而不是单边 teacher guidance。

### 15.3 denoiser 不是在胡乱近似 score

`x_0`-prediction 形式的 denoiser 在 `L_2` 训练下对应后验均值，而高斯加噪核保证了 score 可以由这个后验均值显式写出。所以这里不是拍脑袋近似，而是一条标准的 score-based generative modeling 恒等式。

### 15.4 原始 DMD 理论是漂亮的，但实现并不轻松

真正困难的地方在于：`s_{\mathrm{fake}, t}` 是一个随着生成器不断变化的对象。也就是说 fake score model 必须持续跟踪一个移动目标。这也是后续 DMD2 要重点修补的地方之一。

## 16. 哪些地方是严格等式，哪些地方已经是近似

为了避免把整条推导链误读成“每一步都严格相等”，这里把 DMD 里的关键层次分开总结一下。

### 16.1 严格成立的部分

在常规光滑性与边界条件下，下面这些步骤是理论上的严格结论：

1. 从 reverse KL 的一阶变分推出样本空间中的方向 `s_{\mathrm{real}} - s_{\mathrm{fake}}`。
2. 对 perturbed distribution 重复同样的变分推导，得到 `s_{\mathrm{real}, t} - s_{\mathrm{fake}, t}`。
3. score identity：

$$ \nabla_{x_t}\log p_t(x_t) = \mathbb E_{p(x|x_t)} \left[ \nabla_{x_t}\log q_t(x_t|x) \right]. $$

4. 对高斯加噪核得到

$$ s_t(x_t) = -\frac{x_t - \alpha_t \mathbb E[x|x_t]}{\sigma_t^2}. $$

5. `L_2` 回归器的 Bayes 最优解是后验均值 `\mathbb E[x|x_t]`。

### 16.2 进入近似的部分

真正把理论落到可训练系统时，下面这些地方已经不再是严格等式，而是近似：

1. `\mu_{\mathrm{base}}(x_t, t)` 不是精确后验均值，而是 teacher/base denoiser 对它的近似。
2. `\mu_{\mathrm{fake}}(x_t, t)` 也不是精确后验均值，而是 online fake denoiser 对当前生成分布的近似。
3. fake score model 的训练通常滞后于生成器，所以它是在追一个持续变化的目标。
4. 时间采样分布 `\rho(t)`、权重 `\lambda(t)`、网络容量、优化器、batch size 都会改变最终梯度的实际效果。
5. one-step 生成器本身表达能力有限，即便拿到理论正确的方向，也不一定能精确实现对应的分布流动。

### 16.3 为什么这些近似仍然有意义

虽然实现层面存在近似，但理论主线并没有变：

1. 方向上，我们始终在追 `real score - fake score`。
2. 表达上，我们始终通过 perturbed denoiser 去近似这个方向。
3. 训练上，我们始终在做“让 fake distribution 朝 real distribution 靠拢”的分布级更新，而不是单纯做样本级模仿。

所以 DMD 最重要的价值并不是“每一个符号都能精确等式落地”，而是它给出了一条结构非常清楚的近似路线图。

## 17. 结语

DMD 之所以会让人第一次读起来很“跳”，不是因为它真的用了什么神秘黑魔法，而是因为论文默认读者已经把下面几块背景吃透了：

1. KL 在 Wasserstein 或 transport 视角下的一阶变分。
2. perturbed distribution 的 score identity。
3. `L_2` denoiser 与后验均值之间的等价关系。
4. reverse KL 梯度为什么天然对应 `real score - fake score`。

一旦把这些中间桥梁补起来，DMD 的整体结构其实非常统一：

它不是一堆技巧的堆砌，而是

$$ \text{reverse-KL gradient flow} + \text{perturbed score estimation} + \text{online fake score fitting} + \text{regression stabilization}. $$

这也是我自己理解 DMD 后最大的感受：它最精华的地方不在于“把扩散蒸馏成一步”，而在于它把“一步生成器如何做真正的分布匹配”这个问题写成了一条相当干净的数学闭环。

## 参考资料

1. Yin et al., *One-step Diffusion with Distribution Matching Distillation*, CVPR 2024.
2. Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, ICLR 2021.
3. Wang et al., *ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation*, NeurIPS 2023.
