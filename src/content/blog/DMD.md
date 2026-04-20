---
title: DMD (Distribution Matching Distillation)
date: "2026-2-24"
author: Wander
authorImage: /images/avatar.jpg
category: design
tags:
- Diffusion
- Distillation
description: DMD的一些原paper中省略的数学推导。
---


# 一步登天：深度解析 DMD 蒸馏背后的数学闭环与 Score 之谜

### 前言：消失的采样步骤
在扩散模型（Diffusion Models）的研发中，如何将上千步的采样压缩到一步（One-step Generation）一直是业界的“圣杯”。2024 年，MIT 与 Adobe 联合推出的 **DMD (Distribution Matching Distillation)** 给出了一份近乎完美的答卷。

但在阅读这篇论文时，很多人会被 Section 3.2 中那个简洁得近乎“简陋”的梯度更新公式困扰：
$$s_{real}(x_t, t) = - \frac{x_t - \alpha_t \mu_{base}(x_t, t)}{\sigma_t^2}$$
**质疑随之而来：** 边缘分布 $p(x_t)$ 明明是一个极其复杂的非高斯分布，为什么它的 Score 函数（梯度场）能写成这种高斯形式？直接用两个模型输出之差作为 KL 散度的梯度，难道不差一个常数吗？

今天，我们拨开迷雾，从底层概率逻辑推导 DMD 的数学闭环。

---



### 二、 核心推导：为什么 Marginal Score 具有“高斯形式”？

这是困扰大多数人的核心问题：**非高斯分布的梯度为什么像高斯？** 

#### 1. 边际分数等式（The Score Identity）
根据全概率公式 $p(x_t) = \int p(x_t|x_0) p(x_0) dx_0$，对两边求 $x_t$ 的梯度：
$$ \nabla_{x_t} p(x_t) = \int \nabla_{x_t} p(x_t|x_0) p(x_0) dx_0 $$
利用对数导数技巧 $\nabla f = f \nabla \log f$：
$$ \nabla_{x_t} p(x_t) = \int p(x_t|x_0) [\nabla_{x_t} \log p(x_t|x_0)] p(x_0) dx_0 $$
代入 Score 的定义 $\nabla \log p = \frac{\nabla p}{p}$，并利用贝叶斯公式 $\frac{p(x_t|x_0)p(x_0)}{p(x_t)} = p(x_0|x_t)$：
$$ \nabla_{x_t} \log p(x_t) = \mathbb{E}_{p(x_0|x_t)} \left[ \nabla_{x_t} \log p(x_t|x_0) \right] $$
**结论一：边缘分布的 Score，等于条件分布 Score 的后验期望。**

#### 2. 高斯核的引入
由于 Diffusion 的转移核是高斯的：$p(x_t|x_0) = \mathcal{N}(\alpha_t x_0, \sigma_t^2 \mathbf{I})$，其条件 Score 为：
$$ \nabla_{x_t} \log p(x_t|x_0) = -\frac{x_t - \alpha_t x_0}{\sigma_t^2} $$
将其代入上面的期望公式，由于 $x_t$ 和 $\sigma_t^2$ 在积分时是常数：
$$ \nabla_{x_t} \log p(x_t) = -\frac{x_t - \alpha_t \mathbb{E}[x_0|x_t]}{\sigma_t^2} $$
**这解释了为什么形式像高斯：因为它继承了加噪过程的高斯结构，但其核心由“均值”变成了“后验期望”。**

---

### 三、 神经网络的本能：预测后验期望

公式里的 $\mu_{base}(x_t, t)$ 是从哪来的？

在机器学习中，当我们使用 $L_2$ Loss 训练一个去噪器 $f_\theta$ 时：
$$ \min_\theta \mathbb{E} [ \| f_\theta(x_t) - x_0 \|^2 ] $$

我们要寻找一个函数 $\mu^*(x_t)$，使得上述期望损失最小。我们可以将期望展开：
$$\mathcal{L} = \int p(x_t) \left( \int p(x_0 | x_t) \| \mu(x_t) - x_0 \|^2 dx_0 \right) dx_t$$

为了让总积分为最小值，我们需要对于每一个具体的 $x_t$，都让括号里的项最小。令 $f(\mu) = \int p(x_0 | x_t) \| \mu - x_0 \|^2 dx_0$。
对 $\mu$ 求导并令其为 0：
$$\frac{df}{d\mu} = 2 \int p(x_0 | x_t) (\mu - x_0) dx_0 = 0$$
展开得：
$$2\mu \int p(x_0 | x_t) dx_0 - 2 \int x_0 p(x_0 | x_t) dx_0 = 0$$
因为概率密度函数的积分为 1（$\int p(x_0 | x_t) dx_0 = 1$），所以：
$$\mu^*(x_t) = \int x_0 p(x_0 | x_t) dx_0 = \mathbb{E}[x_0 | x_t]$$

根据变分法，该损失函数的理论最优解（Bayes Optimal Predictor）恰恰就是：
$$ f^*(x_t) = \mathbb{E}[x_0 | x_t] $$

**逻辑闭环了：**
1. 扩散模型通过 $L_2$ Loss 学到了 **后验期望**（预测 $x_0$）。
2. 后验期望代入高斯结构公式，得到了 **边际分布的精确 Score**。
3. DMD 利用这个 Score 来指导生成器进行 **分布匹配**。

---

### 四、 宏观视角：与 Flow Matching 的大一统

如果你研究过 **Flow Matching**，你会惊奇地发现两者背后的数学底色是完全一致的。
在 Flow Matching 中，边缘速度场 $v_t(x)$ 同样被定义为条件速度场 $v_t(x|x_0)$ 的后验期望：
$$ v_t(x) = \mathbb{E}_{p(x_0|x_t)} [v_t(x|x_0)] $$

无论是 Diffusion 里的“梯度场”还是 Flow 中的“速度场”，现代生成模型都在利用同一个真理：**我们虽然只能在训练时观测到单点的条件信息（$x_0$），但神经网络强大的拟合能力自动帮我们完成了对全体分布的积分（边缘信息）。**

---

### 五、 结语：DMD 究竟做了什么？

DMD 论文作者并没有在正文中大篇幅解释这些基础推导。他们默认了读者接受了以下事实：**去噪器的输出差异（$s_{fake} - s_{real}$）就是两个分布之间的“推力”梯度。**

*   **SDS (Score Distillation Sampling)** 曾试图用噪声 $\epsilon$ 来简化这个推力，但因为它没有训练 $s_{fake}$，导致了严重的模式崩塌。
*   **DMD** 通过引入一个专门训练的 **Fake Score Model**，严格遵循了 KL 散度的变分梯度推导，将“蒸馏”推向了数学意义上的精确。

**下一次，当你看到 DMD 那行简洁的公式时，请记住：那不是拍脑袋的近似，而是概率论中“边际”与“条件”之间最优雅的博弈。**

---
*本文由对 DMD 论文深度质疑的讨论整理而成，感谢每一位在公式细节处不妥协的探索者。*

---
这篇博客从读者的疑惑点出发，层层递进到数学推导，最后上升到学科统一的高度。你看是否符合你的要求？如有需要调整的地方请告诉我。


为了让你彻底理解 DMD2 的运作机制，我将根据论文原理和图示，将其训练流程总结为一个标准的伪代码算法，并详细解释每一个关键步骤。

DMD2 的核心是一个**交替优化的框架**：一边训练“考官”（判别器和假图评分函数），一边训练“学生”（生成器）。

---

### DMD2 训练算法流程

**输入：**
*   $G_\theta$：待训练的 Student 模型（参数为 $\theta$）。
*   $T$：预训练好的 Teacher 扩散模型（参数固定）。
*   $D_\phi$：带噪声判别器（参数为 $\phi$）。
*   $S_\psi$：假图评分函数（Fake Score Function，参数为 $\psi$，通常与 $D$ 共享部分参数）。
*   数据集 $X_{real}$，噪声分布 $p_z$。

---

#### **训练循环（直至收敛）：**

**第一阶段：训练“考官” (Update Discriminator & Fake Score)**
*目的是让考官能精准识别出 Student 目前画得哪里假，以及总结出假图的分布。*

1.  **采样**：
    *   从数据集抽样真实图像 $x \sim X_{real}$。
    *   采样随机噪声 $z \sim p_z$。
    *   采样随机时间步 $t \sim [0, T]$。
2.  **生成合成图（Synthetic Images）**：
    *   如果是多步生成器（例如 4 步）：让 Student $G_\theta$ 从 $z$ 开始跑，生成中间态或最终态的图像 $\hat{x}$。
3.  **加噪**：
    *   对真图加噪：$x_t = \text{ForwardDiffusion}(x, t)$
    *   对合成图加噪：$\hat{x}_t = \text{ForwardDiffusion}(\hat{x}, t)$
4.  **更新参数 $\phi, \psi$**：
    *   **优化 $D_\phi$**：通过二分类损失，让 $D$ 学会区分 $x_t$（真）和 $\hat{x}_t$（假）。
    *   **优化 $S_\psi$**：让 $S$ 学习预测 $\hat{x}_t$ 中的噪声，从而建立 Student 生成分布的“得分场”（Score Function）。

---

**第二阶段：训练“学生” (Update Student Generator)**
*目的是让 Student 同时满足 Teacher 的教导（红线）和骗过判别器（绿线）。*

1.  **采样**：采样新的噪声 $z \sim p_z$ 和随机时间步 $t \sim [0, T]$。
2.  **前向计算**：
    *   Student 生成图像：$\hat{x} = G_\theta(z)$（多步则为链式生成）。
    *   注入噪声：$\hat{x}_t = \text{ForwardDiffusion}(\hat{x}, t)$。
3.  **计算梯度（核心步骤）**：
    *   **红线梯度（分布匹配梯度）**：
        $$\nabla_{\hat{x}} \mathcal{L}_{distill} \propto \underbrace{S_{teacher}(\hat{x}_t, t)}_{\text{老师的意见}} - \underbrace{S_{fake}(\hat{x}_t, t)}_{\text{对学生现状的总结}}$$
        *这步告诉学生：你的画风和老师的画风在这个 $t$ 水平上差了多少。*
    *   **绿线梯度（GAN 梯度）**：
        $$\nabla_{\hat{x}} \mathcal{L}_{GAN} = \frac{\partial [-\log D_\phi(\hat{x}_t, t)]}{\partial \hat{x}}$$
        *这步告诉学生：判别器觉得你这里画得假，快改。*
4.  **反向传播更新 $\theta$**：
    *   将上述两个梯度合并，通过链式法则传回给 $G_\theta$：
        $$\theta \leftarrow \theta - \eta \cdot \frac{\partial \hat{x}}{\partial \theta} \left( \lambda_1 \nabla_{\hat{x}} \mathcal{L}_{distill} + \lambda_2 \nabla_{\hat{x}} \mathcal{L}_{GAN} \right)$$
    *   其中 $\frac{\partial \hat{x}}{\partial \theta}$ 包含了多步生成过程中的所有梯度累积。

---

### 关键点详细解析

#### 1. 为什么有“两个”考官？
*   **Teacher ($S_{teacher}$)**：它是一个“博学但固执”的老师。它拥有海量的知识，但它是通用的，不一定完全契合 Student 的精简结构。它的梯度（红线）负责引导 **整体分布的正确性**。
*   **Fake Score ($S_{fake}$)**：它是一个“专门研究该学生”的助教。它专门总结当前这个 Student 模型生成的图有哪些共性错误。红线梯度之所以用 `Teacher - Fake`，就是为了减去 Student 已经学到的部分，只学习还没对齐的那部分差异。

#### 2. GAN 在这里起什么作用？
GAN（绿线）是“**细节修补匠**”。
红线梯度在低噪声（小 $t$）区域往往会变得模糊。GAN 损失通过直接与真图 $x_{real}$ 挂钩，强行要求 Student 生成的图像在视觉感知上具有极高的锐度和逼真度，弥补了扩散模型蒸馏过程中常见的“模糊”现象。

#### 3. 为什么多步训练要用“合成图”？
如算法所示，在训练多步生成器时，我们让 Student **“自己跑几步”** 生成 $\hat{x}$，而不是从真图加噪得到 $x_t$。
*   **原因**：这样梯度就能穿过这几步生成过程。Student 学到的是：“如果我在第一步产生了一个微小的伪影，我该如何在接下来的几步中通过参数优化来消除它。”这彻底解决了**训练和推理不一致**（Training-Inference Mismatch）的问题。

### 算法总结图示

| 角色 | 操作对象 | 目标 | 梯度性质 |
| :--- | :--- | :--- | :--- |
| **Student** | 纯噪声 $z$ | 骗过判别器 + 模仿老师 | 接收红绿双线反馈 |
| **Teacher** | 带噪合成图 $\hat{x}_t$ | 提供理想的概率流场方向 | 静态指导 (Frozen) |
| **Discriminator** | 真图 $x_t$ vs 假图 $\hat{x}_t$ | 区分真伪细节 | 提供动态对抗梯度 |
| **Fake Score** | 假图 $\hat{x}_t$ | 建模学生当前的分布瑕疵 | 辅助红线做“减法” |

这个算法流程是否让你对 DMD2 的整个系统运作有了全局的把握？