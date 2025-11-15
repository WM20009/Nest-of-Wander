---
title: video generation paper reading
date: "2025-11-14"
author: Wander
authorImage: /images/avatar.jpg
category: design
tags:
- video generation
description: 论文阅读，一些记录
---
# MovieGen&Hunyuan Video
两者结构差不多，故放在一起
![moviegen](/images/blog/paper-reading/image.png)
![hunyuan](/images/blog/paper-reading/image1.png)
## [video](https://www.bilibili.com/video/BV1U65SzuE4p?spm_id_from=333.788.videopod.sections&vd_source=09fde7223b2a977074582eca951dfd39)
## Pipline
以下为ai总结：

这是一个 **基于流匹配（Flow Matching）的 DiT 扩散模型**，采用 **3D-VAE 压缩 + Transformer 去噪 + 速度预测** 的范式，并支持**文本驱动**和**图像条件**两种生成模式。

#### Step 1: 视频压缩（3D-VAE 编码）
- **输入**：真实视频 $u$，尺寸为 $(T+1) × H × W$（T+1 帧）
- **编码器**：3D 变分自编码器（同时压缩时空维度）
- **输出**：潜码 $z₀$，形状为  **$(T+1) × C × H/cₛ × W/cₛ$**  ，其中：
  - $C$：潜码通道数
  - $cₛ$：空间压缩率（如 8×）
  - $cₜ$：时间压缩率（隐含在 T+1→T+1 中，通常也是 4× 或 8×）
  - **本质**：将像素空间视频压缩到低维潜空间，大幅减少计算量

#### Step 2: 条件信息准备
- **文本条件**：提示词 $y$ 被编码为文本潜码（通常用 T5 或 CLIP 文本编码器）
- **图像条件（可选）**：若做图生视频，参考图像经 3D-VAE 编码后，**在通道维度上**与 $zₜ$ 拼接（channel-wise concatenation）

#### Step 3: 前向加噪声（扩散过程）
- 对潜码 $z₀$ 按流匹配（Flow Matching）或 RFM（Rectified Flow）方式加噪，得到**任意时刻 t 的噪声潜码** $zₜ$
- **关键**：这里不是传统的 DDPM 加噪，而是在潜空间中进行**插值**：
  - $zₜ = (1-t)·z₀ + t·ε$，其中 ε～N(0, I)
  - **速度真值**：$uₜ = dzₜ/dt = ε - z₀$（这是训练目标）

#### Step 4: 双流到单流的 DiT 去噪
这是核心创新点，参考了 SD3 的多模态融合策略：

**第一阶段：双流处理（Dual-stream）**
- 视频潜码 $zₜ$ → 拆成时空 Patch → **视频流 Transformer Block** $f_v$
- 文本潜码 → **文本流 Transformer Block** $f_t$
- 两个流**独立处理**，各自提取特征（避免文本被视觉信息淹没）

**第二阶段：单流融合（Single-stream）**
- 将双流输出在序列维度**拼接**（concatenate）
- 送入**单流 Transformer Block** $f_s$，让文本和视觉信息**充分交互**
- **目的**：既保留模态特异性，又实现深度耦合

#### Step 5: 速度预测与损失
- **网络输出**：预测的速度 $ûₜ$
- **监督信号**：真实速度 $uₜ = dzₜ/dt$
- **损失函数**：最小化均方误差（MSE）
  $
  loss = E[‖uₜ - ûₜ‖²]
  $
- **优势**：相比预测噪声，速度预测更稳定，采样步数更少

### 三、推理流程（生成视频）

#### Step 1: 初始化
- 从高斯噪声 $z_T$ 开始（T=1）
- 准备文本条件（和可选的图像条件）

#### Step 2: ODE 求解
- 使用 **一阶欧拉法** 迭代求解概率流 ODE：
  $
  z_{t-Δt} = zₜ - ûₜ·Δt
  $
- 重复调用 DiT 网络预测速度，逐步去噪
- **步数**：通常 10-50 步即可（比 DDPM 快很多）

#### Step 3: 解码
- 得到干净的潜码 $z₀$ 后，用 **3D-VAE 解码器** 重建为像素视频 $u$

### 四、关键技术总结

| 模块 | 技术选型 | 作用 |
|------|----------|------|
| **压缩** | 3D-VAE | 时空联合压缩，比 2D 更适配视频 |
| **架构** | DiT (Diffusion Transformer) | 全局注意力，擅长长时序建模 |
| **融合** | Dual → Single Stream | 平衡模态独立性与交互性 |
| **条件** | Channel-wise Concat | 简洁有效的图像条件注入 |
| **目标** | Velocity Prediction | 流匹配，训练稳定、采样快 |
| **求解** | Euler ODE Solver | 一阶常微分方程，简单高效 |

### 五、典型应用场景
1. **文生视频**：只给文本 $y$，生成 $z₀$
2. **图生视频**：给首帧图像，通道拼接后生成后续帧
3. **视频编辑**：对潜码进行插值或局部修改

---

**一句话总结**：该模型用 3D-VAE 压缩视频，用双流-单流 DiT 融合文本与视觉信息，通过预测速度的方式训练，最后用 ODE 求解器快速生成高质量视频，兼顾了效率与效果。
## Notes
- TAE：一般就是3D Conv或者1D+2D，以及再加一些temperal self-attention.对H,W,T同时下采样8倍。
- pachify：在H,W,T上同时下采样立方体。
- text-encoder：同时用好几个encoder再concate,这是因为clip型的适合用于提取globally semantic feature,然后还需要一些文字级别的featrue(可以用ByT5)
- 直接在隐空间中进行flow matching
- 并非自回归结构，依然是同时对所有帧加噪去噪
- 在推理时进行tiling，也就是把整个视频分块生成，块之间有一些overlapping进行线性融合避免伪影。


从这两个工作来看和world model 之间的gap还非常明显，显然没有real-time interaction，用户的自由度还很低，生成的视频也很短，还没使用自回归架构，物理不一致也很多。

下一篇工作是腾讯的Voyager，能够根据图片输入和用户指定的相机轨迹来生成视频，并且进行3D重建，用户可控性更强了一些。

# [Voyager](https://arxiv.org/pdf/2506.04225v1)
![voyager](/images/blog/paper-reading/image2.png)
这篇主要是在做Camera-Controllable View Generation和 Long-Range Video Generation

要实现更精确的控制就必然要加入更explicit的约束，这里主要是进行了对输入image进行点云的构建，后面根据不同的相机参数渲染rgb图和depth图进行显式的几何引导，并且通过每一帧的生成来实时更新点云。

并且对于远距离探索，这里采用了通过点云来进行世界缓存的机制，

“具体而言，我们以逐帧方式向缓存增量添加新点：给定先前帧累积的点云 $\hat{p}$，我们从当前相机视角 $c_i$ 渲染可见性掩码 $𝑀 = render(\hat{p},c_i)$。位于不可见区域的点优先加入$\hat{p}$。对于可见区域，若现有点的表面法向量与当前视角方向夹角超过 90 度，新点同样更新至缓存，因为这些现有点无法在当前视点被观测到。该策略使存储点数减少约 40%，并避免了多帧聚合导致的噪声累积。”

感觉要实现无限的扩展必须要保证所储存的“外部记忆”总量要在一个限度之内，我们也许可以只维护所见范围内的一小部分点云，再根据移动实时生成，但是这样又会造成很多重复的计算。个人感觉这不算太高明的方法。

最后看来效果还是不错的，在各项指标上都比较靠前。

基于Voyager，腾讯又推出了hunyuanworld,这似乎就是他们所谓的世界模型了

# [Huanyuan World](https://arxiv.org/pdf/2507.21809v1)
这篇工作主要就是基于3D的世界生成，并且用了一小部分的视频生成（就是在长距离探索方面使用的voyager的机制）

