---
title: Causal Mask
date: "2026-3-18"
author: Wander
authorImage: /images/avatar.jpg
category: design
tags:
- Details
description: 什么是causal mask
---

长视频生成里说的 causal mask，本质上就是在注意力矩阵上加一个“可见性约束”：
当前位置的 token 只能看见“过去和当前”，不能看“未来”。
这样模型才能按时间顺序自回归地生成视频，而不是在训练/推理时偷看后面的帧。



———

## 1. causal mask 到底是什么

在标准 self-attention 里，注意力分数是：

A = Q K^T / sqrt(d) (Q行K列)

如果不加 mask，第 t 个位置 可以和所有位置做注意力，包括未来位置 t+1, t+2...。

而 causal mask 会构造一个上三角屏蔽矩阵 M：

M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i

然后：

A_masked = A + M
P = softmax(A_masked)
O = P V

因为未来位置对应的是 -inf，softmax 后概率变成 0，于是当前位置无法关注未来。

———

## 2. 为什么长视频生成特别依赖 causal mask

长视频生成和短视频最大区别不是“帧数更多”这么简单，而是：

1. 时间长度很长，token 数暴涨
2. 生成通常要保证严格时序一致性
3. 推理时往往是逐步生成，不能偷看未来
4. 训练和推理要尽量一致

所以 causal mask 有两个核心作用：

1. 保证自回归因果性
  生成第 t 帧时只能依赖 <= t 的内容
2. 控制计算结构
  在长序列上，mask 往往和分块、滑窗、memory token 一起用，变成“局部因果”或者“块因果”

———

## 3. 视频 token 化之后，序列到底长什么样

先从最简单的情况讲。

假设一个视频输入是：

x: [B, T, C, H, W]

- B: batch
- T: 帧数
- C: 通道
- H, W: 分辨率

视频模型通常不会直接对像素做 attention，而是先变成 latent，再切成 patch/token。

例如经过 VAE / tokenizer 后变成：

z: [B, T, C_lat, H_lat, W_lat]

再把每一帧的空间位置展平：

z_flat: [B, T, S, D]

- S = H_lat * W_lat，表示每帧多少个空间 token
- D 是 token embedding dim

再进一步把时间和空间展平成一个长序列：

z_seq: [B, L, D]
L = T * S

这时 attention 就是对长度 L 的序列做的。

———

## 4. 最基础的 causal mask 在视频里怎么定义

如果把视频 token 顺序排成：

(frame1_patch1), (frame1_patch2), ..., (frame1_patchS),
(frame2_patch1), ..., (frameT_patchS)

那么最严格的 causal mask 是按这个一维序列直接做下三角：

mask: [L, L]

其中：

mask[i, j] = 0       if j <= i
mask[i, j] = -inf    if j > i

这意味着：

- 当前 token 可以看见所有过去 token
- 包括过去帧，也包括当前帧里排在它前面的 patch
- 不能看当前序列顺序之后的 token

这种做法最简单，但有个问题：
它把视频当成纯一维序列，空间和时间的结构混在一起了。同一帧内的token也有可能看不到

———

## 5. 长视频里更常见的是“分解式 attention”

视频 token 太多，直接对 L = T*S 做全因果 attention，复杂度是：

O(L^2) = O((T*S)^2)

对长视频非常贵。

所以很多长视频模型不会直接用一个巨大的全序列 causal mask，而会做分解，比如：

1. 空间 attention
2. 时间 attention
3. 块级 attention
4. sliding window causal attention
5. memory / cache attention

———

## 6. 空间 attention + 时间 attention 时，mask 怎么起作用

这是视频 Transformer 最常见的思路之一。

### 6.1 空间 attention

先对每一帧内部做 attention，不跨帧：

输入形状：

x: [B, T, S, D]

把 B 和 T 合并：

x_spatial: [B*T, S, D]

然后做每帧内的 self-attention：

Q, K, V: [B*T, S, d]
attn score: [B*T, S, S]
output: [B*T, S, d]

这里通常 不需要 causal mask，因为同一帧内空间 token 一般是同时可见的。

所以空间层里通常是：

- full bidirectional spatial attention

———

### 6.2 时间 attention

然后固定空间位置，对时间维做 attention。

把张量重排成：

x_temporal: [B, S, T, D]

再合并 B 和 S：

x_temporal: [B*S, T, D]

这时每个空间位置有一条长度为 T 的时间序列。

做 temporal attention 时：

Q, K, V: [B*S, T, d]
score: [B*S, T, T]
mask: [T, T]

这里 causal mask 就非常自然：

mask[t1, t2] = 0      if t2 <= t1
mask[t1, t2] = -inf   if t2 > t1

意思是：
第 t1 帧的某个空间位置，只能看同一空间位置在过去帧和当前帧的信息，不能看未来帧。

这是视频时序建模里最标准的 causal 用法。

但是，如果你希望 T2 时刻的 A 位置去看 T1 时刻的 B 位置，这种“纯按固定 patch 位置做 temporal attention”的设计本身是不够
  的。

  这也是为什么实际视频模型通常不会只靠这一层，而是会配合别的交互机制。常见有几种：

  1. 空间 attention + 时间 attention 交替堆叠
     在空间层里，同一帧内 A 可以看 B；在时间层里，B@T1 又能传到 B@T2；多层叠起来以后，信息可以间接从 B@T1 -> A@T1 ->
     A@T2 或类似路径传播。
     所以虽然单层 temporal attention 不能直接跨空间位置看过去帧，但多层后可以“绕过去”。
     这时 mask 不是只在 T x T 上做，而是在所有视频 token 上定义“哪些过去 token 可见”。
  3. 局部时空窗口 attention
     例如 T2 的 A 只看过去几帧里附近的一片空间区域，而不是只看同坐标位置。
     这比全时空 attention 便宜很多，也更符合运动局部性。
  4. 引入 memory / latent tokens / deformable attention
     让当前位置不必死盯着同一个 patch index，而是通过少量中间 token 或可学习采样位置去找真正相关的过去区域。
———

## 7. 各种张量是怎么交互的

下面把 attention 的张量流完整展开一次。

假设当前做 temporal attention，输入：

x: [B, S, T, D]

先 reshape：

x -> [B*S, T, D]

然后线性映射：

Q = x Wq   -> [B*S, T, H, Dh]
K = x Wk   -> [B*S, T, H, Dh]
V = x Wv   -> [B*S, T, H, Dh]

通常再转成：

Q, K, V -> [B*S, H, T, Dh]

这样每个 head 单独算：

score = Q @ K^T
score: [B*S, H, T, T]

这里的第 4 维和第 3 维分别对应：

- query 的时间位置
- key 的时间位置

加入 causal mask：

mask: [1, 1, T, T]   或 [T, T]
score = score + mask

广播后仍是：

score: [B*S, H, T, T]

softmax：

prob = softmax(score, dim=-1)
prob: [B*S, H, T, T]

再乘 V：

out = prob @ V
out: [B*S, H, T, Dh]

拼回 heads：

out -> [B*S, T, H*Dh]

再 reshape 回视频结构：

out -> [B, S, T, D]

如果需要，再转回：

[B, T, S, D]

这就是 causal mask 在张量层面最标准的交互过程。

———

## 8. 长视频里真正难的地方：mask 不只是一个下三角

在长视频里，真正常见的是“变种 causal mask”。

因为如果 T 很大，[T, T] 或 [L, L] 还是太贵，所以会做稀疏化。

———

## 9. sliding window causal mask

比如只允许当前帧看最近 W 帧：

mask[i, j] = 0       if i-W < j <= i
mask[i, j] = -inf    otherwise

这叫局部因果窗口。

效果：

- 保留因果性
- 只看最近历史
- 复杂度从 O(T^2) 降到 O(T*W)

张量上仍然是：

score: [B*S, H, T, T]

但 mask 大部分位置都是 -inf。

如果底层实现更高效，甚至不会显式构造整个 [T, T]，而是只计算窗口内的 block。

———

## 10. block causal mask

长视频常按 chunk 切分，例如每 8 帧一块。

假设：

T = N_blocks * Bf

- Bf: 每个时间块的帧数

则时间序列按 block 切开，只允许：

- 当前块看当前块内部
- 当前块看所有过去块
- 不看未来块

这就是 block causal。

mask 结构像块下三角矩阵：

[block1  0      0    ]
[block2  block2 0    ]
[block3  block3 block3]

块内可以是：

- 全连接
- 也可以再叠一个局部 causal

这种方法特别适合长视频，因为它和 KV cache、chunked decoding 很配。

———

## 11. prefix / condition + causal generation

很多视频生成不是纯 unconditional，而是有条件输入：

- 文本 token
- 图像首帧
- 参考视频片段
- 音频 token

这时 mask 往往不是纯下三角，而是 prefix-LM mask。

例如序列组织成：

[text tokens] [condition frame tokens] [video tokens to generate]

mask 规则可能是：

1. 文本 token 彼此全可见
2. 所有生成 token 都能看见全部文本 token
3. 生成 token 之间仍然 causal

于是 mask 会分块：

        text   cond   gen
text      full   full   maybe no
cond      full   full   maybe no
gen       full   full   causal

这里最关键的是：
causal 只约束生成段内部，不一定约束条件段。

这在文生视频、图生视频里非常常见。

———

## 12. 训练时和推理时 causal mask 的角色差别

### 训练时

通常是 teacher forcing：

- 把整段视频 token 一次性送进去
- 但通过 causal mask 保证每个位置只能看历史
- 所以虽然并行算了所有位置，语义上还是自回归

这就是 Transformer 训练快的关键。

张量形状例如：

input tokens: [B, L, D]
score: [B, H, L, L]
mask: [1, 1, L, L]

———

### 推理时

逐步生成：

第 1 步生成 x1
第 2 步生成 x2
...
第 t 步生成 xt

如果每步都重新算整个序列 attention，会很慢，所以通常用 KV cache。

这时张量变成：

Q_t: [B, H, 1, Dh]
K_cache: [B, H, t, Dh]
V_cache: [B, H, t, Dh]
score_t: [B, H, 1, t]

注意这时其实“因果性”已经隐含在 cache 里了：
因为 cache 里只有过去，没有未来，所以即使不显式构造完整下三角 mask，也不会看见未来。

这点很重要：
训练时常显式用 causal mask；推理时常通过增量式 cache 隐式实现 causal。

———

## 13. 为什么视频里的 causal mask 比文本更复杂

文本是纯一维序列，位置就是 token index。

视频至少有三种结构：

1. 时间 T
2. 空间 H, W 或 patch index S
3. 条件模态，如 text/audio/reference frames

所以“未来”到底怎么定义，并不唯一。

常见几种定义：

1. 纯序列因果
  flatten 后按 token 顺序下三角
2. 仅时间因果
  同一帧空间内全可见，但跨帧只能看过去
3. 块因果
  以若干帧为单位做下三角
4. 条件前缀 + 生成因果
  条件 token 全可见，生成段内部 causal

因此视频里的 causal mask 实际上是“结构化可见性设计”，不是简单一张三角矩阵。

———

## 14. 一个具体例子：[B,T,S,D] 的长视频 Transformer

假设：

B = 2
T = 16
S = 64
D = 512
H = 8 heads
Dh = 64

输入：

x: [2, 16, 64, 512]

### 第一步：空间 attention

reshape:

x_sp = [2*16, 64, 512] = [32, 64, 512]

投影后：

Q,K,V: [32, 8, 64, 64]
score: [32, 8, 64, 64]

这里一般无 causal mask。

输出再还原：

out_sp: [2, 16, 64, 512]

### 第二步：时间 attention

转置：

x_tm = [2, 64, 16, 512]

reshape:

x_tm = [2*64, 16, 512] = [128, 16, 512]

投影：

Q,K,V: [128, 8, 16, 64]
score: [128, 8, 16, 16]
mask:  [1, 1, 16, 16]

加 mask 后 softmax：

prob: [128, 8, 16, 16]

乘 V：

out: [128, 8, 16, 64]

合并 heads：

out: [128, 16, 512]

还原：

out: [2, 64, 16, 512]

再转回：

[2, 16, 64, 512]

这里 causal mask 只作用在时间维的 16 x 16 上。

———

## 15. 常见误解

### 误解 1：causal mask 就一定是下三角

不一定。

在视频里很多是：

- 块下三角
- 窗口下三角
- prefix + causal 混合 mask
- 不同模态不同可见性

———

### 误解 2：用了 causal mask 就一定能做长视频

也不对。

causal 只解决“不能看未来”的约束，不解决：

- 上下文太长导致算不动
- 长时依赖容易遗忘
- 多镜头切换和身份一致性
- 训练不稳定

所以长视频模型还会配合：

- 稀疏 attention
- memory tokens
- chunked generation
- recurrence / state-space / linear attention
- KV cache 压缩

———

### 误解 3：扩散视频模型就完全不用 causal mask

不准确。

传统 diffusion 不是严格自回归，所以不一定需要标准 causal mask。
但很多长视频 diffusion / AR diffusion / masked video transformer 仍会引入“时间因果约束”或“受限可见性”来稳定长时生成。
所以关键不是模型名字，而是它是不是按某种顺序生成，以及是否禁止未来信息泄露。

———

## 16. 一句话理解“它在做什么”

如果把视频生成想成“边拍边写剧本”，那 causal mask 就是：

- 你写第 20 帧时
- 可以翻前 19 帧的笔记
- 可以看当前帧已写好的局部内容
- 但不能偷看第 21 帧以后会发生什么

它是在注意力层面强行执行这个规则。

———

## 17. 你如果想真正吃透，建议抓住这 4 个层次

1. 数学层
  score + mask -> softmax -> future prob = 0
2. 序列层
  视频被编码成 token 序列后，mask 规定谁能看谁
3. 结构层
  视频不只是 1D，所以 causal 往往只作用在时间维，或以 block/window 形式出现
4. 工程层
  训练时常显式 mask；推理时常靠 KV cache 隐式实现