---
title: VLA Models --探索开源与高效微调
date: "2025-10-6"
author: Wander
authorTwitter: FredKSchott
authorImage: /images/avatar.jpg
category: design
tags:
- VLA
- Flamingo
- open-source
description: some works dedicated for open-source and efficiently fine-tuning
---
<div style="width: 35%; margin: 0 auto;">
  <img src="https://img.freepik.com/free-vector/flamingo-cartoon-style-isolated-white-background_1308-65667.jpg?semt=ais_hybrid&w=740&q=80">
</div>

# Preface
这一系列工作都是由于之前的模型闭源且过于巨大，对于个人开发者很不友好，故开始利用开源的vlm做可以高效微调的vla，为社区做出贡献。
# RoboFlamingo
![roboflamingo](image-6.png)
## 一、Methodology
为了降低RT-2那种模型的微调难度，作者没有采用RT-2那种将action也作为token对齐到语言和视觉模态空间的办法（因为这需要大量数据的微调），而是显式地用了一个policy head接受上游vlm融合以后的历史特征以输出动作。

事实上这和RT-1很像，并且这个方法在cv其他领域如分类分割中广泛地使用，就是用一个基础大模型抽特征，然后加一个head适配下游任务。
## 二、Architecture
### Flamingo
既然模型用的vlm是[flamingo](https://arxiv.org/pdf/2204.14198)，那我们就来回顾一下。
![overview](image-7.png)
![ GATED XATTN-DENSE layers](image-8.png)
![resampler](image-9.png)
(感觉这三张图已经把架构展示得很清楚了)

### Policy Head
在此基础上，我们又设计了一个policy head，利用最后的视觉语言融合表征来输出动作。它解决了如下三个问题：
- 它将具有静态图像输入的视觉-语言模型适应为视频观测
- 它生成机器人控制信号而非仅文本输出
- 它需要有限的下游机器人操作数据，在拥有数十亿可训练参数的情况下实现高性能和泛化能力

policy head作者尝试了三种不同的架构：
- 纯MLP
- decoder-only transformer
- LSTM
最终的结果是LSTM最好
## 三、Loss
![loss](image-10.png)

## 四、Some Comments
- 实验部分的几个baseline都是没有用预训练的vlm，这表明RoboFlamingo确实继承了预训练vlm的一些好处。
- 感觉这个工作就是把动作生成和vlm解耦了，减少了需要训练的参数量，其他似乎没啥特别令人眼前一亮的地方
  
# Openvla
## 一、Methodology
### Open-source
利用Llama 2作为语言编码器DINOv2和SigLIP作视觉编码器
### Compute Effeciency
- low-rank adaptation method
- quantization
## 二、Review
### LoRA
![lora](image-11.png)
如图所示，lora的思想就是把原参数矩阵 $W$ 冻结，然后把微调时的更新矩阵 $\delta W$
