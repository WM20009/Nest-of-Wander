---
title: 基础知识缝缝补补
date: "2025-9-30"
author: Wander
authorImage: /images/avatar.jpg
category: design
tags:
- basic knowledge
description: some common knowledge
---
# 原始tranformer的工作流程
原始的transformer是用来做机器翻译的，具体是首先有一个包含两种语言的tokenizer,然后把待翻译语言序列X输入encoder。
encoder和decoder都是一系列的transform block堆叠而成。

对于decoder；

训练时，采用teacher forcing技术，输入答案序列Y[:-1],输出Y[1:],再用交叉熵作为损失函数。

**注意:** 在self-attention的时候使用下三角mask保证每个词只能看到其前面的词。cross-attention时没有这种机制。

测试时，采用自回归，从< sos >token开始，逐个词进行输出。

# 标签平滑
多分类任务中不使用one-hot编码的gt,而是给一些不可能的类也分配一点点概率。避免模型的置信度过高，导致容错率低、不易校准、不能建模不确定性等。