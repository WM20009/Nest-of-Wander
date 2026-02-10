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
# 标签平滑
多分类任务中不使用one-hot编码的gt,而是给一些不可能的类也分配一点点概率。避免模型的置信度过高，导致容错率低、不易校准、不能建模不确定性等。
# convtranspose
[Understand Transposed Convolutions](https://medium.com/data-science/understand-transposed-convolutions-and-build-your-own-transposed-convolution-layer-from-scratch-4f5d97b2967)