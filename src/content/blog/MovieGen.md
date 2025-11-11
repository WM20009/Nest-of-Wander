---
title: MovieGen
date: "2025-10-22"
author: Wander
authorImage: /images/avatar.jpg
category: design
tags:
- video generation
description: moviegen 论文阅读，一些记录
---
## [video](https://www.bilibili.com/video/BV1U65SzuE4p?spm_id_from=333.788.videopod.sections&vd_source=09fde7223b2a977074582eca951dfd39)
## Notes
- TAE：一般就是3D Conv或者1D+2D，以及再加一些temperal self-attention.
- pachify：在H,W,T上同时下采样立方体。
- text-encoder：同时用好几个encoder再concate,这是因为clip型的适合用于提取globally semantic feature,然后还需要一些文字级别的featrue(可以用ByT5)