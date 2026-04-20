---
title: 《小小算法书》中文版上线
date: 2026-04-20 10:00:00
categories: [译本]
tags: [算法, KaTeX, VitePress]
cover_emoji: 📘
---

花了几天时间，把《The Little Book of Algorithms》翻译成了中文版，部署在 [misterrabbit0w0.github.io/little-book-of-algorithms-zh](https://misterrabbit0w0.github.io/little-book-of-algorithms-zh/)

## 基本情况

- **11 章**，覆盖算法基础 → 字符串 → 图论 → 动态规划 → 机器学习 / 深度学习
- 中文用词尽量自然，保留原作的精炼节奏

## 几个小坑

翻译过程中踩到不少 markdown/LaTeX 边界 bug：

- 表格单元里的 `$|g|_2$` 会被 markdown-it 按 `|` 切列
- `$$...$$` 块里碰到 `_{subscript}` 会被解析成 `<em>`
- 行首的 `+` 会被吃成列表项，把 `$$...$$` 劈成两半
- setext heading：公式块里单独一行 `=` 会把上一行变 `<h1>`

全部在 [.vitepress/theme/index.ts](https://github.com/misterrabbit0w0/little-book-of-algorithms-zh/blob/main/.vitepress/theme/index.ts) 里加了一层客户端 pass-0 解包 + regex 放宽解决了。

## 去哪读

👉 [在线版](https://misterrabbit0w0.github.io/little-book-of-algorithms-zh/) — KaTeX 即时渲染，响应式布局
👉 [GitHub 源码](https://github.com/misterrabbit0w0/little-book-of-algorithms-zh)

issues and prs are welcomed.
