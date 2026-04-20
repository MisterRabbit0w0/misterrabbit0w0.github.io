---
title: 博客换装：Hoshino 主题上线
date: 2026-04-20 11:30:00
categories: [工具]
tags: [Hexo, CSS, oklch]
cover_emoji: 🌸
---

之前一直在用 Butterfly，功能是很全，但有点"什么都有一点"的感觉——想改什么都要在十几层配置里打转。

索性写了个自己的主题 **Hoshino**，从 UI 参考稿直接开搞。

## 设计取向

- **樱花粉 + 薰衣草紫 + 薄荷绿** 的三色系，基于 `oklch` 色彩空间
- M PLUS Rounded 1c 字体，圆润但不幼稚

## 页面清单

| 页面 | 布局特点 |
|------|----------|
| 首页 | Hero + stats + 最近文章 + 精选项目 + 活动时间线 |
| 文章 | 按年分组 + 即时搜索 + tag filter + 分页 |
| 单篇 | 目录侧栏 + 阅读进度条 + 上一篇/下一篇 |
| 项目 | Featured + 卡片网格 + 子项目归档 |
| 关于 | Hero + 技能卡 + 时间线 + 联系卡 |

## 技术栈

- Hexo + EJS 模板（原生 `<%- %>`，不用 Pug/Nunjucks）
- `shared.js` 负责花瓣、主题切换、Tweaks 面板，利用 `page.css` 统一页面
