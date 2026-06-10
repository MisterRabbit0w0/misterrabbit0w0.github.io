---
title: 博客换装：Hoshino 主题上线
date: 2026-05-31 23:30:00
updated: 2026-06-10 12:00:00
categories: [工具]
tags: [Hexo, CSS, oklch]
---

因为没搞懂怎么挂别人主题的依赖，索性自己写了一个，取名 **Hoshino**，从 UI 参考稿直接开搞。

自己造主题有个意外的好处：每个像素都知道是怎么来的，出了问题不用去翻别人仓库的 issue。

## 长什么样

- 米黄色打底，樱花粉做唯一主色。颜色全部用 oklch 写，亮色、暗色、跟随系统三种模式
- 全站字体是 [Maple Mono CN](https://github.com/subframe7536/maple-font)——中英 2:1 的等宽字体，我一直想用它。中文子集走中文网字计划的切片 CDN，按需加载，不至于太重
- 文章头图不放图片，改成「星图」：拿标题做哈希种子，构建时生成一张星座连线 SVG。同一篇文章永远是同一张，每篇文章互不相同
- 花瓣会飘，鼠标指针是自己画的：默认箭头带一片樱花瓣，可点击的地方会变成星星魔法棒，看图时是一只金色放大镜

## 一些实现碎片

- EJS 模板 + 两个 CSS 文件，没上任何框架
- 暗色模式用 `data-color-mode` 切换，颜色全走 CSS 变量，星图封面也跟着一起变色
- 设置了 `prefers-reduced-motion` 的话，花瓣和动画会全部停掉

源码就在[博客仓库](https://github.com/misterrabbit0w0/misterrabbit0w0.github.io)的 `themes/hoshino/` 里，想扒随便扒。
