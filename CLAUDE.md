# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hexo 静态博客，使用 Butterfly v5.5.3 主题，部署在 GitHub Pages (misterrabbit0w0.github.io)。站点语言为中文，时区 Asia/Shanghai。

## Common Commands

```bash
npm run server      # 启动本地开发服务器 (hexo server)
npm run build       # 生成静态文件 (hexo generate)
npm run clean       # 清理构建产物 (hexo clean)
npm run deploy      # 部署 (hexo deploy)
hexo new "标题"     # 创建新文章
hexo new page "名称" # 创建新页面
hexo new draft "标题" # 创建草稿
```

## Architecture

- **`_config.yml`** — Hexo 主配置（站点信息、永久链接格式、分页等）
- **`source/_posts/`** — 博客文章（Markdown + YAML front matter）
- **`scaffolds/`** — 新建内容的模板（post、page、draft）
- **`themes/butterfly/`** — 主题目录（直接包含在仓库中，非 submodule）
  - **`_config.butterfly.yml`** — 主题配置（1100+ 行，控制导航、侧边栏、评论、CDN 等）
  - **`layout/`** — Pug 模板，入口文件为 `index.pug`/`post.pug`/`page.pug`，`includes/` 下按功能分区（header、post、widget、third-party 等）
  - **`scripts/`** — 主题 JS 脚本，分为 common、events、filters、helpers、tag 五类
  - **`source/`** — 主题静态资源（CSS/JS/图片）
  - **`plugins.yml`** — CDN 插件定义（200+ 条目）

## Deployment

推送到 `main` 分支时，GitHub Actions (`.github/workflows/build.yml`) 自动执行 `npm run clean && npm run build`，然后将 `public/` 目录部署到 `gh-pages` 分支。使用 Node 20。

## Key Configuration

- 永久链接格式：`:year/:month/:day/:title/`
- 每页文章数：10
- 语法高亮：highlight.js
- 文章版权协议：CC BY-NC-SA 4.0
- 模板引擎：Pug + Stylus
