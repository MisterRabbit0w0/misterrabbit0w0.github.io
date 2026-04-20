# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hexo 静态博客，使用自制 **Hoshino** 主题（EJS 模板，樱花粉 + 花瓣 + 主题切换），部署在 GitHub Pages (misterrabbit0w0.github.io)。站点语言为中文，时区 Asia/Shanghai。作者署名：织星 / Astraea。

`themes/butterfly/` 是旧主题，保留在仓库里但未启用，可留作参考或后续清理。

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
- **`themes/hoshino/`** — 自制主题（当前启用）
  - **`_config.yml`** — 主题配置（brand、nav、hero、social、repos、projects、about、footer、effects）
  - **`layout/`** — EJS 模板：`layout.ejs`（骨架）、`index.ejs`（首页）、`archive.ejs`（文章列表，被 category/tag 复用）、`post.ejs`（单篇+TOC+阅读进度）、`page.ejs`（按 `page.layout` 分流 projects/about/default）
  - **`source/css/`** — `shared.css`（全局）+ `page.css`（各页面）
  - **`source/js/shared.js`** — 花瓣 spawner + 主题切换（pink/blue/gold/dark）
- **`themes/butterfly/`** — 旧主题（未启用，保留作参考）

## Deployment

推送到 `main` 分支时，GitHub Actions (`.github/workflows/build.yml`) 自动执行 `npm run clean && npm run build`，然后将 `public/` 目录部署到 `gh-pages` 分支。使用 Node 20。

## Key Configuration

- 永久链接格式：`:year/:month/:day/:title/`
- 每页文章数：10
- 语法高亮：highlight.js
- 文章版权协议：CC BY-NC-SA 4.0
- 模板引擎：Pug + Stylus
