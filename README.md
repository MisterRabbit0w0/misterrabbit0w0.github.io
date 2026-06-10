# misterrabbit0w0.github.io

织星 / Astraea 的个人博客，基于 [Hexo](https://hexo.io/) + 自制 **Hoshino** 主题构建，部署在 GitHub Pages。

- 在线访问：<https://misterrabbit0w0.github.io>
- 主题目录：[`themes/hoshino/`](themes/hoshino/)

## 本地开发

```bash
npm install
npm run server    # http://localhost:4000
npm run build     # 生成 public/
npm run clean     # 清缓存
```

详细开发约定见 [CLAUDE.md](CLAUDE.md)。

## 主题特性

- 樱花粉单主色 + oklch 色板，Light / Dark / 跟随系统三种模式
- 文章头图为自制「星图」：以标题哈希为种子，构建时生成每篇独一无二的星座 SVG（front-matter 显式设置 `cover:` 时优先）
- 光标、花瓣、配色均为手绘 / 自制

## 致谢

- 字体：[**Maple Mono CN**](https://github.com/subframe7536/maple-font)（OFL 协议），中文子集由[中文网字计划](https://chinese-font.netlify.app/)的 cn-font-split 切片，经 jsDelivr CDN 分发
- 图标：部分来自 [Feather Icons](https://feathericons.com/)（内联 SVG）

## License

- **代码**（主题、脚本、配置）：[MIT](LICENSE)
- **文章内容**（`source/_posts/`）：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh)
