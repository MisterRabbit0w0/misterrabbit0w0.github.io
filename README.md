# misterrabbit0w0.github.io

织星 / Astraea 的个人博客，基于 [Hexo](https://hexo.io/) + 自制 **Spica** 主题构建，部署在 GitHub Pages。

- 在线访问：<https://misterrabbit0w0.github.io>
- 主题目录：[`themes/spica/`](themes/spica/)

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

## 文章组件

写作时可用的自定义 tag plugin（源码 [`themes/spica/scripts/post-tags.js`](themes/spica/scripts/post-tags.js)，样式在 `themes/spica/source/css/page.css`「文章组件」区块）。

> 示例里的角色与链接出自《ATRI -My Dear Moments-》。

### 剧透条 `spoiler`

行内毛玻璃模糊条，悬停 / 点按显形，未显形时不可选中：

```markdown
结局里{% spoiler %}他被卡车撞了{% endspoiler %}，非常温柔。
```

### 台词 `dialogue`

粉色角色签 + 台词，正文支持 markdown，多段连用自然堆叠：

```markdown
{% dialogue 亚托莉 %}
他说自己必须拯救地球，于是我问道——地球也包括我吗？
{% enddialogue %}
```

### 评分 `rating`

spica 四角闪星评分，0–5 分、0.5 步进，带无障碍标注：

```markdown
{% rating 4.5 %}
```

### 链接卡 `linkcard`

外链书签卡。以 `|` 分段；标题、描述可省略（省略标题时取域名）：

```markdown
{% linkcard https://atri-mdm.com/ | ATRI -My Dear Moments- | 时间流逝吧，你是多么残酷 %}
{% linkcard https://hexo.io/zh-cn/ %}
```

### 图组 `gallery`

2 或 3 列图片网格，可选共同说明；移动端自动降列，图片自动接灯箱放大：

```markdown
{% gallery 2 两张并排的头像 %}
![头像](a.jpg)
![头像二号](b.jpg)
{% endgallery %}
```

### 原生元素（纯 CSS，无需 plugin）

- 折叠块：`<details><summary>标题</summary>内容</details>`，chevron 旋转示意开合
- 按键帽：`<kbd>Ctrl</kbd> + <kbd>C</kbd>`
- 引用：`> 引文` 或 `{% blockquote @作者 https://链接 《作品》 %}`——大引号排版，署名行自动小字右对齐

## 动效维护

Spica 的交互动效采用苹果风格的小幅弹簧回弹，保留原有配色与背景主题，不依赖动画库。

- `themes/spica/source/css/shared.css` 中的 `--motion-spring` 是阻尼比 0.78、角频率 18 的弹簧采样曲线，600ms 归位。不支持 CSS `linear()` 的浏览器使用贝塞尔回退。
- 按钮按下用 120ms，松开、卡片反馈、目录指示条、折叠箭头和面板位移使用弹簧。入场使用独立 `translate`，不覆盖按钮的 `transform`。
- 颜色、透明度、主题揭幕和图表描线保持单向缓动。阅读进度直接跟随滚动，数字不超过真实计数。花瓣下落、摇摆、星光呼吸与流星保留连续运动，不做回弹。
- 星座视差在 `themes/spica/source/js/shared.js` 中按时间积分，指针离开或窗口失焦时归位。系统开启 `prefers-reduced-motion` 后停止视差，隐藏粒子，取消动画延迟并显示全部入场内容。

## 致谢

- 字体：[**Maple Mono CN**](https://github.com/subframe7536/maple-font)（OFL 协议），中文子集由[中文网字计划](https://chinese-font.netlify.app/)的 cn-font-split 切片，经 jsDelivr CDN 分发
- 图标：部分来自 [Feather Icons](https://feathericons.com/)（内联 SVG）

## License

- **代码**（主题、脚本、配置）：[MIT](LICENSE)
- **文章内容**（`source/_posts/`）：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh)
