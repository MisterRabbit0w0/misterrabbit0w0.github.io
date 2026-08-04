'use strict';

// 文章内组件（tag plugins），样式在 source/css/page.css「文章组件」区块。
//
// {% spoiler %}剧透内容{% endspoiler %}                行内模糊条，hover / 键盘聚焦显形
// {% dialogue 角色名 %}台词{% enddialogue %}            台词行：粉色角色签 + 内容（支持 markdown）
// {% rating 4.5 %}                                      五星评分（spica 四角闪星，0.5 步进）
// {% linkcard url | 标题 | 描述 %}                      外链书签卡，标题、描述可省略
// {% gallery 2 可选说明 %} ![a](..) ![b](..) {% endgallery %}  图组网格（2 或 3 列，移动端降列）

var escapeHTML = require('hexo-util').escapeHTML;

function renderBlock(content) {
  return hexo.render.renderSync({ text: String(content || '').trim(), engine: 'markdown' });
}

function renderInline(content) {
  return renderBlock(content).replace(/^<p>([\s\S]*?)<\/p>\s*$/, '$1');
}

/* 剧透：纯 CSS 显形（:hover / :focus-visible），移动端点按即聚焦 */
hexo.extend.tag.register('spoiler', function (args, content) {
  return '<span class="spoiler" tabindex="0" title="剧透">' + renderInline(content) + '</span>';
}, { ends: true });

/* 台词：角色名只能是纯文本，台词正文走 markdown */
hexo.extend.tag.register('dialogue', function (args, content) {
  var speaker = args.join(' ').trim();
  var out = '<div class="dialogue">';
  if (speaker) out += '<span class="dialogue-speaker">' + escapeHTML(speaker) + '</span>';
  out += '<div class="dialogue-line">' + renderBlock(content) + '</div></div>';
  return out;
}, { ends: true });

/* 评分：底排五枚空星 + 顶排实心星按百分比裁切，与星图封面共用同一枚四角闪星 */
var SPICA_STAR = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2 Q12 12 22 12 Q12 12 12 22 Q12 12 2 12 Q12 12 12 2 Z"/></svg>';

hexo.extend.tag.register('rating', function (args) {
  var score = parseFloat(args[0]);
  if (isNaN(score)) score = 0;
  score = Math.max(0, Math.min(5, Math.round(score * 2) / 2));
  var row = SPICA_STAR + SPICA_STAR + SPICA_STAR + SPICA_STAR + SPICA_STAR;
  return '<span class="rating" role="img" aria-label="评分 ' + score + ' / 5">' +
    '<span class="rating-track">' + row +
    '<span class="rating-fill" style="width:' + (score / 5) * 100 + '%">' + row + '</span>' +
    '</span><span class="rating-num">' + score + '</span></span>';
});

/* 外链书签卡：{% linkcard https://example.com | 标题 | 一句话描述 %} */
hexo.extend.tag.register('linkcard', function (args) {
  var parts = args.join(' ').split('|').map(function (s) { return s.trim(); });
  var url = parts[0] || '';
  var domain = url.replace(/^https?:\/\//i, '').split('/')[0];
  var title = parts[1] || domain;
  var desc = parts[2] || '';
  return '<a class="linkcard" href="' + escapeHTML(url) + '" target="_blank" rel="noopener noreferrer">' +
    '<span class="linkcard-main">' +
    '<span class="linkcard-title">' + escapeHTML(title) + '</span>' +
    (desc ? '<span class="linkcard-desc">' + escapeHTML(desc) + '</span>' : '') +
    '<span class="linkcard-domain">' + escapeHTML(domain) + '</span>' +
    '</span><span class="linkcard-arrow" aria-hidden="true">&#8599;</span></a>';
});

/* 图组：{% gallery 2 可选说明 %} ...markdown 图片... {% endgallery %} */
hexo.extend.tag.register('gallery', function (args, content) {
  var cols = parseInt(args[0], 10);
  if (cols !== 2 && cols !== 3) cols = 2;
  var caption = args.slice(1).join(' ').trim();
  return '<figure class="gallery gallery-cols-' + cols + '">' +
    '<div class="gallery-grid">' + renderBlock(content) + '</div>' +
    (caption ? '<figcaption>' + escapeHTML(caption) + '</figcaption>' : '') +
    '</figure>';
}, { ends: true });
