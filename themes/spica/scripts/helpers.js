'use strict';

// 分类 → 标签配色的唯一权威。颜色按内容类型固定：
// 写作/随笔 = pink（站点主体），技术/工具 = lav，开源 = mint，翻译 = gold。
// 同一个词在任何页面都必须是同一个颜色；未收录的词落到 pink。
hexo.extend.helper.register('cat_tag_class', function (name) {
  var map = {
    'articles': 'tag-pink', '随笔': 'tag-pink', '日记': 'tag-pink', '学习笔记': 'tag-pink',
    'tools': 'tag-lav', 'AI': 'tag-lav', '工具': 'tag-lav', '前端': 'tag-lav',
    '算法': 'tag-lav', 'TypeScript': 'tag-lav',
    '开源': 'tag-mint',
    'translation': 'tag-gold', '译本': 'tag-gold'
  };
  return map[name] || 'tag-pink';
});

// 星图封面：以文章标题为种子生成确定性的星座 SVG。
// 同一篇文章永远是同一张星图，新文章自带新星座；颜色走 CSS 变量，亮暗模式自适应。
hexo.extend.helper.register('star_cover', function (seedText) {
  function hash(str) {
    var h1 = 0xdeadbeef, h2 = 0x41c6ce57;
    for (var i = 0; i < str.length; i++) {
      var ch = str.charCodeAt(i);
      h1 = Math.imul(h1 ^ ch, 2654435761);
      h2 = Math.imul(h2 ^ ch, 1597334677);
    }
    h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
    h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
    return (4294967296 * (2097151 & h2) + (h1 >>> 0)) >>> 0;
  }
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function starColor(r) {
    if (r < 0.7) return 'var(--pink)';
    if (r < 0.82) return 'var(--lavender)';
    if (r < 0.94) return 'var(--mint)';
    return 'var(--gold)';
  }

  var seed = hash(String(seedText || 'spica'));
  var rnd = mulberry32(seed);
  var W = 1200, H = 300;
  var parts = [];

  // 环境辉光不在 SVG 内：它以独立的 CSS 层（.cover-glow）存在，不被视口裁切、
  // 不被 mask 横切，向下溢出融入正文——光穿过 mask，切口消失。
  // 这里只产出光心位置（--gx/--gy）。随机数调用次序与旧版一致，已有星图不变。
  var gx = 0.3 + rnd() * 0.4, gy = 0.3 + rnd() * 0.4;

  // 视差分层：辉光（.cover-glow）与远景小星为远层，星座为近层（cover-near）
  parts.push('<g class="cover-far">');

  // 远景小星：带彩度的粉玫瑰灰（--star-dot），少而可辨——浅色下不再是灰尘；
  // 每 7 颗挑 1 颗做低幅错峰闪烁（复用夜空同款 ns-tw）
  var dots = 36 + Math.floor(rnd() * 14);
  for (var i = 0; i < dots; i++) {
    var dop = (0.22 + rnd() * 0.38).toFixed(2);
    var dtw = (i % 7 === 0);
    parts.push('<circle class="cover-dot' + (dtw ? ' ns-tw' : '') + '" cx="' + (rnd() * W).toFixed(1) +
      '" cy="' + (rnd() * H).toFixed(1) + '" r="' + (0.9 + rnd() * 1.2).toFixed(2) + '" opacity="' + dop + '"' +
      (dtw ? ' style="--tw-a:' + (dop * 0.35).toFixed(2) + ';--tw-hi:' + dop +
        ';--tw-dur:' + (4 + (i % 5) * 0.8).toFixed(1) + 's;--tw-del:-' + (i % 7) + 's"' : '') + '/>');
  }
  parts.push('</g><g class="cover-near">');

  // 主星座节点（从左到右蜿蜒）
  var n = 6 + Math.floor(rnd() * 4);
  var padX = 115, step = (W - padX * 2) / (n - 1);
  var pts = [];
  for (var i = 0; i < n; i++) {
    pts.push({
      x: padX + i * step + (rnd() - 0.5) * step * 0.6,
      y: 80 + rnd() * (H - 160)
    });
  }
  var d = pts.map(function (p, i) {
    return (i ? 'L' : 'M') + p.x.toFixed(1) + ' ' + p.y.toFixed(1);
  }).join(' ');
  // pathLength=1 + dashoffset：星座线先自我描绘一遍（一次性仪式），再进入常态
  parts.push('<path class="star-link" pathLength="1" style="--d:.15s" d="' + d + '" fill="none" stroke="var(--pink)" stroke-opacity="0.4" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>');

  // 偶尔分出一条支线（主线画完后再长出）
  if (n >= 7 && rnd() > 0.35) {
    var bi = 1 + Math.floor(rnd() * (n - 2));
    var bx = pts[bi].x + (rnd() - 0.5) * 160;
    var by = pts[bi].y + (rnd() > 0.5 ? 1 : -1) * (40 + rnd() * 70);
    by = Math.max(55, Math.min(H - 55, by));
    parts.push('<path class="star-link" pathLength="1" style="--d:.85s" d="M' + pts[bi].x.toFixed(1) + ' ' + pts[bi].y.toFixed(1) +
      ' L' + bx.toFixed(1) + ' ' + by.toFixed(1) + '" fill="none" stroke="var(--pink)" stroke-opacity="0.4" stroke-width="1.2" stroke-linecap="round"/>');
    pts.push({ x: bx, y: by });
  }

  // 节点星：沿线路径依次绽出（--d 逐颗 stagger），最亮的一颗画成四角闪星
  var brightIdx = Math.floor(rnd() * pts.length);
  pts.forEach(function (p, i) {
    var c = i === brightIdx ? 'var(--pink)' : starColor(rnd());
    var r = 2.2 + rnd() * 1.6;
    var x = p.x.toFixed(1), y = p.y.toFixed(1);
    // 主线描绘近尾声时开始，支线节点排最后
    var delay = (0.55 + i * 0.1).toFixed(2);
    parts.push('<g class="star-node" style="--d:' + delay + 's">' +
      '<circle cx="' + x + '" cy="' + y + '" r="' + (r * 3).toFixed(1) + '" fill="' + c + '" opacity="0.12"/>');
    if (i === brightIdx) {
      var s = (11 + rnd() * 5).toFixed(1);
      parts.push('<path class="star-bright" d="M' + x + ' ' + (p.y - s) +
        ' Q' + x + ' ' + y + ' ' + (p.x + Number(s)).toFixed(1) + ' ' + y +
        ' Q' + x + ' ' + y + ' ' + x + ' ' + (p.y + Number(s)).toFixed(1) +
        ' Q' + x + ' ' + y + ' ' + (p.x - Number(s)).toFixed(1) + ' ' + y +
        ' Q' + x + ' ' + y + ' ' + x + ' ' + (p.y - s) + ' Z" fill="' + c + '"/>');
    } else {
      parts.push('<circle cx="' + x + '" cy="' + y + '" r="' + r.toFixed(1) + '" fill="' + c + '"/>');
    }
    parts.push('</g>');
  });
  parts.push('</g>');

  return '<div class="post-cover-star-wrap" style="--gx:' + (gx * 100).toFixed(1) + '%;--gy:' + (gy * 100).toFixed(1) + '%">' +
    '<div class="cover-glow" aria-hidden="true"></div>' +
    '<svg class="post-cover post-cover-star" viewBox="0 0 ' + W + ' ' + H + '" width="' + W + '" height="' + H +
    '" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">' + parts.join('') + '</svg></div>';
});
