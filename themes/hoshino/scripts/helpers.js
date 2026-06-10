'use strict';

// 分类 → 标签配色，原先在 index/post/archive 三个模板里各复制了一份
hexo.extend.helper.register('cat_tag_class', function (name) {
  var map = {
    '前端': 'tag-pink', 'TypeScript': 'tag-lav', '随笔': 'tag-mint', '算法': 'tag-pink',
    '开源': 'tag-mint', '工具': 'tag-lav', '译本': 'tag-gold', '日记': 'tag-mint'
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

  var seed = hash(String(seedText || 'hoshino'));
  var rnd = mulberry32(seed);
  var W = 1200, H = 300;
  var uid = 'sg' + (seed % 100000);
  var parts = [];

  parts.push(
    '<defs><radialGradient id="' + uid + '" cx="' + (20 + rnd() * 60).toFixed(0) + '%" cy="' + (rnd() * 100).toFixed(0) + '%" r="85%">' +
    '<stop offset="0%" style="stop-color:var(--pink-light)" stop-opacity="0.55"/>' +
    '<stop offset="100%" style="stop-color:var(--pink-light)" stop-opacity="0"/>' +
    '</radialGradient></defs>'
  );
  parts.push('<rect width="' + W + '" height="' + H + '" fill="var(--bg2)"/>');
  parts.push('<rect width="' + W + '" height="' + H + '" fill="url(#' + uid + ')"/>');

  // 远景小星
  var dots = 46 + Math.floor(rnd() * 18);
  for (var i = 0; i < dots; i++) {
    parts.push('<circle cx="' + (rnd() * W).toFixed(1) + '" cy="' + (rnd() * H).toFixed(1) +
      '" r="' + (0.6 + rnd() * 1.1).toFixed(2) + '" fill="var(--muted)" opacity="' + (0.12 + rnd() * 0.35).toFixed(2) + '"/>');
  }

  // 主星座节点（从左到右蜿蜒）
  var n = 6 + Math.floor(rnd() * 4);
  var padX = 90, step = (W - padX * 2) / (n - 1);
  var pts = [];
  for (var i = 0; i < n; i++) {
    pts.push({
      x: padX + i * step + (rnd() - 0.5) * step * 0.6,
      y: 55 + rnd() * (H - 110)
    });
  }
  var d = pts.map(function (p, i) {
    return (i ? 'L' : 'M') + p.x.toFixed(1) + ' ' + p.y.toFixed(1);
  }).join(' ');
  parts.push('<path d="' + d + '" fill="none" stroke="var(--pink)" stroke-opacity="0.4" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>');

  // 偶尔分出一条支线
  if (n >= 7 && rnd() > 0.35) {
    var bi = 1 + Math.floor(rnd() * (n - 2));
    var bx = pts[bi].x + (rnd() - 0.5) * 160;
    var by = pts[bi].y + (rnd() > 0.5 ? 1 : -1) * (40 + rnd() * 70);
    by = Math.max(30, Math.min(H - 30, by));
    parts.push('<path d="M' + pts[bi].x.toFixed(1) + ' ' + pts[bi].y.toFixed(1) +
      ' L' + bx.toFixed(1) + ' ' + by.toFixed(1) + '" fill="none" stroke="var(--pink)" stroke-opacity="0.4" stroke-width="1.2" stroke-linecap="round"/>');
    pts.push({ x: bx, y: by });
  }

  // 节点星：最亮的一颗画成四角闪星
  var brightIdx = Math.floor(rnd() * pts.length);
  pts.forEach(function (p, i) {
    var c = i === brightIdx ? 'var(--pink)' : starColor(rnd());
    var r = 2.2 + rnd() * 1.6;
    var x = p.x.toFixed(1), y = p.y.toFixed(1);
    parts.push('<circle cx="' + x + '" cy="' + y + '" r="' + (r * 3).toFixed(1) + '" fill="' + c + '" opacity="0.12"/>');
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
  });

  return '<svg class="post-cover post-cover-star" viewBox="0 0 ' + W + ' ' + H + '" width="' + W + '" height="' + H +
    '" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">' + parts.join('') + '</svg>';
});
