'use strict';

// 在构建产物的每个 CSS / JS 文件顶部注入署名 banner。
// 浏览器拿到的就是带 banner 的版本，复制走样式/脚本时会一并带上，
// 删掉它需要主动为之——比"做了个相似设计"更易举证。
// after_render:css / after_render:js 在 hexo server 与 hexo generate 下都会触发。

var BANNER =
  '/*! Hoshino theme · © 2026 织星 / Astraea · ' +
  'https://github.com/misterrabbit0w0/misterrabbit0w0.github.io · MIT */\n';

function prepend(str) {
  if (typeof str !== 'string' || str.indexOf('Hoshino theme ·') !== -1) return str;
  return BANNER + str;
}

hexo.extend.filter.register('after_render:css', prepend);
hexo.extend.filter.register('after_render:js', prepend);
