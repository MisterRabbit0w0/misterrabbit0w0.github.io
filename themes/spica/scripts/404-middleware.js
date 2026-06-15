const fs = require('fs');
const path = require('path');

hexo.extend.filter.register('server_middleware', function (app) {
  app.use((req, res) => {
    const file = path.join(hexo.public_dir, '404.html');
    res.statusCode = 404;
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    if (fs.existsSync(file)) fs.createReadStream(file).pipe(res);
    else res.end('404 Not Found');
  });
});
