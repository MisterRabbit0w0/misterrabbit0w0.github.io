const { execSync } = require('child_process');

hexo.extend.helper.register('get_real_sparkline', function(type) {
  const xCoords = [5, 19, 33, 47, 61, 75];
  let data = [0, 0, 0, 0, 0, 0];

  try {
    if (type === 'streak') {
      let log = '';
      try {
        log = execSync('git log --date=short --pretty=format:"%ad"', { encoding: 'utf-8' });
      } catch (e) {
        log = '';
      }
      
      const commitDates = log.split('\n').filter(Boolean);
      const now = new Date();
      const periods = [];
      for (let i = 0; i < 6; i++) {
        periods.push({
          start: new Date(now.getTime() - (6 - i) * 5 * 24 * 60 * 60 * 1000),
          end: new Date(now.getTime() - (5 - i) * 5 * 24 * 60 * 60 * 1000),
          count: 0
        });
      }

      commitDates.forEach(dateStr => {
        const d = new Date(dateStr);
        if (isNaN(d.getTime())) return;
        for (let i = 0; i < 6; i++) {
          if (d >= periods[i].start && d < periods[i].end) {
            periods[i].count++;
            break;
          }
        }
      });

      data = periods.map(p => p.count);
    } else {
      const siteLocals = this.site || (hexo.locals && hexo.locals.toObject());
      const posts = siteLocals && siteLocals.posts ? siteLocals.posts.toArray().sort((a, b) => a.date - b.date) : [];

      if (posts.length === 0) {
        data = [0, 0, 0, 0, 0, 0];
      } else {
        const firstDate = posts[0].date.toDate();
        const lastDate = new Date();
        const spanMs = lastDate - firstDate;
        const intervalMs = spanMs / 6;

        for (let i = 0; i < 6; i++) {
          const cutOffDate = new Date(firstDate.getTime() + (i + 1) * intervalMs);
          const activePosts = posts.filter(p => p.date.toDate() <= cutOffDate);
          
          if (type === 'posts') {
            data[i] = activePosts.length;
          } else if (type === 'categories') {
            const cats = new Set();
            activePosts.forEach(p => {
              if (p.categories) {
                p.categories.forEach(c => cats.add(c.name));
              }
            });
            data[i] = cats.size;
          } else if (type === 'tags') {
            const tags = new Set();
            activePosts.forEach(p => {
              if (p.tags) {
                p.tags.forEach(t => tags.add(t.name));
              }
            });
            data[i] = tags.size;
          }
        }
      }
    }
  } catch (err) {
    console.error('Error generating stats sparkline:', err);
    data = [2, 4, 3, 5, 4, 6];
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const yCoords = [];

  if (min === max) {
    for (let i = 0; i < 6; i++) yCoords.push(12.5);
  } else {
    for (let i = 0; i < 6; i++) {
      const val = data[i];
      const y = 23 - ((val - min) / (max - min)) * 21;
      yCoords.push(Number(y.toFixed(2)));
    }
  }

  const y = yCoords;
  const path = `M 5,${y[0]} ` +
               `C 12,${y[0]} 12,${y[1]} 19,${y[1]} ` +
               `C 26,${y[1]} 26,${y[2]} 33,${y[2]} ` +
               `C 40,${y[2]} 40,${y[3]} 47,${y[3]} ` +
               `C 54,${y[3]} 54,${y[4]} 61,${y[4]} ` +
               `C 68,${y[4]} 68,${y[5]} 75,${y[5]}`;

  return path;
});

hexo.extend.helper.register('get_commits_count_7_days', function() {
  try {
    const countStr = execSync('git rev-list --count --since="7 days ago" HEAD', { encoding: 'utf-8' }).trim();
    const count = parseInt(countStr, 10);
    return isNaN(count) ? 0 : count;
  } catch (e) {
    return 0;
  }
});
