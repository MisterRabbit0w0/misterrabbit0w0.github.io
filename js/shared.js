/*! Spica theme · © 2026 织星 / Astraea · https://github.com/misterrabbit0w0/misterrabbit0w0.github.io · MIT */
// shared.js — petals, theme cursor, Light/Dark/System, live status, GitHub contribution
(function () {
  const STORAGE_KEY = 'blog_tweaks';
  const petalColors = [
    'oklch(82% 0.12 355)', 'oklch(85% 0.09 330)',
    'oklch(85% 0.08 290)', 'oklch(88% 0.06 355)', 'oklch(90% 0.05 290)'
  ];
  const ACTIVITY_TYPES = new Set([
    'PushEvent', 'PullRequestEvent', 'CreateEvent', 'DeleteEvent',
    'IssuesEvent', 'IssueCommentEvent', 'PullRequestReviewEvent',
    'PullRequestReviewCommentEvent', 'ForkEvent', 'ReleaseEvent'
  ]);

  let githubEventsCache = null;
  let githubEventsPromise = null;

  function canUseThemeCursor() {
    return true;
  }

  function applyThemeCursor(enabled) {
    document.documentElement.classList.toggle('has-theme-cursor', enabled && canUseThemeCursor());
    saveTweaks({ cursorFx: enabled });
  }

  function initThemeCursor(defaultEnabled) {
    // Use theme config as source of truth to avoid stale local cache.
    const enabled = defaultEnabled !== false;

    const checkbox = document.getElementById('cursor-fx');
    if (checkbox) {
      checkbox.checked = enabled;
      checkbox.addEventListener('change', () => applyThemeCursor(checkbox.checked));
    }

    applyThemeCursor(enabled);

    if (window.matchMedia) {
      const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
      const onMq = () => applyThemeCursor(checkbox ? checkbox.checked : enabled);
      if (mq.addEventListener) mq.addEventListener('change', onMq);
      else if (mq.addListener) mq.addListener(onMq);
    }
  }

  // 点击星尘：魔法棒的 payoff。
  // 刻意不挂 prefers-reduced-motion：它是交互触发的瞬时动效，
  // 设置面板的「主题光标」开关就是它的禁用机制（关系统动画的用户也能看到）。
  function initClickStardust() {
    const glyphs = ['✦', '✧', '❀'];
    const colors = ['var(--pink)', 'var(--lavender)', 'var(--mint)', 'var(--gold)'];

    document.addEventListener('pointerdown', (e) => {
      if (!document.documentElement.classList.contains('has-theme-cursor')) return;
      if (!e.isPrimary) return;

      const n = 4 + Math.floor(Math.random() * 2);
      for (let i = 0; i < n; i++) {
        const el = document.createElement('span');
        el.className = 'stardust';
        el.textContent = glyphs[Math.floor(Math.random() * glyphs.length)];
        el.style.color = colors[Math.floor(Math.random() * colors.length)];
        el.style.left = e.clientX + 'px';
        el.style.top = e.clientY + 'px';
        el.style.fontSize = (8 + Math.random() * 6) + 'px';
        document.body.appendChild(el);

        const angle = Math.random() * Math.PI * 2;
        const dist = 20 + Math.random() * 24;
        const dx = Math.cos(angle) * dist;
        const dy = Math.sin(angle) * dist - 6;
        const spin = (Math.random() * 120 - 60).toFixed(0);
        el.animate([
          { transform: 'translate(-50%, -50%) scale(1) rotate(0deg)', opacity: 1 },
          { transform: `translate(calc(-50% + ${dx.toFixed(1)}px), calc(-50% + ${dy.toFixed(1)}px)) scale(.35) rotate(${spin}deg)`, opacity: 0 }
        ], {
          duration: 520 + Math.random() * 160,
          easing: 'cubic-bezier(.22,.61,.36,1)'
        }).onfinish = () => el.remove();
      }
    }, { passive: true });
  }

  function isDarkMode() {
    return document.documentElement.getAttribute('data-color-mode') === 'dark';
  }
  function motionReduced() {
    return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  // 夜空星屑尾迹：暗色模式下鼠标划过留下淡淡星屑
  function initSkyTrail() {
    let last = 0;
    document.addEventListener('pointermove', (e) => {
      if (!isDarkMode() || motionReduced() || e.pointerType === 'touch') return;
      const t = e.timeStamp || 0;
      if (t - last < 55) return;
      last = t;
      const el = document.createElement('span');
      el.className = 'skydust';
      el.style.left = e.clientX + 'px';
      el.style.top = e.clientY + 'px';
      document.body.appendChild(el);
      el.animate([
        { transform: 'translate(-50%,-50%) scale(1)', opacity: 0.7 },
        { transform: `translate(-50%, calc(-50% - ${8 + Math.random() * 12}px)) scale(.3)`, opacity: 0 }
      ], { duration: 800 + Math.random() * 300, easing: 'ease-out' }).onfinish = () => el.remove();
    }, { passive: true });
  }

  function burstStardust(x, y) {
    if (motionReduced()) return;
    const glyphs = ['✦', '✧', '·'];
    for (let i = 0; i < 8; i++) {
      const el = document.createElement('span');
      el.className = 'skydust skydust-burst';
      el.textContent = glyphs[i % glyphs.length];
      el.style.left = x + 'px';
      el.style.top = y + 'px';
      el.style.fontSize = (8 + Math.random() * 7) + 'px';
      document.body.appendChild(el);
      const a = (i / 8) * Math.PI * 2, d = 24 + Math.random() * 26;
      el.animate([
        { transform: 'translate(-50%,-50%) scale(1)', opacity: 1 },
        { transform: `translate(calc(-50% + ${(Math.cos(a) * d).toFixed(1)}px), calc(-50% + ${(Math.sin(a) * d).toFixed(1)}px)) scale(.3)`, opacity: 0 }
      ], { duration: 650 + Math.random() * 250, easing: 'cubic-bezier(.22,.61,.36,1)' }).onfinish = () => el.remove();
    }
  }

  // 许愿涟漪：点亮角宿一时，从星点荡开一圈柔光
  function spawnRipple(x, y) {
    if (motionReduced()) return;
    const ring = document.createElement('span');
    ring.className = 'ns-ripple';
    ring.style.left = x + 'px';
    ring.style.top = y + 'px';
    document.body.appendChild(ring);
    ring.animate([
      { opacity: 0.85, transform: 'translate(-50%,-50%) scale(.4)' },
      { opacity: 0, transform: 'translate(-50%,-50%) scale(6)' }
    ], { duration: 950, easing: 'cubic-bezier(.22,.61,.36,1)' }).onfinish = () => ring.remove();
  }

  // 流星：暗色模式下偶发划过夜空的一道细光。
  // 仅在标签页可见、未开启 reduced-motion 时出现，频率克制（约 16–42s 一次）。
  function initShootingStars() {
    if (motionReduced()) return;

    function scheduleNext() {
      setTimeout(fire, 16000 + Math.random() * 26000);
    }
    function fire() {
      if (isDarkMode() && document.visibilityState === 'visible') shoot();
      scheduleNext();
    }
    function shoot() {
      const star = document.createElement('div');
      star.className = 'shooting-star';
      const vw = window.innerWidth, vh = window.innerHeight;
      const startX = vw * (0.08 + Math.random() * 0.5);
      const startY = vh * (0.04 + Math.random() * 0.26);
      const angle = (18 + Math.random() * 16) * Math.PI / 180;
      const dist = 240 + Math.random() * 280;
      const dx = Math.cos(angle) * dist;
      const dy = Math.sin(angle) * dist;
      const deg = (angle * 180 / Math.PI).toFixed(1);
      star.style.left = startX + 'px';
      star.style.top = startY + 'px';
      document.body.appendChild(star);
      star.animate([
        { opacity: 0, transform: `translate(0,0) rotate(${deg}deg) scaleX(.4)` },
        { opacity: 1, offset: 0.2 },
        { opacity: 0.9, offset: 0.72 },
        { opacity: 0, transform: `translate(${dx.toFixed(0)}px, ${dy.toFixed(0)}px) rotate(${deg}deg) scaleX(1)` }
      ], { duration: 950 + Math.random() * 450, easing: 'cubic-bezier(.4,0,.2,1)' })
        .onfinish = () => star.remove();
    }

    scheduleNext();
  }

  // 许愿星：暗色下点亮角宿一(Spica)，浮现一句悄悄话
  function initWishStar() {
    const spica = document.querySelector('.ns-spica');
    const wishEl = document.getElementById('ns-wish');
    if (!spica || !wishEl || !(wishEl.textContent || '').trim()) return;

    let cx = 0, cy = 0;
    function recalc() {
      const r = spica.getBoundingClientRect();
      cx = r.left + r.width / 2;
      cy = r.top + r.height / 2;
    }
    recalc();
    window.addEventListener('resize', recalc);

    const HOT = 30;
    document.addEventListener('pointermove', (e) => {
      if (!isDarkMode()) { spica.classList.remove('ns-spica-hot'); return; }
      spica.classList.toggle('ns-spica-hot', Math.hypot(e.clientX - cx, e.clientY - cy) < HOT);
    }, { passive: true });

    function hideWish() {
      clearTimeout(wishEl._hideT);
      wishEl.classList.remove('show');
      clearTimeout(wishEl._rmT);
      wishEl._rmT = setTimeout(() => { wishEl.hidden = true; }, 600);
    }
    function showWish() {
      recalc();
      burstStardust(cx, cy);
      spawnRipple(cx, cy);
      clearTimeout(wishEl._rmT);
      wishEl.style.left = cx + 'px';
      wishEl.style.top = (cy + 24) + 'px';
      wishEl.hidden = false;
      requestAnimationFrame(() => wishEl.classList.add('show'));
      clearTimeout(wishEl._hideT);
      wishEl._hideT = setTimeout(hideWish, 6000); // 自动隐藏兜底
    }

    document.addEventListener('click', (e) => {
      if (!isDarkMode()) return;
      if (e.target.closest && e.target.closest('a, button, input, select, textarea, label, summary')) return;
      recalc();
      const showing = !wishEl.hidden;
      const nearSpica = Math.hypot(e.clientX - cx, e.clientY - cy) < HOT;
      // 点 Spica：切换显示/隐藏；已显示时点别处也立即关闭（气泡 pointer-events:none，点它会穿透到此处）
      if (nearSpica) { showing ? hideWish() : showWish(); }
      else if (showing) { hideWish(); }
    });
  }

  function spawnPetals(count) {
    const container = document.getElementById('petals');
    if (!container) return;
    container.innerHTML = '';
    let n = Math.max(0, Number(count) || 0);
    // 窄屏/移动端自动减量，照顾低端设备的合成性能
    if (window.matchMedia && window.matchMedia('(max-width: 640px)').matches) {
      n = Math.round(n * 0.55);
    }
    for (let i = 0; i < n; i++) {
      const w = 8 + Math.random() * 6;
      const h = w + 2 + Math.random() * 6;
      const outer = document.createElement('div');
      outer.className = 'petal';
      // 负的 animation-delay：花瓣在载入瞬间就已分散在下落周期的各个相位，
      // 不再出现「等几秒才开始飘」的空窗。
      outer.style.cssText =
        `left:${Math.random() * 100}%;width:${w}px;height:${h}px;` +
        `animation-duration:${7 + Math.random() * 9}s;animation-delay:${(-Math.random() * 14).toFixed(2)}s;`;
      const inner = document.createElement('div');
      inner.className = 'petal-inner';
      const rotBase = Math.random() * 360;
      inner.style.cssText =
        `background:${petalColors[i % petalColors.length]};` +
        `border-radius:${Math.random() > .5 ? '50% 0 50% 0' : '50%'};` +
        `--sway:${(10 + Math.random() * 18).toFixed(1)}px;` +
        `--sway-dur:${(2.6 + Math.random() * 2.4).toFixed(2)}s;` +
        `--sway-delay:${(-Math.random() * 4).toFixed(2)}s;` +
        `--rot-a:${(rotBase - 20 - Math.random() * 20).toFixed(0)}deg;` +
        `--rot-b:${(rotBase + 20 + Math.random() * 30).toFixed(0)}deg;`;
      outer.appendChild(inner);
      container.appendChild(outer);
    }
  }

  // 自动模式：按访客本地时间在日落后切到夜晚。真正的日落随经纬度/日期变化、
  // 需要地理定位权限，对博客过于侵入，故用固定时段近似（可改下方两个常量）。
  var NIGHT_START_HOUR = 18; // 入夜：18:00 起为夜晚
  var DAY_START_HOUR = 6;    // 破晓：06:00 起为白天
  function resolveThemeMode(preference) {
    if (preference === 'system') {
      var h = new Date().getHours();
      return (h >= NIGHT_START_HOUR || h < DAY_START_HOUR) ? 'dark' : 'light';
    }
    return preference === 'dark' ? 'dark' : 'light';
  }

  function applyThemeMode(preference) {
    const resolved = resolveThemeMode(preference);
    document.documentElement.setAttribute('data-color-mode', resolved);
    document.documentElement.setAttribute('data-theme-pref', preference);
    document.querySelectorAll('.theme-switch-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.themeMode === preference);
    });
  }

  function saveTweaks(patch) {
    try {
      const current = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      localStorage.setItem(STORAGE_KEY, JSON.stringify(Object.assign({}, current, patch)));
    } catch (e) {}
  }

  function initThemeSwitch(defaultPref) {
    let preference = defaultPref || 'system';
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      if (saved.themeMode) preference = saved.themeMode;
    } catch (e) {}

    applyThemeMode(preference);

    document.querySelectorAll('.theme-switch-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        preference = btn.dataset.themeMode;
        const apply = () => { applyThemeMode(preference); saveTweaks({ themeMode: preference }); };

        // 仅当真正会发生明暗翻转、且浏览器支持、未开启 reduced-motion 时，
        // 才以点击点为圆心做涟漪式揭开；否则直接切换。
        const willChange = resolveThemeMode(preference) !== document.documentElement.getAttribute('data-color-mode');
        if (!willChange || motionReduced() || !document.startViewTransition) { apply(); return; }

        const root = document.documentElement;
        const r = btn.getBoundingClientRect();
        const x = r.left + r.width / 2;
        const y = r.top + r.height / 2;
        const endR = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y));
        root.style.setProperty('--vt-x', x + 'px');
        root.style.setProperty('--vt-y', y + 'px');
        root.style.setProperty('--vt-r', endR + 'px');
        document.startViewTransition(apply);
      });
    });

    // 自动模式下每分钟复查一次，跨过日落/破晓时刻时无需刷新即自动切换
    window.setInterval(() => {
      try {
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
        if ((saved.themeMode || preference) === 'system') applyThemeMode('system');
      } catch (e) {}
    }, 60000);
  }

  async function fetchGitHubEvents(username, maxPages) {
    if (!username) return [];
    if (githubEventsCache) return githubEventsCache;
    if (githubEventsPromise) return githubEventsPromise;

    githubEventsPromise = (async () => {
      const all = [];
      const pages = maxPages || 3;
      for (let page = 1; page <= pages; page++) {
        const res = await fetch(
          `https://api.github.com/users/${encodeURIComponent(username)}/events/public?per_page=100&page=${page}`
        );
        if (!res.ok) throw new Error(`GitHub events HTTP ${res.status}`);
        const batch = await res.json();
        if (!Array.isArray(batch) || batch.length === 0) break;
        all.push(...batch.filter(e => ACTIVITY_TYPES.has(e.type)));
        if (batch.length < 100) break;
      }
      githubEventsCache = all;
      return all;
    })();

    try {
      return await githubEventsPromise;
    } catch (err) {
      githubEventsPromise = null;
      throw err;
    }
  }

  function formatRelativeTime(dateStr) {
    const diffMs = Date.now() - new Date(dateStr).getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);
    if (diffDays > 0) return `${diffDays} 天前`;
    if (diffHours > 0) return `${diffHours} 小时前`;
    if (diffMins > 0) return `${diffMins} 分钟前`;
    return '刚刚';
  }

  function shortRepoName(fullName, username) {
    if (!fullName) return 'unknown';
    const prefix = `${username}/`;
    return fullName.startsWith(prefix) ? fullName.slice(prefix.length) : fullName;
  }

  function describeEvent(event, username) {
    const repo = shortRepoName(event.repo?.name, username);
    const repoUrl = `https://github.com/${event.repo?.name || ''}`;
    switch (event.type) {
      case 'PushEvent': {
        const n = event.payload?.size || event.payload?.commits?.length || 0;
        const msg = event.payload?.commits?.[0]?.message || '更新了代码';
        return { text: `推送到 ${repo}（${n || 1} commits）`, detail: msg, repoUrl };
      }
      case 'PullRequestEvent': {
        const action = event.payload?.action || 'updated';
        const title = event.payload?.pull_request?.title || 'Pull Request';
        return { text: `PR ${action} · ${repo}`, detail: title, repoUrl };
      }
      case 'IssuesEvent': {
        const action = event.payload?.action || 'updated';
        const title = event.payload?.issue?.title || 'Issue';
        return { text: `Issue ${action} · ${repo}`, detail: title, repoUrl };
      }
      case 'IssueCommentEvent':
      case 'PullRequestReviewCommentEvent':
        return { text: `评论 · ${repo}`, detail: '发表了评论', repoUrl };
      case 'PullRequestReviewEvent':
        return { text: `Review · ${repo}`, detail: '审查了 Pull Request', repoUrl };
      case 'CreateEvent': {
        const ref = event.payload?.ref || event.payload?.ref_type || '资源';
        return { text: `创建 ${ref} · ${repo}`, detail: event.payload?.description || '', repoUrl };
      }
      case 'DeleteEvent':
        return { text: `删除分支 · ${repo}`, detail: event.payload?.ref || '', repoUrl };
      case 'ForkEvent':
        return { text: `Fork · ${repo}`, detail: '分叉了仓库', repoUrl };
      case 'ReleaseEvent':
        return { text: `Release · ${repo}`, detail: event.payload?.release?.name || '发布了版本', repoUrl };
      default:
        return { text: `${event.type} · ${repo}`, detail: '', repoUrl };
    }
  }

  // Renders an arbitrary-length series as a smooth sparkline inside the
  // viewBox "0 0 80 25". The baseline is anchored at 0 (not at the series
  // minimum) so days with no activity sit on the bottom edge instead of being
  // rebased away, and an all-zero series stays flat along the baseline.
  function buildSparklinePath(values) {
    const data = (values && values.length) ? values.slice() : [0];
    if (data.length === 1) data.push(data[0]);
    const n = data.length;
    const X0 = 1, X1 = 79, TOP = 2, BOTTOM = 23;
    const max = Math.max(0, ...data);
    const pts = data.map((val, i) => ({
      x: X0 + (X1 - X0) * (i / (n - 1)),
      y: max > 0 ? BOTTOM - (val / max) * (BOTTOM - TOP) : BOTTOM
    }));

    const f = v => Number(v.toFixed(2));
    const clampY = v => Math.max(TOP, Math.min(BOTTOM, v));
    let d = `M ${f(pts[0].x)},${f(pts[0].y)}`;
    // Catmull-Rom -> cubic Bézier so the curve stays smooth for any point count.
    // Control-point Y is clamped so the smoothing never overshoots past the
    // zero baseline or the top of the chart.
    for (let i = 0; i < n - 1; i++) {
      const p0 = pts[i - 1] || pts[i];
      const p1 = pts[i];
      const p2 = pts[i + 1];
      const p3 = pts[i + 2] || p2;
      const cp1x = p1.x + (p2.x - p0.x) / 6;
      const cp1y = clampY(p1.y + (p2.y - p0.y) / 6);
      const cp2x = p2.x - (p3.x - p1.x) / 6;
      const cp2y = clampY(p2.y - (p3.y - p1.y) / 6);
      d += ` C ${f(cp1x)},${f(cp1y)} ${f(cp2x)},${f(cp2y)} ${f(p2.x)},${f(p2.y)}`;
    }
    return d;
  }

  function countEventsSince(events, days) {
    const cutoff = Date.now() - days * 24 * 60 * 60 * 1000;
    return events.filter(e => new Date(e.created_at).getTime() >= cutoff).length;
  }

  function bucketEvents(events, bucketCount, spanDays) {
    const now = Date.now();
    const buckets = Array(bucketCount).fill(0);
    const bucketMs = (spanDays * 24 * 60 * 60 * 1000) / bucketCount;
    events.forEach(e => {
      const age = now - new Date(e.created_at).getTime();
      if (age < 0 || age > spanDays * 24 * 60 * 60 * 1000) return;
      const idx = Math.min(bucketCount - 1, Math.floor((spanDays * 24 * 60 * 60 * 1000 - age) / bucketMs));
      buckets[idx]++;
    });
    return buckets;
  }

  async function initContributionCard(username) {
    const card = document.getElementById('contrib-card');
    if (!card || !username) return;

    const count7 = document.getElementById('contrib-count-7d');
    const count30 = document.getElementById('contrib-count-30d');
    const sparkPath = document.getElementById('contrib-sparkline-path');
    const list = document.getElementById('contrib-recent');

    try {
      const events = await fetchGitHubEvents(username, 3);
      if (count7) count7.textContent = String(countEventsSince(events, 7));
      if (count30) count30.textContent = String(countEventsSince(events, 30));
      if (sparkPath) sparkPath.setAttribute('d', buildSparklinePath(bucketEvents(events, 30, 30)));

      if (list) {
        list.innerHTML = '';
        const recent = events.slice(0, 6);
        if (recent.length === 0) {
          list.innerHTML = '<li class="contrib-item contrib-placeholder">近期暂无公开 GitHub 活动</li>';
          return;
        }
        recent.forEach(event => {
          const info = describeEvent(event, username);
          const li = document.createElement('li');
          li.className = 'contrib-item';
          li.innerHTML =
            `<a href="${info.repoUrl}" target="_blank" rel="noopener">${info.text}</a>` +
            (info.detail ? `<span>${info.detail}</span>` : '') +
            `<time datetime="${event.created_at}">${formatRelativeTime(event.created_at)}</time>`;
          list.appendChild(li);
        });
      }
    } catch (err) {
      console.warn('Contribution card load failed.', err);
      if (count7) count7.textContent = '—';
      if (count30) count30.textContent = '—';
      if (list) {
        list.innerHTML = '<li class="contrib-item contrib-error">GitHub 活动加载失败（可能是 API 限流）</li>';
      }
    }
  }

  const IDE_NAMES = new Set([
    'Visual Studio Code', 'Cursor', 'Vim', 'Neovim',
    'IntelliJ IDEA', 'WebStorm', 'PyCharm', 'Android Studio'
  ]);

  function applyLanyardStatus(data, cfg, els) {
    const status = data.discord_status || 'offline';
    const { actEl, detailEl, dotEl, titleEl } = els;

    const statusMeta = {
      online: { color: 'var(--mint)', title: '在线' },
      idle: { color: 'oklch(72% 0.16 85)', title: '挂起' },
      dnd: { color: 'var(--pink)', title: '勿扰' },
      offline: { color: 'var(--muted)', title: '离线' }
    };
    const meta = statusMeta[status] || statusMeta.offline;
    if (dotEl) dotEl.style.setProperty('--status-color', meta.color);
    if (titleEl) titleEl.textContent = meta.title;

    if (status === 'offline') {
      if (actEl) actEl.textContent = cfg.defaultAct || '当前离线';
      if (detailEl) detailEl.textContent = cfg.defaultDetail || 'Discord 离线';
      return;
    }

    if (data.listening_to_spotify && data.spotify) {
      if (actEl) actEl.innerHTML = `正在聆听 <strong>${data.spotify.song}</strong>`;
      if (detailEl) detailEl.textContent = `歌手: ${data.spotify.artist}`;
      return;
    }

    const ide = (data.activities || []).find(a => IDE_NAMES.has(a.name));
    if (ide) {
      const workspace = (ide.state || '').replace(/^Workspace: /, '');
      const details = ide.details || '写代码中';
      if (actEl) {
        actEl.innerHTML = workspace
          ? `正在开发 <strong>${workspace}</strong>`
          : `正在使用 <strong>${ide.name}</strong>`;
      }
      if (detailEl) detailEl.textContent = details;
      return;
    }

    const custom = (data.activities || []).find(a => a.type === 4 || a.id === 'custom');
    if (custom && custom.state) {
      if (actEl) actEl.textContent = custom.state;
      if (detailEl) {
        detailEl.textContent = custom.emoji?.name
          ? `Discord 状态 · ${custom.emoji.name}`
          : 'Discord 状态';
      }
      return;
    }

    const playing = (data.activities || []).find(a => a.type === 0 && a.name);
    if (playing) {
      if (actEl) actEl.innerHTML = `正在 <strong>${playing.name}</strong>`;
      if (detailEl) detailEl.textContent = playing.details || playing.state || 'Discord 活动';
      return;
    }

    if (actEl) actEl.textContent = cfg.defaultAct || '正在编织星光...';
    if (detailEl) detailEl.textContent = cfg.defaultDetail || 'Discord 在线中';
  }

  function initLiveStatus(cfg) {
    if (!cfg || !cfg.enable) return;

    (async function updateLiveStatus() {
      const locEl = document.getElementById('status-loc');
      const actEl = document.getElementById('status-act');
      const detailEl = document.getElementById('status-detail');
      const cardEl = document.getElementById('status-card');
      const dotEl = document.querySelector('.status-dot');
      const titleEl = document.querySelector('.status-title');
      if (!cardEl) return;

      cardEl.hidden = false;

      if (locEl && cfg.location) locEl.textContent = cfg.location;
      if (actEl && cfg.defaultAct) actEl.textContent = cfg.defaultAct;
      if (detailEl && cfg.defaultDetail) detailEl.textContent = cfg.defaultDetail;

      if (!cfg.discordId) return;

      try {
        const res = await fetch(`https://api.lanyard.rest/v1/users/${cfg.discordId}`);
        if (!res.ok) throw new Error('Lanyard fetch failed');
        const body = await res.json();
        if (body.success && body.data) {
          applyLanyardStatus(body.data, cfg, { actEl, detailEl, dotEl, titleEl });
        }
      } catch (err) {
        console.warn('Lanyard status load failed, using defaults.', err);
      }
    })();
  }

  function getCodeText(codeEl) {
    return (codeEl.innerText || codeEl.textContent || '').replace(/\n$/, '');
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (e) {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.left = '-9999px';
      document.body.appendChild(ta);
      ta.select();
      let ok = false;
      try { ok = document.execCommand('copy'); } catch (err) {}
      document.body.removeChild(ta);
      return ok;
    }
  }

  function attachCopyButton(toolbar, codeEl) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'code-copy-btn';
    btn.setAttribute('aria-label', '复制代码');
    btn.innerHTML =
      '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      '<rect x="9" y="9" width="13" height="13" rx="2"/>' +
      '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>' +
      '</svg><span class="code-copy-label">复制</span>';

    btn.addEventListener('click', async () => {
      const ok = await copyText(getCodeText(codeEl));
      const label = btn.querySelector('.code-copy-label');
      if (!label) return;
      const prev = label.textContent;
      label.textContent = ok ? '已复制' : '失败';
      btn.classList.toggle('copied', ok);
      window.setTimeout(() => {
        label.textContent = prev;
        btn.classList.remove('copied');
      }, 1600);
    });

    toolbar.appendChild(btn);
  }

  function buildCodeBlockHeader(lang) {
    const header = document.createElement('div');
    header.className = 'code-block-header';
    header.innerHTML = '<div class="code-block-actions"></div>';

    if (lang) {
      const langEl = document.createElement('span');
      langEl.className = 'code-block-lang';
      langEl.textContent = lang;
      header.insertBefore(langEl, header.firstChild);
    }

    return header;
  }

  function wrapCodeBlock(block, codeEl, lang) {
    if (block.parentElement?.classList.contains('code-block-wrap')) return;
    if (block.closest('.code-block-wrap')) return;

    const wrap = document.createElement('div');
    wrap.className = 'code-block-wrap';
    const header = buildCodeBlockHeader(lang);
    attachCopyButton(header.querySelector('.code-block-actions'), codeEl);

    block.parentNode.insertBefore(wrap, block);
    wrap.appendChild(header);
    wrap.appendChild(block);
  }

  function initCodeBlocks() {
    document.querySelectorAll('pre > code[class*="highlight"]').forEach(code => {
      const pre = code.parentElement;
      if (!pre || pre.closest('figure.highlight')) return;
      const lang = Array.from(code.classList).find(c => c !== 'highlight');
      wrapCodeBlock(pre, code, lang);
    });

    document.querySelectorAll('figure.highlight').forEach(fig => {
      if (fig.querySelector('.code-block-wrap')) return;
      const code = fig.querySelector('code');
      const pre = fig.querySelector('pre');
      const codeEl = code || pre;
      if (!codeEl) return;
      const lang = Array.from(fig.classList).find(c => c !== 'highlight');
      wrapCodeBlock(fig, codeEl, lang);
    });

    document.querySelectorAll('.post-body > pre:not(.code-block-wrap pre)').forEach(pre => {
      if (pre.closest('.code-block-wrap')) return;
      const code = pre.querySelector('code') || pre;
      wrapCodeBlock(pre, code, '');
    });
  }

  function initImageLightbox() {
    const targets = document.querySelectorAll('.post-body img, img.post-cover');
    if (!targets.length) return;

    let lb = document.getElementById('img-lightbox');
    if (!lb) {
      lb = document.createElement('div');
      lb.id = 'img-lightbox';
      lb.className = 'img-lightbox';
      lb.hidden = true;
      lb.innerHTML =
        '<button type="button" class="img-lightbox-backdrop" aria-label="关闭预览"></button>' +
        '<figure class="img-lightbox-panel">' +
        '<button type="button" class="img-lightbox-close" aria-label="关闭">&times;</button>' +
        '<img class="img-lightbox-img" alt="">' +
        '<figcaption class="img-lightbox-caption"></figcaption>' +
        '</figure>';

      document.body.appendChild(lb);

      const finishClose = () => {
        lb.classList.remove('img-lightbox-closing');
        lb.hidden = true;
        document.body.classList.remove('lightbox-open');
        lb.querySelector('.img-lightbox-img').removeAttribute('src');
      };
      const close = () => {
        if (lb.hidden || lb.classList.contains('img-lightbox-closing')) return;
        if (motionReduced()) { finishClose(); return; }
        const panel = lb.querySelector('.img-lightbox-panel');
        lb.classList.add('img-lightbox-closing');
        const onEnd = () => { panel.removeEventListener('animationend', onEnd); finishClose(); };
        panel.addEventListener('animationend', onEnd);
      };

      lb.querySelector('.img-lightbox-backdrop').addEventListener('click', close);
      lb.querySelector('.img-lightbox-close').addEventListener('click', close);
      document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && !lb.hidden) close();
      });
    }

    const open = img => {
      const full = img.currentSrc || img.src;
      if (!full) return;
      const lbImg = lb.querySelector('.img-lightbox-img');
      const cap = lb.querySelector('.img-lightbox-caption');
      lbImg.src = full;
      lbImg.alt = img.alt || '';
      if (img.alt) {
        cap.textContent = img.alt;
        cap.hidden = false;
      } else {
        cap.textContent = '';
        cap.hidden = true;
      }
      lb.hidden = false;
      document.body.classList.add('lightbox-open');
    };

    targets.forEach(img => {
      if (img.dataset.lightboxBound) return;
      img.dataset.lightboxBound = '1';
      img.classList.add('img-zoomable');

      const parentA = img.closest('a');
      if (parentA) {
        const href = parentA.getAttribute('href') || '';
        if (href && href !== img.src && !href.endsWith(img.getAttribute('src') || '')) return;
        parentA.addEventListener('click', e => {
          e.preventDefault();
          open(img);
        });
      } else {
        img.addEventListener('click', () => open(img));
      }
    });
  }

  // 入场渐现：页面载入时首屏元素逐一浮现，滚动时分批错峰显现。
  // 刻意不让正文段落参与，阅读时保持安定；只让标题、卡片、列表行等结构块渐入。
  // 健壮性：首屏元素在载入时即显现（不依赖滚动/IO），滚动兜底保证即便
  // IntersectionObserver 在某些环境不回调，内容也绝不会永久隐藏。
  function initReveal() {
    const root = document.documentElement;
    if (motionReduced()) return;
    root.classList.add('js-reveal');

    const selector = [
      '.hero-text > *',
      '.sec-header',
      '.article-row',
      '.post-card', '.post-card-featured',
      '.repo-card', '.project-card', '.project-featured',
      '.archive-card', '.skill-card', '.timeline-item',
      '.widget-card', '.nav-card', '.about-section',
      '.page-hero-left', '.article-count',
      '.post-header > *', '.post-cover', '.post-cover-star',
      '.notfound-badge', '.notfound-num', '.notfound-title', '.notfound-bio', '.notfound-actions'
    ].join(',');

    const els = Array.from(document.querySelectorAll(selector));
    if (!els.length) return;
    els.forEach(el => el.classList.add('reveal'));

    function reveal(el, delayMs) {
      if (el.classList.contains('reveal-in')) return;
      if (delayMs) el.style.transitionDelay = delayMs + 'ms';
      el.classList.add('reveal-in');
      el.addEventListener('transitionend', function te(ev) {
        if (ev.propertyName !== 'opacity') return;
        el.style.transitionDelay = '';
        el.classList.add('reveal-done');
        el.removeEventListener('transitionend', te);
      });
    }
    function inViewport(el) {
      const r = el.getBoundingClientRect();
      const vh = window.innerHeight || document.documentElement.clientHeight;
      return r.top < vh * 0.94 && r.bottom > 0;
    }

    // 载入即显现首屏元素（错峰）：即使没有滚动，「页面进入」也会播放，
    // 且任何可见内容都不会停在隐藏态。用一小段 setTimeout 让隐藏态先绘制一帧
    // （否则同一帧内叠加会跳过淡入），且 setTimeout 即便在被节流的后台标签页也会触发，
    // 比 rAF 更可靠地保证内容最终一定显现。
    setTimeout(() => {
      let firstIndex = 0;
      els.forEach(el => { if (inViewport(el)) reveal(el, Math.min(firstIndex++ * 55, 300)); });
    }, 45);

    // 下方元素：优先用 IntersectionObserver 在进入视口时错峰显现
    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver((entries, obs) => {
        const incoming = entries.filter(e => e.isIntersecting);
        incoming.forEach((entry, i) => {
          reveal(entry.target, Math.min(i * 55, 300));
          obs.unobserve(entry.target);
        });
      }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });
      els.forEach(el => { if (!el.classList.contains('reveal-in')) io.observe(el); });
    }

    // 兜底：滚动时显现进入视口的元素（即便 IO 在当前环境不回调也始终生效）
    function onScroll() {
      els.forEach(el => { if (!el.classList.contains('reveal-in') && inViewport(el)) reveal(el, 0); });
      if (els.every(el => el.classList.contains('reveal-in'))) {
        window.removeEventListener('scroll', onScroll);
      }
    }
    window.addEventListener('scroll', onScroll, { passive: true });
  }

  function initShared(defaults) {
    defaults = defaults || { petalCount: 12, themeMode: 'system' };

    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      defaults = Object.assign({}, defaults, saved);
    } catch (e) {}

    spawnPetals(defaults.petalCount || 12);
    initThemeCursor(defaults.cursorFx);
    initClickStardust();
    initSkyTrail();
    initWishStar();
    initShootingStars();
    initCodeBlocks();
    initImageLightbox();
    initReveal();
    initThemeSwitch(defaults.themeMode || defaults.theme || 'system');
    initLiveStatus(defaults.statusCard);

    const username = defaults.statusCard && defaults.statusCard.username;
    if (document.getElementById('contrib-card') && username) {
      initContributionCard(username);
    }

    const panel = document.getElementById('tweaks-panel');
    const toggle = document.getElementById('tweaks-toggle');
    if (toggle && panel) {
      toggle.addEventListener('click', e => {
        e.stopPropagation();
        const open = panel.classList.toggle('open');
        toggle.classList.toggle('open', open);
      });
      document.addEventListener('click', e => {
        if (!panel.contains(e.target) && !toggle.contains(e.target)) {
          panel.classList.remove('open');
          toggle.classList.remove('open');
        }
      });
      document.addEventListener('keydown', e => {
        if (e.key === 'Escape') {
          panel.classList.remove('open');
          toggle.classList.remove('open');
        }
      });
    }

    const pc = document.getElementById('petal-count');
    if (pc) {
      pc.value = defaults.petalCount || 12;
      pc.addEventListener('input', e => {
        const v = +e.target.value;
        spawnPetals(v);
        saveTweaks({ petalCount: v });
      });
    }
  }

  window.initShared = initShared;
  window.spawnPetals = spawnPetals;
})();
