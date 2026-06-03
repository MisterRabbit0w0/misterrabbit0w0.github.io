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

  function spawnPetals(count) {
    const container = document.getElementById('petals');
    if (!container) return;
    container.innerHTML = '';
    const n = Math.max(0, Number(count) || 0);
    for (let i = 0; i < n; i++) {
      const el = document.createElement('div');
      el.className = 'petal';
      el.style.cssText =
        `left:${Math.random() * 100}%;background:${petalColors[i % petalColors.length]};` +
        `animation-duration:${6 + Math.random() * 8}s;animation-delay:${Math.random() * 10}s;` +
        `width:${8 + Math.random() * 6}px;height:${10 + Math.random() * 8}px;` +
        `transform:rotate(${Math.random() * 360}deg);border-radius:${Math.random() > .5 ? '50% 0 50% 0' : '50%'};`;
      container.appendChild(el);
    }
  }

  function resolveThemeMode(preference) {
    if (preference === 'system') {
      return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
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
        applyThemeMode(preference);
        saveTweaks({ themeMode: preference });
      });
    });

    if (window.matchMedia) {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        try {
          const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
          if ((saved.themeMode || preference) === 'system') applyThemeMode('system');
        } catch (e) {}
      });
    }
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
      online: { color: 'var(--mint)', title: '在线 / Active' },
      idle: { color: 'oklch(72% 0.16 85)', title: '挂起 / Idle' },
      dnd: { color: 'var(--pink)', title: '勿扰 / DND' },
      offline: { color: 'var(--muted)', title: '离线 / Offline' }
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
    header.innerHTML =
      '<div class="code-block-dots" aria-hidden="true"><span></span><span></span><span></span></div>' +
      '<div class="code-block-actions"></div>';

    if (lang) {
      const langEl = document.createElement('span');
      langEl.className = 'code-block-lang';
      langEl.textContent = lang;
      header.querySelector('.code-block-actions').appendChild(langEl);
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
    const targets = document.querySelectorAll('.post-body img, .post-cover');
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

      const close = () => {
        lb.hidden = true;
        document.body.classList.remove('lightbox-open');
        lb.querySelector('.img-lightbox-img').removeAttribute('src');
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

  function initShared(defaults) {
    defaults = defaults || { petalCount: 12, themeMode: 'system' };

    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      defaults = Object.assign({}, defaults, saved);
    } catch (e) {}

    spawnPetals(defaults.petalCount || 12);
    initThemeCursor(defaults.cursorFx);
    initCodeBlocks();
    initImageLightbox();
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
