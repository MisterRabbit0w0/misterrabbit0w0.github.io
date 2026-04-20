// shared.js — petal spawner + theme tweaks
(function () {
  const petalColors = [
    'oklch(82% 0.12 355)','oklch(85% 0.09 330)',
    'oklch(85% 0.08 290)','oklch(88% 0.06 355)','oklch(90% 0.05 290)'
  ];
  function spawnPetals(count) {
    const container = document.getElementById('petals');
    if (!container) return;
    container.innerHTML = '';
    for (let i = 0; i < count; i++) {
      const el = document.createElement('div');
      el.className = 'petal';
      el.style.cssText = `left:${Math.random()*100}%;background:${petalColors[i%petalColors.length]};animation-duration:${6+Math.random()*8}s;animation-delay:${Math.random()*10}s;width:${8+Math.random()*6}px;height:${10+Math.random()*8}px;transform:rotate(${Math.random()*360}deg);border-radius:${Math.random()>.5?'50% 0 50% 0':'50%'};`;
      container.appendChild(el);
    }
  }

  const themes = {
    pink: { '--pink':'oklch(72% 0.18 355)','--lavender':'oklch(72% 0.18 290)','--mint':'oklch(72% 0.18 175)','--bg':'oklch(98% 0.012 330)','--bg2':'oklch(95.5% 0.018 330)','--card':'oklch(99% 0.008 330)','--fg':'oklch(22% 0.02 285)','--muted':'oklch(58% 0.02 285)','--border':'oklch(89% 0.025 330)','--pink-light':'oklch(90% 0.08 355)','--lav-light':'oklch(91% 0.07 290)' },
    blue: { '--pink':'oklch(70% 0.18 220)','--lavender':'oklch(72% 0.18 175)','--mint':'oklch(72% 0.18 145)','--bg':'oklch(98% 0.010 220)','--bg2':'oklch(95% 0.016 220)','--card':'oklch(99% 0.006 220)','--fg':'oklch(20% 0.02 240)','--muted':'oklch(56% 0.02 240)','--border':'oklch(88% 0.02 220)','--pink-light':'oklch(90% 0.07 220)','--lav-light':'oklch(91% 0.06 175)' },
    gold: { '--pink':'oklch(74% 0.16 50)','--lavender':'oklch(72% 0.18 355)','--mint':'oklch(72% 0.14 90)','--bg':'oklch(98.5% 0.012 70)','--bg2':'oklch(96% 0.018 70)','--card':'oklch(99.5% 0.006 70)','--fg':'oklch(22% 0.025 40)','--muted':'oklch(58% 0.02 50)','--border':'oklch(90% 0.025 70)','--pink-light':'oklch(92% 0.07 50)','--lav-light':'oklch(90% 0.07 355)' },
    dark: { '--pink':'oklch(72% 0.20 355)','--lavender':'oklch(72% 0.20 290)','--mint':'oklch(72% 0.18 175)','--bg':'oklch(16% 0.025 290)','--bg2':'oklch(20% 0.025 290)','--card':'oklch(20% 0.025 290)','--fg':'oklch(92% 0.01 290)','--muted':'oklch(62% 0.02 290)','--border':'oklch(30% 0.03 290)','--pink-light':'oklch(30% 0.06 355)','--lav-light':'oklch(28% 0.06 290)' },
  };

  function applyTheme(name) {
    const t = themes[name]; if (!t) return;
    Object.entries(t).forEach(([k,v]) => document.documentElement.style.setProperty(k,v));
  }

  function initCustomCursor() {
    if (!window.matchMedia || !matchMedia('(pointer: fine)').matches) return;
    if (document.querySelector('.cursor-glow')) return;

    const glow = document.createElement('div');
    glow.className = 'cursor-glow';
    document.body.appendChild(glow);

    const interactiveSel = 'a,button,select,summary,label,input[type=range],input[type=checkbox],input[type=radio],input[type=submit],.swatch,.filter-btn,.tag,.social-btn,.link-btn,.post-card,.repo-card,.project-card,.article-row,.nav-card,.archive-card,.skill-card,.stat-card,.page-btn,.toc-item';

    let tx = -100, ty = -100, x = -100, y = -100;
    let visible = false;
    window.addEventListener('mousemove', e => {
      tx = e.clientX; ty = e.clientY;
      if (!visible) { x = tx; y = ty; visible = true; glow.style.opacity = '1'; }
    });
    window.addEventListener('mouseleave', () => { glow.style.opacity = '0'; visible = false; });
    window.addEventListener('mouseenter',  () => { glow.style.opacity = '1'; visible = true;  });

    document.addEventListener('mouseover', e => {
      if (e.target.closest && e.target.closest(interactiveSel)) glow.classList.add('is-hover');
    });
    document.addEventListener('mouseout', e => {
      if (e.target.closest && e.target.closest(interactiveSel) && !(e.relatedTarget && e.relatedTarget.closest && e.relatedTarget.closest(interactiveSel)))
        glow.classList.remove('is-hover');
    });
    window.addEventListener('mousedown', () => glow.classList.add('is-press'));
    window.addEventListener('mouseup',   () => glow.classList.remove('is-press'));

    const burstColors = [
      'oklch(82% 0.12 355)','oklch(85% 0.09 330)',
      'oklch(85% 0.08 290)','oklch(88% 0.06 355)'
    ];
    window.addEventListener('click', e => {
      const n = 6;
      for (let i = 0; i < n; i++) {
        const angle = (i / n) * Math.PI * 2 + Math.random() * 0.5;
        const dist  = 36 + Math.random() * 26;
        const el = document.createElement('div');
        el.className = 'click-petal';
        el.style.cssText =
          `left:${e.clientX}px;top:${e.clientY}px;` +
          `background:${burstColors[i % burstColors.length]};` +
          `--dx:${(Math.cos(angle)*dist).toFixed(1)}px;` +
          `--dy:${(Math.sin(angle)*dist).toFixed(1)}px;` +
          `border-radius:${Math.random()>.5?'50% 0 50% 0':'50%'};`;
        document.body.appendChild(el);
        setTimeout(() => el.remove(), 760);
      }
    });

    (function tick() {
      x += (tx - x) * 0.22;
      y += (ty - y) * 0.22;
      glow.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -50%)`;
      requestAnimationFrame(tick);
    })();
  }

  function initShared(defaults) {
    defaults = defaults || { petalCount:12, theme:'pink' };
    // load from localStorage
    try {
      const saved = JSON.parse(localStorage.getItem('blog_tweaks') || '{}');
      defaults = Object.assign({}, defaults, saved);
    } catch(e){}

    spawnPetals(defaults.petalCount || 12);
    applyTheme(defaults.theme || 'pink');
    initCustomCursor();

    // Tweaks toggle button
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
          panel.classList.remove('open'); toggle.classList.remove('open');
        }
      });
      document.addEventListener('keydown', e => {
        if (e.key === 'Escape') { panel.classList.remove('open'); toggle.classList.remove('open'); }
      });
    }

    // Edit-mode iframe bridge (kept for compatibility)
    window.addEventListener('message', e => {
      if (e.data?.type === '__activate_edit_mode')   panel?.classList.add('open');
      if (e.data?.type === '__deactivate_edit_mode') panel?.classList.remove('open');
    });
    try { window.parent !== window && window.parent.postMessage({type:'__edit_mode_available'}, '*'); } catch(e){}

    document.querySelectorAll('.swatch').forEach(s => {
      if (s.dataset.theme === defaults.theme) s.classList.add('active');
      s.addEventListener('click', () => {
        document.querySelectorAll('.swatch').forEach(x=>x.classList.remove('active'));
        s.classList.add('active');
        applyTheme(s.dataset.theme);
        try { const t=JSON.parse(localStorage.getItem('blog_tweaks')||'{}'); t.theme=s.dataset.theme; localStorage.setItem('blog_tweaks',JSON.stringify(t)); } catch(e){}
        window.parent.postMessage({type:'__edit_mode_set_keys', edits:{theme:s.dataset.theme}}, '*');
      });
    });

    const pc = document.getElementById('petal-count');
    if (pc) {
      pc.value = defaults.petalCount;
      pc.addEventListener('input', e => {
        const v=+e.target.value; spawnPetals(v);
        try { const t=JSON.parse(localStorage.getItem('blog_tweaks')||'{}'); t.petalCount=v; localStorage.setItem('blog_tweaks',JSON.stringify(t)); } catch(e){}
        window.parent.postMessage({type:'__edit_mode_set_keys', edits:{petalCount:v}}, '*');
      });
    }
  }

  window.initShared = initShared;
  window.spawnPetals = spawnPetals;
  window.applyBlogTheme = applyTheme;
})();
