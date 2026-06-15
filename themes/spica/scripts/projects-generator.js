'use strict';

const { readdirSync, readFileSync, existsSync } = require('fs');
const { join, basename, extname } = require('path');
const fm = require('hexo-front-matter');

function loadProjects(hexo) {
  const dir = join(hexo.source_dir, '_projects');
  if (!existsSync(dir)) return [];

  return readdirSync(dir)
    .filter(name => extname(name) === '.md')
    .map(name => {
      const filePath = join(dir, name);
      const parsed = fm.parse(readFileSync(filePath, 'utf8'));
      const data = parsed;
      const slug = data.slug || basename(name, '.md');
      const links = Array.isArray(data.links) ? data.links : [];
      const tags = Array.isArray(data.tags) ? data.tags : [];

      return {
        slug,
        title: data.title || slug,
        description: data.description || '',
        cover: data.cover || '',
        featured: !!data.featured,
        archive: !!data.archive,
        cat: data.cat || '',
        lang: data.lang || '',
        lang_color: data.lang_color || '#c4b5fd',
        stars: data.stars || '',
        tags,
        order: Number(data.order) || 0,
        links,
        primaryUrl: (links.find(l => l.primary) || links[0] || {}).url || '',
        url: (links.find(l => l.primary) || links[0] || {}).url || ''
      };
    })
    .sort((a, b) => {
      if (a.featured !== b.featured) return a.featured ? -1 : 1;
      if (a.order !== b.order) return a.order - b.order;
      return a.title.localeCompare(b.title, 'zh-CN');
    });
}

hexo.extend.helper.register('get_projects', function() {
  return loadProjects(hexo);
});

hexo.extend.helper.register('get_featured_project', function() {
  const items = loadProjects(hexo);
  return items.find(p => p.featured) || items[0] || null;
});

hexo.extend.helper.register('get_project_cards', function() {
  return loadProjects(hexo).filter(p => !p.featured && !p.archive);
});

hexo.extend.helper.register('get_project_archive', function() {
  return loadProjects(hexo).filter(p => p.archive);
});
