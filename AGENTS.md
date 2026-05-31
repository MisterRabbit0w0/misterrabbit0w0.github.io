# AGENTS.md

This file gives coding agents practical guidance for working in this repository.

## Project Overview

This repository is a personal Hexo static blog deployed to GitHub Pages at:

<https://misterrabbit0w0.github.io>

The site is Chinese-language, configured for `Asia/Shanghai`, and authored as
`织星 / Astraea`. The active theme is the custom `hoshino` theme.

## Tech Stack

- Hexo `8.x`
- Node.js `20` in GitHub Actions
- EJS templates
- CSS and client-side JavaScript in the theme
- Markdown posts with YAML front matter
- GitHub Actions deployment to `gh-pages`

## Common Commands

Run commands from the repository root.

```bash
npm install
npm run server
npm run build
npm run clean
npm run deploy
```

Command behavior:

- `npm run server` starts `hexo server`, usually at <http://localhost:4000>.
- `npm run build` runs `hexo generate` and writes static output to `public/`.
- `npm run clean` removes Hexo generated cache/output.
- `npm run deploy` runs Hexo deployment if deployment is configured.

Content helpers:

```bash
npx hexo new "标题"
npx hexo new page "名称"
npx hexo new draft "标题"
```

## Repository Layout

- `_config.yml` - main Hexo site configuration.
- `source/_posts/` - blog posts.
- `source/_projects/` - project entries for the projects page.
- `source/about/`, `source/projects/`, `source/categories/`, `source/tags/` - site pages.
- `source/img/` - images and favicon assets.
- `scaffolds/` - Hexo templates for new posts/pages/drafts.
- `themes/hoshino/` - active custom theme.
- `themes/hoshino/_config.yml` - active theme configuration.
- `themes/hoshino/layout/` - EJS layouts and partials.
- `themes/hoshino/source/css/` - theme styles.
- `themes/hoshino/source/js/` - theme scripts.
- `themes/hoshino/scripts/` - Hexo generators and helpers.
- `.github/workflows/build.yml` - GitHub Pages build and deployment workflow.
- `public/` - generated site output; do not edit manually.

## Configuration Notes

- Active theme: `hoshino`.
- Site language: `zh-CN`.
- Timezone: `Asia/Shanghai`.
- Permalink format: `:year/:month/:day/:title/`.
- Posts per page: `10`.
- Syntax highlighter: `highlight.js`.
- Feed output: `atom.xml`.
- Sitemap output: `sitemap.xml`.

## Development Guidelines

- Keep edits scoped to the requested change.
- Prefer existing Hexo and theme conventions over adding new tooling.
- Keep generated files out of manual edits, especially `public/` and Hexo caches.
- Put reusable theme behavior in `themes/hoshino/source/js/shared.js`.
- Put global styling in `themes/hoshino/source/css/shared.css`.
- Put page-specific styling in `themes/hoshino/source/css/page.css`.
- Keep EJS templates readable and avoid moving presentation logic into content files.
- Preserve the Chinese site voice and existing author identity.
- Use ASCII for code unless the surrounding file or user-facing Chinese content calls for non-ASCII text.

## Content Guidelines

- Blog posts live under `source/_posts/`.
- Use YAML front matter consistently.
- New posts should normally use Hexo scaffolds instead of hand-building front matter.
- Article content is licensed as `CC BY-NC-SA 4.0`; code is MIT licensed.
- Be careful when changing image credits, external image APIs, or attribution text in `README.md`.

## Validation

Before claiming a change is complete, run the most relevant check:

```bash
npm run build
```

For styling or template changes, also run:

```bash
npm run server
```

Then inspect the affected page in a browser when possible. Pay particular attention to:

- home page rendering
- post page rendering
- archive/category/tag pages
- about/projects pages
- mobile layout
- theme switching
- 404 page behavior

## Deployment

Pushing to `main` triggers `.github/workflows/build.yml`.

The workflow:

1. checks out the repository
2. installs Node.js `20`
3. runs `npm install`
4. runs `npm run clean`
5. runs `npm run build`
6. publishes `public/` to `gh-pages`

Do not commit secrets or deployment tokens. The workflow uses `GITHUB_TOKEN`.

## Git Hygiene

- Check `git status --short` before and after substantial edits.
- Do not revert unrelated user changes.
- Do not use destructive git commands unless the user explicitly asks for them.
- Leave untracked or unrelated local directories alone unless they are part of the task.

## Agent-Specific Notes

- Read `CLAUDE.md` as the detailed project context; this file is the cross-agent summary.
- If `CLAUDE.md` and this file differ, prefer the more specific instruction for the current task.
- If changing behavior, update documentation when it would prevent future confusion.
- If dependencies or Hexo plugins change, update `package.json` and `package-lock.json` together.
