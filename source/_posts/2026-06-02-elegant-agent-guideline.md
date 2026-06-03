---
title: 如何优雅地使用 Agent
date: 2026-06-02 10:00:00
categories: [AI]
tags: [Agent, Prompt Engineering]
---

本文相关信息来源截止至 2026 年 6 月，后续可能会有变化，请以最新信息为准。

---

这篇文章属于半扫盲性质的 agent 使用说明，小白也可放心食用~

---

在聊如何使用之前，我们先来了解一些基础概念。

## Agent 是什么？

先来认识几个名词

> agent n. 代理人；代理商；经纪人\
> model n. 模型；模特儿；模范\
> harness v. 利用；驾驭；控制

所以可以简单地认为，agent 就是被系统驾驭的模型，能产出什么内容取决于两个方向，一是模型本身的能力，二是系统提供的能力。就好比电脑不但要有硬件，还要有操作系统和软件，才能发挥作用。

---

大部分人说的 ai 其实并没有区分 *model* 和 *agent*，举个例子，大部分人用的豆包，其实是一个 *agent*（可能能力较弱），它背后的模型是 `doubao-seed-2.0(-pro)`，在豆包这个软件中，模型拥有了更多的能力（比如记忆、工具调用等），所以能产出更多的内容。

---

目前常见的国外模型有：`claude, gemini, gpt`，有的可能还带有版本号，例如 `claude-opus-4.8`, `gpt-5.5-thinking`，`gemini-3.5-flash`。\
命名一般会暗示模型的强度，例如 flash 版本弱于 pro，版本号越新模型一般更强（但是有的厂商会开倒车）。当版本号和档位都不同时就不太好比，例如 `gemini-3.1-pro` 和 `gemini-3.5-flash`（我个人认为前者更强）。\
国内的相对来说种类更多，主流的有：`deepseek`, `glm`, `kimi`, `qwen` 和 `doubao`，相关的版本规范也和国外的差不多。

还有一些额外的后缀，类似 `-thinking`, `-it`，`-xxb-axb`, 等等，这里不做讲解，感兴趣的可以自行搜索。

---

再来谈谈有关的工具，也就是 harness，也可以称之为鞍，毕竟好马配好鞍。\
目前最火的 agent 工具应该是 anthropic 的 `claude code`，简称 `cc`，它本身只原生支持本系列模型，不过现在有工具能方便地接入其他模型了。\
还有类似 `codex`，`opencode`，`pi` 等一系列工具，这里就不一一举例。\
harness 为 skills 提供了接口，skills 则拓展了 AI 的能力，例如写 word, ppt 等文档，调用浏览器或其他软件的能力。

## 如何使用 agent？

这一部分我会分为三节：

- 如何安装 agent （安装 harness）
- 如何接入模型
- 如何拓展 agent 的能力（即安装 skills）

### 如何安装 agent

这里以 `claude code` 为例。

#### 自动安装方法（推荐）

打开任意一个你能连接国外网络的浏览器，搜索 `claude code`，进入官网，目前官网链接是 [这个](https://claude.com/product/claude-code)。官网的样子应该长成这样：

![Claude Code 官网首页](claude-official.png)

看到页面中间的命令了吗？把它复制下来，用你电脑上的终端（terminal），粘贴这行命令，回车，就能安装了！如果你是 Windows 用户，建议使用 PowerShell 终端，Mac 用户可以使用自带的 Terminal 终端。

Windows（PowerShell）：

```powershell
irm https://claude.ai/install.ps1 | iex
```

macOS / Linux：

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

**注意！** 永远不要运行你不知道来源的命令，尤其是涉及到 `iex` 和 `bash` 这样的命令，因为它会执行你输入的命令，所以一定要确认命令的来源和内容。这行命令是直接从官网复制的，可能会随着时间变化而变化，如果你发现官网上的命令和这里不一样，请以官网上的为准。

等待时间可能有点久，可以给终端设置代理加速，设置代理命令为

PowerShell：

```powershell
$env:HTTP_PROXY="your_proxy_url"; $env:HTTPS_PROXY="your_proxy_url"
```

cmd：

```bat
set http_proxy=your_proxy_url
set https_proxy=your_proxy_url
```

上述命令只在当前终端会话中生效，如果关闭终端后再打开，就需要重新设置一次。

#### 手动安装方法

如果自动安装方法遇到了问题，可以使用以下手动安装方法。

首先，安装 Node.js，官网链接是 [这个](https://nodejs.org/)，下载并安装最新的 LTS 版本。

安装完成后，打开终端，先验证 Node.js 是否安装成功，输入以下命令：

```bash
node -v
```

如果显示了版本号，说明 Node.js 安装成功了。接下来，安装 `claude code`，输入以下命令：

```bash
npm install -g @anthropic-ai/claude-code@latest
```

这一步需要的时间可能也比较久（问就是 nodejs 太臃肿），安装完成后，输入以下命令验证是否安装成功：

```bash
claude --version
```

如果显示了版本号，说明 `claude code` 安装成功了。

---

接下来要说明两个概念：**用户级** 和 **项目级**。

用户级（User-level）是指在本电脑用户范围内会生效的设置，工具或技能，此用户的所有项目都可以访问它们。用户级的配置位于 `~/.claude`（Windows 为 `%USERPROFILE%\.claude`） 目录下。

项目级（Project-level）是指在特定项目范围内会生效的设置，工具或技能，只有这个项目能访问它们。项目级的配置位于项目根目录下的 `.claude` 目录中。

### 如何接入模型

目前，国内想要用到完整的 claude/gpt/gemini 模型相对困难，这里给出用 deepseek 官方模型的示例，其他模型的接入方法大同小异。

**本教程不会对任何所谓 *中转站* 做出推荐和评价，也不会为任何因使用中转站而导致的问题负责，请自行判断和选择。**

#### 安装 cc-switch （推荐）

cc-switch 相当于一个懒人包，可以方便接入各种模型。可自行前往 [Github Release](https://github.com/farion1231/cc-switch/releases/latest) 页面下载对应版本并安装。

首先，打开 deepseek 的官网，注册并登录你的账号。进入 [API 开放平台](https://platform.deepseek.com/)。

![DeepSeek API 开放平台](deepseek-official.png)

所有 agent 工具的使用都需要用到 API key，这个和网页端使用不一样，不是免费的！（可能有免费渠道，但请自己甄别）你必须先给模型厂商充值，才能使用 API key 调用他们的模型。一般 deepseek 充值 10 块左右就能用很久（按照用途不同，使用时间也不同）。

然后在 [API keys](https://platform.deepseek.com/api_keys) 页面，选择创建 API key，输入名字（比如 claude-code），点击创建，就能看到你的 API key 了。

![创建 DeepSeek API key](api-key.png)

**注意！** deepseek 的 API key 只在创建时可见，所以一定妥善保管，如果不小心丢失了，请立即删除此 API key 并重新创建。API key 是非常敏感的信息，一旦其他人获得了 API key，他们就能使用你的余额访问模型，就好比你的银行卡密码一样，千万不要泄露给任何人，包括我！如果不小心泄露了，请立即删除此 API key 并重新创建。

如果你下载了 cc-switch，打开软件后点击右上角加号，在预设供应商中选择 `DeepSeek`，一般除了 API key 以外的部分不需要修改，只要把你的 API key 填入对应位置即可，如果其他地方的设置有问题，可以按照我图片中的内容填。

![cc-switch](cc-switch-official.png)

![cc-switch](cc-switch-2-official.png)

如果没有下载此工具，有以下两种方式接入模型：

#### 临时配置

在终端中输入以下命令

Linux / macOS：

```bash
export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
export ANTHROPIC_AUTH_TOKEN=<你的 DeepSeek API Key>
export ANTHROPIC_MODEL=deepseek-v4-pro[1m]
export ANTHROPIC_DEFAULT_OPUS_MODEL=deepseek-v4-pro[1m]
export ANTHROPIC_DEFAULT_SONNET_MODEL=deepseek-v4-pro[1m]
export ANTHROPIC_DEFAULT_HAIKU_MODEL=deepseek-v4-flash
export CLAUDE_CODE_SUBAGENT_MODEL=deepseek-v4-flash
export CLAUDE_CODE_EFFORT_LEVEL=max
```

Windows（PowerShell）：

```powershell
$env:ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
$env:ANTHROPIC_AUTH_TOKEN="<你的 DeepSeek API Key>"
$env:ANTHROPIC_MODEL="deepseek-v4-pro[1m]"
$env:ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-pro[1m]"
$env:ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-pro[1m]"
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
$env:CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"
$env:CLAUDE_CODE_EFFORT_LEVEL="max"
```

**注意！** 这种方式只在当前终端会话中生效，如果关闭终端后再打开，就需要重新设置一次。

#### 永久配置

更推荐直接写入 Claude Code 的配置文件 `~/.claude/settings.json`（Windows 为
  `%USERPROFILE%\.claude\settings.json`）。在文件中添加以下内容：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "<你的 DeepSeek API Key>",
    "ANTHROPIC_MODEL": "deepseek-v4-pro[1m]",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-v4-pro[1m]",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-v4-pro[1m]",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-v4-flash",
    "CLAUDE_CODE_SUBAGENT_MODEL": "deepseek-v4-flash",
    "CLAUDE_CODE_EFFORT_LEVEL": "max"
  }
}
```

### 如何拓展 agent 的能力

Agent 的能力拓展主要是通过安装 skills 来实现的，skills 就好比是 agent 的插件，可以让 agent 拥有更多的功能，例如写文档、调用浏览器等。

skills 分全局级和项目级，具体把一个 skill 安装在全局还是项目取决于其功能范围。

全局 skill 的安装位置是：`~/.claude/skills`。\
项目级 skill 的安装位置是：项目根目录下的 `.claude/skills`。

目前，claude code 已经内置了很多实用的 skills，这些 skills 可以直接使用，无需额外安装。

如果你想要安装第三方的 skills，可以通过以下方式进行安装：

#### 利用 SKillpkg 安装和管理

这里推荐一个朋友写的 [skill 管理工具](https://github.com/Richardlxr/SKillpkg)，可以按照手册进行管理和安装。

#### 手动安装

~~其实可以让你已经安装的 agent 帮你下载和管理 skills~~

一般的 skills，以 [superpowers](https://github.com/obra/superpowers) 为例，会提供完整的安装说明。

如果没有提供安装说明，一般来说，下载后把 skills 文件夹放到 `~/.claude/skills` 目录下就可以了。

第一次安装 skills 可能没有这个文件夹，自行创建即可。

一个 skills 的文件结构一般是这样的：

```text
技能名字/
├── SKILL.md // 技能入口，一般 agent 会尝试发现此文件来预加载 skills
├── requirements.txt // 有的技能可能涉及运行 python 脚本，这个文件会列出所需的 python 包
├── references/ // 参考资料，可能包含一些文档、图片等
└── scripts/ // 可能包含一些脚本文件，例如 python 脚本等
```

安装完成后，可以在 agent 中调用 `/skills` 命令查看 skills 的安装状态。一般通过 `/技能名字` 就可以调用技能。

![skill-list](skill-list.png)
![use-a-skill](use-a-skill.png)

**注意！** 在安装第三方 skills 时，一定要确认 skills 的来源和内容，尤其是那些包含脚本文件的 skills，因为它们可能会执行一些你不希望执行的操作，所以一定要谨慎选择和安装 skills。

### 总结

这是一份简易的关于如何安装和使用 agent 的指南，主要以 `claude code` 为例进行说明，其他 agent 工具的使用方法大同小异。如果有帮助，可以[点一个 star](https://github.com/MisterRabbit0w0/misterrabbit0w0.github.io) 支持一下。
