---
title: 如何优雅地使用 Git
date: 2026-06-27 10:00:00
categories: [tools]
tags: [git, workflow, best practices]
---

本篇为 Git 新手教程，与其他教程直接扔一大堆命令上来并讲解用法不同，我会从新建一个项目开始，展示在真正的开发和多人协作场景下怎么使用 git。

本文使用的示例仓库地址：[calculator](https://GitHub.com/MisterRabbit0w0/calculator)

虽然有了 AI 之后，大部分的分支管理和仓库维护都交给了 AI，但是 git 的使用仍然显得尤为重要，在 AI 无法解决的合并场景或误删库恢复的情况下，必须要人工的介入。

谨以此文致敬仍然在古法编程的开发者们。

---

## 安装 Git

~~最简单的一步~~

[官网链接](https://git-scm.com/)

~~好吧其实可能也没那么简单~~

以下是一份给 Windows 用户在安装过程中使用的各项设置的速查表:

1. Select Components (选择组件)

各选项含义:

+ `Additional icons -> On the Desktop`: 在桌面创建快捷方式。

+ `Git Bash Here / Git GUI Here`: 右键文件夹时，直接打开 Git 命令行(Bash)或图形界面(GUI)。

+ `Check daily for Git for Windows updates`: 每天检查更新。

+ `Add a Git Bash Profile to Windows Terminal`: 将 Git Bash 加入到 Windows 新终端里。

一般保持默认(勾选 Git Bash Here 即可)，其他可按需选择。

2. Choosing the default editor (选择默认编辑器)

各选项含义: 当 Git 需要你输入提交信息(Commit Message)或合并冲突时，默认唤起哪一个文本编辑器。

选项包括: Vim、VS Code、Notepad++ 等。

推荐选择: Notepad++ 或内置的 Nano，VS Code 功能丰富但启动相对较慢，如果不介意仍可选择。

3. Adjusting the name of the initial branch (调整初始分支名称)

各选项含义: 决定你执行 git init 新建仓库时，主分支的名字叫什么。

+ `Let Git decide`: 让 Git 决定，默认叫 master。

+ `Override the default branch name for new repositories`: 自定义名称。

推荐选择: 选择第二项(Override)，并在下方输入 main。现在大部分仓库都将默认分支改为了 main。

4. Adjusting your PATH environment (调整环境变量)

各选项含义: 决定你在哪些终端里可以使用 git 命令。

+ `Use Git from Git Bash only`: 只能在自带的 Git Bash 里用。

+ `Git from the command line and also from 3rd-party software`: 允许在 Windows 终端以及第三方软件(如 VS Code)中使用 Git。

+ `Use Git and optional Unix tools from the Command Prompt`: 把一些 Linux 命令也强行加到 Windows 终端里，此项可能影响终端正常功能。

推荐选择: 第二项。

5. Choosing the SSH executable (选择 SSH 执行文件)

各选项含义: Git 连接 GitHub 等远程服务器时，使用哪个 SSH 工具。

+ `Use bundled OpenSSH`: 使用 Git 自带的 OpenSSH。

+ `Use external OpenSSH`: 使用 Windows 系统自带的或第三方的 OpenSSH。

推荐选择: `Use bundled OpenSSH`

6. Choosing HTTPS transport backend (选择 HTTPS 传输后端)

各选项含义: Git 在通过 HTTPS 协议拉取/推送代码时，如何验证安全证书(SSL/TLS)。

+ `Use the OpenSSL library`: 使用开源的 OpenSSL 库验证。

+ `Use the native Windows Secure Channel library`: 使用 Windows 系统的证书管理中心。

推荐选择: 如果仅为个人开发者，选择 `Use the OpenSSL library`，如果有特殊证书要求，可以选择 `Use the native Windows Secure Channel library`。

7. Configuring the line ending conversions (配置行尾换行符转换)

各选项含义: Windows 的换行符是 CRLF(\r\n)，而 Linux/Mac 的换行符是 LF(\n)。跨平台协作时，格式不一会导致 diff 识别错误。

+ `Checkout Windows-style, commit Unix-style`: 拉取代码时自动转成 Windows 的 CRLF，提交时自动转回 Linux 的 LF。

+ `Checkout as-is, commit Unix-style`: 拉取时维持原样，提交时转成 LF。

+ `Checkout as-is, commit as-is`: 拉取和提交都保持原样。

推荐选择: 第一项。

8. Configuring the terminal emulator to use with Git Bash (配置 Git Bash 终端模拟器)

各选项含义: 打开 Git Bash 命令行时，用什么窗口外壳。

+ `Use MinTTY`: 使用 Linux 风格的 MinTTY 窗口。

+ `Use Windows' default console window`: 使用 Windows 窗口。

推荐选择: `Use MinTTY`。

9. Choose the default behavior of git pull (选择 git pull 的默认行为)

各选项含义: 当你执行拉取代码命令时，如果本地和远程有冲突，该怎么合并。

+ `Default (fast-forward or merge)`: 尝试自动合并，必要时生成一个 Merge 提交节点。

+ `Rebase`: 变基合并两个版本代码。

+ `Only ever fast-forward`: 只允许当前推送代码新于远端代码，如果有冲突则不执行合并。

有关合并，变基和版本的内容，我们会在后面讨论。

推荐选择: Default (fast-forward or merge)。

10. Choose a credential helper (选择凭据管理器)

各选项含义: 管理代码托管平台的账号密码。

+ `Git Credential Manager`: 微软官方提供的跨平台凭据管理器，支持网页弹窗授权登录。

+ `None`: 不使用。每次需要授权的时候使用交互式窗口请求登录。

推荐选择: Git Credential Manager。

11. Configuring extra options (配置额外选项)

各选项含义:

+ `Enable file system caching`: 开启文件系统缓存，可以显著提升大项目的运行速度。

+ `Enable symbolic links`: 开启符号链接，需要 Windows 开发者权限。

推荐选择: 勾选 `Enable file system caching`。

## 配置 Git 基本信息

首先先检查 Git 是否安装成功，打开终端输入：

```bash
git --version
```

如果正常输出版本号，没有类似找不到命令的报错，则说明安装成功。

然后是基本的署名信息，可以使用 `git commit -s` 使得提交带有署名信息，这对部分开源项目是必须的。

```bash
git config --global user.name "Your Name"
git config --global user.email "Your Email"
```

## 实战操作

~~多标题致歉~~

我会用一个简易计算器的项目来展示相关操作，这个项目实现功能很简单，就是让用户输入一个算式，然后输出结果，暂时只考虑加减乘除和乘方五种运算，面向对象设计，相关算法包括中缀转后缀表达式。

### 创建仓库

Git 的操作对象是仓库，仓库可以理解为一个包含了特殊文件夹 `.git` 的文件夹，在这个特殊的文件夹下包含了此项目的所有历史和分支信息。

如果是从头开发一个自己的项目，可以新建一个文件夹/在一个空文件夹下使用命令

```bash
git init
```

以创建一个新的 Git 仓库。

如果是从远程仓库克隆一个已有的项目，可以使用命令

```bash
git clone <远程仓库地址>
```

![](initial-repo.png)

### 修改仓库内容

文件夹成为仓库之后，Git 会自动追踪文件(夹)的变化，可以通过

```bash
git status
```

查看仓库的状态。

比如现在，我写了一个最基础的能计算加减运算的计算器(代码就不贴了，想必大家都会)，命名为 `add_and_sub.cpp`，然后我在仓库里执行 `git status`，就会看到类似下面的输出：

![](git-status.png)

Untracked files 这一部分意味着 Git 发现了一个新文件，但是在历史中没有它的记录。

此时，可以执行 `git add .` 将所有新文件添加到暂存区，或者用 `git add <fileName>` 添加指定名称文件，此后，该文件就会被标记为已跟踪。

---

暂存区的概念：本地的工作区可以简单地认为是电脑上编辑文件可被 Git 发现的那个区域，仓库则是 Git 保存所有提交的记录的地方。暂存区相当于这两者之间的一个缓存区，你在此处的更改仍然还没有被记录到仓库中。

---

加入暂存区后，再次执行 `git status`，就会看到类似下面的输出：

![](git-status-add.png)

`Changes to be committed` 意味着下面的内容已经被添加到暂存区，等待提交。

### 提交更改

提交(commit)是版本管理和协作中最重要的一步操作。

一般建议在 commit 之前，先执行 `git status` 查看当前仓库状态，确认要提交的内容。

最简单的提交命令是 `git commit -m "提交信息"`。

`-m` 参数后面跟的内容是本次提交的说明信息，建议使用 [Conventional Commits 规范](https://www.conventionalcommits.org/en/v1.0.0/)。

例如，对于这一次的修改，我的提交写成

```bash
git commit -m "feat: Add calculator class with expression evaluation"
```

提交后，再次运行 `git status`，应该提示

![](git-status-commit.png)

提交历史可以执行 `git log` 查看

![](git-log.png)

commit 后面那一长串，是 Git 自动生成的提交哈希，用于唯一标识一次提交。后续我们会讲到删除/修改提交历史，如何通过哈希回退到某一次提交。

`HEAD ->` 表示指向当前分支。`master` 是分支名字，指向最新的提交。有关 `HEAD`, `ref`, `:`, `~`, `^` 等特殊名词和符号，我们会在后续分别介绍。

`git commit` 有一个比较简便的用法 `git commit -am "提交信息"`，它会自动将所有已跟踪的文件加入暂存区并提交，但是不会添加未跟踪的文件到暂存区。

另外一个用的多的功能是 signoff，可以在提交时加上 `-s` 参数，表示你同意本次提交的内容，并且愿意为此负责。加入签名参数的提交信息末尾会自动生成

```
Signed-off-by: Your Name <Your Email>
```

此功能一般用于大型开源项目代码追责。

### 分支操作

可以看到，我的 Git 在新建仓库时，会使用 `master` 作为默认分支名称，但是现在大部分仓库都将默认分支改为了 `main`，所以我需要迁移分支。

```bash
git branch -m master main
```

`-m` 参数表示重命名分支。

分支操作是 Git 的核心功能之一，分支表示从某一次提交开始，新建一个独立的开发栈，在此分支上的提交不会影响其他分支。

一般，多人协作开发时，不同的人领取到不同任务的情况下，会在主分支上新建一个分支，完成任务后再合并回主分支，易于解决代码冲突和版本管理问题。

例如，我现在想给计算器加入一个乘除运算功能，而另一个人想给计算器加入一个乘方运算功能，那么我们就可以在主分支上新建两个分支，分别开发这两个功能，最后再合并回主分支。

我们可以使用 `git branch <分支名称>` 新建一个分支，使用 `git checkout <分支名称>` 签出到该分支。

现在 Git 提供了语义更明确的命令 `switch` 用于切换分支。

更快捷的方式是一步创建并切换

```bash
git switch -c <新分支名>   # -c 代表 create
git checkout -b <新分支名> # -b 代表 branch
```

我创建的分支是 `mul_and_div`，另外一个人创建的分支是 `pow`。

可以使用 `git branch` 查看当前仓库分支情况。

![](git-branch.png)

`*` 表示当前所在分支。

最好，分支命名也遵循 [Conventional Branch 规范](https://conventionalbranch.org/zh/)，例如 `feat/mul-and-div`，`feat/pow`。

### 推送代码

在此之前，需要先明确远程仓库的概念。

远程仓库是指托管在 GitHub、GitLab、Gitee 等平台上的仓库，通常用于多人协作开发或发行版本等功能。

commit 之后，代码只会保存在本地仓库中，如果想要将代码推送到远程仓库，需要使用 `git push` 命令。

新仓库一般不带有远程仓库信息，需要通过 `git remote add <仓库名称> <远程仓库地址>` 添加远程仓库信息，一般仓库名称选择 `origin`。

远程克隆的仓库，会自动创建一个名为 `origin` 的远程仓库地址，指向克隆的远程仓库。

如果此远程仓库是一个其他项目的 fork (fork是一个对仓库的操作，用于创建一个现有仓库的副本并进行修改)，那么其通常还带有一个 `upstream` 信息，指向原始仓库。

可以通过 `git remote -v` 查看当前仓库的远程仓库信息。

![](git-remote-v.png)

我关联了我的私人仓库，所以隐藏了仓库地址。

第一行 `fetch` 的含义是，`git pull/git fetch` 命令会从此拉取代码。

第二行 `push` 的含义是，`git push` 命令会将代码推送到此。

一般这两个地址是一致的，如果在某些特殊情况下，可能要从一个只读仓库中获取代码，推送到另一个 fork 或私人仓库中，可以使用

```bash
git remote set-url --push origin <新的push地址>
```

修改 `push` 地址，对 `fetch` 的修改同理。

首次提交代码到远程仓库时，需要指定远程仓库和分支名称，例如：

```bash
git push <远程仓库名称> <远程分支名称>
```

推送时，可能被拒绝

![](git-push-fail.png)

这是因为 Git 在推送时会检查仓库提交历史，如果本地提交历史落后于远程仓库，Git 会拒绝推送。

这里有一个快进(fast-forward) 的概念。**快进** 是指，你要推送的代码的本地的提交历史包含了远程仓库的所有提交历史，并且在此基础上新增了提交。

遇到落后于远程仓库的情况时，可以先拉取远程仓库的代码，再推送。

```bash
git pull <远程仓库名称> <远程分支名称>
git push <远程仓库名称> <远程分支名称>
```

或者当你确定本地的提交历史是正确的，可以使用 `--force` 强制推送。

```bash
git push --force <远程仓库名称> <远程分支名称>
```

一般在本地和远程同时新建仓库会遇到一些兼容问题，因为两者的提交历史完全不一样，所以一般可以用 `--force`，但对已经在开发/公用分支上尽量不要用此危险命令。

### 拉取/更新代码

这个一般用在多人开发中，有人已经完成了开发->提交->推送->合并分支的全流程后，代码比本地的新。需要在最新的版本上继续开发。

分三种情况

+ 本地严格落后于远程仓库的提交历史，且暂存区干净，没有不同于远程仓库的提交。

  这种情况可以直接使用 `git pull` 拉取代码。

+ 本地严格落后于远程仓库的提交历史，但是暂存区有修改。

  有两种解决方案，一种是使用 `git stash` 将暂存区的修改保存起来，拉取代码后再恢复。

  `git stash` 是个很有趣的命令，我们稍后会来讲讲。

  第二种方法，如果本地的开发已经完成的差不多了，可以将暂存区提交，转化成第三种情况

+ 本地有不同于远程仓库的提交历史。

  这种情况比较有意思。理论上来讲，任何新功能的开发都不应该直接在 main 上进行，所以现实中出现的情况应该类似于

  ```
A---B---C  origin/main
     \
      D---E  feat/mul-and-div
  ```

  本地 `main` 停留在 `B`，远程仓库的 `main` 已经更新到了 `C`，而本地的分支 `feat/mul-and-div` 已经提交了两个新的提交 `D` 和 `E`。

  这种处理一般是：先更新本地的 `main`，然后再将功能分支接在新的 `main` 上。

  ```bash
  git switch main # 保证回到 main 分支
  git pull origin main # 拉取远程 main 分支的最新代码
  git switch feat/mul-and-div # 切换回功能分支
  git rebase main # 将功能分支接在新的 main 分支上
  ```

  `rebase` 命令的含义是，将当前分支的提交历史，重新应用到指定分支的最新提交之后。

  在上述例子中，`git rebase main` 会将 `feat/mul-and-div` 分支上的提交 `D` 和 `E` 重新应用到 `main` 分支的最新提交 `C` 之后，提交图变成了

  ```
  A---B---C  main / origin/main
           \
            D'---E'  feat/mul-and-div
  ```

  `D'` 和 `E'` 是 `D` 和 `E` 的重新应用后的提交，如果没有冲突，提交的实际内容和 `D`，`E` 一样，但是哈希不同，Git 会把它们当作不同的提交，所以 **一定** 不能在公共分支上使用 `rebase` 命令，否则会影响所有人的提交历史。

拉取分 `pull` 和 `fetch` 两种方式

+ `fetch`: 拉取远程仓库的最新提交历史到本地，但是不自动执行合并，需要自己手动处理落后的分支和新建的分支。
+ `pull`: 拉取远程仓库的最新提交历史到本地，并自动执行合并，可能会产生冲突，需要自己手动解决。

### 合并分支

合并分支是指将一个分支的修改合并到另一个分支中。常见的合并方式有两种: `merge` 和 `rebase`。

`rebase` 已在 [上文](#拉取-更新代码) 介绍过，是一种把提交历史整理成线性的方式。

`merge` 则是完整地记录不同分支上的提交历史，可以清晰地展现多人开发不同分支的图景。

```
A---B---E-------F  main
     \         /
      C---D---   feat/mul-and-div
```

如上图，分支 `feat/mul-and-div` 在 `B` 处被签出，加入了两个新提交，同时 `main` 上也有一个新提交 `E`，在 `main` 分支上执行 `git merge feat/mul-and-div`，会生成一个新的 `merge` 提交 `F`，合并两个分支的修改。

### 处理冲突

在多人协作环境下，可能在不同分支中，会修改了同一文件的同一部分内容，Git 无法处理这种冲突，需要人工介入。

例如我们的例子，我先合并了 `feat/mul-and-div` 的修改

```bash
git switch main
git merge feat/mul-and-div
```

![](git-merge-no-conflict.png)

这是一次常见的快进提交。

在这之后，另外一个人完成了 `feat/pow` 分支的开发，他先获取最新的 `main` 分支代码

```bash
git switch main
git pull origin main
```

然后他需要先 `rebase` 他的 `feat/pow` 分支到最新的 `main` 分支上

```bash
git switch feat/pow
git rebase main
```

这时，他发现

![](git-rebase-conflict.png)

更新受阻，产生冲突。我们使用 `git status` 查看冲突文件

![](git-status-conflict.png)

此时，用代码编辑器打开你的冲突文件，会看到

![](git-rebase-conflict-code.png)

注意到这是对同一个函数的修改，Git 无法自动选择，需要人工介入。

手动解决冲突后，执行 `git add <冲突文件>`，让 Git 知道冲突已经解决，再执行 `git rebase --continue` 继续 rebase 过程。

![](git-rebase-continue.png)

这个图片是因为我们没有填写 commit message，还记得吗？ `merge` 和 `rebase` 都要重新生成提交节点，commit message 是必须的。自动处理的 `merge` 和 `rebase` 会自动生成信息。

`rebase` 过程可能会遇到多次冲突，需要重复上述操作，直到 rebase 完成。

`rebase` 成功后，能看到类似

![](git-rebase-success.png)

此时，我们解决完冲突，可以将 `feat/pow` 分支的修改合并到 `main` 分支上。

```bash
git switch main
git merge feat/pow
```

就不会再遇到冲突。

一般合并完功能后，旧分支一般不保留，可以选择关闭或者删除

```bash
git branch -d feat/mul-and-div
git branch -d feat/pow
```

但图谱上仍然会保留这些分支的提交历史。很多分支的生命周期都停留在本地，除非是一个多人分工的分支，持续几个小时或几天。

### 撤销/修改操作

这是我认为对新手而言最重要的一个部分，Git 的强大历史和分支管理系统让你的每次提交历史都有迹可循，所以如果你错误提交了一些东西，比如你的个人隐私信息到公开仓库中，最正确的选择不是通过新提交去覆盖错误文件，因为其他人仍然能通过提交历史找到你的隐私信息；而是通过修改提交历史的方式彻底删除这些信息。

**注意！** 一旦你泄露了你的隐私密钥 (例如api-key或token)，最正确的方案是先更换密钥，再来处理历史。

这一块分为多个小节

#### 我不再想要我的工作区修改

在修改之后代码出现了问题，想要撤销修改，使用 `git restore <文件名>` 可以撤销工作区的修改，回退到 `HEAD` 的状态。关于 `HEAD` 指针，我们已经提到，其是一个表明 Git 所在位置的指针，其指向一个分支，如 `HEAD -> main`，当前分支指向该分支的最新提交。其还可以处于游离态 (detached)，即不指向任何分支，而是指向某一次提交。

`git restore` 并不针对某几行或某几次修改，请注意。

#### 我把错误的文件添加到了暂存区

Git 追踪文件行为受到 `.gitignore` 影响。

---

`.gitignore` 的核心语法和常见写法如下。规则以行为单位

```
# `#` 开头为注释，会被忽略；空行也会被忽略

.env # 忽略所有名为 `.env` 的文件，无论其位于何处
node_modules/ # 忽略所有名为 `node_modules` 的目录，无论其位于何处
/test.py # 忽略根目录下的 `test.py` 文件，开头 `/` 表示仅在根目录下搜寻
*.log # 忽略所有 `.log` 文件
password?.txt # 忽略所有 `password1.txt`、`password2.txt` 等文件，`?` 表示匹配任意单个字符
image[12].png # 忽略 `image1.png` 和 `image2.png`，`[]` 表示匹配括号内的任意一个字符
/logs/**/*.log # 忽略 `logs` 目录下的所有子目录中的 `.log` 文件，`**/` 表示匹配任意层级的子目录
!config.json # 取消忽略 `config.json` 文件，`!` 表示取反
```

`.gitignore` 可以放置在任意目录下，生效范围是它所在的文件夹以及该文件夹下的所有子文件夹，子目录下的 `.gitignore` 会覆盖父目录的规则。

`.gitignore` 只会影响其修改之后开始被追踪的文件，已被追踪文件不受影响，如果要取消追踪，可以使用 `git rm --cached <文件名>`，该命令会将文件从暂存区移除，但不会删除工作区的文件。

---

所以，从项目的一开始就建立好的 `.gitignore` 文件非常重要。

误提交到暂存区的文件，可以使用 `git restore --staged <文件名>` 将其从暂存区移除，回到工作区。

#### 我需要修改提交的内容

此处要使用 `git commit --amend` 命令。这步操作会将暂存区的修改和最近一次提交合并成一个新的提交，原来的提交会被覆盖，你需要写入新的提交信息。

#### 我需要撤销提交

`git reset` 命令可以撤销提交，分为三种模式：soft, mixed 和 hard。

soft 模式只回退 `HEAD` 指针， mixed 模式回退 `HEAD` 指针并清空暂存区，hard 模式回退 `HEAD` 指针、清空暂存区并清空工作区。在使用 `hard` 参数时一定要当心。

使用 `git reset HEAD~1` 可以撤销最近一次提交，默认为 `mixed` 模式，`HEAD~1` 表示最近一次提交的父提交。

---

这步操作最好仅作用在未推送的提交上，如果已经推送到远程仓库，建议使用 `git revert` 命令，生成一个新的提交来撤销之前的提交，而不是修改历史。

通常 `git revert HEAD` 用来生成一个“撤销最近提交”的提交。

`git revert HEAD~3..HEAD` 用来生成一个“撤销最近三次提交”的提交。`A..B` 的含义是 (A,B]，左开右闭。

`git revert <commit_hash>` 用来生成一个“撤销指定提交”的提交，它不会删除原提交，也不会移动分支指针。

`git revert -m 1 <merge_commit_hash>` 用来生成一个“撤销指定合并提交”的提交，`-m` 参数指定了保留哪一个父分支的修改。`-m 1` 通常表示保留合并时当前分支这一侧的历史，撤销合并进来的分支的修改。

`--no-commit` (`-n`) 参数可以让你在生成撤销提交后，先不提交，方便你在撤销的基础上做一些修改。

### 分发版本

或许大部分人用的最多的 GitHub 功能应该是直接在 Release 中找已编译好的二进制文件，下面会介绍如何使用 Git 进行版本分发。

Git 本身带有一个 `tag` 功能用来管理版本，使用 `git tag <版本名称>` 用于在当前 HEAD 指针处创建一个特定名称的版本。

有关版本命名规范，可见 [semver](https://semver.org/)。

使用 `git push origin <版本名称>` 推送本地 tag。

如何自动化地创建二进制文件，对 GitHub 来说，需要管理一个 `.GitHub/workflow` 的文件夹。GitHub 的工作流接受 yml 样式的任务清单。

## 写在最后

Git 其实是一个很强大的工具，有的教程为了简化，把很多东西说得太轻；有的教程类似条目说教的我也不太喜欢，因此有了这篇文章。

GitHub 同样也是一个很有用的网站，不只是开源工具分享平台，同样也是很方便的代码托管和管理平台。

Git 不是一个需要一次性背完的工具。很多命令平时用不到，但当代码合并失败、提交写错、文件误删、版本需要回退时，知道它们存在，就已经能让人安心很多。在 AI 已经能帮我们写很多代码的现在，Git 反而更重要了。因为代码可以生成，但历史需要管理；功能可以补全，但协作需要边界；错误可以修复，但我们要知道怎么安全地回到正确的位置。版本管理存在的意义，就是允许我们在试错中继续前进。

本文可能会不定期更新，用来补充我可能实际积累的一些经验。

希望你在下一次推送时，能更充满底气。
