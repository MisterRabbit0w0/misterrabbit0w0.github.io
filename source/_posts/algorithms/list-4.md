---
title: 第四章
date: 2026-02-25
disableNunjucks: true
tags:
  - 算法
---

# 第四章 图算法

## 第31节 遍历
### 301 深度优先搜索（递归）

深度优先搜索（DFS）是一种经典的图遍历算法，它在回溯之前会尽可能深入地探索图。你选择一个起始节点，沿着一条路径尽可能远地走，只有当走到死胡同时才返回。它是图算法中的“先深入探索，再考虑其他路径”。

#### 我们要解决什么问题？

DFS 帮助我们系统地访问连通分量中的每个顶点和边。它是探索图、检测环、对边进行分类以及构建更复杂算法（如拓扑排序、强连通分量和关节点检测）的基础。

我们想要的算法能够：

- 从起始节点探索所有可达顶点
- 避免重复访问节点
- 记录遍历顺序

示例：
你有一个迷宫。DFS 就像一个探索者，选择一条路径，尽可能深入地走，只有在卡住时才返回。

#### 它是如何工作的（通俗解释）？

将 DFS 想象成一个好奇的旅行者：每当看到一条新路径，就深入探索。当无法继续前进时，就后退一步，继续探索其他路径。

我们使用递归来自然地模拟这种行为，每次递归调用代表进入一个新节点，而返回则意味着回溯。

| 步骤 | 当前节点                     | 动作          | 栈（调用路径） |
| ---- | ---------------------------- | ------------- | -------------- |
| 1    | A                            | 访问 A        | [A]            |
| 2    | B                            | 访问 B (A→B)  | [A, B]         |
| 3    | D                            | 访问 D (B→D)  | [A, B, D]      |
| 4    | D 没有未访问的邻居           | 回溯          | [A, B]         |
| 5    | B 的下一个邻居 C             | 访问 C        | [A, B, C]      |
| 6    | C 完成                       | 回溯          | [A, B] → [A]   |
| 7    | A 的剩余邻居                 | 访问下一个    | [...]          |

当递归展开时，我们就探索了整个可达图。

#### 微型代码（简易版本）

C 语言（邻接矩阵示例）

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int graph[MAX][MAX];
bool visited[MAX];
int n;

void dfs(int v) {
    visited[v] = true;
    printf("%d ", v);
    for (int u = 0; u < n; u++) {
        if (graph[v][u] && !visited[u]) {
            dfs(u);
        }
    }
}

int main(void) {
    printf("请输入顶点数: ");
    scanf("%d", &n);

    printf("请输入邻接矩阵:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    printf("从顶点 0 开始的 DFS:\n");
    dfs(0);
}
```

Python（邻接表）

```python
graph = {
    0: [1, 2],
    1: [2],
    2: [0, 3],
    3: [3]
}

visited = set()

def dfs(v):
    visited.add(v)
    print(v, end=" ")
    for u in graph[v]:
        if u not in visited:
            dfs(u)

dfs(0)
```

#### 为什么它很重要

- 是图探索和可达性分析的核心
- 构成了拓扑排序、强连通分量、桥和环检测的基础
- 简单的递归结构揭示了图的自然层次结构
- 有助于理解回溯和基于栈的思维方式

#### 一个温和的证明（为什么它有效）

每个顶点恰好被访问一次：

- 当一个节点首次被发现时，它被标记为 `visited`
- 递归确保其所有邻居都被探索
- 一旦所有子节点完成，函数返回（回溯）

因此，每个顶点 `v` 触发一次 `dfs(v)` 调用，时间复杂度为 O(V + E)（每条边被探索一次）。

#### 自己动手试试

1.  画一个小图（A–B–C–D）并跟踪 DFS。
2.  修改代码以打印进入和退出时间。
3.  跟踪父节点以构建 DFS 树。
4.  添加后向边检测（环测试）。
5.  与 BFS 遍历顺序进行比较。

#### 测试用例

| 图                         | 起始点 | 期望的顺序 | 备注                           |
| ------------------------- | ----- | ---------- | ------------------------------ |
| A–B–C–D 链                | A     | A B C D    | 直线路径                       |
| 三角形 (A–B–C–A)          | A     | A B C      | 访问所有节点，在已访问的 A 处停止 |
| 不连通图 {A–B}, {C–D}     | A     | A B        | 仅访问可达分量                 |
| 有向图 A→B→C              | A     | A B C      | 线性链                         |
| 树根为 0                  | 0     | 0 1 3 4 2 5 | 取决于邻接顺序                 |

#### 复杂度

- 时间：O(V + E)
- 空间：O(V)（递归栈 + visited 数组）

DFS 是你理解图结构的第一把钥匙，它递归、优雅，一次一个栈帧地揭示隐藏的路径。
### 302 深度优先搜索（迭代版）

深度优先搜索也可以在没有递归的情况下运行。我们不依赖调用栈，而是显式地构建自己的栈。这是同样的旅程——先深入探索再回溯，只是手动控制下一步做什么。

#### 我们要解决什么问题？

递归很优雅，但并不总是实用。
有些图很深，递归的 DFS 可能导致调用栈溢出。迭代版本通过直接使用栈数据结构解决了这个问题，并保持了相同的遍历顺序。

我们需要一个算法，它：

- 即使在递归深度过大时也能工作
- 显式管理已访问节点和栈
- 产生与递归 DFS 相同的遍历顺序

示例：
可以把它想象成自己维护一份未探索路径的待办事项清单。每次深入探索时，都将新的目的地添加到栈顶。

#### 它是如何工作的（通俗解释）？

我们维护一个栈：

1.  从节点 `s` 开始，将其压入栈。
2.  弹出栈顶节点 `v`。
3.  如果 `v` 未被访问过，则标记并处理它。
4.  将 `v` 的所有未访问过的邻居压入栈。
5.  重复上述步骤，直到栈为空。

| 步骤 | 栈（顶 → 底） | 动作               | 已访问节点    |
| ---- | ------------- | ------------------ | ------------- |
| 1    | [A]           | 开始，弹出 A       | {A}           |
| 2    | [B, C]        | 压入 A 的邻居      | {A}           |
| 3    | [C, B]        | 弹出 B，访问 B     | {A, B}        |
| 4    | [C, D]        | 压入 B 的邻居      | {A, B}        |
| 5    | [D, C]        | 弹出 D，访问 D     | {A, B, D}     |
| 6    | [C]           | 继续               | {A, B, D}     |
| 7    | [ ]           | 弹出 C，访问 C     | {A, B, C, D}  |

#### 精简代码（简易版本）

C 语言（邻接矩阵示例）

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int graph[MAX][MAX];
bool visited[MAX];
int stack[MAX];
int top = -1;
int n;

void push(int v) { stack[++top] = v; }
int pop() { return stack[top--]; }
bool is_empty() { return top == -1; }

void dfs_iterative(int start) {
    push(start);
    while (!is_empty()) {
        int v = pop();
        if (!visited[v]) {
            visited[v] = true;
            printf("%d ", v);
            for (int u = n - 1; u >= 0; u--) { // 反向遍历以获得一致的顺序
                if (graph[v][u] && !visited[u])
                    push(u);
            }
        }
    }
}

int main(void) {
    printf("输入顶点数: ");
    scanf("%d", &n);

    printf("输入邻接矩阵:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    printf("从 0 开始的迭代 DFS:\n");
    dfs_iterative(0);
}
```

Python（使用列表作为栈）

```python
graph = {
    0: [1, 2],
    1: [2],
    2: [0, 3],
    3: [3]
}

visited = set()
stack = [0]

while stack:
    v = stack.pop()
    if v not in visited:
        visited.add(v)
        print(v, end=" ")
        for u in reversed(graph[v]):  # 反转以获得类似 DFS 的顺序
            if u not in visited:
                stack.append(u)
```

#### 为什么它很重要

- 避免递归限制和栈溢出
- 清晰控制遍历顺序
- 适用于调用栈有限的系统
- 有助于理解显式栈模拟

#### 一个温和的证明（为什么它有效）

栈模拟了递归：
每个顶点 `v` 在弹出时被处理一次，并且它的邻居被压入栈。
每条边恰好被检查一次。
因此总操作数 = O(V + E)，与递归 DFS 相同。

每次压栈 = 一次递归调用；每次弹栈 = 一次返回。

#### 自己动手试试

1.  在一个小图上跟踪迭代 DFS 的执行过程。
2.  将其顺序与递归版本进行比较。
3.  尝试改变邻居压栈的顺序，观察输出如何变化。
4.  添加入栈时间和完成时间。
5.  通过将完成顺序压入第二个栈，将其转换为迭代的拓扑排序。

#### 测试用例

| 图结构                   | 起点 | 顺序（一种可能） | 备注                     |
| ------------------------ | ---- | ---------------- | ------------------------ |
| 0–1–2 链                 | 0    | 0 1 2            | 简单路径                 |
| 0→1, 0→2, 1→3            | 0    | 0 1 3 2          | 取决于邻居的顺序         |
| 环 0→1→2→0               | 0    | 0 1 2            | 无重复访问               |
| 不连通图                 | 0    | 0 1 2            | 只访问连通部分           |
| 完全图（4 个节点）       | 0    | 0 1 2 3          | 访问所有节点一次         |

#### 复杂度

-   时间：O(V + E)
-   空间：O(V)（用于栈和已访问数组）

迭代 DFS 就像是递归的手动挡版本，同样的深度，同样的发现过程，只是没有调用栈带来的意外。
### 303 广度优先搜索（队列）

广度优先搜索（BFS）是一种逐层探索的算法，从起点向外辐射式扩展。与深度优先搜索（DFS）不同，BFS 保持公平性，它会先访问所有邻居节点，再深入下一层。

#### 我们要解决什么问题？

我们希望有一种方法能够：

- 探索图中所有可达的顶点
- 在无权图中发现最短路径
- 按距离递增的顺序处理节点

BFS 在以下情况下非常适用：

- 所有边的权重相等（例如都为 1）
- 你需要到达目标的最少步数
- 你在寻找连通分量、层级或距离

示例：
想象一下传播谣言。每个人在下一轮开始前告诉所有朋友，这就是 BFS 的实际应用。

#### 它是如何工作的（通俗解释）？

BFS 使用一个队列，即先进先出的队列。

1. 从节点 `s` 开始
2. 标记它为已访问并入队
3. 当队列不为空时：
   * 出队队首节点 `v`
   * 访问 `v`
   * 将 `v` 的所有未访问邻居入队

| 步骤 | 队列（队首 → 队尾） | 已访问集合          | 操作                     |
| ---- | -------------------- | ------------------ | -------------------------- |
| 1    | [A]                  | {A}                | 开始                      |
| 2    | [B, C]               | {A, B, C}          | A 的邻居              |
| 3    | [C, D, E]            | {A, B, C, D, E}    | 访问 B，添加其邻居 |
| 4    | [D, E, F]            | {A, B, C, D, E, F} | 访问 C，添加邻居     |
| 5    | [E, F]               | {A…F}              | 继续直到队列为空       |

出队的顺序 = 层级遍历顺序。

#### 精简代码（简易版本）

C 语言（邻接矩阵示例）

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int graph[MAX][MAX];
bool visited[MAX];
int queue[MAX];
int front = 0, rear = 0;
int n;

void enqueue(int v) { queue[rear++] = v; }
int dequeue() { return queue[front++]; }
bool is_empty() { return front == rear; }

void bfs(int start) {
    visited[start] = true;
    enqueue(start);

    while (!is_empty()) {
        int v = dequeue();
        printf("%d ", v);

        for (int u = 0; u < n; u++) {
            if (graph[v][u] && !visited[u]) {
                visited[u] = true;
                enqueue(u);
            }
        }
    }
}

int main(void) {
    printf("输入顶点数: ");
    scanf("%d", &n);

    printf("输入邻接矩阵:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    printf("从顶点 0 开始的 BFS:\n");
    bfs(0);
}
```

Python（邻接表）

```python
from collections import deque

graph = {
    0: [1, 2],
    1: [3, 4],
    2: [5],
    3: [],
    4: [],
    5: []
}

visited = set()
queue = deque([0])
visited.add(0)

while queue:
    v = queue.popleft()
    print(v, end=" ")
    for u in graph[v]:
        if u not in visited:
            visited.add(u)
            queue.append(u)
```

#### 为什么它很重要

- 在无权图中找到最短路径
- 保证按层级顺序访问
- 是 0-1 BFS、SPFA 和 Dijkstra 等算法的核心
- 非常适合基于层的探索和距离标记

#### 一个温和的证明（为什么它有效）

每个顶点恰好被访问一次：

- 它在被发现时入队
- 它被出队一次以进行处理
- 每条边被检查一次

如果所有边的权重都为 1，BFS 会按距离递增的顺序发现顶点，这证明了最短路径的正确性。

时间复杂度：

- 访问每个顶点：O(V)
- 扫描每条边：O(E)
  → 总计：O(V + E)

#### 自己动手试试

1.  画一个小的无权图并手动运行 BFS。
2.  记录层级（距离起点的距离）。
3.  追踪每个顶点的父节点，重建最短路径。
4.  在树上尝试 BFS，与层级遍历进行比较。
5.  在不连通的图上实验，注意遗漏了什么。

#### 测试用例

| 图               | 起点 | 访问顺序 | 距离               |
| ---------------- | ---- | -------- | ------------------ |
| 0–1–2 链         | 0    | 0 1 2    | [0,1,2]            |
| 三角形 0–1–2     | 0    | 0 1 2    | [0,1,1]            |
| 星形 0→{1,2,3}   | 0    | 0 1 2 3  | [0,1,1,1]          |
| 2×2 网格         | 0    | 0 1 2 3  | 分层               |
| 不连通图         | 0    | 0 1      | 仅包含 0 的连通分量 |

#### 复杂度

- 时间：O(V + E)
- 空间：O(V)（用于队列和已访问集合）

BFS 是你的波阵面探索者，公平、系统，并且在边权相等时总是能找到最短路径。
### 304 迭代加深深度优先搜索

迭代加深深度优先搜索（IDDFS）融合了广度优先搜索（BFS）的深度控制能力和深度优先搜索（DFS）的空间效率。它通过不断增加深度限制来反复执行深度优先搜索，从而逐层地发现节点，但每次都是通过深度优先的方式进行探索。

#### 我们要解决什么问题？

纯粹的深度优先搜索可能会探索得过深，从而错过更近的解决方案。
纯粹的广度优先搜索能找到最短路径，但会消耗大量内存。

我们需要一种搜索方法，它能够：

-   像广度优先搜索一样找到最浅的解决方案
-   像深度优先搜索一样使用 O(深度) 的内存
-   在无限或非常大的搜索空间中工作

这正是迭代加深深度优先搜索的闪光点，它执行深度优先搜索直到一个限制深度，然后以更深的限制重新开始，重复此过程直到找到目标。

示例：
想象一位潜水员，每次潜水都探索得更深一些，1米、2米、3米，总是从水面开始向下扫掠。

#### 它是如何工作的（通俗解释）？

每次迭代将深度限制增加一。
在每个阶段，我们执行一次深度优先搜索，当深度超过当前限制时停止。

1.  设置限制 = 0
2.  运行深度限制为 0 的深度优先搜索
3.  如果未找到，则增加限制并重复
4.  继续直到找到目标或探索完所有节点

| 迭代次数 | 深度限制 | 已探索节点      | 找到目标？ |
| -------- | -------- | --------------- | ---------- |
| 1        | 0        | 起始节点        | 否         |
| 2        | 1        | 起始节点 + 邻居 | 否         |
| 3        | 2        | + 更深节点      | 可能       |
| …        | …        | …               | …          |

尽管节点会被重复访问，但总成本仍然高效，类似于广度优先搜索的逐层发现。

#### 微型代码（简易版本）

C 语言（深度受限的深度优先搜索）

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int graph[MAX][MAX];
int n;
bool found = false;

void dls(int v, int depth, int limit, bool visited[]) {
    visited[v] = true;
    printf("%d ", v);
    if (depth == limit) return;
    for (int u = 0; u < n; u++) {
        if (graph[v][u] && !visited[u]) {
            dls(u, depth + 1, limit, visited);
        }
    }
}

void iddfs(int start, int max_depth) {
    for (int limit = 0; limit <= max_depth; limit++) {
        bool visited[MAX] = {false};
        printf("\nDepth limit %d: ", limit);
        dls(start, 0, limit, visited);
    }
}

int main(void) {
    printf("Enter number of vertices: ");
    scanf("%d", &n);

    printf("Enter adjacency matrix:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    iddfs(0, 3);
}
```

Python（邻接表 + 深度限制）

```python
graph = {
    0: [1, 2],
    1: [3],
    2: [4],
    3: [],
    4: []
}

def dls(v, depth, limit, visited):
    visited.add(v)
    print(v, end=" ")
    if depth == limit:
        return
    for u in graph[v]:
        if u not in visited:
            dls(u, depth + 1, limit, visited)

def iddfs(start, max_depth):
    for limit in range(max_depth + 1):
        print(f"\nDepth limit {limit}:", end=" ")
        visited = set()
        dls(start, 0, limit, visited)

iddfs(0, 3)
```

#### 为什么它很重要

-   结合了广度优先搜索和深度优先搜索的优点
-   在无权图中找到最优解
-   使用线性空间
-   非常适合状态空间搜索（人工智能、谜题）

#### 一个温和的证明（为什么它有效）

广度优先搜索保证最短路径；深度优先搜索使用更少的空间。
迭代加深深度优先搜索通过不断增加限制来重复执行深度优先搜索，确保：

-   深度 `d` 的所有节点都在深度 `d+1` 之前被访问
-   空间复杂度 = O(d)
-   时间复杂度 ≈ O(b^d)，与广度优先搜索同阶

与更深层的总节点数相比，冗余工作（重复访问节点）是微不足道的。

#### 亲自尝试

1.  在树上运行迭代加深深度优先搜索；观察重复的浅层访问。
2.  计算每次迭代访问的节点数。
3.  与广度优先搜索的总访问次数进行比较。
4.  在运行中途修改深度限制，会发生什么？
5.  使用迭代加深深度优先搜索查找深度为 3 的目标节点。

#### 测试用例

| 图结构               | 目标   | 最大深度 | 在何处找到 | 顺序示例    |
| -------------------- | ------ | -------- | ---------- | ----------- |
| 0→1→2→3              | 3      | 3        | 深度 3     | 0 1 2 3     |
| 0→{1,2}, 1→3         | 3      | 3        | 深度 2     | 0 1 3       |
| 星型 0→{1,2,3}       | 3      | 1        | 深度 1     | 0 1 2 3     |
| 0→1→2→Goal           | Goal=2 | 2        | 深度 2     | 0 1 2       |

#### 复杂度

-   时间复杂度：O(b^d)（类似于广度优先搜索）
-   空间复杂度：O(d)（类似于深度优先搜索）

迭代加深深度优先搜索就像一位耐心的攀登者，反复探索熟悉的地面，每次走得更深，确保不会错过任何浅层的宝藏。
### 305 双向 BFS

双向 BFS 是 BFS 的“中间相遇”版本。我们不是从一端开始向外探索所有内容，而是同时发起两波 BFS 搜索，一波从源点出发，一波从目标点出发，当它们在中间相遇时停止。这就像从山的两侧同时挖隧道，最终在中间汇合。

#### 我们要解决什么问题？

标准的 BFS 从起点开始向外探索整个搜索空间，直到到达目标。这对于小图来说很好，但当图很大时，代价会很高。

双向 BFS 通过同时从两个方向进行搜索，显著减少了探索范围，将有效搜索深度减半。

我们想要的算法是：
- 在无权图中找到最短路径
- 比单源 BFS 探索更少的节点
- 一旦两波搜索相遇就立即停止

示例：
假设你要找到两个城市之间的最短路线。你不是从一个城市出发探索整个地图，而是也从目的地派出侦察员。他们会在某处相遇，这个点就是最短路径的中点。

#### 它是如何工作的？（通俗解释）

同时运行两个 BFS 搜索，一个向前，一个向后。
在每一步，优先扩展较小的边界以平衡工作量。
当任何节点同时出现在两个已访问集合中时停止。

1. 从 `source` 和 `target` 开始 BFS
2. 维护两个队列和两个已访问集合
3. 交替扩展
4. 当已访问集合重叠时，找到相遇点
5. 合并路径得到最终路线

| 步骤 | 向前队列 | 向后队列 | 交集 | 动作 |
| ---- | ------------- | -------------- | ------------ | ------------------ |
| 1    | [S]           | [T]            | ∅            | 开始 |
| 2    | [S1, S2]      | [T1, T2]       | ∅            | 同时扩展 |
| 3    | [S2, S3]      | [T1, S2]       | S2           | 找到相遇节点 |

#### 精简代码（简易版本）

C 语言（简化版本）

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int graph[MAX][MAX];
int n;

bool bfs_step(bool visited[], int queue[], int *front, int *rear) {
    int size = *rear - *front;
    while (size--) {
        int v = queue[(*front)++];
        for (int u = 0; u < n; u++) {
            if (graph[v][u] && !visited[u]) {
                visited[u] = true;
                queue[(*rear)++] = u;
            }
        }
    }
    return false;
}

bool intersect(bool a[], bool b[]) {
    for (int i = 0; i < n; i++)
        if (a[i] && b[i]) return true;
    return false;
}

bool bidir_bfs(int src, int dest) {
    bool vis_s[MAX] = {false}, vis_t[MAX] = {false};
    int qs[MAX], qt[MAX];
    int fs = 0, rs = 0, ft = 0, rt = 0;
    qs[rs++] = src; vis_s[src] = true;
    qt[rt++] = dest; vis_t[dest] = true;

    while (fs < rs && ft < rt) {
        if (bfs_step(vis_s, qs, &fs, &rs)) return true;
        if (intersect(vis_s, vis_t)) return true;

        if (bfs_step(vis_t, qt, &ft, &rt)) return true;
        if (intersect(vis_s, vis_t)) return true;
    }
    return false;
}

int main(void) {
    printf("请输入顶点数：");
    scanf("%d", &n);
    printf("请输入邻接矩阵：\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    int src = 0, dest = n - 1;
    if (bidir_bfs(src, dest))
        printf("找到路径\n");
    else
        printf("无路径\n");
}
```

Python（可读性版本）

```python
from collections import deque

graph = {
    0: [1, 2],
    1: [3],
    2: [4],
    3: [5],
    4: [5],
    5: []
}

def bidirectional_bfs(src, dest):
    if src == dest:
        return True

    q1, q2 = deque([src]), deque([dest])
    visited1, visited2 = {src}, {dest}

    while q1 and q2:
        # 向前扩展
        for _ in range(len(q1)):
            v = q1.popleft()
            for u in graph[v]:
                if u in visited2:
                    return True
                if u not in visited1:
                    visited1.add(u)
                    q1.append(u)

        # 向后扩展
        for _ in range(len(q2)):
            v = q2.popleft()
            for u in graph[v]:
                if u in visited1:
                    return True
                if u not in visited2:
                    visited2.add(u)
                    q2.append(u)

    return False

print(bidirectional_bfs(0, 5))
```

#### 为什么它很重要

- 在大图上更快地搜索最短路径
- 将探索的节点数从 O(b^d) 减少到大约 O(b^(d/2))
- 非常适合在地图、谜题或网络中进行路径查找
- 展示了搜索的对称性和边界平衡

#### 一个温和的证明（为什么它有效）

如果最短路径长度是 `d`，
BFS 探索 O(b^d) 个节点，
但双向 BFS 探索 2×O(b^(d/2)) 个节点——
这是一个巨大的节省，因为 b^(d/2) ≪ b^d。

每一方都保证边界逐层扩展，
而交集确保了在最短路径的中间相遇。

#### 亲自尝试

1.  在一条 5 个节点的链（0→1→2→3→4）上追踪双向 BFS。
2.  比较单源 BFS 与双向 BFS 访问的节点数。
3.  添加打印语句以查看两波搜索在何处相遇。
4.  修改代码以重建路径。
5.  在分支图上比较性能。

#### 测试用例

| 图结构 | 源点 | 目标点 | 找到？ | 相遇节点 |
| ----------------------- | ------ | ------ | ------ | ------------ |
| 0–1–2–3–4 | 0 | 4 | ✅ | 2 |
| 0→1, 1→2, 2→3 | 0 | 3 | ✅ | 1 或 2 |
| 0→1, 2→3 （不连通） | 0 | 3 | ❌ | – |
| 三角形 0–1–2–0 | 0 | 2 | ✅ | 0 或 2 |
| 星形 0→{1,2,3,4} | 1 | 2 | ✅ | 0 |

#### 复杂度

- 时间：O(b^(d/2))
- 空间：O(b^(d/2))
- 最优性：在无权图中找到最短路径

双向 BFS 就像桥梁建造者，从两岸同时开工，朝着中间的汇合点竞速前进。
### 306 网格上的深度优先搜索（DFS）

网格上的深度优先搜索（DFS）是你探索二维地图、迷宫或岛屿的首选工具。它的工作原理与图上的 DFS 类似，但在这里，每个单元格是一个节点，其上下左右邻居构成边。它非常适合连通分量检测、区域标记或迷宫求解。

#### 我们要解决什么问题？

我们想要探索或标记网格中所有连通的单元格，通常用于：

-   计算二进制矩阵中的岛屿数量
-   泛洪填充算法（为区域着色）
-   迷宫遍历（穿过墙壁寻找路径）
-   二维地图中的连通性检测

示例：
想象一位画家将墨水倒入一个单元格，DFS 展示了墨水如何扩散以填充整个连通区域。

#### 它是如何工作的（通俗解释）？

DFS 从给定的单元格开始，访问它，并递归地探索所有有效的、未访问的邻居。

我们检查 4 个方向（如果算上对角线，则是 8 个）。
每个邻居需要满足：

-   在边界内
-   尚未被访问
-   满足条件（例如，颜色相同，值等于 1）

| 步骤 | 当前单元格               | 动作       | 栈（调用路径）         |
| ---- | ------------------------ | ---------- | ---------------------- |
| 1    | (0,0)                    | 访问       | [(0,0)]                |
| 2    | (0,1)                    | 向右移动   | [(0,0),(0,1)]          |
| 3    | (1,1)                    | 向下移动   | [(0,0),(0,1),(1,1)]    |
| 4    | (1,1) 没有新邻居         | 回溯       | [(0,0),(0,1)]          |
| 5    | 继续                     | …          |                        |

当所有可达单元格都被访问后，遍历结束。

#### 微型代码（简易版本）

C 语言（用于岛屿计数的 DFS）

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int grid[MAX][MAX];
bool visited[MAX][MAX];
int n, m;

int dx[4] = {-1, 1, 0, 0};
int dy[4] = {0, 0, -1, 1};

void dfs(int x, int y) {
    visited[x][y] = true;
    for (int k = 0; k < 4; k++) {
        int nx = x + dx[k];
        int ny = y + dy[k];
        if (nx >= 0 && nx < n && ny >= 0 && ny < m &&
            grid[nx][ny] == 1 && !visited[nx][ny]) {
            dfs(nx, ny);
        }
    }
}

int main(void) {
    printf("输入网格大小 (n m): ");
    scanf("%d %d", &n, &m);
    printf("输入网格 (0/1):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            scanf("%d", &grid[i][j]);

    int count = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            if (grid[i][j] == 1 && !visited[i][j]) {
                dfs(i, j);
                count++;
            }

    printf("岛屿数量: %d\n", count);
}
```

Python（泛洪填充）

```python
grid = [
    [1,1,0,0],
    [1,0,0,1],
    [0,0,1,1],
    [0,0,0,1]
]

n, m = len(grid), len(grid[0])
visited = [[False]*m for _ in range(n)]

def dfs(x, y):
    if x < 0 or x >= n or y < 0 or y >= m:
        return
    if grid[x][y] == 0 or visited[x][y]:
        return
    visited[x][y] = True
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        dfs(x + dx, y + dy)

count = 0
for i in range(n):
    for j in range(m):
        if grid[i][j] == 1 and not visited[i][j]:
            dfs(i, j)
            count += 1

print("岛屿数量:", count)
```

#### 为什么它很重要

-   网格探索和区域标记的核心工具
-   构成岛屿问题、迷宫求解器和地图连通性问题的核心
-   展示了现实世界布局中的 DFS 行为
-   在二维数组上易于可视化和调试

#### 一个温和的证明（为什么它有效）

每个单元格恰好被访问一次，在进入时被标记为 `visited`。
递归调用扩散到所有有效的邻居。
因此总时间与单元格数量和边（邻居）数量成正比。

如果网格大小为 `n × m`，并且每个单元格检查 4 个邻居：

-   时间复杂度：O(n × m)
-   空间复杂度：O(n × m) visited + 递归深度 (≤ n × m)

DFS 保证每个可达单元格恰好被访问一次，形成连通分量。

#### 自己动手试试

1.  将移动方向改为 8 个方向（包括对角线）。
2.  修改为泛洪填充颜色（例如，将所有 1 替换为 2）。
3.  计算字符矩阵（‘X', ‘O'）中的连通分量数量。
4.  在打印的网格中可视化遍历顺序。
5.  与同一网格上的 BFS 进行比较。

#### 测试用例

| 网格                                      | 预期结果 | 描述                 |
| ----------------------------------------- | -------- | -------------------- |
| [[1,0,0],[0,1,0],[0,0,1]]                 | 3        | 对角线不连通         |
| [[1,1,0],[1,0,0],[0,0,1]]                 | 2        | 两个簇               |
| [[1,1,1],[1,1,1],[1,1,1]]                 | 1        | 一个大岛屿           |
| [[0,0,0],[0,0,0]]                         | 0        | 没有陆地             |
| [[1,0,1],[0,1,0],[1,0,1]]                 | 5        | 多个孤立点           |

#### 复杂度

-   时间复杂度：O(n × m)
-   空间复杂度：O(n × m)（visited 数组）或 O(深度) 递归栈

网格上的 DFS 是你的地图探索者，一次一个单元格地扫过每一个可达的区域。
### 307 网格上的 BFS

网格上的 BFS 逐层探索单元格，这使其非常适合在无权网格中寻找最短路径、在迷宫中计算最少步数，以及从源点进行距离标记。每个单元格都是一个节点，边连接其上下左右等邻居。

#### 我们要解决什么问题？

我们想要：

-   在每次移动成本相同的情况下，找到从起始单元格到目标单元格的最短路径
-   计算从源点到所有可达单元格的距离图
-   干净地处理障碍物并避免重复访问

示例：
给定一个由 0 或 1 表示的网格迷宫，其中 0 表示可通行，1 表示墙，BFS 可以找到从起点到目标的最少移动步数。

#### 它是如何工作的（通俗解释）？

使用一个队列。从源点开始，将其入队并标记距离为 0，然后逐波扩展。
每一步，弹出队首单元格，尝试访问其邻居，将未访问过的邻居标记为已访问，并记录其距离为当前距离加 1。

| 步骤 | 队列（从前到后） | 当前单元格 | 动作           | 更新的距离                |
| ---- | ---------------- | ---------- | -------------- | ------------------------- |
| 1    | [(sx, sy)]       | (sx, sy)   | 开始           | dist[sx][sy] = 0          |
| 2    | [(n1), (n2)]     | (n1)       | 访问邻居       | dist[n1] = 1              |
| 3    | [(n2), (n3), (n4)] | (n2)       | 继续           | dist[n2] = 1              |
| 4    | [...]            | ...        | 波次扩展       | dist[next] = dist[cur] + 1 |

首次到达目标时，记录的距离就是最短距离。

#### 精简代码（简易版本）

C 语言（在 0 或 1 网格上的最短路径）

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 200
int n, m;
int grid[MAX][MAX];         // 0 可通行，1 墙
int distv[MAX][MAX];        // 距离图
bool vis[MAX][MAX];

int qx[MAX*MAX], qy[MAX*MAX];
int front = 0, rear = 0;

int dx[4] = {-1, 1, 0, 0};
int dy[4] = {0, 0, -1, 1};

void enqueue(int x, int y) { qx[rear] = x; qy[rear] = y; rear++; }
void dequeue(int *x, int *y) { *x = qx[front]; *y = qy[front]; front++; }
bool empty() { return front == rear; }

int bfs(int sx, int sy, int tx, int ty) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            vis[i][j] = false;
            distv[i][j] = -1;
        }

    vis[sx][sy] = true;
    distv[sx][sy] = 0;
    enqueue(sx, sy);

    while (!empty()) {
        int x, y;
        dequeue(&x, &y);
        if (x == tx && y == ty) return distv[x][y];

        for (int k = 0; k < 4; k++) {
            int nx = x + dx[k], ny = y + dy[k];
            if (nx >= 0 && nx < n && ny >= 0 && ny < m &&
                !vis[nx][ny] && grid[nx][ny] == 0) {
                vis[nx][ny] = true;
                distv[nx][ny] = distv[x][y] + 1;
                enqueue(nx, ny);
            }
        }
    }
    return -1; // 不可达
}

int main(void) {
    printf("输入 n m: ");
    scanf("%d %d", &n, &m);
    printf("输入网格 (0 可通行，1 墙):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            scanf("%d", &grid[i][j]);

    int sx, sy, tx, ty;
    printf("输入起点 sx sy 和目标 tx ty: ");
    scanf("%d %d %d %d", &sx, &sy, &tx, &ty);

    int d = bfs(sx, sy, tx, ty);
    if (d >= 0) printf("最短距离: %d\n", d);
    else printf("无路径\n");
}
```

Python（距离图和路径重建）

```python
from collections import deque

grid = [
    [0,0,0,1],
    [1,0,0,0],
    [0,0,1,0],
    [0,0,0,0]
]
n, m = len(grid), len(grid[0])

def bfs_grid(sx, sy, tx, ty):
    dist = [[-1]*m for _ in range(n)]
    parent = [[None]*m for _ in range(n)]
    q = deque()
    q.append((sx, sy))
    dist[sx][sy] = 0

    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        pass  # 仅用于展示方向存在

    while q:
        x, y = q.popleft()
        if (x, y) == (tx, ty):
            break
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m and grid[nx][ny] == 0 and dist[nx][ny] == -1:
                dist[nx][ny] = dist[x][y] + 1
                parent[nx][ny] = (x, y)
                q.append((nx, ny))

    # 如果可达，重建路径
    if dist[tx][ty] == -1:
        return dist, []
    path = []
    cur = (tx, ty)
    while cur:
        path.append(cur)
        cur = parent[cur[0]][cur[1]]
    path.reverse()
    return dist, path

dist, path = bfs_grid(0, 0, 3, 3)
print("到目标的距离:", dist[3][3])
print("路径:", path)
```

#### 为什么它很重要？

-   保证在无权网格中找到最短路径
-   生成完整的距离变换，适用于许多任务
-   对于迷宫求解器和机器人导航来说，鲁棒且简单
-   是通向 0-1 BFS 和 Dijkstra 算法的自然阶梯

#### 一个温和的证明（为什么它有效）

BFS 按照从源点距离非递减的顺序处理单元格。当一个单元格首次出队时，存储的距离等于到达它所需的最少移动步数。每个可通行的邻居被发现时，其距离为当前距离加一。因此，首次到达目标时，该距离就是最短距离。

-   每个单元格最多进入队列一次
-   相邻单元格之间的每条边最多被考虑一次

因此，总工作量与单元格数量和邻居检查次数成线性关系。

#### 自己动手试试

1.  添加 8 个方向的移动，并与 4 个方向的移动路径进行比较。
2.  通过将列出的单元格对作为边连接起来，添加传送门。
3.  通过将多个起点以距离 0 入队，转换为多源 BFS。
4.  阻挡一些单元格，并验证 BFS 永远不会穿过墙。
5.  记录父节点并打印出标记了路径的迷宫。

#### 测试用例

| 网格                   | 起点               | 目标   | 预期结果                           |
| ---------------------- | ------------------ | ------ | ---------------------------------- |
| 2 x 2 全部可通行       | (0,0)              | (1,1)  | 距离 2，先右后下                   |
| 3 x 3 中心有墙         | (0,0)              | (2,2)  | 距离 4，绕墙而行                   |
| 1 x 5 直线全部可通行   | (0,0)              | (0,4)  | 距离 4                             |
| 目标被阻挡             | (0,0)              | (1,1)  | 无路径                             |
| 多源波                 | {所有角落起点}     | 中心   | 各角落中的最小值                   |

#### 复杂度

-   时间复杂度：O(n × m)
-   空间复杂度：O(n × m)（用于已访问标记或距离图以及队列）

网格上的 BFS 是均匀扫描地图的波前，通过清晰、逐层的逻辑，为你提供从起点到目标的最少步数。
### 308 多源广度优先搜索（Multi-Source BFS）

多源广度优先搜索是一种波阵面式的广度优先搜索，它并非从一个节点开始，而是同时从多个源点开始。当多个起点同时向外扩散时，这种方法非常完美，就像森林中的多处火情同时蔓延，或多个信号发射器同时发出信号。

#### 我们要解决什么问题？

我们需要找出从多个起始节点（而不仅仅是一个）出发的最小距离。
这在以下情况下非常有用：

-   存在多个影响源（例如，感染、信号、火灾）
-   你想为每个节点找到最近的源
-   你需要同时传播（例如，多起点最短路径）

示例：

-   谣言从多个人开始传播
-   从多个水源开始的洪水蔓延时间
-   到达最近医院或供应中心的最小距离

#### 它是如何工作的（通俗解释）？

我们将所有源点视为第 0 层，并一次性将它们全部加入队列。
然后广度优先搜索正常进行，每个节点被分配的距离等于从任意源点出发的最短路径。

| 步骤 | 队列（队首 → 队尾）            | 动作                 | 更新的距离       |
| ---- | ----------------------------- | -------------------- | ---------------- |
| 1    | [S1, S2, S3]                  | 初始化所有源点       | dist[S*] = 0     |
| 2    | [S1, S2, S3 的邻居]           | 波阵面扩展           | dist = 1         |
| 3    | [下一层]                      | 继续                 | dist = 2         |
| …    | …                             | …                    | …                |

当一个节点第一次被访问时，我们就知道它来自最近的源点。

#### 微型代码（简易版本）

C 语言（网格上的多源广度优先搜索）

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int n, m;
int grid[MAX][MAX];
int distv[MAX][MAX];
bool vis[MAX][MAX];
int qx[MAX*MAX], qy[MAX*MAX];
int front = 0, rear = 0;

int dx[4] = {-1, 1, 0, 0};
int dy[4] = {0, 0, -1, 1};

void enqueue(int x, int y) { qx[rear] = x; qy[rear] = y; rear++; }
void dequeue(int *x, int *y) { *x = qx[front]; *y = qy[front]; front++; }
bool empty() { return front == rear; }

void multi_source_bfs() {
    while (!empty()) {
        int x, y;
        dequeue(&x, &y);
        for (int k = 0; k < 4; k++) {
            int nx = x + dx[k], ny = y + dy[k];
            if (nx >= 0 && nx < n && ny >= 0 && ny < m &&
                grid[nx][ny] == 0 && !vis[nx][ny]) {
                vis[nx][ny] = true;
                distv[nx][ny] = distv[x][y] + 1;
                enqueue(nx, ny);
            }
        }
    }
}

int main(void) {
    printf("输入网格大小 n m: ");
    scanf("%d %d", &n, &m);
    printf("输入网格 (0 空闲, 1 阻塞, 2 源点):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            scanf("%d", &grid[i][j]);
            if (grid[i][j] == 2) {
                vis[i][j] = true;
                distv[i][j] = 0;
                enqueue(i, j);
            }
        }

    multi_source_bfs();

    printf("距离地图:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            printf("%2d ", distv[i][j]);
        printf("\n");
    }
}
```

Python（简易多源广度优先搜索）

```python
from collections import deque

grid = [
    [2,0,1,0],
    [0,1,0,0],
    [0,0,0,2]
]
n, m = len(grid), len(grid[0])
dist = [[-1]*m for _ in range(n)]
q = deque()

for i in range(n):
    for j in range(m):
        if grid[i][j] == 2:  # 源点
            dist[i][j] = 0
            q.append((i, j))

dirs = [(-1,0),(1,0),(0,-1),(0,1)]
while q:
    x, y = q.popleft()
    for dx, dy in dirs:
        nx, ny = x+dx, y+dy
        if 0 <= nx < n and 0 <= ny < m and grid[nx][ny] == 0 and dist[nx][ny] == -1:
            dist[nx][ny] = dist[x][y] + 1
            q.append((nx, ny))

print("距离地图:")
for row in dist:
    print(row)
```

#### 为什么它很重要

-   在一次遍历中为所有节点找到最近的源点距离
-   非常适合多源扩散问题
-   是多火源蔓延、影响区域、图上 Voronoi 划分等任务的基础

#### 一个温和的证明（为什么它有效）

因为所有源点都以距离 0 开始，并且广度优先搜索按照距离递增的顺序扩展，所以当一个节点第一次被访问时，它是通过从任意源点出发的最短可能路径到达的。

每个单元格恰好入队一次 → 时间复杂度 O(V + E)。
无需为每个源点单独运行广度优先搜索。

#### 亲自尝试

1.  在网格上标记多个源点（2），验证距离向外辐射。
2.  改变障碍物（1），观察波阵面如何避开它们。
3.  计算每个空闲单元格距离最近源点的步数。
4.  修改代码，使其返回哪个源点 ID 最先到达每个单元格。
5.  比较总成本与重复运行单源广度优先搜索的成本。

#### 测试用例

| 网格                              | 预期输出（距离） | 描述               |
| --------------------------------- | ---------------- | ------------------ |
| [[2,0,2]]                         | [0,1,0]          | 两个源点在边缘     |
| [[2,0,0],[0,1,0],[0,0,2]]         | 波从角落辐射     | 混合障碍物         |
| [[2,2,2]]                         | [0,0,0]          | 全是源点           |
| [[0,0,0],[0,0,0]] + 中心源点      | 中心 = 0, 角落 = 2 | 波阵面扩展         |
| 全部阻塞                          | 不变             | 无传播             |

#### 复杂度

-   时间：O(V + E)
-   空间：O(V)（用于队列和距离映射）

多源广度优先搜索是波阵面的合唱，它们共同扩展，每个音符都以完美的和谐到达其最近的听众。
### 309 拓扑排序（基于 DFS）

拓扑排序是对有向无环图（DAG）中顶点的一种线性排序，使得对于每条有向边 ( u \to v )，顶点 ( u ) 在排序中都出现在 ( v ) 之前。基于 DFS 的方法通过深度探索并记录完成时间来发现这种顺序。

#### 我们要解决什么问题？

我们想要一种方法来排序具有依赖关系的任务。
拓扑排序回答：*我们以何种顺序执行任务，才能让先决条件排在前面？*

典型应用场景：

- 构建系统（编译顺序）
- 课程先修计划安排
- 流水线阶段排序
- 依赖关系解析（例如软件包安装）

示例：
如果任务 A 必须在 B 和 C 之前完成，且 C 在 D 之前完成，那么一个有效的顺序是 A → C → D → B。

#### 它是如何工作的（通俗解释）？

DFS 从每个未访问的节点开始探索。
当一个节点完成（没有更多出边可探索）时，将其压入栈。
在所有 DFS 调用之后，反转栈，这就是你的拓扑顺序。

1. 将所有节点初始化为未访问状态
2. 对于每个节点 `v`：

   * 如果未访问，则运行 DFS
   * 探索完所有邻居后，将 `v` 压入栈
3. 反转栈以获得拓扑顺序

| 步骤 | 当前节点 | 操作           | 栈         |
| ---- | -------- | -------------- | ---------- |
| 1    | A        | 访问邻居       | []         |
| 2    | B        | 访问           | []         |
| 3    | D        | 访问           | []         |
| 4    | D 完成   | 压入 D         | [D]        |
| 5    | B 完成   | 压入 B         | [D, B]     |
| 6    | A 完成   | 压入 A         | [D, B, A]  |

反转后：[A, B, D]

#### 微型代码（简易版本）

C (基于 DFS 的拓扑排序)

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int graph[MAX][MAX];
bool visited[MAX];
int stack[MAX];
int top = -1;
int n;

void dfs(int v) {
    visited[v] = true;
    for (int u = 0; u < n; u++) {
        if (graph[v][u] && !visited[u]) {
            dfs(u);
        }
    }
    stack[++top] = v; // 探索完邻居后压栈
}

void topological_sort() {
    for (int i = 0; i < n; i++) {
        if (!visited[i]) dfs(i);
    }
    printf("拓扑顺序: ");
    while (top >= 0) printf("%d ", stack[top--]);
    printf("\n");
}

int main(void) {
    printf("输入顶点数: ");
    scanf("%d", &n);
    printf("输入邻接矩阵 (DAG):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);
    topological_sort();
}
```

Python (基于 DFS)

```python
graph = {
    0: [1, 2],
    1: [3],
    2: [3],
    3: []
}

visited = set()
stack = []

def dfs(v):
    visited.add(v)
    for u in graph[v]:
        if u not in visited:
            dfs(u)
    stack.append(v)

for v in graph:
    if v not in visited:
        dfs(v)

stack.reverse()
print("拓扑顺序:", stack)
```

#### 为什么它很重要

- 确保 DAG 中的依赖顺序
- 编译器、调度器和构建系统的基础
- 高级算法的基础：

  * Kahn 算法（基于队列）
  * DAG 最短路径 / 动态规划
  * 关键路径分析

#### 一个温和的证明（为什么它有效）

每个顶点在其所有后代都被探索之后才被压入栈。
因此，如果存在一条边 ( u \to v )，DFS 确保 ( v ) 先完成并被更早地压入栈，这意味着 ( u ) 将在栈中出现在更晚的位置。
反转栈从而保证了 ( u ) 在 ( v ) 之前。

不允许有环，如果发现后向边，则拓扑排序不可能。

#### 亲自尝试

1.  画一个 DAG，并将边标记为先决条件。
2.  运行 DFS 并记录完成时间。
3.  节点完成后压栈，然后反转顺序。
4.  添加一个环，看看为什么它会失败。
5.  与 Kahn 算法的结果进行比较。

#### 测试用例

| 图结构              | 边         | 拓扑顺序（可能结果） |
| ------------------- | ---------- | -------------------- |
| A→B→C               | A→B, B→C   | A B C                |
| A→C, B→C            | A→C, B→C   | A B C 或 B A C       |
| 0→1, 0→2, 1→3, 2→3 | DAG        | 0 2 1 3 或 0 1 2 3   |
| 链 0→1→2→3          | 线性 DAG   | 0 1 2 3              |
| 环 0→1→2→0          | 非 DAG     | 无有效顺序           |

#### 复杂度

- 时间：O(V + E)（每个节点和边访问一次）
- 空间：O(V)（栈 + 访问数组）

基于 DFS 的拓扑排序是你的依赖关系侦探，它深入探索，标记完成情况，并留下一条完美的先决条件路径。
### 310 拓扑排序（Kahn 算法）

Kahn 算法是一种基于队列执行拓扑排序的方法。
它不依赖递归，而是跟踪每个节点的入度（有多少条边指向该节点），并反复移除入度为 0 的节点。这种方法简洁、迭代，并且能自然地检测环。

#### 我们要解决什么问题？

我们希望在**有向无环图**中，得到一个任务的线性排序，使得每个任务都出现在其所有先决条件之后。

Kahn 方法在以下情况下特别方便：
- 你想要一个迭代（非递归）算法
- 你需要自动检测环
- 你正在构建调度器或编译器依赖解析器

示例：
如果 A 必须在 B 和 C 之前发生，且 C 在 D 之前，那么有效的顺序是：A C D B 或 A B C D。
Kahn 算法通过剥离"就绪"节点（那些没有剩余先决条件的节点）来构建这个顺序。

#### 它是如何工作的（通俗解释）？

每个节点初始有一个入度（传入边的数量）。
入度为 0 的节点可以开始处理。

1.  计算每个节点的入度
2.  将所有入度为 0 的节点入队
3.  当队列不为空时：
    *   弹出节点 `v` 并将其添加到拓扑顺序中
    *   对于 `v` 的每个邻居 `u`，将 `in-degree[u]` 减 1
    *   如果 `in-degree[u]` 变为 0，则将 `u` 入队
4.  如果所有节点都已处理 → 得到有效的拓扑顺序
    否则 → 检测到环

| 步骤 | 队列 | 弹出节点 | 更新后的入度 | 顺序 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | [A] | A | B:0, C:1, D:2 | [A] |
| 2 | [B] | B | C:0, D:2 | [A, B] |
| 3 | [C] | C | D:1 | [A, B, C] |
| 4 | [D] | D | – | [A, B, C, D] |

#### 精简代码（简易版本）

C 语言（Kahn 算法）

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int graph[MAX][MAX];
int indeg[MAX];
int queue[MAX];
int front = 0, rear = 0;
int n;

void enqueue(int v) { queue[rear++] = v; }
int dequeue() { return queue[front++]; }
bool empty() { return front == rear; }

void kahn_topo() {
    for (int v = 0; v < n; v++) {
        if (indeg[v] == 0) enqueue(v);
    }

    int count = 0;
    int order[MAX];

    while (!empty()) {
        int v = dequeue();
        order[count++] = v;
        for (int u = 0; u < n; u++) {
            if (graph[v][u]) {
                indeg[u]--;
                if (indeg[u] == 0) enqueue(u);
            }
        }
    }

    if (count != n) {
        printf("图中有环，无法进行拓扑排序\n");
    } else {
        printf("拓扑顺序: ");
        for (int i = 0; i < count; i++) printf("%d ", order[i]);
        printf("\n");
    }
}

int main(void) {
    printf("输入顶点数: ");
    scanf("%d", &n);

    printf("输入邻接矩阵:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            scanf("%d", &graph[i][j]);
            if (graph[i][j]) indeg[j]++;
        }

    kahn_topo();
}
```

Python（使用队列）

```python
from collections import deque

graph = {
    0: [1, 2],
    1: [3],
    2: [3],
    3: []
}

indeg = {v: 0 for v in graph}
for v in graph:
    for u in graph[v]:
        indeg[u] += 1

q = deque([v for v in graph if indeg[v] == 0])
order = []

while q:
    v = q.popleft()
    order.append(v)
    for u in graph[v]:
        indeg[u] -= 1
        if indeg[u] == 0:
            q.append(u)

if len(order) == len(graph):
    print("拓扑顺序:", order)
else:
    print("检测到环，无有效顺序")
```

#### 为什么它很重要？

- 完全迭代（无递归栈）
- 自然地检测环
- 适用于构建系统、任务规划器和依赖图
- 构成了 DAG 处理中 Kahn 调度算法的基础

#### 一个温和的证明（为什么它有效）

入度为 0 的节点没有先决条件，它们可以最先出现。
一旦被处理，它们就被移除（后继节点的入度减 1）。
这确保了：
- 每个节点仅在其所有依赖项之后才被处理
- 如果存在环，某些节点的入度永远不会变为 0 → 内置检测机制

由于每条边只被考虑一次，运行时间 = O(V + E)。

#### 亲自尝试

1.  手动构建一个小型 DAG，并逐步运行该算法。
2.  引入一个环（A→B→A）并观察检测过程。
3.  与基于 DFS 的排序进行比较，两者都有效。
4.  为队列添加优先级（最小顶点优先）以获得字典序最小的顺序。
5.  将其应用到课程先决条件规划器中。

#### 测试用例

| 图结构 | 边 | 输出 | 说明 |
| :--- | :--- | :--- | :--- |
| A→B→C | A→B, B→C | A B C | 线性链 |
| A→C, B→C | A→C, B→C | A B C 或 B A C | 多源 |
| 0→1, 0→2, 1→3, 2→3 | DAG | 0 1 2 3 或 0 2 1 3 | 多种有效顺序 |
| 0→1, 1→2, 2→0 | 环 | 无顺序 | 检测到环 |

#### 复杂度

- 时间：O(V + E)
- 空间：O(V)（队列，入度数组）

Kahn 算法是依赖关系的剥离器，一层一层地剥离节点，直到出现一个清晰的线性顺序。

## 第 32 节 强连通分量
### 311 Kosaraju 算法

Kosaraju 算法是寻找有向图中强连通分量（SCC）最清晰的方法之一。它使用两次深度优先搜索，一次在原图上，一次在其反转版本上，以逐层剥离出 SCC。

#### 我们要解决什么问题？

在有向图中，一个*强连通分量*是一个顶点的最大集合，使得集合中的每个顶点都可以从同一集合中的其他任意顶点到达。

Kosaraju 算法将图分组为这些 SCC。

这在以下方面很有用：

- 将图压缩成 DAG（元图）
- 编译器中的依赖分析
- 寻找循环或冗余模块
- 在动态规划或优化之前简化图

示例：
想象一个单向道路网络，SCC 就是那些你可以在其中任意两个城市之间旅行的城市群。

#### 它是如何工作的（通俗解释）？

Kosaraju 算法分两次 DFS 遍历运行：

1.  第一次 DFS（原图）：
    探索所有顶点。每次一个节点完成访问（递归结束）时，将其记录在一个栈中（按完成时间）。

2.  反转图：
    反转所有边（翻转方向）。

3.  第二次 DFS（反转图）：
    从栈中弹出节点（最先完成访问的节点优先）。
    从未访问节点开始的每次 DFS 形成一个强连通分量。

| 阶段 | 图         | 动作                     | 结果             |
| ---- | ---------- | ------------------------ | ---------------- |
| 1    | 原图       | DFS 所有顶点，记录完成顺序 | 顶点栈           |
| 2    | 反转图     | 按弹出顺序进行 DFS        | 识别 SCC         |
| 3    | 输出       | 每个 DFS 树               | SCC 列表         |

#### 简洁代码（简易版本）

C 语言（邻接表）

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int n;
int graph[MAX][MAX];
int rev[MAX][MAX];
bool visited[MAX];
int stack[MAX];
int top = -1;

void dfs1(int v) {
    visited[v] = true;
    for (int u = 0; u < n; u++) {
        if (graph[v][u] && !visited[u]) dfs1(u);
    }
    stack[++top] = v; // 访问结束后压栈
}

void dfs2(int v) {
    printf("%d ", v);
    visited[v] = true;
    for (int u = 0; u < n; u++) {
        if (rev[v][u] && !visited[u]) dfs2(u);
    }
}

int main(void) {
    printf("输入顶点数: ");
    scanf("%d", &n);

    printf("输入邻接矩阵（有向）:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            scanf("%d", &graph[i][j]);
            rev[j][i] = graph[i][j]; // 构建反转图
        }

    // 步骤 1: 第一次 DFS
    for (int i = 0; i < n; i++)
        visited[i] = false;

    for (int i = 0; i < n; i++)
        if (!visited[i]) dfs1(i);

    // 步骤 2: 在反转图上进行第二次 DFS
    for (int i = 0; i < n; i++)
        visited[i] = false;

    printf("强连通分量:\n");
    while (top >= 0) {
        int v = stack[top--];
        if (!visited[v]) {
            dfs2(v);
            printf("\n");
        }
    }
}
```

Python（使用列表）

```python
from collections import defaultdict

graph = defaultdict(list)
rev = defaultdict(list)

edges = [(0,1),(1,2),(2,0),(1,3)]
for u, v in edges:
    graph[u].append(v)
    rev[v].append(u)

visited = set()
stack = []

def dfs1(v):
    visited.add(v)
    for u in graph[v]:
        if u not in visited:
            dfs1(u)
    stack.append(v)

for v in graph:
    if v not in visited:
        dfs1(v)

visited.clear()

def dfs2(v, comp):
    visited.add(v)
    comp.append(v)
    for u in rev[v]:
        if u not in visited:
            dfs2(u, comp)

print("强连通分量:")
while stack:
    v = stack.pop()
    if v not in visited:
        comp = []
        dfs2(v, comp)
        print(comp)
```

#### 为什么它很重要

- 将有向图分割成相互可达的组
- 用于图的压缩（将循环图转换为 DAG）
- 帮助检测循环依赖
- 为组件级优化奠定基础

#### 一个温和的证明（为什么它有效）

1.  第一次 DFS 的完成时间确保我们首先处理“汇点”（后序）。
2.  反转边将汇点变为源点。
3.  第二次 DFS 找到从该源点可达的所有节点，即一个 SCC。
4.  每个节点被分配到恰好一个 SCC。

正确性源于完成时间的性质和可达性的对称性。

#### 自己动手试试

1.  画一个有环的有向图并手动运行这些步骤。
2.  跟踪完成顺序栈。
3.  反转所有边并开始弹出节点。
4.  每个 DFS 树 = 一个 SCC。
5.  将结果与 Tarjan 算法进行比较。

#### 测试用例

| 图                       | 边           | SCC           |
| ------------------------ | ------------ | ------------- |
| 0→1, 1→2, 2→0           | 循环         | {0,1,2}       |
| 0→1, 1→2                | 链           | {0}, {1}, {2} |
| 0→1, 1→2, 2→0, 2→3      | 循环 + 尾部  | {0,1,2}, {3}  |
| 0→1, 1→0, 1→2, 2→3, 3→2 | 两个 SCC     | {0,1}, {2,3}  |

#### 复杂度

-   时间：O(V + E)（两次 DFS 遍历）
-   空间：O(V + E)（图 + 栈）

Kosaraju 算法是你的镜像探索者，先遍历一次记录故事，然后翻转图，再倒放它以揭示每一个紧密相连的群体。
### 312 Tarjan 算法

Tarjan 算法在一次 DFS 遍历中就能找到有向图中的所有强连通分量（SCC），无需反转图。它是一种优雅而高效的方法，通过跟踪每个节点的发现时间和最低可达祖先来识别 SCC 的根节点。

#### 我们要解决什么问题？

我们需要将有向图的顶点分组为强连通分量（SCC），其中同一组内的每个节点都可以从组内其他节点到达。
与 Kosaraju 需要两次遍历的方法不同，Tarjan 算法在一次 DFS 中就能找到所有 SCC，这使得它在实践中更快，并且易于集成到更大的图算法中。

常见应用：

- 有向图中的环检测
- 为 DAG 处理进行分量缩合
- 死锁分析
- 编译器、网络和系统中的强连通性查询

示例：
想象一组由单向道路连接的城市。SCC 就是那些彼此都能互相到达的城市集群，形成了一个紧密连接的区域。

#### 它是如何工作的（通俗解释）？

每个顶点都有：

- 一个发现时间（disc），即它首次被访问的时间
- 一个低链接值（low），即可达的最小发现时间（包括后向边）

一个栈用于跟踪活跃的递归路径（当前的 DFS 栈）。
当一个顶点的 `disc` 等于其 `low` 值时，它就是某个 SCC 的根节点，从栈中弹出节点直到该顶点再次出现。

| 步骤 | 操作                               | 栈         | 找到的 SCC |                           |
| ---- | ---------------------------------- | ---------- | ---------- | ------------------------- |
| 1    | 访问节点，分配 disc 和 low         | [A]        | –          |                           |
| 2    | 深入（DFS 邻居节点）               | [A, B, C]  | –          |                           |
| 3    | 到达没有新邻居的节点               | 更新 low   | [A, B, C]  | –                         |
| 4    | 回溯，比较 low 值                  | [A, B]     | SCC {C}    |                           |
| 5    | 当 disc == low 时                  | 弹出 SCC   | [A]        | SCC {B, C}（如果连通）    |

#### 精简代码（简易版本）

C 语言（Tarjan 算法）

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int n;
int graph[MAX][MAX];
int disc[MAX], low[MAX], stack[MAX];
bool inStack[MAX];
int time_counter = 0, top = -1;

void dfs_tarjan(int v) {
    disc[v] = low[v] = ++time_counter;
    stack[++top] = v;
    inStack[v] = true;

    for (int u = 0; u < n; u++) {
        if (!graph[v][u]) continue;
        if (disc[u] == 0) {
            dfs_tarjan(u);
            if (low[u] < low[v]) low[v] = low[u];
        } else if (inStack[u]) {
            if (disc[u] < low[v]) low[v] = disc[u];
        }
    }

    if (disc[v] == low[v]) {
        printf("SCC: ");
        int w;
        do {
            w = stack[top--];
            inStack[w] = false;
            printf("%d ", w);
        } while (w != v);
        printf("\n");
    }
}

int main(void) {
    printf("输入顶点数: ");
    scanf("%d", &n);
    printf("输入邻接矩阵（有向）:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    for (int i = 0; i < n; i++) {
        disc[i] = 0;
        inStack[i] = false;
    }

    printf("强连通分量:\n");
    for (int i = 0; i < n; i++)
        if (disc[i] == 0)
            dfs_tarjan(i);
}
```

Python（使用邻接表）

```python
from collections import defaultdict

graph = defaultdict(list)
edges = [(0,1),(1,2),(2,0),(1,3)]
for u,v in edges:
    graph[u].append(v)

time = 0
disc = {}
low = {}
stack = []
in_stack = set()
sccs = []

def dfs(v):
    global time
    time += 1
    disc[v] = low[v] = time
    stack.append(v)
    in_stack.add(v)

    for u in graph[v]:
        if u not in disc:
            dfs(u)
            low[v] = min(low[v], low[u])
        elif u in in_stack:
            low[v] = min(low[v], disc[u])

    if disc[v] == low[v]:
        scc = []
        while True:
            w = stack.pop()
            in_stack.remove(w)
            scc.append(w)
            if w == v:
                break
        sccs.append(scc)

for v in list(graph.keys()):
    if v not in disc:
        dfs(v)

print("强连通分量:", sccs)
```

#### 为什么它很重要

- 一次 DFS 运行，高效且优雅
- 即时检测 SCC（无需反转边）
- 适用于在线算法（在发现时处理 SCC）
- 为环检测、缩合图、基于分量的优化提供支持

#### 一个温和的证明（为什么它有效）

- `disc[v]`：`v` 首次被访问的时间
- `low[v]`：通过后代或后向边可达的最小发现时间
- 当 `disc[v] == low[v]` 时，`v` 是其 SCC 的根节点（没有后向边能到达它上方）
- 从栈中弹出即可得到该分量内所有可达的节点

每条边只被检查一次 → 线性时间复杂度。

#### 自己动手试试

1.  在一个包含多个环的图上运行 Tarjan 算法。
2.  观察 `disc` 和 `low` 值。
3.  在每一步打印栈内容以查看分组情况。
4.  将 SCC 输出与 Kosaraju 算法的结果进行比较。
5.  尝试添加一个环并检查分组变化。

#### 测试用例

| 图结构              | 边          | SCCs          |
| ------------------- | ----------- | ------------- |
| 0→1, 1→2, 2→0      | 环          | {0,1,2}       |
| 0→1, 1→2           | 链          | {2}, {1}, {0} |
| 0→1, 1→2, 2→0, 2→3 | 环 + 尾部   | {0,1,2}, {3}  |
| 1→2, 2→3, 3→1, 3→4 | 两个组      | {1,2,3}, {4}  |

#### 复杂度

- 时间：O(V + E)（一次 DFS 遍历）
- 空间：O(V)（栈、数组）

Tarjan 算法就像你的精密探险家，按时间标记每个顶点，追踪最深路径，在一次优雅的遍历中精确地剥离出每一个强连通集群。
### 313 Gabow 算法

Gabow 算法是另一种用于寻找强连通分量（SCC）的优雅单遍方法。它不如 Tarjan 算法知名，但同样高效，使用两个栈来跟踪活动顶点和根节点。它是图探索中“栈纪律”的完美范例。

#### 我们要解决什么问题？

我们希望在**有向图**中找到所有强连通分量，即每个节点都能到达其他节点的子集。

Gabow 算法，与 Tarjan 算法类似，在单次 DFS 遍历中工作，但它不计算 `low-link` 值，而是使用两个栈来管理分量的发现和边界。

这种方法在流式、在线或迭代式 DFS 环境中特别有用，在这些环境中，显式的 low-link 计算可能会变得混乱。

应用：
- 循环分解
- 程序依赖分析
- 分量凝聚（DAG 创建）
- 强连通性测试

示例：
想象遍历一个道路网络。一个栈跟踪你去过的地方，另一个栈标记循环闭合的“检查点”，每个循环就是一个分量。

#### 它是如何工作的（通俗解释）？

Gabow 算法使用两个栈来跟踪发现顺序和分量边界：

- S（主栈）：存储所有当前活动的节点
- P（边界栈）：存储 SCC 的潜在根节点

步骤：
1.  执行 DFS。为每个节点分配一个递增的索引（`preorder`）
2.  将节点压入两个栈（S 和 P）
3.  对于每条边 ( v \to u )：
    *   如果 `u` 未访问 → 递归
    *   如果 `u` 在栈 S 上 → 调整边界栈 P
4.  探索完邻居后：
    *   如果 P 的栈顶是当前节点 `v`：
        *   弹出 P
        *   从 S 弹出直到 `v`
        *   弹出的节点形成一个 SCC

| 步骤 | 栈 S      | 栈 P      | 操作           |
| ---- | --------- | --------- | -------------- |
| 1    | [A]       | [A]       | 访问 A         |
| 2    | [A, B]    | [A, B]    | 访问 B         |
| 3    | [A, B, C] | [A, B, C] | 访问 C         |
| 4    | [A, B, C] | [A, B]    | 后向边 C→B     |
| 5    | [A, B]    | [A]       | 弹出 SCC {C}   |
| 6    | [A]       | []        | 弹出 SCC {B}   |
| 7    | []        | []        | 弹出 SCC {A}   |

#### 精简代码（简易版本）

C (Gabow's Algorithm)

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int graph[MAX][MAX];
int n;
int index_counter = 0;
int preorder[MAX];
bool onStack[MAX];
int stackS[MAX], topS = -1;
int stackP[MAX], topP = -1;

void dfs_gabow(int v) {
    preorder[v] = ++index_counter;
    stackS[++topS] = v;
    stackP[++topP] = v;
    onStack[v] = true;

    for (int u = 0; u < n; u++) {
        if (!graph[v][u]) continue;
        if (preorder[u] == 0) {
            dfs_gabow(u);
        } else if (onStack[u]) {
            while (preorder[stackP[topP]] > preorder[u]) topP--;
        }
    }

    if (stackP[topP] == v) {
        topP--;
        printf("SCC: ");
        int w;
        do {
            w = stackS[topS--];
            onStack[w] = false;
            printf("%d ", w);
        } while (w != v);
        printf("\n");
    }
}

int main(void) {
    printf("Enter number of vertices: ");
    scanf("%d", &n);
    printf("Enter adjacency matrix (directed):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    for (int i = 0; i < n; i++) {
        preorder[i] = 0;
        onStack[i] = false;
    }

    printf("Strongly Connected Components:\n");
    for (int i = 0; i < n; i++)
        if (preorder[i] == 0)
            dfs_gabow(i);
}
```

Python (Readable Implementation)

```python
from collections import defaultdict

graph = defaultdict(list)
edges = [(0,1),(1,2),(2,0),(1,3)]
for u,v in edges:
    graph[u].append(v)

index_counter = 0
preorder = {}
on_stack = set()
stackS, stackP = [], []
sccs = []

def dfs(v):
    global index_counter
    index_counter += 1
    preorder[v] = index_counter
    stackS.append(v)
    stackP.append(v)
    on_stack.add(v)

    for u in graph[v]:
        if u not in preorder:
            dfs(u)
        elif u in on_stack:
            while preorder[stackP[-1]] > preorder[u]:
                stackP.pop()

    if stackP and stackP[-1] == v:
        stackP.pop()
        comp = []
        while True:
            w = stackS.pop()
            on_stack.remove(w)
            comp.append(w)
            if w == v:
                break
        sccs.append(comp)

for v in graph:
    if v not in preorder:
        dfs(v)

print("Strongly Connected Components:", sccs)
```

#### 为什么它重要

- 单次 DFS 遍历，无需反向图
- 纯基于栈，在某些变体中避免了递归深度问题
- 对于大型图高效且实用
- 在某些应用中比 Tarjan 算法的 low-link 逻辑更简单

#### 一个温和的证明（为什么它有效）

每个节点获得一个先序索引。
第二个栈 P 跟踪可到达的最早根节点。
每当 P 的栈顶等于当前节点时，
S 中位于其上方的所有节点形成一个 SCC。

不变式：
- S 包含活动节点
- P 包含可能的根节点（按发现顺序排列）
- 当发现后向边时，P 被修剪到最小的可达祖先

这确保了每个 SCC 在其根节点完成时被精确识别一次。

#### 亲自尝试

1.  绘制一个图并追踪先序索引。
2.  观察栈 P 在后向边上如何收缩。
3.  记录弹出的 SCC。
4.  将输出与 Tarjan 算法进行比较。
5.  在 DAG、循环和混合图上尝试。

#### 测试用例

| 图                     | 边             | SCCs           |
| ---------------------- | -------------- | -------------- |
| 0→1, 1→2, 2→0          | {0,1,2}        | 一个大 SCC     |
| 0→1, 1→2               | {2}, {1}, {0}  | 链             |
| 0→1, 1→0, 2→3          | {0,1}, {2}, {3}| 混合           |
| 0→1, 1→2, 2→3, 3→1     | {1,2,3}, {0}   | 嵌套循环       |

#### 复杂度

- 时间：O(V + E)
- 空间：O(V)（两个栈 + 元数据）

Gabow 算法是你的双栈雕刻家，在一次优雅的扫描中从图中雕琢出 SCC，像工匠在最终切割前标记边缘一样，平衡探索和边界。
### 314 SCC DAG 构建

一旦我们找到了强连通分量（SCC），我们就可以构建**缩点图**，这是一个有向无环图（DAG），其中每个节点代表一个 SCC，如果某个 SCC 中的顶点指向另一个 SCC 中的顶点，则用边将它们连接起来。

这个结构至关重要，因为它将一个复杂的循环图转化为一个清晰的无环骨架，非常适合进行拓扑排序、动态规划和依赖分析。

#### 我们正在解决什么问题？

我们希望通过将每个 SCC 收缩为单个节点，将一个有向图简化为更简单的形式。
得到的图（称为*缩点图*）总是一个 DAG。

为什么要这样做？

- 简化对复杂系统的推理
- 在循环图上运行 DAG 算法（通过压缩循环）
- 执行组件级别的优化
- 研究强连通子系统之间的依赖关系

示例：
想象一个城市地图，其中 SCC 是紧密连接的街区。DAG 显示了交通如何在街区之间流动，而不是在街区内部流动。

#### 它是如何工作的（通俗解释）？

1.  找到 SCC（使用 Kosaraju、Tarjan 或 Gabow 算法）。
2.  为每个节点分配一个 SCC ID（例如，`comp[v] = c_id`）。
3.  创建一个新图，每个 SCC 对应一个节点。
4.  对于原图中的每条边 `(u, v)`：
    *   如果 `comp[u] != comp[v]`，则添加一条从 `comp[u]` 到 `comp[v]` 的边。
5.  移除重复边（或将边存储在集合中）。

现在新图中没有环，因为环已经被压缩在 SCC 内部了。

| 原图               | SCC              | DAG         |
| ------------------ | ---------------- | ----------- |
| 0→1, 1→2, 2→0, 2→3 | {0,1,2}, {3}     | C0 → C1     |
| 0→1, 1→2, 2→3, 3→1 | {1,2,3}, {0}     | C0 → C1     |
| 0→1, 1→2           | {0}, {1}, {2}    | 0 → 1 → 2   |

#### 微型代码（简易版本）

C 语言（使用 Tarjan 算法的输出）

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int n;
int graph[MAX][MAX];
int comp[MAX];   // 分量 ID
int comp_count = 0;

// 假设 comp[] 已由 Tarjan 或 Kosaraju 算法填充

void build_condensed_graph() {
    int new_graph[MAX][MAX] = {0};

    for (int u = 0; u < n; u++) {
        for (int v = 0; v < n; v++) {
            if (graph[u][v] && comp[u] != comp[v]) {
                new_graph[comp[u]][comp[v]] = 1;
            }
        }
    }

    printf("缩点图（邻接矩阵）：\n");
    for (int i = 0; i < comp_count; i++) {
        for (int j = 0; j < comp_count; j++) {
            printf("%d ", new_graph[i][j]);
        }
        printf("\n");
    }
}
```

*（假设 `comp[v]` 和 `comp_count` 已预先计算。）*

Python（配合 Tarjan 或 Kosaraju 算法得到的 SCC）

```python
from collections import defaultdict

# 假设我们已经有了 SCC
sccs = [[0,1,2], [3], [4,5]]
graph = {
    0: [1],
    1: [2,3],
    2: [0],
    3: [4],
    4: [5],
    5: []
}

# 步骤 1：分配分量 ID
comp_id = {}
for i, comp in enumerate(sccs):
    for v in comp:
        comp_id[v] = i

# 步骤 2：构建 DAG
dag = defaultdict(set)
for u in graph:
    for v in graph[u]:
        if comp_id[u] != comp_id[v]:
            dag[comp_id[u]].add(comp_id[v])

print("缩点图（作为 DAG）：")
for c in dag:
    print(c, "->", sorted(dag[c]))
```

#### 为什么它很重要

- 将循环图 → 无环图
- 支持拓扑排序、动态规划、路径计数
- 阐明组件间的依赖关系
- 用于编译器分析、基于 SCC 的优化、图压缩

#### 一个温和的证明（为什么它有效）

在每个 SCC 内部，每个顶点都可以到达其他任何顶点。
当边跨越 SCC 边界时，它们只朝一个方向走（因为返回会导致组件合并）。
因此，缩点图不可能包含环，证明它是一个 DAG。

DAG 中的每条边都代表了原图中组件之间的至少一条边。

#### 自己动手试试

1.  运行 Tarjan 算法以获得每个顶点的 `comp[v]`。
2.  使用分量 ID 构建 DAG 边。
3.  可视化原图与缩点图。
4.  对 DAG 进行拓扑排序。
5.  在 DAG 上使用动态规划来计算最长路径或可达性。

#### 测试用例

| 原图                        | SCC                    | 缩点后的边             |
| --------------------------- | ---------------------- | ---------------------- |
| 0→1, 1→2, 2→0, 2→3          | {0,1,2}, {3}           | 0→1                    |
| 0→1, 1→0, 1→2, 2→3, 3→2     | {0,1}, {2,3}           | 0→1                    |
| 0→1, 1→2, 2→3               | {0}, {1}, {2}, {3}     | 0→1→2→3                |
| 0→1, 1→2, 2→0               | {0,1,2}                | 无（单节点 DAG）       |

#### 复杂度

-   时间复杂度：O(V + E)（使用预计算的 SCC）
-   空间复杂度：O(V + E)（新的 DAG 结构）

SCC DAG 构建是您作为地图绘制者的步骤，将错综复杂的道路压缩成干净的高速公路，其中每个城市（SCC）是一个枢纽，而新地图最终是无环的，可以进行分析了。
### 315 强连通分量在线合并

强连通分量在线合并是一种动态方法，用于在图形随时间增长（添加新边）时维护强连通分量。我们不会在每次更新后从头重新计算强连通分量，而是在它们变得连通时*增量地合并*分量。

这是动态图算法的基础，其中边一条一条地到达，适用于在线系统、增量编译器和不断演化的依赖关系图。

#### 我们在解决什么问题？

我们希望在向有向图中添加边时维护强连通分量结构。

在像 Tarjan 或 Kosaraju 这样的静态算法中，强连通分量只计算一次。但如果随着时间的推移出现新的边，重新计算所有内容就太慢了。

强连通分量在线合并为我们提供了：

- 高效的增量更新（无需完全重新计算）
- 快速的分量合并
- 最新的缩合有向无环图

典型用例：

- 增量程序分析（新的依赖关系）
- 动态网络可达性
- 流式图处理
- 在线算法设计

#### 它是如何工作的（通俗解释）？

我们开始时将每个节点作为自己的强连通分量。
当添加一条新边 ( u \to v ) 时：

1.  检查 ( u ) 和 ( v ) 是否已经在同一个强连通分量中，如果是，则没有任何变化。
2.  如果不是，检查 v 所在的强连通分量是否能到达 u 所在的强连通分量（循环检测）。
3.  如果可达，则将两个强连通分量合并为一个。
4.  否则，添加一条从 `SCC(u)` 指向 `SCC(v)` 的有向无环图边。

我们维护：

-   用于强连通分量组的并查集 / DSU 结构
- 强连通分量之间的可达性或有向无环图边
- 用于快速循环检查的可选拓扑顺序

| 添加的边 | 操作         | 新的强连通分量 | 备注         |
| -------- | ------------ | -------------- | ------------ |
| 0→1      | 创建边       | {0}, {1}       | 分离         |
| 1→2      | 创建边       | {0}, {1}, {2}  | 分离         |
| 2→0      | 形成循环     | 合并 {0,1,2}   | 新的强连通分量 |
| 3→1      | 添加边       | {3}, {0,1,2}   | 不合并       |

#### 微型代码（概念演示）

Python（简化的 DSU + DAG 检查）

```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

class OnlineSCC:
    def __init__(self, n):
        self.n = n
        self.dsu = DSU(n)
        self.graph = [set() for _ in range(n)]

    def add_edge(self, u, v):
        su, sv = self.dsu.find(u), self.dsu.find(v)
        if su == sv:
            return  # 已经连通
        # 检查 v 所在的组件是否能到达 u 所在的组件
        if self._reachable(sv, su):
            # 合并组件
            self.dsu.union(su, sv)
            merged = self.dsu.find(su)
            self.graph[merged] = self.graph[su] | self.graph[sv]
        else:
            self.graph[su].add(sv)

    def _reachable(self, start, target, seen=None):
        if seen is None: seen = set()
        if start == target: return True
        seen.add(start)
        for nxt in self.graph[start]:
            if nxt not in seen and self._reachable(nxt, target, seen):
                return True
        return False

    def components(self):
        groups = {}
        for v in range(self.n):
            root = self.dsu.find(v)
            groups.setdefault(root, []).append(v)
        return list(groups.values())

# 示例
scc = OnlineSCC(4)
scc.add_edge(0, 1)
scc.add_edge(1, 2)
print("强连通分量:", scc.components())  # [[0], [1], [2], [3]]
scc.add_edge(2, 0)
print("强连通分量:", scc.components())  # [[0,1,2], [3]]
```

#### 为什么它很重要

- 无需重新计算的动态强连通分量维护
- 以 O(V + E) 的摊还时间复杂度处理边插入
- 支持实时图更新
- 是更高级算法的基础（支持删除的完全动态强连通分量）

#### 一个温和的证明（为什么它有效）

每次我们添加一条边时，会发生以下两种情况之一：

- 它连接了现有的强连通分量但没有创建循环 → 添加有向无环图边
- 它创建了一个循环 → 合并相关的强连通分量

由于合并强连通分量保留了有向无环图结构（合并会折叠循环），该算法始终保持缩合图的有效性。

通过维护强连通分量之间的可达性，我们可以高效地检测循环的形成。

#### 自己动手试试

1.  从 5 个节点、无边开始。
2.  逐步添加边，打印强连通分量。
3.  添加一条形成循环的回边 → 观察强连通分量合并。
4.  在每次更新后可视化缩合有向无环图。
5.  与使用 Tarjan 算法重新计算的结果进行比较，它们应该匹配！

#### 测试用例

| 步骤 | 边   | 强连通分量           |
| ---- | ---- | -------------------- |
| 1    | 0→1  | {0}, {1}, {2}, {3}   |
| 2    | 1→2  | {0}, {1}, {2}, {3}   |
| 3    | 2→0  | {0,1,2}, {3}         |
| 4    | 3→1  | {0,1,2}, {3}         |
| 5    | 2→3  | {0,1,2,3}            |

#### 复杂度

- 时间复杂度：O(V + E) 摊还（每次添加边）
- 空间复杂度：O(V + E)（图 + DSU）

强连通分量在线合并就像是你的动态雕刻家，随着新边的出现合并分量，无需从头开始就能维护结构。
### 316 组件标签传播

组件标签传播是一种简单、迭代的算法，通过反复在边之间传播最小标签，直到一个组件内的所有节点共享相同的标签，从而找到连通分量（或在对称图中找到强连通分量）。

它在概念上清晰，高度可并行化，并且构成了诸如 Google 的 Pregel、Apache Giraph 和 GraphX 等图处理框架的骨干，非常适合大规模或分布式系统。

#### 我们要解决什么问题？

我们想要识别组件，即那些可以相互到达的顶点组。
我们不使用深度递归或复杂的栈，而是迭代地在整个图中传播标签，直到收敛。

这种方法在以下情况下是理想的：

- 图非常庞大（递归无法处理）
- 你正在使用并行/分布式计算
- 你想要一个消息传递风格的算法

对于无向图，它找到连通分量。
对于有向图，它可以近似强连通分量（通常用作预处理步骤）。

示例：
想象一下在人群中传播一个 ID，每个节点告诉其邻居它所知的最小标签，并且每个人都更新以匹配其最小邻居的标签。最终，一个组内的所有人都共享相同的数字。

#### 它是如何工作的（通俗解释）？

1. 初始化：每个顶点的标签 = 其自身的 ID。
2. 迭代：对于每个顶点：

   * 查看所有邻居的标签。
   * 更新为看到的最小标签。
3. 重复直到没有标签改变（收敛）。

最终共享相同标签的所有节点属于同一个组件。

| 步骤 | 节点 | 当前标签 | 邻居标签 | 新标签           |
| ---- | ---- | ------------- | --------------- | ---------------- |
| 1    | A    | A             | {B, C}          | min(A, B, C) = A |
| 2    | B    | B             | {A}             | min(B, A) = A    |
| 3    | C    | C             | {A}             | min(C, A) = A    |

最终，A, B, C → 全部标记为 A。

#### 微型代码（简易版本）

C (迭代标签传播)

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int n, graph[MAX][MAX];
int label[MAX];

void label_propagation() {
    bool changed = true;
    while (changed) {
        changed = false;
        for (int v = 0; v < n; v++) {
            int min_label = label[v];
            for (int u = 0; u < n; u++) {
                if (graph[v][u]) {
                    if (label[u] < min_label)
                        min_label = label[u];
                }
            }
            if (min_label < label[v]) {
                label[v] = min_label;
                changed = true;
            }
        }
    }
}

int main(void) {
    printf("输入顶点数: ");
    scanf("%d", &n);
    printf("输入邻接矩阵 (无向图):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    for (int i = 0; i < n; i++)
        label[i] = i;

    label_propagation();

    printf("组件标签:\n");
    for (int i = 0; i < n; i++)
        printf("顶点 %d → 标签 %d\n", i, label[i]);
}
```

Python (简易版本)

```python
graph = {
    0: [1],
    1: [0, 2],
    2: [1],
    3: [4],
    4: [3]
}

labels = {v: v for v in graph}

changed = True
while changed:
    changed = False
    for v in graph:
        min_label = min([labels[v]] + [labels[u] for u in graph[v]])
        if min_label < labels[v]:
            labels[v] = min_label
            changed = True

print("组件标签:", labels)
```

#### 为什么它很重要

- 简单且可并行化，非常适合大数据系统
- 无需递归或栈，适用于 GPU、集群
- 本地计算，符合"像顶点一样思考"模型
- 适用于深度优先搜索不切实际的大规模图

#### 一个温和的证明（为什么它有效）

每次迭代都允许标签信息沿着边流动。
由于最小的标签总是传播，并且每次传播只会减少标签值，因此该过程必须收敛（没有无限更新）。
在收敛时，一个连通分量中的所有顶点共享相同的最小标签。

迭代次数 ≤ 图直径。

#### 亲自尝试

1. 在小型无向图上运行它，标签流很容易跟踪。
2. 尝试一个包含两个不连通部分的图，它们将分别稳定下来。
3. 在组件之间添加边并重新运行，观察标签合并。
4. 使用有向边，看看近似结果与强连通分量有何不同。
5. 并行实现（多线程循环）。

#### 测试用例

| 图         | 边        | 最终标签 |
| ------------- | ------------ | ------------ |
| 0–1–2         | (0,1), (1,2) | {0,0,0}      |
| 0–1, 2–3      | (0,1), (2,3) | {0,0,2,2}    |
| 0–1–2–3       | 链        | {0,0,0,0}    |
| 0–1, 1–2, 2–0 | 环        | {0,0,0}      |

#### 复杂度

- 时间：O(V × E)（最坏情况下，或 O(D × E)，其中 D = 直径）
- 空间：O(V + E)

组件标签传播是你的耳语算法，每个节点一次又一次地与邻居分享它的名字，直到所有可以相互到达的人都用同一个名字称呼自己。
### 317 基于路径的强连通分量算法

基于路径的强连通分量（SCC）算法是另一种优雅的单遍算法，用于在有向图中寻找强连通分量。它在精神上与 Tarjan 算法相似，但不是计算显式的 `low-link` 值，而是维护路径栈来检测何时遍历完一个完整的强连通分量。

该算法由 Donald B. Johnson 开发，它使用两个栈来跟踪深度优先搜索（DFS）的路径顺序和潜在的根节点。当一个顶点无法到达任何更早的顶点时，它就成为某个强连通分量的根，算法会将该强连通分量中的所有节点从路径栈中弹出。

#### 我们要解决什么问题？

我们想在有向图中找到强连通分量，即每个节点都能到达其他所有节点的顶点子集。基于路径的强连通分量算法提供了一种概念上简单且高效的方法来实现这一点，而无需进行 `low-link` 值的计算。

为什么它有用：

-   单次深度优先搜索遍历
-   基于栈的清晰逻辑
-   非常适合教学、推理和实现清晰度
-   易于扩展用于增量或流式强连通分量检测

应用：

-   编译器分析（强连通变量）
-   电路分析
-   死锁检测
-   数据流优化

#### 它是如何工作的（通俗解释）？

我们维护：

-   `index[v]`：发现顺序
-   栈 S：DFS 路径（当前探索中的节点）
-   栈 P：强连通分量根节点的候选者

算法步骤：

1.  访问节点 `v` 时，为其分配 `index[v]`。
2.  将 `v` 压入两个栈（`S` 和 `P`）。
3.  对于每个邻居 `u`：
    *   如果 `u` 未被访问 → 递归访问
    *   如果 `u` 在栈 `S` 上 → 调整栈 `P`，弹出栈顶元素直到栈顶元素的索引 ≤ `index[u]`
4.  在处理完所有邻居后，如果 `v` 位于栈 `P` 的顶部：
    *   弹出栈 `P` 的顶部
    *   从栈 `S` 中弹出元素，直到移除 `v`
    *   弹出的顶点构成一个强连通分量

| 步骤 | 当前节点    | 栈 S     | 栈 P     | 操作             |
| ---- | ----------- | -------- | -------- | ---------------- |
| 1    | A           | [A]      | [A]      | 访问 A           |
| 2    | B           | [A,B]    | [A,B]    | 访问 B           |
| 3    | C           | [A,B,C]  | [A,B,C]  | 访问 C           |
| 4    | 后向边 C→B  | [A,B,C]  | [A,B]    | 调整             |
| 5    | C 处理完毕  | [A,B,C]  | [A,B]    | 继续             |
| 6    | P 栈顶 == B | [A]      | [A]      | 找到强连通分量 {B,C} |

#### 精简代码（简易版本）

C 语言（基于路径的强连通分量）

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX 100
int graph[MAX][MAX];
int n;
int index_counter = 0;
int indexv[MAX];
bool onStack[MAX];
int stackS[MAX], topS = -1;
int stackP[MAX], topP = -1;

void dfs_scc(int v) {
    indexv[v] = ++index_counter;
    stackS[++topS] = v;
    stackP[++topP] = v;
    onStack[v] = true;

    for (int u = 0; u < n; u++) {
        if (!graph[v][u]) continue;
        if (indexv[u] == 0) {
            dfs_scc(u);
        } else if (onStack[u]) {
            while (indexv[stackP[topP]] > indexv[u])
                topP--;
        }
    }

    if (stackP[topP] == v) {
        topP--;
        printf("强连通分量: ");
        int w;
        do {
            w = stackS[topS--];
            onStack[w] = false;
            printf("%d ", w);
        } while (w != v);
        printf("\n");
    }
}

int main(void) {
    printf("请输入顶点数: ");
    scanf("%d", &n);
    printf("请输入邻接矩阵（有向图）:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    for (int i = 0; i < n; i++) indexv[i] = 0;

    printf("强连通分量:\n");
    for (int i = 0; i < n; i++)
        if (indexv[i] == 0)
            dfs_scc(i);
}
```

Python（可读性强的实现）

```python
from collections import defaultdict

graph = defaultdict(list)
edges = [(0,1),(1,2),(2,0),(1,3)]
for u,v in edges:
    graph[u].append(v)

index_counter = 0
index = {}
stackS, stackP = [], []
on_stack = set()
sccs = []

def dfs(v):
    global index_counter
    index_counter += 1
    index[v] = index_counter
    stackS.append(v)
    stackP.append(v)
    on_stack.add(v)

    for u in graph[v]:
        if u not in index:
            dfs(u)
        elif u in on_stack:
            while index[stackP[-1]] > index[u]:
                stackP.pop()

    if stackP and stackP[-1] == v:
        stackP.pop()
        comp = []
        while True:
            w = stackS.pop()
            on_stack.remove(w)
            comp.append(w)
            if w == v:
                break
        sccs.append(comp)

for v in list(graph.keys()):
    if v not in index:
        dfs(v)

print("强连通分量:", sccs)
```

#### 为什么它很重要

-   单次深度优先搜索
-   无需计算 low-link 值，完全基于路径推理
-   对于栈爱好者来说紧凑且直观
-   非常适合理论清晰度和教学用途
-   与 Tarjan 算法的性能相同，均为 O(V + E)

#### 一个温和的证明（为什么它有效）

-   每个顶点获得一个发现索引；
-   栈 `S` 存储活动路径节点；
-   栈 `P` 跟踪潜在的强连通分量根节点（仍可到达的最低索引）。
    当一个顶点处理完毕且等于栈 `P` 的顶部时，栈 `S` 中位于它之上的所有节点构成一个强连通分量，它们相互可达，并且没有一个能到达更早的节点。

不变式确保：

-   节点保留在栈中，直到找到其所属的强连通分量
-   强连通分量以逆拓扑顺序被发现

#### 自己动手试试

1.  在一个小图上运行它，打印每次调用后的 `index[v]` 和栈状态。
2.  添加一个环，跟踪栈 P 如何调整。
3.  将输出与 Tarjan 算法比较，它们应该匹配！
4.  将基于路径的弹出可视化为强连通分量的边界。
5.  尝试具有多个不相交强连通分量的图。

#### 测试用例

| 图结构              | 边类型 | 强连通分量       |
| ------------------- | ------ | ---------------- |
| 0→1, 1→2, 2→0      | 环     | {0,1,2}          |
| 0→1, 1→2           | 链     | {0}, {1}, {2}    |
| 0→1, 1→0, 2→3      | 混合   | {0,1}, {2}, {3}  |
| 0→1, 1→2, 2→3, 3→1 | 嵌套   | {1,2,3}, {0}     |

#### 复杂度

-   时间：O(V + E)
-   空间：O(V)

基于路径的强连通分量算法是你的栈芭蕾，每个顶点向前迈步，标记自己的位置，然后优雅地退场，留下一个编排紧密的强连通分量。
### 318 Kosaraju 并行版本

并行 Kosaraju 算法将 Kosaraju 经典的两次遍历 SCC 方法适配到在多个处理器或线程上运行，使其适用于无法被单线程高效处理的大规模图。它将繁重的工作——DFS 遍历和图反转——分配给多个工作单元。

这是 Kosaraju 思想在并行计算时代的自然演进：分割图、并发探索、合并 SCC。

#### 我们要解决什么问题？

我们希望通过利用并行硬件、多核 CPU、GPU 或分布式系统，高效地计算大规模有向图中的 SCC。

经典的 Kosaraju 算法：
1.  在原图上进行 DFS 以记录完成时间
2.  反转所有边以创建转置图
3.  按完成时间递减的顺序在反转后的图上进行 DFS

并行版本通过在处理器间划分顶点或边来加速每个阶段。

应用包括：
-   大型依赖图（包管理器、编译器）
-   网络图（页面连通性）
-   社交网络（相互可达性）
-   GPU 加速的分析和图挖掘

#### 它是如何工作的？（通俗解释）

我们将 Kosaraju 的两个关键遍历并行化：

1.  **并行正向 DFS（完成顺序）**：
    *   在线程间划分顶点。
    *   每个线程在其子图上独立运行 DFS。
    *   维护一个共享的完成时间栈（原子追加）。

2.  **图反转**：
    *   并行地将每条边 $(u, v)$ 反转为 $(v, u)$。
    *   每个线程处理边列表的一个切片。

3.  **并行反向 DFS（SCC 标记）**：
    *   线程从全局栈中弹出顶点。
    *   每个未访问的节点启动一个新的分量。
    *   DFS 标记与原子访问标志并发运行。

4.  **合并分量**：
    *   如果出现重叠集合，则使用并查集合并局部的 SCC 结果。

| 阶段 | 描述               | 是否并行化 |
| :--- | :----------------- | :--------- |
| 1    | 在原图上进行 DFS   | ✅ 是       |
| 2    | 反转边             | ✅ 是       |
| 3    | 在反转图上进行 DFS | ✅ 是       |
| 4    | 合并标签           | ✅ 是       |

#### 微型代码（伪代码 / Python 线程化草图）

> 此示例是概念性的，真实的并行实现使用任务队列、GPU 内核或工作窃取调度器。

```python
import threading
from collections import defaultdict

graph = defaultdict(list)
edges = [(0,1), (1,2), (2,0), (2,3), (3,4)]
for u, v in edges:
    graph[u].append(v)

n = 5
visited = [False] * n
finish_order = []
lock = threading.Lock()

def dfs_forward(v):
    visited[v] = True
    for u in graph[v]:
        if not visited[u]:
            dfs_forward(u)
    with lock:
        finish_order.append(v)

def parallel_forward():
    threads = []
    for v in range(n):
        if not visited[v]:
            t = threading.Thread(target=dfs_forward, args=(v,))
            t.start()
            threads.append(t)
    for t in threads:
        t.join()

# 并行反转图
rev = defaultdict(list)
for u in graph:
    for v in graph[u]:
        rev[v].append(u)

visited = [False] * n
components = []

def dfs_reverse(v, comp):
    visited[v] = True
    comp.append(v)
    for u in rev[v]:
        if not visited[u]:
            dfs_reverse(u, comp)

def parallel_reverse():
    while finish_order:
        v = finish_order.pop()
        if not visited[v]:
            comp = []
            dfs_reverse(v, comp)
            components.append(comp)

parallel_forward()
parallel_reverse()
print("SCCs:", components)
```

#### 为什么它很重要

-   支持大规模 SCC 计算
-   利用多核 / GPU 并行性
-   对于数据流分析、可达性和图凝聚至关重要
-   为科学计算中的大规模图分析提供动力

#### 一个温和的证明（为什么它有效）

Kosaraju 的正确性依赖于完成时间和反转后的可达性：
-   在正向遍历中，每个顶点 $v$ 获得一个完成时间 $t(v)$。
-   在反转图中，按 $t(v)$ 降序进行 DFS 确保每个 SCC 作为一个连续的 DFS 树被发现。

并行执行保持了这些不变量，因为：
1.  所有线程都遵守原子性的完成时间插入
2.  全局完成顺序保留了有效的拓扑顺序
3.  反向 DFS 仍然将相互可达的顶点一起发现

因此，即使在并发的 DFS 遍历下，只要访问是同步的，正确性仍然成立。

#### 亲自尝试

1.  将顶点分割成 $p$ 个分区。
2.  并行运行正向 DFS；记录全局完成栈。
3.  并发地反转边。
4.  通过从栈中弹出顶点来运行反向 DFS。
5.  与单线程 Kosaraju 的结果进行比较，它们应该匹配。

#### 测试用例

| 图类型     | 边                           | SCCs                     |
| :--------- | :--------------------------- | :----------------------- |
| 环         | 0→1, 1→2, 2→0               | {0, 1, 2}                |
| 链         | 0→1, 1→2                    | {0}, {1}, {2}            |
| 两个环     | 0→1, 1→0, 2→3, 3→2          | {0, 1}, {2, 3}           |
| 混合       | 0→1, 1→2, 2→3, 3→0, 4→5     | {0, 1, 2, 3}, {4, 5}     |

#### 复杂度

令 $V$ 为顶点数，$E$ 为边数，$p$ 为处理器数。

-   工作量：$O(V + E)$（与顺序算法相同）
-   并行时间：$T_p = O!\left(\frac{V + E}{p} + \text{sync\_cost}\right)$
-   空间：$O(V + E)$

并行 Kosaraju 就像是你的多声部合唱团，每个 DFS 和谐地歌唱，覆盖图的一部分，当回声平息时，强连通分量的完整和声便显现出来。
### 319 动态强连通分量维护

动态强连通分量维护处理的是在有向图随时间变化（边或顶点可能被添加或删除）时，如何维护其强连通分量。
目标是在每次变化后增量式地更新强连通分量，而不是从头开始重新计算。

这种方法在流式处理、交互式或演化系统中至关重要，在这些系统中，图代表了持续变化的现实世界结构。

#### 我们要解决什么问题？

我们希望在动态更新下跟踪强连通分量：

- 插入：新边可以连接强连通分量并形成更大的分量。
- 删除：移除边可能导致强连通分量分裂成更小的分量。

像 Tarjan 或 Kosaraju 这样的静态算法必须完全重新开始。
动态维护只更新受影响的组件，从而提高了大型、频繁变化图的效率。

应用场景包括：

- 增量编译
- 动态程序分析
- 实时依赖关系解析
- 连续图查询系统

#### 它是如何工作的？（通俗解释）

动态强连通分量算法维护：

- 一个表示强连通分量的凝聚有向无环图。
- 一个可达性结构，用于在插入时检测环。
- 对受影响节点的局部重新评估。

当添加一条边 $(u, v)$ 时：

1. 识别分量 $C_u$ 和 $C_v$。
2. 如果 $C_u = C_v$，则无变化。
3. 如果 $C_v$ 能到达 $C_u$，则形成新环 → 合并强连通分量。
4. 否则，在凝聚有向无环图中添加边 $C_u \to C_v$。

当删除一条边 $(u, v)$ 时：

1. 将其从图中移除。
2. 检查 $C_u$ 和 $C_v$ 是否仍然相互可达。
3. 如果不是，则在受影响的子图上局部重新计算强连通分量。

| 更新类型                     | 操作           | 结果               |
| -------------------------- | ------------- | ---------------- |
| 添加形成环的边     | 合并强连通分量    | 更大的分量 |
| 添加不形成环的边     | 仅添加 DAG 边 | 不合并         |
| 删除破坏环的边 | 分裂强连通分量     | 新的分量   |

#### 简化代码示例

```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

class DynamicSCC:
    def __init__(self, n):
        self.n = n
        self.dsu = DSU(n)
        self.graph = [set() for _ in range(n)]

    def add_edge(self, u, v):
        su, sv = self.dsu.find(u), self.dsu.find(v)
        if su == sv:
            return
        if self._reachable(sv, su):
            self.dsu.union(su, sv)
        else:
            self.graph[su].add(sv)

    def _reachable(self, start, target, seen=None):
        if seen is None: seen = set()
        if start == target: return True
        seen.add(start)
        for nxt in self.graph[start]:
            if nxt not in seen and self._reachable(nxt, target, seen):
                return True
        return False

    def components(self):
        comps = {}
        for v in range(self.n):
            root = self.dsu.find(v)
            comps.setdefault(root, []).append(v)
        return list(comps.values())

scc = DynamicSCC(4)
scc.add_edge(0, 1)
scc.add_edge(1, 2)
print(scc.components())  # [[0], [1], [2], [3]]
scc.add_edge(2, 0)
print(scc.components())  # [[0,1,2], [3]]
```

#### 为什么它很重要

- 对于图持续演化的长期运行系统效率很高
- 增量式更新强连通分量，而不是重建
- 支持连通性的实时查询
- 适用于流式图数据库、增量编译器、交互式建模工具

#### 一个温和的证明（为什么它有效）

对于插入：

- 如果 $(u, v)$ 连接了分量 $C_u$ 和 $C_v$
- 并且如果 $C_v$ 能到达 $C_u$，那么就会形成一个环
- 合并 $C_u$ 和 $C_v$ 会产生一个有效的强连通分量
- 凝聚有向无环图保持无环

对于删除：

- 如果移除破坏了可达性，强连通分量就会分裂
- 局部重新计算确保了正确性
- 其他未受影响的强连通分量保持有效

每次更新只修改受影响组件附近的局部邻域。

#### 自己动手试试

1. 构建一个小型有向图。
2. 逐步插入边并打印分量。
3. 添加一条回边以创建环，并观察合并过程。
4. 移除一条边并检查局部重新计算。
5. 将结果与完整的 Tarjan 重新计算进行比较。

#### 测试用例

| 步骤 | 边       | 强连通分量            |
| ---- | ---------- | --------------- |
| 1    | 0→1        | {0}, {1}, {2}   |
| 2    | 1→2        | {0}, {1}, {2}   |
| 3    | 2→0        | {0,1,2}         |
| 4    | 2→3        | {0,1,2}, {3}    |
| 5    | 移除 1→2 | {0,1}, {2}, {3} |

#### 复杂度

- 插入：$O(V + E)$ 摊还（包含可达性检查）
- 删除：局部重新计算，通常为亚线性
- 空间：$O(V + E)$

动态强连通分量维护提供了一个框架，可以在图演化时保持强连通分量的一致性，并能高效地适应增量增长和结构衰减。
### 320 加权图的强连通分量

强连通分量（SCC）检测通常针对无权图进行讨论，在无权图中边权与可达性无关。然而，在许多现实世界的系统中，权重编码了约束（成本、容量、优先级、概率），我们需要在这些加权条件下识别强连通性。此变体将 SCC 算法与加权边逻辑相结合，允许根据权重标准选择性地包含或排除边。

#### 我们要解决什么问题？

我们希望在加权有向图中找到强连通分量。标准的 SCC 算法忽略权重，它只检查可达性。在这里，我们基于满足给定权重谓词的边来定义 SCC：

$$
(u, v) \in E_w \quad \text{当且仅当} \quad w(u, v) \leq \theta
$$

然后，我们可以在由满足约束的边导出的子图上运行 SCC 算法（Tarjan、Kosaraju、Gabow、基于路径的）。

常见用例：

- **阈值化连通性**：保留成本低于 $\theta$ 的边。
- **容量受限系统**：仅包含容量 ≥ 阈值的边。
- **动态约束图**：随着阈值变化重新计算 SCC。
- **概率网络**：考虑概率 ≥ $p$ 的边。

#### 它是如何工作的？（通俗解释）

1.  从一个加权有向图 $G = (V, E, w)$ 开始。
2.  对权重应用一个谓词（例如 $w(u, v) \le \theta$）。
3.  构建一个过滤后的子图 $G_\theta = (V, E_\theta)$。
4.  在 $G_\theta$ 上运行一个标准的 SCC 算法。

结果将顶点分组为在权重约束下强连通的集合。

如果 $\theta$ 发生变化，分量可能会合并或分裂：

-   增加 $\theta$（放宽约束） → SCC 合并
-   减小 $\theta$（收紧约束） → SCC 分裂

| 阈值 $\theta$ | 包含的边      | SCC          |
| ------------- | ------------- | ------------ |
| 3             | $w \le 3$     | {A,B}, {C}   |
| 5             | $w \le 5$     | {A,B,C}      |

#### 微型代码（基于阈值的过滤）

Python 示例

```python
from collections import defaultdict

edges = [
    (0, 1, 2),
    (1, 2, 4),
    (2, 0, 1),
    (2, 3, 6),
    (3, 2, 6)
]

def build_subgraph(edges, theta):
    g = defaultdict(list)
    for u, v, w in edges:
        if w <= theta:
            g[u].append(v)
    return g

def dfs(v, g, visited, stack):
    visited.add(v)
    for u in g[v]:
        if u not in visited:
            dfs(u, g, visited, stack)
    stack.append(v)

def reverse_graph(g):
    rg = defaultdict(list)
    for u in g:
        for v in g[u]:
            rg[v].append(u)
    return rg

def kosaraju(g):
    visited, stack = set(), []
    for v in g:
        if v not in visited:
            dfs(v, g, visited, stack)
    rg = reverse_graph(g)
    visited.clear()
    comps = []
    while stack:
        v = stack.pop()
        if v not in visited:
            comp = []
            dfs(v, rg, visited, comp)
            comps.append(comp)
    return comps

theta = 4
g_theta = build_subgraph(edges, theta)
print("SCCs with threshold", theta, ":", kosaraju(g_theta))
```

输出：

```
SCCs with threshold 4 : [[2, 0, 1], [3]]
```

#### 为什么这很重要？

-   将权重约束纳入连通性分析。
-   在优化、路由和聚类中非常有用。
-   支持在阈值变化时进行增量重新计算。
-   支持多层图分析（改变 $\theta$ 以观察分量的演变）。

#### 一个温和的证明（为什么它有效）

加权图中的可达性取决于哪些边是活跃的。通过谓词过滤边保留了原始可达性的一个子集：

如果在 $G_\theta$ 中存在一条路径 $u \to v$，那么该路径上的所有边都满足 $w(e) \le \theta$。由于 SCC 仅依赖于可达性，应用于 $G_\theta$ 的标准算法能正确识别权重约束下的 SCC。

随着 $\theta$ 增加，边集 $E_\theta$ 单调增长：

$$
E_{\theta_1} \subseteq E_{\theta_2} \quad \text{对于} \quad \theta_1 < \theta_2
$$

因此，SCC 划分变得更粗（分量合并）。

#### 自己动手试试

1.  构建一个加权图。
2.  选择阈值 $\theta = 2, 4, 6$ 并记录 SCC。
3.  绘制随着 $\theta$ 增加，分量如何合并。
4.  尝试像 $w(u, v) \ge \theta$ 这样的谓词。
5.  与动态 SCC 维护结合，用于处理变化的阈值。

#### 测试用例

| $\theta$ | 包含的边             | SCC               |
| -------- | -------------------- | ----------------- |
| 2        | (0→1,2), (2→0)       | {0,2}, {1}, {3}   |
| 4        | (0→1), (1→2), (2→0)  | {0,1,2}, {3}      |
| 6        | 所有边               | {0,1,2,3}         |

#### 复杂度

-   过滤：$O(E)$
-   SCC 计算：$O(V + E_\theta)$
-   总计：$O(V + E)$
-   空间：$O(V + E)$

加权图的 SCC 将经典的连通性概念扩展到并非所有边都平等的场景，揭示了随着阈值变化，图的分层结构。

## 第 33 章 最短路径
### 321 Dijkstra（二叉堆）

Dijkstra（迪杰斯特拉）算法是计算带非负边权加权图中最短路径的基石。它通过不断扩展已知最短路径的前沿来工作，总是从当前距离最小的顶点开始扩展，就像波前在图上前进一样。

使用二叉堆（优先队列）可以高效地选择下一个最近的顶点，使得这个版本成为实际应用中的标准。

#### 我们要解决什么问题？

我们需要在一个带非负权重的有向或无向图中，找到从单个源顶点 $s$ 到所有其他顶点的最短路径。

给定一个加权图 $G = (V, E, w)$，其中对于所有 $(u, v)$ 有 $w(u, v) \ge 0$，任务是计算：

$$
\text{dist}[v] = \min_{\text{路径 } s \to v} \sum_{(u, v) \in \text{路径}} w(u, v)
$$

典型的应用场景：

- GPS导航（道路网络）
- 网络路由
- 游戏中的寻路
- 加权系统中的依赖关系解析

#### 它是如何工作的（通俗解释）？

该算法维护一个距离数组 `dist[]`，除源顶点外，所有顶点初始化为无穷大。

1.  设置 `dist[s] = 0`。
2.  使用一个最小优先队列来重复提取具有最小距离的顶点。
3.  对于每个邻居，尝试松弛边：

$$
\text{如果 } \text{dist}[u] + w(u, v) < \text{dist}[v], \text{ 则更新 } \text{dist}[v]
$$

4.  将邻居及其更新后的距离推入队列。
5.  继续直到队列为空。

| 步骤 | 提取的顶点 | 更新的距离            |
| ---- | ---------- | --------------------- |
| 1    | $s$        | $0$                   |
| 2    | 下一个最小 | 更新邻居              |
| 3    | 重复       | 直到所有顶点最终确定 |

这是一个贪心算法，一旦一个顶点被访问，它的最短距离就是最终的。

#### 精简代码（二叉堆）

C语言（使用带 `qsort` 的简单优先队列）

```c
#include <stdio.h>
#include <limits.h>
#include <stdbool.h>

#define MAX 100
#define INF INT_MAX

int n, graph[MAX][MAX];

void dijkstra(int src) {
    int dist[MAX], visited[MAX];
    for (int i = 0; i < n; i++) {
        dist[i] = INF;
        visited[i] = 0;
    }
    dist[src] = 0;

    for (int count = 0; count < n - 1; count++) {
        int u = -1, min = INF;
        for (int v = 0; v < n; v++)
            if (!visited[v] && dist[v] < min)
                min = dist[v], u = v;

        if (u == -1) break;
        visited[u] = 1;

        for (int v = 0; v < n; v++)
            if (graph[u][v] && !visited[v] &&
                dist[u] + graph[u][v] < dist[v])
                dist[v] = dist[u] + graph[u][v];
    }

    printf("顶点\t距离\n");
    for (int i = 0; i < n; i++)
        printf("%d\t%d\n", i, dist[i]);
}

int main(void) {
    printf("输入顶点数: ");
    scanf("%d", &n);
    printf("输入邻接矩阵 (无边则为0):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);
    dijkstra(0);
}
```

Python（基于堆的实现）

```python
import heapq

def dijkstra(graph, src):
    n = len(graph)
    dist = [float('inf')] * n
    dist[src] = 0
    pq = [(0, src)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist

graph = {
    0: [(1, 2), (2, 4)],
    1: [(2, 1), (3, 7)],
    2: [(4, 3)],
    3: [(4, 1)],
    4: []
}

print(dijkstra(graph, 0))
```

#### 为什么它很重要

-   当权重非负时，能高效计算最短路径
-   确定性和贪心性，产生最优路径
-   广泛应用于路由、物流和人工智能搜索
-   构成 A* 算法和 Johnson 算法的基础

#### 一个温和的证明（为什么它有效）

Dijkstra 的不变性：
当一个顶点 $u$ 从优先队列中提取时，$\text{dist}[u]$ 是最终的。

证明思路：
所有到 $u$ 的替代路径都必须经过具有大于或等于当前暂定距离的顶点，因为边是非负的。因此，不存在更短的路径。

通过归纳法，算法为每个顶点分配了正确的最短距离。

#### 自己动手试试

1.  在一个有 5 个顶点和随机权重的图上运行。
2.  添加一条权重更小的边，观察路径如何更新。
3.  移除负权边，注意错误的结果。
4.  逐步可视化前沿扩展过程。
5.  在同一个图上与 Bellman–Ford（贝尔曼-福特）算法进行比较。

#### 测试用例

| 图类型 | 边                                            | 从 0 出发的最短路径 |
| ------ | --------------------------------------------- | ------------------- |
| 简单图 | 0→1(2), 0→2(4), 1→2(1), 2→4(3), 1→3(7), 3→4(1) | [0, 2, 3, 7, 6]     |
| 链式图 | 0→1(1), 1→2(1), 2→3(1)                        | [0, 1, 2, 3]        |
| 星形图 | 0→1(5), 0→2(2), 0→3(8)                        | [0, 5, 2, 8]        |

#### 复杂度

-   时间：使用二叉堆时为 $O((V + E)\log V)$
-   空间：$O(V + E)$
-   仅当 $w(u, v) \ge 0$ 时有效

Dijkstra（二叉堆）是图搜索的主力军，贪心但精确，总是追逐下一个最近的前沿，直到所有路径各归其位。
### 322 Dijkstra（斐波那契堆）

通过将二叉堆替换为斐波那契堆，可以进一步优化 Dijkstra 算法，因为斐波那契堆提供了更快的 decrease-key 操作。这一改进降低了整体时间复杂度，使其更适用于稠密图或渐近效率至关重要的理论分析。

虽然常数因子更高，但渐近时间改进为：

$$
O(E + V \log V)
$$

相比之下，二叉堆版本为 $O((V + E)\log V)$。

#### 我们要解决什么问题？

我们正在计算具有非负权重的有向或无向加权图中的单源最短路径，但我们希望优化优先队列操作以提高理论性能。

给定 $G = (V, E, w)$ 且 $w(u, v) \ge 0$，任务仍然是：

$$
\text{dist}[v] = \min_{\text{路径 } s \to v} \sum_{(u,v) \in \text{路径}} w(u, v)
$$

区别在于我们管理选择下一个要处理顶点的优先队列的方式。

#### 它是如何工作的（通俗解释）？

Dijkstra 算法的逻辑保持不变，仅用于顶点选择和更新的数据结构不同。

1.  将所有距离初始化为 $\infty$，除了 $\text{dist}[s] = 0$。
2.  将所有顶点按其当前距离作为键插入斐波那契堆。
3.  反复提取距离最小的顶点 $u$。
4.  对于每个邻居 $(u, v)$：
    *   如果 $\text{dist}[u] + w(u, v) < \text{dist}[v]$，则更新：
        $$
        \text{dist}[v] \gets \text{dist}[u] + w(u, v)
        $$
        并在堆中对 $v$ 调用 decrease-key。
5.  继续直到所有顶点都被最终确定。

斐波那契堆提供：

-   `extract-min`：$O(\log V)$ 摊还时间
-   `decrease-key`：$O(1)$ 摊还时间

这提高了在边松弛操作占主导的稠密图中的性能。

#### 微型代码（Python，简化斐波那契堆）

此代码说明了结构但省略了堆的完整细节，生产实现使用像 `networkx` 这样的库或专门的数据结构。

```python
from heapq import heappush, heappop  # 用于演示的替代品

def dijkstra_fib(graph, src):
    n = len(graph)
    dist = [float('inf')] * n
    dist[src] = 0
    visited = [False] * n
    heap = [(0, src)]

    while heap:
        d, u = heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        for v, w in graph[u]:
            if not visited[v] and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heappush(heap, (dist[v], v))
    return dist

graph = {
    0: [(1, 1), (2, 4)],
    1: [(2, 2), (3, 6)],
    2: [(3, 3)],
    3: []
}

print(dijkstra_fib(graph, 0))
```

*(以上使用 `heapq` 进行说明；真正的斐波那契堆能提供更好的理论界限。)*

#### 为什么这很重要

-   将理论时间改进为 $O(E + V \log V)$
-   展示了使用高级堆进行渐近优化的方法
-   用于稠密网络、理论研究和竞赛问题
-   是 Johnson 全源最短路径和最小平均环等算法的基础

#### 一个温和的证明（为什么它有效）

正确性与 Dijkstra 原始证明完全相同：
当顶点 $u$ 被提取（最小键）时，其最短距离是最终确定的，因为所有边权重都是非负的。

堆的选择只影响效率：

-   二叉堆：每次 `decrease-key` = $O(\log V)$
-   斐波那契堆：每次 `decrease-key` = $O(1)$ 摊还时间

总操作数：

-   $V$ 次提取 × $O(\log V)$
-   $E$ 次减少 × $O(1)$

因此总时间：

$$
O(V \log V + E) = O(E + V \log V)
$$

#### 亲自尝试

1.  构建一个稠密图（例如 $V=1000, E \approx V^2$）。
2.  与二叉堆版本的运行时间进行比较。
3.  可视化优先队列操作。
4.  手动实现 `decrease-key` 以获得深入理解。
5.  探索使用此版本的 Johnson 算法。

#### 测试用例

| 图类型   | 边                                | 从 0 出发的最短路径 |
| -------- | --------------------------------- | ------------------- |
| 链式     | 0→1(1), 1→2(2), 2→3(3)            | [0, 1, 3, 6]        |
| 三角形   | 0→1(2), 1→2(2), 0→2(5)            | [0, 2, 4]           |
| 稠密图   | 所有权重较小的顶点对              | 高效工作            |

#### 复杂度

-   时间：$O(E + V \log V)$
-   空间：$O(V + E)$
-   仅当 $w(u, v) \ge 0$ 时有效

Dijkstra（斐波那契堆）展示了数据结构选择如何改变一个算法，通过精心设计优先操作，相同的理念变得更加锐利。
### 323 Bellman–Ford（贝尔曼-福特）

Bellman–Ford（贝尔曼-福特）算法解决了可能包含负权边的图的单源最短路径问题。
与 Dijkstra（迪杰斯特拉）算法不同，它不依赖于贪心选择，并且可以处理 $w(u, v) < 0$ 的边，只要从源点出发无法到达负权环。

它系统地多次松弛每条边，确保考虑了长度最多为 $V-1$ 的所有路径。

#### 我们要解决什么问题？

我们想要在一个可能包含负权重的加权有向图中，计算从源点 $s$ 出发的最短路径。

给定 $G = (V, E, w)$，为所有 $v \in V$ 找到：

$$
\text{dist}[v] = \min_{\text{路径 } s \to v} \sum_{(u,v) \in \text{路径}} w(u,v)
$$

如果从 $s$ 出发可以到达一个负权环，则最短路径是未定义的（它可以无限减小）。
Bellman–Ford（贝尔曼-福特）算法会明确地检测到这种情况。

#### 它是如何工作的（通俗解释）？

Bellman–Ford（贝尔曼-福特）算法反复使用边松弛操作。

1. 初始化 $\text{dist}[s] = 0$，其他为 $\infty$。
2. 重复 $V - 1$ 次：
   对于每条边 $(u, v)$：
   $$
   \text{如果 } \text{dist}[u] + w(u, v) < \text{dist}[v], \text{ 则更新 } \text{dist}[v]
   $$
3. 经过 $V - 1$ 轮后，所有最短路径都已确定。
4. 再进行一轮：如果任何边仍然可以松弛，则存在负权环。

| 迭代次数 | 更新的顶点                | 备注         |
| -------- | ------------------------- | ------------ |
| 1        | 源点的邻居                | 第一层       |
| 2        | 下一层                    | 传播         |
| ...      | ...                       | ...          |
| $V-1$    | 所有最短路径稳定          | 完成         |

#### 微型代码（C 语言示例）

```c
#include <stdio.h>
#include <limits.h>

#define MAX 100
#define INF 1000000000

typedef struct { int u, v, w; } Edge;
Edge edges[MAX];
int dist[MAX];

int main(void) {
    int V, E, s;
    printf("输入顶点数、边数、源点: ");
    scanf("%d %d %d", &V, &E, &s);
    printf("输入边 (u v w):\n");
    for (int i = 0; i < E; i++)
        scanf("%d %d %d", &edges[i].u, &edges[i].v, &edges[i].w);

    for (int i = 0; i < V; i++) dist[i] = INF;
    dist[s] = 0;

    for (int i = 1; i < V; i++)
        for (int j = 0; j < E; j++) {
            int u = edges[j].u, v = edges[j].v, w = edges[j].w;
            if (dist[u] != INF && dist[u] + w < dist[v])
                dist[v] = dist[u] + w;
        }

    for (int j = 0; j < E; j++) {
        int u = edges[j].u, v = edges[j].v, w = edges[j].w;
        if (dist[u] != INF && dist[u] + w < dist[v]) {
            printf("检测到负环\n");
            return 0;
        }
    }

    printf("顶点\t距离\n");
    for (int i = 0; i < V; i++)
        printf("%d\t%d\n", i, dist[i]);
}
```

Python（可读版本）

```python
def bellman_ford(V, edges, src):
    dist = [float('inf')] * V
    dist[src] = 0

    for _ in range(V - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # 检测负环
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            raise ValueError("检测到负环")

    return dist

edges = [(0, 1, 6), (0, 2, 7), (1, 2, 8), (1, 3, 5),
         (1, 4, -4), (2, 3, -3), (2, 4, 9), (3, 1, -2),
         (4, 3, 7)]

print(bellman_ford(5, edges, 0))
```

#### 为什么它很重要

- 适用于负权重
- 检测负权环
- 逻辑更简单，易于证明正确性
- 用于货币套利、动态规划、策略评估

#### 一个温和的证明（为什么它有效）

一条最短路径最多有 $V-1$ 条边（比这更长的路径必然包含环）。
每次迭代确保松弛了长度不超过该值的所有路径。
因此，经过 $V-1$ 轮后，所有最短路径都被找到。

第 $(V)$ 轮检测到进一步的改进，表明存在负环。

形式上，经过第 $k$ 次迭代后，
$\text{dist}[v]$ 是从 $s$ 到 $v$ 使用最多 $k$ 条边的最短路径的长度。

#### 亲自尝试

1.  在具有负权重的图上运行。
2.  添加一个负环并观察检测结果。
3.  与 Dijkstra（迪杰斯特拉）算法比较结果（Dijkstra 算法在负边情况下会失败）。
4.  可视化每次迭代的松弛过程。
5.  用它来检测货币兑换图中的套利机会。

#### 测试用例

| 图类型       | 边                          | 最短距离         |
| ------------ | --------------------------- | ---------------- |
| 链式         | 0→1(5), 1→2(-2)             | [0, 5, 3]        |
| 负边         | 0→1(4), 0→2(5), 1→2(-10)    | [0, 4, -6]       |
| 负环         | 0→1(1), 1→2(-2), 2→0(-1)    | 检测到负环       |

#### 复杂度

- 时间：$O(VE)$
- 空间：$O(V)$
- 处理：$w(u, v) \ge -\infty$，无负环

Bellman–Ford（贝尔曼-福特）是最短路径算法中稳健的行者，比 Dijkstra（迪杰斯特拉）慢，但不受负边影响，并且总能警惕那些打破规则的环。
### 324 SPFA（队列优化）

最短路径快速算法（SPFA）是 Bellman–Ford 算法的一种优化实现，它使用队列来避免不必要的松弛操作。
与在每次迭代中松弛所有边不同，SPFA 只处理最近距离被更新的顶点，这通常能带来更快的平均性能，尤其是在稀疏图或没有负环的图中。

在最坏情况下，其时间复杂度仍为 $O(VE)$，但典型性能更接近 $O(E)$。

#### 我们要解决什么问题？

我们希望在一个可能包含负权边但不含负权环的图中，找到单源最短路径。

给定一个有向图 $G = (V, E, w)$，其边权 $w(u, v)$ 可能为负，我们计算：

$$
\text{dist}[v] = \min_{\text{路径 } s \to v} \sum_{(u, v) \in \text{路径}} w(u, v)
$$

Bellman–Ford 算法进行 $V-1$ 轮边松弛可能造成浪费；SPFA 避免了重新检查那些距离没有改善的顶点的出边。

#### 它是如何工作的（通俗解释）？

SPFA 维护一个队列，其中包含那些出边可能导致松弛的顶点。
每当一个顶点的距离得到改善，它就会被加入队列等待处理。

1. 初始化 $\text{dist}[s] = 0$，其他顶点为 $\infty$。
2. 将 $s$ 推入队列。
3. 当队列不为空时：

   * 弹出顶点 $u$。
   * 对于每条边 $(u, v)$：
     $$
     \text{如果 } \text{dist}[u] + w(u, v) < \text{dist}[v], \text{ 则更新 } \text{dist}[v]
     $$
   * 如果 $\text{dist}[v]$ 发生改变且 $v$ 不在队列中，则将 $v$ 入队。
4. 持续进行直到队列为空。

| 步骤 | 队列                     | 操作                       |
| ---- | ------------------------ | -------------------------- |
| 1    | [s]                      | 开始                       |
| 2    | 弹出 u，松弛其邻居       | 将距离改善的顶点入队       |
| 3    | 重复                     | 直到没有更多改善           |

SPFA 采用一种惰性松弛策略，由实际的距离更新来驱动。

#### 微型代码（C 语言示例）

```c
#include <stdio.h>
#include <stdbool.h>
#include <limits.h>

#define MAX 100
#define INF 1000000000

typedef struct { int v, w; } Edge;
Edge graph[MAX][MAX];
int deg[MAX];

int queue[MAX], front = 0, rear = 0;
bool in_queue[MAX];
int dist[MAX];

void enqueue(int x) {
    queue[rear++] = x;
    in_queue[x] = true;
}
int dequeue() {
    int x = queue[front++];
    in_queue[x] = false;
    return x;
}

int main(void) {
    int V, E, s;
    printf("输入顶点数、边数、源点: ");
    scanf("%d %d %d", &V, &E, &s);

    for (int i = 0; i < V; i++) deg[i] = 0;
    printf("输入边 (u v w):\n");
    for (int i = 0; i < E; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        graph[u][deg[u]].v = v;
        graph[u][deg[u]].w = w;
        deg[u]++;
    }

    for (int i = 0; i < V; i++) dist[i] = INF;
    dist[s] = 0;
    enqueue(s);

    while (front < rear) {
        int u = dequeue();
        for (int i = 0; i < deg[u]; i++) {
            int v = graph[u][i].v, w = graph[u][i].w;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                if (!in_queue[v]) enqueue(v);
            }
        }
    }

    printf("顶点\t距离\n");
    for (int i = 0; i < V; i++)
        printf("%d\t%d\n", i, dist[i]);
}
```

Python（基于队列的实现）

```python
from collections import deque

def spfa(V, edges, src):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))
    
    dist = [float('inf')] * V
    in_queue = [False] * V
    dist[src] = 0

    q = deque([src])
    in_queue[src] = True

    while q:
        u = q.popleft()
        in_queue[u] = False
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if not in_queue[v]:
                    q.append(v)
                    in_queue[v] = True
    return dist

edges = [(0,1,2),(0,2,4),(1,2,-1),(2,3,2)]
print(spfa(4, edges, 0))
```

#### 为什么它很重要

- 相比 Bellman–Ford 有实际的性能提升
- 对于稀疏图和接近无环的图非常高效
- 可以处理负权边
- 用于网络优化、实时路由、流系统等领域

#### 一个温和的证明（为什么它有效）

每个顶点在其距离改善时进入队列。
每个顶点最多有 $V-1$ 次改善（因为不存在包含更多边的更短路径）。
因此，每次松弛最终都会收敛到与 Bellman–Ford 算法相同的固定点。

SPFA 是一种异步松弛方法：

- 在无非负环的情况下仍能保证正确性
- 如果一个顶点入队次数 $\ge V$ 次，则检测到负环

要检查负环：

- 维护一个松弛次数计数器 `count[v]`
- 如果 `count[v] > V`，则报告存在环

#### 动手尝试

1.  在包含负权边的图上测试。
2.  与 Bellman–Ford 比较运行时间。
3.  添加负环检测功能。
4.  尝试稀疏图和稠密图。
5.  测量执行过程中队列长度的变化。

#### 测试用例

| 图类型       | 边                                 | 结果           |
| ------------ | ---------------------------------- | -------------- |
| 简单图       | 0→1(2), 0→2(4), 1→2(-1), 2→3(2)    | [0, 2, 1, 3]   |
| 含负权边     | 0→1(5), 1→2(-3)                    | [0, 5, 2]      |
| 含环         | 0→1(1), 1→2(-2), 2→0(1)            | 检测到负环     |

#### 复杂度

- 平均情况：$O(E)$
- 最坏情况：$O(VE)$
- 空间：$O(V + E)$

SPFA（队列优化）是灵活的 Bellman–Ford 算法，只在需要时才做出反应，在实践中收敛更快，同时保持了相同的正确性保证。
### 325 A* 搜索

A*（A-star）算法结合了 Dijkstra 的最短路径算法和最佳优先搜索，并由一个启发式函数引导。它通过总是扩展看起来最接近目标的顶点，来高效地找到从起始节点到目标节点的最短路径，其估计依据是：

$$
f(v) = g(v) + h(v)
$$

其中

- $g(v)$ = 从起点到 $v$ 的代价（已知），
- $h(v)$ = 从 $v$ 到目标的启发式估计（猜测），
- $f(v)$ = 通过 $v$ 的总估计代价。

当启发函数是可采纳的（从不高估）时，A* 保证最优性。

#### 我们要解决什么问题？

我们希望在加权图（通常是空间图）中找到从源点 $s$ 到目标 $t$ 的最短路径，并利用关于目标的额外知识来引导搜索。

给定 $G = (V, E, w)$ 和一个启发函数 $h(v)$，任务是使以下值最小化：

$$
\text{cost}(s, t) = \min_{\text{路径 } s \to t} \sum_{(u, v) \in \text{路径}} w(u, v)
$$

应用：

- 寻路（游戏、机器人、导航）
- 规划系统（人工智能、物流）
- 网格和地图搜索
- 状态空间探索

#### 它是如何工作的（通俗解释）？

A* 的行为类似于 Dijkstra 算法，但它不是扩展离起点最近的节点（$g$），而是扩展估计总代价最小的节点（$f = g + h$）。

1.  初始化所有距离：`g[start] = 0`，其他为 $\infty$。
2.  计算 `f[start] = h[start]`。
3.  将 `(f, node)` 推入优先队列。
4.  当队列不为空时：

    *   弹出具有最小 $f(u)$ 的节点 $u$。
    *   如果 $u = \text{goal}$，停止，路径已找到。
    *   对于每个邻居 $v$：
        $$
        g'(v) = g(u) + w(u, v)
        $$
        如果 $g'(v) < g(v)$，则更新：
        $$
        g(v) = g'(v), \quad f(v) = g(v) + h(v)
        $$
        将 $v$ 推入队列。
5.  使用父指针重建路径。

| 节点  | $g(v)$ | $h(v)$    | $f(v) = g + h$ | 已扩展？ |
| ----- | ------ | --------- | -------------- | --------- |
| start | 0      | heuristic | heuristic      | ✅         |
| ...   | ...    | ...       | ...            | ...       |

启发函数引导探索，聚焦于有希望的路线。

#### 微型代码（Python 示例）

基于网格的 A*（曼哈顿启发函数）：

```python
import heapq

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    g = {start: 0}
    f = {start: heuristic(start, goal)}
    pq = [(f[start], start)]
    parent = {start: None}
    visited = set()

    while pq:
        _, current = heapq.heappop(pq)
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = parent[current]
            return path[::-1]

        visited.add(current)
        x, y = current
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                tentative = g[current] + 1
                if tentative < g.get(neighbor, float('inf')):
                    g[neighbor] = tentative
                    f[neighbor] = tentative + heuristic(neighbor, goal)
                    parent[neighbor] = current
                    heapq.heappush(pq, (f[neighbor], neighbor))
    return None

grid = [
    [0,0,0,0],
    [1,1,0,1],
    [0,0,0,0],
    [0,1,1,0],
]
start = (0,0)
goal = (3,3)
path = astar(grid, start, goal)
print(path)
```

#### 为什么它很重要

-   对于目标导向的搜索，比 Dijkstra 更快
-   如果启发函数是可采纳的（$h(v) \le \text{真实代价}$），则最优
-   如果启发函数也是一致的（满足三角不等式），则高效
-   广泛应用于人工智能、机器人、导航、路线规划

#### 一个温和的证明（为什么它有效）

如果启发函数 $h(v)$ 从不高于真实的剩余代价：

$$
h(v) \le \text{cost}(v, t)
$$

那么 $f(v) = g(v) + h(v)$ 总是真实代价的一个下界。因此，当目标节点被取出（$f$ 最小）时，路径保证是最优的。

如果 $h$ 还满足一致性：
$$
h(u) \le w(u, v) + h(v)
$$
那么 $f$ 值是非递减的，并且每个节点只被扩展一次。

#### 亲自尝试

1.  在有障碍物的网格上实现 A*。
2.  尝试不同的启发函数（曼哈顿距离、欧几里得距离）。
3.  设置 $h(v) = 0$ → 变为 Dijkstra 算法。
4.  设置 $h(v) = \text{真实距离}$ → 理想搜索。
5.  尝试不可采纳的 $h$ → 更快但可能不是最优的。

#### 测试用例

| 图类型     | 启发函数   | 结果               |
| ---------- | ---------- | ------------------ |
| 网格 (4×4) | Manhattan  | 找到最短路径       |
| 加权图     | Euclidean  | 最优路线           |
| 所有 $h=0$ | None       | 变为 Dijkstra 算法 |

#### 复杂度

-   时间：$O(E \log V)$（取决于启发函数）
-   空间：$O(V)$
-   最优条件：$h$ 可采纳
-   完备条件：有限分支因子

A* 搜索是具有前瞻性的 Dijkstra 算法，其驱动力不仅来自迄今为止的代价，还来自对前方路程的有根据的猜测。
### 326 Floyd–Warshall

Floyd–Warshall（弗洛伊德-沃舍尔）算法是一种动态规划方法，用于计算加权有向图中的所有点对最短路径（APSP）。
它通过逐步允许中间顶点，迭代地优化每对顶点之间的最短路径估计。

只要图中没有负权环，即使存在负权边，该算法也能正常工作。

#### 我们要解决什么问题？

我们想要计算：

$$
\text{dist}(u, v) = \min_{\text{路径 } u \to v} \sum_{(x, y) \in \text{路径}} w(x, y)
$$

对于图 $G = (V, E, w)$ 中的所有点对 $(u, v)$。

我们允许负权边，但不允许负权环。
它在以下情况下特别有用：

- 我们需要所有点对的最短路径。
- 图是稠密的（$E \approx V^2$）。
- 我们想要计算传递闭包或可达性（设置 $w(u, v) = 1$）。

#### 它是如何工作的（通俗解释）？

我们逐步允许每个顶点作为可能的中间途经点。
最初，从 $i$ 到 $j$ 的最短路径就是直接的边。
然后，对于每个顶点 $k$，我们检查经过 $k$ 的路径是否能改善距离。

递推关系：

$$
d_k(i, j) = \min \big( d_{k-1}(i, j),; d_{k-1}(i, k) + d_{k-1}(k, j) \big)
$$

实现时使用原地更新：

$$
\text{dist}[i][j] = \min(\text{dist}[i][j],; \text{dist}[i][k] + \text{dist}[k][j])
$$

三层嵌套循环：

1. `k`（中间点）
2. `i`（源点）
3. `j`（目标点）

| k | i   | j   | 更新操作                               |
| - | --- | --- | -------------------------------------- |
| 0 | 所有 | 所有 | 考虑顶点 0 作为途经点                  |
| 1 | 所有 | 所有 | 考虑顶点 1 作为途经点                  |
| … | …   | …   | …                                      |

经过 $V$ 次迭代后，所有最短路径最终确定。

#### 微型代码（C 语言示例）

```c
#include <stdio.h>
#define INF 1000000000
#define MAX 100

int main(void) {
    int n;
    printf("输入顶点数: ");
    scanf("%d", &n);
    int dist[MAX][MAX];
    printf("输入邻接矩阵 (INF=9999, 0 表示自身):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &dist[i][j]);

    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (dist[i][k] + dist[k][j] < dist[i][j])
                    dist[i][j] = dist[i][k] + dist[k][j];

    printf("所有点对最短距离:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%8d ", dist[i][j]);
        printf("\n");
    }
}
```

Python 版本

```python
def floyd_warshall(graph):
    n = len(graph)
    dist = [row[:] for row in graph]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

INF = float('inf')
graph = [
    [0, 3, INF, 7],
    [8, 0, 2, INF],
    [5, INF, 0, 1],
    [2, INF, INF, 0]
]

res = floyd_warshall(graph)
for row in res:
    print(row)
```

#### 为什么它很重要

- 一次计算即可得到所有点对最短路径
- 能处理负权边
- 如果 $\text{dist}[i][i] < 0$ 则能检测负权环
- 可用于传递闭包、路由表、图压缩

#### 一个温和的证明（为什么它有效）

令 $d_k(i, j)$ 为从 $i$ 到 $j$ 且仅使用中间顶点集合 ${1, 2, \dots, k}$ 中的顶点的最短路径。

基本情况（$k=0$）：
$d_0(i, j) = w(i, j)$，即直接边。

归纳步骤：
对于每个 $k$，
最短路径要么避开 $k$，要么经过 $k$。
因此：

$$
d_k(i, j) = \min \big(d_{k-1}(i, j),; d_{k-1}(i, k) + d_{k-1}(k, j)\big)
$$

通过归纳，经过 $V$ 次迭代后，所有最短路径都被覆盖。

#### 自己动手试试

1.  构建一个 4×4 的权重矩阵。
2.  引入一条负权边（但不形成环）。
3.  检查每次迭代后的结果。
4.  检测环：观察是否有 $\text{dist}[i][i] < 0$。
5.  用它来计算可达性（将 INF 替换为 0/1）。

#### 测试用例

| 图类型         | 边（权重）                               | 备注                                   |
| -------------- | ---------------------------------------- | -------------------------------------- |
| 简单图         | 0→1(3), 0→3(7), 1→2(2), 2→0(5), 3→0(2)   | 计算所有点对路径                       |
| 含负权边       | 0→1(1), 1→2(-2), 2→0(4)                  | 有效的所有点对最短路径                 |
| 含负权环       | 0→1(1), 1→2(-2), 2→0(-1)                 | $\text{dist}[0][0] < 0$                |

#### 复杂度

-   时间：$O(V^3)$
-   空间：$O(V^2)$
-   检测：如果 $\text{dist}[i][i] < 0$ 则存在负权环

Floyd–Warshall 算法是图的完整记忆，每一个距离，每一条路线，都通过对所有可能中间点的仔细迭代计算出来。
### 327 Johnson 算法

Johnson 算法能够高效地计算稀疏加权有向图中的所有点对最短路径（APSP），即使图中存在负权边（但无负权环）。
它巧妙地结合了 Bellman–Ford 算法和 Dijkstra 算法，通过重新赋权来消除负权边，同时保持最短路径关系。

结果：
$$
O(VE + V^2 \log V)
$$
对于稀疏图，这比 Floyd–Warshall 算法（$O(V^3)$）要高效得多。

#### 我们要解决什么问题？

我们需要计算每对顶点之间的最短路径，即使某些边具有负权重。
直接运行 Dijkstra 算法在负权边上会失败，而对每个顶点运行 Bellman–Ford 算法将花费 $O(V^2E)$。

Johnson 的方法通过重新赋权使边变为非负，然后从每个顶点应用 Dijkstra 算法来解决此问题。

给定一个加权有向图 $G = (V, E, w)$：

$$
w(u, v) \in \mathbb{R}, \quad \text{无负权环。}
$$

我们要求：

$$
\text{dist}(u, v) = \min_{\text{路径 } u \to v} \sum w(x, y)
$$

#### 它是如何工作的（通俗解释）？

1.  添加一个新顶点 $s$，用权重为 $0$ 的边将其连接到所有其他顶点。
2.  从 $s$ 运行 Bellman–Ford 算法计算势能值 $h(v)$：
    $$
    h(v) = \text{dist}_s(v)
    $$
3.  重新赋权边：
    $$
    w'(u, v) = w(u, v) + h(u) - h(v)
    $$
    这确保了所有 $w'(u, v) \ge 0$。
4.  移除 $s$。
5.  对于每个顶点 $u$，在重新赋权后的图 $w'$ 上运行 Dijkstra 算法。
6.  恢复原始距离：
    $$
    \text{dist}(u, v) = \text{dist}'(u, v) + h(v) - h(u)
    $$

| 步骤 | 操作               | 结果                     |
| ---- | ------------------ | ------------------------ |
| 1    | 添加源点 $s$       | 连接到所有顶点           |
| 2    | Bellman–Ford       | 计算势能                 |
| 3    | 重新赋权边         | 所有权重变为非负         |
| 4    | 运行 Dijkstra      | $O(VE + V^2 \log V)$     |
| 5    | 恢复距离           | 使用 $h$ 进行调整        |

#### 微型代码（Python 示例）

```python
import heapq

def bellman_ford(V, edges, s):
    dist = [float('inf')] * V
    dist[s] = 0
    for _ in range(V - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            raise ValueError("检测到负权环")
    return dist

def dijkstra(V, adj, src):
    dist = [float('inf')] * V
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist

def johnson(V, edges):
    # 步骤 1: 添加新的源点 s
    s = V
    new_edges = edges + [(s, v, 0) for v in range(V)]
    h = bellman_ford(V + 1, new_edges, s)

    # 步骤 2: 重新赋权边
    adj = [[] for _ in range(V)]
    for u, v, w in edges:
        adj[u].append((v, w + h[u] - h[v]))

    # 步骤 3: 从每个顶点运行 Dijkstra
    dist_matrix = [[float('inf')] * V for _ in range(V)]
    for u in range(V):
        d = dijkstra(V, adj, u)
        for v in range(V):
            dist_matrix[u][v] = d[v] + h[v] - h[u]
    return dist_matrix

edges = [
    (0, 1, 1),
    (1, 2, -2),
    (2, 0, 4),
    (2, 3, 2),
    (3, 1, 7)
]

res = johnson(4, edges)
for row in res:
    print(row)
```

#### 为何重要

-   适用于负权边
-   结合了 Bellman–Ford 的灵活性和 Dijkstra 的速度
-   对于稀疏图，比 Floyd–Warshall 快得多
-   用于路由、依赖图、AI 导航等领域

#### 一个温和的证明（为何有效）

重新赋权保持了最短路径的顺序：

对于任意路径 $P = (v_0, v_1, \dots, v_k)$：

$$
w'(P) = w(P) + h(v_0) - h(v_k)
$$

因此：

$$
w'(u, v) < w'(u, x) \iff w(u, v) < w(u, x)
$$

$w$ 中的所有最短路径在 $w'$ 中仍然是最短的，但现在所有权重都是非负的，从而允许使用 Dijkstra 算法。

最后，距离被恢复：

$$
\text{dist}(u, v) = \text{dist}'(u, v) + h(v) - h(u)
$$

#### 亲自尝试

1.  添加负权边（无环）并与 Floyd–Warshall 算法的结果进行比较。
2.  可视化重新赋权过程：显示 $w'(u, v)$。
3.  在稀疏图和稠密图上进行测试。
4.  引入负权环以触发检测。
5.  使用斐波那契堆替换 Dijkstra 算法以实现 $O(VE + V^2 \log V)$。

#### 测试用例

| 图类型         | 边                           | 结果                 |
| -------------- | ---------------------------- | -------------------- |
| 三角形         | 0→1(1), 1→2(-2), 2→0(4)     | 所有点对距离         |
| 负权边         | 0→1(-1), 1→2(2)             | 正确                 |
| 负权环         | 0→1(-2), 1→0(-3)            | 被检测到             |

#### 复杂度

-   时间：$O(VE + V^2 \log V)$
-   空间：$O(V^2)$
-   适用条件：无负权环

Johnson 算法是最短路径的协调者，它重新调整了旋律，使每个音符都变为非负，从而让 Dijkstra 算法能够以速度和精度在整个图上演奏。
### 328 0–1 BFS

0–1 BFS 算法是一种专门用于处理边权仅为 0 或 1 的图的最短路径技术。
它使用双端队列（deque）而非优先队列，从而允许在线性时间内高效地进行松弛操作。

通过将权重为 0 的边推入队列前端，权重为 1 的边推入队列后端，该算法维持了一个始终正确的前沿，有效地在 $O(V + E)$ 时间内模拟了 Dijkstra 算法的行为。

#### 我们要解决什么问题？

我们希望在边权属于 ${0, 1}$ 的有向或无向图中计算单源最短路径：

$$
w(u, v) \in {0, 1}
$$

我们需要：

$$
\text{dist}[v] = \min_{\text{路径 } s \to v} \sum_{(u, v) \in \text{路径}} w(u, v)
$$

典型应用：

- 具有特殊转换（例如切换、开关）的无权图
- 包含免费与高代价动作的状态空间搜索
- 二进制网格、位掩码问题或最小操作图

#### 它是如何工作的（通俗解释）？

0–1 BFS 用双端队列替代了 Dijkstra 算法中的优先队列，利用了边权仅为 0 或 1 这一事实。

1.  初始化所有顶点 $v$ 的 `dist[v] = ∞`，设置 `dist[s] = 0`。
2.  将源点 $s$ 推入双端队列。
3.  当双端队列不为空时：

    *   从队列前端弹出顶点。
    *   对于每个邻居边 $(u, v)$：

        *   如果 $w(u, v) = 0$ 且能改进距离，则推入队列前端。
        *   如果 $w(u, v) = 1$ 且能改进距离，则推入队列后端。

这确保了顶点总是按照距离的非递减顺序被处理，就像 Dijkstra 算法一样。

| 步骤  | 边类型  | 操作               |
| ----- | ------- | ------------------ |
| $w=0$ | 推入前端 | 立即处理           |
| $w=1$ | 推入后端 | 稍后处理           |

#### 微型代码（C 语言示例）

```c
#include <stdio.h>
#include <stdbool.h>
#include <limits.h>

#define MAX 1000
#define INF 1000000000

typedef struct { int v, w; } Edge;
Edge graph[MAX][MAX];
int deg[MAX];
int dist[MAX];
int deque[MAX * 2], front = MAX, back = MAX;

void push_front(int x) { deque[--front] = x; }
void push_back(int x) { deque[back++] = x; }
int pop_front() { return deque[front++]; }
bool empty() { return front == back; }

int main(void) {
    int V, E, s;
    scanf("%d %d %d", &V, &E, &s);
    for (int i = 0; i < E; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        graph[u][deg[u]].v = v;
        graph[u][deg[u]].w = w;
        deg[u]++;
    }

    for (int i = 0; i < V; i++) dist[i] = INF;
    dist[s] = 0;
    push_front(s);

    while (!empty()) {
        int u = pop_front();
        for (int i = 0; i < deg[u]; i++) {
            int v = graph[u][i].v;
            int w = graph[u][i].w;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                if (w == 0) push_front(v);
                else push_back(v);
            }
        }
    }

    for (int i = 0; i < V; i++)
        printf("%d: %d\n", i, dist[i]);
}
```

Python（基于双端队列）

```python
from collections import deque

def zero_one_bfs(V, edges, src):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))

    dist = [float('inf')] * V
    dist[src] = 0
    dq = deque([src])

    while dq:
        u = dq.popleft()
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if w == 0:
                    dq.appendleft(v)
                else:
                    dq.append(v)
    return dist

edges = [(0,1,0),(1,2,1),(0,2,1),(2,3,0)]
print(zero_one_bfs(4, edges, 0))
```

#### 为什么它很重要

-   对于 0/1 权重的图，具有线性时间复杂度
-   比 Dijkstra 算法更简单，在特殊情况下更快
-   适用于状态转移图（位翻转、BFS + 代价）
-   常见于竞赛编程、人工智能、机器人学

#### 一个温和的证明（为什么它有效）

因为所有边权要么是 0，要么是 1，所以每一步距离最多增加 1。
双端队列确保了节点按照距离的非递减顺序被处理：

-   当松弛一条 $0$ 权重边时，我们将顶点推入队列前端（距离不变）。
-   当松弛一条 $1$ 权重边时，我们将顶点推入队列后端（距离 +1）。

因此，双端队列就像一个单调的优先队列，保证了与 Dijkstra 算法等效的正确性。

#### 亲自尝试

1.  构建一个包含 0 和 1 权重边的小型图。
2.  将输出与 Dijkstra 算法的结果进行比较。
3.  可视化双端队列的操作。
4.  尝试一个网格，其中直线移动代价为 0，转向代价为 1。
5.  在稀疏图上测量其运行时间并与 Dijkstra 算法对比。

#### 测试用例

| 图类型     | 边                                      | 结果           |
| ---------- | --------------------------------------- | -------------- |
| 简单图     | 0→1(0), 1→2(1), 0→2(1), 2→3(0)          | [0, 0, 1, 1]   |
| 零权重边图 | 0→1(0), 1→2(0), 2→3(0)                  | [0, 0, 0, 0]   |
| 混合权重图 | 0→1(1), 0→2(0), 2→3(1)                  | [0, 1, 0, 1]   |

#### 复杂度

-   时间复杂度：$O(V + E)$
-   空间复杂度：$O(V + E)$
-   条件：$w(u, v) \in {0, 1}$

0–1 BFS 是二进制的 Dijkstra 算法，是一个知道何时可以免费冲刺、何时需要耐心排队等待代价的双速旅行者。
### 329 Dial 算法

Dial 算法是 Dijkstra 算法的一种变体，针对边权为非负整数且范围较小、有界的图进行了优化。
它不使用堆，而是使用一个桶数组，每个桶对应一个可能的距离值（模最大边权）。
这带来了 $O(V + E + C)$ 的性能，其中 $C$ 是最大边权。

当边权是较小范围内的整数时，例如 ${0, 1, \dots, C}$，此算法是理想选择。

#### 我们要解决什么问题？

我们希望在满足以下条件的图中求解单源最短路径：

$$
w(u, v) \in {0, 1, 2, \dots, C}, \quad C \text{ 较小}
$$

给定一个带权有向图 $G = (V, E, w)$，目标是计算：

$$
\text{dist}[v] = \min_{\text{路径 } s \to v} \sum_{(u, v) \in \text{路径}} w(u, v)
$$

如果我们知道最大边权 $C$，就可以用一个队列数组来替代堆，从而高效地循环处理距离。

应用场景：
- 成本较小的网络路由
- 权重级别较少的基于网格的移动
- 电信调度
- 交通流问题

#### 它是如何工作的（通俗解释）？

Dial 算法根据顶点当前的暂定距离对它们进行分组。
它不使用优先队列，而是维护桶 `B[0..C]`，每个桶存储距离模 $(C+1)$ 同余的顶点。

1. 初始化 $\text{dist}[v] = \infty$；设置 $\text{dist}[s] = 0$。
2. 将 $s$ 放入 `B[0]`。
3. 对于当前桶索引 `i`，处理 `B[i]` 中的所有顶点：
   * 对于每条边 $(u, v, w)$：
     $$
     \text{如果 } \text{dist}[u] + w < \text{dist}[v], \text{ 则更新 } \text{dist}[v]
     $$
     将 $v$ 放入桶 `B[(i + w) \bmod (C+1)]`。
4. 将 `i` 移动到下一个非空桶（循环移动）。
5. 当所有桶都为空时停止。

| 步骤 | 桶   | 顶点         | 操作         |
| ---- | ---- | ------------ | ------------ |
| 0    | [0]  | 起始顶点     | 松弛边       |
| 1    | [1]  | 下一层       | 传播         |
| …    | …    | …            | …            |

这种基于桶的松弛操作确保我们总是按照距离递增的顺序处理顶点，就像 Dijkstra 算法一样。

#### 微型代码（Python 示例）

```python
from collections import deque

def dial_algorithm(V, edges, src, C):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))

    INF = float('inf')
    dist = [INF] * V
    dist[src] = 0

    buckets = [deque() for _ in range(C + 1)]
    buckets[0].append(src)

    idx = 0
    processed = 0
    while processed < V:
        while not buckets[idx % (C + 1)]:
            idx += 1
        dq = buckets[idx % (C + 1)]
        u = dq.popleft()
        processed += 1

        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                old = dist[v]
                dist[v] = dist[u] + w
                new_idx = dist[v] % (C + 1)
                buckets[new_idx].append(v)
    return dist

edges = [
    (0, 1, 2),
    (0, 2, 1),
    (1, 3, 3),
    (2, 3, 1)
]
print(dial_algorithm(4, edges, 0, 3))
```

输出：

```
[0, 2, 1, 2]
```

#### 为什么它很重要

- 用固定大小的队列数组替代了堆
- 对于小整数权重更快
- 当 $C$ 为常数时具有线性时间复杂度
- 比斐波那契堆更简单，但在实践中通常同样有效

#### 一个温和的证明（为什么它有效）

所有边权都是非负整数，且以 $C$ 为界。
每次松弛最多使距离增加 $C$，因此我们只需要 $C+1$ 个桶来跟踪模 $C+1$ 的可能余数。

因为每个桶是按循环顺序处理的，并且顶点只有在距离减小时才会被重新访问，所以算法保持了距离的非递减顺序，确保了与 Dijkstra 算法等效的正确性。

#### 亲自尝试

1.  在边权为小整数（0–5）的图上运行。
2.  与二叉堆实现的 Dijkstra 算法比较运行时间。
3.  尝试 $C=1$ → 变为 0–1 BFS。
4.  测试 $C=10$ → 桶更多，但仍然很快。
5.  绘制每个桶的松弛操作次数。

#### 测试用例

| 图类型     | 边                               | 最大权重 $C$ | 距离         |
| ---------- | -------------------------------- | ------------ | ------------ |
| 简单图     | 0→1(2), 0→2(1), 2→3(1), 1→3(3)   | 3            | [0, 2, 1, 2] |
| 均匀权重图 | 0→1(1), 1→2(1), 2→3(1)           | 1            | [0, 1, 2, 3] |
| 零权边图   | 0→1(0), 1→2(0)                   | 1            | [0, 0, 0]    |

#### 复杂度

- 时间：$O(V + E + C)$
- 空间：$O(V + C)$
- 条件：所有边权 $\in [0, C]$

Dial 算法是桶式 Dijkstra 算法，它逐层遍历距离，将每个顶点存储在其成本对应的槽位中，永远不需要堆来决定下一个处理谁。
### 330 多源 Dijkstra

多源 Dijkstra 是 Dijkstra 算法的一种变体，旨在寻找加权图中从多个起始顶点到所有其他顶点的最短距离。
我们无需重复运行 Dijkstra，而是将所有源点的距离初始化为 0 并放入优先队列，让算法同时传播最小距离。

当你有多个起点（城市、服务器、入口点）并希望找到从其中任意一个出发的最短路径时，这是一种强大的技术。

#### 我们要解决什么问题？

给定一个加权图 $G = (V, E, w)$ 和一个源点集合 $S = {s_1, s_2, \dots, s_k}$，我们希望计算：

$$
\text{dist}[v] = \min_{s_i \in S} \text{dist}(s_i, v)
$$

换句话说，就是到最近源点的距离。

典型用例：
- 多仓库路径规划（从任意设施出发的最短路线）
- 最近服务中心（医院、服务器、商店）
- 多种子传播（火灾蔓延、类似 BFS 的效果）
- 图中的 Voronoi 分区

#### 它是如何工作的（通俗解释）？

逻辑很简单：
1. 将所有源点放入优先队列，每个源点的距离为 0。
2. 像往常一样执行 Dijkstra 算法。
3. 每当一个顶点被松弛时，最先到达它的源点决定了它的距离。

因为我们按距离递增的顺序处理顶点，所以每个顶点的距离都反映了最近的源点。

| 步骤                               | 操作 |
| ---------------------------------- | ------ |
| 初始化 dist[v] = ∞             |        |
| 对于每个源点 s ∈ S：dist[s] = 0 |        |
| 将所有 s 推入优先队列     |        |
| 运行标准 Dijkstra              |        |

#### 微型代码（Python 示例）

```python
import heapq

def multi_source_dijkstra(V, edges, sources):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))  # 对于无向图

    INF = float('inf')
    dist = [INF] * V
    pq = []

    for s in sources:
        dist[s] = 0
        heapq.heappush(pq, (0, s))

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist

edges = [
    (0, 1, 2),
    (1, 2, 3),
    (0, 3, 1),
    (3, 4, 4),
    (2, 4, 2)
]

sources = [0, 4]
print(multi_source_dijkstra(5, edges, sources))
```

输出：

```
[0, 2, 3, 1, 0]
```

现在每个顶点都知道它到源点 0 或 4 的最短距离。

#### 为什么它很重要？

- 对多个起点高效（无需运行 $k$ 次独立的 Dijkstra）
- 非常适合最近邻标记或多区域 BFS
- 适用于加权图，不像基础的多源 BFS
- 是构建图 Voronoi 图的基础模块

#### 一个温和的证明（为什么它有效）

Dijkstra 算法确保顶点按距离非递减的顺序被处理。
通过将所有源点的距离初始化为 0，我们将它们视为通过 0 权重边连接的一个超级源点：

$$
S^* \to s_i, \quad w(S^*, s_i) = 0
$$

因此，多源 Dijkstra 等价于从一个连接到所有源点的虚拟节点出发的单源 Dijkstra，这保证了正确性。

#### 亲自尝试

1. 在城市地图图中添加多个源点。
2. 观察每个节点最先连接到哪个源点。
3. 与 $k$ 次独立的 Dijkstra 运行结果进行比较。
4. 修改代码以同时存储源点标签（用于 Voronoi 分配）。
5. 在网格上尝试，其中某些单元格是起始火源或信号源。

#### 测试用例

| 图          | 源点 | 结果                     |
| -------------- | ------- | -------------------------- |
| 线图 0–4 | [0, 4]  | [0, 1, 2, 1, 0]            |
| 三角形 0–1–2 | [0, 2]  | [0, 1, 0]                  |
| 网格           | 角落 | 从角落出发的最小步数 |

#### 复杂度

- 时间：$O((V + E) \log V)$
- 空间：$O(V + E)$
- 条件：非负权重

多源 Dijkstra 是最短路径的合唱，所有源点一起歌唱，每个顶点都聆听着最近的声音。

## 第 34 节 最短路径变体
### 331 0–1 BFS

0–1 BFS 算法是一种专门用于边权仅为 0 或 1 的图的最短路径技术。
它是 Dijkstra（迪杰斯特拉）算法的一个精简版本，利用只有两种可能边权这一事实，用双端队列（deque）替代了优先队列。
这使得我们可以在线性时间内计算所有最短路径：

$$
O(V + E)
$$

#### 我们要解决什么问题？

我们想在一个图中找到从源顶点 $s$ 到所有其他顶点的最短路径，其中边的权重满足：

$$
w(u, v) \in {0, 1}
$$

标准的 Dijkstra 算法可以工作，但当边权如此简单时，维护一个堆是不必要的开销。
关键洞察在于：权重为 0 的边不会增加距离，因此应该立即探索；而权重为 1 的边应该稍后探索。

#### 它是如何工作的（通俗解释）？

我们使用一个双端队列来管理顶点，依据它们当前的最短距离。

1.  初始化所有距离为 $\infty$，除了 $\text{dist}[s] = 0$。
2.  将 $s$ 推入双端队列。
3.  当双端队列不为空时：

    *   从队列前端弹出顶点 $u$。
    *   对于每条权重为 $w$ 的边 $(u, v)$：

        *   如果 $\text{dist}[u] + w < \text{dist}[v]$，则更新 $\text{dist}[v]$。
        *   如果 $w = 0$，将 $v$ 推到队列前端（距离不增加）。
        *   如果 $w = 1$，将 $v$ 推到队列后端（距离 +1）。

因为所有边权都是 0 或 1，这无需堆也能保持正确的顺序。

| 权重 | 操作         | 原因               |
| ---- | ------------ | ------------------ |
| 0    | 推入前端     | 立即探索           |
| 1    | 推入后端     | 稍后探索           |

#### 精简代码（Python 示例）

```python
from collections import deque

def zero_one_bfs(V, edges, src):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))
        # 对于无向图，还需要添加 graph[v].append((u, w))

    dist = [float('inf')] * V
    dist[src] = 0
    dq = deque([src])

    while dq:
        u = dq.popleft()
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if w == 0:
                    dq.appendleft(v)
                else:
                    dq.append(v)
    return dist

edges = [(0, 1, 0), (1, 2, 1), (0, 2, 1), (2, 3, 0)]
print(zero_one_bfs(4, edges, 0))
```

输出：

```
[0, 0, 1, 1]
```

#### 为什么它很重要

-   运行时间为 $O(V + E)$，比 Dijkstra 的 $O(E \log V)$ 更快
-   当权重为 0 或 1 时，简化了实现
-   适用于有向图或无向图
-   完美适用于以下问题：
    *   最小翻转/操作次数
    *   二进制网格中的最短路径
    *   具有特殊代价转换的 BFS

#### 一个温和的证明（为什么它有效）

在每一步，双端队列中的顶点都按非递减距离排序。
当一条权重为 0 的边被松弛时，邻居的距离等于 $\text{dist}[u]$，因此我们立即处理它（推入前端）。
当一条权重为 1 的边被松弛时，邻居的距离增加 1，因此它被放到后端。

这保持了与 Dijkstra 算法相同的不变量：

> 每个顶点在其最短距离被最终确定时被处理。

因此，正确性得以保证。

#### 亲自尝试

1.  在具有 0-1 权重的图上与 Dijkstra 算法进行比较。
2.  创建一个网格，其中直线移动代价为 0，转向代价为 1。
3.  修改代码以处理无向边。
4.  将其用于“最少打破墙壁数”问题。
5.  逐步绘制双端队列内容以可视化进程。

#### 测试用例

| 图类型 | 边                                 | 距离结果       |
| ------ | ---------------------------------- | -------------- |
| 简单图 | 0→1(0), 1→2(1), 0→2(1), 2→3(0)     | [0, 0, 1, 1]   |
| 全 0 图 | 0→1(0), 1→2(0), 2→3(0)             | [0, 0, 0, 0]   |
| 混合图 | 0→1(1), 1→2(0), 0→2(1)             | [0, 1, 1]      |

#### 复杂度

-   时间：$O(V + E)$
-   空间：$O(V + E)$
-   条件：$w(u, v) \in {0, 1}$

0–1 BFS 是一种二进制的 Dijkstra 算法，它先处理零代价边，再处理单位代价边，快速、简单且顺序完美。
### 332 双向 Dijkstra

双向 Dijkstra 是经典 Dijkstra 单对最短路径算法的一种优化。
我们不是只从源点开始搜索，而是同时运行两个 Dijkstra 搜索：一个从源点向前搜索，一个从目标点向后搜索，并在它们相遇时停止。

这极大地减少了需要探索的搜索空间，尤其是在稀疏图或类似道路网络的图中。

#### 我们要解决什么问题？

我们希望在具有非负权重的图中，找到两个特定顶点 $s$（源点）和 $t$（目标点）之间的最短路径。
标准的 Dijkstra 算法会探索整个可达图，如果我们只需要 $s \to t$ 的路径，这样做是浪费的。

双向 Dijkstra 从两端开始搜索并在中间相遇：

$$
\text{dist}(s, t) = \min_{v \in V} \left( \text{dist}*\text{fwd}[v] + \text{dist}*\text{bwd}[v] \right)
$$

#### 它是如何工作的（通俗解释）？

该算法维护两个优先队列，一个用于前向搜索（从 $s$ 开始），一个用于后向搜索（从 $t$ 开始）。
每次搜索都像标准 Dijkstra 一样松弛边，但它们交替进行步骤，直到它们的搜索前沿相交。

1.  将所有 $\text{dist}*\text{fwd}$ 和 $\text{dist}*\text{bwd}$ 初始化为 $\infty$。
2.  设置 $\text{dist}*\text{fwd}[s] = 0$，$\text{dist}*\text{bwd}[t] = 0$。
3.  将 $s$ 插入前向堆，$t$ 插入后向堆。
4.  交替扩展一步前向搜索和一步后向搜索。
5.  当一个顶点 $v$ 被两个搜索都访问到时，计算候选路径：
    $$
    \text{dist}(s, v) + \text{dist}(t, v)
    $$
6.  当两个队列都为空，或者当前最小键值超过最佳候选路径时停止。

| 方向     | 堆           | 距离数组           |
| -------- | ------------ | ------------------ |
| 前向     | 从源点出发   | $\text{dist}_\text{fwd}$ |
| 后向     | 从目标点出发 | $\text{dist}_\text{bwd}$ |

#### 微型代码（Python 示例）

```python
import heapq

def bidirectional_dijkstra(V, edges, s, t):
    graph = [[] for _ in range(V)]
    rev_graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))
        rev_graph[v].append((u, w))  # 为后向搜索反向

    INF = float('inf')
    dist_f = [INF] * V
    dist_b = [INF] * V
    visited_f = [False] * V
    visited_b = [False] * V

    dist_f[s] = 0
    dist_b[t] = 0
    pq_f = [(0, s)]
    pq_b = [(0, t)]
    best = INF

    while pq_f or pq_b:
        if pq_f:
            d, u = heapq.heappop(pq_f)
            if d > dist_f[u]:
                continue
            visited_f[u] = True
            if visited_b[u]:
                best = min(best, dist_f[u] + dist_b[u])
            for v, w in graph[u]:
                if dist_f[u] + w < dist_f[v]:
                    dist_f[v] = dist_f[u] + w
                    heapq.heappush(pq_f, (dist_f[v], v))

        if pq_b:
            d, u = heapq.heappop(pq_b)
            if d > dist_b[u]:
                continue
            visited_b[u] = True
            if visited_f[u]:
                best = min(best, dist_f[u] + dist_b[u])
            for v, w in rev_graph[u]:
                if dist_b[u] + w < dist_b[v]:
                    dist_b[v] = dist_b[u] + w
                    heapq.heappush(pq_b, (dist_b[v], v))

        if best < min(pq_f[0][0] if pq_f else INF, pq_b[0][0] if pq_b else INF):
            break

    return best if best != INF else None

edges = [
    (0, 1, 2),
    (1, 2, 3),
    (0, 3, 1),
    (3, 4, 4),
    (4, 2, 2)
]

print(bidirectional_dijkstra(5, edges, 0, 2))
```

输出：

```
5
```

#### 为什么它很重要

-   平均而言，工作量是标准 Dijkstra 的一半
-   最适合稀疏、类似道路的网络
-   非常适合导航、路由、寻路
-   是 ALT 和收缩层次结构等高级方法的基础

#### 一个温和的证明（为什么它有效）

Dijkstra 的不变性：顶点按距离非递减的顺序被处理。
通过运行两个搜索，我们在两个方向上都保持了这个不变性。
当一个顶点被两个搜索都到达时，任何进一步的扩展只能找到比当前最佳路径更长的路径：

$$
\text{dist}*\text{fwd}[u] + \text{dist}*\text{bwd}[u] = \text{候选最短路径}
$$

因此，第一次相遇就产生了最优距离。

#### 自己动手试试

1.  比较探索的节点数与单次 Dijkstra。
2.  可视化在中间相遇的搜索前沿。
3.  添加一个具有均匀权重的网格图。
4.  结合启发式方法 → 双向 A*。
5.  在反向边上使用后向搜索。

#### 测试用例

| 图类型   | 边（权重）                     | 源点 | 目标点 | 结果 |
| -------- | ------------------------------ | ---- | ------ | ---- |
| 线       | 0→1→2→3                        | 0    | 3      | 3    |
| 三角形   | 0→1(2), 1→2(2), 0→2(5)         | 0    | 2      | 4    |
| 道路     | 0→1(1), 1→2(2), 0→3(3), 3→2(1) | 0    | 2      | 3    |

#### 复杂度

-   时间复杂度：$O((V + E) \log V)$
-   空间复杂度：$O(V + E)$
-   条件：非负权重

双向 Dijkstra 是一种在中间相遇的寻路算法，两位探索者从两端出发，相向而行，直到他们共享最短路线。
### 333 使用欧几里得启发式的 A* 算法

A* 算法是一种启发式引导的最短路径搜索算法，它融合了 Dijkstra 算法的严谨性和有方向性的信息指导。
通过引入一个估计剩余距离的启发式函数 $h(v)$，它扩展更少的节点，并将搜索重点导向目标。
当使用欧几里得距离作为启发式时，A* 算法非常适合空间图、网格、地图和道路网络。

#### 我们要解决什么问题？

我们想在一个具有非负权重的加权图中找到从源点 $s$ 到目标点 $t$ 的最短路径，
但同时我们也希望避免探索不必要的区域。

Dijkstra 算法按照真实成本 $g(v)$（当前已走距离）的顺序扩展所有节点。
A* 算法按照估计的总成本顺序扩展节点：

$$
f(v) = g(v) + h(v)
$$

其中

- $g(v)$ = 从 $s$ 到 $v$ 的当前成本，
- $h(v)$ = 从 $v$ 到 $t$ 的启发式估计值。

如果 $h(v)$ 从不高估真实成本，A* 算法就能保证找到最优路径。

#### 它是如何工作的（通俗解释）？

可以把 A* 想象成带指南针的 Dijkstra。
Dijkstra 算法平等地探索所有方向，而 A* 使用 $h(v)$ 使探索偏向目标。

1. 初始化 $\text{dist}[s] = 0$, $\text{f}[s] = h(s)$
2. 将 $(f(s), s)$ 推入优先队列
3. 当队列不为空时：

   * 弹出具有最小 $f(u)$ 的顶点 $u$
   * 如果 $u = t$，停止，路径已找到
   * 对于每个权重为 $w$ 的邻居 $(u, v)$：

     * 计算试探性成本 $g' = \text{dist}[u] + w$
     * 如果 $g' < \text{dist}[v]$：
       $$
       \text{dist}[v] = g', \quad f(v) = g' + h(v)
       $$
       将 $(f(v), v)$ 推入队列

启发式类型：

- 欧几里得：$h(v) = \sqrt{(x_v - x_t)^2 + (y_v - y_t)^2}$
- 曼哈顿：$h(v) = |x_v - x_t| + |y_v - y_t|$
- 零：$h(v) = 0$ → 退化为 Dijkstra 算法

#### 微型代码（Python 示例）

```python
import heapq
import math

def a_star_euclidean(V, edges, coords, s, t):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))  # 无向图

    def h(v):
        x1, y1 = coords[v]
        x2, y2 = coords[t]
        return math.sqrt((x1 - x2)2 + (y1 - y2)2)

    dist = [float('inf')] * V
    dist[s] = 0
    pq = [(h(s), s)]

    while pq:
        f, u = heapq.heappop(pq)
        if u == t:
            return dist[u]
        for v, w in graph[u]:
            g_new = dist[u] + w
            if g_new < dist[v]:
                dist[v] = g_new
                heapq.heappush(pq, (g_new + h(v), v))
    return None

coords = [(0,0), (1,0), (1,1), (2,1)]
edges = [(0,1,1), (1,2,1), (0,2,2), (2,3,1)]
print(a_star_euclidean(4, edges, coords, 0, 3))
```

输出：

```
3
```

#### 为什么它很重要

- 如果 $h(v)$ 是可采纳的（$h(v) \le$ 真实距离），则保证最优
- 如果 $h(v)$ 是一致的（$h(u) \le w(u,v) + h(v)$），则速度很快
- 非常适合空间导航和基于网格的寻路
- 是许多 AI 系统的基础：游戏、机器人、GPS 路由

#### 一个温和的证明（为什么它有效）

A* 通过可采纳性保证正确性：
$$
h(v) \le \text{dist}(v, t)
$$

这意味着 $f(v) = g(v) + h(v)$ 永远不会低估总路径成本，因此当 $t$ 第一次出队时，就找到了最短路径。

一致性确保了 $f(v)$ 值是非递减的，模仿了 Dijkstra 算法的不变性。

因此，A* 在高效引导探索的同时，保留了 Dijkstra 算法的保证。

#### 亲自尝试

1.  与 Dijkstra 算法比较探索的节点数。
2.  在网格上使用欧几里得和曼哈顿启发式。
3.  尝试一个坏的启发式（例如，真实距离的两倍）→ 观察失败情况。
4.  可视化每一步的搜索边界。
5.  将其应用于迷宫或道路地图。

#### 测试用例

| 图类型       | 启发式                     | 结果           |
| ------------ | ------------------------- | ------------- |
| 网格         | 欧几里得                  | 直线路径       |
| 三角形       | $h=0$                     | Dijkstra      |
| 高估         | $h(v) > \text{dist}(v,t)$ | 可能失败       |

#### 复杂度

- 时间：$O(E \log V)$
- 空间：$O(V)$
- 条件：非负权重，可采纳的启发式

使用欧几里得启发式的 A* 算法是导航员的 Dijkstra，在距离的引导下，它知道自己要去哪里，同时找到最短的路线。
### 334 ALT 算法（A* 地标 + 三角不等式）

ALT 算法通过预计算的地标和三角不等式来增强 A* 搜索，为其提供一个强大且可采纳的启发函数，从而显著加速大型道路网络上的最短路径查询。

"ALT" 这个名字来源于 A*（搜索）、Landmarks（地标）和 Triangle inequality（三角不等式），这三者平衡了预处理和查询时的效率。

#### 我们要解决什么问题？

我们希望在大规模加权图（如道路地图）中高效地找到最短路径，在这种情况下，单源搜索（如 Dijkstra 或 A*）可能需要探索数百万个节点。

为了更有效地引导搜索，我们预先计算从特殊节点（地标）出发的距离，并在 A* 搜索过程中利用它们来构建紧密的启发式边界。

给定非负的边权重，我们基于三角不等式定义一个启发函数：

$$
d(a, b) \le d(a, L) + d(L, b)
$$

由此我们推导出 $d(a, b)$ 的一个下界：

$$
h(a) = \max_{L \in \text{landmarks}} |d(L, t) - d(L, a)|
$$

这个 $h(a)$ 是可采纳的（从不高估）且一致的（单调的）。

#### 它是如何工作的？（通俗解释）

ALT 通过预处理和地标距离来增强 A*：

预处理：

1.  选择一小组地标 $L_1, L_2, \dots, L_k$（分散在图中）。
2.  从每个地标运行 Dijkstra（或 BFS）以计算到所有节点的距离：
    $$
    d(L_i, v) \text{ 和 } d(v, L_i)
    $$

查询阶段：

1.  对于查询 $(s, t)$，计算：
    $$
    h(v) = \max_{i=1}^k |d(L_i, t) - d(L_i, v)|
    $$
2.  使用此启发函数运行 A*：
    $$
    f(v) = g(v) + h(v)
    $$
3.  保证获得最优路径（可采纳且一致）。

| 阶段         | 任务                                 | 成本                 |
| ------------ | ------------------------------------ | -------------------- |
| 预处理       | 从地标进行多源 Dijkstra              | $O(k(V+E)\log V)$    |
| 查询         | 使用地标启发函数的 A*                | 快速 ($O(E'\log V)$) |

#### 微型代码（Python 示例）

```python
import heapq

def dijkstra(V, graph, src):
    dist = [float('inf')] * V
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))
    return dist

def alt_search(V, graph, landmarks, d_landmark_to, s, t):
    def h(v):
        # 三角不等式启发函数的最大值
        return max(abs(d_landmark_to[L][t] - d_landmark_to[L][v]) for L in landmarks)

    dist = [float('inf')] * V
    dist[s] = 0
    pq = [(h(s), s)]

    while pq:
        f, u = heapq.heappop(pq)
        if u == t:
            return dist[u]
        for v, w in graph[u]:
            g_new = dist[u] + w
            if g_new < dist[v]:
                dist[v] = g_new
                heapq.heappush(pq, (g_new + h(v), v))
    return None

# 示例用法
V = 5
graph = [
    [(1, 2), (2, 4)],
    [(2, 1), (3, 7)],
    [(3, 3)],
    [(4, 1)],
    []
]

landmarks = [0, 4]
d_landmark_to = [dijkstra(V, graph, L) for L in landmarks]

print(alt_search(V, graph, landmarks, d_landmark_to, 0, 4))
```

输出：

```
11
```

#### 为什么它很重要

-   比 Dijkstra 或普通 A* 扩展的节点少得多
-   可采纳（从不高估）且一致
-   在道路网络、导航系统、地理信息系统和物流路径规划中尤其有效
-   预处理是离线的，查询是实时快速的

#### 一个温和的证明（为什么它有效）

对于任意节点 $a, b$ 和地标 $L$：

$$
|d(L, b) - d(L, a)| \le d(a, b)
$$

通过对所有选定的地标取最大值：

$$
h(a) = \max_L |d(L, t) - d(L, a)| \le d(a, t)
$$

因此 $h(a)$ 是可采纳的，并且由于三角不等式是对称的，它也是一致的：

$$
h(a) \le w(a, b) + h(b)
$$

因此，使用 $h(a)$ 的 A* 保持了最优性。

#### 自己动手试试

1.  选择 2-3 个分散在图中各处的地标。
2.  比较使用和不使用 ALT 启发函数时 A* 的节点扩展情况。
3.  可视化地标周围的启发函数等值线。
4.  在城市地图中使用它来加速路径查询。
5.  尝试随机地标与中心地标的效果。

#### 测试用例

| 图类型 | 地标           | 结果                 |
| ------ | -------------- | -------------------- |
| 链式图 | [首节点, 末节点] | 精确的启发函数       |
| 网格图 | 4 个角落       | 平滑的引导           |
| 随机图 | 随机节点       | 性能表现不一         |

#### 复杂度

-   预处理：$O(k(V + E) \log V)$
-   查询：$O(E' \log V)$（较小的子集）
-   空间：$O(kV)$
-   条件：非负权重

ALT 算法是强化版的 A*，它借助预计算的智慧（地标）引导，利用几何原理而非猜测在图中飞跃。
### 335 收缩层级

收缩层级（CH）是一种用于大型静态道路网络最短路径查询的强大加速技术。它通过添加捷径并按重要性对顶点排序来预处理图，使得查询速度比朴素 Dijkstra 算法快数个数量级。

它是许多现代路由引擎（如 OSRM、GraphHopper、Valhalla）的核心，这些引擎被用于 GPS 系统。

#### 我们要解决什么问题？

我们想要在一个大型、不变的图（如道路地图）上快速回答许多最短路径查询。对每个查询运行 Dijkstra 甚至 A* 算法都太慢了。

收缩层级通过以下方式解决这个问题：

1.  进行一次预处理以创建节点层级结构。
2.  使用更小的搜索（双向搜索）来回答每个查询。

权衡：昂贵的预处理，极快的查询。

#### 它是如何工作的（通俗解释）？

收缩层级是一个两阶段算法：

##### 1. 预处理阶段（构建层级）

我们按重要性（例如，度数、中心性、交通流量）对节点排序。然后我们逐个收缩它们，添加快捷边，以便最短路径距离保持正确。

对于每个要被移除的节点 $v$：

*   对于每对邻居 $(u, w)$：
    *   如果最短路径 $u \to v \to w$ 是 $u$ 和 $w$ 之间唯一的最短路径，则添加一条权重为 $w(u, v) + w(v, w)$ 的捷径 $(u, w)$。

我们记录节点被收缩的顺序。

| 步骤          | 动作           | 结果           |
| ------------- | -------------- | -------------- |
| 选择节点 $v$  | 收缩           | 添加快捷方式   |
| 继续          | 直到所有节点   | 构建层级       |

图变得分层：低重要性节点先被收缩，高重要性节点最后。

##### 2. 查询阶段（向上-向下搜索）

给定查询 $(s, t)$：

*   运行双向 Dijkstra，但只沿着排名更高的节点“向上”搜索。
*   当两个搜索相遇时停止。

这种向上-向下的约束使搜索范围很小，只探索图的极小部分。

| 方向      | 约束条件                     |
| --------- | ---------------------------- |
| 前向      | 只访问排名更高的节点         |
| 后向      | 只访问排名更高的节点         |

最短路径是两个搜索相遇的最低点。

#### 微型代码（Python 示例，概念性）

```python
import heapq

def add_shortcuts(graph, order):
    V = len(graph)
    shortcuts = [[] for _ in range(V)]
    for v in order:
        neighbors = graph[v]
        for u, wu in neighbors:
            for w, ww in neighbors:
                if u == w:
                    continue
                new_dist = wu + ww
                # 如果 u 和 w 之间不存在更短的路径，则添加快捷方式
                exists = any(n == w and cost <= new_dist for n, cost in graph[u])
                if not exists:
                    shortcuts[u].append((w, new_dist))
        # 移除 v（收缩）
        graph[v] = []
    return [graph[i] + shortcuts[i] for i in range(V)]

def upward_edges(graph, order):
    rank = {v: i for i, v in enumerate(order)}
    up = [[] for _ in range(len(graph))]
    for u in range(len(graph)):
        for v, w in graph[u]:
            if rank[v] > rank[u]:
                up[u].append((v, w))
    return up

def ch_query(up_graph, s, t):
    def dijkstra_dir(start):
        dist = {}
        pq = [(0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if u in dist:
                continue
            dist[u] = d
            for v, w in up_graph[u]:
                heapq.heappush(pq, (d + w, v))
        return dist

    dist_s = dijkstra_dir(s)
    dist_t = dijkstra_dir(t)
    best = float('inf')
    for v in dist_s:
        if v in dist_t:
            best = min(best, dist_s[v] + dist_t[v])
    return best

# 示例图
graph = [
    [(1, 2), (2, 4)],
    [(2, 1), (3, 7)],
    [(3, 3)],
    []
]
order = [0, 1, 2, 3]  # 简化版
up_graph = upward_edges(add_shortcuts(graph, order), order)
print(ch_query(up_graph, 0, 3))
```

输出：

```
8
```

#### 为什么它很重要

*   查询速度：微秒级，即使在百万节点的图上
*   用于 GPS 导航、道路路由、物流规划
*   预处理通过捷径保持正确性
*   易于与 ALT、A* 和多级 Dijkstra 结合使用

#### 一个温和的证明（为什么它有效）

当收缩节点 $v$ 时，我们添加快捷方式来保留所有经过 $v$ 的最短路径。因此，移除 $v$ 永远不会破坏最短路径的正确性。

在查询期间：

*   我们只按排名向上移动。
*   由于所有路径都可以表示为向上-向下-向上，前向和后向搜索的相遇点必然位于真正的最短路径上。

通过将探索限制在“向上”的边上，收缩层级在保持完整性的同时使搜索范围很小。

#### 自己动手试试

1.  构建一个小型图，选择一个顺序，并添加快捷方式。
2.  比较 Dijkstra 与收缩层级查询时间。
3.  可视化层级结构（已收缩的与剩余的）。
4.  尝试随机与启发式节点排序。
5.  添加地标（ALT+CH）以进行进一步优化。

#### 测试用例

| 图类型   | 节点      | 查询          | 结果                |
| -------- | --------- | ------------- | ------------------- |
| 链式     | 0–1–2–3   | 0→3           | 3 条边              |
| 三角形   | 0–1–2     | 0→2           | 添加快捷方式 0–2    |
| 网格     | 3×3       | 角→角         | 捷径减少跳数        |

#### 复杂度

*   预处理：$O(V \log V + E)$（使用启发式排序）
*   查询：$O(\log V)$（极小的搜索空间）
*   空间：$O(V + E + \text{shortcuts})$
*   条件：静态图，非负权重

收缩层级是建筑师的 Dijkstra，它首先重塑城市，然后以近乎即时的精度导航。
### 336 CH 查询算法（基于捷径的路由）

CH 查询算法是收缩层次（CH）的在线阶段。
一旦预处理构建了带有捷径的增强层次结构，查询就可以在微秒内通过执行向上的双向 Dijkstra 搜索来回答，该搜索只沿着指向更重要（更高等级）节点的边进行。

它是导航系统中即时路线规划背后的实用魔法。

#### 我们要解决什么问题？

给定一个收缩后的图（带有捷径）和一个节点等级，
我们想要计算两个顶点 $s$ 和 $t$ 之间的最短距离——
而无需探索整个图。

CH 查询不是扫描每个节点，而是：

1. 从 $s$ 和 $t$ 同时向上搜索（遵循等级顺序）。
2. 当两个搜索相遇时停止。
3. 具有最小前向距离 + 后向距离之和的相遇点给出了最短路径。

#### 它是如何工作的（通俗解释）？

在预处理（收缩节点并插入捷径）之后，
查询算法运行两个同时进行的向上 Dijkstra 搜索：

1. 初始化两个优先队列：

   * 从 $s$ 出发的前向搜索
   * 从 $t$ 出发的后向搜索

2. 只松弛向上的边，即从低等级节点到高等级节点的边。
   （如果 $\text{rank}(v) > \text{rank}(u)$，则边 $(u, v)$ 是向上的。）

3. 每当一个节点 $v$ 被两个搜索都确定时，
   计算潜在路径：
   $$
   d = \text{dist}_f[v] + \text{dist}_b[v]
   $$

4. 跟踪最小的 $d$ 值。

5. 当当前最佳 $d$ 小于剩余未访问边界键值时停止。

| 阶段             | 描述                                   |
| ---------------- | -------------------------------------- |
| 前向搜索         | 从 $s$ 出发的向上边                    |
| 后向搜索         | 从 $t$ 出发的向上边（在反向图中）      |
| 中间相遇         | 在交点处合并距离                       |

#### 微型代码（Python 示例，概念性）

```python
import heapq

def ch_query(up_graph, rank, s, t):
    INF = float('inf')
    n = len(up_graph)
    dist_f = [INF] * n
    dist_b = [INF] * n
    visited_f = [False] * n
    visited_b = [False] * n

    dist_f[s] = 0
    dist_b[t] = 0
    pq_f = [(0, s)]
    pq_b = [(0, t)]
    best = INF

    while pq_f or pq_b:
        # 前向方向
        if pq_f:
            d, u = heapq.heappop(pq_f)
            if d > dist_f[u]:
                continue
            visited_f[u] = True
            if visited_b[u]:
                best = min(best, dist_f[u] + dist_b[u])
            for v, w in up_graph[u]:
                if rank[v] > rank[u] and dist_f[u] + w < dist_f[v]:
                    dist_f[v] = dist_f[u] + w
                    heapq.heappush(pq_f, (dist_f[v], v))

        # 后向方向
        if pq_b:
            d, u = heapq.heappop(pq_b)
            if d > dist_b[u]:
                continue
            visited_b[u] = True
            if visited_f[u]:
                best = min(best, dist_f[u] + dist_b[u])
            for v, w in up_graph[u]:
                if rank[v] > rank[u] and dist_b[u] + w < dist_b[v]:
                    dist_b[v] = dist_b[u] + w
                    heapq.heappush(pq_b, (dist_b[v], v))

        # 提前停止条件
        min_frontier = min(
            pq_f[0][0] if pq_f else INF,
            pq_b[0][0] if pq_b else INF
        )
        if min_frontier >= best:
            break

    return best if best != INF else None

# 示例：带有等级的向上图
up_graph = [
    [(1, 2), (2, 4)],  # 0
    [(3, 7)],          # 1
    [(3, 3)],          # 2
    []                 # 3
]
rank = [0, 1, 2, 3]
print(ch_query(up_graph, rank, 0, 3))
```

输出：

```
8
```

#### 为什么它很重要

- 超快速查询，在大图上通常为微秒级
- 无需启发式，100% 精确的最短路径
- 用于实时 GPS 导航、物流优化、地图路由 API
- 可以与 ALT、转弯惩罚或时间相关权重结合使用

#### 一个温和的证明（为什么它有效）

在预处理期间，每个节点 $v$ 都被收缩，并通过捷径保留了所有最短路径。
每个有效的最短路径都可以表示为一个向上-向下路径：

- 等级递增（向上），然后等级递减（向下）。

通过从 $s$ 和 $t$ 同时运行仅向上的搜索，我们探索了所有此类路径的向上部分。
第一个相遇点 $v$ 满足
$$
\text{dist}_f[v] + \text{dist}_b[v]
$$
最小，对应于最优路径。

#### 自己动手试试

1.  可视化收缩前后的等级和边。
2.  比较访问的节点数与完整 Dijkstra 算法。
3.  结合地标（ALT）以获得更快的查询。
4.  测量网格图与道路状图上的查询时间。
5.  通过存储父指针添加路径重建。

#### 测试用例

| 图               | 查询 (s,t)       | 结果    | 访问的节点数    |
| ---------------- | ---------------- | ------- | --------------- |
| 0–1–2–3          | (0,3)            | 3 条边  | 4 (Dijkstra: 4) |
| 0–1–2–3–4 (直线) | (0,4)            | 路径=4  | 5 (Dijkstra: 5) |
| 网格 (5×5)       | (角落, 角落)     | 正确    | ~减少 20 倍     |

#### 复杂度

- 预处理：由 CH 构建处理（$O(V \log V + E)$）
- 查询：$O(\log V)$（极小的边界）
- 空间：$O(V + E + \text{shortcuts})$
- 条件：非负权重，静态图

CH 查询算法是图搜索的快速通道——
预建的捷径和智能的层次结构让它能在极短的时间内从源点飞到目标点。
### 337 Bellman–Ford 队列变体（提前终止 SPFA）

Bellman–Ford 队列变体，通常被称为 SPFA（*最短路径更快算法*），通过使用队列仅松弛活跃顶点（即距离被更新过的顶点）来改进标准的 Bellman–Ford 算法。在实践中，它通常要快得多，尽管在最坏情况下其时间复杂度仍然是 $O(VE)$。

这是一个巧妙的混合体：兼具 Bellman–Ford 的正确性和 Dijkstra 的选择性。

#### 我们要解决什么问题？

我们希望在可能包含负权边（但不含负权环）的加权图中，计算从单个源点 $s$ 出发的最短路径：

$$
w(u, v) \in \mathbb{R}, \quad \text{并且没有负权环}
$$

经典的 Bellman–Ford 算法会松弛所有边 $V-1$ 次 —— 当大多数节点不经常变化时，这太慢了。

SPFA 通过以下方式进行了优化：

- 跟踪上一轮哪些节点被更新了，
- 将它们推入队列，
- 仅松弛它们的出边。

#### 它是如何工作的（通俗解释）？

该算法使用一个“活跃”节点的队列。当一个节点的距离得到改善时，我们将其入队（如果它不在队列中）。每次迭代弹出一个节点，松弛其邻居节点，并将那些距离得到改善的邻居节点入队。

这就像更新的波前，只传播到需要的地方。

| 步骤 | 操作                                                                 |
| ---- | -------------------------------------------------------------------- |
| 1    | 初始化 $\text{dist}[s] = 0$，其他节点为 $\infty$                     |
| 2    | 将 $s$ 推入队列                                                      |
| 3    | 当队列不为空时：                                                     |
|      | 弹出 $u$，松弛边 $(u, v)$                                            |
|      | 如果 $\text{dist}[u] + w(u,v) < \text{dist}[v]$，则更新 $\text{dist}[v]$ 并将 $v$ 入队 |
| 4    | （可选）检测负权环                                                   |

当每个节点离开队列且没有新的更新时，我们就完成了，提前终止！

#### 微型代码（C 语言示例）

```c
#include <stdio.h>
#include <stdbool.h>
#include <limits.h>

#define MAXV 1000
#define INF 1000000000

typedef struct { int v, w; } Edge;
Edge graph[MAXV][MAXV];
int deg[MAXV], dist[MAXV];
bool in_queue[MAXV];
int queue[MAXV*10], front, back;

void spfa(int V, int s) {
    for (int i = 0; i < V; i++) dist[i] = INF, in_queue[i] = false;
    front = back = 0;
    dist[s] = 0;
    queue[back++] = s;
    in_queue[s] = true;

    while (front != back) {
        int u = queue[front++];
        if (front == MAXV*10) front = 0;
        in_queue[u] = false;

        for (int i = 0; i < deg[u]; i++) {
            int v = graph[u][i].v, w = graph[u][i].w;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                if (!in_queue[v]) {
                    queue[back++] = v;
                    if (back == MAXV*10) back = 0;
                    in_queue[v] = true;
                }
            }
        }
    }
}
```

Python（简化版）

```python
from collections import deque

def spfa(V, edges, src):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))

    dist = [float('inf')] * V
    inq = [False] * V
    dist[src] = 0

    dq = deque([src])
    inq[src] = True

    while dq:
        u = dq.popleft()
        inq[u] = False
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if not inq[v]:
                    dq.append(v)
                    inq[v] = True
    return dist
```

#### 为什么它很重要

- 在稀疏图上通常比 Bellman–Ford 算法更快
- 当没有进一步更新时提前终止
- 安全地处理负权边
- 是最小费用最大流（SPFA 版本）的基础
- 在竞赛编程和网络路由中很流行

#### 一个温和的证明（为什么它有效）

Bellman–Ford 依赖于通过 $V-1$ 次迭代传播最短路径的边松弛操作。

SPFA 动态地调度松弛操作：

- 每个顶点仅在其 $\text{dist}$ 值改善时才进入队列。
- 由于每次松弛都遵守边的约束，最短路径在每个顶点最多经过 $V-1$ 次松弛后收敛。

因此，SPFA 保持了正确性，并且可以在达到收敛时提前终止。

最坏情况（例如负权网格图），每个顶点松弛 $O(V)$ 次，所以时间是 $O(VE)$。但在实践中，它接近 $O(E)$。

#### 自己动手试试

1.  在含有负权但无环的图上运行。
2.  与 Bellman–Ford 比较步骤，松弛次数更少。
3.  为每个节点添加一个计数器以检测负权环。
4.  将其用作最小费用最大流的核心算法。
5.  测试稠密图与稀疏图，观察运行时间差异。

#### 测试用例

| 图结构                           | 边状态       | 结果          |
| -------------------------------- | ------------ | ------------- |
| 0→1(2), 1→2(-1), 0→2(4) | 无负权环 | [0, 2, 1] |
| 0→1(1), 1→2(2), 2→0(-4) | 有负权环    | 检测到负环     |
| 0→1(5), 0→2(2), 2→1(-3) | 无环        | [0, -1, 2] |

#### 复杂度

- 时间：最坏 $O(VE)$，平均通常 $O(E)$
- 空间：$O(V + E)$
- 条件：允许负权边，不允许负权环

Bellman–Ford 队列变体是一个聪明的调度器 —— 它不是盲目地循环，而是监听更新，只在发生变化的地方采取行动。
### 338 带提前终止的 Dijkstra 算法

带提前终止的 Dijkstra 算法是经典 Dijkstra 算法的一种目标感知优化。
它利用了 Dijkstra 算法按距离非递减顺序处理顶点这一事实，因此一旦目标节点从优先队列中取出，其最短距离就已最终确定，搜索可以安全终止。

对于单对最短路径查询，这个简单的调整可以显著减少运行时间。

#### 我们要解决什么问题？

在标准的 Dijkstra 算法中，即使我们只关心一个目的地 $t$，搜索也会持续进行，直到所有可达顶点都被确定（settled）为止。
对于点对点路由，我们只需要 $\text{dist}(s, t)$，这样做是浪费的。

提前终止版本在 $t$ 从优先队列中取出时立即停止：

$$
\text{dist}(t) = \text{最终的最短距离}
$$

#### 它是如何工作的（通俗解释）？

与 Dijkstra 算法设置相同，但增加了一个退出条件：

1.  初始化 $\text{dist}[s] = 0$，其他所有顶点为 $\infty$。
2.  将 $(0, s)$ 推入优先队列。
3.  当队列不为空时：

    *   弹出具有最小暂定距离的 $(d, u)$。
    *   如果 $u = t$：停止，我们找到了最短路径。
    *   对于每个邻居 $(v, w)$：

        *   如果 $\text{dist}[u] + w < \text{dist}[v]$，则松弛并推入。

因为 Dijkstra 算法的不变性保证了我们总是按距离递增的顺序弹出节点，
所以第一次弹出 $t$ 时，我们就找到了它的真实最短距离。

| 步骤                 | 操作           |
| -------------------- | -------------- |
| 提取最小节点 $u$     | 扩展邻居       |
| 如果 $u = t$         | 立即停止       |
| 否则                 | 继续           |

无需探索整个图。

#### 精简代码（Python 示例）

```python
import heapq

def dijkstra_early_stop(V, edges, s, t):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))  # 无向图

    INF = float('inf')
    dist = [INF] * V
    dist[s] = 0
    pq = [(0, s)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == t:
            return dist[t]
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return None  # 不可达
```

示例：

```python
edges = [
    (0, 1, 2),
    (1, 2, 3),
    (0, 3, 1),
    (3, 4, 4),
    (4, 2, 2)
]
print(dijkstra_early_stop(5, edges, 0, 2))
```

输出：

```
5
```

#### 为什么它很重要

-   提前终止 = 更少的扩展 = 更快的查询
-   非常适合点对点路由
-   与 A*（引导式提前终止）和 ALT 算法结合良好
-   易于实现（只需一个 `if` 条件）

用于：

-   GPS 导航（城市到城市路线）
-   网络路由（特定端点）
-   游戏 AI 寻路

#### 一个温和的证明（为什么它有效）

Dijkstra 算法的正确性依赖于一个事实：当一个节点从堆中取出时，其最短距离就是最终的（之后不可能出现更小的距离）。

因此，当 $t$ 被取出时：
$$
\text{dist}(t) = \min_{u \in V} \text{dist}(u)
$$
并且任何其他未访问的顶点都有 $\text{dist}[v] \ge \text{dist}[t]$。

所以，当 $u = t$ 时立即停止能保持正确性。

#### 自己动手试试

1.  在大图上比较提前终止版本与完整 Dijkstra 算法的运行时间。
2.  可视化带和不带提前终止的堆操作。
3.  应用于网格图（起点与目标点在对角）。
4.  与 A* 算法结合，进一步减少访问的节点数。
5.  在不连通的图（目标不可达）上测试。

#### 测试用例

| 图               | 源点          | 目标           | 结果             | 备注           |
| ---------------- | ------------- | -------------- | ---------------- | -------------- |
| 链 0–1–2–3       | 0             | 3              | 3 条边           | 在 3 处停止    |
| 星型（0–其他）   | 0             | 4              | 直接边           | 1 步           |
| 网格             | (0,0)→(n,n)   | 提前终止节省    | 跳过许多节点     |                |

#### 复杂度

-   时间：$O(E \log V)$（最好情况 ≪ 完整图）
-   空间：$O(V + E)$
-   条件：非负权重

带提前终止的 Dijkstra 算法是狙击手版本 —— 它锁定目标，任务完成的那一刻就停止，节省了每一个多余的动作。
### 339 目标导向搜索

目标导向搜索是一种将图探索聚焦于特定目标，而非扫描整个空间的通用策略。它通过利用几何、地标或预计算的启发式信息，在目标方向上偏置扩展，从而修改最短路径算法（如 BFS 或 Dijkstra）。

当偏置是可采纳的（从不高估代价）时，它仍然能保证找到最优路径，同时大大减少探索的节点数。

#### 我们要解决什么问题？

在标准的 BFS 或 Dijkstra 中，探索是均匀向外辐射的，即使是在明显不会通向目标的方向上。对于大型图（网格、道路地图），这是浪费的。

我们需要一种方法，在不失正确性的前提下，引导搜索朝向目的地。

形式化地说，对于源点 $s$ 和目标点 $t$，我们希望找到

$$
\text{dist}(s, t) = \min_{\text{路径 } P: s \to t} \sum_{(u,v)\in P} w(u,v)
$$

同时尽可能少地访问节点。

#### 它是如何工作的（通俗解释）？

目标导向搜索为队列中每个节点的优先级附加一个启发式偏置，这样更接近（或更有希望到达）目标的节点会更早被扩展。

典型的评分规则：

$$
f(v) = g(v) + \lambda \cdot h(v)
$$

其中

- $g(v)$ = 从 $s$ 到当前节点的距离，
- $h(v)$ = 从 $v$ 到 $t$ 的启发式估计距离，
- $\lambda$ = 偏置因子（对于 A* 常为 $1$，对于部分引导则 $<1$）。

可采纳的启发式函数（$h(v) \le \text{真实距离}(v,t)$）能保持最优性。即使是简单的方向性启发式，如欧几里得距离或曼哈顿距离，也能显著减少搜索空间。

| 启发式类型 | 定义                                 | 适用场景         |   |               |   |             |
| -------------- | ------------------------------------ | ---------------- | - | ------------- | - | ----------- |
| 欧几里得距离   | $\sqrt{(x_v-x_t)^2+(y_v-y_t)^2}$     | 几何图           |   |               |   |             |
| 曼哈顿距离     | $                                | x_v-x_t          | + | y_v-y_t       | $ | 网格世界     |
| 地标法 (ALT)   | $                                | d(L,t)-d(L,v)    | $ | 道路网络       |   |             |

#### 微型代码（Python 示例）

```python
import heapq
import math

def goal_directed_search(V, edges, coords, s, t):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))  # 无向图

    def heuristic(v):
        x1, y1 = coords[v]
        x2, y2 = coords[t]
        return math.sqrt((x1 - x2)2 + (y1 - y2)2)

    dist = [float('inf')] * V
    dist[s] = 0
    pq = [(heuristic(s), s)]

    while pq:
        f, u = heapq.heappop(pq)
        if u == t:
            return dist[u]
        for v, w in graph[u]:
            g_new = dist[u] + w
            if g_new < dist[v]:
                dist[v] = g_new
                f_v = g_new + heuristic(v)
                heapq.heappush(pq, (f_v, v))
    return None
```

示例：

```python
coords = [(0,0), (1,0), (1,1), (2,1)]
edges = [(0,1,1), (1,2,1), (0,2,2), (2,3,1)]
print(goal_directed_search(4, edges, coords, 0, 3))
```

输出：

```
3
```

#### 为什么它很重要

- 比无引导的 Dijkstra 算法扩展的节点更少
- 自然地与 A*、ALT、CH、地标启发式等方法结合
- 适用于导航、寻路、规划
- 易于适应网格、3D 空间或加权网络

应用于：

- GPS 导航
- AI 游戏智能体
- 机器人学（运动规划）

#### 一个温和的证明（为什么它有效）

当 $h(v)$ 可采纳时，
$$
h(v) \le \text{真实距离}(v,t),
$$
那么 $f(v) = g(v) + h(v)$ 永远不会低估通过 $v$ 的最优路径的代价。

因此，当目标 $t$ 第一次从队列中弹出时，$\text{dist}(t)$ 保证是真实的最短距离。

如果 $h$ 还是一致的，那么
$$
h(u) \le w(u,v) + h(v),
$$
这确保了优先级顺序的行为类似于 Dijkstra 算法，保持了单调性。

#### 亲自尝试

1.  设置 $h=0$ 运行 → 变为普通的 Dijkstra。
2.  尝试不同的 $\lambda$：
    *   $\lambda=1$ → A*
    *   $\lambda<1$ → 更温和的引导（探索更多）。
3.  在网格上与普通 Dijkstra 比较扩展的节点数。
4.  可视化边界增长，目标导向搜索会形成一个锥形而不是圆形。
5.  测试不可采纳的 $h$（可能加速但失去最优性）。

#### 测试用例

| 图             | 启发式     | 结果     | 访问节点数     |
| -------------- | ---------- | -------- | -------------- |
| 4 节点线       | 欧几里得距离 | 3        | 更少           |
| 5×5 网格       | 曼哈顿距离 | 最优     | ~½ 节点        |
| 零启发式       | $h=0$      | Dijkstra | 所有节点       |

#### 复杂度

- 时间：$O(E \log V)$（实践中松弛操作更少）
- 空间：$O(V)$
- 条件：非负权重，可采纳的 $h(v)$

目标导向搜索是带有指南针的 Dijkstra 算法——它仍然保证找到最短路线，但会自信地朝着目标前进，而不是漫无目的地探索每个方向。
### 340 Yen 的 K 条最短路径

Yen 算法不仅能找到单一的最短路径，还能在加权有向图中找到两个节点之间的 K 条无环最短路径。
它是 Dijkstra 算法的自然延伸，它不会在找到第一个解时停止，而是系统地探索路径偏差，以总长度升序的方式发现次优路线。

广泛应用于网络路由、多路线规划和导航系统的备选方案。

#### 我们要解决什么问题？

给定一个具有非负边权重的图 $G = (V, E)$、一个源节点 $s$ 和一个目标节点 $t$，
我们想要计算前 $K$ 条不同的最短路径：

$$
P_1, P_2, \dots, P_K
$$

按总权重排序：

$$
\text{len}(P_1) \le \text{len}(P_2) \le \cdots \le \text{len}(P_K)
$$

每条路径必须是简单的（无重复节点）。

#### 它是如何工作的（通俗解释）？

Yen 算法建立在 Dijkstra 算法和偏差路径概念之上。

1.  使用 Dijkstra 算法计算最短路径 $P_1$。
2.  对于每个 $i = 2, \dots, K$：

    *   令 $P_{i-1}$ 为前一条路径。
    *   对于 $P_{i-1}$ 中的每个节点（激励节点）：

        *   将路径拆分为根路径（到激励节点的前缀）。
        *   临时移除：

            *   任何会重建先前找到的路径的边。
            *   根路径中的任何节点（激励节点除外）以防止循环。
        *   从激励节点到 $t$ 运行 Dijkstra 算法。
        *   组合根路径 + 激励路径 → 候选路径。
    *   在所有候选路径中，选择尚未被选中的最短路径。
    *   将其添加为 $P_i$。

重复直到找到 $K$ 条路径或没有候选路径剩余。

| 步骤 | 操作                     | 目的               |
| ---- | ------------------------ | ------------------ |
| 1    | $P_1$ = Dijkstra(s, t)   | 基础路径           |
| 2    | 从前缀进行偏差           | 探索备选方案       |
| 3    | 收集候选路径             | 最小堆             |
| 4    | 选择最短路径             | 下一条路径         |

#### 微型代码（Python 示例）

适用于小图的简化版本：

```python
import heapq

def dijkstra(graph, s, t):
    pq = [(0, s, [s])]
    seen = set()
    while pq:
        d, u, path = heapq.heappop(pq)
        if u == t:
            return (d, path)
        if u in seen: 
            continue
        seen.add(u)
        for v, w in graph[u]:
            if v not in seen:
                heapq.heappush(pq, (d + w, v, path + [v]))
    return None

def yen_k_shortest_paths(graph, s, t, K):
    A = []
    B = []
    first = dijkstra(graph, s, t)
    if not first:
        return A
    A.append(first)
    for k in range(1, K):
        prev_path = A[k - 1][1]
        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[:i + 1]
            removed_edges = []
            # 临时移除边
            for d, p in A:
                if p[:i + 1] == root_path and i + 1 < len(p):
                    u = p[i]
                    v = p[i + 1]
                    for e in graph[u]:
                        if e[0] == v:
                            graph[u].remove(e)
                            removed_edges.append((u, e))
                            break
            spur = dijkstra(graph, spur_node, t)
            if spur:
                total_path = root_path[:-1] + spur[1]
                total_cost = sum(graph[u][v][1] for u, v in zip(total_path, total_path[1:])) if False else spur[0] + sum(0 for _ in root_path)
                if (total_cost, total_path) not in B:
                    B.append((total_cost, total_path))
            for u, e in removed_edges:
                graph[u].append(e)
        if not B:
            break
        B.sort(key=lambda x: x[0])
        A.append(B.pop(0))
    return A
```

示例：

```python
graph = {
    0: [(1, 1), (2, 2)],
    1: [(2, 1), (3, 3)],
    2: [(3, 1)],
    3: []
}
print(yen_k_shortest_paths(graph, 0, 3, 3))
```

输出：

```
[(3, [0, 1, 2, 3]), (4, [0, 2, 3]), (5, [0, 1, 3])]
```

#### 为什么它很重要

-   提供多条不同的路线，而不仅仅是一条
-   用于多路径路由、备份规划、物流优化
-   保证简单路径（无循环）
-   重用 Dijkstra 算法，易于与现有求解器集成

#### 一个温和的证明（为什么它有效）

每个 $P_i$ 都是通过在某个激励节点处偏离较早的路径而生成的，这确保了唯一性。
Dijkstra 算法确保每条激励路径都是局部最短的。
由于所有候选路径都按全局顺序存储和选择，因此序列 $P_1, P_2, \dots, P_K$ 是按总成本全局排序的。

因此：
$$
\text{len}(P_1) \le \text{len}(P_2) \le \cdots \le \text{len}(P_K)
$$

并且每条路径都是简单的。

#### 自己动手试试

1.  在一个小网络中生成 3 条最短路径。
2.  与 Eppstein 算法进行比较（对于大的 K 值更快）。
3.  动态移除边并重新运行，观察重新排序。
4.  可视化路径树（根 = $s$，分支 = 偏差）。
5.  适用于无向图或双向图。

#### 测试用例

| 图            | $s$        | $t$ | $K$              | 路径                             |
| ------------- | ---------- | --- | ---------------- | -------------------------------- |
| 线 0–1–2–3    | 0          | 3   | 2                | 一条主路径，一条偏差路径         |
| 三角形        | 0          | 2   | 3                | 0–1–2, 0–2, 0–1–0–2（如果允许）  |
| 加权网格      | 起点–终点  | 3   | 备选路线         |                                  |

#### 复杂度

-   时间：$O(K \cdot V \cdot (E + V \log V))$
-   空间：$O(KV)$
-   条件：非负权重，有限路径

Yen 算法是探索者的 Dijkstra，它不会在找到第一条路时停止，而是一次一个偏差地不断发现更好的绕行路线。

## 第 35 节 最小生成树
### 341 Kruskal（克鲁斯卡尔）算法

Kruskal 算法通过反复添加不会形成环的最轻边来构建最小生成树（MST）。它使用并查集（DSU）数据结构来高效地跟踪连通分量。

#### 我们要解决什么问题？

给定一个连通的、无向的、带权图 $G=(V,E,w)$，找到一个子集 $T\subseteq E$，使得：

- $T$ 连接所有顶点，
- $|T|=|V|-1$，
- $\sum_{e\in T}w(e)$ 最小化。

如果图不连通，Kruskal 算法会构建一个最小生成森林。

#### 它是如何工作的（通俗解释）？

1.  将所有边按权重非递减排序。
2.  初始化 DSU，使每个顶点都在自己的集合中。
3.  按顺序扫描边。对于边 $(u,v)$：
   *   如果 $u$ 和 $v$ 在不同的集合中，则将该边加入 MST 并合并它们的集合。
   *   否则，跳过该边以避免形成环。
4.  当拥有 $|V|-1$ 条边时停止。

| 步骤          | 动作               | DSU 效果               |
| ------------- | ------------------ | ---------------------- |
| 排序边        | 从轻到重           | 无                     |
| 检查 $(u,v)$  | 如果 find(u) ≠ find(v) | union(u,v) 并保留该边 |
| 形成环的情况  | 如果 find(u) = find(v) | 跳过该边               |

#### 精简代码

C 语言 (DSU + Kruskal)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct { int u, v; int w; } Edge;

typedef struct {
    int *p, *r;
    int n;
} DSU;

DSU make_dsu(int n){
    DSU d; d.n = n;
    d.p = malloc(n * sizeof(int));
    d.r = malloc(n * sizeof(int));
    for(int i=0;i<n;i++){ d.p[i]=i; d.r[i]=0; }
    return d;
}
int find(DSU *d, int x){
    if(d->p[x]!=x) d->p[x]=find(d, d->p[x]);
    return d->p[x];
}
void unite(DSU *d, int a, int b){
    a=find(d,a); b=find(d,b);
    if(a==b) return;
    if(d->r[a]<d->r[b]) d->p[a]=b;
    else if(d->r[b]<d->r[a]) d->p[b]=a;
    else { d->p[b]=a; d->r[a]++; }
}
int cmp_edge(const void* A, const void* B){
    Edge *a=(Edge*)A, *b=(Edge*)B;
    return a->w - b->w;
}

int main(void){
    int n, m;
    scanf("%d %d", &n, &m);
    Edge *E = malloc(m * sizeof(Edge));
    for(int i=0;i<m;i++) scanf("%d %d %d", &E[i].u, &E[i].v, &E[i].w);

    qsort(E, m, sizeof(Edge), cmp_edge);
    DSU d = make_dsu(n);

    int taken = 0;
    long long cost = 0;
    for(int i=0;i<m && taken < n-1;i++){
        int a = find(&d, E[i].u), b = find(&d, E[i].v);
        if(a != b){
            unite(&d, a, b);
            cost += E[i].w;
            taken++;
        }
    }
    if(taken != n-1) { printf("图不连通\n"); }
    else { printf("MST 总权重: %lld\n", cost); }
    return 0;
}
```

Python (DSU + Kruskal)

```python
class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b: return False
        if self.r[a] < self.r[b]: self.p[a] = b
        elif self.r[b] < self.r[a]: self.p[b] = a
        else: self.p[b] = a; self.r[a] += 1
        return True

def kruskal(n, edges):
    edges = sorted(edges)
    dsu = DSU(n)
    cost = 0
    mst = []
    for w, u, v in edges:
        if dsu.union(u, v):
            cost += w
            mst.append((u, v, w))
            if len(mst) == n-1:
                break
    return cost, mst

n = 4
edges = [(1,0,1),(4,0,2),(3,1,2),(2,1,3),(5,2,3)]
print(kruskal(n, edges))
```

#### 为什么它很重要

-   结合排序和 DSU，简单且快速。
-   在稀疏图上表现良好。
-   生成的总边权重最小的 MST。
-   易于适应于不连通图的最小生成森林。

#### 一个温和的证明（为什么它有效）

Kruskal 依赖于割性质：
令 $S \subset V$ 为任意真子集，考虑割 $(S, V \setminus S)$。
跨越该割的最轻边可以安全地包含在某个 MST 中。

按权重对边进行排序，并总是选取连接两个不同分量的下一个最轻边，这等价于对当前 DSU 分量定义的划分反复应用割性质。
这永远不会产生环，也永远不会排除最优 MST 的可能性。
通过对所选边数进行归纳，最终集合就是一个 MST。

#### 亲自尝试

1.  生成随机稀疏图，并比较 Kruskal 和 Prim 算法。
2.  从 MST 中移除一条边并重新计算，观察变化。
3.  强制边权重相等，并确认存在多个有效的 MST。
4.  在不连通图上运行以获得最小生成森林。
5.  记录选中的边以可视化分量的增长过程。

#### 测试用例

| 图类型       | 边 $(u,v,w)$                               | MST 边           | 总权重     |
| ------------ | ----------------------------------------- | ---------------- | ---------- |
| 三角形       | (0,1,1), (1,2,2), (0,2,3)                 | (0,1,1), (1,2,2) | 3          |
| 正方形       | (0,1,1), (1,2,1), (2,3,1), (3,0,1), (0,2,2) | 任意 3 条权重为 1 的边 | 3          |
| 不连通图     | 两个独立的三角形                          | 每个分量一个 MST | 两个分量之和 |

#### 复杂度

-   时间复杂度：排序 $O(E \log E)$ 加上几乎线性的 DSU 操作 $O(E \alpha(V))$。通常写作 $O(E \log E)$ 或 $O(E \log V)$。
-   空间复杂度：$O(V + E)$。
-   输出大小：对于连通图，为 $|V|-1$ 条边。

Kruskal 是解决 MST 问题的"先排序，后缝合"方法。全局排序边，然后使用 DSU 在局部缝合分量，直到树最终成形。
### 342 Prim 算法（堆实现）

Prim 算法通过每次向连通子树中添加一个顶点来构建最小生成树（MST），它总是选择连接树内顶点与树外顶点的权重最小的边。它是一种贪心算法，通常使用最小堆来实现以提高效率。

#### 我们要解决什么问题？

给定一个连通、无向、带权重的图 $G=(V,E,w)$，找到一个边的子集 $T\subseteq E$，使得：

- $T$ 连接所有顶点，
- $|T|=|V|-1$，
- $\sum_{e\in T}w(e)$ 最小。

Prim 算法从一个单一的起始顶点开始构建 MST。

#### 它是如何工作的（通俗解释）？

1.  选择任意一个起始顶点 $s$。
2.  初始化所有顶点，令 $\text{key}[v]=\infty$，除了 $\text{key}[s]=0$。
3.  使用一个以边权重为键的最小堆（优先队列）。
4.  重复提取具有最小键值（边权重）的顶点 $u$：
    *   将 $u$ 标记为 MST 的一部分。
    *   对于 $u$ 的每个邻居 $v$：
        *   如果 $v$ 尚未在 MST 中且 $w(u,v)<\text{key}[v]$，则更新 $\text{key}[v]=w(u,v)$ 并设置 $\text{parent}[v]=u$。
5.  继续直到所有顶点都被包含进来。

| 步骤             | 动作               | 描述           |
| ---------------- | ------------------ | -------------- |
| 初始化           | 选择起始顶点       | $\text{key}[s]=0$ |
| 提取最小值       | 添加权重最小的边   | 扩展 MST       |
| 松弛邻居         | 更新更便宜的边     | 维护边界       |

#### 微型代码

Python（最小堆版本）

```python
import heapq

def prim_mst(V, edges, start=0):
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((w, v))
        graph[v].append((w, u))  # 无向图

    visited = [False] * V
    pq = [(0, start, -1)]  # (权重, 顶点, 父节点)
    mst_edges = []
    total_cost = 0

    while pq:
        w, u, parent = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        total_cost += w
        if parent != -1:
            mst_edges.append((parent, u, w))
        for weight, v in graph[u]:
            if not visited[v]:
                heapq.heappush(pq, (weight, v, u))
    return total_cost, mst_edges

edges = [(0,1,2),(0,2,3),(1,2,1),(1,3,4),(2,3,5)]
cost, mst = prim_mst(4, edges)
print(cost, mst)
```

输出：

```
7 [(0,1,2), (1,2,1), (1,3,4)]
```

#### 为什么它重要？

-   适用于稠密图（尤其是使用邻接表和堆时）。
-   增量式构建 MST（类似于 Dijkstra 算法）。
-   非常适合需要保持树连通的在线构建场景。
-   对于小图，更容易与邻接矩阵集成。

#### 一个温和的证明（为什么它有效）

Prim 算法遵循割性质：
在每一步，考虑当前 MST 集合 $S$ 与剩余顶点 $V \setminus S$ 之间的割。
跨越该割的最轻边总是可以安全地包含进来。
通过反复选择这样的最小边，Prim 算法维护了一个有效的 MST 前缀。
当所有顶点都被添加后，生成的树就是最小生成树。

#### 亲自尝试

1.  在稠密图上运行 Prim 算法与 Kruskal 算法，比较边的选择。
2.  可视化增长的边界。
3.  尝试使用邻接矩阵（不使用堆）。
4.  在不连通图上测试，每个连通分量会形成自己的树。
5.  将堆替换为简单的数组，以查看 $O(V^2)$ 版本。

#### 测试用例

| 图       | 边 $(u,v,w)$                    | MST 边                 | 总权重 |
| -------- | ------------------------------- | ---------------------- | ------ |
| 三角形   | (0,1,1), (1,2,2), (0,2,3)       | (0,1,1), (1,2,2)       | 3      |
| 正方形   | (0,1,2), (1,2,3), (2,3,1), (3,0,4) | (2,3,1), (0,1,2), (1,2,3) | 6      |
| 线       | (0,1,5), (1,2,1), (2,3,2)       | (1,2,1), (2,3,2), (0,1,5) | 8      |

#### 复杂度

-   时间复杂度：使用堆时为 $O(E\log V)$，使用数组时为 $O(V^2)$。
-   空间复杂度：$O(V + E)$
-   输出大小：$|V|-1$ 条边

Prim 算法是构建 MST 的“从种子生长”方法。它逐步构建树，总是从边界拉取下一个最轻的边。
### 343 Prim 算法（邻接矩阵）

这是基于数组的 Prim 算法版本，针对稠密图进行了优化。
它不使用堆，而是在每一步直接扫描所有顶点以找到下一个具有最小键值的顶点。
其逻辑与 Prim 的堆版本相同，但用简单循环替代了优先队列。

#### 我们要解决什么问题？

给定一个由邻接矩阵 $W$ 表示的连通、无向、带权图 $G=(V,E,w)$，我们希望构建一个最小生成树（MST），即满足以下条件的边集子集：

- 连接所有顶点，
- 包含 $|V|-1$ 条边，
- 并且最小化总权重 $\sum w(e)$。

邻接矩阵形式简化了边的查找，并且非常适合稠密图，其中 $E\approx V^2$。

#### 它是如何工作的（通俗解释）？

1.  从一个任意顶点（例如 $0$）开始。
2.  初始化所有顶点的 $\text{key}[v]=\infty$，除了 $\text{key}[0]=0$。
3.  维护一个集合 $\text{inMST}[v]$ 来标记已包含在 MST 中的顶点。
4.  重复 $V-1$ 次：

    *   选择尚未在 MST 中且 $\text{key}[u]$ 最小的顶点 $u$。
    *   将 $u$ 加入 MST。
    *   对于每个顶点 $v$，如果 $W[u][v]$ 小于 $\text{key}[v]$，则更新 $\text{key}[v]$ 并记录父节点 $v\gets u$。

在每次迭代中，一个顶点加入 MST，该顶点是通过最轻的边连接到现有集合的顶点。

| 步骤       | 操作             | 描述                     |
| ---------- | ---------------- | ------------------------ |
| 初始化     | $\text{key}[0]=0$ | 从顶点 0 开始            |
| 提取最小值 | 找到最小键值     | 下一个要添加的顶点       |
| 更新       | 松弛边           | 更新邻居顶点的键值       |

#### 精简代码

C 语言（邻接矩阵版本）

```c
#include <stdio.h>
#include <limits.h>
#include <stdbool.h>

#define INF 1000000000
#define N 100

int minKey(int key[], bool inMST[], int n) {
    int min = INF, idx = -1;
    for (int v = 0; v < n; v++)
        if (!inMST[v] && key[v] < min)
            min = key[v], idx = v;
    return idx;
}

void primMatrix(int graph[N][N], int n) {
    int key[N], parent[N];
    bool inMST[N];
    for (int i = 0; i < n; i++)
        key[i] = INF, inMST[i] = false;
    key[0] = 0, parent[0] = -1;

    for (int count = 0; count < n - 1; count++) {
        int u = minKey(key, inMST, n);
        inMST[u] = true;
        for (int v = 0; v < n; v++)
            if (graph[u][v] && !inMST[v] && graph[u][v] < key[v]) {
                parent[v] = u;
                key[v] = graph[u][v];
            }
    }
    int total = 0;
    for (int i = 1; i < n; i++) {
        printf("%d - %d: %d\n", parent[i], i, graph[i][parent[i]]);
        total += graph[i][parent[i]];
    }
    printf("MST cost: %d\n", total);
}

int main() {
    int n = 5;
    int graph[N][N] = {
        {0,2,0,6,0},
        {2,0,3,8,5},
        {0,3,0,0,7},
        {6,8,0,0,9},
        {0,5,7,9,0}
    };
    primMatrix(graph, n);
    return 0;
}
```

输出：

```
0 - 1: 2
1 - 2: 3
0 - 3: 6
1 - 4: 5
MST cost: 16
```

Python 版本

```python
def prim_matrix(graph):
    V = len(graph)
    key = [float('inf')] * V
    parent = [-1] * V
    in_mst = [False] * V
    key[0] = 0

    for _ in range(V - 1):
        u = min((key[v], v) for v in range(V) if not in_mst[v])[1]
        in_mst[u] = True
        for v in range(V):
            if graph[u][v] != 0 and not in_mst[v] and graph[u][v] < key[v]:
                key[v] = graph[u][v]
                parent[v] = u

    edges = [(parent[i], i, graph[i][parent[i]]) for i in range(1, V)]
    cost = sum(w for _, _, w in edges)
    return cost, edges

graph = [
    [0,2,0,6,0],
    [2,0,3,8,5],
    [0,3,0,0,7],
    [6,8,0,0,9],
    [0,5,7,9,0]
]
print(prim_matrix(graph))
```

输出：

```
(16, [(0,1,2), (1,2,3), (0,3,6), (1,4,5)])
```

#### 为什么它重要？

-   使用邻接矩阵易于实现。
-   最适合 $E \approx V^2$ 的稠密图。
-   避免了堆的复杂性。
-   便于在课堂或教学用途中进行可视化和调试。

#### 一个温和的证明（为什么它有效）

与基于堆的版本类似，此变体依赖于割性质：
在每一步，选择的边都以最小权重连接树内的一个顶点和树外的一个顶点，因此它总是安全的。
每次迭代都在不形成环的情况下扩展 MST，经过 $V-1$ 次迭代后，所有顶点都被连接起来。

#### 亲自尝试

1.  在一个完全图上运行，预期得到 $V-1$ 条最轻的边。
2.  修改权重，观察边的选择如何变化。
3.  随着 $V$ 的增长，与基于堆的 Prim 算法比较运行时间。
4.  以 $O(V^2)$ 复杂度实现，并通过实验验证复杂度。
5.  在不连通图上测试，观察算法在何处停止。

#### 测试用例

| 图                     | MST 边                               | 成本                     |            |
| ---------------------- | ------------------------------------ | ------------------------ | ---------- |
| 5 节点矩阵             | (0,1,2), (1,2,3), (0,3,6), (1,4,5)   | 16                       |            |
| 3 节点三角形           | (0,1,1), (1,2,2)                     | 3                        |            |
| 稠密完全图（4 节点）   | 6 条边                               | 选择最轻的 3 条边        | 最小边之和 |

#### 复杂度

-   时间：$O(V^2)$
-   空间：$O(V^2)$（矩阵）
-   输出大小：$|V|-1$ 条边

Prim（邻接矩阵）是经典的稠密图版本，它以牺牲速度为代价，换取了简单性和可预测的访问时间。
### 344 Borůvka（博鲁夫卡）算法

Borůvka 算法是最早的最小生成树（MST）算法之一（1926年），其设计思路是通过反复连接每个连通分量与其最廉价的外向边来构建最小生成树。它分阶段运行，不断合并连通分量，直到只剩下一个树。

#### 我们要解决什么问题？

给定一个连通的、无向的、带权重的图 $G = (V, E, w)$，我们希望找到一个最小生成树（MST），即一个子集 $T \subseteq E$，满足：

- $T$ 连接所有顶点，
- $|T| = |V| - 1$，
- $\sum_{e \in T} w(e)$ 最小化。

Borůvka 的方法并行地生长多个子树，这使得它非常适合于并行或分布式计算。

#### 它是如何工作的（通俗解释）？

该算法分轮次进行。每一轮中，每个连通分量都通过其最轻的外向边连接到另一个连通分量。

1.  开始时，每个顶点自成一个连通分量。
2.  对于每个连通分量，找到连接它到另一个分量的权重最小的边。
3.  将所有找到的边添加到 MST 中，这些边是安全的（由割性质保证）。
4.  合并由这些边连接的连通分量。
5.  重复上述步骤，直到只剩下一个连通分量。

每一轮至少将连通分量的数量减半，因此算法在 $O(\log V)$ 个阶段内完成。

| 步骤 | 操作                             | 描述                     |
| ---- | -------------------------------- | ------------------------ |
| 1    | 初始化连通分量                   | 每个顶点单独成一分量     |
| 2    | 为每个分量找到最廉价的边         | 最轻的外向边             |
| 3    | 添加边                           | 合并连通分量             |
| 4    | 重复                             | 直到只剩下一个连通分量   |

#### 精简代码

Python（并查集版本）

```python
class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b: return False
        if self.r[a] < self.r[b]: self.p[a] = b
        elif self.r[a] > self.r[b]: self.p[b] = a
        else: self.p[b] = a; self.r[a] += 1
        return True

def boruvka_mst(V, edges):
    dsu = DSU(V)
    mst = []
    total_cost = 0
    components = V

    while components > 1:
        cheapest = [-1] * V
        for i, (u, v, w) in enumerate(edges):
            set_u = dsu.find(u)
            set_v = dsu.find(v)
            if set_u != set_v:
                if cheapest[set_u] == -1 or edges[cheapest[set_u]][2] > w:
                    cheapest[set_u] = i
                if cheapest[set_v] == -1 or edges[cheapest[set_v]][2] > w:
                    cheapest[set_v] = i
        for i in range(V):
            if cheapest[i] != -1:
                u, v, w = edges[cheapest[i]]
                if dsu.union(u, v):
                    mst.append((u, v, w))
                    total_cost += w
                    components -= 1
    return total_cost, mst

edges = [(0,1,1), (0,2,3), (1,2,1), (1,3,4), (2,3,5)]
print(boruvka_mst(4, edges))
```

输出：

```
(6, [(0,1,1), (1,2,1), (1,3,4)])
```

#### 为什么它重要？

-   天生可并行化（每个连通分量独立行动）。
-   基于割性质的重复应用，简单而优雅。
-   非常适合稀疏图和分布式系统。
-   常用于混合 MST 算法（与 Kruskal/Prim 结合）。

#### 一个温和的证明（为什么它有效）

根据割性质，对于每个连通分量 $C$，离开 $C$ 的最廉价边总是可以安全地包含在 MST 中。由于边是在所有连通分量中同时选择的，并且在一个阶段内不会产生环（连通分量只通过割进行合并），所以每一轮都保持了正确性。每个阶段之后，连通分量合并，这个过程重复进行，直到所有分量统一为一个。

每次迭代至少将连通分量的数量减半，确保了 $O(\log V)$ 个阶段。

#### 亲自尝试

1.  在一个小图上运行并手动跟踪各个阶段。
2.  与 Kruskal 的排序边方法进行比较。
3.  添加并行日志记录以可视化同时发生的合并。
4.  观察连通分量如何指数级减少。
5.  与 Kruskal 混合使用：使用 Borůvka 直到剩下少量连通分量，然后切换。

#### 测试用例

| 图           | 边 $(u,v,w)$                     | MST                             | 总权重        |
| ------------ | ------------------------------- | ------------------------------- | ------------- |
| 三角形       | (0,1,1),(1,2,2),(0,2,3)         | (0,1,1),(1,2,2)                 | 3             |
| 正方形       | (0,1,1),(1,2,2),(2,3,1),(3,0,2) | (0,1,1),(2,3,1),(1,2,2)         | 4             |
| 稠密 4 节点  | 6 条边                          | 在 2 个阶段内构建 MST           | 已验证的权重  |

#### 复杂度

-   阶段数：$O(\log V)$
-   每阶段时间：$O(E)$
-   总时间：$O(E \log V)$
-   空间：$O(V + E)$

Borůvka 算法是用于寻找最小生成树的并行生长-合并策略，每个连通分量都通过其最轻的边向外连接，直到整个图变成一个连通的树。
### 345 反向删除最小生成树算法

反向删除算法通过从完整图开始，反复删除边来构建最小生成树（MST），但只在删除后不会断开图连接时才进行删除。它在概念上是 Kruskal 算法的镜像。

#### 我们要解决什么问题？

给定一个连通的、无向的、带权重的图 $G = (V, E, w)$，我们想要找到一个最小生成树——一个连接所有顶点且总权重最小的生成树。

与 Kruskal 算法添加边不同，我们从所有边开始，然后一条一条地删除它们，确保每次删除后图仍然保持连通。

#### 它是如何工作的（通俗解释）？

1.  将所有边按权重降序排序。
2.  初始化工作图 $T = G$。
3.  按上述顺序处理每条边 $(u,v)$：
    *   暂时从 $T$ 中移除边 $(u,v)$。
    *   如果 $u$ 和 $v$ 在 $T$ 中仍然连通，则永久删除这条边（它是不需要的）。
    *   否则，恢复它（它是必需的）。
4.  当所有边都处理完毕后，$T$ 就是最小生成树。

这种方法确保只有不可或缺的边被保留下来，形成一个总权重最小的生成树。

| 步骤 | 动作               | 描述                     |   |   |   |     |
| ---- | ------------------ | ------------------------ | - | - | - | --- |
| 1    | 边降序排序         | 权重大的边优先处理       |   |   |   |     |
| 2    | 移除边             | 测试连通性               |   |   |   |     |
| 3    | 必要时保留         | 如果移除会断开连接       |   |   |   |     |
| 4    | 停止               | 当 $                     | T | = | V | -1$ |

#### 微型代码

Python（使用 DFS 进行连通性检查）

```python
def dfs(graph, start, visited):
    visited.add(start)
    for v, _ in graph[start]:
        if v not in visited:
            dfs(graph, v, visited)

def is_connected(graph, u, v):
    visited = set()
    dfs(graph, u, visited)
    return v in visited

def reverse_delete_mst(V, edges):
    edges = sorted(edges, key=lambda x: x[2], reverse=True)
    graph = {i: [] for i in range(V)}
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    mst_cost = sum(w for _, _, w in edges)

    for u, v, w in edges:
        # 移除边
        graph[u] = [(x, wx) for x, wx in graph[u] if x != v]
        graph[v] = [(x, wx) for x, wx in graph[v] if x != u]

        if is_connected(graph, u, v):
            mst_cost -= w  # 边不需要
        else:
            # 恢复边
            graph[u].append((v, w))
            graph[v].append((u, w))

    mst_edges = []
    visited = set()
    def collect(u):
        visited.add(u)
        for v, w in graph[u]:
            if (v, w) not in visited:
                mst_edges.append((u, v, w))
                if v not in visited:
                    collect(v)
    collect(0)
    return mst_cost, mst_edges

edges = [(0,1,1), (0,2,2), (1,2,3), (1,3,4), (2,3,5)]
print(reverse_delete_mst(4, edges))
```

输出：

```
(7, [(0,1,1), (0,2,2), (1,3,4)])
```

#### 为什么它重要

*   是 Kruskal 算法的一个简单对偶视角。
*   演示了环路性质：
    > 在任何环路中，权重最大的边不可能在最小生成树中。
*   适用于教学证明和概念理解。
*   可用于验证最小生成树或以反向方式构建它们。

#### 一个温和的证明（为什么它有效）

环路性质指出：
> 对于图中的任何环路，权重最大的边不可能属于最小生成树。

通过按权重降序排序边，并删除每条位于环路中的最大权重边（即当 $u$ 和 $v$ 在没有它的情况下仍然连通时），我们消除了所有非最小生成树的边。当没有这样的边剩下时，结果就是一个最小生成树。

由于每次删除都保持了连通性，最终的图是一个总权重最小的生成树。

#### 自己动手试试

1.  在一个三角形图上运行，观察最重的边被移除。
2.  将删除顺序与 Kruskal 算法的添加顺序进行比较。
3.  可视化每一步的图。
4.  用 BFS 或并查集替换 DFS 以提高速度。
5.  用它来验证其他算法输出的最小生成树。

#### 测试用例

| 图       | 边 $(u,v,w)$                  | 最小生成树               | 总权重 |
| -------- | ----------------------------- | ------------------------ | ------ |
| 三角形   | (0,1,1),(1,2,2),(0,2,3)       | (0,1,1),(1,2,2)          | 3      |
| 正方形   | (0,1,1),(1,2,2),(2,3,3),(3,0,4) | (0,1,1),(1,2,2),(2,3,3)  | 6      |
| 线       | (0,1,2),(1,2,1),(2,3,3)       | 所有边                   | 6      |

#### 复杂度

*   时间复杂度：使用朴素的 DFS 时为 $O(E(E+V))$（每次删除边都需要检查连通性）。通过并查集优化或预计算结构，可以降低复杂度。
*   空间复杂度：$O(V + E)$。

反向删除算法是构建最小生成树的"减法"视角，剥离掉重的边，直到只剩下必需且轻量的结构。
### 346 通过 Dijkstra 技巧构建最小生成树

这个变体使用类似 Dijkstra 的过程构建最小生成树，通过反复选择连接树内顶点与树外顶点的最轻边来扩展树。
本质上，这是从 Dijkstra 视角重新表述的 Prim 算法，展示了最短路径和生成树增长之间的深刻相似性。

#### 我们要解决什么问题？

给定一个具有非负边权的连通、无向、加权图 $G=(V,E,w)$，我们想要一个最小生成树——
一个子集 $T \subseteq E$，它连接所有顶点，并且满足
$$
|T| = |V| - 1, \quad \text{且} \quad \sum_{e \in T} w(e) \text{ 最小化}.
$$

虽然 Dijkstra 算法基于 *路径成本* 构建最短路径树，但这个版本通过直接使用边权作为从当前树的“距离”来构建最小生成树。

#### 它是如何工作的（通俗解释）？

这是伪装的 Prim 算法：
我们不是跟踪从根出发的最短路径，而是跟踪每个顶点连接到正在生长的树的最轻边。

1.  初始化所有顶点 $\text{key}[v]=\infty$，除了起始顶点 $s$ 设置 $\text{key}[s]=0$。
2.  使用一个以 $\text{key}[v]$ 为键的优先队列（最小堆）。
3.  反复提取具有最小键的顶点 $u$。
4.  对于每个邻居 $v$：

    *   如果 $v$ 尚未在树中且 $w(u,v)<\text{key}[v]$，
        则更新 $\text{key}[v]=w(u,v)$ 并记录父节点。
5.  重复直到所有顶点都被包含。

与 Dijkstra 不同，我们不累加边权，我们只关心到达每个顶点的最小边。

| 步骤            | Dijkstra                                   | 通过 Dijkstra 技巧构建最小生成树                |
| --------------- | ------------------------------------------ | --------------------------------------------- |
| 距离更新        | $\text{dist}[v] = \text{dist}[u] + w(u,v)$ | $\text{key}[v] = \min(\text{key}[v], w(u,v))$ |
| 优先级          | 路径成本                                   | 边成本                                        |
| 目标            | 最短路径                                   | 最小生成树                                    |

#### 微型代码

Python（堆实现）

```python
import heapq

def mst_dijkstra_trick(V, edges, start=0):
    # 构建邻接表
    graph = [[] for _ in range(V)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    in_tree = [False] * V
    key = [float('inf')] * V
    parent = [-1] * V
    key[start] = 0
    pq = [(0, start)]

    total_cost = 0
    while pq:
        w, u = heapq.heappop(pq)
        if in_tree[u]:
            continue
        in_tree[u] = True
        total_cost += w

        for v, wt in graph[u]:
            if not in_tree[v] and wt < key[v]:
                key[v] = wt
                parent[v] = u
                heapq.heappush(pq, (wt, v))

    mst_edges = [(parent[v], v, key[v]) for v in range(V) if parent[v] != -1]
    return total_cost, mst_edges

edges = [(0,1,2),(0,2,3),(1,2,1),(1,3,4),(2,3,5)]
print(mst_dijkstra_trick(4, edges))
```

输出：

```
(7, [(0,1,2), (1,2,1), (1,3,4)])
```

#### 为什么这很重要

-   展示了最小生成树和最短路径搜索之间的对偶性。
-   在 Prim 和 Dijkstra 之间提供了概念桥梁。
-   对教学和算法统一很有用。
-   当已经熟悉 Dijkstra 的结构时，非常直观。

#### 一个温和的证明（为什么它有效）

在每次迭代中，我们维护一个已经包含在最小生成树中的顶点集合 $S$。
割性质保证了连接 $S$ 到 $V\setminus S$ 的最轻边总是可以安全地包含进来。

通过将这些边权存储在优先队列中并总是选择最小的，
我们严格遵循了这个性质，从而逐步构建出一个最小生成树。

当所有顶点都被包含时，算法停止，得到一个最小生成树。

#### 亲自尝试

1.  与 Dijkstra 算法逐行比较，只在松弛步骤有一处改变。
2.  在完全图上运行，观察星形的最小生成树。
3.  尝试具有多条等权边的图，观察如何处理平局。
4.  将堆替换为数组，检查 $O(V^2)$ 版本。
5.  用高亮边界进行可视化，关注边而不是距离。

#### 测试用例

| 图       | 边 $(u,v,w)$                     | 最小生成树                     | 成本 |
| -------- | ------------------------------- | ----------------------- | ---- |
| 三角形   | (0,1,1),(1,2,2),(0,2,3)         | (0,1,1),(1,2,2)         | 3    |
| 正方形   | (0,1,2),(1,2,3),(2,3,1),(3,0,4) | (2,3,1),(0,1,2),(1,2,3) | 6    |
| 线       | (0,1,5),(1,2,2),(2,3,1)         | 所有边                   | 8    |

#### 复杂度

-   时间：$O(E \log V)$（堆）或 $O(V^2)$（数组）
-   空间：$O(V + E)$
-   输出大小：$|V|-1$ 条边

通过 Dijkstra 技巧构建最小生成树是 Prim 算法的重新构想，它用边的最小化取代了距离的累加，证明了两个经典的图算法思想如何共享一个贪婪的核心。
### 347 动态最小生成树维护

动态最小生成树维护要解决的问题是：当底层图发生变化，边被添加、删除或其权重改变时，我们如何高效地更新最小生成树，而无需从头开始重建？

这个问题出现在图随时间演化的系统中，例如网络、道路地图或实时优化系统。

#### 我们要解决什么问题？

给定一个图 $G=(V,E,w)$ 及其最小生成树 $T$，我们希望在以下更新操作下维护 $T$：

- 边插入：添加新边 $(u,v,w)$
- 边删除：移除现有边 $(u,v)$
- 边权重更新：改变权重 $w(u,v)$

在每次变化后朴素地重新计算最小生成树的成本是 $O(E \log V)$。
动态维护通过对 $T$ 进行增量修复，可以显著降低这一成本。

#### 它是如何工作的（通俗解释）？

1. 边插入：

   * 添加新边 $(u,v,w)$。
   * 检查它是否在 $T$ 中形成一个环。
   * 如果新边比该环上最重的边更轻，则替换掉那条重边。
   * 否则，丢弃新边。

2. 边删除：

   * 从 $T$ 中移除 $(u,v)$，这会将 $T$ 分裂成两个连通分量。
   * 在 $G$ 中找到连接这两个分量的最轻边。
   * 添加那条边以恢复连通性。

3. 边权重更新：

   * 如果一条边的权重增加，则将其视为潜在的删除操作。
   * 如果其权重减少，则将其视为潜在的插入操作。

为了高效地完成这些操作，我们需要能够快速实现以下功能的数据结构：

- 快速找到路径上的最大边（用于环检测）
- 找到连通分量之间的最小交叉边

这可以通过动态树、Link-Cut 树或欧拉回路树来实现。

#### 简化代码（简化的静态版本）

这个版本演示了带环检测的边插入维护。

```python
def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, rank, x, y):
    rx, ry = find(parent, x), find(parent, y)
    if rx == ry:
        return False
    if rank[rx] < rank[ry]:
        parent[rx] = ry
    elif rank[rx] > rank[ry]:
        parent[ry] = rx
    else:
        parent[ry] = rx
        rank[rx] += 1
    return True

def dynamic_mst_insert(V, mst_edges, new_edge):
    u, v, w = new_edge
    parent = [i for i in range(V)]
    rank = [0]*V
    for x, y, _ in mst_edges:
        union(parent, rank, x, y)
    if find(parent, u) != find(parent, v):
        mst_edges.append(new_edge)
    else:
        # 会形成环，选择更轻的边
        cycle_edges = mst_edges + [new_edge]
        heaviest = max(cycle_edges, key=lambda e: e[2])
        if heaviest != new_edge:
            mst_edges.remove(heaviest)
            mst_edges.append(new_edge)
    return mst_edges
```

这是概念性的；在实践中，Link-Cut 树可以在对数时间内实现动态操作。

#### 为什么它很重要

- 对于流图、在线网络、实时路由至关重要。
- 避免了每次更新后的重新计算。
- 展示了在变化下贪心不变量的保持，如果维护得当，最小生成树始终保持最小。
- 构成了完全动态图算法的基础。

#### 一个温和的证明（为什么它有效）

最小生成树的不变性通过割性质和环性质得以保持：

- 环性质：在一个环中，最重的边不可能在最小生成树中。
- 割性质：在任何割中，跨越该割的最轻边必须在最小生成树中。

当插入一条边时：

- 它形成一个环，我们丢弃最重的边以保持最小性。

当删除一条边时：

- 它形成一个割，我们插入跨越该割的最轻边以保持连通性。

因此，每次更新都在局部时间内恢复了一个有效的最小生成树。

#### 动手试试

1. 从一个小子图的最小生成树开始。
2. 插入一条边并追踪形成的环。
3. 删除一条最小生成树中的边并找到替代边。
4. 尝试批量更新，观察结构如何演化。
5. 与完全重新计算的运行时间进行比较。

#### 测试用例

| 操作               | 描述               | 结果                          |
| ------------------ | ------------------ | ----------------------------- |
| 插入 $(1,3,1)$     | 创建环 $(1,2,3,1)$ | 移除最重的边                  |
| 删除 $(0,1)$       | 破坏树结构         | 找到最轻的重连边              |
| 减少权重 $(2,3)$   | 重新检查是否包含   | 边可能进入最小生成树          |

#### 复杂度

- 插入/删除（朴素方法）：$O(E)$
- 动态树（Link-Cut）：每次更新 $O(\log^2 V)$
- 空间：$O(V + E)$

动态最小生成树维护展示了局部调整如何保持全局最优性，这是演化系统的一个强大原则。
### 348 最小瓶颈生成树

最小瓶颈生成树（MBST）是一种生成树，它最小化树中所有边的最大边权。与标准的最小生成树（MST）最小化边权总和不同，MBST 关注的是树中最差（最重）的那条边。

在许多现实世界的系统中，例如网络设计或交通规划，你可能不那么关心总成本，而更关心瓶颈约束，即最弱或最慢的连接。

#### 我们要解决什么问题？

给定一个连通的、无向的、带权重的图 $G = (V, E, w)$，我们希望找到一个生成树 $T \subseteq E$，使得

$$
\max_{e \in T} w(e)
$$

尽可能小。

换句话说，在所有生成树中，挑选一棵其中最大边权最小的树。

#### 它是如何工作的（通俗解释）？

你可以使用与 MST 相同的算法（Kruskal 或 Prim）来构建 MBST，因为：

> 每个最小生成树同时也是最小瓶颈生成树。

推理如下：

- MST 最小化了边权总和。
- 在此过程中，它也确保了不会留下不必要的重边。

因此，任何 MST 都自动满足瓶颈性质。

或者，你可以使用二分搜索 + 连通性测试：

1.  按权重对边进行排序。
2.  对阈值 $W$ 进行二分搜索。
3.  检查权重 $w(e) \le W$ 的边是否能连接所有顶点。
4.  能使图连通的最小 $W$ 值就是瓶颈权重。
5.  仅使用权重 $\le W$ 的边提取任意一棵生成树。

#### 微型代码（基于 Kruskal）

```python
def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, rank, x, y):
    rx, ry = find(parent, x), find(parent, y)
    if rx == ry:
        return False
    if rank[rx] < rank[ry]:
        parent[rx] = ry
    elif rank[rx] > rank[ry]:
        parent[ry] = rx
    else:
        parent[ry] = rx
        rank[rx] += 1
    return True

def minimum_bottleneck_spanning_tree(V, edges):
    edges = sorted(edges, key=lambda x: x[2])
    parent = [i for i in range(V)]
    rank = [0]*V
    bottleneck = 0
    count = 0
    for u, v, w in edges:
        if union(parent, rank, u, v):
            bottleneck = max(bottleneck, w)
            count += 1
            if count == V - 1:
                break
    return bottleneck

edges = [(0,1,4),(0,2,3),(1,2,2),(1,3,5),(2,3,6)]
print(minimum_bottleneck_spanning_tree(4, edges))
```

输出：

```
5
```

这里，MST 的总权重是 4+3+2=9，瓶颈边权重是 5。

#### 为什么它很重要

-   对于服务质量或带宽受限的系统很有用。
-   确保树中没有边超过关键容量阈值。
-   说明最小化最大值和最小化总和可以是一致的，这是贪心算法中的一个关键见解。
-   提供了一种测试 MST 正确性的方法：如果一个 MST 没有最小化瓶颈，那么它就是无效的。

#### 一个温和的证明（为什么它有效）

设 $T^*$ 是一个 MST，$B(T^*)$ 是其最大边权。

假设另一棵树 $T'$ 有一个更小的瓶颈：
$$
B(T') < B(T^*)
$$
那么 $T^*$ 中存在一条边 $e$，其权重 $w(e) = B(T^*)$，但 $T'$ 避开了这条边，同时使用更轻的边保持连通。

这与环性质相矛盾，因为 $e$ 将被一条跨越同一割的更轻的边所取代，这意味着 $T^*$ 不是最小的。

因此，每个 MST 都是一个 MBST。

#### 亲自尝试

1.  构建一个具有多个 MST 的图，检查它们是否具有相同的瓶颈。
2.  比较 MST 的总权重与 MBST 的瓶颈。
3.  应用二分搜索方法，确认一致性。
4.  可视化所有生成树，并在每棵树中标记最大边。
5.  构造一个存在多个 MBST 的案例。

#### 测试用例

| 图       | 边 $(u,v,w)$                  | MST                         | 瓶颈 |
| -------- | ----------------------------- | --------------------------- | ---- |
| 三角形   | (0,1,1),(1,2,2),(0,2,3)       | (0,1,1),(1,2,2)             | 2    |
| 正方形   | (0,1,1),(1,2,5),(2,3,2),(3,0,4) | (0,1,1),(2,3,2),(3,0,4)     | 4    |
| 链       | (0,1,2),(1,2,3),(2,3,4)       | 相同                        | 4    |

#### 复杂度

-   时间：$O(E \log E)$（与 Kruskal 相同）
-   空间：$O(V)$
-   输出：瓶颈权重或边

最小瓶颈生成树突出了承载负荷最重的链路，当弹性、延迟或带宽限制比总成本更重要时，这是一个关键指标。
### 349 曼哈顿最小生成树

曼哈顿最小生成树（Manhattan MST）旨在找到一个生成树，使得网格上连接点之间的曼哈顿距离之和最小。
这种变体在 VLSI 设计、城市规划以及基于网格的路径优化中很常见，其中移动被限制在轴对齐方向上。

#### 我们要解决什么问题？

给定二维空间中的 $n$ 个点，坐标为 $(x_i, y_i)$，点 $p_i$ 和 $p_j$ 之间的曼哈顿距离为

$$
d(p_i, p_j) = |x_i - x_j| + |y_i - y_j|
$$

我们希望构建一个连接所有点的生成树，使得总的曼哈顿距离最小。

一个朴素的解决方案会考虑所有 $\binom{n}{2}$ 条边并运行 Kruskal 算法，但对于大的 $n$ 来说这太慢了。
关键在于利用几何特性来限制候选边的数量。

#### 它是如何工作的（通俗解释）？

曼哈顿最小生成树问题利用了一个几何性质：

> 对于每个点，只有少数几个最近邻（在某些变换下）可能属于最小生成树。

因此，我们不是检查所有点对，而是：

1. 按照特定方向（旋转/反射）对点进行排序。
2. 使用扫描线或树状数组（Fenwick tree）来查找候选邻居。
3. 收集 $O(n)$ 条潜在的边。
4. 在这个缩减后的集合上运行 Kruskal 算法。

通过考虑 8 个方向变换，我们确保包含了所有可能的最近邻边。

变换示例：

- $(x, y)$
- $(y, x)$
- $(-x, y)$
- $(x, -y)$
- 等等。

对于每个变换：

- 按照 $(x+y)$ 或 $(x-y)$ 对点排序。
- 对于每个点，追踪最小化 $|x|+|y|$ 的候选邻居。
- 在每个点与其最近邻之间添加边。

最后，在收集到的边上使用 Kruskal 算法计算最小生成树。

#### 简化代码示例

```python
def manhattan_distance(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

def manhattan_mst(points):
    edges = []
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            w = manhattan_distance(points[i], points[j])
            edges.append((i, j, w))
    edges.sort(key=lambda e: e[2])

    parent = [i for i in range(n)]
    def find(x):
        if parent[x]!=x:
            parent[x]=find(parent[x])
        return parent[x]
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx!=ry:
            parent[ry]=rx
            return True
        return False

    mst_edges = []
    cost = 0
    for u, v, w in edges:
        if union(u, v):
            mst_edges.append((u, v, w))
            cost += w
    return cost, mst_edges

points = [(0,0), (2,2), (2,0), (0,2)]
print(manhattan_mst(points))
```

输出：

```
(8, [(0,2,2),(0,3,2),(2,1,4)])
```

这个暴力示例通过检查所有点对来构建曼哈顿最小生成树。
高效的几何变体算法将复杂度从 $O(n^2)$ 降低到 $O(n \log n)$。

#### 为什么它很重要

- 捕捉基于网格的移动（无对角线移动）。
- 在 VLSI 电路布局（导线长度最小化）中至关重要。
- 是城市街区规划和配送网络的基础。
- 展示了在空间问题中几何与图论如何融合。

#### 一个温和的证明（为什么它有效）

曼哈顿距离下的最小生成树满足割性质：跨越任何分割的最轻边必须在最小生成树中。
通过确保包含所有方向上的邻居，我们永远不会错过跨越任何割的最小边。

因此，即使我们剪枝了候选边，正确性仍然得以保留。

#### 自己动手试试

1.  生成随机点，可视化最小生成树的边。
2.  比较曼哈顿最小生成树与欧几里得最小生成树。
3.  添加对角线，看看成本如何变化。
4.  尝试优化的方向邻居搜索。
5.  观察对称性，每个变换覆盖一个象限。

#### 测试用例

| 点坐标                  | 最小生成树边 | 总成本 |
| ----------------------- | ------------ | ------ |
| (0,0),(1,0),(1,1)       | (0,1),(1,2)  | 2      |
| (0,0),(2,2),(2,0),(0,2) | 3 条边       | 8      |
| (0,0),(3,0),(0,4)       | (0,1),(0,2)  | 7      |

#### 复杂度

- 朴素方法：$O(n^2 \log n)$
- 优化方法（基于几何）：$O(n \log n)$
- 空间：$O(n)$

曼哈顿最小生成树连接了网格几何与图优化，是结构指导效率的完美范例。
### 350 欧几里得最小生成树（Kruskal + 几何）

欧几里得最小生成树（EMST）用最短的总欧几里得长度连接平面中的一组点，是直线距离下布线、管道或连接布局的最小可能方案。

它是经典最小生成树问题的几何对应物，是计算几何、网络设计和空间聚类的核心。

#### 我们要解决什么问题？

给定二维（或更高维）中的 $n$ 个点 $P = {p_1, p_2, \dots, p_n}$，我们希望找到一个生成树 $T$，最小化

$$
\text{cost}(T) = \sum_{(p_i,p_j) \in T} \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}
$$

与抽象图的最小生成树不同，这里的图是完全图，每对点之间都有一条以欧几里得距离为权重的边 —— 但对于大的 $n$，我们无法承受 $O(n^2)$ 条边的开销。

#### 它是如何工作的（通俗解释）？

关键的几何见解是：

> EMST 总是 Delaunay 三角剖分（DT）的一个子图。

因此，我们不需要所有 $\binom{n}{2}$ 条边，只需要 Delaunay 图中的边（它有 $O(n)$ 条边）。

算法概述：

1.  计算点的 Delaunay 三角剖分（DT）。
2.  提取所有 DT 边及其欧几里得权重。
3.  在这些边上运行 Kruskal 或 Prim 算法。
4.  得到的树就是 EMST。

这极大地将时间复杂度从二次降低到接近线性。

#### 微型代码（暴力演示）

这是一个使用所有点对的极简版本，适用于小的 $n$：

```python
import math

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])2 + (p1[1]-p2[1])2)

def euclidean_mst(points):
    edges = []
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            w = euclidean_distance(points[i], points[j])
            edges.append((i, j, w))
    edges.sort(key=lambda e: e[2])

    parent = [i for i in range(n)]
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx
            return True
        return False

    mst_edges = []
    cost = 0
    for u, v, w in edges:
        if union(u, v):
            mst_edges.append((u, v, w))
            cost += w
    return cost, mst_edges

points = [(0,0),(1,0),(0,1),(1,1)]
print(euclidean_mst(points))
```

输出：

```
(3.0, [(0,1,1.0),(0,2,1.0),(1,3,1.0)])
```

在实际使用时，请用几何库生成的 Delaunay 边替换掉全点对循环。

#### 为什么它重要

-   以最小总长度最优地连接点。
-   用于地理信息系统、道路网络、聚类（单链接）、传感器网络和模式分析。
-   展示了几何与图论之间的相互作用。
-   是 Steiner 树和 TSP 近似的基础。

#### 一个温和的证明（为什么它有效）

EMST 遵循割性质：对于点的任意划分，最短的连接边必须在 EMST 中。

不在 Delaunay 三角剖分中的边，可以被穿过相同割的、来自 DT 的更短边所替代，因此 EMST 必须位于 DT 内部。

因此，我们可以安全地将候选边限制在 Delaunay 集合中，同时保持最优性。

#### 自己动手试试

1.  在平面上绘制点，画出 EMST 的边。
2.  比较 EMST 与使用曼哈顿距离的 MST。
3.  生成聚类中的点，可视化 EMST 如何连接聚类。
4.  使用几何库（例如 `scipy.spatial.Delaunay`）来构建快速的 DT 边。
5.  测量运行时间差异：暴力法 vs 基于 DT 的方法。

#### 测试用例

| 点集             | EMST                    | 总长度           |
| ---------------- | ----------------------- | ---------------- |
| 正方形角点       | (0,0),(1,0),(1,1),(0,1) | 3.0              |
| 三角形           | (0,0),(2,0),(1,√3)      | 4.0              |
| 随机 5 个点      |,                       | 取决于坐标       |

#### 复杂度

-   暴力法：$O(n^2 \log n)$
-   基于 Delaunay 的方法：$O(n \log n)$
-   空间：$O(n)$

欧几里得最小生成树是几何上的最小骨干，是平面上将所有点连接在一起的最短无形之线。

## 第 36 节 流
### 351 Ford–Fulkerson（福特-富尔克森）

Ford–Fulkerson 方法是计算有向网络中最大流的基础算法。
它将流视为水流在管道系统中移动，从源点推送到汇点，同时遵守容量限制和流量守恒。

#### 我们正在解决什么问题？

给定一个流网络 $G = (V, E)$，其中包含：

- 一个源点 $s$，
- 一个汇点 $t$，
- 以及每条边上的容量 $c(u, v)$，

我们想要找到一个最大流 $f(u, v)$，使得：

1. 容量约束：
   $$0 \le f(u, v) \le c(u, v)$$

2. 流量守恒：
   $$\sum_{(u,v)\in E} f(u,v) = \sum_{(v,u)\in E} f(v,u) \quad \forall u \in V \setminus {s, t}$$

3. 最大化总流量：
   $$
   \text{最大化 } |f| = \sum_{(s,v)\in E} f(s,v)
   $$

#### 它是如何工作的（通俗解释）？

该算法反复搜索增广路径，即从 $s$ 到 $t$ 的、仍可发送额外流量的路径。
每次迭代都会增加总流量，直到没有增广路径剩余。

逐步说明：

1. 将所有流量初始化为 0。
2. 只要在残差图中存在一条从 $s$ 到 $t$ 的增广路径 $P$：

   * 找到沿 $P$ 的瓶颈容量：
     $$b = \min_{(u,v)\in P} (c(u,v) - f(u,v))$$
   * 沿 $P$ 增加流量 $b$：
     $$f(u,v) \gets f(u,v) + b$$
     $$f(v,u) \gets f(v,u) - b$$
3. 更新残差容量并重复。
4. 当没有路径剩余时，当前流量即为最大流。

残差图表示剩余的容量，允许在需要时通过反向边来取消流量。

#### 微型代码

Python（基于 DFS 的增广）

```python
from collections import defaultdict

def ford_fulkerson(capacity, s, t):
    n = len(capacity)
    flow = [[0]*n for _ in range(n)]

    def dfs(u, t, f, visited):
        if u == t:
            return f
        visited[u] = True
        for v in range(n):
            residual = capacity[u][v] - flow[u][v]
            if residual > 0 and not visited[v]:
                pushed = dfs(v, t, min(f, residual), visited)
                if pushed > 0:
                    flow[u][v] += pushed
                    flow[v][u] -= pushed
                    return pushed
        return 0

    max_flow = 0
    while True:
        visited = [False]*n
        pushed = dfs(s, t, float('inf'), visited)
        if pushed == 0:
            break
        max_flow += pushed
    return max_flow

capacity = [
    [0, 16, 13, 0, 0, 0],
    [0, 0, 10, 12, 0, 0],
    [0, 4, 0, 0, 14, 0],
    [0, 0, 9, 0, 0, 20],
    [0, 0, 0, 7, 0, 4],
    [0, 0, 0, 0, 0, 0]
]
print(ford_fulkerson(capacity, 0, 5))
```

输出：

```
23
```

#### 为什么它很重要

- 引入了最大流概念，这是网络优化的基石。
- 支撑了以下算法的基础：

  * 二分图匹配
  * 最小割问题
  * 环流和网络设计
- 阐释了残差图和增广路径，这些概念被用于许多基于流的算法中。

#### 一个温和的证明（为什么它有效）

每次增广都会以一个正数增加总流量，并保持可行性。
因为总容量是有限的，所以该过程最终会终止。

根据最大流最小割定理：

$$
|f^*| = \text{最小 } (S, T) \text{ 割的容量}
$$

因此，当不存在增广路径时，流量即为最大。

如果所有容量都是整数，该算法会在有限步内收敛，因为每次增广至少增加 1 个单位。

#### 亲自尝试

1. 在小网络上运行，绘制每一步的残差图。
2. 追踪增广路径和瓶颈边。
3. 与 Edmonds–Karp（BFS 搜索）进行比较。
4. 将容量改为分数，观察可能的无限循环。
5. 用它来解决二分图匹配问题（转换为流网络）。

#### 测试用例

| 图                      | 最大流           | 备注             |
| ---------------------- | ---------------- | ---------------- |
| 简单的 3 节点图        | 5                | 单次增广         |
| 经典的 6 节点图（如上）| 23               | 教科书示例       |
| 平行边                 | 容量之和         | 可加性           |

#### 复杂度

- 时间复杂度：对于整数容量为 $O(E \cdot |f^*|)$
- 空间复杂度：$O(V^2)$
- 优化版本（Edmonds–Karp）：$O(VE^2)$

Ford–Fulkerson 算法建立了通过容量受限路径推送流量的直观理解，这是网络优化的核心。
### 352 Edmonds–Karp（埃德蒙兹-卡普）

Edmonds–Karp 算法是 Ford–Fulkerson 方法的一个具体、多项式时间的实现，其中每次增广路径都是使用广度优先搜索（BFS）选择的。通过总是选择从源点到汇点的最短路径（按边数计算），它保证了高效的收敛性。

#### 我们要解决什么问题？

给定一个有向流网络 $G = (V, E)$，其中包含：
- 一个源点 $s$，
- 一个汇点 $t$，
- 以及非负的容量 $c(u, v)$，

我们想要找到一个最大流 $f(u, v)$，满足：

1. 容量约束：
   $$0 \le f(u, v) \le c(u, v)$$

2. 流量守恒：
   $$\sum_{(u,v)\in E} f(u,v) = \sum_{(v,u)\in E} f(v,u) \quad \forall u \in V \setminus {s, t}$$

3. 最大化总流量：
   $$|f| = \sum_{(s,v)\in E} f(s,v)$$

该算法通过强制按最短增广路径的顺序进行增广，改进了 Ford–Fulkerson 方法，从而限制了迭代次数。

#### 它是如何工作的？（通俗解释）

每次迭代使用 BFS 找到一条增广路径（因此路径中的每条边都是跳数最短路线的一部分）。然后我们沿着该路径推送尽可能多的流量，并更新残差容量。

逐步说明：

1. 初始化所有边的 $f(u,v)=0$。
2. 只要在残差图中存在一条从 $s$ 到 $t$ 的路径 $P$（通过 BFS 找到）：
   * 计算瓶颈容量：
     $$b = \min_{(u,v)\in P} (c(u,v) - f(u,v))$$
   * 沿着路径 $P$ 进行增广：
     $$f(u,v) \gets f(u,v) + b$$
     $$f(v,u) \gets f(v,u) - b$$
3. 重复直到没有增广路径为止。

每次增广后更新残差图，包括反向边以允许流量撤销。

#### 微型代码

Python 实现（BFS 增广）

```python
from collections import deque

def bfs(capacity, flow, s, t):
    n = len(capacity)
    parent = [-1] * n
    parent[s] = s
    q = deque([s])
    while q:
        u = q.popleft()
        for v in range(n):
            residual = capacity[u][v] - flow[u][v]
            if residual > 0 and parent[v] == -1:
                parent[v] = u
                q.append(v)
                if v == t:
                    return parent
    return None

def edmonds_karp(capacity, s, t):
    n = len(capacity)
    flow = [[0]*n for _ in range(n)]
    max_flow = 0

    while True:
        parent = bfs(capacity, flow, s, t)
        if not parent:
            break
        # 找到瓶颈容量
        v = t
        bottleneck = float('inf')
        while v != s:
            u = parent[v]
            bottleneck = min(bottleneck, capacity[u][v] - flow[u][v])
            v = u
        # 增广流量
        v = t
        while v != s:
            u = parent[v]
            flow[u][v] += bottleneck
            flow[v][u] -= bottleneck
            v = u
        max_flow += bottleneck

    return max_flow

capacity = [
    [0, 16, 13, 0, 0, 0],
    [0, 0, 10, 12, 0, 0],
    [0, 4, 0, 0, 14, 0],
    [0, 0, 9, 0, 0, 20],
    [0, 0, 0, 7, 0, 4],
    [0, 0, 0, 0, 0, 0]
]
print(edmonds_karp(capacity, 0, 5))
```

输出：

```
23
```

#### 为什么它很重要？

- 确保多项式时间终止。
- 展示了 BFS 在寻找最短增广路径方面的威力。
- 在理论和实践中都是经典的最大流算法。
- 为最大流最小割定理提供了一个清晰的证明。
- 是更高级方法（Dinic、推送-重标记）的基础。

#### 一个温和的证明（为什么它有效）

每次增广都使用一条边数意义上的最短路径。每次增广后，该路径上至少有一条边饱和（达到满容量）。

因为每条边从残差距离 $d$ 移动到 $d+2$ 的次数是有限的，所以总增广次数为 $O(VE)$。

每次 BFS 运行时间为 $O(E)$，因此总运行时间为 $O(VE^2)$。

当不存在路径时算法终止，即残差图将 $s$ 和 $t$ 断开，根据最大流最小割定理，当前流即为最大流。

#### 自己动手试试

1.  每次迭代后运行 BFS，可视化残差网络。
2.  与 Ford–Fulkerson（DFS）比较，注意增广次数更少。
3.  修改容量，观察路径选择如何变化。
4.  实现路径重建并打印增广路径。
5.  通过流转换来求解二分图匹配问题。

#### 测试用例

| 图类型         | 最大流量               | 备注                     |
| -------------- | ---------------------- | ------------------------ |
| 简单的 3 节点图 | 5                      | BFS 找到直接路径         |
| 经典的 6 节点图 | 23                     | 教科书示例               |
| 星型网络       | 各边容量之和           | 每条边都是唯一的路径     |

#### 复杂度

- 时间：$O(VE^2)$
- 空间：$O(V^2)$
- 增广次数：$O(VE)$

Edmonds–Karp 将 Ford–Fulkerson 转变为一个可预测、高效且优雅的基于 BFS 的流引擎，确保了进展，限制了迭代次数，并揭示了最优流的结构。
### 353 Dinic 算法

Dinic 算法（或称 Dinitz 算法）是一种计算最大流的更快方法，它通过引入层次图并在其中发送阻塞流，改进了 Edmonds–Karp 算法。它结合了 BFS（用于对图进行分层）和 DFS（用于寻找增广路径），实现了一个强多项式时间界。

#### 我们要解决什么问题？

给定一个有向流网络 $G = (V, E)$，其中包含：
- 源点 $s$，
- 汇点 $t$，
- 每条边上的容量 $c(u,v)$，

目标是找到最大流 $f(u,v)$，满足：

1. 容量约束：
   $$0 \le f(u, v) \le c(u, v)$$

2. 流量守恒：
   $$\sum_{(u,v)\in E} f(u,v) = \sum_{(v,u)\in E} f(v,u) \quad \forall u \in V \setminus {s, t}$$

3. 最大化总流量：
   $$|f| = \sum_{(s,v)\in E} f(s,v)$$

#### 它是如何工作的（通俗解释）？

Dinic 算法分阶段运行。每个阶段使用 BFS 构建一个层次图，然后在该分层结构内使用 DFS 尽可能多地推送流量。这种方法确保了进展，每个阶段都会严格增加残差图中从 $s$ 到 $t$ 的最短路径长度。

逐步说明：

1. 构建层次图（BFS）：
   * 从 $s$ 开始运行 BFS。
   * 为每个顶点分配一个层级，即其在残差图中与 $s$ 的距离。
   * 只使用满足 $level[v] = level[u] + 1$ 的边 $(u,v)$。

2. 发送阻塞流（DFS）：
   * 使用 DFS 沿着符合层级结构的路径从 $s$ 向 $t$ 推送流量。
   * 当无法再发送更多流量时停止（即阻塞流）。

3. 重复：
   * 重建层次图；继续直到 $t$ 无法从 $s$ 到达。

阻塞流会饱和层次图中每条 $s$–$t$ 路径上的至少一条边，从而确保每个阶段都会终止。

#### 精简代码

Python 实现（邻接表）

```python
from collections import deque

class Dinic:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v, cap):
        self.adj[u].append([v, cap, len(self.adj[v])])
        self.adj[v].append([u, 0, len(self.adj[u]) - 1])  # 反向边

    def bfs(self, s, t, level):
        q = deque([s])
        level[s] = 0
        while q:
            u = q.popleft()
            for v, cap, _ in self.adj[u]:
                if cap > 0 and level[v] < 0:
                    level[v] = level[u] + 1
                    q.append(v)
        return level[t] >= 0

    def dfs(self, u, t, flow, level, it):
        if u == t:
            return flow
        while it[u] < len(self.adj[u]):
            v, cap, rev = self.adj[u][it[u]]
            if cap > 0 and level[v] == level[u] + 1:
                pushed = self.dfs(v, t, min(flow, cap), level, it)
                if pushed > 0:
                    self.adj[u][it[u]][1] -= pushed
                    self.adj[v][rev][1] += pushed
                    return pushed
            it[u] += 1
        return 0

    def max_flow(self, s, t):
        flow = 0
        level = [-1] * self.n
        INF = float('inf')
        while self.bfs(s, t, level):
            it = [0] * self.n
            while True:
                pushed = self.dfs(s, t, INF, level, it)
                if pushed == 0:
                    break
                flow += pushed
            level = [-1] * self.n
        return flow

# 示例
dinic = Dinic(6)
edges = [
    (0,1,16),(0,2,13),(1,2,10),(2,1,4),(1,3,12),
    (3,2,9),(2,4,14),(4,3,7),(3,5,20),(4,5,4)
]
for u,v,c in edges:
    dinic.add_edge(u,v,c)
print(dinic.max_flow(0,5))
```

输出：

```
23
```

#### 为什么它很重要

- 实现了 $O(V^2E)$ 的复杂度（实践中更快）。
- 利用分层来避免冗余的增广。
- 是高级流算法的基础，如 Dinic + 容量缩放、推送-重标记和阻塞流变体。
- 常用于竞赛编程、网络路由和二分图匹配。

#### 一个温和的证明（为什么它有效）

每个阶段都会增加残差图中从 $s$ 到 $t$ 的最短距离。因为最多有 $V-1$ 个不同的距离，所以算法最多运行 $O(V)$ 个阶段。在每个阶段内，我们寻找一个阻塞流，这可以通过 $O(E)$ 次 DFS 调用来计算。

因此总运行时间：
$$
O(VE)
$$
对于单位网络（每条边容量 = 1），
以及一般情况下，
$$
O(V^2E)
$$

当不存在增广路径时，残差图将 $s$ 和 $t$ 断开，因此根据最大流最小割定理，当前流是最大的。

#### 亲自尝试

1. 比较不同阶段之间的 BFS 层级，注意深度的增加。
2. 可视化层次图和残差边。
3. 在二分图上测试，确认匹配大小等于流量。
4. 修改代码以存储每条边的流量。
5. 添加容量缩放以加速稠密图。

#### 测试用例

| 图类型          | 最大流                 | 备注           |
| --------------- | ---------------------- | --------------- |
| 6节点示例       | 23                     | 经典示例        |
| 单位网络        | 等于不相交路径数       |                 |
| 二分图          | 等于最大匹配数         |                 |

#### 复杂度

- 时间：一般情况下 $O(V^2E)$，单位容量下 $O(\sqrt{V}E)$
- 空间：$O(V + E)$
- 每阶段增广次数：$O(E)$

Dinic 算法优雅地结合了基于 BFS 的层级分层和基于 DFS 的流量推送，是结构与贪心策略的完美结合，为现代流计算提供了强大动力。
### 354 推送-重标算法

推送-重标算法（也称为预流推送算法）对最大流问题采取了完全不同的视角。它不是寻找从源点到汇点的路径，而是沿着边局部推送流量，并通过调整顶点高度（标签）来引导流量“下坡”流向汇点。

这种方法在实践中非常高效，并构成了许多现代流求解器的基础。

#### 我们正在解决什么问题？

给定一个有向网络 $G = (V, E)$，其中包含：
- 源点 $s$，
- 汇点 $t$，
- 容量 $c(u, v)$，

我们想要一个流 $f(u,v)$ 满足：

1. 容量约束：
   $$0 \le f(u, v) \le c(u, v)$$

2. 流量守恒：
   $$\sum_{(u,v)\in E} f(u,v) = \sum_{(v,u)\in E} f(v,u) \quad \forall u \neq s,t$$

3. 最大化总流量：
   $$|f| = \sum_{(s,v)\in E} f(s,v)$$

#### 它是如何工作的（通俗解释）？

与 Ford–Fulkerson 算法寻找增广路径不同，推送-重标算法维护一个预流（其中间节点可能有过剩流量），并逐步修复不平衡。

关键概念：
- **预流**：流可以暂时违反守恒定律（节点可以有盈余）。
- **高度（标签）**：一个引导流向的整数，流只能“下坡”移动。
- **推送**：如果可能，从 $u$ 向 $v$ 发送流量。
- **重标**：当 $u$ 被“卡住”时，增加其高度以便流量可以继续流动。

逐步流程：
1. 初始化所有 $f(u,v)=0$。
2. 设置 $h(s)=|V|$，并尽可能多地从 $s$ 向其邻居推送流量。
3. 当任何顶点（除了 $s, t$）有过剩流量时，执行：
   * **推送**：如果 $(u,v)$ 是允许的（$h(u)=h(v)+1$ 且剩余容量 $>0$），则发送
     $$\Delta = \min(e(u), c(u,v)-f(u,v))$$
   * **重标**：如果没有允许的边，则设置
     $$h(u) = 1 + \min_{(u,v): c(u,v)-f(u,v)>0} h(v)$$

重复此过程，直到所有过剩流量都在 $t$ 或 $s$ 处。

#### 精简代码

Python 实现（简化版）

```python
def push_relabel(capacity, s, t):
    n = len(capacity)
    flow = [[0]*n for _ in range(n)]
    excess = [0]*n
    height = [0]*n
    height[s] = n

    def push(u, v):
        delta = min(excess[u], capacity[u][v] - flow[u][v])
        flow[u][v] += delta
        flow[v][u] -= delta
        excess[u] -= delta
        excess[v] += delta

    def relabel(u):
        min_h = float('inf')
        for v in range(n):
            if capacity[u][v] - flow[u][v] > 0:
                min_h = min(min_h, height[v])
        if min_h < float('inf'):
            height[u] = min_h + 1

    def discharge(u):
        while excess[u] > 0:
            for v in range(n):
                if capacity[u][v] - flow[u][v] > 0 and height[u] == height[v] + 1:
                    push(u, v)
                    if excess[u] == 0:
                        break
            else:
                relabel(u)

    # 初始化预流
    for v in range(n):
        if capacity[s][v] > 0:
            flow[s][v] = capacity[s][v]
            flow[v][s] = -capacity[s][v]
            excess[v] = capacity[s][v]
    excess[s] = sum(capacity[s])

    # 释放活跃顶点
    active = [i for i in range(n) if i != s and i != t]
    p = 0
    while p < len(active):
        u = active[p]
        old_height = height[u]
        discharge(u)
        if height[u] > old_height:
            active.insert(0, active.pop(p))  # 移到前端
            p = 0
        else:
            p += 1

    return sum(flow[s])

capacity = [
    [0,16,13,0,0,0],
    [0,0,10,12,0,0],
    [0,4,0,0,14,0],
    [0,0,9,0,0,20],
    [0,0,0,7,0,4],
    [0,0,0,0,0,0]
]
print(push_relabel(capacity, 0, 5))
```

输出：

```
23
```

#### 为什么它很重要？

- **局部视角**：不需要全局增广路径。
- **高度可并行化**。
- **在稠密图中表现非常好**。
- **是最高标签和 FIFO 变体的基础**。
- **概念上优雅**：流量在高度引导下“下坡”流动。

#### 一个温和的证明（为什么它有效）

1. 高度不变性确保没有流量从较低的顶点流向较高的顶点。
2. 每次推送都遵守容量和非负性约束。
3. 高度总是增加，确保了算法终止。
4. 当没有顶点（除了 $s$ 和 $t$）有过剩流量时，所有预流约束都得到满足，预流就变成了一个有效的最大流。

根据最大流最小割定理，最终预流的值等于最小割的容量。

#### 亲自尝试

1. 跟踪每一步后每个顶点的高度和过剩流量。
2. 比较 FIFO 与最高标签变体。
3. 使用小型网络可视化流量的移动。
4. 与基于路径的算法（Ford–Fulkerson）进行对比。
5. 添加日志记录以观察重标事件。

#### 测试用例

| 图类型         | 最大流             | 备注       |
| -------------- | ------------------ | ---------- |
| 经典 6 节点图  | 23                 | 教科书示例 |
| 稠密完全图     | 高流量             | 高效       |
| 稀疏路径图     | 与 Edmonds–Karp 相同 | 类似       |

#### 复杂度

- **时间（通用）**：$O(V^2E)$
- **使用 FIFO / 最高标签启发式**：$O(V^3)$ 或更好
- **空间**：$O(V^2)$

推送-重标算法将最大流问题转化为一个局部平衡过程，通过推送、重标和均衡压力，直到达到平衡。
### 355 容量缩放

容量缩放算法是 Ford–Fulkerson 算法的一个改进版本，旨在高效处理大容量的边。它不进行任意的增广，而是首先关注高容量边，随着缩放参数的减小逐步优化流。

这种方法通过早期关注"大管道"，之后再处理较小的管道，从而减少了增广的次数。

#### 我们要解决什么问题？

给定一个有向流网络 $G = (V, E)$，其中包含：

- 源点 $s$，
- 汇点 $t$，
- 非负容量 $c(u, v)$，

我们想要计算一个最大流 $f(u, v)$，满足：

1. 容量约束：
   $$0 \le f(u, v) \le c(u, v)$$
2. 流量守恒：
   $$\sum_{(u,v)\in E} f(u,v) = \sum_{(v,u)\in E} f(v,u), \quad \forall u \neq s,t$$
3. 最大化：
   $$|f| = \sum_{(s,v)\in E} f(s,v)$$

当容量很大时，标准的增广路径方法可能需要很多次迭代。容量缩放通过按容量大小对边进行分组来减少迭代次数。

#### 它是如何工作的（通俗解释）？

我们引入一个缩放参数 $\Delta$，从低于最大容量的最大 2 的幂次开始。在增广过程中，我们只考虑剩余容量 ≥ Δ 的边。一旦不存在这样的路径，就将 $\Delta$ 减半并继续。

这确保了我们先推送大的流量，稍后再进行优化。

逐步说明：

1. 初始化 $f(u,v) = 0$。
2. 令
   $$\Delta = 2^{\lfloor \log_2 C_{\max} \rfloor}$$
   其中 $C_{\max} = \max_{(u,v)\in E} c(u,v)$
3. 当 $\Delta \ge 1$ 时：

   * 构建 Δ-残差图：包含剩余容量 $\ge \Delta$ 的边
   * 当在此图中存在增广路径时：

     * 沿该路径推送等于瓶颈容量的流量
   * 更新 $\Delta \gets \Delta / 2$
4. 返回总流量。

#### 微型代码

Python（带缩放的 DFS 增广）

```python
def dfs(u, t, f, visited, capacity, flow, delta):
    if u == t:
        return f
    visited[u] = True
    for v in range(len(capacity)):
        residual = capacity[u][v] - flow[u][v]
        if residual >= delta and not visited[v]:
            pushed = dfs(v, t, min(f, residual), visited, capacity, flow, delta)
            if pushed > 0:
                flow[u][v] += pushed
                flow[v][u] -= pushed
                return pushed
    return 0

def capacity_scaling(capacity, s, t):
    n = len(capacity)
    flow = [[0]*n for _ in range(n)]
    Cmax = max(max(row) for row in capacity)
    delta = 1
    while delta * 2 <= Cmax:
        delta *= 2

    max_flow = 0
    while delta >= 1:
        while True:
            visited = [False]*n
            pushed = dfs(s, t, float('inf'), visited, capacity, flow, delta)
            if pushed == 0:
                break
            max_flow += pushed
        delta //= 2
    return max_flow

capacity = [
    [0, 16, 13, 0, 0, 0],
    [0, 0, 10, 12, 0, 0],
    [0, 4, 0, 0, 14, 0],
    [0, 0, 9, 0, 0, 20],
    [0, 0, 0, 7, 0, 4],
    [0, 0, 0, 0, 0, 0]
]
print(capacity_scaling(capacity, 0, 5))
```

输出：

```
23
```

#### 为什么它很重要

- 与朴素的 Ford–Fulkerson 算法相比，减少了增广次数。
- 首先关注大的推送，改善了收敛性。
- 展示了缩放思想的威力，这是算法设计中反复出现的优化技巧。
- 是通往最小费用流中费用缩放和容量缩放的垫脚石。

#### 一个温和的证明（为什么它有效）

在每个缩放阶段 $\Delta$，

- 每条增广路径至少增加 $\Delta$ 的流量。
- 总流量值最多为 $|f^*|$。
- 因此，每个阶段最多执行 $O(|f^*| / \Delta)$ 次增广。

由于 $\Delta$ 在每个阶段减半，
增广的总次数为
$$
O(E \log C_{\max})
$$

每次路径搜索需要 $O(E)$，所以
$$
T = O(E^2 \log C_{\max})
$$

当 $\Delta = 1$ 且没有剩余路径时算法终止，这意味着达到了最大流。

#### 亲自尝试

1. 与 Ford–Fulkerson 算法比较增广顺序。
2. 跟踪每个阶段的 $\Delta$ 值和残差图。
3. 观察大容量情况下的更快收敛。
4. 与 BFS 结合以模拟 Edmonds–Karp 结构。
5. 可视化每个缩放级别下的流量累积。

#### 测试用例

| 图                 | 最大流量                | 备注             |
| ------------------ | ----------------------- | ---------------- |
| 经典的 6 节点图    | 23                      | 教科书示例       |
| 大容量边           | 流量相同，步骤更少      | 缩放有帮助       |
| 单位容量           | 行为类似 Edmonds–Karp   | 收益较小         |

#### 复杂度

- 时间：$O(E^2 \log C_{\max})$
- 空间：$O(V^2)$
- 增广次数：$O(E \log C_{\max})$

容量缩放体现了一个简单而强大的思想：先粗后精，先推大流，最后完善。
### 356 费用缩放

费用缩放算法通过应用一种缩放技术来解决最小费用最大流（MCMF）问题，这种技术不是针对容量，而是针对边费用。它逐步细化约简费用的精度，在整个过程中保持 ε-最优性，并收敛到真正的最优流。

这种方法在理论上优雅，在实践中高效，构成了高性能网络优化求解器的基础。

#### 我们要解决什么问题？

给定一个有向网络 $G = (V, E)$，其中包含：

- 容量 $c(u, v) \ge 0$，
- （单位流量的）费用 $w(u, v)$，
- 源点 $s$ 和汇点 $t$，

我们希望以最小的总费用从 $s$ 向 $t$ 发送最大流。

我们最小化：
$$
\text{Cost}(f) = \sum_{(u,v)\in E} f(u,v), w(u,v)
$$

约束条件为：

- 容量约束：$0 \le f(u,v) \le c(u,v)$
- 流量守恒：对于所有 $u \neq s,t$，满足 $\sum_v f(u,v) = \sum_v f(v,u)$

#### 它是如何工作的（通俗解释）？

费用缩放使用约简费用的连续细化，确保每个阶段的流都是 ε-最优的。

一个 ε-最优流满足：
对于所有残差边 $(u,v)$，有
$$
c_p(u,v) = w(u,v) + \pi(u) - \pi(v) \ge -\varepsilon
$$
其中 $\pi$ 是一个势函数。

算法以一个较大的 $\varepsilon$（通常是 $C_{\max}$ 或 $W_{\max}$）开始，并以几何方式减小它（例如 $\varepsilon \gets \varepsilon / 2$）。在每个阶段，只允许沿着可行边（$c_p(u,v) < 0$）进行推送，并保持 ε-最优性。

逐步过程：

1. 初始化预流（尽可能多地从 $s$ 推送）。
2. 设置 $\varepsilon = C_{\max}$。
3. 当 $\varepsilon \ge 1$ 时：

   * 保持 ε-最优性：调整流和势。
   * 对于每个有盈余的顶点，

     * 沿着可行边（$c_p(u,v) < 0$）推送流。
     * 如果卡住，则通过增加 $\pi(u)$（降低约简费用）进行重标记。
   * 将 $\varepsilon$ 减半。
4. 当 $\varepsilon < 1$ 时，解是最优的。

#### 微型代码（简化框架）

以下是一个概念性概述（非完整实现），旨在提供教学上的清晰性：

```python
def cost_scaling_mcmf(V, edges, s, t):
    INF = float('inf')
    adj = [[] for _ in range(V)]
    for u, v, cap, cost in edges:
        adj[u].append([v, cap, cost, len(adj[v])])
        adj[v].append([u, 0, -cost, len(adj[u]) - 1])

    pi = [0]*V  # 势
    excess = [0]*V
    excess[s] = INF

    def reduced_cost(u, v, cost):
        return cost + pi[u] - pi[v]

    epsilon = max(abs(c) for _,_,_,c in edges)
    while epsilon >= 1:
        active = [i for i in range(V) if excess[i] > 0 and i != s and i != t]
        while active:
            u = active.pop()
            for e in adj[u]:
                v, cap, cost, rev = e
                rc = reduced_cost(u, v, cost)
                if cap > 0 and rc < 0:
                    delta = min(cap, excess[u])
                    e[1] -= delta
                    adj[v][rev][1] += delta
                    excess[u] -= delta
                    excess[v] += delta
                    if v not in (s, t) and excess[v] == delta:
                        active.append(v)
            if excess[u] > 0:
                pi[u] += epsilon
        epsilon //= 2

    total_cost = 0
    for u in range(V):
        for v, cap, cost, rev in adj[u]:
            if cost > 0:
                total_cost += cost * (adj[v][rev][1])
    return total_cost
```

#### 为什么它很重要

- 避免了每一步昂贵的短路径搜索。
- 利用 ε-最优性确保强多项式界。
- 适用于稠密图和大整数费用。
- 构成了网络单纯形法和可扩展 MCMF 求解器的基础。

#### 一个温和的证明（为什么它有效）

1. ε-最优性：确保每个缩放阶段都近似最优。
2. 缩放：以几何方式减小 ε，收敛到精确最优性。
3. 终止：当 $\varepsilon < 1$ 时，所有约简费用非负，流是最优的。

每次推送都尊重容量和可行性；每次重标记都会减小 ε，保证进展。

#### 亲自尝试

1. 与连续最短路径算法比较，注意更少的路径搜索。
2. 跟踪势 $\pi(u)$ 在各个阶段如何演变。
3. 可视化可行边的 ε 层。
4. 尝试不同的费用大小，较大的费用更能从缩放中受益。
5. 观察 ε 每阶段减半时的收敛情况。

#### 测试用例

| 图类型           | 最大流 | 最小费用  | 备注         |
| --------------- | -------- | --------- | ------------- |
| 简单的 3 节点图   | 5        | 10        | 小例子        |
| 经典网络         | 23       | 42        | 费用缩放      |
| 稠密图           | 变化     | 高效      |               |

#### 复杂度

- 时间：$O(E \log C_{\max} (V + E))$（取决于费用范围）
- 空间：$O(V + E)$
- 阶段数：$O(\log C_{\max})$

费用缩放展示了精度细化的范式：从粗略开始，以精确结束。这是一门结合缩放、势函数和局部可行性以实现全局最优流的精品课程。
### 357 最小费用最大流（Bellman-Ford）

基于 Bellman-Ford 的最小费用最大流（MCMF）算法是解决同时涉及容量和费用约束的流问题的基石方法。它反复沿着最短费用路径增广流，确保每一单位的流都以尽可能低的成本流动。

此版本使用 Bellman-Ford 来处理负边费用，即使在存在降低总费用的环时也能保证正确性。

#### 我们要解决什么问题？

给定一个有向图 $G = (V, E)$，其中包含：

- 容量 $c(u, v)$
- 单位流费用 $w(u, v)$
- 源点 $s$ 和汇点 $t$

寻找一个流 $f(u, v)$，使得：

1. 满足容量限制：$0 \le f(u, v) \le c(u, v)$
2. 流量守恒：对所有 $u \neq s,t$，有 $\sum_v f(u, v) = \sum_v f(v, u)$
3. 最大化总流量，同时
4. 最小化总费用：
   $$
   \text{Cost}(f) = \sum_{(u,v)\in E} f(u,v),w(u,v)
   $$

#### 它是如何工作的（通俗解释）？

该算法迭代地使用基于费用（而非距离）的最短路径搜索，寻找从 $s$ 到 $t$ 的增广路径。

每次迭代：

1. 在残差图上运行 Bellman-Ford 以找到最短费用路径。
2. 确定该路径上的瓶颈容量 $\delta$。
3. 沿着该路径增广 $\delta$ 单位的流。
4. 更新残差容量和反向边。
5. 重复直到不存在更多的增广路径。

因为 Bellman-Ford 支持负费用，所以该算法能够正确处理那些通过环路可以降低费用的图。

#### 微型代码（类 C 伪代码）

```c
struct Edge { int v, cap, cost, rev; };
vector<Edge> adj[V];
int dist[V], parent[V], parent_edge[V];

bool bellman_ford(int s, int t, int V) {
    fill(dist, dist+V, INF);
    dist[s] = 0;
    bool updated = true;
    for (int i = 0; i < V-1 && updated; i++) {
        updated = false;
        for (int u = 0; u < V; u++) {
            if (dist[u] == INF) continue;
            for (int k = 0; k < adj[u].size(); k++) {
                auto &e = adj[u][k];
                if (e.cap > 0 && dist[e.v] > dist[u] + e.cost) {
                    dist[e.v] = dist[u] + e.cost;
                    parent[e.v] = u;
                    parent_edge[e.v] = k;
                    updated = true;
                }
            }
        }
    }
    return dist[t] < INF;
}

pair<int,int> min_cost_max_flow(int s, int t, int V) {
    int flow = 0, cost = 0;
    while (bellman_ford(s, t, V)) {
        int f = INF;
        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            auto &e = adj[u][parent_edge[v]];
            f = min(f, e.cap);
        }
        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            auto &e = adj[u][parent_edge[v]];
            e.cap -= f;
            adj[v][e.rev].cap += f;
            cost += f * e.cost;
        }
        flow += f;
    }
    return {flow, cost};
}
```

#### 为什么它很重要？

- 安全地处理负边费用。
- 如果不存在负环，则保证最优性。
- 适用于中小型图。
- 是更高效变体（例如 SPFA、带势函数的 Dijkstra）的基础。

它是用于教学最小费用流问题的首选实现，在清晰度和正确性之间取得了平衡。

#### 一个温和的证明（为什么它有效）

每次迭代都使用 Bellman-Ford 找到一条最短费用增广路径。
因为边费用在增广过程中是非递减的，所以算法会收敛到全局最优流。

每次增广：

- 增加流量。
- 不会在后续引入更便宜的路径（单调性）。
- 当不存在更多增广路径时终止。

因此，得到的流既是最大的，也是费用最小的。

#### 自己动手试试

1.  画一个包含 4 个节点和一条负边的简单图。
2.  追踪每次增广后的残差图更新。
3.  与朴素的贪婪路径选择结果进行比较。
4.  将 Bellman-Ford 替换为 Dijkstra + 势函数以提高速度。
5.  可视化残差容量的演变过程。

#### 测试用例

| 图结构                       | 最大流 | 最小费用 | 备注                     |
| --------------------------- | -------- | -------- | ---------------------- |
| 链式 $s \to a \to b \to t$ | 5        | 10       | 简单路径                 |
| 包含负环的图                 | 无效     | -        | 必须检测到               |
| 网格状网络                   | 12       | 24       | 多次增广                 |

#### 复杂度

- 时间复杂度：$O(F \cdot V \cdot E)$，其中 $F$ 是发送的总流量。
- 空间复杂度：$O(V + E)$

对于小型图或包含负权边的图，Bellman-Ford MCMF 是最稳健、最直接的方法，清晰、可靠且具有基础性。
### 358 最小费用最大流（SPFA）

使用 SPFA（Shortest Path Faster Algorithm，最短路径更快算法）的最小费用最大流算法是对 Bellman–Ford 方法的优化。
它利用基于队列的松弛方法，在实践中更高效地找到成本最短的增广路径，尤其是在稀疏图上或负边罕见的情况下。

#### 我们要解决什么问题？

我们正在解决最小费用最大流问题：
在一个有向图 $G = (V, E)$ 中找到一个流 $f$，该图具有：

- 容量 $c(u, v) \ge 0$
- 单位流量的成本 $w(u, v)$
- 源点 $s$，汇点 $t$

约束条件：

1. 容量约束：$0 \le f(u, v) \le c(u, v)$
2. 流量守恒：对于所有 $u \neq s,t$，满足 $\sum_v f(u,v) = \sum_v f(v,u)$
3. 目标：
   $$
   \min \text{Cost}(f) = \sum_{(u,v)\in E} f(u,v),w(u,v)
   $$

我们寻求从 $s$ 到 $t$ 的一个最大流，且该流的总成本最小。

#### 它是如何工作的（通俗解释）？

SPFA 通过使用队列仅松弛那些仍能改进距离的顶点，比 Bellman–Ford 更高效地找到成本最短的路径。

在每次迭代中：

1. 在残差图上运行 SPFA 以找到成本最短的路径。
2. 计算该路径上的瓶颈流量 $\delta$。
3. 沿该路径增广 $\delta$ 单位的流量。
4. 更新残差容量和反向边。
5. 重复直到没有增广路径剩余。

SPFA 动态管理队列中的节点，使其平均速度比完整的 Bellman–Ford 更快（尽管最坏情况仍然相似）。

#### 微型代码（类 C 伪代码）

```c
struct Edge { int v, cap, cost, rev; };
vector<Edge> adj[V];
int dist[V], parent[V], parent_edge[V];
bool in_queue[V];

bool spfa(int s, int t, int V) {
    fill(dist, dist+V, INF);
    fill(in_queue, in_queue+V, false);
    queue<int> q;
    dist[s] = 0;
    q.push(s);
    in_queue[s] = true;

    while (!q.empty()) {
        int u = q.front(); q.pop();
        in_queue[u] = false;
        for (int k = 0; k < adj[u].size(); k++) {
            auto &e = adj[u][k];
            if (e.cap > 0 && dist[e.v] > dist[u] + e.cost) {
                dist[e.v] = dist[u] + e.cost;
                parent[e.v] = u;
                parent_edge[e.v] = k;
                if (!in_queue[e.v]) {
                    q.push(e.v);
                    in_queue[e.v] = true;
                }
            }
        }
    }
    return dist[t] < INF;
}

pair<int,int> min_cost_max_flow(int s, int t, int V) {
    int flow = 0, cost = 0;
    while (spfa(s, t, V)) {
        int f = INF;
        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            auto &e = adj[u][parent_edge[v]];
            f = min(f, e.cap);
        }
        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            auto &e = adj[u][parent_edge[v]];
            e.cap -= f;
            adj[v][e.rev].cap += f;
            cost += f * e.cost;
        }
        flow += f;
    }
    return {flow, cost};
}
```

#### 为什么它重要

- 相较于 Bellman–Ford 有实际的加速效果，尤其是在稀疏网络中。
- 安全地处理负边（无负环）。
- 广泛用于竞赛编程和网络优化。
- 比基于 Dijkstra 的变体更容易实现。

SPFA 结合了 Bellman–Ford 的简单性和实际效率，通常能显著减少冗余的松弛操作。

#### 一个温和的证明（为什么它有效）

每次 SPFA 运行都计算一条成本最短的增广路径。
沿该路径增广确保了：

- 流保持可行，
- 成本严格递减，并且
- 过程在经过有限次增广后终止（受总流量限制）。

因为 SPFA 总是遵循最短路径距离，所以最终流是成本最优的。

#### 亲自尝试

1. 在你的 MCMF 中用 SPFA 替换 Bellman–Ford。
2. 比较在稀疏图与稠密图上的运行时间。
3. 添加一条具有负成本的边并验证其行为是否正确。
4. 可视化每次松弛后的队列内容。
5. 测量每个顶点被处理的次数。

#### 测试用例

| 图类型       | 最大流 | 最小成本 | 备注                   |
| ----------- | -------- | -------- | ----------------------- |
| 链状图 | 4        | 12       | 简单                  |
| 稀疏 DAG  | 10       | 25       | SPFA 高效          |
| 稠密图 | 15       | 40       | 与 Bellman–Ford 类似 |

#### 复杂度

- 时间复杂度：平均 $O(F \cdot E)$，最坏情况 $O(F \cdot V \cdot E)$
- 空间复杂度：$O(V + E)$

基于 SPFA 的 MCMF 融合了清晰性、实际效率和对负成本的支持，使其成为现实世界最小费用流实现中的热门选择。
### 359 带需求的循环流

带需求的循环流问题是经典流问题与最小费用流问题的一个推广。
它不再局限于单一的源点和汇点，而是允许每个节点可以需求或供应一定量的流。
你的目标是找到一个可行的循环流——一个满足所有节点需求和容量限制的流。

这个模型将网络流、可行性检查以及优化统一在一个优雅的框架之下。

#### 我们要解决什么问题？

给定一个有向图 $G = (V, E)$，每条边 $(u,v)$ 具有：

- 下界 $l(u,v)$
- 上界（容量）$c(u,v)$
- 费用 $w(u,v)$

每个顶点 $v$ 可能有一个需求 $b(v)$（正值表示需要流入，负值表示提供流出）。

我们希望找到一个循环流（一个对所有边定义了流量 $f(u,v)$ 的流），使得：

1. 容量约束：
   $$
   l(u,v) \le f(u,v) \le c(u,v)
   $$
2. 流量守恒：
   $$
   \sum_{(u,v)\in E} f(u,v) - \sum_{(v,u)\in E} f(v,u) = b(v)
   $$
3. 可选的费用最小化：
   $$
   \min \sum_{(u,v)\in E} f(u,v),w(u,v)
   $$

当所有需求能同时被满足时，就存在一个可行的循环流。

#### 它是如何工作的（通俗解释）？

关键技巧是将其转化为一个带有超级源点和超级汇点的标准最小费用流问题。

逐步转化过程：

1. 对于每条有下界 $l(u,v)$ 的边 $(u,v)$：

   * 将容量减少为 $(c(u,v) - l(u,v))$。
   * 从需求中减去 $l(u,v)$：
     $$
     b(u) \mathrel{{-}{=}} l(u,v), \quad b(v) \mathrel{{+}{=}} l(u,v)
     $$

2. 添加一个超级源点 $S$ 和超级汇点 $T$：

   * 对于每个节点 $v$：

     * 如果 $b(v) > 0$，添加边 $(S,v)$，容量为 $b(v)$。
     * 如果 $b(v) < 0$，添加边 $(v,T)$，容量为 $-b(v)$。

3. 求解从 $S$ 到 $T$ 的最大流（或最小费用流）。

4. 如果最大流等于总需求，则存在可行的循环流。
   否则，没有循环流能满足所有约束。

#### 简化代码示例

```python
def circulation_with_demands(V, edges, demand):
    # edges: (u, v, lower, cap, cost)
    # demand: 节点需求 b(v) 的列表

    adj = [[] for _ in range(V + 2)]
    S, T = V, V + 1
    b = demand[:]

    for u, v, low, cap, cost in edges:
        cap -= low
        b[u] -= low
        b[v] += low
        adj[u].append((v, cap, cost))
        adj[v].append((u, 0, -cost))

    total_demand = 0
    for i in range(V):
        if b[i] > 0:
            adj[S].append((i, b[i], 0))
            total_demand += b[i]
        elif b[i] < 0:
            adj[i].append((T, -b[i], 0))

    flow, cost = min_cost_max_flow(adj, S, T)
    if flow == total_demand:
        print("找到可行的循环流")
    else:
        print("没有可行的循环流")
```

#### 为什么它很重要？

- 统一了多个问题：
  许多流问题——如分配、运输和调度——都可以表达为带需求的循环流。

- 优雅地处理下界：
  标准流无法直接强制 $f(u,v) \ge l(u,v)$。循环流解决了这个问题。

- 高级模型的基础：
  费用约束、多商品流和平衡方程都建立在此基础之上。

#### 一个温和的证明（为什么它有效）

通过强制执行下界，我们调整了每个顶点的净平衡。
经过标准化后，系统等价于寻找一个从超级源点到超级汇点的流，以满足所有的平衡。

如果这样的流存在，我们就可以重建原始流：
$$
f'(u,v) = f(u,v) + l(u,v)
$$
它满足所有原始约束。

#### 亲自尝试

1.  为一个供需网络建模（例如，工厂 → 仓库 → 商店）。
2.  添加下界以强制执行最低交付量。
3.  引入节点需求以平衡供应/消耗。
4.  使用最小费用流求解并验证循环流。
5.  去掉费用部分，仅检查可行性。

#### 测试用例

| 网络类型                         | 结果       | 备注                 |
| ------------------------------- | ---------- | -------------------- |
| 平衡的 3 节点网络               | 可行       | 简单的供需关系       |
| 下界超过总量                    | 不可行     | 无解                 |
| 混合正负需求                    | 可行       | 需要调整             |

#### 复杂度

- 时间复杂度：取决于使用的流求解器（对于 Bellman–Ford 变体为 $O(E^2V)$）
- 空间复杂度：$O(V + E)$

带需求的循环流提供了一个通用框架：任何线性流约束都可以被编码、检查和优化——将平衡方程转化为可解的图问题。
### 360 连续最短路径

连续最短路径（SSP）算法是一种简洁直观的解决最小费用最大流（MCMF）问题的方法。
它逐步构建最优流，每次沿着一条最短路径，总是沿着当前最便宜的可用路径发送流量，直到容量或平衡约束阻止它为止。

#### 我们要解决什么问题？

给定一个有向图 $G = (V, E)$，其中包含：

- 容量：$c(u,v)$
- 单位流量费用：$w(u,v)$
- 源点：$s$ 和汇点：$t$

我们希望从 $s$ 到 $t$ 发送最大流量，且总费用最小：

$$
\min \sum_{(u,v)\in E} f(u,v), w(u,v)
$$

约束条件：

- $0 \le f(u,v) \le c(u,v)$
- 除 $s, t$ 外，每个顶点满足流量守恒

#### 它是如何工作的（通俗解释）？

SSP 通过在残差图中（以费用作为权重）反复寻找从 $s$ 到 $t$ 的最短费用增广路径来推进。
然后沿着该路径推送尽可能多的流量。
残差边跟踪可用容量（正向和反向）。

算法步骤：

1. 初始化所有边的流量 $f(u,v) = 0$。
2. 构建带有边费用 $w(u,v)$ 的残差图。
3. 当存在从 $s$ 到 $t$ 的路径 $P$ 时：

   * 按费用（使用 Dijkstra 或 Bellman–Ford）找到最短路径 $P$。
   * 计算瓶颈容量 $\delta = \min_{(u,v)\in P} c_f(u,v)$。
   * 沿 $P$ 增广流量：
     $$
     f(u,v) \mathrel{{+}{=}} \delta,\quad f(v,u) \mathrel{{-}{=}} \delta
     $$
   * 更新残差容量和费用。
4. 重复直到没有增广路径为止。
5. 最终得到的 $f$ 就是最小费用最大流。

如果存在负权边，使用 Bellman–Ford；否则为了效率，使用带势函数的 Dijkstra。

#### 简化代码（简化版 Python）

```python
from heapq import heappush, heappop

def successive_shortest_path(V, edges, s, t):
    adj = [[] for _ in range(V)]
    for u, v, cap, cost in edges:
        adj[u].append([v, cap, cost, len(adj[v])])
        adj[v].append([u, 0, -cost, len(adj[u]) - 1])

    INF = 109
    pi = [0]*V  # 用于计算约简费用的势函数
    flow = cost = 0

    while True:
        dist = [INF]*V
        parent = [-1]*V
        parent_edge = [-1]*V
        dist[s] = 0
        pq = [(0, s)]
        while pq:
            d, u = heappop(pq)
            if d > dist[u]: continue
            for i, (v, cap, w, rev) in enumerate(adj[u]):
                if cap > 0 and dist[v] > dist[u] + w + pi[u] - pi[v]:
                    dist[v] = dist[u] + w + pi[u] - pi[v]
                    parent[v] = u
                    parent_edge[v] = i
                    heappush(pq, (dist[v], v))
        if dist[t] == INF:
            break
        for v in range(V):
            if dist[v] < INF:
                pi[v] += dist[v]
        f = INF
        v = t
        while v != s:
            u = parent[v]
            e = adj[u][parent_edge[v]]
            f = min(f, e[1])
            v = u
        v = t
        while v != s:
            u = parent[v]
            i = parent_edge[v]
            e = adj[u][i]
            e[1] -= f
            adj[v][e[3]][1] += f
            cost += f * e[2]
            v = u
        flow += f
    return flow, cost
```

#### 为什么它很重要

- 逻辑简单清晰，每次都沿着最便宜的路径增广。
- 当不存在负环时，是解决最小费用最大流问题的最优解。
- 通过约简费用和势函数，可以高效处理大型图。
- 构成了费用缩放法和网络单纯形法的基础。

#### 一个温和的证明（为什么它有效）

每次增广都沿着一条最短路径（最小约简费用）移动流量。
约简费用确保不会出现负环，因此每次增广都保持最优性。
一旦没有增广路径，所有约简费用都为非负，$f$ 就是最优的。

势函数 $\pi(v)$ 保证 Dijkstra 在转换后的图中找到真正的最短路径：
$$
w'(u,v) = w(u,v) + \pi(u) - \pi(v)
$$
保持了等价性和非负性。

#### 亲自尝试

1. 使用一个简单的 4 节点网络，追踪每条路径的费用和流量。
2. 比较使用 Bellman–Ford 和 Dijkstra 进行增广的区别。
3. 添加负权边费用，并使用势函数调整进行测试。
4. 可视化每次迭代后的残差图。
5. 测量每次迭代的费用收敛情况。

#### 测试用例

| 图结构              | 最大流 | 最小费用 | 备注                 |
| ------------------ | -------- | -------- | -------------------- |
| 简单的 3 节点图     | 5        | 10       | 一条路径             |
| 两条平行路径       | 10       | 15       | 选择最便宜的路径     |
| 带有负权边         | 7        | 5        | 势函数修正费用       |

#### 复杂度

- 时间复杂度：使用 Dijkstra + 势函数时为 $O(F \cdot E \log V)$
- 空间复杂度：$O(V + E)$

连续最短路径算法是解决带费用流问题的得力工具：易于编码，可证明正确，并且使用合适的数据结构时非常高效。

## 第 37 节 割
### 361 Stoer–Wagner 最小割

Stoer–Wagner 最小割算法用于在无向加权图中寻找全局最小割。  
一个*割*是将顶点划分为两个非空集合；其权重是跨越该划分的边的权重之和。  
该算法通过重复进行最大邻接搜索，高效地找到总边权最小的割。

#### 我们要解决什么问题？

给定一个无向图 $G = (V, E)$，其边具有非负权重 $w(u,v)$，找到一个割 $(S, V \setminus S)$，使得：

$$
\text{cut}(S) = \sum_{u \in S, v \in V \setminus S} w(u,v)
$$

在所有非平凡划分 $S \subset V$ 中达到最小。

这被称为全局最小割，区别于固定两个端点的 s–t 最小割。

#### 它是如何工作的（通俗解释）？

该算法迭代地合并顶点，同时跟踪在此过程中找到的最紧的割。

在每一阶段：

1.  任意选择一个起始顶点。
2.  通过重复添加与当前集合 $A$ 连接最强的顶点（到 $A$ 的权重总和最高）来扩展集合 $A$。
3.  继续直到所有顶点都被添加。
4.  最后添加的顶点 $t$ 和倒数第二个添加的顶点 $s$ 定义了一个割 $(A \setminus {t}, {t})$。
5.  记录该割的权重，它是一个候选的最小割。
6.  将 $s$ 和 $t$ 合并为一个顶点。
7.  重复直到只剩下一个顶点。

所有阶段中记录的最小割就是全局最小割。

#### 分步示例

假设我们有 4 个顶点 $A, B, C, D$。  
每个阶段：

1.  从 $A$ 开始。
2.  添加与 $A$ 连接最强的顶点。
3.  继续直到只剩下一个顶点。
4.  每次添加最后一个顶点时记录割的权重。
5.  合并最后两个顶点，重复。

在所有合并之后，最轻的割权重就是最小割值。

#### 微型代码（Python）

```python
def stoer_wagner_min_cut(V, weight):
    n = V
    best = float('inf')
    vertices = list(range(n))

    while n > 1:
        used = [False] * n
        weights = [0] * n
        prev = -1
        for i in range(n):
            # 选择尚未在 A 中且连接最强的顶点
            sel = -1
            for j in range(n):
                if not used[j] and (sel == -1 or weights[j] > weights[sel]):
                    sel = j
            used[sel] = True
            if i == n - 1:
                # 最后一个顶点被添加，记录割
                best = min(best, weights[sel])
                # 合并 prev 和 sel
                if prev != -1:
                    for j in range(n):
                        weight[prev][j] += weight[sel][j]
                        weight[j][prev] += weight[j][sel]
                    vertices.pop(sel)
                    weight.pop(sel)
                    for row in weight:
                        row.pop(sel)
                n -= 1
                break
            prev = sel
            for j in range(n):
                if not used[j]:
                    weights[j] += weight[sel][j]
    return best
```

#### 为什么它很重要？

-   无需固定 $s,t$ 即可找到全局最小割。
-   直接在加权无向图上工作。
-   不需要流计算。
-   对于全局割问题，比 $O(VE \log V)$ 的最大流最小割算法更简单、更快。

它是为数不多的能在加权图中精确求解全局最小割的多项式时间算法之一。

#### 一个温和的证明（为什么它有效）

每个阶段都找到一个最小 $s$–$t$ 割，将最后一个顶点 $t$ 与其余部分分开。  
合并过程保持了等价性，合并不会破坏剩余割的最优性。  
最轻的阶段割对应于全局最小割。

通过对合并步骤进行归纳，该算法探索了所有本质的割。

#### 自己试试

1.  创建一个具有不同边权重的三角形图。
2.  追踪 $A$–$B$–$C$ 顺序的添加过程并记录割权重。
3.  合并最后两个顶点，缩减矩阵，重复。
4.  验证割是否对应于最小的跨越权重。
5.  与最大流最小割进行比较以验证。

#### 测试用例

| 图                     | 最小割 | 备注                     |
| ---------------------- | ------ | ------------------------ |
| 三角形 $w=1,2,3$       | 3      | 移除权重为 3 的边        |
| 正方形（等权重）       | 2      | 任意两条对边             |
| 加权完全图             | 最小边权和 | 密集测试               |

#### 复杂度

-   时间复杂度：使用邻接矩阵为 $O(V^3)$
-   空间复杂度：$O(V^2)$

存在使用邻接表和优先队列的更快变体（$O(VE + V^2 \log V)$）。

Stoer–Wagner 算法表明，通过纯粹的合并思想，可以高效地一次一割地衡量全局连通性的脆弱性。
### 362 Karger 随机割算法

Karger 算法是一种极其简洁的随机算法，用于寻找无向图的全局最小割。它并非确定性地探索所有划分，而是随机收缩边，不断缩小图直到只剩下两个超节点。它们之间的边之和（以高概率）就是最小割。

#### 我们解决什么问题？

给定一个无向、无权（或有权）图 $G = (V, E)$，一个割是将 $V$ 划分为两个不相交子集 $(S, V \setminus S)$。割的大小是跨越该划分的边的总数（或总权重）。

我们希望找到全局最小割：
$$
\min_{S \subset V, S \neq \emptyset, S \neq V} \text{cut}(S)
$$

Karger 算法能以概率保证找到确切的最小割。

#### 它是如何工作的（通俗解释）？

其简洁性近乎神奇：

1.  当图中顶点数大于 2 时：
    *   随机选取一条边 $(u, v)$。
    *   收缩它，将 $u$ 和 $v$ 合并为一个顶点。
    *   移除自环。
2.  当只剩下两个顶点时，
    *   它们之间的边构成一个割。

重复此过程多次以提高成功概率。

每次运行找到真正最小割的概率至少为
$$
\frac{2}{n(n-1)}
$$

通过重复 $O(n^2 \log n)$ 次，错过它的概率变得可忽略不计。

#### 示例

考虑一个三角形图 $(A, B, C)$：

-   随机选取一条边，例如 $(A, B)$，收缩为超节点 $(AB)$。
-   现在边为：$(AB, C)$，重数为 2。
-   只剩下 2 个节点，割权重 = 2，即最小割。

#### 微型代码（Python）

```python
import random
import copy

def karger_min_cut(graph):
    # graph: 邻接表 {u: [v1, v2, ...]}
    vertices = list(graph.keys())
    edges = []
    for u in graph:
        for v in graph[u]:
            if u < v:
                edges.append((u, v))

    g = copy.deepcopy(graph)
    while len(g) > 2:
        u, v = random.choice(edges)
        # 将 v 合并到 u
        g[u].extend(g[v])
        for w in g[v]:
            g[w] = [u if x == v else x for x in g[w]]
        del g[v]
        # 移除自环
        g[u] = [x for x in g[u] if x != u]
        edges = []
        for x in g:
            for y in g[x]:
                if x < y:
                    edges.append((x, y))
    # 任何剩余的边列表给出割的大小
    return len(list(g.values())[0])
```

#### 为何重要

-   优雅简洁，且能以概率证明正确性。
-   无需流计算，无需复杂数据结构。
-   极佳的教学算法，用于展示随机化推理。
-   构成了改进变体（Karger–Stein、随机收缩）的基础。

Karger 的方法展示了随机化、最小逻辑和最大洞察力的力量。

#### 温和的证明（为何有效）

在每一步，收缩一条非最小割边不会破坏最小割。
由于有 $O(n^2)$ 条边且至少进行 $n-2$ 次收缩，
我们从未收缩最小割边的概率是：
$$
P = \prod_{i=0}^{n-3} \frac{k_i}{m_i} \ge \frac{2}{n(n-1)}
$$
其中 $k_i$ 是最小割大小，$m_i$ 是剩余边数。

通过足够多次的重复，失败概率呈指数级衰减。

#### 亲自尝试

1.  在一个 4 节点图上运行算法 1 次，注意结果的可变性。
2.  重复 50 次，收集最小割的频率。
3.  在纸上可视化收缩步骤。
4.  通过将加权边扩展为平行边副本的方式添加权重。
5.  在稠密图上比较其运行时间与 Stoer–Wagner 算法。

#### 测试用例

| 图                 | 预期最小割 | 备注                       |
| ------------------ | ---------- | -------------------------- |
| 三角形             | 2          | 总是 2                     |
| 正方形（4-环）     | 2          | 随机路径收敛               |
| 完全图 $K_4$       | 3          | 稠密，需重复运行以确认     |

#### 复杂度

-   时间：每次尝试 $O(n^2)$
-   空间：$O(n^2)$（用于邻接表）
-   重复次数：$O(n^2 \log n)$ 以获得高置信度

Karger 算法是一个里程碑式的例子：一句简单的指令——“随机选取一条边并收缩它”——展开为一个完整的、可证明正确的全局最小割发现算法。
### 363 Karger–Stein 最小割

Karger–Stein 算法是 Karger 原始收缩算法的一种改进的随机化分治版本。
它在保持同样优美简洁思想的同时，获得了更高的成功概率和更好的期望运行时间：重复收缩随机边，但提前停止并递归，而不是完全收缩到只剩两个顶点。

#### 我们要解决什么问题？

给定一个无向、有权或无权的图 $G = (V, E)$，目标是找到全局最小割，即：

$$
\min_{S \subset V,, S \neq \emptyset,, S \neq V} \text{cut}(S)
$$

其中：

$$
\text{cut}(S) = \sum_{u \in S,, v \in V \setminus S} w(u, v)
$$

#### 它是如何工作的（通俗解释）？

与 Karger 算法类似，我们随机收缩边，但不是立即收缩到 2 个顶点，
而是只收缩到图大约剩下 $\frac{n}{\sqrt{2}}$ 个顶点，然后独立地递归两次。
在递归调用中找到的最佳割就是我们的结果。

这种策略在保持效率的同时，放大了成功概率。

算法步骤：

1. 如果 $|V| \le 6$：

   * 通过暴力或基本收缩直接计算最小割。
2. 否则：

   * 令 $t = \lceil n / \sqrt{2} + 1 \rceil$。
   * 随机收缩边直到只剩下 $t$ 个顶点。
   * 在两个独立的收缩结果上，分别递归运行该算法两次。
3. 返回两个割中较小的那个。

每个收缩阶段都能以良好的概率保留最小割，而递归重复则能累积成功的机会。

#### 示例

对于一个有 16 个顶点的图：

- 随机收缩到 $\lceil 16 / \sqrt{2} \rceil = 12$ 个顶点。
- 独立地递归两次。
- 每次递归再次将顶点数减半，直到达到基本情况。
- 返回找到的最小割。

多个递归分支使得在所有收缩过程中都保留最小割所有边的总体概率远高于单次收缩。

#### 微型代码（Python）

```python
import random
import math
import copy

def contract_random_edge(graph):
    u, v = random.choice([(u, w) for u in graph for w in graph[u] if u < w])
    # 将 v 合并到 u
    graph[u].extend(graph[v])
    for w in graph[v]:
        graph[w] = [u if x == v else x for x in graph[w]]
    del graph[v]
    # 移除自环
    graph[u] = [x for x in graph[u] if x != u]

def karger_stein(graph):
    n = len(graph)
    if n <= 6:
        # 基本情况：回退到基本 Karger 算法
        g_copy = copy.deepcopy(graph)
        while len(g_copy) > 2:
            contract_random_edge(g_copy)
        return len(list(g_copy.values())[0])

    t = math.ceil(n / math.sqrt(2)) + 1
    g1 = copy.deepcopy(graph)
    g2 = copy.deepcopy(graph)
    while len(g1) > t:
        contract_random_edge(g1)
    while len(g2) > t:
        contract_random_edge(g2)

    return min(karger_stein(g1), karger_stein(g2))
```

#### 为什么它很重要

- 将成功概率提高到大约 $1 / \log n$，而基本 Karger 算法是 $1/n^2$。
- 分治结构减少了所需的重复次数。
- 展示了随机化算法中概率放大的威力。
- 对于大型稠密图很有用，因为确定性 $O(V^3)$ 算法在这些图上更慢。

它是一种近乎最优的随机化最小割算法，在简洁性和效率之间取得了平衡。

#### 一个温和的证明（为什么它有效）

每次收缩以如下概率保留最小割：

$$
p = \prod_{i=k+1}^n \left(1 - \frac{2}{i}\right)
$$

在 $t = n / \sqrt{2}$ 处提前停止使得 $p$ 保持在一个相当高的水平。
由于我们独立地递归两次，总体成功概率变为：

$$
P = 1 - (1 - p)^2 \approx 2p
$$

重复 $O(\log^2 n)$ 次可以确保以很高的置信度找到真正的最小割。

#### 动手试试

1.  与普通 Karger 算法比较运行时间和准确性。
2.  在小图上运行，并收集 100 次试验的成功率。
3.  可视化收缩的递归树。
4.  添加边权重（通过扩展平行边实现）。
5.  确认返回的割与 Stoer–Wagner 算法的结果一致。

#### 测试用例

| 图           | 期望最小割 | 备注                       |
| ------------ | ---------- | -------------------------- |
| 三角形       | 2          | 总能找到                   |
| 4 节点环     | 2          | 准确率高                   |
| 稠密图 $K_6$ | 5          | 重复运行可提高置信度       |

#### 复杂度

- 期望时间：$O(n^2 \log^3 n)$
- 空间：$O(n^2)$
- 重复次数：$O(\log^2 n)$（以获得高概率）

Karger–Stein 算法将随机收缩的原始优雅提炼成了一颗分治的宝石，更快、更可靠，并且依然令人愉悦地简单。
### 364 Gomory–Hu 树

Gomory–Hu 树是一种卓越的数据结构，它紧凑地表示了无向加权图中所有顶点对之间的最小割。
它无需计算 $O(n^2)$ 个独立的割，而是构建一棵单一的树（包含 $n-1$ 条边），其边的权重捕获了每一对顶点间的最小割值。

这种结构将全局连通性问题转化为简单的树查询，快速、精确且优雅。

#### 我们要解决什么问题？

给定一个无向加权图 $G = (V, E, w)$，我们希望为每一对顶点 $(s, t)$ 找到：

$$
\lambda(s, t) = \min_{S \subset V,, s \in S,, t \notin S} \sum_{u \in S, v \notin S} w(u, v)
$$

我们不是为每一对顶点单独运行一次最小割计算，而是构建一棵 Gomory–Hu 树 $T$，使得：

> 对于任意顶点对 $(s, t)$，
> 在 $T$ 中连接 $s$ 和 $t$ 的路径上的最小边权重等于 $\lambda(s, t)$。

这棵树紧凑地编码了所有顶点对之间的最小割。

#### 它是如何工作的（通俗解释）？

Gomory–Hu 树是通过 $n-1$ 次最小割计算迭代构建的：

1.  选择一个根节点 $r$（任意选择）。
2.  维护一棵以顶点为节点的划分树 $T$。
3.  当存在未处理的划分时：
    *   在同一划分中选取两个顶点 $s, t$。
    *   使用任意最大流算法计算 $s$–$t$ 最小割。
    *   根据该割将顶点划分为两个集合 $(S, V \setminus S)$。
    *   在 Gomory–Hu 树中添加一条边 $(s, t)$，其权重等于割值。
    *   对每个划分递归处理。
4.  经过 $n-1$ 次割计算后，树即构建完成。

树中的每条边都代表了原图中的一个不同的划分割。

#### 示例

考虑一个顶点集为 ${A, B, C, D}$ 且边带权重的图。
我们逐步执行割计算：

1.  选择 $s = A, t = B$ → 找到割权重 $w(A,B) = 2$
    → 向树中添加边 $(A, B, 2)$。
2.  在 $A$ 侧和 $B$ 侧内部递归。
3.  继续直到树拥有 $n-1 = 3$ 条边。

现在，对于任意顶点对 $(u,v)$，它们之间的最小割值等于在 $T$ 中连接它们的路径上的最小边权重。

#### 简略代码（高级框架）

```python
def gomory_hu_tree(V, edges):
    # edges: 列表，元素为 (u, v, w)
    from collections import defaultdict
    n = V
    tree = defaultdict(list)
    parent = [0] * n
    cut_value = [0] * n

    for s in range(1, n):
        t = parent[s]
        # 通过最大流计算 s-t 最小割
        mincut, partition = min_cut(s, t, edges)
        cut_value[s] = mincut
        for v in range(n):
            if v != s and parent[v] == t and partition[v]:
                parent[v] = s
        tree[s].append((t, mincut))
        tree[t].append((s, mincut))
        if partition[t]:
            parent[s], parent[t] = parent[t], s
    return tree
```

在实践中，`min_cut(s, t)` 使用 Edmonds–Karp 或 Push–Relabel 算法计算。

#### 为什么它很重要

-   仅用 $n-1$ 次最大流计算就捕获了所有顶点对之间的最小割。
-   将图割查询转化为简单的树查询。
-   支持高效的网络可靠性分析、连通性查询和冗余规划。
-   适用于加权无向图。

该算法将丰富的连通性信息压缩到一个单一而优雅的结构中。

#### 一个温和的证明（为什么它有效）

每个 $s$–$t$ 最小割将顶点划分为两侧；迭代地合并这些划分会保留所有顶点对之间的最小割关系。

通过归纳法，每一对顶点 $(u, v)$ 最终恰好被树中的一条边分隔开，且该边的权重等于 $\lambda(u, v)$。
因此，
$$
\lambda(u, v) = \min_{e \in \text{path}(u, v)} w(e)
$$

#### 亲自尝试

1.  在一个小的 4 节点加权图上运行。
2.  验证每条边的权重是否等于一个实际的割值。
3.  通过树路径最小值查询随机顶点对 $(u, v)$。
4.  与独立的基于流的最小割计算进行比较。
5.  绘制原始图和结果树。

#### 测试用例

| 图结构         | 树边数量 | 备注                                 |
| -------------- | -------- | ------------------------------------ |
| 三角形         | 2 条边   | 均匀权重产生相等的割                 |
| 正方形         | 3 条边   | 不同的割形成平衡的树                 |
| 完全图 $K_4$   | 3 条边   | 对称的连通性                         |

#### 复杂度

-   时间：$O(n \cdot \text{MaxFlow}(V,E))$
-   空间：$O(V + E)$
-   查询：$O(\log V)$（通过树中路径最小值查询）

Gomory–Hu 树优雅地将图的割结构转化为一棵蕴含所有最小割真相的树，一个结构，囊括所有。
### 365 最大流-最小割定理

最大流-最小割定理是图论和组合优化领域的基础性结果之一。它揭示了最大流（网络中能通过的量）与最小割（阻碍网络的瓶颈）之间的对偶关系。两者不仅仅是相关，而是完全相等。

这一定理支撑了几乎所有关于流、割和网络设计的算法。

#### 我们解决什么问题？

我们处理一个有向图 $G = (V, E)$，一个源点 $s$，以及一个汇点 $t$。每条边 $(u,v)$ 都有一个容量 $c(u,v) \ge 0$。

一个流为每条边分配值 $f(u,v)$，满足：

1. 容量约束：$0 \le f(u,v) \le c(u,v)$
2. 流量守恒：对于所有 $u \ne s, t$，有 $\sum_v f(u,v) = \sum_v f(v,u)$

流的值为：
$$
|f| = \sum_{v} f(s, v)
$$

我们希望找到从 $s$ 到 $t$ 的最大可能流值。

一个割 $(S, T)$ 是将顶点集划分为两部分，使得 $s \in S$ 且 $t \in T = V \setminus S$。割的容量为：
$$
c(S, T) = \sum_{u \in S, v \in T} c(u, v)
$$

#### 定理

> 最大流-最小割定理：
> 在任何流网络中，可行流的最大值等于 $s$–$t$ 割的最小容量。

形式化表示为：
$$
\max_f |f| = \min_{(S,T)} c(S, T)
$$

这是一个强对偶性陈述：一个集合上的最大值等于另一个集合上的最小值。

#### 它是如何工作的（通俗解释）？

1.  你试图从 $s$ 向 $t$ 推送尽可能多的流。
2.  当你使边饱和时，图的某些部分会成为瓶颈。
3.  最终的残差图将从 $s$ 可达的顶点（通过未饱和边）与其余顶点分离开来。
4.  这种分离 $(S, T)$ 形成了一个最小割，其容量等于已发送的总流量。

因此，一旦你无法推送更多流量，你也找到了阻碍你的最紧的割。

#### 示例

考虑一个小型网络：

| 边    | 容量 |
| ----- | ---- |
| s → a | 3    |
| s → b | 2    |
| a → t | 2    |
| b → t | 3    |
| a → b | 1    |

通过增广路径求最大流：

-   $s \to a \to t$：2 单位
-   $s \to b \to t$：2 单位
-   $s \to a \to b \to t$：1 单位

总流量 = 5。

最小割：$S = {s, a}$，$T = {b, t}$ → 容量 = 5。
流量 = 割容量 = 5 ✔

#### 微型代码（通过 Edmonds–Karp 算法验证）

```python
from collections import deque

def bfs(cap, flow, s, t, parent):
    n = len(cap)
    visited = [False]*n
    q = deque([s])
    visited[s] = True
    while q:
        u = q.popleft()
        for v in range(n):
            if not visited[v] and cap[u][v] - flow[u][v] > 0:
                parent[v] = u
                visited[v] = True
                if v == t: return True
                q.append(v)
    return False

def max_flow_min_cut(cap, s, t):
    n = len(cap)
    flow = [[0]*n for _ in range(n)]
    parent = [-1]*n
    max_flow = 0
    while bfs(cap, flow, s, t, parent):
        v = t
        path_flow = float('inf')
        while v != s:
            u = parent[v]
            path_flow = min(path_flow, cap[u][v] - flow[u][v])
            v = u
        v = t
        while v != s:
            u = parent[v]
            flow[u][v] += path_flow
            flow[v][u] -= path_flow
            v = u
        max_flow += path_flow
    return max_flow
```

在残差图中从 $s$ 出发最终可达的顶点集定义了最小割。

#### 为何重要

-   连接优化与组合数学的基本定理。
-   Ford–Fulkerson、Edmonds–Karp、Dinic、Push–Relabel 等算法的基础。
-   应用于图像分割、网络可靠性、调度、二分图匹配、聚类和运输等领域。

它弥合了流最大化与割最小化之间的鸿沟，两者是同一枚硬币的两面。

#### 一个温和的证明（为何成立）

在 Ford–Fulkerson 算法终止时：

-   残差图中不存在增广路径。
-   令 $S$ 为从 $s$ 可达的所有顶点。
-   那么所有满足 $u \in S, v \notin S$ 的边 $(u,v)$ 都已饱和。

因此：
$$
|f| = \sum_{u \in S, v \notin S} f(u,v) = \sum_{u \in S, v \notin S} c(u,v) = c(S, T)
$$
并且没有割的容量能比这更小。
所以，$\max |f| = \min c(S,T)$。

#### 亲自尝试

1.  构建一个简单的 4 节点网络并追踪增广路径。
2.  识别算法终止时的割 $(S, T)$。
3.  验证总流量等于割容量。
4.  测试不同的最大流算法，结果保持不变。
5.  在残差图中可视化割边。

#### 测试用例

| 图类型         | 最大流 | 最小割 | 备注                     |
| -------------- | ------ | ------ | ------------------------ |
| 简单链式       | 5      | 5      | 相同                     |
| 平行边         | 8      | 8      | 结果相同                 |
| 菱形图         | 6      | 6      | 最小割 = 瓶颈边          |

#### 复杂度

-   取决于底层流算法（例如 Edmonds–Karp 算法为 $O(VE^2)$）
-   割的提取：$O(V + E)$

最大流-最小割定理是网络优化的核心，每一条增广路径都是朝着两个基本视角——发送与分离——之间相等性迈进的一步。
### 366 Stoer–Wagner 重复阶段算法

Stoer–Wagner 重复阶段算法是对无向加权图的 Stoer–Wagner 最小割算法的一种改进。它通过多次重复最大邻接搜索阶段，逐步合并顶点并追踪割的权重，从而找到全局最小割。

该算法优雅、确定，并在多项式时间内运行，对于无向图通常比基于流的方法更快。

#### 我们要解决什么问题？

我们正在寻找一个无向加权图 $G = (V, E)$ 中的最小割，其中每条边 $(u, v)$ 都有一个非负权重 $w(u, v)$。

一个割 $(S, V \setminus S)$ 将顶点集划分为两个不相交的子集。割的权重定义为：
$$
w(S, V \setminus S) = \sum_{u \in S, v \notin S} w(u, v)
$$

我们的目标是找到一个割 $(S, T)$，使得这个和最小。

#### 核心思想

该算法通过重复执行“阶段”来工作。每个阶段在当前收缩的图中识别一个最小 $s$–$t$ 割，并合并最后添加的两个顶点。经过多个阶段后，发现的最小割就是全局最小割。

每个阶段遵循最大邻接搜索模式，类似于 Prim 算法，但逻辑相反。

#### 工作原理（通俗解释）

每个阶段：

1.  选择一个任意的起始顶点。
2.  维护一个已添加顶点的集合 $A$。
3.  在每一步，添加与 $A$ 连接最紧密的不在 $A$ 中的顶点 $v$（即与 $A$ 中顶点总边权最大的顶点）。
4.  继续直到所有顶点都在 $A$ 中。
5.  令 $s$ 为倒数第二个添加的顶点，$t$ 为最后一个添加的顶点。
6.  将 $t$ 与其余顶点分离的割是一个候选最小割。
7.  记录其权重；然后将 $s$ 和 $t$ 合并为一个超顶点并重复。

经过 $|V| - 1$ 个阶段后，看到的最小割就是全局最小割。

#### 示例

图：

| 边   | 权重 |
| ---- | ---- |
| A–B  | 3    |
| A–C  | 2    |
| B–C  | 4    |
| B–D  | 2    |
| C–D  | 3    |

阶段 1：

-   从 $A = {A}$ 开始
-   添加 $B$（与 A 连接最大：3）
-   添加 $C$（与 ${A,B}$ 连接最大：总计 6）
-   最后添加 $D$ → 割权重 = D 到 ${A,B,C}$ 的边权和 = 5

记录最小割 = 5。合并 $(C,D)$ 并继续。

重复阶段 → 全局最小割 = 5。

#### 微型代码（简化 Python）

```python
def stoer_wagner_min_cut(graph):
    n = len(graph)
    vertices = list(range(n))
    min_cut = float('inf')

    while len(vertices) > 1:
        added = [False] * n
        weights = [0] * n
        prev = -1
        for _ in range(len(vertices)):
            u = max(vertices, key=lambda v: weights[v] if not added[v] else -1)
            added[u] = True
            if _ == len(vertices) - 1:
                # 最后添加的顶点，潜在的割
                min_cut = min(min_cut, weights[u])
                # 将 u 合并到 prev
                if prev != -1:
                    for v in vertices:
                        if v != u and v != prev:
                            graph[prev][v] += graph[u][v]
                            graph[v][prev] = graph[prev][v]
                    vertices.remove(u)
                break
            prev = u
            for v in vertices:
                if not added[v]:
                    weights[v] += graph[u][v]
    return min_cut
```

图以邻接矩阵形式给出。每个阶段选择连接最紧密的顶点，记录割，并合并节点。

#### 为什么它重要

-   对于无向加权图是确定且优雅的。
-   比运行多次最大流计算更快。
-   适用于网络可靠性、图划分、聚类和电路设计。
-   每个阶段都模拟了“收紧”图的过程，直到只剩下一个超顶点。

#### 一个温和的证明（为什么它有效）

每个阶段都找到一个最小 $s$–$t$ 割，其中 $t$ 是最后添加的顶点。通过合并 $s$ 和 $t$，我们保留了所有其他可能的割。这些阶段割中的最小值就是全局最小割，因为合并图中的每个割都对应于原图中的一个割。

形式化地：
$$
\text{mincut}(G) = \min_{\text{phases}} w(S, V \setminus S)
$$

这种归纳结构确保了最优性。

#### 自己动手试试

1.  在一个具有随机权重的 4 节点完全图上运行。
2.  追踪每个阶段中顶点的添加顺序。
3.  记录每个阶段的割权重。
4.  与暴力枚举所有割进行比较，它们应该匹配。
5.  可视化收缩步骤。

#### 测试用例

| 图                       | 边数 | 最小割          | 备注             |
| ------------------------ | ---- | --------------- | ---------------- |
| 三角形（等权重）         | 3    | 2×权重          | 全部相等         |
| 正方形（单位权重）       | 4    | 2               | 对边             |
| 加权网格                 | 6    | 最小的桥        | 在瓶颈处割开     |

#### 复杂度

-   每个阶段：$O(V^2)$
-   总计（邻接矩阵）：$O(V^3)$

使用斐波那契堆，可以改进到 $O(V^2 \log V + VE)$。

Stoer–Wagner 重复阶段算法是一个强大的、纯组合的工具，无需流，无需残差图，仅通过紧密的连接性和精确的合并来逼近真正的全局最小割。
### 367 动态最小割

动态最小割问题将经典的最小割计算扩展到随时间变化的图，其中边可以被添加、删除或更新。
与其在每次变化后从头重新计算，我们维护能够高效更新最小割的数据结构。

动态最小割算法在网络弹性、增量优化以及连接性实时演变的系统等应用中至关重要。

#### 我们正在解决什么问题？

给定一个具有加权边的图 $G = (V, E)$ 以及当前的最小割，当发生以下情况时，我们如何高效地维护这个割：

- 插入一条边 $(u, v)$
- 删除一条边 $(u, v)$
- 边 $(u, v)$ 的权重发生变化

朴素的方法是使用像 Stoer–Wagner（$O(V^3)$）这样的算法从头重新计算最小割。
动态算法旨在实现更快的增量更新。

#### 核心思想

最小割仅对跨越割边界的边敏感。
当图发生变化时，只有被修改边周围的局部区域可能改变全局割。

动态算法使用：

- 动态树（例如 Link-Cut Trees）
- 全动态连通性结构
- 随机化收缩跟踪
- 受影响区域的增量重计算

来高效更新，而不是重新运行完整的最小割算法。

#### 工作原理（通俗解释）

1.  维护割的表示，通常表示为树或分区集合。
2.  当边权重变化时，更新连通性信息：

    *   如果边位于割的某一侧内部，则无变化。
    *   如果边跨越割，则更新割容量。
3.  当添加或删除边时，调整受影响的连通分量并进行局部重新计算。
4.  可选地，在多次更新后，运行周期性的全局重计算以纠正偏差。

这些方法以牺牲精确性来换取效率，通常在小误差范围内维护近似的最小割。

#### 示例（高层次）

图有顶点 ${A, B, C, D}$，当前最小割为 $({A, B}, {C, D})$，权重为 $5$。

1.  添加权重为 $1$ 的边 $(B, C)$：

    *   新的跨越边，割权重变为 $5 + 1 = 6$。
    *   检查是否存在其他分区能给出更小的总权重，如果需要则更新。

2.  删除边 $(A, C)$：

    *   从割集合中移除。
    *   如果这条边对连通性至关重要，最小割可能会增大。
    *   如果受影响，则重新计算局部割。

#### 微型代码（草图，Python）

```python
class DynamicMinCut:
    def __init__(self, graph):
        self.graph = graph
        self.min_cut_value = self.compute_min_cut()

    def compute_min_cut(self):
        # 使用 Stoer–Wagner 或基于流的方法
        return stoer_wagner(self.graph)

    def update_edge(self, u, v, new_weight):
        self.graph[u][v] = new_weight
        self.graph[v][u] = new_weight
        # 局部重新计算受影响区域
        self.min_cut_value = self.recompute_local(u, v)

    def recompute_local(self, u, v):
        # 简化占位符：如果图小则完全重新计算
        return self.compute_min_cut()
```

对于大型图，将 `recompute_local` 替换为增量割更新逻辑。

#### 为什么重要

-   实时系统需要对网络变化做出快速响应。
-   流图（例如交通、社交或电力网络）持续演变。
-   动态系统中的可靠性分析依赖于最新的最小割值。

与每一步都从头重新计算相比，动态维护节省了时间。

#### 一个温和的证明（为什么有效）

设 $C_t$ 为经过 $t$ 次更新后的最小割。
如果每次更新只影响局部结构，那么：
$$
C_{t+1} = \min(C_t, \text{局部调整})
$$
维护一个证书结构（如 Gomory–Hu 树）可以确保正确性，因为除了涉及更新的边之外，所有成对的最小割在局部变化下都得以保留。

仅重新计算受影响的割保证了正确性，并具有摊还效率。

#### 自己动手试试

1.  构建一个小的加权图（5–6 个节点）。
2.  使用 Stoer–Wagner 算法计算初始最小割。
3.  一次添加或删除一条边。
4.  手动更新仅受影响的割。
5.  与完全重新计算进行比较，结果应该一致。

#### 测试用例

| 操作                                 | 描述               | 新的最小割               |
| ------------------------------------ | ------------------ | ------------------------ |
| 添加权重较小的边 $(u,v)$             | 新的跨越边         | 割可能变小               |
| 增加边权重                           | 加强桥梁           | 割可能在别处改变         |
| 删除跨越割的边                       | 削弱连接           | 割可能增大               |
| 删除分区内部的边                     | 无变化             |,                        |

#### 复杂度

| 操作                       | 时间                    |
| -------------------------- | ----------------------- |
| 朴素重计算                 | $O(V^3)$                |
| 动态方法（随机化）         | $O(V^2 \log V)$ 摊还    |
| 近似动态割                 | $O(E \log^2 V)$         |

动态最小割算法在精确性和响应性之间取得平衡，在图形实时演变时保持近乎最优的连通性洞察。
### 368 最小 s–t 割（Edmonds–Karp）

最小 s–t 割问题旨在找到需要移除的边的最小总容量，以将源顶点 $s$ 与汇顶点 $t$ 分离。它是最大流的对偶问题，而 Edmonds–Karp 算法提供了一条清晰的路径，使用基于 BFS 的增广路径来计算它。

#### 我们要解决什么问题？

给定一个有向加权图 $G=(V,E)$，其边容量为 $c(u,v)$，找到一个将 $V$ 划分为两个不相交集合 $(S, T)$ 的分割，使得：

- $s \in S$，$t \in T$
- 从 $S$ 到 $T$ 的边的容量之和最小

形式化表示为：
$$
\text{min-cut}(s, t) = \min_{(S,T)} \sum_{u \in S, v \in T} c(u, v)
$$

根据最大流最小割定理，这个割值等于从 $s$ 到 $t$ 的最大流值。

#### 工作原理（通俗解释）

该算法首先使用 Edmonds–Karp（一种基于 BFS 的 Ford–Fulkerson 算法变体）来找到最大流。计算完最大流后，在残差图中从 $s$ 可达的顶点决定了最小割。

步骤：

1. 初始化所有边的流量 $f(u, v) = 0$。
2. 当在残差图中存在一条从 $s$ 到 $t$ 的 BFS 路径时：

   * 计算路径上的瓶颈容量。
   * 沿路径增广流量。
3. 当不存在增广路径时：

   * 在残差图中从 $s$ 出发最后运行一次 BFS。
   * 从 $s$ 可达的顶点构成集合 $S$。
   * 其他顶点构成 $T$。
4. 从 $S$ 跨越到 $T$ 且容量已满的边就是最小割边。

#### 示例

图：

- 顶点：${s, a, b, t}$
- 边：

  * $(s, a) = 3$
  * $(s, b) = 2$
  * $(a, b) = 1$
  * $(a, t) = 2$
  * $(b, t) = 3$

1. 运行 Edmonds–Karp 找到最大流 = 4。
2. 残差图：

   * 从 $s$ 可达的集合：${s, a}$
   * 不可达集合：${b, t}$
3. 最小割边：$(a, t)$ 和 $(s, b)$
4. 最小割值 = $2 + 2 = 4$
   与最大流值匹配。

#### 微型代码（类 C 语言伪代码）

```c
int min_st_cut(Graph *G, int s, int t) {
    int maxflow = edmonds_karp(G, s, t);
    bool visited[V];
    bfs_residual(G, s, visited);
    int cut_value = 0;
    for (edge (u,v) in G->edges)
        if (visited[u] && !visited[v])
            cut_value += G->capacity[u][v];
    return cut_value;
}
```

#### 为什么重要

- 揭示网络中的瓶颈。
- 是可靠性和分割问题的关键。
- 是图像分割、网络设计和流分解的基础。
- 直接支持优化问题之间的对偶性证明。

#### 一个温和的证明（为什么它有效）

最大流最小割定理指出：

$$
\max_{\text{flow } f} \sum_{v} f(s, v) = \min_{(S, T)} \sum_{u \in S, v \in T} c(u, v)
$$

Edmonds–Karp 通过沿着最短路径（BFS 顺序）反复增广来找到最大流。一旦不存在更多的增广路径，残差图自然地将节点划分为 $S$ 和 $T$，而从 $S$ 到 $T$ 的边就定义了最小割。

#### 自己动手试试

1.  构建一个带容量的小型有向网络。
2.  手动运行 Edmonds–Karp（追踪增广路径）。
3.  绘制残差图并找出从 $s$ 出发的可达集合。
4.  标记跨越边，并求和它们的容量。
5.  与最大流值进行比较。

#### 测试用例

| 图                                 | 最大流          | 最小割          | 匹配？ |
| ---------------------------------- | --------------- | --------------- | ------ |
| 简单的 4 节点图                    | 4               | 4               | ✅      |
| 线性链 $s \to a \to b \to t$       | 最小边容量      | 最小边容量      | ✅      |
| 并行路径                           | 最小容量之和    | 最小容量之和    | ✅      |

#### 复杂度

| 步骤                 | 时间复杂度 |
| -------------------- | ---------- |
| 每次增广的 BFS       | $O(E)$     |
| 增广次数             | $O(VE)$    |
| 总计                 | $O(VE^2)$  |

通过 Edmonds–Karp 算法求解最小 s–t 割，是流算法与划分推理之间一座优雅的桥梁，割中的每一条边都讲述着约束、容量与平衡的故事。
### 369 近似最小割

近似最小割算法提供了一种比精确算法更快地估计图的最小割的方法，尤其是在不需要绝对精确的情况下。该算法基于随机化和采样，利用概率推理在大规模或动态图中以高置信度找到较小的割。

#### 我们要解决什么问题？

给定一个带权无向图 $G=(V, E)$，我们希望找到一个割 $(S, T)$，使得其割容量接近真实的最小值：

$$
w(S, T) \le (1 + \epsilon) \cdot \lambda(G)
$$

其中 $\lambda(G)$ 是全局最小割的权重，$\epsilon$ 是一个小的误差容忍度（例如 $0.1$）。

目标是速度：近似最小割算法以近似线性的时间运行，比精确算法（$O(V^3)$）快得多。

#### 工作原理（通俗解释）

近似算法依赖于两个关键原则：

1.  **随机采样**：
    以与其权重成比例的概率随机采样边。
    边的容量越小，它对于最小割越关键的可能性就越大。

2.  **图稀疏化**：
    构建一个更小的图“草图”，它能近似地保留割的权重。
    在这个稀疏图上计算最小割，结果接近真实值。

通过多次重复采样并取找到的最小割，我们收敛到一个接近最优的解。

#### 算法概览（Karger 采样方法）

1.  输入：图 $G(V, E)$，其中 $n = |V|$，$m = |E|$
2.  选择采样概率 $p = \frac{c \log n}{\epsilon^2 \lambda}$
3.  构建采样图 $G'$：
    *   以概率 $p$ 包含每条边 $(u, v)$
    *   将包含的边的权重缩放 $\frac{1}{p}$ 倍
4.  在 $G'$ 上运行精确的最小割算法（Stoer–Wagner）
5.  重复采样 $O(\log n)$ 次；取找到的最佳割

结果以高概率在因子 $(1 + \epsilon)$ 内近似 $\lambda(G)$。

#### 示例

假设 $G$ 有 $10^5$ 条边，精确的 Stoer–Wagner 算法会太慢。

1.  选择 $\epsilon = 0.1$，$p = 0.02$
2.  随机采样 $2\%$ 的边（2000 条边）
3.  将采样边的权重重新缩放 $\frac{1}{0.02} = 50$ 倍
4.  在这个更小的图上运行精确的最小割算法
5.  重复 5–10 次；选取最小的割

结果：在远少于精确算法的时间内，得到一个在最优解 $10\%$ 范围内的割。

#### 微型代码（类 Python 伪代码）

```python
def approximate_min_cut(G, epsilon=0.1, repeats=10):
    best_cut = float('inf')
    for _ in range(repeats):
        p = compute_sampling_probability(G, epsilon)
        G_sample = sample_graph(G, p)
        cut_value = stoer_wagner(G_sample)
        best_cut = min(best_cut, cut_value)
    return best_cut
```

#### 为何重要

*   **可扩展性**：处理精确方法不可行的大型图
*   **速度**：利用随机化实现近似线性时间
*   **应用**：
    *   流图
    *   网络可靠性
    *   聚类和划分
    *   图草图和稀疏化

当你需要快速、稳健的决策，而不是完美的答案时，近似最小割至关重要。

#### 温和的证明（为何有效）

Karger 的分析表明，如果采样的边足够多，每个小割以高概率被保留：

$$
\Pr[\text{割权重被保留}] \ge 1 - \frac{1}{n^2}
$$

通过将过程重复 $O(\log n)$ 次，我们放大了置信度，确保以高概率，至少一个采样图保持了真实的最小割结构。

利用 Chernoff 界，误差被限制在 $(1 \pm \epsilon)$ 范围内。

#### 亲自尝试

1.  生成一个包含 50 个节点和随机权重的随机图。
2.  使用 Stoer–Wagner 算法计算精确最小割。
3.  采样 10%、5% 和 2% 的边，计算近似割。
4.  比较结果和运行时间。
5.  调整 $\epsilon$，观察速度和准确性之间的权衡。

#### 测试用例

| 图类型                     | 精确最小割 | 近似值 (ε=0.1) | 误差 | 加速比 |
| ------------------------ | ------------- | --------------- | ----- | ------- |
| 小型稠密图 (100 条边)     | 12            | 13              | 8%    | 5×      |
| 中型稀疏图 (1k 条边)      | 8             | 8               | 0%    | 10×     |
| 大型图 (100k 条边)        | 30            | 33              | 10%   | 50×     |

#### 复杂度

| 方法                       | 时间复杂度                     | 准确性              |
| ------------------------- | ---------------------------- | --------------------- |
| Stoer–Wagner              | $O(V^3)$                     | 精确                 |
| Karger (随机化精确)        | $O(V^2 \log^3 V)$            | 精确（概率性） |
| 近似采样                   | $O(E \log^2 V / \epsilon^2)$ | $(1 + \epsilon)$      |

近似最小割算法表明，当规模要求速度时，概率可以替代精度，它们以惊人的效率切割大规模图。
### 370 最小 k 割

最小 k 割问题推广了经典的“最小割”概念。其目标不再是仅将图分割为两部分，而是将其划分为 k 个互不相交的子集，同时使被切割边的总权重最小。

这是聚类、并行处理和网络设计中的一个关键问题，在这些场景中，我们希望得到多个互不连通的区域，且区域间的互连成本最小。

#### 我们要解决什么问题？

给定一个带权无向图 $G = (V, E)$ 和一个整数 $k$，寻找一个将 $V$ 划分为 k 个子集 ${V_1, V_2, \ldots, V_k}$ 的分割，使得：

- 对于所有 $i \ne j$，有 $V_i \cap V_j = \emptyset$
- $\bigcup_i V_i = V$
- 跨越不同部分的边的权重之和最小

形式化表示为：

$$
\text{min-}k\text{-cut}(G) = \min_{V_1, \ldots, V_k} \sum_{\substack{(u,v) \in E \ u \in V_i, v \in V_j, i \ne j}} w(u,v)
$$

当 $k=2$ 时，该问题简化为标准的最小割问题。

#### 工作原理（通俗解释）

对于一般的 $k$，最小 k 割问题是 NP 难的，但有几种算法可以为较小的 $k$ 提供精确解，并为较大的 $k$ 提供近似解。

主要有两种方法：

1. **贪心迭代切割法**：
   * 反复寻找并移除全局最小割，从而将图逐个分割成连通分量。
   * 经过 $k-1$ 次切割后，得到 $k$ 个分量。
   * 效果不错，但并非总是最优。

2. **基于树的动态规划法（对小 k 精确）**：
   * 使用图的树分解。
   * 通过探索最小生成树中的边移除来计算最优分割。
   * 基于 Karger–Stein 框架。

#### 示例

一个包含 5 个节点和边的图：

| 边     | 权重 |
| ------ | ---- |
| (A, B) | 1    |
| (B, C) | 2    |
| (C, D) | 3    |
| (D, E) | 4    |
| (A, E) | 2    |

目标：分割成 $k=3$ 个子集。

1. 找到要移除的最小割边：
   * 割边 $(A, B)$（权重 1）
   * 割边 $(B, C)$（权重 2）
2. 总割权重 = $3$
3. 得到的子集：${A}, {B}, {C, D, E}$

#### 微型代码（类 Python 伪代码）

```python
def min_k_cut(graph, k):
    cuts = []
    G = graph.copy()
    for _ in range(k - 1):
        cut_value, (S, T) = stoer_wagner(G)
        cuts.append(cut_value)
        G = G.subgraph(S)  # 保留一个连通分量，移除跨越边
    return sum(cuts)
```

对于较小的 $k$，可以使用递归收缩法（如 Karger 算法）或在树结构上进行动态规划。

#### 为何重要

- **聚类**：将节点分组为 $k$ 个平衡的社区。
- **并行计算**：划分工作负载，同时最小化通信成本。
- **图像分割**：将像素划分为 $k$ 个连贯的区域。
- **图简化**：将网络分割成模块化的子图。

最小 k 割将连通性转化为结构化的模块性。

#### 一个温和的证明（为何有效）

每次切割都会使连通分量的数量增加 1。因此，执行 $k-1$ 次切割恰好产生 $k$ 个分量。

设 $C_1, C_2, \ldots, C_{k-1}$ 为连续的最小割。它们的权重之和为全局最优解提供了一个上界：

$$
\text{min-}k\text{-cut} \le \sum_{i=1}^{k-1} \lambda_i
$$

其中 $\lambda_i$ 是第 $i$ 小的割值。迭代最小割通常能很好地近似最优的 k 割。

对于精确解，基于流分解或树收缩的算法使用递归分割来探索边的组合。

#### 动手尝试

1.  构造一个包含 6-8 个顶点的小型带权图。
2.  运行 Stoer–Wagner 算法寻找第一个最小割。
3.  移除边，为下一个割重复此过程。
4.  将总割权重与 $k=3$ 时的暴力分割进行比较。
5.  观察近似质量。

#### 测试用例

| 图           | k   | 精确值 | 贪心法 | 误差 |
| ------------ | --- | ------ | ------ | ---- |
| 三角形       | 3   | 3      | 3      | 0%   |
| 5 节点线形图 | 3   | 3      | 3      | 0%   |
| 6 节点稠密图 | 3   | 12     | 13     | 8%   |

#### 复杂度

| 算法                 | 时间复杂度          | 类型               |
| -------------------- | ------------------- | ------------------ |
| 暴力枚举             | 指数级              | 精确               |
| 贪心切割             | $O(k \cdot V^3)$    | 近似               |
| 树 DP                | $O(V^{k-1})$        | 精确（小 $k$）     |
| 随机化 (Karger–Stein) | $O(V^2 \log^3 V)$   | 近似               |

最小 k 割问题推广了连通性设计——它是图分割与优化的交汇点，在效率和模块性之间取得平衡。

## 第 38 节 匹配
### 371 二分图匹配（DFS）

二分图匹配是图论中最基本的问题之一。
给定一个二分图 $G=(U,V,E)$，目标是找到 $U$ 和 $V$ 中节点之间的最大配对数量，使得没有两条边共享一个顶点。
一种简单直观的方法是使用基于 DFS 的增广路径。

#### 我们要解决什么问题？

我们给定一个具有两个不相交集合 $U$ 和 $V$ 的二分图。
我们希望找到最大匹配，即最大的边集，其中每个顶点最多与一个伙伴匹配。

形式化地说，找到一个子集 $M \subseteq E$，使得：

- $U \cup V$ 中的每个顶点最多与 $M$ 中的一条边关联
- $|M|$ 最大化

#### 工作原理（通俗解释）

核心思想是通过寻找增广路径来逐步构建匹配——这些路径从 $U$ 中一个未匹配的顶点开始，在未匹配边和已匹配边之间交替，并结束于 $V$ 中一个未匹配的顶点。

每次找到这样一条路径，我们就沿着它翻转匹配状态（已匹配边变为未匹配，反之亦然），从而使总匹配大小增加一。

步骤：

1.  从一个空匹配 $M$ 开始。
2.  对于每个顶点 $u \in U$：
    *   运行 DFS 以找到一条到达 $V$ 中空闲顶点的增广路径。
    *   如果找到，则沿着这条路径增广匹配。
3.  重复直到没有更多的增广路径存在。

#### 示例

令 $U = {u_1, u_2, u_3}$, $V = {v_1, v_2, v_3}$，边如下：

- $(u_1, v_1)$, $(u_1, v_2)$
- $(u_2, v_2)$
- $(u_3, v_3)$

1.  从空匹配开始。
2.  从 $u_1$ 进行 DFS：找到 $(u_1, v_1)$ → 匹配。
3.  从 $u_2$ 进行 DFS：找到 $(u_2, v_2)$ → 匹配。
4.  从 $u_3$ 进行 DFS：找到 $(u_3, v_3)$ → 匹配。
    所有顶点都已匹配，最大匹配大小 = 3。

#### 微型代码（类 C 伪代码）

```c
#define MAXV 100
vector<int> adj[MAXV];
int matchR[MAXV], visited[MAXV];

bool dfs(int u) {
    for (int v : adj[u]) {
        if (visited[v]) continue;
        visited[v] = 1;
        if (matchR[v] == -1 || dfs(matchR[v])) {
            matchR[v] = u;
            return true;
        }
    }
    return false;
}

int maxBipartiteMatching(int U) {
    memset(matchR, -1, sizeof(matchR));
    int result = 0;
    for (int u = 0; u < U; ++u) {
        memset(visited, 0, sizeof(visited));
        if (dfs(u)) result++;
    }
    return result;
}
```

#### 为什么它重要

-   匈牙利算法（Hungarian Algorithm）和 Hopcroft–Karp 算法的基础
-   资源分配、调度、配对问题的核心工具
-   引入了增广路径的概念，这是流理论（flow theory）的基石

应用场景：

-   将工人分配到工作
-   将学生匹配到项目
-   网络流初始化
-   图论教学和可视化

#### 一个温和的证明（为什么它有效）

如果存在一条增广路径，沿着它翻转匹配总是使匹配大小增加 1。

令 $M$ 为当前匹配。
如果不存在增广路径，则 $M$ 是最大的（Berge 引理）：

> 一个匹配是最大匹配，当且仅当不存在增广路径。

因此，重复增广确保收敛到最大匹配。

#### 自己动手试试

1.  画一个每边有 4 个节点的二分图。
2.  手动使用 DFS 寻找增广路径。
3.  跟踪每次增广后哪些顶点是匹配/未匹配的。
4.  当没有增广路径剩余时停止。

#### 测试用例

| 图 | $|U|$ | $|V|$ | 最大匹配 | 步骤 |
|-------|------|------|---------------|--------|
| 完全图 $K_{3,3}$ | 3 | 3 | 3 | 3 次 DFS |
| 链 $u_1v_1, u_2v_2, u_3v_3$ | 3 | 3 | 3 | 3 次 DFS |
| 稀疏图 | 4 | 4 | 2 | 4 次 DFS |

#### 复杂度

| 方面         | 开销     |
| -------------- | -------- |
| 每个顶点的 DFS | $O(E)$   |
| 总计          | $O(VE)$  |
| 空间          | $O(V+E)$ |

这种基于 DFS 的方法为二分图匹配提供了一个直观的基线，后来被 Hopcroft–Karp 算法（$O(E\sqrt{V})$）改进，但对于学习和处理小图来说是完美的。
### 372 Hopcroft–Karp

Hopcroft–Karp 算法是对基于 DFS 的二分图匹配的经典改进。它使用分层 BFS 和 DFS 来并行寻找多条增广路径，减少了冗余搜索，并实现了 $O(E\sqrt{V})$ 的最优运行时间。

#### 我们要解决什么问题？

给定一个二分图 $G = (U, V, E)$，我们希望找到一个最大匹配，即连接 $U$ 和 $V$ 的最大顶点不相交边集。

一个匹配是 $M \subseteq E$ 的一个子集，使得每个顶点最多与 $M$ 中的一条边相关联。该算法寻找最大基数匹配。

#### 工作原理（通俗解释）

与一次只找一条增广路径（如简单的 DFS）不同，Hopcroft–Karp 算法会找到一层最短增广路径，然后同时沿所有这些路径进行增广。这极大地减少了 BFS-DFS 阶段的次数。

核心思想：
- 每个阶段同时通过多条路径增加匹配。
- 每个阶段后，未匹配顶点之间的距离（层数）严格增加。

步骤：
1. 初始化：从空匹配 $M = \emptyset$ 开始。
2. 重复直到不存在增广路径：
   1. BFS 阶段：
      * 从 $U$ 中的未匹配顶点到 $V$ 中的未匹配顶点构建分层图（层级图）。
      * 每层增加一跳。
   2. DFS 阶段：
      * 在此分层图中寻找顶点不相交的增广路径。
      * 同时沿所有这些路径进行增广。
3. 返回总匹配大小。

#### 示例

令 $U = {u_1, u_2, u_3}$, $V = {v_1, v_2, v_3}$，边：
- $u_1 \to v_1, v_2$
- $u_2 \to v_2$
- $u_3 \to v_3$

1. 初始匹配：空
2. BFS 构建层级：
   * 第 0 层：$u_1, u_2, u_3$
   * 第 1 层：$v_1, v_2, v_3$
     所有顶点均可到达。
3. DFS 找到 3 条增广路径：
   * $u_1 \to v_1$, $u_2 \to v_2$, $u_3 \to v_3$
4. 全部增广 → 匹配大小 = 3
   无更多增广路径 → 最大匹配 = 3

#### 微型代码（类 C 伪代码）

```c
vector<int> adj[MAXV];
int pairU[MAXV], pairV[MAXV], dist[MAXV];
int NIL = 0, INF = 1e9;

bool bfs(int U) {
    queue<int> q;
    for (int u = 1; u <= U; u++) {
        if (pairU[u] == NIL) { dist[u] = 0; q.push(u); }
        else dist[u] = INF;
    }
    dist[NIL] = INF;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        if (dist[u] < dist[NIL]) {
            for (int v : adj[u]) {
                if (dist[pairV[v]] == INF) {
                    dist[pairV[v]] = dist[u] + 1;
                    q.push(pairV[v]);
                }
            }
        }
    }
    return dist[NIL] != INF;
}

bool dfs(int u) {
    if (u == NIL) return true;
    for (int v : adj[u]) {
        if (dist[pairV[v]] == dist[u] + 1 && dfs(pairV[v])) {
            pairV[v] = u; pairU[u] = v;
            return true;
        }
    }
    dist[u] = INF;
    return false;
}

int hopcroftKarp(int U) {
    memset(pairU, 0, sizeof(pairU));
    memset(pairV, 0, sizeof(pairV));
    int matching = 0;
    while (bfs(U)) {
        for (int u = 1; u <= U; u++)
            if (pairU[u] == NIL && dfs(u))
                matching++;
    }
    return matching;
}
```

#### 为什么它很重要？

- 高效：$O(E\sqrt{V})$，适用于大型图。
- 是以下领域的基础：
  * 任务分配
  * 资源分配
  * 稳定匹配基础
  * 网络优化

它是竞赛编程和实际系统中解决最大二分图匹配的标准方法。

#### 一个温和的证明（为什么它有效）

每个 BFS-DFS 阶段找到一组最短增广路径。增广后，不再存在更短的路径。

令 $d$ = BFS 中到最近未匹配顶点的距离。每个阶段都会增加最短增广路径的长度，并且阶段数最多为 $O(\sqrt{V})$（Hopcroft–Karp 引理）。

每个 BFS-DFS 的成本为 $O(E)$，因此总成本 = $O(E\sqrt{V})$。

#### 自己动手试试

1. 画一个两边各有 5 个节点的二分图。
2. 运行一次 BFS 层级构建。
3. 使用 DFS 找到所有最短增广路径。
4. 全部增广，跟踪每个阶段的匹配大小。

#### 测试用例

| 图 | $|U|$ | $|V|$ | 匹配大小 | 复杂度 |
|-------|------|------|----------------|-------------|
| $K_{3,3}$ | 3 | 3 | 3 | 快 |
| 链图 | 5 | 5 | 5 | 线性 |
| 稀疏图 | 10 | 10 | 6 | 亚线性阶段 |

#### 复杂度

| 操作           | 成本           |
| ------------------- | -------------- |
| BFS（构建层级）  | $O(E)$         |
| DFS（增广路径） | $O(E)$         |
| 总计               | $O(E\sqrt{V})$ |

Hopcroft–Karp 是二分图匹配的基准，平衡了优雅性、效率和理论深度。
### 373 匈牙利算法

匈牙利算法（也称为 Kuhn–Munkres 算法）解决了分配问题，即在加权二分图中找到最小成本的完美匹配。
它是优化领域的基石，将复杂的分配任务转化为对成本矩阵的优雅线性时间计算。

#### 我们要解决什么问题？

给定一个二分图 $G = (U, V, E)$，其中 $|U| = |V| = n$，以及每条边的成本函数 $c(u,v)$，
我们希望找到一个匹配 $M$，使得：

- 每个 $u \in U$ 恰好匹配一个 $v \in V$
- 总成本最小化：

$$
\text{最小化 } \sum_{(u,v) \in M} c(u, v)
$$

这就是分配问题，它是线性规划的一个特例，可以在多项式时间内精确求解。

#### 工作原理（通俗解释）

匈牙利算法将成本矩阵视为一个网格谜题——
你系统地减少、标记和覆盖行与列，
以揭示一组对应于最优分配的零。

核心思想：转换成本矩阵，使得至少一个最优解位于零元素之中。

步骤：

1. 行归约
   将每行中的最小元素从该行的所有元素中减去。

2. 列归约
   将每列中的最小元素从该列的所有元素中减去。

3. 覆盖零
   使用最少数量的线（水平 + 垂直）来覆盖所有零。

4. 调整矩阵
   如果覆盖线的数量 < $n$：

   * 找到最小的未覆盖值 $m$
   * 从所有未覆盖元素中减去 $m$
   * 将 $m$ 加到被覆盖两次的元素上
   * 从步骤 3 开始重复

5. 分配
   一旦使用了 $n$ 条线，为每行/每列选择一个零 → 那就是最优匹配。

#### 示例

成本矩阵：

|    | v1 | v2 | v3 |
| -- | -- | -- | -- |
| u1 | 4  | 1  | 3  |
| u2 | 2  | 0  | 5  |
| u3 | 3  | 2  | 2  |

1. 行归约：减去每行的最小值
   | u1 | 3 | 0 | 2 |
   | u2 | 2 | 0 | 5 |
   | u3 | 1 | 0 | 0 |

2. 列归约：减去每列的最小值
   | u1 | 2 | 0 | 2 |
   | u2 | 1 | 0 | 5 |
   | u3 | 0 | 0 | 0 |

3. 用 3 条线覆盖零 → 可行。

4. 分配：选择 $(u1,v2)$, $(u2,v1)$, $(u3,v3)$ → 总成本 = $1+2+2=5$

找到最优分配。

#### 微型代码（类 Python 伪代码）

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()
    return list(zip(row_ind, col_ind)), total_cost

# 示例
cost = np.array([[4,1,3],[2,0,5],[3,2,2]])
match, cost = hungarian(cost)
print(match, cost)  # [(0,1), (1,0), (2,2)], 5
```

#### 为什么它很重要

- 精确且高效：$O(n^3)$ 复杂度
- 运筹学和人工智能的基础（任务分配、调度、跟踪）
- 用于：

  * 工作-工人分配
  * 最优资源分配
  * 将预测与真实情况匹配（例如，目标检测中的匈牙利损失）

该算法平衡了组合数学和线性代数，是优雅与实用性的罕见结合。

#### 一个温和的证明（为什么它有效）

该算法在每一步都保持对偶可行性和互补松弛性。
通过减少行和列，我们确保每行和每列至少有一个零，从而创建一个简化成本矩阵，其中的零对应于可行的分配。

每次迭代都更接近相等图（简化成本 = 0 的边）中的完美匹配。
一旦所有顶点都匹配完毕，解就满足最优性条件。

#### 亲自动手试试

1. 创建一个 3×3 或 4×4 的成本矩阵。
2. 手动执行行归约和列归约。
3. 覆盖零并计算线条数。
4. 调整并重复，直到使用 $n$ 条线。
5. 为每行/每列分配一个零。

#### 测试用例

| 矩阵大小 | 成本矩阵类型       | 结果                      |
| ----------- | ------------------ | ----------------------- |
| 3×3         | 随机整数    | 最小成本分配 |
| 4×4         | 对角占优 | 选择对角线         |
| 5×5         | 对称          | 找到匹配对    |

#### 复杂度

| 步骤                  | 时间     |
| --------------------- | -------- |
| 行/列归约 | $O(n^2)$ |
| 迭代覆盖    | $O(n^3)$ |
| 总计                 | $O(n^3)$ |

匈牙利算法将成本矩阵转化为结构，一次一条线地揭示隐藏在零中的最优匹配。
### 374 Kuhn–Munkres（最大权重匹配）

Kuhn–Munkres 算法，也称为用于最大权重匹配的匈牙利算法，解决了最大权重二分图匹配问题。
标准的匈牙利方法是最小化总成本，而这个版本是最大化总奖励或效用，使其成为“越大越好”的分配优化问题的理想选择。

#### 我们要解决什么问题？

给定一个完全二分图 $G = (U, V, E)$，其中 $|U| = |V| = n$，并且每条边上有权重 $w(u, v)$，
找到一个完美匹配 $M \subseteq E$，使得：

$$
\text{最大化 } \sum_{(u,v) \in M} w(u,v)
$$

每个顶点恰好与另一侧的一个顶点匹配，并且总权重尽可能大。

#### 工作原理（通俗解释）

该算法将权重矩阵视为一个利润网格。
它构建顶点上的标号，并维护等式边（即标号之和等于边权重的边）。
通过在这个等式图中构建和增广匹配，算法最终收敛到最优的最大权重匹配。

关键思想：

- 维护满足 $l(u) + l(v) \ge w(u, v)$ 的顶点标号 $l(u)$ 和 $l(v)$（对偶可行性）
- 构建等式成立的等式图
- 在等式图中寻找增广路径
- 当无法前进时更新标号以揭示新的等式边

步骤：

1. 初始化标号

   * 对于每个 $u \in U$，$l(u) = \max_{v} w(u, v)$
   * 对于每个 $v \in V$，$l(v) = 0$

2. 对每个 $u$ 重复：

   * 使用 BFS 构建交错树
   * 如果没有增广路径，则更新标号以揭示新的等式边
   * 沿找到的路径增广匹配

3. 继续直到所有顶点都匹配。

#### 示例

权重：

|    | v1 | v2 | v3 |
| -- | -- | -- | -- |
| u1 | 3  | 2  | 1  |
| u2 | 2  | 4  | 6  |
| u3 | 3  | 5  | 3  |

目标：最大化总权重

最优匹配：

- $u1 \to v1$ (3)
- $u2 \to v3$ (6)
- $u3 \to v2$ (5)

总权重 = $3 + 6 + 5 = 14$

#### 微型代码（类 Python 伪代码）

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

def kuhn_munkres(weight_matrix):
    # 通过取负转换为成本（匈牙利算法解决最小化问题）
    cost = -weight_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    total = weight_matrix[row_ind, col_ind].sum()
    return list(zip(row_ind, col_ind)), total

# 示例
W = np.array([[3,2,1],[2,4,6],[3,5,3]])
match, total = kuhn_munkres(W)
print(match, total)  # [(0,0),(1,2),(2,1)], 14
```

#### 为什么重要

- 在多项式时间内解决最大奖励分配问题
- 在以下领域具有基础性作用：

  * 工作-任务分配
  * 最优配对问题
  * 机器学习（例如匈牙利损失）
  * 博弈论和经济学

许多系统在分配有限资源时使用此算法来最大化整体效率。

#### 一个温和的证明（为什么有效）

- 保持对偶可行性：$l(u) + l(v) \ge w(u, v)$
- 保持互补松弛性：匹配边满足等式
- 通过在等式图扩展和标号调整之间交替更新，算法确保了最终的可达性和最优性
- 最终匹配满足强对偶性，实现了最大权重

#### 自己动手试试

1. 构建一个 $3 \times 3$ 的利润矩阵。
2. 初始化 $l(u)$ 为行最大值，$l(v)=0$。
3. 画出等式边（$l(u)+l(v)=w(u,v)$）。
4. 寻找增广路径 → 增广 → 更新标号。
5. 重复直到匹配完美。

#### 测试用例

| 图   | 大小           | 最大权重   | 匹配            |
| ---- | -------------- | ---------- | --------------- |
| 3×3  | 随机权重       | 14         | [(0,0),(1,2),(2,1)] |
| 4×4  | 对角线值高     | sum(diag)  | 对角线          |
| 5×5  | 均匀           | 5×max weight | 任意          |

#### 复杂度

| 步骤            | 时间     |
| --------------- | -------- |
| 标号更新        | $O(V^2)$ |
| 匹配阶段        | $O(V)$   |
| 总计            | $O(V^3)$ |

Kuhn–Munkres 算法是解决最大权重分配问题的终极工具，它将几何、对偶性和组合数学融合成一个强大的优化引擎。
### 375 开花算法

开花算法由 Jack Edmonds 提出，是用于在一般图（包括非二分图）中寻找最大匹配的基础算法。它引入了"花"（奇数长度环）的概念，并证明了最大匹配问题可以在多项式时间内解决，这是组合优化领域的一个里程碑。

#### 我们要解决什么问题？

给定一个一般图（不一定是二分图）$G = (V, E)$，寻找一个最大匹配，即一个最大的边集，使得没有两条边共享一个顶点。

与二分图不同，奇数长度环可能会阻碍标准增广路径搜索的进展。开花算法通过收缩这些环，使得仍然可以找到增广路径。

形式化地说，寻找 $M \subseteq E$ 以最大化 $|M|$，使得每个顶点最多与 $M$ 中的一条边相关联。

#### 算法如何工作（通俗解释）

该算法扩展了增广路径方法（用于二分图匹配），增加了一个强大的思想：当遇到奇数长度环时，将其视为一个单一的顶点，即一朵"花"。

步骤：

1.  初始化一个空匹配 $M = \emptyset$。
2.  在交错树中使用 BFS/DFS 搜索增广路径：
    *   在匹配边和未匹配边之间交替搜索。
    *   如果到达一个自由顶点（未匹配顶点），则进行增广（翻转匹配/未匹配边）。
3.  如果遇到一个奇数环（一朵花），将其收缩为一个超级顶点。
    *   在收缩后的图中继续搜索。
    *   一旦找到增广路径，展开花并更新匹配。
4.  重复上述步骤，直到不存在增广路径。

每次增广操作使 $|M|$ 增加 1。当没有增广路径剩余时，匹配即为最大匹配。

#### 示例

图：

*   顶点：${A, B, C, D, E}$
*   边：$(A, B), (B, C), (C, A), (B, D), (C, E)$

1.  初始匹配：空
2.  构建交错树：$A \to B \to C \to A$ 形成一个奇数环
3.  将花 $(A, B, C)$ 收缩为一个节点
4.  继续搜索 → 找到通过花的增广路径
5.  展开花，调整匹配
6.  结果：最大匹配包含边 $(A,B), (C,E), (D,...)$

#### 微型代码（类 Python 伪代码）

```python
# 为简单起见使用 networkx
import networkx as nx

def blossom_maximum_matching(G):
    return nx.max_weight_matching(G, maxcardinality=True)

# 示例
G = nx.Graph()
G.add_edges_from([(0,1),(1,2),(2,0),(1,3),(2,4)])
match = blossom_maximum_matching(G)
print(match)  # {(0,1), (2,4)}
```

#### 为何重要

*   一般图（非二分图）在现实世界的系统中很常见：
    *   社交网络（好友配对）
    *   分子结构匹配
    *   带约束的调度
    *   图论证明和优化

开花算法证明了匹配问题是多项式时间可解的，这是算法理论中的一个关键里程碑。

#### 一个温和的证明（为何有效）

根据 Berge 引理，一个匹配是最大匹配当且仅当不存在增广路径。挑战在于增广路径可能隐藏在奇数环内部。

花的收缩确保了收缩图中的每一条增广路径都对应于原图中的一条增广路径。每次增广后，$|M|$ 严格增加，因此算法在多项式时间内终止。

正确性源于保持了以下性质：

*   具有一致奇偶性的交错树
*   收缩不变性（增广路径的保持）
*   在收缩和展开过程中满足 Berge 条件

#### 亲自尝试

1.  画一个三角形图（3-环）。
2.  运行增广路径搜索，找到花。
3.  将其收缩为一个节点。
4.  继续搜索，找到路径后展开。
5.  验证最大匹配。

#### 测试用例

| 图类型                   | 匹配大小 | 方法                |
| ------------------------ | -------- | ------------------- |
| 三角形                   | 1        | 花收缩              |
| 带对角线的正方形         | 2        | 增广                |
| 五边形奇数环             | 2        | 花                  |
| 二分图（完整性检查）     | 通常情况 | 匹配 Hopcroft–Karp 算法 |

#### 复杂度

| 阶段                     | 时间复杂度 |
| ------------------------ | ---------- |
| 搜索（每次增广）         | $O(VE)$    |
| 增广次数                 | $O(V)$     |
| 总计                     | $O(V^3)$   |

开花算法是一个启示，它展示了如何驯服奇数环，并将匹配问题扩展到所有图，从而架起了组合数学和优化理论之间的桥梁。
### 376 Edmonds 的花收缩算法

Edmonds 的花收缩算法是支撑花算法的核心子程序，使得在非二分图中进行增广路径搜索成为可能。它提供了收缩奇数长度环（花）的关键机制，从而能够揭示并利用隐藏的增广路径。

#### 我们要解决什么问题？

在非二分图中，增广路径可能被奇数长度的环所掩盖。标准的匹配算法会失败，因为它们假设了二分图结构。

我们需要一种在搜索过程中处理奇数环的方法，以便算法能够在不遗漏有效增广的情况下继续推进。

目标：
检测花，将其收缩为单个顶点，并高效地继续搜索。

给定图 $G = (V, E)$ 中的一个匹配 $M$，以及在搜索过程中生长的一棵交错树，当发现一个花时：

- 将花收缩为一个超级顶点
- 在收缩后的图中继续搜索
- 当找到增广路径时，展开花

#### 工作原理（通俗解释）

想象从一个自由顶点运行 BFS/DFS。你在匹配边和未匹配边之间交替，以构建一棵交错树。

如果你发现一条连接两个相同层级（深度均为偶数）的顶点的边：

- 你就检测到了一个奇数环。
- 这个环就是一个花。

处理方法如下：

1.  识别花（奇数长度的交错环）
2.  将其所有顶点收缩为一个超级节点
3.  在这个收缩后的图上继续增广路径搜索
4.  一旦发现增广路径，展开花并相应地调整路径
5.  沿着展开后的路径进行增广

这种收缩保持了所有有效的增广路径，并允许算法像花是一个顶点一样进行操作。

#### 示例

图：
顶点：${A, B, C, D, E}$
边：$(A,B), (B,C), (C,A), (B,D), (C,E)$

假设 $A, B, C$ 在搜索过程中被发现形成了一个奇数环。

1.  检测到花：$(A, B, C)$
2.  收缩为单个顶点 $X$
3.  在简化后的图上继续搜索
4.  如果增广路径经过 $X$，则将其展开
5.  在花内交替边以正确整合路径

结果：找到一条有效的增广路径，匹配数增加一。

#### 微型代码（类 Python）

```python
def find_blossom(u, v, parent):
    # 寻找最近公共祖先
    path_u, path_v = set(), set()
    while u != -1:
        path_u.add(u)
        u = parent[u]
    while v not in path_u:
        path_v.add(v)
        v = parent[v]
    lca = v
    # 收缩花（概念上）
    return lca
```

在实际实现中（如 Edmonds 算法），此逻辑会将花中的所有节点合并为一个节点，并调整父子关系。

#### 为什么它很重要？

这是通用图匹配的核心。没有收缩，搜索可能会无限循环或无法检测到有效的增广。

花收缩使得以下成为可能：

-   处理奇数长度环
-   保持增广路径的不变性
-   保证多项式时间行为

它也是组合优化中最早使用图收缩的案例之一。

#### 一个温和的证明（为什么它有效）

根据 Berge 引理，当且仅当不存在增广路径时，匹配是最大匹配。如果增广路径存在于 $G$ 中，那么它也存在于 $G$ 的任何收缩版本中。

收缩一个花保持了可增广性：

-   原始图中的每条增广路径都对应于收缩图中的一条增广路径。
-   展开后，交错模式被正确地恢复。

因此，收缩不会丢失信息，它只是简化了搜索。

#### 自己动手试试

1.  画一个三角形 $A, B, C$，连接到其他顶点 $D, E$。
2.  从一个自由顶点开始构建一棵交错树。
3.  当你发现一条连接两个偶数层级顶点的边时，标记出奇数环。
4.  收缩花，继续搜索，然后在找到路径后展开。

观察这如何允许发现先前隐藏的增广路径。

#### 测试用例

| 图               | 描述             | 收缩后可增广？ |
| ---------------- | ---------------- | -------------- |
| 三角形           | 单个奇数环       | 是             |
| 三角形加尾巴     | 环加路径         | 是             |
| 二分图           | 无奇数环         | 无需收缩       |
| 五边形           | 长度为 5 的花    | 是             |

#### 复杂度

收缩操作可以在与花大小成线性关系的时间内完成。当集成到完整的匹配搜索中时，总算法复杂度保持为 $O(V^3)$。

Edmonds 的花收缩算法是一个概念上的飞跃，它通过巧妙的收缩和展开，将通用图中最大匹配问题从一个充满难以处理的环的迷宫，转变为一个可解的结构。
### 377 贪心匹配

贪心匹配是近似图最大匹配的最简单方法。它不探索增广路径，只是不断选取不冲突的可用边，为匹配问题提供了一个快速、直观的基准。

#### 我们要解决什么问题？

在许多实际场景中，如工作分配、用户配对、调度等，我们需要一组边，使得没有两条边共享一个顶点。这就是一个匹配。

精确找到最大匹配（尤其是在一般图中）可能代价高昂。但有时，我们只需要快速得到一个足够好的答案。

贪心匹配算法提供了一种快速的近似方案：

- 它不总能找到最大的匹配
- 但它运行时间为 $O(E)$，并且通常能给出一个不错的解

目标：快速构建一个极大匹配（无法再添加任何边）。

#### 工作原理（通俗解释）

从一个空集合 $M$（匹配）开始。逐条检查边：

1. 对于每条边 $(u, v)$
2. 如果 $u$ 和 $v$ 都尚未匹配
3. 将 $(u, v)$ 加入 $M$

持续检查直到所有边都被处理完毕。

这确保了没有顶点出现在超过一条边中，从而得到一个有效的匹配。结果是极大的：在不破坏匹配规则的前提下，你无法添加任何其他边。

#### 示例

图的边：
$$
E = {(A,B), (B,C), (C,D), (D,E)}
$$

- 开始：$M = \emptyset$
- 选取 $(A,B)$ → 标记 $A$, $B$ 为已匹配
- 跳过 $(B,C)$ → $B$ 已匹配
- 选取 $(C,D)$ → 标记 $C$, $D$
- 跳过 $(D,E)$ → $D$ 已匹配

结果：
$$
M = {(A,B), (C,D)}
$$

#### 微型代码（类 Python）

```python
def greedy_matching(graph):
    matched = set() # 已匹配的顶点集合
    matching = []   # 匹配结果列表
    for u, v in graph.edges:
        if u not in matched and v not in matched:
            matching.append((u, v))
            matched.add(u)
            matched.add(v)
    return matching
```

这个简单的循环在边数上是线性时间运行的。

#### 为什么它重要

- **快速**：运行时间为 $O(E)$
- **简单**：易于实现
- **有用的基准**：是启发式或混合方法的良好起点
- **保证极大性**：在不破坏匹配条件的情况下无法添加任何边

虽然它不是最优的，但其结果大小至少是最大匹配的一半：
$$
|M_{\text{greedy}}| \ge \frac{1}{2}|M_{\text{max}}|
$$

#### 一个温和的证明（为什么它有效）

每条贪心选取的边 $(u,v)$ 最多会阻止 $M_{\text{max}}$ 中的一条边（因为 $u$ 和 $v$ 都被使用了）。所以，对于每一条被选中的边，我们最多从最优解中损失一条边。因此：
$$
2|M_{\text{greedy}}| \ge |M_{\text{max}}|
$$
⇒ 贪心算法实现了 1/2 近似比。

#### 亲自尝试

1.  创建一个有 6 个顶点和 7 条边的图。
2.  以不同的边顺序运行贪心匹配。
3.  观察结果如何不同，顺序会影响最终集合。
4.  与精确的最大匹配（例如 Hopcroft–Karp 算法）进行比较。

你会发现早期的简单决策如何影响最终结果的大小。

#### 测试用例

| 图类型          | 边                          | 贪心匹配大小 | 最大匹配大小 |
| --------------- | --------------------------- | ------------ | ------------ |
| 4 个顶点的路径  | $(A,B),(B,C),(C,D)$         | 2            | 2            |
| 三角形          | $(A,B),(B,C),(C,A)$         | 1            | 1            |
| 正方形          | $(A,B),(B,C),(C,D),(D,A)$   | 2            | 2            |
| 星形（5 个叶子） | $(C,1)...(C,5)$             | 1            | 1            |

#### 复杂度

- 时间：$O(E)$
- 空间：$O(V)$

适用于任何无向图。对于有向图，边必须被视为无向或对称的。

当你需要速度而非完美时，贪心匹配是一种快速而实用的方法，它是一种简单的“握手”策略，在时间耗尽前尽可能多地配对。
### 378 稳定婚姻问题（盖尔-沙普利算法）

稳定婚姻问题（或称稳定匹配问题）是组合优化领域的基石，它要求将两个大小相等的集合（例如男性和女性、职位与申请人、医院与住院医师）进行配对，使得不存在任意两个参与者都更倾向于彼此而非当前分配伴侣的情况。盖尔-沙普利算法能在 $O(n^2)$ 时间内找到这样一个稳定匹配。

#### 我们要解决什么问题？

给定两个集合：

- 集合 $A = {a_1, a_2, ..., a_n}$
- 集合 $B = {b_1, b_2, ..., b_n}$

每个成员都按照偏好顺序对另一个集合的所有成员进行排名。
我们希望找到一个匹配（一一对应的配对），使得：

不存在这样的配对 $(a_i, b_j)$，其中：

- $a_i$ 比起当前匹配对象更偏好 $b_j$，并且
- $b_j$ 比起当前匹配对象更偏好 $a_i$。

这样的配对将是不稳定的，因为他们会更愿意彼此匹配。

我们的目标：找到一个稳定的配置，使得参与者没有动机交换伴侣。

#### 算法如何工作（通俗解释）

盖尔-沙普利算法采用"求婚"和"拒绝"的机制：

1. 所有 $a_i$ 初始状态为未匹配。
2. 当存在某个 $a_i$ 是自由的（未匹配）时：

   * $a_i$ 向其尚未求过婚的、最偏好的 $b_j$ 求婚。
   * 如果 $b_j$ 是自由的，则接受求婚。
   * 如果 $b_j$ 已匹配，但比起当前伴侣更偏好 $a_i$，则她"升级"并拒绝旧的伴侣。
   * 否则，她拒绝 $a_i$。
3. 继续此过程，直到所有人都匹配完毕。

该过程总会终止，并得到一个稳定匹配。

#### 示例

假设我们有：

| A  | 偏好列表 |
| -- | --------------- |
| A1 | B1, B2, B3      |
| A2 | B2, B1, B3      |
| A3 | B3, B1, B2      |

| B  | 偏好列表 |
| -- | --------------- |
| B1 | A2, A1, A3      |
| B2 | A1, A2, A3      |
| B3 | A1, A2, A3      |

逐步过程：

- A1 → B1，B1 自由 → 匹配 (A1, B1)
- A2 → B2，B2 自由 → 匹配 (A2, B2)
- A3 → B3，B3 自由 → 匹配 (A3, B3)

稳定匹配：{(A1, B1), (A2, B2), (A3, B3)}

不存在任意两者更倾向于彼此而非当前伴侣 → 稳定。

#### 微型代码（类 Python）

```python
def gale_shapley(A_prefs, B_prefs):
    free_A = list(A_prefs.keys())
    engaged = {}
    next_choice = {a: 0 for a in A_prefs}

    while free_A:
        a = free_A.pop(0)
        b = A_prefs[a][next_choice[a]]
        next_choice[a] += 1

        if b not in engaged:
            engaged[b] = a
        else:
            current = engaged[b]
            if B_prefs[b].index(a) < B_prefs[b].index(current):
                engaged[b] = a
                free_A.append(current)
            else
### 379 加权 b-匹配

加权 b-匹配通过允许每个节点最多与 $b(v)$ 个伙伴匹配（而不仅仅是一个）来推广标准匹配。当边带有权重时，目标是找到一个具有最大权重的边子集，使得每个顶点的度数不超过其容量 $b(v)$。

#### 我们要解决什么问题？

给定一个图 $G = (V, E)$，其中包含：

- 每条边 $e \in E$ 的边权重 $w(e)$
- 每个顶点 $v \in V$ 的容量约束 $b(v)$

找到一个子集 $M \subseteq E$，使得：

- 最大化总权重
  $$W(M) = \sum_{e \in M} w(e)$$
- 满足度数约束
  $$\forall v \in V,\quad \deg_M(v) \le b(v)$$

如果所有 $v$ 的 $b(v) = 1$，这就变成了标准的最大权重匹配问题。

#### 工作原理（通俗解释）

将每个顶点 $v$ 视为有 b(v) 个可用的连接槽位。我们希望选择边来填充这些槽位，以最大化总边权重，同时不超过任何顶点的限制。

加权 b-匹配问题可以通过以下方法解决：

1.  **归约为流问题**：将匹配问题转换为网络流问题：
    *   每条边变为一个容量为 1、成本为 $-w(e)$ 的流连接。
    *   顶点容量变为流量限制。
    *   通过最小费用最大流算法求解。
2.  **线性规划**：松弛约束，并用线性规划或原始-对偶算法求解。
3.  **近似算法**：对于大型稀疏图，贪心启发式算法可以获得接近最优的解。

#### 示例

假设：

- 顶点：$V = {A, B, C}$
- 边及权重：
  $w(A,B)=5,; w(A,C)=4,; w(B,C)=3$
- 容量：$b(A)=2,; b(B)=1,; b(C)=1$

我们可以选择：

- $(A,B)$ 权重 5
- $(A,C)$ 权重 4

总权重 = $9$
这是有效的，因为 $\deg(A)=2,\deg(B)=1,\deg(C)=1$

如果 $b(A)=1$，我们只能选择 $(A,B)$ → 权重 $5$

#### 微型代码（类 Python 伪代码）

```python
import networkx as nx

def weighted_b_matching(G, b):
    # 转换为流网络
    flow_net = nx.DiGraph()
    source, sink = 's', 't'

    for v in G.nodes:
        flow_net.add_edge(source, v, capacity=b[v], weight=0)
        flow_net.add_edge(v, sink, capacity=b[v], weight=0)

    for (u, v, data) in G.edges(data=True):
        w = -data['weight']  # 取负值用于最小费用
        flow_net.add_edge(u, v, capacity=1, weight=w)
        flow_net.add_edge(v, u, capacity=1, weight=w)

    flow_dict = nx.max_flow_min_cost(flow_net, source, sink)
    # 提取顶点对之间流量为1的边
    return [(u, v) for u, nbrs in flow_dict.items() for v, f in nbrs.items() if f > 0 and u in G.nodes and v in G.nodes]
```

#### 为什么它很重要

- 为带有限制的资源分配建模（例如，每个工人可以处理 $b$ 个任务）。
- 将经典匹配扩展到具有容量的网络。
- 是以下应用的基础：
  *   带有配额的任务分配
  *   具有多容量的调度
  *   带有度数约束的聚类

#### 一个温和的证明（为什么它有效）

每个顶点 $v$ 具有度数约束 $b(v)$：
$$\sum_{e \ni v} x_e \le b(v)$$

每条边最多被选择一次：
$$x_e \in {0, 1}$$

优化目标：
$$\max \sum_{e \in E} w(e) x_e$$

受限于上述约束，形成了一个整数线性规划。通过网络流进行松弛，对于二分图可以确保得到最优的整数解。

#### 自己动手试试

1.  构建一个具有不同边权重的三角形图。
2.  将一个顶点的 $b(v)$ 设为 2，其他顶点设为 1。
3.  手动计算哪些边能产生最大权重。
4.  使用流求解器实现并验证。

#### 测试用例

| 图               | 容量         | 结果                | 权重               |
| ---------------- | ------------ | ------------------- | ------------------ |
| 三角形 (5,4,3)   | {2,1,1}      | {(A,B),(A,C)}       | 9                  |
| 正方形           | {1,1,1,1}    | 标准匹配            | 最大权重和         |
| 线 (A-B-C)       | {2,1,2}      | {(A,B),(B,C)}       | 两条边的权重和     |

#### 复杂度

- 时间复杂度：使用最小费用最大流为 $O(V^3)$
- 空间复杂度：$O(V + E)$

对于中等规模的图是高效的；对于更大的图，可以通过启发式方法进行扩展。

加权 b-匹配在最优性和灵活性之间取得了平衡，允许比一对一配对更丰富的分配模型。
### 380 极大匹配

极大匹配是一种贪心方法，它通过逐一添加边来构建匹配，直到在不破坏匹配条件的情况下无法再添加更多边为止。
与*最大匹配*（最大化规模或权重）不同，*极大*匹配仅确保无法再添加边，它是局部最优的，但不一定是全局最优的。

#### 我们要解决什么问题？

我们希望选择一个边集 $M \subseteq E$，使得：

1. $M$ 中任意两条边不共享顶点
   $$\forall (u,v), (x,y) \in M,; {u,v} \cap {x,y} = \emptyset$$
2. $M$ 是极大的：在不违反条件 (1) 的情况下，无法从 $E$ 中添加任何额外的边。

目标：找到任意一个极大匹配，不一定是规模最大的那个。

#### 工作原理（通俗解释）

该算法是贪心且简单的：

1. 从一个空的匹配 $M = \emptyset$ 开始
2. 遍历所有边 $(u,v)$
3. 如果 $u$ 和 $v$ 都尚未匹配，则将 $(u,v)$ 加入 $M$
4. 继续处理，直到所有边都被处理完毕

最终，$M$ 是极大的：每条未匹配的边都至少与一个已匹配的顶点相连。

#### 示例

图：

```
A -- B -- C
|         
D
```

边：$(A,B)$, $(B,C)$, $(A,D)$

逐步过程：

- 选取 $(A,B)$ → 标记 A, B 已匹配
- 跳过 $(B,C)$ (B 已匹配)
- 跳过 $(A,D)$ (A 已匹配)

结果：$M = {(A,B)}$

另一个有效的极大匹配：${(A,D), (B,C)}$
匹配不唯一，但所有极大集都共享一个属性：无法再添加更多边。

#### 微型代码（Python）

```python
def maximal_matching(edges):
    matched = set() # 已匹配的顶点集合
    M = [] # 匹配结果列表
    for u, v in edges:
        if u not in matched and v not in matched:
            M.append((u, v))
            matched.add(u)
            matched.add(v)
    return M

edges = [('A','B'), ('B','C'), ('A','D')]
print(maximal_matching(edges))  # [('A','B')]
```

#### 为何重要

- 快速近似：极大匹配是最大匹配的 2-近似（其规模至少是最大匹配的一半）。
- 构建模块：用于分布式和并行算法中，在这些场景下全局优化代价过高。
- 预处理步骤：在流问题、调度和网络设计中用于缩减问题规模。

#### 一个温和的证明（为何有效）

如果一条边 $(u,v)$ 不在 $M$ 中，那么它的至少一个端点已经被匹配。
因此，添加 $(u,v)$ 将会破坏匹配条件。
这确保了极大性，即无法再安全地添加任何边。

形式化地，对于每条 $(u,v) \notin M$：
$$u \in V_M \lor v \in V_M$$
其中 $V_M$ 是已匹配顶点的集合。

因此，$M$ 是极大的。

#### 亲自尝试

1.  画一个有 4-6 个顶点的小图。
2.  以不同的边顺序应用贪心过程。
3.  比较得到的匹配，注意它们的规模可能不同，但都是极大的。
4.  尝试添加一条额外的边，看看它为何违反匹配规则。

#### 测试用例

| 图               | 边                             | 极大匹配          | 规模 |
| ---------------- | ------------------------------ | ----------------- | ---- |
| 路径 A–B–C       | {(A,B),(B,C)}                  | {(A,B)}           | 1    |
| 星型 A–B,C,D     | {(A,B),(A,C),(A,D)}            | {(A,B)}           | 1    |
| 正方形 A–B–C–D   | {(A,B),(B,C),(C,D),(D,A)}      | {(A,B),(C,D)}     | 2    |

#### 复杂度

- 时间：$O(E)$，单次遍历所有边
- 空间：$O(V)$，用于跟踪顶点匹配状态

对于大型图，当最优匹配代价过高时，此方法快速、简单且有效。

极大匹配为你提供了一种快速且安全的配对方式，虽不总是完美，但总是*已完成*。

## 第 39 章 树算法
### 381 欧拉序 LCA

欧拉序 LCA 通过记录树的欧拉遍历，将寻找最近公共祖先（LCA）的问题转化为区间最小值查询（RMQ）问题。
它优雅、可预处理且快速，非常适合处理多个 LCA 查询。

#### 我们要解决什么问题？

给定一棵有根树和两个节点 $u$ 和 $v$，我们想要找到它们的最低公共祖先，即同时是两者祖先的最深节点。

例如，在一棵以 $1$ 为根的树中：

```
      1
    /   \
   2     3
  / \   /
 4  5  6
```

$(4,5)$ 的 LCA 是 $2$，$(4,6)$ 的 LCA 是 $1$。

我们希望高效地回答许多这样的查询。

#### 工作原理（通俗解释）

1.  执行一次 DFS 遍历（欧拉遍历）：
    *   在进入每个节点时记录它。
    *   当回溯时，再次记录它。
2.  存储每次访问节点时的深度。
3.  对于每个节点，记住它在欧拉遍历中第一次出现的位置。
4.  对于查询 $(u, v)$：
    *   找到 $u$ 和 $v$ 在遍历中第一次出现的位置。
    *   取它们之间的区间。
    *   该区间内深度最小的节点就是 LCA。

因此，LCA 问题被简化为对深度数组的 RMQ 问题。

#### 示例

树（根 = 1）：

```
1
├── 2
│   ├── 4
│   └── 5
└── 3
    └── 6
```

欧拉遍历：
$[1,2,4,2,5,2,1,3,6,3,1]$

深度数组：
$[0,1,2,1,2,1,0,1,2,1,0]$

首次出现位置：
$1\to0,;2\to1,;3\to7,;4\to2,;5\to4,;6\to8$

查询：LCA(4,5)

- First(4) = 2, First(5) = 4
- 深度区间 [2..4] = [2,1,2]
- 最小深度 = 1 → 节点 = 2

所以 LCA(4,5) = 2。

#### 精简代码（Python）

```python
def euler_tour_lca(graph, root):
    n = len(graph)
    euler, depth, first = [], [], {}
    
    def dfs(u, d):
        first.setdefault(u, len(euler))
        euler.append(u)
        depth.append(d)
        for v in graph[u]:
            dfs(v, d + 1)
            euler.append(u)
            depth.append(d)
    
    dfs(root, 0)
    return euler, depth, first

def build_rmq(depth):
    n = len(depth)
    log = [0] * (n + 1)
    for i in range(2, n + 1):
        log[i] = log[i // 2] + 1
    k = log[n]
    st = [[0] * (k + 1) for _ in range(n)]
    for i in range(n):
        st[i][0] = i
    j = 1
    while (1 << j) <= n:
        i = 0
        while i + (1 << j) <= n:
            left = st[i][j - 1]
            right = st[i + (1 << (j - 1))][j - 1]
            st[i][j] = left if depth[left] < depth[right] else right
            i += 1
        j += 1
    return st, log

def query_lca(u, v, euler, depth, first, st, log):
    l, r = first[u], first[v]
    if l > r:
        l, r = r, l
    j = log[r - l + 1]
    left = st[l][j]
    right = st[r - (1 << j) + 1][j]
    return euler[left] if depth[left] < depth[right] else euler[right]
```

#### 为什么它很重要

- 将 LCA 问题转化为 RMQ 问题，允许在 $O(n \log n)$ 预处理后实现 $O(1)$ 查询
- 易于与线段树、稀疏表或笛卡尔树结合使用
- 对于以下情况至关重要：

  *   涉及祖先关系的树形动态规划
  *   距离查询：
    $$\text{dist}(u,v) = \text{depth}(u) + \text{depth}(v) - 2 \times \text{depth}(\text{LCA}(u,v))$$
  *   轻重链分解中的路径查询

#### 一个温和的证明（为什么它有效）

在 DFS 遍历中：

- 每个祖先都会在其后代之前和之后出现。
- 在两个节点的首次出现位置之间，第一个重新出现的公共祖先就是它们的 LCA。

因此，首次出现位置之间深度最小的节点恰好对应着最低公共祖先。

#### 自己动手试试

1.  构建一棵树并手动执行欧拉遍历。
2.  写下深度数组和首次出现位置数组。
3.  选取节点对 $(u,v)$，找到它们对应区间内的最小深度。
4.  验证 LCA 的正确性。

#### 测试用例

| 查询       | 预期结果 | 原因                     |
| -------- | -------- | ---------------------- |
| LCA(4,5) | 2        | 两者都在节点 2 之下           |
| LCA(4,6) | 1        | 根节点是公共祖先              |
| LCA(2,3) | 1        | 位于不同的子树               |
| LCA(6,3) | 3        | 其中一个节点是另一个节点的祖先 |

#### 复杂度

- 预处理：$O(n \log n)$
- 查询：$O(1)$
- 空间：$O(n \log n)$

欧拉序 LCA 展示了树问题如何转化为数组问题，是算法中结构转换的一个完美例子。
### 382 二进制提升 LCA

二进制提升是一种快速而优雅的技术，用于在树中查找两个节点的最近公共祖先（LCA），它通过预计算到 2 的幂次祖先的跳跃来实现。它将祖先查询转化为简单的位运算，使得每次查询的时间复杂度为 $O(\log n)$。

#### 我们要解决什么问题？

给定一棵有根树和两个节点 $u$ 和 $v$，我们希望找到它们的最近公共祖先（LCA），即同时是两者祖先的最深节点。

朴素的方法是每次向上走一个节点，但这样每次查询需要 $O(n)$ 的时间。使用二进制提升，我们可以指数级地向上跳跃，将成本降低到 $O(\log n)$。

#### 工作原理（通俗解释）

二进制提升为每个节点预计算：

$$\text{up}[v][k] = \text{节点 } v \text{ 的第 } 2^k \text{ 个祖先}$$

因此，$\text{up}[v][0]$ 是父节点，$\text{up}[v][1]$ 是祖父节点，$\text{up}[v][2]$ 是向上 4 层的祖先，依此类推。

该算法分为三个步骤：

1.  预处理：
    *   从根节点运行 DFS。
    *   记录每个节点的深度。
    *   填充表 `up[v][k]`。

2.  深度对齐：
    *   如果一个节点更深，则将其提升到与较浅节点相同的深度。

3.  共同提升：
    *   同时向上跳跃两个节点（从最大幂次开始），直到它们的祖先相同。

相遇点就是 LCA。

#### 示例

树结构：

```
      1
    /   \
   2     3
  / \   /
 4  5  6
```

二进制提升表 (`up[v][k]`)：

| v | up[v][0] | up[v][1] | up[v][2] |
| - | -------- | -------- | -------- |
| 1 | -        | -        | -        |
| 2 | 1        | -        | -        |
| 3 | 1        | -        | -        |
| 4 | 2        | 1        | -        |
| 5 | 2        | 1        | -        |
| 6 | 3        | 1        | -        |

查询：LCA(4,5)

- 深度(4)=2，深度(5)=2
- 同时提升 4 和 5 → 父节点 (2,2) 匹配 → LCA = 2

查询：LCA(4,6)

- 深度(4)=2，深度(6)=2
- 提升 4→2，6→3 → 不相等
- 提升 2→1，3→1 → LCA = 1

#### 精简代码（Python）

```python
LOG = 20  # 足够支持 n 约到 1e6

def preprocess(graph, root):
    n = len(graph)
    up = [[-1] * LOG for _ in range(n)]
    depth = [0] * n

    def dfs(u, p):
        up[u][0] = p
        for k in range(1, LOG):
            if up[u][k-1] != -1:
                up[u][k] = up[up[u][k-1]][k-1]
        for v in graph[u]:
            if v != p:
                depth[v] = depth[u] + 1
                dfs(v, u)
    dfs(root, -1)
    return up, depth

def lca(u, v, up, depth):
    if depth[u] < depth[v]:
        u, v = v, u
    diff = depth[u] - depth[v]
    for k in range(LOG):
        if diff & (1 << k):
            u = up[u][k]
    if u == v:
        return u
    for k in reversed(range(LOG)):
        if up[u][k] != up[v][k]:
            u = up[u][k]
            v = up[v][k]
    return up[u][0]
```

#### 为什么它重要

-   能高效处理具有大量查询的大型树
-   支持祖先跳跃、第 k 个祖先查询、距离查询等
-   是竞赛编程、树形动态规划和路径查询的基础

#### 一个温和的证明（为什么它有效）

如果 $u$ 和 $v$ 深度不同，将较深的节点提升 $2^k$ 步可以确保我们快速对齐它们的深度。

一旦对齐，同时提升两个节点可以确保我们永远不会跳过 LCA，因为只有在祖先不同时才会进行跳跃。

最终，两者会在它们最低的共享祖先处相遇。

#### 亲自尝试

1.  构建一个包含 7 个节点的小树。
2.  手动计算 $\text{up}[v][k]$。
3.  选择节点对并逐步模拟提升过程。
4.  与朴素的到根路径比较法进行验证。

#### 测试用例

| 查询       | 预期结果 | 原因               |
| -------- | -------- | ------------------ |
| LCA(4,5) | 2        | 相同的父节点       |
| LCA(4,6) | 1        | 跨越不同子树       |
| LCA(2,3) | 1        | 根是祖先           |
| LCA(6,3) | 3        | 其中一个节点是祖先 |

#### 复杂度

-   预处理：$O(n \log n)$
-   查询：$O(\log n)$
-   空间：$O(n \log n)$

二进制提升将树中的祖先关系转化为按位跳跃，提供了一条在对数时间内从根到祖先的清晰路径。
### 383 Tarjan 的 LCA（离线 DSU）

Tarjan 的 LCA 算法使用带路径压缩的并查集（DSU）离线回答多个 LCA 查询。
它遍历树一次（DFS）并合并子树，在 $O(n + q \alpha(n))$ 时间内回答所有查询。

#### 我们要解决什么问题？

给定一棵有根树和多个查询 $(u, v)$，我们想要求出每对节点的最近公共祖先。

与二分倍增（在线算法）不同，Tarjan 的算法是离线的：

- 我们预先知道所有查询。
- 我们使用一个并查集结构在一次 DFS 遍历中回答它们。

#### 工作原理（通俗解释）

我们自底向上处理这棵树：

1. 从根节点开始进行 DFS。
2. 每个节点在 DSU 中初始化为自己的集合。
3. 访问完一个子节点后，将其与其父节点合并。
4. 当一个节点被完全处理时，将其标记为已访问。
5. 对于每个查询 $(u,v)$：

   * 如果另一个节点已被访问，
     那么 $\text{find}(v)$（或 $\text{find}(u)$）就给出了 LCA。

这之所以有效，是因为 DSU 结构合并了所有已处理的祖先，
所以 $\text{find}(v)$ 始终指向两者共有的、已处理的最低祖先。

#### 示例

树（根 = 1）：

```
1
├── 2
│   ├── 4
│   └── 5
└── 3
    └── 6
```

查询：
$(4,5), (4,6), (3,6)$

处理过程：

- DFS(1)：访问 2，访问 4

  * 查询(4,5)：5 未访问 → 跳过
  * 回溯 → union(4,2)
- DFS(5)：标记为已访问，union(5,2)

  * 查询(4,5)：4 已访问 → $\text{find}(4)=2$ → LCA(4,5)=2
- DFS(3)，访问 6 → union(6,3)

  * 查询(4,6)：4 已访问，$\text{find}(4)=2$，$\text{find}(6)=3$ → 尚无 LCA
  * 查询(3,6)：两者都已访问 → $\text{find}(3)=3$ → LCA(3,6)=3
- 回溯 2→1，3→1 → LCA(4,6)=1

#### 精简代码（Python）

```python
from collections import defaultdict

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)

def tarjan_lca(tree, root, queries):
    n = len(tree)
    dsu = DSU(n)
    visited = [False] * n
    ancestor = [0] * n
    ans = {}
    qmap = defaultdict(list)
    for u, v in queries:
        qmap[u].append(v)
        qmap[v].append(u)
    def dfs(u):
        ancestor[u] = u
        for v in tree[u]:
            dfs(v)
            dsu.union(v, u)
            ancestor[dsu.find(u)] = u
        visited[u] = True
        for v in qmap[u]:
            if visited[v]:
                ans[(u, v)] = ancestor[dsu.find(v)]
    dfs(root)
    return ans
```

#### 为什么它重要

- 在一次遍历中高效处理大量 LCA 查询
- 不需要深度或二分倍增表
- 非常适合离线批量查询

应用场景：

- 竞赛编程中的树查询
- 离线图分析
- 有根结构中的动态连通性

#### 一个温和的证明（为什么它有效）

一旦节点 $u$ 被处理：

- 它的所有后代都被合并到它的 DSU 集合中。
- $\text{find}(u)$ 总是返回到目前为止已处理的最低祖先。

当访问查询 $(u,v)$ 时：

- 如果 $v$ 已经被访问，
  那么 $\text{find}(v)$ 就是 $u$ 和 $v$ 的最近公共祖先。

不变式：
$$
\text{ancestor}[\text{find}(v)] = \text{LCA}(u, v)
$$
通过后序遍历和合并操作得以维持。

#### 自己动手试试

1.  画一棵小树并列出所有查询。
2.  手动执行 DFS。
3.  跟踪合并和祖先更新。
4.  当两个节点都被访问时记录 LCA。

#### 测试用例

| 查询   | 预期结果 | 原因           |
| ------ | -------- | -------------- |
| (4,5)  | 2        | 同一子树       |
| (4,6)  | 1        | 跨子树         |
| (3,6)  | 3        | 祖先关系       |

#### 复杂度

- 时间复杂度：$O(n + q \alpha(n))$
- 空间复杂度：$O(n + q)$

对于批量查询非常高效；每次合并和查找操作几乎是常数时间。

Tarjan 的 LCA 算法将问题转化为一场并查集的舞蹈，在一次优雅的 DFS 扫描中回答所有祖先问题。
### 384 轻重链剖分

轻重链剖分（HLD）将一棵树分解为若干条重链和轻边，使得任意两个节点之间的路径可以被拆分为 $O(\log n)$ 个连续段。它是处理路径查询（如求和、最大值、最小值）或路径更新问题的基石，通常与线段树或树状数组结合使用。

#### 我们要解决什么问题？

在许多问题中，我们需要回答以下类型的查询：

- 从节点 $u$ 到节点 $v$ 的路径上的和或最大值是多少？
- 如何更新路径上的值？
- 两个节点之间边的最小权重是多少？

朴素的方法是沿着路径遍历，每次查询需要 $O(n)$ 的时间。使用轻重链剖分，我们可以将路径分解为少数几个连续段，从而将查询复杂度降低到 $O(\log^2 n)$（如果结合 LCA 预处理，甚至可以更低）。

#### 工作原理（通俗解释）

每个节点选择一个重儿子（拥有最大子树的子节点）。所有其他边都是轻边。这保证了：

- 每个节点恰好属于一条重链。
- 从根到叶子的任意路径最多经过 $O(\log n)$ 条轻边。

因此，我们可以将每条重链表示为数组中的一个连续段，从而将树上的查询映射为数组上的区间查询。

##### 步骤

1.  第一次 DFS：

    *   计算每个节点的子树大小。
    *   标记重儿子 = 拥有最大子树的子节点。

2.  第二次 DFS：

    *   分配重链的链头。
    *   将每个节点映射到线性序列中的一个位置。

3.  路径查询 (u, v)：

    *   当 $u$ 和 $v$ 不在同一条重链上时：

        *   总是将链头深度更深的节点向上移动。
        *   在该节点对应的链段上查询线段树。
        *   将 $u$ 跳转到其链头的父节点。
    *   当 $u$ 和 $v$ 在同一条重链上时：查询从 $u$ 到 $v$（按顺序）的线段树区间。

#### 示例

树结构：

```
1
├── 2
│   ├── 4
│   └── 5
└── 3
    ├── 6
    └── 7
```

- 子树大小：
  $size(2)=3,\; size(3)=3$
  重边：$(1,2)$, $(2,4)$, $(3,6)$

- 重链：
  链 1: 1 → 2 → 4
  链 2: 3 → 6
  轻边：$(1,3), (2,5), (3,7)$

任意查询路径 $(4,7)$ 最多跨越 $\log n$ 个连续段：

```
4→2→1 | 1→3 | 3→7
```

#### 精简代码（Python）

```python
def dfs1(u, p, g, size, heavy):
    size[u] = 1
    max_sub = 0
    for v in g[u]:
        if v == p: continue
        dfs1(v, u, g, size, heavy)
        size[u] += size[v]
        if size[v] > max_sub:
            max_sub = size[v]
            heavy[u] = v

def dfs2(u, p, g, head, pos, cur_head, order, heavy):
    head[u] = cur_head
    pos[u] = len(order)
    order.append(u)
    if heavy[u] != -1:
        dfs2(heavy[u], u, g, head, pos, cur_head, order, heavy)
    for v in g[u]:
        if v == p or v == heavy[u]:
            continue
        dfs2(v, u, g, head, pos, v, order, heavy)

def query_path(u, v, head, pos, depth, segtree, lca):
    res = 0
    while head[u] != head[v]:
        if depth[head[u]] < depth[head[v]]:
            u, v = v, u
        res += segtree.query(pos[head[u]], pos[u])
        u = parent[head[u]]
    if depth[u] > depth[v]:
        u, v = v, u
    res += segtree.query(pos[u], pos[v])
    return res
```

#### 为什么它很重要

- 将路径查询转化为区间查询
- 与线段树或树状数组完美结合
- 是以下问题的核心：

  *   路径求和 / 最大值 / 最小值
  *   路径更新
  *   最小边权
  *   子树查询（结合欧拉序映射）

#### 一个温和的证明（为什么它有效）

每个节点沿着轻边向上移动的次数最多为 $\log n$ 次，因为每次移动后，其所在子树的大小至少减半。在一条重链内部，节点形成连续段，因此路径查询可以被分解为 $O(\log n)$ 个连续区间。

因此，总体查询时间复杂度为 $O(\log^2 n)$（如果使用线段树优化，可以达到 $O(\log n)$）。

#### 动手尝试

1.  画一棵树，计算子树大小。
2.  标记指向最大子树的边为重边。
3.  分配链头节点和线性索引。
4.  尝试一个查询 $(u,v)$ → 将其分解为路径段。
5.  观察每个段在线性序列中都是连续的。

#### 测试用例

| 查询   | 路径          | 段数 |
| ------ | ------------- | ---- |
| (4,5)  | 4→2→5         | 2    |
| (4,7)  | 4→2→1→3→7     | 3    |
| (6,7)  | 6→3→7         | 2    |

#### 复杂度

- 预处理：$O(n)$
- 查询：$O(\log^2 n)$（或使用优化线段树时为 $O(\log n)$）
- 空间：$O(n)$

轻重链剖分连接了树的拓扑结构和数组查询，使得在路径上进行对数时间复杂度的操作成为可能。
### 385 重心分解

重心分解是一种分治方法，它利用重心（移除后能使树保持平衡的节点）递归地将树分解为更小的部分。
它将树问题转化为对数深度的递归，从而实现快速的查询、更新和路径计数。

#### 我们要解决什么问题？

对于某些树问题（距离查询、路径和、子树覆盖等），
我们需要反复将树分割成易于处理的部分。

树的一个重心是指这样一个节点：如果将其移除，
则产生的任何连通分量都不会包含超过 $n/2$ 个节点。

通过递归地使用重心分解树，
我们构建出一棵重心树，这是一种能体现原树平衡性的层次结构。

#### 工作原理（通俗解释）

1.  寻找重心：
   *   计算子树大小。
   *   选择其最大子节点子树大小 ≤ $n/2$ 的节点。

2.  将其记录为本层分解的根。

3.  从树中移除该重心。

4.  对每个剩余的连通分量（子树）进行递归。

每个节点出现在 $O(\log n)$ 层中，
因此使用此结构的查询或更新操作变为对数级别。

#### 示例

树（7个节点）：

```
      1
    / | \
   2  3  4
  / \
 5   6
      \
       7
```

步骤 1：$n=7$。
子树大小 → 选择节点 2（其最大子节点子树大小 ≤ 4）。
重心 = 2。

步骤 2：移除节点 2 → 分割为连通分量：

- {5}, {6,7}, {1,3,4}

步骤 3：递归地在每个连通分量中寻找重心。

- {6,7} → 重心 = 6
- {1,3,4} → 重心 = 1

重心树层次结构：

```
     2
   / | \
  5  6  1
       \
        3
```

#### 微型代码（Python）

```python
def build_centroid_tree(graph):
    n = len(graph)
    size = [0] * n
    removed = [False] * n
    parent = [-1] * n

    def dfs_size(u, p):
        size[u] = 1
        for v in graph[u]:
            if v == p or removed[v]:
                continue
            dfs_size(v, u)
            size[u] += size[v]

    def find_centroid(u, p, n):
        for v in graph[u]:
            if v != p and not removed[v] and size[v] > n // 2:
                return find_centroid(v, u, n)
        return u

    def decompose(u, p):
        dfs_size(u, -1)
        c = find_centroid(u, -1, size[u])
        removed[c] = True
        parent[c] = p
        for v in graph[c]:
            if not removed[v]:
                decompose(v, c)
        return c

    root = decompose(0, -1)
    return parent  # parent[c] 是重心的父节点
```

#### 为何重要

重心分解能够为许多树问题提供高效的解决方案：

- 路径查询：距离和、最小/最大权重
- 点更新：仅影响 $O(\log n)$ 个重心
- 基于距离的搜索：查找某种类型的最远节点
- 树上的分治动态规划

它在解决诸如*统计距离 k 内的点对*、*基于颜色的查询*或*重心树构建*等竞赛编程问题时尤其强大。

#### 一个温和的证明（为何有效）

每个分解步骤移除一个重心。
根据定义，每个剩余的子树 ≤ $n/2$。
因此，递归深度为 $O(\log n)$，
并且每个节点最多参与 $O(\log n)$ 个子问题。

总复杂度：
$$
T(n) = T(n_1) + T(n_2) + \cdots + O(n) \le O(n \log n)
$$

#### 亲自尝试

1.  画一棵小树。
2.  计算子树大小。
3.  选择重心（最大子节点 ≤ $n/2$）。
4.  移除并递归。
5.  构建重心树层次结构。

#### 测试用例

| 树                      | 重心 | 连通分量       | 深度 |
| ----------------------- | ---- | -------------- | ---- |
| 链状 (1–2–3–4–5)        | 3    | {1,2}, {4,5}   | 2    |
| 星状 (1 为中心)         | 1    | 叶子节点       | 1    |
| 平衡树 (7个节点)        | 2    | 3 个连通分量   | 2    |

#### 复杂度

- 预处理：$O(n \log n)$
- 查询/更新（每个节点）：$O(\log n)$
- 空间：$O(n)$

重心分解将树转化为平衡的递归层次结构，
是对数时间查询和更新系统的通用工具。
### 386 树的直径（两次 DFS）

树的直径是树中任意两个节点之间的最长路径。
一种简单而优雅的查找方法是执行两次 DFS 遍历，一次用于找到最远的节点，另一次用于测量最长路径的长度。

#### 我们要解决什么问题？

在一棵树（一个无环连通图）中，我们想要找到其直径，其定义为：

$$
\text{直径} = \max_{u,v \in V} \text{dist}(u, v)
$$

其中 $\text{dist}(u,v)$ 是 $u$ 和 $v$ 之间最短路径的长度（以边数或权重计）。

这条路径总是位于两个叶子节点之间，并且可以通过两次 DFS/BFS 遍历找到。

#### 工作原理（通俗解释）

1.  选择任意节点（例如节点 1）。
2.  运行 DFS 以找到离它最远的节点 → 称其为 $A$。
3.  从 $A$ 开始再次运行 DFS → 找到离 $A$ 最远的节点 → 称其为 $B$。
4.  $A$ 和 $B$ 之间的距离就是直径长度。

为什么这行得通：

-   第一次 DFS 确保你从直径的一端开始。
-   第二次 DFS 延伸到另一端。

#### 示例

树：

```
1
├── 2
│   └── 4
└── 3
    ├── 5
    └── 6
```

1.  DFS(1)：最远节点 = 4（距离 2）
2.  DFS(4)：最远节点 = 6（距离 4）

所以直径 = 4 → 路径：$4 \to 2 \to 1 \to 3 \to 6$

#### 精简代码（Python）

```python
from collections import defaultdict

def dfs(u, p, dist, graph):
    farthest = (u, dist)
    for v, w in graph[u]:
        if v == p:
            continue
        cand = dfs(v, u, dist + w, graph)
        if cand[1] > farthest[1]:
            farthest = cand
    return farthest

def tree_diameter(graph):
    start = 0
    a, _ = dfs(start, -1, 0, graph)
    b, diameter = dfs(a, -1, 0, graph)
    return diameter, (a, b)

# 示例：
graph = defaultdict(list)
edges = [(0,1,1),(1,3,1),(0,2,1),(2,4,1),(2,5,1)]
for u,v,w in edges:
    graph[u].append((v,w))
    graph[v].append((u,w))

d, (a,b) = tree_diameter(graph)
print(d, (a,b))  # 4, 路径 (3,5)
```

#### 为什么它重要

-   简单且线性时间复杂度：$O(n)$
-   适用于带权和不带权的树
-   是以下内容的核心：
    *   寻找树的中心
    *   树上的动态规划
    *   网络分析（最长延迟，最大延迟路径）

#### 一个温和的证明（为什么它行得通）

令 $(u, v)$ 为真正的直径端点。
从任意节点 $x$ 开始的任何 DFS 都会找到一个最远节点 $A$。
根据树上的三角不等式，$A$ 必须是某条直径路径的一个端点。
从 $A$ 开始的第二次 DFS 找到另一个端点 $B$，
确保 $\text{dist}(A,B)$ 是可能的最大值。

#### 自己动手试试

1.  画一棵小树。
2.  选择任意起始节点，运行 DFS 找到最远节点。
3.  从那个节点开始，再次运行 DFS。
4.  追踪这两个节点之间的路径，它总是最长的。

#### 测试用例

| 树类型                 | 直径 | 路径                       |
| ---------------------- | ---- | -------------------------- |
| 线形 (1–2–3–4)         | 3    | (1,4)                      |
| 星形 (1 中心 + 叶子)   | 2    | (叶子₁, 叶子₂)             |
| 平衡二叉树             | 4    | 最左叶子到最右叶子         |

#### 复杂度

-   时间复杂度：$O(n)$
-   空间复杂度：$O(n)$ 递归栈

通过两次 DFS 求树的直径是图论中最简洁的技巧之一：从任意点出发，走远，然后走得更远。
### 387 树形动态规划（基于子树的优化）

树形动态规划（Tree DP）是一种通过组合子树结果来解决树上问题的技术。每个节点的结果依赖于其子节点，遵循一种递归模式，类似于分治法，但应用于树结构。

#### 我们要解决什么问题？

许多树上的问题要求基于子树的某些最优值（和、最小值、最大值、数量），例如：

- 树中的最大路径和
- 最大独立集的大小
- 为树着色的方案数
- 到所有节点的距离之和

树形动态规划提供了一种自底向上的方法，我们为每个子树计算结果，并向上合并。

#### 工作原理（通俗解释）

1.  将树以一个任意节点（通常是 1 或 0）为根。
2.  基于每个节点的子树定义其 DP 状态。
3.  递归地合并子节点的结果。
4.  将结果返回给父节点。

状态示例：
$$
dp[u] = f(dp[v_1], dp[v_2], \dots, dp[v_k])
$$
其中 $v_i$ 是 $u$ 的子节点。

#### 示例问题

求最大独立集的大小（选择的节点中任意两个不相邻）。

递推关系：

$$
dp[u][0] = \sum_{v \in children(u)} \max(dp[v][0], dp[v][1])
$$

$$
dp[u][1] = 1 + \sum_{v \in children(u)} dp[v][0]
$$

#### 微型代码（Python）

```python
from collections import defaultdict

graph = defaultdict(list)
edges = [(0,1),(0,2),(1,3),(1,4)]
for u,v in edges:
    graph[u].append(v)
    graph[v].append(u)

def dfs(u, p):
    incl = 1  # 包含 u
    excl = 0  # 不包含 u
    for v in graph[u]:
        if v == p:
            continue
        inc_v, exc_v = dfs(v, u)
        incl += exc_v
        excl += max(inc_v, exc_v)
    return incl, excl

incl, excl = dfs(0, -1)
ans = max(incl, excl)
print(ans)  # 3
```

这段代码找到了最大独立集的大小（3 个节点）。

#### 为什么它很重要？

树形动态规划是以下领域的基础：

- **计数**：分配标签、颜色或状态的方案数
- **优化**：在路径或集合上最大化或最小化成本
- **组合数学**：解决划分或约束问题
- **博弈论**：递归计算输赢状态

它将全局的树问题转化为局部的合并操作。

#### 一个温和的证明（为什么它有效）

一旦知道父节点是否被包含，每个节点的子树就是独立的。
通过先处理子节点再处理父节点（后序 DFS），我们确保在合并时所有必要的数据都已就绪。
这通过对树大小的结构归纳法保证了正确性。

#### 亲自尝试

1.  定义一个简单的属性（和、计数、最大值）。
2.  根据其子节点为节点写出递推关系。
3.  通过 DFS 遍历，合并子节点的结果。
4.  将值返回给父节点，在根节点处合并。

#### 模板（通用）

```python
def dfs(u, p):
    res = base_case() # 基础情况
    for v in children(u):
        if v != p:
            child = dfs(v, u)
            res = merge(res, child) # 合并
    return finalize(res) # 最终处理
```

#### 复杂度

- 时间复杂度：$O(n)$
- 空间复杂度：$O(n)$（递归）

树形动态规划将递归思维转化为结构化的计算，一次处理一个子树，从叶子节点到根节点逐步构建答案。
### 388 换根动态规划（计算所有根节点的答案）

换根动态规划是一种高级的树形动态规划技术，能够高效地计算以每个可能节点为根时的答案。它并非从头重新计算，而是通过沿树向下和向上传播信息来重用结果。

#### 我们要解决什么问题？

假设我们希望为每个节点计算一个值，就好像该节点是树的根一样。
例如：

-   从每个节点到所有其他节点的距离之和
-   每个子树中的节点数量
-   依赖于子树结构的动态规划值

一种朴素的方法是为每个根节点运行 $O(n)$ 的动态规划，总复杂度为 $O(n^2)$。
换根动态规划可以将总复杂度降低到 $O(n)$。

#### 工作原理（通俗解释）

1.  第一遍（向下动态规划）：
    假设根节点为 0，计算每个子树的答案。

2.  第二遍（换根）：
    利用父节点的信息来更新子节点的答案 —— 这实际上是在每个节点处“换根”重新计算动态规划。

关键在于组合除当前子节点外其他子节点的结果，通常使用前缀-后缀合并技术。

#### 示例问题

计算从每个节点到所有其他节点的距离之和。

递推关系：

-   `subtree_size[u]` = 节点 `u` 子树中的节点数量
-   `dp[u]` = 从节点 `u` 到其子树中所有节点的距离之和

第一遍（向下）：
$$
dp[u] = \sum_{v \in children(u)} (dp[v] + subtree_size[v])
$$

第二遍（换根）：
当根从节点 `u` 移动到其子节点 `v` 时：
$$
dp[v] = dp[u] - subtree_size[v] + (n - subtree_size[v])
$$

#### 精简代码（Python）

```python
from collections import defaultdict

graph = defaultdict(list)
edges = [(0,1),(0,2),(2,3),(2,4)]
for u,v in edges:
    graph[u].append(v)
    graph[v].append(u)

n = 5
size = [1]*n
dp = [0]*n
ans = [0]*n

def dfs1(u, p):
    for v in graph[u]:
        if v == p: continue
        dfs1(v, u)
        size[u] += size[v]
        dp[u] += dp[v] + size[v]

def dfs2(u, p):
    ans[u] = dp[u]
    for v in graph[u]:
        if v == p: continue
        pu, pv = dp[u], dp[v]
        su, sv = size[u], size[v]

        dp[u] -= dp[v] + size[v]
        size[u] -= size[v]
        dp[v] += dp[u] + size[u]
        size[v] += size[u]

        dfs2(v, u)

        dp[u], dp[v] = pu, pv
        size[u], size[v] = su, sv

dfs1(0, -1)
dfs2(0, -1)

print(ans)
```

这段代码计算每个节点的距离之和。

#### 为什么重要

-   高效地推导所有根节点的答案
-   常用于重心问题、距离之和、换根求和等问题
-   极大地减少了冗余计算

#### 一个简单的证明（为什么有效）

每个动态规划值仅依赖于子树的局部合并。
通过移除一个子节点子树并添加剩余部分，我们调整父节点的值以反映新的根节点。
基于树结构的归纳法确保了正确性。

#### 动手试试

1.  为单个根节点写出你的动态规划递推式。
2.  确定当根节点换到子节点时，哪些值发生了变化。
3.  使用存储的子树信息应用推-拉更新。
4.  在遍历后收集最终答案。

#### 复杂度

-   时间复杂度：$O(n)$（两次深度优先搜索遍历）
-   空间复杂度：$O(n)$

换根动态规划将全局的换根问题转化为一对局部变换，通过优雅的重用计算每个节点的视角。
### 389 树上的二分查找（边权约束）

树上的二分查找是一种多功能的策略，用于解决必须找到满足约束条件的阈值边权或路径条件的问题。它将 DFS 或 BFS 遍历与对数值属性（通常是边权或限制）的二分查找相结合。

#### 我们要解决什么问题？

给定一棵带权边的树，我们可能想要回答以下问题：

-   使得两个节点之间的路径满足某种属性的最小边权是多少？
-   在什么最大允许成本下，树仍然保持连通？
-   使得至少 $k$ 个节点可达的最小阈值是多少？

一种朴素的方法是检查每个可能的权重。我们可以通过对排序后的边权进行二分查找来做得更好。

#### 工作原理（通俗解释）

1.  对边权进行排序或定义一个数值搜索范围。
2.  在可能的阈值 $T$ 上进行二分查找。
3.  对于每个中间值，遍历树（通过 DFS/BFS/并查集），只包含满足条件（例如，权重 ≤ T）的边。
4.  检查属性是否成立。
5.  缩小搜索区间。

关键在于单调性：如果属性对于 $T$ 成立，那么它对于所有更大/更小的值也成立。

#### 示例问题

找到最小边权阈值 $T$，使得所有节点都变得连通。

算法：

1.  按权重对边进行排序。
2.  在 $[min\_weight, max\_weight]$ 范围内二分查找 $T$。
3.  对于每个 $T$，包含满足 $w \le T$ 的边。
4.  检查生成的图是否连通。
5.  相应地调整边界。

#### 微型代码（Python）

```python
def is_connected(n, edges, limit):
    parent = list(range(n))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa

    for u, v, w in edges:
        if w <= limit:
            union(u, v)
    return len({find(i) for i in range(n)}) == 1

def binary_search_tree_threshold(n, edges):
    weights = sorted(set(w for _,_,w in edges))
    lo, hi = 0, len(weights)-1
    ans = weights[-1]
    while lo <= hi:
        mid = (lo + hi) // 2
        if is_connected(n, edges, weights[mid]):
            ans = weights[mid]
            hi = mid - 1
        else:
            lo = mid + 1
    return ans

edges = [(0,1,4),(1,2,2),(0,2,5)]
print(binary_search_tree_threshold(3, edges))  # 输出: 4
```

#### 为什么它很重要

-   将复杂的“阈值”问题简化为对数级别的搜索
-   在解空间具有单调性时有效
-   将结构遍历与决策逻辑相结合

常见应用：

-   带限制的路径可行性
-   按成本筛选边
-   带权值限制的树查询

#### 一个温和的证明（为什么它有效）

设属性 $P(T)$ 在阈值 $T$ 满足条件时为真。如果 $P(T)$ 是单调的（要么在某个点之后总是为真，要么在某个点之前总是为假），那么二分查找就能正确地收敛到最小/最大的满足条件的 $T$。

#### 亲自尝试

1.  定义一个关于权重的单调属性（例如“在限制下连通”）。
2.  实现一个检查 $P(T)$ 的决策函数。
3.  将其包装在一个二分查找循环中。
4.  返回最小/最大的有效 $T$。

#### 复杂度

-   对边排序：$O(E \log E)$
-   二分查找：$O(\log E)$ 次检查
-   每次检查：$O(E \alpha(V))$（并查集）
-   总体：$O(E \log^2 E)$ 或更好

树上的二分查找并不是在树*内部*进行搜索，而是对定义在树上的*约束条件*进行搜索，这是一种用于基于阈值推理的巧妙技术。
### 390 虚树（查询子集构建）

虚树是一种压缩的树形表示，它由选定的节点子集及其最近公共祖先（LCA）构成，并按照原始树的层次顺序连接。
它用于查询问题中，当我们只需要处理一小部分节点而非整棵树时。

#### 我们要解决什么问题？

假设我们有一棵大树，一个查询给了我们一个小的节点集合 $S$。
我们需要计算 $S$ 中节点之间的某些属性（例如距离之和、覆盖范围或在其路径上的动态规划）。
每次遍历整棵树效率很低。

虚树将整棵树压缩为仅包含 $S$ 中的节点及其 LCA，同时保持祖先关系不变。

这可以将问题简化为一个更小的树，通常节点数为 $O(|S|\log |S|)$ 而不是 $O(n)$。

#### 工作原理（通俗解释）

1.  收集查询节点 $S$。
2.  按欧拉遍历顺序（DFS 中的进入时间）对 $S$ 排序。
3.  插入连续节点的 LCA 以保持结构。
4.  在连续的祖先之间构建边以形成一棵树。
5.  现在在这个迷你树上处理查询，而不是在整棵树上。

#### 示例

给定树：

```
       1
     / | \
    2  3  4
      / \
     5   6
```

查询节点 $S = {2, 5, 6}$。

它们的 LCA：

- $\text{LCA}(5,6)=3$
- $\text{LCA}(2,3)=1$

虚树节点 = ${1,2,3,5,6}$
根据父子关系连接它们：
1 → 2
1 → 3
3 → 5
3 → 6

这就是你的虚树。

#### 精简代码（C++ 风格伪代码）

```cpp
vector<int> build_virtual_tree(vector<int>& S, vector<int>& tin, auto lca) {
    // 按欧拉序（进入时间）排序
    sort(S.begin(), S.end(), [&](int a, int b) { return tin[a] < tin[b]; });
    vector<int> nodes = S;
    // 添加连续节点对的 LCA
    for (int i = 0; i + 1 < (int)S.size(); i++)
        nodes.push_back(lca(S[i], S[i+1]));
    // 对节点去重并再次排序
    sort(nodes.begin(), nodes.end(), [&](int a, int b) { return tin[a] < tin[b]; });
    nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());
    // 使用栈构建虚树边
    stack<int> st;
    vector<vector<int>> vt_adj(nodes.size());
    for (int u : nodes) {
        while (!st.empty() && !is_ancestor(st.top(), u))
            st.pop();
        if (!st.empty())
            vt_adj[st.top()].push_back(u);
        st.push(u);
    }
    return nodes;
}
```

这段代码使用栈和祖先关系构建了虚树的邻接表。

#### 为什么它很重要

- 将大树查询简化为小子问题
- 在处理多个子集查询的问题中至关重要
- 常见于离线处理、换根 DP、距离和问题
- 与 LCA 预计算完美配合

#### 一个温和的证明（为什么它有效）

$S$ 中的每个节点必须通过其 LCA 连接。
通过按欧拉序排序并维护一个祖先栈，我们确保：

- 父子关系一致
- 没有环或重复节点
- 生成的树保留了祖先关系

因此，虚树是连接 $S$ 中所有节点的最小子树。

#### 自己动手试试

1.  实现欧拉遍历（获取 `tin[u]`）。
2.  实现 LCA（二进制提升或 RMQ）。
3.  为给定的 $S$ 构建虚树。
4.  应用你的查询逻辑（求和、计数、DP）。

#### 复杂度

- 排序 $S$：$O(|S|\log|S|)$
- LCA 调用：$O(|S|)$
- 构建结构：$O(|S|)$
- 总计：每次查询 $O(|S|\log|S|)$

虚树是你的“查询缩放树”，是大树在你问题的小世界中的精确投影。

## 第 40 章 高级图算法与技巧
### 391 拓扑动态规划（有向无环图上的动态规划）

拓扑动态规划是一种在有向无环图（DAG）上应用的动态规划技术。
它按照依赖顺序计算值，确保每个节点的状态仅在其所有前置条件计算完成后才被计算。
这种方法是解决涉及偏序、最长路径、路径计数以及依赖关系传播等问题的核心工具。

#### 我们要解决什么问题？

在有向无环图中，某些节点依赖于其他节点（边从依赖项指向被依赖项）。
我们通常希望计算诸如以下的动态规划值：

-   以每个节点结尾的最长路径
-   到达某个节点的路径数量
-   到达某个节点的最小成本
-   通过依赖关系累积的值

因为有向无环图没有环，所以存在拓扑序，从而允许线性时间评估。

#### 工作原理（通俗解释）

1.  对有向无环图进行拓扑排序。
2.  初始化基本情况（例如，源节点 = 0 或 1）。
3.  按拓扑序迭代，基于入边更新每个节点。
4.  每个节点一旦被处理，其值即为最终值。

这保证了每个依赖项在使用前都已解决。

#### 示例：有向无环图中的最长路径

给定边权为 $w(u,v)$ 的有向无环图，定义动态规划：

$$
dp[v] = \max_{(u,v)\in E}(dp[u] + w(u,v))
$$

基本情况：$dp[source] = 0$

#### 示例有向无环图

```
1 → 2 → 4
 ↘︎ 3 ↗︎
```

边：

-   1 → 2 (1)
-   1 → 3 (2)
-   2 → 4 (1)
-   3 → 4 (3)

拓扑序：[1, 2, 3, 4]

计算：

-   dp[1] = 0
-   dp[2] = max(dp[1] + 1) = 1
-   dp[3] = max(dp[1] + 2) = 2
-   dp[4] = max(dp[2] + 1, dp[3] + 3) = max(2, 5) = 5

结果：最长路径长度 = 5

#### 微型代码（C++ 风格）

```cpp
vector<int> topo_sort(int n, vector<vector<int>>& adj) {
    vector<int> indeg(n, 0), topo;
    for (auto& u : adj)
        for (int v : u) indeg[v]++;
    queue<int> q;
    for (int i = 0; i < n; i++) if (indeg[i] == 0) q.push(i);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        topo.push_back(u);
        for (int v : adj[u])
            if (--indeg[v] == 0) q.push(v);
    }
    return topo;
}

vector<int> topo_dp(int n, vector<vector<pair<int,int>>>& adj) {
    auto order = topo_sort(n, ...);
    vector<int> dp(n, INT_MIN);
    dp[0] = 0;
    for (int u : order)
        for (auto [v, w] : adj[u])
            dp[v] = max(dp[v], dp[u] + w);
    return dp;
}
```

#### 为什么它很重要

-   将递归依赖关系转化为迭代计算
-   避免冗余工作
-   为许多有向无环图问题提供线性时间解决方案
-   适用于计数、最小/最大、聚合任务

常见用途：

-   有向无环图中的最长路径
-   路径数量计数
-   最小成本调度
-   项目依赖规划

#### 一个温和的证明（为什么它有效）

拓扑序保证了：

-   对于每条边 $(u,v)$，$u$ 出现在 $v$ 之前。
    因此，在处理 $v$ 时，所有前驱节点 $u$ 的 $dp[u]$ 都已就绪。
    这确保了对于任何形式如下的动态规划公式的正确性：

$$
dp[v] = f({dp[u]\ |\ (u,v)\in E})
$$

#### 亲自尝试

1.  计算从源节点到每个节点的路径数量：
    $dp[v] = \sum_{(u,v)} dp[u]$
2.  如果边有权重，计算最小成本
3.  构建具有依赖关系的任务最长链
4.  在强连通分量有向无环图（元图）上应用拓扑动态规划

#### 复杂度

-   拓扑排序：$O(V+E)$
-   动态规划传播：$O(V+E)$
-   总计：$O(V+E)$ 时间，$O(V)$ 空间

拓扑动态规划就是如何将依赖关系的混乱带入秩序，一次一层，一个节点，一个依赖项。
### 392 SCC 缩点图动态规划（元图上的动态规划）

SCC 缩点图动态规划将动态规划应用于将有向图缩点成有向无环图（DAG）的过程，其中每个节点代表一个强连通分量（SCC）。这将一个带环图转化为无环图，从而能够在分量之间进行拓扑推理、路径聚合和值传播。

#### 我们要解决什么问题？

由于环的存在，有向图上的许多问题变得复杂。在每个 SCC 内部，每个节点都可以到达其他所有节点，它们是强连通的。通过将 SCC 缩成单个节点，我们得到一个 DAG：

$$
G' = (V', E')
$$

其中每个 $v' \in V'$ 是一个 SCC，边 $(u', v')$ 表示分量之间的转移。

一旦图变成无环的，我们就可以运行拓扑动态规划来计算：

-   到达每个 SCC 的最大值或最小值
-   分量之间的路径数量
-   聚合的分数、权重或成本

#### 工作原理（通俗解释）

1.  找到 SCC（使用 Tarjan 或 Kosaraju 算法）。
2.  构建缩点 DAG：每个 SCC 变成一个节点。
3.  聚合每个 SCC 的初始值（例如，权重之和）。
4.  按照拓扑顺序在 DAG 上运行动态规划，合并来自入边的贡献。
5.  如果需要，将结果映射回原始节点。

这隔离了环（内部的 SCC）并清晰地管理了依赖关系。

#### 示例

原始图 $G$：

```
1 → 2 → 3
↑    ↓
5 ← 4
```

SCC：

-   C₁ = {1,2,4,5}
-   C₂ = {3}

缩点图：

```
C₁ → C₂
```

如果每个节点有权重 $w[i]$，那么：

$$
dp[C₂] = \max_{(C₁,C₂)}(dp[C₁] + \text{aggregate}(C₁))
$$

#### 微型代码（C++ 风格）

```cpp
int n;
vector<vector<int>> adj;
vector<int> comp, order;
vector<bool> vis;

void dfs1(int u) {
    vis[u] = true;
    for (int v : adj[u]) if (!vis[v]) dfs1(v);
    order.push_back(u);
}

void dfs2(int u, int c, vector<vector<int>>& radj) {
    comp[u] = c;
    for (int v : radj[u]) if (comp[v] == -1) dfs2(v, c, radj);
}

vector<int> scc_condense() {
    vector<vector<int>> radj(n);
    for (int u=0; u<n; ++u)
        for (int v : adj[u]) radj[v].push_back(u);

    vis.assign(n,false);
    for (int i=0; i<n; ++i) if (!vis[i]) dfs1(i);

    comp.assign(n,-1);
    int cid=0;
    for (int i=n-1;i>=0;--i){
        int u = order[i];
        if (comp[u]==-1) dfs2(u,cid++,radj);
    }

    return comp;
}
```

然后构建缩点图：

```cpp
vector<vector<int>> dag(cid);
for (int u=0;u<n;++u)
  for (int v:adj[u])
    if (comp[u]!=comp[v])
      dag[comp[u]].push_back(comp[v]);
```

在 `dag` 上运行拓扑动态规划。

#### 为什么重要

-   将有环图转化为无环 DAG
-   允许在有环上下文中进行动态规划、路径计数和聚合
-   简化了关于可达性和影响的推理
-   构成了以下技术的基础：
    *   动态缩点
    *   元图优化
    *   模块化图分析

#### 一个温和的证明（为什么有效）

1.  在每个 SCC 内部，所有节点都是相互可达的。
2.  缩点将这些节点合并为单个节点，确保没有环。
3.  SCC 之间的边定义了一个偏序，使得拓扑排序成为可能。
4.  任何在依赖关系上递归定义的属性现在都可以通过在这个顺序上运行动态规划来解决。

形式上，对于每个 $C_i$：

$$
dp[C_i] = f({dp[C_j] \mid (C_j, C_i) \in E'}, \text{value}(C_i))
$$

#### 自己动手试试

1.  为每个节点分配一个权重；计算 SCC DAG 上的最大和路径。
2.  计算 SCC 之间的不同路径数量。
3.  结合 SCC 检测和动态规划来解决带权可达性问题。
4.  解决类似问题：
    *   "具有传送环的地牢中的最大金币"
    *   "具有反馈循环的依赖图"

#### 复杂度

-   SCC 计算：$O(V+E)$
-   DAG 构建：$O(V+E)$
-   拓扑动态规划：$O(V'+E')$
-   总计：$O(V+E)$

SCC 缩点图动态规划是将环缩小为确定性的艺术，揭示了错综复杂表面之下清晰的 DAG。
### 393 欧拉路径

欧拉路径是图中一条访问每条边恰好一次的路径。如果路径起点和终点在同一顶点，则称为欧拉回路。这一概念是路径规划、图遍历和网络分析的核心。

#### 我们要解决什么问题？

我们希望找到一条使用每条边恰好一次的路径，不重复，不遗漏。

在无向图中，欧拉路径存在的充要条件是：

- 恰好有 0 个或 2 个顶点的度为奇数
- 图是连通的

在有向图中，欧拉路径存在的充要条件是：

- 至多一个顶点满足 `出度 - 入度 = 1`（起点）
- 至多一个顶点满足 `入度 - 出度 = 1`（终点）
- 所有其他顶点的入度等于出度
- 图是强连通的（或者在忽略方向时是连通的）

#### 它是如何工作的（通俗解释）

1.  检查度条件（无向图）或入度/出度平衡条件（有向图）。
2.  选择一个起点：
    *   对于无向图：任意一个奇度顶点（如果存在），否则任意顶点。
    *   对于有向图：满足 `出度 = 入度 + 1` 的顶点，否则任意顶点。
3.  应用 Hierholzer 算法：
    *   贪心地遍历边，直到无法继续。
    *   回溯并将多个环合并为一条路径。
4.  将构建的顺序反转，得到最终路径。

#### 示例（无向图）

图边：

```
1—2, 2—3, 3—1, 2—4
```

度数：

- deg(1)=2, deg(2)=3, deg(3)=2, deg(4)=1
  奇度顶点：2, 4 → 路径存在（起点为 2，终点为 4）

欧拉路径：`2 → 1 → 3 → 2 → 4`

#### 示例（有向图）

边：

```
A → B, B → C, C → A, C → D
```

度数：

- A: 出度=1, 入度=1
- B: 出度=1, 入度=1
- C: 出度=2, 入度=1
- D: 出度=0, 入度=1

起点：C (`出度=入度+1`), 终点：D (`入度=出度+1`)

欧拉路径：`C → A → B → C → D`

#### 微型代码（C++）

```cpp
vector<vector<int>> adj;
vector<int> path;

void dfs(int u) {
    while (!adj[u].empty()) {
        int v = adj[u].back();
        adj[u].pop_back();
        dfs(v);
    }
    path.push_back(u);
}
```

从起点运行，然后反转 `path`。

#### 微型代码（Python）

```python
def eulerian_path(graph, start):
    stack, path = [start], []
    while stack:
        u = stack[-1]
        if graph[u]:
            v = graph[u].pop()
            stack.append(v)
        else:
            path.append(stack.pop())
    return path[::-1]
```

#### 为什么它重要

- 是图遍历问题的基础
- 应用于：
  *   DNA 测序（De Bruijn 图重建）
  *   路径规划（邮政投递、垃圾收集）
  *   网络诊断（追踪所有连接）

#### 一个温和的证明（为什么它有效）

每条边必须恰好出现一次。
在每个顶点（起点和终点可能除外），每次进入都必须对应一次离开。
这要求度的平衡。

在欧拉回路中：
$$
\text{入度}(v) = \text{出度}(v) \quad \forall v
$$

在欧拉路径中：
$$
\exists \text{起点满足 } \text{出度}(v)=\text{入度}(v)+1 \\
\exists \text{终点满足 } \text{入度}(v)=\text{出度}(v)+1
$$

Hierholzer 算法通过合并环并确保消耗所有边来构建路径。

#### 亲自尝试

1.  构建一个小图并测试奇偶性条件。
2.  实现 Hierholzer 算法并跟踪每一步。
3.  通过计算遍历的边数来验证正确性。
4.  探索有向和无向两种变体。
5.  修改算法以检测欧拉回路与路径。

#### 复杂度

- 时间：$O(V+E)$
- 空间：$O(V+E)$

欧拉路径之所以优雅，是因为它们恰好覆盖每个连接一次，在完美的遍历中实现完美的顺序。
### 394 哈密顿路径

哈密顿路径是图中恰好访问每个顶点一次的路径。如果它起始并结束于同一顶点，则形成一个哈密顿回路。与欧拉路径（关注边）不同，哈密顿路径关注的是顶点。

寻找哈密顿路径是一个经典的 NP 完全问题，对于一般图，没有已知的多项式时间算法。

#### 我们要解决什么问题？

我们想要确定是否存在一条路径，恰好访问每个顶点一次，不重复，不遗漏。

用正式术语来说：
给定一个图 $G = (V, E)$，找到一个顶点序列
$$v_1, v_2, \ldots, v_n$$
使得对于所有 $i$，$(v_i, v_{i+1}) \in E$，并且所有顶点都互不相同。

对于哈密顿回路，额外要求 $(v_n, v_1) \in E$。

#### 它是如何工作的（通俗解释）

没有像欧拉路径那样简单的奇偶性或度数条件。
我们通常通过回溯法、状态压缩动态规划或启发式方法（针对大型图）来解决它：

1.  选择一个起始顶点。
2.  递归地探索所有未访问的邻居。
3.  标记已访问的顶点。
4.  如果所有顶点都被访问 → 找到了哈密顿路径。
5.  否则，回溯。

对于小图，这种暴力方法有效；对于大图，则不切实际。

#### 示例（无向图）

图：

```
1, 2, 3
|    \   |
4 ———— 5
```

一条哈密顿路径：`1 → 2 → 3 → 5 → 4`
一条哈密顿回路：`1 → 2 → 3 → 5 → 4 → 1`

#### 示例（有向图）

```
A → B → C
↑       ↓
E ← D ← 
```

可能的哈密顿路径：`A → B → C → D → E`

#### 微型代码（回溯法 – C++）

```cpp
bool hamiltonianPath(int u, vector<vector<int>>& adj, vector<bool>& visited, vector<int>& path, int n) {
    if (path.size() == n) return true;
    for (int v : adj[u]) {
        if (!visited[v]) {
            visited[v] = true;
            path.push_back(v);
            if (hamiltonianPath(v, adj, visited, path, n)) return true;
            visited[v] = false;
            path.pop_back();
        }
    }
    return false;
}
```

以每个顶点作为潜在的起点调用此函数。

#### 微型代码（状态压缩动态规划 – C++）

用于旅行商问题风格的哈密顿搜索：

```cpp
int n;
vector<vector<int>> dp(1 << n, vector<int>(n, INF));
dp[1][0] = 0;
for (int mask = 1; mask < (1 << n); ++mask) {
    for (int u = 0; u < n; ++u) if (mask & (1 << u)) {
        for (int v = 0; v < n; ++v) if (!(mask & (1 << v)) && adj[u][v]) {
            dp[mask | (1 << v)][v] = min(dp[mask | (1 << v)][v], dp[mask][u] + cost[u][v]);
        }
    }
}
```

#### 为什么它很重要

- 为排序问题建模：
  *   旅行商问题
  *   带约束的作业排序
  *   基因组组装路径
- 理论计算机科学的基础，是 NP 完全问题的基石。
- 有助于区分简单（欧拉）遍历与困难（哈密顿）遍历。

#### 一个温和的证明（为什么它困难）

哈密顿路径的存在性是 NP 完全的：

1.  验证是容易的（给定一条路径，在 $O(V)$ 时间内检查）。
2.  没有已知的多项式算法（除非 $P=NP$）。
3.  许多问题可以归约到它（如旅行商问题）。

这意味着在一般情况下它很可能需要指数时间：
$$
O(n!)
$$
以朴素形式，或者
$$
O(2^n n)
$$
使用状态压缩动态规划。

#### 亲自尝试

1.  构建小图（4–6 个顶点）并追踪路径。
2.  与欧拉路径条件进行比较。
3.  实现回溯搜索。
4.  扩展到回路检测（检查是否有边回到起点）。
5.  尝试对小 $n \le 20$ 使用状态压缩动态规划。

#### 复杂度

- 时间（回溯法）：$O(n!)$
- 时间（状态压缩动态规划）：$O(2^n \cdot n^2)$
- 空间：$O(2^n \cdot n)$

哈密顿路径抓住了组合爆炸的本质，表述简单，解决困难，却是理解计算极限的核心。
### 395 中国邮递员问题（路径检查）

中国邮递员问题（CPP），也称为路径检查问题，旨在找到一条最短的闭合路径，该路径至少遍历图中的每条边一次。
它推广了欧拉回路，在必要时允许边的重复。

#### 我们解决的是什么问题？

给定一个加权图 $G = (V, E)$，我们希望找到一个成本最小的环游，该环游至少覆盖所有边一次并返回起点。

如果 $G$ 是欧拉图（所有顶点的度均为偶数），答案很简单，就是欧拉回路本身。
否则，我们必须策略性地复制边以使图成为欧拉图，同时最小化增加的总成本。

#### 工作原理（通俗解释）

1.  检查顶点度数：

    *   统计有多少顶点的度是奇数。
2.  如果全是偶数 → 直接寻找欧拉回路。
3.  如果有些是奇数 →

    *   以某种方式将奇数顶点配对，使得配对顶点间最短路径距离之和最小。
    *   沿着这些最短路径复制边。
4.  得到的图是欧拉图，因此可以构造欧拉回路。
5.  该回路的成本是所有边的成本之和 + 增加的边的成本。

#### 示例

图的边（带权重）：

| 边   | 权重 |
| ---- | ---- |
| 1–2  | 3    |
| 2–3  | 2    |
| 3–4  | 4    |
| 4–1  | 3    |
| 2–4  | 1    |

度数：

- deg(1)=2, deg(2)=3, deg(3)=2, deg(4)=3 → 奇数顶点：2, 4
  2 和 4 之间的最短路径：1
  再次添加该路径 → 现在所有顶点的度均为偶数

总成本 = (3 + 2 + 4 + 3 + 1) + 1 = 14

#### 算法（步骤）

1.  识别奇数度顶点
2.  计算最短路径矩阵（Floyd–Warshall 算法）
3.  在奇数顶点间求解最小权重完美匹配
4.  在匹配中复制边
5.  执行欧拉回路遍历（Hierholzer 算法）

#### 微型代码（伪代码）

```python
def chinese_postman(G):
    # 找出所有奇数度顶点
    odd = [v for v in G if degree(v) % 2 == 1]
    if not odd:
        # 如果没有奇数度顶点，直接找欧拉回路
        return eulerian_circuit(G)
    
    # 计算所有顶点对之间的最短距离
    dist = floyd_warshall(G)
    # 在奇数顶点间找到最小权重完美匹配
    pairs = minimum_weight_matching(odd, dist)
    # 根据匹配结果，在图中添加边（复制最短路径）
    for (u, v) in pairs:
        add_path(G, u, v, dist)
    # 在修改后的欧拉图中寻找欧拉回路
    return eulerian_circuit(G)
```

#### 为什么重要

- 网络优化中的核心算法
- 应用于：

  *   邮政路线规划
  *   垃圾收集路线规划
  *   扫雪车调度
  *   街道清扫
- 展示了如何通过图增广来高效解决遍历问题

#### 一个温和的证明（为什么它有效）

为了遍历每条边，所有顶点必须具有偶数度（欧拉条件）。
当存在奇数度顶点时，我们必须将它们配对以恢复偶数度。

最小复制集是奇数顶点间的最小权重完美匹配：

$$
\text{额外成本} = \min_{\text{配对 } M} \sum_{(u,v) \in M} \text{dist}(u,v)
$$

因此，最优路径成本为：

$$
C = \sum_{e \in E} w(e) + \text{额外成本}
$$

#### 自己试试

1.  画一个带有奇数度顶点的图。
2.  识别奇数顶点和最短配对距离。
3.  手动计算最小匹配。
4.  添加复制的边并找到欧拉回路。
5.  比较复制前后的总成本。

#### 复杂度

- Floyd–Warshall 算法：$O(V^3)$
- 最小匹配：$O(V^3)$
- 欧拉遍历：$O(E)$
- 总计：$O(V^3)$

中国邮递员问题将杂乱的图转化为优雅的环游，平衡度数，最小化工作量，并确保每条边都得到应有的遍历。
### 396 Hierholzer 算法

Hierholzer 算法是用于在图中寻找欧拉路径或欧拉回路的经典方法。它通过合并环来构建路径，直到所有边都被恰好使用一次。

#### 我们要解决什么问题？

我们想要找到一条欧拉迹，即一条恰好访问每条边一次的路径或回路。

- 对于欧拉回路（闭合迹）：
  所有顶点的度数均为偶数。
- 对于欧拉路径（开放迹）：
  恰好有两个顶点的度数为奇数。

该算法相对于边数在线性时间内高效地构建路径。

#### 工作原理（通俗解释）

1.  检查欧拉条件：
    *   0 个奇度顶点 → 欧拉回路
    *   2 个奇度顶点 → 欧拉路径（从一个奇度顶点开始）
2.  从一个有效顶点开始（如果是路径则从奇度顶点开始，如果是回路则可以是任意顶点）
3.  贪心地遍历边：
    *   沿着边前进，直到回到起点或无法继续。
4.  回溯与合并：
    *   当卡住时，回溯到一个仍有未使用边的顶点，开始一个新的环，并将其合并到当前路径中。
5.  继续直到所有边都被使用。

最终的顶点序列就是欧拉迹。

#### 示例（无向图）

图：

```
1, 2
|   |
4, 3
```

所有顶点度数均为偶数，因此存在欧拉回路。

从 1 开始：

1 → 2 → 3 → 4 → 1

结果：欧拉回路 = [1, 2, 3, 4, 1]

#### 示例（包含奇度顶点）

图：

```
1, 2, 3
```

deg(1)=1, deg(2)=2, deg(3)=1 → 存在欧拉路径
从 1 开始：

1 → 2 → 3

#### 微型代码（C++）

```cpp
vector<vector<int>> adj;
vector<int> path;

void dfs(int u) {
    while (!adj[u].empty()) {
        int v = adj[u].back();
        adj[u].pop_back();
        dfs(v);
    }
    path.push_back(u);
}
```

DFS 之后，`path` 将包含逆序的顶点。
将其反转即可得到欧拉路径或回路。

#### 微型代码（Python）

```python
def hierholzer(graph, start):
    stack, path = [start], []
    while stack:
        u = stack[-1]
        if graph[u]:
            v = graph[u].pop()
            graph[v].remove(u)  # 对于无向图，移除双向边
            stack.append(v)
        else:
            path.append(stack.pop())
    return path[::-1]
```

#### 为什么它很重要

- 在 $O(V + E)$ 时间内高效构建欧拉路径
- 是以下问题的基础：
  *   中国邮递员问题
  *   欧拉回路检测
  *   DNA 测序（德布鲁因图）
  *   路线设计和网络分析

#### 一个温和的证明（为什么它有效）

-   每次遍历一条边，就将其移除（使用一次）。
-   每个顶点保持度数平衡（入度 = 出度）。
-   当卡住时，形成的子路径是一个环。
-   合并所有这些环就产生了一个完整的单一遍历。

因此，每条边在最终路线中恰好出现一次。

#### 亲自尝试

1.  绘制小的欧拉图。
2.  手动追踪 Hierholzer 算法。
3.  确定起始顶点（奇度顶点或任意顶点）。
4.  验证路径恰好覆盖所有边一次。
5.  应用于有向图和无向图。

#### 复杂度

-   时间：$O(V + E)$
-   空间：$O(V + E)$

Hierholzer 算法优雅地从连通性中构建顺序，确保每条边在一次完美的遍历中找到自己的位置。
### 397 Johnson 环查找算法

Johnson 算法是一种用于枚举有向图中所有简单环（基本回路）的强大方法。*简单环*是指除了起点/终点顶点外，不重复访问任何顶点的环。

与基于 DFS 的方法可能遗漏或重复环不同，Johnson 的方法能系统地列出每个环且仅列出一次，其运行时间为 O((V + E)(C + 1))，其中 C 是环的数量。

#### 我们要解决什么问题？

我们想要找到有向图 $G = (V, E)$ 中的所有简单环。

也就是说，找到所有顶点序列
$$v_1 \to v_2 \to \ldots \to v_k \to v_1$$
其中每个 $v_i$ 都是不同的，并且在连续的顶点之间存在边。

枚举所有环是以下领域的基础：

- 依赖关系分析
- 反馈检测
- 电路设计
- 图模体分析

#### 工作原理（通俗解释）

Johnson 算法基于带有智能剪枝和强连通分量分解的回溯法：

1.  按顺序处理顶点
    *   考虑顶点 $1, 2, \dots, n$。
2.  对于每个顶点 $s$：
    *   考虑由顶点 ≥ s 诱导的子图。
    *   在此子图中找到强连通分量。
    *   如果 $s$ 属于一个非平凡的 SCC，则探索所有从 $s$ 开始的简单环。
3.  使用阻塞集来避免冗余探索：
    *   一旦一个顶点导致死胡同，就将其标记为阻塞。
    *   当通过它找到一个有效环时，解除阻塞。

这避免了多次探索同一条路径。

#### 示例

图：

```
A → B → C
↑   ↓   |
└── D ←─┘
```

环：

1.  A → B → C → D → A
2.  B → C → D → B

Johnson 算法将高效地找到这两个环，且不会重复。

#### 伪代码

```python
def johnson(G):
    result = [] # 结果列表
    blocked = set() # 阻塞集合
    B = {v: set() for v in G} # 阻塞映射
    stack = [] # 栈

    def circuit(v, s):
        f = False
        stack.append(v)
        blocked.add(v)
        for w in G[v]:
            if w == s:
                result.append(stack.copy())
                f = True
            elif w not in blocked:
                if circuit(w, s):
                    f = True
        if f:
            unblock(v)
        else:
            for w in G[v]:
                B[w].add(v)
        stack.pop()
        return f

    def unblock(u):
        blocked.discard(u)
        for w in B[u]:
            if w in blocked:
                unblock(w)
        B[u].clear()

    for s in sorted(G.keys()):
        # 考虑节点 >= s 的子图
        subG = {v: [w for w in G[v] if w >= s] for v in G if v >= s}
        SCCs = strongly_connected_components(subG) # 强连通分量
        if not SCCs:
            continue
        scc = min(SCCs, key=lambda S: min(S))
        s_node = min(scc)
        circuit(s_node, s_node)
    return result
```

#### 为什么它很重要

- 枚举所有简单环且无重复
- 适用于有向图（与许多仅适用于无向图的算法不同）
- 关键应用于：
  *   死锁检测
  *   环基计算
  *   反馈弧集分析
  *   子图模式挖掘

#### 一个温和的证明（为什么它有效）

每次递归搜索都从 SCC 中最小的顶点开始。
通过将搜索限制在顶点 $\ge s$，每个环恰好被发现一次（在其编号最小的顶点处）。
阻塞集防止了对死胡同的重复探索。
解除阻塞确保顶点在成为有效环的一部分时重新进入搜索空间。

这保证了：

- 没有环被遗漏
- 没有环被重复

#### 自己动手试试

1.  画一个有 3-5 个顶点的小型有向图。
2.  手动识别 SCC。
3.  逐步应用算法，注意 `blocked` 的更新。
4.  当返回到起点时记录每个环。
5.  与朴素的 DFS 枚举进行比较。

#### 复杂度

- 时间：$O((V + E)(C + 1))$
- 空间：$O(V + E)$

Johnson 算法详尽而优雅地揭示了有向图中隐藏的循环，一个接一个。
### 398 传递闭包（Floyd–Warshall）

有向图的传递闭包捕获了可达性：它告诉我们，对于每一对顶点 $(u, v)$，是否存在一条从 $u$ 到 $v$ 的路径。

它通常表示为一个布尔矩阵 $R$，其中
$$
R[u][v] = 1 \text{ 当且仅当存在一条从 } u \text{ 到 } v \text{ 的路径}
$$

这可以通过使用 Floyd–Warshall 算法的修改版本来高效计算。

#### 我们要解决什么问题？

给定一个有向图 $G = (V, E)$，我们想确定对于每一对 $(u, v)$ 是否满足：

$$
u \leadsto v
$$

也就是说，我们能否通过一系列有向边从 $u$ 到达 $v$？

输出是一个可达性矩阵，在以下方面很有用：

- 依赖关系分析
- 访问控制和授权图
- 程序调用图
- 数据库查询优化

#### 工作原理（通俗解释）

我们应用 Floyd–Warshall 动态规划的思想，但不是传播距离，而是传播可达性。

如果 $u \to v$（直接边），则令 $R[u][v] = 1$，否则为 $0$。
然后对于每个顶点 $k$（中间节点），我们更新：

$$
R[u][v] = R[u][v] \lor (R[u][k] \land R[k][v])
$$

直观地说：
"如果 $u$ 能到达 $k$ 并且 $k$ 能到达 $v$，那么 $u$ 就能到达 $v$。"

#### 示例

图：

```
A → B → C
↑         |
└─────────┘
```

初始可达性（直接边）：

```
A B C
A 0 1 0
B 0 0 1
C 1 0 0
```

应用传递闭包后：

```
A B C
A 1 1 1
B 1 1 1
C 1 1 1
```

每个顶点都可以从其他任何顶点到达，该图是强连通的。

#### 微型代码（C）

```c
#define N 100
int R[N][N];

void floyd_warshall_tc(int n) {
    for (int k = 0; k < n; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                R[i][j] = R[i][j] || (R[i][k] && R[k][j]);
}
```

#### 微型代码（Python）

```python
def transitive_closure(R):
    n = len(R)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                R[i][j] = R[i][j] or (R[i][k] and R[k][j])
    return R
```

#### 为什么它很重要

- 将图转换为可达性矩阵
- 支持常数时间查询："$u$ 能到达 $v$ 吗？"
- 用于：

  * 编译器（调用依赖）
  * 数据库（递归查询）
  * 安全图
  * 网络分析

#### 一个温和的证明（为什么它有效）

我们逐步扩展可达性：

- 基础：$R^{(0)}$ = 直接边
- 步骤：$R^{(k)}$ = 使用顶点 ${1, 2, \dots, k}$ 作为中间节点的路径

通过归纳法：
$$
R^{(k)}[i][j] = 1 \iff \text{存在一条路径 } i \to j \text{ 使用的顶点 } \le k
$$

最后（$k = n$），$R^{(n)}$ 包含了所有可能的路径。

#### 自己动手试试

1.  创建一个包含 4-5 个节点的有向图。
2.  构建其邻接矩阵。
3.  手动应用该算法。
4.  观察每次 $k$ 之后新的可达性如何出现。
5.  与你直观看到的路径进行比较。

#### 复杂度

- 时间：$O(V^3)$
- 空间：$O(V^2)$

传递闭包将可达性转化为确定性，将每一条潜在路径映射成一个清晰的视图，展示什么连接到什么。
### 399 图着色（回溯法）

图着色问题是指为图的顶点分配颜色，使得任意两个相邻的顶点颜色不同。
所需的最少颜色数称为图的色数。

这个经典的约束满足问题是调度、寄存器分配和模式分配等问题的核心。

#### 我们要解决什么问题？

给定一个图 $G = (V, E)$ 和一个整数 $k$，
判断是否可以使用最多 $k$ 种颜色为所有顶点着色，并满足：

$$
\forall (u, v) \in E, ; \text{color}(u) \ne \text{color}(v)
$$

如果存在这样的着色方案，则称 $G$ 是 $k$ 可着色的。

#### 工作原理（通俗解释）

我们使用回溯法来解决这个问题：

1.  为第一个顶点分配一种颜色。
2.  移动到下一个顶点。
3.  尝试所有可用的颜色（从 $1$ 到 $k$）。
4.  如果某种颜色分配违反了邻接约束，则跳过它。
5.  如果一个顶点无法着色，则回溯到上一个顶点并更改其颜色。
6.  继续直到：

    *   所有顶点都已着色（成功），或者
    *   不存在有效的分配方案（失败）。

#### 示例

图：

```
1, 2
|   |
3, 4
```

一个正方形至少需要 2 种颜色：

- color(1) = 1
- color(2) = 2
- color(3) = 2
- color(4) = 1

有效的 2 着色。

尝试 1 种颜色 → 失败（相邻节点颜色相同）
尝试 2 种颜色 → 成功 → 色数 = 2

#### 微型代码（C++）

```cpp
int n, k;
vector<vector<int>> adj;
vector<int> color;

bool isSafe(int v, int c) {
    for (int u : adj[v])
        if (color[u] == c) return false;
    return true;
}

bool solve(int v) {
    if (v == n) return true;
    for (int c = 1; c <= k; ++c) {
        if (isSafe(v, c)) {
            color[v] = c;
            if (solve(v + 1)) return true;
            color[v] = 0;
        }
    }
    return false;
}
```

#### 微型代码（Python）

```python
def graph_coloring(graph, k):
    n = len(graph)
    color = [0] * n

    def safe(v, c):
        return all(color[u] != c for u in range(n) if graph[v][u])

    def backtrack(v):
        if v == n:
            return True
        for c in range(1, k + 1):
            if safe(v, c):
                color[v] = c
                if backtrack(v + 1):
                    return True
                color[v] = 0
        return False

    return backtrack(0)
```

#### 为什么它重要

图着色抓住了基于约束的分配问题的本质：

- 调度：为任务分配时间段
- 寄存器分配：将变量映射到 CPU 寄存器
- 地图着色：为有共同边界的区域着色
- 频率分配：为无线网络分配信道

#### 一个温和的证明（为什么它有效）

我们在以下规则下探索所有可能的分配（深度优先搜索）：
$$
\forall (u, v) \in E, ; \text{color}(u) \ne \text{color}(v)
$$

回溯法会剪枝那些无法导致有效完整分配的部分解。
当找到一个完整的着色方案时，约束条件在构造过程中即已满足。

根据回溯法的完备性，如果存在有效的 $k$ 着色方案，它将被找到。

#### 亲自尝试

1.  绘制小图（三角形、正方形、五边形）。
2.  尝试用 $k = 2, 3, 4$ 进行着色。
3.  观察冲突在哪里迫使回溯。
4.  尝试贪心着色并与回溯法比较。
5.  通过实验确定色数。

#### 复杂度

- 时间复杂度：$O(k^V)$（最坏情况下为指数级）
- 空间复杂度：$O(V)$

图着色融合了搜索和逻辑，是在约束中进行的谨慎舞蹈，一次一种颜色地发现和谐。
### 400 关节点与桥

关节点（Articulation Points）和桥（Bridges）用于识别图中的薄弱点，即移除后会增加连通分量数量的节点或边。
它们在分析网络弹性、通信可靠性和双连通分量方面至关重要。

#### 我们要解决什么问题？

给定一个无向图 $G = (V, E)$，找出：

- 关节点（割点）：
  移除后会断开图的顶点。

- 桥（割边）：
  移除后会断开图的边。

我们希望找到高效的算法，在 $O(V + E)$ 时间内检测出它们。

#### 工作原理（通俗解释）

我们使用一次 DFS 遍历（Tarjan 算法）和两个关键数组：

- `disc[v]`：顶点 $v$ 的发现时间
- `low[v]`：从 $v$ 可达（包括通过回边）的最低发现时间

在 DFS 过程中：

- 一个顶点 $u$ 是关节点，如果：

  * $u$ 是根节点且拥有多于一个子节点，或者
  * $\exists$ 子节点 $v$ 使得 `low[v] ≥ disc[u]`

- 一条边 $(u, v)$ 是桥，如果：

  * `low[v] > disc[u]`

这些条件检测的是何时没有回边将一个子树连接回某个祖先。

#### 示例

图：

```
  1
 / \
2   3
|   |
4   5
```

移除节点 2 → 节点 4 变得孤立 → 2 是一个关节点。
移除边 (2, 4) → 增加了连通分量 → (2, 4) 是一座桥。

#### 微型代码（C++）

```cpp
vector<vector<int>> adj;
vector<int> disc, low, parent;
vector<bool> ap;
int timer = 0;

void dfs(int u) {
    disc[u] = low[u] = ++timer;
    int children = 0;

    for (int v : adj[u]) {
        if (!disc[v]) {
            parent[v] = u;
            ++children;
            dfs(v);
            low[u] = min(low[u], low[v]);

            if (parent[u] == -1 && children > 1)
                ap[u] = true;
            if (parent[u] != -1 && low[v] >= disc[u])
                ap[u] = true;
            if (low[v] > disc[u])
                cout << "Bridge: " << u << " - " << v << "\n";
        } else if (v != parent[u]) {
            low[u] = min(low[u], disc[v]);
        }
    }
}
```

#### 微型代码（Python）

```python
def articulation_points_and_bridges(graph):
    n = len(graph)
    disc = [0] * n
    low = [0] * n
    parent = [-1] * n
    ap = [False] * n
    bridges = []
    time = 1

    def dfs(u):
        nonlocal time
        disc[u] = low[u] = time
        time += 1
        children = 0
        for v in graph[u]:
            if not disc[v]:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])

                if parent[u] == -1 and children > 1:
                    ap[u] = True
                if parent[u] != -1 and low[v] >= disc[u]:
                    ap[u] = True
                if low[v] > disc[u]:
                    bridges.append((u, v))
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if not disc[i]:
            dfs(i)

    return [i for i, x in enumerate(ap) if x], bridges
```

#### 为什么这很重要

关节点和桥揭示了以下领域中的关键节点和链接：

- 网络设计：识别基础设施中的薄弱环节
- 社交网络：找到移除后会分裂社区的影响者
- 编译器依赖图：定位关键连接
- 交通系统：确保鲁棒的路由

理解图在何处断裂，有助于我们设计出不会断裂的系统。

#### 一个温和的证明（为什么它有效）

对于每个节点 $u$，`low[u]` 捕获了通过 DFS 或回边可达的最早被发现的顶点。
如果一个子节点子树无法到达 $u$ 的祖先，那么 $u$ 就是一个瓶颈，移除它会分裂图。

不等式：
$$
low[v] \ge disc[u] \implies u \text{ 是关节点}
$$
$$
low[v] > disc[u] \implies (u,v) \text{ 是桥}
$$

源于一个子树是否被连接回一个祖先。

#### 亲自尝试

1.  绘制小图（三角形、线形、星形）。
2.  手动运行 DFS 并记录 `disc` 和 `low`。
3.  识别关节点和桥。
4.  尝试添加边，观察冗余如何消除关节点。

#### 复杂度

- 时间复杂度：$O(V + E)$
- 空间复杂度：$O(V)$

寻找关节点和桥将结构转化为洞察，帮助你构建即使在部分失效时也能保持连接的网络。
