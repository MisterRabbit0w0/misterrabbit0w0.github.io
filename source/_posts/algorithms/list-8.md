---
title: 第八章
date: 2026-02-25 06:00:00
permalink: algorithms/list-8/
disableNunjucks: true
tags:
  - 算法
---

# 第八章：几何、图形与空间算法

# 第七十一节：凸包
### 701 礼品包装算法（Jarvis March）

礼品包装算法，或称 Jarvis March，是用于寻找点集凸包（即包围所有点的最小凸多边形）的最简单、最直观的算法之一。可以想象成用一根橡皮筋缠绕在板子上的钉子上。

它通过反复选择最逆时针的点，一次一个点地“包裹”凸包，直到回到起点。

#### 我们要解决什么问题？

给定平面上的 n 个点，我们希望计算它们的凸包，即按顺序连接最外层点所形成的多边形。凸包在几何学、图形学和机器人学中都是基础概念。

形式上，集合 S 的凸包 H(S) 是包含 S 的最小凸集。

我们希望算法能够：

*   找到凸包上的所有点。
*   按周长顺序排列它们。
*   即使在存在共线点的情况下也能可靠工作。

Jarvis March 概念简单，适用于小型或接近凸集的点集。

#### 它是如何工作的（通俗解释）？

想象你站在最左边的点，然后沿着外围行走，总是尽可能地向左转（逆时针方向）。这确保了我们沿着凸包的边界行进。

算法步骤：

| 步骤 | 描述                                                                                           |
| ---- | ----------------------------------------------------------------------------------------------------- |
| 1    | 从最左边（或最低）的点开始。                                                            |
| 2    | 选择下一个点 p，使得所有其他点都位于直线（当前点, p）的右侧。 |
| 3    | 移动到 p，将其添加到凸包中。                                                                    |
| 4    | 重复直到回到起点。                                                                 |

这模仿了“包裹”所有点的过程，因此得名礼品包装。

#### 示例演练

假设我们有 6 个点：
A(0,0), B(2,1), C(1,2), D(3,3), E(0,3), F(3,0)

从最左边的点 A(0,0) 开始。
从 A 出发，最逆时针的点是 E(0,3)。
从 E 出发，再次向左转 → D(3,3)。
从 D → F(3,0)。
从 F → 回到 A。

凸包 = [A, E, D, F]

#### 微型代码（简易版）

C

```c
#include <stdio.h>

typedef struct { double x, y; } Point;

int orientation(Point a, Point b, Point c) {
    double val = (b.y - a.y) * (c.x - b.x) - 
                 (b.x - a.x) * (c.y - b.y);
    if (val == 0) return 0;      // 共线
    return (val > 0) ? 1 : 2;    // 1: 顺时针, 2: 逆时针
}

void convexHull(Point pts[], int n) {
    if (n < 3) return;
    int hull[100], h = 0;
    int l = 0;
    for (int i = 1; i < n; i++)
        if (pts[i].x < pts[l].x)
            l = i;

    int p = l, q;
    do {
        hull[h++] = p;
        q = (p + 1) % n;
        for (int i = 0; i < n; i++)
            if (orientation(pts[p], pts[i], pts[q]) == 2)
                q = i;
        p = q;
    } while (p != l);

    printf("凸包:\n");
    for (int i = 0; i < h; i++)
        printf("(%.1f, %.1f)\n", pts[hull[i]].x, pts[hull[i]].y);
}

int main(void) {
    Point pts[] = {{0,0},{2,1},{1,2},{3,3},{0,3},{3,0}};
    int n = sizeof(pts)/sizeof(pts[0]);
    convexHull(pts, n);
}
```

Python

```python
def orientation(a, b, c):
    val = (b[1]-a[1])*(c[0]-b[0]) - (b[0]-a[0])*(c[1]-b[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

def convex_hull(points):
    n = len(points)
    if n < 3: return []
    l = min(range(n), key=lambda i: points[i][0])
    hull = []
    p = l
    while True:
        hull.append(points[p])
        q = (p + 1) % n
        for i in range(n):
            if orientation(points[p], points[i], points[q]) == 2:
                q = i
        p = q
        if p == l: break
    return hull

pts = [(0,0),(2,1),(1,2),(3,3),(0,3),(3,0)]
print("凸包:", convex_hull(pts))
```

#### 为什么它很重要

*   简单直观：易于可视化和实现。
*   适用于任何点集，即使是未排序的。
*   输出敏感：时间复杂度取决于凸包点数 *h*。
*   是比较更高级算法（Graham, Chan）的良好基准。

应用：

*   机器人与路径规划（边界检测）
*   计算机图形学（碰撞包络）
*   地理信息系统与制图（区域轮廓）
*   聚类与异常值检测

#### 亲自尝试

1.  尝试用构成正方形、三角形或凹形的点进行测试。
2.  添加共线点，观察它们是否被包含在内。
3.  可视化每个方向判断步骤（绘制箭头）。
4.  统计比较次数（以验证 O(nh)）。
5.  与 Graham Scan 和 Andrew's Monotone Chain 算法进行比较。

#### 测试用例

| 点集                         | 凸包输出         | 备注             |
| ------------------------------ | ------------------- | ----------------- |
| 正方形 (0,0),(0,1),(1,0),(1,1) | 全部 4 个点        | 完美矩形 |
| 三角形 (0,0),(2,0),(1,1)     | 3 个点            | 简单凸形     |
| 凹形                  | 仅外边界 | 忽略凹度 |
| 随机点                  | 变化              | 总是凸形     |

#### 复杂度

*   时间：O(nh)，其中 *h* = 凸包点数
*   空间：O(h) 用于输出列表

礼品包装算法是你在计算几何中的第一个指南针，跟随最左转的方向，数据的形状便会显现出来。
### 702 格雷厄姆扫描法

格雷厄姆扫描法（Graham Scan）是一种快速、优雅的算法，用于寻找一组点的凸包。它的工作原理是围绕一个锚点按角度对点进行排序，然后在扫描过程中通过维护一个转向方向的栈来构建凸包。

可以把它想象成：围绕一个基点排列好所有的星星，然后沿着最外层的环进行描边，而不会退回到内部。

#### 我们要解决什么问题？

给定平面上的 n 个点，我们希望找到凸包，即能包围所有点的最小凸多边形。

与"礼品包装"算法（Gift Wrapping）逐个点地环绕不同，格雷厄姆扫描法首先对点进行排序，然后高效地一次扫描就勾勒出凸包。

我们需要：

* 一致的排序方式（极角）
* 一种测试转向的方法（方向判定）

#### 它是如何工作的？（通俗解释）

1.  选择锚点，即 y 坐标最小的点（如果 y 相同，则选择 x 最小的点）。
2.  根据相对于锚点的极角对所有点进行排序。
3.  扫描排序后的点，维护一个凸包顶点的栈。
4.  对于每个点，检查栈中的最后两个点：
    *   如果它们构成非左转（顺时针），则弹出栈顶的点。
    *   持续此操作直到转向变为左转（逆时针）。
    *   将新点压入栈中。
5.  最后，栈中按顺序保存着凸包的顶点。

#### 示例演练

点集：
A(0,0), B(2,1), C(1,2), D(3,3), E(0,3), F(3,0)

1.  锚点：A(0,0)
2.  按极角排序 → F(3,0), B(2,1), D(3,3), C(1,2), E(0,3)
3.  扫描：
    *   开始 [A, F, B]
    *   检查下一个 D → 左转 → 压入栈
    *   下一个 C → 右转 → 弹出 D
    *   压入 C → 与 B 检查，仍是右转 → 弹出 B
    *   继续直到所有点扫描完毕
        凸包：A(0,0), F(3,0), D(3,3), E(0,3)

#### 精简代码（简易版本）

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct { double x, y; } Point;

Point anchor;

int orientation(Point a, Point b, Point c) {
    double val = (b.y - a.y) * (c.x - b.x) - 
                 (b.x - a.x) * (c.y - b.y);
    if (val == 0) return 0;
    return (val > 0) ? 1 : 2;
}

double dist(Point a, Point b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return dx * dx + dy * dy;
}

int compare(const void *p1, const void *p2) {
    Point a = *(Point *)p1, b = *(Point *)p2;
    int o = orientation(anchor, a, b);
    if (o == 0)
        return dist(anchor, a) < dist(anchor, b) ? -1 : 1;
    return (o == 2) ? -1 : 1;
}

void grahamScan(Point pts[], int n) {
    int ymin = 0;
    for (int i = 1; i < n; i++)
        if (pts[i].y < pts[ymin].y ||
           (pts[i].y == pts[ymin].y && pts[i].x < pts[ymin].x))
            ymin = i;

    Point temp = pts[0];
    pts[0] = pts[ymin];
    pts[ymin] = temp;
    anchor = pts[0];

    qsort(pts + 1, n - 1, sizeof(Point), compare);

    Point stack[100];
    int top = 2;
    stack[0] = pts[0];
    stack[1] = pts[1];
    stack[2] = pts[2];

    for (int i = 3; i < n; i++) {
        while (orientation(stack[top - 1], stack[top], pts[i]) != 2)
            top--;
        stack[++top] = pts[i];
    }

    printf("凸包:\n");
    for (int i = 0; i <= top; i++)
        printf("(%.1f, %.1f)\n", stack[i].x, stack[i].y);
}

int main() {
    Point pts[] = {{0,0},{2,1},{1,2},{3,3},{0,3},{3,0}};
    int n = sizeof(pts)/sizeof(pts[0]);
    grahamScan(pts, n);
}
```

Python

```python
def orientation(a, b, c):
    val = (b[1]-a[1])*(c[0]-b[0]) - (b[0]-a[0])*(c[1]-b[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

def graham_scan(points):
    n = len(points)
    anchor = min(points, key=lambda p: (p[1], p[0]))
    sorted_pts = sorted(points, key=lambda p: (
        atan2(p[1]-anchor[1], p[0]-anchor[0]), (p[0]-anchor[0])2 + (p[1]-anchor[1])2
    ))

    hull = []
    for p in sorted_pts:
        while len(hull) >= 2 and orientation(hull[-2], hull[-1], p) != 2:
            hull.pop()
        hull.append(p)
    return hull

from math import atan2
pts = [(0,0),(2,1),(1,2),(3,3),(0,3),(3,0)]
print("凸包:", graham_scan(pts))
```

#### 为什么它很重要？

*   **高效**：排序带来 O(n log n) 的时间复杂度；扫描是线性的。
*   **鲁棒**：通过打破平局的方式处理共线情况。
*   **经典**：计算几何中基础的凸包算法。

应用：

*   **图形学**：凸轮廓、网格简化
*   **碰撞检测和物理模拟**
*   **地理信息系统（GIS）边界分析**
*   **聚类凸包和凸包围**

#### 亲自尝试

1.  绘制 10 个随机点，按角度对它们排序。
2.  手动追踪转向以观察凸包形状。
3.  添加共线点，测试平局处理。
4.  与 Jarvis March 算法对相同数据进行对比。
5.  测量 n 增长时的性能。

#### 测试用例

| 点集                      | 凸包           | 备注                 |
| ------------------------- | -------------- | -------------------- |
| 正方形角点                | 全部 4 个点    | 经典凸包             |
| 三角形 + 内部点           | 3 个外部点     | 内部点被忽略         |
| 共线点                    | 仅端点         | 正确处理             |
| 随机散点                  | 外部环         | 已验证形状           |

#### 复杂度

*   时间：O(n log n)
*   空间：O(n)（用于排序和栈）

格雷厄姆扫描法融合了几何与顺序，对星星排序，跟随转向，凸包便清晰而锐利地显现出来。
### 703 Andrew 单调链算法

Andrew 单调链算法是一种简洁高效的凸包算法，既易于实现，在实践中速度也很快。它本质上是 Graham 扫描算法的一个简化变体，但不是按角度排序，而是按 x 坐标排序，并通过两次扫描构建凸包：一次用于下凸包，一次用于上凸包。

可以把它想象成建造两次围栏，一次沿着底部，一次沿着顶部，然后将它们连接成一个完整的边界。

#### 我们要解决什么问题？

给定 n 个点，找出它们的凸包，即包围这些点的最小凸多边形。

Andrew 算法提供了：

* 按 x（和 y）的确定性排序
* 基于循环的简单构建（无需角度计算）
* 一个 O(n log n) 的解决方案，与 Graham 扫描算法相当

因其简单性和数值稳定性而被广泛使用。

#### 它是如何工作的（通俗解释）？

1.  按 x 坐标，然后按 y 坐标对所有点进行字典序排序。
2.  构建下凸包：
    *   从左到右遍历点。
    *   当最后两个点加上新点构成非左转时，弹出最后一个点。
    *   压入新点。
3.  构建上凸包：
    *   从右到左遍历点。
    *   重复相同的弹出规则。
4.  连接下凸包和上凸包，排除重复的端点。

最终会得到一个按逆时针顺序排列的完整凸包。

#### 示例演练

点：
A(0,0), B(2,1), C(1,2), D(3,3), E(0,3), F(3,0)

1.  按 x 排序 → A(0,0), E(0,3), C(1,2), B(2,1), D(3,3), F(3,0)

2.  下凸包
    *   起始 A(0,0), E(0,3) → 右转 → 弹出 E
    *   添加 C(1,2), B(2,1), F(3,0) → 只保留左转
        → 下凸包: [A, B, F]

3.  上凸包
    *   起始 F(3,0), D(3,3), E(0,3), A(0,0) → 保持左转
        → 上凸包: [F, D, E, A]

4.  合并（移除重复点）：
    凸包: [A, B, F, D, E]

#### 精简代码（简易版）

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct { double x, y; } Point;

int cmp(const void *a, const void *b) {
    Point p = *(Point*)a, q = *(Point*)b;
    if (p.x == q.x) return (p.y > q.y) - (p.y < q.y);
    return (p.x > q.x) - (p.x < q.x);
}

double cross(Point o, Point a, Point b) {
    return (a.x - o.x)*(b.y - o.y) - (a.y - o.y)*(b.x - o.x);
}

void monotoneChain(Point pts[], int n) {
    qsort(pts, n, sizeof(Point), cmp);

    Point hull[200];
    int k = 0;

    // 构建下凸包
    for (int i = 0; i < n; i++) {
        while (k >= 2 && cross(hull[k-2], hull[k-1], pts[i]) <= 0)
            k--;
        hull[k++] = pts[i];
    }

    // 构建上凸包
    for (int i = n-2, t = k+1; i >= 0; i--) {
        while (k >= t && cross(hull[k-2], hull[k-1], pts[i]) <= 0)
            k--;
        hull[k++] = pts[i];
    }

    k--; // 最后一个点与第一个点相同

    printf("凸包:\n");
    for (int i = 0; i < k; i++)
        printf("(%.1f, %.1f)\n", hull[i].x, hull[i].y);
}

int main() {
    Point pts[] = {{0,0},{2,1},{1,2},{3,3},{0,3},{3,0}};
    int n = sizeof(pts)/sizeof(pts[0]);
    monotoneChain(pts, n);
}
```

Python

```python
def cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def monotone_chain(points):
    points = sorted(points)
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]

pts = [(0,0),(2,1),(1,2),(3,3),(0,3),(3,0)]
print("凸包:", monotone_chain(pts))
```

#### 为什么它很重要

*   比 Graham 扫描算法更简单，无需极角排序
*   稳定且对共线点鲁棒
*   由于实现简洁，在实践中常用
*   是学习二维计算几何的良好起点

应用：

*   二维碰撞检测
*   图形学中的凸包络
*   地图绘制中的边界区域
*   高级几何算法（Voronoi， Delaunay）的凸包预处理

#### 自己动手试试

1.  按 x 排序并手动画出点。
2.  逐步执行两次扫描（下凸包，上凸包）。
3.  可视化非左转时的弹出过程。
4.  添加共线点，验证处理方式。
5.  与 Graham 扫描算法和 Jarvis 步进法比较凸包结果。

#### 测试用例

| 点集                      | 凸包                | 备注             |
| ------------------------- | ------------------- | ---------------- |
| 正方形（4 个角点）        | 4 个角点            | 经典矩形         |
| 三角形 + 中心点           | 外部的 3 个点       | 中心点被忽略     |
| 共线点                    | 2 个端点            | 正确处理         |
| 随机散点                  | 正确的凸环          | 稳定             |

#### 复杂度

*   时间：O(n log n)（排序占主导）
*   空间：O(n)

Andrew 单调链算法是几何算法简洁性的典范：排序、扫描、缝合，一个简单的循环就能勾勒出完美的边界。
### 704 Chan 算法

Chan 算法是一种巧妙的输出敏感凸包算法，这意味着它的运行时间不仅取决于点的总数 *n*，还取决于实际构成凸包的点的数量 *h*。它巧妙地结合了 Graham Scan 和 Jarvis March，取两者之长。

可以把它想象成组织一大群人：先分组，描绘每组的边界，然后将这些外部线条合并成一个平滑的凸包。

#### 我们要解决什么问题？

我们想找到一个包含 *n* 个点的集合的凸包，但如果只有少数点在凸包上，我们不想承担对所有点进行排序的全部代价。

Chan 算法通过以下方式解决这个问题：

*   子问题分解（分成若干块）
*   快速局部凸包计算（通过 Graham Scan）
*   高效合并（通过包装法）

结果：
O(n log h) 的时间复杂度，当 *h* 较小时更快。

#### 它是如何工作的（通俗解释）？

Chan 算法主要分为三个步骤：

| 步骤 | 描述                                                                                                                          |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | 将点集划分为大小为 *m* 的组。                                                                                        |
| 2    | 对每个组，计算局部凸包（使用 Graham Scan）。                                                                   |
| 3    | 在所有局部凸包上使用礼品包装法（Jarvis March）来寻找全局凸包，但将探索的凸包顶点数量限制在 *m* 个以内。 |

如果失败（h > m），则将 m 加倍并重复此过程。

这种"猜测并检查"的方法确保你能在 *O(n log h)* 时间内找到完整的凸包。

#### 示例演练

假设有 30 个点，但只有 6 个点构成凸包。

1.  选择 *m = 4*，因此大约有 8 个组。
2.  用 Graham Scan（快速）计算每个组的凸包。
3.  通过包装法合并，在每一步中，从所有局部凸包中选择下一个切点。
4.  如果需要超过 *m* 步，则将 *m* 加倍 → *m = 8*，重复。
5.  找到所有凸包顶点后，停止。

结果：以最少的额外工作得到全局凸包。

#### 微型代码（概念性伪代码）

这个算法很复杂，但这里有一个简单的概念版本：

```python
def chans_algorithm(points):
    import math
    n = len(points)
    m = 1
    while True:
        m = min(2*m, n)
        groups = [points[i:i+m] for i in range(0, n, m)]

        # 步骤 1: 计算局部凸包
        local_hulls = [graham_scan(g) for g in groups]

        # 步骤 2: 使用包装法合并
        hull = []
        start = min(points)
        p = start
        for k in range(m):
            hull.append(p)
            q = None
            for H in local_hulls:
                # 在每个局部凸包上选择切点
                cand = tangent_from_point(p, H)
                if q is None or orientation(p, q, cand) == 2:
                    q = cand
            p = q
            if p == start:
                return hull
```

核心思想：高效地合并小凸包，而无需每次都重新处理所有点。

#### 为什么它重要

*   输出敏感：当凸包尺寸较小时性能最佳。
*   连接理论与实践，展示了如何结合算法来降低渐近成本。
*   展示了分治法与包装法的协同作用。
*   是高维凸包的重要理论基础。

应用：

*   几何计算框架
*   机器人路径包络
*   计算几何库
*   性能关键的制图或碰撞检测系统

#### 亲自尝试

1.  尝试 *h* 较小（凸包点少）而 *n* 较大的情况，注意更快的性能。
2.  与 Graham Scan 比较运行时间。
3.  可视化分组及其局部凸包。
4.  跟踪每次迭代中 *m* 的翻倍情况。
5.  测量随着凸包增大性能的增长情况。

#### 测试用例

| 点集                       | 凸包                | 备注               |
| ---------------------------- | ------------------- | ------------------- |
| 6 点凸集           | 所有点          | 单次迭代    |
| 密集簇 + 少量离群点 | 仅外部边界 | 输出敏感    |
| 随机二维点集                    | 正确凸包        | 与 Graham Scan 匹配 |
| 1000 个点，10 个凸包点        | O(n log 10)         | 非常快           |

#### 复杂度

*   时间：O(n log h)
*   空间：O(n)
*   最佳适用场景：凸包尺寸相对于总点数较小

Chan 算法是几何学中安静的优化器，它猜测、测试、折返，一次一层地包裹着世界。
### 705 快速凸包（QuickHull）

快速凸包是一种用于寻找凸包的分治算法，在概念上类似于快速排序，但应用于几何领域。它递归地将点集分割成更小的组，寻找极值点并逐步构建凸包。

想象一下你在用橡皮筋套住钉子：选取最远的钉子，画一条线，然后将其余的点分成位于该线上方和下方的两组。重复此过程，直到每个线段都“绷紧”。

#### 我们要解决什么问题？

给定 n 个点，我们希望构建凸包，即包含所有点的最小凸多边形。

快速凸包通过以下方式实现：

* 选择极值点作为锚点
* 将点集划分为子问题
* 递归地寻找构成凸包边缘的最远点

这种方法很直观，并且平均情况下通常很快，但在最坏情况下（例如所有点都在凸包上）可能会退化到 *O(n²)*。

#### 它是如何工作的？（通俗解释）

1.  找到最左边和最右边的点（A 和 B）。它们构成了凸包的基线。
2.  将点分成两组：
    *   在线段 AB 上方
    *   在线段 AB 下方
3.  对于每一侧：
    *   找到距离线段 AB 最远的点 C。
    *   这形成了一个三角形 ABC。
    *   任何位于三角形 ABC 内部的点都不在凸包上。
    *   对位于外部的子集（A–C 和 C–B）进行递归处理。
4.  合并来自两侧的递归凸包。

每个递归步骤添加一个顶点，即最远的点，从而逐步构建凸包。

#### 示例演练

点集：
A(0,0), B(4,0), C(2,3), D(1,1), E(3,1)

1.  最左点 = A(0,0), 最右点 = B(4,0)
2.  AB 上方的点 = {C}, AB 下方的点 = {}
3.  距离 AB 最远的点（上方）= C(2,3)
   → 凸包边：A–C–B
4.  AB 下方没有剩余点 → 完成

凸包 = [A(0,0), C(2,3), B(4,0)]

#### 精简代码（简易版本）

C

```c
#include <stdio.h>
#include <math.h>

typedef struct { double x, y; } Point;

double cross(Point a, Point b, Point c) {
    return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x);
}

double distance(Point a, Point b, Point c) {
    return fabs(cross(a, b, c));
}

void quickHullRec(Point pts[], int n, Point a, Point b, int side) {
    int idx = -1;
    double maxDist = 0;

    for (int i = 0; i < n; i++) {
        double val = cross(a, b, pts[i]);
        if ((side * val) > 0 && fabs(val) > maxDist) {
            idx = i;
            maxDist = fabs(val);
        }
    }

    if (idx == -1) {
        printf("(%.1f, %.1f)\n", a.x, a.y);
        printf("(%.1f, %.1f)\n", b.x, b.y);
        return;
    }

    quickHullRec(pts, n, a, pts[idx], -cross(a, pts[idx], b) < 0 ? 1 : -1);
    quickHullRec(pts, n, pts[idx], b, -cross(pts[idx], b, a) < 0 ? 1 : -1);
}

void quickHull(Point pts[], int n) {
    int min = 0, max = 0;
    for (int i = 1; i < n; i++) {
        if (pts[i].x < pts[min].x) min = i;
        if (pts[i].x > pts[max].x) max = i;
    }
    Point A = pts[min], B = pts[max];
    quickHullRec(pts, n, A, B, 1);
    quickHullRec(pts, n, A, B, -1);
}

int main() {
    Point pts[] = {{0,0},{4,0},{2,3},{1,1},{3,1}};
    int n = sizeof(pts)/sizeof(pts[0]);
    printf("凸包:\n");
    quickHull(pts, n);
}
```

Python

```python
def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def distance(a, b, c):
    return abs(cross(a, b, c))

def quickhull_rec(points, a, b, side):
    idx, max_dist = -1, 0
    for i, p in enumerate(points):
        val = cross(a, b, p)
        if side * val > 0 and abs(val) > max_dist:
            idx, max_dist = i, abs(val)
    if idx == -1:
        return [a, b]
    c = points[idx]
    return (quickhull_rec(points, a, c, -1 if cross(a, c, b) > 0 else 1) +
            quickhull_rec(points, c, b, -1 if cross(c, b, a) > 0 else 1))

def quickhull(points):
    points = sorted(points)
    a, b = points[0], points[-1]
    return list({*quickhull_rec(points, a, b, 1), *quickhull_rec(points, a, b, -1)})

pts = [(0,0),(4,0),(2,3),(1,1),(3,1)]
print("凸包:", quickhull(pts))
```

#### 为什么它很重要

*   优雅且递归，概念简单。
*   对于随机点，具有良好的平均性能。
*   分治设计教授了几何递归。
*   直观的可视化，便于教学凸包概念。

应用领域：

*   几何建模
*   游戏开发（碰撞包络）
*   路径规划和网格简化
*   空间数据集的可视化工具

#### 亲自尝试

1.  绘制随机点并逐步进行递归分割。
2.  添加共线点，观察如何处理它们。
3.  与 Graham Scan 算法比较步骤数。
4.  测试在稀疏与密集凸包上的运行时间。
5.  可视化追踪递归树，每个节点代表一条凸包边。

#### 测试用例

| 点集                    | 凸包           | 备注                   |
| ----------------------- | -------------- | ----------------------- |
| 三角形                  | 3 个点         | 简单凸包                |
| 正方形角点 + 中心点     | 4 个角点       | 中心点被忽略            |
| 随机散点                | 外环           | 与其他算法结果一致      |
| 所有点共线              | 仅端点         | 处理退化情况            |

#### 复杂度

*   平均：O(n log n)
*   最坏：O(n²)
*   空间：O(n)（递归栈）

快速凸包是快速排序的几何兄弟，分割、递归，并将片段连接成一个清晰的凸边界。
### 706 增量凸包

增量凸包算法逐步构建凸包，从一个小的凸集（如三角形）开始，每次插入一个点，并在添加每个点时动态更新凸包。

这就像围绕点生长一个肥皂泡：每个新点要么漂浮在内部（被忽略），要么将泡壁向外推（更新凸包）。

#### 我们要解决什么问题？

给定 n 个点，我们想要构建它们的凸包。

与排序或分割的方法（如 Graham 扫描或快速凸包）不同，增量方法：

* 从少数几个点构建初始凸包
* 添加每个剩余的点
* 当新点扩展边界时，更新凸包的边

这种模式可以很好地推广到更高维度，使其成为 3D 凸包和计算几何库的基础。

#### 它是如何工作的（通俗解释）？

1.  从一个小的凸包开始（例如，前 3 个不共线的点）。
2.  对于每个新点 P：
    *   检查 P 是否在当前凸包内部。
    *   如果不是：
        *   找到所有可见的边（面向 P 的边）。
        *   将这些边从凸包中移除。
        *   将 P 连接到可见区域的边界。
3.  继续处理，直到所有点都处理完毕。

凸包会增量地增长，始终保持凸性。

#### 示例演练

点：
A(0,0), B(4,0), C(2,3), D(1,1), E(3,2)

1.  用 {A, B, C} 开始构建凸包。
2.  添加 D(1,1)：位于凸包内部 → 忽略。
3.  添加 E(3,2)：位于边界上或内部 → 忽略。

凸包保持为 [A, B, C]。

如果你添加 F(5,1)：

*   F 位于外部，因此更新凸包以包含它 → [A, B, F, C]

#### 微型代码（简易版）

C（概念版）

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct { double x, y; } Point;

double cross(Point a, Point b, Point c) {
    return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x);
}

// 简化的 2D 增量凸包（无边剪枝）
void incrementalHull(Point pts[], int n) {
    // 从前 3 个点构成的三角形开始
    Point hull[100];
    int h = 3;
    for (int i = 0; i < 3; i++) hull[i] = pts[i];

    for (int i = 3; i < n; i++) {
        Point p = pts[i];
        int visible[100], count = 0;

        // 标记从 p 点可见的边
        for (int j = 0; j < h; j++) {
            Point a = hull[j];
            Point b = hull[(j+1)%h];
            if (cross(a, b, p) > 0) visible[count++] = j;
        }

        // 如果没有可见边，则点在内部
        if (count == 0) continue;

        // 移除可见边并插入新连接（简化版）
        // 这里：我们仅为演示打印添加的点
        printf("将点 (%.1f, %.1f) 添加到凸包\n", p.x, p.y);
    }

    printf("最终凸包（近似）：\n");
    for (int i = 0; i < h; i++)
        printf("(%.1f, %.1f)\n", hull[i].x, hull[i].y);
}

int main() {
    Point pts[] = {{0,0},{4,0},{2,3},{1,1},{3,2},{5,1}};
    int n = sizeof(pts)/sizeof(pts[0]);
    incrementalHull(pts, n);
}
```

Python（简化版）

```python
def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def is_inside(hull, p):
    for i in range(len(hull)):
        a, b = hull[i], hull[(i+1)%len(hull)]
        if cross(a, b, p) > 0:
            return False
    return True

def incremental_hull(points):
    hull = points[:3]
    for p in points[3:]:
        if not is_inside(hull, p):
            hull.append(p)
            # 实际上，按逆时针顺序重新排序凸包
            hull = sorted(hull, key=lambda q: (q[0], q[1]))
    return hull

pts = [(0,0),(4,0),(2,3),(1,1),(3,2),(5,1)]
print("凸包:", incremental_hull(pts))
```

#### 为什么它很重要？

*   概念简单，易于扩展到 3D 及更高维度。
*   在线性：可以在点流式输入时动态更新凸包。
*   用于实时模拟、碰撞检测和几何库。
*   是动态凸包维护的基础（下一节）。

应用：

*   增量几何算法
*   数据流和实时凸性检查
*   增量构建 Delaunay 三角剖分或 Voronoi 图结构

#### 自己动手试试

1.  逐个添加点，在每一步绘制凸包。
2.  观察内部点如何不改变凸包。
3.  尝试随机的插入顺序，凸包保持一致。
4.  与 Graham 扫描的静态方法进行比较。
5.  使用可见面检测扩展到 3D。

#### 测试用例

| 点集                     | 凸包            | 备注           |
| ------------------------ | --------------- | -------------- |
| 三角形 + 内部点          | 外部的 3 个点   | 内部点被忽略   |
| 正方形 + 中心点          | 仅角点          | 有效           |
| 随机点                   | 外环            | 已验证         |
| 增量添加                 | 正确更新        | 动态凸包       |

#### 复杂度

*   时间：朴素方法 O(n²)，优化后 O(n log n)
*   空间：O(h)

增量方法教会了几何学的耐心，一次一个点，随着世界的增长重塑边界。
### 707 分治凸包算法

分治凸包算法通过将点集分成两半，递归计算每一半的凸包，然后合并它们来构建凸包，很像归并排序，但应用于几何问题。

想象一下，将你的点集切成两片云，分别包裹每片云，然后将两个包裹缝合成一个平滑的边界。

#### 我们要解决什么问题？

给定平面上的 n 个点，我们想要构建它们的凸包。

分治方法提供了：

* 清晰的 O(n log n) 时间复杂度
* 优雅的结构（递归 + 合并）
* 为高维凸包提供了坚实的基础

这是将分治策略应用于几何数据的经典示例。

#### 它是如何工作的（通俗解释）？

1. 将所有点按 x 坐标排序。
2. 将点分成两半。
3. 递归计算每一半的凸包。
4. 合并两个凸包：
   * 找到上切线：从上方接触两个凸包的直线
   * 找到下切线：从下方接触两个凸包的直线
   * 移除切线之间的内部点
   * 连接剩余的点以形成合并后的凸包

这个过程重复进行，直到所有点都被一个凸边界包围。

#### 示例演练

点：
A(0,0), B(4,0), C(2,3), D(1,1), E(3,2)

1. 按 x 排序: [A, D, C, E, B]
2. 分割: 左半 = [A, D, C], 右半 = [E, B]
3. 凸包(左半) = [A, C]
   凸包(右半) = [E, B]
4. 合并:
   * 找到上切线 → 连接 C 和 E
   * 找到下切线 → 连接 A 和 B
     凸包 = [A, B, E, C]

#### 微型代码（概念性伪代码）

为了说明逻辑（省略了底层的切线查找细节）：

```python
def divide_conquer_hull(points):
    n = len(points)
    if n <= 3:
        # 基本情况：简单的凸多边形
        return sorted(points)
    
    mid = n // 2
    left = divide_conquer_hull(points[:mid])
    right = divide_conquer_hull(points[mid:])
    return merge_hulls(left, right)

def merge_hulls(left, right):
    # 找到上切线和下切线
    upper = find_upper_tangent(left, right)
    lower = find_lower_tangent(left, right)
    # 合并切线之间的点
    hull = []
    i = left.index(upper[0])
    while left[i] != lower[0]:
        hull.append(left[i])
        i = (i + 1) % len(left)
    hull.append(lower[0])
    j = right.index(lower[1])
    while right[j] != upper[1]:
        hull.append(right[j])
        j = (j + 1) % len(right)
    hull.append(upper[1])
    return hull
```

在实践中，切线查找使用方向测试和循环遍历。

#### 为什么它很重要

* 优雅的递归：几何学与算法设计的结合。
* 平衡的性能：确定性的 O(n log n)。
* 非常适合批处理或并行实现。
* 能很好地扩展到 3D 凸包（在平面上分割）。

应用：

* 计算几何工具包
* 空间分析和地图合并
* 并行几何处理
* 基于几何的聚类

#### 亲自尝试

1.  画 10 个点，按 x 中点分割。
2.  手动构建左半和右半的凸包。
3.  找到上/下切线并合并。
4.  将结果与 Graham Scan 算法进行比较。
5.  追踪递归树（像归并排序一样）。

#### 测试用例

| 点集             | 凸包           | 备注               |
| ---------------- | -------------- | ------------------ |
| 三角形           | 3 个点         | 简单基本情况       |
| 正方形           | 所有角点       | 完美合并           |
| 随机散点         | 外边界         | 已验证             |
| 共线点           | 仅端点         | 正确               |

#### 复杂度

* 时间: O(n log n)
* 空间: O(n)
* 最佳情况: 平衡分割 → 高效合并

分治凸包算法是几何的和谐，每一半找到自己的形状，然后它们共同勾勒出所有点的完美轮廓。
### 708 三维凸包

三维凸包是平面凸包在空间中的自然延伸。不同于将点连接成多边形，而是将它们连接成一个多面体，一个包裹所有给定点的三维外壳。

可以将其想象为在三维空间中用收缩膜包裹散落的鹅卵石，它会收紧成一个由三角形面构成的表面。

#### 我们要解决什么问题？

给定三维空间中的 n 个点，找到完全包裹它们的凸多面体（一组三角形面）。

我们需要计算：

*   顶点（位于凸包上的点）
*   边（顶点之间的连线）
*   面（构成表面的平面小面）

目标：
一个最小的面集合，使得每个点都位于凸包内部或表面上。

#### 它是如何工作的（通俗解释）？

有多种算法可以从二维扩展到三维，但一种经典的方法是增量式三维凸包算法：

| 步骤 | 描述                                                                   |
| ---- | ----------------------------------------------------------------------------- |
| 1    | 从一个非退化四面体（4个不共面的点）开始。 |
| 2    | 对于每个剩余的点 P：                                                   |
|      | – 识别可见面（P 位于其外部的面）。                      |
|      | – 移除这些面（形成一个"洞"）。                                      |
|      | – 创建连接 P 到该洞边界的新面。              |
| 3    | 继续处理直到所有点都被处理完毕。                                      |
| 4    | 剩余的面定义了三维凸包。                            |

每次插入要么添加新面，要么点位于内部而被忽略。

#### 示例演练

点：
A(0,0,0), B(1,0,0), C(0,1,0), D(0,0,1), E(1,1,1)

1.  从基础四面体开始：A, B, C, D
2.  添加 E(1,1,1)：

    *   找出从 E 可见的面
    *   移除它们
    *   将 E 连接到可见区域的边界边
3.  新的凸包有 5 个顶点，形成一个凸多面体。

#### 微型代码（概念性伪代码）

这是一个高层次的想法，实际版本使用复杂的数据结构（面邻接关系、冲突图）：

```python
def incremental_3d_hull(points):
    hull = initialize_tetrahedron(points)
    for p in points:
        if point_inside_hull(hull, p):
            continue
        visible_faces = [f for f in hull if face_visible(f, p)]
        hole_edges = find_boundary_edges(visible_faces)
        hull = [f for f in hull if f not in visible_faces]
        for e in hole_edges:
            hull.append(make_face(e, p))
    return hull
```

每个面由三个点 (a, b, c) 表示，通过行列式或三重积进行方向测试。

#### 为什么它很重要？

*   是三维几何、网格、实体和物理的基础。
*   用于计算几何、图形学、CAD、物理引擎。
*   构成以下内容的构建模块：

    *   Delaunay 三角剖分（三维）
    *   Voronoi 图（三维）
    *   凸分解和碰撞检测

应用：

*   三维建模和渲染
*   凸分解（物理引擎）
*   空间分析，凸包络
*   游戏几何，网格简化

#### 亲自尝试

1.  从 4 个不共面的点开始，可视化四面体。
2.  添加一个外部的点并勾勒出新面。
3.  添加一个内部的点，确认凸包没有变化。
4.  比较立方体角点、随机点、球面采样点的三维凸包。
5.  使用几何查看器逐步可视化更新过程。

#### 测试用例

| 点集                  | 凸包输出 | 说明            |
| ----------------------- | ----------- | ---------------- |
| 4 个不共面的点          | 四面体 | 基础情况        |
| 立方体角点            | 8 个顶点  | 经典盒状凸包 |
| 球面上的随机点 | 所有点  | 凸集       |
| 随机内部点  | 仅外部点  | 内部点被忽略    |

#### 复杂度

*   时间：平均 O(n log n)，最坏情况 O(n²)
*   空间：O(n)

三维凸包将几何提升到空间，从包裹一根线到包裹一个表面，它将散乱的点转化为形状。
### 709 动态凸包

动态凸包是一种数据结构（及算法族），它能在插入点（有时也包括删除点）时维护凸包，而无需从头重新计算整个凸包。

可以把它想象成一个有生命的橡皮筋，当你添加或移除图钉时，它会弯曲并收紧，始终调整自身以保持凸性。

#### 我们要解决什么问题？

给定一系列更新操作（点的插入或删除），我们希望高效地维护当前的凸包，以便：

*   `Insert(point)` 能在亚线性时间内调整凸包。
*   `Query()` 能返回凸包或回答问题（面积、直径、点定位）。
*   `Delete(point)`（可选）能删除一个点并修复凸包。

当数据不断演变、点流式输入、智能体移动或数据集增量更新时，动态凸包至关重要。

#### 它是如何工作的（通俗解释）？

根据是否需要完全动态（插入+删除）还是半动态（仅插入），存在多种策略：

| 变体                     | 思路                                         | 复杂度                     |
| ------------------------ | -------------------------------------------- | -------------------------- |
| 半动态                   | 仅支持插入，增量式维护凸包                   | 每次插入摊销 O(log n)      |
| 完全动态                 | 同时支持插入和删除                           | 每次更新 O(log² n)         |
| 在线凸包（1D / 2D）      | 分别维护上链和下链                           | 对数级更新                 |

常见结构：

1.  将凸包分割为上链和下链。
2.  将每条链存储在平衡二叉搜索树或有序集合中。
3.  插入时：
    *   按 x 坐标定位插入位置。
    *   检查转向方向（方向测试）。
    *   移除内部点（非凸点）并添加新顶点。
4.  删除时：
    *   移除顶点，重新连接相邻点，重新检查凸性。

#### 示例演练（半动态）

从空凸包开始。
逐个插入点：

1.  添加 A(0,0) → 凸包 = [A]
2.  添加 B(2,0) → 凸包 = [A, B]
3.  添加 C(1,2) → 凸包 = [A, B, C]
4.  添加 D(3,1)：
    *   上链 = [A, C, D]
    *   下链 = [A, B, D]
      凸包动态更新，无需重新计算所有点。

如果 D 位于内部，则跳过它。
如果 D 扩展了凸包，则移除被覆盖的边并重新插入。

#### 微型代码（Python 草图）

使用排序链的简单增量凸包：

```python
def cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

class DynamicHull:
    def __init__(self):
        self.upper = []
        self.lower = []

    def insert(self, p):
        self._insert_chain(self.upper, p, 1)
        self._insert_chain(self.lower, p, -1)

    def _insert_chain(self, chain, p, sign):
        chain.append(p)
        chain.sort()  # 按 x 坐标维护顺序
        while len(chain) >= 3 and sign * cross(chain[-3], chain[-2], chain[-1]) <= 0:
            del chain[-2]

    def get_hull(self):
        return self.lower[:-1] + self.upper[::-1][:-1]

# 示例
dh = DynamicHull()
for p in [(0,0),(2,0),(1,2),(3,1)]:
    dh.insert(p)
print("凸包:", dh.get_hull())
```

#### 为什么它很重要

*   **实时几何**：用于移动点集、游戏、机器人技术。
*   **流式分析**：实时数据的凸包络。
*   **增量算法**：无需完全重建即可维护凸性。
*   **数据结构研究**：将几何与平衡树联系起来。

应用：

*   碰撞检测（物体逐步移动）
*   实时可视化
*   几何中位数或边界区域更新
*   计算几何库（CGAL, Boost.Geometry）

#### 动手尝试

1.  逐个插入点，每次插入后画出凸包草图。
2.  尝试插入一个内部点（凸包不变）。
3.  插入一个外部点，观察边的移除和添加。
4.  扩展代码以处理删除操作。
5.  与增量凸包（静态顺序）进行比较。

#### 测试用例

| 操作               | 结果             | 备注                 |
| ------------------ | ---------------- | -------------------- |
| 插入外部点         | 凸包扩展         | 预期增长             |
| 插入内部点         | 无变化           | 保持稳定             |
| 插入共线点         | 添加端点         | 内部点被忽略         |
| 删除凸包顶点       | 重新连接边界     | 完全动态变体         |

#### 复杂度

*   半动态（仅插入）：每次插入摊销 O(log n)
*   完全动态：每次更新 O(log² n)
*   查询（返回凸包）：O(h)

动态凸包是一种随时间增长的形状，是对极值点的记忆，时刻准备着让下一个点来弯曲它的边界。
### 710 旋转卡尺

旋转卡尺技术是一种几何学利器，它通过围绕凸多边形的边界“旋转”一组假想的卡尺，来系统地探索其上的点对、边或方向。

这就像在凸包周围放置一对测量臂，同步旋转它们，并在每一步记录距离、宽度或直径。

#### 我们要解决什么问题？

一旦你有了一个凸包，许多几何量都可以使用旋转卡尺高效计算：

* 最远点对（直径）
* 最小宽度 / 边界框
* 最近平行边对
* 对跖点对
* 给定方向上的多边形面积和宽度

它将几何扫描转化为一次 O(n) 的遍历，无需嵌套循环。

#### 它是如何工作的（通俗解释）？

1.  从一个凸多边形开始（点按逆时针顺序排列）。
2.  想象一把卡尺：一条线接触一个顶点，另一条平行线接触对边。
3.  围绕凸包旋转这些卡尺：
    * 在每一步，推进其下一条边导致较小旋转的那一侧。
    * 测量你需要的任何量（距离、面积、宽度）。
4.  当卡尺完成一整圈旋转时停止。

每一个“事件”（顶点对齐）都对应一个对跖点对，这对于寻找极值距离很有用。

#### 示例演练：最远点对（直径）

凸包：A(0,0), B(4,0), C(4,3), D(0,3)

1.  从边 AB 开始，找到距离 AB 最远的点 (D)。
2.  将卡尺旋转到下一条边 (BC)，根据需要推进对侧点。
3.  继续旋转直到完成完整扫描。
4.  跟踪找到的最大距离 → 此处：A(0,0) 和 C(4,3) 之间

结果：直径 = 5

#### 微型代码（Python）

在凸包上使用旋转卡尺寻找最远点对（直径）：

```python
from math import dist

def rotating_calipers(hull):
    n = len(hull)
    if n == 1:
        return (hull[0], hull[0], 0)
    if n == 2:
        return (hull[0], hull[1], dist(hull[0], hull[1]))

    def area2(a, b, c):
        return abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))

    max_d = 0
    best_pair = (hull[0], hull[0])
    j = 1
    for i in range(n):
        ni = (i + 1) % n
        while area2(hull[i], hull[ni], hull[(j+1)%n]) > area2(hull[i], hull[ni], hull[j]):
            j = (j + 1) % n
        d = dist(hull[i], hull[j])
        if d > max_d:
            max_d = d
            best_pair = (hull[i], hull[j])
    return best_pair + (max_d,)

# 示例凸包（矩形）
hull = [(0,0),(4,0),(4,3),(0,3)]
a, b, d = rotating_calipers(hull)
print(f"最远点对: {a}, {b}, 距离={d:.2f}")
```

#### 为何重要

*   为许多几何问题提供了优雅的 O(n) 解决方案
*   将几何搜索转化为同步扫描
*   广泛应用于计算几何、图形学和机器人学
*   是边界框、最小宽度和碰撞算法中的核心步骤

应用：

*   形状分析（直径、宽度、边界框）
*   碰撞检测（物理引擎中的支撑函数）
*   机器人学（间隙计算）
*   GIS 和地图绘制（有向凸包属性）

#### 动手尝试

1.  画一个凸多边形。
2.  放置一对平行线，使其与两条相对的边相切。
3.  旋转它们并记录最远点对。
4.  与暴力 O(n²) 距离检查进行比较。
5.  扩展以计算最小面积边界框。

#### 测试用例

| 凸包              | 结果               | 备注         |
| ----------------- | ------------------ | ------------ |
| 4×3 矩形          | A(0,0)-C(4,3)      | 对角线 = 5   |
| 三角形            | 最长边             | 有效         |
| 正六边形          | 相对顶点           | 对称         |
| 不规则多边形      | 对跖最大点对       | 已验证       |

#### 复杂度

*   时间：O(n)（围绕凸包的线性扫描）
*   空间：O(1)

旋转卡尺是几何学的精密仪器，流畅、同步且精确，它通过轻柔地绕其边缘旋转来测量世界。

# 第 72 节 最近点对与线段算法
### 711 最近点对（分治法）

最近点对（分治法）算法能比暴力方法更快地找出集合中距离最近的两个点。它巧妙地结合了排序、递归和几何洞察力，实现了 O(n log n) 的时间复杂度。

可以把它想象成一步一步地放大观察点对：分割平面，分别解决两边的问题，然后只检查跨边界点对可能隐藏的狭窄条带区域。

#### 我们要解决什么问题？

给定平面上的 n 个点，找出具有最小欧几里得距离的点对 (p, q)：

$$
d(p, q) = \sqrt{(p_x - q_x)^2 + (p_y - q_y)^2}
$$

一个朴素的方法是检查所有点对（O(n²)），但分治法通过将问题一分为二，并且只合并靠近边界的候选点对，从而减少了工作量。

#### 它是如何工作的（通俗解释）？

1.  将所有点按 x 坐标排序。
2.  将点分成两半：左半部分和右半部分。
3.  递归地找出每一半中的最近点对 → 得到距离 $d_L$ 和 $d_R$。
4.  令 $d = \min(d_L, d_R)$。
5.  合并步骤：
    *   收集距离分割线（一条垂直线）在 $d$ 范围内的点（一个垂直条带）。
    *   将这些条带中的点按 y 坐标排序。
    *   对于每个点，只检查按 y 顺序排列的下几个邻居（最多 7 个）。
6.  在这些检查中找到的最小距离就是答案。

这种"只检查附近少数几个点"的限制，正是该算法保持 $O(n \log n)$ 复杂度的关键。

#### 示例演练

点集：
A(0,0), B(3,4), C(1,1), D(4,5), E(2,2)

1.  按 x 排序 → [A(0,0), C(1,1), E(2,2), B(3,4), D(4,5)]
2.  分割为左半部分 [A, C, E] 和右半部分 [B, D]。
3.  左半部分递归 → 最近点对 = A–C = $\sqrt{2}$
    右半部分递归 → 最近点对 = B–D = $\sqrt{2}$
    所以 $d = \min(\sqrt{2}, \sqrt{2}) = \sqrt{2}$。
4.  分割线附近（$x \approx 2$）的条带 → E(2,2), B(3,4), D(4,5)
    检查点对：
    *   E–B = $\sqrt{5}$
    *   E–D = $\sqrt{10}$
    没有找到更小的距离。

结果：最近点对 = (A, C)，距离 = $\sqrt{2}$。

#### 精简代码（简易版本）

C

```c
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

typedef struct { double x, y; } Point;

int cmpX(const void* a, const void* b) {
    Point *p = (Point*)a, *q = (Point*)b;
    return (p->x > q->x) - (p->x < q->x);
}

int cmpY(const void* a, const void* b) {
    Point *p = (Point*)a, *q = (Point*)b;
    return (p->y > q->y) - (p->y < q->y);
}

double dist(Point a, Point b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}

double brute(Point pts[], int n) {
    double min = DBL_MAX;
    for (int i=0; i<n; i++)
        for (int j=i+1; j<n; j++)
            if (dist(pts[i], pts[j]) < min)
                min = dist(pts[i], pts[j]);
    return min;
}

double stripClosest(Point strip[], int size, double d) {
    double min = d;
    qsort(strip, size, sizeof(Point), cmpY);
    for (int i=0; i<size; i++)
        for (int j=i+1; j<size && (strip[j].y - strip[i].y) < min; j++)
            if (dist(strip[i], strip[j]) < min)
                min = dist(strip[i], strip[j]);
    return min;
}

double closestRec(Point pts[], int n) {
    if (n <= 3) return brute(pts, n);
    int mid = n/2;
    Point midPoint = pts[mid];

    double dl = closestRec(pts, mid);
    double dr = closestRec(pts+mid, n-mid);
    double d = dl < dr ? dl : dr;

    Point strip[1000];
    int j=0;
    for (int i=0; i<n; i++)
        if (fabs(pts[i].x - midPoint.x) < d)
            strip[j++] = pts[i];
    return fmin(d, stripClosest(strip, j, d));
}

double closestPair(Point pts[], int n) {
    qsort(pts, n, sizeof(Point), cmpX);
    return closestRec(pts, n);
}

int main() {
    Point pts[] = {{0,0},{3,4},{1,1},{4,5},{2,2}};
    int n = sizeof(pts)/sizeof(pts[0]);
    printf("Closest distance = %.3f\n", closestPair(pts, n));
}
```

Python

```python
from math import sqrt

def dist(a,b):
    return sqrt((a[0]-b[0])2 + (a[1]-b[1])2)

def brute(pts):
    n = len(pts)
    d = float('inf')
    for i in range(n):
        for j in range(i+1,n):
            d = min(d, dist(pts[i], pts[j]))
    return d

def strip_closest(strip, d):
    strip.sort(key=lambda p: p[1])
    m = len(strip)
    for i in range(m):
        for j in range(i+1, m):
            if (strip[j][1] - strip[i][1]) >= d:
                break
            d = min(d, dist(strip[i], strip[j]))
    return d

def closest_pair(points):
    n = len(points)
    if n <= 3:
        return brute(points)
    mid = n // 2
    midx = points[mid][0]
    d = min(closest_pair(points[:mid]), closest_pair(points[mid:]))
    strip = [p for p in points if abs(p[0]-midx) < d]
    return min(d, strip_closest(strip, d))

pts = [(0,0),(3,4),(1,1),(4,5),(2,2)]
pts.sort()
print("Closest distance:", closest_pair(pts))
```

#### 为什么它很重要

*   几何中分治法的经典示例。
*   高效而优雅，实现了从 O(n²) 到 O(n log n) 的跨越。
*   为其他平面算法（如 Delaunay 三角剖分、Voronoi 图）建立直觉。

应用场景：

*   聚类（检测邻近邻居）
*   碰撞检测（寻找最小间隔）
*   天文学 / 地理信息系统（最近的恒星、城市）
*   机器学习（最近邻初始化）

#### 亲自尝试

1.  尝试随机二维点，验证结果是否与暴力方法一致。
2.  添加共线点，确认沿线的距离。
3.  可视化分割和条带，绘制分割线和条带区域。
4.  扩展到 3D 最近点对（同时检查 z 坐标）。
5.  测量当 n 翻倍时的运行时间。

#### 测试用例

| 点集                              | 最近点对       | 距离       |
| --------------------------------- | -------------- | ---------- |
| (0,0),(1,1),(2,2)                 | (0,0)-(1,1)    | √2         |
| (0,0),(3,4),(1,1),(4,5),(2,2)     | (0,0)-(1,1)    | √2         |
| 随机点集                          | 已验证         | O(n log n) |
| 重复点                            | 距离 = 0       | 边界情况   |

#### 复杂度

*   时间：O(n log n)
*   空间：O(n)
*   暴力方法：O(n²)（用于比较）

分治法在混沌中寻找结构，通过排序、分割和合并，直到最近点对脱颖而出。
### 712 最近点对（扫描线法）

最近点对（扫描线法）算法是一种高效且优美的 O(n log n) 技术，它从左到右扫描平面，维护一个可能构成最近点对的候选点滑动窗口（或称“活动集”）。

可以把它想象成在星空中扫过一条垂直线，每当一颗星星出现时，你只检查它附近的邻居，而不是整个天空。

#### 我们要解决什么问题？

给定二维空间中的 n 个点，我们希望找到具有最小欧几里得距离的点对。

与递归分割的“分治法”不同，扫描线解决方案以增量方式逐个处理点，维护一个在 x 方向上足够接近、可能成为候选点的活动集。

这种方法直观、迭代，并且特别适合使用平衡搜索树或有序集合来实现。

#### 它是如何工作的（通俗解释）？

1.  按 x 坐标对点进行排序。
2.  初始化一个空的活动集（按 y 排序）。
3.  从左到右扫描：
    *   对于每个点 p，
        *   移除那些与 p 的 x 距离超过当前最佳距离 d 的点（它们太靠左了）。
        *   在剩余的活动集中，只检查那些 y 距离 < d 的点。
        *   如果找到更近的点对，则更新 d。
    *   将 p 插入活动集。
4.  继续处理，直到所有点都处理完毕。

由于每个点只进入和离开活动集一次，并且每个点只与常数个附近的点进行比较，所以总时间为 O(n log n)。

#### 示例演练

点：
A(0,0), B(3,4), C(1,1), D(2,2), E(4,5)

1.  按 x 排序 → [A, C, D, B, E]
2.  从 A 开始 → 活动集 = {A}, d = ∞
3.  添加 C：距离(A,C) = √2 → d = √2
4.  添加 D：检查邻居 (A,C) → C–D = √2（无改进）
5.  添加 B：移除 A (B.x - A.x > √2)，检查 C–B (距离 > √2)，D–B (距离 = √5)
6.  添加 E：移除 C (E.x - C.x > √2)，检查 D–E, B–E
    最近点对：(A, C)，距离 √2

#### 精简代码（简易版）

Python

```python
from math import sqrt
import bisect

def dist(a, b):
    return sqrt((a[0]-b[0])2 + (a[1]-b[1])2)

def closest_pair_sweep(points):
    points.sort(key=lambda p: p[0])  # 按 x 排序
    active = []
    best = float('inf')
    best_pair = None

    for p in points:
        # 移除 x 方向上过远的点
        while active and p[0] - active[0][0] > best:
            active.pop(0)

        # 按 y 范围过滤活动集中的点
        candidates = [q for q in active if abs(q[1] - p[1]) < best]

        # 检查每个候选点
        for q in candidates:
            d = dist(p, q)
            if d < best:
                best = d
                best_pair = (p, q)

        # 插入当前点（保持按 y 排序）
        bisect.insort(active, p, key=lambda r: r[1] if hasattr(bisect, "insort") else 0)

    return best_pair, best

# 示例
pts = [(0,0),(3,4),(1,1),(2,2),(4,5)]
pair, d = closest_pair_sweep(pts)
print("最近点对:", pair, "距离:", round(d,3))
```

*(注意：`bisect` 不能直接按 key 排序；在实际代码中请使用 `sortedcontainers` 或平衡树。)*

C (伪代码)
在 C 语言中，使用以下方式实现：

*   使用 `qsort` 按 x 排序
*   使用平衡二叉搜索树（按 y）作为活动集
*   窗口更新和邻居检查
    （实际实现使用 AVL 树或有序数组）

#### 为什么它很重要

*   增量式和在线处理：一次处理一个点。
*   概念简单，是一个几何滑动窗口。
*   是分治法的实用替代方案。

应用：

*   流式几何处理
*   实时碰撞检测
*   最近邻估计
*   计算几何可视化

#### 亲自尝试

1.  手动逐步执行已排序的点。
2.  跟踪活动集如何收缩和增长。
3.  添加内部点，观察需要比较多少个点。
4.  尝试 1000 个随机点，验证快速运行时间。
5.  与分治法进行比较，结果相同，但路径不同。

#### 测试用例

| 点集                  | 最近点对              | 距离       | 备注           |
| --------------------- | --------------------- | ---------- | -------------- |
| (0,0),(1,1),(2,2)     | (0,0)-(1,1)           | √2         | 简单直线       |
| 随机散点              | 正确的点对            | O(n log n) | 高效           |
| 聚集在原点附近        | 找到最近邻居          | 有效       |                |
| 重复点                | 距离 0                | 边界情况   |                |

#### 复杂度

*   时间：O(n log n)
*   空间：O(n)
*   活动集大小：O(n)（通常窗口很小）

扫描线法是几何学中稳定跳动的脉搏，从左到右移动，修剪过去，只专注于附近的现在，以找到最近的点对。
### 713 暴力最近点对

暴力最近点对算法是寻找集合中最近两点最简单的方法：检查每一对可能的点，并选择距离最小的那一对。

这是几何问题中“全部尝试一遍”的等价方法，是理解更智能算法如何在其基础上进行改进的完美第一步。

#### 我们要解决什么问题？

给定平面上的 n 个点，我们希望找到具有最小欧几里得距离的点对 (p, q)：

$$
d(p, q) = \sqrt{(p_x - q_x)^2 + (p_y - q_y)^2}
$$

暴力方法意味着：

*   比较每一对点一次。
*   追踪目前找到的最小距离。
*   返回具有该距离的点对。

它很慢，时间复杂度为 O(n²)，但思路直接，在简单性上无可匹敌。

#### 它是如何工作的？（通俗解释）

1.  初始化最佳距离 ( d = \infty )。
2.  循环遍历所有点 ( i = 1..n-1 )：
    *   对于每个 ( j = i+1..n )，计算距离 ( d(i, j) )。
    *   如果 ( d(i, j) < d )，更新 ( d ) 并存储该点对。
3.  返回最小的 ( d ) 及其点对。

因为每一对点都被精确检查一次，所以很容易推理，非常适合小型数据集或测试。

#### 示例演练

点：
A(0,0), B(3,4), C(1,1), D(2,2)

点对及其距离：

*   A–B = 5
*   A–C = √2
*   A–D = √8
*   B–C = √13
*   B–D = √5
*   C–D = √2

最小距离 = √2（点对 A–C 和 C–D）
返回第一个或所有最小距离的点对。

#### 微型代码（简易版）

C

```c
#include <stdio.h>
#include <math.h>
#include <float.h>

typedef struct { double x, y; } Point;

double dist(Point a, Point b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}

void closestPairBrute(Point pts[], int n) {
    double best = DBL_MAX;
    Point p1, p2;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double d = dist(pts[i], pts[j]);
            if (d < best) {
                best = d;
                p1 = pts[i];
                p2 = pts[j];
            }
        }
    }
    printf("Closest Pair: (%.1f, %.1f) and (%.1f, %.1f)\n", p1.x, p1.y, p2.x, p2.y);
    printf("Distance: %.3f\n", best);
}

int main() {
    Point pts[] = {{0,0},{3,4},{1,1},{2,2}};
    int n = sizeof(pts)/sizeof(pts[0]);
    closestPairBrute(pts, n);
}
```

Python

```python
from math import sqrt

def dist(a, b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def closest_pair_brute(points):
    best = float('inf')
    pair = None
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            d = dist(points[i], points[j])
            if d < best:
                best = d
                pair = (points[i], points[j])
    return pair, best

pts = [(0,0),(3,4),(1,1),(2,2)]
pair, d = closest_pair_brute(pts)
print("Closest pair:", pair, "distance:", round(d,3))
```

#### 为何重要

*   是理解分治法和扫描线改进的基础。
*   对于小的 n → 最简单、最可靠的方法。
*   用于测试优化后的算法。
*   几何迭代和距离函数的温和入门。

应用：

*   几何问题的教学基线
*   计算几何工具包中的验证
*   调试优化后的实现
*   非常小的点集（n < 100）

#### 动手尝试

1.  添加 5–10 个随机点，手动列出所有点对距离。
2.  对照优化版本检查正确性。
3.  扩展到 3D，只需添加 z 项。
4.  修改为曼哈顿距离。
5.  打印所有距离相等的最小点对（平局情况）。

#### 测试用例

| 点集                      | 最近点对       | 距离   |
| ------------------------- | -------------- | ------ |
| (0,0),(1,1),(2,2)         | (0,0)-(1,1)    | √2     |
| (0,0),(3,4),(1,1),(2,2)   | (0,0)-(1,1)    | √2     |
| 随机点集                  | 已验证         | 与优化版本匹配 |
| 重复点                    | 距离 = 0       | 边界情况 |

#### 复杂度

*   时间复杂度：O(n²)
*   空间复杂度：O(1)

暴力法是处理几何问题的第一直觉，简单、确定但缓慢，却是所有后续巧妙算法坚实的基石。
### 714 Bentley–Ottmann

Bentley–Ottmann 算法是一种经典的扫描线方法，它能高效地找出平面上给定线段集合中的所有交点。其运行时间为

$$
O\big((n + k)\log n\big)
$$

其中 $n$ 是线段数量，$k$ 是交点的数量。

其核心思想是让一条垂直的扫描线扫过整个平面，维护一个按 $y$ 坐标排序的、与扫描线相交的线段活动集合，并使用一个事件队列来处理仅有的三种类型的点：线段起点、线段终点以及已发现的交点。

#### 我们要解决什么问题？

给定 $n$ 条线段，我们希望计算它们之间的所有交点。

一种朴素的方法是检查所有线段对：

$$
\binom{n}{2} = \frac{n(n-1)}{2}
$$

这导致 $O(n^2)$ 的时间复杂度。Bentley–Ottmann 算法通过只测试扫描线活动集合中相邻的线段，将复杂度降低到 $O\big((n + k)\log n\big)$。

#### 它是如何工作的（通俗解释）？

在扫描过程中，我们维护两个数据结构：

1.  **事件队列 (EQ)**：所有按 x 排序的事件，包括线段起点、线段终点和已发现的交点。
2.  **活动集合 (AS)**：当前与扫描线相交的所有线段，按 y 坐标排序。

扫描过程从左向右进行：

| 步骤 | 描述                                                                                                                              |
| ---- | -------------------------------------------------------------------------------------------------------------------------------- |
| 1    | 用所有线段的端点初始化事件队列。                                                                                               |
| 2    | 从左到右扫描处理所有事件。                                                                                                    |
| 3    | 对于每个事件 $p$：                                                                                                                |
| a.   | 如果 $p$ 是线段起点，将该线段插入 AS，并测试其与直接相邻线段是否相交。                                                       |
| b.   | 如果 $p$ 是线段终点，将该线段从 AS 中移除。                                                                                  |
| c.   | 如果 $p$ 是一个交点，记录它，在 AS 中交换两条相交线段的位置，并检查它们的新邻居。                                           |
| 4    | 继续直到事件队列为空。                                                                                                           |

使用平衡搜索树，事件队列或活动集合上的每个操作耗时 $O(\log n)$。

#### 示例演练

线段：

*   $S_1: (0,0)\text{–}(4,4)$
*   $S_2: (0,4)\text{–}(4,0)$
*   $S_3: (1,3)\text{–}(3,3)$

事件队列（按 $x$ 排序）：
$(0,0), (0,4), (1,3), (2,2), (3,3), (4,0), (4,4)$

处理过程：

1.  在 $x=0$ 处：插入 $S_1, S_2$。它们在 $(2,2)$ 处相交 → 安排交点事件。
2.  在 $x=1$ 处：插入 $S_3$；检查 $S_1, S_2, S_3$ 的局部相交情况。
3.  在 $x=2$ 处：处理 $(2,2)$，交换 $S_1, S_2$，重新检查邻居。
4.  继续；所有交点均已发现。

输出：交点 $(2,2)$。

#### 微型代码（概念性 Python）

算法的简化示意（实际实现需要优先队列和平衡树）：

```python
from collections import namedtuple
Event = namedtuple("Event", ["x", "y", "type", "segment"])

def orientation(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def intersects(s1, s2):
    a, b = s1
    c, d = s2
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)

def bentley_ottmann(segments):
    events = []
    for s in segments:
        (x1, y1), (x2, y2) = s
        if x1 > x2:
            s = ((x2, y2), (x1, y1))
        events.append((x1, y1, 'start', s))
        events.append((x2, y2, 'end', s))
    events.sort()

    active = []
    intersections = []

    for x, y, etype, s in events:
        if etype == 'start':
            active.append(s)
            for other in active:
                if other != s and intersects(s, other):
                    intersections.append((x, y))
        elif etype == 'end':
            active.remove(s)

    return intersections

segments = [((0,0),(4,4)), ((0,4),(4,0)), ((1,3),(3,3))]
print("Intersections:", bentley_ottmann(segments))
```

#### 为什么它很重要？

*   **高效**：$O((n + k)\log n)$ 对比 $O(n^2)$
*   **优雅**：只检查相邻线段
*   **通用**：事件驱动几何的基础

应用：

*   CAD 系统（曲线交叉）
*   GIS（地图叠加、道路交叉口）
*   图形学（线段碰撞检测）
*   机器人学（运动规划、可见性图）

#### 一个温和的证明（为什么它有效）

在任何扫描位置，活动集合中的线段都按其 $y$ 坐标排序。当两条线段相交时，它们的顺序必须在交点处交换。

因此：

*   每个交点恰好会在扫描线到达其 $x$ 坐标时被发现一次。
*   只有相邻的线段才可能交换；因此只需要进行局部检查。
*   每个事件（插入、删除或交点）在平衡树操作中需要 $O(\log n)$ 时间。

总成本：

$$
O\big((n + k)\log n\big)
$$

其中 $n$ 贡献端点，$k$ 贡献已发现的交点。

#### 亲自尝试

1.  画出几条在不同点相交的线段。
2.  将所有端点按 $x$ 坐标排序。
3.  模拟扫描过程：维护一个按 $y$ 排序的活动集合。
4.  在每个事件处，只检查相邻的线段。
5.  验证每个交点出现一次且仅出现一次。
6.  与暴力 $O(n^2)$ 方法进行比较。

#### 测试用例

| 线段                     | 交点数量 | 备注                     |
| ------------------------ | -------- | ------------------------ |
| 正方形的两条对角线       | 1        | 交点位于中心             |
| 五角星                   | 10       | 所有线段对都相交         |
| 平行线                   | 0        | 没有交点                 |
| 随机交叉                 | 已验证   | 与预期输出匹配           |

#### 复杂度

$$
\text{时间: } O\big((n + k)\log n\big), \quad
\text{空间: } O(n)
$$

Bentley–Ottmann 算法是几何精确性的典范，它扫过平面，维持顺序，并恰好一次地揭示每一个交叉点。
### 715 线段相交测试

线段相交测试是用于检查平面上两条线段是否相交的基本几何例程。它是许多大型算法的构建模块，从多边形裁剪到 Bentley–Ottmann 等扫描线方法。

其核心是一个简单的原理：两条线段相交当且仅当它们相互跨越，这通过使用叉积的方向测试来确定。

#### 我们要解决什么问题？

给定两条线段：

* $S_1 = (p_1, q_1)$
* $S_2 = (p_2, q_2)$

我们想要确定它们是否相交，无论是在两条线段内部的一个点相交，还是在端点处相交。

从数学上讲，$S_1$ 和 $S_2$ 相交的条件是：

1. 两条线段相互交叉，或者
2. 它们共线并且重叠。

#### 它是如何工作的（通俗解释）？

我们使用方向测试来检查点的相对位置。

对于任意三个点 $a, b, c$，定义：

$$
\text{orient}(a, b, c) = (b_x - a_x)(c_y - a_y) - (b_y - a_y)(c_x - a_x)
$$

* $\text{orient}(a, b, c) > 0$：$c$ 在直线 $ab$ 的左侧
* $\text{orient}(a, b, c) < 0$：$c$ 在直线 $ab$ 的右侧
* $\text{orient}(a, b, c) = 0$：点共线

对于线段 $(p_1, q_1)$ 和 $(p_2, q_2)$：

计算方向：

* $o_1 = \text{orient}(p_1, q_1, p_2)$
* $o_2 = \text{orient}(p_1, q_1, q_2)$
* $o_3 = \text{orient}(p_2, q_2, p_1)$
* $o_4 = \text{orient}(p_2, q_2, q_1)$

两条线段严格相交的条件是：

$$
(o_1 \neq o_2) \quad \text{并且} \quad (o_3 \neq o_4)
$$

如果任何 $o_i = 0$，则检查对应的点是否位于线段上（共线重叠）。

#### 示例演练

线段：

* $S_1: (0,0)\text{–}(4,4)$
* $S_2: (0,4)\text{–}(4,0)$

计算方向：

| 配对                                   | 值    | 含义       |
| -------------------------------------- | ----- | ---------- |
| $o_1 = \text{orient}(0,0),(4,4),(0,4)$ | $> 0$ | 左转       |
| $o_2 = \text{orient}(0,0),(4,4),(4,0)$ | $< 0$ | 右转       |
| $o_3 = \text{orient}(0,4),(4,0),(0,0)$ | $< 0$ | 右转       |
| $o_4 = \text{orient}(0,4),(4,0),(4,4)$ | $> 0$ | 左转       |

由于 $o_1 \neq o_2$ 且 $o_3 \neq o_4$，线段在 $(2,2)$ 处相交。

#### 微型代码（C）

```c
#include <stdio.h>

typedef struct { double x, y; } Point;

double orient(Point a, Point b, Point c) {
    return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x);
}

int onSegment(Point a, Point b, Point c) {
    return b.x <= fmax(a.x, c.x) && b.x >= fmin(a.x, c.x) &&
           b.y <= fmax(a.y, c.y) && b.y >= fmin(a.y, c.y);
}

int intersect(Point p1, Point q1, Point p2, Point q2) {
    double o1 = orient(p1, q1, p2);
    double o2 = orient(p1, q1, q2);
    double o3 = orient(p2, q2, p1);
    double o4 = orient(p2, q2, q1);

    if (o1*o2 < 0 && o3*o4 < 0) return 1;

    if (o1 == 0 && onSegment(p1, p2, q1)) return 1;
    if (o2 == 0 && onSegment(p1, q2, q1)) return 1;
    if (o3 == 0 && onSegment(p2, p1, q2)) return 1;
    if (o4 == 0 && onSegment(p2, q1, q2)) return 1;

    return 0;
}

int main() {
    Point a={0,0}, b={4,4}, c={0,4}, d={4,0};
    printf("Intersect? %s\n", intersect(a,b,c,d) ? "Yes" : "No");
}
```

#### 微型代码（Python）

```python
def orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def on_segment(a, b, c):
    return (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
            min(a[1], c[1]) <= b[1] <= max(a[1], c[1]))

def intersect(p1, q1, p2, q2):
    o1 = orient(p1, q1, p2)
    o2 = orient(p1, q1, q2)
    o3 = orient(p2, q2, p1)
    o4 = orient(p2, q2, q1)

    if o1*o2 < 0 and o3*o4 < 0:
        return True
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True
    return False

print(intersect((0,0),(4,4),(0,4),(4,0)))
```

#### 为什么它很重要

* 许多几何算法的核心原语
* 支持多边形相交、裁剪和三角剖分
* 用于计算几何、地理信息系统、计算机辅助设计和物理引擎

应用：

* 检测碰撞或交叉
* 构建可见性图
* 检查多边形中的自相交
* 扫描线和裁剪算法的基础

#### 一个温和的证明（为什么它有效）

要使线段 $AB$ 和 $CD$ 相交，它们必须相互跨越。也就是说，$C$ 和 $D$ 必须位于 $AB$ 的不同侧，并且 $A$ 和 $B$ 必须位于 $CD$ 的不同侧。

方向函数 $\text{orient}(a,b,c)$ 给出了三角形 $(a,b,c)$ 的有向面积。如果 $\text{orient}(A,B,C)$ 和 $\text{orient}(A,B,D)$ 的符号不同，则 $C$ 和 $D$ 在 $AB$ 的异侧。

因此，如果：

$$
\text{sign}(\text{orient}(A,B,C)) \neq \text{sign}(\text{orient}(A,B,D))
$$

并且

$$
\text{sign}(\text{orient}(C,D,A)) \neq \text{sign}(\text{orient}(C,D,B))
$$

那么两条线段必然相交。共线情况（$\text{orient}=0$）通过检查重叠来单独处理。

#### 自己动手试试

1.  画两条相交的线段，验证方向的符号。
2.  尝试平行且不相交的线段，确认测试返回 false。
3.  测试共线重叠的线段。
4.  扩展到 3D（使用向量叉积）。
5.  结合边界框检查以进行更快的过滤。

#### 测试用例

| 线段                            | 结果     | 备注               |
| ------------------------------- | -------- | ------------------ |
| $(0,0)-(4,4)$ 和 $(0,4)-(4,0)$ | 相交     | 在 $(2,2)$ 处交叉  |
| $(0,0)-(4,0)$ 和 $(5,0)-(6,0)$ | 不相交   | 不共线的共线线段   |
| $(0,0)-(4,0)$ 和 $(2,0)-(6,0)$ | 相交     | 重叠               |
| $(0,0)-(4,0)$ 和 $(0,1)-(4,1)$ | 不相交   | 平行               |

#### 复杂度

$$
\text{时间: } O(1), \quad \text{空间: } O(1)
$$

线段相交测试是几何学的原子操作，是一个基于叉积和方向逻辑构建的单一、精确的检查。
### 716 线段扫描算法

线段扫描算法是一种通用的事件驱动框架，用于高效地检测多条线段之间的交点、重叠或覆盖关系。它通过一个垂直移动的扫描线和一个平衡树来追踪活动线段，按排序顺序处理事件（线段起点、终点和交点）。

这是 Bentley–Ottmann 算法、矩形并集面积计算以及重叠计数等算法背后的核心概念。

#### 我们要解决什么问题？

给定平面上的一组 $n$ 条线段（或区间），我们希望高效地：

* 检测它们之间的交点
* 计算重叠或覆盖情况
* 计算并集或交集区域

一种朴素的方法是两两比较所有线段（$O(n^2)$），但扫描线算法通过仅维护当前与扫描线相交的局部邻域线段，避免了不必要的检查。

#### 它是如何工作的（通俗解释）？

我们概念上从左到右滑动一条垂直线穿过平面，按 x 坐标排序的顺序处理关键事件：

| 步骤                                                                                       | 描述                                                                          |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| 1                                                                                          | 事件队列（EQ）：所有线段端点和已知的交点，按 $x$ 排序。  |
| 2                                                                                          | 活动集（AS）：当前与扫描线相交的线段，按 $y$ 排序。 |
| 3                                                                                          | 从左到右处理每个事件 $e$：                                           |
|   a. 起点事件：将线段插入 AS；检查与直接相邻线段的交点。 |                                                                                      |
|   b. 终点事件：从 AS 中移除线段。                                                |                                                                                      |
|   c. 交点事件：报告交点；交换线段顺序；检查新的相邻线段。 |                                                                                      |
| 4                                                                                          | 继续处理直到 EQ 为空。                                                          |

在每一步，活动集仅包含当前在扫描线下方“存活”的线段。只有 AS 中的相邻线段对才可能相交。

#### 示例演练

线段：

* $S_1: (0,0)\text{–}(4,4)$
* $S_2: (0,4)\text{–}(4,0)$
* $S_3: (1,3)\text{–}(3,3)$

事件（按 x 排序）：
$(0,0), (0,4), (1,3), (2,2), (3,3), (4,0), (4,4)$

步骤：

1. 在 $x=0$ 处：插入 $S_1, S_2$ → 检查交点 $(2,2)$ → 将事件加入队列。
2. 在 $x=1$ 处：插入 $S_3$ → 检查与相邻线段 $S_1$、$S_2$ 的关系。
3. 在 $x=2$ 处：处理交点事件 $(2,2)$ → 交换 $S_1$、$S_2$ 的顺序。
4. 继续处理直到所有线段处理完毕。

 输出：交点 $(2,2)$。

#### 微型代码（Python 概念）

线段扫描的概念框架：

```python
from bisect import insort
from collections import namedtuple

Event = namedtuple("Event", ["x", "y", "type", "segment"])

def orientation(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def intersect(s1, s2):
    a, b = s1
    c, d = s2
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    return (o1*o2 < 0 and o3*o4 < 0)

def sweep_segments(segments):
    events = []
    for s in segments:
        (x1, y1), (x2, y2) = s
        if x1 > x2: s = ((x2, y2), (x1, y1))
        events += [(x1, y1, 'start', s), (x2, y2, 'end', s)]
    events.sort()

    active = []
    intersections = []

    for x, y, t, s in events:
        if t == 'start':
            insort(active, s)
            # 检查相邻线段
            for other in active:
                if other != s and intersect(s, other):
                    intersections.append((x, y))
        elif t == 'end':
            active.remove(s)
    return intersections

segments = [((0,0),(4,4)), ((0,4),(4,0)), ((1,3),(3,3))]
print("交点:", sweep_segments(segments))
```

#### 为什么它重要？

* 为许多几何问题提供了统一的方法
* 构成了 Bentley–Ottmann 算法、矩形并集和扫描圆算法的基础
* 高效：进行局部检查而非全局比较

应用：

* 检测碰撞或交点
* 计算形状的并集面积
* 事件驱动模拟
* 可见性图和运动规划

#### 一个温和的证明（为什么它有效）

在每个 $x$ 坐标处，活动集代表了扫描线下方线段的当前“切片”。

关键不变量：

1. 活动集按 y 坐标排序，反映了扫描线处的垂直顺序。
2. 两条线段只有在此排序中相邻时才可能相交。
3. 每个交点都对应一次顺序交换，因此每个交点只被发现一次。

每个事件（插入、移除、交换）在使用平衡树时耗时 $O(\log n)$。
每个交点增加一个事件，因此总复杂度为：

$$
O\big((n + k)\log n\big)
$$

其中 $k$ 是交点的数量。

#### 自己动手试试

1. 画出几条线段，标记起点和终点事件。
2. 按 $x$ 对事件排序，逐步执行扫描。
3. 在每一步维护垂直顺序。
4. 添加一条水平线段，观察它与多条活动线段的重叠情况。
5. 计算交点并验证正确性。

#### 测试用例

| 线段                         | 交点 | 备注                   |
| -------------------------------- | ------------- | ----------------------- |
| $(0,0)$–$(4,4)$, $(0,4)$–$(4,0)$ | 1             | 在 $(2,2)$ 处相交        |
| 平行且不重叠         | 0             | 无交点         |
| 水平重叠              | 多个      | 共享区域           |
| 随机交叉                 | 已验证      | 与预期输出匹配 |

#### 复杂度

$$
\text{时间: } O\big((n + k)\log n\big), \quad
\text{空间: } O(n)
$$

扫描线框架是几何调度器，稳定地扫过平面，追踪活动形状，并在事件发生时精确捕获每个事件。
### 717 基于方向的线段相交判定（CCW 测试）

基于方向的相交判定方法，通常称为 CCW 测试（逆时针测试），是计算几何中最简单、最优雅的工具之一。它通过分析点的方向（即三个点是顺时针还是逆时针转向）来判断两条线段是否相交。

这是一种简洁、纯粹的代数方法，用于几何推理，而无需显式求解直线相交的方程。

#### 我们要解决什么问题？

给定两条线段：

* $S_1 = (p_1, q_1)$
* $S_2 = (p_2, q_2)$

我们想要判断它们是否相交，无论是在两条线段内部的某个点，还是在端点处。

CCW 测试完全使用行列式（叉积）进行计算，避免了浮点数除法，并能处理共线等边界情况。

#### 它是如何工作的（通俗解释）？

对于三个点 $a, b, c$，定义方向函数：

$$
\text{orient}(a, b, c) = (b_x - a_x)(c_y - a_y) - (b_y - a_y)(c_x - a_x)
$$

* $\text{orient}(a,b,c) > 0$ → 逆时针转向 (CCW)
* $\text{orient}(a,b,c) < 0$ → 顺时针转向 (CW)
* $\text{orient}(a,b,c) = 0$ → 共线

对于两条线段 $(p_1, q_1)$ 和 $(p_2, q_2)$，我们计算：

$$
\begin{aligned}
o_1 &= \text{orient}(p_1, q_1, p_2) \\
o_2 &= \text{orient}(p_1, q_1, q_2) \\
o_3 &= \text{orient}(p_2, q_2, p_1) \\
o_4 &= \text{orient}(p_2, q_2, q_1)
\end{aligned}
$$

两条线段相交当且仅当：

$$
(o_1 \neq o_2) \quad \text{且} \quad (o_3 \neq o_4)
$$

这确保了每条线段都跨越了另一条线段。

如果任何 $o_i = 0$，我们使用边界框测试来检查共线重叠。

#### 示例演练

线段：

* $S_1: (0,0)\text{–}(4,4)$
* $S_2: (0,4)\text{–}(4,0)$

计算方向：

| 表达式                              | 值    | 含义   |
| ----------------------------------- | ----- | ------ |
| $o_1 = \text{orient}(0,0,4,4,0,4)$ | $> 0$ | CCW    |
| $o_2 = \text{orient}(0,0,4,4,4,0)$ | $< 0$ | CW     |
| $o_3 = \text{orient}(0,4,4,0,0,0)$ | $< 0$ | CW     |
| $o_4 = \text{orient}(0,4,4,0,4,4)$ | $> 0$ | CCW    |

因为 $o_1 \neq o_2$ 且 $o_3 \neq o_4$，所以线段在 $(2,2)$ 处相交。

#### 精简代码 (Python)

```python
def orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def on_segment(a, b, c):
    return (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
            min(a[1], c[1]) <= b[1] <= max(a[1], c[1]))

def intersect(p1, q1, p2, q2):
    o1 = orient(p1, q1, p2)
    o2 = orient(p1, q1, q2)
    o3 = orient(p2, q2, p1)
    o4 = orient(p2, q2, q1)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True  # 一般情况

    # 特殊情况：共线重叠
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False

# 示例
print(intersect((0,0),(4,4),(0,4),(4,0)))  # True
```

#### 精简代码 (C)

```c
#include <stdio.h>

typedef struct { double x, y; } Point;

double orient(Point a, Point b, Point c) {
    return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x);
}

int onSegment(Point a, Point b, Point c) {
    return b.x <= fmax(a.x, c.x) && b.x >= fmin(a.x, c.x) &&
           b.y <= fmax(a.y, c.y) && b.y >= fmin(a.y, c.y);
}

int intersect(Point p1, Point q1, Point p2, Point q2) {
    double o1 = orient(p1, q1, p2);
    double o2 = orient(p1, q1, q2);
    double o3 = orient(p2, q2, p1);
    double o4 = orient(p2, q2, q1);

    if (o1*o2 < 0 && o3*o4 < 0) return 1;

    if (o1 == 0 && onSegment(p1, p2, q1)) return 1;
    if (o2 == 0 && onSegment(p1, q2, q1)) return 1;
    if (o3 == 0 && onSegment(p2, p1, q2)) return 1;
    if (o4 == 0 && onSegment(p2, q1, q2)) return 1;

    return 0;
}

int main() {
    Point a={0,0}, b={4,4}, c={0,4}, d={4,0};
    printf("Intersect? %s\n", intersect(a,b,c,d) ? "Yes" : "No");
}
```

#### 为什么它很重要

* 几何和计算图形学中的基本原语
* 构成多边形相交、裁剪和三角剖分的核心
* 数值稳定，避免了除法或浮点数斜率
* 用于碰撞检测、路径规划和几何内核

应用：

* 检测多边形网格中的相交
* 检查导航系统中的路径交叉
* 实现裁剪算法（例如 Weiler–Atherton）

#### 温和的证明（为什么它有效）

线段 $AB$ 和 $CD$ 相交，当且仅当每一对端点都跨越了另一条线段。
方向函数 $\text{orient}(A,B,C)$ 给出了三角形 $(A,B,C)$ 的有向面积。

* 如果 $\text{orient}(A,B,C)$ 和 $\text{orient}(A,B,D)$ 符号相反，那么 $C$ 和 $D$ 在 $AB$ 的不同侧。
* 类似地，如果 $\text{orient}(C,D,A)$ 和 $\text{orient}(C,D,B)$ 符号相反，那么 $A$ 和 $B$ 在 $CD$ 的不同侧。

因此，如果：

$$
\text{sign}(\text{orient}(A,B,C)) \neq \text{sign}(\text{orient}(A,B,D))
$$

并且

$$
\text{sign}(\text{orient}(C,D,A)) \neq \text{sign}(\text{orient}(C,D,B))
$$

那么两条线段必然相交。
如果任何方向值为 $0$，我们只需检查共线的点是否位于线段边界内。

#### 自己动手试试

1.  画出两条相交的线段；在每个顶点处标出方向符号。
2.  尝试不相交和平行的情况，确认方向测试结果不同。
3.  检查共线重叠的线段。
4.  实现一个能计算多条线段之间相交次数的版本。
5.  与暴力坐标相交法进行比较。

#### 测试用例

| 线段                                 | 结果     | 备注               |
| ------------------------------------ | -------- | ------------------ |
| $(0,0)$–$(4,4)$ 和 $(0,4)$–$(4,0)$   | 相交     | 在 $(2,2)$ 处交叉  |
| $(0,0)$–$(4,0)$ 和 $(5,0)$–$(6,0)$   | 不相交   | 不相连的共线线段   |
| $(0,0)$–$(4,0)$ 和 $(2,0)$–$(6,0)$   | 相交     | 重叠               |
| $(0,0)$–$(4,0)$ 和 $(0,1)$–$(4,1)$   | 不相交   | 平行线             |

#### 复杂度

$$
\text{时间: } O(1), \quad \text{空间: } O(1)
$$

CCW 测试将相交检测提炼为单一的代数测试，这是基于方向符号构建的几何推理的基础。
### 718 圆相交问题

圆相交问题询问两个圆是否相交，如果相交，则计算它们的交点。
这是一个将代数几何与空间推理相结合的经典例子，常用于碰撞检测、维恩图和范围查询。

根据相对位置，两个圆可以有 0、1、2 或无限（重合）个交点。

#### 我们要解决什么问题？

给定两个圆：

* $C_1$：圆心 $(x_1, y_1)$，半径 $r_1$
* $C_2$：圆心 $(x_2, y_2)$，半径 $r_2$

我们想要确定：

1. 它们是否相交？
2. 如果相交，交点是什么？

#### 它是如何工作的（通俗解释）？

设圆心之间的距离为：

$$
d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
$$

现在将 $d$ 与 $r_1$ 和 $r_2$ 进行比较：

| 条件               | 含义                                       |                  |                                                      |
| ------------------ | ------------------------------------------ | ---------------- | ---------------------------------------------------- |
| $d > r_1 + r_2$    | 两圆分离（无交点）                         |                  |                                                      |
| $d = r_1 + r_2$    | 两圆外切（1 个交点）                       |                  |                                                      |
| $                  | r_1 - r_2                                  | < d < r_1 + r_2$ | 两圆相交（2 个交点）                                 |
| $d =               | r_1 - r_2                                  | $                | 两圆内切（1 个交点）                                 |
| $d <               | r_1 - r_2                                  | $                | 一个圆在另一个圆内部（无交点）                       |
| $d = 0, r_1 = r_2$ | 两圆重合（无限个交点）                     |                  |                                                      |

如果它们相交（$|r_1 - r_2| < d < r_1 + r_2$），则可以通过几何方法计算交点。

#### 交点推导

我们找到两个圆之间的相交线。

令：

$$
a = \frac{r_1^2 - r_2^2 + d^2}{2d}
$$

那么，连接两圆圆心的直线上，交点弦穿过的点 $P$ 为：

$$
P_x = x_1 + a \cdot \frac{x_2 - x_1}{d}
$$
$$
P_y = y_1 + a \cdot \frac{y_2 - y_1}{d}
$$

从 $P$ 到每个交点的垂直高度为：

$$
h = \sqrt{r_1^2 - a^2}
$$

交点为：

$$
(x_3, y_3) = \big(P_x \pm h \cdot \frac{y_2 - y_1}{d},; P_y \mp h \cdot \frac{x_2 - x_1}{d}\big)
$$

这两个点表示圆的交点。

#### 示例演练

圆：

* $C_1: (0, 0), r_1 = 5$
* $C_2: (6, 0), r_2 = 5$

计算：

* $d = 6$
* $r_1 + r_2 = 10$, $|r_1 - r_2| = 0$
  因此 $|r_1 - r_2| < d < r_1 + r_2$ → 2 个交点

然后：

$$
a = \frac{5^2 - 5^2 + 6^2}{2 \cdot 6} = 3
$$
$$
h = \sqrt{5^2 - 3^2} = 4
$$

$P = (3, 0)$ → 交点：

$$
(x_3, y_3) = (3, \pm 4)
$$

交点：$(3, 4)$ 和 $(3, -4)$

#### 微型代码（Python）

```python
from math import sqrt

def circle_intersection(x1, y1, r1, x2, y2, r2):
    dx, dy = x2 - x1, y2 - y1
    d = sqrt(dx*dx + dy*dy)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0 and r1 == r2:
        return []
    a = (r1*r1 - r2*r2 + d*d) / (2*d)
    h = sqrt(r1*r1 - a*a)
    xm = x1 + a * dx / d
    ym = y1 + a * dy / d
    rx = -dy * (h / d)
    ry =  dx * (h / d)
    return [(xm + rx, ym + ry), (xm - rx, ym - ry)]

print(circle_intersection(0, 0, 5, 6, 0, 5))
```

#### 微型代码（C）

```c
#include <stdio.h>
#include <math.h>

void circle_intersection(double x1, double y1, double r1,
                         double x2, double y2, double r2) {
    double dx = x2 - x1, dy = y2 - y1;
    double d = sqrt(dx*dx + dy*dy);

    if (d > r1 + r2 || d < fabs(r1 - r2) || (d == 0 && r1 == r2)) {
        printf("No unique intersection\n");
        return;
    }

    double a = (r1*r1 - r2*r2 + d*d) / (2*d);
    double h = sqrt(r1*r1 - a*a);
    double xm = x1 + a * dx / d;
    double ym = y1 + a * dy / d;
    double rx = -dy * (h / d);
    double ry =  dx * (h / d);

    printf("Intersection points:\n");
    printf("(%.2f, %.2f)\n", xm + rx, ym + ry);
    printf("(%.2f, %.2f)\n", xm - rx, ym - ry);
}

int main() {
    circle_intersection(0,0,5,6,0,5);
}
```

#### 为什么它很重要

* 基础的几何构建模块
* 用于碰撞检测、维恩图、圆填充、传感器范围重叠
* 支持圆裁剪、透镜面积计算和圆图构建

应用：

* 图形学（绘制圆弧、混合圆）
* 机器人学（感知重叠）
* 物理引擎（球体-球体碰撞）
* 地理信息系统（圆形缓冲区相交）

#### 一个温和的证明（为什么它有效）

两个圆的方程为：

$$
(x - x_1)^2 + (y - y_1)^2 = r_1^2
$$
$$
(x - x_2)^2 + (y - y_2)^2 = r_2^2
$$

相减消去平方项，得到连接交点的直线（根轴）的线性方程。
将此直线与一个圆的方程联立求解，得到两个对称点，通过弦的几何关系中的 $a$ 和 $h$ 推导得出。

因此，该解法是精确且对称的，并能根据 $d$ 自然地处理 0、1 或 2 个交点的情况。

#### 自己动手试试

1.  绘制两个重叠的圆，并计算 $d$、$a$、$h$。
2.  将几何草图与计算出的点进行比较。
3.  测试相切的圆（$d = r_1 + r_2$）。
4.  测试嵌套的圆（$d < |r_1 - r_2|$）。
5.  扩展到 3D 球体-球体相交（相交圆）。

#### 测试用例

| 圆 1          | 圆 2          | 结果                     |
| ------------ | ------------ | ------------------------ |
| $(0,0), r=5$ | $(6,0), r=5$ | $(3, 4)$, $(3, -4)$      |
| $(0,0), r=3$ | $(6,0), r=3$ | 相切（1 个交点）         |
| $(0,0), r=2$ | $(0,0), r=2$ | 重合（无限个交点）       |
| $(0,0), r=2$ | $(5,0), r=2$ | 无交点                   |

#### 复杂度

$$
\text{时间: } O(1), \quad \text{空间: } O(1)
$$

圆相交问题融合了代数与几何，是一种精确的构造，揭示了两个圆形世界相遇之处。
### 719 多边形求交

多边形求交问题要求我们计算两个多边形之间的重叠区域（或交集）。这是计算几何中的一项基本操作，构成了裁剪、布尔运算、地图叠加和碰撞检测的基础。

有几种标准方法：

* Sutherland–Hodgman 算法（用凸裁剪多边形裁剪主体多边形）
* Weiler–Atherton 算法（适用于带洞的一般多边形）
* Greiner–Hormann 算法（对复杂形状鲁棒）

#### 我们解决的是什么问题？

给定两个多边形 $P$ 和 $Q$，我们想要计算：

$$
R = P \cap Q
$$

其中 $R$ 是交集多边形，代表两者共有的区域。

对于凸多边形，求交是直接的；对于凹多边形或自相交多边形，则需要仔细裁剪。

#### 它是如何工作的（通俗解释）？

让我们描述经典的 Sutherland–Hodgman 方法（针对凸裁剪多边形）：

1. 初始化：令输出多边形 = 主体多边形。
2. 遍历裁剪多边形的每条边。
3. 用当前裁剪边对输出多边形进行裁剪：
   * 保留边内侧的点。
   * 计算跨越边界的边的交点。
4. 处理完所有边后，剩下的多边形就是交集。

之所以有效，是因为每条边都逐步修剪了主体多边形。

#### 核心思想

对于裁剪多边形的一条有向边 $(C_i, C_{i+1})$，一个点 $P$ 在其内侧的条件是：

$$
(C_{i+1} - C_i) \times (P - C_i) \ge 0
$$

这里使用叉积来检查相对于裁剪边的方向。

每对多边形边最多可能产生一个交点。

#### 示例演练

裁剪多边形（正方形）：
$(0,0)$, $(5,0)$, $(5,5)$, $(0,5)$

主体多边形（三角形）：
$(2,-1)$, $(6,2)$, $(2,6)$

处理各边：

1. 对底边 $(0,0)$–$(5,0)$ 裁剪 → 移除 $y=0$ 以下的点
2. 对右边 $(5,0)$–$(5,5)$ 裁剪 → 切掉 $x>5$ 的部分
3. 对顶边 $(5,5)$–$(0,5)$ 裁剪 → 修剪 $y>5$ 以上的部分
4. 对左边 $(0,5)$–$(0,0)$ 裁剪 → 修剪 $x<0$ 的部分

输出多边形：一个代表正方形内部重叠区域的五边形。

#### 微型代码（Python）

```python
def inside(p, cp1, cp2):
    return (cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

def intersection(s, e, cp1, cp2):
    dc = (cp1[0]-cp2[0], cp1[1]-cp2[1])
    dp = (s[0]-e[0], s[1]-e[1])
    n1 = cp1[0]*cp2[1] - cp1[1]*cp2[0]
    n2 = s[0]*e[1] - s[1]*e[0]
    denom = dc[0]*dp[1] - dc[1]*dp[0]
    if denom == 0: return e
    x = (n1*dp[0] - n2*dc[0]) / denom
    y = (n1*dp[1] - n2*dc[1]) / denom
    return (x, y)

def suth_hodg_clip(subject, clip):
    output = subject
    for i in range(len(clip)):
        input_list = output
        output = []
        cp1 = clip[i]
        cp2 = clip[(i+1)%len(clip)]
        for j in range(len(input_list)):
            s = input_list[j-1]
            e = input_list[j]
            if inside(e, cp1, cp2):
                if not inside(s, cp1, cp2):
                    output.append(intersection(s, e, cp1, cp2))
                output.append(e)
            elif inside(s, cp1, cp2):
                output.append(intersection(s, e, cp1, cp2))
    return output

subject = [(2,-1),(6,2),(2,6)]
clip = [(0,0),(5,0),(5,5),(0,5)]
print(suth_hodg_clip(subject, clip))
```

#### 为何重要

* 多边形操作的核心：求交、并集、差集
* 用于裁剪流水线、渲染、CAD、GIS
* 对于 $n$ 个顶点的主体多边形和 $m$ 个顶点的裁剪多边形，效率高（$O(nm)$）
* 对凸裁剪多边形稳定

应用：

* 图形学：将多边形裁剪到视口
* 地图绘制：叠加形状、划分区域
* 模拟：检测重叠区域
* 计算几何：多边形布尔运算

#### 一个温和的证明（为何有效）

每条裁剪边定义了一个半平面。凸多边形的交集等于所有界定裁剪多边形的半平面的交集。

形式化地：
$$
R = P \cap \bigcap_{i=1}^{m} H_i
$$
其中 $H_i$ 是裁剪边 $i$ 内侧的半平面。

在每一步，我们取多边形与半平面的交集，该交集本身也是凸的。因此，对所有边进行裁剪后，我们得到了精确的交集。

由于每个顶点相对于每条边最多产生一个交点，总复杂度为 $O(nm)$。

#### 动手尝试

1.  画一个三角形并用正方形裁剪它，跟随每一步。
2.  尝试交换裁剪多边形和主体多边形。
3.  测试退化情况（无交集、完全包含）。
4.  比较凸裁剪多边形与凹裁剪多边形。
5.  扩展到 Weiler–Atherton 算法以处理非凸形状。

#### 测试用例

| 主体多边形        | 裁剪多边形 | 结果                 |
| ---------------------- | ------------ | ---------------------- |
| 跨越正方形的三角形 | 正方形       | 裁剪后的五边形       |
| 完全在内部           | 正方形       | 保持不变              |
| 完全在外部          | 正方形       | 空集                  |
| 重叠的矩形 | 两者         | 交集矩形 |

#### 复杂度

$$
\text{时间: } O(nm), \quad \text{空间: } O(n + m)
$$

多边形求交是几何学的布尔运算符，逐步修剪形状，直到只剩下共享区域。
### 720 最近邻点对（使用 KD 树）

最近邻点对问题要求我们在给定的点集中找到距离最近的一对点，这是计算几何和空间数据分析中的一个基本问题。

它是聚类、图形学、机器学习和碰撞检测等算法的基础，可以使用分治法、扫描线法或像 KD 树这样的空间数据结构来高效解决。

#### 我们要解决什么问题？

给定平面上的一个包含 $n$ 个点的集合 $P = {p_1, p_2, \dots, p_n}$，找到两个不同的点 $(p_i, p_j)$，使得欧几里得距离

$$
d(p_i, p_j) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}
$$

最小。

朴素地检查所有 $\binom{n}{2}$ 对点需要 $O(n^2)$ 的时间。
我们想要一个 $O(n \log n)$ 或更好的解决方案。

#### 它是如何工作的（通俗解释）？

我们将重点介绍 KD 树方法，它在低维空间中高效支持最近邻查询。

KD 树（k 维树）递归地沿着坐标轴划分空间：

1. 构建阶段

   * 按 $x$ 坐标对点排序，在中位数处分割 → 根节点
   * 递归地构建左子树（较小的 $x$）和右子树（较大的 $x$）
   * 在每一层深度交替使用坐标轴（$x$, $y$, $x$, $y$, …）

2. 查询阶段（针对每个点）

   * 遍历 KD 树以找到最近的候选点
   * 回溯检查可能包含更近点的子树
   * 维护全局最小距离和点对

通过利用轴对齐的边界框，许多区域可以提前被剪枝（忽略）。

#### 分步详解（概念性）

1. 以 $O(n \log n)$ 的时间复杂度构建 KD 树。
2. 对于每个点 $p$，在 $O(\log n)$ 的期望时间内搜索其最近邻点。
3. 跟踪具有最小距离的全局最小点对 $(p, q)$。

#### 示例演练

点集：
$$
P = {(1,1), (4,4), (5,1), (7,2)}
$$

1. 按 $x$ 坐标分割构建 KD 树：
   根节点 = $(4,4)$
   左子树 = $(1,1)$
   右子树 = $(5,1),(7,2)$

2. 为每个点查询最近邻：

   * $(1,1)$ → 最近邻 = $(4,4)$ ($d=4.24$)
   * $(4,4)$ → 最近邻 = $(5,1)$ ($d=3.16$)
   * $(5,1)$ → 最近邻 = $(7,2)$ ($d=2.24$)
   * $(7,2)$ → 最近邻 = $(5,1)$ ($d=2.24$)

 最近点对：$(5,1)$ 和 $(7,2)$

#### 微型代码（Python）

```python
from math import sqrt

def dist(a, b):
    return sqrt((a[0]-b[0])2 + (a[1]-b[1])2)

def build_kdtree(points, depth=0):
    if not points: return None
    k = 2
    axis = depth % k
    points.sort(key=lambda p: p[axis])
    mid = len(points) // 2
    return {
        'point': points[mid],
        'left': build_kdtree(points[:mid], depth+1),
        'right': build_kdtree(points[mid+1:], depth+1)
    }

def nearest_neighbor(tree, target, depth=0, best=None):
    if tree is None: return best
    point = tree['point']
    if best is None or dist(target, point) < dist(target, best):
        best = point
    axis = depth % 2
    next_branch = tree['left'] if target[axis] < point[axis] else tree['right']
    best = nearest_neighbor(next_branch, target, depth+1, best)
    return best

points = [(1,1), (4,4), (5,1), (7,2)]
tree = build_kdtree(points)
best_pair = None
best_dist = float('inf')
for p in points:
    q = nearest_neighbor(tree, p)
    if q != p:
        d = dist(p, q)
        if d < best_dist:
            best_pair = (p, q)
            best_dist = d
print("最近点对:", best_pair, "距离:", best_dist)
```

#### 微型代码（C 语言，概念性草图）

在 C 语言中构建完整的 KD 树更为复杂，但核心逻辑如下：

```c
double dist(Point a, Point b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}

// 根据深度递归地沿 x 或 y 轴分割点
Node* build_kdtree(Point* points, int n, int depth) {
    // 按轴排序，选择中位数作为根节点
    // 递归处理左子树和右子树
}

// 通过剪枝递归搜索最近邻
void nearest_neighbor(Node* root, Point target, Point* best, double* bestDist, int depth) {
    // 比较当前点，在可能有希望的子树中递归
    // 如果另一侧子树可能包含更近的点，则回溯
}
```

#### 为什么它很重要

* 避免了 $O(n^2)$ 的暴力方法
* 对于中等维度（2D、3D）具有良好的扩展性
* 可推广到范围搜索、半径查询、聚类

应用：

* 图形学（对象邻近度、网格简化）
* 机器学习（k-NN 分类）
* 机器人学（最近障碍物检测）
* 空间数据库（地理查询）

#### 一个温和的证明（为什么它有效）

每个递归分区定义了一个存储点的半空间。
在搜索时，我们总是探索包含查询点的一侧，但如果查询点周围的超球面与分割平面相交，则必须检查另一侧。

由于每一层都将数据大致分成两半，访问节点的期望数量是 $O(\log n)$。
通过递归寻找中位数，构建树的时间复杂度是 $O(n \log n)$。

总的最远点对复杂度：

$$
O(n \log n)
$$

#### 亲自尝试

1.  绘制 10 个随机点，计算暴力法得到的点对。
2.  手动构建一个 KD 树（交替使用 x/y 轴）。
3.  追踪最近邻搜索的步骤。
4.  比较搜索顺序和剪枝决策。
5.  扩展到 3D 点。

#### 测试用例

| 点集                  | 结果          | 备注               |
| --------------------- | ------------- | ------------------- |
| (0,0), (1,1), (3,3)   | (0,0)-(1,1)   | $d=\sqrt2$          |
| (1,1), (2,2), (2,1.1) | (2,2)-(2,1.1) | 最近点对             |
| 随机 10 个点          | 已验证        | 与暴力法结果匹配     |

#### 复杂度

$$
\text{时间: } O(n \log n), \quad \text{空间: } O(n)
$$

最近邻点对问题是几何学的本能，它通过优雅的分治与搜索推理，在拥挤的空间中找到最亲密的伴侣。

# 第 73 节 扫描线与平面扫描算法
### 721 扫描线算法处理事件

扫描线算法是一个统一的框架，通过沿着一条移动的线（通常是垂直线）按排序顺序处理事件，来解决许多几何问题。它将空间关系转化为时间序列，允许我们使用动态活动集高效地跟踪交点、重叠或活动对象。

这种范式是 Bentley–Ottmann、最近点对、矩形并集和天际线等算法的核心。

#### 我们要解决什么问题？

我们想要处理在平面中相互作用的几何事件、点、线段、矩形、圆等。挑战在于：如果我们只考虑在特定扫描位置处于活动状态的对象，许多空间问题就会变得简单。

例如：

* 在交点检测中，只有相邻的线段才可能相交。
* 在矩形并集中，只有活动区间才对总面积有贡献。
* 在天际线计算中，只有当前最高的高度才重要。

因此，我们重新表述问题：

> 让一条扫描线扫过平面，逐个处理事件，并在几何对象进入或离开时更新活动集。

#### 它是如何工作的（通俗解释）？

1. 事件队列 (EQ)

   * 所有关键点按 $x$（或时间）排序。
   * 每个事件标记一个开始、结束或变化（如交点）。

2. 活动集 (AS)

   * 存储当前与扫描线相交的“活动”对象。
   * 维护在一个按另一个坐标（如 $y$）排序的结构中。

3. 主循环
   按排序顺序处理每个事件：

   * 将新几何对象插入 AS。
   * 移除过期的几何对象。
   * 查询或更新关系（邻居、计数、交点）。

4. 继续直到 EQ 为空。

使用平衡树时，每一步都是对数复杂度，因此总复杂度为 $O((n+k)\log n)$，其中 $k$ 是交互次数（例如交点数）。

#### 示例演练

我们以线段交点为例：

线段：

* $S_1: (0,0)$–$(4,4)$
* $S_2: (0,4)$–$(4,0)$

事件：端点按 $x$ 排序：
$(0,0)$, $(0,4)$, $(4,0)$, $(4,4)$

步骤：

1. 在 $x=0$ 处，插入 $S_1$, $S_2$。
2. 检查活动集顺序 → 检测到交点 $(2,2)$ → 将交点事件加入队列。
3. 在 $x=2$ 处，处理交点 → 交换 AS 中的顺序。
4. 继续 → 报告所有交点。

结果：通过事件驱动的扫描找到交点 $(2,2)$。

#### 微型代码（Python 草图）

```python
import heapq

def sweep_line(events):
    heapq.heapify(events)  # 按 x 的最小堆
    active = set()
    while events:
        x, event_type, obj = heapq.heappop(events)
        if event_type == 'start':
            active.add(obj)
        elif event_type == 'end':
            active.remove(obj)
        elif event_type == 'intersection':
            print("Intersection at x =", x)
        # 如果需要，处理活动集中的邻居
```

用法：

* 用元组 `(x, type, object)` 填充 `events`
* 随着扫描进行，从 `active` 中插入/移除对象

#### 微型代码（C 语言框架）

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct { double x; int type; int id; } Event;

int cmp(const void* a, const void* b) {
    double x1 = ((Event*)a)->x, x2 = ((Event*)b)->x;
    return (x1 < x2) ? -1 : (x1 > x2);
}

void sweep_line(Event* events, int n) {
    qsort(events, n, sizeof(Event), cmp);
    for (int i = 0; i < n; i++) {
        if (events[i].type == 0) printf("Start event at x=%.2f\n", events[i].x);
        if (events[i].type == 1) printf("End event at x=%.2f\n", events[i].x);
    }
}
```

#### 为什么它很重要

* 计算几何中的通用模式
* 将二维问题转化为排序的一维扫描
* 实现交点、并集和计数的高效检测
* 用于图形学、GIS、模拟、CAD

应用：

* Bentley–Ottmann（线段交点）
* 矩形并集面积
* 范围计数和查询
* 平面细分和可见性图

#### 一个温和的证明（为什么它有效）

在任何时刻，只有与扫描线相交的对象才能影响结果。通过按排序顺序处理事件，我们保证：

* 几何关系的每一次变化都发生在一个事件处。
* 在事件之间，活动集的结构保持稳定。

因此，我们可以增量地维护局部状态（邻居、计数、最大值），永远不需要重新访问旧的位置。

对于 $n$ 个输入元素和 $k$ 次交互，总成本为：

$$
O((n + k)\log n)
$$

因为每次插入、删除或邻居检查都是 $O(\log n)$。

#### 自己动手试试

1.  画出线段并按 $x$ 对端点排序。
2.  扫描一条垂直线并跟踪它穿过了哪些线段。
3.  记录每次两条线段改变顺序的时刻 → 交点！
4.  尝试矩形或区间，观察活动集如何变化。

#### 测试用例

| 输入                         | 预期结果        | 备注                   |
| ---------------------------- | --------------- | ---------------------- |
| 2 条相交线段                 | 1 个交点        | 在中心                 |
| 3 条两两相交的线段           | 3 个交点        | 全部检测到             |
| 不重叠的线段                 | 无交点          | 活动集保持很小         |

#### 复杂度

$$
\text{时间: } O((n + k)\log n), \quad \text{空间: } O(n)
$$

扫描线是几何学的传送带，滑过空间，一次一个事件地更新世界。
### 722 区间调度

区间调度算法是线性贪心优化的基石。给定一组时间区间，每个区间代表一个作业或任务，目标是选择最大数量的不重叠区间。这个简单而深刻的算法构成了资源分配、时间线规划和空间调度问题的核心。

#### 我们要解决什么问题？

给定 $n$ 个区间：

$$
I_i = [s_i, f_i), \quad i = 1, \ldots, n
$$

我们希望找到最大的区间子集，使得其中任意两个区间不重叠。
形式化地，找到 $S \subseteq \{1, \ldots, n\}$，使得对于所有 $i, j \in S$，

$$
[s_i, f_i) \cap [s_j, f_j) = \emptyset
$$

并且 $|S|$ 最大化。

示例：

| 区间 | 开始时间 | 结束时间 |
| --------- | ------ | ------- |
| $I_1$     | 1      | 4       |
| $I_2$     | 3      | 5       |
| $I_3$     | 0      | 6       |
| $I_4$     | 5      | 7       |
| $I_5$     | 8      | 9       |

最优调度：$I_1, I_4, I_5$ (3 个区间)

#### 它是如何工作的（通俗解释）？

贪心思想：

> 总是选择最早结束的区间，然后丢弃所有与之重叠的区间，并重复此过程。

推理：

- 尽早结束可以为后续任务留出更多空间。
- 没有更早的结束时间能增加数量；它只会阻塞后面的区间。

#### 算法（贪心策略）

1. 按结束时间 $f_i$ 对区间进行排序。
2. 初始化空集合 $S$。
3. 按顺序处理每个区间 $I_i$：

   * 如果 $I_i$ 的开始时间晚于或等于最后一个被选中间区的结束时间 → 选择它。
4. 返回 $S$。

#### 示例演练

输入：
$(1,4), (3,5), (0,6), (5,7), (8,9)$

1. 按结束时间排序：
   $(1,4), (3,5), (0,6), (5,7), (8,9)$

2. 从 $(1,4)$ 开始

   * 下一个 $(3,5)$ 重叠 → 跳过
   * $(0,6)$ 重叠 → 跳过
   * $(5,7)$ 符合条件 → 选择
   * $(8,9)$ 符合条件 → 选择

 输出：$(1,4), (5,7), (8,9)$

#### 微型代码（Python）

```python
def interval_scheduling(intervals):
    intervals.sort(key=lambda x: x[1])  # 按结束时间排序
    selected = []
    current_end = float('-inf')
    for (s, f) in intervals:
        if s >= current_end:
            selected.append((s, f))
            current_end = f
    return selected

intervals = [(1,4),(3,5),(0,6),(5,7),(8,9)]
print("最优调度:", interval_scheduling(intervals))
```

#### 微型代码（C）

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct { int s, f; } Interval;

int cmp(const void *a, const void *b) {
    return ((Interval*)a)->f - ((Interval*)b)->f;
}

void interval_scheduling(Interval arr[], int n) {
    qsort(arr, n, sizeof(Interval), cmp);
    int last_finish = -1;
    for (int i = 0; i < n; i++) {
        if (arr[i].s >= last_finish) {
            printf("(%d, %d)\n", arr[i].s, arr[i].f);
            last_finish = arr[i].f;
        }
    }
}

int main() {
    Interval arr[] = {{1,4},{3,5},{0,6},{5,7},{8,9}};
    interval_scheduling(arr, 5);
}
```

#### 为何重要

* 贪心证明：最早结束的区间永远不会损害最优性
* 是资源调度、CPU 作业选择、会议室规划的基础
* 是加权变体、区间划分、线段树的基础

应用：

* CPU 进程调度
* 铁路或跑道时段分配
* 活动规划和预订系统
* 非重叠任务分配

#### 一个温和的证明（为何有效）

令 $S^*$ 是一个最优解，$I_g$ 是贪心算法选择的那个最早结束的区间。
我们可以将 $S^*$ 转换，使其也包含 $I_g$ 而不减少其大小，方法是用 $I_g$ 替换任何与之重叠的区间。

因此，通过归纳法：

* 贪心算法总能找到一个最优子集。

总运行时间主要由排序决定：

$$
O(n \log n)
$$

#### 亲自尝试

1. 在一条线上画出区间，模拟贪心选择过程。
2. 添加重叠区间，观察哪些被跳过。
3. 与暴力方法（检查所有子集）进行比较。
4. 扩展到加权区间调度（使用动态规划）。

#### 测试用例

| 区间                     | 最优调度  | 数量 |
| ----------------------------- | ----------------- | ----- |
| (1,4),(3,5),(0,6),(5,7),(8,9) | (1,4),(5,7),(8,9) | 3     |
| (0,2),(1,3),(2,4),(3,5)       | (0,2),(2,4)       | 2     |
| (1,10),(2,3),(3,4),(4,5)      | (2,3),(3,4),(4,5) | 3     |

#### 复杂度

$$
\text{时间: } O(n \log n), \quad \text{空间: } O(1)
$$

区间调度算法是贪心优雅的典范，它每次决策都选择最早结束的那个，从而在时间线上描绘出最长的不重叠路径。
### 723 矩形并集面积

矩形并集面积算法计算一组与坐标轴对齐的矩形所覆盖的总面积。重叠区域只应计算一次，即使多个矩形覆盖了同一区域。

这个问题是扫描线技术与区间管理相结合的经典演示，将一个二维几何问题转化为一系列一维范围计算。

#### 我们要解决什么问题？

给定 $n$ 个与坐标轴对齐的矩形，
每个矩形 $R_i = [x_1, x_2) \times [y_1, y_2)$，

我们想要计算它们并集的总面积：

$$
A = \text{面积}\left(\bigcup_{i=1}^n R_i\right)
$$

重叠区域必须只计算一次。
暴力网格枚举成本太高，我们需要一种几何的、事件驱动的方法。

#### 它是如何工作的（通俗解释）？

我们使用一条沿 $x$ 轴移动的垂直扫描线：

1. 事件：
   每个矩形生成两个事件：
   * 在 $x_1$ 处：添加垂直区间 $[y_1, y_2)$
   * 在 $x_2$ 处：移除垂直区间 $[y_1, y_2)$

2. 活动集：
   在扫描过程中，维护一个存储活动 y 区间的结构，代表扫描线当前与矩形相交的区域。

3. 面积累加：
   当扫描线从 $x_i$ 移动到 $x_{i+1}$ 时，
   从活动集计算出覆盖的 y 长度 ($L$)，
   贡献的面积为：

   $$
   A += L \times (x_{i+1} - x_i)
   $$

通过按排序顺序处理所有 $x$ 事件，我们捕获所有添加/移除操作并精确累加面积。

#### 示例演练

矩形：

1. $(1, 1, 3, 3)$
2. $(2, 2, 4, 4)$

事件：

* $x=1$: 添加 [1,3]
* $x=2$: 添加 [2,4]
* $x=3$: 移除 [1,3]
* $x=4$: 移除 [2,4]

逐步计算：

| 区间                      | x 范围 | 覆盖的 y 长度 | 面积 |
| ------------------------- | ------- | --------- | ---- |
| [1,3]                     | 1→2     | 2         | 2    |
| [1,3]+[2,4] → 合并为 [1,4] | 2→3     | 3         | 3    |
| [2,4]                     | 3→4     | 2         | 2    |

 总面积 = 2 + 3 + 2 = 7

#### 精简代码 (Python)

```python
def union_area(rectangles):
    events = []
    for (x1, y1, x2, y2) in rectangles:
        events.append((x1, 1, y1, y2))  # 开始
        events.append((x2, -1, y1, y2)) # 结束
    events.sort()  # 按 x 排序

    def compute_y_length(active):
        # 合并区间
        merged, last_y2, total = [], -float('inf'), 0
        for y1, y2 in sorted(active):
            y1 = max(y1, last_y2)
            if y2 > y1:
                total += y2 - y1
                last_y2 = y2
        return total

    active, prev_x, area = [], 0, 0
    for x, typ, y1, y2 in events:
        area += compute_y_length(active) * (x - prev_x)
        if typ == 1:
            active.append((y1, y2))
        else:
            active.remove((y1, y2))
        prev_x = x
    return area

rects = [(1,1,3,3),(2,2,4,4)]
print("Union area:", union_area(rects))  # 7
```

#### 精简代码 (C, 概念性)

```c
typedef struct { double x; int type; double y1, double y2; } Event;
```

* 按 $x$ 对事件排序
* 维护活动区间（链表或线段树）
* 计算合并后的 $y$ 长度并累加 $L \times \Delta x$

高效的实现使用线段树来跟踪覆盖计数和总长度，每次更新复杂度为 $O(\log n)$。

#### 为什么它很重要？

* 是计算几何、地理信息系统、图形学的基础
* 处理并集面积、周长、体积（高维类似问题）
* 是碰撞面积、覆盖计算、地图叠加的基础

应用：

* 渲染重叠矩形
* 土地或地块并集面积
* 碰撞检测（2D 包围盒）
* CAD 和布局设计工具

#### 一个温和的证明（为什么它有效）

在每个扫描位置，所有变化都发生在事件边界 ($x_i$) 处。
在 $x_i$ 和 $x_{i+1}$ 之间，活动区间的集合保持不变。
因此，面积可以增量计算：

$$
A = \sum_{i} L_i \cdot (x_{i+1} - x_i)
$$

其中 $L_i$ 是在 $x_i$ 处覆盖的总 $y$ 长度。
由于每次插入/移除只更新局部区间，通过维护活动区间的并集，算法的正确性得以保证。

#### 亲自尝试

1.  绘制 2–3 个重叠的矩形。
2.  列出它们的 $x$ 事件。
3.  扫描并跟踪活动的 $y$ 区间。
4.  合并重叠以计算 $L_i$。
5.  每一步乘以 $\Delta x$。

#### 测试用例

| 矩形                     | 期望面积 | 备注             |
| ------------------------ | ------------- | ----------------- |
| (1,1,3,3), (2,2,4,4)     | 7             | 部分重叠   |
| (0,0,1,1), (1,0,2,1)     | 2             | 不相邻          |
| (0,0,2,2), (1,1,3,3)     | 7             | 角落重叠 |

#### 复杂度

$$
\text{时间: } O(n \log n), \quad \text{空间: } O(n)
$$

矩形并集面积算法将一个复杂的二维并集问题转化为带有活动区间合并的一维扫描，精确、优雅且可扩展。
### 724 线段相交（Bentley–Ottmann 变体）

线段相交问题要求我们找出平面上 $n$ 条线段中所有的交点。
Bentley–Ottmann 算法是经典的扫描线方法，它将朴素的 $O(n^2)$ 两两检查改进为

$$
O\big((n + k)\log n\big)
$$

其中 $k$ 是交点的数量。

这个变体是专门针对线段的、事件驱动的扫描线方法的直接应用。

#### 我们要解决什么问题？

给定 $n$ 条线段

$$
S = { s_1, s_2, \ldots, s_n }
$$

我们想要计算任意两条线段之间所有交点的集合。
我们既需要知道哪些线段相交，也需要知道在哪里相交。

#### 朴素方法与扫描线法

* 朴素方法：
  检查所有 $\binom{n}{2}$ 对线段 → $O(n^2)$ 时间。
  即使对于小的 $n$，当交点很少时，这种方法也很浪费。

* 扫描线法（Bentley–Ottmann）：

  * 按 $x$ 坐标递增顺序处理事件
  * 维护按 $y$ 坐标排序的活动线段集合
  * 只有相邻的线段才可能相交 → 仅进行局部检查

这将二次搜索变成了一个输出敏感的算法。

#### 它是如何工作的（通俗解释）

我们让一条垂直的扫描线从左向右移动，处理三种事件类型：

| 事件类型   | 描述                                   |
| ---------- | -------------------------------------- |
| 起点       | 将线段添加到活动集合中                 |
| 终点       | 将线段从活动集合中移除                 |
| 交点       | 两条线段相交；记录交点，交换顺序       |

活动集合在扫描线处按线段的高度（$y$ 坐标）排序。
当插入一条新线段时，我们只测试它与邻居是否相交。
交换顺序后，我们只测试新形成的相邻线段对。

#### 示例演练

线段：

* $S_1: (0,0)$–$(4,4)$
* $S_2: (0,4)$–$(4,0)$
* $S_3: (1,0)$–$(1,3)$

事件队列（按 $x$ 排序）：
$(0,0)$, $(0,4)$, $(1,0)$, $(1,3)$, $(4,0)$, $(4,4)$

1. $x=0$：插入 $S_1$, $S_2$ → 检查这对线段 → 发现交点 $(2,2)$。
2. $x=1$：插入 $S_3$，检查与邻居，没有新交点。
3. $x=2$：处理交点 $(2,2)$，交换 $S_1$, $S_2$ 的顺序。
4. 继续 → 当扫描线经过线段端点时移除线段。

输出：交点 $(2,2)$。

#### 几何测试：方向判定

给定线段 $AB$ 和 $CD$，它们相交当且仅当

$$
\text{orient}(A, B, C) \ne \text{orient}(A, B, D)
$$

且

$$
\text{orient}(C, D, A) \ne \text{orient}(C, D, B)
$$

这使用叉积方向判定来测试点是否位于线段的两侧。

#### 微型代码（Python）

```python
import heapq

def orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def intersect(a, b, c, d):
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return o1*o2 < 0 and o3*o4 < 0

def bentley_ottmann(segments):
    events = []
    for s in segments:
        (x1,y1),(x2,y2) = s
        if x1 > x2:
            s = ((x2,y2),(x1,y1))
        events.append((x1, 'start', s))
        events.append((x2, 'end', s))
    heapq.heapify(events)

    active, intersections = [], []
    while events:
        x, typ, seg = heapq.heappop(events)
        if typ == 'start':
            active.append(seg)
            for other in active:
                if other != seg and intersect(seg[0], seg[1], other[0], other[1]):
                    intersections.append(x)
        elif typ == 'end':
            active.remove(seg)
    return intersections

segments = [((0,0),(4,4)), ((0,4),(4,0)), ((1,0),(1,3))]
print("Intersections:", bentley_ottmann(segments))
```

#### 为什么它很重要

* 输出敏感：随实际交点数量扩展
* 几何引擎、CAD 工具和图形管线的核心
* 用于多边形裁剪、网格叠加和地图相交

应用：

* 检测矢量地图中的线段交叉
* 在 GIS 中叠加几何图层
* 路径相交检测（道路、电线、边）
* 三角剖分和可见性图的预处理

#### 一个温和的证明（为什么它有效）

每个交点事件都对应于线段垂直顺序的一次交换。
因为顺序只在交点处改变，所以所有交点都可以通过处理以下事件来发现：

1. 插入/删除（起点/终点事件）
2. 交换（交点事件）

我们永远不会遗漏或重复一个交点，因为在事件之间只有相邻的线段对才可能相交。

总操作数：

* $n$ 个起点，$n$ 个终点，$k$ 个交点 → $O(n + k)$ 个事件
* 每个事件使用 $O(\log n)$ 次操作（堆/树）

因此

$$
O\big((n + k)\log n\big)
$$

#### 自己动手试试

1.  绘制有多处交叉的线段。
2.  按 $x$ 坐标对端点排序。
3.  扫描并维护有序的活动集合。
4.  在发生交换时记录交点。
5.  与暴力两两检查进行比较。

#### 测试用例

| 线段               | 交点数量 |
| ------------------ | -------- |
| 正方形的对角线     | 1        |
| 网格交叉           | 多个     |
| 平行线             | 0        |
| 随机线段           | 已验证   |

#### 复杂度

$$
\text{时间: } O((n + k)\log n), \quad \text{空间: } O(n + k)
$$

线段相交的 Bentley–Ottmann 变体是基准技术，是事件与交换的精确舞蹈，能捕获每一个交点，且仅捕获一次。
### 725 天际线问题

天际线问题是一个经典的几何扫描线挑战：给定城市景观中一系列矩形建筑，计算从远处观看时形成的轮廓（或剪影）。

这是一个典型的分治法和扫描线示例，它将重叠的矩形转换为一个分段高度函数，随着扫描的进行，该函数会上升和下降。

#### 我们要解决什么问题？

每个建筑 $B_i$ 由三个数字定义：

$$
B_i = (x_{\text{left}}, x_{\text{right}}, h)
$$

我们想要计算天际线，即一系列关键点：

$$
$$(x_1, h_1), (x_2, h_2), \ldots, (x_m, 0)]
$$

使得所有建筑的上轮廓被精确地描绘一次。

示例输入：

| 建筑 | 左边界 | 右边界 | 高度 |
| -------- | ---- | ----- | ------ |
| 1        | 2    | 9     | 10     |
| 2        | 3    | 7     | 15     |
| 3        | 5    | 12    | 12     |

输出：

$$
$$(2,10), (3,15), (7,12), (12,0)]
$$

#### 它是如何工作的（通俗解释）

天际线只在建筑的边缘（左侧或右侧）发生变化。
我们将每条边视为扫描线从左向右移动时的一个事件：

1. 在左边缘 ($x_\text{left}$)：将建筑高度添加到活动集合中。
2. 在右边缘 ($x_\text{right}$)：从活动集合中移除高度。
3. 每个事件后，天际线高度 = max(活动集合)。
4. 如果高度发生变化，则将 $(x, h)$ 追加到结果中。

通过跟踪当前最高的建筑，这种方法高效地构建了轮廓。

#### 示例演练

输入：
$(2,9,10), (3,7,15), (5,12,12)$

事件：

* (2, 开始, 10)
* (3, 开始, 15)
* (5, 开始, 12)
* (7, 结束, 15)
* (9, 结束, 10)
* (12, 结束, 12)

步骤：

| x  | 事件    | 活动高度 | 最大高度 | 输出 |
| -- | -------- | -------------- | ---------- | ------ |
| 2  | 开始 10 | {10}           | 10         | (2,10) |
| 3  | 开始 15 | {10,15}        | 15         | (3,15) |
| 5  | 开始 12 | {10,15,12}     | 15         | –      |
| 7  | 结束 15 | {10,12}        | 12         | (7,12) |
| 9  | 结束 10 | {12}           | 12         | –      |
| 12 | 结束 12 | {}             | 0          | (12,0) |

输出天际线：
$$
$$(2,10), (3,15), (7,12), (12,0)]
$$

#### 精简代码（Python）

```python
import heapq

def skyline(buildings):
    events = []
    for L, R, H in buildings:
        events.append((L, -H))  # 开始
        events.append((R, H))   # 结束
    events.sort()

    result = []
    heap = [0]  # 最大堆（存储负值）
    prev_max = 0
    active = {}

    for x, h in events:
        if h < 0:  # 开始
            heapq.heappush(heap, h)
        else:      # 结束
            active[h] = active.get(h, 0) + 1  # 标记为待移除
        # 清理已结束的高度
        while heap and active.get(-heap[0], 0):
            active[-heap[0]] -= 1
            if active[-heap[0]] == 0:
                del active[-heap[0]]
            heapq.heappop(heap)
        curr_max = -heap[0]
        if curr_max != prev_max:
            result.append((x, curr_max))
            prev_max = curr_max
    return result

buildings = [(2,9,10), (3,7,15), (5,12,12)]
print("Skyline:", skyline(buildings))
```

#### 精简代码（C，概念性）

```c
typedef struct { int x, h, type; } Event;
```

1. 按 $x$（以及平局时按高度）对事件排序。
2. 使用平衡树（多重集合）来维护活动高度。
3. 开始时，插入高度；结束时，移除高度。
4. 将最大高度的变化记录为输出点。

#### 为什么它很重要

* 展示了使用优先队列的基于事件的扫描
* 在渲染、城市建模、区间聚合中的核心作用
* 矩形并集的对偶问题，这里我们关心的是上轮廓，而不是面积

应用：

* 城市景观渲染
* 范围聚合可视化
* 直方图或条形图合并轮廓
* 阴影或覆盖范围分析

#### 一个温和的证明（为什么它有效）

天际线只在边缘处变化，因为内部点是连续覆盖的。
在边缘之间，活动建筑的集合是恒定的，因此最大高度也是恒定的。

通过按顺序处理所有边缘并记录每次高度变化，我们重建了精确的上包络线。

每次插入/移除的时间复杂度为 $O(\log n)$（堆操作），并且有 $2n$ 个事件：

$$
O(n \log n)
$$

#### 自己动手试试

1. 画出 2–3 个重叠的建筑。
2. 按 $x$ 对所有边缘排序。
3. 扫描并跟踪活动高度。
4. 每次最大高度变化时记录输出。
5. 通过手动描绘轮廓来验证。

#### 测试用例

| 建筑                   | 天际线                       |
| --------------------------- | ----------------------------- |
| (2,9,10),(3,7,15),(5,12,12) | (2,10),(3,15),(7,12),(12,0)   |
| (1,3,3),(2,4,4),(5,6,1)     | (1,3),(2,4),(4,0),(5,1),(6,0) |
| (1,2,1),(2,3,2),(3,4,3)     | (1,1),(2,2),(3,3),(4,0)       |

#### 复杂度

$$
\text{时间: } O(n \log n), \quad \text{空间: } O(n)
$$

天际线问题捕捉了几何的起伏节奏，一个由重叠形状构建的阶梯状剪影，以及扫描事件过程中的优雅。
### 726 最近点对扫描算法

最近点对扫描算法利用扫描线和活动集，寻找平面上任意两点之间的最小距离。它是结合排序、几何和局部性的最优雅示例之一，将 $O(n^2)$ 的搜索转化为 $O(n \log n)$ 的算法。

#### 我们要解决什么问题？

给定平面上的 $n$ 个点 $P = {p_1, p_2, \ldots, p_n}$，找到两个不同的点 $(p_i, p_j)$，使得它们的欧几里得距离最小：

$$
d(p_i, p_j) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}
$$

我们既想要这个最小距离，也想要达到这个距离的点对。

一个朴素的 $O(n^2)$ 算法会检查所有点对。我们将使用扫描线和空间剪枝来做得更好。

#### 它是如何工作的？（通俗解释）

关键洞见：当从左向右扫描时，只有位于一个狭窄垂直条带内的点才可能成为最近点对。

算法概要：

1.  按 $x$ 坐标对点进行排序。
2.  维护一个活动集（按 $y$ 排序），其中包含位于当前条带宽度（等于目前找到的最佳距离）内的点。
3.  对于每个新点：
    *   移除 $x$ 坐标太靠左的点。
    *   仅与垂直方向上距离在 $\delta$ 内的点进行比较，其中 $\delta$ 是当前最佳距离。
    *   如果找到更小的距离，则更新最佳距离。
4.  继续直到处理完所有点。

这之所以有效，是因为在一个 $\delta \times 2\delta$ 的条带内，最多只有 6 个点可能足够接近以改进最佳距离。

#### 示例演练

点集：
$$
P = {(1,1), (2,3), (3,2), (5,5)}
$$

1.  按 $x$ 排序：$(1,1), (2,3), (3,2), (5,5)$
2.  从第一个点 $(1,1)$ 开始
3.  添加 $(2,3)$ → $d=\sqrt{5}$
4.  添加 $(3,2)$ → 与最后 2 个点比较
    *   $d((2,3),(3,2)) = \sqrt{2}$ → 最佳 $\delta = \sqrt{2}$
5.  添加 $(5,5)$ → $(1,1)$ 的 $x$ 差值 > $\delta$，移除它
    *   比较 $(5,5)$ 与 $(2,3),(3,2)$ → 未找到更小的

输出：最近点对 $(2,3),(3,2)$，距离 $\sqrt{2}$。

#### 微型代码（Python）

```python
from math import sqrt
import bisect

def dist(a, b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def closest_pair(points):
    points.sort()  # 按 x 排序
    best = float('inf')
    best_pair = None
    active = []  # 按 y 排序
    j = 0

    for i, p in enumerate(points):
        x, y = p
        while (x - points[j][0]) > best:
            active.remove(points[j])
            j += 1
        pos = bisect.bisect_left(active, (y - best, -float('inf')))
        while pos < len(active) and active[pos][0] <= y + best:
            d = dist(p, active[pos])
            if d < best:
                best, best_pair = d, (p, active[pos])
            pos += 1
        bisect.insort(active, p)
    return best, best_pair

points = [(1,1), (2,3), (3,2), (5,5)]
print("最近点对:", closest_pair(points))
```

#### 微型代码（C，概念性草图）

```c
typedef struct { double x, y; } Point;
double dist(Point a, Point b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}
// 按 x 排序，维护活动集（按 y 的平衡二叉搜索树）
// 对于每个新点，移除 x 坐标过远的点，搜索 y 坐标附近的点，更新最佳距离
```

高效的实现使用平衡搜索树或有序列表。

#### 为什么它重要

*   计算几何中的经典算法
*   结合了排序 + 扫描 + 局部搜索
*   使用几何剪枝的空间算法模型

应用：

*   最近邻搜索
*   聚类和模式识别
*   运动规划（最小间隔）
*   空间索引和范围查询

#### 一个温和的证明（为什么它有效）

在每一步中，$x$ 坐标距离超过 $\delta$ 的点无法改进最佳距离。在 $\delta$ 条带内，每个点最多有 6 个邻居（$\delta \times \delta$ 网格中的打包论证）。因此，排序后的总比较次数是线性的。

总体复杂度：

$$
O(n \log n)
$$

来自初始排序和对数时间的插入/删除操作。

#### 自己动手试试

1.  绘制一些点。
2.  按 $x$ 坐标排序。
3.  从左向右扫描。
4.  保持条带宽度为 $\delta$，仅检查局部邻居。
5.  与暴力法比较以验证。

#### 测试用例

| 点集                      | 最近点对       | 距离               |
| ------------------------- | -------------- | ------------------ |
| (1,1),(2,3),(3,2),(5,5)   | (2,3),(3,2)    | $\sqrt{2}$         |
| (0,0),(1,0),(2,0)         | (0,0),(1,0)    | 1                  |
| 随机 10 个点              | 已验证         | 与暴力法结果一致   |

#### 复杂度

$$
\text{时间: } O(n \log n), \quad \text{空间: } O(n)
$$

最近点对扫描算法是几何学的精密工具，它将搜索范围缩小到一个移动的条带，并且只比较那些真正重要的邻居。
### 727 圆排列扫描算法

圆排列扫描算法用于计算一组圆的排列，即由所有圆弧及其交点诱导出的平面细分。它是直线和线段排列的推广，扩展到曲线边，需要事件驱动的扫描和几何推理。

#### 我们要解决什么问题？

给定 $n$ 个圆
$$
C_i: (x_i, y_i, r_i)
$$
我们想要计算它们的排列：由圆之间的交点形成的平面分解，包括面、边和顶点。

一个更简单的变体侧重于计算交点数量和构造交点。

每对圆最多可以有两个交点，因此最多有

$$
O(n^2)
$$
个交点。

#### 为什么比直线更难？

* 圆引入了非线性边界。
* 扫描线必须处理弧段，而不仅仅是直线区间。
* 事件发生在圆的起始/结束 x 坐标以及交点处。

这意味着每个圆在扫描过程中会进入和退出两次，并且新的交点可以动态出现。

#### 它是如何工作的（通俗解释）

扫描线从左向右移动，将圆作为垂直切片进行相交。我们维护一个当前与扫描线相交的圆弧的活动集合。

在每个事件点：

1. 最左侧点 ($x_i - r_i$)：插入圆弧。
2. 最右侧点 ($x_i + r_i$)：移除圆弧。
3. 交点：当两个弧相交时，安排交点事件。

每次插入或交换弧时，检查局部相邻弧是否有交点（类似于 Bentley–Ottmann 算法，但处理的是曲线段）。

#### 圆-圆相交公式

两个圆：

$$
(x_1, y_1, r_1), \quad (x_2, y_2, r_2)
$$

圆心距离：

$$
d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
$$

如果 $|r_1 - r_2| \le d \le r_1 + r_2$，则它们相交于两点：

$$
a = \frac{r_1^2 - r_2^2 + d^2}{2d}
$$

$$
h = \sqrt{r_1^2 - a^2}
$$

然后交点坐标为：

$$
x_3 = x_1 + a \cdot \frac{x_2 - x_1}{d} \pm h \cdot \frac{y_2 - y_1}{d}
$$

$$
y_3 = y_1 + a \cdot \frac{y_2 - y_1}{d} \mp h \cdot \frac{x_2 - x_1}{d}
$$

每个交点成为扫描过程中的一个事件。

#### 示例（3 个圆）

圆：

* $C_1: (0,0,2)$
* $C_2: (3,0,2)$
* $C_3: (1.5,2,1.5)$

每对圆相交于 2 个点 → 最多 6 个交点。
该排列包含顶点（交点）、边（弧）和面（区域）。

扫描过程：

* $x = -2$: $C_1$ 开始
* $x = 1$: $C_2$ 开始
* $x = 1.5$: 交点事件
* $x = 3$: $C_3$ 开始
* $x = 5$: 圆结束

#### 微型代码（Python 草图）

```python
from math import sqrt

def circle_intersections(c1, c2):
    (x1, y1, r1), (x2, y2, r2) = c1, c2
    dx, dy = x2 - x1, y2 - y1
    d = sqrt(dx*dx + dy*dy)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return []
    a = (r1*r1 - r2*r2 + d*d) / (2*d)
    h = sqrt(r1*r1 - a*a)
    xm = x1 + a * dx / d
    ym = y1 + a * dy / d
    xs1 = xm + h * dy / d
    ys1 = ym - h * dx / d
    xs2 = xm - h * dy / d
    ys2 = ym + h * dx / d
    return [(xs1, ys1), (xs2, ys2)]

def circle_arrangement(circles):
    events = []
    for i, c1 in enumerate(circles):
        for j, c2 in enumerate(circles[i+1:], i+1):
            pts = circle_intersections(c1, c2)
            events.extend(pts)
    return sorted(events)

circles = [(0,0,2), (3,0,2), (1.5,2,1.5)]
print("交点:", circle_arrangement(circles))
```

这个简化版本枚举了交点，适用于事件调度。

#### 为什么它很重要？

* 是处理曲线对象几何排列的基础
* 用于运动规划、机器人学、蜂窝覆盖、CAD
* 迈向完整代数几何排列（圆锥曲线、椭圆）的一步

应用：

* 蜂窝网络规划（覆盖重叠）
* 机器人路径区域
* 维恩图和空间推理
* 圆弧上的图嵌入

#### 一个温和的证明（为什么它有效）

每个圆最多与其他圆有两个交点；
每个交点事件被处理一次；
最多 $O(n^2)$ 个交点，每个处理时间为 $O(\log n)$（树插入/移除）。

因此：

$$
O(n^2 \log n)
$$

正确性源于局部邻接性：在事件期间只有相邻的弧可以交换，因此所有交点都被捕获。

#### 自己动手试试

1.  绘制 3 个部分重叠的圆。
2.  计算两两之间的交点。
3.  标记点，按顺时针顺序连接弧。
4.  从最左到最右进行扫描。
5.  计算形成的面（区域）数量。

#### 测试用例

| 圆的情况       | 交点数量 | 面数量     |
| -------------- | -------- | --------- |
| 2 个重叠圆     | 2        | 3 个区域  |
| 3 个重叠圆     | 6        | 8 个区域  |
| 不相交的圆     | 0        | n 个区域  |

#### 复杂度

$$
\text{时间: } O(n^2 \log n), \quad \text{空间: } O(n^2)
$$

圆排列扫描算法将平滑的几何结构转化为离散的结构，通过耐心地扫描平面，追踪每一条弧、每一个交叉点和每一个面。
### 728 扫描重叠矩形

扫描重叠矩形算法用于检测一组轴对齐矩形之间的相交或碰撞。它是线扫描方法在二维碰撞检测、空间连接和布局引擎中实用而优雅的应用。

#### 我们要解决什么问题？

给定 $n$ 个轴对齐矩形

$$
R_i = [x_{1i}, x_{2i}] \times [y_{1i}, y_{2i}]
$$

我们想要找到所有重叠的矩形对 $(R_i, R_j)$，即满足

$$
 [x_{1i}, x_{2i}] \cap [x_{1j}, x_{2j}] \ne \emptyset
$$
且
$$
 [y_{1i}, y_{2i}] \cap [y_{1j}, y_{2j}] \ne \emptyset
$$

这是图形学、地理信息系统（GIS）和物理引擎中常见的子问题。

#### 朴素方法

检查每一对矩形：
$$
O(n^2)
$$

当 $n$ 很大时太慢。

我们将使用一条沿 $x$ 轴扫描的线，维护一个活跃矩形集合，这些矩形的 x 区间与当前位置重叠。

#### 它是如何工作的（通俗解释）

我们按 $x$ 递增的顺序处理事件：

*   开始事件：在 $x_{1i}$ 处，矩形进入活跃集合。
*   结束事件：在 $x_{2i}$ 处，矩形离开活跃集合。

在每次插入时，我们将新矩形与所有活跃矩形检查 y 方向是否重叠。

因为活跃矩形在 $x$ 方向都重叠，所以我们只需要测试 $y$ 区间。

#### 示例演练

矩形：

| ID | $x_1$ | $x_2$ | $y_1$ | $y_2$ |
| -- | ----- | ----- | ----- | ----- |
| R1 | 1     | 4     | 1     | 3     |
| R2 | 2     | 5     | 2     | 4     |
| R3 | 6     | 8     | 0     | 2     |

事件（按 $x$ 排序）：
$(1,\text{start},R1)$, $(2,\text{start},R2)$, $(4,\text{end},R1)$, $(5,\text{end},R2)$, $(6,\text{start},R3)$, $(8,\text{end},R3)$

扫描过程：

1.  $x=1$：添加 R1 → 活跃集合 = {R1}。
2.  $x=2$：添加 R2 → 检查与 R1 的重叠：

    *   $[1,3] \cap [2,4] = [2,3] \ne \emptyset$ → 发现重叠 (R1, R2)。
3.  $x=4$：移除 R1。
4.  $x=5$：移除 R2。
5.  $x=6$：添加 R3。
6.  $x=8$：移除 R3。

输出：重叠对 (R1, R2)。

#### 重叠条件

两个矩形 $R_i, R_j$ 重叠当且仅当

$$
x_{1i} < x_{2j} \ \text{且}\ x_{2i} > x_{1j}
$$

且

$$
y_{1i} < y_{2j} \ \text{且}\ y_{2i} > y_{1j}
$$

#### 微型代码（Python）

```python
def overlaps(r1, r2):
    # 检查两个矩形是否重叠
    return not (r1[1] <= r2[0] or r2[1] <= r1[0] or
                r1[3] <= r2[2] or r2[3] <= r1[2])

def sweep_rectangles(rects):
    events = []
    for i, (x1, x2, y1, y2) in enumerate(rects):
        events.append((x1, 'start', i))
        events.append((x2, 'end', i))
    events.sort()
    active = []
    result = []
    for x, typ, idx in events:
        if typ == 'start':
            for j in active:
                if overlaps(rects[idx], rects[j]):
                    result.append((idx, j))
            active.append(idx)
        else:
            active.remove(idx)
    return result

rects = [(1,4,1,3),(2,5,2,4),(6,8,0,2)]
print("重叠对:", sweep_rectangles(rects))
```

#### 微型代码（C 语言草图）

```c
typedef struct { double x1, x2, y1, y2; } Rect;
int overlaps(Rect a, Rect b) {
    // 检查两个矩形是否重叠
    return !(a.x2 <= b.x1 || b.x2 <= a.x1 ||
             a.y2 <= b.y1 || b.y2 <= a.y1);
}
```

使用事件数组，按 $x$ 排序，维护活跃列表。

#### 为什么它很重要

*   是宽相位碰撞检测的核心思想
*   用于 2D 游戏、UI 布局引擎、空间连接
*   通过多轴扫描可轻松扩展到 3D 包围盒相交检测

应用：

*   物理模拟（包围盒重叠）
*   空间查询系统（R 树验证）
*   CAD 布局约束检查

#### 一个温和的证明（为什么它有效）

*   活跃集合恰好包含与当前 $x$ 重叠的矩形。
*   通过仅检查这些矩形，我们一次性覆盖了所有可能的重叠。
*   每次插入/移除：$O(\log n)$（使用平衡树）。
*   每对矩形仅在 $x$ 区间重叠时被测试。

总时间：

$$
O((n + k) \log n)
$$

其中 $k$ 是重叠对的数量。

#### 自己动手试试

1.  在网格上绘制重叠的矩形。
2.  按 $x$ 对边进行排序。
3.  扫描并维护活跃列表。
4.  在每次插入时，测试与活跃矩形的 $y$ 重叠。
5.  记录重叠，并可视化验证。

#### 测试用例

| 矩形                     | 重叠对          |
| ------------------------ | --------------- |
| R1(1,4,1,3), R2(2,5,2,4) | (R1,R2)         |
| 不相交的矩形             | 无              |
| 嵌套矩形                 | 全部重叠        |

#### 复杂度

$$
\text{时间: } O((n + k)\log n), \quad \text{空间: } O(n)
$$

扫描重叠矩形算法就像一个几何哨兵，滑过平面，跟踪活跃的形状，并精确地发现碰撞。
### 729 区间计数

区间计数（Range Counting）提出的问题是：给定平面上的许多点，有多少点位于一个与坐标轴对齐的查询矩形内。它是几何数据查询的基础，为交互式图表、地图和数据库索引提供支持。

#### 我们要解决什么问题？

输入：一个包含 $n$ 个点的静态集合 $P = {(x_i,y_i)}_{i=1}^n$。
查询：对于矩形 $R = [x_L, x_R] \times [y_B, y_T]$，返回

$$
\#\{(x,y) \in P \mid x_L \le x \le x_R,\; y_B \le y \le y_T\}.
$$

我们希望在一次性的预处理步骤之后，获得快速的查询时间，最好是亚线性的。

#### 它是如何工作的（通俗解释）

有几种经典的数据结构支持正交区间计数。

1.  **按 x 排序 + 基于 y 的 Fenwick 树（离线或扫描线法）**：
    将点按 $x$ 排序。将查询按 $x_R$ 排序。沿 $x$ 方向扫描，将点添加到以其压缩后的 $y$ 值为键的 Fenwick 树中。
    对于区间 $[x_L,x_R]\times[y_B,y_T]$ 的计数等于：
    $$
    \text{count}(x \le x_R, y \in [y_B,y_T]) - \text{count}(x < x_L, y \in [y_B,y_T]).
    $$
    时间复杂度：离线 $O((n + q)\log n)$。

2.  **区间树（静态，在线）**：
    在 $x$ 上构建一棵平衡二叉搜索树。每个节点存储其子树中 $y$ 值的排序列表。
    一个二维查询将 $x$ 区间分解为 $O(\log n)$ 个规范节点，在每个节点中，我们对 $y$ 列表进行二分查找，以统计有多少个 $y$ 值落在 $[y_B,y_T]$ 内。
    时间复杂度：查询 $O(\log^2 n)$，空间 $O(n \log n)$。
    使用分数级联（fractional cascading），查询可以优化到 $O(\log n)$。

3.  **Fenwick 树的 Fenwick 树 或 线段树的 Fenwick 树**：
    使用 Fenwick 树按 $x$ 索引。每个 Fenwick 节点存储另一个基于 $y$ 的 Fenwick 树。
    在坐标压缩后，支持完全在线的更新和查询，时间复杂度 $O(\log^2 n)$，空间复杂度 $O(n \log n)$。

#### 示例演练

点：$(1,1), (2,3), (3,2), (5,4), (6,1)$
查询：$R = [2,5] \times [2,4]$

内部的点：$(2,3), (3,2), (5,4)$
答案：$3$。

#### 微型代码 1：使用 Fenwick 树的离线扫描线法（Python）

```python
# 离线正交区间计数：
# 对于每个查询 [xL,xR]x[yB,yT]，计算 F(xR, yB..yT) - F(xL-ε, yB..yT)

from bisect import bisect_left, bisect_right

class Fenwick:
    def __init__(self, n):
        self.n = n
        self.fw = [0]*(n+1)
    def add(self, i, v=1):
        while i <= self.n:
            self.fw[i] += v
            i += i & -i
    def sum(self, i):
        s = 0
        while i > 0:
            s += self.fw[i]
            i -= i & -i
        return s
    def range_sum(self, l, r):
        if r < l: return 0
        return self.sum(r) - self.sum(l-1)

def offline_range_count(points, queries):
    # points: 点列表，格式为 (x,y)
    # queries: 查询列表，格式为 (xL,xR,yB,yT)
    ys = sorted({y for _,y in points} | {q[2] for q in queries} | {q[3] for q in queries})
    def y_id(y): return bisect_left(ys, y) + 1

    # 准备事件：在达到某个 x 时添加点，然后回答在该 x 处结束的查询
    events = []
    for i,(x,y) in enumerate(points):
        events.append((x, 0, i))  # 点事件
    Fq = []  # 在 xR 处的查询
    Gq = []  # 在 xL-1 处的查询
    for qi,(xL,xR,yB,yT) in enumerate(queries):
        Fq.append((xR, 1, qi))
        Gq.append((xL-1, 2, qi))
    events += Fq + Gq
    events.sort()

    fw = Fenwick(len(ys))
    ansR = [0]*len(queries)
    ansL = [0]*len(queries)

    for x,typ,idx in events:
        if typ == 0:
            _,y = points[idx]
            fw.add(y_id(y), 1)
        elif typ == 1:
            xL,xR,yB,yT = queries[idx]
            l = bisect_left(ys, yB) + 1
            r = bisect_right(ys, yT)
            ansR[idx] = fw.range_sum(l, r)
        else:
            xL,xR,yB,yT = queries[idx]
            l = bisect_left(ys, yB) + 1
            r = bisect_right(ys, yT)
            ansL[idx] = fw.range_sum(l, r)

    return [ansR[i] - ansL[i] for i in range(len(queries))]

# 演示
points = [(1,1),(2,3),(3,2),(5,4),(6,1)]
queries = [(2,5,2,4), (1,6,1,1)]
print(offline_range_count(points, queries))  # [3, 2]
```

#### 微型代码 2：静态区间树查询思路（Python，概念性）

```python
# 构建：按 x 对点排序，递归分割；
# 每个节点存储其 y 排序列表用于二分计数。

from bisect import bisect_left, bisect_right

class RangeTree:
    def __init__(self, pts):
        # pts 已按 x 排序
        self.xs = [p[0] for p in pts]
        self.ys = sorted(p[1] for p in pts)
        self.left = self.right = None
        if len(pts) > 1:
            mid = len(pts)//2
            self.left = RangeTree(pts[:mid])
            self.right = RangeTree(pts[mid:])

    def count_y(self, yB, yT):
        L = bisect_left(self.ys, yB)
        R = bisect_right(self.ys, yT)
        return R - L

    def query(self, xL, xR, yB, yT):
        # 统计 x 在 [xL,xR] 且 y 在 [yB,yT] 内的点数
        if xR < self.xs[0] or xL > self.xs[-1]:
            return 0
        if xL <= self.xs[0] and self.xs[-1] <= xR:
            return self.count_y(yB, yT)
        if not self.left:  # 叶子节点
            return int(xL <= self.xs[0] <= xR and yB <= self.ys[0] <= yT)
        return self.left.query(xL,xR,yB,yT) + self.right.query(xL,xR,yB,yT)

pts = sorted([(1,1),(2,3),(3,2),(5,4),(6,1)])
rt = RangeTree(pts)
print(rt.query(2,5,2,4))  # 3
```

#### 为什么它很重要

*   是空间数据库和分析仪表板的核心原语
*   是热力图、密度查询和窗口聚合的基础
*   可以通过 $k$-d 树和区间树扩展到更高维度

应用场景：
地图视口、时间窗口计数、地理信息系统过滤、交互式刷选与链接。

#### 一个温和的证明（为什么它有效）

对于区间树：$x$ 区间 $[x_L,x_R]$ 被分解为一棵平衡二叉搜索树的 $O(\log n)$ 个规范节点。
每个规范节点按排序顺序存储其子树的 $y$ 值。
在一个节点上对 $[y_B,y_T]$ 进行计数，通过二分查找需要 $O(\log n)$ 时间。
对 $O(\log n)$ 个节点的结果求和，得到每次查询 $O(\log^2 n)$ 的时间复杂度。
使用分数级联，第二层的搜索可以复用指针，因此所有计数都可以在 $O(\log n)$ 时间内找到。

#### 动手试试

1.  使用 Fenwick 树和坐标压缩实现离线计数。
2.  与每次查询 $O(n)$ 的朴素方法进行比较验证。
3.  构建一个区间树，并对不同的 $n$ 值计时 $q$ 次查询。
4.  添加更新功能：切换到 Fenwick 树的 Fenwick 树以支持动态点。
5.  扩展到 3D，使用树的树来处理正交长方体。

#### 测试用例

| 点集                              | 查询矩形               | 期望结果 |
| --------------------------------- | ---------------------- | -------- |
| $(1,1),(2,3),(3,2),(5,4),(6,1)$ | $[2,5]\times[2,4]$    | 3        |
| 同上                              | $[1,6]\times[1,1]$    | 2        |
| $(0,0),(10,10)$                   | $[1,9]\times[1,9]$    | 0        |
| 网格 $3\times 3$                  | 中心 $[1,2]\times[1,2]$ | 4        |

#### 复杂度

*   使用 Fenwick 树的离线扫描线法：预处理加查询 $O((n+q)\log n)$
*   区间树：构建 $O(n \log n)$，查询 $O(\log^2 n)$ 或使用分数级联 $O(\log n)$
*   线段树或 Fenwick 树的 Fenwick 树：动态更新和查询 $O(\log^2 n)$

区间计数通过分层搜索树和排序的辅助列表，将空间选择转化为对数时间的查询。
### 729 范围计数

范围计数（Range Counting）要解决的问题是：给定平面上的许多点，有多少个点位于一个轴对齐的查询矩形内。它是几何数据查询的基础，为交互式图表、地图和数据库索引提供支持。

#### 我们要解决什么问题？

输入：一个包含 $n$ 个点的静态集合 $P = {(x_i,y_i)}_{i=1}^n$。
查询：对于矩形 $R = [x_L, x_R] \times [y_B, y_T]$，返回

$$
\#\{(x,y) \in P \mid x_L \le x \le x_R,\; y_B \le y \le y_T\}.
$$

我们希望在一次性预处理步骤之后，获得快速的查询时间，理想情况下是亚线性的。

#### 它是如何工作的（通俗解释）

有几种经典的数据结构支持正交范围计数。

1.  **按 x 排序 + 基于 y 的 Fenwick 树（离线或扫描法）**：
    将点按 $x$ 排序。将查询按 $x_R$ 排序。沿 $x$ 轴扫描，将点添加到以其压缩后的 $y$ 值为键的 Fenwick 树中。
    对于查询 $[x_L,x_R]\times[y_B,y_T]$，其计数等于：
    $$
    \text{count}(x \le x_R, y \in [y_B,y_T]) - \text{count}(x < x_L, y \in [y_B,y_T]).
    $$
    时间复杂度：离线 $O((n + q)\log n)$。

2.  **范围树（静态，在线）**：
    在 $x$ 上构建一棵平衡二叉搜索树（BST）。每个节点存储其子树中 $y$ 值的排序列表。
    一个二维查询将 $x$ 范围分解为 $O(\log n)$ 个规范节点，在每个节点中，我们对 $y$ 列表进行二分查找，以统计有多少个 $y$ 值落在 $[y_B,y_T]$ 内。
    时间复杂度：查询 $O(\log^2 n)$，空间 $O(n \log n)$。
    使用分数级联（fractional cascading）后，查询可优化至 $O(\log n)$。

3.  **Fenwick 树的 Fenwick 树 或 线段树的 Fenwick 树**：
    使用 Fenwick 树按 $x$ 索引。每个 Fenwick 节点再存储一个基于 $y$ 的 Fenwick 树。
    在坐标压缩后，支持完全在线的更新和查询，时间复杂度 $O(\log^2 n)$，空间复杂度 $O(n \log n)$。

#### 示例演练

点：$(1,1), (2,3), (3,2), (5,4), (6,1)$
查询：$R = [2,5] \times [2,4]$

内部的点：$(2,3), (3,2), (5,4)$
答案：$3$。

#### 微型代码 1：使用 Fenwick 树的离线扫描法（Python）

```python
# 离线正交范围计数：
# 对于每个查询 [xL,xR]x[yB,yT]，计算 F(xR, yB..yT) - F(xL-ε, yB..yT)

from bisect import bisect_left, bisect_right

class Fenwick:
    def __init__(self, n):
        self.n = n
        self.fw = [0]*(n+1)
    def add(self, i, v=1):
        while i <= self.n:
            self.fw[i] += v
            i += i & -i
    def sum(self, i):
        s = 0
        while i > 0:
            s += self.fw[i]
            i -= i & -i
        return s
    def range_sum(self, l, r):
        if r < l: return 0
        return self.sum(r) - self.sum(l-1)

def offline_range_count(points, queries):
    # points: 点列表，格式为 (x,y)
    # queries: 查询列表，格式为 (xL,xR,yB,yT)
    ys = sorted({y for _,y in points} | {q[2] for q in queries} | {q[3] for q in queries})
    def y_id(y): return bisect_left(ys, y) + 1

    # 准备事件：在 x 处添加点，然后回答在该 x 处结束的查询
    events = []
    for i,(x,y) in enumerate(points):
        events.append((x, 0, i))  # 点事件
    Fq = []  # 在 xR 处的查询
    Gq = []  # 在 xL-1 处的查询
    for qi,(xL,xR,yB,yT) in enumerate(queries):
        Fq.append((xR, 1, qi))
        Gq.append((xL-1, 2, qi))
    events += Fq + Gq
    events.sort()

    fw = Fenwick(len(ys))
    ansR = [0]*len(queries)
    ansL = [0]*len(queries)

    for x,typ,idx in events:
        if typ == 0:
            _,y = points[idx]
            fw.add(y_id(y), 1)
        elif typ == 1:
            xL,xR,yB,yT = queries[idx]
            l = bisect_left(ys, yB) + 1
            r = bisect_right(ys, yT)
            ansR[idx] = fw.range_sum(l, r)
        else:
            xL,xR,yB,yT = queries[idx]
            l = bisect_left(ys, yB) + 1
            r = bisect_right(ys, yT)
            ansL[idx] = fw.range_sum(l, r)

    return [ansR[i] - ansL[i] for i in range(len(queries))]

# 演示
points = [(1,1),(2,3),(3,2),(5,4),(6,1)]
queries = [(2,5,2,4), (1,6,1,1)]
print(offline_range_count(points, queries))  # [3, 2]
```

#### 微型代码 2：静态范围树查询思路（Python，概念性）

```python
# 构建：按 x 对点排序，递归分割；
# 每个节点存储其 y 排序列表用于二分计数。

from bisect import bisect_left, bisect_right

class RangeTree:
    def __init__(self, pts):
        # pts 已按 x 排序
        self.xs = [p[0] for p in pts]
        self.ys = sorted(p[1] for p in pts)
        self.left = self.right = None
        if len(pts) > 1:
            mid = len(pts)//2
            self.left = RangeTree(pts[:mid])
            self.right = RangeTree(pts[mid:])

    def count_y(self, yB, yT):
        L = bisect_left(self.ys, yB)
        R = bisect_right(self.ys, yT)
        return R - L

    def query(self, xL, xR, yB, yT):
        # 统计 x 在 [xL,xR] 且 y 在 [yB,yT] 内的点的数量
        if xR < self.xs[0] or xL > self.xs[-1]:
            return 0
        if xL <= self.xs[0] and self.xs[-1] <= xR:
            return self.count_y(yB, yT)
        if not self.left:  # 叶节点
            return int(xL <= self.xs[0] <= xR and yB <= self.ys[0] <= yT)
        return self.left.query(xL,xR,yB,yT) + self.right.query(xL,xR,yB,yT)

pts = sorted([(1,1),(2,3),(3,2),(5,4),(6,1)])
rt = RangeTree(pts)
print(rt.query(2,5,2,4))  # 3
```

#### 为什么它很重要

*   是空间数据库和分析仪表板的核心原语
*   是热力图、密度查询和窗口聚合的基础
*   可通过 $k$-d 树和范围树扩展到更高维度

应用场景：
地图视口、时间窗口计数、地理信息系统（GIS）过滤、交互式刷选和链接。

#### 一个温和的证明（为什么它有效）

对于范围树：$x$ 范围 $[x_L,x_R]$ 被分解为一棵平衡二叉搜索树（BST）的 $O(\log n)$ 个规范节点。
每个规范节点将其子树中的 $y$ 值按顺序存储。
通过二分查找，在一个节点中统计 $[y_B,y_T]$ 内的数量成本为 $O(\log n)$。
对 $O(\log n)$ 个节点的结果求和，得到每次查询 $O(\log^2 n)$ 的时间复杂度。
使用分数级联后，第二级的查找可以重用指针，从而所有计数都可以在 $O(\log n)$ 时间内找到。

#### 自己动手试试

1.  使用 Fenwick 树和坐标压缩实现离线计数。
2.  与每次查询 $O(n)$ 的朴素方法进行比较验证。
3.  构建一个范围树，并对不同的 $n$ 值计时 $q$ 次查询。
4.  添加更新功能：切换到 Fenwick 树的 Fenwick 树以支持动态点。
5.  使用树套树扩展到 3D 正交长方体查询。

#### 测试用例

| 点集                              | 查询矩形               | 预期结果 |
| --------------------------------- | ---------------------- | -------- |
| $(1,1),(2,3),(3,2),(5,4),(6,1)$ | $[2,5]\times[2,4]$    | 3        |
| 同上                              | $[1,6]\times[1,1]$    | 2        |
| $(0,0),(10,10)$                   | $[1,9]\times[1,9]$    | 0        |
| $3\times 3$ 网格                  | 中心 $[1,2]\times[1,2]$ | 4        |

#### 复杂度

*   使用 Fenwick 树的离线扫描法：预处理加查询 $O((n+q)\log n)$
*   范围树：构建 $O(n \log n)$，查询 $O(\log^2 n)$ 或使用分数级联 $O(\log n)$
*   线段树或 Fenwick 树的 Fenwick 树：动态更新和查询 $O(\log^2 n)$

范围计数通过分层搜索树和排序辅助列表，将空间选择转化为对数时间的查询。
### 730 三角形的平面扫描

三角形平面扫描算法用于计算平面上一组三角形之间的交点、重叠区域或排列关系。它将基于直线和线段的扫描扩展到多边形元素，将边和面都作为事件进行管理。

#### 我们要解决什么问题？

给定 $n$ 个三角形
$$
T_i = {(x_{i1}, y_{i1}), (x_{i2}, y_{i2}), (x_{i3}, y_{i3})}
$$
我们希望计算：

* 三角形边之间的所有交点
* 重叠区域（并集面积或相交多边形）
* 覆盖分解：由三角形边界诱导的完整平面细分

此类扫描在网格覆盖、计算几何内核和计算机图形学中至关重要。

#### 朴素方法

比较所有三角形对 $(T_i, T_j)$ 及其 9 条边对。
时间复杂度：
$$
O(n^2)
$$
对于大型网格或空间数据来说过于昂贵。

我们通过基于边和事件的平面扫描来改进。

#### 它是如何工作的（通俗解释）

一个三角形由 3 条线段组成。
我们将每个三角形的边视为线段事件，并使用线段扫描线进行处理：

1. 将所有三角形的边转换为线段列表。
2. 按 $x$ 坐标对所有线段的端点进行排序。
3. 扫描线从左向右移动。
4. 维护一个与扫描线相交的活动边集合。
5. 当两条边相交时，记录交点，并在需要时细分几何形状。

如果要计算覆盖，交点会将三角形细分为平面面片。

#### 示例演练

三角形：

* $T_1$: $(1,1)$, $(4,1)$, $(2,3)$
* $T_2$: $(2,0)$, $(5,2)$, $(3,4)$

1. 提取边：

   * $T_1$: $(1,1)-(4,1)$, $(4,1)-(2,3)$, $(2,3)-(1,1)$
   * $T_2$: $(2,0)-(5,2)$, $(5,2)-(3,4)$, $(3,4)-(2,0)$

2. 收集所有端点，按 $x$ 排序：
   $x = 1, 2, 3, 4, 5$

3. 扫描：

   * $x=1$: 添加 $T_1$ 的边
   * $x=2$: 添加 $T_2$ 的边；检查与当前活动集合的交点
   * 找到 $T_1$ 的斜边与 $T_2$ 的底边的交点
   * 记录交点
   * 如果需要覆盖，则更新几何形状

输出：交点、重叠区域多边形。

#### 几何谓词

对于边 $(A,B)$ 和 $(C,D)$：
使用方向测试检查交点：

$$
\text{orient}(A,B,C) \ne \text{orient}(A,B,D)
$$
和
$$
\text{orient}(C,D,A) \ne \text{orient}(C,D,B)
$$

交点会细分边并更新事件队列。

#### 微型代码（Python 草图）

```python
def orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def intersect(a,b,c,d):
    o1 = orient(a,b,c)
    o2 = orient(a,b,d)
    o3 = orient(c,d,a)
    o4 = orient(c,d,b)
    return o1*o2 < 0 and o3*o4 < 0

def sweep_triangles(triangles):
    segments = []
    for tri in triangles:
        for i in range(3):
            a, b = tri[i], tri[(i+1)%3]
            if a[0] > b[0]:
                a, b = b, a
            segments.append((a,b))
    events = []
    for s in segments:
        events.append((s[0][0],'start',s))
        events.append((s[1][0],'end',s))
    events.sort()
    active = []
    intersections = []
    for x,typ,seg in events:
        if typ == 'start':
            for s in active:
                if intersect(seg[0],seg[1],s[0],s[1]):
                    intersections.append(x)
            active.append(seg)
        else:
            active.remove(seg)
    return intersections

triangles = [[(1,1),(4,1),(2,3)],[(2,0),(5,2),(3,4)]]
print("交点:", sweep_triangles(triangles))
```

这个基本形式可以扩展为计算实际的交点坐标和多边形。

#### 为什么它很重要？

* 是网格覆盖、多边形并集、相交面积计算的基础
* 用于有限元网格划分、地图覆盖、几何引擎
* 将线段扫描推广到多边形输入

应用：

* CAD/CAE 分析
* GIS 覆盖操作
* 三角化地图相交
* 渲染和遮挡检测

#### 一个温和的证明（为什么它有效）

每个三角形贡献三条边，总共 $3n$ 条边。
当两条边相交时发生每个交点事件。
Bentley–Ottmann 框架通过活动集合中的局部邻接性确保每个交点被检测一次。

总复杂度：

$$
O((n + k)\log n)
$$

其中 $k$ 是边之间的交点数量。

#### 自己动手试试

1.  画出两个部分重叠的三角形。
2.  提取边，按 $x$ 坐标对端点排序。
3.  扫描，跟踪活动边。
4.  标记每个交点。
5.  与所有边对暴力求交进行比较。

#### 测试用例

| 三角形                 | 交点数量 | 描述                 |
| --------------------- | ------------- | --------------------------- |
| 不相交                | 0             | 不重叠             |
| 部分重叠              | >0            | 边相交              |
| 嵌套                | 0             | 一个三角形在另一个内部 |
| 相交边                | 2             | 边界相交     |

#### 复杂度

$$
\text{时间: } O((n + k)\log n), \quad \text{空间: } O(n + k)
$$

三角形的平面扫描将多边形边穿过扫描线，精确追踪每个交叉点，为多边形覆盖和网格操作奠定了基础。

# 第 74 节 Delaunay 和 Voronoi 图
### 731 Delaunay 三角剖分（增量法）

Delaunay 三角剖分是计算几何中的一个基础结构。
给定平面上一组点，它将它们连接成三角形，使得没有任何点位于任何三角形的外接圆内部。
增量算法逐步构建这种三角剖分，每次插入一个点，并在局部恢复 Delaunay 条件。

#### 我们要解决什么问题？

给定平面上的 $n$ 个点 $P = {p_1, p_2, \ldots, p_n}$，
构造一个三角剖分 $T$，使得对于 $T$ 中的每个三角形 $\triangle abc$：

$$
\text{没有其他点 } p \in P \text{ 位于 } \triangle abc \text{ 的外接圆内部。}
$$

这个性质会产生形状良好的三角形并最大化最小角，
使得 Delaunay 三角剖分成为网格生成、插值和图形处理的理想选择。

#### 核心思想

从一个包含所有点的超级三角形开始。
逐个插入点，每次插入后，更新局部连接以保持空圆性质。

#### 它是如何工作的（通俗解释）

1. 初始化：
   创建一个包围所有输入点的大三角形。

2. 插入每个点 $p_i$：

   * 找到包含 $p_i$ 的三角形。
   * 将其分割成连接 $p_i$ 与其顶点的子三角形。

3. 合法化边：

   * 对于每条新边，检查 Delaunay 条件。
   * 如果违反条件（邻接三角形的对顶点位于外接圆内），则翻转该边。

4. 重复直到所有点都被插入。

5. 移除与超级三角形顶点相邻的三角形。

#### 示例演练

点集：
$P = {A(0,0), B(2,0), C(1,2), D(1,1)}$

1. 超级三角形覆盖所有点。
2. 插入 $A, B, C$ → 初始三角形 $\triangle ABC$。
3. 插入 $D(1,1)$ → 分割为 $\triangle ABD$, $\triangle BCD$, $\triangle CAD$。
4. 检查每条边是否违反外接圆条件。
5. 如果需要，翻转边。

输出：满足空圆条件的三角剖分。

#### Delaunay 条件（空圆测试）

对于顶点为 $a,b,c$ 的三角形和点 $p$，
如果行列式为正，则 $p$ 位于外接圆内：

$$
\begin{vmatrix}
a_x & a_y & a_x^2 + a_y^2 & 1 \\
b_x & b_y & b_x^2 + b_y^2 & 1 \\
c_x & c_y & c_x^2 + c_y^2 & 1 \\
p_x & p_y & p_x^2 + p_y^2 & 1
\end{vmatrix} > 0
$$

如果为真，则翻转 $p$ 的对边以恢复 Delaunay 性质。

#### 微型代码（Python 草图）

```python
import math

def circumcircle_contains(a, b, c, p):
    ax, ay = a
    bx, by = b
    cx, cy = c
    px, py = p
    mat = [
        [ax - px, ay - py, (ax - px)2 + (ay - py)2],
        [bx - px, by - py, (bx - px)2 + (by - py)2],
        [cx - px, cy - py, (cx - px)2 + (cy - py)2],
    ]
    det = (
        mat[0][0] * (mat[1][1]*mat[2][2] - mat[2][1]*mat[1][2])
        - mat[1][0] * (mat[0][1]*mat[2][2] - mat[2][1]*mat[0][2])
        + mat[2][0] * (mat[0][1]*mat[1][2] - mat[1][1]*mat[0][2])
    )
    return det > 0

def incremental_delaunay(points):
    # 占位符：实际实现会使用边翻转结构
    # 这里我们以伪代码形式返回三角形列表
    return [("triangulation", points)]
```

这段伪代码展示了外接圆测试，这是合法化步骤的核心。
完整的实现需要维护边邻接关系和三角形翻转。

#### 为什么它很重要

* 产生高质量的网格（没有狭长的三角形）
* 用于地形建模、网格细化、有限元方法
* 构成 Voronoi 图的基础（其对偶图）

应用：

* 3D 建模和渲染
* 科学计算和仿真
* GIS 插值（TIN 模型）
* 计算几何工具包（CGAL, Shapely）

#### 一个温和的证明（为什么它有效）

增量算法在每一步都保持 Delaunay 性质：

* 最初，超级三角形显然满足该性质。
* 每次插入都会细分现有的三角形。
* 边翻转恢复了局部最优性。

因为每次插入都保持了空圆条件，
所以最终的三角剖分是全局 Delaunay 的。

时间复杂度取决于插入顺序和点分布：

$$
O(n^2) \text{ 最坏情况}, \quad O(n \log n) \text{ 平均情况。}
$$

#### 自己动手试试

1.  画三个点，构成一个三角形。
2.  在内部添加第四个点，连接到所有顶点。
3.  检查每条边的外接圆测试。
4.  翻转任何违反条件的边。
5.  对更多点重复此过程。

观察三角剖分如何调整以保持 Delaunay 性质。

#### 测试用例

| 点集                     | 三角剖分               |
| ------------------------ | ---------------------- |
| (0,0),(2,0),(1,2)        | 单个三角形             |
| + (1,1)                  | 3 个三角形，Delaunay   |
| 随机 10 个点             | 有效的三角剖分         |

#### 复杂度

$$
\text{时间: } O(n \log n) \text{ (平均)}, \quad O(n^2) \text{ (最坏)}
$$
$$
\text{空间: } O(n)
$$

增量 Delaunay 三角剖分就像雕塑家一样构建几何图形，逐点进行，翻转边，直到每个三角形都符合空圆的和谐。
### 732 Delaunay（分治法）

分治法 Delaunay 三角剖分算法通过递归地划分点集、三角剖分子问题并以几何精度合并它们来构建 Delaunay 三角剖分。
这是最优雅且高效的方法之一，在保证空圆性质的同时，实现了
$$O(n \log n)$$
的时间复杂度。

#### 我们要解决什么问题？

给定平面上的 $n$ 个点 $P = {p_1, p_2, \ldots, p_n}$，
找到一个三角剖分，使得对于每个三角形 $\triangle abc$：

$$
\text{没有其他点 } p \in P \text{ 位于 } \triangle abc \text{ 的外接圆内}
$$

我们寻求一个全局的 Delaunay 结构，它由局部解递归构建而成。

#### 它是如何工作的（通俗解释）

分治法的思路类似于归并排序：

1. 按 $x$ 坐标对点进行排序。
2. 将点集分成两半 $P_L$ 和 $P_R$。
3. 递归地对每一半进行三角剖分，得到 $T_L$ 和 $T_R$。
4. 合并两个三角剖分：

   * 找到连接 $T_L$ 和 $T_R$ 的下公共切线。
   * 然后向上"拉链式"合并，添加新的 Delaunay 边，直到到达上切线。
   * 在合并过程中，移除违反空圆条件的边。

合并后，$T = T_L \cup T_R$ 就是完整的 Delaunay 三角剖分。

#### 关键的几何步骤：合并

要合并两个 Delaunay 三角剖分：

1. 找到连接最低可见点的基边（下切线）。
2. 迭代地添加连接点的边，这些边形成有效的 Delaunay 三角形。
3. 如果边违反外接圆条件，则进行边翻转。
4. 继续向上，直到到达上切线。

这种"拉链式"合并创建了一个无缝的、全局有效的三角剖分。

#### 示例演练

点集：$P = {(0,0), (2,0), (1,2), (4,0), (5,2)}$

1. 按 $x$ 排序：$(0,0), (1,2), (2,0), (4,0), (5,2)$
2. 划分：左半部分 $(0,0),(1,2),(2,0)$，右半部分 $(4,0),(5,2)$
3. 对每一半进行三角剖分：

   * $T_L$：$\triangle (0,0),(1,2),(2,0)$
   * $T_R$：$\triangle (4,0),(5,2)$
4. 合并：

   * 找到下切线 $(2,0)-(4,0)$
   * 添加连接边，用空圆条件测试
   * 最终三角剖分：所有五个点的 Delaunay 三角剖分

#### Delaunay 测试（空圆检查）

对于每条连接左右两侧的候选边 $(a,b)$，
测试添加第三个顶点 $c$ 后是否保持 Delaunay 性质：

$$
\begin{vmatrix}
a_x & a_y & a_x^2 + a_y^2 & 1 \\
b_x & b_y & b_x^2 + b_y^2 & 1 \\
c_x & c_y & c_x^2 + c_y^2 & 1 \\
p_x & p_y & p_x^2 + p_y^2 & 1
\end{vmatrix} \le 0
$$

如果违反条件（行列式为正），则移除或翻转该边。

#### 微型代码（Python 草图）

```python
def delaunay_divide(points):
    points = sorted(points)
    if len(points) <= 3:
        # 基本情况：直接三角剖分
        return [tuple(points)]
    mid = len(points)//2
    left = delaunay_divide(points[:mid])
    right = delaunay_divide(points[mid:])
    return merge_delaunay(left, right)

def merge_delaunay(left, right):
    # 占位合并函数；实际版本会查找切线并翻转边
    return left + right
```

这个框架展示了递归结构；实际实现会维护邻接关系、计算切线并应用空圆检查。

#### 为什么它很重要

* 最优时间复杂度 $O(n \log n)$
* 优雅的分治范式
* 是 Fortune 扫描线算法和高级三角剖分器的基础
* 适用于静态点集、地形网格、GIS 模型

应用：

* 地形建模（TIN 生成）
* 科学模拟（有限元网格）
* Voronoi 图构建（通过对偶图）
* 计算几何库（CGAL, Triangle）

#### 一个温和的证明（为什么它有效）

1. 基本情况：小点集显然是 Delaunay 的。
2. 归纳步骤：合并保持了 Delaunay 性质，因为：

   * 合并过程中创建的所有边都满足局部空圆测试。
   * 合并只连接彼此可见的边界顶点。

因此，通过归纳，最终的三角剖分是 Delaunay 的。

每个合并步骤需要线性时间，并且有 $\log n$ 层：

$$
T(n) = 2T(n/2) + O(n) = O(n \log n)
$$

#### 自己动手试试

1. 绘制 6–8 个按 $x$ 排序的点。
2. 分成两半，分别对每一半进行三角剖分。
3. 画出下切线，连接可见的顶点。
4. 翻转任何违反空圆性质的边。
5. 验证最终的三角剖分满足 Delaunay 规则。

#### 测试用例

| 点集             | 三角剖分类型           |
| ---------------- | ---------------------- |
| 3 个点           | 单个三角形             |
| 5 个点           | 合并后的三角形         |
| 随机 10 个点     | 有效的 Delaunay 三角剖分 |

#### 复杂度

$$
\text{时间： } O(n \log n), \quad \text{空间： } O(n)
$$

分治法 Delaunay 算法通过平衡构建和谐，分割平面，局部求解，并全局合并成一个完美的空圆马赛克。
### 733 Delaunay（Fortune 扫描线算法）

Fortune 扫描线算法是一种巧妙的平面扫描方法，用于在
$$
O(n \log n)
$$
时间内构建 Delaunay 三角剖分及其对偶图——Voronoi 图。
它优雅地在平面上滑动一条扫描线（或抛物线），维护一个称为海滩线的动态结构，以追踪 Voronoi 边的演变过程，并由此推导出 Delaunay 边。

#### 我们要解决什么问题？

给定 $n$ 个点 $P = {p_1, p_2, \ldots, p_n}$（称为*站点*），构建它们的 Delaunay 三角剖分，即一组三角形，使得没有任何点位于任何三角形的外接圆内。

其对偶图，即 Voronoi 图，将平面划分为多个单元，每个单元对应一个点，包含所有到该点距离比到其他任何点都近的位置。

Fortune 算法能够同时高效地构建这两种结构。

#### 核心洞见

当扫描线向下移动时，每个站点的影响前沿形成一个抛物线弧。
海滩线是所有活动弧的并集。
Voronoi 边出现在弧相交的地方；Delaunay 边则连接那些其弧共享边界的站点。

算法处理两种类型的事件：

1.  **站点事件**：当扫描线触及一个新站点时
2.  **圆事件**：当海滩线重塑导致弧消失时（三个弧在一个圆上相遇）

#### 它是如何工作的（通俗解释）

1.  将所有站点按 $y$ 坐标排序（从上到下）。
2.  向下扫描一条水平线：
    *   在每个站点事件处，向海滩线插入一个新的抛物线弧。
    *   更新交点以创建 Voronoi/Delaunay 边。
3.  在每个圆事件处，移除消失的弧（当三个弧在 Voronoi 图的一个顶点相遇时）。
4.  维护：
    *   事件队列：即将到来的站点/圆事件
    *   海滩线：弧的平衡树
    *   输出边：Voronoi 边 / Delaunay 边（对偶）
5.  继续处理，直到所有事件都被处理完毕。
6.  在边界框处闭合所有剩余的开放边。

通过连接共享 Voronoi 边的站点来恢复 Delaunay 三角剖分。

#### 示例演练

点：

*   $A(2,6), B(5,5), C(3,3)$

1.  按 $y$ 排序：$A, B, C$
2.  向下扫描：
    *   站点 $A$：创建新弧
    *   站点 $B$：新弧分割现有弧，新的断点 → Voronoi 边开始
    *   站点 $C$：另一次分割，更多断点
3.  圆事件：弧合并 → Voronoi 顶点，记录 Delaunay 三角形
4.  输出：三个 Voronoi 单元，连接 $A, B, C$ 的 Delaunay 三角形

#### 数据结构

| 结构                         | 用途                                 |
| ---------------------------- | ------------------------------------ |
| 事件队列（优先队列）         | 按 $y$ 排序的站点和圆事件            |
| 海滩线（平衡二叉搜索树）     | 活动弧（抛物线）                     |
| 输出边列表                   | Voronoi / Delaunay 边                |

#### 微型代码（伪代码）

```python
def fortunes_algorithm(points):
    points.sort(key=lambda p: -p[1])  # 从上到下排序
    event_queue = [(p[1], 'site', p) for p in points]
    beach_line = []
    voronoi_edges = []
    while event_queue:
        y, typ, data = event_queue.pop(0)
        if typ == 'site':
            insert_arc(beach_line, data)
        else:
            remove_arc(beach_line, data)
        update_edges(beach_line, voronoi_edges)
    return voronoi_edges
```

这个草图省略了细节，但展示了事件驱动的扫描结构。

#### 为什么它很重要

*   最优的 $O(n \log n)$ Delaunay / Voronoi 构造
*   避免了复杂的全局翻转
*   优美的几何解释：抛物线 + 扫描线
*   计算几何库的基础（例如 CGAL、Boost、Qhull）

应用：

*   最近邻搜索（Voronoi 区域）
*   地形和网格生成
*   蜂窝覆盖建模
*   运动规划和影响图

#### 一个温和的证明（为什么它有效）

每个站点和圆事件最多触发一个事件，因此有 $O(n)$ 个事件。
每个事件花费 $O(\log n)$ 时间（在平衡树中进行插入/删除/搜索）。
所有边都满足局部 Delaunay 条件，因为弧仅在抛物线相遇（等距前沿）时创建。

因此，总复杂度为：

$$
O(n \log n)
$$

正确性源于：

*   扫描线维护了有效的部分 Voronoi/Delaunay 结构
*   每条 Delaunay 边恰好被创建一次（与 Voronoi 边对偶）

#### 自己动手试试

1.  在纸上画出 3–5 个点。
2.  想象一条线向下扫描。
3.  从每个点画出抛物线弧（距离轨迹）。
4.  标记交点（Voronoi 边）。
5.  连接相邻的点，Delaunay 边自然出现。

#### 测试用例

| 点集                       | Delaunay 三角形 | Voronoi 单元    |
| ------------------------- | --------------- | --------------- |
| 构成三角形的 3 个点       | 1               | 3               |
| 4 个非共线点              | 2               | 4               |
| 网格点                    | 多个            | 网格状单元      |

#### 复杂度

$$
\text{时间: } O(n \log n), \quad \text{空间: } O(n)
$$

Fortune 扫描线算法揭示了几何的深刻对偶性，移动的抛物线追踪着无形的边界，三角形和单元从纯粹的距离对称性中结晶而出。
### 734 沃罗诺伊图（Fortune 扫描线算法）

沃罗诺伊图将平面划分为若干区域，每个区域包含所有距离某个特定站点最近的点。
Fortune 扫描线算法使用与 Delaunay 扫描相同的框架构建此结构，因为两者互为对偶，其时间复杂度为
$$O(n \log n)$$

#### 我们要解决什么问题？

给定一个包含 $n$ 个站点的集合
$$
P = {p_1, p_2, \ldots, p_n}
$$
每个站点位于 $(x_i, y_i)$，沃罗诺伊图将平面划分为单元：

$$
V(p_i) = { q \mid d(q, p_i) \le d(q, p_j), \ \forall j \ne i }
$$

每个沃罗诺伊单元 $V(p_i)$ 都是一个凸多边形（对于不同的站点）。
边是站点对之间的垂直平分线。
顶点是站点三元组的外心。

#### 为什么使用 Fortune 算法？

朴素方法：计算所有成对平分线（$O(n^2)$），然后求它们的交点。
Fortune 方法通过扫描一条线并维护定义海滩线的抛物线弧（海滩线是已处理区域和未处理区域之间不断演化的边界），将复杂度改进到
$$O(n \log n)$$

#### 它是如何工作的（通俗解释）

扫描线自上而下移动（$y$ 坐标递减），动态追踪每个站点的影响边界。

1. 站点事件

   * 当扫描线到达一个新站点时，在海滩线中插入一个新的抛物线弧。
   * 弧之间的交点成为断点，这些断点形成沃罗诺伊边。

2. 圆事件

   * 当三个连续的弧汇聚时，中间的弧消失。
   * 汇聚点是一个沃罗诺伊顶点（三个站点的外心）。

3. 事件队列

   * 按 $y$ 坐标排序（优先队列）。
   * 每个被处理的事件都会更新海滩线并输出边。

4. 终止

   * 当所有事件处理完毕后，将未完成的边延伸到边界框。

输出是一个完整的沃罗诺伊图，根据对偶性，同时得到其 Delaunay 三角剖分。

#### 示例演练

站点：
$A(2,6), B(5,5), C(3,3)$

步骤：

1.  扫描从顶部开始（站点 A）。
2.  插入 A → 海滩线 = 单个弧。
3.  到达 B → 插入新弧，两个弧相交 → 开始形成沃罗诺伊边。
4.  到达 C → 插入新弧，形成更多边。
5.  发生圆事件，弧汇聚 → 在外心处生成沃罗诺伊顶点。
6.  扫描完成 → 边最终确定，图闭合。

输出：3 个沃罗诺伊单元，3 个顶点，3 条 Delaunay 边。

#### 海滩线表示

海滩线是一系列抛物线弧，存储在以 $x$ 顺序为键的平衡二叉搜索树中。
弧之间的断点追踪沃罗诺伊边。

当插入一个站点时，它会分割一个现有的弧。
当触发圆事件时，一个弧消失，创建一个顶点。

#### 微型代码（伪代码）

```python
def voronoi_fortune(points):
    points.sort(key=lambda p: -p[1])  # 从上到下排序
    event_queue = [(p[1], 'site', p) for p in points]
    beach_line = []
    voronoi_edges = []
    while event_queue:
        y, typ, data = event_queue.pop(0)
        if typ == 'site':
            insert_arc(beach_line, data)
        else:
            remove_arc(beach_line, data)
        update_edges(beach_line, voronoi_edges)
    return voronoi_edges
```

这个高层结构强调了算法的事件驱动性质。
具体实现会使用专门的数据结构来处理弧、断点和圆事件调度。

#### 为什么它很重要

*   同时构建沃罗诺伊图和 Delaunay 三角剖分
*   最优的 $O(n \log n)$ 复杂度
*   适用于大规模几何数据的稳健算法
*   计算几何中空间结构的基础

应用：

*   最近邻搜索
*   游戏和模拟中的空间划分
*   设施选址和影响图
*   网格生成（通过对偶的 Delaunay 三角剖分）

#### 一个温和的证明（为什么它有效）

每个站点事件恰好添加一个弧 → $O(n)$ 个站点事件。
每个圆事件移除一个弧 → $O(n)$ 个圆事件。
每个事件的处理时间为 $O(\log n)$（树更新、优先队列操作）。

因此，总复杂度为：

$$
O(n \log n)
$$

正确性源于抛物线的几何性质：

*   断点总是单调移动
*   每个沃罗诺伊顶点只被创建一次
*   海滩线演化过程中没有回溯

#### 自己动手试试

1.  绘制 3–5 个点。
2.  画出成对的垂直平分线。
3.  注意交点（沃罗诺伊顶点）。
4.  将边连接成凸多边形。
5.  与 Fortune 扫描线的行为进行比较。

#### 测试用例

| 站点             | 沃罗诺伊区域 | 顶点数 | Delaunay 边数 |
| ---------------- | ------------ | ------ | ------------- |
| 3 个点           | 3            | 1      | 3             |
| 4 个非共线点     | 4            | 3      | 5             |
| 3x3 网格         | 9            | 多个   | 网格状        |

#### 复杂度

$$
\text{时间: } O(n \log n), \quad \text{空间: } O(n)
$$

Fortune 扫描线算法构建沃罗诺伊图是动态的几何过程，抛物线在移动的水平线下起伏，描绘出定义邻近性和结构的无形边界。
### 735 增量式 Voronoi

增量式 Voronoi 算法通过一次插入一个站点，逐步构建 Voronoi 图，它更新的是现有图的局部区域，而不是从头开始重新计算。
它在概念上很简单，并且构成了动态和在线 Voronoi 系统的基础。

#### 我们要解决什么问题？

我们想要为一组点
$$
P = {p_1, p_2, \ldots, p_n}
$$
构造或更新一个 Voronoi 图，使得对于每个站点 $p_i$，其 Voronoi 单元包含所有距离 $p_i$ 比距离任何其他站点更近的点。

在静态算法（如 Fortune 扫描线算法）中，所有点必须预先已知。
但是，如果我们想要增量地添加站点，一次一个，并局部更新图表呢？

这正是本算法所实现的。

#### 它是如何工作的（通俗解释）

1.  从简单开始：从单个站点开始，其单元是整个平面（由一个大边界框界定）。
2.  插入下一个站点：
    *   定位它位于哪个单元内。
    *   计算新站点与现有站点之间的垂直平分线。
    *   使用该平分线裁剪现有单元。
    *   新站点的单元由比任何其他站点更靠近它的区域形成。
3.  对所有站点重复此过程。

每次插入只修改附近的单元，而不是整个图，这种局部性质是关键。

#### 示例演练

站点：
$A(2,2)$ → $B(6,2)$ → $C(4,5)$

1.  从 A 开始：单个单元（整个边界框）。
2.  添加 B：
    *   绘制 A 和 B 之间的垂直平分线。
    *   垂直分割平面 → 两个单元。
3.  添加 C：
    *   绘制与 A 和 B 的平分线。
    *   相交平分线以形成三个 Voronoi 区域。

每个新站点通过切割现有单元来"雕刻"出其影响区域。

#### 几何步骤（插入站点 $p$）

1.  定位包含单元：找到 $p$ 位于哪个单元内。
2.  查找受影响的单元：这些是其区域有部分面积比 $p$ 更靠近其站点的邻居单元。
3.  计算 $p$ 与每个受影响站点之间的平分线。
4.  裁剪并重建单元多边形。
5.  更新相邻单元的邻接关系图。

#### 数据结构

| 结构               | 用途                               |
| ------------------ | ---------------------------------- |
| 单元列表           | 每个站点的多边形边界               |
| 站点邻接关系图     | 用于高效的邻居查找                 |
| 边界框             | 用于有限图的截断                   |

可选加速结构：

*   Delaunay 三角剖分：用于更快定位单元的对偶结构
*   空间索引（KD 树）用于单元搜索

#### 微型代码（伪代码）

```python
def incremental_voronoi(points, bbox):
    diagram = init_diagram(points[0], bbox) # 用第一个点和边界框初始化图表
    for p in points[1:]:
        cell = locate_cell(diagram, p) # 定位点 p 所在的单元
        neighbors = find_neighbors(cell) # 查找该单元的邻居
        for q in neighbors:
            bisector = perpendicular_bisector(p, q) # 计算 p 和 q 的垂直平分线
            clip_cells(diagram, p, q, bisector) # 使用平分线裁剪单元
        add_cell(diagram, p) # 将新站点 p 的单元添加到图表中
    return diagram
```

这段伪代码突出了通过平分线裁剪进行渐进式构建的过程。

#### 为什么它很重要

*   概念简单，易于可视化和实现
*   局部更新，只有附近区域发生变化
*   适用于动态系统（添加/删除点）
*   与增量式 Delaunay 三角剖分对偶

应用：

*   在线设施选址
*   动态传感器覆盖
*   实时影响力映射
*   游戏 AI 区域（单位领地）

#### 一个温和的证明（为什么它有效）

每一步都保持 Voronoi 性质：

*   每个区域都是半平面的交集
*   每次插入都会添加新的平分线，细化分区
*   无需重新计算未受影响的区域

时间复杂度取决于我们定位受影响单元的效率。
朴素方法为 $O(n^2)$，但使用 Delaunay 对偶和点定位可以达到：
$$
O(n \log n)
$$

#### 自己动手试试

1.  绘制边界框和一个点（站点）。
2.  插入第二个点，绘制垂直平分线。
3.  插入第三个点，绘制与所有站点的平分线，裁剪重叠区域。
4.  为每个 Voronoi 单元着色，检查边界是否与两个站点等距。
5.  对更多点重复此过程。

#### 测试用例

| 站点数   | 结果                         |
| -------- | ---------------------------- |
| 1 个站点 | 整个边界框                   |
| 2 个站点 | 两个半平面                   |
| 3 个站点 | 三个凸多边形                 |
| 5 个站点 | 复杂的多边形排列             |

#### 复杂度

$$
\text{时间：} O(n^2) \text{ （朴素方法）}, \quad O(n \log n) \text{ （借助 Delaunay 辅助）}
$$
$$
\text{空间：} O(n)
$$

增量式 Voronoi 算法就像晶体形成一样构建图表，每个新点都"雕刻"出自己的区域，用清晰的几何切割重塑其周围的世界。
### 736 Bowyer–Watson

Bowyer–Watson 算法是一种简单而强大的增量方法，用于构建 Delaunay 三角剖分。
每次插入一个新点，算法会局部地重新三角化受该插入影响的区域，确保空圆性质始终成立。

它是最直观且应用最广泛的 Delaunay 构造方法之一。

#### 我们要解决什么问题？

我们希望为一组点
$$
P = {p_1, p_2, \ldots, p_n}
$$
构造一个 Delaunay 三角剖分，使得每个三角形都满足空圆性质：

$$
\text{对于每个三角形 } \triangle abc，\text{ 没有其他点 } p \in P \text{ 位于其外接圆内。}
$$

我们以增量方式构建三角剖分，在每次插入后保持其有效性。

#### 它是如何工作的（通俗解释）

将平面想象成一个有弹性的网格。每次添加一个点时：

1.  找到所有外接圆包含新点的三角形（即"坏三角形"）。
2.  移除这些三角形，它们不再满足 Delaunay 条件。
3.  被移除区域的边界形成一个多边形空洞。
4.  将新点连接到该边界上的每个顶点。
5.  结果是一个新的、仍然是 Delaunay 的三角剖分。

重复此过程，直到所有点都被插入。

#### 逐步示例

点：$A(0,0), B(5,0), C(2.5,5), D(2.5,2)$

1.  用一个包含所有点的超级三角形进行初始化。
2.  插入 $A, B, C$ → 得到基础三角形。
3.  插入 $D$：
   *   找到外接圆包含 $D$ 的三角形。
   *   移除它们（形成一个"洞"）。
   *   将 $D$ 连接到洞的边界顶点。
4.  得到的三角剖分满足 Delaunay 性质。

#### 几何核心：空洞

对于每个新点 $p$：

*   找到所有满足
    $$p \text{ 位于 } \text{circumcircle}(a, b, c) \text{ 内}$$
    的三角形 $\triangle abc$。
*   移除这些三角形。
*   收集所有仅被一个坏三角形共享的边界边，它们构成了空洞多边形。
*   将 $p$ 连接到每条边界边，形成新的三角形。

#### 微型代码（伪代码）

```python
def bowyer_watson(points):
    tri = [super_triangle(points)]
    for p in points:
        bad_tris = [t for t in tri if in_circumcircle(p, t)]
        boundary = find_boundary(bad_tris)
        for t in bad_tris:
            tri.remove(t)
        for edge in boundary:
            tri.append(make_triangle(edge[0], edge[1], p))
    tri = [t for t in tri if not shares_vertex_with_super(t)]
    return tri
```

关键辅助函数：

*   `in_circumcircle(p, triangle)` 测试点是否位于外接圆内
*   `find_boundary` 识别未被两个移除三角形共享的边

#### 为什么它很重要

*   简单、鲁棒，易于实现
*   自然地处理增量插入
*   是许多动态 Delaunay 系统的基础
*   与增量 Voronoi 图对偶（每次插入更新局部单元）

应用：

*   网格生成（有限元，2D/3D）
*   GIS 地形建模
*   粒子模拟
*   空间插值（例如自然邻域法）

#### 一个温和的证明（为什么它有效）

每次插入只移除违反空圆性质的三角形，然后添加保持该性质的新三角形。

通过归纳法：

1.  基础三角剖分（超级三角形）是有效的。
2.  每次插入都保持局部 Delaunay 条件。
3.  因此，整个三角剖分保持为 Delaunay。

复杂度：

*   朴素搜索坏三角形：每次插入 $O(n)$
*   总计：$O(n^2)$
*   使用空间索引/点定位：
    $$O(n \log n)$$

#### 自己试一试

1.  画 3 个点 → 初始三角形。
2.  在内部添加一个新点。
3.  为所有三角形画出外接圆，标记出包含新点的那些。
4.  移除它们；将新点连接到边界。
5.  观察所有三角形现在都满足空圆规则。

#### 测试用例

| 点集            | 三角形数量          | 性质                    |
| --------------- | ------------------ | ----------------------- |
| 3 个点          | 1 个三角形         | 平凡地是 Delaunay       |
| 4 个点          | 2 个三角形         | 两者空圆性质均有效      |
| 随机 6 个点     | 多个三角形         | 有效的三角剖分          |

#### 复杂度

$$
\text{时间：} O(n^2) \text{ 朴素方法}, \quad O(n \log n) \text{ 优化后}
$$
$$
\text{空间：} O(n)
$$

Bowyer–Watson 算法就像用三角形进行雕刻，每个新点都温和地重塑网格，雕刻出空洞，并以完美的几何平衡将它们缝合回去。
### 737 对偶变换

对偶变换揭示了 Delaunay 三角剖分与 Voronoi 图之间的深层联系，它们是几何对偶体。
每条 Voronoi 边对应一条 Delaunay 边，每个 Voronoi 顶点对应一个 Delaunay 三角形的外心。

通过理解这种对偶性，我们可以从一个结构构造出另一个结构，无需分别计算两者。

#### 我们要解决什么问题？

我们常常既需要 Delaunay 三角剖分（用于连通性），又需要 Voronoi 图（用于空间划分）。

与其各自独立构建，我们可以利用对偶性：

* 构建 Delaunay 三角剖分，推导出 Voronoi 图。
* 或者构建 Voronoi 图，推导出 Delaunay 三角剖分。

这节省了计算量，并凸显了几何的对称性。

#### 对偶关系

令 $P = {p_1, p_2, \ldots, p_n}$ 为平面上一组站点。

1. 顶点：

   * 每个 Voronoi 顶点对应一个 Delaunay 三角形的外心。

2. 边：

   * 每条 Voronoi 边与其对偶的 Delaunay 边垂直。
   * 它连接相邻 Delaunay 三角形的外心。

3. 面：

   * 每个 Voronoi 单元对应 Delaunay 中的一个站点顶点。

因此：
$$
\text{Voronoi(对偶)} = \text{Delaunay(原像)}
$$

反之亦然。

#### 它是如何工作的（通俗解释）

从 Delaunay 三角剖分开始：

1. 对于每个三角形，计算其外心。
2. 连接相邻三角形（共享一条边的三角形）的外心。
3. 这些连接形成 Voronoi 边。
4. 这些边的集合构成了 Voronoi 图。

或者，从 Voronoi 图开始：

1. 每个单元的站点成为一个顶点。
2. 如果两个单元的边界相邻，则连接它们的站点 → Delaunay 边。
3. 通过连接三个相互邻接的单元形成三角形。

#### 示例演练

站点：$A(2,2), B(6,2), C(4,5)$

1. Delaunay 三角剖分：三角形 $ABC$。
2. $\triangle ABC$ 的外心 = Voronoi 顶点。
3. 在点对 $(A,B), (B,C), (C,A)$ 之间绘制垂直平分线。
4. 这些线在外心处相交，形成 Voronoi 边。

现在：

* Voronoi 边 ⟷ Delaunay 边
* Voronoi 顶点 ⟷ Delaunay 三角形

对偶完成。

#### 代数对偶（点-线变换）

在计算几何中，我们经常使用点-线对偶：

$$
(x, y) \longleftrightarrow y = mx - c
$$

或者更常见地：

$$
(x, y) \mapsto y = ax - b
$$

在这个意义上：

* 原像空间中的一个点对应对偶空间中的一条线。
* 关联性和顺序得以保持：

  * 点在线上/下 ↔ 线在点上/下。

用于凸包和半平面交集计算。

#### 微型代码（Python 草图）

```python
def delaunay_to_voronoi(delaunay):
    voronoi_vertices = [circumcenter(t) for t in delaunay.triangles]
    voronoi_edges = []
    for e in delaunay.shared_edges():
        c1 = circumcenter(e.tri1)
        c2 = circumcenter(e.tri2)
        voronoi_edges.append((c1, c2))
    return voronoi_vertices, voronoi_edges
```

这里，`circumcenter(triangle)` 计算外接圆的圆心。

#### 为什么它很重要

* 统一了两个核心几何结构
* 实现了三角剖分与划分之间的转换
* 对于网格生成、路径规划、空间查询至关重要
* 简化算法：计算一个，得到两个

应用：

* 地形建模：三角化高程，推导区域
* 最近邻：Voronoi 搜索
* 计算物理：Delaunay 网格，Voronoi 体积
* AI 导航：通过对偶性实现区域邻接

#### 一个温和的证明（为什么它有效）

在 Delaunay 三角剖分中：

* 每个三角形满足空圆性质。
* 相邻三角形的外心到两个站点的距离相等。

因此，根据定义，连接相邻三角形的外心得到的边到两个站点的距离相等，即 Voronoi 边。

所以 Delaunay 三角剖分的对偶图正是 Voronoi 图。

形式上：
$$
\text{Delaunay}(P) = \text{Voronoi}^*(P)
$$

#### 自己动手试试

1. 绘制 4 个不共线的点。
2. 构造 Delaunay 三角剖分。
3. 绘制外接圆并定位外心。
4. 连接相邻三角形的外心 → Voronoi 边。
5. 观察其与原始 Delaunay 边的垂直关系。

#### 测试用例

| 站点     | Delaunay 三角形 | Voronoi 顶点 |
| -------- | --------------- | ------------ |
| 3        | 1               | 1            |
| 4        | 2               | 2            |
| 随机 6   | 4–6             | 多个         |

#### 复杂度

如果任一结构已知：
$$
\text{转换时间：} O(n)
$$
$$
\text{空间：} O(n)
$$

对偶变换是几何的镜子，每条边、每个面、每个顶点都在一个充满垂直关系的世界中反射，揭示了同一优雅真理的两个侧面。
### 738 幂图（加权 Voronoi 图）

幂图（也称为 Laguerre–Voronoi 图）是 Voronoi 图的一种推广，其中每个站点都有一个关联的权重。
我们不再使用简单的欧几里得距离，而是使用幂距离，该距离会根据这些权重来移动或收缩区域。

这允许对某些点比其他点"影响力更大"的影响区域进行建模，非常适合诸如加权最近邻和圆填充等应用。

#### 我们要解决什么问题？

在标准的 Voronoi 图中，每个站点 $p_i$ 拥有所有比任何其他站点更接近它的点 $q$：
$$
V(p_i) = { q \mid d(q, p_i) \le d(q, p_j), \ \forall j \ne i }.
$$

在幂图中，每个站点 $p_i$ 都有一个权重 $w_i$，单元由幂距离定义：
$$
\pi_i(q) = | q - p_i |^2 - w_i.
$$

一个点 $q$ 属于 $p_i$ 的幂单元，当且仅当：
$$
\pi_i(q) \le \pi_j(q) \quad \forall j \ne i.
$$

当所有权重 $w_i = 0$ 时，我们就得到了经典的 Voronoi 图。

#### 它是如何工作的（通俗解释）

将每个站点视为一个圆（或球体），其半径由其权重决定。
我们比较的是幂距离，而不是纯粹的距离：

*   较大的权重意味着更强的影响力（更大的圆）。
*   较小的权重意味着更弱的影响力。

一个点 $q$ 会选择幂距离最小的站点。

这会产生倾斜的平分线（不垂直），并且如果某个单元被相邻单元完全主导，它可能会完全消失。

#### 示例演练

具有权重的站点：

*   $A(2,2), w_A = 1$
*   $B(6,2), w_B = 0$
*   $C(4,5), w_C = 4$

计算 $A$ 和 $B$ 之间的幂平分线：

$$
|q - A|^2 - w_A = |q - B|^2 - w_B
$$

展开并简化后得到一个线性方程，即一条平移后的平分线：
$$
2(x_B - x_A)x + 2(y_B - y_A)y = (x_B^2 + y_B^2 - w_B) - (x_A^2 + y_A^2 - w_A)
$$

因此，边界仍然是直线，但不在站点之间的中心位置。

#### 算法（高层概述）

1.  输入：站点 $p_i = (x_i, y_i)$ 及其权重 $w_i$。
2.  使用幂距离计算所有成对平分线。
3.  相交平分线以形成多边形单元。
4.  将单元裁剪到边界框内。
5.  （可选）使用对偶加权 Delaunay 三角剖分（正则三角剖分）以提高效率。

#### 几何对偶：正则三角剖分

幂图的对偶是一个正则三角剖分，通过使用三维空间中的提升点来构建：

将每个站点 $(x_i, y_i, w_i)$ 映射到三维点 $(x_i, y_i, x_i^2 + y_i^2 - w_i)$。

这些提升点的下凸包，投影回二维平面，就给出了幂图。

#### 微型代码（Python 草图）

```python
def power_bisector(p1, w1, p2, w2):
    (x1, y1), (x2, y2) = p1, p2
    a = 2 * (x2 - x1)
    b = 2 * (y2 - y1)
    c = (x22 + y22 - w2) - (x12 + y12 - w1)
    return (a, b, c)  # 直线 ax + by = c

def power_diagram(points, weights):
    cells = []
    for i, p in enumerate(points):
        cell = bounding_box()
        for j, q in enumerate(points):
            if i == j: continue
            a, b, c = power_bisector(p, weights[i], q, weights[j])
            cell = halfplane_intersect(cell, a, b, c)
        cells.append(cell)
    return cells
```

每个单元都是通过加权平分线定义的半平面的交集构建的。

#### 为什么它很重要

*   Voronoi 图在加权影响力方面的推广
*   产生正则三角剖分对偶
*   支持非均匀密度建模

应用：

*   物理学：加权叠加场
*   地理信息系统：具有不同影响力的区域
*   计算几何：圆填充
*   机器学习：用于加权聚类的幂图

#### 一个温和的证明（为什么它有效）

每个单元由线性不等式定义：
$$
\pi_i(q) \le \pi_j(q)
$$
这些是半平面。
这些半平面的交集形成一个凸多边形（可能为空）。

因此，每个单元：

*   是凸的
*   覆盖所有空间（单元的并集）
*   与其他单元不相交

对偶结构：正则三角剖分，保持加权 Delaunay 性质（空*幂圆*）。

复杂度：
$$
O(n \log n)
$$
使用增量法或提升法。

#### 自己动手试试

1.  绘制两个具有不同权重的点。
2.  计算幂平分线，注意它不是等距的。
3.  添加第三个站点，观察区域如何因权重而移动。
4.  增加一个站点的权重，观察其单元如何扩张。

#### 测试用例

| 站点      | 权重      | 图示               |
| --------- | --------- | ------------------ |
| 2 个相同  | 相同      | 垂直平分线         |
| 2 个不同  | 一个较大  | 平移的边界         |
| 3 个变化  | 混合      | 倾斜的多边形       |

#### 复杂度

$$
\text{时间: } O(n \log n), \quad \text{空间: } O(n)
$$

幂图根据影响力弯曲几何，每个权重都扭曲了空间的平衡，重新绘制了邻近与权力的边界。
### 739 劳埃德松弛法

劳埃德松弛法（也称为劳埃德算法）是一种迭代过程，通过反复将每个站点移动到其 Voronoi 单元的质心来优化 Voronoi 图。其结果是得到一个质心 Voronoi 剖分（CVT），即每个区域的站点同时也是其质心的图。

这是一种几何平滑方法，能将不规则的分区转化为美观、均匀、平衡的布局。

#### 我们要解决什么问题？

标准的 Voronoi 图根据邻近性来划分空间，但如果站点分布不均匀，单元形状可能会不规则或扭曲。

我们想要一个平衡的图，其中：

* 单元紧凑且大小相似
* 站点位于单元的质心

劳埃德松弛法通过迭代优化来解决这个问题。

#### 它是如何工作的（通俗解释）

从一个随机点集和一个边界区域开始。

然后重复以下步骤：

1.  计算当前站点的 Voronoi 图。
2.  找到每个 Voronoi 单元的质心（该区域内所有点的平均值）。
3.  将每个站点移动到其单元的质心。
4.  重复直到站点收敛（移动量很小）。

随着时间的推移，站点会均匀分布，形成一种蓝噪声分布，非常适合采样和网格划分。

#### 分步示例

1.  在一个正方形内初始化 10 个随机站点。
2.  计算 Voronoi 图。
3.  对于每个单元，计算质心：
   $$
   c_i = \frac{1}{A_i} \int_{V_i} (x, y) , dA
   $$
4.  将站点位置 $p_i$ 替换为质心 $c_i$。
5.  重复 5–10 次迭代。

结果：更平滑、更规则且面积几乎相等的单元。

#### 微型代码（Python 草图）

```python
import numpy as np
from scipy.spatial import Voronoi

def lloyd_relaxation(points, bounds, iterations=5):
    for _ in range(iterations):
        vor = Voronoi(points)
        new_points = []
        for region_index in vor.point_region:
            region = vor.regions[region_index]
            if -1 in region or len(region) == 0:
                continue
            polygon = [vor.vertices[i] for i in region]
            centroid = np.mean(polygon, axis=0)
            new_points.append(centroid)
        points = np.array(new_points)
    return points
```

这个简单的实现使用了 Scipy 的 Voronoi 函数，并将质心计算为多边形的平均值。

#### 为什么它很重要

*   产生均匀、平滑、平衡的分区
*   生成蓝噪声分布（对采样很有用）
*   用于网格划分、纹理生成和泊松圆盘采样
*   收敛速度快（通常几次迭代就足够了）

应用：

*   网格生成（有限元、模拟）
*   图形/程序化纹理的采样
*   聚类（k-means 是一种离散的类似方法）
*   晶格设计和区域优化

#### 一个温和的证明（为什么它有效）

每次迭代都会减少一个能量泛函：

$$
E = \sum_i \int_{V_i} | q - p_i |^2 , dq
$$

这衡量了站点到其区域内点的总平方距离。将 $p_i$ 移动到质心可以在局部最小化 $E_i$。

随着迭代的进行：

*   能量单调递减
*   系统收敛到固定点，此时每个 $p_i$ 都是 $V_i$ 的质心

收敛时：
$$
p_i = c_i
$$
每个单元都是一个质心 Voronoi 区域。

#### 自己动手试试

1.  在纸上随机散布一些点。
2.  画出 Voronoi 单元。
3.  估算质心（目视或用网格）。
4.  将点移动到质心。
5.  重新绘制 Voronoi 图。
6.  重复，观察图案变得均匀。

#### 测试用例

| 站点       | 迭代次数 | 结果                 |
| ---------- | -------- | -------------------- |
| 10 个随机点 | 0        | 不规则的 Voronoi 图  |
| 10 个随机点 | 3        | 更平滑、更平衡       |
| 10 个随机点 | 10       | 均匀的 CVT           |

#### 复杂度

每次迭代：

*   Voronoi 计算：$O(n \log n)$
*   质心更新：$O(n)$

总计：
$$
O(k n \log n)
$$
其中 $k$ 为迭代次数。

劳埃德松弛法将随机性打磨成秩序，每次迭代都是向和谐的一次温和推动，将散乱的点转化为平衡的几何镶嵌图案。
### 740 Voronoi 最近邻查询

Voronoi 最近邻查询是 Voronoi 图的一个自然应用。一旦构建了 Voronoi 图，最近邻查找就变得瞬时完成。每个查询点只需落入一个 Voronoi 单元，而定义该单元的站点就是其最近邻。

这使得 Voronoi 结构成为空间搜索、邻近度分析和几何分类的理想选择。

#### 我们要解决什么问题？

给定一组站点
$$
P = {p_1, p_2, \ldots, p_n}
$$
和一个查询点 $q$，我们希望找到最近的站点：
$$
p^* = \arg\min_{p_i \in P} | q - p_i |.
$$

Voronoi 图划分空间，使得单元 $V(p_i)$ 内的每个点 $q$ 都满足：
$$
| q - p_i | \le | q - p_j |, \ \forall j \ne i.
$$

因此，定位 $q$ 所在的单元立即揭示了其最近邻。

#### 它是如何工作的（通俗解释）

1.  预处理：根据站点构建 Voronoi 图。
2.  查询：给定一个新点 $q$，确定它位于哪个 Voronoi 单元。
3.  回答：生成该单元的站点就是最近邻。

这将最近邻搜索从计算（距离比较）转变为几何（区域查找）。

#### 示例演练

站点：

*   $A(2,2)$
*   $B(6,2)$
*   $C(4,5)$

构建 Voronoi 图 → 得到三个凸单元。

查询：$q = (5,3)$

*   检查哪个区域包含 $q$ → 属于 $B$ 的单元。
*   所以最近邻是 $B(6,2)$。

#### 算法（高层次描述）

1.  构建 Voronoi 图（任何方法，例如 Fortune 扫描线算法）。
2.  点定位：
    *   使用空间索引或平面细分搜索（例如梯形图）。
    *   查询点 $q$ → 找到包含它的多边形。
3.  返回关联的站点。

可选优化：如果预期有很多查询，可以构建一个点定位数据结构以实现 $O(\log n)$ 的查询时间。

#### 微型代码（Python 草图）

```python
from scipy.spatial import Voronoi, KDTree

def voronoi_nearest(points, queries):
    vor = Voronoi(points)
    tree = KDTree(points)
    result = []
    for q in queries:
        dist, idx = tree.query(q)
        result.append((q, points[idx], dist))
    return result
```

这里我们结合了 Voronoi 几何（用于理解）和 KD 树（用于实际速度）。

在精确的 Voronoi 查找中，每个查询都使用平面细分中的点定位。

#### 为什么它很重要

*   将最近邻搜索转变为常数时间查找（预处理后）
*   为聚类、导航、模拟等实现空间划分
*   构成以下内容的基础：
    *   最近设施定位
    *   路径规划（区域过渡）
    *   插值（例如最近站点分配）
    *   密度估计、资源分配

应用于：

*   地理信息系统（查找最近的医院、学校等）
*   机器人学（导航区域）
*   物理学（粒子系统中的 Voronoi 单元）
*   机器学习（最近质心分类器）

#### 一个温和的证明（为什么它有效）

根据定义，每个 Voronoi 单元 $V(p_i)$ 满足：
$$
V(p_i) = { q \mid | q - p_i | \le | q - p_j | \ \forall j \ne i }.
$$

所以如果 $q \in V(p_i)$，那么：
$$
| q - p_i | = \min_{p_j \in P} | q - p_j |.
$$

因此，定位 $q$ 所在的单元给出了正确的最近邻。高效的点定位（通过平面搜索）确保了 $O(\log n)$ 的查询时间。

#### 自己动手试试

1.  在纸上画出 4 个站点。
2.  构建 Voronoi 图。
3.  选择一个随机查询点。
4.  查看哪个单元包含它，那就是你的最近站点。
5.  通过手动计算距离来验证。

#### 测试用例

| 站点                      | 查询点       | 最近邻       |
| ------------------------ | ------------ | ------------ |
| A(0,0), B(4,0)           | (1,1)        | A            |
| A(2,2), B(6,2), C(4,5)   | (5,3)        | B            |
| 随机 5 个站点             | 随机查询点   | 包含单元的站点 |

#### 复杂度

*   预处理（构建 Voronoi 图）：$O(n \log n)$
*   查询（点定位）：$O(\log n)$
*   空间：$O(n)$

Voronoi 最近邻方法用优雅的几何取代了暴力距离检查，每个查询都通过找到它所在的位置来解决，而不是计算它走了多远。

# 第 75 节. 点在多边形内与多边形三角剖分
### 741 光线投射算法

光线投射算法（也称为奇偶规则）是一种简单而优雅的方法，用于判断一个点位于多边形内部还是外部。
其工作原理是从查询点发射一条假想的射线，并计算该射线与多边形边相交的次数。

如果相交次数为奇数，则该点在多边形内部。
如果为偶数，则该点在多边形外部。

#### 我们要解决什么问题？

给定：

* 由顶点定义的多边形
  $$P = {v_1, v_2, \ldots, v_n}$$
* 一个查询点
  $$q = (x_q, y_q)$$

判断点 $q$ 位于多边形内部、外部还是边界上。

这个测试在以下领域是基础性的：

* 计算几何
* 计算机图形学（命中测试）
* 地理信息系统（点包含于多边形）
* 碰撞检测

#### 它是如何工作的（通俗解释）

想象从查询点 $q$ 水平向右发射一条光线。
每当光线与多边形的一条边相交时，我们就翻转一个内部/外部标志。

* 如果光线与边相交次数为奇数 → 点在内部
* 如果为偶数 → 点在外部

需要特别注意以下情况：

* 光线恰好穿过一个顶点
* 点恰好位于一条边上

#### 分步过程

1. 设置 `count = 0`。
2. 对于多边形的每条边 $(v_i, v_{i+1})$：

   * 检查从 $q$ 出发的水平射线是否与该边相交。
   * 如果相交，则递增 `count`。
3. 如果 `count` 为奇数，则 $q$ 在内部。
   如果为偶数，则 $q$ 在外部。

边相交条件（对于边 $(x_i, y_i)$ 和 $(x_j, y_j)$ 之间）：

* 射线相交的条件是：
  $$
  y_q \in [\min(y_i, y_j), \max(y_i, y_j))
  $$
  并且
  $$
  x_q < x_i + \frac{(y_q - y_i)(x_j - x_i)}{(y_j - y_i)}
  $$

#### 示例演练

多边形：正方形
$$
(1,1), (5,1), (5,5), (1,5)
$$
查询点 $q(3,3)$

* 从 $(3,3)$ 向右投射射线
* 与左边 $(1,1)-(1,5)$ 相交一次 → count = 1
* 与上/下边相交吗？没有
  → 奇数次相交 → 内部

查询点 $q(6,3)$

* 没有相交 → count = 0 → 外部

#### 微型代码（Python 示例）

```python
def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersect:
                inside = not inside
    return inside
```

这段代码通过简单的奇偶翻转实现了奇偶规则。

#### 为什么它很重要

* 直观且易于实现
* 适用于任何简单多边形（凸多边形或凹多边形）
* 是以下内容的基础：

  * 点包含于区域测试
  * 多边形填充（图形光栅化）
  * GIS 空间连接

应用：

* 图形学：命中检测、裁剪
* 机器人学：占用检查
* 地图绘制：地理包含
* 模拟：空间包含测试

#### 一个温和的证明（为什么它有效）

每次射线穿过一条边时，点就从外部转换到内部，反之亦然。
由于多边形边界是闭合的，总的相交次数决定了最终的奇偶性。

形式化地：
$$
\text{Inside}(q) = \text{count}(q) \bmod 2
$$

如果处理得当（上界开放），共享顶点的边不会被重复计数。

#### 自己动手试试

1. 在方格纸上绘制任意多边形。
2. 选择一个点 $q$ 并向右画一条射线。
3. 计算边相交次数。
4. 检查奇偶性（奇数 → 内部，偶数 → 外部）。
5. 将 $q$ 移动到边附近以测试特殊情况。

#### 测试用例

| 多边形              | 点              | 结果          |
| ------------------ | --------------- | ------------- |
| 正方形 (1,1)-(5,5) | (3,3)           | 内部          |
| 正方形 (1,1)-(5,5) | (6,3)           | 外部          |
| 三角形             | (边中点)        | 在边界上      |
| 凹多边形           | 凹口内部        | 仍然正确      |

#### 复杂度

$$
\text{时间: } O(n), \quad \text{空间: } O(1)
$$

光线投射算法就像用光线照射几何图形，每次相交都会翻转你的视角，揭示点是在形状的阴影之内还是之外。
### 742 环绕数

环绕数算法是一种用于点与多边形包含性测试的鲁棒方法。与仅计算交叉次数的射线投射法不同，它测量多边形围绕查询点旋转了多少次，不仅捕获了内部/外部状态，还捕获了方向（顺时针与逆时针）。

如果环绕数非零，则点在多边形内部；如果为零，则在外部。

#### 我们要解决什么问题？

给定：

* 一个多边形 $P = {v_1, v_2, \ldots, v_n}$
* 一个查询点 $q = (x_q, y_q)$

确定点 $q$ 位于多边形内部还是外部，包括凹多边形和自相交多边形的情况。

环绕数定义为多边形边围绕该点扫过的总角度：
$$
w(q) = \frac{1}{2\pi} \sum_{i=1}^{n} \Delta\theta_i
$$
其中 $\Delta\theta_i$ 是从 $q$ 出发的连续边之间的有向角度。

#### 它是如何工作的（通俗解释）

想象你沿着多边形的边行走，并从你的路径上观察查询点：

* 当你遍历时，该点似乎围绕你旋转。
* 每次转向都会为环绕总和贡献一个角度。
* 如果总转向等于 $2\pi$（或 $-2\pi$），则表示你已围绕该点环绕一次 → 点在内部。
* 如果总转向等于 $0$，则表示你从未环绕该点 → 点在外部。

这就像计算你围绕该点循环了多少次。

#### 逐步过程

1. 初始化 $w = 0$。
2. 对于每条边 $(v_i, v_{i+1})$：

   * 计算向量：
     $$
     \mathbf{u} = v_i - q, \quad \mathbf{v} = v_{i+1} - q
     $$
   * 计算有向角度：
     $$
     \Delta\theta = \text{atan2}(\det(\mathbf{u}, \mathbf{v}), \mathbf{u} \cdot \mathbf{v})
     $$
   * 加到总和上： $w += \Delta\theta$
3. 如果 $|w| > \pi$，则点在内部；否则，在外部。

或者等价地：
$$
\text{如果 } w / 2\pi \ne 0 \text{，则在内部}
$$

#### 示例演练

多边形：
$(0,0), (4,0), (4,4), (0,4)$
查询点 $q(2,2)$

对于每条边，计算围绕 $q$ 的有向转向。
总角度和 = $2\pi$ → 内部

查询点 $q(5,2)$
总角度和 = $0$ → 外部

#### 方向处理

$\Delta\theta$ 的符号取决于多边形方向：

* 逆时针 (CCW) → 正角度
* 顺时针 (CW) → 负角度

因此，环绕数也可以揭示方向：

* $+1$ → 在逆时针多边形内部
* $-1$ → 在顺时针多边形内部
* $0$ → 在外部

#### 微型代码（Python 示例）

```python
import math

def winding_number(point, polygon):
    xq, yq = point
    w = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        u = (x1 - xq, y1 - yq)
        v = (x2 - xq, y2 - yq)
        det = u[0]*v[1] - u[1]*v[0]
        dot = u[0]*v[0] + u[1]*v[1]
        angle = math.atan2(det, dot)
        w += angle
    return abs(round(w / (2 * math.pi))) > 0
```

这段代码计算扫过的总角度，并检查它是否近似为 $2\pi$。

#### 为什么它很重要

* 比射线投射法更鲁棒（处理自相交情况）
* 适用于凹多边形和复杂多边形
* 捕获方向信息
* 用于计算几何库（CGAL、GEOS、Shapely）

应用：

* 地理空间分析（边界内检测）
* 图形学（填充规则，奇偶规则与非零环绕规则）
* 不规则形状的碰撞检测
* 矢量渲染（SVG 使用环绕规则）

#### 一个温和的证明（为什么它有效）

每条边都贡献了一个围绕 $q$ 的角度转向。
通过对所有这些转向求和，我们测量了净旋转。
如果多边形包围了 $q$，则路径会环绕一次（总计 $2\pi$）。
如果 $q$ 在外部，则转向相互抵消（总计 $0$）。

正式地：
$$
w(q) = \frac{1}{2\pi} \sum_{i=1}^{n} \text{atan2}(\det(\mathbf{u_i}, \mathbf{v_i}), \mathbf{u_i} \cdot \mathbf{v_i})
$$
并且 $w(q) \ne 0$ 当且仅当 $q$ 被包围。

#### 自己尝试

1.  画一个凹多边形（例如星形）。
2.  在凹处内部选择一个点。
3.  射线投射法可能会错误分类，但环绕数不会。
4.  直观地计算角度，将它们加起来。
5.  注意符号表示方向。

#### 测试用例

| 多边形              | 点     | 结果                |
| ------------------- | ------ | ------------------- |
| 正方形 (0,0)-(4,4)  | (2,2)  | 内部                |
| 正方形              | (5,2)  | 外部                |
| 星形                | 中心   | 内部                |
| 星形                | 尖端   | 外部                |
| 顺时针多边形        | (2,2)  | 环绕数 = -1         |

#### 复杂度

$$
\text{时间: } O(n), \quad \text{空间: } O(1)
$$

环绕数算法不仅仅是询问一条射线穿过边界多少次，它还倾听空间围绕该点的旋转，计算完整的旋转次数以揭示包围关系。
### 743 凸多边形点测试

凸多边形点测试是一种快速而优雅的方法，用于判断一个点位于凸多边形的内部、外部还是边界上。它完全依赖于方向测试，即查询点与多边形每条边之间的叉积符号。

因为凸多边形具有一致的“转向”方向，所以此方法能以线性时间运行，且没有复杂的分支逻辑。

#### 我们要解决什么问题？

给定：
* 一个凸多边形 $P = {v_1, v_2, \ldots, v_n}$
* 一个查询点 $q = (x_q, y_q)$

我们想要测试点 $q$ 位于：
* $P$ 的内部
* $P$ 的边界上
* $P$ 的外部

此测试专用于凸多边形，其中所有内角 $\le 180^\circ$，且边具有一致的方向（顺时针或逆时针）。

#### 它是如何工作的（通俗解释）

在凸多边形中，所有顶点都朝同一个方向转向（例如逆时针）。如果一个点始终位于每条边的同一侧，则该点在多边形内部。

测试方法如下：
1. 遍历所有边 $(v_i, v_{i+1})$。
2. 对于每条边，计算边向量与从顶点到查询点的向量之间的叉积：
   $$
   \text{cross}((v_{i+1} - v_i), (q - v_i))
   $$
3. 记录符号（正、负或零）。
4. 如果所有符号均为非负（或非正）→ 点在内部或边界上。
5. 如果符号不同 → 点在外部。

#### 叉积测试

对于两个向量 $\mathbf{a} = (x_a, y_a)$, $\mathbf{b} = (x_b, y_b)$

二维叉积为：
$$
\text{cross}(\mathbf{a}, \mathbf{b}) = a_x b_y - a_y b_x
$$

在几何中：
* $\text{cross} > 0$: $\mathbf{b}$ 在 $\mathbf{a}$ 的左侧（逆时针转向）
* $\text{cross} < 0$: $\mathbf{b}$ 在 $\mathbf{a}$ 的右侧（顺时针转向）
* $\text{cross} = 0$: 点共线

#### 逐步示例

多边形（逆时针）：
$(0,0), (4,0), (4,4), (0,4)$

查询点 $q(2,2)$

为每条边计算：

| 边           | 叉积计算                       | 符号 |
| ------------ | ----------------------------- | ---- |
| (0,0)-(4,0)  | $(4,0) \times (2,2) = 8$      | +    |
| (4,0)-(4,4)  | $(0,4) \times (-2,2) = 8$     | +    |
| (4,4)-(0,4)  | $(-4,0) \times (-2,-2) = 8$   | +    |
| (0,4)-(0,0)  | $(0,-4) \times (2,-2) = 8$    | +    |

全部为正 → 内部

#### 微型代码（Python 示例）

```python
def convex_point_test(point, polygon):
    xq, yq = point
    n = len(polygon)
    sign = 0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (yq - y1) - (y2 - y1) * (xq - x1)
        if cross != 0:
            if sign == 0:
                sign = 1 if cross > 0 else -1
            elif sign * cross < 0:
                return "Outside"
    return "Inside/On Boundary"
```

此版本能有效检测符号变化，并在出现不匹配时提前停止。

#### 为什么它很重要

* 快速，线性时间，常数因子小
* 鲁棒，能处理所有凸多边形
* 无需三角学、角度或相交测试
* 自然地适用于整数坐标

应用：
* 凸形状的碰撞检测
* 图形裁剪（Sutherland–Hodgman）
* 凸包成员测试
* 计算几何库（CGAL, Shapely）

#### 一个温和的证明（为什么它有效）

在凸多边形中，所有内部的点必须位于每条边的同一侧。方向符号指示了点位于哪一侧。如果符号不同，点必然跨越了边界 → 外部。

因此：
$$
q \in P \iff \forall i, \ \text{sign}(\text{cross}(v_{i+1} - v_i, q - v_i)) = \text{constant}
$$

这源于凸性：对于每条边，多边形完全位于一个半平面内。

#### 亲自尝试

1.  画一个凸多边形（三角形、正方形、六边形）。
2.  选取一个内部点，测试叉积的符号。
3.  选取一个外部点，注意至少有一个符号翻转。
4.  尝试一个边界上的点，一个叉积 = 0，其他符号相同。

#### 测试用例

| 多边形             | 点              | 结果         |
| ------------------ | --------------- | ------------ |
| 正方形 (0,0)-(4,4) | (2,2)           | 内部         |
| 正方形             | (5,2)           | 外部         |
| 三角形             | (边中点)        | 边界上       |
| 六边形             | (中心)          | 内部         |

#### 复杂度

$$
\text{时间: } O(n), \quad \text{空间: } O(1)
$$

凸多边形点测试像指南针一样解读几何，始终检查方向，确保点安全地位于凸路径的一致转向之内。
### 744 耳切三角剖分

耳切算法（Ear Clipping Algorithm）是一种简单、几何化的方法，用于对简单多边形（凸或凹）进行三角剖分。
它通过迭代地移除“耳朵”——即那些可以被安全切除而不会穿过多边形内部的小三角形——直到只剩下一个三角形为止。

这种方法因其易于实现和数值稳定，被广泛应用于计算机图形学、网格生成和几何处理中。

#### 我们要解决什么问题？

给定一个简单多边形
$$
P = {v_1, v_2, \ldots, v_n}
$$
我们希望将其分解为若干个互不重叠的三角形，这些三角形的并集恰好等于 $P$。

三角剖分是以下领域的基础：

* 渲染和光栅化
* 有限元分析
* 计算几何算法

对于一个有 $n$ 个顶点的多边形，每次三角剖分恰好产生 $n-2$ 个三角形。

#### 它是如何工作的（通俗解释）

多边形的一个“耳朵”是由三个连续顶点 $(v_{i-1}, v_i, v_{i+1})$ 形成的三角形，满足：

1. 该三角形完全位于多边形内部，并且
2. 其内部不包含多边形的任何其他顶点。

该算法反复地“剪切”耳朵：

1. 找到一个能形成耳朵的顶点。
2. 将其（以及耳朵三角形）从多边形中移除。
3. 重复此过程，直到只剩下一个三角形。

每次“剪切”都会使多边形的顶点数减少一个。

#### 耳朵的定义（正式）

三角形 $\triangle (v_{i-1}, v_i, v_{i+1})$ 是一个耳朵，如果：

1. $\triangle$ 是凸的：
   $$
   \text{cross}(v_i - v_{i-1}, v_{i+1} - v_i) > 0
   $$
2. 没有其他顶点 $v_j$（其中 $j \ne i-1,i,i+1$）位于 $\triangle$ 内部。

#### 分步示例

多边形（逆时针方向）：$(0,0), (4,0), (4,4), (2,2), (0,4)$

1. 检查每个顶点的凸性。
2. 顶点 $(4,0)$ 形成一个耳朵，三角形 $(0,0),(4,0),(4,4)$ 内部不包含其他顶点。
3. 剪切耳朵 → 移除 $(4,0)$。
4. 对更小的多边形重复此过程。
5. 继续直到只剩下一个三角形。

结果：三角剖分 = 3 个三角形。

#### 微型代码（Python 示例）

```python
def is_convex(a, b, c):
    return (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]) > 0

def point_in_triangle(p, a, b, c):
    cross1 = (b[0] - a[0])*(p[1] - a[1]) - (b[1] - a[1])*(p[0] - a[0])
    cross2 = (c[0] - b[0])*(p[1] - b[1]) - (c[1] - b[1])*(p[0] - b[0])
    cross3 = (a[0] - c[0])*(p[1] - c[1]) - (a[1] - c[1])*(p[0] - c[0])
    return (cross1 >= 0 and cross2 >= 0 and cross3 >= 0) or (cross1 <= 0 and cross2 <= 0 and cross3 <= 0)

def ear_clipping(polygon):
    triangles = []
    vertices = polygon[:]
    while len(vertices) > 3:
        n = len(vertices)
        for i in range(n):
            a, b, c = vertices[i-1], vertices[i], vertices[(i+1) % n]
            if is_convex(a, b, c):
                ear = True
                for p in vertices:
                    if p not in (a, b, c) and point_in_triangle(p, a, b, c):
                        ear = False
                        break
                if ear:
                    triangles.append((a, b, c))
                    vertices.pop(i)
                    break
    triangles.append(tuple(vertices))
    return triangles
```

这个版本每次迭代移除一个耳朵，并在 $n-3$ 次迭代后终止。

#### 为什么它很重要

* 易于理解和实现
* 适用于任何简单多边形（凸或凹）
* 产生一致的三角剖分
* 构成许多高级网格生成算法的基础

应用领域：

* 渲染多边形（OpenGL 曲面细分）
* 物理碰撞网格
* 几何建模（例如 GIS、FEM）

#### 一个温和的证明（为什么它有效）

每个简单多边形至少有两个耳朵（迈斯特斯定理）。
每个耳朵都是一个不与其它三角形重叠的有效三角形。
通过每一步剪切一个耳朵，多边形的边界会收缩，同时保持其简单性。
因此，该算法总是以 $n-2$ 个三角形终止。

时间复杂度（朴素实现）：
$$
O(n^2)
$$
使用空间加速（例如，邻接表）：
$$
O(n \log n)
$$

#### 自己动手试试

1. 画一个凹多边形。
2. 找到凸顶点。
3. 测试每个顶点是否满足耳朵条件（内部没有其他顶点）。
4. 剪切耳朵，重画多边形。
5. 重复直到完成整个三角剖分。

#### 测试用例

| 多边形形状     | 顶点数 | 三角形数 |
| -------------- | ------ | -------- |
| 三角形         | 3      | 1        |
| 凸四边形       | 4      | 2        |
| 凹五边形       | 5      | 3        |
| 星形           | 8      | 6        |

#### 复杂度

$$
\text{时间: } O(n^2), \quad \text{空间: } O(n)
$$

耳切三角剖分就像折纸一样切割几何形状，一次一个耳朵，直到每一个折叠都变成一个完美的三角形。
### 745 单调多边形三角剖分

单调多边形三角剖分算法是一种强大而高效的方法，用于对 y-单调多边形进行三角剖分。y-单调多边形是指其边永远不会沿 y 轴“回溯”的多边形。由于这个特性，我们可以从上到下扫描，以有序的方式连接对角线，从而实现优雅的 $O(n)$ 时间复杂度。

#### 我们要解决什么问题？

给定一个 y-单调多边形（其边界可以拆分为在 y 方向上都是单调的左链和右链），我们希望将其分割成不重叠的三角形。

如果一个多边形满足任何水平线与其边界最多相交两次，那么它就是 y-单调的。这种结构保证了可以使用基于栈的扫描增量地处理每个顶点。

我们希望得到一个具有以下特性的三角剖分：

*   无边相交
*   线性时间构造
*   用于渲染和几何的稳定结构

#### 它是如何工作的（通俗解释）

想象一下从上到下扫描一条水平线。在每个顶点处，你决定是否将其与之前的顶点用对角线连接起来，形成新的三角形。

核心思想：

1.  按 y 坐标（降序）对顶点排序
2.  将每个顶点分类为属于左链还是右链
3.  使用一个栈来管理当前活跃的顶点链
4.  当可以形成有效的对角线时，弹出栈顶并连接
5.  继续直到只剩下基边

最后，你就得到了多边形的完整三角剖分。

#### 分步说明（概念流程）

1.  输入：一个 y-单调多边形
2.  按 y 坐标降序对顶点排序
3.  用前两个顶点初始化栈
4.  对于每个后续顶点 $v_i$：
    *   如果 $v_i$ 在相对的链上，则将 $v_i$ 连接到栈中的所有顶点，然后重置栈。
    *   否则，弹出形成凸拐角的顶点，添加对角线，并将 $v_i$ 压入栈中。
5.  继续直到只剩下一条链。

#### 示例

多边形（y-单调）：

```
v1 (顶部)
|\
| \
|  \
v2  v3
|    \
|     v4
|    /
v5--v6 (底部)
```

1.  按 y 坐标对顶点排序
2.  识别左链 (v1, v2, v5, v6)，右链 (v1, v3, v4, v6)
3.  从上到下扫描
4.  在下移过程中添加链之间的对角线
5.  在线性时间内完成三角剖分。

#### 微型代码（Python 伪代码）

```python
def monotone_triangulation(vertices):
    # 顶点已按 y 坐标降序排序
    stack = [vertices[0], vertices[1]]
    triangles = []
    for i in range(2, len(vertices)):
        current = vertices[i]
        if on_opposite_chain(current, stack[-1]):
            while len(stack) > 1:
                top = stack.pop()
                triangles.append((current, top, stack[-1]))
            stack = [stack[-1], current]
        else:
            top = stack.pop()
            while len(stack) > 0 and is_convex(current, top, stack[-1]):
                triangles.append((current, top, stack[-1]))
                top = stack.pop()
            stack.extend([top, current])
    return triangles
```

这里的 `on_opposite_chain` 和 `is_convex` 是使用叉积和链标记的几何测试。

#### 为什么它很重要？

*   针对单调多边形的最优 $O(n)$ 算法
*   通用多边形三角剖分中的关键步骤（在分解后使用）
*   应用于：
    *   图形渲染（OpenGL 曲面细分）
    *   地图引擎（GIS）
    *   网格生成和计算几何库

#### 一个温和的证明（为什么它有效）

在一个 y-单调多边形中：

*   边界没有自相交
*   扫描线总是以一致的拓扑顺序遇到顶点
*   每个新顶点只能连接到可见的前驱顶点

因此，每条边和每个顶点只被处理一次，产生 $n-2$ 个三角形，没有冗余操作。

时间复杂度：
$$
O(n)
$$
每个顶点最多被压入和弹出一次。

#### 自己动手试试

1.  画一个 y-单调多边形（像一个山坡）。
2.  标记左链和右链。
3.  从上到下扫描，连接对角线。
4.  跟踪栈操作和形成的三角形。
5.  验证三角剖分产生 $n-2$ 个三角形。

#### 测试用例

| 多边形             | 顶点数 | 三角形数 | 时间     |
| ------------------ | ------ | -------- | -------- |
| 凸多边形           | 5      | 3        | $O(5)$   |
| Y-单调六边形       | 6      | 4        | $O(6)$   |
| 凹的单调多边形     | 7      | 5        | $O(7)$   |

#### 复杂度

$$
\text{时间: } O(n), \quad \text{空间: } O(n)
$$

单调多边形三角剖分就像瀑布一样流动，平滑地扫过多边形的形状，以优雅的精度将其分割成完美、不重叠的三角形。
### 746 Delaunay 三角剖分（最优三角形质量）

Delaunay 三角剖分是计算几何中最优雅、最基础的构造之一。
它为一组点生成一个三角剖分，使得没有任何点位于任何三角形的外接圆内部。
这一特性最大化所有三角形的最小角，避免了狭长、条状的三角形，使其成为网格划分、插值和图形处理的理想选择。

#### 我们要解决什么问题？

给定平面中一个有限点集
$$
P = {p_1, p_2, \ldots, p_n}
$$
我们希望将它们连接成互不重叠的三角形，并满足 Delaunay 条件：

> 对于剖分中的每个三角形，其外接圆内部不包含 $P$ 中的任何其他点。

由此我们得到 Delaunay 三角剖分，它以以下特点著称：

* 最优的角度质量（最大-最小角特性）
* 与 Voronoi 图的对偶性
* 插值和模拟的鲁棒性

#### 它是如何工作的（通俗解释）

想象一下，通过每三个点膨胀出一个圆。
如果一个圆内部没有其他点，它就“属于”一个三角形。
遵守此规则的三角剖分就是 Delaunay 三角剖分。

有几种方法可以构造它：

1. 增量插入法（Bowyer–Watson）：每次添加一个点
2. 分治法：递归地合并 Delaunay 集合
3. Fortune 扫描线法：$O(n \log n)$ 算法
4. 边翻转法：强制执行空圆特性

每种方法都确保没有三角形违反外接圆为空的条件。

#### Delaunay 条件（空外接圆测试）

对于顶点为 $a(x_a,y_a)$, $b(x_b,y_b)$, $c(x_c,y_c)$ 的三角形和一个查询点 $p(x_p,y_p)$：

计算行列式：

$$
\begin{vmatrix}
x_a & y_a & x_a^2 + y_a^2 & 1 \\
x_b & y_b & x_b^2 + y_b^2 & 1 \\
x_c & y_c & x_c^2 + y_c^2 & 1 \\
x_p & y_p & x_p^2 + y_p^2 & 1
\end{vmatrix}
$$


* 如果结果 > 0，点 $p$ 在外接圆内部 → 违反 Delaunay 条件
* 如果 ≤ 0，三角形满足 Delaunay 条件

#### 逐步指南（Bowyer–Watson 方法）

1. 从一个包含所有点的超级三角形开始。
2. 对于每个点 $p$：

   * 找到所有外接圆包含 $p$ 的三角形
   * 移除它们（形成一个空腔）
   * 将 $p$ 连接到空腔边界上的所有边
3. 重复直到所有点都被添加。
4. 移除连接到超级三角形顶点的三角形。

#### 微型代码（Python 草图）

```python
def delaunay(points):
    # 假设有辅助函数：circumcircle_contains, super_triangle
    triangles = [super_triangle(points)]
    for p in points:
        bad = [t for t in triangles if circumcircle_contains(t, p)]
        edges = []
        for t in bad:
            for e in t.edges():
                if e not in edges:
                    edges.append(e)
                else:
                    edges.remove(e)
        for t in bad:
            triangles.remove(t)
        for e in edges:
            triangles.append(Triangle(e[0], e[1], p))
    return [t for t in triangles if not t.shares_super()]
```

这种增量构造的运行时间为 $O(n^2)$，若使用加速技术则为 $O(n \log n)$。

#### 为什么它很重要

* 质量保证：避免狭长三角形
* 对偶结构：构成 Voronoi 图的基础
* 稳定性：输入的小变化 → 三角剖分的小变化
* 应用：

  * 地形建模
  * 网格生成（有限元法，计算流体力学）
  * 插值（自然邻点法，Sibson 插值）
  * 计算机图形学和地理信息系统

#### 一个温和的证明（为什么它有效）

对于一般位置（任意四点不共圆）的任何点集：

* Delaunay 三角剖分存在且唯一
* 它在所有三角剖分中最大化最小角
* 边翻转可以恢复 Delaunay 条件：
  如果两个三角形共享一条边并违反了条件，
  翻转这条边会增加最小的角。

因此，重复翻转直到没有违反条件，就会产生一个有效的 Delaunay 三角剖分。

#### 亲自尝试

1. 在平面上绘制随机点。
2. 任意连接它们，然后检查外接圆。
3. 翻转违反 Delaunay 条件的边。
4. 比较前后变化，注意改进的三角形形状。
5. 叠加 Voronoi 图（它们是对偶结构）。

#### 测试用例

| 点集                 | 方法           | 三角形数量 | 备注                       |
| -------------------- | -------------- | ---------- | -------------------------- |
| 3 个点               | 平凡情况       | 1          | 总是 Delaunay              |
| 形成正方形的 4 个点  | 基于翻转       | 2          | 对角线满足空圆条件         |
| 随机 10 个点         | 增量法         | 16         | Delaunay 网格              |
| 网格点               | 分治法         | 许多       | 均匀网格                   |

#### 复杂度

$$
\text{时间: } O(n \log n), \quad \text{空间: } O(n)
$$

Delaunay 三角剖分在平面上构建了和谐，每个三角形都平衡，每个圆都为空，每个角都宽阔，这是一种既高效又优美的几何。
### 747 凸分解

凸分解算法将复杂的多边形分解为更小的凸多边形块。
由于凸多边形在处理碰撞检测、渲染和几何运算时更为简便，因此在计算几何和图形系统中，这一分解步骤通常至关重要。

#### 我们正在解决什么问题？

给定一个简单多边形（可能是凹多边形），我们希望将其分割为凸子多边形，使得：

1. 所有子多边形的并集等于原多边形。
2. 子多边形互不重叠。
3. 每个子多边形都是凸的，所有内角 ≤ 180°。

凸分解有助于将困难的几何任务（如求交、裁剪、物理模拟）转化为更简单的凸多边形情况。

#### 它是如何工作的（通俗解释）

凹多边形会“向内弯曲”。
为了使它们变成凸的，我们绘制对角线来分割凹区域。
其思路如下：

1. 找到反射顶点（内角 > 180°）。
2. 从每个反射顶点向多边形内部可见的非相邻顶点绘制对角线。
3. 沿着这些对角线分割多边形。
4. 重复此过程，直到每个结果部分都是凸的。

你可以将其想象成从一个纸形状上剪掉折叠部分，直到每一块都能平放。

#### 反射顶点测试

对于顶点序列 $(v_{i-1}, v_i, v_{i+1})$（按逆时针顺序），
计算叉积：

$$
\text{cross}(v_{i+1} - v_i, v_{i-1} - v_i)
$$

* 如果结果 < 0，则 $v_i$ 是反射顶点（凹拐点）。
* 如果结果 > 0，则 $v_i$ 是凸顶点。

反射顶点标记了可以绘制对角线的位置。

#### 逐步示例

多边形（逆时针）：
$(0,0), (4,0), (4,2), (2,1), (4,4), (0,4)$

1. 计算每个顶点的方向，$(2,1)$ 是反射顶点。
2. 从 $(2,1)$ 出发，在对侧链上找到一个可见顶点（例如 $(0,4)$）。
3. 添加对角线 $(2,1)$–$(0,4)$ → 多边形被分割成两个凸部分。
4. 每个结果多边形都通过了凸性测试。

#### 微型代码（Python 示例）

```python
def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def is_reflex(prev, curr, nxt):
    return cross(curr, nxt, prev) < 0

def convex_decomposition(polygon):
    parts = [polygon]
    i = 0
    while i < len(parts):
        poly = parts[i]
        n = len(poly)
        found = False
        for j in range(n):
            if is_reflex(poly[j-1], poly[j], poly[(j+1)%n]):
                for k in range(n):
                    if k not in (j-1, j, (j+1)%n):
                        # 简单的可见性检查（为简化起见）
                        parts.append(poly[j:k+1])
                        parts.append(poly[k:] + poly[:j+1])
                        parts.pop(i)
                        found = True
                        break
            if found: break
        if not found: i += 1
    return parts
```

这个基本结构会查找反射顶点并递归地分割多边形。

#### 为什么它很重要

凸分解是许多几何系统的基础：

* 物理引擎（Box2D、Chipmunk、Bullet）：
  碰撞检测针对每个凸部分进行计算。
* 图形管线：
  光栅化和细分简化为凸多边形。
* 计算几何：
  许多算法（例如，点包含测试、求交）对于凸集来说更容易。

#### 一个温和的证明（为什么它有效）

每个简单多边形都可以使用完全位于多边形内部的对角线分解为凸多边形。
存在一个确定的上限 $n-3$ 条对角线（源自多边形三角剖分）。
由于每个凸多边形显然可以分解为其自身，递归切割过程会终止。

因此，凸分解既是有限的，也是完备的。

#### 自己动手试试

1.  画一个凹多边形（例如箭头形或“L”形）。
2.  标记反射顶点。
3.  添加连接反射顶点到可见点的对角线。
4.  验证每个结果部分都是凸的。
5.  计数：三角形总数 ≤ $n-2$。

#### 测试用例

| 多边形       | 顶点数 | 凸部分数量 | 备注                       |
| ----------- | ------ | ---------- | -------------------------- |
| 凸多边形     | 5      | 1          | 已经是凸的                 |
| 凹“L”形     | 6      | 2          | 单条对角线分割             |
| 星形        | 8      | 5          | 多个反射顶点切割           |
| 不规则多边形 | 10     | 4          | 顺序分解                   |

#### 复杂度

$$
\text{时间复杂度: } O(n^2), \quad \text{空间复杂度: } O(n)
$$

凸分解算法逐块解开几何形状，
将一个复杂的形状转变为简单凸形的马赛克，这些凸形是计算几何的基石。
### 748 多边形面积（鞋带公式）

鞋带公式（也称为高斯面积公式）是一种简单而优雅的方法，可以直接根据顶点坐标计算任何简单多边形（凸多边形或凹多边形）的面积。

它之所以被称为“鞋带”方法，是因为当你以交叉模式乘法和求和坐标时，看起来就像在系鞋带一样。

#### 我们要解决什么问题？

给定一个由其有序顶点定义的多边形
$$
P = {(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)}
$$
我们想要高效地找到它的面积，而无需进行细分或积分。

假设多边形是简单的（边不相交）且闭合的，这意味着 $v_{n+1} = v_1$。

#### 它是如何工作的（通俗解释）

要找到多边形的面积，取连续坐标的叉积之和，一个方向减去另一个方向：

1.  将每个 $x_i$ 乘以下一个顶点的 $y_{i+1}$。
2.  将每个 $y_i$ 乘以下一个顶点的 $x_{i+1}$。
3.  将两个和相减。
4.  取绝对值的一半。

就是这样。当写出来时，乘积的模式形成了一个“鞋带”，因此得名。

#### 公式

$$
A = \frac{1}{2} \Bigg| \sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i) \Bigg|
$$

其中 $(x_{n+1}, y_{n+1}) = (x_1, y_1)$ 以闭合多边形。

#### 示例

多边形：
$(0,0), (4,0), (4,3), (0,4)$

逐步计算：

| i | $x_i$ | $y_i$ | $x_{i+1}$ | $y_{i+1}$ | $x_i y_{i+1}$ | $y_i x_{i+1}$ |
| - | ----- | ----- | --------- | --------- | ------------- | ------------- |
| 1 | 0     | 0     | 4         | 0         | 0             | 0             |
| 2 | 4     | 0     | 4         | 3         | 12            | 0             |
| 3 | 4     | 3     | 0         | 4         | 0             | 12            |
| 4 | 0     | 4     | 0         | 0         | 0             | 0             |

现在计算：
$$
A = \frac{1}{2} |(0 + 12 + 0 + 0) - (0 + 0 + 12 + 0)| = \frac{1}{2} |12 - 12| = 0
$$

哎呀，这意味着我们必须检查顶点顺序（顺时针 CW 与逆时针 CCW）。
重新排序得到正面积：

$$
A = \frac{1}{2} |12 + 16 + 0 + 0 - (0 + 0 + 0 + 0)| = 14
$$

所以面积 = 14 平方单位。

#### 微型代码（Python 示例）

```python
def polygon_area(points):
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

poly = [(0,0), (4,0), (4,3), (0,4)]
print(polygon_area(poly))  # 输出: 14.0
```

只要顶点顺序一致（顺时针或逆时针），这个版本对凸多边形和凹多边形都适用。

#### 为什么它重要

*   简单且精确（整数算术完美适用）
*   不需要三角学或分解
*   应用广泛：GIS、CAD、图形学、机器人学
*   适用于任何由顶点坐标定义的 2D 多边形。

应用：

*   计算地块面积
*   多边形裁剪算法
*   基于几何的物理
*   矢量图形（SVG 路径面积）

#### 一个温和的证明（为什么它有效）

鞋带公式源自格林定理的线积分形式：

$$
A = \frac{1}{2} \oint (x,dy - y,dx)
$$

沿着多边形边离散化得到：

$$
A = \frac{1}{2} \sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i)
$$

绝对值确保无论方向如何（顺时针或逆时针），面积都是正的。

#### 自己试试

1.  取任意多边形、三角形、正方形或不规则形状。
2.  按顺序写下坐标。
3.  交叉相乘，一个方向求和然后减去另一个方向的和。
4.  取绝对值的一半。
5.  与已知的几何面积进行比较验证。

#### 测试用例

| 多边形                               | 顶点数 | 预期面积            |
| ------------------------------------- | -------- | ------------------------ |
| 三角形 (0,0),(4,0),(0,3)            | 3        | 6                        |
| 矩形 (0,0),(4,0),(4,3),(0,3)     | 4        | 12                       |
| 平行四边形 (0,0),(5,0),(6,3),(1,3) | 4        | 15                       |
| 凹形                         | 5        | 与几何形状一致 |

#### 复杂度

$$
\text{时间复杂度: } O(n), \quad \text{空间复杂度: } O(1)
$$

鞋带公式是几何学的算术诗篇——
数字巧妙地交叉编织，在一行代数中悄然围合出形状的整个面积。
### 749 闵可夫斯基和

闵可夫斯基和是一种几何运算，通过逐点相加两个形状的坐标来组合它们。它是计算几何、机器人学和运动规划领域的基石，用于建模可达空间、扩展障碍物，以及以数学上精确的方式组合形状。

#### 我们要解决什么问题？

给定平面上的两个点集（或形状）：

$$
A, B \subset \mathbb{R}^2
$$

闵可夫斯基和定义为从 $A$ 中取一点和从 $B$ 中取一点的所有可能和构成的集合：

$$
A \oplus B = {, a + b \mid a \in A,, b \in B ,}
$$

直观地说，我们让一个形状沿着另一个形状"扫过"，将它们的坐标相加，结果是一个新的形状，代表了所有可能的位置组合。

#### 它是如何工作的（通俗解释）

将 $A$ 和 $B$ 视为两个多边形。计算 $A \oplus B$：

1.  取 $A$ 中的每个顶点，与 $B$ 中的每个顶点相加。
2.  收集所有得到的点。
3.  计算该点集的凸包。

如果 $A$ 和 $B$ 都是凸的，那么它们的闵可夫斯基和也是凸的，并且可以通过按排序的角度顺序合并边来高效计算（类似于合并两个凸多边形）。

如果 $A$ 或 $B$ 是凹的，你可以先将它们分解为凸部分，计算所有成对的和，然后合并结果。

#### 几何意义

如果你将 $B$ 视为一个"物体"，将 $A$ 视为一个"区域"，那么 $A \oplus B$ 表示如果 $B$ 的参考点沿着 $A$ 移动，$B$ 可以占据的所有位置。

例如：

*   在机器人学中，$A$ 可以是机器人，$B$ 可以是障碍物，它们的和给出了所有可能的碰撞配置。
*   在图形学中，它用于形状扩展、偏移和碰撞检测。

#### 逐步示例

令：
$$
A = {(0,0), (2,0), (1,1)}, \quad B = {(0,0), (1,0), (0,1)}
$$

计算所有成对的和：

| $a$   | $b$   | $a+b$ |
| ----- | ----- | ----- |
| (0,0) | (0,0) | (0,0) |
| (0,0) | (1,0) | (1,0) |
| (0,0) | (0,1) | (0,1) |
| (2,0) | (0,0) | (2,0) |
| (2,0) | (1,0) | (3,0) |
| (2,0) | (0,1) | (2,1) |
| (1,1) | (0,0) | (1,1) |
| (1,1) | (1,0) | (2,1) |
| (1,1) | (0,1) | (1,2) |

所有这些点的凸包 = 闵可夫斯基和多边形。

#### 微型代码（Python 示例）

```python
from itertools import product

def minkowski_sum(A, B):
    # 计算所有点对的和
    points = [(a[0]+b[0], a[1]+b[1]) for a, b in product(A, B)]
    return convex_hull(points)

def convex_hull(points):
    # 计算点集的凸包
    points = sorted(set(points))
    if len(points) <= 1:
        return points
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]
```

这段代码计算所有和，并围绕它们构建一个凸包。

#### 为什么它很重要

*   **碰撞检测**：
    $A \oplus (-B)$ 可以判断形状是否相交（如果原点 ∈ 和集）。
*   **运动规划**：
    通过机器人形状扩展障碍物可以简化路径规划。
*   **图形学和 CAD**：
    用于偏移、缓冲和形态学操作。
*   **凸分析**：
    为凸函数和支持集的加法建模。

#### 一个温和的证明（为什么它有效）

对于凸集 $A$ 和 $B$，闵可夫斯基和保持凸性：

$$
\lambda_1 (a_1 + b_1) + \lambda_2 (a_2 + b_2)
= (\lambda_1 a_1 + \lambda_2 a_2) + (\lambda_1 b_1 + \lambda_2 b_2)
\in A \oplus B
$$

对于所有满足 $\lambda_1, \lambda_2 \ge 0$ 且 $\lambda_1 + \lambda_2 = 1$ 的 $\lambda_1, \lambda_2$ 成立。

因此，$A \oplus B$ 是凸的。从几何上看，这个和代表了所有点的向量加法，这是凸性在线性组合下封闭性的直接应用。

#### 自己动手试试

1.  从两个凸多边形开始（比如一个正方形和一个三角形）。
2.  将每个顶点对相加并绘制这些点。
3.  取凸包，那就是你的闵可夫斯基和。
4.  尝试翻转一个形状（$-B$），这个和会缩小成一个交集测试。

#### 测试用例

| 形状 A       | 形状 B           | 结果形状         |
| ------------ | ---------------- | ---------------- |
| 三角形       | 三角形           | 六边形           |
| 正方形       | 正方形           | 更大的正方形     |
| 线段         | 圆形             | 加粗的线         |
| 多边形       | 负多边形         | 碰撞区域         |

#### 复杂度

$$
\text{时间: } O(n + m), \quad \text{对于大小为 } n, m \text{ 的凸多边形}
$$

（使用角度合并算法）

闵可夫斯基和是几何学中"加法"思想的方式——每个形状都扩展了另一个，它们共同定义了空间中一切可达、可组合和可能的事物。
### 750 多边形求交（Weiler–Atherton 裁剪算法）

Weiler–Atherton 算法是一种经典且通用的方法，用于计算两个任意多边形（即使是带孔的凹多边形）的交集、并集或差集。它是计算机图形学、CAD 和地理空间分析中使用的裁剪系统的几何核心。

#### 我们要解决什么问题？

给定两个多边形：

* 主体多边形 $S$
* 裁剪多边形 $C$

我们想要找到交集区域 $S \cap C$，或者可选地找到并集 ($S \cup C$) 或差集 ($S - C$)。

与仅处理凸多边形的简单算法（如 Sutherland–Hodgman）不同，Weiler–Atherton 算法适用于任何简单多边形，无论是凸多边形、凹多边形还是带孔多边形。

#### 它是如何工作的（通俗解释）

其思想是沿着两个多边形的边行进，在交点处切换多边形，以追踪最终的裁剪区域。

可以把它想象成沿着 $S$ 行走，每当碰到 $C$ 的边界时，就决定是进入还是离开裁剪区域。这种路径追踪构建了最终的交集多边形。

#### 分步概述

1.  **查找交点**
    计算 $S$ 和 $C$ 的边之间的所有交点。将这些点按正确顺序插入到两个多边形的顶点列表中。

2.  **将交点标记为“进入”或“离开”**
    这取决于沿着 $S$ 的边界行进时，是进入还是离开 $C$。

3.  **遍历多边形**

    *   从一个未访问的交点开始。
    *   如果它是“进入”点，则沿着 $S$ 行进，直到遇到下一个交点。
    *   切换到 $C$ 并沿着其边界继续追踪。
    *   在两个多边形之间交替，直到回到起点。

4.  **重复直到所有交点都被访问。**
    每次闭合的遍历给出最终结果的一部分（可能是多个不相连的多边形）。

#### 相交几何（数学测试）

对于线段 $A_1A_2$ 和 $B_1B_2$，我们使用参数直线方程计算交点：

$$
A_1 + t(A_2 - A_1) = B_1 + u(B_2 - B_1)
$$

求解 $t$ 和 $u$：

$$
t = \frac{(B_1 - A_1) \times (B_2 - B_1)}{(A_2 - A_1) \times (B_2 - B_1)}, \quad
u = \frac{(B_1 - A_1) \times (A_2 - A_1)}{(A_2 - A_1) \times (B_2 - B_1)}
$$

如果 $0 \le t, u \le 1$，则线段相交于：

$$
P = A_1 + t(A_2 - A_1)
$$

#### 微型代码（Python 示例）

此草图展示了概念结构（省略了数值边缘情况）：

```python
def weiler_atherton(subject, clip):
    intersections = []
    for i in range(len(subject)):
        for j in range(len(clip)):
            p1, p2 = subject[i], subject[(i+1)%len(subject)]
            q1, q2 = clip[j], clip[(j+1)%len(clip)]
            ip = segment_intersection(p1, p2, q1, q2)
            if ip:
                intersections.append(ip)
                subject.insert(i+1, ip)
                clip.insert(j+1, ip)

    result = []
    visited = set()
    for ip in intersections:
        if ip in visited: continue
        polygon = []
        current = ip
        in_subject = True
        while True:
            polygon.append(current)
            visited.add(current)
            next_poly = subject if in_subject else clip
            idx = next_poly.index(current)
            current = next_poly[(idx + 1) % len(next_poly)]
            if current in intersections:
                in_subject = not in_subject
            if current == ip:
                break
        result.append(polygon)
    return result
```

这捕捉了算法结构，实际上，几何库（如 Shapely、CGAL、GEOS）会稳健地处理精度和拓扑问题。

#### 为什么它很重要

*   处理复杂多边形（凹多边形、带孔、多个交点）
*   适用于所有布尔运算（交集、并集、差集）
*   是以下应用的基础：
    *   计算机图形学裁剪（在视口内渲染多边形）
    *   GIS 空间分析（叠加操作）
    *   2D CAD 建模（切割和合并形状）

#### 一个温和的证明（为什么它有效）

通过在交点处交替遍历多边形，算法保持了拓扑连续性，最终的多边形边界遵循 $S$ 和 $C$ 的有效边。因为交点将多边形划分为连通的边界片段，并且每次遍历都在“内部”和“外部”区域之间交替，所以每条闭合路径都对应于交集的一个有效部分。

因此，正确性源于：

*   一致的方向（顺时针或逆时针）
*   准确的内部/外部测试
*   对所有交点的完整遍历

#### 自己动手试试

1.  画两个重叠的多边形（一个凸多边形，一个凹多边形）。
2.  找到边之间的所有交点。
3.  将每个点标记为进入或离开。
4.  沿着边在两个多边形之间交替，追踪交集区域。
5.  填充它，那就是 $S \cap C$。

#### 测试用例

| 主体多边形 | 裁剪多边形 | 操作 | 结果 |
| --------------- | ------------ | -------------------- | ------------- |
| 矩形 | 三角形 | 交集 | 三角形帽 |
| 凹多边形 | 矩形 | 交集 | 裁剪后的形状 |
| 两个矩形 | 重叠 | 并集 | 组合框 |
| 星形和圆形 | 交集 | 复杂曲线区域 | |

#### 复杂度

$$
\text{时间: } O((n + m)^2), \quad \text{空间: } O(n + m)
$$

优化的实现使用空间索引来加速相交测试。

Weiler–Atherton 算法将多边形重叠问题转化为沿着边界的行走——进入、离开和重新汇合，精确地追踪两个形状如何真正相遇的几何形状。

# 第 76 节 空间数据结构
### 751 KD 树构建

KD 树（*k-dimensional tree* 的缩写）是一种用于组织 k 维空间中点的数据结构，以实现快速的最近邻查询和范围查询。
它是一种递归的、空间划分的结构，通过轴对齐的超平面分割空间，很像一次又一次地将世界切成两半。

#### 我们要解决什么问题？

给定一组点
$$
P = {p_1, p_2, \ldots, p_n} \subset \mathbb{R}^k
$$
我们希望构建一种结构，使我们能够高效地回答几何查询，例如：

* "哪个点离 $(x, y, z)$ 最近？"
* "哪些点位于这个边界框内？"

与其每次检查所有点（每次查询 $O(n)$），我们一次性构建一棵 KD 树，从而能够在平均 $O(\log n)$ 的时间内进行搜索。

#### 它是如何工作的（通俗解释）

KD 树是一种二叉树，它递归地按坐标轴分割点：

1.  选择一个分割轴（例如，按 $x$，然后 $y$，然后 $x$，… 循环进行）。
2.  沿着该轴找到中位数点。
3.  创建一个存储该点的节点，这就是你的分割平面。
4.  递归构建：
    *   左子树 → 坐标小于中位数的点
    *   右子树 → 坐标大于中位数的点

每个节点将空间划分为两个半空间，创建了一个嵌套边界框的层次结构。

#### 逐步示例（2D）

点：
$(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)$

构建过程：

| 步骤 | 轴   | 中位数点 | 分割线 | 左侧点             | 右侧点        |
| ---- | ---- | -------- | ------ | ------------------ | ------------- |
| 1    | x    | (7,2)    | x=7    | (2,3),(5,4),(4,7)  | (8,1),(9,6)   |
| 2    | y    | (5,4)    | y=4    | (2,3)              | (4,7)         |
| 3    | y    | (8,1)    | y=1    | ,                  | (9,6)         |

最终结构：

```
        (7,2)
       /     \
   (5,4)     (8,1)
   /   \        \
(2,3) (4,7)     (9,6)
```

#### 微型代码（Python 示例）

```python
def build_kdtree(points, depth=0):
    if not points:
        return None
    k = len(points[0])
    axis = depth % k
    points.sort(key=lambda p: p[axis])
    median = len(points) // 2
    return {
        'point': points[median],
        'left': build_kdtree(points[:median], depth + 1),
        'right': build_kdtree(points[median + 1:], depth + 1)
    }
```
这个递归构建器通过交替轴对点进行排序，并在每一层选取中位数点。

#### 为什么它很重要

KD 树是计算几何中的核心结构之一，应用广泛：

*   最近邻搜索，在 $O(\log n)$ 时间内找到最近的点
*   范围查询，统计或收集轴对齐框内的点
*   光线追踪与图形学，加速可见性和相交性检查
*   机器学习，加速 k-NN 分类或聚类
*   机器人学/运动规划，组织配置空间

#### 一个温和的证明（为什么它有效）

在每次递归中，中位数分割确保了：

*   树的高度大约是 $\log_2 n$
*   每次搜索在每个维度上只下降一个分支，从而剪枝掉空间的大部分

因此，构建过程平均是 $O(n \log n)$（由于排序），
并且在平衡条件下查询是对数级的。

形式上，在每一层：
$$
T(n) = 2T(n/2) + O(n) \Rightarrow T(n) = O(n \log n)
$$

#### 自己动手试试

1.  写下 8 个随机的 2D 点。
2.  按 x 轴对它们排序，选取中位数 → 根节点。
3.  递归地按 y 轴对左右两半排序 → 下一次分割。
4.  为每次分割绘制边界线（垂直线和水平线）。
5.  将分区可视化为矩形区域。

#### 测试用例

| 点集           | 维度 | 预期深度 | 说明             |
| -------------- | ---- | -------- | ---------------- |
| 7 个随机点     | 2D   | ~3       | 平衡分割         |
| 1000 个随机点  | 3D   | ~10      | 基于中位数       |
| 10 个共线点    | 1D   | 10       | 退化链           |
| 网格点         | 2D   | log₂(n)  | 均匀区域         |

#### 复杂度

| 操作           | 时间              | 空间       |
| -------------- | ----------------- | ---------- |
| 构建           | $O(n \log n)$     | $O(n)$     |
| 搜索           | $O(\log n)$ (平均) | $O(\log n)$ |
| 最坏情况搜索   | $O(n)$            | $O(\log n)$ |

KD 树就像一个几何文件柜 ——
每次分割都将空间整齐地折叠成两半，让你只需进行几次优雅的比较就能找到最近的点，而无需搜索整个世界。
### 752 KD 树搜索

一旦构建好 KD 树，其真正的威力来自于快速的搜索操作，即无需扫描整个数据集就能找到查询位置附近的点。搜索利用了 KD 树的递归空间划分特性，剪枝掉那些不可能包含最近点的大片空间。

#### 我们要解决什么问题？

给定：

* 一个组织在 KD 树中的点集 $P \subset \mathbb{R}^k$
* 一个查询点 $q = (q_1, q_2, \ldots, q_k)$

我们想要找到：

1. $q$ 的最近邻（欧几里得距离最小的点）
2. 或者给定范围内的所有点（轴对齐区域或半径范围内）

与 $O(n)$ 的暴力搜索不同，KD 树搜索平均能达到 $O(\log n)$ 的查询时间。

#### 它是如何工作的（通俗解释）

搜索递归地下降遍历 KD 树：

1. 在每个节点，比较查询点在当前分割轴上的坐标。
2. 移动到包含查询点的子树。
3. 当到达叶子节点时，将其记录为当前最佳点。
4. 回溯：
   * 如果到目前为止最佳点周围的超球体与分割平面相交，则也搜索另一棵子树（可能存在更近的点）。
   * 否则，剪掉那个分支，因为它不可能包含更近的点。
5. 返回找到的最接近的点。

这种剪枝是 KD 树效率的核心。

#### 逐步示例（二维最近邻）

点集（来自之前的构建）：
$(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)$

树根：$(7,2)$，在 x 轴上分割

查询：$q = (9,2)$

1. 比较 $q_x=9$ 与分割点 $x=7$ → 进入右子树
2. 比较 $q_y=2$ 与分割点 $y=1$ → 进入 $(9,6)$ 的子树
3. 计算 $(9,6)$ 的距离 → $d=4$
4. 回溯：检查半径为 4 的圆是否与 x=7 平面相交 → 是 → 探索左侧
5. 左子节点 $(8,1)$ → $d=1.41$ → 更优
6. 完成 → 最近邻 = $(8,1)$

#### 微型代码（Python 示例）

```python
import math

def distance2(a, b):
    return sum((a[i] - b[i])2 for i in range(len(a)))

def nearest_neighbor(tree, point, depth=0, best=None):
    if tree is None:
        return best

    k = len(point)
    axis = depth % k
    next_branch = None
    opposite_branch = None

    if point[axis] < tree['point'][axis]:
        next_branch = tree['left']
        opposite_branch = tree['right']
    else:
        next_branch = tree['right']
        opposite_branch = tree['left']

    best = nearest_neighbor(next_branch, point, depth + 1, best)

    if best is None or distance2(point, tree['point']) < distance2(point, best):
        best = tree['point']

    if (point[axis] - tree['point'][axis])2 < distance2(point, best):
        best = nearest_neighbor(opposite_branch, point, depth + 1, best)

    return best
```

此函数仅递归探索必要的分支，剪掉那些不可能包含更近点的分支。

#### 为什么它很重要

KD 树搜索是许多算法的支柱：

* 机器学习：k-最近邻（k-NN）、聚类
* 计算机图形学：光线-物体相交
* 机器人学/运动规划：最近样本搜索
* 仿真/物理：邻近检测
* 地理信息系统/空间数据库：区域和半径查询

如果没有 KD 树搜索，这些任务的计算量将随数据规模线性增长。

#### 一个温和的证明（为什么它有效）

KD 树搜索的正确性依赖于两个几何事实：

1. 分割平面将空间划分为不相交的区域。
2. 最近邻点必须位于：
   * 与查询点相同的区域，或者
   * 跨越分割平面，但距离小于当前最佳半径。

因此，基于 $q$ 与分割平面之间的距离进行剪枝，永远不会排除可能的更近点。通过仅在必要时访问子树，搜索既完整又高效。

#### 亲自尝试

1. 为二维点构建一个 KD 树。
2. 查询一个随机点，追踪递归调用。
3. 绘制搜索区域并可视化被剪枝的子树。
4. 增加数据规模，注意查询时间如何保持在 $O(\log n)$ 附近。

#### 测试用例

| 查询点 | 期望最近邻 | 距离   | 备注               |
| ------ | ---------- | ------ | ------------------ |
| (9,2)  | (8,1)      | 1.41   | 在右侧分支         |
| (3,5)  | (4,7)      | 2.23   | 左侧区域           |
| (5,3)  | (5,4)      | 1.00   | 精确匹配坐标轴     |

#### 复杂度

| 操作                     | 时间        | 空间       |
| ------------------------ | ----------- | ---------- |
| 平均搜索                 | $O(\log n)$ | $O(\log n)$ |
| 最坏情况（退化的树）     | $O(n)$      | $O(\log n)$ |

KD 树搜索就像一位拥有完美直觉的侦探在几何空间中穿行——只检查必须检查的地方，跳过可以跳过的地方，并且一旦找到可能的最接近答案就立即停止。
### 753 KD 树中的范围搜索

KD 树中的范围搜索是一种几何查询，用于检索给定轴对齐区域（2D 中的矩形、3D 中的长方体或更高维中的超矩形）内的所有点。
这是 KD 树遍历的自然延伸，但我们不是寻找一个最近邻，而是收集位于目标窗口内的所有点。

#### 我们要解决什么问题？

给定：

* 一个包含 $k$ 维空间中 $n$ 个点的 KD 树
* 一个查询区域（例如，在 2D 中）：
  $$
  R = [x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]
  $$

我们想要找到所有点 $p = (p_1, \ldots, p_k)$，使得：

$$
x_{\min} \le p_1 \le x_{\max}, \quad
y_{\min} \le p_2 \le y_{\max}, \quad \ldots
$$

换句话说，就是所有位于轴对齐框 $R$ *内部*的点。

#### 它是如何工作的（通俗解释）

该算法递归地访问 KD 树节点，并剪枝掉不可能与查询区域相交的分支：

1. 在每个节点处，将分割坐标与区域的边界进行比较。
2. 如果节点的点位于 $R$ 内，则记录它。
3. 如果左子树可能包含位于 $R$ 内的点，则搜索左子树。
4. 如果右子树可能包含位于 $R$ 内的点，则搜索右子树。
5. 当子树完全落在区域外时停止。

这种方法避免了访问大多数节点，只访问那些其区域与查询框重叠的节点。

#### 分步示例（2D）

KD 树（如前所述）：

```
        (7,2)
       /     \
   (5,4)     (8,1)
   /   \        \
(2,3) (4,7)     (9,6)
```

查询区域：
$$
x \in [4, 8], \quad y \in [1, 5]
$$

搜索过程：

1. 根节点 (7,2) 在区域内 → 记录。
2. 左子节点 (5,4) 在区域内 → 记录。
3. (2,3) 在区域左侧 → 剪枝。
4. (4,7) y=7 > 5 → 剪枝。
5. 右子节点 (8,1) 在区域内 → 记录。
6. (9,6) x > 8 → 剪枝。

结果：
区域内的点 = (7,2), (5,4), (8,1)

#### 微型代码（Python 示例）

```python
def range_search(tree, region, depth=0, found=None):
    if tree is None:
        return found or []
    if found is None:
        found = []
    k = len(tree['point'])
    axis = depth % k
    point = tree['point']

    # 检查点是否在区域内
    if all(region[i][0] <= point[i] <= region[i][1] for i in range(k)):
        found.append(point)

    # 如果子树与区域重叠，则探索子树
    if region[axis][0] <= point[axis]:
        range_search(tree['left'], region, depth + 1, found)
    if region[axis][1] >= point[axis]:
        range_search(tree['right'], region, depth + 1, found)
    return found
```

使用示例：

```python
region = [(4, 8), (1, 5)]  # x 和 y 的边界
results = range_search(kdtree, region)
```

#### 为什么它很重要

范围查询是空间计算的基础：

* 数据库索引（R 树，KD 树）→ 快速过滤
* 图形学 → 查找视口或相机视锥体内的对象
* 机器人学 → 检索用于碰撞检查的局部邻居
* 机器学习 → 空间限制内的聚类
* GIS 系统 → 空间连接和地图查询

KD 树范围搜索将几何逻辑与高效剪枝相结合，使其适用于高速应用。

#### 一个温和的证明（为什么它有效）

KD 树中的每个节点定义了一个空间的超矩形区域。
如果该区域完全位于查询框之外，则其内部的任何点都不可能位于查询框内，因此我们可以安全地跳过它。
否则，我们进行递归。

访问的节点总数为：
$$
O(n^{1 - \frac{1}{k}} + m)
$$
其中 $m$ 是报告的点数，
这是多维搜索理论中的一个已知界限。

因此，范围搜索是输出敏感的：它的规模取决于实际找到的点数。

#### 亲自尝试

1. 用随机的 2D 点构建一个 KD 树。
2. 定义一个边界框 $[x_1,x_2]\times[y_1,y_2]$。
3. 跟踪递归调用，注意哪些分支被剪枝。
4. 可视化查询区域，确认返回的点落在内部。

#### 测试用例

| 区域                     | 预期点              | 备注               |
| ------------------------ | ------------------- | ------------------ |
| $x\in[4,8], y\in[1,5]$   | (5,4), (7,2), (8,1) | 3 个点在内部       |
| $x\in[0,3], y\in[2,4]$   | (2,3)               | 单个匹配           |
| $x\in[8,9], y\in[0,2]$   | (8,1)               | 在边界上           |
| $x\in[10,12], y\in[0,5]$ | ∅                   | 空结果             |

#### 复杂度

| 操作         | 时间                 | 空间         |
| ------------ | -------------------- | ------------ |
| 范围搜索     | $O(n^{1 - 1/k} + m)$ | $O(\log n)$  |
| 平均（2D–3D）| $O(\sqrt{n} + m)$    | $O(\log n)$  |

KD 树范围搜索就像用探照灯扫过几何空间——
它只照亮你关心的部分，将其余部分留在黑暗中，
并只揭示在你查询窗口内闪烁的点。
### 754 KD 树中的最近邻搜索

最近邻（NN）搜索是 KD 树上最重要的操作之一。它能在数据集中找到欧几里得空间中距离给定查询点最近的一个（或多个）点，这个问题在聚类、机器学习、图形学和机器人学中都有出现。

#### 我们要解决什么问题？

给定：

* 一个点集 $P = {p_1, p_2, \ldots, p_n} \subset \mathbb{R}^k$
* 基于这些点构建的 KD 树
* 一个查询点 $q \in \mathbb{R}^k$

我们想要找到：

$$
p^* = \arg\min_{p_i \in P} |p_i - q|
$$

即通过欧几里得距离（有时是曼哈顿距离或余弦距离）找到距离 $q$ 最近的点 $p^*$。

#### 它是如何工作的（通俗解释）

KD 树的最近邻搜索通过递归地下降到树中工作，就像在多维空间中进行二分搜索一样。

1.  从根节点开始。
    沿着节点的分割轴比较查询坐标。
    根据查询值是更小还是更大来决定向左或向右走。

2.  递归直到叶节点。
    该叶节点的点成为你初始的最佳点。

3.  回溯到树上。
    在每个节点：
    *   如果该节点的点更近，则更新最佳点。
    *   检查查询点周围的超球面（半径 = 当前最佳距离）是否与分割平面相交。
    *   如果相交，则探索另一个子树，因为平面另一侧可能存在更近的点。
    *   如果不相交，则剪掉该分支。

4.  当返回到根节点时终止。

结果：保证最佳点就是真正的最近邻。

#### 分步示例（二维）

点：
$(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)$

KD 树根节点：$(7,2)$，按 $x$ 轴分割

查询点：$q = (9,2)$

| 步骤 | 节点      | 轴                 | 动作                     | 最佳点（距离平方）                 |              |
| ---- | --------- | ------------------ | ------------------------ | ---------------------------------- | ------------ |
| 1    | (7,2)     | x                  | 向右走 (9 > 7)           | (7,2), $d=4$                       |              |
| 2    | (8,1)     | y                  | 向上走 (2 > 1)           | (8,1), $d=2$                       |              |
| 3    | (9,6)     | y                  | 距离 = 17 → 更差         | (8,1), $d=2$                       |              |
| 4    | 回溯      | 检查分割平面 $     | 9-7                      | =2$, 等于 $r=√2$ → 探索左侧        | (8,1), $d=2$ |
| 5    | 完成      |,                   |,                         | 最近邻 = (8,1)                     |              |

#### 微型代码（Python 示例）

```python
import math

def dist2(a, b):
    return sum((a[i] - b[i])2 for i in range(len(a)))

def kd_nearest(tree, query, depth=0, best=None):
    if tree is None:
        return best
    k = len(query)
    axis = depth % k
    next_branch = None
    opposite_branch = None

    point = tree['point']
    if query[axis] < point[axis]:
        next_branch, opposite_branch = tree['left'], tree['right']
    else:
        next_branch, opposite_branch = tree['right'], tree['left']

    best = kd_nearest(next_branch, query, depth + 1, best)
    if best is None or dist2(query, point) < dist2(query, best):
        best = point

    # 检查另一分支是否可能包含更近的点
    if (query[axis] - point[axis])2 < dist2(query, best):
        best = kd_nearest(opposite_branch, query, depth + 1, best)

    return best
```

用法：

```python
nearest = kd_nearest(kdtree, (9,2))
print("Nearest:", nearest)
```

#### 为什么它很重要

最近邻搜索无处不在：

*   机器学习
    *   k-NN 分类器
    *   聚类（k-means, DBSCAN）
*   计算机图形学
    *   光线追踪加速
    *   纹理查找、采样
*   机器人学
    *   路径规划（PRM, RRT*）
    *   障碍物接近度
*   仿真
    *   粒子系统和空间交互

KD 树的最近邻搜索将平均查询时间从 $O(n)$ 减少到 $O(\log n)$，使其适用于实时应用。

#### 一个温和的证明（为什么它有效）

剪枝规则在几何上是合理的，因为两个属性：

1.  每个子树完全位于分割平面的一侧。
2.  如果查询的超球面（半径 = 当前最佳距离）与该平面不相交，则另一侧不可能存在更近的点。

因此，只有其边界区域与球面重叠的子树才会被探索，这保证了正确性和效率。

在平衡的情况下：
$$
T(n) \approx O(\log n)
$$
而在退化（不平衡）树中：
$$
T(n) = O(n)
$$

#### 自己试试

1.  为 10 个随机的二维点构建一个 KD 树。
2.  查询一个点并跟踪递归过程。
3.  绘制最佳距离的超球面，看看哪些分支被跳过。
4.  与暴力最近邻搜索比较，验证结果相同。

#### 测试用例

| 查询点 | 预期最近邻 | 距离   | 备注                 |
| ------ | ---------- | ------ | -------------------- |
| (9,2)  | (8,1)      | 1.41   | 右重查询             |
| (3,5)  | (4,7)      | 2.23   | 深度左侧搜索         |
| (7,2)  | (7,2)      | 0      | 精确命中             |

#### 复杂度

| 操作     | 平均情况   | 最坏情况 |
| -------- | ---------- | -------- |
| 搜索     | $O(\log n)$ | $O(n)$   |
| 空间占用 | $O(n)$      | $O(n)$   |

KD 树的最近邻搜索就像是形式化的直觉——它直接跳向答案必然存在的地方，只在几何要求时瞥一眼侧面，然后让空间的其余部分保持安静。
### 755 R-树构建

R-树是一种强大的空间索引结构，设计用于处理矩形、多边形和空间对象，而不仅仅是点。它被用于数据库、地理信息系统（GIS）和图形引擎中，以进行高效的范围查询、重叠检测和最近对象搜索。

虽然 KD-树通过坐标轴划分空间，但 R-树通过紧密包围数据对象或更小组对象的边界框来划分空间。

#### 我们要解决什么问题？

我们需要一种支持以下功能的索引结构：

*   快速搜索与查询区域重叠的对象
*   高效的插入和删除操作
*   动态增长，无需从头开始重新平衡

R-树提供了所有这三种功能，使其成为动态、多维空间数据（矩形、多边形、区域）的理想选择。

#### 工作原理（通俗解释）

其核心思想是将附近的对象分组，并用它们的最小边界矩形来表示：

1.  每个叶子节点存储形式为 `(MBR, 对象)` 的条目。
2.  每个内部节点存储形式为 `(MBR, 子节点指针)` 的条目，其中 MBR 覆盖所有子矩形。
3.  根节点的 MBR 覆盖整个数据集。

当插入或搜索时，算法遍历这些嵌套的边界框，剪枝掉不与查询区域相交的子树。

#### 构建 R-树（批量加载）

构建 R-树主要有两种方法：

##### 1. 增量插入（动态）

使用 ChooseSubtree 规则逐个插入每个对象：

1.  从根节点开始。
2.  在每一层，选择其 MBR 需要最小扩展就能包含新对象的子节点。
3.  如果子节点溢出（条目过多），则使用启发式方法（如二次分裂或线性分裂）对其进行分裂。
4.  向上更新父节点的 MBR。

##### 2. 批量加载（静态）

对于大型静态数据集，按空间顺序（例如，希尔伯特曲线或 Z 序曲线）对对象进行排序，然后逐层打包它们以最小化重叠。

#### 示例（二维矩形）

假设我们有 8 个对象，每个对象都有边界框：

| 对象 | 矩形 $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$ |
| :--- | :--------------------------------------------- |
| A    | (1, 1, 2, 2)                                   |
| B    | (2, 2, 3, 3)                                   |
| C    | (8, 1, 9, 2)                                   |
| D    | (9, 3, 10, 4)                                  |
| E    | (5, 5, 6, 6)                                   |
| F    | (6, 6, 7, 7)                                   |
| G    | (3, 8, 4, 9)                                   |
| H    | (4, 9, 5, 10)                                  |

如果每个节点最多能容纳 4 个条目，我们可能会这样分组：

*   节点 1 → {A, B, C, D}
    MBR = (1,1,10,4)
*   节点 2 → {E, F, G, H}
    MBR = (3,5,7,10)
*   根节点 → {节点 1, 节点 2}
    MBR = (1,1,10,10)

这种层次化的嵌套使得快速区域查询成为可能。

#### 微型代码（Python 示例）

一个简化的静态 R-树构建器：

```python
def build_rtree(objects, max_entries=4):
    if len(objects) <= max_entries:
        return {'children': objects, 'leaf': True,
                'mbr': compute_mbr(objects)}

    # 按 x 中心排序以便分组
    objects.sort(key=lambda o: (o['mbr'][0] + o['mbr'][2]) / 2)
    groups = [objects[i:i+max_entries] for i in range(0, len(objects), max_entries)]

    children = [{'children': g, 'leaf': True, 'mbr': compute_mbr(g)} for g in groups]
    return {'children': children, 'leaf': False, 'mbr': compute_mbr(children)}

def compute_mbr(items):
    xmin = min(i['mbr'][0] for i in items)
    ymin = min(i['mbr'][1] for i in items)
    xmax = max(i['mbr'][2] for i in items)
    ymax = max(i['mbr'][3] for i in items)
    return (xmin, ymin, xmax, ymax)
```

#### 为什么重要

R-树广泛应用于：

*   空间数据库（PostGIS，SQLite 的 R-树扩展）
*   游戏引擎（碰撞和可见性查询）
*   地理信息系统（地图数据索引）
*   CAD 和图形学（对象选择和剔除）
*   机器人/仿真（空间占用网格）

R-树将 KD-树推广到处理*具有大小和形状的对象*，而不仅仅是点。

#### 一个温和的证明（为什么它有效）

R-树的正确性依赖于两个几何不变性：

1.  每个子节点的边界框完全包含在其父节点的 MBR 内。
2.  每个叶子节点的 MBR 覆盖其存储的对象。

因为该结构保持了这些包含关系，任何与父节点框相交的查询都只需检查相关的子树，从而确保了完备性和正确性。

其效率来自于最小化兄弟节点 MBR 之间的重叠，这减少了不必要的子树访问。

#### 自己动手试试

1.  创建几个矩形并可视化它们的边界框。
2.  手动将它们分组到 MBR 簇中。
3.  画出代表父节点的嵌套矩形。
4.  执行一个查询，例如"所有与 (2,2)-(6,6) 相交的对象"，并跟踪访问了哪些框。

#### 测试用例

| 查询框         | 预期结果       | 备注           |
| :------------- | :------------- | :------------- |
| (1,1)-(3,3)    | A, B           | 在节点 1 内    |
| (5,5)-(7,7)    | E, F           | 在节点 2 内    |
| (8,2)-(9,4)    | C, D           | 右侧分组       |
| (0,0)-(10,10)  | 全部           | 完全重叠       |

#### 复杂度

| 操作   | 平均情况       | 最坏情况   |
| :----- | :------------- | :--------- |
| 搜索   | $O(\log n)$    | $O(n)$     |
| 插入   | $O(\log n)$    | $O(n)$     |
| 空间   | $O(n)$         | $O(n)$     |

R-树就像一个安静的几何图书管理员——它将形状整齐地归档到嵌套的盒子中，这样当你问"附近有什么？"时，它只打开相关的抽屉。
### 756 R*-树

R*-树是 R-树的一个改进版本，其核心在于最小化边界框之间的重叠和覆盖范围。通过精心选择插入和分裂条目的位置及方式，它在实际空间查询中实现了更优的性能。

由于 R*-树能够高效处理动态插入，同时保持较低的查询时间，因此它是许多现代空间数据库（如 PostGIS 和 SQLite）的默认索引。

#### 我们要解决什么问题？

在标准的 R-树中，边界框可能存在显著的重叠。这会导致搜索效率低下，因为查询区域可能需要探索多个重叠的子树。

R*-树通过优化两个操作来解决这个问题：

1.  **插入**：尝试最小化面积和重叠度的增加。
2.  **分裂**：重新组织条目以减少未来的重叠。

因此，该树能维持更紧凑的边界框和更快的搜索时间。

#### 工作原理（通俗解释）

R*-树在常规 R-树算法的基础上增加了一些增强功能：

1.  **选择子树**

    *   选择其边界框需要最小**扩展**就能包含新条目的子节点。
    *   如果存在多个选择，则优先选择**重叠面积**更小且**总面积**更小的那个。

2.  **强制重插入**

    *   当一个节点溢出时，不立即分裂，而是移除一小部分条目（通常是 30%），并将它们重新插入到树中更高的位置。
    *   这种“震荡”操作重新分配了对象，改善了空间聚类。

3.  **分裂优化**

    *   当分裂不可避免时，使用启发式方法来最小化重叠和周长，而不仅仅是面积。

4.  **重插入级联**

    *   重插入操作可以向上传播，这会略微增加插入成本，但能产生更紧凑、更平衡的树。

#### 示例（二维矩形）

假设我们要向一个已包含以下矩形的节点中插入一个新矩形 $R_{\text{new}}$：

| 矩形 | 面积 | 与其他矩形的重叠度 |
| :--- | :--- | :----------------- |
| A    | 4    | 小                 |
| B    | 6    | 大                 |
| C    | 5    | 中等               |

在普通的 R-树中，如果扩展程度相似，我们可能会随意选择 A 或 B。
在 R*-树中，我们优先选择能最小化以下指标的子节点：

$$
\Delta \text{Overlap} + \Delta \text{Area}
$$

如果仍然平局，则选择周长更小的那个。

这会产生空间紧凑、低重叠度的分区。

#### 微型代码（概念性伪代码）

```python
def choose_subtree(node, rect):
    best = None
    best_metric = float('inf')
    for child in node.children:
        enlargement = area_enlargement(child.mbr, rect) # 计算面积扩展
        overlap_increase = overlap_delta(node.children, child, rect) # 计算重叠度增量
        metric = (overlap_increase, enlargement, area(child.mbr)) # 组合度量指标
        if metric < best_metric:
            best_metric = metric
            best = child
    return best

def insert_rstar(node, rect, obj):
    if node.is_leaf:
        node.entries.append((rect, obj))
        if len(node.entries) > MAX_ENTRIES:
            handle_overflow(node) # 处理溢出
    else:
        child = choose_subtree(node, rect)
        insert_rstar(child, rect, obj)
        node.mbr = recompute_mbr(node.entries) # 重新计算节点的最小边界矩形
```

#### 为何重要

R*-树几乎被用于所有注重性能的空间系统中：

*   **数据库**：PostgreSQL / PostGIS、SQLite、MySQL
*   **地理信息系统和地图绘制**：实时区域和邻近查询
*   **计算机图形学**：视锥体裁剪和碰撞检测
*   **模拟和机器人学**：空间占用网格
*   **机器学习**：嵌入向量或高维数据的范围查询

它在更新成本和查询速度之间取得了平衡，在静态和动态数据集中都表现良好。

#### 一个温和的证明（为何有效）

设每个节点的最小边界矩形为 $B_i$，查询区域为 $Q$。
对于每个子节点，重叠度定义为：

$$
\text{Overlap}(B_i, B_j) = \text{Area}(B_i \cap B_j)
$$

当插入一个新条目时，R*-树尝试最小化：

$$
\Delta \text{Overlap} + \Delta \text{Area} + \lambda \times \Delta \text{Margin}
$$

其中 $\lambda$ 是一个较小的值。经验表明，这种启发式方法可以最小化查询期间预期访问的节点数量。

随着时间的推移，树会收敛到一个平衡的、低重叠度的层次结构，这就是为什么它始终优于基本 R-树的原因。

#### 动手尝试

1.  将矩形插入到一个 R-树和一个 R*-树中。
2.  比较每一层边界框的重叠度。
3.  运行一个范围查询，统计每种算法访问的节点数。
4.  可视化，你会发现 R*-树的边界框更紧凑且更不相交。

#### 测试用例

| 操作                   | 基本 R-树      | R*-树           | 注释                     |
| :--------------------- | :------------- | :-------------- | :----------------------- |
| 插入 1000 个矩形       | 重叠度 60%     | 重叠度 20%      | R*-树聚类效果更好        |
| 查询（区域）           | 访问 45 个节点 | 访问 18 个节点  | 搜索更快                 |
| 批量加载               | 时间相近       | 稍慢            | 但结构更好               |

#### 复杂度

| 操作   | 平均情况       | 最坏情况 |
| :----- | :------------- | :------- |
| 搜索   | $O(\log n)$    | $O(n)$   |
| 插入   | $O(\log n)$    | $O(n)$   |
| 空间   | $O(n)$         | $O(n)$   |

R*-树是耐心的制图师的升级版——它不仅仅是把形状归档到抽屉里，而是会重新组织它们，直到地图上的每条边都清晰地对齐。这样，当你寻找某物时，就能快速而准确地找到它。
### 757 四叉树

四叉树是一种简洁而优雅的空间数据结构，用于递归地将二维空间细分为四个象限（或区域）。它非常适合索引空间数据，如图像、地形、游戏地图以及占据平面不同区域的几何对象。

与按坐标值分割的 KD 树不同，四叉树分割的是空间本身，而不是数据，它在每一层将平面划分为相等的象限。

#### 我们要解决什么问题？

我们想要一种高效表示二维数据空间占用或层次细分的方法。典型目标包括：

* 存储和查询几何数据（点、矩形、区域）。
* 支持快速查找：*"这个区域里有什么？"*
* 支持层次简化或渲染（例如，在计算机图形学或地理信息系统中）。

四叉树通过局部调整其深度，使得高效存储稀疏和密集区域成为可能。

#### 工作原理（通俗解释）

想象一个包含你所有数据的大正方形区域。

1.  从根正方形（整个区域）开始。
2.  如果该区域包含的点数超过阈值（例如 1 或 4），则将其细分为 4 个相等的象限：
    *   NW（西北）
    *   NE（东北）
    *   SW（西南）
    *   SE（东南）
3.  对每个仍然包含过多点的象限递归重复细分过程。
4.  每个叶节点随后保存少量点或对象。

这样就创建了一棵树，其结构反映了数据的空间分布：数据密集处更深，稀疏处更浅。

#### 示例（二维空间中的点）

假设我们在一个 10×10 的网格中有以下二维点：
$(1,1), (2,3), (8,2), (9,8), (4,6)$

*   根正方形覆盖 $(0,0)$–$(10,10)$。
*   它在中心点 $(5,5)$ 处进行细分。
    *   NW：$(0,5)$–$(5,10)$ → 包含 $(4,6)$
    *   NE：$(5,5)$–$(10,10)$ → 包含 $(9,8)$
    *   SW：$(0,0)$–$(5,5)$ → 包含 $(1,1), (2,3)$
    *   SE：$(5,0)$–$(10,5)$ → 包含 $(8,2)$

这种层次化布局使得区域查询直观且快速。

#### 微型代码（Python 示例）

```python
class QuadTree:
    def __init__(self, boundary, capacity=1):
        self.boundary = boundary  # (x, y, w, h)
        self.capacity = capacity
        self.points = []
        self.divided = False

    def insert(self, point):
        x, y = point
        bx, by, w, h = self.boundary
        if not (bx <= x < bx + w and by <= y < by + h):
            return False  # 越界

        if len(self.points) < self.capacity:
            self.points.append(point)
            return True
        else:
            if not self.divided:
                self.subdivide()
            return (self.nw.insert(point) or self.ne.insert(point) or
                    self.sw.insert(point) or self.se.insert(point))

    def subdivide(self):
        bx, by, w, h = self.boundary
        hw, hh = w / 2, h / 2
        self.nw = QuadTree((bx, by + hh, hw, hh), self.capacity)
        self.ne = QuadTree((bx + hw, by + hh, hw, hh), self.capacity)
        self.sw = QuadTree((bx, by, hw, hh), self.capacity)
        self.se = QuadTree((bx + hw, by, hw, hh), self.capacity)
        self.divided = True
```

用法：

```python
qt = QuadTree((0, 0, 10, 10), 1)
for p in [(1,1), (2,3), (8,2), (9,8), (4,6)]:
    qt.insert(p)
```

#### 为什么它重要

四叉树是计算机图形学、地理信息系统和机器人学的基础：

*   图像处理：为压缩和过滤存储像素或区域。
*   游戏引擎：碰撞检测、可见性查询、地形简化。
*   地理数据：用于地图渲染的层次化瓦片。
*   机器人学：用于路径规划的占用栅格。

它们自然地适应空间密度，在需要的地方存储更多细节。

#### 一个温和的证明（为什么它有效）

假设数据集有 $n$ 个点，均匀分布在面积为 $A$ 的二维区域中。每次细分将每个节点的面积减少 4 倍，如果分布不是病态的，则节点的期望数量与 $O(n)$ 成正比。

对于均匀分布的点：
$$
\text{高度} \approx O(\log_4 n)
$$

对于矩形区域的查询成本是：
$$
T(n) = O(\sqrt{n})
$$
在实践中，因为只访问相关的象限。

自适应深度确保密集簇被紧凑地表示，而稀疏区域保持浅层。

#### 自己动手试试

1.  向四叉树中插入 20 个随机点并绘制它（每个细分表示为一个更小的正方形）。
2.  执行查询："矩形 (3,3)-(9,9) 内的所有点"，并统计访问的节点数。
3.  与暴力扫描进行比较。
4.  尝试将容量减少到 1 或 2，观察结构如何加深。

#### 测试用例

| 查询矩形         | 期望的点         | 备注                 |
| ---------------- | ---------------- | -------------------- |
| (0,0)-(5,5)      | (1,1), (2,3)     | 左下象限             |
| (5,0)-(10,5)     | (8,2)            | 右下象限             |
| (5,5)-(10,10)    | (9,8)            | 右上象限             |
| (0,5)-(5,10)     | (4,6)            | 左上象限             |

#### 复杂度

| 操作             | 平均情况         | 最坏情况 |
| ---------------- | ---------------- | -------- |
| 插入             | $O(\log n)$      | $O(n)$   |
| 搜索（区域）     | $O(\sqrt{n})$    | $O(n)$   |
| 空间             | $O(n)$           | $O(n)$   |

四叉树就像画家的网格——
它将世界划分得恰到好处，以便注意到颜色变化之处，
使画布既详细又易于导航。
### 758 八叉树

八叉树是四叉树在三维空间中的扩展。它不是将空间划分为四个象限，而是递归地将立方体划分为八个卦限。这个简单的想法从二维地图优雅地扩展到三维世界，非常适合图形学、物理和空间模拟。

如果说四叉树帮助我们处理像素和瓦片，那么八叉树则帮助我们处理体素、体积和三维空间中的物体。

#### 我们要解决什么问题？

我们需要一种数据结构来高效地表示和查询三维空间信息。

典型目标包括：

* 存储和定位三维点、网格或物体。
* 执行碰撞检测或视锥体裁剪。
* 表示体数据（例如，三维扫描、密度、占用网格）。
* 通过层次化剪枝加速光线追踪或渲染。

八叉树在细节和效率之间取得平衡，对密集区域进行精细划分，同时保持稀疏区域的粗粒度。

#### 工作原理（通俗解释）

八叉树递归地划分空间：

1.  从一个包含所有数据点或物体的立方体开始。
2.  如果一个立方体包含的项目数量超过阈值（例如 4 个），则将其细分为 8 个相等的子立方体（卦限）。
3.  每个节点存储指向其子节点的指针，这些子节点分别覆盖：
    * 前-上-左 (FTL)
    * 前-上-右 (FTR)
    * 前-下-左 (FBL)
    * 前-下-右 (FBR)
    * 后-上-左 (BTL)
    * 后-上-右 (BTR)
    * 后-下-左 (BBL)
    * 后-下-右 (BBR)
4.  递归细分，直到每个叶立方体包含足够少的物体。

这种递归的空间划分形成了三维空间的层次化地图。

#### 示例（三维空间中的点）

想象以下三维点（位于从 $(0,0,0)$ 到 $(8,8,8)$ 的立方体中）：
$(1,2,3), (7,6,1), (3,5,4), (6,7,7), (2,1,2)$

第一次细分发生在立方体的中心 $(4,4,4)$。
每个子立方体覆盖八个卦限之一：

*   $(0,0,0)-(4,4,4)$ → 包含 $(1,2,3), (2,1,2)$
*   $(4,0,0)-(8,4,4)$ → 包含 $(7,6,1)$（后来因 y>4 而被排除）
*   $(0,4,4)-(4,8,8)$ → 包含 $(3,5,4)$
*   $(4,4,4)-(8,8,8)$ → 包含 $(6,7,7)$

每个子立方体仅在需要时才进行细分，从而创建局部自适应的表示。

#### 微型代码（Python 示例）

```python
class Octree:
    def __init__(self, boundary, capacity=2):
        self.boundary = boundary  # (x, y, z, size)
        self.capacity = capacity
        self.points = []
        self.children = None

    def insert(self, point):
        x, y, z = point
        bx, by, bz, s = self.boundary
        if not (bx <= x < bx + s and by <= y < by + s and bz <= z < bz + s):
            return False  # 点超出边界

        if len(self.points) < self.capacity:
            self.points.append(point)
            return True

        if self.children is None:
            self.subdivide()

        for child in self.children:
            if child.insert(point):
                return True
        return False

    def subdivide(self):
        bx, by, bz, s = self.boundary
        hs = s / 2
        self.children = []
        for dx in [0, hs]:
            for dy in [0, hs]:
                for dz in [0, hs]:
                    self.children.append(Octree((bx + dx, by + dy, bz + dz, hs), self.capacity))
```

#### 为什么它很重要

八叉树是现代三维计算的基石：

*   计算机图形学：视锥体裁剪、阴影映射、光线追踪。
*   物理引擎：宽阶段碰撞检测。
*   三维重建：存储体素化场景（例如，Kinect、LiDAR）。
*   GIS 和模拟：体数据和空间查询。
*   机器人技术：三维环境中的占用地图。

因为八叉树能适应数据密度，所以它能显著减少三维问题中的内存使用和查询时间。

#### 一个温和的证明（为什么它有效）

在每一层，立方体被划分为 $8$ 个更小的立方体。
如果一个区域被均匀填充，树的高度为：

$$
h = O(\log_8 n) = O(\log n)
$$

每个查询只访问与查询区域重叠的立方体。
因此，预期的查询时间是次线性的：

$$
T_{\text{query}} = O(n^{2/3})
$$

对于稀疏数据，活动节点的数量远小于 $n$，
所以在实践中，插入和查询的运行时间都接近 $O(\log n)$。

#### 亲自尝试

1.  在立方体 $(0,0,0)$–$(8,8,8)$ 中插入随机三维点。
2.  绘制一个递归的立方体图，显示哪些区域被细分。
3.  查询："哪些点位于 $(2,2,2)$–$(6,6,6)$ 范围内？"
4.  与暴力搜索进行比较。

#### 测试用例

| 查询立方体       | 预期点           | 备注             |
| ---------------- | ---------------- | ---------------- |
| (0,0,0)-(4,4,4)  | (1,2,3), (2,1,2) | 下部卦限         |
| (4,4,4)-(8,8,8)  | (6,7,7)          | 上部远端卦限     |
| (2,4,4)-(4,8,8)  | (3,5,4)          | 上部近端卦限     |

#### 复杂度

| 操作             | 平均情况     | 最坏情况 |
| ---------------- | ------------ | -------- |
| 插入             | $O(\log n)$  | $O(n)$   |
| 搜索（区域）     | $O(n^{2/3})$ | $O(n)$   |
| 空间             | $O(n)$       | $O(n)$   |

八叉树是三维空间的无声建筑师——
它在体积和光线内部构建无形的脚手架，
每个立方体对其世界知之甚少，却足以让一切保持快速、简洁和无限。
### 759 BSP 树（二叉空间分割树）

BSP 树，即*二叉空间分割树*，是一种使用平面对空间进行递归细分的数据结构。
虽然四叉树和八叉树将空间划分为固定的象限或立方体，但 BSP 树可以通过任意超平面进行分割，这使得它在几何、可见性和渲染方面具有难以置信的灵活性。

这种结构是计算机图形学和计算几何学的一项重大突破，曾用于早期的 3D 引擎（如*DOOM*），至今仍在 CAD、物理和空间推理系统中发挥着重要作用。

#### 我们要解决什么问题？

我们需要一种通用、高效的方法来：

* 表示和查询复杂的 2D 或 3D 场景。
* 确定可见性（哪些表面最先被看到）。
* 执行碰撞检测、光线追踪或 CSG（构造实体几何）。

与假设轴对齐分割的四叉树或八叉树不同，BSP 树可以通过任意平面对空间进行分割，完美地适应复杂的几何形状。

#### 工作原理（通俗解释）

1.  从一组几何图元（线、多边形或多面体）开始。
2.  选取一个作为分割平面。
3.  将所有其他对象分成两组：
    * 前向集：位于平面前方的对象。
    * 后向集：位于平面后方的对象。
4.  使用新的平面对每一侧进行递归分割，直到每个区域只包含少量图元。

结果是一棵二叉树：

* 每个内部节点代表一个分割平面。
* 每个叶节点代表一个凸子空间（一个被完全分割的空间区域）。

#### 示例（2D 图示）

想象有三条线分割一个 2D 平面：

* 线 A：垂直
* 线 B：对角线
* 线 C：水平

每条线都将空间分成两个半平面。
所有分割完成后，你会得到凸区域（不重叠的单元）。

每个区域对应 BSP 树中的一个叶子节点，
按从前到后的顺序遍历这棵树，可以得到正确的画家算法渲染结果——将较近的表面绘制在较远的表面之上。

#### 分步总结

1.  选择一个分割多边形或平面（例如，从对象列表中选取一个）。
2.  将其他每个对象分类为位于平面前方、后方或与之相交。
    * 如果相交，则沿平面将其分割。
3.  为前向集和后向集递归构建树。
4.  对于可见性或光线追踪，根据观察者位置相对于平面的关系，按顺序遍历节点。

#### 微型代码（简化的 Python 伪代码）

```python
class BSPNode:
    def __init__(self, plane, front=None, back=None):
        self.plane = plane
        self.front = front
        self.back = back

def build_bsp(objects):
    if not objects:
        return None
    plane = objects[0]  # 选取分割平面
    front, back = [], []
    for obj in objects[1:]:
        side = classify(obj, plane)
        if side == 'front':
            front.append(obj)
        elif side == 'back':
            back.append(obj)
        else:  # 相交
            f_part, b_part = split(obj, plane)
            front.append(f_part)
            back.append(b_part)
    return BSPNode(plane, build_bsp(front), build_bsp(back))
```

这里的 `classify` 用于确定对象位于平面的哪一侧，
而 `split` 则沿该平面分割相交的对象。

#### 为何重要

BSP 树在以下领域至关重要：

*   3D 渲染引擎，为画家算法对多边形进行排序。
*   游戏开发，高效的可见性和碰撞查询。
*   计算几何，点包含和多边形以及光线相交测试。
*   CSG 建模，通过布尔运算（并集、交集、差集）组合实体。
*   机器人和仿真，表示自由和占用的 3D 空间。

#### 一个温和的证明（为何有效）

每个分割平面都将空间分成两个凸子集。
由于凸区域从不重叠，空间中的每个点恰好属于一个叶子节点。

对于 $n$ 个分割平面，形成的凸区域数量在 2D 中是 $O(n^2)$，在 3D 中是 $O(n^3)$，
但通过仅遍历相关分支，查询平均可以在对数时间内得到答案。

从数学上讲，
如果 $Q$ 是查询点，$P_i$ 是平面，
那么每次比较
$$
\text{sign}(a_i x + b_i y + c_i z + d_i)
$$
都会指导遍历，从而产生一个确定性的、空间一致的分区。

#### 动手尝试

1.  绘制 3 个多边形，并将每个都用作分割平面。
2.  在每次分割后为得到的区域着色。
3.  将它们存储在 BSP 树中（前向和后向）。
4.  从给定视点按从后到前的顺序渲染多边形，你会注意到没有深度排序错误。

#### 测试用例

| 场景                     | 平面数 | 区域数 | 用途                     |
| ------------------------ | ------ | ------ | ------------------------ |
| 简单房间                 | 3      | 8      | 可见性排序               |
| 室内地图                 | 20     | 200+   | 碰撞检测和渲染           |
| CSG 模型（立方体 ∩ 球体） | 6      | 50+    | 布尔建模                 |

#### 复杂度

| 操作     | 平均情况      | 最坏情况 |
| -------- | ------------- | -------- |
| 构建     | $O(n \log n)$ | $O(n^2)$ |
| 查询     | $O(\log n)$   | $O(n)$   |
| 空间占用 | $O(n)$        | $O(n^2)$ |

BSP 树是几何哲学家的工具——
它用思想的平面切割世界，
区分前后、可见与隐藏，
直到每个区域都清晰明了，没有任何重叠混淆。
### 760 莫顿序（Z 曲线）

莫顿序，也称为 Z 序曲线，是一种将多维数据（2D、3D 等）映射到一维的巧妙方法，同时能保持空间局部性。
它本身不是一棵树，但它支撑着许多空间数据结构，包括四叉树、八叉树和 R 树，因为它允许进行分层索引而无需显式存储树结构。

之所以称为 "Z 序"，是因为当可视化时，曲线的遍历路径在空间中看起来像一个重复的 Z 字形图案。

#### 我们要解决什么问题？

我们想要一种线性化空间数据的方法，使得空间上邻近的点在排序后也保持邻近。
这在以下方面很有用：

* 高效地对空间数据进行排序和索引。
* 批量加载空间树，如 R 树或 B 树。
* 改善数据库中的缓存局部性和磁盘访问。
* 构建内存高效的分层结构。

莫顿序通过使用位交错技术，提供了一种紧凑且计算成本低廉的方法来实现这一点。

#### 工作原理（通俗解释）

取两个或三个坐标，例如 2D 中的 $(x, y)$ 或 3D 中的 $(x, y, z)$ ——
然后交错它们的比特位以创建一个单一的莫顿码（整数）。

对于 2D：

1. 将 $x$ 和 $y$ 转换为二进制。
   示例：$x = 5 = (101)_2$, $y = 3 = (011)_2$。
2. 交错比特位：从 $x$ 取一位，从 $y$ 取一位，交替进行：
   $x_2 y_2 x_1 y_1 x_0 y_0$。
3. 结果 $(100111)_2 = 39$ 就是点 $(5, 3)$ 的莫顿码。

这个数字代表了该点在 Z 序中的位置。

当你按莫顿码对点进行排序时，邻近的坐标在排序后的顺序中也倾向于彼此靠近 ——
因此，2D 或 3D 的邻近性大致转化为 1D 的邻近性。

#### 示例（2D 可视化）

| 点 $(x, y)$ | 二进制 $(x, y)$ | 莫顿码 | 序号 |
| -------------- | --------------- | ----------- | ----- |
| (0, 0)         | (000, 000)      | 000000      | 0     |
| (1, 0)         | (001, 000)      | 000001      | 1     |
| (0, 1)         | (000, 001)      | 000010      | 2     |
| (1, 1)         | (001, 001)      | 000011      | 3     |
| (2, 2)         | (010, 010)      | 001100      | 12    |

在 2D 中绘制这些点会得到特征性的 "Z" 字形，在每个尺度上递归重复。

#### 微型代码（Python 示例）

```python
def interleave_bits(x, y):
    z = 0
    for i in range(32):  # 假设是 32 位坐标
        z |= ((x >> i) & 1) << (2 * i)
        z |= ((y >> i) & 1) << (2 * i + 1)
    return z

def morton_2d(points):
    return sorted(points, key=lambda p: interleave_bits(p[0], p[1]))

points = [(1,0), (0,1), (1,1), (2,2), (0,0)]
print(morton_2d(points))
```

这将生成点的 Z 序遍历顺序。

#### 为什么它很重要

莫顿序连接了几何与数据系统：

* 数据库：用于批量加载 R 树（称为 *打包 R 树*）。
* 图形学：纹理 Mipmapping 和空间采样。
* 并行计算：网格的块分解（空间缓存效率）。
* 数值模拟：自适应网格细化索引。
* 向量数据库：快速的近似最近邻分组。

因为它保持了*空间局部性*并支持*位运算*，所以它比按欧几里得距离排序或使用复杂数据结构进行初始索引要快得多。

#### 一个温和的证明（为什么它有效）

Z 曲线递归地将空间细分为象限（2D）或八分圆（3D），并以深度优先的顺序访问它们。
在每个递归层级，最高有效位的交错比特决定了点属于哪个象限或八分圆。

对于一个 2D 点 $(x, y)$：

$$
M(x, y) = \sum_{i=0}^{b-1} \left[ (x_i \cdot 2^{2i}) + (y_i \cdot 2^{2i+1}) \right]
$$

其中 $x_i, y_i$ 是 $x$ 和 $y$ 的比特位。

这种映射保持了层次化的邻近性：
如果两个点在交错形式下共享前 $k$ 个比特位，
那么它们位于同一个 $2^{-k}$ 大小的空间区域内。

#### 亲自尝试

1.  为 4×4 网格中的 8 个点 $(x, y)$ 写下二进制坐标。
2.  交错它们的比特位以获得莫顿码。
3.  按码排序，然后绘制点以观察 "Z" 形图案。
4.  观察邻近的点在其码中共享许多前导比特位。

#### 测试用例

| $(x, y)$ | 莫顿码 | 二进制 | 结果        |
| -------- | ----------- | ------ | ------------- |
| (0, 0)   | 0           | 0000   | 起点         |
| (1, 0)   | 1           | 0001   | 右         |
| (0, 1)   | 2           | 0010   | 上            |
| (1, 1)   | 3           | 0011   | 右上   |
| (2, 0)   | 4           | 0100   | 下一个象限 |

#### 复杂度

| 操作      | 时间                 | 空间  |
| -------------- | -------------------- | ------ |
| 编码 (2D)  | $O(b)$               | $O(1)$ |
| 排序        | $O(n \log n)$        | $O(n)$ |
| 查询局部性 | $O(1)$ (近似) |,      |

莫顿序（Z 曲线）是数学家的罗盘 ——
它描绘出一条单一的线，在网格的每个单元中舞动，
将多维世界折叠成一维的线，
却不忘谁与谁邻近。

# 第 77 节 光栅化与扫描线技术
### 761 布雷森汉姆直线算法

布雷森汉姆直线算法是计算机图形学中的基础算法，它仅使用整数运算在两点之间绘制一条直线。
它避免了浮点运算，因此既快速又精确，非常适合光栅显示器、像素艺术和嵌入式系统。

该算法由杰克·布雷森汉姆于1962年为早期的IBM绘图仪发明，至今仍是将连续几何转化为离散计算的最优雅范例之一。

#### 我们要解决什么问题？

我们想在像素网格上从点 $(x_0, y_0)$ 到点 $(x_1, y_1)$ 绘制一条直线。
但计算机只能点亮离散的像素，而不是连续的值。

一种简单的方法是计算 $y = m x + c$ 并对每个结果进行舍入，
但这使用了缓慢的浮点运算并且会累积舍入误差。

布雷森汉姆算法通过使用增量整数更新
和一个决策变量来选择下一个要点亮的像素来解决这个问题。

#### 工作原理（通俗解释）

想象一下，你从直线的一端走到另一端，每次走一个像素。
在每一步，你需要决定：

> “我应该直接向东走，还是向东北方向走？”

这个决定取决于真实直线与这两个候选像素之间的中点距离有多远。

布雷森汉姆使用一个决策参数 $d$ 来跟踪理想直线与光栅化路径之间的差异。

对于斜率 $0 \le m \le 1$ 的直线，算法工作流程如下：

1. 从 $(x_0, y_0)$ 开始
2. 计算差值：
   $$
   \Delta x = x_1 - x_0, \quad \Delta y = y_1 - y_0
   $$
3. 初始化决策参数：
   $$
   d = 2\Delta y - \Delta x
   $$
4. 对于从 $x_0$ 到 $x_1$ 的每个 $x$：

   * 绘制点 $(x, y)$
   * 如果 $d > 0$，则递增 $y$ 并更新
     $$
     d = d + 2(\Delta y - \Delta x)
     $$
   * 否则，更新
     $$
     d = d + 2\Delta y
     $$

这个过程仅使用加法和减法来追踪直线。

#### 示例

让我们绘制一条从 $(2, 2)$ 到 $(8, 5)$ 的直线。

$$
\Delta x = 6, \quad \Delta y = 3
$$
初始 $d = 2\Delta y - \Delta x = 0$。

| 步骤 | (x, y) | d  | 操作                         |
| ---- | ------ | -- | -------------------------- |
| 1    | (2, 2) | 0  | 绘制                       |
| 2    | (3, 2) | +6 | $d>0$, 递增 y → (3,3) |
| 3    | (4, 3) | -6 | $d<0$, 保持                |
| 4    | (5, 3) | +6 | 递增 y → (5,4)        |
| 5    | (6, 4) | -6 | 保持                       |
| 6    | (7, 4) | +6 | 递增 y → (7,5)        |
| 7    | (8, 5) |,  | 完成                       |

绘制的直线点：(2,2), (3,3), (4,3), (5,4), (6,4), (7,5), (8,5)。

#### 微型代码（C 语言示例）

```c
#include <stdio.h>
#include <stdlib.h>

void bresenham_line(int x0, int y0, int x1, int y1) {
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    while (1) {
        printf("(%d, %d)\n", x0, y0);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx)  { err += dx; y0 += sy; }
    }
}
```

这个版本对称地处理了所有斜率和方向。

#### 为什么它很重要

布雷森汉姆算法是最早且最具影响力的光栅化方法之一。
时至今日，它仍被用于：

* 2D 和 3D 图形渲染器
* CAD 软件
* 打印机驱动程序和绘图仪
* 微控制器和显示系统
* 计算机科学中整数运算和几何的教学

它不仅仅是一个算法，更是几何与计算之间的桥梁。

#### 一个温和的证明（为什么它有效）

真实直线方程是 $y = m x + b$，其中 $m = \frac{\Delta y}{\Delta x}$。
两个候选像素之间的中点与真实直线相差一个误差 $\varepsilon$。
布雷森汉姆算法将这个误差的缩放版本作为 $d$ 进行跟踪，并将其加倍以避免分数：

$$
d = 2(\Delta y x - \Delta x y + C)
$$

当 $d > 0$ 时，中点位于真实直线下方，因此我们沿对角线步进。
当 $d < 0$ 时，中点位于真实直线上方，因此我们水平步进。
由于更新是常数时间的整数加法，因此保证了准确性和效率。

#### 亲自尝试

1. 在方格纸上绘制 $(0, 0)$ 和 $(10, 6)$ 之间的直线。
2. 手动应用更新规则，你会看到相同的模式出现。
3. 通过交换 x 和 y 的角色，修改算法以适应陡峭的斜率（$m > 1$）。
4. 可视化决策变量如何控制垂直步进。

#### 测试用例

| 点      | 斜率 | 绘制的像素      |
| ----------- | ----- | ----------------- |
| (0,0)-(5,2) | 0.4   | 平缓的直线       |
| (0,0)-(2,5) | >1    | 交换角色        |
| (2,2)-(8,5) | 0.5   | 经典测试      |
| (5,5)-(0,0) | -1    | 反向方向 |

#### 复杂度

| 操作 | 时间复杂度                     | 空间复杂度  |
| --------- | ------------------------ | ------ |
| 绘制直线 | $O(\Delta x + \Delta y)$ | $O(1)$ |

布雷森汉姆直线算法是像素世界里的诗人尺规——
它以精确度绘制，一次一个整数，
在数字画布上将代数转化为艺术。
### 762 中点圆算法

中点圆算法是 Bresenham 直线算法的圆形对应物。它通过利用圆的对称性和一个巧妙的中点决策规则，仅使用整数运算（无需三角函数，无需浮点计算）绘制完美的圆。

该算法是经典光栅图形的核心，驱动着从复古游戏到低级图形库和显示驱动程序的一切。

#### 我们要解决什么问题？

我们希望在离散的像素网格上绘制一个圆心为 $(x_c, y_c)$、半径为 $r$ 的圆。圆的方程为：

$$
x^2 + y^2 = r^2
$$

最简单的方法是，我们可以使用公式 $y = \sqrt{r^2 - x^2}$ 从 $x$ 计算每个 $y$，但这需要缓慢的平方根和浮点运算。

中点圆算法通过一种增量的、基于整数的方法消除了这些需求。

#### 工作原理（通俗解释）

1.  从最顶端的点 $(0, r)$ 开始。
2.  沿 x 方向向外移动，并在每一步根据哪个像素的中心更接近真实圆来决定是向南移动还是向东南移动。
3.  利用圆的对称性，每次迭代绘制八个点——围绕圆的每个八分圆各一个。

该算法依赖于一个决策变量 $d$，它衡量中点距离圆边界的远近。

#### 逐步推导

在每一步，我们评估圆函数：

$$
f(x, y) = x^2 + y^2 - r^2
$$

我们想知道候选像素之间的中点是在圆内还是圆外。决策参数在我们移动时增量更新。

1.  初始化：
   $$
   x = 0, \quad y = r
   $$
   $$
   d = 1 - r
   $$

2.  重复直到 $x > y$：

   * 绘制八个对称点：
     $(\pm x + x_c, \pm y + y_c)$ 和 $(\pm y + x_c, \pm x + y_c)$
   * 如果 $d < 0$，选择东 (E) 像素并更新
     $$
     d = d + 2x + 3
     $$
   * 否则，选择东南 (SE) 像素并更新
     $$
     d = d + 2(x - y) + 5, \quad y = y - 1
     $$
   * 在两种情况下，都递增 $x = x + 1$

#### 示例

圆心 $(0, 0)$，半径 $r = 5$。

| 步骤 | (x, y) | d  | 操作          |
| ---- | ------ | -- | ------------ |
| 0    | (0, 5) | -4 | E → (1, 5)   |
| 1    | (1, 5) | -1 | E → (2, 5)   |
| 2    | (2, 5) | +4 | SE → (3, 4)  |
| 3    | (3, 4) | +1 | SE → (4, 3)  |
| 4    | (4, 3) | +7 | SE → (5, 2)  |
| 5    | (5, 2) |,  | 停止 (x > y) |

为每次迭代绘制八个对称点即可完成圆。

#### 微型代码（C 语言示例）

```c
#include <stdio.h>

void midpoint_circle(int xc, int yc, int r) {
    int x = 0, y = r;
    int d = 1 - r;

    while (x <= y) {
        // 8 个对称点
        printf("(%d,%d) (%d,%d) (%d,%d) (%d,%d)\n",
               xc + x, yc + y, xc - x, yc + y,
               xc + x, yc - y, xc - x, yc - y);
        printf("(%d,%d) (%d,%d) (%d,%d) (%d,%d)\n",
               xc + y, yc + x, xc - y, yc + x,
               xc + y, yc - x, xc - y, yc - x);

        if (d < 0) {
            d += 2 * x + 3;
        } else {
            d += 2 * (x - y) + 5;
            y--;
        }
        x++;
    }
}
```

#### 为什么它很重要

中点圆算法用于：

*   低级图形库（例如 SDL、OpenGL 光栅化器基础）
*   嵌入式系统和显示固件
*   数字艺术和游戏中用于绘制圆和圆弧
*   对称性和整数几何示例的几何推理

它与 Bresenham 直线算法完美配对，两者都基于离散的决策逻辑而非连续数学。

#### 一个温和的证明（为什么它有效）

中点测试评估两个候选像素之间的中点是在理想圆内还是圆外：

如果 $f(x + 1, y - 0.5) < 0$，中点在圆内 → 选择 E。
否则，中点在圆外 → 选择 SE。

通过重新排列项，推导出增量更新公式：

$$
d_{k+1} =
\begin{cases}
d_k + 2x_k + 3, & \text{如果 } d_k < 0 \\
d_k + 2(x_k - y_k) + 5, & \text{如果 } d_k \ge 0
\end{cases}
$$

由于所有项都是整数，因此可以使用整数运算精确地对圆进行光栅化。

#### 亲自尝试

1.  绘制一个圆心在 $(0,0)$、$r=5$ 的圆。
2.  使用上述规则逐步计算 $d$。
3.  在每次迭代时标记八个对称点。
4.  与数学上的圆进行比较，它们完美对齐。

#### 测试用例

| 圆心   | 半径 | 绘制的点数 | 对称性 |
| -------- | ------ | ------------ | -------- |
| (0, 0)   | 3      | 24           | 完美     |
| (10, 10) | 5      | 40           | 完美     |
| (0, 0)   | 10     | 80           | 完美     |

#### 复杂度

| 操作       | 时间   | 空间  |
| ----------- | ------ | ------ |
| 绘制圆 | $O(r)$ | $O(1)$ |

中点圆算法是几何学中安静的工匠——它仅用整数和对称性就绘制出完美的循环，将纯粹的方程转化为方形网格上像素的舞蹈。
### 763 扫描线填充

扫描线填充算法是计算机图形学中一种经典的多边形填充技术。
它逐条水平线（或称*扫描线*）高效地为多边形内部着色。
该方法无需测试每个像素，而是确定每条扫描线进入和离开多边形的位置，并仅填充这些点之间的区域。

这种方法构成了光栅图形、渲染器以及矢量到像素转换的基础。

#### 我们要解决什么问题？

我们需要填充多边形内部，即其边界内的所有像素——使用一种在离散网格上运行的高效、确定性的过程。

一种暴力方法是测试每个像素是否在多边形内部（使用射线投射或环绕数规则），但这非常耗时。

扫描线填充算法利用交点将这个问题转化为逐行填充问题。

#### 工作原理（通俗解释）

1.  想象水平线从多边形的顶部到底部进行扫描。
2.  每条扫描线可能与多边形的边相交多次。
3.  规则是：
    *   填充成对交点（进入和离开多边形）之间的像素。

因此，每条扫描线变成了一系列简单的*开-关*区域：在每一对交替的 x 交点之间进行填充。

#### 逐步过程

1.  构建边表
    *   对于每条多边形边，记录：
        *   最小 y 值（起始扫描线）
        *   最大 y 值（结束扫描线）
        *   下端点的 x 坐标
        *   斜率倒数 ($1/m$)
    *   按最小 y 值排序存储这些边。

2.  初始化活动边表，开始时为空。

3.  对于每条扫描线 y：
    *   将 ET 中最小 y 值等于当前扫描线的边添加到 AET。
    *   从 AET 中移除最大 y 值等于当前扫描线的边。
    *   按当前 x 值对 AET 排序。
    *   填充每对 x 交点之间的像素。
    *   对于 AET 中的每条边，更新其 x 值：
      $$
      x_{\text{新}} = x_{\text{旧}} + \frac{1}{m}
      $$

4.  重复直到 AET 为空。

此过程能高效处理凸多边形和凹多边形。

#### 示例

多边形：顶点 $(2,2), (6,2), (4,6)$

| 边           | y_min | y_max | x_at_y_min | 1/m  |
| ------------ | ----- | ----- | ---------- | ---- |
| (2,2)-(6,2) | 2     | 2     | 2          |,    |
| (6,2)-(4,6) | 2     | 6     | 6          | -0.5 |
| (4,6)-(2,2) | 2     | 6     | 2          | +0.5 |

扫描线进度：

| y | 活动边                 | x 交点        | 填充区域           |
| - | ---------------------- | --------------- | ----------------- |
| 2 |,                      |,               | 边开始            |
| 3 | (2,6,-0.5), (2,6,+0.5) | x = 2.5, 5.5    | 填充 (3, 2.5→5.5) |
| 4 | ...                    | x = 3, 5        | 填充 (4, 3→5)     |
| 5 | ...                    | x = 3.5, 4.5    | 填充 (5, 3.5→4.5) |
| 6 |,                      |,               | 完成              |

#### 微型代码（Python 示例）

```python
def scanline_fill(polygon):
    # polygon = [(x0,y0), (x1,y1), ...]
    n = len(polygon)
    edges = []
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        if y0 == y1:
            continue  # 跳过水平边
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        inv_slope = (x1 - x0) / (y1 - y0)
        edges.append([y0, y1, x0, inv_slope])

    edges.sort(key=lambda e: e[0])
    y = int(edges[0][0])
    active = []

    while active or edges:
        # 添加新边
        while edges and edges[0][0] == y:
            active.append(edges.pop(0))
        # 移除已完成的边
        active = [e for e in active if e[1] > y]
        # 排序并找到交点
        x_list = [e[2] for e in active]
        x_list.sort()
        # 在成对交点之间填充
        for i in range(0, len(x_list), 2):
            print(f"在 y={y} 处填充，从 x={x_list[i]} 到 x={x_list[i+1]}")
        # 更新 x
        for e in active:
            e[2] += e[3]
        y += 1
```

#### 为什么它很重要

*   是 2D 渲染引擎中多边形光栅化的核心。
*   用于填充工具、图形 API 和硬件光栅化器。
*   能高效处理凹多边形和复杂多边形。
*   展示了增量更新和扫描线连贯性在图形学中的强大力量。

它是你屏幕上矢量图形区域填充或 CAD 软件为多边形着色的背后算法。

#### 一个温和的证明（为什么它有效）

多边形在每次穿过边时，其状态在*内部*和*外部*之间交替。
对于每条扫描线，在每对交点之间填充保证了：

$$
\forall x \in [x_{2i}, x_{2i+1}], \ (x, y) \text{ 在多边形内部。}
$$

由于我们只处理活动边并增量更新 x，每个操作对于每条边每条扫描线是 $O(1)$，从而在边数乘以扫描线数上得到总的线性复杂度。

#### 自己试试

1.  在方格纸上画一个三角形。
2.  对于每条水平线，标记它进入和离开三角形的位置。
3.  在这些交点之间填充。
4.  观察填充区域如何精确匹配多边形内部。

#### 测试用例

| 多边形         | 顶点数 | 填充扫描线数 |
| --------------- | -------- | ---------------- |
| 三角形        | 3        | 4                |
| 矩形       | 4        | 4                |
| 凹 L 形 | 6        | 8                |
| 复杂多边形 | 8        | 10–12            |

#### 复杂度

| 操作    | 时间复杂度       | 空间复杂度  |
| ------------ | ---------- | ------ |
| 填充多边形 | $O(n + H)$ | $O(n)$ |

其中 $H$ = 边界框内的扫描线数量。

扫描线填充算法就像用尺子作画——它逐行滑过画布，以平静的精确度填充每个空间，直到整个形状被牢固地填满。
### 764 边表填充

边表填充算法是扫描线多边形填充的一种优化且高效的形式。它使用显式的边表（ET）和活动边表（AET）来管理多边形边界，从而能够快速、结构性地填充即使是复杂的形状。

这种方法通常内置于图形硬件和渲染库中，因为它能最大限度地减少冗余工作，同时确保精确的多边形填充。

#### 我们要解决什么问题？

在使用扫描线填充多边形时，我们需要确切地知道每条扫描线在多边形的何处进入和退出。边表填充算法不再每次都重新计算交点，而是将边组织起来，以便在扫描线移动时增量式地更新交点。

边表填充算法改进了基本的扫描线填充，它将预先计算好的边数据存储在按 y 坐标分桶的桶中。

#### 工作原理（通俗解释）

1.  构建边表（ET），为每条边开始的扫描线 $y$ 设置一个桶。
2.  构建活动边表（AET），这是一个动态列表，包含与当前扫描线相交的边。
3.  对于每条扫描线 $y$：
    *   将 ET 中起始于 $y$ 的边添加到 AET。
    *   移除终止于 $y$ 的边。
    *   按当前 x 坐标对活动边进行排序。
    *   在成对的 x 值之间填充像素。
    *   使用边的斜率增量式地更新每条边的 x 坐标。

#### 边表（ET）结构

每条边存储以下信息：

| 字段 | 含义                                   |
| :--- | :------------------------------------- |
| y_max | 边终止的扫描线                         |
| x     | 在 y_min 处的 x 坐标                   |
| 1/m   | 斜率倒数（每步 y 的增量）              |

边被插入到与其起始 y_min 对应的 ET 桶中。

#### 逐步示例

考虑一个顶点为以下坐标的多边形：
$(3,2), (6,5), (3,8), (1,5)$

计算各边：

| 边            | y_min | y_max | x | 1/m  |
| :------------ | :---- | :---- | :- | :--- |
| (3,2)-(6,5) | 2     | 5     | 3 | +1   |
| (6,5)-(3,8) | 5     | 8     | 6 | -1   |
| (3,8)-(1,5) | 5     | 8     | 1 | +1   |
| (1,5)-(3,2) | 2     | 5     | 1 | 0.67 |

ET（按 y_min 分组）：

| y | 边                           |
| :- | :--------------------------- |
| 2 | [(5, 3, 1), (5, 1, 0.67)] |
| 5 | [(8, 6, -1), (8, 1, +1)]  |

然后从 y=2 开始扫描线填充。

每一步：
*   将 ET[y] 中的边添加到 AET。
*   按 x 对 AET 排序。
*   在成对的边之间填充。
*   通过 $x = x + 1/m$ 更新 x。

#### 微型代码（Python 示例）

```python
def edge_table_fill(polygon):
    ET = {}
    for i in range(len(polygon)):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i+1) % len(polygon)]
        if y0 == y1:
            continue
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        inv_slope = (x1 - x0) / (y1 - y0)
        ET.setdefault(int(y0), []).append({
            'ymax': int(y1),
            'x': float(x0),
            'inv_slope': inv_slope
        })

    y = min(ET.keys())
    AET = []
    while AET or y in ET:
        if y in ET:
            AET.extend(ET[y])
        AET = [e for e in AET if e['ymax'] > y]
        AET.sort(key=lambda e: e['x'])
        for i in range(0, len(AET), 2):
            x1, x2 = AET[i]['x'], AET[i+1]['x']
            print(f"在 y={y} 处填充线: 从 x={x1:.2f} 到 x={x2:.2f}")
        for e in AET:
            e['x'] += e['inv_slope']
        y += 1
```

#### 为何重要

边表填充算法是多边形光栅化的核心，应用于：
*   2D 图形渲染器（例如，OpenGL 的多边形管线）
*   用于填充矢量绘图的 CAD 系统
*   字体光栅化和游戏图形
*   GPU 扫描转换器

它减少了冗余计算，使其成为硬件或软件光栅化循环的理想选择。

#### 一个温和的证明（为何有效）

对于每条扫描线，AET 精确地维护了与该线相交的边集。由于每条边都是线性的，其交点 x 坐标每扫描线增加 $\frac{1}{m}$。因此，该算法确保了一致性：

$$
x_{y+1} = x_y + \frac{1}{m}
$$

交替填充规则（内部-外部）保证了每个内部像素被填充一次且仅一次。

#### 自己动手试试

1.  在方格纸上画一个五边形。
2.  创建一个包含 y_min、y_max、x 和 1/m 的边表。
3.  对于每条扫描线，标记进入和退出的 x 值，并在它们之间填充。
4.  将你填充的区域与精确的多边形进行比较，它们将完全匹配。

#### 测试用例

| 多边形   | 顶点数 | 类型              | 填充是否正确          |
| :------- | :----- | :---------------- | :-------------------- |
| 三角形  | 3        | 凸                | 是                    |
| 矩形 | 4        | 凸                | 是                    |
| 凹多边形   | 6        | 非凸        | 是                    |
| 星形      | 10       | 自相交 | 部分（取决于规则）    |

#### 复杂度

| 操作         | 时间复杂度 | 空间复杂度 |
| :----------- | :--------- | :--------- |
| 填充多边形 | $O(n + H)$ | $O(n)$     |

其中 $n$ 是边的数量，$H$ 是扫描线的数量。

边表填充算法是多边形填充领域训练有素的工匠——它像整理工具箱里的工具一样组织边，然后逐条扫描线稳步工作，将抽象的顶点转化为坚实、填充的形状。
### 765 Z-Buffer 算法

Z-Buffer 算法（或称深度缓冲）是现代 3D 渲染的基础。它通过比较深度（z 值）来确定每个像素上重叠的 3D 物体中哪个表面是可见的。

该算法简单、鲁棒，并且被广泛地硬件实现。你今天使用的每一块 GPU 每秒都会执行其某种变体数十亿次。

#### 我们要解决什么问题？

当将 3D 物体投影到 2D 屏幕上时，许多表面会沿着同一像素列重叠。我们需要决定哪一个离相机最近，因此是可见的。

简单的解决方案是对多边形进行全局排序，但对于相交或形状复杂的物体来说，这变得很困难。Z-Buffer 算法通过*逐像素*工作来解决这个问题，维护一个迄今为止最近物体的运行记录。

#### 工作原理（通俗解释）

其核心思想是维护两个与屏幕大小相同的缓冲区：

1.  **帧缓冲区（颜色缓冲区）**，存储每个像素的最终颜色。
2.  **深度缓冲区（Z-Buffer）**，存储迄今为止看到的最近表面的 z 坐标（深度）。

算法步骤：

1.  用一个大值（例如，无穷大）初始化深度缓冲区。
2.  对于每个多边形：
    *   计算其在屏幕上的投影。
    *   对于多边形内部的每个像素：
        *   计算其深度 z。
        *   如果 $z < z_{\text{buffer}}[x, y]$，则更新两个缓冲区：
          $$
          z_{\text{buffer}}[x, y] = z
          $$
          $$
          \text{frame}[x, y] = \text{polygon\_color}
          $$
3.  处理完所有多边形后，帧缓冲区就包含了可见的图像。

#### 逐步示例

假设我们在屏幕空间中渲染两个重叠的三角形：三角形 A（蓝色）和三角形 B（红色）。

对于给定的像素 $(x, y)$：
*   三角形 A 的深度为 $z_A = 0.45$
*   三角形 B 的深度为 $z_B = 0.3$

因为 $z_B < z_A$，所以来自三角形 B 的红色像素是可见的。

#### 数学细节

如果多边形是由以下方程给出的平面：
$$
ax + by + cz + d = 0,
$$
那么我们可以为每个像素计算 $z$ 值：
$$
z = -\frac{ax + by + d}{c}.
$$
在光栅化过程中，$z$ 值可以像颜色或纹理坐标一样，在多边形上进行增量插值。

#### 微型代码（C 语言示例）

```c
#include <stdio.h>
#include <float.h>

#define WIDTH 800
#define HEIGHT 600

typedef struct {
    float zbuffer[HEIGHT][WIDTH];
    unsigned int framebuffer[HEIGHT][WIDTH];
} Scene;

void clear(Scene* s) {
    for (int y = 0; y < HEIGHT; y++)
        for (int x = 0; x < WIDTH; x++) {
            s->zbuffer[y][x] = FLT_MAX; // 初始化为最大浮点数
            s->framebuffer[y][x] = 0; // 背景颜色
        }
}

void plot(Scene* s, int x, int y, float z, unsigned int color) {
    if (z < s->zbuffer[y][x]) {
        s->zbuffer[y][x] = z;
        s->framebuffer[y][x] = color;
    }
}
```

每个像素都将其新的深度与存储的深度进行比较，一个简单的 `if` 语句确保了正确的可见性。

#### 为何重要

*   用于所有现代 GPU（OpenGL, Direct3D, Vulkan）。
*   无需排序即可处理任意重叠的几何体。
*   与混合结合使用时，支持纹理映射、光照和透明度。
*   提供了逐像素精度的可见性模型，对于照片级真实感渲染至关重要。

#### 一个温和的证明（为何有效）

对于任意像素 $(x, y)$，可见表面是所有投影到该像素的多边形中 z 值最小的那个：
$$
z_{\text{visible}}(x, y) = \min_i z_i(x, y).
$$
通过在我们绘制时增量地检查和更新这个最小值，Z-Buffer 算法确保了更远的表面不会覆盖更近的表面。

因为深度缓冲区初始化为 $\infty$，所以每个像素的第一次写入都会成功，而后续的写入只有在更近时才会被有条件地替换。

#### 自己动手试试

1.  渲染两个具有不同 z 值的重叠矩形。
2.  以相反的顺序绘制它们，注意前面的矩形仍然显示在前面。
3.  可视化深度缓冲区，更近的表面具有更小的值（如果反向可视化则更亮）。

#### 测试用例

| 场景                         | 预期结果                 |
| ---------------------------- | ------------------------ |
| 两个重叠的三角形             | 最前面的可见             |
| 空间中旋转的立方体           | 面被正确遮挡             |
| 多个相交的物体               | 每个像素的可见性都正确   |

#### 复杂度

| 操作       | 时间复杂度      | 空间复杂度      |
| ---------- | --------------- | --------------- |
| 每像素操作 | $O(1)$          | $O(1)$          |
| 整帧操作   | $O(W \times H)$ | $O(W \times H)$ |

Z-Buffer 算法是每一幅渲染图像的沉默守护者——它监视着每个像素的深度，确保你看到的正是你虚拟世界中最近的东西。
### 766 画家算法

画家算法是三维图形学中最早、最简单的隐藏面消除方法之一。
它模仿画家的作画方式：先绘制远处的表面，然后在其上绘制更近的表面，直到最终可见的图像呈现出来。

尽管在现代系统中，它已很大程度上被 Z 缓冲区所取代，但其概念上依然优雅，并且在某些渲染管线及可视化任务中仍然有用。

#### 我们要解决什么问题？

当多个三维多边形在屏幕空间中重叠时，我们需要确定每个多边形的哪些部分应该可见。
与 Z 缓冲区中测试每个像素的深度不同，画家算法通过按深度排序后绘制整个多边形来解决此问题。

画家先画最远的墙，然后画更近的墙，这样更近的表面自然会覆盖其后面的表面。

#### 工作原理（通俗解释）

1.  计算每个多边形的平均深度（z）。
2.  按深度降序（最远优先）对所有多边形进行排序。
3.  将多边形逐个绘制到图像缓冲区中，更近的多边形会覆盖更远多边形的像素。

当对象不相交且其深度顺序一致时，此方法效果良好。

#### 逐步示例

想象三个在深度上堆叠的矩形：

| 多边形 | 平均 z | 颜色 |
| ------- | --------- | ----- |
| A       | 0.9       | 蓝色  |
| B       | 0.5       | 红色   |
| C       | 0.2       | 绿色 |

按 z 排序：A → B → C

按顺序绘制它们：

1.  绘制 A（蓝色，最远）
2.  绘制 B（红色，中间）
3.  绘制 C（绿色，最近）

结果：最近（绿色）的多边形遮挡了其他多边形的部分区域。

#### 处理重叠

如果两个多边形在投影中重叠且无法轻易进行深度排序（例如，它们相交或循环重叠），
则需要递归细分或采用混合方法：

1.  沿相交线分割多边形。
2.  对生成的片段重新排序。
3.  按正确顺序绘制它们。

这确保了可见性正确性，但代价是额外的几何计算。

#### 微型代码（Python 示例）

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

polygons = [
    {'points': [(1,1),(5,1),(3,4)], 'z':0.8, 'color':'skyblue'},
    {'points': [(2,2),(6,2),(4,5)], 'z':0.5, 'color':'salmon'},
    {'points': [(3,3),(7,3),(5,6)], 'z':0.2, 'color':'limegreen'},
$$

# 按 z 排序（最远优先）
sorted_polygons = sorted(polygons, key=lambda p: p['z'], reverse=True)

fig, ax = plt.subplots()
for p in sorted_polygons:
    ax.add_patch(Polygon(p['points'], closed=True, facecolor=p['color'], edgecolor='black'))
ax.set_xlim(0,8)
ax.set_ylim(0,7)
ax.set_aspect('equal')
plt.show()
```

这段代码从后向前绘制多边形，就像画家在画布上逐层叠加颜色一样。

#### 为什么它重要

*   直观，易于实现。
*   直接处理多边形层级的数据，无需逐像素深度比较。
*   用于 2D 渲染引擎、矢量图形和场景排序。
*   构成了更高级可见性算法的概念基础。

通常在以下情况下使用：

*   渲染顺序可以预先计算（无相交）。
*   模拟透明表面或简单的正交场景。

#### 一个温和的证明（为什么它有效）

设多边形 $P_1, P_2, ..., P_n$ 具有深度 $z_1, z_2, ..., z_n$。
如果对于 $P_i$ 位于 $P_j$ 后面的所有像素都有 $z_i > z_j$，那么按 $z$ 降序绘制确保：

$$
\forall (x, y): \text{color}(x, y) = \text{该像素处最近可见多边形的颜色}.
$$

这是因为在帧缓冲区中，后绘制的多边形会覆盖先绘制的多边形。

然而，当多边形相交时，这种深度顺序不具备传递性并会失效，因此需要细分或采用 Z 缓冲区等其他算法。

#### 亲自尝试

1.  在纸上绘制三个重叠的多边形。
2.  为每个多边形分配 z 值，并按从后到前的顺序排列它们。
3.  按该顺序"绘制"它们，观察近处的如何覆盖远处的。
4.  现在创建相交的形状，观察排序在何处失效。

#### 测试用例

| 场景                     | 能正确工作吗？          |
| ------------------------ | ------------------------- |
| 非重叠多边形 | 是                       |
| 嵌套多边形          | 是                       |
| 相交多边形    | 否（需要细分） |
| 透明多边形     | 是（使用 Alpha 混合） |

#### 复杂度

| 操作     | 时间复杂度          | 空间复杂度                              |
| ------------- | ------------- | ---------------------------------- |
| 多边形排序 | $O(n \log n)$ | $O(n)$                             |
| 多边形绘制 | $O(n)$        | $O(W \times H)$（用于帧缓冲区） |

画家算法捕捉了图形学的一个基本事实：
有时可见性不在于计算，而在于顺序——
一种逐层叠加直到场景呈现的艺术，一笔一画，皆是如此。
### 767 Gouraud 着色

Gouraud 着色是一种经典方法，用于在多边形表面上产生平滑的颜色过渡。它不是为整个面分配单一平坦的颜色，而是在顶点处插值颜色，并通过逐渐混合这些颜色来为每个像素着色。

这是最早将*平滑光照*引入计算机图形的算法之一，快速、优雅且易于实现。

#### 我们要解决什么问题？

平面着色为每个多边形分配统一的颜色。这看起来很人工，因为相邻多边形之间的边界清晰可见。

Gouraud 着色通过使颜色在表面上平滑变化来解决这个问题，模拟光线在弯曲物体上逐渐反射的效果。

#### 工作原理（通俗解释）

1.  计算顶点法线，即共享该顶点的所有面的法线的平均值。
2.  使用光照模型（通常是朗伯反射）计算顶点强度：

    $$
    I_v = k_d (L \cdot N_v) + I_{\text{ambient}}
    $$

    其中
    *   $L$ 是光照方向
    *   $N_v$ 是顶点法线
    *   $k_d$ 是漫反射率

3.  对于每个多边形：
    *   沿着每条扫描线插值顶点强度。
    *   通过水平插值强度来填充内部像素。

这以较低的计算成本在表面上提供了平滑的渐变。

#### 数学形式

设顶点强度为 $I_1, I_2, I_3$。对于任意内部点 $(x, y)$，其强度 $I(x, y)$ 通过重心插值计算：

$$
I(x, y) = \alpha I_1 + \beta I_2 + \gamma I_3
$$

其中 $\alpha + \beta + \gamma = 1$，且 $\alpha, \beta, \gamma$ 是 $(x, y)$ 相对于三角形的重心坐标。

#### 逐步示例

假设一个三角形的顶点强度为：

| 顶点 | 坐标    | 强度 |
| ---- | ------- | ---- |
| A    | (1, 1)  | 0.2  |
| B    | (5, 1)  | 0.8  |
| C    | (3, 4)  | 0.5  |

那么三角形内的每个点都会平滑地混合这些值，产生从 A 处的暗到 B 处的亮再到 C 处的中等亮度的渐变。

#### 微型代码（Python 示例）

```python
import numpy as np
import matplotlib.pyplot as plt

def barycentric(x, y, x1, y1, x2, y2, x3, y3):
    det = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / det
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / det
    c = 1 - a - b
    return a, b, c

# 三角形顶点和强度
x1, y1, i1 = 1, 1, 0.2
x2, y2, i2 = 5, 1, 0.8
x3, y3, i3 = 3, 4, 0.5

img = np.zeros((6, 7))
for y in range(6):
    for x in range(7):
        a, b, c = barycentric(x, y, x1, y1, x2, y2, x3, y3)
        if a >= 0 and b >= 0 and c >= 0:
            img[y, x] = a*i1 + b*i2 + c*i3

plt.imshow(img, origin='lower', cmap='inferno')
plt.show()
```

这段代码演示了如何基于顶点光照强度平滑地混合像素颜色。

#### 为什么它很重要

*   将逼真的着色引入了多边形图形。
*   构成了 OpenGL 和 Direct3D 中硬件光照的基础。
*   高效，所有操作都是线性插值，适合光栅化硬件。
*   在 Phong 着色普及之前，被用于 3D 建模软件和实时引擎中。

#### 一个温和的证明（为什么它有效）

如果在顶点计算光照并进行插值，平面上的强度呈线性变化。对于由顶点 $A, B, C$ 定义的三角形，任何内部点的光照强度满足：

$$
\nabla^2 I(x, y) = 0
$$

因为插值是线性的，因此在跨越边时是连续的。共享顶点的相邻多边形在这些顶点处具有匹配的强度，从而产生平滑的整体外观。

#### 自己动手试试

1.  创建一个三角形网格（甚至一个立方体）。
2.  通过平均面法线来计算顶点法线。
3.  对每个顶点使用公式 $I_v = k_d (L \cdot N_v)$。
4.  在每个三角形上插值顶点强度并可视化结果。

尝试旋转光照向量，你会看到着色如何动态变化。

#### 测试用例

| 模型          | 着色类型     | 视觉效果             |
| ------------- | ------------ | -------------------- |
| 立方体（平面）| 平面着色     | 刻面外观             |
| 立方体（Gouraud） | 平滑着色 | 混合的边缘           |
| 球体          | Gouraud      | 柔和的光照           |
| 地形          | Gouraud      | 自然的渐变光照       |

#### 复杂度

| 操作                 | 时间            | 空间           |
| -------------------- | --------------- | -------------- |
| 逐顶点光照计算       | $O(V)$          | $O(V)$         |
| 逐像素插值           | $O(W \times H)$ | $O(W \times H)$ |

Gouraud 着色算法是图形学真实感演进中的关键一步——它是几何形状和视觉平滑度之间的桥梁，光线在表面上柔和地滑过，而不是从一个面突然跳到另一个面。
### 768 Phong 着色

Phong 着色通过插值法向量而非强度值来改进 Gouraud 着色，从而在曲面上产生更精确的高光和更平滑的光照效果。
这是计算机图形学中实现真实感的一次突破，它优雅地捕捉了光泽反射、镜面高光和柔和的光线衰减。

#### 我们要解决什么问题？

Gouraud 着色在顶点之间插值颜色，如果高光（例如球体上的一个亮点）出现在顶点之间，这种方法可能会错过这些小而亮的高光。
Phong 着色通过为每个像素插值表面法向量，然后在每个像素上重新计算光照来解决这个问题。

这产生了更平滑、物理上更准确的结果，特别是对于弯曲和反射表面。

#### 工作原理（通俗解释）

1.  将顶点法向量计算为所有相邻面法向量的平均值。
2.  对于多边形内的每个像素：
    *   使用重心插值法插值法向量 $N(x, y)$。
    *   将其归一化为单位长度。
    *   使用 $N(x, y)$ 在该像素处应用光照方程。
3.  使用标准光照模型（如 Phong 反射模型）计算（每个像素的）光照：

    $$
    I(x, y) = k_a I_a + k_d (L \cdot N) I_l + k_s (R \cdot V)^n I_l
    $$

    其中
    *   $k_a, k_d, k_s$ 分别是环境光、漫反射和镜面反射系数
    *   $L$ 是光线方向
    *   $N$ 是像素处的表面法向量
    *   $R$ 是反射向量
    *   $V$ 是观察方向
    *   $n$ 是光泽度（镜面反射指数）

#### 逐步示例

1.  对于三角形的每个顶点，存储其法向量 $N_1, N_2, N_3$。
2.  对于三角形内的每个像素：
    *   使用以下公式插值 $N(x, y)$：
        $$
        N(x, y) = \alpha N_1 + \beta N_2 + \gamma N_3
        $$
    *   归一化：
        $$
        N'(x, y) = \frac{N(x, y)}{|N(x, y)|}
        $$
    *   使用 Phong 模型在该像素处计算光照。

高光强度在表面上平滑变化，产生真实的反射光斑。

#### 微型代码（Python 示例）

```python
import numpy as np

def normalize(v):
    return v / np.linalg.norm(v)

def phong_shading(N, L, V, ka=0.1, kd=0.7, ks=0.8, n=10):
    N = normalize(N)
    L = normalize(L)
    V = normalize(V)
    R = 2 * np.dot(N, L) * N - L
    I = ka + kd * max(np.dot(N, L), 0) + ks * (max(np.dot(R, V), 0)  n)
    return np.clip(I, 0, 1)
```

在每个像素处，插值 `N`，然后调用 `phong_shading(N, L, V)` 来计算其颜色强度。

#### 为什么它重要

*   产生视觉上平滑的着色和精确的镜面高光。
*   成为现代图形硬件中逐像素光照的基础。
*   无需增加多边形数量即可准确建模曲面。
*   非常适合光泽、金属或反射材质。

#### 一个温和的证明（为什么它有效）

光照是表面方向的一个非线性函数：镜面反射项 $(R \cdot V)^n$ 强烈依赖于局部角度。
通过插值法向量，Phong 着色保留了每个多边形内的这种角度变化。

从数学上讲，Gouraud 着色计算：

$$
I(x, y) = \alpha I_1 + \beta I_2 + \gamma I_3,
$$

而 Phong 着色计算：

$$
I(x, y) = f(\alpha N_1 + \beta N_2 + \gamma N_3),
$$

其中 $f(N)$ 是光照函数。
由于光照在 $N$ 上是非线性的，插值法向量能给出更忠实的近似。

#### 自己动手试试

1.  使用平面着色、Gouraud 着色和 Phong 着色渲染一个球体，比较结果。
2.  将单个光源放在一侧，只有 Phong 着色能捕捉到圆形的镜面高光。
3.  尝试调整 $n$（光泽度）：
    *   低 $n$ → 哑光表面。
    *   高 $n$ → 闪亮的反射。

#### 测试用例

| 模型     | 着色类型 | 结果                               |
| -------- | -------- | ---------------------------------- |
| 立方体   | 平面着色 | 分块的面                           |
| 球体     | Gouraud  | 平滑，但缺少高光                   |
| 球体     | Phong    | 平滑，带有明亮的镜面高光           |
| 汽车车身 | Phong    | 逼真的金属反射                     |

#### 复杂度

| 操作                 | 时间            | 空间           |
| -------------------- | --------------- | -------------- |
| 逐像素光照           | $O(W \times H)$ | $O(W \times H)$ |
| 法向量插值           | $O(W \times H)$ | $O(1)$         |

Phong 着色是从*平滑的颜色*到*平滑的光线*的飞跃。
通过引入逐像素光照，它连接了几何与光学——使表面闪耀、曲线流畅、反射如现实世界般闪烁。
### 769 抗锯齿（超采样）

抗锯齿技术能够平滑在像素网格上绘制对角线或曲线时出现的锯齿状边缘。最常见的方法——超采样抗锯齿（SSAA）——通过以更高分辨率渲染场景并对相邻像素进行平均来产生更平滑的边缘。

它是高质量图形的基石，将生硬的阶梯状边缘转变为柔和、连续的形状。

#### 我们正在解决什么问题？

数字图像由方形像素构成，但现实世界中的大多数形状并非如此。当我们渲染一条对角线或曲线时，像素化会产生可见的锯齿，即那些在移动时看起来粗糙或闪烁的“阶梯状”边缘。

锯齿源于欠采样——没有足够的像素样本来表现精细细节。抗锯齿通过增加采样密度或在区域之间进行混合来修复此问题。

#### 工作原理（通俗解释）

超采样为每个像素采集多个颜色样本并对它们进行平均：

1.  对于每个像素，将其划分为 $k \times k$ 个子像素。
2.  使用场景几何和着色计算每个子像素的颜色。
3.  平均所有子像素颜色以生成最终像素颜色。

这样，像素颜色反映了部分覆盖率，即像素被物体覆盖的部分相对于背景的比例。

#### 示例

想象一条黑色对角线穿过白色背景。如果一个像素被线条覆盖了一半，经过超采样后它将呈现灰色，因为白色（背景）和黑色（线条）子像素的平均值是灰色。

因此，你得到的是边缘处的平滑渐变，而不是生硬的过渡。

#### 数学形式

如果每个像素被划分为 $m$ 个子像素，最终颜色为：

$$
C_{\text{pixel}} = \frac{1}{m} \sum_{i=1}^{m} C_i
$$

其中 $C_i$ 是每个子像素样本的颜色。

$m$ 越高，图像越平滑，但代价是更多的计算量。

#### 逐步算法

1.  选择超采样因子 $s$（例如，2×2、4×4、8×8）。
2.  对于每个像素 $(x, y)$：
    *   对于每个子像素 $(i, j)$：
        $$
        x' = x + \frac{i + 0.5}{s}, \quad y' = y + \frac{j + 0.5}{s}
        $$
        *   计算 $(x', y')$ 处的颜色 $C_{ij}$。
    *   求平均：
        $$
        C(x, y) = \frac{1}{s^2} \sum_{i=0}^{s-1}\sum_{j=0}^{s-1} C_{ij}
        $$
3.  将 $C(x, y)$ 存储到最终帧缓冲区中。

#### 微型代码（Python 伪代码）

```python
import numpy as np

def supersample(render_func, width, height, s=4):
    image = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            color_sum = np.zeros(3)
            for i in range(s):
                for j in range(s):
                    x_sub = x + (i + 0.5) / s
                    y_sub = y + (j + 0.5) / s
                    color_sum += render_func(x_sub, y_sub)
            image[y, x] = color_sum / (s * s)
    return image
```

这里的 `render_func` 计算子像素的颜色，是渲染器的核心。

#### 为什么它很重要

*   减少锯齿状边缘（空间锯齿）。
*   当物体移动时提高运动平滑度（时间锯齿）。
*   增强整体图像真实感和视觉舒适度。
*   仍然是现代技术（如 MSAA、FXAA 和 TAA）的概念基础。

#### 一个温和的证明（为什么它有效）

当我们以低于奈奎斯特率的频率对连续信号（图像）进行采样时，就会出现锯齿——高频细节“折叠”成可见的伪影。

超采样提高了有效采样率，而平均操作充当了一个低通滤波器，去除了高于像素网格限制的频率。

从数学上讲，如果 $I(x, y)$ 是真实的图像强度，渲染的像素值变为：

$$
I_{\text{pixel}} = \frac{1}{A} \iint_{A} I(x, y) , dx, dy
$$

这是连续图像在像素区域上的*面积平均值*——一个关于真实显示器如何发射光线的物理精确模型。

#### 自己动手试试

1.  在白色 100×100 网格上渲染一条黑色对角线。
2.  不使用抗锯齿，观察锯齿状边缘。
3.  应用 4× 超采样（每个像素 2×2）。
4.  进行比较，边缘将显得更平滑、更自然。

#### 测试用例

| 图像           | 采样率 | 结果               |
| -------------- | ------ | ------------------ |
| 对角线         | 1×1    | 锯齿状边缘         |
| 对角线         | 2×2    | 明显更平滑         |
| 圆形轮廓       | 4×4    | 平滑的曲率         |
| 文本渲染       | 8×8    | 清晰易读           |

#### 复杂度

| 操作                 | 时间复杂度      | 空间复杂度        |
| -------------------- | --------------- | ----------------- |
| 超采样因子 $s$       | $O(s^2)$        | $O(W \times H)$   |
| 滤波                 | $O(W \times H)$ | $O(1)$            |

超采样抗锯齿将像素的硬几何形状柔化成眼睛感知为连续的东西。这就是数字画布学会低语曲线，而不仅仅是高喊方块的方式。
### 770 扫描线多边形裁剪

扫描线多边形裁剪是一种使用水平扫描（扫描线）方法将多边形修剪到给定窗口或视口的高效技术。
它用于二维渲染管线中，在光栅化之前裁剪多边形，只保留位于显示区域内的可见部分。

该算法结合了几何精度和光栅效率，逐行操作而非逐边操作。

#### 我们要解决什么问题？

在屏幕上绘制多边形时，可能只有部分多边形位于观察窗口内。
我们必须裁剪（切割）多边形，使得窗口外的像素不被绘制。

像 Sutherland–Hodgman 这样的经典多边形裁剪算法是逐边工作的。
而扫描线多边形裁剪则是按行（扫描线）操作，这与光栅化的工作方式相匹配——
使其更快且更容易集成到渲染管线中。

#### 工作原理（通俗解释）

1.  表示裁剪区域（通常是一个矩形）和要绘制的多边形。
2.  从上到下水平扫描一条扫描线。
3.  对于每条扫描线：
    *   找出多边形边与该扫描线的所有交点。
    *   按 x 坐标对交点进行排序。
    *   填充位于裁剪区域内的每*对*交点之间的像素。
4.  对窗口边界内的所有扫描线重复此过程。

这样，算法自然地裁剪了多边形——
因为只考虑了视口内的交点。

#### 示例

考虑一个与 10×10 窗口边缘重叠的三角形。
在扫描线 $y = 5$ 处，它可能与多边形边在 $x = 3$ 和 $x = 7$ 处相交。
像素 $(4, 5)$ 到 $(6, 5)$ 被填充；其他像素被忽略。

在下一条扫描线 $y = 6$ 处，交点可能移动到 $x = 4$ 和 $x = 6$，
自动形成裁剪后的内部区域。

#### 数学形式

对于连接 $(x_1, y_1)$ 和 $(x_2, y_2)$ 的每条多边形边，
使用线性插值找到与扫描线 $y = y_s$ 的交点：

$$
x = x_1 + (y_s - y_1) \frac{(x_2 - x_1)}{(y_2 - y_1)}
$$

只包含 $y_s$ 位于边垂直跨度内的交点。

对交点 $(x_1', x_2', x_3', x_4', ...)$ 排序后，
填充每对 $(x_1', x_2'), (x_3', x_4'), ...$ 之间的区域——
每对代表多边形的一个内部线段。

#### 微型代码（简化的 C 语言示例）

```c
typedef struct { float x1, y1, x2, y2; } Edge;

void scanline_clip(Edge *edges, int n, int ymin, int ymax, int width) {
    for (int y = ymin; y <= ymax; y++) {
        float inter[100]; int k = 0;
        for (int i = 0; i < n; i++) {
            float y1 = edges[i].y1, y2 = edges[i].y2;
            if ((y >= y1 && y < y2) || (y >= y2 && y < y1)) {
                float x = edges[i].x1 + (y - y1) * (edges[i].x2 - edges[i].x1) / (y2 - y1);
                inter[k++] = x;
            }
        }
        // 对交点进行排序
        for (int i = 0; i < k - 1; i++)
            for (int j = i + 1; j < k; j++)
                if (inter[i] > inter[j]) { float t = inter[i]; inter[i] = inter[j]; inter[j] = t; }

        // 在成对交点之间填充
        for (int i = 0; i < k; i += 2)
            for (int x = (int)inter[i]; x < (int)inter[i+1]; x++)
                if (x >= 0 && x < width) plot_pixel(x, y);
    }
}
```

此示例逐扫描线地裁剪并填充多边形。

#### 为什么它很重要

*   与光栅化完美集成，扫描线顺序相同。
*   避免了复杂的多边形裁剪数学计算。
*   在硬件管线和软件渲染器上都能高效工作。
*   至今仍用于嵌入式系统、2D 游戏和矢量图形引擎。

#### 一个温和的证明（为什么它有效）

当你水平移动穿过一条扫描线时，多边形边界在填充区域的"进入"和"退出"之间交替。
因此，交点总是成对出现，
填充它们之间的区域恰好再现了多边形的内部。

如果裁剪限制是 $x_{\min}$ 和 $x_{\max}$，
该算法只在这些边界内填充，
因此输出区域实际上是：

$$
\text{裁剪后的多边形} = P \cap [x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]
$$

确保了精确的裁剪，而无需重新进行几何计算。

#### 自己动手试试

1.  绘制一个部分位于矩形窗口外的多边形。
2.  从上到下移动一条水平线，每一步都标记交点。
3.  将它们成对连接，阴影区域就是裁剪后的多边形。
4.  观察裁剪区域如何自动修剪边缘。

#### 测试用例

| 多边形                     | 窗口   | 结果                           |
| -------------------------- | ------ | ------------------------------ |
| 三角形完全在内部           | 10×10  | 无变化                         |
| 正方形跨越边缘             | 10×10  | 在边界处被修剪                 |
| 星形部分在外部             | 10×10  | 仅渲染可见的内部               |
| 多边形完全在外部           | 10×10  | 无输出                         |

#### 复杂度

| 操作         | 时间复杂度      | 空间复杂度 |
| ------------ | --------------- | ---------- |
| 每条扫描线   | $O(E)$          | $O(E)$     |
| 总体         | $O(E \times H)$ | $O(E)$     |

其中 $E$ 是边的数量，$H$ 是高度（扫描线数量）。

扫描线多边形裁剪将几何降维到光栅本身的层面——
一条稳定的扫描线，只揭示真正属于屏幕的内容，
将多边形逐行转化为可见的艺术。

# 第 78 节 计算机视觉
### 771 Canny 边缘检测器

Canny 边缘检测器是计算机视觉中最具影响力的算法之一，用于精确且稳定地检测边缘。它结合了梯度分析、噪声抑制和非极大值抑制，能够从复杂图像中提取清晰、单像素宽的边缘。

该算法由 John F. Canny 于 1986 年提出，至今仍是边缘检测的黄金标准。

#### 我们要解决什么问题？

边缘标记了具有明显强度变化的对象或区域之间的边界。检测边缘对于对象识别、分割和形状分析等任务至关重要。

使用梯度或 Sobel 滤波器的简单边缘检测方法通常会产生噪声大、边缘粗或断裂的边缘。Canny 方法提供了：

*   低错误率（检测到真实边缘）
*   良好的定位（边缘位置精确）
*   最小响应（每条边缘只检测一次）

#### 工作原理（通俗解释）

Canny 算法包含五个概念性步骤：

1.  **噪声抑制**
    使用高斯滤波器平滑图像以减少高频噪声：
    $$
    I_s = I * G_{\sigma}
    $$
    其中 $G_{\sigma}$ 是标准差为 $\sigma$ 的高斯核。

2.  **梯度计算**
    使用偏导数计算强度梯度：
    $$
    G_x = \frac{\partial I_s}{\partial x}, \quad G_y = \frac{\partial I_s}{\partial y}
    $$
    然后计算梯度幅值和方向：
    $$
    M(x, y) = \sqrt{G_x^2 + G_y^2}, \quad \theta(x, y) = \arctan\left(\frac{G_y}{G_x}\right)
    $$

3.  **非极大值抑制**
    通过仅保留梯度方向上的局部最大值来细化边缘。
    对于每个像素，将 $M(x, y)$ 与沿 $\theta(x, y)$ 方向的邻居进行比较，仅在其值更大时保留。

4.  **双阈值处理**
    使用两个阈值 $T_{\text{高}}$ 和 $T_{\text{低}}$ 对像素进行分类：

    *   $M > T_{\text{高}}$：强边缘
    *   $T_{\text{低}} < M \leq T_{\text{高}}$：弱边缘
    *   $M \leq T_{\text{低}}$：非边缘

5.  **滞后边缘跟踪**
    连接到强边缘的弱边缘被保留；其他弱边缘被丢弃。
    这确保了真实边缘的连续性，同时过滤了噪声。

#### 逐步示例

对于灰度图像：

1.  使用 $5 \times 5$ 高斯滤波器平滑图像（$\sigma = 1.0$）。
2.  使用 Sobel 算子计算 $G_x$ 和 $G_y$。
3.  计算梯度幅值 $M$。
4.  抑制非极大值，仅保留局部峰值。
5.  应用阈值（例如，$T_{\text{低}} = 0.1$, $T_{\text{高}} = 0.3$）。
6.  利用连通性将弱边缘连接到强边缘。

最终结果：勾勒出真实结构的细长、连续的轮廓。

#### 微型代码（使用 NumPy 的 Python 示例）

```python
import cv2
import numpy as np

# 加载灰度图像
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# 应用高斯模糊
blur = cv2.GaussianBlur(img, (5, 5), 1.0)

# 计算梯度
Gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
Gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
mag = np.sqrt(Gx2 + Gy2)
angle = np.arctan2(Gy, Gx)

# 为简便起见，使用 OpenCV 的滞后阈值处理
edges = cv2.Canny(img, 100, 200)

cv2.imwrite("edges.jpg", edges)
```

这段代码利用 OpenCV 的内置流程，简洁地实现了所有五个阶段。

#### 为何重要

*   即使在有噪声的条件下也能可靠地检测边缘。
*   当结合插值实现时，可提供亚像素精度。
*   通过高斯平滑和滞后阈值，平衡了敏感性和噪声控制。
*   为更高级的视觉任务（如轮廓跟踪、特征提取和分割）奠定了基础。

#### 一个温和的证明（为何有效）

Canny 将边缘检测表述为一个优化问题，旨在寻找一个算子，该算子能在保持定位和最小响应的同时，最大化信噪比。

通过将边缘建模为受高斯噪声污染的强度斜坡，他推导出最优边缘检测器基于高斯函数的一阶导数：

$$
h(x) = -x e^{-\frac{x^2}{2\sigma^2}}
$$

因此，该算法的设计自然地平衡了平滑（以抑制噪声）和微分（以检测边缘）。

#### 亲自尝试

1.  将 Canny 算法应用于不同 $\sigma$ 值的照片，观察较大的 $\sigma$ 如何模糊小细节。
2.  尝试不同的阈值 $(T_{\text{低}}, T_{\text{高}})$。
    *   太低：噪声会表现为边缘。
    *   太高：真实边缘会消失。
3.  将 Canny 的结果与简单的 Sobel 或 Prewitt 滤波器进行比较。

#### 测试用例

| 图像类型         | $\sigma$ | 阈值        | 结果                     |
| ---------------- | -------- | ----------- | ------------------------ |
| 简单形状         | 1.0      | (50, 150)   | 清晰的边界               |
| 有噪声的纹理     | 2.0      | (80, 200)   | 干净的边缘               |
| 人脸照片         | 1.2      | (70, 180)   | 保留面部轮廓             |
| 卫星图像         | 3.0      | (100, 250)  | 大尺度轮廓               |

#### 复杂度

| 操作               | 时间复杂度      | 空间复杂度      |
| ------------------ | --------------- | --------------- |
| 梯度计算           | $O(W \times H)$ | $O(W \times H)$ |
| 非极大值抑制       | $O(W \times H)$ | $O(1)$          |
| 滞后跟踪           | $O(W \times H)$ | $O(W \times H)$ |

Canny 边缘检测器改变了计算机感知图像结构的方式——它是微积分、概率论和几何学的结合，在事物的边界中发现了美。
### 772 Sobel 算子

Sobel 算子是一种简单而强大的工具，用于图像中的边缘检测和梯度估计。它测量亮度在水平和垂直方向上的变化情况，生成一幅图像，其中边缘表现为高强度的区域。

尽管概念简单，它仍然是计算机视觉和数字图像处理领域的基石。

#### 我们要解决什么问题？

边缘是图像中强度发生急剧变化的地方，通常表示物体边界、纹理或特征。为了找到它们，我们需要一种方法来估计图像强度的梯度（变化率）。

Sobel 算子使用卷积掩码提供了该导数的离散近似，同时应用轻微的平滑来减少噪声。

#### 工作原理（通俗解释）

Sobel 方法使用两个 $3 \times 3$ 卷积核来估计梯度：

* 水平梯度 ($G_x$)：
  $$
  G_x =
  \begin{bmatrix}
  -1 & 0 & +1 \\
  -2 & 0 & +2 \\
  -1 & 0 & +1
  \end{bmatrix}
  $$

* 垂直梯度 ($G_y$)：
  $$
  G_y =
  \begin{bmatrix}
  +1 & +2 & +1 \\
  0 & 0 & 0 \\
  -1 & -2 & -1
  \end{bmatrix}
  $$

将这些核与图像进行卷积，即可得到 x 和 y 方向上的强度变化率。

#### 计算梯度

1. 对于每个像素 $(x, y)$：
   $$
   G_x(x, y) = (I * K_x)(x, y), \quad G_y(x, y) = (I * K_y)(x, y)
   $$
2. 计算梯度幅度：
   $$
   M(x, y) = \sqrt{G_x^2 + G_y^2}
   $$
3. 计算梯度方向：
   $$
   \theta(x, y) = \arctan\left(\frac{G_y}{G_x}\right)
   $$

高幅度值对应强边缘。

#### 逐步示例

对于一个小的 3×3 图像块：

| 像素 | 值   |     |
| ---- | ---- | --- |
| 10   | 10   | 10  |
| 10   | 50   | 80  |
| 10   | 80   | 100 |

与 $G_x$ 和 $G_y$ 卷积得到：

* $G_x = (+1)(80 - 10) + (+2)(100 - 10) = 320$
* $G_y = (+1)(10 - 10) + (+2)(10 - 80) = -140$

因此：
$$
M = \sqrt{320^2 + (-140)^2} \approx 349.3
$$

此处检测到一个强边缘。

#### 微型代码（Python 示例）

```python
import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# 计算 Sobel 梯度
Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# 幅度和角度
magnitude = np.sqrt(Gx2 + Gy2)
angle = np.arctan2(Gy, Gx)

cv2.imwrite("sobel_edges.jpg", np.uint8(np.clip(magnitude, 0, 255)))
```

#### 为什么它很重要

* 快速且易于实现。
* 对于光照良好、低噪声的图像能产生良好的边缘图。
* 构成许多更复杂算法（例如 Canny 边缘检测器）的核心部分。
* 非常适合机器人技术、医学成像和计算机视觉预处理中的特征提取。

#### 一个温和的证明（为什么它有效）

Sobel 核是具有内置平滑效果的偏导数的离散近似。对于图像强度函数 $I(x, y)$，连续导数为：

$$
\frac{\partial I}{\partial x} \approx I(x + 1, y) - I(x - 1, y)
$$

中心差分方案与垂直（或水平）权重 `[1, 2, 1]` 相结合，以抑制噪声并强调中心像素，使得 Sobel 对小的波动具有鲁棒性。

#### 亲自尝试

1.  分别对 $x$ 和 $y$ 应用 Sobel 滤波器。
2.  可视化 $G_x$（垂直边缘）和 $G_y$（水平边缘）。
3.  组合幅度以查看完整的边缘强度。
4.  尝试不同类型的图像：肖像、文本、自然场景。

#### 测试用例

| 图像类型               | 核大小 | 输出特性             |
| ---------------------- | ------ | -------------------- |
| 白色背景上的文本       | 3×3    | 清晰的字母边缘       |
| 风景                   | 3×3    | 良好的物体轮廓       |
| 有噪声的照片           | 5×5    | 轻微模糊但稳定       |
| 医学 X 光片            | 3×3    | 突出骨骼轮廓         |

#### 复杂度

| 操作                   | 时间            | 空间           |
| ---------------------- | --------------- | -------------- |
| 卷积                   | $O(W \times H)$ | $O(W \times H)$ |
| 幅度 + 方向            | $O(W \times H)$ | $O(1)$         |

Sobel 算子以其极致的简洁性而著称——一个小的 3×3 窗口揭示了光的几何结构，将细微的强度变化转化为定义形态和结构的边缘。
### 773 霍夫变换（直线检测）

霍夫变换是一种几何算法，用于检测图像中的直线、圆和其他参数化形状。它将图像空间中的边缘点转换到参数空间，在该空间中模式表现为峰值，从而使其对噪声和缺失数据具有鲁棒性。

对于直线检测，它是寻找图像中所有直线的最优雅方法之一，即使边缘是断裂或分散的。

#### 我们要解决什么问题？

在边缘检测（如 Canny 或 Sobel）之后，我们得到一组可能属于边缘的像素。但我们仍然需要找到连续的几何结构，特别是连接这些点的直线。

一种简单的方法是尝试直接在图像中拟合直线，但当边缘不完整时，这种方法不稳定。霍夫变换通过在变换空间中累积投票来解决这个问题，在该空间中所有可能的直线都可以被表示。

#### 工作原理（通俗解释）

笛卡尔坐标系中的一条直线可以写成
$$
y = mx + b,
$$
但这种形式对于垂直线（$m \to \infty$）会失效。因此，我们改用极坐标形式：

$$
\rho = x \cos \theta + y \sin \theta
$$

其中

* $\rho$ 是从原点到直线的垂直距离，
* $\theta$ 是 x 轴与直线法线之间的角度。

每个边缘像素 $(x, y)$ 代表所有可能穿过它的直线。在参数空间 $(\rho, \theta)$ 中，该像素对应一条正弦曲线。

当多条曲线相交时 → 该点 $(\rho, \theta)$ 代表一条由许多边缘像素支持的直线。

#### 逐步算法

1.  初始化一个累加器数组 $A[\rho, \theta]$（全部为零）。
2.  对于每个边缘像素 $(x, y)$：
    *   对于 $\theta$ 从 $0$ 到 $180^\circ$：
        $$
        \rho = x \cos \theta + y \sin \theta
        $$
        递增累加器单元 $A[\rho, \theta]$。
3.  找到所有投票数超过阈值的累加器峰值。每个峰值 $(\rho_i, \theta_i)$ 对应一条检测到的直线。
4.  将这些参数转换回图像空间以进行可视化。

#### 示例

假设三个点大致沿着一条对角线边缘。它们在 $(\rho, \theta)$ 空间中的每条正弦曲线都在 $(\rho = 50, \theta = 45^\circ)$ 附近相交 —— 因此在那里出现一个强投票。

该点对应于直线
$$
x \cos 45^\circ + y \sin 45^\circ = 50,
$$
或者在图像空间中，等价于 $y = -x + c$。

#### 微型代码（使用 OpenCV 的 Python 示例）

```python
import cv2
import numpy as np

# 读取并预处理图像
img = cv2.imread("edges.jpg", cv2.IMREAD_GRAYSCALE)

# 使用 Canny 获取边缘图
edges = cv2.Canny(img, 100, 200)

# 应用霍夫变换
lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

# 绘制检测到的直线
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for rho, theta in lines[:, 0]:
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
    x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
    cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite("hough_lines.jpg", output)
```

#### 为什么它很重要

*   即使存在间隙或噪声，也能检测直线、边界和轴线。
*   对缺失像素具有容忍性，直线通过共识而非连续性显现。
*   是许多任务的基础：
    *   自动驾驶汽车中的车道线检测
    *   文档对齐
    *   形状识别
    *   工业检测

#### 一个温和的证明（为什么它有效）

每个边缘像素为所有穿过它的直线提供证据。如果 $N$ 个点大致位于同一条直线上，它们的正弦曲线在 $(\rho, \theta)$ 空间中相交，产生一个大的投票数 $A[\rho, \theta] = N$。

这种相交特性有效地将图像空间中的共线性转化为参数空间中的集中性，从而允许通过简单的阈值处理进行检测。

形式上：
$$
A(\rho, \theta) = \sum_{x, y \in \text{edges}} \delta(\rho - x \cos \theta - y \sin \theta)
$$

$A$ 中的峰值对应于主要的线性结构。

#### 亲自尝试

1.  在一个简单形状（例如矩形）上运行 Canny 边缘检测。
2.  应用霍夫变换并可视化累加器峰值。
3.  更改投票阈值，观察较小或较弱的直线如何出现/消失。
4.  尝试不同的 $\Delta \theta$ 分辨率，权衡精度与速度。

#### 测试用例

| 图像             | 预期直线数 | 备注                                 |
| ---------------- | ---------- | ------------------------------------ |
| 正方形形状       | 4          | 检测所有边缘                         |
| 道路照片         | 2–3        | 找到车道线                           |
| 网格图案         | 许多       | 累加器中出现规则的峰值               |
| 有噪声的背景     | 很少       | 只有强烈且一致的边缘得以保留         |

#### 复杂度

| 操作             | 时间                 | 空间                 |
| ---------------- | -------------------- | -------------------- |
| 投票累积         | $O(N \cdot K)$       | $O(R \times \Theta)$ |
| 峰值检测         | $O(R \times \Theta)$ | $O(1)$               |

其中

*   $N$ = 边缘像素数量
*   $K$ = 采样的 $\theta$ 值数量

霍夫变换将几何转化为统计学 —— 每个边缘像素投出它的票，当足够多的像素达成一致时，一条直线便从噪声中悄然浮现，清晰而确定。
### 774 霍夫变换（圆检测）

霍夫圆检测是霍夫直线检测的扩展，用于检测圆形形状。它不再寻找直线对齐，而是寻找位于可能圆周长上的点集。当圆形部分可见或被噪声遮挡时，这种方法尤其有用。

#### 我们要解决什么问题？

边缘检测为我们提供了边界候选像素，但我们通常需要检测特定的几何形状，如圆形、椭圆形或圆弧。圆形检测在以下任务中至关重要：

* 检测物体中的硬币、瞳孔或孔洞
* 识别道路标志和圆形标志
* 定位显微镜或天文学中的圆形图案

一个圆由其圆心 $(a, b)$ 和半径 $r$ 定义。目标是找到所有能拟合足够多边缘点的 $(a, b, r)$。

#### 工作原理（通俗解释）

一个圆可以表示为：
$$
(x - a)^2 + (y - b)^2 = r^2
$$

对于每个边缘像素 $(x, y)$，任何经过它的可能圆都满足这个方程。因此，对于给定的半径 $r$，每个 $(x, y)$ 为所有可能的圆心 $(a, b)$ 投票。

投票大量累积的地方 → 那就是圆心。

当半径未知时，算法在三维参数空间 $(a, b, r)$ 中搜索：

* $a$: 圆心的 x 坐标
* $b$: 圆心的 y 坐标
* $r$: 半径

#### 逐步算法

1.  **边缘检测**
    使用 Canny 或 Sobel 算子获取边缘图。

2.  **初始化累加器**
    创建一个三维数组 $A[a, b, r]$ 并填充为零。

3.  **投票过程**
    对于每个边缘像素 $(x, y)$ 和每个候选半径 $r$：

    * 计算可能的圆心：
      $$
      a = x - r \cos \theta, \quad b = y - r \sin \theta
      $$
      其中 $\theta$ 在 $[0, 2\pi]$ 范围内。
    * 增加累加器单元 $A[a, b, r]$ 的值。

4.  **寻找峰值**
    $A[a, b, r]$ 中的局部最大值表示检测到的圆。

5.  **输出**
    转换回图像空间，用检测到的 $(a, b, r)$ 绘制圆。

#### 示例

想象一个 100×100 的图像，其中有一个以 (50, 50) 为圆心、半径为 30 的边缘圆。每个边缘点都为对应于该半径的所有可能 $(a, b)$ 圆心投票。在 $(50, 50)$ 处，投票对齐并产生一个强峰值，从而揭示了圆心。

#### 微型代码（使用 OpenCV 的 Python 示例）

```python
import cv2
import numpy as np

# 读取灰度图像
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 100, 200)

# 检测圆
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1.2,
                           minDist=20, param1=100, param2=30,
                           minRadius=10, maxRadius=80)

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for (x, y, r) in circles[0, :]:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2) # 绘制圆
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3) # 绘制圆心

cv2.imwrite("hough_circles.jpg", output)
```

#### 为什么它很重要

* 即使部分可见也能检测到圆形物体。
* 对噪声和边缘间隙具有鲁棒性。
* 通过优化实现（例如 OpenCV 的 `HOUGH_GRADIENT`）高效处理不同半径范围。
* 在从机器人学到生物学再到天文学等各个领域都很有用。

#### 一个温和的证明（为什么它有效）

对于每个点 $(x, y)$，圆方程
$$
(x - a)^2 + (y - b)^2 = r^2
$$
描述了一组可能的圆心 $(a, b)$ 的轨迹。

通过累积来自许多点的投票，真正的圆心在参数空间中作为强交点出现。从数学上讲：
$$
A(a, b, r) = \sum_{x, y \in \text{edges}} \delta((x - a)^2 + (y - b)^2 - r^2)
$$

$A$ 中的峰值对应于被许多边缘点支持的圆。

#### 自己动手试试

1.  使用一个包含一个圆的简单图像，测试检测精度。
2.  添加高斯噪声，观察阈值如何影响结果。
3.  检测具有不同半径的多个圆。
4.  在真实图像（硬币、车轮、钟面）上尝试。

#### 测试用例

| 图像             | 半径范围 | 结果               | 备注                         |
| ---------------- | -------- | ------------------ | ---------------------------- |
| 合成圆           | 10–50    | 完美检测           | 简单的边缘模式               |
| 硬币照片         | 20–100   | 多重检测           | 重叠的圆                     |
| 钟表盘           | 30–80    | 清晰的边缘         | 即使部分圆弧也能工作         |
| 噪声图像         | 10–80    | 一些误报           | 可以调整 `param2`            |

#### 复杂度

| 操作         | 时间复杂度           | 空间复杂度           |
| ------------ | -------------------- | -------------------- |
| 投票         | $O(N \cdot R)$       | $O(A \cdot B \cdot R)$ |
| 峰值检测     | $O(A \cdot B \cdot R)$ | $O(1)$               |

其中：

* $N$ = 边缘像素数量
* $R$ = 测试的半径值数量
* $(A, B)$ = 可能的圆心坐标

霍夫圆检测让几何变得生动起来——每个像素对曲率的低语累积成形状的清晰声音，揭示隐藏在噪声中的圆以及图像中交织的几何结构。
### 775 Harris 角点检测器

Harris 角点检测器用于识别*角点*，即图像强度在多个方向上发生剧烈变化的点。这些点非常适合用于跨帧或跨视图的跟踪、匹配和模式识别。与边缘检测器（仅对单一方向的变化有响应）不同，角点检测器对两个方向的变化都有响应。

#### 我们要解决什么问题？

角点是稳定且独特的特征，是以下任务的理想地标：

*   物体识别
*   图像拼接
*   光流
*   三维重建

一个好的角点检测器应具备：

1.  可重复性（在不同光照/视角条件下都能找到）
2.  准确性（精确定位）
3.  高效性（计算速度快）

Harris 检测器利用图像梯度和一个简单的数学测试，实现了以上三点。

#### 工作原理（通俗解释）

想象一下在图像上移动一个小窗口。
如果窗口位于平坦区域，像素值几乎不变。
如果窗口沿着边缘移动，强度在*一个方向*上变化。
如果窗口位于角点，强度在*两个方向*上变化。

我们可以使用局部梯度信息来量化这种变化。

#### 数学公式

1.  对于一个以 $(x, y)$ 为中心的窗口，定义在偏移 $(u, v)$ 后的强度变化为：

    $$
    E(u, v) = \sum_{x, y} w(x, y) [I(x + u, y + v) - I(x, y)]^2
    $$

    其中 $w(x, y)$ 是高斯加权函数。

2.  对小偏移使用泰勒展开：

    $$
    I(x + u, y + v) \approx I(x, y) + I_x u + I_y v
    $$

    代入并简化后得到：

    $$
    E(u, v) = [u \ v]
    \begin{bmatrix}
    A & C \
    C & B
    \end{bmatrix}
    \begin{bmatrix}
    u \
    v
    \end{bmatrix}
    $$

    其中
    $A = \sum w(x, y) I_x^2$,
    $B = \sum w(x, y) I_y^2$,
    $C = \sum w(x, y) I_x I_y$.

    这个 $2\times2$ 矩阵
    $$
    M =
    \begin{bmatrix}
    A & C \
    C & B
    \end{bmatrix}
    $$
    被称为结构张量或二阶矩矩阵。

#### 角点响应函数

为了判断一个点是平坦区域、边缘还是角点，我们检查 $M$ 的特征值 $\lambda_1, \lambda_2$：

| 情况 | $\lambda_1$ | $\lambda_2$ | 类型 |
| ---- | ----------- | ----------- | ---- |
| 小   | 小          | 平坦区域    |      |
| 大   | 小          | 边缘        |      |
| 大   | 大          | 角点        |      |

为了避免显式计算特征值，Harris 提出了一个更简单的函数：

$$
R = \det(M) - k (\operatorname{trace}(M))^2
$$

其中
$\det(M) = AB - C^2$,
$\operatorname{trace}(M) = A + B$,
$k$ 通常取值在 $0.04$ 到 $0.06$ 之间。

如果 $R$ 是大的正数 → 角点。
如果 $R$ 是负数 → 边缘。
如果 $R$ 很小 → 平坦区域。

#### 微型代码（使用 OpenCV 的 Python 示例）

```python
import cv2
import numpy as np

img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)

img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv2.imwrite('harris_corners.jpg', img)
```

#### 为什么它很重要

*   检测稳定、独特的特征点用于匹配和跟踪。
*   简单且计算高效。
*   是现代检测器（如 Shi–Tomasi、FAST 和 ORB）的基础。
*   非常适合相机运动分析、SLAM 和立体视觉。

#### 一个温和的证明（为什么它有效）

在一个真正的角点处，两个梯度方向都携带重要信息。结构张量 $M$ 通过其特征值捕捉了这些梯度。

当 $\lambda_1$ 和 $\lambda_2$ 都很大时，无论偏移方向如何，局部强度函数都会发生剧烈变化，这正是角点的定义。

响应函数 $R$ 通过 $\det(M)$ 和 $\operatorname{trace}(M)$ 间接测量了这种曲率，避免了昂贵的特征值计算，同时保留了其几何意义。

#### 自己动手试试

1.  将 Harris 检测器应用于棋盘格图像，这是角点的完美测试对象。
2.  改变参数 $k$ 和阈值，观察检测到的角点数量如何变化。
3.  尝试应用于自然图像或人脸，注意纹理丰富的区域会产生许多响应。

#### 测试用例

| 图像类型     | 预期角点数量 | 备注                                 |
| ------------ | ------------ | ------------------------------------ |
| 棋盘格       | ~80          | 清晰锐利的角点                       |
| 路标         | 4–8          | 强边缘，稳定角点                     |
| 自然场景     | 很多         | 纹理会产生多个响应                   |
| 模糊的照片   | 很少         | 随着梯度减弱，角点会消失             |

#### 复杂度

| 操作               | 时间            | 空间           |
| ------------------ | --------------- | -------------- |
| 梯度计算           | $O(W \times H)$ | $O(W \times H)$ |
| 张量 + 响应计算    | $O(W \times H)$ | $O(W \times H)$ |
| 非极大值抑制       | $O(W \times H)$ | $O(1)$         |

Harris 角点检测器寻找图像中光线弯曲最剧烈的地方——亮度的十字路口，那里信息密度达到顶峰，几何与感知在此悄然达成共识："这里有重要的东西。"
### 776 FAST 角点检测器

FAST（来自加速段测试的特征）角点检测器是 Harris 检测器的一个极速替代方案。
它跳过了繁重的矩阵运算，而是对每个像素周围使用简单的强度比较测试来确定其是否为角点。
FAST 因其卓越的速度和简单性，被广泛应用于实时应用，如 SLAM、AR 跟踪和移动视觉。

#### 我们要解决什么问题？

Harris 检测器虽然精确，但涉及为每个像素计算梯度和矩阵运算，对于大图像或实时图像来说代价高昂。
FAST 则测试一个像素的邻域是否在多个方向上表现出明显的亮度对比，
这是类角点行为的标志，但无需使用导数。

核心思想：

> 如果一个像素周围的一组像素的亮度明显高于或低于它（超过某个阈值），则该像素是一个角点。

#### 工作原理（通俗解释）

1.  考虑每个候选像素 $p$ 周围的一个由 16 个像素组成的圆。
    这些像素均匀分布（半径为 3 的 Bresenham 圆）。

2.  对于每个邻域像素 $x$，将其强度 $I(x)$ 与 $I(p)$ 进行比较：
    *   如果 $I(x) > I(p) + t$，则更亮
    *   如果 $I(x) < I(p) - t$，则更暗

3.  如果存在一个连续的弧段，包含 $n$ 个像素（通常 $n = 12$，总共 16 个），这些像素的强度都高于或都低于 $I(p)$ 超过阈值 $t$，则声明像素 $p$ 为角点。

4.  执行非极大值抑制，只保留最强的角点。

这个测试完全避免了浮点运算，因此非常适合嵌入式或实时系统。

#### 数学描述

令 $I(p)$ 为像素 $p$ 的强度，$S_{16}$ 为其周围的 16 个像素。
如果存在 $S_{16}$ 中 $n$ 个连续的像素 $x_i$ 满足以下条件之一，则 $p$ 是一个角点：

$$
I(x_i) > I(p) + t \quad \forall i
$$
或
$$
I(x_i) < I(p) - t \quad \forall i
$$

其中 $t$ 是一个固定的阈值。

#### 逐步算法

1.  预计算 16 个圆形偏移量。
2.  对于每个像素 $p$：
    *   比较四个关键像素（1, 5, 9, 13）以快速排除大多数候选像素。
    *   如果其中至少三个都更亮或都更暗，则进行完整的 16 像素测试。
3.  如果存在一个连续的像素段满足强度规则，则将 $p$ 标记为角点。
4.  应用非极大值抑制来精炼角点位置。

#### 微型代码（使用 OpenCV 的 Python 示例）

```python
import cv2

img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# 初始化 FAST 检测器
fast = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)

# 检测关键点
kp = fast.detect(img, None)

# 绘制并保存结果
img_out = cv2.drawKeypoints(img, kp, None, color=(0,255,0))
cv2.imwrite('fast_corners.jpg', img_out)
```

#### 为什么它很重要

*   极其快速和简单，无需梯度，无需矩阵运算。
*   适用于实时跟踪、移动 AR 和机器人导航。
*   用作更高级描述符（如 ORB）的基础。
*   角点响应完全基于强度对比，使其在低功耗硬件上效率很高。

#### 一个温和的证明（为什么它有效）

角点是亮度在多个方向上急剧变化的地方。
圆形测试通过要求中心周围存在一系列持续更亮或更暗的像素来模拟这一点。
如果强度仅在一个方向上变化，则连续条件不满足，该模式是边缘，而不是角点。

该测试有效地测量了多方向对比度，近似于与 Harris 相同的直觉，
但使用的是简单的整数比较，而不是微分分析。

#### 亲自尝试

1.  在高分辨率图像上运行 FAST；注意角点出现得有多快。
2.  增加或减少阈值 $t$ 以控制灵敏度。
3.  与 Harris 的结果进行比较，角点位置是否相似但计算更快？
4.  禁用 `nonmaxSuppression` 以查看原始响应图。

#### 测试用例

| 图像           | 阈值 | 检测到的角点 | 观察结果                     |
| -------------- | ---- | ------------ | ---------------------------- |
| 棋盘格         | 30   | ~100         | 检测非常稳定                 |
| 纹理墙         | 20   | 300–400      | 由于纹理导致高密度           |
| 自然照片       | 40   | 60–120       | 减少到强特征                 |
| 低对比度       | 15   | 很少         | 在平坦光照下失败             |

#### 复杂度

| 操作             | 时间            | 空间  |
| ---------------- | --------------- | ----- |
| 像素比较         | $O(W \times H)$ | $O(1)$ |
| 非极大值抑制     | $O(W \times H)$ | $O(1)$ |

运行时间仅取决于图像大小，与梯度或窗口大小无关。

FAST 角点检测器以速度和实用性换取了数学上的优雅。
它倾听每个像素周围的亮度节奏——
当这个节奏在许多方向上急剧变化时，它简单而高效地说：
"这里有一个角点。"
### 777 SIFT（尺度不变特征变换）

SIFT（尺度不变特征变换）算法能在图像中找到独特且可重复的关键点，对尺度、旋转和光照变化具有鲁棒性。
它不仅检测角点或斑点，还构建描述子——一种小的数字指纹，使得特征能在不同图像间进行匹配。
这使得 SIFT 成为图像拼接、三维重建和物体识别的基础。

#### 我们要解决什么问题？

像 Harris 或 FAST 这样的角点检测器只能在固定的尺度和方向上工作。
但在现实世界的视觉任务中，物体以不同的大小、角度和光照出现。

SIFT 通过检测对尺度和旋转不变的特征来解决这个问题。
其核心思想是：构建一个*尺度空间*，并定位在图像模糊的不同层级中持续存在的稳定模式（极值点）。

#### 工作原理（通俗解释）

该算法有四个主要阶段：

1.  **尺度空间构建**：使用高斯函数逐步模糊图像。
2.  **关键点检测**：在空间和尺度上寻找局部极值点。
3.  **方向分配**：计算梯度方向以实现旋转不变性。
4.  **描述子生成**：将局部梯度模式捕捉到一个 128 维向量中。

每一步都增强了不变性：首先是尺度，然后是旋转，最后是光照。

#### 1. 尺度空间构建

通过使用标准差 $\sigma$ 递增的高斯滤波器反复模糊图像来创建*尺度空间*。

$$
L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)
$$

其中

$$
G(x, y, \sigma) = \frac{1}{2\pi\sigma^2} e^{-(x^2 + y^2)/(2\sigma^2)}
$$

为了检测稳定的结构，计算高斯差分：

$$
D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma)
$$

高斯差分近似于高斯拉普拉斯算子，后者是一种斑点检测器。

#### 2. 关键点检测

如果一个像素在 $3\times3\times3$ 的邻域（跨越位置和尺度）内是局部最大值或最小值，则它是一个关键点。
这意味着它在空间和尺度上都大于或小于其 26 个邻居。

为了提高稳定性，会丢弃低对比度和类似边缘的点。

#### 3. 方向分配

对于每个关键点，计算局部图像梯度：

$$
m(x, y) = \sqrt{(L_x)^2 + (L_y)^2}, \quad \theta(x, y) = \tan^{-1}(L_y / L_x)
$$

在关键点周围的邻域内构建梯度方向（0–360°）的直方图。
该直方图的峰值定义了关键点的方向。
如果有多个强峰值，则分配多个方向。

这提供了旋转不变性。

#### 4. 描述子生成

对于每个已分配方向的关键点，取其周围的 $16 \times 16$ 区域，划分为 $4 \times 4$ 个单元。
对于每个单元，计算一个 8 个区间的梯度方向直方图，并用梯度幅值和高斯衰减进行加权。

这样就得到了 $4 \times 4 \times 8 = 128$ 个数字，即 SIFT 描述子向量。

最后，对描述子进行归一化以减少光照影响。

#### 微型代码（使用 OpenCV 的 Python 示例）

```python
import cv2

img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# 创建 SIFT 检测器
sift = cv2.SIFT_create()

# 检测关键点和描述子
kp, des = sift.detectAndCompute(img, None)

# 绘制关键点
img_out = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_features.jpg', img_out)
```

#### 为何重要

*   对尺度和旋转不变。
*   对噪声、光照和仿射变换具有鲁棒性。
*   构成许多现代特征匹配器（如 SURF、ORB、AKAZE）的基础。
*   对全景拼接、三维重建和定位至关重要。

#### 一个温和的证明（为何有效）

高斯尺度空间确保了关键点在尺度变化下持续存在。
因为高斯拉普拉斯算子对缩放具有不变性，在高斯差分中检测极值点有效地近似了这种行为。

分配主导梯度方向确保了旋转不变性：
$$
f'(x', y') = f(R_\theta x, R_\theta y)
$$
描述子的归一化直方图使其对光照缩放具有鲁棒性：
$$
\frac{f'(x, y)}{||f'(x, y)||} = \frac{k f(x, y)}{||k f(x, y)||} = \frac{f(x, y)}{||f(x, y)||}
$$

#### 亲自尝试

1.  在不同尺度下对同一物体运行 SIFT，观察一致的关键点。
2.  将图像旋转 45°，检查 SIFT 是否匹配对应的点。
3.  使用 `cv2.BFMatcher()` 可视化两幅图像之间的匹配。

#### 测试用例

| 场景                       | 预期匹配数 | 观察结果                     |
| -------------------------- | ---------- | ---------------------------- |
| 同一物体，不同缩放         | 50–100     | 稳定的匹配                   |
| 旋转视角                   | 50+        | 关键点得以保留               |
| 低光照                     | 30–60      | 梯度仍然可区分               |
| 不同物体                   | 0          | 描述子拒绝错误匹配           |

#### 复杂度

| 步骤                   | 时间复杂度                 | 空间复杂度                 |
| ---------------------- | -------------------------- | -------------------------- |
| 高斯金字塔构建         | $O(W \times H \times S)$   | $O(W \times H \times S)$   |
| DoG 极值点检测         | $O(W \times H \times S)$   | $O(W \times H)$            |
| 描述子计算             | $O(K)$                     | $O(K)$                     |

其中 $S$ = 每个八度的尺度数，$K$ = 关键点数量。

SIFT 算法捕捉的是在变换中幸存下来的视觉结构——
就像图像皮肤之下的骨骼，当它生长、转动或变暗时保持不变。
它看到的不是像素，而是*在变化中持续存在的模式*。
### 778 SURF（加速稳健特征）

SURF（加速稳健特征）算法是 SIFT 的一种精简、更快的替代方案。
它保留了对于尺度、旋转和光照的鲁棒性，但用盒式滤波器和积分图像替换了繁重的高斯运算，
使其成为跟踪和识别等近实时应用的理想选择。

#### 我们要解决什么问题？

SIFT 功能强大但计算成本高昂——
尤其是高斯金字塔和 128 维描述符。

SURF 通过以下方式解决这个问题：

*   使用积分图像实现恒定时间的盒式滤波。
*   使用 Hessian 行列式近似进行关键点检测。
*   压缩描述符以实现更快的匹配。

结果：以一小部分成本实现 SIFT 级别的精度。

#### 工作原理（通俗解释）

1.  使用近似的 Hessian 矩阵检测兴趣点。
2.  使用 Haar 小波响应分配方向。
3.  根据强度梯度构建描述符（但比 SIFT 更少、更粗略）。

每个部分都设计为使用整数运算并通过积分图像进行快速求和。

#### 1. 积分图像

积分图像允许快速计算盒式滤波器总和：

$$
I_{\text{int}}(x, y) = \sum_{i \le x, j \le y} I(i, j)
$$

然后，任何矩形区域的和都可以仅使用四次数组访问在 $O(1)$ 时间内计算出来。

#### 2. 关键点检测（Hessian 近似）

SURF 使用 Hessian 行列式来寻找类斑点区域：

$$
H(x, y, \sigma) =
\begin{bmatrix}
L_{xx}(x, y, \sigma) & L_{xy}(x, y, \sigma) \
L_{xy}(x, y, \sigma) & L_{yy}(x, y, \sigma)
\end{bmatrix}
$$

并计算行列式：

$$
\det(H) = L_{xx} L_{yy} - (0.9L_{xy})^2
$$

其中导数用不同大小的盒式滤波器近似。
在空间和尺度上的局部最大值被保留为关键点。

#### 3. 方向分配

对于每个关键点，在圆形区域内计算 $x$ 和 $y$ 方向上的 Haar 小波响应。
一个滑动方向窗口（通常宽 $60^\circ$）用于找到主导方向。

这确保了旋转不变性。

#### 4. 描述符生成

每个关键点周围的区域被划分为一个 $4 \times 4$ 的网格。
对于每个单元格，基于 Haar 响应计算四个特征：

$$
(v_x, v_y, |v_x|, |v_y|)
$$

这些特征被连接成一个 64 维的描述符（而 SIFT 是 128 维）。

为了更好的匹配，描述符被归一化：

$$
\hat{v} = \frac{v}{||v||}
$$

#### 微型代码（使用 OpenCV 的 Python 示例）

```python
import cv2

img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# 初始化 SURF（可能需要 nonfree 模块）
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

# 检测关键点和描述符
kp, des = surf.detectAndCompute(img, None)

# 绘制并保存结果
img_out = cv2.drawKeypoints(img, kp, None, (255,0,0), 4)
cv2.imwrite('surf_features.jpg', img_out)
```

#### 为什么它很重要

*   比 SIFT 更快，对模糊、尺度和旋转具有鲁棒性。
*   在物体识别、配准和跟踪方面表现良好。
*   降低的描述符维度（64）实现了更快的匹配。
*   可以在移动和嵌入式硬件上高效运行。

#### 一个温和的证明（为什么它有效）

Hessian 矩阵的行列式捕捉了局部曲率——
两个方向上强烈的正曲率表示一个斑点或角点状结构。
使用积分图像确保即使是大型滤波器也可以在恒定时间内计算：

$$
\text{BoxSum}(x_1, y_1, x_2, y_2) =
I_{\text{int}}(x_2, y_2) - I_{\text{int}}(x_2, y_1) - I_{\text{int}}(x_1, y_2) + I_{\text{int}}(x_1, y_1)
$$

因此，SURF 的加速直接来自于数学简化——
用盒式差分和替换卷积，而不丢失特征的几何本质。

#### 亲自尝试

1.  在同一图像上比较 SURF 和 SIFT 的关键点。
2.  调整 `hessianThreshold`，更高的值会产生更少但更稳定的关键点。
3.  在图像的旋转或缩放版本上进行测试以验证不变性。

#### 测试用例

| 图像           | 检测器阈值 | 关键点数量 | 描述符维度 | 备注                       |
| -------------- | ---------- | ---------- | ---------- | -------------------------- |
| 棋盘格         | 400        | 80         | 64         | 稳定的网格角点             |
| 风景           | 300        | 400        | 64         | 丰富的纹理                 |
| 旋转的物体     | 400        | 70         | 64         | 方向保持不变               |
| 噪声图像       | 200        | 200        | 64         | 仍能检测到稳定的斑点       |

#### 复杂度

| 步骤             | 时间            | 空间           |
| ---------------- | --------------- | --------------- |
| 积分图像         | $O(W \times H)$ | $O(W \times H)$ |
| Hessian 响应     | $O(W \times H)$ | $O(1)$          |
| 描述符           | $O(K)$          | $O(K)$          |

其中 $K$ 是检测到的关键点数量。

SURF 算法以一半的时间捕捉了 SIFT 的精髓——
这是数学效率的壮举，
将连续高斯空间的优雅转化为一组快速、离散的滤波器，
从而敏锐而迅捷地观察世界。
### 779 ORB（Oriented FAST and Rotated BRIEF）

ORB（Oriented FAST and Rotated BRIEF）算法结合了 FAST 的速度与 BRIEF 的描述能力——产生了一种轻量级但高效的特征检测器和描述符。它专为 SLAM、AR 跟踪和图像匹配等实时视觉任务而设计，并且与 SIFT 或 SURF 不同，它是完全开源且无专利的。

#### 我们要解决什么问题？

SIFT 和 SURF 功能强大但计算成本高，并且历史上是有专利的。FAST 速度极快但缺乏方向信息或描述符。BRIEF 紧凑但不具备旋转不变性。

ORB 统一了所有三个目标：

*   FAST 关键点
*   旋转不变性
*   二进制描述符（用于快速匹配）

所有这些都在一个高效的流程中完成。

#### 工作原理（通俗解释）

1.  使用 FAST 检测角点。
2.  基于图像矩为每个关键点分配方向。
3.  围绕关键点计算旋转的 BRIEF 描述符。
4.  使用汉明距离进行匹配。

它同时具备旋转和尺度不变性，紧凑且速度极快。

#### 1. 关键点检测（FAST）

ORB 从 FAST 检测器开始寻找候选角点。

对于每个像素 $p$ 及其圆形邻域 $S_{16}$：

*   如果 $S_{16}$ 中至少有 12 个连续像素都比 $p$ 亮或暗一个阈值 $t$，那么 $p$ 就是一个角点。

为了提高稳定性，ORB 在高斯金字塔上应用 FAST，以捕获多个尺度上的特征。

#### 2. 方向分配

使用强度矩为每个关键点分配方向：

$$
m_{pq} = \sum_x \sum_y x^p y^q I(x, y)
$$

图像块的中心为：

$$
C = \left( \frac{m_{10}}{m_{00}}, \frac{m_{01}}{m_{00}} \right)
$$

方向由下式给出：

$$
\theta = \tan^{-1}\left(\frac{m_{01}}{m_{10}}\right)
$$

这确保了描述符可以与主导方向对齐。

#### 3. 描述符生成（旋转 BRIEF）

BRIEF（二进制鲁棒独立基本特征）通过对图像块中的像素对进行强度比较来构建二进制字符串。

对于关键点周围图像块中的 $n$ 个随机像素对 $(p_i, q_i)$：

$$
\tau(p_i, q_i) =
\begin{cases}
1, & \text{如果 } I(p_i) < I(q_i) \\
0, & \text{否则}
\end{cases}
$$

描述符是这些比特位的串联，通常长度为 256 位。

在 ORB 中，这种采样模式会根据关键点的方向 $\theta$ 进行旋转，从而实现旋转不变性。

#### 4. 匹配（汉明距离）

ORB 描述符是二进制字符串，因此特征匹配使用汉明距离——两个描述符之间不同比特位的数量。

这使得匹配通过按位异或操作变得极快。

#### 微型代码（使用 OpenCV 的 Python 示例）

```python
import cv2

img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# 初始化 ORB
orb = cv2.ORB_create(nfeatures=500)

# 检测关键点和描述符
kp, des = orb.detectAndCompute(img, None)

# 绘制结果
img_out = cv2.drawKeypoints(img, kp, None, color=(0,255,0))
cv2.imwrite('orb_features.jpg', img_out)
```

#### 为什么它重要

*   像 FAST 一样快，像 SIFT 一样具有描述性，像 BRIEF 一样紧凑。
*   二进制描述符使得匹配速度比 SIFT/SURF 快 10 倍。
*   完全免费和开源，非常适合商业用途。
*   SLAM、机器人和移动计算机视觉的核心组件。

#### 一个温和的证明（为什么它有效）

方向分配步骤确保了旋转不变性。令 $I'(x, y)$ 为 $I(x, y)$ 旋转角度 $\theta$ 后的版本。那么基于中心的定向保证了：

$$
BRIEF'(p_i, q_i) = BRIEF(R_{-\theta} p_i, R_{-\theta} q_i)
$$

这意味着同一个关键点在旋转后会产生相同的二进制描述符。

汉明距离是二进制向量的度量标准，因此即使在中等光照变化下，匹配仍然高效且鲁棒。

#### 自己动手试试

1.  在同一图像的两个旋转版本上检测 ORB 关键点。
2.  使用 `cv2.BFMatcher(cv2.NORM_HAMMING)` 来匹配特征。
3.  与 SIFT 比较速度，注意 ORB 运行有多快。
4.  增加 `nfeatures` 并测试准确性与运行时间之间的权衡。

#### 测试用例

| 场景             | 关键点数量 | 描述符长度 | 匹配速度 | 备注                     |
| ---------------- | ---------- | ---------- | -------- | ------------------------ |
| 棋盘格           | ~500       | 256 位     | 快       | 稳定的网格角点           |
| 旋转物体         | ~400       | 256 位     | 快       | 旋转不变性得以保持       |
| 低对比度         | ~200       | 256 位     | 快       | 对比度影响 FAST 检测     |
| 实时视频         | 300–1000   | 256 位     | 实时     | 可在嵌入式设备上运行     |

#### 复杂度

| 步骤               | 时间复杂度      | 空间复杂度 |
| ------------------ | --------------- | ---------- |
| FAST 检测          | $O(W \times H)$ | $O(1)$     |
| BRIEF 描述符       | $O(K)$          | $O(K)$     |
| 匹配（汉明距离）   | $O(K \log K)$   | $O(K)$     |

其中 $K$ = 关键点数量。

ORB 算法是计算机视觉中精明务实的混合体——它懂得 SIFT 的优雅、BRIEF 的节俭和 FAST 的迅捷。它行动迅速，感知旋转，按位高效，以硬件也青睐的速度捕捉结构。
### 780 RANSAC（随机抽样一致）

RANSAC（随机抽样一致）算法是一种鲁棒的方法，用于从包含异常值的数据中估计模型。
它反复将模型拟合到随机选取的点子集，并选择最能解释大多数数据的模型。
在计算机视觉中，RANSAC 是特征匹配、单应性估计和运动跟踪的基石，它能在噪声中找到结构。

#### 我们要解决什么问题？

现实世界的数据是混乱的。
当在两幅图像之间匹配点时，有些对应关系是错误的，这些就是异常值。
如果运行标准的最小二乘拟合，即使少数几个错误匹配也可能毁掉你的模型。

RANSAC 通过拥抱随机性来解决这个问题：
它测试许多小子集，相信共识而非任何单个样本的精确度。

#### 工作原理（通俗解释）

RANSAC 的思想很简单：

1.  随机选取一个最小数据点子集。
2.  用这个子集拟合一个模型。
3.  计算在容差范围内有多少其他点符合这个模型，这些点就是内点。
4.  保留具有最大内点集的模型。
5.  （可选）使用所有内点重新拟合模型以提高精度。

你不需要所有数据，只需要足够的共识。

#### 数学概述

设：

*   $N$ = 数据点总数
*   $s$ = 拟合模型所需的点数（例如，直线需要 $s=2$，单应性需要 $s=4$）
*   $p$ = 至少有一个随机样本不包含异常值的概率
*   $\epsilon$ = 异常值的比例

那么所需的迭代次数 $k$ 为：

$$
k = \frac{\log(1 - p)}{\log(1 - (1 - \epsilon)^s)}
$$

这告诉我们，对于给定的置信度，需要测试多少个随机样本。

#### 示例：直线拟合

给定二维点，我们想找到最佳直线 $y = mx + c$。

1.  随机选择两个点。
2.  计算斜率 $m$ 和截距 $c$。
3.  计算有多少其他点位于该直线距离 $d$ 的范围内：

$$
\text{error}(x_i, y_i) = \frac{|y_i - (mx_i + c)|}{\sqrt{1 + m^2}}
$$

4.  选择具有最多内点的直线作为最佳直线。

#### 微型代码（Python 示例）

```python
import numpy as np
import random

def ransac_line(points, n_iter=1000, threshold=1.0):
    best_m, best_c, best_inliers = None, None, []
    for _ in range(n_iter):
        sample = random.sample(points, 2)
        (x1, y1), (x2, y2) = sample
        if x2 == x1:
            continue
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1
        inliers = []
        for (x, y) in points:
            err = abs(y - (m*x + c)) / np.sqrt(1 + m2)
            if err < threshold:
                inliers.append((x, y))
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_m, best_c = m, c
    return best_m, best_c, best_inliers
```

#### 为什么它很重要

*   对异常值鲁棒，即使 50–80% 的数据是坏的也能工作。
*   与模型无关，可以拟合直线、平面、基础矩阵、单应性等。
*   简单灵活，只需要一个模型拟合例程和一个误差度量。

应用广泛，例如：

*   图像拼接（单应性估计）
*   立体视觉（对极几何）
*   三维重建
*   机器人运动估计

#### 一个温和的证明（为什么它有效）

每个随机子集仅包含内点的概率为 $(1 - \epsilon)^s$。
经过 $k$ 次迭代后，没有样本是纯净的概率为 $(1 - (1 - \epsilon)^s)^k$。
令其等于 $1 - p$ 就得到了上面的迭代公式。

因此，经过足够多的随机试验后，RANSAC 几乎肯定能找到得到大多数点支持的模型——
而不会被少数异常值所左右。

#### 自己动手试试

1.  生成一个包含 20% 异常值的带噪声数据集，并使用 RANSAC 拟合一条直线。
2.  与最小二乘法比较，注意 RANSAC 如何保持稳定。
3.  使用 `cv2.findHomography(..., cv2.RANSAC)` 将其应用于两张照片之间的特征匹配。

#### 测试用例

| 数据                | 异常值比例 | 模型                  | 内点率 | 备注                   |
| ------------------- | ---------- | --------------------- | ------ | ---------------------- |
| 直线点 + 噪声       | 20%        | $y = mx + c$          | 95%    | 完美恢复               |
| 三维空间中的平面    | 40%        | $ax + by + cz + d = 0$| 90%    | 对坏数据鲁棒           |
| 单应性              | 50%        | 3×3 矩阵              | 85%    | 用于图像拼接           |
| 随机噪声            | 90%        | N/A                   | 低     | 无法收敛               |

#### 复杂度

| 步骤               | 时间复杂度     | 空间复杂度 |
| ------------------ | -------------- | ---------- |
| 抽样与拟合         | $O(k \cdot s)$ | $O(1)$     |
| 内点计数           | $O(kN)$        | $O(1)$     |

总体：$O(kN)$，其中 $k$ 取决于所需的置信度和异常值比例。

RANSAC 算法是怀疑论者发现真理的方式——
它忽略人群，倾听少数诚实的声音，
并不断抽样，直到共识在噪声中揭示出正确的直线。

# 第 79 节 空间路径规划
### 781 A* 搜索

A*（A-star）算法是迄今为止设计的最优雅、最高效的寻路算法之一。它能在图中找到起点节点和目标节点之间的最短路径，同时最小化总成本。从导航系统、机器人技术到游戏和人工智能规划，A* 算法无处不在，它通过巧妙运用启发式函数，在贪婪搜索和统一成本搜索之间取得了平衡。

#### 我们要解决什么问题？

给定一组相互连接的节点（或网格地图），以及节点之间的移动成本，我们希望找到从起点到目标点的最短、成本最低的路径。

与向所有方向探索的 Dijkstra（迪杰斯特拉）算法不同，A* 使用启发式函数来引导搜索朝向目标，速度要快得多，并且（在一定条件下）仍然保证能找到最优路径。

#### 工作原理（通俗解释）

A* 为每个节点维护两个关键量：

*   $g(n)$，从起点到该节点的成本
*   $h(n)$，从该节点到目标的估计成本（启发式函数）
*   $f(n) = g(n) + h(n)$，经过该节点的总估计成本

它不断扩展 $f(n)$ 值最低的节点，直到到达目标。启发式函数使搜索保持聚焦；$g$ 函数则确保最优性。

#### 逐步算法

1.  初始化两个集合：
    *   **开放列表**：待评估的节点（初始时只有起点节点）
    *   **封闭列表**：已评估过的节点

2.  对于当前节点：
    *   计算 $f(n) = g(n) + h(n)$
    *   在开放列表中选择 $f(n)$ 最低的节点
    *   将其移动到封闭列表

3.  对于每个邻居节点：
    *   计算试探性的 $g_{new} = g(\text{当前节点}) + \text{cost(当前节点, 邻居节点)}$
    *   如果邻居节点不在开放列表中，或者 $g_{new}$ 更小，则更新它：
        *   $g(\text{邻居节点}) = g_{new}$
        *   $f(\text{邻居节点}) = g_{new} + h(\text{邻居节点})$
        *   将其父节点设置为当前节点

4.  当目标节点被选中进行扩展时停止。

#### 启发式函数示例

| 领域               | 启发式函数 $h(n)$                                     | 性质         |   |           |   |            |
| ------------------ | ----------------------------------------------------- | ------------ | - | --------- | - | ---------- |
| 网格（4邻域）      | 曼哈顿距离 $                                  | x_1 - x_2  | + | y_1 - y_2 | $ | 可采纳的   |
| 网格（8邻域）      | 欧几里得距离 $\sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$       | 可采纳的     |   |           |   |            |
| 加权图             | 最小边权 × 剩余节点数                                  | 可采纳的     |   |           |   |            |

如果一个启发式函数从不高估到达目标的真实成本，那么它就是**可采纳的**。如果它同时还是**一致的**，那么 A* 算法就能保证在不重复访问节点的情况下找到最优路径。

#### 微型代码（Python 示例）

```python
import heapq

def a_star(start, goal, neighbors, heuristic):
    open_set = [(0, start)]
    came_from = {}
    g = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for next_node, cost in neighbors(current):
            new_g = g[current] + cost
            if next_node not in g or new_g < g[next_node]:
                g[next_node] = new_g
                f = new_g + heuristic(next_node, goal)
                heapq.heappush(open_set, (f, next_node))
                came_from[next_node] = current
    return None
```

#### 为什么它很重要

*   **最优且完备**（使用可采纳的启发式函数时）
*   **高效**，只探索有希望的路径
*   **广泛应用**于：
    *   GPS 导航
    *   视频游戏 AI（非玩家角色移动）
    *   机器人运动规划
    *   基于图的优化问题

A* 是一个绝佳的例子，展示了如何将实际成本和估计成本相结合的简单想法，产生出强大的实际效用。

#### 一个温和的证明（为什么它有效）

令 $f(n) = g(n) + h(n)$。如果 $h(n)$ 从不高估到达目标的真实距离，那么当目标节点第一次被选中进行扩展时，找到的路径必定具有最小成本。

形式化地说，对于可采纳的 $h$：

$$
h(n) \le h^*(n)
$$

其中 $h^*(n)$ 是到达目标的真实成本。因此，$f(n)$ 总是经过节点 $n$ 的总成本的下界，A* 算法永远不会错过全局最优路径。

#### 自己动手试试

1.  在二维网格上实现 A* 算法，将墙壁标记为障碍物。
2.  尝试不同的启发式函数（曼哈顿距离、欧几里得距离、零启发式）。
3.  与 Dijkstra 算法比较，注意 A* 扩展的节点更少。
4.  可视化开放列表和封闭列表，就像在地图上观看推理过程展开一样。

#### 测试用例

| 网格大小 | 障碍物     | 启发式函数     | 结果                     |
| -------- | ---------- | -------------- | ------------------------ |
| 5×5      | 无         | 曼哈顿距离     | 直线路径                 |
| 10×10    | 随机 20%   | 曼哈顿距离     | 找到绕行路径             |
| 50×50    | 迷宫       | 欧几里得距离   | 高效的最短路径           |
| 100×100  | 30%        | 零启发式（Dijkstra） | 较慢但路径相同           |

#### 复杂度

| 项     | 含义                               | 典型值                   |
| ------ | ---------------------------------- | ------------------------ |
| 时间   | 最坏情况 $O(E)$，通常远小于此值    | 取决于启发式函数         |
| 空间   | $O(V)$                             | 存储开放集和封闭集       |

使用可采纳的启发式函数时，在稀疏地图中，A* 算法的时间复杂度可以接近线性，这对于一个通用的最优搜索算法来说是非常高效的。

A* 算法在远见和纪律之间取得了平衡——它不会像 Dijkstra 算法那样漫无目的地游荡，也不会像贪婪最佳优先搜索那样盲目地跳跃。它进行**规划**，平衡对已走道路的认知和对前方道路的直觉。
### 782 网格上的 Dijkstra 算法

Dijkstra 算法是计算最短路径的经典基础。
在其基于网格的版本中，它按照成本递增的顺序系统地探索所有可达节点，保证到每个目的地的最短路径。
虽然 A* 算法加入了启发式，但 Dijkstra 算法完全基于累积距离运行，
使其在不知道目标方向或没有启发式信息时，成为无偏、最优路径查找的黄金标准。

#### 我们要解决什么问题？

给定一个二维网格（或任何加权图），
每个单元格通过边与其邻居相连，移动成本为 $w \ge 0$。
我们希望找到从源点到所有其他节点——或者到特定目标（如果存在的话）——的最小总成本路径。

Dijkstra 算法确保一旦一个节点的成本被最终确定，
就不存在到达该节点的更短路径。

#### 工作原理（通俗解释）

1. 将起始单元格的距离 $d$ 赋值为 0，其他所有单元格赋值为 $∞$。
2. 将起始单元格放入优先队列。
3. 重复弹出当前成本最低的节点。
4. 对于每个邻居，计算试探性距离：

   $$
   d_{new} = d_{current} + w(current, neighbor)
   $$

   如果 $d_{new}$ 更小，则更新邻居的距离并将其重新插入队列。
5. 继续直到所有节点都被处理完毕或到达目标。

每个节点恰好被"松弛"一次，
确保了效率和最优性。

#### 示例（4邻域网格）

考虑一个网格，其中水平或垂直移动的成本为 1：

$$
\text{起点} = (0, 0), \quad \text{目标} = (3, 3)
$$

每次扩展后，已知最小距离的波前向外扩展：

| 步骤 | 前沿单元格           | 成本 |
| ---- | -------------------- | ---- |
| 1    | (0,0)                | 0    |
| 2    | (0,1), (1,0)         | 1    |
| 3    | (0,2), (1,1), (2,0)  | 2    |
| ...  | ...                  | ...  |
| 6    | (3,3)                | 6    |

#### 微型代码（Python 示例）

```python
import heapq

def dijkstra_grid(start, goal, grid):
    rows, cols = len(grid), len(grid[0])
    dist = {start: 0}
    pq = [(0, start)]
    came_from = {}

    def neighbors(cell):
        x, y = cell
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                yield (nx, ny), 1  # 成本为 1

    while pq:
        d, current = heapq.heappop(pq)
        if current == goal:
            break
        for (nxt, cost) in neighbors(current):
            new_d = d + cost
            if new_d < dist.get(nxt, float('inf')):
                dist[nxt] = new_d
                came_from[nxt] = current
                heapq.heappush(pq, (new_d, nxt))
    return dist, came_from
```

#### 为什么它很重要

*   最优且完备，总能找到最短路径。
*   是现代算法（如 A*、Bellman–Ford 和 Floyd–Warshall）的基础。
*   用途广泛，适用于网格、网络和加权图。
*   确定性的，在没有启发式偏差的情况下探索所有同等好的路径。

应用于：

*   网络路由（例如，OSPF、BGP）
*   游戏 AI 中的探索区域
*   自主机器人的路径规划

#### 一个温和的证明（为什么它有效）

关键不变量：
当一个节点 $u$ 从优先队列中移除时，其最短路径距离 $d(u)$ 就是最终值。

证明概要：
如果存在一条到 $u$ 的更短路径，那么该路径上的某个中间节点 $v$ 将具有更小的试探性距离，
因此 $v$ 会在 $u$ 之前被提取出来。
因此，$d(u)$ 之后不可能再被改进。

这保证了在边权重非负情况下的最优性。

#### 亲自尝试

1.  在具有不同障碍物模式的网格上运行 Dijkstra 算法。
2.  修改边权重以模拟地形（例如，草地 = 1，泥地 = 3）。
3.  比较与 A* 算法探索的节点，注意 Dijkstra 算法如何均匀扩展，而 A* 算法则向目标聚焦。
4.  实现一个 8 方向版本，并测量路径差异。

#### 测试用例

| 网格    | 障碍物百分比 | 成本度量       | 结果                     |
| ------- | ------------ | -------------- | ------------------------ |
| 5×5     | 0            | 均匀           | 直线                     |
| 10×10   | 20           | 均匀           | 找到绕行路径             |
| 10×10   | 0            | 可变权重       | 遵循低成本路径           |
| 100×100 | 30           | 均匀           | 扩展所有可达单元格       |

#### 复杂度

| 操作                     | 时间                              | 空间           |
| ------------------------ | --------------------------------- | -------------- |
| 优先队列操作             | $O((V + E)\log V)$                | $O(V)$         |
| 网格遍历                 | $O(W \times H \log (W \times H))$ | $O(W \times H)$ |

在均匀成本的网格中，它的行为类似于具有加权精度的广度优先搜索。

Dijkstra 算法是算法世界中冷静而有序的探索者——
它按照完美的顺序向外行走，考虑每一条可能的道路，直到所有道路都被测量完毕，
保证每个目的地都能获得可能的最短、最公平的路径。
### 783 Theta*（任意角度寻路）

Theta* 是 A* 算法的一种扩展，允许在网格上进行任意角度的移动，从而产生比局限于 4 或 8 个方向的路径看起来更自然、更短的路径。
它弥合了离散网格搜索与连续几何优化之间的差距，使其成为机器人学、无人机导航以及智能体在开放空间中自由移动的游戏中的热门选择。

#### 我们要解决什么问题？

在经典的 A* 算法中，移动被限制在网格方向上（上、下、对角线）。
即使最优的几何路径是直线，A* 也会产生锯齿状的“阶梯”路线。

Theta* 通过检查节点之间的视线来消除此限制：
如果当前节点的父节点可以直接看到后继节点，
它就会直接连接它们，而无需遵循网格边，从而产生更平滑、更短的路径。

#### 工作原理（通俗解释）

Theta* 的工作方式类似于 A*，但修改了父节点连接的方式。

对于当前节点 `s` 的每个邻居 `s'`：

1.  检查 `parent(s)` 是否对 `s'` 有视线。
    *   如果是，则设置
        $$
        g(s') = g(parent(s)) + \text{dist}(parent(s), s')
        $$
        和
        $$
        parent(s') = parent(s)
        $$
2.  否则，行为类似于标准 A*：
    $$
    g(s') = g(s) + \text{dist}(s, s')
    $$
    和
    $$
    parent(s') = s
    $$
3.  使用以下公式更新优先队列：
    $$
    f(s') = g(s') + h(s')
    $$

这种简单的几何松弛提供了接近最优的连续路径，
而不会增加渐近复杂度。

#### 微型代码（Python 示例）

```python
import heapq, math

def line_of_sight(grid, a, b):
    # Bresenham 风格的直线检查
    x0, y0 = a
    x1, y1 = b
    dx, dy = abs(x1-x0), abs(y1-y0)
    sx, sy = (1 if x1 > x0 else -1), (1 if y1 > y0 else -1)
    err = dx - dy
    while True:
        if grid[x0][y0] == 1:
            return False
        if (x0, y0) == (x1, y1):
            return True
        e2 = 2*err
        if e2 > -dy: err -= dy; x0 += sx
        if e2 < dx: err += dx; y0 += sy

def theta_star(grid, start, goal, heuristic):
    rows, cols = len(grid), len(grid[0])
    g = {start: 0}
    parent = {start: start}
    open_set = [(heuristic(start, goal), start)]

    def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

    while open_set:
        _, s = heapq.heappop(open_set)
        if s == goal:
            path = []
            while s != parent[s]:
                path.append(s)
                s = parent[s]
            path.append(start)
            return path[::-1]
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx == dy == 0: continue
                s2 = (s[0]+dx, s[1]+dy)
                if not (0 <= s2[0] < rows and 0 <= s2[1] < cols): continue
                if grid[s2[0]][s2[1]] == 1: continue
                if line_of_sight(grid, parent[s], s2):
                    new_g = g[parent[s]] + dist(parent[s], s2)
                    if new_g < g.get(s2, float('inf')):
                        g[s2] = new_g
                        parent[s2] = parent[s]
                        f = new_g + heuristic(s2, goal)
                        heapq.heappush(open_set, (f, s2))
                else:
                    new_g = g[s] + dist(s, s2)
                    if new_g < g.get(s2, float('inf')):
                        g[s2] = new_g
                        parent[s2] = s
                        f = new_g + heuristic(s2, goal)
                        heapq.heappush(open_set, (f, s2))
    return None
```

#### 为什么它很重要

*   为智能体和机器人产生平滑、逼真的路径。
*   比基于网格的 A* 更接近欧几里得最短路径。
*   如果启发函数是一致的且网格具有统一成本，则保持可采纳性。
*   在开阔地、无人机导航、自动驾驶和即时战略游戏中表现良好。

#### 一个温和的证明（为什么它有效）

Theta* 修改了 A* 的父节点链接以减少路径长度：
如果 `parent(s)` 和 `s'` 有视线，
那么

$$
g'(s') = g(parent(s)) + d(parent(s), s')
$$

总是 ≤

$$
g(s) + d(s, s')
$$

因为直接连接更短或相等。
因此，Theta* 永远不会高估成本，在欧几里得距离和无障碍可见性假设下，它保留了 A* 的最优性。

#### 自己动手试试

1.  在障碍物较少的网格上运行 Theta*。
2.  将路径与 A* 进行比较：Theta* 产生平缓的对角线，而不是锯齿状的转角。
3.  增加障碍物密度，观察路径如何平滑适应。
4.  尝试不同的启发函数（曼哈顿距离 vs 欧几里得距离）。

#### 测试用例

| 地图类型         | A* 路径长度 | Theta* 路径长度 | 视觉平滑度       |
| ---------------- | ----------- | --------------- | ---------------- |
| 开放网格         | 28.0        | 26.8            | 平滑             |
| 稀疏障碍物       | 33.2        | 30.9            | 自然弧线         |
| 迷宫式           | 52.5        | 52.5            | 相等（被阻挡）   |
| 随机场地         | 41.7        | 38.2            | 更干净的运动     |

#### 复杂度

| 操作               | 时间                         | 空间     |
| ------------------ | ---------------------------- | -------- |
| 搜索               | $O(E \log V)$                | $O(V)$   |
| 视线检查           | $O(L)$ 平均每次扩展          |          |

Theta* 的运行复杂度接近 A*，但以微小的开销为代价换取了更平滑的路径和更少的转弯。

Theta* 是 A* 的几何感知进化：
它不仅关注成本，还关注*可见性*，
在别人只看到方格的地方编织直线——
将锯齿状的运动转变为优雅、连续的旅行。
### 784 跳点搜索（网格加速）

跳点搜索（JPS）是专门针对等代价网格的 A* 算法的优化。
它通过沿直线“跳跃”直到抵达一个重要的决策点（一个跳点），从而剪枝掉冗余节点。
其结果是找到与 A* 相同的最优路径，但速度要快得多，通常快数倍，且节点扩展更少。

#### 我们要解决什么问题？

在等代价网格上使用 A* 会扩展许多不必要的节点：
当在开阔区域直线移动时，A* 会逐个探索每个单元格。
但如果所有代价都相等，我们并不需要在每个单元格都停下——
只有当某些情况*发生变化*时（遇到障碍物或被迫转向）才需要。

JPS 通过跳过这些“无趣”的单元格来加速搜索，
同时保持完全的寻路最优性。

#### 工作原理（通俗解释）

1.  从当前节点出发，沿方向 $(dx, dy)$ 移动。
2.  继续沿该方向跳跃，直到：
    *   碰到障碍物，或者
    *   找到一个强制邻居（一个旁边有障碍物迫使转向的节点），或者
    *   到达目标点。
3.  每个跳点都被视为 A* 中的一个节点。
4.  从每个跳点出发，递归地向可能的方向进行跳跃。

这大大减少了需要考虑的节点数量，同时保持了正确性。

#### 精简代码（简化版 Python 实现）

```python
import heapq

def jump(grid, x, y, dx, dy, goal):
    rows, cols = len(grid), len(grid[0])
    while 0 <= x < rows and 0 <= y < cols and grid[x][y] == 0:
        if (x, y) == goal:
            return (x, y)
        # 强制邻居检查
        if dx != 0 and dy != 0:
            if (grid[x - dx][y + dy] == 1 and grid[x - dx][y] == 0) or \
               (grid[x + dx][y - dy] == 1 and grid[x][y - dy] == 0):
                return (x, y)
        elif dx != 0:
            if (grid[x + dx][y + 1] == 1 and grid[x][y + 1] == 0) or \
               (grid[x + dx][y - 1] == 1 and grid[x][y - 1] == 0):
                return (x, y)
        elif dy != 0:
            if (grid[x + 1][y + dy] == 1 and grid[x + 1][y] == 0) or \
               (grid[x - 1][y + dy] == 1 and grid[x - 1][y] == 0):
                return (x, y)
        x += dx
        y += dy
    return None

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def jump_point_search(grid, start, goal):
    open_set = [(0, start)]
    g = {start: 0}
    came_from = {}
    directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for dx, dy in directions:
            jp = jump(grid, current[0]+dx, current[1]+dy, dx, dy, goal)
            if not jp: continue
            new_g = g[current] + heuristic(current, jp)
            if new_g < g.get(jp, float('inf')):
                g[jp] = new_g
                f = new_g + heuristic(jp, goal)
                heapq.heappush(open_set, (f, jp))
                came_from[jp] = current
    return None
```

#### 为什么重要

*   产生与 A* 完全相同的**最优路径**，但访问的节点**少得多**。
*   非常适合大型开阔网格或导航网格。
*   保留了 A* 的**最优性**和**完备性**。

应用包括：

*   游戏 AI 寻路（特别是实时移动）
*   均匀环境中的模拟和机器人学
*   大规模地图路由

#### 一个温和的证明（为什么有效）

JPS 找到的每条路径都对应一条有效的 A* 路径。
关键观察是：
如果直线移动没有揭示任何新的邻居或强制选择，
那么中间节点就不会贡献任何额外的最优路径。

形式上，剪枝这些节点保留了所有最短路径，
因为它们可以通过跳点之间的线性插值来重建。
因此，在等代价条件下，JPS 在路径上与 A* 等价。

#### 亲自尝试

1.  在一个开阔的 100×100 网格上运行 A* 和 JPS。
    *   比较节点扩展次数和时间。
2.  添加随机障碍物，观察跳跃次数的变化。
3.  可视化跳点，它们出现在拐角和转向点。
4.  测量加速比：JPS 通常能将扩展次数减少 5 到 20 倍。

#### 测试用例

| 网格类型             | A* 扩展次数 | JPS 扩展次数 | 加速比          |
| -------------------- | ----------- | ------------ | --------------- |
| 50×50 开阔           | 2500        | 180          | 13.9×           |
| 100×100 开阔         | 10,000      | 450          | 22×             |
| 100×100 20% 障碍物   | 7,200       | 900          | 8×              |
| 迷宫                 | 4,800       | 4,700        | 1× (与 A* 相同) |

#### 复杂度

| 术语         | 时间复杂度      | 空间复杂度 |
| ------------ | --------------- | ---------- |
| 平均情况     | $O(k \log n)$   | $O(n)$     |
| 最坏情况     | $O(n \log n)$   | $O(n)$     |

JPS 的性能提升很大程度上取决于障碍物的布局——
障碍物越少，加速效果越显著。

跳点搜索是搜索剪枝的典范——
它认识到直线路径已经是最优的，
跳过了均匀探索的单调过程，
只在必须做出真正决策时才向前跃进。
### 785 快速探索随机树（RRT）

快速探索随机树（RRT）算法是机器人学和自主导航中运动规划的基石。
它通过在空间中随机采样点，并将其连接到最近且能无碰撞到达的已知节点来构建一棵树。
RRT 在高维、连续的构型空间中特别有用，因为基于网格的算法在这些空间中效率低下。

#### 我们要解决什么问题？

在为机器人、车辆或机械臂规划运动时，构型空间可能是连续且复杂的。
RRT 不是将空间离散化为单元格，而是随机采样构型，并逐步探索可达区域，
最终找到一条从起点到目标点的有效路径。

#### 工作原理（通俗解释）

1.  从起点位置初始化一棵树 `T`。
2.  在构型空间中随机采样一个点 `x_rand`。
3.  在现有树中找到最近的节点 `x_near`。
4.  从 `x_near` 向 `x_rand` 移动一小步，得到 `x_new`。
5.  如果它们之间的线段是无碰撞的，则将 `x_new` 添加到树中，并以 `x_near` 作为其父节点。
6.  重复此过程，直到到达目标点或达到最大采样次数。

随着时间的推移，这棵树会迅速扩展到未探索的区域——
因此得名 *快速探索*。

#### 微型代码（Python 示例）

```python
import random, math

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def steer(a, b, step):
    d = distance(a, b)
    if d < step:
        return b
    return (a[0] + step*(b[0]-a[0])/d, a[1] + step*(b[1]-a[1])/d)

def rrt(start, goal, is_free, step=1.0, max_iter=5000, goal_bias=0.05):
    tree = {start: None}
    for _ in range(max_iter):
        x_rand = goal if random.random() < goal_bias else (random.uniform(0,100), random.uniform(0,100))
        x_near = min(tree.keys(), key=lambda p: distance(p, x_rand))
        x_new = steer(x_near, x_rand, step)
        if is_free(x_near, x_new):
            tree[x_new] = x_near
            if distance(x_new, goal) < step:
                tree[goal] = x_new
                return tree
    return tree

def reconstruct(tree, goal):
    path = [goal]
    while tree[path[-1]] is not None:
        path.append(tree[path[-1]])
    return path[::-1]
```

这里的 `is_free(a, b)` 是一个碰撞检查函数，用于确保点之间的运动是有效的。

#### 为什么它很重要

*   **可扩展至高维度**：适用于网格或 Dijkstra 算法变得不可行的空间。
*   **概率完备性**：如果存在解，随着采样次数的增加，找到解的概率趋近于 1。
*   **RRT* 和 PRM 算法的基础**。
*   **常见应用领域**：
    *   自主无人机和汽车导航
    *   机械臂运动规划
    *   游戏和仿真环境

#### 一个温和的证明（为什么它有效）

设 $X_{\text{free}}$ 为构型空间中无障碍物的区域。
在每次迭代中，RRT 从 $X_{\text{free}}$ 中均匀采样。
由于 $X_{\text{free}}$ 是有界的且具有非零测度，
每个区域都有被采样的正概率。

树的最近邻扩展确保了新节点总是向未探索区域靠近。
因此，随着迭代次数的增加，树到达目标区域的概率趋于 1——
这就是概率完备性。

#### 亲自尝试

1.  在带有圆形障碍物的 2D 网格上模拟 RRT。
2.  可视化树的扩展过程，它会从起点"扇形"展开到自由空间。
3.  添加更多障碍物，观察分支如何自然地绕过它们生长。
4.  调整 `step` 和 `goal_bias` 参数以获得更平滑或更快的收敛。

#### 测试用例

| 场景         | 空间     | 障碍物           | 成功率 | 平均路径长度 |
| ------------ | -------- | ---------------- | ------ | ------------ |
| 空旷空间     | 2D       | 0%               | 100%   | 140          |
| 20% 阻挡     | 2D       | 随机             | 90%    | 165          |
| 迷宫         | 2D       | 狭窄通道         | 75%    | 210          |
| 3D 空间      | 球形     | 30%              | 85%    | 190          |

#### 复杂度

| 操作                 | 时间复杂度                                  | 空间复杂度 |
| -------------------- | ------------------------------------------- | ---------- |
| 最近邻搜索           | $O(N)$（朴素），$O(\log N)$（KD 树）        | $O(N)$     |
| 总迭代次数（期望值） | $O(N \log N)$                               | $O(N)$     |

RRT 是运动规划领域中的冒险探索者：
它不会绘制世界的每一寸地图，
而是派出探测分支深入未知区域，
直到其中一个分支找到回家的路。
### 786 快速探索随机树星（RRT*）

RRT* 是经典快速探索随机树（RRT）的最优变体。
虽然 RRT 能快速找到一条有效路径，但它不能保证这条路径是*最短的*。
RRT* 通过逐步优化树结构来改进——重新连接附近的节点以最小化总路径成本，并随着时间推移收敛到最优解。

#### 我们要解决什么问题？

RRT 快速且完备，但*非最优*：其路径可能曲折或不必要地冗长。
在运动规划中，最优性很重要，无论是出于能耗、安全还是美观的考虑。

RRT* 保留了 RRT 的探索特性，但增加了一个重新连接的步骤，以局部优化路径。
随着采样的持续进行，路径成本单调递减并收敛到最优路径长度。

#### 工作原理（通俗解释）

每次迭代执行三个主要步骤：

1. 采样与扩展：
   选取一个随机点 `x_rand`，找到最近的节点 `x_nearest`，
   并向其方向扩展以创建 `x_new`（与 RRT 相同）。

2. 选择最佳父节点：
   找到 `x_new` 半径 $r_n$ 内的所有节点。
   在其中，选择能使得到达 `x_new` 的总成本最低的节点。

3. 重新连接：
   对于 $r_n$ 内的每个邻居节点 `x_near`，
   检查是否通过 `x_new` 能提供更短的路径。
   如果是，则将 `x_near` 的父节点更新为 `x_new`。

这种持续的优化使得 RRT* 具有*渐近最优性*：
随着采样点数量的增加，解会收敛到全局最优。

#### 微型代码（Python 示例）

```python
import random, math

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def steer(a, b, step):
    d = distance(a, b)
    if d < step:
        return b
    return (a[0] + step*(b[0]-a[0])/d, a[1] + step*(b[1]-a[1])/d)

def rrt_star(start, goal, is_free, step=1.0, max_iter=5000, radius=5.0):
    tree = {start: None}
    cost = {start: 0}
    for _ in range(max_iter):
        x_rand = (random.uniform(0,100), random.uniform(0,100))
        x_nearest = min(tree.keys(), key=lambda p: distance(p, x_rand))
        x_new = steer(x_nearest, x_rand, step)
        if not is_free(x_nearest, x_new): 
            continue
        # 查找附近的节点
        neighbors = [p for p in tree if distance(p, x_new) < radius and is_free(p, x_new)]
        # 选择最佳父节点
        x_parent = min(neighbors + [x_nearest], key=lambda p: cost[p] + distance(p, x_new))
        tree[x_new] = x_parent
        cost[x_new] = cost[x_parent] + distance(x_parent, x_new)
        # 重新连接
        for p in neighbors:
            new_cost = cost[x_new] + distance(x_new, p)
            if new_cost < cost[p] and is_free(x_new, p):
                tree[p] = x_new
                cost[p] = new_cost
        # 检查是否到达目标
        if distance(x_new, goal) < step:
            tree[goal] = x_new
            cost[goal] = cost[x_new] + distance(x_new, goal)
            return tree, cost
    return tree, cost
```

#### 为什么它很重要

* 渐近最优：随着采样点增加，路径质量提高。
* 保留了与 RRT 相同的概率完备性。
* 产生平滑、高效且安全的轨迹。
* 应用于：
  * 自动驾驶车辆路径规划
  * 无人机导航
  * 机器人机械臂
  * 高维构型空间

#### 一个温和的证明（为什么它有效）

令 $c^*$ 为起点到目标之间的最优路径成本。
RRT* 确保当采样数 $n \to \infty$ 时：

$$
P(\text{cost}(RRT^*) \to c^*) = 1
$$

因为：

* 采样在自由空间上是均匀分布的。
* 每次重新连接都局部最小化了成本函数。
* 连接半径 $r_n \sim (\log n / n)^{1/d}$ 确保了所有附近节点最终都能以高概率连接。

因此，该算法几乎必然收敛到最优解。

#### 亲自尝试

1. 在同一障碍物地图上运行 RRT 和 RRT*。
2. 可视化差异：RRT* 的树看起来更密集、更平滑。
3. 观察随着迭代次数增加，总路径成本如何下降。
4. 调整半径参数以平衡探索与优化。

#### 测试用例

| 场景             | RRT 路径长度 | RRT* 路径长度 | 改进程度       |
| ---------------- | ------------ | ------------- | -------------- |
| 空旷空间         | 140          | 123           | 缩短 12%       |
| 稀疏障碍物       | 165          | 142           | 缩短 14%       |
| 迷宫走廊         | 210          | 198           | 缩短 6%        |
| 3D 环境          | 190          | 172           | 缩短 9%        |

#### 复杂度

| 操作                 | 时间                    | 空间    |
| -------------------- | ----------------------- | ------- |
| 最近邻搜索           | $O(\log N)$ (KD-树)     | $O(N)$  |
| 每次迭代的重新连接   | $O(\log N)$ 平均        | $O(N)$  |
| 总迭代次数           | $O(N \log N)$           | $O(N)$  |

RRT* 是规划器中的精炼梦想家——
它像其祖先 RRT 一样从快速猜测开始，
然后停下来重新思考，重新连接其路径，
直到每一步都不仅是向前，而且是变得更好。
### 787 概率路线图（PRM）

概率路线图（PRM）算法是一种两阶段的运动规划方法，用于在高维连续空间中进行多查询路径查找。
与 RRT 那样从单个起点开始探索不同，PRM 首先采样许多随机点，将它们连接成一个图（路线图），然后使用标准的图搜索算法（如 Dijkstra 或 A*）来查找任意两个构型之间的路径。

#### 我们要解决什么问题？

对于需要在同一环境中执行多次查询的机器人或系统，例如在不同的目的地之间导航 —— 每次都从头开始重建一棵树（像 RRT 那样）是低效的。
PRM 通过预先计算一个穿过自由构型空间的可行连接路线图来解决这个问题。
一旦构建完成，就可以快速回答查询。

#### 工作原理（通俗解释）

PRM 包含两个阶段：

1. 学习阶段（路线图构建）：

   * 在自由空间中随机采样 $N$ 个点（构型）。
   * 丢弃与障碍物发生碰撞的点。
   * 对于每个有效点，如果它们之间的直线连接是无碰撞的，则将其连接到其 $k$ 个最近邻点。
   * 将这些节点和边存储为一个图（即路线图）。

2. 查询阶段（路径搜索）：

   * 将起点和目标点连接到路线图中附近的节点。
   * 使用图搜索算法（如 Dijkstra 或 A*）在路线图上找到最短路径。

随着时间的推移，路线图会变得更加密集，从而增加了找到最优路径的可能性。

#### 微型代码（Python 示例）

```python
import random, math, heapq

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def is_free(a, b):
    # 占位碰撞检查器（始终自由）
    return True

def build_prm(num_samples=100, k=5):
    points = [(random.uniform(0,100), random.uniform(0,100)) for _ in range(num_samples)]
    edges = {p: [] for p in points}
    for p in points:
        neighbors = sorted(points, key=lambda q: distance(p, q))[1:k+1]
        for q in neighbors:
            if is_free(p, q):
                edges[p].append(q)
                edges[q].append(p)
    return points, edges

def dijkstra(edges, start, goal):
    dist = {p: float('inf') for p in edges}
    prev = {p: None for p in edges}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            break
        for v in edges[u]:
            alt = d + distance(u, v)
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))
    path, u = [], goal
    while u:
        path.append(u)
        u = prev[u]
    return path[::-1]
```

#### 为何重要

* 适用于多查询运动规划。
* 概率完备性：随着样本数量的增加，找到路径（如果存在）的概率趋近于 1。
* 常见于：

  * 移动机器人导航
  * 自动驾驶车辆路线图
  * 高维机械臂规划
  * 虚拟环境和游戏

#### 一个温和的证明（为何有效）

令 $X_{\text{free}}$ 为自由构型空间。
均匀随机采样确保了当样本数量 $n \to \infty$ 时，样本集在 $X_{\text{free}}$ 中变得稠密。

如果连接半径 $r_n$ 满足：

$$
r_n \ge c \left( \frac{\log n}{n} \right)^{1/d}
$$

（其中 $d$ 是空间维度），
那么路线图以高概率变得连通。

因此，在同一自由空间连通分量中的任意两个构型，都可以通过路线图中的一条路径连接起来，这使得 PRM 具有*概率完备性*。

#### 亲自尝试

1.  用 100 个随机点构建一个 PRM，并将每个点连接到其 5 个最近邻。
2.  添加圆形障碍物，观察路线图如何避开它们。
3.  使用同一个路线图查询多个起点-目标点对。
4.  测量随着样本量增加，路径质量的变化。

#### 测试用例

| 样本数 | 邻居数 (k) | 连通性 | 平均路径长度 | 查询时间 (毫秒) |
| ------- | ------------- | ------------ | ---------------- | --------------- |
| 50      | 3             | 80%          | 160              | 0.8             |
| 100     | 5             | 95%          | 140              | 1.2             |
| 200     | 8             | 99%          | 125              | 1.5             |
| 500     | 10            | 100%         | 118              | 2.0             |

#### 复杂度

| 操作               | 时间          | 空间      |
| ----------------------- | ------------- | ---------- |
| 采样                | $O(N)$        | $O(N)$     |
| 最近邻搜索 | $O(N \log N)$ | $O(N)$     |
| 路径查询 (A*)         | $O(E \log V)$ | $O(V + E)$ |

PRM 是运动规划的制图师 ——
它首先用分散的地标勘测地形，
将可达的地标连接成一张活地图，
让旅行者能够通过其概率之路快速规划路线。
### 788 可见性图

可见性图算法是一种经典的几何方法，用于在具有多边形障碍物的二维环境中进行最短路径规划。它连接所有能够直接"看见"彼此的成对点（顶点），这意味着它们之间的直线不与任何障碍物相交。然后，在该图上应用 Dijkstra 或 A* 等最短路径算法来找到最优路线。

#### 我们要解决什么问题？

想象一个在有墙壁或障碍物的房间中导航的机器人。我们想要在起点和目标点之间找到一条无碰撞的最短路径。与基于网格或采样的方法不同，可见性图提供了一条精确的几何路径，通常会触及障碍物的角落。

#### 工作原理（通俗解释）

1.  收集所有障碍物的顶点，以及起点和目标点。
2.  对于每对顶点 $(v_i, v_j)$：
    *   在它们之间画一条线段。
    *   如果该线段不与任何障碍物的边相交，则它们是*可见的*。
    *   在图（graph）中添加一条边 $(v_i, v_j)$，其权重为欧几里得距离。
3.  在起点和目标点之间运行最短路径算法（Dijkstra 或 A*）。
4.  生成的路径沿着可见性发生变化的障碍物角落前进。

这就在多边形世界中产生了（在欧几里得距离上）最优路径。

#### 微型代码（Python 示例）

```python
import math, itertools, heapq

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def intersect(a, b, c, d):
    # 基本的线段相交测试
    def ccw(p, q, r):
        return (r[1]-p[1])*(q[0]-p[0]) > (q[1]-p[1])*(r[0]-p[0])
    return (ccw(a,c,d) != ccw(b,c,d)) and (ccw(a,b,c) != ccw(a,b,d))

def build_visibility_graph(points, obstacles):
    edges = {p: [] for p in points}
    for p, q in itertools.combinations(points, 2):
        if not any(intersect(p, q, o[i], o[(i+1)%len(o)]) for o in obstacles for i in range(len(o))):
            edges[p].append((q, distance(p,q)))
            edges[q].append((p, distance(p,q)))
    return edges

def dijkstra(graph, start, goal):
    dist = {v: float('inf') for v in graph}
    prev = {v: None for v in graph}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == goal: break
        for v, w in graph[u]:
            alt = d + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))
    path = []
    u = goal
    while u is not None:
        path.append(u)
        u = prev[u]
    return path[::-1]
```

#### 为什么它重要？

*   在多边形环境中产生精确的最短路径。
*   完全依赖于几何，无需离散化或随机采样。
*   常见于：
    *   机器人学（绕障碍物路径规划）
    *   电子游戏（导航网格和寻路）
    *   计算几何教学和测试
    *   建筑布局和城市规划工具

#### 一个温和的证明（为什么它有效）

如果所有障碍物都是多边形的，并且允许在可见顶点之间沿直线运动，那么任何最优路径都可以表示为一个可见顶点序列（起点 → 角落 → 角落 → 目标）。

形式上，在与障碍物的两个连续切点之间，路径必须是一条直线段；否则，它可以被缩短。

因此，避开障碍物的最短路径存在于可见性图的边中。

#### 自己动手试试

1.  创建一个带有多边形障碍物（矩形、三角形等）的地图。
2.  绘制可见性图，即连接可见顶点的边。
3.  观察最短路径如何"紧贴"障碍物角落。
4.  将结果与基于网格的 A* 进行比较，你会看到几何方法如何给出精确的最小路径。

#### 测试用例

| 场景             | 障碍物数量 | 顶点数量 | 路径类型              | 结果   |
| ---------------- | ---------- | -------- | --------------------- | ------ |
| 空平面           | 0          | 2        | 直线                  | 最优   |
| 一个矩形         | 4          | 6        | 切向角落路径          | 最优   |
| 迷宫墙壁         | 12         | 20       | 多角落路径            | 最优   |
| 三角形障碍物     | 9          | 15       | 混合边                | 最优   |

#### 复杂度

| 操作                 | 时间复杂度 | 空间复杂度 |
| -------------------- | ---------- | ---------- |
| 边可见性检查         | $O(V^2 E)$ | $O(V^2)$   |
| 最短路径（Dijkstra） | $O(V^2)$   | $O(V)$     |

此处 $V$ 是顶点数，$E$ 是障碍物边数。

可见性图是运动规划器中的几何纯粹主义者——它相信直线和清晰的视线，描绘出刚好擦过障碍物边缘的路径，仿佛几何本身在低声指引着前进的方向。
### 789 势场路径规划

势场路径规划将导航视为一个物理问题。
机器人在人工势场的影响下移动：
目标像重力一样吸引它，障碍物像电荷一样排斥它。
这种方法将规划转化为一个连续优化问题，其中运动自然地沿着势能下降方向流动。

#### 我们要解决什么问题？

在杂乱空间中寻找路径可能很棘手。
像 A* 这样的经典算法在离散网格上工作，但许多真实环境是连续的。
势场为导航提供了一个平滑的、实值的框架，它直观、轻量且具有反应性。

挑战是什么？避免局部最小值，即机器人在到达目标之前被困在力的"山谷"中。

#### 工作原理（通俗解释）

1.  在空间上定义一个势函数：

    *   指向目标的吸引势：
        $$
        U_{att}(x) = \frac{1}{2} k_{att} , |x - x_{goal}|^2
        $$

    *   远离障碍物的排斥势：
        $$
        U_{rep}(x) =
        \begin{cases}
        \frac{1}{2} k_{rep} \left(\frac{1}{d(x)} - \frac{1}{d_0}\right)^2, & d(x) < d_0 \\
        0, & d(x) \ge d_0
        \end{cases}
        $$

        其中 $d(x)$ 是到最近障碍物的距离，$d_0$ 是影响半径。

2.  计算合力（势的负梯度）：
    $$
    F(x) = -\nabla U(x) = F_{att}(x) + F_{rep}(x)
    $$

3.  让机器人沿着力的方向移动一小步，直到到达目标（或被困住）。

#### 微型代码（Python 示例）

```python
import numpy as np

def attractive_force(pos, goal, k_att=1.0):
    """计算吸引力"""
    return -k_att * (pos - goal)

def repulsive_force(pos, obstacles, k_rep=100.0, d0=5.0):
    """计算排斥力"""
    total = np.zeros(2)
    for obs in obstacles:
        diff = pos - obs
        d = np.linalg.norm(diff)
        if d < d0 and d > 1e-6:
            total += k_rep * ((1/d - 1/d0) / d3) * diff
    return total

def potential_field_path(start, goal, obstacles, step=0.5, max_iter=1000):
    """使用势场规划路径"""
    pos = np.array(start, dtype=float)
    goal = np.array(goal, dtype=float)
    path = [tuple(pos)]
    for _ in range(max_iter):
        F_att = attractive_force(pos, goal)
        F_rep = repulsive_force(pos, obstacles)
        F = F_att + F_rep
        pos += step * F / np.linalg.norm(F)
        path.append(tuple(pos))
        if np.linalg.norm(pos - goal) < 1.0:
            break
    return path
```

#### 为何重要

*   **连续空间路径规划**：直接在 $\mathbb{R}^2$ 或 $\mathbb{R}^3$ 中工作。
*   **计算量小**：无需构建网格或图。
*   **反应式**：动态适应障碍物的变化。
*   **应用于**：
    *   自主无人机和机器人
    *   人群模拟
    *   局部运动控制系统

#### 一个温和的证明（为何有效）

总势函数
$$
U(x) = U_{att}(x) + U_{rep}(x)
$$
除了在障碍物边界处外都是可微的。
在任何一点，最速下降方向 $-\nabla U(x)$ 指向 $U(x)$ 的最近最小值。
如果 $U$ 是凸的（除了目标外没有局部最小值），梯度下降路径将收敛到目标构型。

然而，在非凸环境中，可能存在多个最小值。
混合方法（如添加随机扰动或与 A* 结合）可以逃离这些陷阱。

#### 亲自尝试

1.  定义一个带有圆形障碍物的 2D 地图。
2.  将势场可视化为热力图。
3.  追踪路径如何平滑地滑向目标。
4.  引入一个狭窄通道，观察调整 $k_{rep}$ 如何影响避障。
5.  与 A* 结合，实现全局 + 局部规划。

#### 测试用例

| 环境         | 障碍物数量 | 行为描述                     | 结果           |
| ------------ | ---------- | ---------------------------- | -------------- |
| 空旷空间     | 0          | 直接路径                     | 到达目标       |
| 一个障碍物   | 1          | 平滑绕过障碍物               | 成功           |
| 两个障碍物   | 2          | 避开两者                     | 成功           |
| 狭窄缝隙     | 2（靠近）  | 可能出现局部最小值           | 部分成功       |

#### 复杂度

| 操作                     | 时间复杂度         | 空间复杂度 |
| ------------------------ | ------------------ | ---------- |
| 每步的力计算             | $O(N_{obstacles})$ | $O(1)$     |
| 总迭代次数               | $O(T)$             | $O(T)$     |

其中 $T$ 是移动步数。

势场路径规划就像通过无形的重力导航——
空间中的每一点都在低语一个方向，
目标轻柔地牵引，墙壁坚定地推拒，
而旅行者通过运动本身感知世界的形状。
### 790 Bug 算法

Bug 算法是一类简单的反应式路径规划方法，适用于移动机器人。它们仅使用局部感知，无需地图，无需全局规划，只需感知目标的大致方向和是否有障碍物阻挡路径。
它们非常适合极简主义机器人或在不确定性很高的现实世界中进行导航。

#### 我们要解决什么问题？

当机器人向目标移动但遇到未预见的障碍物时，它需要一种在没有全局地图的情况下恢复的方法。
像 A* 或 RRT 这样的传统规划器假设完全了解环境。
相比之下，Bug 算法仅利用机器人能感知到的信息，实时做出决策。

#### 工作原理（通俗解释）

所有 Bug 算法都包含两个阶段：

1.  沿直线向目标移动，直到碰到障碍物。
2.  沿着障碍物的边界移动，直到出现一条通往目标的更好路线。

不同版本对"更好路线"的定义不同：

| 变体        | 策略                                                                   |
| ----------- | ---------------------------------------------------------------------- |
| Bug1        | 绕行整个障碍物，找到离目标最近的点，然后离开。                         |
| Bug2        | 沿着障碍物移动，直到到目标的直线路径再次畅通。                         |
| TangentBug  | 使用距离传感器估计可见性，并智能地切换路径。                           |

#### 示例：Bug2 算法

1.  从 $S$ 点开始，沿着直线 $SG$ 向目标 $G$ 移动。
2.  如果碰到障碍物，则沿着其边界移动，同时测量到 $G$ 的距离。
3.  当到 $G$ 的直线路径再次可见时，离开障碍物并继续前进。
4.  当到达 $G$ 或无法继续前进时停止。

这*仅*使用局部感知和相对于目标的位置感知。

#### 微型代码（Python 示例）

```python
import math

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def bug2(start, goal, obstacles, step=1.0, max_iter=1000):
    pos = list(start)
    path = [tuple(pos)]
    for _ in range(max_iter):
        # 直接向目标移动
        dir_vec = [goal[0]-pos[0], goal[1]-pos[1]]
        dist = math.hypot(*dir_vec)
        if dist < 1.0:
            path.append(tuple(goal))
            break
        dir_vec = [dir_vec[0]/dist, dir_vec[1]/dist]
        next_pos = [pos[0]+step*dir_vec[0], pos[1]+step*dir_vec[1]]
        # 简单的障碍物检查
        hit = any(distance(next_pos, o) < 2.0 for o in obstacles)
        if hit:
            # 沿着边界移动（简化版）
            next_pos[0] += step * dir_vec[1]
            next_pos[1] -= step * dir_vec[0]
        pos = next_pos
        path.append(tuple(pos))
    return path
```

#### 为什么它重要

*   仅需局部感知，无需预先计算的地图。
*   适用于未知或动态环境。
*   计算成本低，对传感器噪声鲁棒性强。
*   常用于：
    *   低成本自主机器人
    *   简单的无人机或漫游车
    *   嵌入式微控制器系统

#### 一个温和的证明（为什么它有效）

如果满足以下条件，Bug 算法能保证到达目标：

*   机器人能够检测到何时到达目标，并且
*   环境是有界的，且障碍物数量有限。

因为每个边界都是确定性地被跟随，并且会避免重访已访问过的位置，
所以路径长度是有界的，机器人将要么：

1.  到达目标，要么
2.  证明不存在路径（在探索完所有障碍物之后）。

用正式术语来说，Bug2 在传感器约束下实现了*完备性*。

#### 亲自尝试

1.  在起点和目标之间放置一个圆形障碍物。
2.  模拟 Bug2，观察机器人碰到障碍物，沿着其边缘移动，然后继续向目标前进。
3.  添加更多障碍物，注意路径复杂度如何增加。
4.  与 A* 或 RRT 比较，Bug2 的路径更长，但计算是即时的。

#### 测试用例

| 环境             | 障碍物数量 | 结果               | 路径类型       |
| ---------------- | ---------- | ------------------ | -------------- |
| 空旷空间         | 0          | 直线               | 直接           |
| 单个障碍物       | 1          | 绕行               | 成功           |
| 多个障碍物       | 3          | 顺序避障           | 成功           |
| 目标被包围       | 1 个包围圈 | 无路径             | 检测到失败     |

#### 复杂度

| 操作                 | 时间     | 空间     |
| -------------------- | -------- | -------- |
| 局部感知和更新       | $O(1)$   | $O(1)$   |
| 总路径遍历           | $O(L)$   | $O(1)$   |

其中 $L$ 是遇到的所有障碍物边界总长度。

Bug 算法是机器人学中的漫游者——
它们看不到整个地图，只能看到眼前的事物，
然而凭借耐心和坚持，它们找到了回家的路。

# 第 80 节 计算几何的变体与应用
### 791 凸多边形求交

凸多边形求交算法用于计算两个凸多边形重叠所形成的区域。由于凸多边形没有内部凹陷，其交集本身也是凸的，可以通过几何裁剪或增量边遍历高效地找到。

#### 我们解决什么问题？

给定两个凸多边形 \( P \) 和 \( Q \)，我们希望找到它们的交集多边形 \( R = P \cap Q \)。这是计算几何、计算机图形学（裁剪）和碰撞检测中的基础问题。

凸性保证了：

* 多边形内任意两点之间的线段都完全位于多边形内部。
* 交集的计算时间复杂度相对于边数是线性的。

#### 工作原理（通俗解释）

有两种常见方法：

1.  **半平面求交（Sutherland–Hodgman 算法）**：
    用另一个多边形的每条边去裁剪一个多边形。

    *   从 \( P \) 的所有顶点开始。
    *   对于 \( Q \) 的每条边，只保留位于该半平面内部的点。
    *   处理完所有边后，结果就是 \( P \cap Q \)。

2.  **边遍历（分治并进法）**：
    同时绕两个多边形行走，通过比较角度来推进边，并收集交点和包含点。

两种方法都依赖于凸性：每对边最多有两个交点，并且边按角度顺序保持有序。

#### 数学核心

对于多边形 \( Q \) 的每条有向边，表示为 \( (q_i, q_{i+1}) \)，定义一个半平面：
$$
H_i = { x \in \mathbb{R}^2 : (q_{i+1} - q_i) \times (x - q_i) \ge 0 }
$$

那么，交集多边形为：
$$
P \cap Q = P \cap \bigcap_i H_i
$$

每次裁剪步骤都会通过切掉当前半平面外部的部分来缩减 \( P \)。

#### 微型代码（Python 示例）

```python
def cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def intersect(p1, p2, q1, q2):
    A1, B1 = p2[1]-p1[1], p1[0]-p2[0]
    C1 = A1*p1[0] + B1*p1[1]
    A2, B2 = q2[1]-q1[1], q1[0]-q2[0]
    C2 = A2*q1[0] + B2*q1[1]
    det = A1*B2 - A2*B1
    if abs(det) < 1e-9:
        return None
    return ((B2*C1 - B1*C2)/det, (A1*C2 - A2*C1)/det)

def clip_polygon(poly, edge_start, edge_end):
    new_poly = []
    for i in range(len(poly)):
        curr, nxt = poly[i], poly[(i+1)%len(poly)]
        inside_curr = cross(edge_start, edge_end, curr) >= 0
        inside_next = cross(edge_start, edge_end, nxt) >= 0
        if inside_curr and inside_next:
            new_poly.append(nxt)
        elif inside_curr and not inside_next:
            new_poly.append(intersect(curr, nxt, edge_start, edge_end))
        elif not inside_curr and inside_next:
            new_poly.append(intersect(curr, nxt, edge_start, edge_end))
            new_poly.append(nxt)
    return [p for p in new_poly if p]

def convex_intersection(P, Q):
    result = P
    for i in range(len(Q)):
        result = clip_polygon(result, Q[i], Q[(i+1)%len(Q)])
        if not result:
            break
    return result
```

#### 为何重要

*   多边形裁剪中的核心操作（用于渲染管线）。
*   凸对象之间碰撞检测的基础。
*   应用于计算几何、地理信息系统（GIS）和物理引擎。
*   作为更复杂几何算法（例如闵可夫斯基和、分离轴定理）的构建模块。

#### 一个温和的证明（为何有效）

多边形 \( Q \) 的每条边定义了一个描述其内部的线性不等式。将 \( P \) 与一个半平面相交保持了凸性。连续应用来自 \( Q \) 的所有约束，既保持了凸性，也保持了有界性。

由于每次裁剪步骤线性地移除顶点，对于具有 \( n \) 和 \( m \) 个顶点的多边形，总复杂度为 \( O(n + m) \)。

#### 动手尝试

1.  创建两个凸多边形 \( P \) 和 \( Q \)。
2.  使用裁剪代码计算 \( P \cap Q \)。
3.  将它们可视化，结果形状总是凸的。
4.  尝试不相交、相切和完全包含的配置。

#### 测试用例

| 多边形 P             | 多边形 Q     | 交集类型           |
| --------------------- | ------------- | ---------------------- |
| 重叠的三角形          | 四边形        | 凸四边形               |
| 正方形内含正方形      | 偏移          | 较小的凸多边形         |
| 不相交                | 相距较远      | 空集                   |
| 边相切                | 相邻          | 线段                   |

#### 复杂度

| 操作         | 时间复杂度      | 空间复杂度 |
| ------------ | --------------- | ---------- |
| 裁剪         | $O(n + m)$      | $O(n)$     |
| 半平面测试   | $O(n)$ 每条边   | $O(1)$     |

凸多边形求交是几何重叠的建筑师——不是通过蛮力，而是通过逻辑来切割形状，描绘出两个凸世界相遇并共享共同点的静谧边界。
### 792 闵可夫斯基和

闵可夫斯基和是一种基础的几何运算，它通过向量加法将两组点集组合起来。
在计算几何中，它常用于建模形状扩张、碰撞检测和路径规划，例如，将一个物体按照另一个物体的形状进行“膨胀”。

#### 我们要解决什么问题？

假设我们有两个凸形状 ( A ) 和 ( B )。
我们想要一个新的形状 $A \oplus B$，它表示从一个形状中取一个点与另一个形状中取一个点进行求和的所有可能结果。

形式上，这捕捉了一个物体如果“围绕”另一个物体滑动会占据多少空间——
这是运动规划和碰撞几何中的一个关键思想。

#### 它是如何工作的（通俗解释）

给定两个集合 $A, B \subset \mathbb{R}^2$：
$$
A \oplus B = { a + b \mid a \in A, b \in B }
$$

换句话说，取 ( A ) 中的每一个点，并用 ( B ) 中的每一个点对其进行平移，
然后取所有这些平移结果的并集。

当 ( A ) 和 ( B ) 是凸多边形时，闵可夫斯基和也是凸的。
其边界可以通过按角度顺序合并边来高效地构造。

#### 几何直观

* 将一个圆加到一个多边形上会使其角变“圆滑”（用于构型空间扩张）。
* 将机器人形状加到障碍物上，实际上是用机器人的尺寸扩大了障碍物——
  从而将路径规划问题简化为在扩张空间中的点导航问题。

#### 数学形式

如果 ( A ) 和 ( B ) 是顶点分别为
$A = (a_1, \dots, a_n)$ 和 $B = (b_1, \dots, b_m)$ 的凸多边形，
并且两者都按逆时针顺序列出，
那么闵可夫斯基和多边形可以通过逐边合并来计算：

$$
A \oplus B = \text{conv}{ a_i + b_j }
$$

#### 微型代码（Python 示例）

```python
import math

def cross(a, b):
    return a[0]*b[1] - a[1]*b[0]

def minkowski_sum(A, B):
    # 假设为凸多边形，按逆时针顺序排列
    i, j = 0, 0
    n, m = len(A), len(B)
    result = []
    while i < n or j < m:
        result.append((A[i % n][0] + B[j % m][0],
                       A[i % n][1] + B[j % m][1]))
        crossA = cross((A[(i+1)%n][0]-A[i%n][0], A[(i+1)%n][1]-A[i%n][1]),
                       (B[(j+1)%m][0]-B[j%m][0], B[(j+1)%m][1]-B[j%m][1]))
        if crossA >= 0:
            i += 1
        if crossA <= 0:
            j += 1
    return result
```

#### 为什么它很重要

* 碰撞检测：
  两个凸形状 ( A ) 和 ( B ) 相交，当且仅当
  $(A \oplus (-B))$ 包含原点。

* 运动规划：
  用机器人的形状扩张障碍物可以简化寻路。

* 计算几何：
  用于构建构型空间和近似复杂的形状交互。

#### 一个温和的证明（为什么它有效）

对于凸多边形，闵可夫斯基和可以通过将它们的支撑函数相加得到：
$$
h_{A \oplus B}(u) = h_A(u) + h_B(u)
$$
其中 $h_S(u) = \max_{x \in S} u \cdot x$ 给出了形状 ( S ) 沿方向 ( u ) 的最远延伸。
$A \oplus B$ 的边界是通过按角度升序合并 ( A ) 和 ( B ) 的边形成的，
并保持了凸性。

这产生了一个 $O(n + m)$ 的构造算法。

#### 自己动手试试

1.  从两个凸多边形开始（例如，三角形和正方形）。
2.  计算它们的闵可夫斯基和，结果应该“融合”了它们的形状。
3.  加上一个小圆形形状，观察角是如何变圆的。
4.  可视化这个过程如何通过另一个形状的几何来扩大一个形状。

#### 测试用例

| 形状 A          | 形状 B          | 结果形状         | 备注                     |
| ---------------- | ---------------- | ----------------- | ---------------------- |
| 三角形         | 正方形           | 六边形形状   | 凸的                 |
| 矩形        | 圆形           | 圆角矩形 | 用于机器人规划 |
| 两个正方形      | 相同朝向 | 更大的正方形     | 按比例放大              |
| 不规则凸形 | 小多边形    | 平滑的边缘    | 保持凸性       |

#### 复杂度

| 操作           | 时间                  | 空间      |
| ------------------- | --------------------- | ---------- |
| 边合并        | $O(n + m)$            | $O(n + m)$ |
| 凸包清理 | $O((n + m)\log(n+m))$ | $O(n + m)$ |

闵可夫斯基和是几何的组合旋律——
一个形状中的每个点都与另一个形状中的每个点和声歌唱，
产生一个新的、统一的图形，揭示了物体在空间中真正的相遇方式。
### 793 旋转卡尺

旋转卡尺（Rotating Calipers）技术是一种几何方法，用于高效解决各种凸多边形问题。
其名称源于脑海中浮现的一幅图像：一对卡尺围绕一个凸形旋转，始终与两条平行的支撑线相切。

这种方法可以优雅地在线性时间内计算几何量，如宽度、直径、最小包围盒或最远点对。

#### 我们要解决什么问题？

给定一个凸多边形，我们通常需要计算以下几何度量：

*   直径（两个顶点之间的最大距离）。
*   宽度（包含该多边形的两条平行线之间的最小距离）。
*   最小包围矩形（最小面积的包围盒）。

一种朴素的方法是检查所有点对，需要 $O(n^2)$ 的工作量。
旋转卡尺则利用凸性和几何特性，在线性时间内完成计算。

#### 工作原理（通俗解释）

1.  从按逆时针顺序排列的凸多边形顶点开始。
2.  确定一对初始的对跖点，即位于平行支撑线上的点。
3.  围绕多边形的边旋转一对卡尺，始终保持与对跖顶点接触。
4.  对于每条边的方向，计算相关的度量（距离、宽度等）。
5.  根据需要记录最小值或最大值。

因为每条边和每个顶点最多被访问一次，所以总时间为 $O(n)$。

#### 示例：寻找凸多边形的直径

直径是凸包上任意两点之间的最长距离。

1.  计算点的凸包（如果还不是凸的）。
2.  在对跖点处初始化两个指针。
3.  对于每个顶点 $i$，在面积（或叉积）增加时移动相对的顶点 $j$：
   $$
   |(P_{i+1} - P_i) \times (P_{j+1} - P_i)| > |(P_{i+1} - P_i) \times (P_j - P_i)|
   $$
4.  记录最大距离 $d = | P_i - P_j |$。

#### 微型代码（Python 示例）

```python
import math

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def rotating_calipers(points):
    # points: 按逆时针顺序排列的凸包顶点列表
    n = len(points)
    if n < 2:
        return 0
    max_dist = 0
    j = 1
    for i in range(n):
        next_i = (i + 1) % n
        while abs((points[next_i][0]-points[i][0]) * 
                  (points[(j+1)%n][1]-points[i][1]) - 
                  (points[next_i][1]-points[i][1]) * 
                  (points[(j+1)%n][0]-points[i][0])) > abs(
                  (points[next_i][0]-points[i][0]) * 
                  (points[j][1]-points[i][1]) - 
                  (points[next_i][1]-points[i][1]) * 
                  (points[j][0]-points[i][0])):
            j = (j + 1) % n
        max_dist = max(max_dist, distance(points[i], points[j]))
    return max_dist
```

#### 为何重要

*   **高效**：对于朴素方法需要 $O(n^2)$ 时间的问题，仅需 $O(n)$ 时间。
*   **通用**：适用于多种几何任务，如距离、宽度、包围盒。
*   **几何直观**：模拟了围绕形状的物理测量。
*   **应用于**：
    *   碰撞检测和包围盒
    *   形状分析和凸几何
    *   机器人学和计算几何教育

#### 一个温和的证明（为何有效）

对于一个凸多边形，旋转的每个方向都对应唯一的一对支撑线。
每条线接触多边形的一个顶点或一条边。
当多边形旋转 180° 时，每个顶点恰好成为一次支撑点。

因此，总步数等于顶点数，并且最大距离或最小宽度必定出现在这些对跖对之一上。

这是凸性和支撑函数 $h_P(u) = \max_{x \in P} (u \cdot x)$ 的直接几何结果。

#### 亲自尝试

1.  生成一个凸多边形（例如，六边形）。
2.  应用旋转卡尺计算：
    *   最大距离（直径）。
    *   平行边之间的最小距离（宽度）。
    *   最小包围矩形面积。
3.  可视化旋转的卡尺，它们总是与相对的边相切。

#### 测试用例

| 多边形   | 顶点数 | 度量         | 结果               |
| -------- | ------ | ------------ | ------------------ |
| 正方形   | 4      | 直径         | √2 × 边长          |
| 矩形     | 4      | 宽度         | 较短边             |
| 三角形   | 3      | 直径         | 最长边             |
| 六边形   | 6      | 包围盒       | 与对称性匹配       |

#### 复杂度

| 操作                     | 时间          | 空间  |
| ------------------------ | ------------- | ----- |
| 边遍历                   | $O(n)$        | $O(1)$ |
| 凸包预处理               | $O(n \log n)$ | $O(n)$ |

旋转卡尺技术是几何学中运动的罗盘——
优雅地滑过凸形，
在完美的旋转和谐中测量距离和宽度。
### 794 半平面交

半平面交算法用于寻找满足一组线性不等式定义的公共区域，其中每个不等式代表平面上的一个半平面。
这是计算几何、线性规划和可见性计算中的核心几何操作，能够高效地定义凸区域。

#### 我们要解决什么问题？

给定平面上的一组直线，每条直线定义一个半平面（直线某一侧的区域），
找出所有这些半平面的交多边形。

每个半平面可以写成一个线性不等式：
$$
a_i x + b_i y + c_i \le 0
$$
这些区域的交集形成一个凸多边形（可能为空或无界）。

应用包括：

* 线性可行域
* 可见性多边形
* 凸形状裁剪
* 几何方法求解小型二维线性规划

#### 工作原理（通俗解释）

1.  用其边界直线和一个方向（"内部"方向）来表示每个半平面。
2.  根据边界直线的角度对所有半平面进行排序。
3.  逐个处理它们，维护当前的交多边形（或双端队列）。
4.  每当添加一个新的半平面时，就用该半平面对当前多边形进行裁剪。
5.  处理完所有半平面后得到的结果就是交集区域。

半平面的凸性保证了它们的交集是凸的。

#### 数学形式

半平面由不等式定义：
$$
a_i x + b_i y + c_i \le 0
$$

交集区域为：
$$
R = \bigcap_{i=1}^n { (x, y) : a_i x + b_i y + c_i \le 0 }
$$

每条边界直线将平面分成两部分；
我们迭代地消除"外部"部分。

#### 微型代码（Python 示例）

```python
import math

EPS = 1e-9

def intersect(L1, L2):
    a1, b1, c1 = L1
    a2, b2, c2 = L2
    det = a1*b2 - a2*b1
    if abs(det) < EPS:
        return None
    x = (b1*c2 - b2*c1) / det
    y = (c1*a2 - c2*a1) / det
    return (x, y)

def inside(point, line):
    a, b, c = line
    return a*point[0] + b*point[1] + c <= EPS

def clip_polygon(poly, line):
    result = []
    n = len(poly)
    for i in range(n):
        curr, nxt = poly[i], poly[(i+1)%n]
        inside_curr = inside(curr, line)
        inside_next = inside(nxt, line)
        if inside_curr and inside_next:
            result.append(nxt)
        elif inside_curr and not inside_next:
            result.append(intersect((nxt[0]-curr[0], nxt[1]-curr[1], 0), line))
        elif not inside_curr and inside_next:
            result.append(intersect((nxt[0]-curr[0], nxt[1]-curr[1], 0), line))
            result.append(nxt)
    return [p for p in result if p]

def half_plane_intersection(lines, bound_box=10000):
    # 从一个大的正方形区域开始
    poly = [(-bound_box,-bound_box), (bound_box,-bound_box),
            (bound_box,bound_box), (-bound_box,bound_box)]
    for line in lines:
        poly = clip_polygon(poly, line)
        if not poly:
            break
    return poly
```

#### 为什么重要

*   计算几何核心：是凸裁剪和线性可行性的基础。
*   线性规划可视化：单纯形法的几何版本。
*   图形学与视觉：用于裁剪、阴影投射和可见性计算。
*   路径规划与机器人学：定义安全导航区域。

#### 温和的证明（为什么有效）

每个半平面对应于 $\mathbb{R}^2$ 中的一个线性约束。
凸集的交集是凸的，因此结果也必须是凸的。

迭代裁剪过程连续应用交集运算：
$$
P_{k+1} = P_k \cap H_{k+1}
$$
在每一步，多边形保持凸性并单调缩小（或变为空集）。

最终的多边形 $P_n$ 同时满足所有约束。

#### 自己动手试试

1.  表示如下约束：

    *   $x \ge 0$
    *   $y \ge 0$
    *   $x + y \le 5$
2.  将它们转换为直线系数并传递给 `half_plane_intersection()`。
3.  绘制生成的多边形，它将是这些不等式所界定的三角形。

尝试添加或移除约束，观察可行域如何变化。

#### 测试用例

| 约束条件                             | 结果形状           | 备注                     |
| ------------------------------------ | ------------------ | ------------------------ |
| 构成三角形的 3 个不等式             | 有限凸多边形       | 可行                     |
| 彼此相对的平行约束                   | 无限长条带         | 无界                     |
| 不一致的不等式                       | 空集               | 无交集                   |
| 矩形约束                             | 正方形             | 简单有界多边形           |

#### 复杂度

| 操作             | 时间复杂度     | 空间复杂度 |
| ---------------- | -------------- | ---------- |
| 多边形裁剪       | $O(n \log n)$  | $O(n)$     |
| 增量更新         | $O(n)$         | $O(n)$     |

半平面交是几何的约束语言——
每条直线是一条规则，每个半平面是一个承诺，
而它们的交集，是所有可能性的优雅形状。
### 795 直线排列

直线排列是由一组直线形成的平面细分。它是计算几何中最基本的结构之一，用于研究平面结构的组合复杂性，并构建用于点定位、可见性和几何优化的算法。

#### 我们要解决什么问题？

给定平面上的 ( n ) 条直线，我们想要找出它们如何将平面划分为称为面的区域，以及它们的边和顶点。

例如：

* 2 条直线将平面划分为 4 个区域。
* 3 条直线（无平行线，无三线共点）将平面划分为 7 个区域。
* 一般来说，( n ) 条直线最多将平面划分为
  $$
  \frac{n(n+1)}{2} + 1
  $$
  个区域。

应用包括：

* 计算交点和可见性图
* 运动规划和路径分解
* 为点定位构建梯形图
* 研究组合几何和对偶性

#### 工作原理（通俗解释）

直线排列是增量构建的：

1. 从空平面开始（1 个区域）。
2. 每次添加一条直线。
3. 每条新直线都会与所有之前的直线相交，将某些区域一分为二。

如果这些直线处于一般位置（无平行线，无三线共点），那么第 ( k ) 条直线形成的新区域数量是 ( k )。

因此，( n ) 条直线后的总区域数为：
$$
R(n) = 1 + \sum_{k=1}^{n} k = 1 + \frac{n(n+1)}{2}
$$

#### 几何结构

每个排列将平面划分为：

* 顶点（直线的交点）
* 边（交点之间的线段）
* 面（由边界定的区域）

总数满足欧拉平面公式：
$$
V - E + F = 1 + C
$$
其中 ( C ) 是连通分量的数量（对于直线，( C = 1 )）。

#### 微型代码（Python 示例）

此代码片段为小规模输入构建交点并计算面数。

```python
import itertools

def intersect(l1, l2):
    (a1,b1,c1), (a2,b2,c2) = l1, l2
    det = a1*b2 - a2*b1
    if abs(det) < 1e-9:
        return None
    x = (b1*c2 - b2*c1) / det
    y = (c1*a2 - c2*a1) / det
    return (x, y)

def line_arrangement(lines):
    points = []
    for (l1, l2) in itertools.combinations(lines, 2):
        p = intersect(l1, l2)
        if p:
            points.append(p)
    return len(points), len(lines), 1 + len(points) + len(lines)
```

示例：

```python
lines = [(1, -1, 0), (0, 1, -1), (1, 1, -2)]
print(line_arrangement(lines))
```

#### 为什么它很重要

* 组合几何：帮助界定几何结构的复杂性。
* 点定位：高效空间查询的基础。
* 运动规划：将空间细分为可导航区域。
* 算法设计：催生了梯形图和对偶空间排列等数据结构。

#### 一个温和的证明（为什么它有效）

当添加第 ( k ) 条直线时：

* 它可能与之前的所有 ( k - 1 ) 条直线在相异点相交。
* 这些交点将新直线分成 ( k ) 段。
* 每段穿过一个区域，恰好创建 ( k ) 个新区域。

因此：
$$
R(n) = R(n-1) + n
$$
其中 ( R(0) = 1 )。
通过求和：
$$
R(n) = 1 + \frac{n(n+1)}{2}
$$
这个论证仅依赖于一般位置假设，即无平行线或重合线。

#### 自己动手试试

1. 绘制处于一般位置的 1、2、3 和 4 条直线。
2. 计算区域数，你将得到 2、4、7、11。
3. 验证递推关系 ( R(n) = R(n-1) + n )。
4. 尝试让直线平行或共点，区域数将会减少。

#### 测试用例

| 直线数 (n) | 最大区域数 | 备注                         |
| --------- | ----------- | ---------------------------- |
| 1         | 2           | 将平面一分为二               |
| 2         | 4           | 相交直线                     |
| 3         | 7           | 无平行线，无三线共点         |
| 4         | 11          | 增加 4 个新区域              |
| 5         | 16          | 继续二次增长                 |

#### 复杂度

| 操作                     | 时间     | 空间     |
| ------------------------ | -------- | -------- |
| 交点计算                 | $O(n^2)$ | $O(n^2)$ |
| 增量式排列构建           | $O(n^2)$ | $O(n^2)$ |

直线排列是几何的组合游乐场 —— 每条新直线都增加了复杂性、交点和秩序，将简单的平面转变为关系和区域的网格。
### 796 点定位（梯形图）

点定位问题询问：给定一个平面细分（例如，一组不相交的线段将平面划分为若干区域），确定哪个区域包含给定的点。
梯形图方法利用几何和随机化高效地解决此问题。

#### 我们要解决什么问题？

给定一组不相交的线段，对它们进行预处理，以便我们能回答以下形式的查询：

> 对于点 $(x, y)$，细分中的哪个面（区域）包含它？

应用包括：

*   在平面地图或网格中查找点的位置
*   光线追踪和可见性问题
*   地理信息系统
*   使用平面细分的计算几何算法

#### 工作原理（通俗解释）

1.  构建梯形分解：
   从每个端点向上和向下延伸一条垂直线，直到它碰到另一条线段或无穷远。
   这些线将平面分割成梯形（可能无界）。

2.  构建搜索结构（DAG）：
   将梯形及其邻接关系存储在一个有向无环图中。
   每个内部节点代表一个测试（点是在线段的左侧/右侧，还是在顶点的上方/下方？）。
   每个叶节点对应一个梯形。

3.  查询：
   为了定位一个点，使用几何测试遍历 DAG，直到到达一个叶节点，该叶节点对应的梯形包含该点。

这种结构允许在 $O(n \log n)$ 的期望预处理时间后，实现 $O(\log n)$ 的期望查询时间。

#### 数学概览

对于每个线段集合 $S$：

*   在端点处构建垂直延伸线 → 形成一组垂直条带。
*   每个梯形最多由四条边界定：
  *   顶部和底部由输入线段界定
  *   左侧和右侧由垂直线界定

梯形的总数与 $n$ 呈线性关系。

#### 微型代码（Python 示例，简化版）

以下是概念性框架；实际实现使用几何库。

```python
import bisect

class TrapezoidMap:
    def __init__(self, segments):
        self.segments = sorted(segments, key=lambda s: min(s[0][0], s[1][0]))
        self.x_coords = sorted({x for seg in segments for (x, _) in seg})
        self.trapezoids = self._build_trapezoids()

    def _build_trapezoids(self):
        traps = []
        for i in range(len(self.x_coords)-1):
            x1, x2 = self.x_coords[i], self.x_coords[i+1]
            traps.append(((x1, x2), None))
        return traps

    def locate_point(self, x):
        i = bisect.bisect_right(self.x_coords, x) - 1
        return self.trapezoids[max(0, min(i, len(self.trapezoids)-1))]
```

这个玩具版本将 x 轴划分为梯形；真实版本包含 y 轴边界和邻接关系。

#### 为什么它重要

*   快速查询：期望 $O(\log n)$ 的点定位时间。
*   可扩展的结构：空间复杂度与线段数量呈线性关系。
*   广泛的实用性：是构建 Voronoi 图、可见性和多边形裁剪的基础构件。
*   优雅的随机化：随机增量构造使其保持简单且健壮。

#### 温和的证明（为什么它有效）

在随机增量构造中：

1.  每个新线段在期望上只与 $O(1)$ 个梯形交互。
2.  该结构在期望上维护 $O(n)$ 个梯形和 DAG 中的 $O(n)$ 个节点。
3.  搜索平均只需要 $O(\log n)$ 次决策。

因此，期望的性能界限是：
$$
\text{预处理: } O(n \log n), \quad \text{查询: } O(\log n)
$$

#### 自己动手试试

1.  画几条不相交的线段。
2.  从端点延伸垂直线以形成梯形。
3.  选取随机点并追踪它们落在哪个梯形中。
4.  观察查询如何简化为坐标的比较。

#### 测试用例

| 输入线段          | 查询点    | 输出区域       |
| ----------------- | --------- | -------------- |
| 水平线 y = 1      | (0, 0)    | 线段下方       |
| 两条交叉对角线    | (1, 1)    | 交叉区域       |
| 多边形边          | (2, 3)    | 多边形内部     |
| 空集              | (x, y)    | 无界区域       |

#### 复杂度

| 操作         | 期望时间     | 空间复杂度 |
| ------------ | ------------ | ---------- |
| 构建结构     | $O(n \log n)$ | $O(n)$     |
| 点查询       | $O(\log n)$   | $O(1)$     |

梯形图将几何转化为逻辑——
每条线段定义一个规则，
每个梯形代表一种情况，
而每次查询都通过优雅的空间推理找到其归属。
### 797 沃罗诺伊最近设施

沃罗诺伊最近设施算法将平面上的每个点分配给给定设施集合中距离它最近的设施。由此产生的结构称为沃罗诺伊图，它将空间划分为多个单元，每个单元代表最接近某个特定设施的点集区域。

#### 我们要解决什么问题？

给定一组 $n$ 个设施（点）$S = {p_1, p_2, \dots, p_n}$ 和一个查询点 $q$，我们希望找到使距离最小化的设施 $p_i$：
$$
d(q, p_i) = \min_{1 \le i \le n} \sqrt{(x_q - x_i)^2 + (y_q - y_i)^2}
$$

一个设施 $p_i$ 的沃罗诺伊区域是所有比任何其他设施更接近 $p_i$ 的点的集合：
$$
V(p_i) = { q \in \mathbb{R}^2 \mid d(q, p_i) \le d(q, p_j), , \forall j \ne i }
$$

#### 工作原理（通俗解释）

1.  为给定的设施计算沃罗诺伊图，这是空间的一个平面划分。
2.  每个单元对应一个设施，并包含所有以该设施为最近设施的点。
3.  要回答最近设施查询：
    *   定位查询点位于哪个单元。
    *   该单元的生成点就是最近的设施。

高效的数据结构允许在 $O(n \log n)$ 的预处理后，实现 $O(\log n)$ 的查询时间。

#### 数学几何

两个设施 $p_i$ 和 $p_j$ 之间的边界是连接它们的线段的中垂线：
$$
(x - x_i)^2 + (y - y_i)^2 = (x - x_j)^2 + (y - y_j)^2
$$
简化后得到：
$$
2(x_j - x_i)x + 2(y_j - y_i)y = (x_j^2 + y_j^2) - (x_i^2 + y_i^2)
$$

每一对设施贡献一条中垂线，它们的交点定义了沃罗诺伊顶点。

#### 微型代码（Python 示例）

```python
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

points = [(1,1), (5,2), (3,5), (7,7)]
vor = Voronoi(points)

fig = voronoi_plot_2d(vor)
plt.plot([p[0] for p in points], [p[1] for p in points], 'ro')
plt.show()
```

要定位一个点的最近设施：

```python
import numpy as np
def nearest_facility(points, q):
    points = np.array(points)
    dists = np.linalg.norm(points - np.array(q), axis=1)
    return np.argmin(dists)
```

#### 为什么它很重要？

*   **位置优化**：将客户分配到最近的仓库或服务中心。
*   **计算几何**：空间分析和网格划分中的核心原语。
*   **地理信息系统和物流**：用于区域划分和需求建模。
*   **机器人和覆盖**：在区域规划、聚类和传感器分布中很有用。

#### 一个温和的证明（为什么它有效）

沃罗诺伊图中的每条边界都是由两个设施之间的等距点定义的。平面被划分，使得每个位置都属于最近站点的区域。

凸性成立是因为：
$$
V(p_i) = \bigcap_{j \ne i} { q : d(q, p_i) \le d(q, p_j) }
$$
并且每个不等式定义了一个半平面，因此它们的交集是凸的。

因此，每个沃罗诺伊区域都是凸的，并且每个查询都有一个唯一的最近设施。

#### 自己动手试试

1.  在网格上放置三个设施。
2.  在每对设施之间画出中垂线。
3.  每个交点定义一个沃罗诺伊顶点。
4.  任意选择一个随机点，检查它落在哪个区域。该设施就是它的最近邻。

#### 测试用例

| 设施                | 查询点 | 最近设施 |
| ------------------- | ----------- | ------- |
| (0,0), (5,0)        | (2,1)       | (0,0)   |
| (1,1), (4,4), (7,1) | (3,3)       | (4,4)   |
| (2,2), (6,6)        | (5,3)       | (6,6)   |

#### 复杂度

| 操作                 | 时间          | 空间  |
| -------------------- | ------------- | ------ |
| 构建沃罗诺伊图       | $O(n \log n)$ | $O(n)$ |
| 查询最近设施         | $O(\log n)$   | $O(1)$ |

沃罗诺伊最近设施算法揭示了一个简单而深刻的真理：地图上的每个地方都属于它最"爱"的设施——那个纯粹由几何命运决定的、距离最近的设施。
### 798 Delaunay（德劳内）网格生成

Delaunay 网格生成从一组点出发，创建高质量的三角形网格，旨在优化数值稳定性和平滑度。它是计算几何、有限元方法（FEM）和计算机图形学领域的基石。

#### 我们要解决什么问题？

给定一组点 $P = {p_1, p_2, \dots, p_n}$，我们希望构建一个三角剖分（即分割成三角形），使得：

1.  没有任何点位于任何三角形的外接圆内部。
2.  三角形尽可能“形状良好”，避免出现细长、退化的形状。

这被称为 Delaunay 三角剖分。

#### 工作原理（通俗解释）

1.  从一个包含所有点的边界三角形开始。
2.  逐个插入点：
    *   对于每个新点，找出所有外接圆包含该点的三角形。
    *   移除这些三角形，形成一个多边形空洞。
    *   将新点连接到空洞的各个顶点，形成新的三角形。
3.  移除任何连接到边界顶点的三角形。

最终结果是最大化所有三角形中最小角度的三角剖分。

#### 数学准则

对于任意三角形 $\triangle ABC$，其外接圆经过点 $A$、$B$ 和 $C$，如果第四个点 $D$ 位于该圆内，则违反了 Delaunay 条件。

这可以通过行列式进行测试：

$$
\begin{vmatrix}
x_A & y_A & x_A^2 + y_A^2 & 1 \\
x_B & y_B & x_B^2 + y_B^2 & 1 \\
x_C & y_C & x_C^2 + y_C^2 & 1 \\
x_D & y_D & x_D^2 + y_D^2 & 1
\end{vmatrix} > 0
$$

如果行列式为正，则 $D$ 位于外接圆内——因此，必须翻转三角剖分以恢复 Delaunay 性质。

#### 微型代码（Python 示例）

```python
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

points = np.random.rand(10, 2)
tri = Delaunay(points)

plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')
plt.show()
```

这段代码片段生成一个二维 Delaunay 三角剖分并将其绘制出来。

#### 为什么它很重要？

*   **有限元分析（FEA）**：为模拟提供条件良好的网格。
*   **地形和曲面建模**：构建平滑、无重叠的三角剖分。
*   **计算机图形学**：用于细分、着色和 3D 建模。
*   **科学计算**：实现稳定的数值插值。

#### 一个温和的证明（为什么它有效）

Delaunay 三角剖分在所有 $P$ 的三角剖分中最大化最小角度。这避免了导致不稳定的细长三角形。

关键的几何对偶性：

*   Delaunay 三角剖分是 Voronoi 图的对偶。
*   每条 Delaunay 边连接其 Voronoi 单元共享边界的点。

因此，构造其中一个就自动定义了另一个。

#### 自己动手试试

1.  在纸上画出几个随机点。
2.  画出它们的外接圆，并找出不包含任何其他点的交点。
3.  连接这些点，你就手动构建了一个 Delaunay 三角剖分。
4.  现在稍微扰动一个点，注意结构如何在保持有效的同时进行调整。

#### 测试用例

| 输入点         | 生成的三角形 | 备注                     |
| -------------- | ------------ | ------------------------ |
| 4 个角点       | 2 个三角形   | 简单的正方形分割         |
| 随机 5 个点    | 5–6 个三角形 | 取决于凸包               |
| 10 个随机点    | ≈ 2n 个三角形 | 典型的平面密度           |

#### 复杂度

| 操作             | 期望时间       | 空间   |
| ---------------- | -------------- | ------ |
| 构建三角剖分     | $O(n \log n)$  | $O(n)$ |
| 点插入           | $O(\log n)$    | $O(1)$ |
| 边翻转           | $O(1)$ 摊还    |,      |

#### 变体与扩展

*   **约束 Delaunay 三角剖分（CDT）**：保留特定的边。
*   **3D Delaunay 四面体化**：扩展到空间网格。
*   **自适应细化**：通过插入新点来改善三角形质量。
*   **各向异性 Delaunay**：考虑方向性度量。

Delaunay 网格是几何与稳定性的交汇点——一个懂得如何保持平衡、优雅和高效的三角形网络。
### 799 最小包围圆（Welzl 算法）

最小包围圆问题旨在找到一个平面内能包含所有给定点的最小可能圆。它也被称为最小包围圆或边界圆问题。

#### 我们正在解决什么问题？

给定二维空间中的一组点 $P = {p_1, p_2, \dots, p_n}$，找到一个半径为 $r$、圆心为 $c = (x, y)$ 的圆，使得：

$$
\forall p_i \in P, \quad |p_i - c| \le r
$$

这个圆尽可能紧密地“包裹”所有点，就像在它们周围拉伸一根橡皮筋并拟合出尽可能小的圆。

#### 工作原理（通俗解释）

Welzl 算法使用随机增量构造高效地解决此问题：

1. 随机打乱点的顺序。
2. 增量式构建包围圆：

   * 从没有点开始，此时圆未定义。
   * 对于每个新点：

     * 如果该点在当前圆内，则不做任何操作。
     * 如果它在圆外，则重建圆以包含这个新点。
3. 圆可以由以下方式定义：

   * 两个点（当它们构成直径时），或
   * 三个点（当它们定义一个通过所有三点的唯一圆时）。

期望时间复杂度：O(n)。

#### 几何构造

1. 两个点 (A, B)：
   圆心是中点，半径是距离的一半：
   $$
   c = \frac{A + B}{2}, \quad r = \frac{|A - B|}{2}
   $$

2. 三个点 (A, B, C)：
   圆是通过所有三点的唯一圆。
   使用垂直平分线：

   $$
   \begin{aligned}
   D &= 2(A_x(B_y - C_y) + B_x(C_y - A_y) + C_x(A_y - B_y)) \\
   U_x &= \frac{(A_x^2 + A_y^2)(B_y - C_y) + (B_x^2 + B_y^2)(C_y - A_y) + (C_x^2 + C_y^2)(A_y - B_y)}{D} \\
   U_y &= \frac{(A_x^2 + A_y^2)(C_x - B_x) + (B_x^2 + B_y^2)(A_x - C_x) + (C_x^2 + C_y^2)(B_x - A_x)}{D}
   \end{aligned}
   $$

   圆心是 $(U_x, U_y)$，半径 $r = |A - U|$。

#### 微型代码（Python 示例）

```python
import math, random

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def circle_two_points(a, b):
    center = ((a[0]+b[0])/2, (a[1]+b[1])/2)
    radius = dist(a, b)/2
    return center, radius

def circle_three_points(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    d = 2*(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    ux = ((ax**2+ay**2)*(by-cy) + (bx**2+by**2)*(cy-ay) + (cx**2+cy**2)*(ay-by)) / d
    uy = ((ax**2+ay**2)*(cx-bx) + (bx**2+by**2)*(ax-cx) + (cx**2+cy**2)*(bx-ax)) / d
    center = (ux, uy)
    radius = dist(center, a)
    return center, radius

def welzl(points):
    random.shuffle(points)
    def mec(pts, boundary):
        if not pts or len(boundary) == 3:
            if len(boundary) == 0:
                return ((0, 0), 0)
            if len(boundary) == 1:
                return (boundary[0], 0)
            if len(boundary) == 2:
                return circle_two_points(*boundary)
            return circle_three_points(*boundary)
        p = pts.pop()
        c, r = mec(pts, boundary)
        if dist(c, p) <= r:
            pts.append(p)
            return c, r
        res = mec(pts, boundary + [p])
        pts.append(p)
        return res
    return mec(points[:], [])
```

#### 为什么它重要

* 几何边界：用于碰撞检测和包围体层次结构。
* 聚类和空间统计：紧密包围点以进行面积估计。
* 图形学和机器人学：简化形状近似。
* 数据可视化：计算紧凑的包围形状。

#### 一个温和的证明（为什么它有效）

最多三个点定义最小包围圆：

* 一个点 → 半径为 0 的圆。
* 两个点 → 以该线段为直径的最小圆。
* 三个点 → 通过它们的最小圆。

通过随机插入，每个点需要重建的概率很小，从而实现了期望 O(n) 的时间复杂度。

#### 亲自尝试

1. 选择几个点并在方格纸上画出它们。
2. 找到最远的一对点，画出通过它们的圆。
3. 添加另一个在外的点，调整圆以包含它。
4. 观察何时三个点定义了确切的最小圆。

#### 测试用例

| 输入点              | 最小包围圆（圆心，半径） |
| ------------------- | ------------------------ |
| (0,0), (1,0)        | ((0.5, 0), 0.5)          |
| (0,0), (0,2), (2,0) | ((1,1), √2)              |
| (1,1), (2,2), (3,1) | ((2,1.5), √1.25)         |

#### 复杂度

| 操作         | 期望时间 | 空间   |
| ------------ | -------- | ------ |
| 构建圆       | $O(n)$   | $O(1)$ |
| 验证         | $O(n)$   | $O(1)$ |

Welzl 算法揭示了几何学中的一个简单真理：
拥抱所有点的最小圆从不脆弱——
它完美平衡，由触及其边缘的少数点定义。
### 800 碰撞检测（分离轴定理）

分离轴定理（SAT）是用于检测两个凸形状是否相交的基本几何原理。它既提供了相交的证明，也提供了在它们不重叠时计算最小分离距离的方法。

#### 我们正在解决什么问题？

给定两个凸多边形（或三维中的凸多面体），确定它们是否碰撞，即它们的内部是否重叠，或者是否不相交。

对于凸形状 $A$ 和 $B$，分离轴定理指出：

> 两个凸形状不相交，当且仅当存在一条直线（轴），使得它们在该轴上的投影不重叠。

这条直线被称为分离轴。

#### 工作原理（通俗解释）

1.  对于两个多边形的每条边：
    *   计算法向量（垂直于边）。
    *   将该法向量视为一个潜在的分离轴。
2.  将两个多边形投影到该轴上：
   $$
   \text{投影} = [\min(v \cdot n), \max(v \cdot n)]
   $$
   其中 $v$ 是顶点，$n$ 是单位法向量。
3.  如果存在一个轴，其上的投影不重叠，则多边形未发生碰撞。
4.  如果所有投影都重叠，则多边形相交。

#### 数学测试

对于给定轴 $n$：

$$
A_{\text{min}} = \min_{a \in A}(a \cdot n), \quad A_{\text{max}} = \max_{a \in A}(a \cdot n)
$$
$$
B_{\text{min}} = \min_{b \in B}(b \cdot n), \quad B_{\text{max}} = \max_{b \in B}(b \cdot n)
$$

如果
$$
A_{\text{max}} < B_{\text{min}} \quad \text{或} \quad B_{\text{max}} < A_{\text{min}}
$$
则存在分离轴 → 无碰撞。

否则，投影重叠 → 碰撞。

#### 微型代码（Python 示例）

```python
import numpy as np

def project(polygon, axis):
    # 将多边形投影到轴上
    dots = [np.dot(v, axis) for v in polygon]
    return min(dots), max(dots)

def overlap(a_proj, b_proj):
    # 检查两个投影区间是否重叠
    return not (a_proj[1] < b_proj[0] or b_proj[1] < a_proj[0])

def sat_collision(polygon_a, polygon_b):
    polygons = [polygon_a, polygon_b]
    for poly in polygons:
        for i in range(len(poly)):
            p1, p2 = poly[i], poly[(i+1) % len(poly)]
            edge = np.subtract(p2, p1)
            axis = np.array([-edge[1], edge[0]])  # 垂直法向量
            axis = axis / np.linalg.norm(axis)
            if not overlap(project(polygon_a, axis), project(polygon_b, axis)):
                return False
    return True
```

#### 为何重要

*   **物理引擎**：检测 2D 和 3D 中物体间碰撞的核心。
*   **游戏开发**：对于凸多边形、包围盒和多面体非常高效。
*   **机器人学**：用于运动规划和避障。
*   **CAD 系统**：帮助测试零件或曲面之间的相交。

#### 一个温和的证明（为何有效）

每个凸多边形都可以描述为半平面的交集。如果两个凸集不相交，则必须至少存在一个超平面将它们完全分离。

投影到所有边的法向量上，覆盖了所有可能的分离方向。如果找不到分离轴，则集合重叠。

这直接源于凸几何中的超平面分离定理。

#### 亲自尝试

1.  在纸上画两个矩形或凸多边形。
2.  计算每条边的法向量。
3.  将两个多边形投影到每个法向量上并比较区间。
4.  如果找到一个投影不重叠的轴，那就是你的分离轴。

#### 测试用例

| 形状 A                 | 形状 B                    | 结果           |
| ----------------------- | -------------------------- | ------------ |
| 重叠的正方形     | 平移距离小于宽度 | 碰撞    |
| 不重叠的正方形 | 平移距离大于宽度 | 无碰撞 |
| 三角形 vs 矩形   | 边接触              | 碰撞    |
| 三角形 vs 矩形   | 完全分离            | 无碰撞 |

#### 复杂度

| 操作                            | 时间复杂度       | 空间复杂度  |
| ------------------------------------ | ---------- | ------ |
| 碰撞测试（2D 凸形）           | $O(n + m)$ | $O(1)$ |
| 碰撞测试（3D 凸多面体） | $O(n + m)$ | $O(1)$ |

$n$ 和 $m$ 是边（或面）的数量。

#### 扩展

*   **3D 版本**：使用面法向量和边的叉积作为轴。
*   **GJK 算法**：对于任意凸形状的更快速替代方案。
*   **EPA（扩展多面体算法）**：在碰撞后计算穿透深度。
*   **粗检测阶段**：结合 SAT 和包围体以提高效率。

分离轴定理抓住了碰撞检测逻辑的本质——要找到接触，我们只需要寻找它们之间的空隙。如果没有空隙，物体就已经相遇了。
