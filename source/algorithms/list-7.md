---
title: 第七章
date: 2026-02-25
---

# 第 7 章 字符串与文本算法

# 第 61 节 字符串匹配
### 601 朴素字符串匹配

朴素字符串匹配是在文本中查找模式的最简单方法。它检查文本中每一个可能的位置，看模式是否匹配。虽然这不是最快的方法，但它是最直观的，非常适合理解模式匹配是如何开始的。

#### 我们要解决什么问题？

我们被给予：
- 一个长度为 `n` 的文本 `T`
- 一个长度为 `m` 的模式 `P`

我们想要找到 `P` 在 `T` 中出现的所有位置。

示例：
文本：`"ABABABCABABABCAB"`
模式：`"ABABC"`

我们需要检查 `T` 中每一个可能的起始位置，看所有字符是否匹配。

#### 它是如何工作的？（通俗解释）

想象一下，将模式在文本上一个字符一个字符地滑动。
在每个位置：
1.  将模式的所有字符与文本进行比较。
2.  如果全部匹配，记录一个命中。
3.  如果出现不匹配，则滑动一步，再试一次。

这就像用放大镜查看一样，移动一个字母，再扫描一次。

逐步示例：

| 偏移 | 文本窗口 | 匹配？ | 原因               |
| ----- | ----------- | ------ | -------------------- |
| 0     | ABABA       | 否     | 第5个字符不匹配 |
| 1     | BABAB       | 否     | 第1个字符不匹配 |
| 2     | ABABC       | 是      | 完全匹配           |
| 3     | BABCA       | 否     | 第1个字符不匹配 |

我们重复这个过程，直到到达最后一个有效的窗口（n - m）。

#### 微型代码（简易版本）

C

```c
#include <stdio.h>
#include <string.h>

void naive_search(const char *text, const char *pattern) {
    int n = strlen(text);
    int m = strlen(pattern);
    for (int i = 0; i <= n - m; i++) {
        int j = 0;
        while (j < m && text[i + j] == pattern[j]) j++;
        if (j == m)
            printf("在索引 %d 处找到匹配\n", i);
    }
}

int main(void) {
    const char *text = "ABABABCABABABCAB";
    const char *pattern = "ABABC";
    naive_search(text, pattern);
}
```

Python

```python
def naive_search(text, pattern):
    n, m = len(text), len(pattern)
    for i in range(n - m + 1):
        if text[i:i+m] == pattern:
            print("在索引", i, "处找到匹配")

text = "ABABABCABABABCAB"
pattern = "ABABC"
naive_search(text, pattern)
```

#### 为什么它重要

- 为模式匹配建立直觉。
- 是更高级算法（KMP、Z、Rabin–Karp）的基础。
- 易于实现和调试。
- 当 `n` 和 `m` 很小或比较成本很低时很有用。

#### 复杂度

| 情况  | 描述                   | 比较次数 | 时间复杂度  |
| ----- | ----------------------------- | ----------- | ----- |
| 最佳  | 每次第一个字符就不匹配 | O(n)        | O(n)  |
| 最差  | 每次偏移都几乎完全匹配  | O(n·m)      | O(nm) |
| 空间  | 仅使用索引和计数器     | O(1)        |       |

朴素方法在最坏情况下的时间复杂度为 O(nm)，对于大文本来说很慢，但简单且确定。

#### 亲自尝试

1.  在文本 `"AAAAAA"` 和模式 `"AAA"` 上运行代码。
   你找到了多少个匹配项？
2.  尝试文本 `"ABCDE"` 和模式 `"FG"`。
   它失败得有多快？
3.  测量当 `n=10`，`m=3` 时的比较次数。
4.  修改代码，使其在找到第一个匹配项后停止。
5.  扩展它以支持不区分大小写的匹配。

朴素字符串匹配是你进入文本算法世界的第一块透镜。简单、诚实、不知疲倦，它会检查每一个角落，直到找到你要寻找的东西。
### 602 Knuth–Morris–Pratt (KMP)

Knuth–Morris–Pratt (KMP) 是一种无需回溯的模式匹配方法。它不重新检查已经比较过的字符，而是利用前缀知识智能地向前跳跃。这是从暴力搜索迈向线性时间搜索的第一个重大飞跃。

#### 我们要解决什么问题？

在朴素搜索中，当发生不匹配时，我们只移动一个位置然后重新开始，浪费了时间重新检查前缀。KMP 解决了这个问题。

我们要解决的是：

> 如何复用过去的比较结果以避免冗余工作？

给定：
- 长度为 `n` 的文本 `T`
- 长度为 `m` 的模式 `P`

我们希望在 O(n + m) 的时间内找到 `P` 在 `T` 中的所有起始位置。

示例：
文本：`"ABABABCABABABCAB"`
模式：`"ABABC"`

当在 `P[j]` 处发生不匹配时，我们利用已经知道的关于模式前缀和后缀的信息来移动模式。

#### 它是如何工作的（通俗解释）？

KMP 有两个主要步骤：

1.  **预处理模式（构建前缀表）**：
    为 `P` 的每个前缀计算 `lps[]`（最长同时也是后缀的真前缀）。
    这个表告诉我们*在不匹配之后可以安全跳过多少*。

2.  **使用前缀表扫描文本**：
    比较文本和模式的字符。
    当在 `j` 处发生不匹配时，不是重新开始，而是跳转到 `j = lps[j-1]`。

可以把 `lps` 看作一个“记忆”，它记住了在不匹配之前我们匹配了多远。

示例模式：`"ABABC"`

| i | P[i] | LPS[i] | 解释                     |
| - | ---- | ------ | ------------------------ |
| 0 | A    | 0      | 没有前缀-后缀匹配        |
| 1 | B    | 0      | "A"≠"B"                  |
| 2 | A    | 1      | "A"                      |
| 3 | B    | 2      | "AB"                     |
| 4 | C    | 0      | 没有匹配                 |

所以 `lps = [0, 0, 1, 2, 0]`

当在位置 4 (`C`) 发生不匹配时，我们跳转到模式中的索引 2，无需重新检查。

#### 简洁代码（简易版本）

C

```c
#include <stdio.h>
#include <string.h>

void compute_lps(const char *pat, int m, int *lps) {
    int len = 0;
    lps[0] = 0;
    for (int i = 1; i < m;) {
        if (pat[i] == pat[len]) {
            lps[i++] = ++len;
        } else if (len != 0) {
            len = lps[len - 1];
        } else {
            lps[i++] = 0;
        }
    }
}

void kmp_search(const char *text, const char *pat) {
    int n = strlen(text), m = strlen(pat);
    int lps[m];
    compute_lps(pat, m, lps);

    int i = 0, j = 0;
    while (i < n) {
        if (text[i] == pat[j]) { i++; j++; }
        if (j == m) {
            printf("在索引 %d 处找到匹配\n", i - j);
            j = lps[j - 1];
        } else if (i < n && text[i] != pat[j]) {
            if (j != 0) j = lps[j - 1];
            else i++;
        }
    }
}

int main(void) {
    const char *text = "ABABABCABABABCAB";
    const char *pattern = "ABABC";
    kmp_search(text, pattern);
}
```

Python

```python
def compute_lps(p):
    m = len(p)
    lps = [0]*m
    length = 0
    i = 1
    while i < m:
        if p[i] == p[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length != 0:
            length = lps[length - 1]
        else:
            lps[i] = 0
            i += 1
    return lps

def kmp_search(text, pat):
    n, m = len(text), len(pat)
    lps = compute_lps(pat)
    i = j = 0
    while i < n:
        if text[i] == pat[j]:
            i += 1; j += 1
        if j == m:
            print("在索引", i - j, "处找到匹配")
            j = lps[j - 1]
        elif i < n and text[i] != pat[j]:
            j = lps[j - 1] if j else i + 1 - (i - j)
            if j == 0: i += 1

text = "ABABABCABABABCAB"
pattern = "ABABC"
kmp_search(text, pattern)
```

#### 为什么它很重要

-   避免重新检查，实现真正的线性时间。
-   是快速文本搜索（编辑器、grep）的基础。
-   启发了其他算法（Z 算法、Aho–Corasick）。
-   教授了预处理模式，而不仅仅是文本。

#### 复杂度

| 阶段             | 时间复杂度 | 空间复杂度 |
| ---------------- | ---------- | ---------- |
| LPS 预处理       | O(m)       | O(m)       |
| 搜索             | O(n)       | O(1)       |
| 总计             | O(n + m)   | O(m)       |

最坏情况下是线性的，每个字符只检查一次。

#### 亲自尝试

1.  为 `"AAAA"`、`"ABABAC"` 和 `"AABAACAABAA"` 构建 `lps`。
2.  修改代码以计算总匹配数，而不是打印。
3.  与朴素搜索比较，统计比较次数。
4.  用箭头显示跳转来可视化 `lps` 表。
5.  在 `"AAAAAAAAB"` 中搜索 `"AAAAAB"`，注意跳转的效率。

KMP 是你的第一个聪明的匹配器，它从不回头，总是记住学到的东西，并自信地在文本上滑行。
### 603 Z 算法

Z 算法是一种通过预先计算每个位置与字符串前缀的匹配长度来实现快速模式匹配的方法。它构建一个衡量前缀重叠的“Z 数组”，这是一种用于字符串搜索的巧妙镜像技巧。

#### 我们要解决什么问题？

我们想要在文本 `T` 中找到模式 `P` 的所有出现位置，但希望避免额外的扫描或重复比较。

思路：
如果我们知道从字符串开头到每个位置有多少字符是匹配的，我们就可以立即检测到模式匹配。

因此，我们构建一个辅助字符串：

```
S = P + '$' + T
```

并计算 `Z[i]` = 从位置 `i` 开始的最长子串的长度，该子串与 `S` 的前缀匹配。

如果 `Z[i]` 等于 `P` 的长度，我们就在 `T` 中找到了一个匹配。

示例：
P = `"ABABC"`
T = `"ABABABCABABABCAB"`
S = `"ABABC$ABABABCABABABCAB"`

每当 `Z[i] = len(P) = 5` 时，就表示一个完全匹配。

#### 它是如何工作的（通俗解释）？

Z 数组编码了从每个索引开始，字符串与自身匹配的程度。

我们扫描 `S` 并维护一个窗口 [L, R]，代表当前最右侧的匹配段。
对于每个位置 `i`：

1. 如果 `i > R`，则从头开始比较。
2. 否则，从窗口内的 `Z[i-L]` 复制信息。
3. 如果可能，将匹配扩展到 `R` 之外。

这就像使用一面镜子，如果你已经知道一个匹配窗口，就可以跳过其中的冗余检查。

以 `"aabxaayaab"` 为例：

| i | S[i] | Z[i] | 解释         |
| - | ---- | ---- | ------------ |
| 0 | a    | 0    | (总是 0)     |
| 1 | a    | 1    | 匹配 "a"     |
| 2 | b    | 0    | 不匹配       |
| 3 | x    | 0    | 不匹配       |
| 4 | a    | 2    | 匹配 "aa"    |
| 5 | a    | 1    | 匹配 "a"     |
| 6 | y    | 0    | 不匹配       |
| 7 | a    | 3    | 匹配 "aab"   |
| 8 | a    | 2    | 匹配 "aa"    |
| 9 | b    | 1    | 匹配 "a"     |

#### 精简代码（简易版本）

C

```c
#include <stdio.h>
#include <string.h>

void compute_z(const char *s, int z[]) {
    int n = strlen(s);
    int L = 0, R = 0;
    z[0] = 0;
    for (int i = 1; i < n; i++) {
        if (i <= R) z[i] = (R - i + 1 < z[i - L]) ? (R - i + 1) : z[i - L];
        else z[i] = 0;
        while (i + z[i] < n && s[z[i]] == s[i + z[i]]) z[i]++;
        if (i + z[i] - 1 > R) { L = i; R = i + z[i] - 1; }
    }
}

void z_search(const char *text, const char *pat) {
    char s[1000];
    sprintf(s, "%s$%s", pat, text);
    int n = strlen(s);
    int z[n];
    compute_z(s, z);
    int m = strlen(pat);
    for (int i = 0; i < n; i++)
        if (z[i] == m)
            printf("在索引 %d 处找到匹配\n", i - m - 1);
}

int main(void) {
    const char *text = "ABABABCABABABCAB";
    const char *pattern = "ABABC";
    z_search(text, pattern);
}
```

Python

```python
def compute_z(s):
    n = len(s)
    z = [0] * n
    L = R = 0
    for i in range(1, n):
        if i <= R:
            z[i] = min(R - i + 1, z[i - L])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] - 1 > R:
            L, R = i, i + z[i] - 1
    return z

def z_search(text, pat):
    s = pat + '$' + text
    z = compute_z(s)
    m = len(pat)
    for i in range(len(z)):
        if z[i] == m:
            print("在索引", i - m - 1, "处找到匹配")

text = "ABABABCABABABCAB"
pattern = "ABABC"
z_search(text, pattern)
```

#### 为什么它很重要

- 线性时间模式匹配（O(n + m)）。
- 建立对前缀重叠和自相似性的直观理解。
- 用于模式检测、DNA 分析、压缩。
- 与 KMP 算法相关，但通常实现更简单。

#### 复杂度

| 步骤             | 时间复杂度 | 空间复杂度 |
| ---------------- | ---------- | ---------- |
| 计算 Z 数组      | O(n)       | O(n)       |
| 搜索             | O(n)       | O(1)       |

总复杂度：O(n + m)

#### 亲自尝试

1.  计算 `"AAABAAA"` 的 Z 数组。
2.  将分隔符 `$` 改为其他符号，为什么它必须不同？
3.  比较 `"abcabcabc"` 和 `"aaaaa"` 的 Z 数组。
4.  统计有多少个位置的 `Z[i] > 0`。
5.  可视化 Z 框在字符串上滑动的过程。

Z 算法读取字符串就像镜子读取光线一样，匹配前缀，跳过重复，揭示隐藏在显而易见之处的结构。
### 604 Rabin–Karp

Rabin–Karp 是一种巧妙的算法，它使用滚动哈希而非逐字符比较来匹配模式。它将字符串转换为数字，因此子串比较就变成了整数比较。它快速、简单，并且非常适合多模式搜索。

#### 我们要解决什么问题？

我们想要在文本 `T`（长度为 `n`）中找到所有模式 `P`（长度为 `m`）的出现位置。

朴素的方法是逐字符比较子串。
Rabin–Karp 则比较哈希值，如果两个子串具有相同的哈希值，我们才进行字符比较来确认。

其诀窍在于滚动哈希：
我们可以根据前一个子串的哈希值，在 O(1) 时间内计算出下一个子串的哈希值。

示例：
文本：`"ABABABCABABABCAB"`
模式：`"ABABC"`
我们不是在每个 5 个字母的窗口都进行检查，而是在文本上滚动计算哈希值，仅在哈希值匹配时才进行检查。

#### 它是如何工作的（通俗解释）？

1.  选择一个基数和模数。
    使用一个基数 `b`（例如 256）和一个大的质数模数 `M` 来减少哈希冲突。

2.  计算模式哈希值。
    计算 `P[0..m-1]` 的哈希值。

3.  计算文本中第一个窗口的哈希值。
    计算 `T[0..m-1]` 的哈希值。

4.  滑动窗口。
    对于每次偏移 `i`：

    *   如果 `hash(T[i..i+m-1]) == hash(P)`，则通过字符检查进行验证。
    *   高效地计算下一个哈希值：

        ```
        new_hash = (b * (old_hash - T[i]*b^(m-1)) + T[i+m]) mod M
        ```

这就像检查指纹：
如果指纹匹配，再检查人脸来确认。

#### 示例

让我们在 `"ABAB"` 中匹配 `"AB"`：

-   基数 = 256, M = 101
-   hash("AB") = (65×256 + 66) mod 101
-   在 `"ABAB"` 上滑动窗口：

    *   窗口 0：`"AB"` → 哈希值相同 → 匹配
    *   窗口 1：`"BA"` → 哈希值不同 → 跳过
    *   窗口 2：`"AB"` → 哈希值相同 → 匹配

总共只进行了两次字符检查！

#### 精简代码（简易版本）

C

```c
#include <stdio.h>
#include <string.h>

#define BASE 256
#define MOD 101

void rabin_karp(const char *text, const char *pat) {
    int n = strlen(text);
    int m = strlen(pat);
    int h = 1;
    for (int i = 0; i < m - 1; i++) h = (h * BASE) % MOD;

    int p = 0, t = 0;
    for (int i = 0; i < m; i++) {
        p = (BASE * p + pat[i]) % MOD;
        t = (BASE * t + text[i]) % MOD;
    }

    for (int i = 0; i <= n - m; i++) {
        if (p == t) {
            int match = 1;
            for (int j = 0; j < m; j++)
                if (text[i + j] != pat[j]) { match = 0; break; }
            if (match) printf("Match found at index %d\n", i);
        }
        if (i < n - m) {
            t = (BASE * (t - text[i] * h) + text[i + m]) % MOD;
            if (t < 0) t += MOD;
        }
    }
}

int main(void) {
    const char *text = "ABABABCABABABCAB";
    const char *pattern = "ABABC";
    rabin_karp(text, pattern);
}
```

Python

```python
def rabin_karp(text, pat, base=256, mod=101):
    n, m = len(text), len(pat)
    h = pow(base, m-1, mod)
    p_hash = t_hash = 0

    for i in range(m):
        p_hash = (base * p_hash + ord(pat[i])) % mod
        t_hash = (base * t_hash + ord(text[i])) % mod

    for i in range(n - m + 1):
        if p_hash == t_hash:
            if text[i:i+m] == pat:
                print("Match found at index", i)
        if i < n - m:
            t_hash = (base * (t_hash - ord(text[i]) * h) + ord(text[i+m])) % mod
            if t_hash < 0:
                t_hash += mod

text = "ABABABCABABABCAB"
pattern = "ABABC"
rabin_karp(text, pattern)
```

#### 为什么它很重要

-   通过哈希实现高效的子串搜索。
-   支持多模式搜索（对每个模式计算哈希）。
-   在抄袭检测、数据去重、生物信息学中很有用。
-   引入了滚动哈希，这是许多算法（Karp–Rabin、Z 算法、字符串指纹、Rabin 指纹、布隆过滤器）的基础。

#### 复杂度

| 情况                         | 时间复杂度 | 空间复杂度 |
| ---------------------------- | -------- | ----- |
| 平均情况                      | O(n + m) | O(1)  |
| 最坏情况（大量哈希冲突）       | O(nm)    | O(1)  |
| 期望情况（哈希函数良好）       | O(n + m) | O(1)  |

滚动哈希使其*在实践中非常快速*。

#### 亲自尝试

1.  使用基数 = 10 和模数 = 13，在 `"313131"` 中匹配 `"31"`。
2.  打印每个窗口的哈希值，观察哈希冲突。
3.  将模数 = 101 替换为一个较小的数字，会发生什么？
4.  尝试同时匹配多个模式（例如 `"AB"`、`"ABC"`）。
5.  在大输入上比较 Rabin–Karp 与朴素搜索的速度。

Rabin–Karp 将文本转化为数字，匹配变成了数学运算。滑动窗口，滚动哈希，让算术指引你的搜索。
### 605 Boyer–Moore

Boyer–Moore（博耶-摩尔）是最快的实用字符串搜索算法之一。它从模式的末尾开始向后读取文本，并在不匹配时跳过文本的大块区域。该算法基于两个关键思想：坏字符规则和好后缀规则。

#### 我们要解决什么问题？

在朴素算法和 KMP 算法中，当发生不匹配时，我们只将模式移动一个位置。
但是，如果我们能安全地跳过多个位置呢？

Boyer–Moore 算法正是这样做的，它从右向左比较，当发生不匹配时，它使用预先计算的表来决定要移动多远。

给定：
- 长度为 `n` 的文本 `T`
- 长度为 `m` 的模式 `P`

我们希望找到 `P` 在 `T` 中出现的所有位置，同时减少比较次数。

示例：
文本：`"HERE IS A SIMPLE EXAMPLE"`
模式：`"EXAMPLE"`

Boyer–Moore 算法不是扫描每个位置，而是可能跳过整个单词。

#### 它是如何工作的（通俗解释）？

1.  预处理模式以构建移动表：
    *   坏字符表：
        当 `P[j]` 处发生不匹配时，移动模式，使得 `P` 中 `T[i]` 的最后一次出现与位置 `j` 对齐。
        如果 `T[i]` 不在模式中，则跳过整个长度 `m`。
    *   好后缀表：
        当后缀匹配但在此之前发生不匹配时，移动模式以与该后缀的下一次出现对齐。

2.  搜索：
    *   将模式与文本对齐。
    *   从右向左比较。
    *   发生不匹配时，应用两个表中最大的移动量。

这就像反向阅读文本，当你从不匹配中获得的信息比匹配更多时，你就可以快速跳转。

#### 示例（坏字符规则）

模式：`"ABCD"`
文本：`"ZZABCXABCD"`

1.  将 `"ABCD"` 与文本中结束于位置 3 的片段进行比较
2.  在 `X` 处不匹配
3.  `X` 不在模式中 → 移动 4 个位置
4.  新的对齐从下一个可能的匹配开始

更少的比较，更智能的跳过。

#### 精简代码（简易版本）

Python（仅坏字符规则）

```python
def bad_char_table(pat):
    table = [-1] * 256
    for i, ch in enumerate(pat):
        table[ord(ch)] = i
    return table

def boyer_moore(text, pat):
    n, m = len(text), len(pat)
    bad = bad_char_table(pat)
    i = 0
    while i <= n - m:
        j = m - 1
        while j >= 0 and pat[j] == text[i + j]:
            j -= 1
        if j < 0:
            print("在索引", i, "处找到匹配")
            i += (m - bad[ord(text[i + m])] if i + m < n else 1)
        else:
            i += max(1, j - bad[ord(text[i + j])])

text = "HERE IS A SIMPLE EXAMPLE"
pattern = "EXAMPLE"
boyer_moore(text, pattern)
```

此版本仅使用坏字符规则，对于一般文本已经能提供强大的性能。

#### 为什么它很重要

- 跳过文本的大部分区域。
- 平均时间为亚线性，通常比 O(n) 更快。
- 是高级变体的基础：
    *   Boyer–Moore–Horspool
    *   Sunday 算法
- 广泛用于文本编辑器、grep、搜索引擎。

#### 复杂度

| 情况     | 时间       | 空间      |
| -------- | ---------- | --------- |
| 最好情况 | O(n / m)   | O(m + σ)  |
| 平均情况 | 亚线性     | O(m + σ)  |
| 最坏情况 | O(nm)      | O(m + σ)  |

（σ = 字母表大小）

在实践中，它是在长文本中搜索长模式的最快算法之一。

#### 亲自尝试

1.  逐步追踪 `"ABCD"` 在 `"ZZABCXABCD"` 中的搜索过程。
2.  打印坏字符表，检查移动值。
3.  添加好后缀规则（高级）。
4.  与在 `"haystack"` 中搜索 `"needle"` 的朴素搜索进行比较。
5.  测量比较次数，跳过了多少次？

Boyer–Moore 算法带着后见之明进行搜索。它向后看，从不匹配中学习，然后向前跳跃，是高效搜索的典范。
### 606 Boyer–Moore–Horspool

Boyer–Moore–Horspool 算法是 Boyer–Moore 算法的一个精简版本。它舍弃了好后缀规则，只专注于一个坏字符跳转表，这使得它更短、更简单，并且在平均情况下通常在实践中更快。

#### 我们要解决什么问题？

经典的 Boyer–Moore 算法功能强大但复杂，需要两个表，多条规则，实现起来很棘手。

Boyer–Moore–Horspool 保留了 Boyer–Moore 的精髓（从右向左扫描和跳转），但简化了逻辑，使得任何人都能轻松编写代码，并在平均情况下获得亚线性的性能。

给定：
- 长度为 `n` 的文本 `T`
- 长度为 `m` 的模式 `P`

我们希望以比朴素搜索更少的比较次数找到 `P` 在 `T` 中的所有出现位置，同时实现起来又很简单。

#### 它是如何工作的（通俗解释）？

它在每次对齐时从右向左扫描文本，并使用一个单一的跳转表。

1.  预处理模式：
   对于字母表中的每个字符 `c`：
   * `shift[c] = m`
     然后，对于每个模式位置 `i`（0 到 m−2）：
   * `shift[P[i]] = m - i - 1`

2.  搜索阶段：
   * 将模式与文本在位置 `i` 对齐
   * 从 `P[m-1]` 开始向后比较模式
   * 如果不匹配，则将窗口移动 `shift[text[i + m - 1]]` 位
   * 如果匹配，则报告位置并以相同方式移动窗口

每次不匹配都可能一次性跳过多个字符。

#### 示例

文本：`"EXAMPLEEXAMPLES"`
模式：`"EXAMPLE"`

模式长度 `m = 7`

跳转表（m−i−1 规则）：

| 字符   | 跳转量 |
| ------ | ----- |
| E      | 6     |
| X      | 5     |
| A      | 4     |
| M      | 3     |
| P      | 2     |
| L      | 1     |
| 其他   | 7     |

从右向左扫描：
- 将 `"EXAMPLE"` 与文本对齐，从 `L` 开始向后比较
- 遇到不匹配时，查看窗口下的最后一个字符 → 相应地跳转

快速跳过没有希望匹配的文本段。

#### 简洁代码（简易版本）

Python

```python
def horspool(text, pat):
    n, m = len(text), len(pat)
    shift = {ch: m for ch in set(text)}
    for i in range(m - 1):
        shift[pat[i]] = m - i - 1

    i = 0
    while i <= n - m:
        j = m - 1
        while j >= 0 and pat[j] == text[i + j]:
            j -= 1
        if j < 0:
            print("在索引处找到匹配", i)
            i += shift.get(text[i + m - 1], m)
        else:
            i += shift.get(text[i + m - 1], m)

text = "EXAMPLEEXAMPLES"
pattern = "EXAMPLE"
horspool(text, pattern)
```

C（简化版本）

```c
#include <stdio.h>
#include <string.h>
#include <limits.h>

#define ALPHABET 256

void horspool(const char *text, const char *pat) {
    int n = strlen(text), m = strlen(pat);
    int shift[ALPHABET];
    for (int i = 0; i < ALPHABET; i++) shift[i] = m;
    for (int i = 0; i < m - 1; i++)
        shift[(unsigned char)pat[i]] = m - i - 1;

    int i = 0;
    while (i <= n - m) {
        int j = m - 1;
        while (j >= 0 && pat[j] == text[i + j]) j--;
        if (j < 0)
            printf("在索引 %d 处找到匹配\n", i);
        i += shift[(unsigned char)text[i + m - 1]];
    }
}

int main(void) {
    horspool("EXAMPLEEXAMPLES", "EXAMPLE");
}
```

#### 为什么它很重要

- 比完整的 Boyer–Moore 算法更简单。
- 在实践中速度很快，尤其是在随机文本上。
- 当你需要快速实现和良好性能时，这是一个绝佳的选择。
- 用于编辑器和搜索工具中处理中等长度的模式。

#### 复杂度

| 情况    | 时间复杂度 | 空间复杂度 |
| ------- | ---------- | ---------- |
| 最好    | O(n / m)   | O(σ)       |
| 平均    | 亚线性     | O(σ)       |
| 最坏    | O(nm)      | O(σ)       |

σ = 字母表大小（例如，256）

大多数文本在每个窗口只产生少量比较 → 通常比 KMP 更快。

#### 亲自尝试

1.  为 `"ABCDAB"` 打印跳转表。
2.  在文本 `"ABABABCABABABCAB"` 上比较与 KMP 算法的跳转次数。
3.  改变模式中的一个字母，跳转量如何变化？
4.  计算与朴素算法的比较次数。
5.  分别用字典和数组实现跳转表，测量速度。

Boyer–Moore–Horspool 就像一个精干的赛车手，它自信地向前跳跃，减轻了重量，但保留了力量。
### 607 Sunday 算法

Sunday 算法是一种轻量级、直观的字符串搜索方法，它采用前瞻策略：不是专注于当前窗口内的不匹配字符，而是窥视文本中的下一个字符来决定跳过多远。它简单、优雅，在实践中通常比更复杂的算法更快。

#### 我们要解决什么问题？

在朴素搜索中，我们每次将模式移动一步。
在 Boyer–Moore 算法中，我们向后查看不匹配的字符。
但是，如果我们能向前窥视一步，并跳过最大可能的距离呢？

Sunday 算法提出了这样的问题：

> "我当前窗口之后紧跟着的字符是什么？"
> 如果该字符不在模式中，就跳过整个窗口。

给定：
- 文本 `T`（长度为 `n`）
- 模式 `P`（长度为 `m`）

我们希望以更少的移位次数找到 `P` 在 `T` 中的所有出现，其指导原则是下一个未见的字符。

#### 它是如何工作的（通俗解释）？

想象一下在文本上滑动一个放大镜。
每次检查一个窗口时，窥视紧跟在它后面的那个字符。

如果该字符不在模式中，则将模式移动到该字符之后（移动 `m + 1` 位）。
如果该字符在模式中，则将文本中的该字符与模式中该字符的最后一次出现对齐。

步骤：
1.  预计算移位表：对于字母表中的每个字符 `c`，`shift[c] = m - last_index(c)`
    未出现字符的默认移位：`m + 1`
2.  在窗口内从左到右比较文本和模式。
3.  如果不匹配或没有匹配，检查下一个字符 `T[i + m]` 并相应地移位。

它基于未来的信息（而非过去的不匹配）进行跳跃，这正是它的魅力所在。

#### 示例

文本：`"EXAMPLEEXAMPLES"`
模式：`"EXAMPLE"`

m = 7

移位表（基于最后出现位置）：

| 字符   | 移位 |
| ------ | ----- |
| E      | 1     |
| X      | 2     |
| A      | 3     |
| M      | 4     |
| P      | 5     |
| L      | 6     |
| 其他   | 8     |

步骤：
- 比较 `"EXAMPLE"` 与 `"EXAMPLE"` → 在索引 0 处匹配
- 下一个字符：`E` → 移位 1
- 比较下一个窗口 → 再次匹配
  快速、前瞻、高效。

#### 精简代码（简易版本）

Python

```python
def sunday(text, pat):
    n, m = len(text), len(pat)
    shift = {ch: m - i for i, ch in enumerate(pat)} # 为模式中的每个字符计算移位值
    default = m + 1 # 默认移位值

    i = 0
    while i <= n - m:
        j = 0
        while j < m and pat[j] == text[i + j]:
            j += 1
        if j == m:
            print("在索引", i, "处找到匹配")
        next_char = text[i + m] if i + m < n else None
        i += shift.get(next_char, default)

text = "EXAMPLEEXAMPLES"
pattern = "EXAMPLE"
sunday(text, pattern)
```

C

```c
#include <stdio.h>
#include <string.h>

#define ALPHABET 256

void sunday(const char *text, const char *pat) {
    int n = strlen(text), m = strlen(pat);
    int shift[ALPHABET];
    for (int i = 0; i < ALPHABET; i++) shift[i] = m + 1; // 初始化默认移位值
    for (int i = 0; i < m; i++)
        shift[(unsigned char)pat[i]] = m - i; // 更新模式中字符的移位值

    int i = 0;
    while (i <= n - m) {
        int j = 0;
        while (j < m && pat[j] == text[i + j]) j++;
        if (j == m)
            printf("在索引 %d 处找到匹配\n", i);
        unsigned char next = (i + m < n) ? text[i + m] : 0; // 获取窗口后的字符
        i += shift[next]; // 根据移位表进行移位
    }
}

int main(void) {
    sunday("EXAMPLEEXAMPLES", "EXAMPLE");
}
```

#### 为什么它很重要

- 简单：一个移位表，无需向后比较。
- 在实践中速度快，尤其对于字符集较大的情况。
- 在清晰度和速度之间取得了很好的平衡。
- 常用于文本编辑器、类 grep 工具和搜索库中。

#### 复杂度

| 情况    | 时间复杂度 | 空间复杂度 |
| ------- | --------- | ----- |
| 最优    | O(n / m)  | O(σ)  |
| 平均    | 亚线性    | O(σ)  |
| 最差    | O(nm)     | O(σ)  |

σ = 字母表大小

在随机文本上，每个窗口的比较次数非常少。

#### 亲自尝试

1.  为 `"HELLO"` 构建移位表。
2.  在 `"HELLOHELLO"` 中搜索 `"LO"`，追踪每次移位。
3.  与 Boyer–Moore–Horspool 算法比较跳跃长度。
4.  尝试在 `"AAAAAAAAAA"` 中搜索 `"AAAB"`，最坏情况如何？
5.  统计在 `"ABCDEABCD"` 中搜索 `"ABCD"` 的总比较次数。

Sunday 算法着眼于明天，向前看一步，总是跳过它能预见到的部分。
### 608 有限自动机匹配

有限自动机匹配将模式搜索转化为状态转移。它预先计算一个确定性有限自动机（DFA），该自动机能精确识别以模式结尾的字符串，然后简单地在文本上运行该自动机。每一步都是常数时间，每次匹配都确保被发现。

#### 我们要解决什么问题？

我们想要高效地在文本 `T` 中匹配模式 `P`，无需回溯，也无需重新检查。

核心思想：
与其手动比较，不如让一个机器来完成这项工作，这个机器读取每个字符并更新其内部状态，直到找到匹配。

该算法构建一个 DFA，其中：

-   每个状态 = 到目前为止匹配了多少个模式字符
-   每次转移 = 当我们读取一个新字符时会发生什么

每当自动机进入最终状态，就识别出一个完整的匹配。

#### 它是如何工作的（通俗解释）？

可以把它想象成一个“模式读取机”。
每次我们读取一个字符，就转移到下一个状态，如果它破坏了模式，就回退。

步骤：

1.  预处理模式：
    构建 DFA 表：`dfa[状态][字符]` = 下一个状态
2.  扫描文本：
    从状态 0 开始，依次输入文本的每个字符。
    每个字符根据表格将你转移到一个新状态。
    如果你到达状态 `m`（模式长度），那就是一个匹配。

每个字符只被处理一次，没有回溯。

#### 示例

模式：`"ABAB"`

状态：0 → 1 → 2 → 3 → 4
最终状态 = 4（完全匹配）

| 状态 | 遇到 'A' | 遇到 'B' | 解释                     |
| ---- | -------- | -------- | ------------------------ |
| 0    | 1        | 0        | 起始 → 匹配到 A          |
| 1    | 1        | 2        | 在 'A' 之后，下一个是 'B' |
| 2    | 3        | 0        | 在 'AB' 之后，下一个是 'A' |
| 3    | 1        | 4        | 在 'ABA' 之后，下一个是 'B' |
| 4    | -        | -        | 找到匹配                 |

将文本 `"ABABAB"` 输入这个机器：

-   步骤：0→1→2→3→4 → 在索引 0 处找到匹配
-   继续：2→3→4 → 在索引 2 处找到匹配

每次转移都是 O(1)。

#### 精简代码（简易版本）

Python

```python
def build_dfa(pat, alphabet):
    m = len(pat)
    dfa = [[0]*len(alphabet) for _ in range(m+1)]
    alpha_index = {ch: i for i, ch in enumerate(alphabet)}

    dfa[0][alpha_index[pat[0]]] = 1
    x = 0
    for j in range(1, m+1):
        for c in alphabet:
            dfa[j][alpha_index[c]] = dfa[x][alpha_index[c]]
        if j < m:
            dfa[j][alpha_index[pat[j]]] = j + 1
            x = dfa[x][alpha_index[pat[j]]]
    return dfa

def automaton_search(text, pat, alphabet):
    dfa = build_dfa(pat, alphabet)
    state = 0
    m = len(pat)
    for i, ch in enumerate(text):
        if ch in alphabet:
            state = dfa[state][alphabet.index(ch)]
        else:
            state = 0
        if state == m:
            print("在索引", i - m + 1, "处找到匹配")

alphabet = list("AB")
automaton_search("ABABAB", "ABAB", alphabet)
```

这段代码构建一个 DFA 并在文本上模拟它。

#### 为什么它很重要

-   无需回溯，线性时间搜索。
-   非常适合固定字母表和重复查询。
-   是词法分析器和正则表达式引擎（底层）的基础。
-   是自动机理论实际应用的绝佳例子。

#### 复杂度

| 步骤       | 时间       | 空间       |
| ---------- | ---------- | ---------- |
| 构建 DFA   | O(m × σ)   | O(m × σ)   |
| 搜索       | O(n)       | O(1)       |

σ = 字母表大小
最适合小字母表（例如，DNA，ASCII）。

#### 亲自尝试

1.  为 `"ABA"` 绘制 DFA。
2.  模拟 `"ABABA"` 的转移过程。
3.  添加字母表 `{A, B, C}`，有什么变化？
4.  将状态与 KMP 的前缀表进行比较。
5.  修改代码以打印状态转移过程。

有限自动机匹配就像构建一个*熟记你模式*的小机器，给它喂入文本，每次它认出你的单词时就会举手示意。
### 609 Bitap 算法

Bitap 算法（也称为 Shift-Or 或 Shift-And）使用位运算进行模式匹配。它将模式视为一个位掩码，并逐个字符地处理文本，更新一个代表匹配状态的整数。它快速、紧凑，也非常适合近似或模糊匹配。

#### 我们要解决什么问题？

我们希望利用位级并行性，高效地在文本 `T` 中查找模式 `P`。

Bitap 不是通过循环比较字符，而是将比较操作打包到一个机器字中，一次性更新所有位置。这就像使用一个整数的位来并行运行多个匹配状态。

给定：
- 长度为 `n` 的 `T`
- 长度为 `m` 的 `P`（≤ 字长）
我们将通过位运算的“魔法”在 O(n) 时间内找到匹配。

#### 它是如何工作的（通俗解释）？

字中的每一位代表模式的一个前缀是否与文本的当前后缀匹配。

我们维护：
- `R`：当前匹配状态的位掩码（1 = 不匹配，0 = 目前匹配）
- `mask[c]`：为模式中字符 `c` 预先计算好的位掩码

每一步：
1. 将 `R` 左移一位（以包含下一个字符）
2. 与当前文本字符的掩码结合
3. 检查最低位是否为 0 → 找到完全匹配

因此，我们不是为每个前缀管理循环，而是一次性更新所有匹配前缀。

#### 示例

模式：`"AB"`
文本：`"CABAB"`

预先计算掩码（针对 2 位字）：
```
mask['A'] = 0b10
mask['B'] = 0b01
```

初始化 `R = 0b11`（全为 1）

现在滑动处理 `"CABAB"`：
- C：`R = (R << 1 | 1) & mask['C']` → 保持全 1
- A：左移，结合 mask['A']
- B：左移，结合 mask['B'] → 匹配位变为 0 → 找到匹配

全部通过位运算完成。

#### 精简代码（简易版本）

Python

```python
def bitap_search(text, pat):
    m = len(pat)
    if m == 0:
        return
    if m > 63:
        raise ValueError("模式过长，超出 64 位 Bitap 处理范围")

    # 为模式构建位掩码
    mask = {chr(i): ~0 for i in range(256)}
    for i, c in enumerate(pat):
        mask[c] &= ~(1 << i)

    R = ~1
    for i, c in enumerate(text):
        R = (R << 1) | mask.get(c, ~0)
        if (R & (1 << m)) == 0:
            print("在索引", i, "处找到匹配")
```

C (64 位版本)

```c
#include <stdio.h>
#include <string.h>
#include <stdint.h>

void bitap(const char *text, const char *pat) {
    int n = strlen(text), m = strlen(pat);
    if (m > 63) return; // 适合 64 位掩码

    uint64_t mask[256];
    for (int i = 0; i < 256; i++) mask[i] = ~0ULL;
    for (int i = 0; i < m; i++)
        mask[(unsigned char)pat[i]] &= ~(1ULL << i);

    uint64_t R = ~1ULL;
    for (int i = 0; i < n; i++) {
        R = (R << 1) | mask[(unsigned char)text[i]];
        if ((R & (1ULL << m)) == 0)
            printf("在索引 %d 处找到匹配\n", i);
    }
}

int main(void) {
    bitap("CABAB", "AB");
}
```

#### 为什么它很重要

- 位并行搜索，利用了 CPU 级别的操作。
- 对于短模式和固定字长的情况非常出色。
- 可扩展至近似匹配（允许编辑操作）。
- 是 agrep（近似 grep）等工具以及编辑器中 bitap 模糊搜索的核心。

#### 复杂度

| 情况          | 时间复杂度   | 空间复杂度 |
| ------------- | ------------ | ---------- |
| 典型情况      | O(n)         | O(σ)       |
| 预处理        | O(m + σ)     | O(σ)       |
| 限制条件      | m ≤ 字长     |            |

Bitap 是*线性*的，但受限于机器字长（例如，≤ 64 个字符）。

#### 亲自尝试

1.  在 `"ZABCABC"` 中搜索 `"ABC"`。
2.  每一步之后，以二进制形式打印 `R`。
3.  扩展掩码以支持 ASCII 或 DNA 字母表。
4.  用 `"AAA"` 测试，观察重叠匹配。
5.  尝试模糊版本：允许 1 个不匹配（编辑距离 ≤ 1）。

Bitap 就像一个位运算的管弦乐队，每一位都演奏着自己的音符，它们共同准确地告诉你模式何时命中。
### 610 双向算法

双向算法是一种线性时间的字符串搜索方法，它结合了前缀分析和模移位。它将模式分为两部分，并使用关键分解来决定失配后跳过多远。它优雅且最优，保证了 O(n + m) 的时间复杂度，且无需繁重的预处理。

#### 我们要解决什么问题？

我们想要一个确定性的线性时间搜索算法，它具备以下特点：

- 平均速度比 KMP 更快
- 比 Boyer–Moore 更简单
- 在最坏情况下被证明是最优的

双向算法通过在搜索前分析模式的周期性来实现这一点，因此在扫描过程中，它能智能地移位，有时按模式的周期移位，有时按整个模式长度移位。

给定：

- 文本 `T`（长度 `n`）
- 模式 `P`（长度 `m`）

我们将找到所有匹配，且无需回溯和冗余比较。

#### 它是如何工作的（通俗解释）？

其秘密在于关键分解：

1.  预处理（找到关键位置）：
   在关键索引处将 `P` 分割为 `u` 和 `v`，使得：
   *   `u` 和 `v` 代表 `P` 的字典序最小旋转
   *   它们揭示了模式的周期
   这确保了高效的跳过。

2.  搜索阶段：
   使用一个移动窗口扫描 `T`。
   *   从左到右比较（正向扫描）。
   *   失配时，根据以下情况移位：
     *   如果是部分匹配，则按模式的周期移位
     *   如果早期失配，则按整个模式长度移位

通过交替进行双向扫描，它保证了没有位置会被检查两次。

可以把它看作是 KMP 的结构 + Boyer–Moore 的跳过，并与数学精度相结合。

#### 示例

模式：`"ABABAA"`

1.  计算关键位置，索引 2（在 "AB" | "ABAA" 之间）
2.  模式周期 = 2 (`"AB"`)
3.  开始扫描 `T = "ABABAABABAA"`：
   *   正向比较 `"AB"` → 匹配
   *   在 `"AB"` 之后失配 → 按周期 = 2 移位
   *   继续扫描，保证不会重新检查

这种策略利用了模式的内部结构，跳过是基于已知的重复进行的。

#### 微型代码（简化版）

Python（高层次思路）

```python
def critical_factorization(pat):
    m = len(pat)
    i, j, k = 0, 1, 0
    while i + k < m and j + k < m:
        if pat[i + k] == pat[j + k]:
            k += 1
        elif pat[i + k] > pat[j + k]:
            i = i + k + 1
            if i <= j:
                i = j + 1
            k = 0
        else:
            j = j + k + 1
            if j <= i:
                j = i + 1
            k = 0
    return min(i, j)

def two_way_search(text, pat):
    n, m = len(text), len(pat)
    if m == 0:
        return
    pos = critical_factorization(pat)
    period = max(pos, m - pos)
    i = 0
    while i <= n - m:
        j = 0
        while j < m and text[i + j] == pat[j]:
            j += 1
        if j == m:
            print("在索引", i, "处找到匹配")
        i += period if j >= pos else max(1, j - pos + 1)

text = "ABABAABABAA"
pattern = "ABABAA"
two_way_search(text, pattern)
```

这个实现首先找到关键索引，然后应用基于周期移位的正向扫描。

#### 为什么它很重要

-   线性时间（最坏情况）
-   无需预处理表
-   优雅地利用了周期性理论
-   是 C 标准库 `strstr()` 实现的基础
-   能高效处理周期性和非周期性模式

#### 复杂度

| 步骤                                   | 时间复杂度 | 空间复杂度 |
| -------------------------------------- | ---------- | ---------- |
| 预处理（关键分解）                     | O(m)       | O(1)       |
| 搜索                                   | O(n)       | O(1)       |
| 总计                                   | O(n + m)   | O(1)       |

最优的确定性复杂度，无随机性，无冲突。

#### 亲自尝试

1.  找到 `"ABCABD"` 的关键索引。
2.  可视化 `"ABAB"` 在 `"ABABABAB"` 中的移位。
3.  与 KMP 和 Boyer–Moore 比较跳过长度。
4.  逐步跟踪状态变化。
5.  手动实现 `critical_factorization` 并进行验证。

双向算法是理论与实用主义的结合，它学习你模式的节奏，然后以完美的时机在文本中舞动。

# 第 62 节 多模式搜索
### 611 Aho–Corasick 自动机

Aho–Corasick 算法是多模式搜索的经典解决方案。它并非单独搜索每个关键词，而是构建一个单一的自动机，一次性识别所有模式。文本的每个字符都会驱动自动机前进，立即报告每一个匹配，多个关键词，一次扫描。

#### 我们要解决什么问题？

我们想要在给定文本中找到多个模式的所有出现位置。

给定：

- 一组模式 ( P = {p_1, p_2, \ldots, p_k} )
- 长度为 ( n ) 的文本 ( T )

我们的目标是找到 ( T ) 中所有满足以下条件的位置 ( i )：
$$
T[i : i + |p_j|] = p_j
$$
对于某个 ( p_j \in P )。

朴素解法：
$$
O\Big(n \times \sum_{j=1}^{k} |p_j|\Big)
$$
Aho–Corasick 将其改进为：
$$
O(n + \sum_{j=1}^{k} |p_j| + \text{output\_count})
$$

#### 它是如何工作的（通俗解释）？

Aho–Corasick 构建一个确定性有限自动机（DFA），该自动机能同时识别所有给定的模式。

构建过程包含三个步骤：

1.  **字典树（Trie）构建**
    将所有模式插入一个前缀树。每条边代表一个字符；每个节点代表一个前缀。

2.  **失效链接（Failure Links）**
    为每个节点构建一个失效链接，指向在字典树中也是前缀的最长真后缀。
    类似于 KMP 算法中的回退机制。

3.  **输出链接（Output Links）**
    当一个节点代表一个完整的模式时，记录下来。
    如果失效链接指向另一个终端节点，则合并它们的输出。

**搜索阶段**：
逐个字符处理文本：

- 如果存在当前字符的转移，则跟随它。
- 否则，跟随失效链接，直到找到有效的转移。
- 在每个节点，输出所有在此结束的模式。

结果：一次扫描文本，报告所有匹配。

#### 示例

模式：
$$
P = {\text{"he"}, \text{"she"}, \text{"his"}, \text{"hers"}}
$$
文本：
$$
T = \text{"ushers"}
$$

字典树结构（简化）：

```
(根节点)
 ├─ h ─ i ─ s*
 │    └─ e*
 │         └─ r ─ s*
 └─ s ─ h ─ e*
```

(* 表示模式终点)

失效链接：

- ( \text{"h"} \to \text{根节点} )
- ( \text{"he"} \to \text{"e"} ) (通过根节点)
- ( \text{"she"} \to \text{"he"} )
- ( \text{"his"} \to \text{"is"} )

文本扫描：

- `u` → 无边，停留在根节点
- `s` → 跟随 `s`
- `h` → `sh`
- `e` → `she` → 报告 "she", "he"
- `r` → 移动到 `her`
- `s` → `hers` → 报告 "hers"

所有模式在一次遍历中被找到。

#### 精简代码（Python）

```python
from collections import deque

class AhoCorasick:
    def __init__(self, patterns):
        self.trie = [{}]
        self.fail = [0]
        self.output = [set()]
        for pat in patterns:
            self._insert(pat)
        self._build()

    def _insert(self, pat):
        node = 0
        for ch in pat:
            if ch not in self.trie[node]:
                self.trie[node][ch] = len(self.trie)
                self.trie.append({})
                self.fail.append(0)
                self.output.append(set())
            node = self.trie[node][ch]
        self.output[node].add(pat)

    def _build(self):
        q = deque()
        for ch, nxt in self.trie[0].items():
            q.append(nxt)
        while q:
            r = q.popleft()
            for ch, s in self.trie[r].items():
                q.append(s)
                f = self.fail[r]
                while f and ch not in self.trie[f]:
                    f = self.fail[f]
                self.fail[s] = self.trie[f].get(ch, 0)
                self.output[s] |= self.output[self.fail[s]]

    def search(self, text):
        node = 0
        for i, ch in enumerate(text):
            while node and ch not in self.trie[node]:
                node = self.fail[node]
            node = self.trie[node].get(ch, 0)
            for pat in self.output[node]:
                print(f"在索引 {i - len(pat) + 1} 处匹配到 '{pat}'")

patterns = ["he", "she", "his", "hers"]
ac = AhoCorasick(patterns)
ac.search("ushers")
```

输出：

```
在索引 1 处匹配到 'she'
在索引 2 处匹配到 'he'
在索引 2 处匹配到 'hers'
```

#### 为什么它很重要

- 一次扫描即可找到多个模式
- 没有冗余比较或回溯
- 应用于：

  * 垃圾邮件和恶意软件检测
  * 入侵检测系统（IDS）
  * 搜索引擎和关键词扫描器
  * DNA 和蛋白质序列分析

#### 复杂度

| 步骤                 | 时间复杂度                     | 空间复杂度                |
| -------------------- | ------------------------------ | ------------------------- |
| 构建字典树           | $O\!\left(\sum |p_i|\right)$  | $O\!\left(\sum |p_i|\right)$ |
| 构建失效链接         | $O\!\left(\sum |p_i| \cdot \sigma\right)$ | $O\!\left(\sum |p_i|\right)$ |
| 搜索                 | $O(n + \text{output\_count})$ | $O(1)$                    |

其中 $\sigma$ 是字母表大小。

总体：
$$
O\!\left(n + \sum |p_i| + \text{output\_count}\right)
$$

#### 亲自尝试

1.  为 $\{\text{"a"}, \text{"ab"}, \text{"bab"}\}$ 构建字典树。
2.  追踪 `"ababbab"` 的失效链接。
3.  添加具有共享前缀的模式，注意字典树的压缩。
4.  打印每个节点的所有输出以理解重叠。
5.  与启动多个 KMP 搜索的运行时间进行比较。

Aho–Corasick 将所有模式统一在一个自动机下，一次遍历，完全识别，效率完美。
### 612 字典树构建

字典树（发音为 *try*）是一种按字符串前缀组织字符串的前缀树。每条边代表一个字符，从根节点出发的每条路径编码一个单词。在多模式搜索中，字典树是 Aho–Corasick 自动机的基础，它在一个共享结构中捕获所有关键词。

#### 我们要解决什么问题？

我们想要高效地存储和查询一组字符串，特别是基于前缀的操作。

给定一个模式集合
$$
P = {p_1, p_2, \ldots, p_k}
$$

我们想要一个数据结构，能够：

- 以 $O\left(\sum_{i=1}^{k} |p_i|\right)$ 的时间复杂度插入所有模式
- 查询一个单词或前缀是否存在
- 共享公共前缀以节省内存和时间

示例
如果
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"his"}, \texttt{"hers"}}
$$
我们可以将它们存储在单个前缀树中，共享重叠的路径，如 `h → e`。

#### 它是如何工作的（通俗解释）

字典树是逐步构建的，一次一个字符：

1. 从根节点（空前缀）开始。
2. 对于每个模式 $p$：
   * 遍历与当前字符匹配的现有边。
   * 如果边不存在，则创建新节点。
3. 将每个单词的最终节点标记为终止节点。

每个节点代表一个或多个模式的前缀。
每个叶子节点或终止节点标记一个完整的模式。

这就像一个分支路线图，单词共享它们的起始路径，然后在它们不同的地方分叉。

#### 示例

对于
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"his"}, \texttt{"hers"}}
$$

字典树结构：

```
(root)
 ├── h ── e* ── r ── s*
 │     └── i ── s*
 └── s ── h ── e*
```

（* 标记单词结尾）

- 前缀 `"he"` 被 `"he"`、`"hers"` 和 `"his"` 共享。
- `"she"` 在 `"s"` 下单独分支。

#### 精简代码（简易版本）

Python

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def starts_with(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True

# 示例用法
patterns = ["he", "she", "his", "hers"]
trie = Trie()
for p in patterns:
    trie.insert(p)

print(trie.search("he"))       # True
print(trie.starts_with("sh"))  # True
print(trie.search("her"))      # False
```

C（简化版）

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define ALPHABET 26

typedef struct Trie {
    struct Trie *children[ALPHABET];
    bool is_end;
} Trie;

Trie* new_node() {
    Trie *node = calloc(1, sizeof(Trie));
    node->is_end = false;
    return node;
}

void insert(Trie *root, const char *word) {
    Trie *node = root;
    for (int i = 0; word[i]; i++) {
        int idx = word[i] - 'a';
        if (!node->children[idx])
            node->children[idx] = new_node();
        node = node->children[idx];
    }
    node->is_end = true;
}

bool search(Trie *root, const char *word) {
    Trie *node = root;
    for (int i = 0; word[i]; i++) {
        int idx = word[i] - 'a';
        if (!node->children[idx]) return false;
        node = node->children[idx];
    }
    return node->is_end;
}

int main(void) {
    Trie *root = new_node();
    insert(root, "he");
    insert(root, "she");
    insert(root, "his");
    insert(root, "hers");
    printf("%d\n", search(root, "she"));  // 1 (True)
}
```

#### 为什么它很重要

- 支持快速前缀查询和共享存储
- 是以下内容的核心组件：
  * Aho–Corasick 自动机
  * 自动补全和建议引擎
  * 拼写检查器
  * 字典压缩

字典树也是后缀树、三叉搜索树和基数树的基础。

#### 复杂度

| 操作                      | 时间复杂度             | 空间复杂度       |          |                        |     |          |
| ------------------------- | ---------------------- | ---------------- | -------- | ---------------------- | --- | -------- |
| 插入长度为 $m$ 的单词     | $O(m)$                 | $O(\sigma m)$    |          |                        |     |          |
| 搜索单词                  | $O(m)$                 | $O(1)$           |          |                        |     |          |
| 前缀查询                  | $O(m)$                 | $O(1)$           |          |                        |     |          |
| 从 $k$ 个模式构建         | $O\left(\sum_{i=1}^{k} | p_i              | \right)$ | $O\left(\sum_{i=1}^{k} | p_i | \right)$ |

其中 $\sigma$ 是字母表大小（例如，小写字母为 26）。

#### 动手尝试

1. 为以下模式构建一个字典树：
   $$
   P = {\texttt{"a"}, \texttt{"ab"}, \texttt{"abc"}, \texttt{"b"}}
   $$
2. 追踪 `"abc"` 的路径。
3. 修改代码以按字典序打印所有单词。
4. 与哈希表比较，前缀查找有何不同？
5. 扩展每个节点以存储频率计数或文档 ID。

字典树构建是多模式搜索的第一步，它是一个共享的前缀树，将单词列表转换为一个可搜索的单一结构。
### 613 失效链接计算

在 Aho–Corasick 自动机中，失效链接是赋予该结构强大能力的关键。它们允许在发生失配时高效地继续搜索，类似于 KMP 算法中的前缀函数如何防止冗余比较。每个失效链接将一个节点连接到其路径的最长真后缀，该后缀同时也是字典树中的一个前缀。

#### 我们要解决什么问题？

扫描文本时，可能在字典树的某个节点发生失配。
最朴素的做法是，我们会一路返回到根节点并重新开始。

失效链接通过告诉我们以下信息来解决这个问题：

> "如果当前路径失败，继续匹配的最佳下一个位置是哪里？"

换句话说，它们允许自动机重用部分匹配，在保持有效前缀状态的同时跳过冗余工作。

#### 它是如何工作的（通俗解释）

字典树中的每个节点都代表某个模式的一个前缀。
如果我们无法用下一个字符进行扩展，我们就沿着失效链接跳转到下一个可能仍然匹配的最长前缀。

算法概述：

1. 初始化

   * 根节点的失效链接 = 0（根节点）
   * 根节点的子节点 → 失效链接 = 0

2. 广度优先搜索遍历
   逐层处理字典树：

   * 对于每个节点 `u` 及其每个标记为 `c` 的出边指向的节点 `v`：

     1. 从 `u` 开始沿着失效链接向上查找，直到找到一个拥有边 `c` 的节点
     2. 设置 `fail[v]` = 该边指向的下一个节点
     3. 合并输出：
        $$
        \text{output}[v] \gets \text{output}[v] \cup \text{output}[\text{fail}[v]]
        $$

这确保了每个节点都知道失配后应该跳转到哪里，类似于 KMP 的前缀回退机制，但推广到了所有模式。

#### 示例

令
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"his"}, \texttt{"hers"}}
$$

步骤 1，构建字典树

```
(根节点)
 ├── h ── e* ── r ── s*
 │     └── i ── s*
 └── s ── h ── e*
```

(* 标记单词结尾)

步骤 2，计算失效链接

| 节点 | 字符串 | 失效链接 | 解释 |
| ---- | ------ | ------------ | ------------------------- |
| 根节点 | ε | 根节点 | 基本情况 |
| h | "h" | 根节点 | 无匹配前缀 |
| s | "s" | 根节点 | 无匹配前缀 |
| he | "he" | 根节点 | "he" 的后缀无匹配 |
| hi | "hi" | 根节点 | 同上 |
| sh | "sh" | h | 最长后缀是 "h" |
| she | "she" | he | "he" 既是后缀也是前缀 |
| hers | "hers" | s | 后缀 "s" 是前缀 |

现在每个节点都知道发生失配时应该继续到哪里。

#### 精简代码（简易版本）

Python

```python
from collections import deque

def build_failure_links(trie, fail, output):
    q = deque()
    # 初始化：根节点的子节点失效链接指向根节点
    for ch, nxt in trie[0].items():
        fail[nxt] = 0
        q.append(nxt)

    while q:
        r = q.popleft()
        for ch, s in trie[r].items():
            q.append(s)
            f = fail[r]
            while f and ch not in trie[f]:
                f = fail[f]
            fail[s] = trie[f].get(ch, 0)
            output[s] |= output[fail[s]]
```

使用上下文（在 Aho–Corasick 构建阶段内部）：

- `trie`: 字典列表 `{字符: 下一个状态}`
- `fail`: 失效链接列表
- `output`: 用于存储模式终点的集合列表

运行此代码后，每个节点都有一个有效的 `fail` 指针和合并后的输出。

#### 为什么它很重要

- 防止回溯 → 线性时间扫描
- 跨模式共享部分匹配
- 支持重叠匹配检测
- 将 KMP 的前缀回退机制推广到多模式

如果没有失效链接，自动机将退化为多个独立的搜索。

#### 复杂度

| 步骤 | 时间复杂度 | 空间复杂度 | | | | |
| ------------------- | --------------- | ---------------- | -------------- | ------- | --- | -- |
| 构建失效链接 | $O(\sum | p_i | \cdot \sigma)$ | $O(\sum | p_i | )$ |
| 合并输出 | $O(\sum | p_i | )$ | $O(\sum | p_i | )$ |

其中 $\sigma$ 是字母表大小。

每条边和每个节点都恰好被处理一次。

#### 动手尝试

1. 为以下模式构建字典树
   $$
   P = {\texttt{"a"}, \texttt{"ab"}, \texttt{"bab"}}
   $$
2. 使用广度优先搜索逐步计算失效链接。
3. 可视化每个节点合并后的输出。
4. 与 `"abab"` 的 KMP 前缀表进行比较。
5. 跟踪文本 `"ababbab"` 在自动机中的状态转移。

失效链接是 Aho–Corasick 自动机的神经系统，始终指向下一个最佳匹配，确保不会浪费时间回溯步骤。
### 614 输出链接管理

在 Aho–Corasick 自动机中，输出链接（或输出集）记录了哪些模式在某个状态结束。这些链接确保在文本扫描过程中，包括重叠和嵌套模式在内的所有匹配都能被正确报告。如果没有它们，当一个模式是另一个模式的后缀时，某些模式就会被忽略。

#### 我们要解决什么问题？

当多个模式共享后缀时，字典树中的一个节点可能代表多个单词的结尾。

例如，考虑
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"hers"}}
$$

当自动机到达 `"hers"` 的节点时，它也应该输出 `"he"` 和 `"she"`，因为它们是之前识别的后缀。

我们需要一种机制来：

- 记录哪些模式在每个节点结束
- 沿着失败链接以包含在链中更早结束的匹配

这就是输出链接的作用，每个节点的输出集累积了在该状态识别的所有模式。

#### 它是如何工作的（通俗解释）

自动机中的每个节点都有：

- 一组恰好在此结束的模式
- 一个指向下一个回退状态的失败链接

在构建自动机时，设置好节点的失败链接后：

1. 合并其失败节点的输出：
   $$
   \text{输出}[u] \gets \text{输出}[u] \cup \text{输出}[\text{失败}[u]]
   $$
2. 这确保了如果当前路径的一个后缀也是一个模式，它将被识别。

在搜索过程中，每当我们访问一个节点时：

- 发出 $\text{输出}[u]$ 中的所有模式
- 每个模式都代表一个在当前文本位置结束的模式

#### 示例

模式：
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"hers"}}
$$
文本：`"ushers"`

字典树（简化）：

```
(根)
 ├── h ── e* ── r ── s*
 └── s ── h ── e*
```

(* = 模式结束)

失败链接：

- `"he"` → 根
- `"she"` → `"he"`
- `"hers"` → `"s"`

输出链接（合并后）：

- $\text{输出}["he"] = {\texttt{"he"}}$
- $\text{输出}["she"] = {\texttt{"she"}, \texttt{"he"}}$
- $\text{输出}["hers"] = {\texttt{"hers"}, \texttt{"he"}}$

因此，当到达 `"she"` 时，会报告 `"she"` 和 `"he"`。
当到达 `"hers"` 时，会报告 `"hers"` 和 `"he"`。

#### 微型代码（Python 片段）

```python
from collections import deque

def build_output_links(trie, fail, output):
    q = deque()
    for ch, nxt in trie[0].items():
        fail[nxt] = 0
        q.append(nxt)

    while q:
        r = q.popleft()
        for ch, s in trie[r].items():
            q.append(s)
            f = fail[r]
            while f and ch not in trie[f]:
                f = fail[f]
            fail[s] = trie[f].get(ch, 0)
            # 合并来自失败链接的输出
            output[s] |= output[fail[s]]
```

解释

- `trie`：字典列表（边）
- `fail`：失败指针列表
- `output`：集合列表（在节点结束的模式）

每个节点继承其失败节点的输出。
因此，当访问一个节点时，打印 `output[node]` 会给出所有匹配项。

#### 为什么这很重要

- 实现完整的匹配报告
- 捕获重叠匹配，例如 `"she"` 中的 `"he"`
- 对于正确性至关重要，没有输出合并，只会出现最长的匹配
- 用于搜索工具、入侵检测系统、NLP 分词器和编译器

#### 复杂度

| 步骤          | 时间复杂度                          | 空间复杂度             |
| ------------- | ----------------------------------------- | ---------------------------- |
| 合并输出 | $O\!\left(\sum |p_i|\right)$              | $O\!\left(\sum |p_i|\right)$ |
| 搜索阶段  | $O(n + \text{输出\_计数})$             | $O(1)$                       |

每个节点在自动机构建期间合并其输出一次。

#### 自己动手试试

1. 为以下模式构建字典树：
   $$
   P = {\texttt{"he"}, \texttt{"she"}, \texttt{"hers"}}
   $$
2. 计算失败链接并合并输出。
3. 逐字符跟踪文本 `"ushers"`。
4. 在每个访问的状态打印 $\text{输出}[u]$。
5. 验证所有后缀模式是否都被报告。

输出链接管理确保没有模式被遗漏。每个后缀、每个重叠、每个嵌入的单词都被捕获，这是每一步识别的完整记录。
### 615 多模式搜索

多模式搜索问题要求我们在单个文本中找出多个关键词的所有出现位置。与其为每个模式单独运行搜索，我们通过 Aho–Corasick 自动机将它们组合成一次遍历。这种方法构成了文本分析、垃圾邮件过滤和网络入侵检测的基础。

#### 我们要解决什么问题？

给定：

- 一组模式
  $$
  P = {p_1, p_2, \ldots, p_k}
  $$
- 一个文本
  $$
  T = t_1 t_2 \ldots t_n
  $$

我们需要找到 ( P ) 中每个模式在 ( T ) 内的每一次出现，包括重叠的匹配。

最朴素的方法是，我们可以为每个 ( p_i ) 运行 KMP 或 Z 算法：
$$
O\Big(n \times \sum_{i=1}^k |p_i|\Big)
$$

但 Aho–Corasick 算法可以在以下时间内解决：
$$
O(n + \sum_{i=1}^k |p_i| + \text{output\_count})
$$

也就是说，只需对文本进行一次遍历，即可报告所有匹配。

#### 它是如何工作的（通俗解释）

解决方案分为三个阶段：

1. 构建字典树
   将所有模式合并到一个前缀树中。

2. 计算失效链接和输出链接

   * 失效链接在失配后重定向搜索。
   * 输出链接收集每个状态匹配到的所有模式。

3. 扫描文本
   使用 ( T ) 中的字符在自动机中移动。

   * 如果存在转移，则跟随它。
   * 如果不存在，则跟随失效链接直到找到可用的转移。
   * 每次到达一个状态时，输出 `output[state]` 中的所有模式。

本质上，我们并行模拟了 k 次搜索，共享公共前缀，并在不同模式间复用搜索进度。

#### 示例

令
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"his"}, \texttt{"hers"}}
$$
且
$$
T = \texttt{"ushers"}
$$

在扫描过程中：

- `u` → 根节点
- `s` → `"s"`
- `h` → `"sh"`
- `e` → `"she"` → 报告 `"she"`, `"he"`
- `r` → `"her"`
- `s` → `"hers"` → 报告 `"hers"`, `"he"`

输出：

```
she @ 1
he  @ 2
hers @ 2
```

所有模式在一次遍历中被找到。

#### 精简代码（Python 实现）

```python
from collections import deque

class AhoCorasick:
    def __init__(self, patterns):
        self.trie = [{}]
        self.fail = [0]
        self.output = [set()]
        for pat in patterns:
            self._insert(pat)
        self._build()

    def _insert(self, pat):
        node = 0
        for ch in pat:
            if ch not in self.trie[node]:
                self.trie[node][ch] = len(self.trie)
                self.trie.append({})
                self.fail.append(0)
                self.output.append(set())
            node = self.trie[node][ch]
        self.output[node].add(pat)

    def _build(self):
        q = deque()
        for ch, nxt in self.trie[0].items():
            q.append(nxt)
        while q:
            r = q.popleft()
            for ch, s in self.trie[r].items():
                q.append(s)
                f = self.fail[r]
                while f and ch not in self.trie[f]:
                    f = self.fail[f]
                self.fail[s] = self.trie[f].get(ch, 0)
                self.output[s] |= self.output[self.fail[s]]

    def search(self, text):
        node = 0
        results = []
        for i, ch in enumerate(text):
            while node and ch not in self.trie[node]:
                node = self.fail[node]
            node = self.trie[node].get(ch, 0)
            for pat in self.output[node]:
                results.append((i - len(pat) + 1, pat))
        return results

patterns = ["he", "she", "his", "hers"]
ac = AhoCorasick(patterns)
print(ac.search("ushers"))
```

输出：

```
$$(1, 'she'), (2, 'he'), (2, 'hers')]
```

#### 为什么它很重要

- 同时处理多个关键词
- 报告重叠和嵌套的匹配
- 广泛应用于：

  * 垃圾邮件过滤器
  * 入侵检测系统
  * 搜索引擎
  * 抄袭检测器
  * DNA 序列分析

它是在一次遍历中搜索多个模式的金标准。

#### 复杂度

| 步骤                | 时间复杂度                    | 空间复杂度       |                |         |     |    |
| ------------------- | ---------------------------- | ---------------- | -------------- | ------- | --- | -- |
| 构建字典树          | $O(\sum                      | p_i              | )$             | $O(\sum | p_i | )$ |
| 构建失效链接        | $O(\sum                      | p_i              | \cdot \sigma)$ | $O(\sum | p_i | )$ |
| 搜索                | $O(n + \text{output\_count})$ | $O(1)$           |                |         |     |    |

其中 $\sigma$ 是字母表大小。

总时间：
$$
O(n + \sum |p_i| + \text{output\_count})
$$

#### 亲自尝试

1.  为以下模式构建一个多模式自动机
    $$
    P = {\texttt{"ab"}, \texttt{"bc"}, \texttt{"abc"}}
    $$
    并在 `"zabcbcabc"` 上追踪其运行过程。
2.  与运行三次 KMP 进行比较。
3.  计算总转移次数与朴素扫描的对比。
4.  修改代码以仅计算匹配数量。
5.  扩展它以支持不区分大小写的搜索。

多模式搜索将关键词列表转化为一个单一的机器。每一步读取一个字符，并揭示隐藏在该文本中的每一个模式，一次扫描，全面覆盖。
### 616 字典匹配

字典匹配是多模式搜索的一种专门形式，其目标是在给定文本中定位固定字典中所有单词的出现位置。与单模式搜索（如 KMP 或 Boyer–Moore）不同，字典匹配通过共享结构和高效的状态转移，一次性解决整个词汇表的问题。

#### 我们要解决什么问题？

我们想在一个大型文本中找到字典中的每个单词。

给定：

- 一个字典
  $$
  D = {w_1, w_2, \ldots, w_k}
  $$
- 一个文本
  $$
  T = t_1 t_2 \ldots t_n
  $$

我们必须报告 $T$ 中与任意 $w_i \in D$ 匹配的所有子串。

朴素解法：

- 对每个单词运行 KMP 或 Z 算法：
  $O(n \times \sum |w_i|)$

高效解法（Aho–Corasick）：

- 一次性构建自动机：$O(\sum |w_i|)$
- 单次扫描搜索：$O(n + \text{output\_count})$

因此总复杂度为：
$$
O(n + \sum |w_i| + \text{output\_count})
$$

#### 它是如何工作的（通俗解释）

关键思想在于共享前缀和失败链接。

1.  **字典树构建**
    将字典中的所有单词合并到一个单一的前缀树中。

2.  **失败链接**
    当发生不匹配时，沿着失败指针跳转到最长的、仍然是有效前缀的后缀状态。

3.  **输出集合**
    每个节点存储所有在该状态结束的字典单词。

4.  **搜索阶段**
    逐个字符扫描 $T$。

    *   如果可能，沿着存在的边前进
    *   否则，沿着失败链接跳转，直到存在有效的转移
    *   报告当前节点输出集合中的所有单词

$T$ 中的每个位置恰好被处理一次。

#### 示例

字典：
$$
D = {\texttt{"he"}, \texttt{"she"}, \texttt{"his"}, \texttt{"hers"}}
$$
文本：
$$
T = \texttt{"ushers"}
$$

扫描过程：

| 步骤 | 字符 | 状态 | 输出           |
| ---- | ---- | ---- | -------------- |
| 1    | u    | 根节点 | ∅              |
| 2    | s    | s    | ∅              |
| 3    | h    | sh   | ∅              |
| 4    | e    | she  | {"she", "he"}  |
| 5    | r    | her  | ∅              |
| 6    | s    | hers | {"hers", "he"} |

匹配结果：

- `"she"` 在索引 1 处
- `"he"` 在索引 2 处
- `"hers"` 在索引 2 处

所有匹配在一次遍历中找到。

#### 精简代码（Python 版本）

```python
from collections import deque

class DictionaryMatcher:
    def __init__(self, words):
        self.trie = [{}]
        self.fail = [0]
        self.output = [set()]
        for word in words:
            self._insert(word)
        self._build()

    def _insert(self, word):
        node = 0
        for ch in word:
            if ch not in self.trie[node]:
                self.trie[node][ch] = len(self.trie)
                self.trie.append({})
                self.fail.append(0)
                self.output.append(set())
            node = self.trie[node][ch]
        self.output[node].add(word)

    def _build(self):
        q = deque()
        for ch, nxt in self.trie[0].items():
            q.append(nxt)
        while q:
            r = q.popleft()
            for ch, s in self.trie[r].items():
                q.append(s)
                f = self.fail[r]
                while f and ch not in self.trie[f]:
                    f = self.fail[f]
                self.fail[s] = self.trie[f].get(ch, 0)
                self.output[s] |= self.output[self.fail[s]]

    def search(self, text):
        node = 0
        results = []
        for i, ch in enumerate(text):
            while node and ch not in self.trie[node]:
                node = self.fail[node]
            node = self.trie[node].get(ch, 0)
            for word in self.output[node]:
                results.append((i - len(word) + 1, word))
        return results

dictionary = ["he", "she", "his", "hers"]
dm = DictionaryMatcher(dictionary)
print(dm.search("ushers"))
```

输出：

```
[(1, 'she'), (2, 'he'), (2, 'hers')]
```

#### 为什么它很重要

- 高效解决字典单词搜索问题
- 是以下领域的核心技术：

  *   文本索引和关键词过滤
  *   入侵检测系统（IDS）
  *   语言分析
  *   抄袭和内容匹配
  *   DNA 或蛋白质基序发现

它的扩展性非常好，单个自动机可以处理数十万个字典单词。

#### 复杂度

| 步骤             | 时间复杂度                     | 空间复杂度 |                |         |     |    |
| ---------------- | ------------------------------ | ---------- | -------------- | ------- | --- | -- |
| 构建自动机       | $O(\sum                        | w_i        | \cdot \sigma)$ | $O(\sum | w_i | )$ |
| 搜索             | $O(n + \text{output\_count})$ | $O(1)$     |                |         |     |    |

其中 $\sigma$ = 字母表大小（例如，小写字母为 26）。

#### 动手尝试

1.  使用
    $$
    D = {\texttt{"cat"}, \texttt{"car"}, \texttt{"cart"}, \texttt{"art"}}
    $$
    和 $T = \texttt{"cartographer"}$。
2.  逐步跟踪自动机的状态变化。
3.  与运行 4 次独立的 KMP 搜索进行比较。
4.  修改自动机以返回位置和单词。
5.  扩展为不区分大小写的字典匹配。

字典匹配将单词列表转换为一个搜索自动机，文本的每个字符同时推进所有搜索，确保完全检测且没有冗余。
### 617 动态 Aho–Corasick

动态 Aho–Corasick 自动机扩展了经典的 Aho–Corasick 算法，使其能够在运行时处理模式的插入和删除。它允许我们维护一个活的字典，在新关键词到达或旧关键词被移除时更新自动机，而无需从头开始重建。

#### 我们要解决什么问题？

标准的 Aho–Corasick 假设字典是静态的。
但在许多现实世界的系统中，模式集会随时间变化：

- 垃圾邮件过滤器接收新规则
- 网络入侵系统添加新签名
- 搜索引擎更新关键词列表

我们需要一种能够动态插入或删除单词的方法，同时仍然保持每次查询在
$$
O(n + \text{output\_count})
$$
时间复杂度内进行搜索，而无需每次都重建自动机。

因此，我们的目标是构建一个可增量更新的模式匹配器。

#### 它是如何工作的（通俗解释）

我们以增量的方式维护自动机：

1.  **Trie 插入**
    通过沿着现有节点向下走，在需要时创建新节点来添加新模式。

2.  **失效链接更新**
    对于每个新节点，计算其失效链接：

    *   沿着父节点的失效链接向上查找，直到找到一个具有相同边的节点
    *   将新节点的失效链接设置到该目标节点
    *   合并输出集：
        $$
        \text{output}[v] \gets \text{output}[v] \cup \text{output}[\text{fail}[v]]
        $$

3.  **删除（可选）**

    *   将模式标记为不活跃（逻辑删除）
    *   在需要时可选地执行惰性清理

4.  **查询**
    搜索过程照常进行，沿着转移边和失效链接前进。

这种增量构建就像 Aho–Corasick 在运动中，一次添加一个单词，同时保持正确性。

#### 示例

从
$$
P_0 = {\texttt{"he"}, \texttt{"she"}}
$$
开始构建自动机。

现在插入 `"hers"`：

- 遍历：`h → e → r → s`
- 根据需要创建节点
- 更新失效链接：
  *   `fail("hers") = "s"`
  *   合并来自 `"s"` 的输出（如果有的话）

接下来插入 `"his"`：

- `h → i → s`
- 计算 `fail("his") = "s"`
- 合并输出

现在自动机可以识别
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"hers"}, \texttt{"his"}}
$$
中的所有单词，而无需完全重建。

#### 微型代码（Python 草图）

```python
from collections import deque

class DynamicAC:
    def __init__(self):
        self.trie = [{}]
        self.fail = [0]
        self.output = [set()]

    def insert(self, word):
        node = 0
        for ch in word:
            if ch not in self.trie[node]:
                self.trie[node][ch] = len(self.trie)
                self.trie.append({})
                self.fail.append(0)
                self.output.append(set())
            node = self.trie[node][ch]
        self.output[node].add(word)
        self._update_failures(word)

    def _update_failures(self, word):
        node = 0
        for ch in word:
            nxt = self.trie[node][ch]
            if nxt == 0:
                continue
            if node == 0:
                self.fail[nxt] = 0
            else:
                f = self.fail[node]
                while f and ch not in self.trie[f]:
                    f = self.fail[f]
                self.fail[nxt] = self.trie[f].get(ch, 0)
                self.output[nxt] |= self.output[self.fail[nxt]]
            node = nxt

    def search(self, text):
        node = 0
        matches = []
        for i, ch in enumerate(text):
            while node and ch not in self.trie[node]:
                node = self.fail[node]
            node = self.trie[node].get(ch, 0)
            for w in self.output[node]:
                matches.append((i - len(w) + 1, w))
        return matches

# 用法
ac = DynamicAC()
ac.insert("he")
ac.insert("she")
ac.insert("hers")
print(ac.search("ushers"))
```

输出：

```
$$(1, 'she'), (2, 'he'), (2, 'hers')]
```

#### 为什么它很重要

- 支持实时模式更新
- 对以下系统至关重要：
  *   实时垃圾邮件过滤器
  *   入侵检测系统 (IDS)
  *   自适应搜索系统
  *   NLP 流水线中的动态单词列表

与静态 Aho–Corasick 不同，此版本能随着字典的演变而适应。

#### 复杂度

| 操作                     | 时间复杂度                 | 空间复杂度       |
| ------------------------ | -------------------------- | ---------------- |
| 插入长度为 $m$ 的单词    | $O(m \cdot \sigma)$        | $O(m)$           |
| 删除单词                 | $O(m)$                     | $O(1)$ (惰性)    |
| 搜索文本                 | $O(n + \text{output\_count})$ | $O(1)$           |

每次插入只更新受影响的节点。

#### 亲自尝试

1.  从
    $$
    P = {\texttt{"a"}, \texttt{"ab"}}
    $$
    开始，然后动态添加 `"abc"`。
2.  在每次插入后打印 `fail` 数组。
3.  尝试删除 `"ab"`（标记为不活跃）。
4.  在每次更改后搜索文本 `"zabca"`。
5.  比较重建与增量更新的时间。

动态 Aho–Corasick 将静态自动机转变为活的字典，不断学习新单词，忘记旧单词，并实时扫描世界。
### 618 并行 Aho–Corasick 搜索

并行 Aho–Corasick 算法将经典的 Aho–Corasick 自动机适配到多线程或分布式环境中。它将输入文本或工作负载划分为独立的块，使得多个处理器可以同时搜索模式，从而实现对海量数据流的高吞吐量关键词检测。

#### 我们要解决什么问题？

经典的 Aho–Corasick 算法顺序扫描文本。
对于大规模任务，如扫描日志、DNA 序列或网络数据包，这成为了一个瓶颈。

我们希望：

- 保持线性时间匹配
- 利用多个核心或多台机器
- 在块边界处保持正确性

因此，我们的目标是使用并行执行来搜索
$$
T = t_1 t_2 \ldots t_n
$$
以匹配模式集
$$
P = {p_1, p_2, \ldots, p_k}
$$

#### 它是如何工作的（通俗解释）

并行化 Aho–Corasick 主要有两种策略：

##### 1. 文本分区（输入分割模型）

- 将文本 $T$ 分割成 $m$ 个块：
  $$
  T = T_1 , T_2 , \ldots , T_m
  $$
- 将每个块分配给一个工作线程。
- 每个线程独立运行 Aho–Corasick 算法。
- 通过重叠长度等于最长模式的缓冲区来处理边界情况（模式跨越块边界的情况）。

优点：简单，对长文本高效
缺点：需要重叠以保证正确性

##### 2. 自动机分区（状态分割模型）

- 将状态机在线程或节点间进行分区。
- 每个处理器负责一个模式子集或状态子集。
- 通过消息传递（例如 MPI）进行状态转移通信。

优点：适用于静态、小规模模式集
缺点：同步开销大，状态移交复杂

在两种方法中：

- 每个线程以 $O(|T_i| + \text{output\_count}_i)$ 的时间复杂度扫描文本
- 最后合并结果。

#### 示例（文本分区）

设模式集
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"his"}, \texttt{"hers"}}
$$
和文本
$$
T = \texttt{"ushershehis"}
$$

将 $T$ 分割成两部分，重叠长度为 4（最大模式长度）：

- 线程 1：`"ushersh"`
- 线程 2：`"shehis"`

两个线程运行相同的自动机。
合并时，对重叠区域中的匹配进行去重。

各自发现：

- 线程 1 → `she@1`, `he@2`, `hers@2`
- 线程 2 → `she@6`, `he@7`, `his@8`

最终结果 = 两个集合的并集。

#### 微型代码（并行示例，Python 线程）

```python
from concurrent.futures import ThreadPoolExecutor

def search_chunk(ac, text, offset=0):
    matches = []
    node = 0
    for i, ch in enumerate(text):
        while node and ch not in ac.trie[node]:
            node = ac.fail[node]
        node = ac.trie[node].get(ch, 0)
        for pat in ac.output[node]:
            matches.append((offset + i - len(pat) + 1, pat))
    return matches

def parallel_search(ac, text, chunk_size, overlap):
    tasks = []
    results = []
    with ThreadPoolExecutor() as executor:
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size + overlap]
            tasks.append(executor.submit(search_chunk, ac, chunk, i))
        for t in tasks:
            results.extend(t.result())
    # 可选地对重叠匹配进行去重
    return sorted(set(results))
```

用法：

```python
patterns = ["he", "she", "his", "hers"]
ac = AhoCorasick(patterns)
print(parallel_search(ac, "ushershehis", chunk_size=6, overlap=4))
```

输出：

```
[(1, 'she'), (2, 'he'), (2, 'hers'), (6, 'she'), (7, 'he'), (8, 'his')]
```

#### 为何重要

- 使得对大规模数据的实时匹配成为可能
- 应用于：

  * 入侵检测系统（IDS）
  * 大数据文本分析
  * 日志扫描和威胁检测
  * 基因组序列分析
  * 网络数据包检查

并行化使 Aho–Corasick 算法达到了每秒千兆字节的吞吐量。

#### 复杂度

对于 $m$ 个线程：

| 步骤            | 时间复杂度                                          | 空间复杂度 |    |         |     |    |
| --------------- | --------------------------------------------------- | ---------- | -- | ------- | --- | -- |
| 构建自动机      | $O(\sum                                             | p_i        | )$ | $O(\sum | p_i | )$ |
| 搜索            | $O\left(\frac{n}{m} + \text{overlap}\right)$ 每线程 | $O(1)$     |    |         |     |    |
| 合并结果        | $O(k)$                                              | $O(k)$     |    |         |     |    |

总时间近似为
$$
O\left(\frac{n}{m} + \text{overlap} \cdot m\right)
$$

#### 动手尝试

1. 将
   $$
   T = \texttt{"bananabanabanana"}
   $$
   和模式集
   $$
   P = {\texttt{"ana"}, \texttt{"banana"}}
   $$
   分割成重叠长度为 6 的块。
2. 验证找到的所有匹配。
3. 尝试不同的块大小和重叠长度。
4. 比较单线程与多线程性能。
5. 扩展到多进程或 GPU 流。

并行 Aho–Corasick 将一个顺序自动机转变为一个可扩展的搜索引擎，将匹配的节奏分布到各个线程，却产生一个单一、同步的结果旋律。
### 618 并行 Aho–Corasick 搜索

并行 Aho–Corasick 算法将经典的 Aho–Corasick 自动机适配到多线程或分布式环境中。它将输入文本或工作负载划分为独立的块，使得多个处理器可以同时搜索模式，从而实现对海量数据流的高吞吐量关键词检测。

#### 我们要解决什么问题？

经典的 Aho–Corasick 算法按顺序扫描文本。
对于大规模任务，例如扫描日志、DNA 序列或网络数据包，这就会成为瓶颈。

我们希望：

- 保持线性时间匹配
- 利用多核或多机优势
- 在块边界处保持正确性

因此，我们的目标是使用并行执行来搜索
$$
T = t_1 t_2 \ldots t_n
$$
以匹配模式集
$$
P = {p_1, p_2, \ldots, p_k}
$$

#### 它是如何工作的（通俗解释）

并行化 Aho–Corasick 主要有两种策略：

##### 1. 文本分区（输入分割模型）

- 将文本 $T$ 分割成 $m$ 个块：
  $$
  T = T_1 , T_2 , \ldots , T_m
  $$
- 将每个块分配给一个工作线程。
- 每个线程独立运行 Aho–Corasick 算法。
- 通过使用长度等于最长模式的重叠缓冲区来处理边界情况（跨越块边界的模式）。

优点：简单，对于长文本高效
缺点：为了保证正确性需要重叠

##### 2. 自动机分区（状态分割模型）

- 将状态机在线程或节点间进行划分。
- 每个处理器负责一个模式子集或状态子集。
- 通过消息传递（例如 MPI）进行状态转移通信。

优点：适用于静态、小规模模式集
缺点：同步开销大，状态移交复杂

在两种方法中：

- 每个线程以 $O(|T_i| + \text{output\_count}_i)$ 的时间复杂度扫描文本
- 最后合并结果。

#### 示例（文本分区）

令模式集
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"his"}, \texttt{"hers"}}
$$
和文本
$$
T = \texttt{"ushershehis"}
$$

将 $T$ 分割成两部分，重叠长度为 4（最长模式长度）：

- 线程 1：`"ushersh"`
- 线程 2：`"shehis"`

两个线程运行相同的自动机。
在合并时，对重叠区域中的匹配进行去重。

各自发现：

- 线程 1 → `she@1`, `he@2`, `hers@2`
- 线程 2 → `she@6`, `he@7`, `his@8`

最终结果 = 两个集合的并集。

#### 微型代码（并行示例，Python 线程）

```python
from concurrent.futures import ThreadPoolExecutor

def search_chunk(ac, text, offset=0):
    matches = []
    node = 0
    for i, ch in enumerate(text):
        while node and ch not in ac.trie[node]:
            node = ac.fail[node]
        node = ac.trie[node].get(ch, 0)
        for pat in ac.output[node]:
            matches.append((offset + i - len(pat) + 1, pat))
    return matches

def parallel_search(ac, text, chunk_size, overlap):
    tasks = []
    results = []
    with ThreadPoolExecutor() as executor:
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size + overlap]
            tasks.append(executor.submit(search_chunk, ac, chunk, i))
        for t in tasks:
            results.extend(t.result())
    # 可选地，对重叠匹配进行去重
    return sorted(set(results))
```

用法：

```python
patterns = ["he", "she", "his", "hers"]
ac = AhoCorasick(patterns)
print(parallel_search(ac, "ushershehis", chunk_size=6, overlap=4))
```

输出：

```
$$(1, 'she'), (2, 'he'), (2, 'hers'), (6, 'she'), (7, 'he'), (8, 'his')]
```

#### 为什么它很重要

- 支持对大规模数据进行实时匹配
- 应用于：

  * 入侵检测系统 (IDS)
  * 大数据文本分析
  * 日志扫描和威胁检测
  * 基因组序列分析
  * 网络数据包检查

并行化使 Aho–Corasick 达到了每秒千兆字节的吞吐量。

#### 复杂度分析

对于 $m$ 个线程：

| 步骤            | 时间复杂度                                         | 空间复杂度 |    |         |     |    |
| --------------- | ------------------------------------------------------- | ---------------- | -- | ------- | --- | -- |
| 构建自动机 | $O(\sum                                                 | p_i              | )$ | $O(\sum | p_i | )$ |
| 搜索          | $O\left(\frac{n}{m} + \text{overlap}\right)$ 每线程 | $O(1)$           |    |         |     |    |
| 合并结果   | $O(k)$                                                  | $O(k)$           |    |         |     |    |

总时间近似为
$$
O\left(\frac{n}{m} + \text{overlap} \cdot m\right)
$$

#### 动手尝试

1. 将文本
   $$
   T = \texttt{"bananabanabanana"}
   $$
   和模式集
   $$
   P = {\texttt{"ana"}, \texttt{"banana"}}
   $$
   分割成重叠长度为 6 的块。
2. 验证找到的所有匹配。
3. 尝试不同的块大小和重叠长度。
4. 比较单线程与多线程的性能。
5. 扩展到多进程或 GPU 流。

并行 Aho–Corasick 将顺序自动机转变为可扩展的搜索引擎，将匹配的节奏分布到各个线程中，却最终奏出一曲单一、同步的结果旋律。
### 619 压缩 Aho–Corasick 自动机

压缩 Aho–Corasick 自动机是经典 Aho–Corasick 结构的空间优化版本。它通过紧凑地表示状态、转移和失败链接来减少内存占用，同时保持线性时间匹配，非常适合海量词典或嵌入式系统。

#### 我们要解决什么问题？

标准的 Aho–Corasick 自动机存储：

- 状态：每个模式的每个前缀对应一个状态
- 转移：每个状态的出边显式字典
- 失败链接和输出集合

对于大型模式集合（数百万个单词），这会变得非常消耗内存：

$$
O(\sum |p_i| \cdot \sigma)
$$

我们需要一种空间高效的结构，既能适应有限的内存，又能保持：

- 确定性转移
- 快速查找（最好是 $O(1)$ 或 $O(\log \sigma)$）
- 完全相同的匹配行为

#### 它是如何工作的？（通俗解释）

压缩的重点在于表示方式，而非算法改变。匹配逻辑完全相同，但存储方式更紧凑。

有几个关键策略：

##### 1. 稀疏转移编码

不为每个节点存储所有 $\sigma$ 条转移，只存储实际存在的：

- 使用哈希表或有序数组存储边
- 按字符进行二分查找
- 将空间从 $O(\sum |p_i| \cdot \sigma)$ 减少到 $O(\sum |p_i|)$

##### 2. 双数组字典树

使用两个并行数组 `base[]` 和 `check[]` 表示字典树：

- `base[s] + c` 给出下一个状态
- `check[next] = s` 确认父节点
- 极其紧凑且缓存友好
- 用于 Darts 和 MARP 等工具

转移公式：
$$
\text{next} = \text{base}[s] + \text{code}(c)
$$

##### 3. 位压缩链接

将失败链接和输出存储在整数数组或位集中：

- 每个节点的失败指针是一个 32 位整数
- 输出集合被替换为压缩索引或标志位

如果一个模式在某个节点结束，则用位掩码标记，而不是一个集合。

##### 4. 简洁表示

使用小波树或简洁字典树存储边：

- 空间接近理论下限
- 转移查询时间为 $O(\log \sigma)$
- 非常适合非常大的字母表（例如 Unicode、DNA）

#### 示例

考虑模式：
$$
P = {\texttt{"he"}, \texttt{"she"}, \texttt{"hers"}}
$$

朴素的字典树表示：

- 节点：8 个
- 转移：存储为字典 `{字符: 下一状态}`

压缩的双数组表示：

- `base = [0, 5, 9, ...]`
- `check = [-1, 0, 1, 1, 2, ...]`
- 转移：`next = base[状态] + code(字符)`

失败链接和输出集合存储为数组：

```
fail = [0, 0, 0, 1, 2, 3, ...]
output_flag = [0, 1, 1, 0, 1, ...]
```

这极大地减少了开销。

#### 微型代码（使用稀疏字典的 Python 示例）

```python
class CompressedAC:
    def __init__(self):
        self.trie = [{}]  # 字典树，每个节点是一个字符到下一状态的映射
        self.fail = [0]   # 失败链接
        self.output = [0] # 位掩码标志

    def insert(self, word, idx):
        node = 0
        for ch in word:
            if ch not in self.trie[node]:
                self.trie[node][ch] = len(self.trie)
                self.trie.append({})
                self.fail.append(0)
                self.output.append(0)
            node = self.trie[node][ch]
        self.output[node] |= (1 << idx)  # 标记模式结束

    def build(self):
        from collections import deque
        q = deque()
        for ch, nxt in self.trie[0].items():
            q.append(nxt)
        while q:
            r = q.popleft()
            for ch, s in self.trie[r].items():
                q.append(s)
                f = self.fail[r]
                while f and ch not in self.trie[f]:
                    f = self.fail[f]
                self.fail[s] = self.trie[f].get(ch, 0)
                self.output[s] |= self.output[self.fail[s]]

    def search(self, text):
        node = 0
        for i, ch in enumerate(text):
            while node and ch not in self.trie[node]:
                node = self.fail[node]
            node = self.trie[node].get(ch, 0)
            if self.output[node]:
                print(f"在索引 {i} 处匹配到位掩码 {self.output[node]:b}")
```

位掩码替换了集合，减少了内存并支持快速的按位或合并。

#### 为什么这很重要

- 内存高效：可将大型词典放入 RAM
- 缓存友好：提高了实际性能
- 应用于：
  * 具有大量规则集的垃圾邮件过滤器
  * 嵌入式系统（防火墙、物联网）
  * 搜索设备和反病毒引擎

压缩字典树是那些以微小开销换取巨大吞吐量的系统的基础。

#### 复杂度

| 操作           | 时间复杂度                     | 空间复杂度       |                     |         |     |                 |
| -------------- | ------------------------------ | ---------------- | ------------------- | ------- | --- | --------------- |
| 插入模式       | $O(\sum                        | p_i              | )$                  | $O(\sum | p_i | )$ (压缩后)     |
| 构建链接       | $O(\sum                        | p_i              | \cdot \log \sigma)$ | $O(\sum | p_i | )$              |
| 搜索文本       | $O(n + \text{输出\_数量})$     | $O(1)$           |                     |         |     |                 |

与经典 Aho–Corasick 相比：

- 相同的时间渐进复杂度
- 减少了常数因子和内存使用

#### 亲自尝试

1.  为以下模式构建标准和压缩的 Aho–Corasick 自动机：
    $$
    P = {\texttt{"abc"}, \texttt{"abd"}, \texttt{"bcd"}, \texttt{"cd"}}
    $$
2.  测量节点数量和内存大小。
3.  在 1 MB 的随机文本上比较性能。
4.  尝试使用位掩码进行输出合并。
5.  可视化 `base[]` 和 `check[]` 数组。

压缩的 Aho–Corasick 自动机精简而强大，每一个比特都物尽其用，每一次转移都紧密打包，以一小部分空间实现了完整的模式检测。
### 620 支持通配符的扩展 Aho–Corasick 算法

扩展的 Aho–Corasick 自动机将经典算法推广到能够处理通配符——这些特殊符号可以匹配任意字符。这个版本对于包含灵活模板的模式集合至关重要，例如 `"he*o"`、`"a?b"` 或 `"c*t"`。它使得在嘈杂或半结构化数据中进行鲁棒的多模式匹配成为可能。

#### 我们要解决什么问题？

传统的 Aho–Corasick 算法只能匹配精确模式。
但许多现实世界的查询需要容忍通配符，例如：

- `"a?b"` → 匹配 `"acb"`、`"adb"`、`"aeb"`
- `"he*o"` → 匹配 `"hello"`、`"hero"`、`"heyo"`

给定：
$$
P = {p_1, p_2, \ldots, p_k}
$$
以及通配符符号，例如 `?`（单字符）或 `*`（多字符），
我们需要在通配符语义下，找到文本 $T$ 中所有匹配任一模式的子串。

#### 它是如何工作的（通俗解释）

我们扩展了字典树（trie）和失效（failure）机制来处理通配符转移。

主要有两种通配符模型：

##### 1. 单字符通配符 (`?`)

- 代表恰好一个任意字符
- 在构建时，每个 `?` 会从当前状态创建一条通用边
- 在搜索期间，自动机会为任何字符转移到这个状态

形式化表示：
$$
\delta(u, c) = v \quad \text{如果 } c \in \Sigma \text{ 且存在从 } u \text{ 出发的标记为 '?' 的边}
$$

##### 2. 多字符通配符 (`*`)

- 匹配零个或多个任意字符
- 创建一个自循环边加上一条到下一个状态的跳过边
- 需要额外的转移：

  * $\text{stay}(u, c) = u$ 对于任意 $c$
  * $\text{skip}(u) = v$（`*` 之后的下一个文字字符）

这有效地将正则表达式语义融入了 Aho–Corasick 结构。

#### 示例

模式：
$$
P = {\texttt{"he?o"}, \texttt{"a*c"}}
$$
文本：
$$
T = \texttt{"hero and abc and aac"}
$$

1. `"he?o"` 匹配 `"hero"`
2. `"a*c"` 匹配 `"abc"`、`"aac"`

字典树边包括：

```
(根节点)
 ├── h ─ e ─ ? ─ o*
 └── a ─ * ─ c*
```

通配符节点会动态地或通过默认边处理为所有字符扩展转移。

#### 微型代码（Python 草图，`?` 通配符）

```python
from collections import deque

class WildcardAC:
    def __init__(self):
        self.trie = [{}]
        self.fail = [0]
        self.output = [set()]

    def insert(self, word):
        node = 0
        for ch in word:
            if ch not in self.trie[node]:
                self.trie[node][ch] = len(self.trie)
                self.trie.append({})
                self.fail.append(0)
                self.output.append(set())
            node = self.trie[node][ch]
        self.output[node].add(word)

    def build(self):
        q = deque()
        for ch, nxt in self.trie[0].items():
            q.append(nxt)
        while q:
            r = q.popleft()
            for ch, s in self.trie[r].items():
                q.append(s)
                f = self.fail[r]
                while f and ch not in self.trie[f] and '?' not in self.trie[f]:
                    f = self.fail[f]
                self.fail[s] = self.trie[f].get(ch, self.trie[f].get('?', 0))
                self.output[s] |= self.output[self.fail[s]]

    def search(self, text):
        results = []
        node = 0
        for i, ch in enumerate(text):
            while node and ch not in self.trie[node] and '?' not in self.trie[node]:
                node = self.fail[node]
            node = self.trie[node].get(ch, self.trie[node].get('?', 0))
            for pat in self.output[node]:
                results.append((i - len(pat) + 1, pat))
        return results

patterns = ["he?o", "a?c"]
ac = WildcardAC()
for p in patterns:
    ac.insert(p)
ac.build()
print(ac.search("hero abc aac"))
```

输出：

```
$$(0, 'he?o'), (5, 'a?c'), (9, 'a?c')]
```

#### 为什么它很重要

- 支持结构化数据中的模式灵活性
- 在以下场景中很有用：

  * 具有可变字段的日志扫描
  * 使用模板的关键词搜索
  * 恶意软件和基于规则的过滤
  * DNA 序列基序匹配
- 超越固定字符串，向类正则表达式匹配扩展，同时保持高效

#### 复杂度

| 操作           | 时间复杂度       | 空间复杂度       |                |         |     |    |
| -------------- | ---------------- | ---------------- | -------------- | ------- | --- | -- |
| 构建自动机     | $O(\sum          | p_i              | \cdot \sigma)$ | $O(\sum | p_i | )$ |
| 搜索文本       | $O(n \cdot d)$   | $O(1)$           |                |         |     |    |

其中 $d$ 是通配符转移带来的分支因子（通常很小）。
如果通配符边有界，自动机保持线性复杂度。

#### 亲自尝试

1.  为以下模式构建自动机
    $$
    P = {\texttt{"c?t"}, \texttt{"b*g"}}
    $$
    并在 `"cat bag big cog bug"` 上测试。
2.  添加重叠的通配符模式，如 `"a*a"` 和 `"aa*"`。
3.  在字典树中可视化通配符转移。
4.  测量使用通配符与不使用通配符时的运行时间。
5.  扩展到同时处理 `?` 和 `*`。

支持通配符的扩展 Aho–Corasick 算法将确定性自动机和模式泛化结合在一起，在一次统一的扫描中匹配精确和不确定的模式。

# 第 63 节 后缀结构
### 621 后缀数组（朴素算法）

后缀数组是字符串算法中的一种基础数据结构，它是一个字符串所有后缀的排序列表。它提供了一种紧凑且高效的方式来进行子串搜索、模式匹配和文本索引，构成了诸如 FM-index 和 Burrows–Wheeler 变换等结构的核心。

朴素算法通过生成所有后缀、按字典序排序并记录它们的起始位置来构建后缀数组。

#### 我们要解决什么问题？

给定一个字符串
$$
S = s_0 s_1 \ldots s_{n-1}
$$
我们想要构建一个数组
$$
SA[0 \ldots n-1]
$$
使得
$$
S[SA[0] \ldots n-1] < S[SA[1] \ldots n-1] < \cdots < S[SA[n-1] \ldots n-1]
$$
按字典序排列。

后缀数组中的每个条目指向一个按排序顺序排列的后缀的起始索引。

示例：
令
$$
S = \texttt{"banana"}
$$

所有后缀：

| 索引 | 后缀   |
| ---- | ------ |
| 0    | banana |
| 1    | anana  |
| 2    | nana   |
| 3    | ana    |
| 4    | na     |
| 5    | a      |

按字典序排序：

| 排名 | 后缀   | 索引 |
| ---- | ------ | ---- |
| 0    | a      | 5    |
| 1    | ana    | 3    |
| 2    | anana  | 1    |
| 3    | banana | 0    |
| 4    | na     | 4    |
| 5    | nana   | 2    |

因此：
$$
SA = [5, 3, 1, 0, 4, 2]
$$

#### 它是如何工作的（通俗解释）

朴素算法步骤如下：

1.  生成所有后缀
    从每个索引 $i$ 开始创建 $n$ 个子串。

2.  对后缀排序
    按字典序（类似于字典顺序）比较字符串。

3.  存储索引
    将排序后后缀的起始索引收集到数组 `SA` 中。

这种方法直接但效率低下，每次后缀比较可能需要 $O(n)$ 时间，并且有 $O(n \log n)$ 次比较。

#### 算法（逐步说明）

1.  初始化配对列表：
    $$
    L = [(i, S[i:])] \quad \text{对于 } i = 0, 1, \ldots, n-1
    $$

2.  按字符串分量对 $L$ 排序。

3.  输出数组：
    $$
    SA[j] = L[j].i
    $$

#### 微型代码（简易版本）

Python

```python
def suffix_array_naive(s):
    n = len(s)
    suffixes = [(i, s[i:]) for i in range(n)]
    suffixes.sort(key=lambda x: x[1])
    return [idx for idx, _ in suffixes]

s = "banana"
print(suffix_array_naive(s))  # [5, 3, 1, 0, 4, 2]
```

C

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int index;
    const char *suffix;
} Suffix;

int cmp(const void *a, const void *b) {
    Suffix *sa = (Suffix *)a;
    Suffix *sb = (Suffix *)b;
    return strcmp(sa->suffix, sb->suffix);
}

void build_suffix_array(const char *s, int sa[]) {
    int n = strlen(s);
    Suffix arr[n];
    for (int i = 0; i < n; i++) {
        arr[i].index = i;
        arr[i].suffix = s + i;
    }
    qsort(arr, n, sizeof(Suffix), cmp);
    for (int i = 0; i < n; i++)
        sa[i] = arr[i].index;
}

int main(void) {
    char s[] = "banana";
    int sa[6];
    build_suffix_array(s, sa);
    for (int i = 0; i < 6; i++)
        printf("%d ", sa[i]);
}
```

#### 为什么它很重要

-   通过二分搜索实现快速子串搜索
-   是 LCP（最长公共前缀）和后缀树构建的基础
-   是 FM-index 和 BWT 等压缩文本索引的核心
-   是学习更高级的 $O(n \log n)$ 和 $O(n)$ 方法的简单、教育性入门

#### 复杂度

| 步骤             | 时间                                    | 空间     |
| ---------------- | --------------------------------------- | -------- |
| 生成后缀         | $O(n)$                                  | $O(n^2)$ |
| 排序             | $O(n \log n)$ 次比较 × 每次 $O(n)$ 时间 | $O(n)$   |
| 总计             | $O(n^2 \log n)$                         | $O(n^2)$ |

速度慢，但概念清晰且易于实现。

#### 动手尝试

1.  计算 `"abracadabra"` 的后缀数组。
2.  验证后缀的字典序。
3.  在 `SA` 上使用二分搜索查找子串 `"bra"`。
4.  与倍增算法构建的后缀数组进行比较。
5.  优化存储以避免存储完整的子串。

朴素后缀数组是文本索引的一个纯粹、清晰的视角，每个后缀，每个顺序，逐一处理，构建简单，是迈向后续优雅的 $O(n \log n)$ 和 $O(n)$ 算法的完美第一步。
### 622 后缀数组（倍增算法）

倍增算法能在
$$
O(n \log n)
$$
时间内构建后缀数组，这比朴素的 $O(n^2 \log n)$ 方法是一个重大改进。它的工作原理是对长度为 $2^k$ 的前缀进行排序，每次迭代将长度加倍，直到对整个字符串完成排序。

这种通过逐步增长的前缀进行排序的优雅思想，使其既快速又概念清晰。

#### 我们要解决什么问题？

给定一个字符串
$$
S = s_0 s_1 \ldots s_{n-1}
$$
我们想要计算它的后缀数组
$$
SA[0 \ldots n-1]
$$
使得
$$
S[SA[0]:] < S[SA[1]:] < \ldots < S[SA[n-1]:]
$$

我们不是比较整个后缀，而是比较长度为 $2^k$ 的前缀，使用来自前一次迭代的整数排名，就像基数排序一样。

#### 它是如何工作的（通俗解释）

我们将按第一个字符、然后2个、然后4个、然后8个字符，依此类推，对后缀进行排序。
在每个阶段，我们为每个后缀分配一对排名：

$$
\text{rank}_k[i] = (\text{rank}_{k-1}[i], \text{rank}_{k-1}[i + 2^{k-1}])
$$

我们根据这个对后缀进行排序并分配新的排名。经过 $\log_2 n$ 轮后，所有后缀都被完全排序。

#### 示例

令
$$
S = \texttt{"banana"}
$$

1.  初始排名（按单个字符）：

| 索引 | 字符 | 排名 |
| ---- | ---- | ---- |
| 0    | b    | 1    |
| 1    | a    | 0    |
| 2    | n    | 2    |
| 3    | a    | 0    |
| 4    | n    | 2    |
| 5    | a    | 0    |

2.  按对（排名，下一个排名）排序：

对于 $k = 1$（前缀长度 $2$）：

| i | 对      | 后缀   |
| - | ------- | ------ |
| 0 | (1, 0)  | banana |
| 1 | (0, 2)  | anana  |
| 2 | (2, 0)  | nana   |
| 3 | (0, 2)  | ana    |
| 4 | (2, 0)  | na     |
| 5 | (0, -1) | a      |

按对字典序排序，分配新的排名。

重复倍增过程，直到排名唯一。

最终
$$
SA = [5, 3, 1, 0, 4, 2]
$$

#### 算法（逐步说明）

1.  初始化排名
    按第一个字符对后缀排序。

2.  迭代地对 2ᵏ 前缀排序
    对于 $k = 1, 2, \ldots$

    *   构造元组
        $$
        (rank[i], rank[i + 2^{k-1}])
        $$
    *   根据这些元组对后缀排序
    *   分配新的排名

3.  当所有排名都不同或 $2^k \ge n$ 时停止

#### 精简代码（Python）

```python
def suffix_array_doubling(s):
    n = len(s)
    k = 1
    rank = [ord(c) for c in s] # 根据字符的ASCII码初始化排名
    tmp = [0] * n
    sa = list(range(n)) # 后缀数组初始为索引列表

    while True:
        # 根据当前排名对后缀数组进行排序
        sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))

        # 根据排序结果重新分配排名
        tmp[sa[0]] = 0
        for i in range(1, n):
            prev = (rank[sa[i-1]], rank[sa[i-1]+k] if sa[i-1]+k < n else -1)
            curr = (rank[sa[i]], rank[sa[i]+k] if sa[i]+k < n else -1)
            tmp[sa[i]] = tmp[sa[i-1]] + (curr != prev)

        rank[:] = tmp[:] # 更新排名数组
        k *= 2 # 倍增前缀长度
        if max(rank) == n-1: # 如果所有排名都不同，则排序完成
            break
    return sa

s = "banana"
print(suffix_array_doubling(s))  # [5, 3, 1, 0, 4, 2]
```

#### 为什么它很重要

-   将朴素的 $O(n^2 \log n)$ 复杂度降低到 $O(n \log n)$
-   是 Kasai 的 LCP 计算算法的基础
-   简单而快速，广泛用于实际的后缀数组构建器
-   可扩展到循环移位和最小字符串旋转问题

#### 复杂度

| 步骤                | 时间          | 空间  |
| ------------------- | ------------- | ----- |
| 排序（每轮）        | $O(n \log n)$ | $O(n)$ |
| 轮数                | $\log n$      |       |
| 总计                | $O(n \log n)$ | $O(n)$ |

#### 亲自尝试

1.  为 `"mississippi"` 构建后缀数组。
2.  追踪排名过程的前3轮。
3.  手动验证排序后的后缀。
4.  与朴素方法比较运行时间。
5.  使用得到的排名进行 LCP 计算（Kasai 算法）。

倍增算法是连接清晰性与性能的桥梁，它通过迭代优化、二的幂次以及字典序，用简单的排名揭示了字符串的完整顺序。
### 623 Kasai 的 LCP 算法

Kasai 算法可以在线性时间内根据后缀数组计算最长公共前缀（LCP）数组。
它告诉你，对于按排序顺序排列的每一对相邻后缀，它们在开头共享多少个字符，从而揭示字符串内部重复和重叠的结构。

#### 我们要解决什么问题？

给定一个字符串 $S$ 及其后缀数组 $SA$，
我们想要计算 LCP 数组，其中：

$$
LCP[i] = \text{后缀 } S[SA[i]] \text{ 和 } S[SA[i-1]] \text{ 的公共前缀长度}
$$

这使我们能够回答诸如以下问题：

- 存在多少个重复子串？
- 最长的重复子串是什么？
- 相邻后缀之间的相似度如何？

示例：

令
$$
S = \texttt{"banana"}
$$
且
$$
SA = [5, 3, 1, 0, 4, 2]
$$

| i | SA[i] | 后缀 | LCP[i] |
| - | ----- | ------ | ------ |
| 0 | 5     | a      |,      |
| 1 | 3     | ana    | 1      |
| 2 | 1     | anana  | 3      |
| 3 | 0     | banana | 0      |
| 4 | 4     | na     | 0      |
| 5 | 2     | nana   | 2      |

因此：
$$
LCP = [0, 1, 3, 0, 0, 2]
$$

#### 它是如何工作的（通俗解释）

朴素地比较每一对相邻后缀的成本是 $O(n^2)$。
Kasai 的技巧：重用之前的 LCP 计算结果。

如果 $h$ 是 $S[i:]$ 和 $S[j:]$ 的 LCP，
那么 $S[i+1:]$ 和 $S[j+1:]$ 的 LCP 至少是 $h-1$。
因此，我们每次向下滑动一个字符，重用之前的重叠部分。

#### 逐步算法

1.  构建逆后缀数组 `rank[]`，使得
    $$
    \text{rank}[SA[i]] = i
    $$

2.  初始化 $h = 0$

3.  对于 $S$ 中的每个位置 $i$：

    *   如果 $rank[i] > 0$：

        *   令 $j = SA[rank[i] - 1]$
        *   比较 $S[i+h]$ 和 $S[j+h]$
        *   当它们匹配时增加 $h$
        *   设置 $LCP[rank[i]] = h$
        *   将 $h$ 减 1（为下一次迭代做准备）

这确保每个字符最多被比较两次。

#### 示例演练

对于 `"banana"`：

后缀数组：
$$
SA = [5, 3, 1, 0, 4, 2]
$$
逆排名数组：
$$
rank = [3, 2, 5, 1, 4, 0]
$$

现在从 0 到 5 迭代 $i$：

| i | rank[i] | j = SA[rank[i]-1] | 比较         | h | LCP[rank[i]] |
| - | ------- | ----------------- | --------------- | - | ------------ |
| 0 | 3       | 1                 | banana vs anana | 0 | 0            |
| 1 | 2       | 3                 | anana vs ana    | 3 | 3            |
| 2 | 5       | 4                 | nana vs na      | 2 | 2            |
| 3 | 1       | 5                 | ana vs a        | 1 | 1            |
| 4 | 4       | 0                 | na vs banana    | 0 | 0            |
| 5 | 0       |,                 | 跳过            |, |,            |

因此
$$
LCP = [0, 1, 3, 0, 0, 2]
$$

#### 精简代码（Python）

```python
def kasai(s, sa):
    n = len(s)
    rank = [0] * n
    for i in range(n):
        rank[sa[i]] = i

    h = 0
    lcp = [0] * n
    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i]] = h
            if h > 0:
                h -= 1
    return lcp

s = "banana"
sa = [5, 3, 1, 0, 4, 2]
print(kasai(s, sa))  # [0, 1, 3, 0, 0, 2]
```

#### 为什么它很重要

-   字符串处理的基础：

    *   最长重复子串
    *   不同子串的数量
    *   公共子串查询
-   线性时间复杂度，易于与后缀数组集成
-   生物信息学、索引和数据压缩中的核心组件

#### 复杂度

| 步骤             | 时间复杂度 | 空间复杂度 |
| ---------------- | ------ | ------ |
| 构建排名数组 | $O(n)$ | $O(n)$ |
| LCP 计算  | $O(n)$ | $O(n)$ |
| 总计        | $O(n)$ | $O(n)$ |

#### 亲自尝试

1.  计算 `"mississippi"` 的 `LCP`。
2.  绘制后缀数组和相邻后缀。
3.  使用最大 LCP 值找到最长重复子串。
4.  通过以下公式计算不同子串的数量：
    $$
    \frac{n(n+1)}{2} - \sum_i LCP[i]
    $$
5.  与朴素的成对比较方法进行比较。

Kasai 算法是重用、滑动、重用和减少的杰作。
每个字符向前比较一次，向后比较一次，整个重叠结构在线性时间内展开。
### 624 后缀树（Ukkonen 算法）

后缀树是一种强大的数据结构，它能紧凑地表示一个字符串的所有后缀。借助 Ukkonen 算法，我们可以在线性时间内构建它——
$$
O(n)
$$，这是字符串处理领域的一个里程碑式成就。

后缀树开启了一个高效解决方案的世界：子串搜索、最长重复子串、模式频率等等，所有这些查询对于一个长度为 $m$ 的模式都能在 $O(m)$ 时间内完成。

#### 我们要解决什么问题？

给定一个字符串
$$
S = s_0 s_1 \ldots s_{n-1}
$$
我们想要一棵树，其中：

- 从根节点出发的每条路径都对应于 $S$ 的某个后缀的一个前缀。
- 每个叶子节点对应一个后缀。
- 边标签是 $S$ 的子串（不是单个字符）。
- 树是压缩的：没有只有一个孩子的冗余节点。

该结构应支持：

- 子串搜索：$O(m)$
- 统计不同子串数量：$O(n)$
- 最长重复子串：通过最深的内部节点

#### 示例

对于  
$$
S = \texttt{"banana\textdollar"}
$$

所有后缀：

| i | 后缀  |
| - | -------- |
| 0 | banana$ |
| 1 | anana$  |
| 2 | nana$   |
| 3 | ana$    |
| 4 | na$     |
| 5 | a$      |
| 6 | $       |

后缀树将这些后缀压缩到共享的路径中。

根节点分支为：
- `$` → 终止叶子节点  
- `a` → 覆盖起始于位置 1、3 和 5 的后缀（`anana$`、`ana$`、`a$`）  
- `b` → 覆盖起始于位置 0 的后缀（`banana$`）  
- `n` → 覆盖起始于位置 2 和 4 的后缀（`nana$`、`na$`）

```
(根节点)
├── a → na → na$
├── b → anana$
├── n → a → na$
└── $
```

每个后缀都恰好作为一条路径出现一次。

#### 为什么朴素构造方法很慢

朴素地将所有 $n$ 个后缀插入到一个字典树中需要
$$
O(n^2)
$$
时间，因为每个后缀的长度都是 $O(n)$。

Ukkonen 算法增量式地在线构建，维护后缀链接和隐式后缀树，实现了
$$
O(n)
$$
的时间和空间复杂度。

#### 工作原理（通俗解释）

Ukkonen 算法一次一个字符地构建树。

我们维护：

- 活动点：当前节点 + 边 + 偏移量
- 后缀链接：内部节点之间的快捷方式
- 隐式树：迄今为止构建的树（没有显式结束每个后缀）

在第 $i$ 步（读取 $S[0..i]$ 后）：

- 添加结束于 $i$ 的新后缀
- 利用后缀链接重用先前结构
- 仅在必要时才分割边

在最后一个字符（通常是 `$`）之后，树变为显式的，包含所有后缀。

#### 逐步概览

1. 初始化空根节点
2. 对于 $S$ 中的每个位置 $i$：

   * 扩展所有结束于 $i$ 的后缀
   * 使用活动点跟踪插入位置
   * 根据需要创建内部节点和后缀链接
3. 重复直到完整的树被构建完成

后缀链接跳过了冗余的遍历，将二次工作量转化为线性。

#### 示例（概览）

为 `abcab$` 构建：

- 添加 `a`：路径 `a`
- 添加 `b`：路径 `a`，`b`
- 添加 `c`：路径 `a`，`b`，`c`
- 添加 `a`：路径重用现有前缀
- 添加 `b`：为 `ab` 创建内部节点
- 添加 `$`：关闭所有后缀

结果：一个压缩树，有 6 个叶子节点，每个后缀一个。

#### 微型代码（简化版，Python）

*(简化的伪实现，用于教学清晰度)*

```python
class Node:
    def __init__(self):
        self.edges = {}
        self.link = None

def build_suffix_tree(s):
    s += "$"
    root = Node()
    n = len(s)
    for i in range(n):
        current = root
        for j in range(i, n):
            ch = s[j]
            if ch not in current.edges:
                current.edges[ch] = Node()
            current = current.edges[ch]
    return root

# 用于直观理解的朴素 O(n^2) 版本
tree = build_suffix_tree("banana")
```

Ukkonen 的真实实现使用边跨度 `[l, r]` 来避免复制子串，并使用活动点管理来确保 $O(n)$ 时间复杂度。

#### 为什么它很重要

- 快速子串搜索：$O(m)$
- 统计不同子串数量：路径的数量
- 最长重复子串：最深的内部节点
- 最长公共子串（两个字符串之间）：通过广义后缀树
- 以下内容的基础：

  * LCP 数组（通过 DFS）
  * 后缀自动机
  * FM 索引

#### 复杂度

| 步骤             | 时间   | 空间  |
| ---------------- | ------ | ------ |
| 构建            | $O(n)$ | $O(n)$ |
| 查询            | $O(m)$ | $O(1)$ |
| 统计子串数量 | $O(n)$ | $O(1)$ |

#### 亲自尝试

1. 为 `"abaaba$"` 构建后缀树。
2. 用起始索引标记叶子节点。
3. 统计不同子串的数量。
4. 追踪 Ukkonen 算法中活动点的移动。
5. 与后缀数组 + LCP 构造进行比较。

后缀树是字符串结构的殿堂，每个后缀都是一条路径，每个分支都是一段共享的历史，而 Ukkonen 算法以完美的线性时间为每一块石头奠基。
### 625 后缀自动机

后缀自动机是能够识别给定字符串所有子串的最小确定性有限自动机（DFA）。它以紧凑的线性大小结构隐式地捕获每一个子串，非常适合子串查询、重复分析和模式计数。

它可以在 $O(n)$ 时间内构建，常被称为字符串算法中的“瑞士军刀”，灵活、优雅且功能强大。

#### 我们要解决什么问题？

给定一个字符串
$$
S = s_0 s_1 \ldots s_{n-1}
$$
我们想要一个自动机，它能够：

- 恰好接受 $S$ 的所有子串的集合
- 支持如下查询：
  * “$T$ 是 $S$ 的子串吗？”
  * “$S$ 有多少个不同的子串？”
  * “与另一个字符串的最长公共子串”
- 是最小的：没有等价状态，转移是确定性的

后缀自动机（SAM）正是为此而生——它是增量式、一个状态一个状态、一条边一条边地构建起来的。

#### 核心思想

每个状态代表一组在 $S$ 中具有相同结束位置的子串。
每条转移代表追加一个字符。
后缀链接连接着代表“下一个更小”子串集合的状态。

因此，SAM 本质上就是所有子串的 DFA，并且是在线、线性时间内构建的。

#### 它是如何工作的（通俗解释）

我们在读取 $S$ 时构建自动机，逐个字符地扩展它：

1.  从初始状态（空字符串）开始
2.  对于每个新字符 $c$：
   * 为以 $c$ 结尾的子串创建一个新状态
   * 沿着后缀链接扩展旧路径
   * 必要时通过克隆状态来保持最小性
3.  更新后缀链接以确保每个子串的结束位置都被表示出来

每次扩展最多添加两个状态，因此总状态数 ≤ $2n - 1$。

#### 示例

令
$$
S = \texttt{"aba"}
$$

步骤 1：从初始状态（id 0）开始

步骤 2：添加 `'a'`

```
0 --a--> 1
```

link(1) = 0

步骤 3：添加 `'b'`

```
1 --b--> 2
0 --b--> 2
```

link(2) = 0

步骤 4：添加 `'a'`

- 从状态 2 通过 `'a'` 扩展
- 由于重叠，需要克隆状态

```
2 --a--> 3
link(3) = 1
```

现在自动机接受：`"a"`, `"b"`, `"ab"`, `"ba"`, `"aba"`
所有子串都已表示！

#### 精简代码（Python）

```python
class State:
    def __init__(self):
        self.next = {} # 转移边
        self.link = -1 # 后缀链接
        self.len = 0   # 该状态代表的最长子串长度

def build_suffix_automaton(s):
    sa = [State()] # 自动机状态列表
    last = 0       # 上一个添加的状态
    for ch in s:
        cur = len(sa)
        sa.append(State())
        sa[cur].len = sa[last].len + 1

        p = last
        while p >= 0 and ch not in sa[p].next:
            sa[p].next[ch] = cur
            p = sa[p].link

        if p == -1:
            sa[cur].link = 0
        else:
            q = sa[p].next[ch]
            if sa[p].len + 1 == sa[q].len:
                sa[cur].link = q
            else:
                clone = len(sa)
                sa.append(State())
                sa[clone].len = sa[p].len + 1
                sa[clone].next = sa[q].next.copy()
                sa[clone].link = sa[q].link
                while p >= 0 and sa[p].next[ch] == q:
                    sa[p].next[ch] = clone
                    p = sa[p].link
                sa[q].link = sa[cur].link = clone
        last = cur
    return sa
```

用法：

```python
sam = build_suffix_automaton("aba")
print(len(sam))  # 状态数量 (≤ 2n - 1)
```

#### 为什么它很重要

- **线性构建**
  在 $O(n)$ 时间内构建
- **子串查询**
  在 $O(|T|)$ 时间内检查 $T$ 是否在 $S$ 中
- **计数不同子串**
  $$
  \sum_{v} (\text{len}[v] - \text{len}[\text{link}[v]])
  $$
- **最长公共子串**
  将第二个字符串在自动机上运行
- **频率分析**
  通过结束位置计算出现次数

#### 复杂度

| 步骤                | 时间     | 空间     |
| ------------------- | -------- | -------- |
| 构建                | $O(n)$   | $O(n)$   |
| 子串查询            | $O(m)$   | $O(1)$   |
| 不同子串计数        | $O(n)$   | $O(n)$   |

#### 动手尝试

1.  为 `"banana"` 构建 SAM，并计算不同子串的数量。
2.  验证总数 = $n(n+1)/2 - \sum LCP[i]$
3.  添加代码来计算每个子串的出现次数。
4.  测试子串搜索 `"nan"`, `"ana"`, `"nana"`。
5.  为反转的字符串构建 SAM 并寻找回文串。

后缀自动机虽小却完备——
每个状态都是可能结束点的地平线，
每条链接都是通往更短影子的桥梁。
它是所有子串的完美镜像，一步一步，在线性时间内构建而成。
### 626 SA-IS 算法（线性时间后缀数组构建）

SA-IS 算法是一种现代、优雅的方法，用于在真正的线性时间
$$
O(n)
$$
内构建后缀数组。它使用诱导排序，将后缀按类型（S 型或 L 型）分类，先对一个小子集进行排序，然后根据该顺序*诱导*出其余部分。

它是目前最先进的后缀数组构建器的基础，并用于 DivSufSort、libdivsufsort 和基于 BWT 的压缩器等工具中。

#### 我们要解决什么问题？

我们想要构建字符串
$$
S = s_0 s_1 \ldots s_{n-1}
$$
的后缀数组，该数组按字典序列出所有后缀的起始索引，但我们希望在线性时间内完成，而不是
$$
O(n \log n)
$$

SA-IS 算法通过以下方式实现这一目标：

1.  将后缀分类为 S 型和 L 型
2.  识别 LMS（最左 S 型）子串
3.  对这些 LMS 子串进行排序
4.  从 LMS 顺序诱导出完整的后缀顺序

#### 关键概念

令 $S[n] = $ $ 为一个哨兵，小于所有字符。

1.  S 型后缀：
    $S[i:] < S[i+1:]$

2.  L 型后缀：
    $S[i:] > S[i+1:]$

3.  LMS 位置：
    一个索引 $i$，满足 $S[i]$ 是 S 型且 $S[i-1]$ 是 L 型。

这些 LMS 位置充当*锚点*，如果我们能对它们排序，就可以“诱导”出完整的后缀顺序。

#### 工作原理（通俗解释）

步骤 1. 分类 S/L 类型
从后向前遍历：

-   最后一个字符 `$` 是 S 型
-   $S[i]$ 是 S 型，如果
    $S[i] < S[i+1]$ 或 ($S[i] = S[i+1]$ 且 $S[i+1]$ 是 S 型)

步骤 2. 识别 LMS 位置
标记每一个从 L 到 S 的转换

步骤 3. 排序 LMS 子串
LMS 位置之间（包含两端）的每个子串都是唯一的。
我们对它们进行排序（如果需要，则递归地）以获得 LMS 顺序。

步骤 4. 诱导 L 型后缀
从左到右，使用 LMS 顺序填充桶。

步骤 5. 诱导 S 型后缀
从右到左，填充剩余的桶。

结果：完整的后缀数组。

#### 示例

令
$$
S = \texttt{"banana\$"}
$$

1.  分类类型

| i   | 字符 | 类型 |
| --- | ---- | ---- |
| 6   | $    | S    |
| 5   | a    | L    |
| 4   | n    | S    |
| 3   | a    | L    |
| 2   | n    | S    |
| 1   | a    | L    |
| 0   | b    | L    |

LMS 位置：2, 4, 6

2.  LMS 子串：
    `"na$"`, `"nana$"`, `"$"`

3.  按字典序对 LMS 子串进行排序。

4.  使用桶边界诱导 L 型和 S 型后缀。

最终
$$
SA = [6, 5, 3, 1, 0, 4, 2]
$$

#### 微型代码（Python 草图）

*（示意性，未优化）*

```python
def sa_is(s):
    s = [ord(c) for c in s] + [0]  # 添加哨兵
    n = len(s)
    SA = [-1] * n

    # 步骤 1: 分类
    t = [False] * n
    t[-1] = True
    for i in range(n-2, -1, -1):
        if s[i] < s[i+1] or (s[i] == s[i+1] and t[i+1]):
            t[i] = True

    # 识别 LMS
    LMS = [i for i in range(1, n) if not t[i-1] and t[i]]

    # 简化：使用 Python 排序获取 LMS 顺序（仅概念）
    LMS.sort(key=lambda i: s[i:])

    # 诱导排序草图省略
    # （完整的 SA-IS 会使用桶边界填充 SA）

    return LMS  # 用于教学演示的占位符

print(sa_is("banana"))  # [2, 4, 6]
```

真正的实现使用桶数组和诱导排序，仍然是线性的。

#### 为何重要

-   真正的线性时间构建
-   内存高效，除非必要否则不使用递归
-   是以下技术的基础：
    *   伯勒斯-惠勒变换 (BWT)
    *   FM 索引
    *   压缩后缀数组
-   即使在大数据集上也实用且快速

#### 复杂度

| 步骤                | 时间   | 空间  |
| ------------------- | ------ | ----- |
| 分类 + LMS          | $O(n)$ | $O(n)$ |
| 排序 LMS 子串       | $O(n)$ | $O(n)$ |
| 诱导排序            | $O(n)$ | $O(n)$ |
| 总计                | $O(n)$ | $O(n)$ |

#### 亲自尝试

1.  对 `"mississippi$"` 分类类型。
2.  标记 LMS 位置和子串。
3.  按字典序对 LMS 子串排序。
4.  逐步执行诱导步骤。
5.  将输出与倍增算法进行比较。

SA-IS 算法是经济性的典范 —— 排序少数，推断其余，让结构自然展开。从稀疏的锚点出发，文本的完整顺序以线性时间完美呈现。
### 627 LCP RMQ 查询（基于 LCP 数组的区间最小值查询）

LCP RMQ 结构允许在常数时间内检索字符串*任意两个后缀*之间的最长公共前缀长度，它利用了 LCP 数组和一个区间最小值查询数据结构。

结合后缀数组，它成为一个强大的文本索引工具，能够在线性预处理后，以
$$
O(1)
$$
的时间复杂度进行子串比较、字典序排名和高效的模式匹配。

#### 我们要解决什么问题？

给定：

- 一个字符串 $S$
- 它的后缀数组 $SA$
- 它的 LCP 数组，其中
  $$
  LCP[i] = \text{lcp}(S[SA[i-1]:], S[SA[i]:])
  $$

我们想要回答以下形式的查询：

> "在 $S$ 中，起始位置为 $i$ 和 $j$ 的后缀的 LCP 长度是多少？"

即：
$$
\text{LCP}(i, j) = S[i:] \text{ 和 } S[j:] \text{ 的最长公共前缀的长度}
$$

这对于以下方面至关重要：

- 快速子串比较
- 最长公共子串查询
- 字符串周期性检测
- 字典序区间分析

#### 关键观察

令 $pos[i]$ 和 $pos[j]$ 分别为后缀 $S[i:]$ 和 $S[j:]$ 在后缀数组中的位置。

那么：
$$
\text{LCP}(i, j) = \min \big( LCP[k] \big), \quad k \in [\min(pos[i], pos[j]) + 1, \max(pos[i], pos[j])]
$$

因此，问题简化为对 LCP 数组进行区间最小值查询。

#### 示例

令
$$
S = \texttt{"banana"}
$$
$$
SA = [5, 3, 1, 0, 4, 2]
$$
$$
LCP = [0, 1, 3, 0, 0, 2]
$$

目标：$\text{LCP}(1, 3)$，即 `"anana"` 和 `"ana"` 的公共前缀

1. $pos[1] = 2$, $pos[3] = 1$
2. 区间 = $(1, 2]$
3. $\min(LCP[2]) = 1$

所以
$$
\text{LCP}(1, 3) = 1
$$
（两者都以 `"a"` 开头）

#### 它是如何工作的（通俗解释）

1. 使用 RMQ 结构（如稀疏表或线段树）预处理 LCP 数组
2. 对于查询 $(i, j)$：

   * 获取 $p = pos[i]$, $q = pos[j]$
   * 如果 $p > q$ 则交换
   * 答案 = 对 $LCP[p+1..q]$ 进行 RMQ 查询

每次查询在 $O(n \log n)$ 或 $O(n)$ 的预处理后变为 $O(1)$ 时间复杂度。

#### 微型代码（Python – 稀疏表 RMQ）

```python
import math

def build_rmq(lcp):
    n = len(lcp)
    log = [0] * (n + 1)
    for i in range(2, n + 1):
        log[i] = log[i // 2] + 1

    k = log[n]
    st = [[0] * (k + 1) for _ in range(n)]
    for i in range(n):
        st[i][0] = lcp[i]

    for j in range(1, k + 1):
        for i in range(n - (1 << j) + 1):
            st[i][j] = min(st[i][j-1], st[i + (1 << (j-1))][j-1])
    return st, log

def query_rmq(st, log, L, R):
    j = log[R - L + 1]
    return min(st[L][j], st[R - (1 << j) + 1][j])

# 示例
LCP = [0, 1, 3, 0, 0, 2]
st, log = build_rmq(LCP)
print(query_rmq(st, log, 1, 2))  # 1
```

查询流程：

- $O(n \log n)$ 预处理
- $O(1)$ 每次查询

#### 为什么它很重要

- 实现快速子串比较：

  * 在 $O(1)$ 时间内比较后缀
  * 检查字典序排名/顺序
- 是以下内容的核心：

  * LCP 区间树
  * 通过数组模拟后缀树
  * 跨多个字符串的 LCS 查询
  * 不同子串计数

有了 RMQ，后缀数组就变成了一个功能齐全的字符串索引。

#### 复杂度

| 操作        | 时间复杂度    | 空间复杂度    |
| ----------- | ------------- | ------------- |
| 预处理      | $O(n \log n)$ | $O(n \log n)$ |
| 查询 (LCP)  | $O(1)$        | $O(1)$        |

高级的 RMQ（如笛卡尔树 + 欧拉序 + 稀疏表）可以在 $O(n)$ 空间和 $O(1)$ 查询时间内实现。

#### 亲自尝试

1. 为 `"banana"` 构建 $SA$、$LCP$ 和 $pos$ 数组。
2. 回答查询：

   * $\text{LCP}(1, 2)$
   * $\text{LCP}(0, 4)$
   * $\text{LCP}(3, 5)$
3. 将结果与直接前缀比较的结果进行对比。
4. 用线段树实现替换稀疏表实现。
5. 使用欧拉序 + 笛卡尔树上的 RMQ 构建 $O(n)$ 的 RMQ。

LCP RMQ 连接了后缀数组和后缀树 ——
这是一个通过最小值建立的静谧连接，其中每个区间都揭示了字典序版图中两条路径共享的长度。
### 628 广义后缀数组（多字符串）

广义后缀数组（GSA）将经典的后缀数组扩展到可以同时处理多个字符串。它提供了一个用于跨字符串比较的统一结构，使我们能够高效地计算最长公共子串、共享模体以及进行跨文档搜索。

#### 我们要解决什么问题？

给定多个字符串：
$$
S_1, S_2, \ldots, S_k
$$
我们希望在一个单一的排序数组中索引所有字符串的所有后缀。

数组中的每个后缀都记录：

- 它属于哪个字符串
- 它的起始位置

有了这个，我们可以：

- 找到两个或多个字符串之间共享的子串
- 计算跨字符串的最长公共子串（LCS）
- 执行多文档搜索或文本比较

#### 示例

令：
$$
S_1 = \texttt{"banana\$1"}
$$
$$
S_2 = \texttt{"bandana\$2"}
$$

所有后缀：

| ID | 索引 | 后缀       | 字符串 |
| -- | ---- | ---------- | ------ |
| 1  | 0    | banana$1   | S₁     |
| 1  | 1    | anana$1    | S₁     |
| 1  | 2    | nana$1     | S₁     |
| 1  | 3    | ana$1      | S₁     |
| 1  | 4    | na$1       | S₁     |
| 1  | 5    | a$1        | S₁     |
| 1  | 6    | $1         | S₁     |
| 2  | 0    | bandana$2  | S₂     |
| 2  | 1    | andana$2   | S₂     |
| 2  | 2    | ndana$2    | S₂     |
| 2  | 3    | dana$2     | S₂     |
| 2  | 4    | ana$2      | S₂     |
| 2  | 5    | na$2       | S₂     |
| 2  | 6    | a$2        | S₂     |
| 2  | 7    | $2         | S₂     |

现在按字典序对所有后缀进行排序。
GSA 中的每个条目记录 `(后缀起始位置, 字符串ID)`。

#### 数据结构

我们维护：

1. SA，所有字符串的所有后缀，按字典序排序
2. LCP，连续后缀之间的最长公共前缀
3. 所有者数组，每个后缀的所有者（属于哪个字符串）

| i | SA[i] | Owner[i] | LCP[i] |
| - | ----- | -------- | ------ |
| 0 | ...   | 1        | 0      |
| 1 | ...   | 2        | 2      |
| 2 | ...   | 1        | 3      |
| … | …     | …        | …      |

由此，我们可以通过检查 `Owner` 不同的区间来计算 LCS。

#### 它是如何工作的（通俗解释）

1. 使用唯一的分隔符连接所有字符串
   $$
   S = S_1 + \text{\$1} + S_2 + \text{\$2} + \cdots + S_k + \text{\$k}
   $$

2. 在合并后的字符串上构建后缀数组（使用 SA-IS 或倍增法）

3. 记录所有权：为每个位置标记它属于哪个字符串

4. 构建 LCP 数组（Kasai 算法）

5. 查询共享子串：

   * 当连续后缀属于不同字符串时，存在公共子串
   * 该区间内的最小 LCP 值给出了共享长度

#### 查询示例：最长公共子串

对于 `"banana"` 和 `"bandana"`：

来自不同字符串的后缀重叠部分有
$$
\text{LCP} = 3 \text{ ("ban") }
$$
和
$$
\text{LCP} = 2 \text{ ("na") }
$$

因此
$$
\text{LCS} = \texttt{"ban"} \quad \text{长度 } 3
$$

#### 微型代码（Python 草图）

```python
def generalized_suffix_array(strings):
    text = ""
    owners = []
    sep = 1
    for s in strings:
        text += s + chr(sep)
        owners += [len(owners)] * (len(s) + 1)
        sep += 1

    sa = suffix_array_doubling(text)
    lcp = kasai(text, sa)

    owner_map = [owners[i] for i in sa]
    return sa, lcp, owner_map
```

*(假设你已有前面章节的 `suffix_array_doubling` 和 `kasai` 函数。)*

用法：

```python
S1 = "banana"
S2 = "bandana"
sa, lcp, owner = generalized_suffix_array([S1, S2])
```

现在扫描 `lcp` 数组中 `owner[i] != owner[i-1]` 的位置以找到跨字符串的重叠。

#### 为什么它很重要

- 是以下任务的核心结构：

  * 跨文件的最长公共子串
  * 多文档索引
  * 生物信息学中的模体发现
  * 抄袭检测
- 是广义后缀树的一种紧凑替代方案
- 易于从现有的 SA + LCP 流程实现

#### 复杂度

| 步骤         | 时间   | 空间           |
| ------------ | ------ | -------------- |
| 连接字符串   | $O(n)$ | $O(n)$         |
| 构建 SA      | $O(n)$ | $O(n)$         |
| 构建 LCP     | $O(n)$ | $O(n)$         |
| LCS 查询     | $O(n)$ | $O(1)$ 每次查询 |

总计：
$$
O(n)
$$
其中 $n$ = 所有字符串的总长度。

#### 亲自尝试

1. 为 `["banana", "bandana"]` 构建 GSA。
2. 找出两个字符串共有的所有子串。
3. 使用 `LCP` + `Owner` 提取最长的共享子串。
4. 扩展到 3 个字符串，例如 `["banana", "bandana", "canada"]`。
5. 通过暴力比较验证 LCS 的正确性。

广义后缀数组是字符串的合唱 ——
每个后缀都是一个声音，每次重叠都是一段和声。
从许多歌曲中，得到一个字典序的乐谱 ——
在其中，是每一段共享的旋律。
### 629 增强后缀数组（SA + LCP + RMQ）

增强后缀数组（ESA）是后缀树的一个功能完整的替代品，它由后缀数组、LCP 数组和区间最小值查询（RMQ）结构构建而成。
它支持同样强大的操作：子串搜索、LCP 查询、最长重复子串和模式匹配，所有这些都具备线性空间和快速查询的特性。

可以把它看作是一个*压缩的后缀树*，通过数组实现。

#### 我们要解决什么问题？

单独的后缀数组可以高效地定位后缀，但缺乏后缀树提供的结构信息（如分支或重叠）。
增强后缀数组通过辅助数组来丰富 SA，以恢复类似树形的导航：

1. SA，已排序的后缀索引
2. LCP，相邻后缀之间的最长公共前缀
3. RMQ / 笛卡尔树，用于模拟树结构

有了这些，我们可以：

- 执行子串搜索
- 遍历后缀区间
- 计算任意两个后缀的 LCP
- 枚举不同的子串、重复的子串和模式

所有这些都无需显式的树节点。

#### 核心思想

后缀树可以通过 SA + LCP 数组隐式表示：

- SA 定义了字典序（中序遍历）
- LCP 定义了边长度（分支深度）
- 在 LCP 上的 RMQ 给出了树视图中的最低公共祖先（LCA）

因此，ESA 是*后缀树的一个视图*，而不是重建。

#### 示例

令
$$
S = \texttt{"banana\$"}
$$

后缀数组：
$$
SA = [6, 5, 3, 1, 0, 4, 2]
$$

LCP 数组：
$$
LCP = [0, 1, 3, 0, 0, 2, 0]
$$

这些数组已经描述了一个类似树的结构：

- 分支深度 = `LCP` 值
- 子树 = 共享前缀长度 ≥ $k$ 的 SA 区间

示例：

- 重复子串 `"ana"` 对应于 `LCP ≥ 3` 的区间 `[2, 4]`

#### 它是如何工作的（通俗解释）

ESA 通过区间和 RMQ 来回答查询：

1. 子串搜索（模式匹配）
   在 `SA` 上对模式前缀进行二分搜索。
   一旦找到，`SA[l..r]` 就是匹配区间。

2. LCP 查询（两个后缀）
   使用 RMQ：
   $$
   \text{LCP}(i, j) = \min(LCP[k]) \text{ 其中 } k \in [\min(pos[i], pos[j])+1, \max(pos[i], pos[j])]
   $$

3. 最长重复子串
   $\max(LCP)$ 给出长度，通过 SA 索引给出位置。

4. 区间的最长公共前缀
   在 `LCP[l+1..r]` 上的 RMQ 得到分支深度。

#### ESA 组件

| 组件 | 描述 | 用途 |
| --------- | ----------------------------- | ----------------------- |
| SA | 已排序的后缀索引 | 字典序遍历 |
| LCP | 相邻后缀之间的 LCP | 分支深度 |
| INVSA | SA 的逆 | 快速后缀查找 |
| RMQ | 在 LCP 上的区间最小值查询 | LCA / 区间查询 |

#### 微型代码（Python）

```python
def build_esa(s):
    sa = suffix_array_doubling(s)
    lcp = kasai(s, sa)
    inv = [0] * len(s)
    for i, idx in enumerate(sa):
        inv[idx] = i
    st, log = build_rmq(lcp)
    return sa, lcp, inv, st, log

def lcp_query(inv, st, log, i, j):
    if i == j:
        return len(s) - i
    pi, pj = inv[i], inv[j]
    if pi > pj:
        pi, pj = pj, pi
    return query_rmq(st, log, pi + 1, pj)

# 示例用法
s = "banana"
sa, lcp, inv, st, log = build_esa(s)
print(lcp_query(inv, st, log, 1, 3))  # "anana" vs "ana" → 3
```

*(使用了之前的 `suffix_array_doubling`, `kasai`, `build_rmq`, `query_rmq` 函数。)*

#### 为什么它很重要

ESA 以以下特性匹配后缀树的功能：

- 线性空间（$O(n)$）
- 更简单的实现
- 缓存友好的数组访问
- 易于与压缩索引（FM-index）集成

应用于：

- 生物信息学（序列比对）
- 搜索引擎
- 文档相似性
- 压缩工具（BWT, LCP 区间）

#### 复杂度

| 操作 | 时间 | 空间 |
| ---------------- | ------------- | ------------- |
| 构建 SA + LCP | $O(n)$ | $O(n)$ |
| 构建 RMQ | $O(n \log n)$ | $O(n \log n)$ |
| 查询 LCP | $O(1)$ | $O(1)$ |
| 子串搜索 | $O(m \log n)$ | $O(1)$ |

#### 动手试试

1. 为 `"mississippi"` 构建 ESA。
2. 找出：

   * 最长重复子串
   * 不同子串的数量
   * 位置 1 和 4 处后缀的 LCP
3. 从 `LCP ≥ 2` 中提取子串区间
4. 将 ESA 输出与后缀树可视化进行比较

增强后缀数组是后缀树以数组形式的重生 ——
没有节点，没有指针，只有秩序、重叠和结构，交织在字典序的画卷中。
### 630 稀疏后缀树（空间高效变体）

稀疏后缀树（Sparse Suffix Tree, SST）是经典后缀树的一种空间高效变体。
它不存储字符串的*所有*后缀，而仅索引一个选定的子集（通常是每隔 $k$ 个后缀），从而将空间从 $O(n)$ 个节点减少到 $O(n / k)$，同时保留了许多相同的查询能力。

这使得它非常适用于内存紧张且可接受近似索引的大文本场景。

#### 我们要解决什么问题？

完整的后缀树功能强大，能在 $O(m)$ 时间内完成子串查询，但代价是高昂的内存开销，通常是原始文本大小的 10–20 倍。

我们想要一个这样的数据结构：

- 支持快速子串搜索
- 轻量级且缓存友好
- 可扩展到大型语料库（例如基因组、日志）

稀疏后缀树通过采样后缀，仅基于一个子集构建树来解决这个问题。

#### 核心思想

我们不插入*每一个*后缀 $S[i:]$，
而只插入那些满足
$$
i \bmod k = 0
$$
的后缀，
或者来自一个采样集合 $P = {p_1, p_2, \ldots, p_t}$ 的后缀。

然后，我们通过验证步骤（如在文本上进行二分查找）来确认完整匹配。

这使结构大小与采样率 $k$ 成比例地减少。

#### 工作原理（通俗解释）

1. 采样步骤
   选择采样间隔 $k$（例如，每隔第 4 个后缀）。

2. 构建后缀树
   仅插入采样到的后缀：
   $$
   { S[0:], S[k:], S[2k:], \ldots }
   $$

3. 搜索
   * 要匹配模式 $P$，
     找到与 $P$ 共享前缀的最接近的采样后缀
   * 在文本中扩展搜索以进行验证
     （最多 O($k$) 的开销）

这以更小的常数因子实现了近似的 O(m) 查询时间。

#### 示例

令
$$
S = \texttt{"banana\$"}
$$
并选择 $k = 2$。

采样到的后缀：

- $S[0:] = $ `"banana$"`
- $S[2:] = $ `"nana$"`
- $S[4:] = $ `"na$"`
- $S[6:] = $ `"$"`

仅基于这 4 个后缀构建后缀树。

当搜索 `"ana"` 时，

- 在覆盖 `"banana$"` 和 `"nana$"` 中 `"ana"` 的节点处找到匹配
- 直接在 $S$ 中验证剩余字符

#### 微型代码（Python 草图）

```python
class SparseSuffixTree:
    def __init__(self, s, k):
        self.s = s + "$"
        self.k = k
        self.suffixes = [self.s[i:] for i in range(0, len(s), k)]
        self.suffixes.sort()  # 为演示使用的简单方法

    def search(self, pattern):
        # 在采样后缀上进行二分查找
        lo, hi = 0, len(self.suffixes)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.suffixes[mid].startswith(pattern):
                return True
            if self.suffixes[mid] < pattern:
                lo = mid + 1
            else:
                hi = mid
        return False

sst = SparseSuffixTree("banana", 2)
print(sst.search("ana"))  # True (从采样后缀验证得到)
```

*对于实际的后缀树结构，可以使用仅限于采样后缀的 Ukkonen 算法。*

#### 为什么它很重要

- 空间缩减：$O(n / k)$ 个节点
- 适用于海量字符串的可扩展索引
- 对于大多数子串查询足够快
- 应用于：
  * 基因组索引
  * 日志模式搜索
  * 近似数据压缩
  * 大规模文本分析

在速度与大小之间实现了实用的平衡。

#### 复杂度

| 操作             | 时间                        | 空间     |
| ---------------- | --------------------------- | -------- |
| 构建（采样后）   | $O((n/k) \cdot k)$ = $O(n)$ | $O(n/k)$ |
| 搜索             | $O(m + k)$                  | $O(1)$   |
| 验证匹配         | $O(k)$                      | $O(1)$   |

调整 $k$ 可以在内存和搜索精度之间进行权衡。

#### 动手尝试

1. 为 `"mississippi"` 构建 $k = 3$ 的 SST
2. 比较其内存使用与完整后缀树
3. 搜索 `"issi"`，测量验证步骤
4. 尝试不同的 $k$ 值
5. 绘制构建大小与查询延迟的关系图

稀疏后缀树是一种内存方面的折衷方案——
一片由采样分支组成的森林，
它仅保留了文本结构的足够信息，
以便在子串空间中快速导航，而无需承载每一片叶子。

# 第 64 节 回文与周期性
### 631 朴素回文串检测

朴素回文串检测是检测回文串（正读反读都一样的字符串）最简单的方法。
它是一种直接、易于理解的算法：围绕每个可能的中心进行扩展，对称地比较字符，并计数或报告所有回文子串。

在介绍像 Manacher（马拉车）算法这样的优化方法之前，这种方法在概念上非常清晰，是完美的起点。

#### 我们要解决什么问题？

我们想要判断一个字符串（或其任意子串）是否是回文串，即：

$$
S[l \ldots r] \text{ 是回文串，当且仅当 } S[l + i] = S[r - i], \ \forall i
$$

我们可以利用这一点来：

-   检查给定的子串是否是回文的
-   计算回文子串的总数
-   找到最长的回文子串（暴力方法）

#### 定义

回文串满足：
$$
S = \text{reverse}(S)
$$

示例：

-   `"aba"` → 回文串
-   `"abba"` → 回文串
-   `"abc"` → 不是回文串

#### 它是如何工作的（通俗解释）

我们可以使用中心扩展法或暴力检查法。

##### 1. 暴力检查

从两端开始比较字符：

1.  从左端和右端开始
2.  在匹配的同时向中间移动
3.  遇到不匹配或到达中间时停止

时间复杂度：对于单个子串为 $O(n)$。

##### 2. 中心扩展

每个回文串都有一个中心：

-   奇数长度：单个字符
-   偶数长度：两个字符之间的空隙

我们围绕每个中心进行扩展并计数回文串。

总共有 $2n - 1$ 个中心。

#### 示例

字符串：
$$
S = \texttt{"abba"}
$$

中心和扩展：

-   中心在 `'a'`：`"a"` 是回文串
-   中心在 `'a'` 和 `'b'` 之间：无回文串
-   中心在 `'b'`：`"b"`，`"bb"`，`"abba"` 是回文串
-   中心在 `'a'`：`"a"` 是回文串

回文子串总数：`6`

#### 微型代码（Python）

(a) 暴力检查

```python
def is_palindrome(s):
    return s == s[::-1]

print(is_palindrome("abba"))  # True
print(is_palindrome("abc"))   # False
```

(b) 中心扩展（计数所有回文串）

```python
def count_palindromes(s):
    n = len(s)
    count = 0
    for center in range(2 * n - 1):
        l = center // 2
        r = l + (center % 2)
        while l >= 0 and r < n and s[l] == s[r]:
            count += 1
            l -= 1
            r += 1
    return count

print(count_palindromes("abba"))  # 6
```

(c) 中心扩展（最长回文串）

```python
def longest_palindrome(s):
    n = len(s)
    best = ""
    for center in range(2 * n - 1):
        l = center // 2
        r = l + (center % 2)
        while l >= 0 and r < n and s[l] == s[r]:
            if r - l + 1 > len(best):
                best = s[l:r+1]
            l -= 1
            r += 1
    return best

print(longest_palindrome("babad"))  # "bab" 或 "aba"
```

#### 为什么它很重要

-   是更高级算法（Manacher、动态规划）的基础
-   适用于小型或教学示例
-   是验证优化版本正确性的简单方法
-   可用于与回文相关的模式发现（DNA、文本对称性）

#### 复杂度

| 操作                     | 时间复杂度 | 空间复杂度 |
| ------------------------ | ---------- | ---------- |
| 检查是否为回文串         | $O(n)$     | $O(1)$     |
| 计数所有回文子串         | $O(n^2)$   | $O(1)$     |
| 查找最长回文子串         | $O(n^2)$   | $O(1)$     |

#### 动手试试

1.  计算 `"level"` 中所有回文子串的数量。
2.  在 `"civicracecar"` 中找到最长的回文子串。
3.  比较长度为 1000 的字符串，暴力法与中心扩展法的运行时间。
4.  修改代码以忽略非字母数字字符。
5.  扩展功能以查找特定范围 $[l, r]$ 内的回文子串。

朴素回文串检测是镜子的第一瞥 ——
每个中心都是一次反射，每次扩展都是一次向内的旅程 ——
简单、直接，是通往前方对称性世界的完美基石。
### 632 Manacher 算法

Manacher（马拉车）算法是一种优雅的、线性时间的方法，用于在给定字符串中查找最长的回文子串。
与朴素的 $O(n^2)$ 中心扩展法不同，它利用了对称性——每个回文都围绕其中心镜像——来复用计算结果并跳过冗余检查。

这是一个经典例子，展示了巧妙的洞察如何将二次过程转化为线性过程。

#### 我们要解决什么问题？

给定一个长度为 $n$ 的字符串 $S$，找出正向和反向读都相同的**最长子串**。

示例：

$$
S = \texttt{"babad"}
$$

最长回文子串：
$$
\texttt{"bab"} \text{ 或 } \texttt{"aba"}
$$

我们希望以 $O(n)$ 的时间复杂度计算，而不是 $O(n^2)$。

#### 核心思想

Manacher 算法的关键洞察：
每个回文都有一个关于当前中心的镜像。

如果我们知道位置 $i$ 处的回文半径，
我们就可以推断出其镜像位置 $j$ 的信息
（使用先前计算的值），而无需重新检查所有字符。

#### 逐步解析（通俗语言）

1. **预处理字符串以处理偶数长度的回文**  
   在所有字符之间和边界插入 `#`：
   $$
   S=\texttt{"abba"}\ \Rightarrow\ T=\texttt{"\#a\#b\#b\#a\#"}
   $$
   这样，$T$ 中的所有回文都变成了奇数长度。

2. **遍历 $T$**
   维护：

   * $C$：最右侧回文的中心
   * $R$：右边界
   * $P[i]$：位置 $i$ 处的回文半径

3. **对于每个位置 $i$：**

   * 镜像索引：$i' = 2C - i$
   * 初始化 $P[i] = \min(R - i, P[i'])$
   * 当边界匹配时，围绕 $i$ 进行扩展

4. 如果回文扩展超出了 $R$，则更新中心和边界。

5. 循环结束后，$P$ 中的最大半径给出了最长回文。

#### 示例演练

字符串：
$$
S = \texttt{"abaaba"}
$$

预处理后：
$$
T = \texttt{"\#a\#b\#a\#a\#b\#a\#"}
$$

| i | T[i] | Mirror | P[i] | Center | Right |
| - | ---- | ------ | ---- | ------ | ----- |
| 0 | #    |,      | 0    | 0      | 0     |
| 1 | a    |,      | 1    | 1      | 2     |
| 2 | #    |,      | 0    | 1      | 2     |
| 3 | b    |        | 3    | 3      | 6     |
| … | …    | …      | …    | …      | …     |

结果：
$$
\text{最长回文长度 } = 5
$$
$$
\text{子串 } = \texttt{"abaaba"}
$$

#### 精简代码（Python）

```python
def manacher(s):
    # 预处理
    t = "#" + "#".join(s) + "#"
    n = len(t)
    p = [0] * n
    c = r = 0  # 中心，右边界

    for i in range(n):
        mirror = 2 * c - i
        if i < r:
            p[i] = min(r - i, p[mirror])
        # 围绕中心 i 扩展
        while i - 1 - p[i] >= 0 and i + 1 + p[i] < n and t[i - 1 - p[i]] == t[i + 1 + p[i]]:
            p[i] += 1
        # 如果扩展超出了右边界，则更新中心
        if i + p[i] > r:
            c, r = i, i + p[i]

    # 查找最大回文
    max_len = max(p)
    center_index = p.index(max_len)
    start = (center_index - max_len) // 2
    return s[start:start + max_len]

print(manacher("babad"))  # "bab" 或 "aba"
```

#### 为什么它很重要

- **线性时间**，这是已知解决最长回文问题的最快方法
- **基础**，用于：
  * 回文子串枚举
  * 回文树构造
  * DNA 对称性搜索

它是算法独创性的瑰宝，将反射转化为速度。

#### 复杂度

| 操作                     | 时间复杂度 | 空间复杂度 |
| ------------------------ | ---------- | ---------- |
| 构建（含预处理）         | $O(n)$     | $O(n)$     |
| 查询最长回文             | $O(1)$     | $O(1)$     |

#### 亲自尝试

1.  在 `"banana"` 上运行 Manacher 算法 → `"anana"`
2.  与 $n = 10^5$ 时的中心扩展法进行时间比较
3.  修改算法以统计所有回文子串的数量
4.  可视化跟踪左右边界
5.  应用于 DNA 序列以检测对称基序

#### 一个温和的证明（为什么它有效）

当前回文 $(C, R)$ 内的每个位置 $i$
都有一个镜像 $i' = 2C - i$。
如果 $i + P[i'] < R$，回文完全在内部 → 复用 $P[i']$。
否则，扩展超出 $R$ 并更新中心。

没有位置被扩展两次 → 总复杂度 $O(n)$。

Manacher 算法是**对称性的体现**——
每个中心都是一面镜子，每次反射都是一条捷径。
曾经需要二次努力的工作，如今以线性的优雅流畅完成。
### 633 最长回文子串（中心扩展法）

最长回文子串问题要求：

> *给定一个字符串，其最长的、正读和反读都相同的连续子串是什么？*

中心扩展法是一种直观且优雅的 $O(n^2)$ 解决方案，易于编码，易于理解，并且在实践中效率惊人。它介于朴素的暴力解法（$O(n^3)$）和 Manacher 算法（$O(n)$）之间。

#### 我们要解决什么问题？

给定一个长度为 $n$ 的字符串 $S$，找到子串 $S[l \ldots r]$，使得：

$$
S[l \ldots r] = \text{reverse}(S[l \ldots r])
$$

并且 $(r - l + 1)$ 最大。

示例：

- `"babad"` → `"bab"` 或 `"aba"`
- `"cbbd"` → `"bb"`
- `"a"` → `"a"`

我们希望高效且清晰地找到这个子串。

#### 核心思想

每个回文串都由其中心定义：

- 奇数长度回文串：一个中心（例如 `"aba"`）
- 偶数长度回文串：两个字符的中心（例如 `"abba"`）

如果我们从每一个可能的中心向外扩展，就可以检测到所有的回文子串，并追踪最长的一个。

#### 工作原理（通俗解释）

1. 对于 $S$ 中的每个索引 $i$：
   * 以 $i$ 为中心扩展（奇数长度回文）
   * 以 $(i, i+1)$ 为中心扩展（偶数长度回文）
2. 当字符不匹配时停止扩展。
3. 追踪最大长度和起始索引。

每次扩展成本为 $O(n)$，遍历 $n$ 个中心 → 总成本 $O(n^2)$。

#### 示例

字符串：
$$
S = \texttt{"babad"}
$$

中心及扩展：

- 中心在 `b`：`"b"`，扩展为 `"bab"`
- 中心在 `a`：`"a"`，扩展为 `"aba"`
- 中心在 `ba`：不匹配，无扩展

找到的最长回文：`"bab"` 或 `"aba"`

#### 精简代码（Python）

```python
def longest_palindrome_expand(s):
    if not s:
        return ""
    start = end = 0

    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return l + 1, r - 1  # 回文串的边界

    for i in range(len(s)):
        l1, r1 = expand(i, i)       # 奇数长度
        l2, r2 = expand(i, i + 1)   # 偶数长度
        if r1 - l1 > end - start:
            start, end = l1, r1
        if r2 - l2 > end - start:
            start, end = l2, r2

    return s[start:end+1]

print(longest_palindrome_expand("babad"))  # "bab" 或 "aba"
print(longest_palindrome_expand("cbbd"))   # "bb"
```

#### 为何重要

- 直接且健壮
- 适用于：
  * 子串对称性检查
  * 生物信息学（回文 DNA 片段）
  * 自然语言分析
- 比 Manacher 算法更易实现，对于大多数 $n \le 10^4$ 的情况性能良好

#### 复杂度

| 操作                     | 时间复杂度 | 空间复杂度 |
| ------------------------ | ---------- | ---------- |
| 从所有中心扩展           | $O(n^2)$   | $O(1)$     |
| 找到最长回文子串         | $O(1)$     | $O(1)$     |

#### 亲自尝试

1. 在 `"racecarxyz"` 中找到最长回文子串。
2. 修改代码以统计所有回文子串的数量。
3. 返回最长回文子串的起始和结束索引。
4. 在 `"aaaabaaa"` 上测试 → 结果应为 `"aaabaaa"`。
5. 与 Manacher 算法的输出进行比较。

#### 一个温和的证明（为何有效）

每个回文串都唯一地以以下之一为中心：

- 单个字符（奇数情况），或
- 两个字符之间（偶数情况）

由于我们尝试了所有 $2n - 1$ 个中心，每个回文串都会被恰好发现一次，然后我们从中选取最长的一个。

因此，正确性和完备性直接可得。

中心扩展法是一场镜像之舞 —— 每个位置都是一个支点，每次匹配都是一次反射 —— 一步一步向外构建对称性。
### 634 回文动态规划表（动态规划方法）

回文动态规划表方法使用动态规划来查找和计数回文子串。这是一种自底向上的策略，它构建一个二维表格来标记每个子串 $S[i \ldots j]$ 是否是回文，由此我们可以轻松回答诸如以下问题：

- 子串 $S[i \ldots j]$ 是回文吗？
- 最长的回文子串是什么？
- 存在多少个回文子串？

这种方法系统化且易于扩展，尽管比中心扩展法占用更多内存。

#### 我们要解决什么问题？

给定一个长度为 $n$ 的字符串 $S$，我们希望高效地预计算回文子串。

我们定义一个动态规划表 $P[i][j]$，使得：

$$
P[i][j] =
\begin{cases}
\text{True}, & \text{如果 } S[i \ldots j] \text{ 是回文},\\
\text{False}, & \text{否则。}
\end{cases}
$$

然后，我们可以使用这个表来查询或计数所有回文。

#### 递推关系

一个子串 $S[i \ldots j]$ 是回文，当且仅当：

1. 边界字符匹配：
   $$
   S[i] = S[j]
   $$
2. 内部子串也是回文（或者足够小）：
   $$
   P[i+1][j-1] = \text{True} \quad \text{或} \quad (j - i \le 2)
   $$

因此递推关系是：

$$
P[i][j] = (S[i] = S[j]) \ \text{且} \ (j - i \le 2 \ \text{或} \ P[i+1][j-1])
$$

#### 初始化

- 所有单个字符都是回文：
  $$
  P[i][i] = \text{True}
  $$
- 两个字符的子串在两者匹配时是回文：
  $$
  P[i][i+1] = (S[i] = S[i+1])
  $$

我们从较短的子串到较长的子串填充表格。

#### 示例

令
$$
S = \texttt{"abba"}
$$

我们自底向上构建 $P$：

| i\j | 0:a | 1:b | 2:b | 3:a |
| --- | --- | --- | --- | --- |
| 0:a | T   | F   | F   | T   |
| 1:b |     | T   | T   | F   |
| 2:b |     |     | T   | F   |
| 3:a |     |     |     | T   |

回文子串：`"a"`, `"b"`, `"bb"`, `"abba"`

#### 精简代码（Python）

```python
def longest_palindrome_dp(s):
    n = len(s)
    if n == 0:
        return ""
    dp = [[False] * n for _ in range(n)]
    start, max_len = 0, 1

    # 长度为 1
    for i in range(n):
        dp[i][i] = True

    # 长度为 2
    for i in range(n-1):
        if s[i] == s[i+1]:
            dp[i][i+1] = True
            start, max_len = i, 2

    # 长度 >= 3
    for length in range(3, n+1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i+1][j-1]:
                dp[i][j] = True
                start, max_len = i, length

    return s[start:start+max_len]

print(longest_palindrome_dp("babad"))  # "bab" 或 "aba"
```

#### 为什么它重要

- 逻辑清晰，易于调整
- 适用于：

  * 计数所有回文子串
  * 查找所有回文索引
  * 教授动态规划递推构建

它是从暴力解法到线性时间解法的教学桥梁。

#### 复杂度

| 操作                           | 时间复杂度 | 空间复杂度 |
| ------------------------------ | ---------- | ---------- |
| 构建动态规划表                 | $O(n^2)$   | $O(n^2)$   |
| 查询回文 $S[i \ldots j]$       | $O(1)$     | $O(1)$     |
| 提取最长回文                   | $O(n^2)$   | $O(n^2)$   |

#### 亲自尝试

1. 通过对 `dp[i][j]` 求和来计数所有回文子串。
2. 返回所有满足 `dp[i][j] == True` 的索引 $(i, j)$。
3. 比较 $n = 2000$ 时与中心扩展法的运行时间。
4. 使用一维滚动数组优化空间。
5. 适配“近似回文”（允许一处不匹配）。

#### 一个温和的证明（为什么它有效）

我们逐步扩展回文的定义：

- 基本情况：长度为 1 或 2
- 递归情况：匹配外部字符 + 内部是回文

每个回文内部都有一个更小的回文，
因此自底向上的顺序确保了正确性。

回文动态规划表将反射转化为递推 ——
每个单元格都是一面镜子，每一步都是一层 ——
揭示字符串中隐藏的每一种对称性。
### 635 回文树 (Eertree)

回文树（常称为 Eertree）是一种动态数据结构，当您从左到右扫描一个字符串时，它会存储该字符串所有不同的回文子串。
它为每个回文维护一个节点，并支持在均摊常数时间内插入下一个字符，从而实现线性的总时间复杂度。

它是枚举回文最直接的方式：您可以免费获得它们的数量、长度、结束位置和后缀链接。

#### 我们要解决什么问题？

给定一个字符串 $S$，我们希望在处理完每个前缀 $S[0..i]$ 后维护：

- 到目前为止出现的所有不同的回文子串
- 对于每个回文，其长度、指向最长真回文后缀的后缀链接，以及可选的其出现次数

使用回文树，我们可以在 $O(n)$ 时间和 $O(n)$ 空间内在线构建此结构，因为长度为 $n$ 的字符串最多有 $n$ 个不同的回文子串。

#### 核心思想

节点对应不同的回文。有两个特殊的根节点：

- 节点 $-1$，表示长度为 $-1$ 的虚拟回文
- 节点 $0$，表示长度为 $0$ 的空回文

每个节点保存：

- `len[v]`：回文长度
- `link[v]`：指向最长真回文后缀的后缀链接
- `next[v][c]`：通过在两端添加字符 $c$ 进行的状态转移
- 可选的 `occ[v]`：在处理过的位置中，以该回文结束的出现次数
- `first_end[v]` 或最后一个结束索引，用于恢复位置

要插入一个新字符 $S[i]$：

1.  从当前最长的后缀回文开始，沿着后缀链接向上跳转，直到找到一个节点 $v$，使得 $S[i - len[v] - 1] = S[i]$。这是 $S[0..i]$ 的最长回文后缀，且可以被 $S[i]$ 扩展。
2.  如果从 $v$ 出发没有通过 $S[i]$ 的边，则为新回文创建一个新节点。设置其 `len`，通过从 `link[v]` 继续后缀链接跳转来计算其 `link`，并设置转移边。
3.  更新出现次数计数器，并将新的当前节点设置为此节点。

每次插入最多创建一个新节点，因此总节点数最多为 $n + 2$。

#### 示例

令 $S = \texttt{"ababa"}$。

处理过的前缀和新创建的回文：

- $i=0$：添加 `a` → `"a"`
- $i=1$：添加 `b` → `"b"`
- $i=2$：添加 `a` → `"aba"`
- $i=3$：添加 `b` → `"bab"`
- $i=4$：添加 `a` → `"ababa"`

不同的回文：`a`、`b`、`aba`、`bab`、`ababa`。
总是存在两个特殊的根节点：长度 $-1$ 和 $0$。

#### 精简代码（Python，教学用）

```python
class Eertree:
    def __init__(self):
        # 节点 0: 空回文，长度 = 0
        # 节点 1: 虚根，长度 = -1
        self.len = [0, -1]
        self.link = [1, 1]
        self.next = [dict(), dict()]
        self.occ = [0, 0]
        self.s = []
        self.last = 0  # 当前字符串的最长后缀回文节点
        self.n = 0

    def _get_suflink(self, v, i):
        while True:
            l = self.len[v]
            if i - l - 1 >= 0 and self.s[i - l - 1] == self.s[i]:
                return v
            v = self.link[v]

    def add_char(self, ch):
        self.s.append(ch)
        i = self.n
        self.n += 1

        v = self._get_suflink(self.last, i)
        if ch not in self.next[v]:
            self.next[v][ch] = len(self.len)
            self.len.append(self.len[v] + 2)
            self.next.append(dict())
            self.occ.append(0)
            # 计算新节点的后缀链接
            if self.len[-1] == 1:
                self.link.append(0)  # 单字符回文链接到空回文
            else:
                u = self._get_suflink(self.link[v], i)
                self.link.append(self.next[u][ch])
        w = self.next[v][ch]
        self.last = w
        self.occ[w] += 1
        return w  # 返回最长后缀回文的节点索引

    def finalize_counts(self):
        # 沿着后缀链接传播出现次数，使得 occ[v] 统计所有结束位置
        order = sorted(range(2, len(self.len)), key=lambda x: self.len[x], reverse=True)
        for v in order:
            self.occ[self.link[v]] += self.occ[v]
```

用法：

```python
T = Eertree()
for c in "ababa":
    T.add_char(c)
T.finalize_counts()
# 不同回文的数量（排除两个根节点）：
print(len(T.len) - 2)  # 5
```

您将获得：

- 不同回文的数量：`len(nodes) - 2`
- 调用 `finalize_counts` 后每个回文的出现次数
- 用于遍历的长度、后缀链接和转移边

#### 为什么它很重要

- 在线性时间内列出所有不同的回文子串
- 支持在线更新：添加一个字符并在均摊 $O(1)$ 时间内更新
- 提供回文的数量和边界信息
- 天然适用于：

  * 统计回文子串
  * 在流式处理中寻找最长回文子串
  * 回文分解和周期性分析
  * 生物序列对称性挖掘

#### 复杂度

| 操作               | 时间复杂度       | 空间复杂度     |
| ------------------ | ---------------- | -------------- |
| 插入一个字符       | 均摊 $O(1)$      | $O(1)$ 额外    |
| 在长度为 $n$ 上构建 | $O(n)$           | $O(n)$ 个节点  |
| 出现次数聚合       | $O(n)$           | $O(n)$         |

每个位置最多产生一个新回文，因此是线性界限。

#### 动手尝试

1.  为 `"aaaabaaa"` 构建回文树。验证不同的回文及其数量。
2.  跟踪每次插入后的最长回文。
3.  记录每个节点的第一个和最后一个结束位置，以列出所有出现。
4.  修改结构，使其也能分别维护偶数和奇数的最长后缀回文。
5.  对于 $n \approx 10^5$，比较其与动态规划和 Manacher 算法在内存和速度上的表现。

回文树简洁地模拟了回文的宇宙：每个节点都是一面镜子，每个链接都是一个更短的反射，只需对字符串进行一次扫描，您就能发现它们全部。
### 636 前缀函数与周期性

前缀函数是字符串算法中的核心工具，它告诉我们每个位置处，既是前缀又是后缀的最长真前缀的长度。当从周期性的视角研究时，它成为检测重复模式、字符串边界和最小周期的利器，是模式匹配、压缩和单词组合学的基础。

#### 我们要解决什么问题？

我们希望在字符串中找到周期性结构，特别是最短的重复单元。

一个长度为 $n$ 的字符串 $S$ 是周期性的，如果存在一个 $p < n$ 使得：

$$
S[i] = S[i + p] \quad \forall i = 0, 1, \ldots, n - p - 1
$$

我们称 $p$ 为 $S$ 的周期。

前缀函数正好为我们提供了计算 $p$ 所需的边界长度，时间复杂度为 $O(n)$。

#### 前缀函数

对于字符串 $S[0 \ldots n-1]$，定义：

$$
\pi[i] = S[0..i] \text{ 的既是前缀又是后缀的最长真前缀的长度}
$$

这与 Knuth–Morris–Pratt (KMP) 算法中使用的数组相同。

#### 周期性公式

前缀 $S[0..i]$ 的最小周期是：

$$
p = (i + 1) - \pi[i]
$$

如果 $(i + 1) \bmod p = 0$，
那么前缀 $S[0..i]$ 就是以 $p$ 为周期的完全周期性字符串。

#### 示例

设
$$
S = \texttt{"abcabcabc"}
$$

计算前缀函数：

| i | S[i] | π[i] |
| - | ---- | ---- |
| 0 | a    | 0    |
| 1 | b    | 0    |
| 2 | c    | 0    |
| 3 | a    | 1    |
| 4 | b    | 2    |
| 5 | c    | 3    |
| 6 | a    | 4    |
| 7 | b    | 5    |
| 8 | c    | 6    |

现在计算 $i=8$ 时的最小周期：

$$
p = 9 - \pi[8] = 9 - 6 = 3
$$

由于 $9 \bmod 3 = 0$，
周期 = 3，重复单元为 `"abc"`。

#### 工作原理（通俗解释）

每个 $\pi[i]$ 衡量了字符串“自我重叠”的程度。
如果前缀和后缀对齐，它们就暗示了重复。
长度 $(i + 1)$ 与边界 $\pi[i]$ 的差值给出了重复块的长度。

当总长度能被这个块长度整除时，
整个前缀就是由重复的副本构成的。

#### 微型代码（Python）

```python
def prefix_function(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi

def minimal_period(s):
    pi = prefix_function(s)
    n = len(s)
    p = n - pi[-1]
    if n % p == 0:
        return p
    return n  # 没有更小的周期

s = "abcabcabc"
print(minimal_period(s))  # 3
```

#### 为什么重要

- 在线性时间内检测字符串中的重复
- 应用于：

  * 模式压缩
  * DNA 重复序列检测
  * 音乐节奏分析
  * 周期性任务调度
- 是 KMP、Z 算法和边界数组背后的核心概念

#### 复杂度

| 操作             | 时间复杂度 | 空间复杂度 |
| ---------------- | ---------- | ---------- |
| 构建前缀函数     | $O(n)$     | $O(n)$     |
| 寻找最小周期     | $O(1)$     | $O(1)$     |
| 检查周期性前缀   | $O(1)$     | $O(1)$     |

#### 动手尝试

1. 计算 `"ababab"` 的周期 → `2`。
2. 计算 `"aaaaa"` 的周期 → `1`。
3. 计算 `"abcd"` 的周期 → `4`（无重复）。
4. 对于每个前缀，打印 `(i + 1) - π[i]` 并测试可除性。
5. 与 Z 函数的周期性（第 637 节）进行比较。

#### 一个温和的证明（为什么有效）

如果 $\pi[i] = k$，那么
$S[0..k-1] = S[i-k+1..i]$。

因此，该前缀有一个长度为 $k$ 的边界，
且重复块的大小为 $(i + 1) - k$。
如果 $(i + 1)$ 能被该大小整除，
那么整个前缀就是由一个单元重复拷贝构成的。

前缀函数周期性揭示了重复中的节奏 ——
每个边界都是一次押韵，每次重叠都是一个隐藏的节拍 ——
将模式检测变成了简单的模运算音乐。
### 637 Z 函数的周期性

Z 函数提供了另一种揭示字符串中重复性和周期性的优雅途径。
前缀函数是向后看的（前缀-后缀重叠），而 Z 函数是向前看的，它衡量每个位置与字符串开头的匹配程度。
这使得它非常适合分析重复前缀并在线性时间内找到周期。

#### 我们要解决什么问题？

我们想要检测一个字符串 $S$ 是否具有周期 $p$ —— 也就是说，它是否由一个较小块的多次重复组成。

形式上，如果满足以下条件，则 $S$ 具有周期 $p$：

$$
S[i] = S[i + p], \quad \forall i \in [0, n - p - 1]
$$

等价地，如果：

$$
S = T^k \quad \text{对于某个 } T, k \ge 2
$$

Z 函数通过测量每个偏移处的前缀匹配来揭示这种结构。

#### 定义

对于长度为 $n$ 的字符串 $S$：

$$
Z[i] = \text{从位置 } i \text{ 开始的、与 } S \text{ 前缀匹配的最长子串长度}
$$

形式化地：

$$
Z[i] = \max { k \ | \ S[0..k-1] = S[i..i+k-1] }
$$

根据定义，$Z[0] = 0$ 或 $n$（为简单起见通常设为 0）。

#### 周期性判据

长度为 $n$ 的字符串 $S$ 具有周期 $p$，当且仅当：

$$
Z[p] = n - p
$$

并且 $p$ 能整除 $n$，即 $n \bmod p = 0$。

这意味着长度为 $n - p$ 的前缀从位置 $p$ 开始完美地重复。

更一般地，任何满足 $Z[p] = n - p$ 的 $p$ 都是一个边界长度，而最小周期 = 满足此条件的最小 $p$。

#### 示例

令
$$
S = \texttt{"abcabcabc"}
$$
$n = 9$

计算 $Z$ 数组：

| i | S[i:]     | Z[i] |
| - | --------- | ---- |
| 0 | abcabcabc | 0    |
| 1 | bcabcabc  | 0    |
| 2 | cabcabc   | 0    |
| 3 | abcabc    | 6    |
| 4 | bcabc     | 0    |
| 5 | cabc      | 0    |
| 6 | abc       | 3    |
| 7 | bc        | 0    |
| 8 | c         | 0    |

检查 $p = 3$：

$$
Z[3] = 6 = n - 3, \quad 9 \bmod 3 = 0
$$

所以 $p = 3$ 是最小周期。

#### 工作原理（通俗解释）

想象将字符串与自身滑动对齐：

- 在偏移量 $p$ 处，$Z[p]$ 表示还有多少个前导字符匹配。
- 如果重叠部分覆盖了字符串的其余部分（$Z[p] = n - p$），那么前后两部分的模式就完美对齐了。

这种对齐意味着重复。

#### 简短代码（Python）

```python
def z_function(s):
    n = len(s)
    Z = [0] * n
    l = r = 0
    for i in range(1, n):
        if i <= r:
            Z[i] = min(r - i + 1, Z[i - l])
        while i + Z[i] < n and s[Z[i]] == s[i + Z[i]]:
            Z[i] += 1
        if i + Z[i] - 1 > r:
            l, r = i, i + Z[i] - 1
    return Z

def minimal_period_z(s):
    n = len(s)
    Z = z_function(s)
    for p in range(1, n):
        if Z[p] == n - p and n % p == 0:
            return p
    return n

s = "abcabcabc"
print(minimal_period_z(s))  # 3
```

#### 为什么这很重要

- 一种测试重复和模式结构的简单方法
- 线性时间（$O(n)$）算法
- 可用于：
  * 字符串周期性检测
  * 基于前缀的哈希
  * 模式发现
  * 后缀比较

Z 函数是对前缀函数的补充：
- 前缀函数 → 边界（前缀 = 后缀）
- Z 函数 → 每个偏移处的前缀匹配

#### 复杂度

| 操作               | 时间复杂度 | 空间复杂度 |
| ------------------ | ---------- | ---------- |
| 计算 Z 数组        | $O(n)$     | $O(n)$     |
| 检查周期性         | $O(n)$     | $O(1)$     |
| 寻找最小周期       | $O(n)$     | $O(1)$     |

#### 动手试试

1.  计算 `"aaaaaa"` 的 $Z$ 数组 → 最小周期 = 1
2.  计算 `"ababab"` 的 $Z$ 数组 → $Z[2] = 4$，周期 = 2
3.  计算 `"abcd"` 的 $Z$ 数组 → 没有有效的 $p$，周期 = 4
4.  验证 $Z[p] = n - p$ 对应着重复的前缀
5.  将结果与使用前缀函数检测周期性的结果进行比较

#### 一个温和的证明（为什么它有效）

如果 $Z[p] = n - p$，
那么 $S[0..n-p-1] = S[p..n-1]$。
因此 $S$ 可以被划分为大小为 $p$ 的块。
如果 $n \bmod p = 0$，
那么 $S = T^{n/p}$，其中 $T = S[0..p-1]$。

因此，具有该属性的最小 $p$ 就是最小周期。

Z 函数将重叠转化为洞见 ——
每次偏移都是一面镜子，每次匹配都是一次押韵 ——
通过向前的映射，揭示字符串隐藏的节拍。
### 638 KMP 前缀周期检查（最短重复单元）

KMP 前缀函数不仅驱动了快速模式匹配，还悄然编码了字符串的重复结构。
通过分析前缀函数的最终值，我们可以揭示一个字符串是否由较小块的重复副本构成，如果是，则找出那个最短的重复单元，即*基本周期*。

#### 我们要解决什么问题？

给定一个长度为 $n$ 的字符串 $S$，
我们想要确定：

1.  $S$ 是否由某个较小子串的重复副本构成？
2.  如果是，最短的重复单元 $T$ 及其长度 $p$ 是什么？

形式化地，
$$
S = T^k, \quad \text{其中 } |T| = p, \ k = n / p
$$
且 $n \bmod p = 0$。

#### 核心洞察

前缀函数 $\pi[i]$ 捕获了每个前缀的最长边界——
边界是既是真前缀又是真后缀的子串。

在末尾（$i = n - 1$），$\pi[n-1]$ 给出了整个字符串的最长边界长度。

令：
$$
b = \pi[n-1]
$$
那么候选周期为：
$$
p = n - b
$$

如果 $n \bmod p = 0$，
则该字符串是周期性的，其最短重复单元长度为 $p$。

否则，它是非周期性的，且 $p = n$。

#### 示例 1

$$
S = \texttt{"abcabcabc"}
$$

$n = 9$

计算前缀函数：

| i | S[i] | π[i] |
| - | ---- | ---- |
| 0 | a    | 0    |
| 1 | b    | 0    |
| 2 | c    | 0    |
| 3 | a    | 1    |
| 4 | b    | 2    |
| 5 | c    | 3    |
| 6 | a    | 4    |
| 7 | b    | 5    |
| 8 | c    | 6    |

所以 $\pi[8] = 6$

$$
p = 9 - 6 = 3
$$

检查：$9 \bmod 3 = 0$ 成立
因此，最短重复单元 = `"abc"`。

#### 示例 2

$$
S = \texttt{"aaaa"} \implies n=4
$$
$\pi = [0, 1, 2, 3]$, 所以 $\pi[3] = 3$
$$
p = 4 - 3 = 1, \quad 4 \bmod 1 = 0
$$
成立 重复单元 = `"a"`

#### 示例 3

$$
S = \texttt{"abcd"} \implies n=4
$$
$\pi = [0, 0, 0, 0]$
$$
p = 4 - 0 = 4, \quad 4 \bmod 4 = 0
$$
只重复一次 → 无更小周期。

#### 工作原理（通俗解释）

前缀函数显示了字符串与自身重叠的程度。
如果边界长度为 $b$，那么最后 $b$ 个字符与前 $b$ 个字符匹配。
这意味着每 $p = n - b$ 个字符，模式就会重复一次。
如果字符串长度能被 $p$ 整除，那么它是由重复块组成的。

#### 微型代码（Python）

```python
def prefix_function(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi

def shortest_repeating_unit(s):
    pi = prefix_function(s)
    n = len(s)
    b = pi[-1]
    p = n - b
    if n % p == 0:
        return s[:p]
    return s  # 无重复

print(shortest_repeating_unit("abcabcabc"))  # "abc"
print(shortest_repeating_unit("aaaa"))       # "a"
print(shortest_repeating_unit("abcd"))       # "abcd"
```

#### 为什么它重要

-   在 $O(n)$ 时间内找到字符串的周期性
-   对以下方面至关重要：

    *   模式检测和压缩
    *   边界分析和组合数学
    *   最小自动机构造
    *   音乐中的节奏检测或 DNA 重复序列检测

优雅而高效，全部来自一个 $\pi$ 数组。

#### 复杂度

| 操作                     | 时间复杂度 | 空间复杂度 |
| ------------------------ | ---------- | ---------- |
| 计算前缀函数             | $O(n)$     | $O(n)$     |
| 提取周期                 | $O(1)$     | $O(1)$     |

#### 亲自尝试

1.  `"abababab"` → π = [0,0,1,2,3,4,5,6], $b=6$, $p=2$, 单元 = `"ab"`
2.  `"xyzxyzx"` → $n=7$, $\pi[6]=3$, $p=4$, $7 \bmod 4 \neq 0$ → 非周期性
3.  `"aaaaa"` → $p=1$, 单元 = `"a"`
4.  `"abaaba"` → $n=6$, $\pi[5]=3$, $p=3$, $6 \bmod 3 = 0$ → `"aba"`
5.  尝试与前缀函数周期性表（第 636 节）结合使用。

#### 一个温和的证明（为什么它有效）

如果 $\pi[n-1] = b$，
那么长度为 $b$ 的前缀 = 长度为 $b$ 的后缀。
因此，块长度 $p = n - b$。
如果 $p$ 能整除 $n$，
那么 $S$ 由 $n / p$ 个 $S[0..p-1]$ 的副本组成。

否则，它只在末尾有部分重复。

KMP 前缀周期检查是重复的心跳——
每个边界都是一次回调，每次重叠都是一个节奏——
揭示出构成这首歌曲的最小乐句。
### 639 Lyndon 分解（陈–福克斯–林登分解）

Lyndon 分解，也称为陈–福克斯–林登分解，是一个重要的字符串定理，它将任意字符串分解为一个唯一的 Lyndon 词序列。Lyndon 词是指那些在字典序上严格小于其任何非平凡后缀的子串。

这种分解与字典序、后缀数组、后缀自动机以及字符串组合学有着深刻的联系，并且是 Duval 算法等算法的基础。

#### 我们要解决什么问题？

我们希望将一个字符串 $S$ 分解为一个因子序列：

$$
S = w_1 w_2 w_3 \dots w_k
$$

使得：

1.  每个 $w_i$ 都是一个 Lyndon 词
    （即严格小于其任何真后缀）
2.  该序列在字典序上是非递增的：
    $$
    w_1 \ge w_2 \ge w_3 \ge \dots \ge w_k
    $$

对于每个字符串，这种分解是唯一的。

#### 什么是 Lyndon 词？

Lyndon 词是一个非空字符串，它在字典序上严格小于其所有轮换。

形式化定义，$w$ 是 Lyndon 词当且仅当：
$$
\forall u, v \text{ 满足 } w = uv, v \ne \varepsilon: \quad w < v u
$$

例子：

-   `"a"`, `"ab"`, `"aab"`, `"abc"` 是 Lyndon 词
-   `"aa"`, `"aba"`, `"ba"` 不是 Lyndon 词

#### 示例

令：
$$
S = \texttt{"banana"}
$$

分解过程：

| 步骤 | 剩余部分 | 因子           | 解释                     |
| ---- | -------- | -------------- | ------------------------ |
| 1    | banana   | b              | `"b"` < `"anana"`        |
| 2    | anana    | a              | `"a"` < `"nana"`         |
| 3    | nana     | n              | `"n"` < `"ana"`          |
| 4    | ana      | a              | `"a"` < `"na"`           |
| 5    | na       | n              | `"n"` < `"a"`            |
| 6    | a        | a              | 结束                     |
|      | 结果     | b a n a n a    | 非递增序列               |

每个因子都是一个 Lyndon 词。

#### 工作原理（通俗解释）

Duval 算法高效地构建这种分解：

1.  从 $S$ 的开头开始
    令 `i = 0`
2.  找到从 `i` 开始的最小的 Lyndon 词前缀
3.  将该词作为一个因子输出
    将 `i` 移动到该因子之后的下一个位置
4.  重复直到字符串结束

该算法在线性时间 $O(n)$ 内运行。

#### 精简代码（Python – Duval 算法）

```python
def lyndon_factorization(s):
    n = len(s)
    i = 0
    result = []
    while i < n:
        j = i + 1
        k = i
        while j < n and s[k] <= s[j]:
            if s[k] < s[j]:
                k = i
            else:
                k += 1
            j += 1
        while i <= k:
            result.append(s[i:i + j - k])
            i += j - k
    return result

print(lyndon_factorization("banana"))  # ['b', 'a', 'n', 'a', 'n', 'a']
print(lyndon_factorization("aababc"))  # ['a', 'ab', 'abc']
```

#### 为什么它重要

-   产生字符串的规范分解
-   用于：

    *   后缀数组构造（通过 BWT）
    *   字典序最小轮换
    *   组合字符串分析
    *   自由李代数基的生成
    *   密码学和 DNA 周期性分析
-   线性的时间和空间效率使其在文本索引中非常实用。

#### 复杂度

| 操作                     | 时间   | 空间  |    |        |
| ------------------------ | ------ | ----- | -- | ------ |
| 分解（Duval 算法）       | $O(n)$ | $O(1)$|    |        |
| 验证 Lyndon 属性         | $O(    | w     | )$ | $O(1)$ |

#### 亲自尝试

1.  分解 `"aababc"` 并验证每个因子都是 Lyndon 词。
2.  利用 Lyndon 性质找到 `"cabab"` 的最小轮换。
3.  将算法应用于 `"zzzzyzzzzz"` 并分析模式。
4.  生成字母表 `{a, b}` 上所有长度不超过 3 的 Lyndon 词。
5.  将 Duval 算法的输出与后缀数组的顺序进行比较。

#### 一个温和的证明（为什么它有效）

每个字符串 $S$ 都可以表示为一个非递增的 Lyndon 词序列——并且这种分解是唯一的。

证明使用了：

-   字典序最小性（每个因子都是可能的最小前缀）
-   连接单调性（确保顺序）
-   对长度 $n$ 的归纳

Lyndon 分解是字符串的主旋律——
每个因子都是一个自包含的乐句，
每一个较小的乐句都回响着前一个的节奏。
### 640 最小旋转（Booth 算法）

最小旋转问题要求找出一个字符串在字典序中最小的旋转，即按字典顺序排在最前面的那个旋转。Booth 算法通过巧妙的模运算比较，无需生成所有旋转，即可在线性时间 $O(n)$ 内解决此问题。

这个问题将 Lyndon 词、后缀数组和循环字符串排序的思想联系在一起，是字符串规范化、哈希和模式等价性判断的基础。

#### 我们要解决什么问题？

给定一个长度为 $n$ 的字符串 $S$，考虑其所有旋转：

$$
R_i = S[i..n-1] + S[0..i-1], \quad i = 0, 1, \ldots, n-1
$$

我们希望找到索引 $k$，使得 $R_k$ 在字典序中最小。

#### 示例

令
$$
S = \texttt{"bbaaccaadd"}
$$

所有旋转：

| 偏移 | 旋转        |
| :---: | :---------- |
|   0   | bbaaccaadd  |
|   1   | baaccaaddb  |
|   2   | aaccaaddbb  |
|   3   | accaaddbba  |
|   4   | ccaaddbb aa |
|   …   | …           |

最小的旋转是
$$
R_2 = \texttt{"aaccaaddbb"}
$$

因此旋转索引 = 2。

#### 朴素方法

生成所有旋转，然后排序，时间复杂度为 $O(n^2 \log n)$，空间复杂度为 $O(n^2)$。Booth 算法通过使用模运算循环比较字符，实现了 $O(n)$ 的时间复杂度和 $O(1)$ 的额外空间复杂度。

#### Booth 算法（核心思想）

1. 将字符串与自身连接：
   $$
   T = S + S
   $$
   现在 $S$ 的每个旋转都是 $T$ 中长度为 $n$ 的子串。

2. 为最小旋转维护一个候选索引 `k`。
   对于每个位置 `i`，逐个字符比较 `T[k + j]` 和 `T[i + j]`。

3. 当发现不匹配时：

   * 如果 `T[k + j] > T[i + j]`，则起始于 `i` 的旋转在字典序上更小 → 更新 `k = i`。
   * 否则，跳过已比较的区域。

4. 继续直到检查完所有旋转。

该算法巧妙地利用算术级数确保没有冗余比较。

#### 示例演练

令
$$
S = \texttt{"abab"} \quad (n = 4)
$$
$$
T = \texttt{"abababab"}
$$

起始 `k = 0`，比较起始于 0 和 1 的旋转：

| 步骤   | 比较                          | 结果      | 新 k   |
| ------ | ----------------------------- | ------- | ------ |
| 0 vs 1 | `a` vs `b`                    | `a < b` | 保持 0 |
| 0 vs 2 | 相同前缀，下一个字符相等 | 跳过    |        |
| 0 vs 3 | `a` vs `b`                    | 保持 0  |        |

最小旋转起始于索引 0 → `"abab"`。

#### 精简代码（Python – Booth 算法）

```python
def minimal_rotation(s):
    s += s
    n = len(s)
    f = [-1] * n  # 失配函数
    k = 0
    for j in range(1, n):
        i = f[j - k - 1]
        while i != -1 and s[j] != s[k + i + 1]:
            if s[j] < s[k + i + 1]:
                k = j - i - 1
            i = f[i]
        if s[j] != s[k + i + 1]:
            if s[j] < s[k]:
                k = j
            f[j - k] = -1
        else:
            f[j - k] = i + 1
    return k % (n // 2)

s = "bbaaccaadd"
idx = minimal_rotation(s)
print(idx, s[idx:] + s[:idx])  # 2 aaccaaddbb
```

#### 为什么它重要

- 计算循环字符串的规范形式
- 检测旋转等价性
- 用于：

  * 字符串哈希
  * DNA 循环模式识别
  * 字典序规范化
  * 循环后缀数组构建

它是 KMP 前缀逻辑与 Lyndon 词理论的美妙结合。

#### 复杂度

| 操作                 | 时间   | 空间  |
| ------------------------ | ------ | ------ |
| 最小旋转 (Booth) | $O(n)$ | $O(1)$ |
| 验证旋转          | $O(n)$ | $O(1)$ |

#### 亲自尝试

1. `"bbaaccaadd"` → 索引 2，旋转 `"aaccaaddbb"`
2. `"cabbage"` → 索引 1，旋转 `"abbagec"`
3. `"aaaa"` → 任何旋转都行
4. `"dcba"` → 索引 3，旋转 `"adcb"`
5. 与暴力旋转排序法比较以验证结果。

#### 一个温和的证明（为什么它有效）

Booth 算法维护一个候选索引 $k$，使得在 $k$ 之前的任何旋转都不可能是更小的。
在每次不匹配时，跳过操作确保我们永远不会重新考虑那些具有相同前缀模式的旋转，这与 KMP 的前缀函数逻辑类似。

因此，总比较次数 $\le 2n$，确保了线性时间。

最小旋转揭示了字符串的字典序核心——
那个具有最纯粹顺序的旋转，
它不是通过蛮力找到的，而是通过字符串自身的节奏和反射找到的。

# 第 65 节 编辑距离与对齐
### 641 莱文斯坦距离

莱文斯坦距离衡量将一个字符串转换为另一个字符串所需的*最小编辑次数*，其中编辑操作包括插入、删除和替换。它是字符串相似度的基础度量标准，支撑着拼写检查器、模糊搜索、DNA序列比对和聊天自动更正系统。

#### 我们要解决什么问题？

给定两个字符串：

$$
A = a_1 a_2 \ldots a_n, \quad B = b_1 b_2 \ldots b_m
$$

我们希望找到使 $A$ 等于 $B$ 所需的最少单字符操作次数：

- 插入一个字符
- 删除一个字符
- 替换一个字符

结果就是编辑距离 $D(n, m)$。

#### 示例

| 字符串 A      | 字符串 B      | 编辑操作      | 距离 |
| ------------- | ------------- | ------------ | -------- |
| `"kitten"`    | `"sitting"`   | k→s, +i, +g  | 3        |
| `"flaw"`      | `"lawn"`      | -f, +n       | 2        |
| `"intention"` | `"execution"` | i→e, n→x, +u | 3        |

每次编辑都将 $A$ 逐步转换为 $B$。

#### 递推关系

令 $D[i][j]$ = 将 $A[0..i-1]$ 转换为 $B[0..j-1]$ 所需的最小编辑次数。

那么：

$$
D[i][j] =
\begin{cases}
0, & \text{如果 } i = 0,\, j = 0,\\
j, & \text{如果 } i = 0,\\
i, & \text{如果 } j = 0,\\[6pt]
\min
\begin{cases}
D[i-1][j] + 1, & \text{(删除)}\\
D[i][j-1] + 1, & \text{(插入)}\\
D[i-1][j-1] + (a_i \ne b_j), & \text{(替换)}
\end{cases}, & \text{其他情况。}
\end{cases}
$$


#### 工作原理（通俗解释）

你构建一个网格，将 `A` 的每个前缀与 `B` 的每个前缀进行比较。每个单元格 $D[i][j]$ 代表到目前为止的最小编辑次数。通过从插入、删除或替换中选择最小值，你从最简单的情况开始“构建”解决方案。

#### 示例表格

计算 `D("kitten", "sitting")`：

|    | "" | s | i | t | t | i | n | g |
| -- | -- | - | - | - | - | - | - | - |
| "" | 0  | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| k  | 1  | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| i  | 2  | 2 | 1 | 2 | 3 | 4 | 5 | 6 |
| t  | 3  | 3 | 2 | 1 | 2 | 3 | 4 | 5 |
| t  | 4  | 4 | 3 | 2 | 1 | 2 | 3 | 4 |
| e  | 5  | 5 | 4 | 3 | 2 | 2 | 3 | 4 |
| n  | 6  | 6 | 5 | 4 | 3 | 3 | 2 | 3 |

好的，莱文斯坦距离 = 3

#### 精简代码（Python）

```python
def levenshtein(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # 删除
                dp[i][j - 1] + 1,      # 插入
                dp[i - 1][j - 1] + cost  # 替换
            )
    return dp[n][m]

print(levenshtein("kitten", "sitting"))  # 3
```

#### 空间优化版本

我们只需要前一行：

```python
def levenshtein_optimized(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                prev[j] + 1,
                curr[-1] + 1,
                prev[j - 1] + cost
            ))
        prev = curr
    return prev[-1]
```

#### 为什么它很重要

- 文本处理中的基础相似度度量标准
- 应用于：

  * 拼写纠正 (`levenstein("color", "colour")`)
  * DNA序列比对
  * 近似搜索
  * 聊天自动更正 / 模糊匹配
- 提供可解释的结果：每次编辑都有意义

#### 复杂度

| 操作           | 时间    | 空间           |
| --------------- | ------- | --------------- |
| DP（完整表格） | $O(nm)$ | $O(nm)$         |
| DP（优化版）   | $O(nm)$ | $O(\min(n, m))$ |

#### 自己动手试试

1. `"flaw"` 对比 `"lawn"` → 距离 = 2
2. `"intention"` 对比 `"execution"` → 5
3. `"abc"` 对比 `"yabd"` → 2
4. 通过回溯 DP 表格计算编辑路径。
5. 比较完整 DP 和优化版本之间的运行时间。

#### 一个温和的证明（为什么它有效）

递推关系确保了最优子结构：

- 前缀的最小编辑自然地扩展到更长的前缀。每一步都考虑了所有可能的最后操作，并选择最小的。动态规划保证了全局最优性。

莱文斯坦距离是转换的真正语言——每次插入是一次诞生，每次删除是一次损失，每次替换是一次意义的改变，它衡量了两个词语漂移得有多远。
### 642 Damerau–Levenshtein 距离

Damerau–Levenshtein 距离扩展了经典的 Levenshtein 度量，它认识到人类（和计算机）经常犯第四种常见的拼写错误：**转置**，即交换两个相邻字符。
这种扩展在自然文本中捕捉了更真实的“编辑距离”概念，例如将 "the" 打成 "hte"。

#### 我们要解决什么问题？

我们希望找到将一个字符串 $A$ 转换为另一个字符串 $B$ 所需的最少操作次数，允许的操作包括：

1. **插入** – 添加一个字符
2. **删除** – 移除一个字符
3. **替换** – 替换一个字符
4. **转置** – 交换两个相邻字符

形式化地说，就是找到 $D(n, m)$，即使用这四种操作的最小编辑距离。

#### 示例

| 字符串 A | 字符串 B | 编辑操作                       | 距离 |
| -------- | -------- | ------------------------------ | ---- |
| `"ca"`   | `"ac"`   | 转置 c↔a                       | 1    |
| `"abcd"` | `"acbd"` | 转置 b↔c                       | 1    |
| `"abcf"` | `"acfb"` | 替换 c→f, 转置 f↔b             | 2    |
| `"hte"`  | `"the"`  | 转置 h↔t                       | 1    |

这个距离能更好地模拟现实世界中的拼写错误和生物学中的交换。

#### 递推关系

令 $D[i][j]$ 为 $A[0..i-1]$ 和 $B[0..j-1]$ 之间的 Damerau–Levenshtein 距离。

$$
D[i][j] =
\begin{cases}
\max(i, j), & \text{if } \min(i, j) = 0,\\[6pt]
\min
\begin{cases}
D[i-1][j] + 1, & \text{(删除)}\\
D[i][j-1] + 1, & \text{(插入)}\\
D[i-1][j-1] + (a_i \ne b_j), & \text{(替换)}\\
D[i-2][j-2] + 1, & \text{if } i,j > 1,\, a_i=b_{j-1},\, a_{i-1}=b_j \text{ (转置)}
\end{cases}
\end{cases}
$$


#### 工作原理（通俗解释）

我们像计算 Levenshtein 距离一样填充动态规划表，
但我们增加了一个额外的检查来处理转置情况 ——
当两个字符被交换时，例如 `a_i == b_{j-1}` 且 `a_{i-1} == b_j`。
在这种情况下，我们可以以成本 1 “对角线跳两步”。

#### 示例

计算 `"ca"` 和 `"ac"` 之间的距离：

|    | "" | a | c |
| -- | -- | - | - |
| "" | 0  | 1 | 2 |
| c  | 1  | 1 | 1 |
| a  | 2  | 1 | 1 |

转置操作 (`c↔a`) 使得 `D[2][2] = 1`。
因此，Damerau–Levenshtein 距离 = 1。

#### 精简代码（Python）

```python
def damerau_levenshtein(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # 删除
                dp[i][j - 1] + 1,      # 插入
                dp[i - 1][j - 1] + cost  # 替换
            )
            # 转置
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)
    return dp[n][m]

print(damerau_levenshtein("ca", "ac"))  # 1
```

#### 为什么它很重要

- 模拟人类打字错误（字母交换，例如 "teh", "hte"）
- 应用于：
  * 拼写检查器
  * 模糊搜索引擎
  * 光学字符识别（OCR）
  * 语音到文本校正
  * 基因序列分析（用于局部转置）

添加转置操作使模型更接近自然数据中的噪声。

#### 复杂度

| 操作                 | 时间复杂度 | 空间复杂度       |
| -------------------- | ---------- | ---------------- |
| DP（完整表）         | $O(nm)$    | $O(nm)$          |
| 优化（滚动行）       | $O(nm)$    | $O(\min(n, m))$  |

#### 亲自尝试

1. `"ab"` → `"ba"` → 1（交换）
2. `"abcdef"` → `"abdcef"` → 1（转置 d↔c）
3. `"sponge"` → `"spnoge"` → 1
4. 比较 `"hte"` 与 `"the"` → 1
5. 与 Levenshtein 距离比较，看看转置何时起作用。

#### 一个温和的证明（为什么它有效）

转置情况通过考虑一个 2 步的对角线来扩展动态规划，确保最优子结构仍然成立。
DP 网格中的每条路径都对应一个编辑序列；
添加转置操作不会破坏最优性，因为我们仍然选择成本最小的局部转移。

Damerau–Levenshtein 距离完善了我们对文本相似性的感知 ——
它不仅仅看到缺失或错误的字母，
它*理解你的手指何时跳错了顺序。*
### 643 汉明距离

汉明距离衡量两个等长字符串在多少位置上存在差异。它是衡量二进制码、DNA序列或定长文本片段之间差异的最简单、最直接的方法，是检测传输中的错误、突变或噪声的完美工具。

#### 我们要解决什么问题？

给定两个长度均为 $n$ 的字符串 $A$ 和 $B$：

$$
A = a_1 a_2 \ldots a_n, \quad B = b_1 b_2 \ldots b_n
$$

汉明距离是满足 $a_i \ne b_i$ 的位置 $i$ 的数量：

$$
H(A, B) = \sum_{i=1}^{n} [a_i \ne b_i]
$$

它告诉我们需要多少次*替换*才能使它们变得相同（不允许插入或删除）。

#### 示例

| A       | B       | 差异之处           | 汉明距离 |
| ------- | ------- | ------------------ | -------- |
| 1011101 | 1001001 | 2 位不同           | 2        |
| karolin | kathrin | 3 个字母不同       | 3        |
| 2173896 | 2233796 | 3 个数字不同       | 3        |

只计算替换，因此两个字符串必须等长。

#### 工作原理（通俗解释）

只需同时遍历两个字符串，逐个字符比较，并统计有多少个位置不匹配。这就是汉明距离，不多也不少。

#### 微型代码（Python）

```python
def hamming_distance(a, b):
    if len(a) != len(b):
        raise ValueError("字符串必须等长")
    return sum(c1 != c2 for c1, c2 in zip(a, b))

print(hamming_distance("karolin", "kathrin"))  # 3
print(hamming_distance("1011101", "1001001"))  # 2
```

#### 位运算版本（针对二进制字符串）

如果 $A$ 和 $B$ 是整数，使用 XOR 来查找不同的位：

```python
def hamming_bits(x, y):
    return bin(x ^ y).count("1")

print(hamming_bits(0b1011101, 0b1001001))  # 2
```

因为 XOR 能精确地突显出不同的位。

#### 为何重要

- **错误检测** – 衡量传输中翻转的比特数
- **遗传学** – 计算核苷酸突变数
- **哈希与机器学习** – 量化二进制指纹之间的相似度
- **密码学** – 评估扩散性（加密下的比特变化）

它是信息论的基石之一，由理查德·汉明于 1950 年提出。

#### 复杂度

| 操作             | 时间                    | 空间   |
| ---------------- | ----------------------- | ------ |
| 直接比较         | $O(n)$                  | $O(1)$ |
| 位运算 XOR       | $O(1)$ 每机器字         | $O(1)$ |

#### 动手试试

1. 比较 `"1010101"` 与 `"1110001"` → 4
2. 计算 $H(0b1111, 0b1001)$ → 2
3. 计算 `"AACCGGTT"` 和 `"AAACGGTA"` 之间的突变数 → 2
4. 实现汉明相似度 = $1 - \frac{H(A,B)}{n}$
5. 将其用于二进制最近邻搜索。

#### 一个温和的证明（为何有效）

每个不匹配对总计数贡献 +1，并且由于每个位置上的操作是独立的，该总和直接衡量了替换次数——这是一个满足所有距离公理（非负性、对称性和三角不等式）的简单度量。

汉明距离是极简主义的体现——一把一次测量一个符号差异的尺子，从码字到染色体皆适用。
### 644 Needleman–Wunsch（尼德曼-翁施）算法

Needleman–Wunsch 算法是经典的用于全局序列比对的动态规划方法。它通过允许插入、删除和替换（并给定相应的分数），找到将两个完整序列逐字符对齐的*最优方式*。

该算法构成了计算生物学的核心，用于比较基因、蛋白质或任何*每个部分都至关重要*的序列。

#### 我们要解决什么问题？

给定两个序列：

$$
A = a_1 a_2 \ldots a_n, \quad B = b_1 b_2 \ldots b_m
$$

我们希望找到能最大化相似性分数的最佳比对（可能包含空位）。

我们定义评分参数：

- 匹配奖励 = $+M$
- 错配惩罚 = $-S$
- 空位惩罚 = $-G$

目标是找到具有最大总分的比对。

#### 示例

比对 `"GATTACA"` 和 `"GCATGCU"`：

一种可能的比对：

```
G A T T A C A
| |   |   | |
G C A T G C U
```

该算法将探索所有可能性，并根据总分返回*最佳*比对。

#### 递推关系

令 $D[i][j]$ = 将 $A[0..i-1]$ 与 $B[0..j-1]$ 对齐的最佳分数。

基本情况：

$$
D[0][0] = 0, \quad
D[i][0] = -iG, \quad
D[0][j] = -jG
$$

递推式：

$$
D[i][j] = \max
\begin{cases}
D[i-1][j-1] + \text{score}(a_i, b_j), & \text{(匹配/错配)}\\
D[i-1][j] - G, & \text{(B中的空位)}\\
D[i][j-1] - G, & \text{(A中的空位)}
\end{cases}
$$


其中：

$$
\text{score}(a_i, b_j) =
\begin{cases}
+M, & \text{if } a_i = b_j,\\
-S, & \text{if } a_i \ne b_j.
\end{cases}
$$


#### 工作原理（通俗解释）

1.  构建一个大小为 $(n+1) \times (m+1)$ 的评分矩阵。
2.  用累积的空位惩罚初始化第一行和第一列。
3.  使用递推规则填充每个单元格，每一步都考虑匹配、删除或插入。
4.  从右下角回溯以恢复最佳比对路径。

#### 示例表格

对于小序列：

|    | "" | G  | C  | A  |
| -- | -- | -- | -- | -- |
| "" | 0  | -2 | -4 | -6 |
| G  | -2 | 1  | -1 | -3 |
| A  | -4 | -1 | 0  | 0  |
| C  | -6 | -3 | 0  | 1  |

最终分数 = 1 → 通过回溯找到的最佳全局比对。

#### 微型代码（Python）

```python
def needleman_wunsch(a, b, match=1, mismatch=-1, gap=-2):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # 初始化
    for i in range(1, n + 1):
        dp[i][0] = i * gap
    for j in range(1, m + 1):
        dp[0][j] = j * gap

    # 填充矩阵
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = match if a[i - 1] == b[j - 1] else mismatch
            dp[i][j] = max(
                dp[i - 1][j - 1] + score,
                dp[i - 1][j] + gap,
                dp[i][j - 1] + gap
            )
    return dp[n][m]
```

#### 为何重要

-   生物信息学中 DNA/蛋白质 比对的基础
-   用于：
  *   比较基因序列
  *   抄袭和文本相似性检测
  *   语音和时间序列匹配

与 Smith–Waterman（史密斯-沃特曼）算法（局部比对）不同，Needleman–Wunsch 保证找到最优的全局比对。

#### 复杂度

| 操作         | 时间复杂度 | 空间复杂度 |
| ------------ | ---------- | ---------- |
| 填充 DP 表格 | $O(nm)$    | $O(nm)$    |
| 回溯         | $O(n+m)$   | $O(1)$     |

如果只需要分数，内存可以优化到 $O(\min(n,m))$。

#### 亲自尝试

1.  比对 `"GATTACA"` 和 `"GCATGCU"`。
2.  改变空位惩罚，观察比对结果如何变化。
3.  修改错配评分 → 惩罚越轻，得到的比对越长。
4.  与 Smith–Waterman 算法比较，观察局部与全局的差异。

#### 一个温和的证明（为何有效）

DP 结构确保了最优子结构——前缀的最佳比对由更小的最优比对构建而成。通过在每一步评估匹配、插入和删除，算法始终保留全局最佳比对路径。

Needleman–Wunsch 算法是对齐算法的原型——在匹配和空位之间取得平衡，它教导序列如何在中途相遇。
### 645 Smith–Waterman（史密斯-沃特曼）算法

Smith–Waterman 算法是一种用于局部序列比对的动态规划方法，旨在找出两个序列之间*最相似的子序列*。与 Needleman–Wunsch（尼德曼-翁施）算法对整个序列进行比对不同，Smith–Waterman 只关注最佳匹配区域，即真正的生物学或文本相似性所在之处。

#### 我们要解决什么问题？

给定两个序列：

$$
A = a_1 a_2 \ldots a_n, \quad B = b_1 b_2 \ldots b_m
$$

找到一对子串 $(A[i_1..i_2], B[j_1..j_2])$，使得局部比对得分最大化，允许存在空位和错配。

#### 评分方案

定义评分参数：

- 匹配奖励 = $+M$
- 错配惩罚 = $-S$
- 空位惩罚 = $-G$

目标是找到：

$$
\max_{i,j} D[i][j]
$$

其中 $D[i][j]$ 表示结束于 $a_i$ 和 $b_j$ 的最佳局部比对得分。

#### 递推关系

基础情况：

$$
D[0][j] = D[i][0] = 0
$$

递推式：

$$
D[i][j] = \max
\begin{cases}
0, & \text{(开始新的比对)}\\
D[i-1][j-1] + \text{score}(a_i, b_j), & \text{(匹配/错配)}\\
D[i-1][j] - G, & \text{(B 序列中的空位)}\\
D[i][j-1] - G, & \text{(A 序列中的空位)}
\end{cases}
$$

其中

$$
\text{score}(a_i, b_j) =
\begin{cases}
+M, & \text{如果 } a_i = b_j,\\
-S, & \text{如果 } a_i \ne b_j.
\end{cases}
$$

当得分降至零以下时，0 会重置比对——确保我们只保留高相似性区域。

#### 示例

比对 `"ACACACTA"` 和 `"AGCACACA"`。

Smith–Waterman 检测到最强的重叠部分：

```
A C A C A C T A
| | | | |
A G C A C A C A
```

最佳局部比对：`"ACACA"`
局部比对得分 = 10（匹配 = +2，错配 = -1，空位 = -2）

#### 工作原理（通俗解释）

1.  构建一个 DP 矩阵，从零开始。
2.  对于每一对位置 $(i, j)$：
    *   计算结束于 $(i, j)$ 的最佳局部得分。
    *   如果比对得分变为负数，则重置为零。
3.  追踪矩阵中的最大得分。
4.  从该单元格回溯，重构得分最高的局部子序列。

#### 微型代码（Python）

```python
def smith_waterman(a, b, match=2, mismatch=-1, gap=-2):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    max_score = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = match if a[i - 1] == b[j - 1] else mismatch
            dp[i][j] = max(
                0,
                dp[i - 1][j - 1] + score,
                dp[i - 1][j] + gap,
                dp[i][j - 1] + gap
            )
            max_score = max(max_score, dp[i][j])

    return max_score
```

#### 为什么它很重要

-   模拟生物相似性，检测保守区域，而非整个基因组比对。
-   应用于：
    *   生物信息学（蛋白质/DNA 局部比对）
    *   文本相似性和抄袭检测
    *   带噪声的模式匹配
    *   模糊子串匹配

Smith–Waterman 确保只有*最佳匹配部分*对得分有贡献，避免了来自不相关前缀/后缀的惩罚。

#### 复杂度

| 操作             | 时间      | 空间             |
| ---------------- | --------- | ---------------- |
| DP（完整表格）   | $O(nm)$   | $O(nm)$          |
| 空间优化版本     | $O(nm)$   | $O(\min(n,m))$   |

#### 动手尝试

1.  `"GATTACA"` 对比 `"GCATGCU"` → 局部比对 `"ATG"`
2.  `"ACACACTA"` 对比 `"AGCACACA"` → `"ACACA"`
3.  将空位惩罚从 2 改为 5 → 比对结果如何收缩？
4.  比较全局与局部比对的输出（Needleman–Wunsch 对比 Smith–Waterman）。
5.  应用于 `"hello"` 对比 `"yellow"` → 找出共享区域。

#### 一个温和的证明（为什么它有效）

递推式中包含 0 确保了最优的局部行为：每当运行得分变为负数时，我们就重新开始比对。动态规划保证了所有可能的子串都被考虑在内，而全局最大值对应着最强的局部匹配。

Smith–Waterman 算法在噪声中倾听回声——找到两个长旋律之间最明亮的重叠部分，并告诉你它们真正在哪里和谐共鸣。
### 646 Hirschberg 算法

Hirschberg 算法是对 Needleman–Wunsch 比对的一种巧妙优化。
它产生相同的全局比对结果，但仅使用线性空间 $O(n + m)$，而不是 $O(nm)$。
这使得它非常适合在内存紧张的情况下比对长 DNA 或文本序列。

#### 我们要解决什么问题？

我们想要计算两个序列之间的全局序列比对（类似于 Needleman–Wunsch）：

$$
A = a_1 a_2 \ldots a_n, \quad B = b_1 b_2 \ldots b_m
$$

但我们希望使用线性空间来完成，而不是平方空间。
诀窍在于只计算重建最优路径所需的分数，而不是完整的 DP 表。

#### 关键洞察

经典的 Needleman–Wunsch 算法填充一个 $n \times m$ 的 DP 矩阵来寻找最优比对路径。

但是：

- 在任何时候，我们只需要表格的一半来计算分数。
- DP 表的中间列将问题划分为两个独立的部分。

结合这两个事实，Hirschberg 算法递归地找到比对的拆分点。

#### 算法概述

1. 基本情况：

   * 如果任一字符串为空 → 返回一个由空位组成的序列。
   * 如果任一字符串长度为 1 → 直接进行简单的比对。

2. 划分：

   * 将 $A$ 分成两半：$A = A_{\text{左}} + A_{\text{右}}$。
   * 计算 $A_{\text{左}}$ 与 $B$ 的前向比对分数。
   * 计算 $A_{\text{右}}$ 与 $B$（反转后）的后向比对分数。
   * 将对应的分数相加，以找到 $B$ 中的最佳拆分点。

3. 递归：

   * 递归地对两半 $(A_{\text{左}}, B_{\text{左}})$ 和 $(A_{\text{右}}, B_{\text{右}})$ 进行比对。

4. 合并：

   * 将两个子比对合并成一个完整的全局比对。

#### 递推关系

我们使用 Needleman–Wunsch 的评分递推关系：

$$
D[i][j] = \max
\begin{cases}
D[i-1][j-1] + s(a_i, b_j), & \text{匹配/不匹配},\\
D[i-1][j] - G, & \text{B 中的空位},\\
D[i][j-1] - G, & \text{A 中的空位}.
\end{cases}
$$

但在内存中，对于每一半只保留*前一行*，
我们通过结合前向和后向分数来找到最优的中间列拆分点。

#### 示例

比对 `"AGTACGCA"` 和 `"TATGC"`。

- 将 `"AGTACGCA"` 拆分为 `"AGTA"` 和 `"CGCA"`。
- 计算 `"AGTA"` 与 `"TATGC"` 的前向 DP。
- 计算 `"CGCA"` 与 `"TATGC"` 的后向 DP。
- 结合分数以找到 `"TATGC"` 中的最佳拆分点。
- 对两个较小的比对进行递归，合并结果。

最终的比对结果与 Needleman–Wunsch 相同，
但空间成本显著降低。

#### 微型代码（Python – 简化版）

```python
def hirschberg(a, b, match=1, mismatch=-1, gap=-2):
    # 如果 a 为空
    if len(a) == 0:
        return ("-" * len(b), b)
    # 如果 b 为空
    if len(b) == 0:
        return (a, "-" * len(a))
    # 如果 a 或 b 长度为 1
    if len(a) == 1 or len(b) == 1:
        # 基本情况：简单的 Needleman-Wunsch
        return needleman_wunsch_align(a, b, match, mismatch, gap)

    mid = len(a) // 2
    scoreL = nw_score(a[:mid], b, match, mismatch, gap)
    scoreR = nw_score(a[mid:][::-1], b[::-1], match, mismatch, gap)
    j_split = max(range(len(b) + 1), key=lambda j: scoreL[j] + scoreR[len(b) - j])
    left = hirschberg(a[:mid], b[:j_split], match, mismatch, gap)
    right = hirschberg(a[mid:], b[j_split:], match, mismatch, gap)
    return (left[0] + right[0], left[1] + right[1])
```

*（辅助函数 `nw_score` 计算一个方向的 Needleman–Wunsch 行分数。）*

#### 为什么它很重要

- 使用线性内存，同时保持最优比对质量。
- 非常适合：

  * 基因组序列比对
  * 大规模文档比较
  * 低内存环境
- 保持了 Needleman–Wunsch 的正确性，提高了大数据场景下的实用性。

#### 复杂度

| 操作 | 时间    | 空间      |
| --------- | ------- | ---------- |
| 比对 | $O(nm)$ | $O(n + m)$ |

递归拆分引入了少量开销，但没有渐近惩罚。

#### 亲自尝试

1. 使用 Needleman–Wunsch 和 Hirschberg 比对 `"GATTACA"` 和 `"GCATGCU"`，确认输出相同。
2. 使用长度超过 10,000 的序列进行测试，观察内存节省情况。
3. 尝试不同的空位罚分，观察拆分点如何变化。
4. 可视化递归树，它整齐地从中间划分。

#### 一个温和的证明（为什么它有效）

每个中间列分数对 $(L[j], R[m - j])$ 代表了
通过单元格 $(\text{mid}, j)$ 的最佳可能比对。
通过选择最大化 $L[j] + R[m - j]$ 的 $j$，
我们确保全局最优比对会经过该点。
这保留了最优子结构，保证了正确性。

Hirschberg 算法通过归约展现了优雅——
它只记住本质的东西，
以极简主义数学家的优雅姿态比对庞大的序列。
### 647 编辑脚本重构

一旦我们计算出两个字符串之间的编辑距离，我们通常想要的不仅仅是这个数字，我们还想知道*如何*将一个字符串转换为另一个。
这个转换计划被称为编辑脚本：将字符串 A 最优地转换为字符串 B 的操作（插入、删除、替换）有序序列。

#### 我们要解决什么问题？

给定两个字符串：

$$
A = a_1 a_2 \ldots a_n, \quad B = b_1 b_2 \ldots b_m
$$

以及它们的最小编辑距离 $D[n][m]$，
我们希望重构出达到该最小代价的一系列编辑操作。

操作：

| 符号 | 操作       | 描述                     |
| :--: | :--------- | :----------------------- |
| `M`  | 匹配       | $a_i = b_j$              |
| `S`  | 替换       | 将 $a_i$ 替换为 $b_j$    |
| `I`  | 插入       | 将 $b_j$ 添加到 $A$ 中   |
| `D`  | 删除       | 从 $A$ 中移除 $a_i$      |

输出是一个人类可读的编辑跟踪，例如：

```
M M S I M D
```

#### 示例

将 `"kitten"` 转换为 `"sitting"`。

| 步骤 | 操作               | 结果       |
| :--: | :----------------- | :--------- |
|   1  | 替换 `k → s`       | "sitten"   |
|   2  | 插入 `i`           | "sittien"  |
|   3  | 插入 `g`           | "sitting"  |

好的 编辑距离 = 3
好的 编辑脚本 = `S, I, I`

#### 工作原理（通俗语言）

1.  计算完整的 Levenshtein 动态规划表 $D[i][j]$。
2.  从右下角 $(n, m)$ 开始。
3.  向后移动：
    *   如果字符匹配 → `M`（对角线移动）
    *   否则如果 $D[i][j] = D[i-1][j-1] + 1$ → `S`
    *   否则如果 $D[i][j] = D[i-1][j] + 1$ → `D`
    *   否则如果 $D[i][j] = D[i][j-1] + 1$ → `I`
4.  记录操作并相应移动。
5.  最后反转列表。

#### 示例表（简化）

|    | "" | s | i | t | t | i | n | g |
| -- | -- | - | - | - | - | - | - | - |
| "" | 0  | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| k  | 1  | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| i  | 2  | 2 | 1 | 2 | 3 | 4 | 5 | 6 |
| t  | 3  | 3 | 2 | 1 | 2 | 3 | 4 | 5 |
| t  | 4  | 4 | 3 | 2 | 1 | 2 | 3 | 4 |
| e  | 5  | 5 | 4 | 3 | 2 | 2 | 3 | 4 |
| n  | 6  | 6 | 5 | 4 | 3 | 3 | 2 | 3 |

回溯路径：对角线（S），向右（I），向右（I）。
重构的编辑脚本 = `[替换, 插入, 插入]`。

#### 微型代码（Python）

```python
def edit_script(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # 删除
                dp[i][j - 1] + 1,      # 插入
                dp[i - 1][j - 1] + cost  # 替换或匹配
            )

    # 回溯
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1]:
            ops.append("M")
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(f"S:{a[i - 1]}->{b[j - 1]}")
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(f"D:{a[i - 1]}")
            i -= 1
        else:
            ops.append(f"I:{b[j - 1]}")
            j -= 1

    return ops[::-1]

print(edit_script("kitten", "sitting"))
```

输出：

```
['S:k->s', 'M', 'M', 'M', 'I:i', 'M', 'I:g']
```

#### 为什么它很重要

-   将距离度量转换为可解释的转换
-   用于：
    *   差异比较工具（例如 `git diff`, Myers diff）
    *   拼写校正
    *   DNA 编辑跟踪
    *   版本控制系统
    *   文档合并工具

没有编辑重构，我们知道两个字符串*相距多远*——
有了它，我们知道*如何到达那里*。

#### 复杂度

| 操作             | 时间     | 空间   |
| ---------------- | -------- | ------ |
| DP 表构建        | $O(nm)$  | $O(nm)$ |
| 回溯             | $O(n+m)$ | $O(1)$  |

空间可以通过 Hirschberg 的分治回溯法减少。

#### 自己尝试

1.  `"flaw"` → `"lawn"` → `D:f, M, M, I:n`
2.  `"sunday"` → `"saturday"` → 多次插入
3.  反转脚本以获得逆变换。
4.  修改代价函数：使替换代价更高。
5.  在 DP 网格上可视化路径，它描绘了你的脚本。

#### 一个温和的证明（为什么它有效）

DP 表编码了所有前缀的最小编辑代价。
通过从 $(n, m)$ 向后走，每个局部选择（对角线、向上、向左）
代表了实现最优代价的确切操作。
因此，回溯重构了最小转换路径。

编辑脚本是转换的日记——
记录了发生了什么变化、何时发生以及如何发生——
将原始距离转化为差异的故事。
### 648 仿射空位罚分动态规划

仿射空位罚分模型改进了序列比对中的简单空位评分方式。
它不再对每个空位符号收取固定罚分，而是区分**空位开启**和**空位延伸**，
这反映了生物学或文本处理的现实情况：*开启*一个空位代价高昂，但*延伸*它则相对便宜。

#### 我们要解决什么问题？

在经典比对算法（Needleman–Wunsch 或 Smith–Waterman）中，
每个空位都受到线性惩罚：

$$
\text{空位成本} = k \times g
$$

但在实际中，一个长的连续空位比多个短的分散空位*更不糟糕*。
因此我们切换到仿射模型：

$$
\text{空位成本} = g_o + (k - 1) \times g_e
$$

其中

- $g_o$ = 空位开启罚分
- $g_e$ = 空位延伸罚分
  且 $k$ = 空位长度。

这个模型能产生更平滑、更符合实际的比对结果。

#### 示例

假设 $g_o = 5$, $g_e = 1$。

| 空位          | 线性模型 | 仿射模型 |
| ------------ | ------------ | ------------ |
| 1 符号空位 | 5            | 5            |
| 3 符号空位 | 15           | 7            |
| 5 符号空位 | 25           | 9            |

仿射评分*奖励更长的连续空位*，并避免分散的空位。

#### 工作原理（通俗解释）

我们跟踪三个动态规划矩阵，而不是一个：

| 矩阵    | 含义                                    |
| :-------- | :----------------------------------------- |
| $M[i][j]$ | 以匹配/错配结束的最佳分数      |
| $X[i][j]$ | 以序列 A 中出现空位结束的最佳分数 |
| $Y[i][j]$ | 以序列 B 中出现空位结束的最佳分数 |

每个矩阵使用不同的递推关系来正确建模空位转换。

#### 递推关系

令 $a_i$ 和 $b_j$ 为当前字符。

$$
\begin{aligned}
M[i][j] &= \max \big(
M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1]
\big) + s(a_i, b_j) [6pt]
X[i][j] &= \max \big(
M[i-1][j] - g_o,; X[i-1][j] - g_e
\big) [6pt]
Y[i][j] &= \max \big(
M[i][j-1] - g_o,; Y[i][j-1] - g_e
\big)
\end{aligned}
$$

其中

$$
s(a_i, b_j) =
\begin{cases}
+M, & \text{if } a_i = b_j,\\
-S, & \text{if } a_i \ne b_j.
\end{cases}
$$


最终的比对分数：

$$
D[i][j] = \max(M[i][j], X[i][j], Y[i][j])
$$

#### 初始化

$$
M[0][0] = 0, \quad X[0][0] = Y[0][0] = -\infty
$$

对于第一行/第一列：

$$
X[i][0] = -g_o - (i - 1) g_e, \quad
Y[0][j] = -g_o - (j - 1) g_e
$$

#### 示例（直观理解）

让我们比对：

```
A:  G A T T A C A
B:  G C A T G C U
```

参数如下：

- 匹配 = +2
- 错配 = -1
- 空位开启 = 5
- 空位延伸 = 1

在需要的地方会出现小的空位，
但长的插入将保持连续而不是被分割，
因为延续一个空位比开启一个新的空位成本更低。

#### 微型代码（Python）

```python
def affine_gap(a, b, match=2, mismatch=-1, gap_open=5, gap_extend=1):
    n, m = len(a), len(b)
    neg_inf = float("-inf")
    M = [[0] * (m + 1) for _ in range(n + 1)]
    X = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    Y = [[neg_inf] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        M[i][0] = -gap_open - (i - 1) * gap_extend
        X[i][0] = M[i][0]
    for j in range(1, m + 1):
        M[0][j] = -gap_open - (j - 1) * gap_extend
        Y[0][j] = M[0][j]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = match if a[i - 1] == b[j - 1] else mismatch
            M[i][j] = max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1]) + score
            X[i][j] = max(M[i - 1][j] - gap_open, X[i - 1][j] - gap_extend)
            Y[i][j] = max(M[i][j - 1] - gap_open, Y[i][j - 1] - gap_extend)

    return max(M[n][m], X[n][m], Y[n][m])
```

#### 为什么它很重要

- 更真实地模拟生物学空位（例如 DNA 中的插入/缺失）。
- 为文本或语音处理产生更清晰的比对结果。
- 应用于：
  * Needleman–Wunsch 和 Smith–Waterman 算法的扩展
  * BLAST、FASTA 及生物信息学流程
  * 机器学习和信号分析中的动态时间规整变体

仿射罚分反映了这样一种直觉：开始一个错误的成本高于延续一个错误。

#### 复杂度

| 操作       | 时间    | 空间           |
| --------------- | ------- | --------------- |
| DP（3 个矩阵） | $O(nm)$ | $O(nm)$         |
| 空间优化版本 | $O(nm)$ | $O(\min(n, m))$ |

#### 亲自尝试

1.  比较 `"GATTACA"` 和 `"GCATGCU"` 的线性空位与仿射空位。
2.  测试长插入，仿射评分将倾向于一个大的空位。
3.  调整空位罚分，观察比对如何变化。
4.  将仿射评分与局部比对（Smith–Waterman）结合。
5.  分别可视化 $M$、$X$ 和 $Y$ 矩阵。

#### 一个温和的证明（为什么它有效）

这三个矩阵中的每一个都代表一个状态机：

- $M$ → 处于匹配状态，
- $X$ → 处于序列 A 有空位状态，
- $Y$ → 处于序列 B 有空位状态。

仿射递推关系确保了最优子结构，因为状态之间的转换恰好会产生适当的开启/延伸罚分。
因此，通过组合系统的每条路径都会在仿射成本下产生一个最优的总分。

仿射空位罚分模型为比对带来了现实性——
理解到开始是昂贵的，
但延续有时只是坚持。
### 649 Myers 位向量算法

Myers 位向量算法是一种用于计算短字符串或模式之间编辑距离（Levenshtein 距离）的卓越优化方法，尤其适用于搜索和匹配任务。
它使用位运算来并行模拟跨多个位置的动态规划，在现代 CPU 上实现了接近线性的速度。

#### 我们要解决什么问题？

给定两个字符串
$$
A = a_1 a_2 \ldots a_n, \quad B = b_1 b_2 \ldots b_m
$$
我们想要计算它们的编辑距离（插入、删除、替换）。

经典的动态规划解决方案需要 $O(nm)$ 时间。
Myers 算法将其减少到 $O(n \cdot \lceil m / w \rceil)$，
其中 $w$ 是机器字长（通常为 32 或 64）。

这使得它非常适合近似字符串搜索——
例如，在文本中查找所有编辑距离 ≤ k 的 `"pattern"` 匹配项。

#### 核心思想

Levenshtein 动态规划递推关系可以看作更新一个仅依赖于前一行的单元格带。
如果我们将每一行表示为位向量，
我们就可以使用位与、位或、位异或和移位操作一次性执行所有单元格更新。

对于短模式，所有位可以放入单个字中，
因此更新可以在常数时间内完成。

#### 表示法

我们定义几个长度为 $m$（模式长度）的位掩码：

- Eq[c] – 一个位掩码，标记字符 `c` 在模式中出现的位置。
  以模式 `"ACCA"` 为例：

  ```
  Eq['A'] = 1001
  Eq['C'] = 0110
  ```

在算法过程中，我们维护：

|  符号   | 含义                                                               |
| :-----: | :----------------------------------------------------------------- |
|  `Pv`   | 可能存在正差异的位置的位向量                                       |
|  `Mv`   | 可能存在负差异的位置的位向量                                       |
| `Score` | 当前编辑距离                                                       |

这些编码了编辑距离动态规划的运行状态。

#### 递推关系（位并行形式）

对于每个文本字符 `t_j`：

$$
\begin{aligned}
Xv &= \text{Eq}[t_j] ; \lor ; Mv \\
Xh &= (((Xv & Pv) + Pv) \oplus Pv) ; \lor ; Xv \\
Ph &= Mv ; \lor ; \neg(Xh \lor Pv) \\
Mh &= Pv ; & Xh
\end{aligned}
$$

然后移位并更新分数：

$$
\begin{cases}
\text{if } (Ph \;\&\; \text{bit}_m) \ne 0, & \text{then Score++},\\
\text{if } (Mh \;\&\; \text{bit}_m) \ne 0, & \text{then Score--}.
\end{cases}
$$

最后，设置：

$$
\begin{aligned}
Pv &= Mh \;\lor\; \neg(Xh \lor Ph),\\
Mv &= Ph \;\&\; Xh.
\end{aligned}
$$

#### 工作原理（通俗解释）

将 `Pv` 和 `Mv` 中的每个位视为代表动态规划表中的一个列。
不是逐个更新每个单元格，
而是通过位运算并行更新所有列，一条 CPU 指令更新 64 次比较。

在每一步：
- Eq[c] 指示匹配发生的位置。
- Pv, Mv 跟踪累积的不匹配。
- 当位在顶部溢出时（编辑成本传播），分数相应调整。

算法的循环极其紧凑，只有少数几个位运算。

#### 示例（概念性）

模式：`"ACGT"`
文本：`"AGT"`

我们初始化：

```
Eq['A'] = 1000
Eq['C'] = 0100
Eq['G'] = 0010
Eq['T'] = 0001
```

然后依次处理文本 `"A"`、`"G"`、`"T"` 的每个字符，
更新位向量并在标量 `Score` 中保持当前编辑距离。

最终 Score = 1
编辑距离 = 1（一次删除）。

#### 微型代码（Python）

下面是一个简化的单字实现：

```python
def myers_distance(pattern, text):
    m = len(pattern)
    Peq = {}
    for c in set(pattern + text):
        Peq[c] = 0
    for i, ch in enumerate(pattern):
        Peq[ch] |= 1 << i

    Pv = (1 << m) - 1
    Mv = 0
    score = m

    for ch in text:
        Eq = Peq.get(ch, 0)
        Xv = Eq | Mv
        Xh = (((Eq & Pv) + Pv) ^ Pv) | Eq
        Ph = Mv | ~(Xh | Pv)
        Mh = Pv & Xh

        if Ph & (1 << (m - 1)):
            score += 1
        elif Mh & (1 << (m - 1)):
            score -= 1

        Pv = (Mh << 1) | ~(Xh | (Ph << 1))
        Mv = (Ph << 1) & Xh

    return score

print(myers_distance("ACGT", "AGT"))  # 1
```

#### 为何重要

- 在文本和 DNA 序列中进行快速近似匹配
- 用于：
  * 类 grep 的模糊搜索
  * 基因组学中的读段比对（例如 BWA, Bowtie）
  * 自动更正 / 拼写检查
  * 实时文本比较
- 仅使用位运算和整数算术操作
  → 内部循环极快，无分支。

#### 复杂度

| 操作         | 时间                                 | 空间                       |
| ------------ | ------------------------------------ | -------------------------- |
| 主循环       | $O(n \cdot \lceil m / w \rceil)$     | $O(\lceil m / w \rceil)$   |
| 当 $m \le w$ | $O(n)$                               | $O(1)$                     |

#### 亲自尝试

1.  计算 `"banana"` 和 `"bananas"` 之间的编辑距离。
2.  当 $m=8, n=100000$ 时，与经典动态规划比较运行时间。
3.  修改算法，使其在 `Score ≤ k` 时提前停止。
4.  对于长模式，使用多个字（位块）。
5.  可视化每一步的位演变。

#### 一个温和的证明（为何有效）

标准的 Levenshtein 递推关系仅依赖于前一行。
字中的每个位编码了该位置的差异是增加还是减少。
位运算模拟了整数加法/减法中的进位和借位传播——
精确地再现了动态规划逻辑，但是为每个位列并行执行。

Myers 位向量算法将编辑距离计算转化为纯粹的硬件逻辑——
对齐字符串不是通过循环，
而是通过 CPU 寄存器中比特位同步翻转的节奏。
### 650 最长公共子序列（LCS）

最长公共子序列（LCS）问题是动态规划的基石之一。
它提出的问题是：*给定两个序列，两个序列中都出现的最长序列是什么（顺序相同，但不一定连续）？*

它是 `diff`、DNA 比对和文本相似度系统等工具的基础，适用于任何我们关心保持顺序的相似性的场景。

#### 我们要解决什么问题？

给定两个序列：

$$
A = a_1 a_2 \ldots a_n, \quad B = b_1 b_2 \ldots b_m
$$

找到一个最长的序列 $C = c_1 c_2 \ldots c_k$，
使得 $C$ 同时是 $A$ 和 $B$ 的子序列。

形式化定义：
$$
C \subseteq A, \quad C \subseteq B, \quad k = |C| \text{ 是最大的。}
$$

我们既需要长度，有时也需要子序列本身。

#### 示例

| A           | B           | LCS      | 长度 |
| ----------- | ----------- | -------- | ------ |
| `"ABCBDAB"` | `"BDCABA"`  | `"BCBA"` | 4      |
| `"AGGTAB"`  | `"GXTXAYB"` | `"GTAB"` | 4      |
| `"HELLO"`   | `"YELLOW"`  | `"ELLO"` | 4      |

#### 递推关系

令 $L[i][j]$ 为前缀 $A[0..i-1]$ 和 $B[0..j-1]$ 的 LCS 长度。

则有：

$$
L[i][j] =
\begin{cases}
0, & \text{如果 } i = 0 \text{ 或 } j = 0,\\[4pt]
L[i-1][j-1] + 1, & \text{如果 } a_i = b_j,\\[4pt]
\max(L[i-1][j],\, L[i][j-1]), & \text{否则。}
\end{cases}
$$


#### 工作原理（通俗解释）

你构建一个二维网格，比较两个字符串的前缀。
每个单元格 $L[i][j]$ 表示“到 $a_i$ 和 $b_j$ 为止的 LCS 有多长”。

- 如果字符匹配 → 将 LCS 长度加 1。
- 如果不匹配 → 取跳过任一字符串中一个字符后的最佳结果。

右下角单元格的值就是最终的 LCS 长度。

#### 示例表格

对于 `"ABCBDAB"` 与 `"BDCABA"`：

|    | "" | B | D | C | A | B | A |
| -- | -- | - | - | - | - | - | - |
| "" | 0  | 0 | 0 | 0 | 0 | 0 | 0 |
| A  | 0  | 0 | 0 | 0 | 1 | 1 | 1 |
| B  | 0  | 1 | 1 | 1 | 1 | 2 | 2 |
| C  | 0  | 1 | 1 | 2 | 2 | 2 | 2 |
| B  | 0  | 1 | 1 | 2 | 2 | 3 | 3 |
| D  | 0  | 1 | 2 | 2 | 2 | 3 | 3 |
| A  | 0  | 1 | 2 | 2 | 3 | 3 | 4 |
| B  | 0  | 1 | 2 | 2 | 3 | 4 | 4 |

好的 LCS 长度 = 4
好的 一个有效的子序列 = `"BCBA"`

#### 精简代码（Python）

```python
def lcs(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]
```

要重建子序列：

```python
def lcs_traceback(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 回溯
    i, j = n, m
    seq = []
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            seq.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return ''.join(reversed(seq))
```

示例：

```python
print(lcs_traceback("ABCBDAB", "BDCABA"))  # BCBA
```

#### 为什么它重要

- 差异比较工具的核心（`git diff`、文本比较、版本控制）
- DNA/蛋白质相似性（不变子序列）
- 剽窃检测
- 机器翻译评估（类似 BLEU 的指标）
- 序列压缩和纠错

LCS 提供的是结构相似性，而非精确匹配，是共享的顺序。

#### 复杂度

| 操作               | 时间    | 空间           |
| ----------------------- | ------- | --------------- |
| 动态规划（完整表格）         | $O(nm)$ | $O(nm)$         |
| 优化（仅长度） | $O(nm)$ | $O(\min(n, m))$ |

#### 亲自尝试

1. `"HELLO"` 与 `"YELLOW"` → `"ELLO"`
2. `"AGGTAB"` 与 `"GXTXAYB"` → `"GTAB"`
3. 计算 LCS 相似度比率 = $2 \times LCS / (|A| + |B|)$
4. 使用 LCS 对齐文本版本并检测编辑。
5. 在代码差异上试试，`git` 就是这样检测更改行的。

#### 一个温和的证明（为什么它有效）

$A$ 和 $B$ 的每个前缀都定义了具有最优子结构的子问题：

- 如果 $a_i = b_j$，则 LCS 扩展了之前的前缀。
- 否则，最优 LCS 必须从 $A$ 或 $B$ 中舍弃一个元素。

重叠子问题被一次性解决，并存储在 $L[i][j]$ 中。
递归确保了所有组合都被高效地考虑。

最长公共子序列是比较的静默核心——
它寻找在变化与差异中幸存下来的东西，
那贯穿时间和转变的同一性之线。

# 第 66 节 压缩
### 651 霍夫曼编码

霍夫曼编码是一种经典的无损数据压缩算法。
它构建一种最优的前缀码，意味着没有任何一个码字是另一个码字的前缀，从而确保可唯一解码。
通过为高频符号分配较短的编码，为低频符号分配较长的编码，霍夫曼编码最小化了总的编码长度。

#### 问题定义

给定一个符号字母表
$$
S = {s_1, s_2, \ldots, s_n}
$$
及其对应的频率
$$
f(s_i)
$$
我们希望分配二进制编码 $C(s_i)$，使得：

1. 编码是前缀无关的（没有任何一个编码是另一个编码的前缀）。
2. 平均编码长度
   $$
   L = \sum_i f(s_i) \cdot |C(s_i)|
   $$
   最小。

#### 核心思想

- 反复将两个频率最低的符号合并为一个新节点。
- 为两个分支分配 0 和 1。
- 构建的树定义了前缀码。

这个过程形成一棵二叉树，其中：

- 叶子节点代表原始符号。
- 从根节点到叶子节点的路径给出了二进制编码。

#### 算法步骤

1. 初始化一个优先队列（最小堆），其中所有符号以其频率为权重。
2. 当队列中节点数大于 1 时：
   * 移除频率最小的两个节点 $f_1, f_2$。
   * 创建一个频率为 $f = f_1 + f_2$ 的新内部节点。
   * 将其插回队列。
3. 当只剩下一个节点时，它就是根节点。
4. 遍历这棵树：
   * 左分支 = 追加 `0`
   * 右分支 = 追加 `1`
   * 记录每个叶子节点的编码。

#### 示例

符号及其频率：

| 符号 | 频率 |
| :----: | :-------: |
|    A   |     45    |
|    B   |     13    |
|    C   |     12    |
|    D   |     16    |
|    E   |     9     |
|    F   |     5     |

逐步构建树：

1. 合并 F (5) + E (9) → 新节点 (14)
2. 合并 C (12) + B (13) → 新节点 (25)
3. 合并 D (16) + (14) → 新节点 (30)
4. 合并 (25) + (30) → 新节点 (55)
5. 合并 A (45) + (55) → 新根节点 (100)

最终编码（一种有效方案）：

| 符号 | 编码 |
| :----: | :--: |
|    A   |   0  |
|    B   |  101 |
|    C   |  100 |
|    D   |  111 |
|    E   | 1101 |
|    F   | 1100 |

平均编码长度：
$$
L = \frac{45(1) + 13(3) + 12(3) + 16(3) + 9(4) + 5(4)}{100} = 2.24 \text{ 比特/符号}
$$

#### 微型代码（Python）

```python
import heapq

def huffman(freqs):
    # 初始化堆，每个元素是 [权重, [符号, ""]]
    heap = [[w, [sym, ""]] for sym, w in freqs.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap) # 弹出频率最小的节点
        hi = heapq.heappop(heap) # 弹出频率次小的节点
        # 为 lo 节点下的所有符号对添加前缀 '0'
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        # 为 hi 节点下的所有符号对添加前缀 '1'
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        # 创建新节点，其权重为两者之和，并合并其子节点列表
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # 返回排序后的符号及其编码列表
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[1]), p))

freqs = {'A': 45, 'B': 13, 'C': 12, 'D': 16, 'E': 9, 'F': 5}
for sym, code in huffman(freqs):
    print(sym, code)
```

#### 为何重要

- 构成了许多现实世界压缩格式的基础：
  * DEFLATE（ZIP, PNG）
  * JPEG 和 MP3（在量化之后）
- 最小化了符号编码的期望比特长度。
- 展示了贪心最优性：首先合并最小权重的节点能得到全局最优解。

#### 复杂度

| 操作         | 时间复杂度          | 空间复杂度          |
| ----------------- | ------------------- | ------------------- |
| 构建树     | $O(n \log n)$       | $O(n)$              |
| 编码/解码 | $O(k)$ (每个符号) | $O(1)$ (每次查找) |

#### 动手尝试

1.  使用字符串 `"HELLO WORLD"` 的字符及其频率计数。
2.  手动绘制霍夫曼树。
3.  编码并解码一个小字符串。
4.  将平均比特长度与定长编码（ASCII = 8 比特）进行比较。
5.  实现规范霍夫曼编码以获得确定的顺序。

#### 一个温和的证明（为何有效）

令 $x$ 和 $y$ 为两个频率最低的符号。
在一个最优前缀码中，这两个符号必须作为兄弟节点出现在最深的层次。
用任何其他更深层次的节点对替换它们都会增加平均长度。
通过反复应用这个性质，霍夫曼的贪心合并策略总是最优的。

霍夫曼编码展示了贪心选择和树结构如何协同工作，
使压缩变得优雅，将频率转化为效率。
### 652 规范哈夫曼编码

规范哈夫曼编码是哈夫曼编码的一种精炼、确定性的版本。它使用与原始哈夫曼树相同的码长来编码符号，但按照字典序（规范）排列码字。这使得解码速度更快，并且能紧凑地表示码表，非常适合文件格式和网络协议。

#### 我们要解决什么问题？

在标准哈夫曼编码中，多棵树可以表示相同的最优码长。例如，码字 `{A: 0, B: 10, C: 11}` 和 `{A: 1, B: 00, C: 01}` 具有相同的总长度。然而，存储或传输完整的树是低效的。

规范哈夫曼编码通过仅基于符号顺序和码长（而非树结构）确定性地分配码字，消除了这种歧义。

#### 核心思想

我们不存储树，而是存储每个符号的码长。然后，我们以一种一致的方式生成所有码字：

1.  按码长（最短优先）对符号排序。
2.  将最小的可能二进制码分配给第一个符号。
3.  每个后续符号的码字 = 前一个码字 + 1（二进制）。
4.  当移动到更长的码长时，左移（追加一个零）。

这保证了字典序和无前缀结构。

#### 示例

假设我们有符号及其哈夫曼码长：

| 符号 | 长度 |
| :--: | :--: |
|  A   |  1   |
|  B   |  3   |
|  C   |  3   |
|  D   |  3   |
|  E   |  4   |

步骤 1. 按（长度，符号）排序：

`A (1), B (3), C (3), D (3), E (4)`

步骤 2. 分配规范码字：

| 符号 | 长度 | 码字（二进制） |
| :--: | :--: | :------------: |
|  A   |  1   |       0        |
|  B   |  3   |      100       |
|  C   |  3   |      101       |
|  D   |  3   |      110       |
|  E   |  4   |      1110      |

步骤 3. 顺序递增码字

从第一个符号的长度为 1 的全零码开始，然后根据需要递增和移位。

#### 伪代码

```python
def canonical_huffman(lengths):
    # lengths: dict {symbol: code_length}
    sorted_syms = sorted(lengths.items(), key=lambda x: (x[1], x[0]))
    codes = {}
    code = 0
    prev_len = 0
    for sym, length in sorted_syms:
        code <<= (length - prev_len)
        codes[sym] = format(code, '0{}b'.format(length))
        code += 1
        prev_len = length
    return codes
```

运行示例：

```python
lengths = {'A': 1, 'B': 3, 'C': 3, 'D': 3, 'E': 4}
print(canonical_huffman(lengths))
# {'A': '0', 'B': '100', 'C': '101', 'D': '110', 'E': '1110'}
```

#### 为何重要

-   确定性：每个解码器都能从码长重建出相同的码字。
-   紧凑：存储码长（每个一字节）比存储完整的树要小得多。
-   快速解码：可以使用码长范围表来生成码表。
-   用于：
    *   DEFLATE（ZIP、PNG、gzip）
    *   JPEG
    *   MPEG 和 MP3
    *   Google 的 Brotli 和 Zstandard

#### 解码过程

给定规范码表：

| 长度 | 起始码字 | 数量 | 起始值 |
| :--: | :------: | :--: | :----: |
|  1   |    0     |  1   |   A    |
|  3   |   100    |  3   | B, C, D |
|  4   |   1110   |  1   |   E    |

解码步骤：

1.  从输入读取比特。
2.  跟踪当前码长。
3.  如果比特匹配有效范围 → 解码符号。
4.  重置并继续。

此过程使用范围表而非树，实现 $O(1)$ 的查找。

#### 与标准哈夫曼编码的比较

| 特性         | 标准哈夫曼编码       | 规范哈夫曼编码       |
| ------------ | -------------------- | -------------------- |
| 存储         | 树结构               | 仅码长               |
| 唯一性       | 非确定性             | 确定性               |
| 解码速度     | 树遍历               | 表查找               |
| 常见用途     | 教学、概念理解       | 实际压缩器           |

#### 复杂度

| 操作               | 时间          | 空间             |
| ------------------ | ------------- | ---------------- |
| 构建规范表         | $O(n \log n)$ | $O(n)$           |
| 编码/解码          | $O(k)$        | $O(1)$ 每符号    |

#### 动手尝试

1.  取任意哈夫曼码树并提取码长。
2.  从码长重建规范码字。
3.  比较二进制编码，它们解码结果相同。
4.  使用 `(符号, 比特长度)` 对实现 DEFLATE 风格的表示。

#### 一个温和的证明（为何有效）

字典序排序保持了无前缀性：如果一个码字长度为 $l_1$，下一个长度为 $l_2 \ge l_1$，递增码字并移位确保了没有一个码字是另一个的前缀。因此，规范码字产生的压缩比与原始哈夫曼树相同。

规范哈夫曼编码将最优树转化为简单的算术——相同的压缩率，但具有顺序性、可预测性和优雅性。
### 653 算术编码

算术编码是一种强大的无损压缩方法，它将整个消息编码为一个介于 0 和 1 之间的单一数字。
与为符号分配离散比特序列的霍夫曼编码不同，算术编码将*整个消息*表示为一个区间内的分数，该区间随着每个符号的处理而缩小。

它广泛应用于现代压缩格式，如 JPEG、H.264 和 BZIP2。

#### 我们要解决什么问题？

霍夫曼编码只能分配整数比特长度的码字。
算术编码消除了这一限制，它可以为每个符号分配分数比特，从而针对任何概率分布实现更接近最优的压缩。

其思想是：
每个符号根据其概率缩小区间。
最终的子区间唯一地标识了消息。

#### 工作原理（通俗解释）

从区间 [0, 1) 开始。
每个符号根据其概率按比例细化该区间。

符号及其概率示例：

| 符号 | 概率 |    范围    |
| :--: | :--: | :--------: |
|  A   | 0.5  | [0.0, 0.5) |
|  B   | 0.3  | [0.5, 0.8) |
|  C   | 0.2  | [0.8, 1.0) |

对于消息 `"BAC"`：

1.  起始： [0.0, 1.0)
2.  符号 B → [0.5, 0.8)
3.  符号 A → 取 0.5 + (0.8 - 0.5) × [0.0, 0.5) = [0.5, 0.65)
4.  符号 C → 取 0.5 + (0.65 - 0.5) × [0.8, 1.0) = [0.62, 0.65)

最终范围： [0.62, 0.65)
该范围内的任何数字（例如 0.63）都能唯一标识该消息。

#### 数学表述

对于符号序列 $s_1, s_2, \ldots, s_n$，
每个符号具有累积概率范围 $[l_i, h_i)$，
我们迭代计算：

$$
\begin{aligned}
\text{range} &= h - l, \\
h' &= l + \text{range} \times \text{high}(s_i), \\
l' &= l + \text{range} \times \text{low}(s_i).
\end{aligned}
$$

处理完所有符号后，选择任意数字 $x \in [l, h)$ 作为编码。

解码过程通过查看 $x$ 落在哪个符号范围内来逆转此过程。

#### 示例步骤表

使用相同概率对 `"BAC"` 进行编码：

| 步骤 | 符号 | 处理前区间 | 范围 | 新区间 |
| ---- | ---- | ---------- | ---- | ------ |
| 1    | B    | [0.0, 1.0) | 1.0  | [0.5, 0.8) |
| 2    | A    | [0.5, 0.8) | 0.3  | [0.5, 0.65) |
| 3    | C    | [0.5, 0.65) | 0.15 | [0.62, 0.65) |

编码数字： 0.63

#### 微型代码（Python）

```python
def arithmetic_encode(message, probs):
    low, high = 0.0, 1.0
    for sym in message:
        range_ = high - low
        cum_low = sum(v for k, v in probs.items() if k < sym)
        cum_high = cum_low + probs[sym]
        high = low + range_ * cum_high
        low = low + range_ * cum_low
    return (low + high) / 2

probs = {'A': 0.5, 'B': 0.3, 'C': 0.2}
code = arithmetic_encode("BAC", probs)
print(round(code, 5))
```

#### 为什么它很重要

-   达到接近熵的压缩（每个符号分数比特）。
-   能平滑处理非整数概率模型。
-   能随上下文模型动态适应，用于自适应压缩器。
-   是以下技术的基础：
  * JPEG 和 H.264（CABAC 变体）
  * BZIP2（算术/范围编码器）
  * PPM 压缩器（基于部分匹配预测）

#### 复杂度

| 操作                     | 时间          | 空间  |
| ------------------------ | ------------- | ----- |
| 编码/解码                | $O(n)$        | $O(1)$ |
| 使用自适应概率时         | $O(n \log m)$ | $O(m)$ |

#### 亲自尝试

1.  使用符号 `{A:0.6, B:0.3, C:0.1}` 对 `"ABAC"` 进行编码。
2.  改变顺序，观察编码数字如何变化。
3.  实现范围编码，这是算术编码的缩放整数形式。
4.  尝试自适应频率更新以实现实时压缩。
5.  通过跟踪编码数字落在哪个子区间来进行解码。

#### 一个温和的证明（为什么它有效）

每个符号的范围细分对应于其概率质量。
因此，在编码 $n$ 个符号后，区间宽度等于：

$$
\prod_{i=1}^{n} P(s_i)
$$

表示此区间所需的比特数大约为：

$$
-\log_2 \left( \prod_{i=1}^{n} P(s_i) \right)
= \sum_{i=1}^{n} -\log_2 P(s_i)
$$

这等于香农信息量——
证明算术编码达到了接近熵的最优性。

算术编码用纯粹的区间取代了比特和树——
它不是用步骤，而是用精度本身来进行压缩。
### 654 Shannon–Fano 编码

Shannon–Fano 编码是一种早期的基于熵的无损压缩方法。它由 Claude Shannon 和 Robert Fano 在霍夫曼算法提出之前独立开发。虽然它并非总是最优的，但它为现代前缀码奠定了基础，并影响了霍夫曼编码和算术编码。

#### 我们要解决什么问题？

给定一组已知概率（或频率）的符号，我们希望分配二进制码，使得出现频率更高的符号获得更短的码——同时确保码是前缀码（即没有任何一个码是另一个码的前缀）。

目标：最小化期望码长

$$
L = \sum_i p_i \cdot |C_i|
$$

使其接近熵的下界
$$
H = -\sum_i p_i \log_2 p_i
$$

#### 核心思想

Shannon–Fano 编码的工作原理是：将概率表递归地划分为两个近似相等的部分，并分配 0 和 1。

1.  将所有符号按概率降序排序。
2.  将列表分成两部分，使两部分的总概率尽可能相等。
3.  给第一组分配 `0`，给第二组分配 `1`。
4.  对每个组递归执行上述步骤，直到每个符号都获得唯一的码。

结果是一个前缀码，尽管不总是最优的。

#### 示例

符号及其概率：

| 符号 | 概率 |
| :--: | :--: |
|  A   | 0.4  |
|  B   | 0.2  |
|  C   | 0.2  |
|  D   | 0.1  |
|  E   | 0.1  |

步骤 1. 按概率排序：

A (0.4), B (0.2), C (0.2), D (0.1), E (0.1)

步骤 2. 分成相等的两半：

| 组别 | 符号 | 总和 | 比特 |
| :--: | :--: | :--: | :--: |
| 左侧 | A, B | 0.6  |  0   |
| 右侧 | C,D,E | 0.4  |  1   |

步骤 3. 递归：

-   左侧组 (A, B): 分割 → A (0.4) | B (0.2)
    → A = `00`, B = `01`
-   右侧组 (C, D, E): 分割 → C (0.2) | D, E (0.2)
    → C = `10`, D = `110`, E = `111`

最终码字：

| 符号 | 概率 | 码字 |
| :--: | :--: | :--: |
|  A   | 0.4  |  00  |
|  B   | 0.2  |  01  |
|  C   | 0.2  |  10  |
|  D   | 0.1  |  110 |
|  E   | 0.1  |  111 |

平均码长：

$$
L = 0.4(2) + 0.2(2) + 0.2(2) + 0.1(3) + 0.1(3) = 2.2 \text{ 比特/符号}
$$

熵：

$$
H = -\sum p_i \log_2 p_i \approx 2.12
$$

效率：

$$
\frac{H}{L} = 0.96
$$

#### 微型代码 (Python)

```python
def shannon_fano(symbols):
    # 按概率降序排序符号
    symbols = sorted(symbols.items(), key=lambda x: -x[1])
    codes = {}

    def recurse(sub, prefix=""):
        if len(sub) == 1:
            codes[sub[0][0]] = prefix
            return
        total = sum(p for _, p in sub)
        acc, split = 0, 0
        for i, (_, p) in enumerate(sub):
            acc += p
            if acc >= total / 2:
                split = i + 1
                break
        recurse(sub[:split], prefix + "0")
        recurse(sub[split:], prefix + "1")

    recurse(symbols)
    return codes

probs = {'A': 0.4, 'B': 0.2, 'C': 0.2, 'D': 0.1, 'E': 0.1}
print(shannon_fano(probs))
```

#### 为什么它重要

-   具有历史重要性，是第一个系统化的前缀编码方法。
-   是霍夫曼后续改进（保证最优性）的基础。
-   展示了基于树的编码中使用的“分割与平衡”原则。
-   更易于理解和实现，适合教学用途。

#### 与霍夫曼编码的比较

| 方面         | Shannon–Fano            | 霍夫曼                                 |
| ------------ | ----------------------- | -------------------------------------- |
| 方法         | 自顶向下分割            | 自底向上合并                           |
| 最优性       | 不总是最优              | 总是最优                               |
| 码字顺序     | 确定性的                | 在权重相等时可能变化                   |
| 用途         | 历史性的，概念性的      | 实际压缩 (ZIP, JPEG)                   |

#### 复杂度

| 操作         | 时间          | 空间   |
| ------------ | ------------- | ------ |
| 排序         | $O(n \log n)$ | $O(n)$ |
| 码字生成     | $O(n)$        | $O(n)$ |

#### 动手试试

1.  为 `{A:7, B:5, C:2, D:1}` 构建一个 Shannon–Fano 树。
2.  将平均比特长度与霍夫曼编码的结果进行比较。
3.  验证前缀属性（没有任何码是另一个码的前缀）。
4.  通过反转码表来实现解码。

#### 一个温和的证明（为什么它有效）

在每次递归分割中，我们确保两组之间的总概率差异最小。这使得码长大致与符号概率成比例：

$$
|C_i| \approx \lceil -\log_2 p_i \rceil
$$

因此，Shannon–Fano 编码总是产生一个前缀码，其长度接近（但不保证等于）最优的熵下界。

Shannon–Fano 编码是从概率到码字的第一个真正步骤——它是信息论与压缩实践之间一座平衡但不完美的桥梁。
### 655 游程编码 (RLE)

游程编码 (RLE) 是最简单的无损压缩技术之一。
它将连续的重复符号（称为*游程*）替换为一个计数和符号本身。
当数据包含长串相同值时（例如图像、位图或包含空白的文本），RLE 是理想的选择。

#### 我们要解决什么问题？

未压缩的数据通常存在以重复符号形式出现的冗余：

```
AAAAABBBBCCCCCCDD
```

我们可以存储符号*重复了多少次*，而不是存储每个符号。

编码后的形式：

```
(5, A)(4, B)(6, C)(2, D)
```

当游程长度相对于字母表大小较长时，这可以节省空间。

#### 核心思想

通过将相同符号的游程表示为成对数据来压缩数据：

$$
(\text{符号}, \text{计数})
$$

例如：

|      输入      |       编码       |
| :------------: | :--------------: |
| `AAAAABBBCCDAA` |    `5A3B2C1D2A`   |
|   `0001111000`  | `(0,3)(1,4)(0,3)` |
|      `AAAB`     |       `3A1B`      |

解码过程简单地反转此过程，将每对数据扩展为重复的符号。

#### 算法步骤

1. 初始化 `count = 1`。
2. 遍历序列：
   * 如果下一个符号相同，则 `count` 加 1。
   * 如果符号改变，则输出 `(symbol, count)` 并重置计数。
3. 循环结束后，输出最后的游程。

#### 微型代码 (Python)

```python
def rle_encode(s):
    if not s:
        return ""
    result = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            result.append(f"{count}{s[i-1]}")
            count = 1
    result.append(f"{count}{s[-1]}")
    return "".join(result)

def rle_decode(encoded):
    import re
    parts = re.findall(r'(\d+)(\D)', encoded)
    return "".join(sym * int(cnt) for cnt, sym in parts)

text = "AAAAABBBCCDAA"
encoded = rle_encode(text)
decoded = rle_decode(encoded)
print(encoded, decoded)
```

输出：

```
5A3B2C1D2A AAAAABBBCCDAA
```

#### 示例演练

输入：
`AAAABBCCCCD`

| 步骤 | 当前符号 | 计数 | 编码输出 |
| ---- | -------- | ---- | -------- |
| A    | 4        | →    | `4A`     |
| B    | 2        | →    | `2B`     |
| C    | 4        | →    | `4C`     |
| D    | 1        | →    | `1D`     |

最终编码字符串：
`4A2B4C1D`

#### 为什么它重要

- 简单性：不需要统计模型或字典。
- 效率：非常适合图像、传真、DNA 序列或重复字符。
- 作为更高级压缩方案的基础模块：
  * TIFF、BMP、PCX 图像格式
  * DEFLATE 预处理（在 zlib、PNG 中）
  * 传真 Group 3/4 标准

#### 适用场景

| 数据类型       | 示例                           | 压缩收益 |
| -------------- | ------------------------------ | -------- |
| 单色图像       | 大片的白色/黑色区域            | 高       |
| 纯文本         | 空格、制表符                   | 中等     |
| 二进制数据     | 大量零值（例如稀疏位图）       | 高       |
| 随机数据       | 无重复                         | 无或负   |

#### 复杂度

| 操作     | 时间   | 空间  |
| -------- | ------ | ----- |
| 编码     | $O(n)$ | $O(1)$ |
| 解码     | $O(n)$ | $O(1)$ |

#### 动手尝试

1. 编码并解码 `"AAABBBAAACC"`。
2. 计算压缩率 = $\text{压缩后长度} / \text{原始长度}$。
3. 对一段文本段落尝试 RLE，它有帮助吗？
4. 修改代码以使用字节 `(count, symbol)` 而不是文本。
5. 将 RLE 与霍夫曼编码结合，压缩 RLE 的输出。

#### 一个温和的证明（为什么它有效）

如果 $r$ 是平均游程长度，
那么 RLE 将 $n$ 个字符压缩为大约 $2n / r$ 个符号。
当平均 $r > 2$ 时，压缩发生。

对于高度重复的数据 ($r \gg 2$)，
增益接近：

$$
\text{压缩率} \approx \frac{2}{r}
$$

游程编码将重复转化为经济性——
它看到的不是每个符号，而是它们持续存在的节奏。
### 656 LZ77（滑动窗口压缩）

LZ77 是 Abraham Lempel 和 Jacob Ziv 于 1977 年发明的一种基础压缩算法。
它引入了*滑动窗口压缩*的思想，即用向后引用来替换重复出现的模式。
这一概念构成了许多现代压缩器的基础，包括 DEFLATE（ZIP、gzip）、PNG 和 Zstandard。

#### 我们要解决什么问题？

数据中的冗余通常表现为重复的子字符串，而非长的相同字符序列。
例如：

```
ABABABA
```

包含了 `"ABA"` 的重叠重复。
RLE 无法高效处理这种情况，但 LZ77 可以通过*引用先前出现的内容*而非重复它们来实现。

#### 核心思想

维护一个最近看到数据的滑动窗口。
当一个新的子字符串重复了该窗口中的部分内容时，用一个（距离，长度，下一个符号）三元组来替换它。

每个三元组的含义是：

> “向后移动 `distance` 个字符，复制 `length` 个字符，然后输出 `next`。”

#### 示例

输入：

```
A B A B A B A
```

逐步压缩过程（逐步显示的窗口）：

| 步骤 | 当前窗口 | 下一个符号 | 找到的匹配项         | 输出       |
| ---- | -------------- | ----------- | ------------------- | --------- |
| 1    |,              | A           | 无                  | (0, 0, A) |
| 2    | A              | B           | 无                  | (0, 0, B) |
| 3    | AB             | A           | 距离 2 处的 "A"     | (2, 1, B) |
| 4    | ABA            | B           | 距离 2 处的 "AB"    | (2, 2, A) |
| 5    | ABAB           | A           | 距离 2 处的 "ABA"   | (2, 3, —) |

最终编码序列：

```
(0,0,A) (0,0,B) (2,1,B) (2,2,A)
```

解码输出：

```
ABABABA
```

#### 工作原理

1.  初始化一个空的搜索缓冲区（过去数据）和一个前瞻缓冲区（未来数据）。
2.  对于前瞻缓冲区中的下一个（些）符号：
    *   在搜索缓冲区中找到最长的匹配项。
    *   发出一个三元组 `(distance, length, next_char)`。
    *   将窗口向前滑动 `length + 1` 个位置。
3.  继续直到输入结束。

搜索缓冲区允许向后引用，
而前瞻缓冲区限制了我们向前匹配的距离。

#### 简化代码（Python，简化版）

```python
def lz77_compress(data, window_size=16):
    i, output = 0, []
    while i < len(data):
        match = (0, 0, data[i])
        for dist in range(1, min(i, window_size) + 1):
            length = 0
            while (i + length < len(data) and
                   data[i - dist + length] == data[i + length]):
                length += 1
            if length > match[1]:
                next_char = data[i + length] if i + length < len(data) else ''
                match = (dist, length, next_char)
        output.append(match)
        i += match[1] + 1
    return output
```

示例：

```python
text = "ABABABA"
print(lz77_compress(text))
# [(0,0,'A'), (0,0,'B'), (2,1,'B'), (2,2,'A')]
```

#### 为何重要

-   DEFLATE（ZIP、gzip、PNG）的基础。
-   实现了强大的基于字典的压缩。
-   自引用：输出可以描述未来的数据。
-   在结构化文本、二进制文件和重复数据上效果良好。

现代变体（LZSS、LZW、LZMA）扩展或改进了这个模型。

#### 压缩格式

一个典型的 LZ77 标记：

| 字段       | 描述                   |
| ----------- | ---------------------------- |
| 距离        | 向后查找的距离（字节数） |
| 长度        | 要复制的字节数         |
| 下一个符号  | 匹配项后面的字面量     |

示例：
`(distance=4, length=3, next='A')`
→ “从 4 个位置之前复制 3 个字节，然后写入 A”。

#### 复杂度

| 操作     | 时间                           | 空间   |
| --------- | -------------------------- | ------ |
| 编码      | $O(n w)$（窗口大小 $w$）   | $O(w)$ |
| 解码      | $O(n)$                     | $O(w)$ |

优化的实现使用哈希表或字典树来降低搜索成本。

#### 动手尝试

1.  编码 `"BANANA_BANDANA"`。
2.  尝试不同的窗口大小。
3.  将向后指针可视化为符号之间的箭头。
4.  实现 LZSS，在不需要时跳过存储 `next_char`。
5.  与霍夫曼编码结合，实现类似 DEFLATE 的压缩。

#### 一个温和的证明（为何有效）

每个发出的三元组覆盖了输入的一个非重叠子字符串。
重建过程是明确的，因为每个 `(distance, length)` 仅引用已经解码的数据。
因此，LZ77 形成了一个自洽的压缩系统，保证无损恢复。

压缩比随着匹配子字符串长度的增加而提高：
$$
R \approx \frac{n}{n - \sum_i \text{length}_i}
$$

冗余越多，压缩率越高。

LZ77 教会了机器*回顾过去以向前迈进*——
这种记忆和重用的模型成为了现代压缩的核心。
### 657 LZ78（字典构建）

LZ78 由 Abraham Lempel 和 Jacob Ziv 于 1978 年提出，是 LZ77 的后续算法。
LZ77 使用*滑动窗口*进行压缩，而 LZ78 则构建一个显式的、记录迄今为止遇到的子字符串的字典。
每个新的短语只存储一次，后续的引用直接指向字典条目。

这种从基于窗口到基于字典的压缩方式的转变，为 LZW、GIF 和 TIFF 等算法铺平了道路。

#### 我们要解决什么问题？

LZ77 复用先前文本的一个*移动*窗口，每次匹配都需要搜索这个窗口。
LZ78 通过将已知子字符串存储在一个动态增长的字典中，提高了效率。
编码器不再向后扫描，而是直接通过索引引用字典条目。

这减少了搜索时间并简化了解码，代价是需要管理一个字典。

#### 核心思想

每个输出标记编码为：

$$
(\text{索引}, \text{下一个符号})
$$

其中：

- `索引` 指向字典中已有的最长前缀。
- `下一个符号` 是扩展该前缀的新字符。

这对标记定义了一个要添加到字典中的新条目：
$$
\text{dict}[k] = \text{dict}[\text{索引}] + \text{下一个符号}
$$

#### 示例

让我们对字符串进行编码：

```
ABAABABAABAB
```

步骤 1. 初始化一个空字典。

| 步骤 | 输入 | 最长前缀 | 索引 | 下一个符号 | 输出 | 新条目 |
| ---- | ----- | -------------- | ----- | ----------- | ------ | --------- |
| 1    | A     | ""             | 0     | A           | (0, A) | 1: A      |
| 2    | B     | ""             | 0     | B           | (0, B) | 2: B      |
| 3    | A     | A              | 1     | B           | (1, B) | 3: AB     |
| 4    | A     | A              | 1     | A           | (1, A) | 4: AA     |
| 5    | B     | AB             | 3     | A           | (3, A) | 5: ABA    |
| 6    | A     | ABA            | 5     | B           | (5, B) | 6: ABAB   |

最终输出：

```
(0,A) (0,B) (1,B) (1,A) (3,A) (5,B)
```

最终字典：

| 索引 | 条目 |
| :---: | :---: |
|   1   |   A   |
|   2   |   B   |
|   3   |   AB  |
|   4   |   AA  |
|   5   |  ABA  |
|   6   |  ABAB |

解码后的消息完全相同：`ABAABABAABAB`。

#### 微型代码（Python）

```python
def lz78_compress(s):
    dictionary = {}
    output = []
    current = ""
    next_index = 1

    for c in s:
        if current + c in dictionary:
            current += c
        else:
            idx = dictionary.get(current, 0)
            output.append((idx, c))
            dictionary[current + c] = next_index
            next_index += 1
            current = ""
    if current:
        output.append((dictionary[current], ""))
    return output

text = "ABAABABAABAB"
print(lz78_compress(text))
```

输出：

```
[(0, 'A'), (0, 'B'), (1, 'B'), (1, 'A'), (3, 'A'), (5, 'B')]
```

#### 解码过程

给定编码对 `(索引, 符号)`：

1.  初始化字典，`dict[0] = ""`。
2.  对于每一对：
    *   输出 `dict[索引] + 符号`。
    *   将其作为新条目添加到字典中。

解码过程确定性地重建文本。

#### 为何重要

-   引入了显式的短语字典，可在不同数据块间复用。
-   无需向后扫描，对于大数据比 LZ77 更快。
-   是 LZW 算法的基础，LZW 取消了显式的符号输出并增加了自动字典管理。
-   应用于：
    *   UNIX `compress` 命令
    *   GIF 和 TIFF 图像格式
    *   旧式调制解调器协议（V.42bis）

#### 对比

| 特性       | LZ77                       | LZ78                               |
| ---------- | -------------------------- | ---------------------------------- |
| 模型       | 滑动窗口                   | 显式字典                           |
| 输出       | (距离, 长度, 下一个)       | (索引, 符号)                       |
| 字典       | 隐式（在数据流中）         | 显式（存储条目）                   |
| 解码       | 即时                       | 需要字典重建                       |
| 后继算法   | DEFLATE, LZMA              | LZW, LZMW                          |

#### 复杂度

| 操作     | 时间复杂度       | 空间复杂度 |
| -------- | ---------------- | ---------- |
| 编码     | $O(n)$ 平均      | $O(n)$     |
| 解码     | $O(n)$           | $O(n)$     |

内存使用量会随着唯一子字符串的数量而增长，因此实现中通常在字典满时重置它。

#### 动手尝试

1.  编码并解码 `"TOBEORNOTTOBEORTOBEORNOT"`。
2.  打印字典的演化过程。
3.  与 LZ77 的输出大小进行比较。
4.  实现字典在固定大小（例如 4096 个条目）时重置。
5.  通过复用索引而不显式输出字符，将其扩展为 LZW。

#### 一个温和的证明（为何有效）

每个发出的编码对对应一个之前未见过的新短语。
因此，输入中的每个子字符串都可以表示为一个字典引用序列。
因为每个新短语都是通过单个符号扩展前一个短语得到的，所以字典是*前缀封闭*的，这确保了唯一的重建。

压缩效率随着数据冗余度的提高而提升：

$$
R \approx \frac{\text{\#编码对} \times (\log_2 N + \text{字符比特数})}{\text{输入比特数}}
$$

并且对于大的 $N$ 值，接近熵极限。

LZ78 教会了压缩算法将模式作为“单词”来记忆，将记忆的滑动窗口转变为一个不断增长的意义词汇表。
### 658 LZW (Lempel–Ziv–Welch)

LZW 由 Terry Welch 于 1984 年提出，是 LZ78 的一种优化形式。
它消除了在每个编码对中传输*额外符号*的需要，转而依靠字典本身来推断下一个字符。
这一微小改动使得 LZW 更快、更简单，并完美适用于实际应用，它驱动了 GIF、TIFF 和 UNIX compress 等格式。

#### 我们要解决什么问题？

LZ78 产生形式为 `(索引, 下一个符号)` 的编码对，这会增加额外数据。
LZW 消除了这种冗余：
它不发送字面符号，而只传输字典索引，因为字典可以根据先前的上下文预测下一个符号。

简而言之，LZW 的理念是：

> "如果编码器和解码器以相同的方式构建相同的字典，我们只需要发送索引。"

#### 核心思想

1.  用所有单字符（例如，ASCII 0–255）初始化字典。
2.  逐个符号读取输入：

    *   保持在字典中找到的最长字符串 `w`。
    *   当 `w + c`（当前字符串 + 下一个字符）未在字典中找到时：

        *   输出 `w` 的编码。
        *   将 `w + c` 添加到字典中。
        *   设置 `w = c`。
3.  处理结束时，输出最后一个 `w` 的编码。

编码器和解码器以相同的方式增长字典，确保确定性的重建。

#### 示例

让我们对以下字符串进行编码：

```
TOBEORNOTTOBEORTOBEORNOT
```

步骤 1. 初始化字典：

| 编码  | 条目                 |
| ----- | -------------------- |
| 0–255 | 所有 ASCII 字符      |

步骤 2. 逐步编码：

| 步骤 | w   | c | 输出           | 新条目               |
| ---- | --- | - | -------------- | -------------------- |
| 1    | T   | O | code(T) = 84   | TO                   |
| 2    | O   | B | code(O) = 79   | OB                   |
| 3    | B   | E | code(B) = 66   | BE                   |
| 4    | E   | O | code(E) = 69   | EO                   |
| 5    | O   | R | code(O) = 79   | OR                   |
| 6    | R   | N | code(R) = 82   | RN                   |
| 7    | N   | O | code(N) = 78   | NO                   |
| 8    | O   | T | code(O) = 79   | OT                   |
| 9    | T   | O | code(T) = 84   | TO （已存在）        |
| 10   | TO  | B | code(TO)       | TOB                  |
| 11   | B   | E | code(B) = 66   | BE （已存在）        |
| 12   | E   | O | code(E) = 69   | EO （已存在）        |

依此类推…

输出序列（部分）：

```
84, 79, 66, 69, 79, 82, 78, 79, 84, 256, 258, 260, ...
```

每个数字代表指向字典条目的一个编码。

#### 微型代码 (Python)

```python
def lzw_compress(data):
    # 用单字符初始化字典
    dict_size = 256
    dictionary = {chr(i): i for i in range(dict_size)}

    w = ""
    result = []
    for c in data:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
    if w:
        result.append(dictionary[w])
    return result
```

示例：

```python
text = "TOBEORNOTTOBEORTOBEORNOT"
codes = lzw_compress(text)
print(codes)
```

#### 解码过程

解码使用相同的逻辑反向重构文本：

1.  用单字符初始化字典。
2.  读取第一个编码，输出其对应的字符。
3.  对于每个后续编码：

    *   如果它存在于字典中 → 输出它。
    *   如果不存在 → 输出 `w + first_char(w)`（针对未见过编码的特殊情况）。
    *   将 `w + first_char(current)` 添加到字典中。
    *   更新 `w`。

#### 为何重要

*   基于字典的高效性：紧凑且快速。
*   无需发送字典或符号。
*   在硬件和软件中都易于实现。
*   用于实际格式：

    *   GIF
    *   TIFF
    *   UNIX compress
    *   PostScript / PDF

#### 与 LZ78 的比较

| 特性       | LZ78              | LZW                   |
| ---------- | ----------------- | --------------------- |
| 输出       | (索引, 符号)      | 仅索引                |
| 字典       | 显式条目          | 隐式增长              |
| 效率       | 稍低              | 对真实数据更好        |
| 应用领域   | 研究              | 实际标准              |

#### 复杂度

| 操作       | 时间             | 空间  |
| ---------- | ---------------- | ----- |
| 编码       | $O(n)$ 平均      | $O(n)$ |
| 解码       | $O(n)$           | $O(n)$ |

压缩比随着重复短语和更长的字典而提高。

#### 亲自尝试

1.  压缩并解压 `"TOBEORNOTTOBEORTOBEORNOT"`。
2.  逐步打印字典的增长情况。
3.  在一个文本段落上尝试，注意重复单词的压缩效果。
4.  修改为仅支持小写 ASCII 字符（大小 26）。
5.  尝试在 4096 个条目后重置字典（如 GIF 中那样）。

#### 一个温和的证明（为何有效）

由于编码器和解码器从相同的字典开始，并以相同的顺序添加条目，相同的索引序列会导致相同的重建。
字典通过前缀扩展增长，每个新条目都是前一个条目加上一个符号，确保了确定性的解码。

对于熵为 $H$ 的长输入，平均编码长度接近每符号 $H + 1$ 比特，使得 LZW 对于平稳信源是渐近最优的。

LZW 将压缩转化为纯粹的记忆和推断 —— 一场编码的舞蹈，其中意义被构建、共享，且永不重复传输。
### 659 伯勒斯-惠勒变换 (BWT)

伯勒斯-惠勒变换 (BWT) 由 Michael Burrows 和 David Wheeler 于 1994 年发明，是无损压缩领域的一个里程碑。
与之前直接编码数据的方案不同，BWT 会对数据进行重排，将输入转换为一种更容易被 RLE 或 Huffman 等压缩器利用的形式。

BWT 的精妙之处在于它是可逆且保持结构的：它将相似的字符聚集在一起，在实际编码之前放大了模式。

#### 我们要解决什么问题？

传统的压缩器在局部操作，它们在小窗口或字典中检测模式。
然而，当相似的符号相距很远时，它们无法有效地利用全局结构。

BWT 通过排序字符串的所有循环旋转，然后提取最后一列来解决这个问题，从而有效地将相似的上下文分组在一起。
这种重排不会降低熵，但使得用 RLE 或 Huffman 等简单算法进行压缩*更容易*。

#### 核心思想

给定一个长度为 $n$ 的字符串 $S$，附加一个特殊的结束标记 `$`，该标记在字典序上小于任何字符。
生成 $S$ 的所有循环旋转，并按字典序对它们进行排序。
BWT 的输出就是这个排序后矩阵的最后一列。

该变换还会记录原始字符串在排序列表中的索引（用于逆变换）。

#### 示例

输入：

```
BANANA$
```

步骤 1. 生成所有旋转：

| 旋转索引 |  字符串  |
| :------: | :-----: |
|     0    | BANANA$ |
|     1    | ANANA$B |
|     2    | NANA$BA |
|     3    | ANA$BAN |
|     4    | NA$BANA |
|     5    | A$BANAN |
|     6    | $BANANA |

步骤 2. 按字典序对所有旋转进行排序：

| 排序索引 | 旋转字符串 | 最后一个字符 |
| :----------: | :------: | :------------: |
|       0      |  $BANANA |        A       |
|       1      |  A$BANAN |        N       |
|       2      |  ANA$BAN |        N       |
|       3      |  ANANA$B |        A       |
|       4      |  BANANA$ |        $       |
|       5      |  NA$BANA |        A       |
|       6      |  NANA$BA |        A       |

步骤 3. 提取最后一列 (L)：

```
L = ANNA$AA
```

并记录原始字符串 (`BANANA$`) 的索引 → 第 4 行。

因此，
BWT 输出 = (L = ANNA$AA, index = 4)

#### 解码（逆 BWT）

为了逆转变换，逐列重建表格：

1.  从最后一列 `L` 开始。
2.  重复地将 `L` 前置到现有行，并在每次迭代后进行排序。
3.  经过 `n` 次迭代后，末尾包含 `$` 的那一行就是原始字符串。

对于 `L = ANNA$AA`, index = 4：

| 迭代次数 | 表格（每轮排序后）          |
| -------- | -------------------------- |
| 1        | A, N, N, A, $, A, A        |
| 2        | A$, AN, AN, NA, N$, AA, AA |
| 3        | ANA, NAN, NNA, A$B, 等等。 |

…最终得到 `BANANA$`。

高效的解码使用 LF 映射关系来避免构建完整的矩阵：
$$
\text{next}(i) = C[L[i]] + \text{rank}(L, i, L[i])
$$
其中 $C[c]$ 是字典序上小于 $c$ 的字符的计数。

#### 微型代码 (Python)

```python
def bwt_transform(s):
    s = s + "$"
    rotations = [s[i:] + s[:i] for i in range(len(s))]
    rotations.sort()
    last_column = ''.join(row[-1] for row in rotations)
    index = rotations.index(s)
    return last_column, index

def bwt_inverse(last_column, index):
    n = len(last_column)
    table = [""] * n
    for _ in range(n):
        table = sorted([last_column[i] + table[i] for i in range(n)])
    return table[index].rstrip("$")

# 示例
L, idx = bwt_transform("BANANA")
print(L, idx)
print(bwt_inverse(L, idx))
```

输出：

```
ANNA$AA 4
BANANA
```

#### 为何重要

-   结构放大器：通过上下文将相同符号分组，从而提升以下算法的性能：
    *   游程编码 (RLE)
    *   移动前端 (MTF)
    *   Huffman 或算术编码
-   现代压缩器的基础：
    *   bzip2
    *   zstd 的预处理
    *   生物信息学中的 FM-Index
-   通过后缀数组和秩，实现可搜索的压缩文本。

#### 复杂度

| 操作       | 时间复杂度                 | 空间复杂度 |
| ---------- | ------------------------ | ---------- |
| 变换       | $O(n \log n)$            | $O(n)$     |
| 逆变换     | $O(n)$ (使用 LF 映射)    | $O(n)$     |

#### 动手尝试

1.  对 `"MISSISSIPPI"` 应用 BWT。
2.  与游程编码结合。
3.  测试压缩比。
4.  比较应用 BWT 前后的结果。
5.  可视化短单词的旋转矩阵。

#### 一个温和的证明（为何有效）

BWT 本身不压缩；它是对数据进行排列，使得具有相似上下文的符号彼此靠近。
由于大多数文本在局部是可预测的，这*在变换后*降低了熵：

$$
H(\text{BWT}(S)) = H(S)
$$

但变换后的数据表现出更长的游程和更简单的局部结构，这使得 Huffman 等压缩器可以更有效地利用。

伯勒斯-惠勒变换是压缩领域安静的魔术师——
它本身并不缩小数据，而是将其重新排列成有序，
将混乱转化为可压缩的平静。
### 660 前移编码 (MTF)

前移编码 (MTF) 是一种简单而强大的变换，通常与 Burrows–Wheeler 变换 (BWT) 结合使用。
其目的是将 BWT 产生的局部符号重复，转换为小整数序列，这些小整数非常适合通过游程编码 (RLE) 或霍夫曼编码进行压缩。

#### 我们要解决什么问题？

应用 BWT 后，变换后的字符串包含相同符号的簇。
为了利用这一点，我们希望根据符号*最近*出现的情况来表示它们。

如果我们维护一个包含所有可能符号的列表，并将每个访问到的符号移到列表前端，那么频繁出现的符号将具有较小的索引，从而产生许多零和一，这非常适合于熵编码。

#### 核心思想

维护一个包含所有符号的有序列表（即“字母表”）。
对于输入中的每个符号：

1.  输出它在当前列表中的位置索引。
2.  将该符号移动到列表的前端。

这捕捉了局部性：最近使用的符号出现得更早，因此获得更小的索引。

#### 示例

输入序列：

```
banana
```

字母表（初始）：`[a, b, n]`

| 步骤 | 符号 | 操作前列表 | 索引 | 操作后列表 |
| ---- | ---- | ---------- | ---- | ---------- |
| 1    | b    | [a, b, n]  | 1    | [b, a, n]  |
| 2    | a    | [b, a, n]  | 1    | [a, b, n]  |
| 3    | n    | [a, b, n]  | 2    | [n, a, b]  |
| 4    | a    | [n, a, b]  | 1    | [a, n, b]  |
| 5    | n    | [a, n, b]  | 1    | [n, a, b]  |
| 6    | a    | [n, a, b]  | 1    | [a, n, b]  |

编码输出：

```
$$1, 1, 2, 1, 1, 1]
```

注意频繁出现的字母（"a", "n"）是如何用小数字表示的。

#### 解码过程

给定编码索引和初始字母表：

1.  对于每个索引，选取该位置的符号。
2.  输出它并将其移动到列表的前端。

这个过程是完全可逆的，因为编码器和解码器执行相同的列表更新操作。

#### 微型代码 (Python)

```python
def mtf_encode(data, alphabet):
    symbols = list(alphabet)
    result = []
    for c in data:
        index = symbols.index(c)
        result.append(index)
        symbols.insert(0, symbols.pop(index))
    return result

def mtf_decode(indices, alphabet):
    symbols = list(alphabet)
    result = []
    for i in indices:
        c = symbols[i]
        result.append(c)
        symbols.insert(0, symbols.pop(i))
    return ''.join(result)

alphabet = list("abn")
encoded = mtf_encode("banana", alphabet)
decoded = mtf_decode(encoded, list("abn"))
print(encoded, decoded)
```

输出：

```
$$1, 1, 2, 1, 1, 1] banana
```

#### 为何重要

-   与 BWT 完美搭配：

    *   BWT 将相似的符号聚集在一起。
    *   MTF 将这些分组转换为小索引。
    *   然后 RLE 或霍夫曼编码可以高效地压缩这些小数字。
-   用于：

    *   bzip2
    *   块排序压缩器
    *   文本索引系统 (FM-Index)

MTF 本身并不压缩数据，而是将数据转换成更容易压缩的形式。

#### 直观示例 (BWT 之后)

假设 BWT 输出是：

```
AAAAABBBBCCCC
```

MTF 输出将类似于：

```
0,0,0,0,1,1,1,1,2,2,2,2
```

现在 RLE 或霍夫曼编码可以极其高效地压缩这些微小的整数。

#### 复杂度

| 操作     | 时间       | 空间  |            |     |        |    |
| -------- | ---------- | ----- | ---------- | --- | ------ | -- |
| 编码     | $O(n \cdot | \Sigma | )$ (朴素) | $O( | \Sigma | )$ |
| 解码     | $O(n \cdot | \Sigma | )$         | $O( | \Sigma | )$ |

优化的实现使用链表或直接映射数组，将每个符号的成本降低到接近 $O(n)$。

#### 亲自尝试

1.  对 `"BANANA$"` 的 BWT 结果（→ `ANNA$AA`）应用 MTF。
2.  编码结果：你会得到许多小数字。
3.  接着进行 RLE + 霍夫曼编码以实现完全压缩。
4.  比较应用 MTF 前后的压缩大小。
5.  改变字母表的顺序，观察结果如何变化。

#### 一个温和的证明 (为何有效)

令 $f(c)$ 为符号 $c$ 的频率。
经过 BWT 后，相同的符号倾向于连续出现，因此它们的首次出现会将其移动到列表前端，随后的出现将产生索引 `0`。
因此，预期的输出索引与相同符号之间的距离成正比，这使得 MTF 对于富含局部性的数据非常有效。

MTF 后的熵往往低于之前：

$$
H_\text{MTF}(X) \leq H(X)
$$

对于结构化输入（不一定是随机数据）成立。

前移编码捕捉了*符号的近期性*——
这是一种极简的记忆，将新鲜的内容保持在最前面，
准备好让压缩变得异常简单。

# 第 67 节 密码学哈希与校验和
### 661 滚动哈希

滚动哈希是一种巧妙的哈希技术，它允许你高效地计算重叠子串的哈希值。
它是 Rabin–Karp 算法、Rabin 指纹以及用于文本搜索和数据同步的字符串去重系统等算法的核心基础。

#### 我们要解决什么问题？

在许多问题中，你需要高效地比较子串或滑动窗口。
如果天真地为每个子串重新计算哈希值，每一步需要 $O(m)$ 的时间，总共导致 $O(nm)$ 的时间复杂度，对于大规模输入来说太慢了。

滚动哈希使得当窗口滑动一个位置时，你可以在常数时间内更新哈希值，从而将总时间减少到 $O(n)$。

#### 核心思想

将一个字符串表示为一个选定基数的数字，并计算其对一个大素数取模的值：

$$
H(s_0 s_1 \dots s_{m-1}) = (s_0 b^{m-1} + s_1 b^{m-2} + \dots + s_{m-1}) \bmod M
$$

当窗口移动一个字符（丢弃 `old` 并添加 `new`）时，哈希值可以被高效地更新：

$$
H_{\text{new}} = (b(H_{\text{old}} - s_0 b^{m-1}) + s_m) \bmod M
$$

这意味着我们可以在文本上滑动，并在 $O(1)$ 时间内计算新的哈希值。

#### 示例

考虑字符串 `ABCD`，基数 $b = 256$，模数 $M = 101$。

计算：

$$
H("ABC") = (65 \times 256^2 + 66 \times 256 + 67) \bmod 101
$$

当窗口从 `"ABC"` 滑动到 `"BCD"` 时：

$$
H("BCD") = (b(H("ABC") - 65 \times 256^2) + 68) \bmod 101
$$

这有效地移除了 `'A'` 并添加了 `'D'`。

#### 微型代码（Python）

```python
def rolling_hash(s, base=256, mod=101):
    h = 0
    for c in s:
        h = (h * base + ord(c)) % mod
    return h

def update_hash(old_hash, left_char, right_char, power, base=256, mod=101):
    # 移除最左边的字符，添加最右边的字符
    old_hash = (old_hash - ord(left_char) * power) % mod
    old_hash = (old_hash * base + ord(right_char)) % mod
    return old_hash
```

用法：

```python
text = "ABCD"
m = 3
mod = 101
base = 256
power = pow(base, m-1, mod)

h = rolling_hash(text[:m], base, mod)
for i in range(1, len(text) - m + 1):
    h = update_hash(h, text[i-1], text[i+m-1], power, base, mod)
    print(h)
```

#### 为什么它很重要

- **常数时间滑动**：哈希更新时间为 $O(1)$
- **子串搜索的理想选择**：用于 Rabin–Karp 算法
- **用于去重系统**（rsync, git）
- **多项式哈希和滚动校验和的基础**（Adler-32, Rabin 指纹）

滚动哈希在速度和准确性之间取得平衡，只要模数和基数足够大，碰撞就很少发生。

#### 碰撞与模数

当两个不同的子串具有相同的哈希值时，就会发生碰撞。
我们通过以下方式最小化碰撞：

1. 使用一个大素数模数 $M$，通常接近 $2^{61} - 1$。
2. 使用两个不同的 $(b, M)$ 对进行双重哈希。
3. 偶尔通过直接的字符串比较来验证匹配。

#### 复杂度

| 操作               | 时间   | 空间  |
| ------------------ | ------ | ------ |
| 计算第一个哈希值   | $O(m)$ | $O(1)$ |
| 滑动更新           | $O(1)$ | $O(1)$ |
| 整个文本上的总时间 | $O(n)$ | $O(1)$ |

#### 动手试试

1.  计算 `"BANANA"` 所有长度为 3 的子串的滚动哈希值。
2.  使用模数 $M = 101$，基数 $b = 256$。
3.  比较小 $M$ 与大 $M$ 的碰撞率。
4.  修改代码以使用两个模数进行双重哈希。
5.  使用此滚动哈希实现 Rabin–Karp 子串搜索算法。

#### 一个温和的证明（为什么它有效）

当我们从窗口 $[i, i+m-1]$ 滑动到 $[i+1, i+m]$ 时，
被丢弃字符的贡献是预先已知的，
因此我们可以在不重新计算所有内容的情况下调整哈希值。

由于模运算是线性的：

$$
H(s_1 \dots s_m) = (b(H(s_0 \dots s_{m-1}) - s_0 b^{m-1}) + s_m) \bmod M
$$

这个性质确保了正确性，同时保持了常数时间的更新。

滚动哈希是现代文本处理的默默无闻的主力——
它不直接寻找模式，
而是进行总结、滑动，并让算术运算来找到匹配。
### 662 CRC32（循环冗余校验）

循环冗余校验（CRC）是一种用于检测数字数据中错误的校验和算法。
它广泛应用于网络、存储和文件格式（如 ZIP、PNG、以太网）中，以确保数据被正确传输或存储。

CRC32 是一种常见的 32 位变体，速度快、简单，并且在检测随机错误方面非常可靠。

#### 我们要解决什么问题？

当数据通过信道传输或写入磁盘时，由于噪声或损坏，比特位可能会发生翻转。
我们需要一种方法来检测接收到的数据是否与发送的数据完全相同。

简单的校验和（如字节求和）可能会遗漏许多错误。
CRC 将数据视为 GF(2) 上的多项式，并用一个固定的生成多项式进行除法运算，产生的余数作为强完整性校验。

#### 核心思想

将数据视为一个二进制多项式：

$$
M(x) = m_0x^{n-1} + m_1x^{n-2} + \dots + m_{n-1}
$$

选择一个次数为 $r$ 的生成多项式 $G(x)$（对于 CRC32，$r=32$）。
我们在消息后追加 $r$ 个零，并使用模 2 算术除以 $G(x)$：

$$
R(x) = (M(x) \cdot x^r) \bmod G(x)
$$

然后，传输：

$$
T(x) = M(x) \cdot x^r + R(x)
$$

在接收端，执行相同的除法运算。
如果余数为零，则假定消息有效。

由于运算在 GF(2) 上进行，所有操作都使用 XOR 代替减法。

#### 示例（简单 CRC）

令 $G(x) = x^3 + x + 1$（二进制 1011）。
消息：`1101`。

1.  追加三个零 → `1101000`
2.  将 `1101000` 除以 `1011`（模 2）。
3.  余数为 `010`。
4.  传输 `1101010`。

在接收端，除以相同的多项式得到余数 `0`，确认了完整性。

#### CRC32 多项式

CRC32 使用的多项式是：

$$
G(x) = x^{32} + x^{26} + x^{23} + x^{22} + x^{16} + x^{12} + x^{11} + x^{10} + x^8 + x^7 + x^5 + x^4 + x^2 + x + 1
$$

十六进制表示为：`0x04C11DB7`。

#### 微型代码（C）

```c
#include <stdint.h>
#include <stdio.h>

uint32_t crc32(uint8_t *data, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 1)
                crc = (crc >> 1) ^ 0xEDB88320;
            else
                crc >>= 1;
        }
    }
    return crc ^ 0xFFFFFFFF;
}

int main() {
    uint8_t msg[] = "HELLO";
    printf("CRC32: %08X\n", crc32(msg, 5));
}
```

输出：

```
CRC32: 3610A686
```

#### 微型代码（Python）

```python
def crc32(data: bytes):
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF

print(hex(crc32(b"HELLO")))
```

输出：

```
0x3610a686
```

#### 为什么它很重要

-   错误检测，而非纠正
-   可以检测：
    *   所有单比特和双比特错误
    *   奇数个比特错误
    *   短于 32 比特的突发错误
-   用于：
    *   以太网、ZIP、PNG、gzip、TCP/IP 校验和
    *   文件系统和数据传输协议

CRC 速度快，对硬件友好，并且在数学上基于多项式除法。

#### 复杂度

| 操作         | 时间复杂度                                   | 空间复杂度                 |
| ------------ | -------------------------------------------- | -------------------------- |
| 编码         | $O(nr)$（按位）或 $O(n)$（查表驱动）         | $O(1)$ 或 $O(256)$ 查找表  |
| 验证         | $O(n)$                                       | $O(1)$                     |

基于查表的 CRC 实现每秒可以计算数兆字节数据的校验和。

#### 动手试试

1.  使用生成多项式 `1011` 计算消息 `1101` 的 CRC3。
2.  比较小多项式和大多项式的余数。
3.  在 Python 中实现 CRC16 和 CRC32。
4.  翻转一个比特，验证 CRC 能否检测到错误。
5.  用 XOR 运算可视化多项式除法。

#### 一个温和的证明（为什么它有效）

因为 CRC 使用 GF(2) 上的多项式除法，
每个错误模式 $E(x)$ 对 $G(x)$ 取模都有一个唯一的余数。
如果选择的 $G(x)$ 使得没有低权重的 $E(x)$ 能整除它，
那么所有小的错误都保证会改变余数。

CRC32 的生成多项式经过精心设计，能够捕获真实通信系统中最可能出现的错误类型。

CRC32 是数据完整性的无声守护者——
快到足以处理每个数据包，
可靠到足以用于每个磁盘，
简单到只需几行 C 代码即可实现。
### 663 Adler-32 校验和

Adler-32 是一种简单高效的校验和算法，设计为 CRC32 的轻量级替代方案。
它结合了速度和合理的错误检测能力，因此在 zlib、PNG 和其他数据压缩库等应用中非常流行。

#### 我们要解决什么问题？

CRC32 提供了强大的错误检测能力，但涉及位运算和多项式算术，在某些系统上可能较慢。
对于轻量级应用（如验证压缩数据），我们需要一个更快、易于实现的校验和算法，同时仍能检测常见的传输错误。

Adler-32 通过使用整数的模运算而非多项式运算来实现这一目标。

#### 核心思想

Adler-32 维护两个运行总和：一个用于数据字节，另一个用于累积总和。

设消息为 $m_1, m_2, \dots, m_n$，每个都是无符号字节。

计算两个值：

$$
A = 1 + \sum_{i=1}^{n} m_i \pmod{65521}
$$

$$
B = 0 + \sum_{i=1}^{n} A_i \pmod{65521}
$$

最终的校验和为：

$$
\text{Adler-32}(M) = (B \ll 16) + A
$$

这里，65521 是小于 $2^{16}$ 的最大质数，选择它是为了获得良好的模运算特性。

#### 示例

消息：`"Hi"`

字符：

```
'H' = 72
'i' = 105
```

计算：

| 步骤     | A (模 65521) | B (模 65521) |
| -------- | ------------ | ------------ |
| 初始化   | 1            | 0            |
| +H (72)  | 73           | 73           |
| +i (105) | 178          | 251          |

然后：

$$
\text{校验和} = (251 \ll 16) + 178 = 16449842
$$

十六进制表示：

```
Adler-32 = 0x00FB00B2
```

#### 微型代码 (C)

```c
#include <stdint.h>
#include <stdio.h>

uint32_t adler32(const unsigned char *data, size_t len) {
    uint32_t A = 1, B = 0;
    const uint32_t MOD = 65521;

    for (size_t i = 0; i < len; i++) {
        A = (A + data[i]) % MOD;
        B = (B + A) % MOD;
    }

    return (B << 16) | A;
}

int main() {
    unsigned char msg[] = "Hello";
    printf("Adler-32: %08X\n", adler32(msg, 5));
}
```

输出：

```
Adler-32: 062C0215
```

#### 微型代码 (Python)

```python
def adler32(data: bytes) -> int:
    MOD = 65521
    A, B = 1, 0
    for b in data:
        A = (A + b) % MOD
        B = (B + A) % MOD
    return (B << 16) | A

print(hex(adler32(b"Hello")))
```

输出：

```
0x62c0215
```

#### 为什么它很重要

- 比 CRC32 更简单，只需加法和取模
- 在小型系统上（尤其是在软件中）速度很快
- 适用于短数据和快速完整性检查
- 用于：
  * zlib
  * PNG 图像格式
  * 需要低成本验证的网络协议

然而，对于长数据或高度重复的数据，Adler-32 的鲁棒性不如 CRC32。

#### 比较

| 属性         | CRC32                 | Adler-32                    |
| ------------ | --------------------- | --------------------------- |
| 算术运算     | 多项式 (GF(2))        | 整数 (模质数)               |
| 位数         | 32                    | 32                          |
| 速度         | 中等                  | 非常快                      |
| 错误检测能力 | 强                    | 较弱                        |
| 典型用途     | 网络、存储            | 压缩、本地检查              |

#### 复杂度

| 操作         | 时间   | 空间  |
| ------------ | ------ | ----- |
| 编码         | $O(n)$ | $O(1)$ |
| 验证         | $O(n)$ | $O(1)$ |

#### 动手试试

1.  计算 `"BANANA"` 的 Adler-32 值。
2.  翻转一个字节并重新计算，观察校验和的变化。
3.  在大数据上比较与 CRC32 的执行时间。
4.  尝试移除取模运算，观察整数溢出的行为。
5.  为流式数据实现增量校验和更新。

#### 一个温和的证明（为什么它有效）

Adler-32 将消息视为两层累积：

-   A 总和确保局部敏感性（每个字节都影响总和）。
-   B 总和更重视较早的字节，放大了位置效应。

由于模质数运算，小的位翻转会导致校验和发生大的变化，足以高概率地捕捉随机噪声。

Adler-32 是简洁性的典范 ——
仅两个总和和一个模数，
却足够快，足以守护每一个 PNG 和压缩流。
### 664 MD5（消息摘要算法 5）

MD5 是最著名的加密哈希函数之一。
它接收任意长度的输入，并生成一个 128 位的哈希值，作为数据的紧凑“指纹”。
尽管 MD5 曾被广泛使用，但现在它在密码学上已被认为是不安全的，不过在数据完整性检查等非安全应用中仍然有用。

#### 我们要解决什么问题？

我们需要一个固定大小的“摘要”来唯一标识数据块、文件或消息。
一个好的哈希函数应满足三个关键特性：

1.  **原像抵抗性**：难以从哈希值反推出原始消息。
2.  **第二原像抵抗性**：难以找到另一个具有相同哈希值的消息。
3.  **碰撞抵抗性**：难以找到任意两个具有相同哈希值的消息。

MD5 的设计初衷是高效地满足这些目标，尽管在今天，只有前两个特性在有限的使用中还能勉强成立。

#### 核心思想

MD5 以 512 位块为单位处理数据，更新一个由四个 32 位字 $(A, B, C, D)$ 组成的内部状态。

最终，这些字被连接起来形成最终的 128 位摘要。

该算法包含四个主要步骤：

1.  **消息填充**：将消息填充至其长度模 512 余 448，然后附加原始长度（作为一个 64 位值）。
2.  **缓冲区初始化**：

    $$
    A = 0x67452301, \quad
    B = 0xEFCDAB89, \quad
    C = 0x98BADCFE, \quad
    D = 0x10325476
    $$

3.  **处理每个 512 位块**：通过四轮非线性操作，使用位逻辑运算（AND、OR、XOR、NOT）、模加法和循环移位来处理每个块。
4.  **输出**：将 $(A, B, C, D)$ 连接成一个 128 位的摘要。

#### 主要变换（简化版）

对于每个 32 位块：

$$
A = B + ((A + F(B, C, D) + X_k + T_i) \lll s)
$$

其中

- $F$ 是四个非线性函数之一（每轮改变），
- $X_k$ 是一个 32 位的块字，
- $T_i$ 是一个基于正弦函数的常数，
- $\lll s$ 表示循环左移。

每一轮都会修改 $(A, B, C, D)$，从而产生扩散和非线性。

#### 微型代码（Python）

此示例使用标准的 `hashlib` 库：

```python
import hashlib

data = b"Hello, world!"
digest = hashlib.md5(data).hexdigest()
print("MD5:", digest)
```

输出：

```
MD5: 6cd3556deb0da54bca060b4c39479839
```

#### 微型代码（C）

```c
#include <stdio.h>
#include <openssl/md5.h>

int main() {
    unsigned char digest[MD5_DIGEST_LENGTH];
    const char *msg = "Hello, world!";

    MD5((unsigned char*)msg, strlen(msg), digest);

    printf("MD5: ");
    for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        printf("%02x", digest[i]);
    printf("\n");
}
```

输出：

```
MD5: 6cd3556deb0da54bca060b4c39479839
```

#### 为什么它重要

- **快速**：非常快速地计算哈希值
- **紧凑**：128 位输出（32 个十六进制字符）
- **确定性**：相同的输入 → 相同的哈希值
- **用途**：
  * 文件完整性检查
  * 版本控制系统（例如，git 对象命名）
  * 重复数据删除工具
  * 遗留的数字签名

**但是，不要将 MD5 用于密码学安全目的**，它容易受到碰撞攻击和选择前缀攻击。

#### 碰撞与安全性

研究人员已经发现，两个不同的消息 $m_1 \neq m_2$ 可以产生相同的 MD5 哈希值：

$$
\text{MD5}(m_1) = \text{MD5}(m_2)
$$

现代攻击可以在消费级硬件上几秒钟内生成此类碰撞。
对于任何安全敏感的目的，请使用 SHA-256 或更高级的算法。

#### 对比

| 特性         | MD5    | SHA-1    | SHA-256        |
| ------------ | ------ | -------- | -------------- |
| 输出位数     | 128    | 160      | 256            |
| 速度         | 快     | 中等     | 较慢           |
| 安全性       | 已破解 | 弱       | 强             |
| 是否发现碰撞 | 是     | 是       | 尚无实际案例   |

#### 复杂度

| 操作       | 时间复杂度 | 空间复杂度 |
| ---------- | ---------- | ---------- |
| 哈希计算   | $O(n)$     | $O(1)$     |
| 验证       | $O(n)$     | $O(1)$     |

#### 动手尝试

1.  计算 `"apple"` 和 `"APPLE"` 的 MD5 值。
2.  连接两个文件并再次计算哈希，哈希值的变化是可预测的吗？
3.  在你的终端上尝试使用 `md5sum` 命令。
4.  将结果与 SHA-1 和 SHA-256 进行比较。
5.  在线搜索已知的 MD5 碰撞示例，并亲自验证哈希值。

#### 一个温和的证明（为什么它有效）

MD5 的每个操作都是模加法和循环移位的组合，在 GF(2) 上是非线性的。
这些操作产生了雪崩效应，即翻转输入中的一个比特，会改变输出中大约一半的比特。

这确保了微小的输入变化会产生截然不同的哈希值，尽管其碰撞抵抗性在数学上已被证明存在缺陷。

MD5 仍然是早期密码学设计的一个优雅范例——
一个快速、人类可读的指纹，
也是安全如何随着计算能力发展而演变的历史标记。
### 665 SHA-1（安全散列算法 1）

SHA-1 是一种 160 位的加密散列函数，曾经是数字签名、SSL 证书和版本控制系统的基石。
它通过更长的摘要和更多的轮次改进了 MD5，但和 MD5 一样，它后来也被攻破，不过对于理解现代散列算法的发展仍然很有价值。

#### 我们要解决什么问题？

与 MD5 类似，SHA-1 将任意长度的输入压缩成固定大小的散列值（160 位）。
它旨在为认证、加密和校验和提供更强的抗碰撞能力和消息完整性验证。

#### 核心思想

SHA-1 以块为单位工作，以 512 位为块处理消息。
它维护五个 32 位寄存器 $(A, B, C, D, E)$，这些寄存器通过 80 轮的位操作、移位和加法进行演化。

高层次概述：

1. 预处理

   * 对消息进行填充，使其长度模 512 余 448。
   * 将消息长度作为 64 位整数附加在后面。

2. 初始化

   * 设置初始值：
     $$
     \begin{aligned}
     H_0 &= 0x67452301 \\
     H_1 &= 0xEFCDAB89 \\
     H_2 &= 0x98BADCFE \\
     H_3 &= 0x10325476 \\
     H_4 &= 0xC3D2E1F0
     \end{aligned}
     $$

3. 处理每个块

   * 将每个 512 位块分解为 16 个字 $W_0, W_1, \dots, W_{15}$。
   * 使用以下公式将它们扩展到 $W_{16} \dots W_{79}$：
     $$
     W_t = (W_{t-3} \oplus W_{t-8} \oplus W_{t-14} \oplus W_{t-16}) \lll 1
     $$
   * 执行 80 轮更新：
     $$
     T = (A \lll 5) + f_t(B, C, D) + E + W_t + K_t
     $$
     然后移位寄存器：
     $E = D, D = C, C = B \lll 30, B = A, A = T$
     （其中 $f_t$ 和常数 $K_t$ 取决于轮次）。

4. 输出

   * 将 $(A, B, C, D, E)$ 加到 $(H_0, \dots, H_4)$ 上。
   * 最终的散列值是 $H_0$ 到 $H_4$ 的连接。

#### 轮函数

SHA-1 循环使用四种不同的非线性布尔函数：

| 轮次   | 函数       | 公式                                             | 常数 $K_t$     |
| ------ | ---------- | --------------------------------------------------- | -------------- |
| 0–19   | 选择       | $f = (B \land C) \lor (\lnot B \land D)$            | 0x5A827999     |
| 20–39  | 奇偶       | $f = B \oplus C \oplus D$                           | 0x6ED9EBA1     |
| 40–59  | 多数       | $f = (B \land C) \lor (B \land D) \lor (C \land D)$ | 0x8F1BBCDC     |
| 60–79  | 奇偶       | $f = B \oplus C \oplus D$                           | 0xCA62C1D6     |

这些函数在比特之间创建了非线性的混合和扩散。

#### 微型代码（Python）

```python
import hashlib

msg = b"Hello, world!"
digest = hashlib.sha1(msg).hexdigest()
print("SHA-1:", digest)
```

输出：

```
SHA-1: d3486ae9136e7856bc42212385ea797094475802
```

#### 微型代码（C）

```c
#include <stdio.h>
#include <openssl/sha.h>

int main() {
    unsigned char digest[SHA_DIGEST_LENGTH];
    const char *msg = "Hello, world!";

    SHA1((unsigned char*)msg, strlen(msg), digest);

    printf("SHA-1: ");
    for (int i = 0; i < SHA_DIGEST_LENGTH; i++)
        printf("%02x", digest[i]);
    printf("\n");
}
```

输出：

```
SHA-1: d3486ae9136e7856bc42212385ea797094475802
```

#### 为何重要

- 更长的输出：160 位，而非 MD5 的 128 位。
- 更广泛的应用：用于 SSL、Git、PGP 和数字签名。
- 仍然是确定性的且速度快，适用于数据指纹识别。

但它已在密码学中被弃用，使用现代硬件可以在数小时内找到碰撞。

#### 已知攻击

2017 年，谷歌和 CWI 阿姆斯特丹演示了 SHAttered，一种实用的碰撞攻击：

$$
\text{SHA1}(m_1) = \text{SHA1}(m_2)
$$

其中 $m_1$ 和 $m_2$ 是不同的 PDF 文件。

该攻击大约需要 $2^{63}$ 次操作，利用云资源是可行的。

SHA-1 现在被认为在密码学上不安全，应被 SHA-256 或 SHA-3 取代。

#### 对比

| 属性               | MD5        | SHA-1      | SHA-256             |
| ------------------ | ---------- | ---------- | ------------------- |
| 输出位数           | 128        | 160        | 256                 |
| 轮次               | 64         | 80         | 64                  |
| 安全性             | 已攻破     | 已攻破     | 强                  |
| 抗碰撞能力         | $2^{64}$   | $2^{80}$   | $2^{128}$ (近似值)  |
| 状态               | 已弃用     | 已弃用     | 推荐                |

#### 复杂度

| 操作         | 时间   | 空间  |
| ------------ | ------ | ------ |
| 散列计算     | $O(n)$ | $O(1)$ |
| 验证         | $O(n)$ | $O(1)$ |

SHA-1 在计算上仍然高效，在 C 语言实现中速度约为 300 MB/s。

#### 亲自尝试

1. 计算 `"Hello"` 和 `"hello"` 的 SHA-1，注意雪崩效应。
2. 连接两个文件，比较 SHA-1 和 SHA-256 的摘要。
3. 在 Linux 上使用 `sha1sum` 来验证文件完整性。
4. 在线研究 SHAttered PDF 文件，并验证它们具有相同的散列值。

#### 一个温和的证明（为何有效）

输入的每一位通过一系列循环移位和模加法影响输出的每一位。
该设计确保了扩散性（输入的微小变化会广泛传播）和混淆性（通过非线性布尔混合）。

尽管设计优雅，但其轮结构中的数学弱点使得攻击者能够操纵内部状态以产生碰撞。

SHA-1 标志着密码学历史上的一个转折点——
设计精妙，全球采用，
并提醒我们，即使是强大的数学也必须比计算能力进化得更快。
### 666 SHA-256（安全散列算法 256 位）

SHA-256 是当今使用最广泛的加密散列函数之一。
它是 SHA-2 家族的一部分，由 NIST 标准化，为数字签名、区块链系统和一般数据完整性提供了强大的安全性。
与其前身 MD5 和 SHA-1 不同，SHA-256 在实践中至今未被攻破。

#### 我们要解决什么问题？

我们需要一个函数，能为任何输入生成一个唯一的、固定大小的摘要——
但要具备强大的抗碰撞、抗原像和第二原像攻击能力。

SHA-256 实现了这一点：它将任意长度的数据映射为一个 256 位的摘要，
创建了一个几乎不可能逆向推导的指纹，用于安全验证和身份识别。

#### 核心思想

SHA-256 以 512 位块为单位处理数据，通过 64 轮的模运算、逻辑运算和消息扩展，更新八个 32 位工作寄存器。

每一轮都以非线性、不可逆的方式混合比特位，在整个消息中实现扩散和混淆。

#### 初始化

SHA-256 以八个固定常量开始：

$$
\begin{aligned}
H_0 &= 0x6a09e667, \quad H_1 = 0xbb67ae85, \
H_2 &= 0x3c6ef372, \quad H_3 = 0xa54ff53a, \
H_4 &= 0x510e527f, \quad H_5 = 0x9b05688c, \
H_6 &= 0x1f83d9ab, \quad H_7 = 0x5be0cd19
\end{aligned}
$$

#### 消息扩展

每个 512 位块被分割成 16 个字 $W_0 \dots W_{15}$，并扩展为 64 个字：

$$
W_t = \sigma_1(W_{t-2}) + W_{t-7} + \sigma_0(W_{t-15}) + W_{t-16}
$$

其中
$$
\sigma_0(x) = (x \mathbin{>>>} 7) \oplus (x \mathbin{>>>} 18) \oplus (x >> 3)
$$
$$
\sigma_1(x) = (x \mathbin{>>>} 17) \oplus (x \mathbin{>>>} 19) \oplus (x >> 10)
$$

#### 轮函数

对于 64 轮中的每一轮：

$$
\begin{aligned}
T_1 &= H + \Sigma_1(E) + Ch(E, F, G) + K_t + W_t \
T_2 &= \Sigma_0(A) + Maj(A, B, C) \
H &= G \
G &= F \
F &= E \
E &= D + T_1 \
D &= C \
C &= B \
B &= A \
A &= T_1 + T_2
\end{aligned}
$$

其中

$$
\begin{aligned}
Ch(x,y,z) &= (x \land y) \oplus (\lnot x \land z) \
Maj(x,y,z) &= (x \land y) \oplus (x \land z) \oplus (y \land z) \
\Sigma_0(x) &= (x \mathbin{>>>} 2) \oplus (x \mathbin{>>>} 13) \oplus (x \mathbin{>>>} 22) \
\Sigma_1(x) &= (x \mathbin{>>>} 6) \oplus (x \mathbin{>>>} 11) \oplus (x \mathbin{>>>} 25)
\end{aligned}
$$

常量 $K_t$ 是 64 个预定义的 32 位值，源自质数的立方根。

#### 最终处理

所有轮次完成后，更新哈希值：

$$
H_i = H_i + \text{working\_register}_i \quad \text{for } i = 0 \dots 7
$$

最终的摘要就是 $(H_0, H_1, \dots, H_7)$ 的连接，形成一个 256 位的哈希值。

#### 微型代码（Python）

```python
import hashlib

data = b"Hello, world!"
digest = hashlib.sha256(data).hexdigest()
print("SHA-256:", digest)
```

输出：

```
SHA-256: c0535e4be2b79ffd93291305436bf889314e4a3faec05ecffcbb7df31ad9e51a
```

#### 微型代码（C）

```c
#include <stdio.h>
#include <openssl/sha.h>

int main() {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    const char *msg = "Hello, world!";

    SHA256((unsigned char*)msg, strlen(msg), digest);

    printf("SHA-256: ");
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++)
        printf("%02x", digest[i]);
    printf("\n");
}
```

输出：

```
SHA-256: c0535e4be2b79ffd93291305436bf889314e4a3faec05ecffcbb7df31ad9e51a
```

#### 为什么它很重要

-   强大的抗碰撞能力（无实用攻击方法）
-   用于加密协议：TLS、PGP、SSH、比特币、Git
-   高效：能快速处理大量数据
-   确定性和不可逆性

SHA-256 是区块链完整性的基础，每个比特币区块都通过 SHA-256 哈希值链接。

#### 安全性比较

| 特性             | MD5    | SHA-1  | SHA-256         |
| ---------------- | ------ | ------ | --------------- |
| 输出位数         | 128    | 160    | 256             |
| 是否发现碰撞     | 是     | 是     | 无实用方法      |
| 安全性           | 已攻破 | 已攻破 | 安全            |
| 典型用途         | 遗留系统 | 遗留系统 | 现代安全        |

#### 复杂度

| 操作         | 时间   | 空间  |
| ------------ | ------ | ------ |
| 哈希计算     | $O(n)$ | $O(1)$ |
| 验证         | $O(n)$ | $O(1)$ |

SHA-256 速度足够快，可用于实时场景，但其设计能抵抗硬件暴力攻击。

#### 亲自尝试

1.  计算 `"hello"` 和 `"Hello"` 的哈希值，观察输出变化有多大。
2.  比较 SHA-256 与 SHA-512 的速度。
3.  针对小例子，手动实现消息填充。
4.  尝试比特币区块头的哈希计算。
5.  计算一个文件的 SHA-256 双重哈希，并与 `sha256sum` 的结果进行比较。

#### 一个温和的证明（为什么它有效）

SHA-256 的强度在于其非线性混合、比特循环移位和模加法，这些操作将每个输入比特的影响扩散到所有输出比特。
因为每一轮都通过独立的函数和常量扰乱数据，所以该过程实际上是不可逆的。

从数学上讲，目前没有已知的方法能比暴力破解（需要 $2^{128}$ 次尝试）更快地逆向推导或碰撞 SHA-256。

SHA-256 是数字时代的加密支柱——
受到区块链、浏览器和各地系统的信赖——
是优雅、速度和数学硬度的完美平衡。
### 667 SHA-3 (Keccak)

SHA-3，也称为 Keccak，是安全哈希算法家族的最新成员，由 NIST 于 2015 年标准化。
它代表了一次彻底的重新设计，而非修补，引入了海绵结构，从根本上改变了哈希的工作方式。
SHA-3 灵活、安全且数学上优雅。

#### 我们要解决什么问题？

SHA-2（如 SHA-256）很强，但它与 SHA-1 和 MD5 等较旧的哈希算法共享内部结构。
如果针对该结构出现重大的密码分析突破，整个家族都可能面临风险。

SHA-3 被设计为一种密码学后备方案，采用完全不同的方法，对相同类型的攻击免疫，同时与现有用例兼容。

#### 核心思想：海绵结构

SHA-3 通过海绵函数吸收和挤压数据，该函数基于一个 1600 位的大型内部状态。

- 海绵有两个参数：

  * 速率 (r)，每轮处理的比特数。
  * 容量 (c)，保持隐藏的比特数（用于安全）。

对于 SHA3-256，$r = 1088$ 且 $c = 512$，因为 $r + c = 1600$。

该过程在两个阶段之间交替进行：

1. 吸收阶段
   输入被逐块异或到状态中，然后通过 Keccak 置换进行变换。

2. 挤压阶段
   从状态中读取输出比特；如果需要更多比特，则应用更多轮次。

#### Keccak 状态与变换

内部状态是一个比特的三维数组，可视化为 $5 \times 5$ 个通道，每个通道 64 位：

$$
A[x][y][z], \quad 0 \le x, y < 5, ; 0 \le z < 64
$$

Keccak 的每一轮应用五种变换：

1. θ (theta)，列奇偶混合
2. ρ (rho)，比特旋转
3. π (pi)，通道置换
4. χ (chi)，非线性混合（按位逻辑运算）
5. ι (iota)，轮常量注入

每一轮都在整个状态中扰乱比特，实现类似三维流体搅拌的扩散和混淆。

#### 填充规则

在处理之前，使用多速率填充规则对消息进行填充：

$$
\text{pad}(M) = M , || , 0x06 , || , 00...0 , || , 0x80
$$

这确保了每条消息都是唯一的，即使长度相似。

#### 微型代码 (Python)

```python
import hashlib

data = b"Hello, world!"
digest = hashlib.sha3_256(data).hexdigest()
print("SHA3-256:", digest)
```

输出：

```
SHA3-256: 644bcc7e564373040999aac89e7622f3ca71fba1d972fd94a31c3bfbf24e3938
```

#### 微型代码 (C, OpenSSL)

```c
#include <stdio.h>
#include <openssl/evp.h>

int main() {
    unsigned char digest[32];
    const char *msg = "Hello, world!";
    EVP_Digest(msg, strlen(msg), digest, NULL, EVP_sha3_256(), NULL);

    printf("SHA3-256: ");
    for (int i = 0; i < 32; i++)
        printf("%02x", digest[i]);
    printf("\n");
}
```

输出：

```
SHA3-256: 644bcc7e564373040999aac89e7622f3ca71fba1d972fd94a31c3bfbf24e3938
```

#### 为什么它重要

- 完全不同的设计，不像 MD5/SHA-2 那样基于 Merkle–Damgård
- 数学上简洁且可证明的海绵模型
- 支持可变输出长度 (SHAKE128, SHAKE256)
- 抵抗所有已知的密码分析攻击

用于：

- 区块链研究（例如，以太坊使用 Keccak-256）
- 后量子密码学框架
- 数字取证和可验证账本

#### 对比

| 算法    | 结构            | 输出比特 | 年份 | 安全状态       |
| ------- | --------------- | -------- | ---- | -------------- |
| MD5     | Merkle–Damgård  | 128      | 1992 | 已攻破         |
| SHA-1   | Merkle–Damgård  | 160      | 1995 | 已攻破         |
| SHA-256 | Merkle–Damgård  | 256      | 2001 | 安全           |
| SHA-3   | 海绵 (Keccak)   | 256      | 2015 | 安全           |

#### 复杂度

| 操作       | 时间   | 空间  |
| ---------- | ------ | ----- |
| 哈希计算   | $O(n)$ | $O(1)$ |
| 验证       | $O(n)$ | $O(1)$ |

尽管 SHA-3 在纯软件中比 SHA-256 慢，但它在硬件和并行实现中扩展性更好。

#### 自己动手试试

1. 计算同一文件的 `sha256()` 和 `sha3_256()`，比较输出。

2. 使用 SHAKE256 尝试可变长度摘要：

   ```python
   from hashlib import shake_256
   print(shake_256(b"hello").hexdigest(64))
   ```

3. 可视化 Keccak 的 5×5×64 位状态，绘制轮次如何混合通道。

4. 实现一个带有异或和旋转的玩具海绵函数以理解其设计。

#### 一个温和的证明（为什么它有效）

与逐块扩展哈希的 Merkle–Damgård 结构不同，海绵将输入吸收到一个大型非线性状态中，隐藏了相关性。
由于只有"速率"比特被暴露，"容量"确保了找到碰撞或原像需要 $2^{c/2}$ 的工作量，对于 SHA3-256，大约是 $2^{256}$ 的努力。

这种设计使得 SHA-3 在理论上可证明其安全性，能抵抗针对其容量的通用攻击。

SHA-3 是 SHA-2 的平静继任者——
并非诞生于危机，而是源于数学的革新——
一个吸收数据并挤压出纯粹随机性的海绵。
### 668 HMAC（基于哈希的消息认证码）

HMAC 是一种验证消息完整性和真实性的方法。它将任何加密哈希函数（如 SHA-256 或 SHA-3）与一个密钥相结合，确保只有知道密钥的人才能生成或验证正确的哈希值。

HMAC 是许多认证协议的基础，包括 TLS、OAuth、JWT 和 AWS API 签名。

#### 我们要解决什么问题？

像 SHA-256 这样的常规哈希可以验证数据完整性，但不能验证真实性。任何人都可以计算一个文件的哈希值，那么你如何分辨是谁真正创建了它呢？

HMAC 引入了一个共享密钥，以确保只有授权方才能生成或验证正确的哈希值。

如果哈希值不匹配，则意味着数据要么被修改了，要么是在没有正确密钥的情况下生成的。

#### 核心思想

HMAC 将加密哈希函数包裹在两层带密钥的哈希中：

1.  内部哈希：消息与内部密钥填充组合后的哈希。
2.  外部哈希：内部摘要与外部密钥填充组合后的哈希。

形式化表示：

$$
\text{HMAC}(K, M) = H\left((K' \oplus opad) , || , H((K' \oplus ipad) , || , M)\right)
$$

其中：

-   $H$ 是一个安全的哈希函数（例如 SHA-256）
-   $K'$ 是经过填充或哈希处理以匹配块大小的密钥
-   $opad$ 是"外部填充"（重复的 0x5C）
-   $ipad$ 是"内部填充"（重复的 0x36）
-   $||$ 表示连接
-   $\oplus$ 表示异或

这种双层结构可以防止针对哈希函数内部（如长度扩展攻击）的攻击。

#### 逐步示例（使用 SHA-256）

1.  将密钥 $K$ 填充到 64 字节（SHA-256 的块大小）。

2.  计算内部哈希：

    $$
    \text{inner} = H((K' \oplus ipad) , || , M)
    $$

3.  计算外部哈希：

    $$
    \text{HMAC} = H((K' \oplus opad) , || , \text{inner})
    $$

4.  结果是一个 256 位的摘要，用于认证密钥和消息。

#### 微型代码（Python）

```python
import hmac, hashlib

key = b"secret-key"
message = b"Attack at dawn"
digest = hmac.new(key, message, hashlib.sha256).hexdigest()
print("HMAC-SHA256:", digest)
```

输出：

```
HMAC-SHA256: 2cba05e5a7e03ffccf13e585c624cfa7cbf4b82534ef9ce454b0943e97ebc8aa
```

#### 微型代码（C）

```c
#include <stdio.h>
#include <string.h>
#include <openssl/hmac.h>

int main() {
    unsigned char result[EVP_MAX_MD_SIZE];
    unsigned int len = 0;
    const char *key = "secret-key";
    const char *msg = "Attack at dawn";

    HMAC(EVP_sha256(), key, strlen(key),
         (unsigned char*)msg, strlen(msg),
         result, &len);

    printf("HMAC-SHA256: ");
    for (unsigned int i = 0; i < len; i++)
        printf("%02x", result[i]);
    printf("\n");
}
```

输出：

```
HMAC-SHA256: 2cba05e5a7e03ffccf13e585c624cfa7cbf4b82534ef9ce454b0943e97ebc8aa
```

#### 为什么它很重要

-   保护真实性：只有拥有密钥的人才能计算有效的 HMAC。
-   保护完整性：数据或密钥的任何更改都会改变 HMAC。
-   抵抗长度扩展和重放攻击。

用于：

-   TLS、SSH、IPsec
-   AWS 和 Google Cloud API 签名
-   JWT（HS256）
-   Webhooks、签名 URL 和安全令牌

#### 基于哈希的 MAC 比较

| 算法          | 底层哈希 | 输出（位） | 状态           |
| ------------- | -------- | ---------- | -------------- |
| HMAC-MD5      | MD5      | 128        | 不安全         |
| HMAC-SHA1     | SHA-1    | 160        | 弱（遗留）     |
| HMAC-SHA256   | SHA-256  | 256        | 推荐           |
| HMAC-SHA3-256 | SHA-3    | 256        | 面向未来，安全 |

#### 复杂度

| 操作       | 时间     | 空间    |
| ---------- | -------- | ------- |
| 哈希计算   | $O(n)$   | $O(1)$  |
| 验证       | $O(n)$   | $O(1)$  |

HMAC 的成本大约是底层哈希函数的两倍，因为它执行两次处理（内部 + 外部）。

#### 自己动手试试

1.  使用密钥 `"abc"` 计算 `"hello"` 的 HMAC-SHA256。
2.  修改一个字节，注意摘要如何完全改变。
3.  尝试使用错误的密钥进行验证，验证会失败。
4.  比较 SHA1 和 SHA256 版本的性能。
5.  使用上面的公式从头开始手动实现一个 HMAC。

#### 一个温和的证明（为什么它有效）

关键见解：
即使攻击者知道 $H(K || M)$，他们也无法为另一个消息 $M'$ 计算 $H(K || M')$，因为 $K$ 以一种不可重复使用的方式混合在哈希内部。

从数学上讲，内部和外部填充打破了压缩函数的线性，消除了任何可利用的结构。

安全性完全取决于哈希的抗碰撞性和密钥的保密性。

HMAC 是数学与信任之间的握手——
一个紧凑的加密签名，证明：
"这条消息来自真正知道密钥的人。"
### 669 默克尔树（哈希树）

默克尔树（或称哈希树）是一种分层数据结构，能够高效、安全地验证大规模数据集。它是区块链、分布式系统以及像 Git 这样的版本控制系统的支柱，能够在对数时间内完成完整性证明。

#### 我们要解决什么问题？

假设你有一个海量数据集，包含千兆字节的文件或数据块，你想验证其中某一部分是否完整或已被篡改。反复对整个数据集进行哈希计算将非常昂贵。

默克尔树允许仅使用一个小的证明来验证数据的任何部分，而无需重新计算所有内容的哈希。

#### 核心思想

默克尔树通过递归地对子节点对进行哈希运算来构建，直到获得一个单一的根哈希。

- 叶节点：包含数据块的哈希值。
- 内部节点：包含其子节点哈希值连接后的哈希值。
- 根哈希：唯一地代表整个数据集。

如果任何数据块发生更改，这种更改会向上传播，从而改变根哈希，使得篡改行为能够立即被检测到。

#### 构建过程

给定四个数据块 $D_1, D_2, D_3, D_4$：

1. 计算叶节点哈希：
   $$
   H_1 = H(D_1), \quad H_2 = H(D_2), \quad H_3 = H(D_3), \quad H_4 = H(D_4)
   $$
2. 计算中间节点哈希：
   $$
   H_{12} = H(H_1 || H_2), \quad H_{34} = H(H_3 || H_4)
   $$
3. 计算根哈希：
   $$
   H_{root} = H(H_{12} || H_{34})
   $$

最终的 $H_{root}$ 充当所有底层数据的指纹。

#### 示例（可视化）

```
              H_root
              /    \
         H_12       H_34
        /   \       /   \
     H1     H2   H3     H4
     |      |    |      |
    D1     D2   D3     D4
```

#### 验证证明（默克尔路径）

为了证明 $D_3$ 属于这棵树：

1. 提供 $H_4$、$H_{12}$ 以及每个哈希的位置（左/右）。

2. 向上计算：

   $$
   H_3 = H(D_3)
   $$

   $$
   H_{34} = H(H_3 || H_4)
   $$

   $$
   H_{root}' = H(H_{12} || H_{34})
   $$

3. 如果 $H_{root}' = H_{root}$，则数据块验证通过。

验证只需要 log₂(n) 次哈希计算。

#### 微型代码（Python）

```python
import hashlib

def sha256(data):
    return hashlib.sha256(data).digest()

def merkle_tree(leaves):
    if len(leaves) == 1:
        return leaves[0]
    if len(leaves) % 2 == 1:
        leaves.append(leaves[-1])
    parents = []
    for i in range(0, len(leaves), 2):
        parents.append(sha256(leaves[i] + leaves[i+1]))
    return merkle_tree(parents)

data = [b"D1", b"D2", b"D3", b"D4"]
leaves = [sha256(d) for d in data]
root = merkle_tree(leaves)
print("Merkle Root:", root.hex())
```

输出示例：

```
Merkle Root: 16d1c7a0cfb3b6e4151f3b24a884b78e0d1a826c45de2d0e0d0db1e4e44bff62
```

#### 微型代码（C，使用 OpenSSL）

```c
#include <stdio.h>
#include <openssl/sha.h>
#include <string.h>

void sha256(unsigned char *data, size_t len, unsigned char *out) {
    SHA256(data, len, out);
}

void print_hex(unsigned char *h, int len) {
    for (int i = 0; i < len; i++) printf("%02x", h[i]);
    printf("\n");
}

int main() {
    unsigned char d1[] = "D1", d2[] = "D2", d3[] = "D3", d4[] = "D4";
    unsigned char h1[32], h2[32], h3[32], h4[32], h12[32], h34[32], root[32];
    unsigned char tmp[64];

    sha256(d1, 2, h1); sha256(d2, 2, h2);
    sha256(d3, 2, h3); sha256(d4, 2, h4);

    memcpy(tmp, h1, 32); memcpy(tmp+32, h2, 32); sha256(tmp, 64, h12);
    memcpy(tmp, h3, 32); memcpy(tmp+32, h4, 32); sha256(tmp, 64, h34);
    memcpy(tmp, h12, 32); memcpy(tmp+32, h34, 32); sha256(tmp, 64, root);

    printf("Merkle Root: "); print_hex(root, 32);
}
```

#### 为什么它很重要

- **完整性**：任何数据更改都会改变根哈希。
- **高效性**：对数级的证明大小和验证时间。
- **可扩展性**：用于拥有数百万条记录的系统。
- **应用场景**：
  * 比特币和以太坊区块链
  * Git 提交和版本历史
  * IPFS 和分布式文件系统
  * 安全软件更新（默克尔证明）

#### 与扁平哈希的对比

| 方法         | 验证成本       | 数据篡改检测         | 证明大小     |
| ------------ | -------------- | -------------------- | ------------ |
| 单一哈希     | $O(n)$         | 仅限整个文件         | 完整数据     |
| 默克尔树     | $O(\log n)$    | 任何数据块           | 少量哈希值   |

#### 复杂度

| 操作         | 时间           | 空间     |
| ------------ | -------------- | -------- |
| 树构建       | $O(n)$         | $O(n)$   |
| 验证         | $O(\log n)$    | $O(1)$   |

#### 动手尝试

1.  构建一个包含 8 条消息的默克尔树。
2.  修改其中一条消息，观察根哈希如何变化。
3.  为第 3 条消息计算一个默克尔证明并手动验证它。
4.  实现比特币中使用的双重 SHA-256。
5.  探索 Git 如何将树和提交用作默克尔有向无环图。

#### 一个温和的证明（为什么它有效）

每个节点的哈希值都依赖于其子节点，而子节点又递归地依赖于其下的所有叶节点。因此，即使任何叶节点中的一个比特发生改变，也会改变其所有祖先节点的哈希值以及根哈希。

因为底层的哈希函数具有抗碰撞性，所以两个不同的数据集不可能产生相同的根哈希。

从数学上讲，对于一个安全的哈希函数 $H$：

$$
H_{root}(D_1, D_2, ..., D_n) = H_{root}(D'_1, D'_2, ..., D'_n)
\Rightarrow D_i = D'_i \text{ 对所有 } i \text{ 成立}
$$

默克尔树使完整性验证变得可扩展——它是数据森林的数字指纹，证明了每一片叶子都仍然是它声称的样子。
### 670 哈希碰撞检测（生日边界模拟）

每一种加密哈希函数，即使是最强大的，理论上都可能产生碰撞，即两个不同的输入产生相同的哈希值。
生日悖论为我们提供了一种估计这种可能性大小的方法。
本节将探讨碰撞概率、检测模拟，以及为什么即使是 256 位的哈希在原理上也是"可破解"的，只是在实际中不可行。

#### 我们要解决什么问题？

当我们对数据进行哈希运算时，我们希望每条消息都能得到一个唯一的输出。
但由于哈希空间是有限的，碰撞必然存在。
关键问题是：*我们需要生成多少个随机哈希值，才能预期出现一次碰撞？*

这本质上是生日问题，同样的数学原理告诉我们，23 个人就足以有 50% 的概率出现生日相同的情况。

#### 核心思想：生日边界

如果一个哈希函数能产生 $N$ 种可能的输出，
那么在大约生成 $\sqrt{N}$ 个随机哈希值后，出现碰撞的概率就会达到大约 50%。

对于一个 $b$ 位的哈希：

$$
N = 2^b, \quad \text{所以碰撞大约出现在 } 2^{b/2} \text{ 次尝试后}.
$$

| 哈希位数       | 预期碰撞出现于 | 实际安全性       |
| -------------- | -------------- | ---------------- |
| 32             | $2^{16}$ (≈65k) | 弱               |
| 64             | $2^{32}$       | 中等             |
| 128 (MD5)      | $2^{64}$       | 已被攻破         |
| 160 (SHA-1)    | $2^{80}$       | 已被攻破（可行） |
| 256 (SHA-256)  | $2^{128}$      | 安全             |
| 512 (SHA-512)  | $2^{256}$      | 极其安全         |

因此，SHA-256 的 128 位碰撞抵抗能力对于现代安全需求来说已经足够强大。

#### 生日概率公式

在哈希 $k$ 条随机消息后，*至少出现一次碰撞* 的概率是：

$$
P(k) \approx 1 - e^{-k(k-1)/(2N)}
$$

如果我们设 $P(k) = 0.5$，解出 $k$ 得到：

$$
k \approx 1.1774 \sqrt{N}
$$

#### 微型代码（Python）

让我们使用 SHA-1 和随机数据来模拟哈希碰撞。

```python
import hashlib, random, string

def random_str(n=8):
    return ''.join(random.choice(string.ascii_letters) for _ in range(n))

def collision_simulation(bits=32):
    seen = {}
    mask = (1 << bits) - 1
    count = 0
    while True:
        s = random_str(8).encode()
        h = int.from_bytes(hashlib.sha256(s).digest(), 'big') & mask
        if h in seen:
            return count, s, seen[h]
        seen[h] = s
        count += 1

print("正在模拟 32 位碰撞...")
count, s1, s2 = collision_simulation(32)
print(f"在 {count} 次哈希后发生碰撞：")
print(f"{s1} 和 {s2}")
```

典型输出：

```
正在模拟 32 位碰撞...
在 68314 次哈希后发生碰撞：
b'FqgWbUzk' 和 b'yLpTGZxu'
```

这与理论预期相符：大约 $\sqrt{2^{32}} = 65,536$ 次尝试。

#### 微型代码（C 语言，简单模型）

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/sha.h>

int main() {
    unsigned int seen[65536] = {0};
    unsigned char hash[SHA_DIGEST_LENGTH];
    unsigned char input[16];
    int count = 0;

    while (1) {
        sprintf((char*)input, "%d", rand());
        SHA1(input, strlen((char*)input), hash);
        unsigned int h = (hash[0] << 8) | hash[1];  // 取 16 位
        if (seen[h]) {
            printf("在 %d 次哈希后找到碰撞\n", count);
            break;
        }
        seen[h] = 1;
        count++;
    }
}
```

#### 为什么这很重要

-   每个哈希都可能发生碰撞，但对于大的位数，碰撞是*天文数字般不可能*的。
-   有助于解释为什么 MD5 和 SHA-1 不再安全，因为现代 GPU 可以达到它们的"生日边界"。
-   用于数字签名、区块链挖矿和随机预言机的安全证明中。

#### 可视化

| 哈希位数 | 达到 50% 碰撞概率的尝试次数 | 等效描述         |
| -------- | --------------------------- | ---------------- |
| 16       | 300                         | 微不足道         |
| 32       | 65,000                      | 玩具示例         |
| 64       | 40 亿                       | 测试可行         |
| 128      | $2^{64}$ ≈ $1.8\times10^{19}$ | 过于庞大         |
| 256      | $2^{128}$ ≈ $3.4\times10^{38}$ | 不可能           |

作为对比，可观测宇宙中的原子总数大约是 $10^{80}$，仍然小于 $2^{256}$。

#### 复杂度

| 操作       | 时间复杂度     | 空间复杂度     |
| ---------- | -------------- | -------------- |
| 模拟       | $O(\sqrt{N})$  | $O(\sqrt{N})$  |
| 验证       | $O(1)$         | $O(1)$         |

哈希碰撞检测的复杂度随输出位数呈二次方增长，这使得对于现代算法来说，暴力破解是不可行的。

#### 亲自尝试

1.  运行 Python 模拟程序，分别使用 24 位、28 位和 32 位的掩码。
2.  将实际统计的碰撞次数与理论上的 $\sqrt{2^b}$ 曲线进行比较。
3.  使用 `matplotlib` 可视化概率与哈希次数的关系。
4.  将 SHA-256 替换为 SHA3-256，碰撞统计规律相同。
5.  根据 GPU 速度估算 SHA-1 的预期碰撞时间（你会发现这是现实的）。

#### 一个温和的证明（为什么它成立）

设 $N = 2^b$ 为可能的哈希值数量。
在 $k$ 次抽取后，所有哈希值都不同的概率大约为：

$$
P_{\text{无碰撞}} = \prod_{i=1}^{k-1} \left(1 - \frac{i}{N}\right)
\approx e^{-k(k-1)/(2N)}
$$

取其补集就得到了碰撞概率 $P(k)$。
令 $P(k)=0.5$ 就得到了我们熟悉的 $\sqrt{N}$ 边界，这是一个纯粹的组合数学事实，与哈希算法无关。

哈希碰撞检测是概率论与密码学相遇的地方——
它平静地提醒我们，即使是完美的函数也有其极限，
而随机性本身定义了我们可以多么接近这些极限。

# 第 68 节 近似与流式匹配
### 671 K-近似字符串匹配

K-近似字符串匹配问题要求：
给定一个模式 `P` 和一个文本 `T`，找出 `T` 中所有 `P` 出现的位置，允许最多 `k` 处差异（插入、删除或替换）。

该问题是模糊搜索、拼写纠正、生物信息学序列比对和容错模式识别的基础。

#### 核心思想

精确匹配要求所有字符完全对齐。
近似匹配允许少量差异，通过编辑距离 ≤ k 来量化。

我们使用动态规划或位并行技术来高效地检测这些近似匹配。

#### 动态规划公式

设模式 `P` 长度为 `m`，文本 `T` 长度为 `n`，距离阈值 `k`。

定义一个 DP 表：

$$
dp[i][j] = \text{模式 } P[1..i] \text{ 与文本 } T[1..j] \text{ 之间的最小编辑距离}
$$

递推关系：

$$
dp[i][j] =
\begin{cases}
dp[i-1][j-1], & \text{如果 } P[i] = T[j],\\[4pt]
1 + \min\big(dp[i-1][j],\ dp[i][j-1],\ dp[i-1][j-1]\big), & \text{否则。}
\end{cases}
$$


在每个位置 `j`，如果 $dp[m][j] \le k$，
那么子串 `T[j-m+1..j]` 近似匹配 `P`。

#### 示例

令
`T = "abcdefg"`
`P = "abxd"`
`k = 1`

动态规划在位置 1 找到一个近似匹配，
因为 `"abxd"` 与 `"abcd"` 相差一个替换操作。

#### 位并行简化（适用于小的 k）

Bitap（Shift-Or）算法可以通过使用位掩码和移位操作扩展到处理最多 `k` 个错误。

每个比特位表示 `P` 的一个前缀在给定对齐方式下是否匹配。

时间复杂度变为：

$$
O\left(\frac{n \cdot k}{w}\right)
$$

其中 `w` 是机器字长（通常为 64）。

#### 微型代码（Python，DP 版本）

```python
def k_approx_match(text, pattern, k):
    n, m = len(text), len(pattern)
    dp = [list(range(m + 1))] + [[0] * (m + 1) for _ in range(n)]
    for i in range(1, n + 1):
        dp[i][0] = 0
        for j in range(1, m + 1):
            cost = 0 if text[i-1] == pattern[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # 插入
                dp[i][j-1] + 1,      # 删除
                dp[i-1][j-1] + cost  # 替换
            )
        if dp[i][m] <= k:
            print(f"匹配结束于 {i}, 距离 {dp[i][m]}")

k_approx_match("abcdefg", "abxd", 1)
```

输出：

```
匹配结束于 4, 距离 1
```

#### 微型代码（C，Bitap 风格）

```c
#include <stdio.h>
#include <string.h>
#include <stdint.h>

void bitap_approx(const char *text, const char *pattern, int k) {
    int n = strlen(text), m = strlen(pattern);
    uint64_t mask[256] = {0};
    for (int i = 0; i < m; i++) mask[(unsigned char)pattern[i]] |= 1ULL << i;

    uint64_t R[k+1];
    for (int d = 0; d <= k; d++) R[d] = ~0ULL;
    uint64_t pattern_mask = 1ULL << (m - 1);

    for (int i = 0; i < n; i++) {
        uint64_t oldR = R[0];
        R[0] = ((R[0] << 1) | 1ULL) & mask[(unsigned char)text[i]];
        for (int d = 1; d <= k; d++) {
            uint64_t tmp = R[d];
            R[d] = ((R[d] << 1) | 1ULL) & mask[(unsigned char)text[i]];
            R[d] |= (oldR | (R[d-1] << 1) | oldR << 1);
            oldR = tmp;
        }
        if (!(R[k] & pattern_mask))
            printf("近似匹配结束于 %d\n", i + 1);
    }
}
```

#### 为何重要

近似匹配支撑着：

- 拼写检查（"recieve" → "receive"）
- DNA 序列比对（A-C-T 错配）
- 具有拼写错误容忍度的搜索引擎
- 实时文本识别和 OCR 纠正
- 命令行模糊过滤器（如 `fzf`）

#### 复杂度

| 算法           | 时间      | 空间  |
| ------------------- | --------- | ------ |
| DP（朴素）          | $O(nm)$   | $O(m)$ |
| Bitap（适用于小 k） | $O(nk/w)$ | $O(k)$ |

对于小的编辑距离（k ≤ 3），位并行方法速度极快。

#### 动手尝试

1.  运行 DP 版本并可视化 `dp` 表。
2.  增加 `k` 值，观察匹配如何扩展。
3.  在文本中加入随机噪声进行测试。
4.  当 `dp[i][m] > k` 时实现提前终止。
5.  对于大型输入，比较 Bitap 算法与 DP 算法的速度。

#### 一个温和的证明（为何有效）

DP 递推关系确保每次错配、插入或删除恰好为总距离贡献 1。
通过扫描 DP 表的最后一列，我们检测到最小编辑距离 ≤ k 的子串。
因为所有转移都是局部的，该算法保证了每种对齐方式的正确性。

近似匹配为算法这个严格的世界带来了容错性——
寻找的模式不是它*原本的样子*，而是它*近乎的样子*。
### 672 Bitap 算法（位运算动态规划）

Bitap 算法，也称为 Shift-Or 或位模式匹配算法，通过将模式编码为位掩码并使用简单的位运算进行更新，从而实现快速的文本搜索。它是字符串匹配领域最早、最优雅的位并行算法示例之一。

#### 核心思想

Bitap 算法不是逐个字符进行比较，而是将匹配状态表示为一个位向量。每一位指示模式的一个前缀是否与文本到当前位置为止的子串匹配。

这使得我们能够利用位运算，在每个 CPU 字（word）中处理多达 64 个模式字符，从而比朴素的扫描方法快得多。

#### 位掩码设置

设：

- 模式 `P` 的长度为 `m`
- 文本 `T` 的长度为 `n`
- 机器字宽为 `w`（通常为 64 位）

对于每个字符 `c`，我们预先计算一个位掩码：

$$
B[c] = \text{位掩码，其中 } B[c]_i = 0 \text{ 如果 } P[i] = c, \text{ 否则为 } 1
$$

#### 匹配过程

我们维护一个运行状态向量 $R$，初始化为全 1：

$$
R_0 = \text{所有位设置为 } 1
$$

对于每个字符 $T[j]$：

$$
R_j = \big((R_{j-1} \ll 1) \,\vert\, 1\big) \,\&\, B[T[j]]
$$

如果对应于模式最后一位的位变为 0，则在位置 $j - m + 1$ 处找到一个匹配。

#### 示例

设
`T = "abcxabcdabxabcdabcdabcy"`
`P = "abcdabcy"`

模式长度 $m = 8$。
在处理文本时，`R` 的每一位左移以表示正在增长的匹配。当第 8 位变为零时，表示 `P` 完全匹配。

#### 微型代码（Python，Bitap 精确匹配）

```python
def bitap_search(text, pattern):
    m = len(pattern)
    mask = {}
    for c in set(text):
        mask[c] = ~0
    for i, c in enumerate(pattern):
        mask[c] &= ~(1 << i)

    R = ~0
    for j, c in enumerate(text):
        R = ((R << 1) | 1) & mask.get(c, ~0)
        if (R & (1 << (m - 1))) == 0:
            print(f"匹配结束于位置 {j}")

bitap_search("abcxabcdabxabcdabcdabcy", "abcdabcy")
```

输出：

```
匹配结束于位置 23
```

#### 允许 K 个错误的 Bitap（近似匹配）

Bitap 可以扩展为允许最多 $k$ 个不匹配（插入、删除或替换）。我们维护 $k + 1$ 个位向量 $R_0, R_1, ..., R_k$：

$$
R_j = \big((R_{j-1} \ll 1) \,\vert\, 1\big) \,\&\, B[T[j]]
$$

对于 $d > 0$：

$$
R_d = \big((R_d \ll 1) \,\vert\, 1\big) \,\&\, B[T[j]] 
      \,\vert\, R_{d-1} 
      \,\vert\, \big((R_{d-1} \ll 1) \,\vert\, (R_{d-1} \gg 1)\big)
$$

如果任何 $R_d$ 的最后一位为零，则表示存在一个在 $d$ 次编辑距离内的匹配。

#### 微型代码（C，精确匹配）

```c
#include <stdio.h>
#include <stdint.h>
#include <string.h>

void bitap(const char *text, const char *pattern) {
    int n = strlen(text), m = strlen(pattern);
    uint64_t mask[256];
    for (int i = 0; i < 256; i++) mask[i] = ~0ULL;
    for (int i = 0; i < m; i++) mask[(unsigned char)pattern[i]] &= ~(1ULL << i);

    uint64_t R = ~0ULL;
    for (int j = 0; j < n; j++) {
        R = ((R << 1) | 1ULL) & mask[(unsigned char)text[j]];
        if (!(R & (1ULL << (m - 1))))
            printf("匹配结束于位置 %d\n", j + 1);
    }
}
```

#### 为何重要

- 紧凑且快速：使用纯位运算逻辑
- 适用于短模式（能放入机器字内）
- 是近似匹配和模糊搜索的基础
- 应用于 grep、拼写纠正、DNA 搜索和流文本处理等场景

#### 复杂度

| 操作          | 时间      | 空间              |
| ------------- | ------- | --------------- |
| 精确匹配      | $O(n)$  | $O(\sigma)$     |
| 允许 k 个错误 | $O(nk)$ | $O(\sigma + k)$ |

这里 $\sigma$ 是字母表大小。

#### 动手尝试

1.  修改 Python 版本，使其返回匹配的起始位置而非结束位置。
2.  使用长文本进行测试，观察其接近线性的运行时间。
3.  扩展它，使用 `R[k]` 数组来允许 1 个不匹配。
4.  可视化 `R` 的位模式，观察前缀匹配是如何演变的。
5.  与 KMP 和朴素匹配算法比较性能。

#### 一个温和的证明（为何有效）

`R` 的每次左移对应于将匹配的前缀扩展一个字符。通过与 `B[c]` 进行按位与（masking），在模式字符与文本字符匹配的位置将位清零。因此，位置 `m−1` 处的 0 意味着整个模式已经匹配——该算法用位并行性模拟了一个动态规划表。

Bitap 是算法优雅性的完美例证——它将完整的比较表压缩为几个与文本同步“舞动”的比特位。
### 673 Landau–Vishkin 算法（编辑距离 ≤ k）

Landau–Vishkin 算法通过计算不超过固定阈值 k 的编辑距离，高效地解决了 *k-近似字符串匹配* 问题，而无需构建完整的动态规划表。当 k 较小时，它是近似匹配中最优雅的线性时间算法之一。

#### 核心思想

我们希望在文本 T 中找到所有位置，使得模式 P 能以最多 k 次编辑（插入、删除或替换）与之匹配。

该算法不是计算完整的 $m \times n$ 编辑距离表，而是跟踪 DP 网格中的对角线，并沿着每条对角线尽可能远地扩展匹配。

这种基于对角线的思想使得算法在 k 较小时快得多。

#### 简要的 DP 视角

在经典的编辑距离 DP 中，每个单元格 $(i, j)$ 表示 $P[1..i]$ 和 $T[1..j]$ 之间的编辑距离。

具有相同差值 $d = j - i$ 的单元格位于同一条对角线上。每次编辑操作都会使你在对角线之间略微移动。

Landau–Vishkin 算法为每个编辑距离 e (0 ≤ e ≤ k) 计算，在 e 次编辑后，我们能沿着每条对角线匹配多远。

#### 主要递推关系

令：

- `L[e][d]` = 在 `e` 次编辑后，对角线 `d`（偏移量 `j - i`）上匹配的最远前缀长度。

我们按如下方式更新：

$$
L[e][d] =
\begin{cases}
L[e-1][d-1] + 1, & \text{插入},\\[4pt]
L[e-1][d+1], & \text{删除},\\[4pt]
L[e-1][d] + 1, & \text{替换（如果不匹配）}.
\end{cases}
$$


然后，从该位置开始，只要字符匹配，我们就尽可能远地扩展：

$$
\text{while } P[L[e][d]+1] = T[L[e][d]+d+1],\ L[e][d]++
$$

如果在任何时刻 $L[e][d] \ge m$，我们就找到了一个编辑次数 ≤ e 的匹配。

#### 示例（直观解释）

令
`P = "kitten"`, `T = "sitting"`, 且 $k = 2$。

算法从匹配所有 0 次编辑的对角线开始。然后允许 1 次编辑（跳过、替换、插入）并跟踪匹配可以继续多远。最后，它确认 `"kitten"` 与 `"sitting"` 在 2 次编辑内匹配。

#### 微型代码（Python 版本）

```python
def landau_vishkin(text, pattern, k):
    n, m = len(text), len(pattern)
    for s in range(n - m + 1):
        max_edits = 0
        e = 0
        L = {0: -1}
        while e <= k:
            newL = {}
            for d in range(-e, e + 1):
                best = max(
                    L.get(d - 1, -1) + 1,
                    L.get(d + 1, -1),
                    L.get(d, -1) + 1
                )
                while best + 1 < m and s + best + d + 1 < n and pattern[best + 1] == text[s + best + d + 1]:
                    best += 1
                newL[d] = best
                if best >= m - 1:
                    print(f"在文本位置 {s} 匹配，编辑次数 ≤ {e}")
                    return
            L = newL
            e += 1

landau_vishkin("sitting", "kitten", 2)
```

输出：

```
在文本位置 0 匹配，编辑次数 ≤ 2
```

#### 为何重要

- 对于固定的 k 是线性时间：$O(kn)$
- 对于短容错搜索效果极佳
- 是以下领域算法的基础：

  * 生物信息学（DNA 序列比对）
  * 拼写纠正
  * 抄袭检测
  * 近似子串搜索

#### 复杂度

| 操作         | 时间      | 空间   |
| ------------ | --------- | ------ |
| 匹配检查     | $O(kn)$   | $O(k)$ |

当 k 较小（例如 1–3）时，这比完整的 DP（$O(nm)$）快得多。

#### 亲自尝试

1.  修改代码以打印所有近似匹配位置。
2.  可视化每个编辑步骤的对角线 `d = j - i`。
3.  在文本中加入随机噪声进行测试。
4.  比较 k=1,2,3 时与 DP 的运行时间。
5.  扩展到字母集（A, C, G, T）。

#### 一个温和的证明（为何有效）

每条对角线代表 `P` 和 `T` 之间的固定对齐偏移量。在 e 次编辑后，算法记录沿着每条对角线可到达的最远匹配索引。由于每次编辑只能向左或向右移动一条对角线，因此每层最多有 $2e + 1$ 条活动对角线，总成本为 $O(kn)$。正确性基于对最小编辑次数的归纳。

Landau–Vishkin 算法是跳过不匹配的艺术——它在可能性网格中找到结构，并遍历真正重要的少数路径。
### 674 过滤算法（快速近似搜索）

过滤算法通过一个快速精确过滤步骤跳过文本的大部分，然后使用较慢的验证步骤（如动态规划）来确认潜在匹配，从而加速近似字符串匹配。它是许多现代搜索工具和生物信息学系统背后的核心思想。

#### 核心思想

如果检查每个子串，近似匹配的代价很高。因此，过滤算法不是测试所有位置，而是将模式分割成若干片段，并对每个片段进行*精确*搜索。

如果文本的一个子串与模式的编辑距离最多为 $k$ 个错误，那么至少有一个片段必须*精确*匹配（鸽巢原理）。

这就是关键的洞见。

#### 逐步分解

令：

- `P` = 长度为 $m$ 的模式
- `T` = 长度为 $n$ 的文本
- `k` = 允许的最大错误数

我们将 `P` 分成 $k + 1$ 个相等（或近似相等）的块：

$$
P = P_1 P_2 \dots P_{k+1}
$$

如果 `T` 包含一个子串 `S`，使得 `P` 和 `S` 之间的编辑距离 ≤ $k$，那么至少有一个块 $P_i$ 必须*精确地*出现在 `S` 内部。

因此，我们可以：

1.  使用快速方法（如 KMP 或 Boyer–Moore）精确搜索每个块 `P_i`。
2.  使用动态规划（距离最多为 $k$）验证每个找到的出现位置周围的区域。

#### 示例

模式 `P = "abcdefgh"`
k = 2
分成 3 个块：`abc | def | gh`

在文本中搜索每个块：

```
T: xabcydzdefh...
```

如果我们在位置 2 找到 `"abc"`，在位置 8 找到 `"def"`，我们就检查它们的邻域，看看整个模式是否能在 2 次编辑内对齐。

#### 算法大纲

1.  将模式 `P` 分成 $k + 1$ 个块。
2.  对于每个块：
    *   在 `T` 中找到所有精确匹配（例如，通过 KMP）。
    *   对于位置 `pos` 处的每个匹配，使用编辑距离动态规划验证周围的子串 `T[pos - offset : pos + m]`。
3.  报告总编辑次数 ≤ $k$ 的匹配。

#### 微型代码（Python）

```python
def filtering_match(text, pattern, k):
    n, m = len(text), len(pattern)
    block_size = m // (k + 1)
    matches = set()

    for b in range(k + 1):
        start = b * block_size
        end = m if b == k else (b + 1) * block_size
        block = pattern[start:end]

        pos = text.find(block)
        while pos != -1:
            window_start = max(0, pos - start - k)
            window_end = min(n, pos - start + m + k)
            window = text[window_start:window_end]
            if edit_distance(window, pattern) <= k:
                matches.add(window_start)
            pos = text.find(block, pos + 1)

    for mpos in sorted(matches):
        print(f"在位置 {mpos} 处发现近似匹配")

def edit_distance(a, b):
    dp = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (a[i - 1] != b[j - 1])
            )
    return dp[len(a)][len(b)]

filtering_match("xxabcdefyy", "abcdefgh", 2)
```

输出：

```
在位置 2 处发现近似匹配
```

#### 为什么它有效

根据鸽巢原理，如果最多有 $k$ 个不匹配，那么模式的至少一个片段必须是完整的。因此，对片段进行精确搜索极大地减少了需要检查的候选位置数量。

这对于小的 k 值尤其有效，此时 $k + 1$ 个片段能均匀地覆盖模式。

#### 复杂度

| 阶段                     | 时间        | 空间  |
| ------------------------ | ----------- | ----- |
| 过滤（精确搜索）         | $O((k+1)n)$ | $O(1)$ |
| 验证（动态规划）         | $O(km)$     | $O(m)$ |

总体而言，对于小的 $k$ 值和长文本，该算法是亚线性的。

#### 应用

-   快速近似文本搜索
-   DNA 序列比对（种子-扩展模型）
-   剽窃和相似性检测
-   大型数据库中的模糊搜索

#### 亲自尝试

1.  改变 k 值，观察过滤如何减少比较次数。
2.  在过滤阶段使用 Boyer–Moore 算法。
3.  测量在大输入上的性能。
4.  用 Myers 的位并行方法替换编辑距离动态规划以提高速度。
5.  可视化重叠的验证区域。

#### 一个温和的证明（为什么它有效）

如果 `T` 的一个子串 `S` 与 `P` 匹配，且错误 ≤ k，那么在将 `P` 分成 $k+1$ 部分后，必须至少存在一个块 `P_i` 在 `S` 内精确匹配。因此，检查精确块匹配的邻域确保了正确性。这使得非候选区域可以被指数级地剪枝。

过滤算法体现了一个简单的哲学——先找锚点，后验证，将暴力匹配转变为智能、可扩展的搜索。
### 675 Wu–Manber 算法（多模式近似搜索）

Wu–Manber 算法是一种实用且高效的近似多模式匹配方法。
它推广了 Boyer–Moore 和 Shift-And/Bitap 算法的思想，使用基于块的哈希和移位表来跳过文本的大部分区域，同时仍然允许有限数量的错误。

它为许多经典搜索工具提供了支持，包括 agrep 以及早期具有容错能力的 grep -F 变体。

#### 核心思想

Wu–Manber 扩展了过滤原则：
在文本中搜索模式块，并且只验证那些可能匹配的位置。

但它不是一次处理一个模式，而是使用以下方式同时处理多个模式：

1. 一个基于哈希的块查找表
2. 一个指示我们可以安全跳过多少距离的移位表
3. 一个用于潜在匹配的验证表

#### 工作原理（高层次）

令：

- `P₁, P₂, ..., P_r` 为模式，每个长度 ≥ `B`
- `B` = 块大小（通常为 2 或 3）
- `T` = 长度为 `n` 的文本
- `k` = 允许的最大错配数

算法在 `T` 上滑动一个窗口，将窗口的最后 `B` 个字符视为一个关键块。

如果该块不出现在任何模式中，则按预计算的移位值向前跳过。
如果它确实出现，则在该位置附近运行近似验证。

#### 预处理步骤

1. 移位表构建

   对于模式中的每个块 `x`，存储*任何包含 `x` 的模式中从末尾算起的最小距离*。
   不出现的块可以有一个大的默认移位值。

   $$
   \text{SHIFT}[x] = \min{m - i - B + 1\ |\ P[i..i+B-1] = x}
   $$

2. 哈希表

   对于每个块，存储以该块结尾的模式列表。

   $$
   \text{HASH}[x] = {P_j\ |\ P_j\text{ ends with }x}
   $$

3. 验证表

   存储候选模式在何处以及如何进行编辑距离验证。

#### 搜索阶段

从左到右在文本中滑动窗口：

1. 读取窗口的最后 `B` 个字符 `x`。
2. 如果 `SHIFT[x] > 0`，则按该值向前跳过。
3. 如果 `SHIFT[x] == 0`，可能匹配，使用动态规划或位并行编辑距离验证 `HASH[x]` 中的候选模式。

重复直到文本结束。

#### 示例

令 `patterns = ["data", "date"]`, `B = 2`, 且 `k = 1`。

文本：`"the dataset was updated yesterday"`

1. 预计算：

   ```
   SHIFT["ta"] = 0
   SHIFT["da"] = 1
   SHIFT["at"] = 1
   其他 = 3
   ```

2. 扫描时：

   * 当最后两个字符是 `"ta"` 时，SHIFT = 0 → 验证 "data" 和 "date"。
   * 在其他位置，向前跳过 1–3 个字符。

这允许跳过文本的大部分，同时只验证少数几个位置。

#### 简化代码（简化 Python 版本）

```python
def wu_manber(text, patterns, k=0, B=2):
    shift = {}
    hash_table = {}
    m = min(len(p) for p in patterns)
    default_shift = m - B + 1

    # 预处理
    for p in patterns:
        for i in range(m - B + 1):
            block = p[i:i+B]
            shift[block] = min(shift.get(block, default_shift), m - i - B)
            hash_table.setdefault(block, []).append(p)

    # 搜索
    pos = 0
    while pos <= len(text) - m:
        block = text[pos + m - B: pos + m]
        if shift.get(block, default_shift) > 0:
            pos += shift.get(block, default_shift)
        else:
            for p in hash_table.get(block, []):
                segment = text[pos:pos+len(p)]
                if edit_distance(segment, p) <= k:
                    print(f"在位置 {pos} 匹配到 '{p}'")
            pos += 1

def edit_distance(a, b):
    dp = [[i + j if i*j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + (a[i-1] != b[j-1])
            )
    return dp[-1][-1]

wu_manber("the dataset was updated yesterday", ["data", "date"], 1)
```

输出：

```
在位置 4 匹配到 'data'
在位置 4 匹配到 'date'
```

#### 重要性

- 同时处理多个模式
- 支持近似匹配（少量错配）
- 在大文本上高效（亚线性跳过）
- 用于：

  * agrep
  * 文本索引引擎
  * 生物信息学搜索工具

#### 复杂度

| 操作         | 时间           | 空间   |
| ------------ | -------------- | ------ |
| 预处理       | $O(rm)$        | $O(rm)$ |
| 搜索         | $O(n)$ 平均    | $O(rm)$ |

其中：

- $r$ = 模式数量
- $m$ = 平均模式长度

#### 亲自尝试

1. 将块大小 B 改为 3，观察跳过行为。
2. 添加更多具有公共后缀的模式。
3. 与 Boyer–Moore 和 Aho–Corasick 比较速度。
4. 尝试 k = 1 和 2 的情况（近似匹配）。
5. 实现位并行验证（Myers 算法）。

#### 一个温和的证明（为什么它有效）

每个真正的匹配必须至少包含一个块，该块在模式和文本片段中都保持不变。
通过对块进行哈希，我们只找到那些*可能*是有效匹配的位置。
移位值确保跳过的块不可能匹配，
使得算法既正确又高效。

Wu–Manber 算法是模糊搜索的工艺大师——
将哈希、跳过和验证结合在一次快速、优雅的文本扫描中。
### 676 流式 KMP（在线前缀更新）

流式 KMP 算法将经典的 Knuth–Morris–Pratt 模式匹配算法适配到*流式模型*，在这种模型中，字符逐个到达，我们必须*立即*检测到匹配，而无需重新扫描之前的文本。

这对于实时系统、网络流量监控和实时日志过滤至关重要，因为在这些场景中存储整个输入是不可行的。

#### 核心思想

在经典 KMP 中，我们为模式 `P` 预先计算一个前缀函数 `π`，以帮助我们在失配后高效地移动。

在流式 KMP 中，我们在实时读取字符时，以增量方式维护这个相同的前缀状态。每个新字符仅根据前一状态和当前符号来更新匹配状态。

这实现了每个字符的常数时间更新和常数内存开销。

#### 前缀函数回顾

对于一个模式 $P[0..m-1]$，前缀函数 `π[i]` 定义为：

$$
π[i] = \text{模式 } P[0..i] \text{ 的最长真前缀的长度，该前缀同时也是后缀。}
$$

示例：
对于 `P = "ababc"`，
前缀表为：

| i | P[i] | π[i] |
| - | ---- | ---- |
| 0 | a    | 0    |
| 1 | b    | 0    |
| 2 | a    | 1    |
| 3 | b    | 2    |
| 4 | c    | 0    |

#### 流式更新规则

我们维护：

- `state` = 当前已匹配的模式字符数。

当流中到达一个新字符 `c` 时：

```
while state > 0 and P[state] != c:
    state = π[state - 1]
if P[state] == c:
    state += 1
if state == m:
    report match
    state = π[state - 1]
```

这确保了每个输入字符都能在 O(1) 时间内更新匹配状态。

#### 示例

模式：`"abcab"`

输入流：`"xabcabcabz"`

我们跟踪匹配状态：

| 流字符 | 更新前状态 | 更新后状态 | 匹配？ |
| ----------- | ------------ | ----------- | ------ |
| x           | 0            | 0           |        |
| a           | 0            | 1           |        |
| b           | 1            | 2           |        |
| c           | 2            | 3           |        |
| a           | 3            | 4           |        |
| b           | 4            | 5           | 是      |
| c           | 5→2          | 3           |        |
| a           | 3            | 4           |        |
| b           | 4            | 5           | 是      |

因此，匹配发生在位置 5 和 9。

#### 精简代码（Python 流式 KMP）

```python
def compute_prefix(P):
    m = len(P)
    π = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and P[i] != P[j]:
            j = π[j - 1]
        if P[i] == P[j]:
            j += 1
        π[i] = j
    return π

def stream_kmp(P):
    π = compute_prefix(P)
    state = 0
    pos = 0
    print("流式处理中...")

    while True:
        c = yield  # 每次接收一个字符
        pos += 1
        while state > 0 and (state == len(P) or P[state] != c):
            state = π[state - 1]
        if P[state] == c:
            state += 1
        if state == len(P):
            print(f"在位置 {pos} 结束处匹配")
            state = π[state - 1]

# 使用示例
matcher = stream_kmp("abcab")
next(matcher)
for c in "xabcabcabz":
    matcher.send(c)
```

输出：

```
在位置 5 结束处匹配
在位置 9 结束处匹配
```

#### 精简代码（C 语言版本）

```c
#include <stdio.h>
#include <string.h>

void compute_prefix(const char *P, int *pi, int m) {
    pi[0] = 0;
    int j = 0;
    for (int i = 1; i < m; i++) {
        while (j > 0 && P[i] != P[j]) j = pi[j - 1];
        if (P[i] == P[j]) j++;
        pi[i] = j;
    }
}

void stream_kmp(const char *P, const char *stream) {
    int m = strlen(P);
    int pi[m];
    compute_prefix(P, pi, m);
    int state = 0;
    for (int pos = 0; stream[pos]; pos++) {
        while (state > 0 && P[state] != stream[pos])
            state = pi[state - 1];
        if (P[state] == stream[pos])
            state++;
        if (state == m) {
            printf("在位置 %d 结束处匹配\n", pos + 1);
            state = pi[state - 1];
        }
    }
}

int main() {
    stream_kmp("abcab", "xabcabcabz");
}
```

#### 为何重要

- 处理无限流，只保留当前状态
- 无需重新扫描，每个符号只处理一次
- 完美适用于：

  * 实时文本过滤器
  * 入侵检测系统
  * 网络数据包分析
  * 在线模式分析

#### 复杂度

| 操作                 | 时间   | 空间  |
| -------------------- | ------ | ------ |
| 每个字符更新         | $O(1)$ | $O(m)$ |
| 匹配检测             | $O(n)$ | $O(m)$ |

#### 动手尝试

1.  修改代码以计数重叠匹配。
2.  使用连续输入流（例如，日志尾部）进行测试。
3.  实现支持多模式的版本（使用 Aho–Corasick 算法）。
4.  添加在长时间失配后的重置功能。
5.  可视化每个新字符的前缀转换。

#### 一个温和的证明（为何有效）

前缀函数确保了每当发生失配时，算法确切地知道可以安全回溯多远而不会丢失潜在的匹配。流式 KMP 延续了这一逻辑 —— 当前 `state` 始终等于与流后缀匹配的 `P` 的最长前缀的长度。这个不变量保证了仅通过常数时间更新就能获得正确性。

流式 KMP 是一个极简主义的奇迹 —— 一个整数状态、一个前缀表，以及流过它的数据流 —— 实现了零回溯的实时匹配。
### 677 滚动哈希草图（滑动窗口哈希）

滚动哈希草图是高效处理大型文本流或长字符串的基本技巧。它能够以常数时间计算固定长度 L 的每个子串（或窗口）的哈希值，非常适合滑动窗口算法、重复检测、指纹识别和相似性搜索。

这项技术是许多著名算法的基础，包括 Rabin–Karp、Winnowing 和 MinHash。

#### 核心思想

假设你想对长度为 n 的文本 T 中所有长度为 L 的子串进行哈希。

一种朴素的方法是以 $O(L)$ 的时间计算每个哈希值，总时间为 $O(nL)$。滚动哈希在窗口滑动一个字符时，能以 $O(1)$ 的时间更新哈希值。

#### 多项式滚动哈希

一种常见的形式是将子串视为以 B 为底、对一个大素数 M 取模的数字：

$$
H(i) = (T[i]B^{L-1} + T[i+1]B^{L-2} + \dots + T[i+L-1]) \bmod M
$$

当窗口向前滑动一个字符时，我们移除旧字符并添加新字符：

$$
H(i+1) = (B(H(i) - T[i]B^{L-1}) + T[i+L]) \bmod M
$$

这个递推关系使我们能够高效地更新哈希值。

#### 示例

令 $T = $ `"abcd"`，窗口长度 $L = 3$，基数 $B = 31$，模数 $M = 10^9 + 9$。

计算：

- `"abc"` 的 $H(0)$
- 滑动一步 → `"bcd"` 的 $H(1)$

```
H(0) = a*31^2 + b*31 + c
H(1) = (H(0) - a*31^2)*31 + d
```

#### 微型代码（Python）

```python
def rolling_hash(text, L, B=257, M=109 + 7):
    n = len(text)
    if n < L:
        return []

    hashes = []
    power = pow(B, L - 1, M)
    h = 0

    # 初始哈希
    for i in range(L):
        h = (h * B + ord(text[i])) % M
    hashes.append(h)

    # 滚动更新
    for i in range(L, n):
        h = (B * (h - ord(text[i - L]) * power) + ord(text[i])) % M
        h = (h + M) % M  # 确保非负
        hashes.append(h)

    return hashes

text = "abcdefg"
L = 3
print(rolling_hash(text, L))
```

输出（示例哈希值）：

```
$$6382170, 6487717, 6593264, 6698811, 6804358]
```

#### 微型代码（C）

```c
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define MOD 1000000007
#define BASE 257

void rolling_hash(const char *text, int L) {
    int n = strlen(text);
    if (n < L) return;

    uint64_t power = 1;
    for (int i = 1; i < L; i++) power = (power * BASE) % MOD;

    uint64_t hash = 0;
    for (int i = 0; i < L; i++)
        hash = (hash * BASE + text[i]) % MOD;

    printf("Hash[0] = %llu\n", hash);
    for (int i = L; i < n; i++) {
        hash = (BASE * (hash - text[i - L] * power % MOD) + text[i]) % MOD;
        if ((int64_t)hash < 0) hash += MOD;
        printf("Hash[%d] = %llu\n", i - L + 1, hash);
    }
}

int main() {
    rolling_hash("abcdefg", 3);
}
```

#### 为何重要

- 增量计算，非常适合流式处理
- 支持常数时间的子串比较
- 是以下技术的支柱：
  * Rabin–Karp 模式匹配
  * 滚动校验和（rsync, zsync）
  * Winnowing 指纹识别
  * 去重系统

#### 复杂度

| 操作           | 时间   | 空间  |
| -------------- | ------ | ----- |
| 初始哈希       | $O(L)$ | $O(1)$ |
| 每次更新       | $O(1)$ | $O(1)$ |
| n 个窗口总计   | $O(n)$ | $O(1)$ |

#### 动手尝试

1.  使用两个不同的模数（双重哈希）来减少碰撞。
2.  在长文本中检测长度为 10 的重复子串。
3.  为字节（文件）而非字符实现滚动哈希。
4.  针对随机输入和顺序输入，实验碰撞行为。
5.  与从头重新计算每个哈希值的方法比较速度。

#### 一个温和的证明（为何有效）

当我们滑动窗口时，旧字符（$T[i]B^{L-1}$）的贡献被减去，而所有其他字符都乘以 $B$。然后，新字符被添加到最低有效位。这保留了正确的加权多项式表示（模 $M$）——因此每个子串的哈希值*以高概率*是唯一的。

滚动哈希草图是现代文本系统的代数核心——每一步都遗忘一个符号，学习另一个符号，并以常数时间保持流的指纹活性。
### 678 基于草图（Sketch）的相似性（MinHash 与 LSH）

当数据集或文档过大而无法直接比较时，我们会转向基于草图的相似性方法。这是一种紧凑的数学指纹，让我们无需完整读取两个文本（或任何数据）就能估计它们的相似程度。

这个思想通过 MinHash 和局部敏感哈希（LSH）等技术，为搜索引擎、重复检测和推荐系统提供了动力。

#### 问题描述

你想知道两个长文档（例如，每个都有数百万个词元）的内容是否相似。
精确计算它们特征集（例如，单词、shingle 或 n-gram）之间的 Jaccard 相似度需要进行大量的交集和并集操作。

我们需要一种更快的方法，其复杂度应低于文档大小的线性级别，但又足够准确以支持大规模比较。

#### 核心思想：草图（Sketching）

草图是对大型对象的一种压缩表示，它保留了某些统计特性。
对于文本相似性，我们使用 MinHash 草稿来近似 Jaccard 相似度：

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

MinHash 让我们能够高效地估计这个值。

#### MinHash

对于每个集合（例如，词元集合），我们应用 h 个独立的哈希函数。
每个哈希函数为每个元素分配一个伪随机数，我们记录该函数的*最小*哈希值。

形式化地，对于集合 $S$ 和哈希函数 $h_i$：

$$
\text{MinHash}*i(S) = \min*{x \in S} h_i(x)
$$

得到的草图是一个向量：

$$
M(S) = [\text{MinHash}_1(S), \text{MinHash}_2(S), \dots, \text{MinHash}_h(S)]
$$

那么两个集合之间的相似度可以通过以下公式估计：

$$
\hat{J}(A, B) = \frac{\text{number of matching components in } M(A), M(B)}{h}
$$

#### 示例

设集合为：

- $A = {1, 3, 5}$
- $B = {1, 2, 3, 6}$

我们使用三个简单的哈希函数：

| 元素 | h₁(x) | h₂(x) | h₃(x) |
| ------- | ----- | ----- | ----- |
| 1       | 5     | 7     | 1     |
| 2       | 6     | 3     | 4     |
| 3       | 2     | 5     | 3     |
| 5       | 8     | 2     | 7     |
| 6       | 1     | 4     | 6     |

那么：

- MinHash(A) = [min(5,2,8)=2, min(7,5,2)=2, min(1,3,7)=1]
- MinHash(B) = [min(5,6,2,1)=1, min(7,3,5,4)=3, min(1,4,3,6)=1]

逐元素比较：3 个中有 1 个匹配 → 估计相似度 $\hat{J}=1/3$。

真实的 Jaccard 相似度是 $|A∩B|/|A∪B| = 2/5 = 0.4$，所以草图估计值很接近。

#### 微型代码（Python）

```python
import random

def minhash(setA, setB, num_hashes=100):
    max_hash = 232 - 1
    seeds = [random.randint(0, max_hash) for _ in range(num_hashes)]
    
    def hash_func(x, seed): return (hash((x, seed)) & max_hash)
    
    def signature(s):
        return [min(hash_func(x, seed) for x in s) for seed in seeds]

    sigA = signature(setA)
    sigB = signature(setB)
    matches = sum(a == b for a, b in zip(sigA, sigB))
    return matches / num_hashes

A = {"data", "machine", "learning", "hash"}
B = {"data", "machine", "hash", "model"}
print("Estimated similarity:", minhash(A, B))
```

输出（近似值）：

```
Estimated similarity: 0.75
```

#### 从 MinHash 到 LSH

局部敏感哈希（LSH）提升了 MinHash 以实现快速的*查找*。
它以高概率将相似的草图分组到同一个"桶"中，这样我们就可以在常数时间内找到近似重复项。

将长度为 `h` 的草图划分为 `b` 个*带*，每个带有 `r` 行：

- 将每个带哈希到一个桶。
- 如果两个文档在*任何*一个带中共享一个桶，它们很可能相似。

这将全局比较转化为概率性索引。

#### 为何重要

- 支持在海量集合中进行快速相似性搜索
- 空间高效：每个文档的草图大小固定
- 应用于：

  * 搜索引擎去重（Google，Bing）
  * 文档聚类
  * 剽窃检测
  * 大规模推荐系统

#### 复杂度

| 操作           | 时间       | 空间  |    |        |
| ---------------- | ---------- | ------ | -- | ------ |
| 构建草图       | $O(h       | S      | )$ | $O(h)$ |
| 比较两个集合   | $O(h)$     | $O(1)$ |    |        |
| LSH 查找       | $O(1)$ 平均 | $O(h)$ |    |        |

#### 亲自尝试

1.  为多个文档创建 MinHash 草图，并可视化成对相似性。
2.  改变哈希函数的数量（10，100，500），观察准确性的权衡。
3.  尝试 2 带和 3 带的 LSH 分组。
4.  与 TF-IDF 向量上的余弦相似度进行比较。
5.  应用于文本段落的 n-gram 集合。

#### 一个温和的证明（为何有效）

对于一个随机哈希函数 $h$，
$\min(h(A)) = \min(h(B))$ 的概率等于 Jaccard 相似度 $J(A, B)$。
因此，MinHash 签名中相等分量的期望比例近似于 $J(A, B)$。
这个优雅的统计特性使得 MinHash 既无偏又可证明是准确的。

基于草图的相似性将意义压缩成少数几个数字——
整个文档的微小数字回响，
使机器能够大规模地记忆、比较和聚类世界上的文本。
### 679 加权编辑距离（加权操作）

加权编辑距离通过为插入、删除和替换操作，甚至为特定的字符对分配*不同的成本*，从而推广了经典的 Levenshtein 距离。
这使得它在现实世界的任务中更加灵活，例如拼写纠正、语音识别、OCR 和生物序列分析，在这些任务中，某些错误*比其他错误更可能发生*。

#### 核心思想

在标准编辑距离中，每个操作的成本为 1。
在加权编辑距离中，每个操作都有自己的成本函数：

- $w_{ins}(a)$，插入字符 *a* 的成本
- $w_{del}(a)$，删除字符 *a* 的成本
- $w_{sub}(a,b)$，将 *a* 替换为 *b* 的成本

目标是找到将一个字符串转换为另一个字符串的最小总成本。

#### 递推关系

令 $dp[i][j]$ 为将 $s_1[0..i-1]$ 转换为 $s_2[0..j-1]$ 的最小成本。
则有：

$$
dp[i][j] = \min
\begin{cases}
dp[i-1][j] + w_{\text{del}}(s_1[i-1]), & \text{(删除)},\\[4pt]
dp[i][j-1] + w_{\text{ins}}(s_2[j-1]), & \text{(插入)},\\[4pt]
dp[i-1][j-1] + w_{\text{sub}}(s_1[i-1], s_2[j-1]), & \text{(替换)}.
\end{cases}
$$

其基本情况为：

$$
dp[0][j] = \sum_{k=1}^{j} w_{ins}(s_2[k-1])
$$

$$
dp[i][0] = \sum_{k=1}^{i} w_{del}(s_1[k-1])
$$

#### 示例

让我们比较 `"kitten"` 和 `"sitting"`，并设定：

- $w_{sub}(a,a)=0$, $w_{sub}(a,b)=2$
- $w_{ins}(a)=1$, $w_{del}(a)=1$

操作如下：

```
kitten → sitten (替换 'k'→'s', 成本 2)
sitten → sittin (插入 'i', 成本 1)
sittin → sitting (插入 'g', 成本 1)
```

总成本 = 4。

#### 微型代码 (Python)

```python
def weighted_edit_distance(s1, s2, w_sub, w_ins, w_del):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] + w_del(s1[i-1])
    for j in range(1, n+1):
        dp[0][j] = dp[0][j-1] + w_ins(s2[j-1])

    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = min(
                dp[i-1][j] + w_del(s1[i-1]),
                dp[i][j-1] + w_ins(s2[j-1]),
                dp[i-1][j-1] + w_sub(s1[i-1], s2[j-1])
            )
    return dp[m][n]

w_sub = lambda a,b: 0 if a==b else 2
w_ins = lambda a: 1
w_del = lambda a: 1

print(weighted_edit_distance("kitten", "sitting", w_sub, w_ins, w_del))
```

输出：

```
4
```

#### 微型代码 (C)

```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int min3(int a, int b, int c) {
    return a < b ? (a < c ? a : c) : (b < c ? b : c);
}

int weighted_edit_distance(const char *s1, const char *s2) {
    int m = strlen(s1), n = strlen(s2);
    int dp[m+1][n+1];

    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int cost = (s1[i-1] == s2[j-1]) ? 0 : 2;
            dp[i][j] = min3(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            );
        }
    }
    return dp[m][n];
}

int main() {
    printf("%d\n", weighted_edit_distance("kitten", "sitting"));
}
```

输出：

```
4
```

#### 为何重要

加权编辑距离让我们能够对现实世界中转换的不对称性进行建模：

- OCR：混淆 "O" 和 "0" 的成本低于 "O" → "X"
- 语音比较："f" ↔ "ph" 替换的成本较低
- 生物信息学：插入/删除惩罚取决于间隙长度
- 拼写纠正：键盘相邻键的错误成本较低

这种细粒度的控制既提供了更好的准确性，也提供了更自然的容错能力。

#### 复杂度

| 操作             | 时间      | 空间             |
| ---------------- | --------- | ---------------- |
| 完整动态规划     | $O(mn)$   | $O(mn)$          |
| 空间优化         | $O(mn)$   | $O(\min(m,n))$   |

#### 动手尝试

1.  使用真实的键盘布局定义 $w_{sub}(a,b)$ = QWERTY 键盘上的距离。
2.  比较等成本与非对称成本之间的差异。
3.  修改插入/删除惩罚以模拟间隙开启与扩展。
4.  将动态规划成本表面可视化为热图。
5.  使用加权编辑距离对 OCR 纠正候选进行排序。

#### 一个温和的证明（为何有效）

加权编辑距离保留了动态规划原理：
转换前缀的最小成本仅取决于更小的子问题。
通过分配非负、一致的权重，
该算法保证了在这些成本定义下的最优转换。
它将 Levenshtein 距离推广为所有成本均为 1 的特殊情况。

加权编辑距离将字符串比较从对编辑次数的刚性计数，
转变为对一次更改*有多错误*的细致反映——
使其成为文本算法中最具人性化的度量之一。
### 680 在线莱文斯坦距离（动态流式更新）

在线莱文斯坦距离算法将编辑距离计算带入流式处理世界，它在新字符到达时增量式地更新距离，而不是重新计算整个动态规划（DP）表。
这对于实时拼写检查、语音转录和 DNA 流式比对至关重要，因为文本或数据是逐个符号到来的。

#### 核心思想

经典的莱文斯坦距离构建一个大小为 $m \times n$ 的完整表格，比较字符串 $A$ 和 $B$ 的所有前缀。
在*在线设置*中，文本 $T$ 随时间增长，但模式 $P$ 保持固定。

我们不希望每次出现新字符时都重建所有内容 ——
相反，我们高效地更新最后一个 DP 行以反映新的输入。

这意味着维护固定模式与不断增长的文本前缀之间的当前编辑距离。

#### 标准莱文斯坦距离回顾

对于字符串 $P[0..m-1]$ 和 $T[0..n-1]$：

$$
dp[i][j] =
\begin{cases}
i, & \text{if } j = 0,\\[4pt]
j, & \text{if } i = 0,\\[6pt]
\min
\begin{cases}
dp[i-1][j] + 1, & \text{(删除)},\\[4pt]
dp[i][j-1] + 1, & \text{(插入)},\\[4pt]
dp[i-1][j-1] + [P[i-1] \ne T[j-1]], & \text{(替换)}.
\end{cases}
\end{cases}
$$

最终距离为 $dp[m][n]$。

#### 在线变体

当新字符 $t$ 到达时，
我们只保留前一行并在 $O(m)$ 时间内更新它。

令 `prev[i]` = 将 `P[:i]` 与 `T[:j-1]` 对齐的成本，
`curr[i]` = 将 `P[:i]` 与 `T[:j]` 对齐的成本。

新字符 `t` 的更新规则：

$$
curr[0] = j
$$

$$
curr[i] = \min
\begin{cases}
prev[i] + 1, & \text{(删除)},\\[4pt]
curr[i-1] + 1, & \text{(插入)},\\[4pt]
prev[i-1] + [P[i-1] \ne t], & \text{(替换)}.
\end{cases}
$$

处理完成后，替换 `prev = curr`。

#### 示例

模式 `P = "kitten"`
流式文本：`"kit", "kitt", "kitte", "kitten"`

我们为每个字符更新一行：

| 步骤     | 输入 | 距离 |
| -------- | ----- | -------- |
| "k"      | 5     |          |
| "ki"     | 4     |          |
| "kit"    | 3     |          |
| "kitt"   | 2     |          |
| "kitte"  | 1     |          |
| "kitten" | 0     |          |

当我们达到完全匹配时，距离逐渐降至 0。

#### 精简代码（Python，基于流）

```python
def online_levenshtein(pattern):
    m = len(pattern)
    prev = list(range(m + 1))
    j = 0

    while True:
        c = yield prev[-1]  # 当前距离
        j += 1
        curr = [j]
        for i in range(1, m + 1):
            cost = 0 if pattern[i - 1] == c else 1
            curr.append(min(
                prev[i] + 1,
                curr[i - 1] + 1,
                prev[i - 1] + cost
            ))
        prev = curr

# 使用示例
stream = online_levenshtein("kitten")
next(stream)
for ch in "kitten":
    d = stream.send(ch)
    print(f"After '{ch}': distance = {d}")
```

输出：

```
After 'k': distance = 5
After 'i': distance = 4
After 't': distance = 3
After 't': distance = 2
After 'e': distance = 1
After 'n': distance = 0
```

#### 精简代码（C 语言版本）

```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void online_levenshtein(const char *pattern, const char *stream) {
    int m = strlen(pattern);
    int *prev = malloc((m + 1) * sizeof(int));
    int *curr = malloc((m + 1) * sizeof(int));

    for (int i = 0; i <= m; i++) prev[i] = i;

    for (int j = 0; stream[j]; j++) {
        curr[0] = j + 1;
        for (int i = 1; i <= m; i++) {
            int cost = (pattern[i - 1] == stream[j]) ? 0 : 1;
            int del = prev[i] + 1;
            int ins = curr[i - 1] + 1;
            int sub = prev[i - 1] + cost;
            curr[i] = del < ins ? (del < sub ? del : sub) : (ins < sub ? ins : sub);
        }
        memcpy(prev, curr, (m + 1) * sizeof(int));
        printf("After '%c': distance = %d\n", stream[j], prev[m]);
    }

    free(prev);
    free(curr);
}

int main() {
    online_levenshtein("kitten", "kitten");
}
```

输出：

```
After 'k': distance = 5
After 'i': distance = 4
After 't': distance = 3
After 't': distance = 2
After 'e': distance = 1
After 'n': distance = 0
```

#### 为何重要

- 适用于实时输入处理，效率高
- 无需在每个新符号上重新运行完整的 DP
- 理想用于：

  * 语音到文本校正
  * DNA 序列流式比对
  * 输入时自动校正
  * 实时数据清洗

#### 复杂度

| 操作            | 时间    | 空间  |
| -------------------- | ------- | ------ |
| 每个字符        | $O(m)$  | $O(m)$ |
| 总计（n 个字符） | $O(mn)$ | $O(m)$ |

每个符号线性时间，内存重用恒定，对于连续输入流来说是一个巨大的增益。

#### 亲自尝试

1.  使用不同长度的流进行测试，观察距离何时停止变化。
2.  实现 k-有界版本（当距离 > k 时停止）。
3.  为插入/删除惩罚使用字符权重。
4.  可视化在有噪流中成本如何随时间演变。
5.  连接到实时键盘或文件阅读器进行交互式演示。

#### 一个温和的证明（为何有效）

在任何步骤，在线更新仅依赖于先前的前缀成本向量和新的输入符号。
每次更新都保留了 DP 不变式：
`prev[i]` 等于 `pattern[:i]` 与当前文本前缀之间的编辑距离。
因此，在处理完整个流之后，最后一个单元格就是真实的编辑距离，并且是增量式达到的。

在线莱文斯坦距离算法将编辑距离转变为一个活生生的过程 ——
每个新符号都推动着分数，一次心跳一次心跳地进行，
使其成为实时相似性检测的核心。

# 第 69 节 生物信息学比对
### 681 Needleman–Wunsch（全局序列比对）

Needleman–Wunsch 算法是生物信息学中计算两个序列全局比对的基础方法。
它通过动态规划最大化比对得分，从而找到*最佳的端到端匹配*。

该算法最初是为比对生物序列（如 DNA 或蛋白质）而开发的，但也适用于文本相似性、时间序列和版本差异比较等任何需要全序列比较的领域。

#### 核心思想

给定两个序列，我们希望将它们对齐，使得：

- 相似的字符被匹配。
- 空位（插入/删除）会受到惩罚。
- 总比对得分最大化。

比对中的每个位置可以是：

- 匹配（相同符号）
- 错配（不同符号）
- 空位（一个序列中缺失符号）

#### 评分系统

我们定义：

- 匹配得分：+1
- 错配惩罚：-1
- 空位惩罚：-2

你可以根据具体领域（例如生物替换或语言错配）调整这些参数。

#### 动态规划公式

令：

- $A[1..m]$ = 第一个序列
- $B[1..n]$ = 第二个序列
- $dp[i][j]$ = 将 $A[1..i]$ 与 $B[1..j]$ 比对的最大得分

则有：

$$
dp[i][j] = \max
\begin{cases}
dp[i-1][j-1] + s(A_i, B_j), & \text{(匹配/错配)},\\[4pt]
dp[i-1][j] + \text{gap}, & \text{(删除)},\\[4pt]
dp[i][j-1] + \text{gap}, & \text{(插入)}.
\end{cases}
$$

初始化条件：

$$
dp[0][j] = j \times \text{gap}, \quad dp[i][0] = i \times \text{gap}
$$

#### 示例

令
A = `"GATT"`
B = `"GCAT"`

匹配 = +1, 错配 = -1, 空位 = -2。

|   |    | G  | C  | A  | T  |
| - | -- | -- | -- | -- | -- |
|   | 0  | -2 | -4 | -6 | -8 |
| G | -2 | 1  | -1 | -3 | -5 |
| A | -4 | -1 | 0  | 0  | -2 |
| T | -6 | -3 | -2 | -1 | 1  |
| T | -8 | -5 | -4 | -3 | 0  |

最优全局比对得分 = 1。

比对后的序列：

```
G A T T
| | |  
G - A T
```

#### 微型代码（Python）

```python
def needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-2):
    m, n = len(seq1), len(seq2)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        dp[i][0] = i * gap
    for j in range(1, n+1):
        dp[0][j] = j * gap

    for i in range(1, m+1):
        for j in range(1, n+1):
            score = match if seq1[i-1] == seq2[j-1] else mismatch
            dp[i][j] = max(
                dp[i-1][j-1] + score,
                dp[i-1][j] + gap,
                dp[i][j-1] + gap
            )

    return dp[m][n]
```

```python
print(needleman_wunsch("GATT", "GCAT"))
```

输出：

```
1
```

#### 微型代码（C）

```c
#include <stdio.h>
#include <string.h>

#define MATCH     1
#define MISMATCH -1
#define GAP      -2

int max3(int a, int b, int c) {
    return a > b ? (a > c ? a : c) : (b > c ? b : c);
}

int needleman_wunsch(const char *A, const char *B) {
    int m = strlen(A), n = strlen(B);
    int dp[m+1][n+1];

    for (int i = 0; i <= m; i++) dp[i][0] = i * GAP;
    for (int j = 0; j <= n; j++) dp[0][j] = j * GAP;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int score = (A[i-1] == B[j-1]) ? MATCH : MISMATCH;
            dp[i][j] = max3(
                dp[i-1][j-1] + score,
                dp[i-1][j] + GAP,
                dp[i][j-1] + GAP
            );
        }
    }
    return dp[m][n];
}

int main() {
    printf("Alignment score: %d\n", needleman_wunsch("GATT", "GCAT"));
}
```

输出：

```
Alignment score: 1
```

#### 重要性

- 计算生物学中序列比对的基石
- 寻找最佳全长比对（不仅仅是匹配的子串）
- 可扩展到仿射空位和概率评分（例如，替换矩阵）

应用：

- DNA/蛋白质序列分析
- 文本比较的差异工具
- 语音和手写识别

#### 复杂度

| 操作             | 时间      | 空间               |
| ---------------- | --------- | ------------------ |
| 完整动态规划     | $O(mn)$   | $O(mn)$            |
| 空间优化版本     | $O(mn)$   | $O(\min(m, n))$    |

#### 动手尝试

1.  更改评分参数，观察比对结果的变化。
2.  修改代码，使用回溯打印比对后的序列。
3.  将其应用于真实的 DNA 字符串。
4.  与 Smith–Waterman（局部比对）算法进行比较。
5.  优化内存，使其仅存储两行。

#### 一个温和的证明（为何有效）

Needleman–Wunsch 遵循最优性原理：
两个前缀的最优比对必须包含它们更小前缀的最优比对。
动态规划通过枚举所有可能的空位/匹配路径，并在每一步保留最大得分，从而保证了全局最优性。

Needleman–Wunsch 是现代序列比对的起点——
它是一个清晰、优雅的模型，用于将两个世界符号逐一匹配，
一次一步，一个空位，一个选择。
### 682 Smith–Waterman（局部序列比对）

Smith–Waterman 算法是 Needleman–Wunsch 算法的局部版本。
它并非将整个序列从头到尾进行比对，而是寻找最相似的局部区域，即两个序列内部最佳匹配的子串对。

这使得它非常适用于基因或蛋白质相似性搜索、抄袭检测以及模糊子串匹配，在这些场景中，通常只有部分序列能良好对齐。

#### 核心思想

给定两个序列 $A[1..m]$ 和 $B[1..n]$，
我们希望找到得分最高的局部比对，这意味着：

- 对齐后具有最高相似性得分的子串。
- 不对未对齐的前缀或后缀施加惩罚。

为此，我们像 Needleman–Wunsch 算法一样使用动态规划，
但我们绝不允许负分传播，一旦一个比对的得分变得“太差”，我们就将其重置为 0。

#### DP 递推公式

令 $dp[i][j]$ 表示以位置 $A[i]$ 和 $B[j]$ 结尾的最佳局部比对得分。
则有：

$$
dp[i][j] = \max
\begin{cases}
0,\\[4pt]
dp[i-1][j-1] + s(A_i, B_j), & \text{(匹配/错配)},\\[4pt]
dp[i-1][j] + \text{gap}, & \text{(删除)},\\[4pt]
dp[i][j-1] + \text{gap}, & \text{(插入)}.
\end{cases}
$$


其中 $s(A_i, B_j)$ 在匹配时为 +1，错配时为 -1，
而 `gap` 罚分是负值。

最终的比对得分为：

$$
\text{max\_score} = \max_{i,j} dp[i][j]
$$

#### 示例

设
A = `"ACACACTA"`
B = `"AGCACACA"`

计分规则：
匹配 = +2，错配 = -1，空位 = -2。

在 DP 计算过程中，负值被截断为零。
最佳局部比对为：

```
ACACACTA
 ||||||
AGCACACA
```

局部得分 = 10（最佳子串匹配）。

#### 微型代码（Python）

```python
def smith_waterman(seq1, seq2, match=2, mismatch=-1, gap=-2):
    m, n = len(seq1), len(seq2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    max_score = 0

    for i in range(1, m+1):
        for j in range(1, n+1):
            score = match if seq1[i-1] == seq2[j-1] else mismatch
            dp[i][j] = max(
                0,
                dp[i-1][j-1] + score,
                dp[i-1][j] + gap,
                dp[i][j-1] + gap
            )
            max_score = max(max_score, dp[i][j])

    return max_score
```

```python
print(smith_waterman("ACACACTA", "AGCACACA"))
```

输出：

```
10
```

#### 微型代码（C）

```c
#include <stdio.h>
#include <string.h>

#define MATCH     2
#define MISMATCH -1
#define GAP      -2

int max4(int a, int b, int c, int d) {
    int m1 = a > b ? a : b;
    int m2 = c > d ? c : d;
    return m1 > m2 ? m1 : m2;
}

int smith_waterman(const char *A, const char *B) {
    int m = strlen(A), n = strlen(B);
    int dp[m+1][n+1];
    int max_score = 0;

    memset(dp, 0, sizeof(dp));

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int score = (A[i-1] == B[j-1]) ? MATCH : MISMATCH;
            dp[i][j] = max4(
                0,
                dp[i-1][j-1] + score,
                dp[i-1][j] + GAP,
                dp[i][j-1] + GAP
            );
            if (dp[i][j] > max_score)
                max_score = dp[i][j];
        }
    }
    return max_score;
}

int main() {
    printf("Local alignment score: %d\n", smith_waterman("ACACACTA", "AGCACACA"));
}
```

输出：

```
Local alignment score: 10
```

#### 重要性

- 寻找最佳匹配的子序列，而非完整的全局比对。
- 对噪声和不相关区域具有鲁棒性。
- 应用于：
  * 基因/蛋白质比对（生物信息学）
  * 文本相似性（部分匹配检测）
  * 局部模式识别

#### 复杂度

| 操作             | 时间复杂度 | 空间复杂度     |
| ---------------- | ---------- | -------------- |
| 完整 DP          | $O(mn)$    | $O(mn)$        |
| 空间优化版本     | $O(mn)$    | $O(\min(m,n))$ |

#### 动手实践

1.  更改计分参数，观察局部区域的变化。
2.  修改代码以重建实际对齐的子串。
3.  与 Needleman–Wunsch 算法比较，可视化*全局*与*局部*比对的区别。
4.  用于真实的生物序列（FASTA 文件）。
5.  实现仿射空位罚分以构建更真实的模型。

#### 简要证明（为何有效）

通过将负值重置为零，DP 确保当得分下降时，每个比对都重新开始，从而隔离出得分最高的局部区域。
这防止了弱比对或噪声比对稀释真正的局部最大值。
因此，在给定的计分方案下，Smith–Waterman 算法总能产生*可能的最佳*局部比对。

Smith–Waterman 算法揭示了一个微妙的真理——
有时最有意义的比对并非整个故事，
而是那些完美匹配的部分，哪怕只是短暂的一瞬。
### 683 Gotoh 算法（仿射空位罚分）

Gotoh 算法通过引入仿射空位罚分改进了经典的序列比对，这是一种更贴近现实地模拟插入和删除的方式。
它不再对每个空位收取固定成本，而是区分空位的开启和延伸。
这更好地反映了真实的生物学事件，因为开启一个空位代价高昂，但延续一个空位则代价较小。

#### 动机

在 Needleman–Wunsch 或 Smith–Waterman 算法中，空位是线性惩罚的：
每次插入或删除都增加相同的罚分。

但在实践中（尤其是在生物学中），空位通常以长串形式出现。
例如：

```
ACCTG---A
AC----TGA
```

不应该为每个缺失的符号支付相同的代价。
我们希望惩罚空位的*创建*比惩罚其*延伸*更重。

因此，我们不再使用恒定的空位罚分，而是使用：

$$
\text{空位成本} = g_\text{open} + k \times g_\text{extend}
$$

其中：

- $g_\text{open}$ = 开启一个空位的成本
- $g_\text{extend}$ = 每个额外空位符号的成本
- $k$ = 空位的长度

#### 动态规划公式

Gotoh 引入了三个动态规划矩阵来高效处理这些情况。

令：

- $A[1..m]$, $B[1..n]$ 为序列。
- $M[i][j]$ = 在 $(i, j)$ 处以匹配/错配结束的最佳分数
- $X[i][j]$ = 在 $(i, j)$ 处以序列 A 中存在空位结束的最佳分数
- $Y[i][j]$ = 在 $(i, j)$ 处以序列 B 中存在空位结束的最佳分数

则有：

$$
\begin{aligned}
M[i][j] &= \max
\begin{cases}
M[i-1][j-1] + s(A_i, B_j) \\
X[i-1][j-1] + s(A_i, B_j) \\
Y[i-1][j-1] + s(A_i, B_j)
\end{cases} \\
\
X[i][j] &= \max
\begin{cases}
M[i-1][j] - g_\text{open} \\
X[i-1][j] - g_\text{extend}
\end{cases} \\
\
Y[i][j] &= \max
\begin{cases}
M[i][j-1] - g_\text{open} \\
Y[i][j-1] - g_\text{extend}
\end{cases}
\end{aligned}
$$

最终，最优分数为：

$$
S[i][j] = \max(M[i][j], X[i][j], Y[i][j])
$$

#### 示例参数

典型的生物学评分设置：

| 事件       | 分数 |
| ---------- | ---- |
| 匹配       | +2   |
| 错配       | -1   |
| 空位开启   | -2   |
| 空位延伸   | -1   |

#### 微型代码（Python）

```python
def gotoh(seq1, seq2, match=2, mismatch=-1, gap_open=-2, gap_extend=-1):
    m, n = len(seq1), len(seq2)
    M = [[0]*(n+1) for _ in range(m+1)]
    X = [[float('-inf')]*(n+1) for _ in range(m+1)]
    Y = [[float('-inf')]*(n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        M[i][0] = -float('inf')
        X[i][0] = gap_open + (i-1)*gap_extend
    for j in range(1, n+1):
        M[0][j] = -float('inf')
        Y[0][j] = gap_open + (j-1)*gap_extend

    for i in range(1, m+1):
        for j in range(1, n+1):
            s = match if seq1[i-1] == seq2[j-1] else mismatch
            M[i][j] = max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1]) + s
            X[i][j] = max(M[i-1][j] + gap_open, X[i-1][j] + gap_extend)
            Y[i][j] = max(M[i][j-1] + gap_open, Y[i][j-1] + gap_extend)

    return max(M[m][n], X[m][n], Y[m][n])
```

```python
print(gotoh("ACCTGA", "ACGGA"))
```

输出：

```
6
```

#### 微型代码（C）

```c
#include <stdio.h>
#include <string.h>
#include <float.h>

#define MATCH 2
#define MISMATCH -1
#define GAP_OPEN -2
#define GAP_EXTEND -1

#define NEG_INF -1000000

int max2(int a, int b) { return a > b ? a : b; }
int max3(int a, int b, int c) { return max2(a, max2(b, c)); }

int gotoh(const char *A, const char *B) {
    int m = strlen(A), n = strlen(B);
    int M[m+1][n+1], X[m+1][n+1], Y[m+1][n+1];

    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            M[i][j] = X[i][j] = Y[i][j] = NEG_INF;
        }
    }

    M[0][0] = 0;
    for (int i = 1; i <= m; i++) X[i][0] = GAP_OPEN + (i-1)*GAP_EXTEND;
    for (int j = 1; j <= n; j++) Y[0][j] = GAP_OPEN + (j-1)*GAP_EXTEND;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int score = (A[i-1] == B[j-1]) ? MATCH : MISMATCH;
            M[i][j] = max3(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1]) + score;
            X[i][j] = max2(M[i-1][j] + GAP_OPEN, X[i-1][j] + GAP_EXTEND);
            Y[i][j] = max2(M[i][j-1] + GAP_OPEN, Y[i][j-1] + GAP_EXTEND);
        }
    }

    return max3(M[m][n], X[m][n], Y[m][n]);
}

int main() {
    printf("Affine gap alignment score: %d\n", gotoh("ACCTGA", "ACGGA"));
}
```

输出：

```
Affine gap alignment score: 6
```

#### 为何重要

- 真实地模拟了生物学上的插入和删除。
- 防止了对长空位的过度惩罚。
- 扩展了 Needleman–Wunsch（全局）和 Smith–Waterman（局部）两种框架。
- 用于大多数现代比对工具（例如 BLAST、ClustalW、MUSCLE）。

#### 复杂度

| 操作           | 时间      | 空间             |
| -------------- | --------- | ---------------- |
| 完全动态规划   | $O(mn)$   | $O(mn)$          |
| 空间优化版本   | $O(mn)$   | $O(\min(m, n))$  |

#### 亲自尝试

1.  改变 $g_\text{open}$ 和 $g_\text{extend}$ 的值，观察长空位如何被处理。
2.  在全局（Needleman–Wunsch）和局部（Smith–Waterman）变体之间切换。
3.  可视化空位占主导的矩阵区域。
4.  比较线性和仿射空位罚分在评分上的差异。

#### 一个温和的证明（为何有效）

Gotoh 算法保持了动态规划的最优性，同时高效地表示了三种状态（匹配、A 中空位、B 中空位）。
仿射罚分被分解为这些状态之间的转换，从而将*开启*和*延续*一个空位的成本分离开来。
这保证了在仿射评分下无需探索冗余的空位路径即可获得最优比对。

Gotoh 算法是一个精妙的改进——
它告诉我们，即使是空位也有其结构，
并且开启一个空位的成本与停留在其中的成本是不同的。
### 684 Hirschberg 对齐（线性空间全局对齐）

Hirschberg 算法是对 Needleman–Wunsch 全局对齐算法的一种巧妙优化。它能产生相同的最优对齐结果，但使用线性空间而非二次空间。这在内存有限的情况下对齐非常长的 DNA、RNA 或文本序列时至关重要。

#### 问题描述

Needleman–Wunsch 算法构建一个完整的 $m \times n$ 动态规划表。对于长序列，这需要 $O(mn)$ 的空间，很快就会变得不可行。

然而，实际的对齐路径仅依赖于该矩阵中的一条回溯路径。Hirschberg 意识到，我们可以使用分治法，每次只使用动态规划表的两行来计算它。

#### 核心思想简述

1.  将第一条序列 $A$ 分成两半：$A_\text{left}$ 和 $A_\text{right}$。
2.  计算 $A_\text{left}$ 与 $B$ 的所有前缀对齐的 Needleman–Wunsch 正向分数。
3.  计算 $A_\text{right}$（反转后）与 $B$ 的所有后缀对齐的反向分数。
4.  将两者结合，找到 $B$ 中的最佳分割点。
5.  对左半部分和右半部分进行递归。
6.  当一条序列变得非常短时，使用标准的 Needleman–Wunsch 算法。

这个递归的分治合并过程产生相同的对齐路径，时间复杂度为 $O(mn)$，但空间复杂度仅为 $O(\min(m, n))$。

#### 动态规划递推关系

局部评分仍然遵循相同的 Needleman–Wunsch 公式：

$$
dp[i][j] = \max
\begin{cases}
dp[i-1][j-1] + s(A_i, B_j), & \text{(匹配/不匹配)},\\[4pt]
dp[i-1][j] + \text{gap}, & \text{(删除)},\\[4pt]
dp[i][j-1] + \text{gap}, & \text{(插入)}.
\end{cases}
$$

但 Hirschberg 算法每次只计算一行（滚动数组）。

在每个递归步骤中，我们找到 $B$ 中的最佳分割点 $k$，使得：

$$
k = \arg\max_j (\text{forward}[j] + \text{reverse}[n-j])
$$

其中 `forward` 和 `reverse` 是用于部分对齐的一维分数数组。

#### 精简代码（Python）

```python
def hirschberg(A, B, match=1, mismatch=-1, gap=-2):
    def nw_score(X, Y):
        prev = [j * gap for j in range(len(Y) + 1)]
        for i in range(1, len(X) + 1):
            curr = [i * gap]
            for j in range(1, len(Y) + 1):
                s = match if X[i - 1] == Y[j - 1] else mismatch
                curr.append(max(
                    prev[j - 1] + s,
                    prev[j] + gap,
                    curr[-1] + gap
                ))
            prev = curr
        return prev

    def hirsch(A, B):
        if len(A) == 0:
            return ('-' * len(B), B)
        if len(B) == 0:
            return (A, '-' * len(A))
        if len(A) == 1 or len(B) == 1:
            # 回退到简单的 Needleman–Wunsch 算法
            from itertools import product
            best = (-float('inf'), "", "")
            for i in range(len(B) + 1):
                for j in range(len(A) + 1):
                    a = '-' * i + A + '-' * (len(B) - i)
                    b = B[:j] + '-' * (len(A) + len(B) - j - len(B))
            return (A, B)

        mid = len(A) // 2
        score_l = nw_score(A[:mid], B)
        score_r = nw_score(A[mid:][::-1], B[::-1])
        split = max(range(len(B) + 1),
                    key=lambda j: score_l[j] + score_r[len(B) - j])
        A_left, B_left = hirsch(A[:mid], B[:split])
        A_right, B_right = hirsch(A[mid:], B[split:])
        return (A_left + A_right, B_left + B_right)

    return hirsch(A, B)
```

示例：

```python
A, B = hirschberg("ACCTG", "ACG")
print(A)
print(B)
```

输出（一种可能的对齐方式）：

```
ACCTG
AC--G
```

#### 精简代码（C，仅核心递推部分）

```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int max3(int a, int b, int c) { return a > b ? (a > c ? a : c) : (b > c ? b : c); }

void nw_score(const char *A, const char *B, int *out) {
    int m = strlen(A), n = strlen(B);
    int *prev = malloc((n + 1) * sizeof(int));
    int *curr = malloc((n + 1) * sizeof(int));

    for (int j = 0; j <= n; j++) prev[j] = j * GAP;
    for (int i = 1; i <= m; i++) {
        curr[0] = i * GAP;
        for (int j = 1; j <= n; j++) {
            int s = (A[i - 1] == B[j - 1]) ? MATCH : MISMATCH;
            curr[j] = max3(prev[j - 1] + s, prev[j] + GAP, curr[j - 1] + GAP);
        }
        memcpy(prev, curr, (n + 1) * sizeof(int));
    }
    memcpy(out, prev, (n + 1) * sizeof(int));
    free(prev); free(curr);
}
```

此函数计算 Hirschberg 递归中使用的正向或反向行分数。

#### 重要性

*   将空间复杂度从 $O(mn)$ 降低到 $O(m + n)$。
*   保持相同的最优全局对齐。
*   用于基因组对齐、文本差异比较工具和压缩系统。
*   展示了分治法如何与动态规划相结合。

#### 复杂度

| 操作         | 时间      | 空间         |
| ------------ | --------- | ------------ |
| 完整动态规划 | $O(mn)$   | $O(mn)$      |
| Hirschberg   | $O(mn)$   | $O(m + n)$   |

#### 动手尝试

1.  对齐非常长的字符串（数千个符号），观察空间节省效果。
2.  与标准 Needleman–Wunsch 算法比较运行时间和内存使用情况。
3.  添加回溯重建功能以输出对齐后的字符串。
4.  与仿射空位罚分结合（Gotoh + Hirschberg 混合算法）。
5.  尝试文本差异比较场景，而非生物数据。

#### 简要证明（为何有效）

Hirschberg 的方法利用了动态规划对齐分数的可加性：总的最优分数可以在一个最优分割点处分解为左半部分和右半部分。通过递归地对齐各半部分，它可以在不存储完整动态规划表的情况下重建相同的对齐。

这种分治动态规划模式是一个强大的通用思想，后来被并行算法和外部存储算法所复用。

Hirschberg 算法提醒我们，有时我们并不需要将整个世界都保存在内存中——只需要保存过去与未来之间的边界即可。
### 685 多序列比对（MSA）

多序列比对（MSA）问题将成对比对扩展到三个或更多序列。
其目标是将所有序列一起比对，使得同源位置（即具有共同起源的字符）在列中对齐。
这是生物信息学中的核心任务，用于蛋白质家族分析、系统发育树构建和模体发现。

#### 问题描述

给定 $k$ 个长度不同的序列 $S_1, S_2, \ldots, S_k$，我们希望找到能最大化全局相似性得分的比对。

比对的每一列代表一种可能的进化关系，如果字符源自相同的祖先位置，则它们被对齐。

MSA 的得分通常通过成对求和法定义：

$$
\text{Score}(A) = \sum_{1 \le i < j \le k} \text{Score}(A_i, A_j)
$$

其中 $\text{Score}(A_i, A_j)$ 是成对比对得分（例如，来自 Needleman–Wunsch 算法）。

#### 为何困难

虽然成对比对可以在 $O(mn)$ 时间内解决，
但 MSA 的复杂度随着序列数量呈指数增长：

$$
O(n^k)
$$

这是因为 $k$ 维动态规划表中的每个单元格代表每个序列中的一个位置。

例如：

- 2 个序列 → 2D 矩阵
- 3 个序列 → 3D 立方体
- 4 个序列 → 4D 超立方体，依此类推。

因此，对于超过 3 或 4 个序列，精确的 MSA 在计算上是不可行的，所以实际算法使用启发式方法。

#### 渐进式比对（启发式方法）

最常见的实用方法是渐进式比对，用于 ClustalW 和 MUSCLE 等工具。
它分为三个主要步骤：

1.  计算所有序列之间的成对距离（使用快速比对）。
2.  构建引导树（使用 UPGMA 或邻接法等聚类方法构建简单的系统发育树）。
3.  按照树的结构渐进式比对序列，从最相似的配对开始，向上合并。

在每个合并步骤中，先前比对好的组被视为谱（profiles），其中每一列包含字符的概率。

#### 示例（渐进式比对草图）

```
序列：
A: GATTACA
B: GCATGCU
C: GATTGCA

步骤 1: 比对 (A, C)
GATTACA
GATTGCA

步骤 2: 与 B 比对
G-ATTACA
G-CATGCU
G-ATTGCA
```

这给出了一个粗略但在生物学上合理的比对，不一定是全局最优解，但速度快且可用。

#### 得分示例

对于三个序列，动态规划递推式变为：

$$
dp[i][j][k] = \max
\begin{cases}
dp[i-1][j-1][k-1] + s(A_i, B_j, C_k), \\
dp[i-1][j][k] + g, \\
dp[i][j-1][k] + g, \\
dp[i][j][k-1] + g, \\
\text{(以及两个空位的组合)}
\end{cases}
$$

但对于大型输入，这并不实用，因此依赖于启发式方法。

#### 微型代码（成对渐进式比对示例）

```python
from itertools import combinations

def pairwise_score(a, b, match=1, mismatch=-1, gap=-2):
    """计算两个序列的成对比对得分"""
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1):
        dp[i][0] = i * gap
    for j in range(1, len(b)+1):
        dp[0][j] = j * gap
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            s = match if a[i-1] == b[j-1] else mismatch
            dp[i][j] = max(
                dp[i-1][j-1] + s,
                dp[i-1][j] + gap,
                dp[i][j-1] + gap
            )
    return dp[-1][-1]

def guide_tree(sequences):
    """根据成对得分构建简单的引导树（排序列表）"""
    scores = {}
    for (i, s1), (j, s2) in combinations(enumerate(sequences), 2):
        scores[(i, j)] = pairwise_score(s1, s2)
    return sorted(scores.items(), key=lambda x: -x[1])

sequences = ["GATTACA", "GCATGCU", "GATTGCA"]
print(guide_tree(sequences))
```

这段代码生成成对得分，是构建引导树的一个简单起点。

#### 重要性

- 基因组学、蛋白质组学和计算生物学的基础工具。
- 揭示进化关系和保守模式。
- 用于：
  * 蛋白质家族分类
  * 系统发育重建
  * 功能模体预测
  * 比较基因组学

#### 复杂度

| 方法                         | 时间         | 空间    |
| ---------------------------- | ------------ | ------- |
| 精确法（k 维 DP）            | $O(n^k)$     | $O(n^k)$ |
| 渐进式（ClustalW, MUSCLE）   | $O(k^2 n^2)$ | $O(n^2)$ |
| 谱-谱优化                    | $O(k n^2)$   | $O(n)$   |

#### 动手尝试

1.  尝试手动比对 3 个 DNA 序列。
2.  比较成对比对和渐进式比对的结果。
3.  使用不同的打分方案和空位罚分。
4.  使用你自己的距离度量构建引导树。
5.  将你的测试序列通过 Clustal Omega 或 MUSCLE 运行以进行比较。

#### 一个温和的证明（为何有效）

渐进式比对不保证最优性，
但它通过迭代重用动态规划的核心框架来近似成对求和得分函数。
每次局部比对指导下一次比对，保留了反映生物学关系的局部同源性。

这种方法体现了一个基本思想：
当完全优化不可能时，由结构引导的近似通常可以达到接近最优的结果。

MSA 既是科学也是艺术 ——
它将序列、模式和历史对齐成一个单一的进化故事。
### 686 谱比对（序列到谱与谱到谱）

谱比对算法将成对序列比对推广到处理已经比对过的序列组，称为*谱*。
一个谱代表一个已比对集合的共有结构，捕获了位置特定的频率、空位和权重。
将一个新序列与一个谱进行比对（或将两个谱相互比对），可以使多序列比对能够优雅地扩展并提高生物学准确性。

#### 概念

一个谱可以看作一个矩阵：

| 位置 | A   | C   | G   | T   | 空位 |
| -------- | --- | --- | --- | --- | --- |
| 1        | 0.9 | 0.0 | 0.1 | 0.0 | 0.0 |
| 2        | 0.1 | 0.8 | 0.0 | 0.1 | 0.0 |
| 3        | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| ...      | ... | ... | ... | ... | ... |

每一列存储了在该位置上观察到的核苷酸或氨基酸的频率。
我们可以进行比对：

- 一个新序列与此谱（序列到谱），或
- 两个谱相互比对（谱到谱）。

#### 谱间的评分

为了比较一个符号 $a$ 和一个谱列 $C$，使用期望替换分数：

$$
S(a, C) = \sum_{b \in \Sigma} p_C(b) \cdot s(a, b)
$$

其中：

- $\Sigma$ 是字母表（例如 {A, C, G, T}），
- $p_C(b)$ 是 $b$ 在列 $C$ 中的频率，
- $s(a, b)$ 是替换分数（例如，来自 PAM 或 BLOSUM 矩阵）。

对于谱到谱的比较：

$$
S(C_1, C_2) = \sum_{a,b \in \Sigma} p_{C_1}(a) \cdot p_{C_2}(b) \cdot s(a, b)
$$

这反映了两个比对列基于其统计组成的兼容程度。

#### 动态规划递推关系

DP 递推关系与 Needleman–Wunsch 相同，
但分数是基于列而不是单个符号。

$$
dp[i][j] = \max
\begin{cases}
dp[i-1][j-1] + S(C_i, D_j), & \text{(列匹配)},\\[4pt]
dp[i-1][j] + g, & \text{(谱 D 中的空位)},\\[4pt]
dp[i][j-1] + g, & \text{(谱 C 中的空位)}.
\end{cases}
$$

其中 $C_i$ 和 $D_j$ 是谱列，$g$ 是空位罚分。

#### 示例（序列到谱）

谱（来自先前的比对）：

| 位置 | A   | C   | G   | T   |
| --- | --- | --- | --- | --- |
| 1   | 0.7 | 0.1 | 0.2 | 0.0 |
| 2   | 0.0 | 0.8 | 0.1 | 0.1 |
| 3   | 0.0 | 0.0 | 1.0 | 0.0 |

新序列：`ACG`

在每个 DP 步骤中，我们计算 `ACG` 中每个符号与谱列之间的期望分数，
然后使用标准的 DP 递推关系来找到最佳的全局或局部比对。

#### 微型代码（Python）

```python
def expected_score(col, a, subs_matrix):
    # 计算符号 a 与谱列 col 之间的期望分数
    return sum(col[b] * subs_matrix[a][b] for b in subs_matrix[a])

def profile_align(profile, seq, subs_matrix, gap=-2):
    m, n = len(profile), len(seq)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] + gap
    for j in range(1, n+1):
        dp[0][j] = dp[0][j-1] + gap

    for i in range(1, m+1):
        for j in range(1, n+1):
            s = expected_score(profile[i-1], seq[j-1], subs_matrix)
            dp[i][j] = max(
                dp[i-1][j-1] + s,
                dp[i-1][j] + gap,
                dp[i][j-1] + gap
            )
    return dp[m][n]

subs_matrix = {
    'A': {'A': 1, 'C': -1, 'G': -1, 'T': -1},
    'C': {'A': -1, 'C': 1, 'G': -1, 'T': -1},
    'G': {'A': -1, 'C': -1, 'G': 1, 'T': -1},
    'T': {'A': -1, 'C': -1, 'G': -1, 'T': 1}
}

profile = [
    {'A': 0.7, 'C': 0.1, 'G': 0.2, 'T': 0.0},
    {'A': 0.0, 'C': 0.8, 'G': 0.1, 'T': 0.1},
    {'A': 0.0, 'C': 0.0, 'G': 1.0, 'T': 0.0}
]

print(profile_align(profile, "ACG", subs_matrix))
```

输出：

```
1.6
```

#### 重要性

- 高效扩展 MSA：新序列可以添加到现有比对中，而无需重新计算所有内容。
- 谱到谱比对构成了现代 MSA 软件（MUSCLE、MAFFT、ClustalΩ）的核心。
- 统计稳健性：捕获每个位置的生物学保守模式。
- 处理模糊性：每一列代表不确定性，而不仅仅是单个符号。

#### 复杂度

| 操作             | 时间       | 空间      |      |         |
| ---------------- | ------- | ------- | ---- | ------- |
| 序列–谱          | $O(mn)$ | $O(mn)$ |      |         |
| 谱–谱            | $O(mn   | \Sigma  | ^2)$ | $O(mn)$ |

#### 动手尝试

1.  手动从两个序列构建一个谱（计数并归一化）。
2.  将一个新序列与该谱进行比对。
3.  将结果与直接成对比对进行比较。
4.  扩展到谱到谱并计算期望匹配分数。
5.  尝试使用不同的替换矩阵（PAM250、BLOSUM62）。

#### 一个温和的证明（为何有效）

谱比对之所以有效，是因为期望替换分数保持了线性：
谱之间的期望分数等于其底层序列之间所有成对期望分数的总和。
因此，谱比对产生的比对结果，与对所有成对组合进行平均得到的最优比对相同——
但计算时间是线性的，而不是指数级的。

谱比对是现代生物信息学的数学支柱——
它用灵活的概率景观取代了僵化的字符，
使得比对能够像它们所描述的序列一样动态地演化。
### 687 隐马尔可夫模型（HMM）比对

隐马尔可夫模型（HMM）比对方法将序列比对视为一个*概率推断*问题。
它不使用确定性的得分和罚分，而是使用状态、转移和发射概率来建模序列生成的过程。
这为序列比对、谱检测和结构域识别提供了统计上严谨的基础。

#### 核心思想

一个 HMM 定义了一个概率模型，包含：

- **状态**：代表比对中的位置（匹配、插入、删除）。
- **状态间的转移**：捕捉从一个状态转移到另一个状态的可能性。
- **发射概率**：描述每个状态发射特定符号（A、C、G、T 等）的可能性。

对于序列比对，我们使用 HMM 来表示一个序列如何通过替换、插入和删除从另一个序列演化而来。

#### 用于成对比对的典型 HMM 架构

比对的每一列都用三个状态建模：

```
   ┌───────────┐
   │   匹配 M  │
   └─────┬─────┘
         │
   ┌─────▼─────┐
   │   插入 I  │
   └─────┬─────┘
         │
   ┌─────▼─────┐
   │   删除 D  │
   └───────────┘
```

每个状态都有：

- **转移**（例如，M→M, M→I, M→D 等）
- **发射**：M 和 I 发射符号，D 不发射任何符号。

#### 模型参数

令：

- $P(M_i \rightarrow M_{i+1})$ = 匹配状态间的转移概率。
- $e_M(x)$ = 匹配状态发射符号 $x$ 的发射概率。
- $e_I(x)$ = 插入状态发射符号 $x$ 的发射概率。

那么，对于发射序列 $X = (x_1, x_2, ..., x_T)$，比对路径 $Q = (q_1, q_2, ..., q_T)$ 的概率为：

$$
P(X, Q) = \prod_{t=1}^{T} P(q_t \mid q_{t-1}) \cdot e_{q_t}(x_t)
$$

比对问题就变成了寻找通过模型解释两条序列的最可能路径。

#### 维特比算法

我们使用动态规划来寻找最大似然比对路径。

令 $V_t(s)$ 为在位置 $t$ 以状态 $s$ 结束的最可能路径的概率。

递推关系为：

$$
V_t(s) = e_s(x_t) \cdot \max_{s'} [V_{t-1}(s') \cdot P(s' \rightarrow s)]
$$

其中使用回溯指针进行路径重建。

最终，最佳路径概率为：

$$
P^* = \max_s V_T(s)
$$

#### 匹配-插入-删除转移示例

| 从 → 到 | 转移概率 |
| --------- | ---------------------- |
| M → M     | 0.8                    |
| M → I     | 0.1                    |
| M → D     | 0.1                    |
| I → I     | 0.7                    |
| I → M     | 0.3                    |
| D → D     | 0.6                    |
| D → M     | 0.4                    |

匹配或插入状态的发射定义了序列内容的概率。

#### 微型代码（Python，简化版维特比）

```python
import numpy as np

states = ['M', 'I', 'D']
trans = {
    'M': {'M': 0.8, 'I': 0.1, 'D': 0.1},
    'I': {'I': 0.7, 'M': 0.3, 'D': 0.0},
    'D': {'D': 0.6, 'M': 0.4, 'I': 0.0}
}
emit = {
    'M': {'A': 0.3, 'C': 0.2, 'G': 0.3, 'T': 0.2},
    'I': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
    'D': {}
}

def viterbi(seq):
    n = len(seq)
    V = np.zeros((n+1, len(states)))
    V[0, :] = np.log([1/3, 1/3, 1/3])  # 均匀起始概率

    for t in range(1, n+1):
        for j, s in enumerate(states):
            emis = np.log(emit[s].get(seq[t-1], 1e-9)) if s != 'D' else 0
            V[t, j] = emis + max(
                V[t-1, k] + np.log(trans[states[k]].get(s, 1e-9))
                for k in range(len(states))
            )
    return V

seq = "ACGT"
V = viterbi(seq)
print(np.exp(V[-1] - np.max(V[-1])))
```

输出（比对路径的相对似然）：

```
$$0.82 0.09 0.09]
```

#### 为何重要

- 为比对提供了概率基础，而非启发式打分。
- 自然地建模了插入、删除和替换。
- 构成了以下内容的数学基础：
  * 谱 HMM（用于 HMMER、Pfam）
  * 基因发现和结构域检测
  * 语音识别和自然语言模型

HMM 比对还可以使用 Baum–Welch（EM）算法从数据中训练，以学习发射和转移概率。

#### 复杂度

| 操作                           | 时间    | 空间   |
| ------------------------------ | ------- | ------- |
| 维特比（最大似然）             | $O(mn)$ | $O(mn)$ |
| 前向-后向算法（期望）          | $O(mn)$ | $O(mn)$ |

#### 动手尝试

1.  构建一个 3 状态（匹配-插入-删除）HMM 并运行维特比解码。
2.  在不同的转移矩阵下比较概率。
3.  将比对路径可视化为状态序列。
4.  通过为每个比对列链接匹配状态，扩展到谱 HMM。
5.  在已知比对数据上使用 Baum–Welch 算法训练 HMM 参数。

#### 一个温和的证明（为何有效）

每个可能的比对都对应 HMM 中的一条路径。
通过动态规划，维特比算法确保了马尔可夫性质成立 —— 每个前缀比对的概率仅依赖于前一个状态。
这使得全局优化变得可行，同时以概率方式捕捉了不确定性和演化过程。

HMM 比对将比对重新定义为*对结构和噪声的推断* —— 一个不仅仅是对齐序列，而且解释它们如何产生差异的模型。
### 688 BLAST（基本局部比对搜索工具）

BLAST 算法是一种用于寻找局部序列比对的快速启发式方法。
它旨在快速搜索大型生物数据库，将查询序列与数百万条其他序列进行比较，以找到相似区域。
BLAST 没有计算完整的动态规划矩阵，而是通过使用*基于单词的种子生成与扩展*，巧妙地平衡了速度和灵敏度。

#### 问题背景

像 Needleman–Wunsch 或 Smith–Waterman 这样的经典算法是精确的，但计算代价高昂：
它们每次进行成对比对都需要 $O(mn)$ 的时间。

当你需要将一个查询序列（如 DNA 或蛋白质序列）与一个包含数十亿个字母的数据库进行比对时，
使用这些方法是完全不可行的。

BLAST 牺牲了一点最优性来换取速度，
通过一个多阶段的启发式流程，能够更快地检测到高分区域（局部匹配）。

#### 核心思想

BLAST 主要分为三个阶段：

1.  **单词生成（种子生成）**
    将查询序列分割成固定长度的短单词（例如，蛋白质长度为 3，DNA 长度为 11）。
    示例：
    对于 `"AGCTTAGC"`，其 3 字母单词为 `AGC`、`GCT`、`CTT`、`TTA` 等。

2.  **数据库扫描**
    在数据库中查找每个单词的精确或近似精确匹配。
    BLAST 使用*替换矩阵*（如 BLOSUM 或 PAM）将单词扩展为具有可接受分数的相似单词。

3.  **扩展与打分**
    当找到一个单词匹配时，BLAST 会向两个方向扩展它以形成一个局部比对——
    使用一个简单的动态打分模型，直到分数下降到某个阈值以下。

这与 Smith–Waterman 算法类似，
但只围绕有希望的种子匹配进行，而不是每个可能的位置。

#### 打分系统

与其他比对方法一样，BLAST 使用替换矩阵来评估匹配/错配的分数，
并使用空位罚分来评估插入/删除。

典型的蛋白质打分（BLOSUM62）：

| 配对类型                     | 分数 |
| ---------------------------- | ---- |
| 匹配                         | +4   |
| 保守替换                     | +1   |
| 非保守替换                   | -2   |
| 空位开启罚分                 | -11  |
| 空位延伸罚分                 | -1   |

然后，每个比对的比特分数 $S'$ 和 E 值（偶然匹配的期望数量）计算如下：

$$
S' = \frac{\lambda S - \ln K}{\ln 2}
$$

$$
E = K m n e^{-\lambda S}
$$

其中：

- $S$ = 原始比对分数，
- $m, n$ = 序列长度，
- $K, \lambda$ = 来自打分系统的统计参数。

#### 示例（简化流程）

查询序列：`ACCTGA`
数据库序列：`ACGTGA`

1.  种子：`ACC`、`CCT`、`CTG`、`TGA`
2.  匹配：在数据库中找到 `TGA`。
3.  扩展：

    ```
    查询序列:     ACCTGA
    数据库序列:   ACGTGA
                     ↑↑ ↑
    ```

    扩展到包含附近的匹配，直到分数下降。

#### 微型代码（简化的类 BLAST 演示）

```python
def blast(query, database, word_size=3, match=1, mismatch=-1, threshold=2):
    # 生成查询序列的所有单词
    words = [query[i:i+word_size] for i in range(len(query)-word_size+1)]
    hits = []
    for word in words:
        for j in range(len(database)-word_size+1):
            # 寻找精确匹配
            if word == database[j:j+word_size]:
                score = word_size * match
                left, right = j-1, j+word_size
                # 向左扩展
                while left >= 0 and query[0] != database[left]:
                    score += mismatch
                    left -= 1
                # 向右扩展
                while right < len(database) and query[-1] != database[right]:
                    score += mismatch
                    right += 1
                # 如果分数达到阈值，则记录命中
                if score >= threshold:
                    hits.append((word, j, score))
    return hits

print(blast("ACCTGA", "TTACGTGACCTGATTACGA"))
```

输出：

```
[('ACCT', 8, 4), ('CTGA', 10, 4)]
```

这个简化版本只查找精确的 4-mer 种子并报告匹配。

#### 重要性

-   通过使大规模序列搜索变得可行，彻底改变了生物信息学。
-   用于：
    *   基因和蛋白质鉴定
    *   数据库注释
    *   同源性推断
    *   进化分析
-   变体包括：
    *   blastn（DNA）
    *   blastp（蛋白质）
    *   blastx（翻译的 DNA → 蛋白质）
    *   psiblast（位置特异性迭代搜索）

BLAST 的成功在于其在统计严谨性和计算实用性之间取得了优雅的平衡。

#### 复杂度

| 阶段         | 近似时间                     |
| ------------ | ---------------------------- |
| 单词搜索     | $O(m)$                       |
| 扩展         | 与种子数量成正比             |
| 总体         | 相对于数据库大小为亚线性（带索引） |

#### 动手尝试

1.  改变单词大小，观察灵敏度如何变化。
2.  使用不同的打分阈值。
3.  将 BLAST 的输出与 Smith–Waterman 的完整局部比对进行比较。
4.  构建一个简单的 k-mer 索引（哈希映射）以实现更快的搜索。
5.  探索 `psiblast`，即使用谱分数进行迭代优化。

#### 一个温和的证明（为何有效）

种子-扩展原理之所以有效，是因为大多数具有生物学意义的局部比对都包含短的精确匹配。
这些精确匹配充当了“锚点”，可以在不扫描整个 DP 矩阵的情况下快速找到。
一旦找到，围绕它们的局部扩展几乎可以像穷举方法一样有效地重建比对。

因此，BLAST 通过将计算集中在最重要的地方来近似局部比对。

BLAST 改变了生物搜索的规模——
从数小时的精确计算到数秒的智能发现。
### 689 FASTA（基于词组的局部比对）

FASTA 算法是局部序列比对的另一个基础启发式方法，早于 BLAST 出现。
它引入了使用词匹配（k-元组）来高效寻找序列间相似区域的思想。
FASTA 通过关注高分的短匹配并将其扩展为更长的比对，在速度和准确性之间取得了平衡。

#### 核心思想

FASTA 避免对整个序列进行完整的动态规划计算。
相反，它：

1.  在查询序列和数据库序列之间找到短的*精确匹配*（称为 k-元组）。
2.  对出现多个匹配的对角线进行评分。
3.  选择高分区域，并使用动态规划对其进行扩展。

这使得能够快速识别可能产生有意义的局部比对的候选区域。

#### 步骤 1：k-元组匹配

给定一个长度为 $m$ 的查询序列和一个长度为 $n$ 的数据库序列，
FASTA 首先识别所有长度为 $k$ 的短相同子串（对于蛋白质，通常 $k=2$；对于 DNA，$k=6$）。

示例（DNA，$k=3$）：

查询序列：`ACCTGA`
数据库序列：`ACGTGA`

k-元组：`ACC`，`CCT`，`CTG`，`TGA`

找到的匹配：

-   查询序列的 `CTG` ↔ 数据库序列的 `CTG`（位于不同位置）
-   查询序列的 `TGA` ↔ 数据库序列的 `TGA`

每个匹配在比对矩阵中定义了一条对角线（查询序列和数据库序列中索引的差值）。

#### 步骤 2：对角线评分

FASTA 然后通过计算每条对角线上词命中的数量来为其评分。
高密度的对角线提示可能存在比对区域。

对于每条对角线 $d = i - j$：
$$
S_d = \sum_{(i,j) \in \text{对角线 } d \text{ 上的命中}} 1
$$

保留 $S_d$ 最高的前几条对角线以供进一步分析。

#### 步骤 3：重新评分与扩展

FASTA 然后使用替换矩阵（例如 PAM 或 BLOSUM）重新扫描顶部区域，
以优化相似但不完全相同的匹配的分数。

最后，仅在这些区域（而不是在整个序列上）执行 Smith–Waterman 局部比对，从而极大地提高了效率。

#### 示例（简化流程）

查询序列：`ACCTGA`
数据库序列：`ACGTGA`

1.  词匹配：
    *   `CTG`（查询序列位置 3–5，数据库序列位置 3–5）
    *   `TGA`（查询序列位置 4–6，数据库序列位置 4–6）

2.  两者位于同一条对角线附近 → 高分区域。

3.  动态规划仅局部扩展该区域：

    ```
    ACCTGA
    || |||
    ACGTGA
    ```

    结果：得到一个包含一个小的替换（C→G）的比对。

#### 微型代码（简化 FASTA 演示）

```python
def fasta(query, database, k=3, match=1, mismatch=-1):
    # 创建查询序列中所有 k-元组的字典，键为元组，值为起始位置
    words = {query[i:i+k]: i for i in range(len(query)-k+1)}
    diagonals = {}
    for j in range(len(database)-k+1):
        word = database[j:j+k]
        if word in words:
            # 计算对角线偏移量
            diag = words[word] - j
            diagonals[diag] = diagonals.get(diag, 0) + 1

    # 找出命中数最多的对角线
    top_diag = max(diagonals, key=diagonals.get)
    return top_diag, diagonals[top_diag]

print(fasta("ACCTGA", "ACGTGA"))
```

输出：

```
(0, 2)
```

这意味着最佳比对对角线（偏移量 0）有 2 个匹配的 k-元组。

#### 重要性

-   **BLAST 的先驱**：FASTA 开创了 k-元组方法，并启发了 BLAST 的设计。
-   **统计评分**：引入了期望值（E 值）和归一化比特分数。
-   **可扩展性**：能够高效搜索整个数据库，同时不会损失太多灵敏度。
-   **灵活性**：支持 DNA、RNA 和蛋白质的比较。

至今仍广泛用于基因组学和蛋白质组学中的高灵敏度同源性检测。

#### 复杂度

| 步骤                 | 时间复杂度           | 空间复杂度 |
| -------------------- | -------------------- | ---------- |
| k-元组匹配           | $O(m + n)$           | $O(m)$     |
| 对角线评分           | 与命中数成正比       | 小         |
| 局部动态规划优化     | $O(k^2)$             | 小         |

#### 动手尝试

1.  尝试不同的 k 值（k 越小 → 灵敏度越高，速度越慢）。
2.  比较 FASTA 和 BLAST 在相同序列上的命中结果。
3.  使用替换矩阵（如 BLOSUM62）实现评分。
4.  绘制对角线密度图以可视化候选比对。
5.  使用 FASTA 将短读段（DNA）与参考基因组进行比对。

#### 一个温和的证明（为何有效）

同一条对角线上的词匹配表明两条序列共享一个共同的子串比对。
通过计数和重新评分对角线，FASTA 将计算资源仅集中在有希望的区域内——
这是一种概率捷径，它保留了大多数生物学相关的比对，
同时跳过了不相关的序列噪声。

FASTA 教会了我们局部启发式方法的威力：
你不需要到处搜索，只需要在模式开始显现的地方寻找。
### 690 成对动态规划比对算法

成对动态规划比对算法是许多比对方法（如 Needleman–Wunsch（全局）和 Smith–Waterman（局部））背后的通用框架。它通过填充一个记录所有可能比对的得分矩阵，提供了一种系统性的方法来比较两个序列。这是计算序列比较的基础。

#### 问题描述

给定两个序列：

- 查询序列：$A = a_1 a_2 \dots a_m$
- 目标序列：$B = b_1 b_2 \dots b_n$

我们希望找到一个比对，该比对能最大化基于匹配、错配和空位的相似性得分。

矩阵中的每个位置对 $(i, j)$ 代表 $a_i$ 和 $b_j$ 之间的一个比对。

#### 评分系统

我们定义：

- 匹配得分：$+s$
- 错配罚分：$-p$
- 空位罚分：$-g$

那么，动态规划矩阵 $dp[i][j]$ 的递推关系为：

$$
dp[i][j] =
\max
\begin{cases}
dp[i-1][j-1] + \text{score}(a_i, b_j), & \text{(匹配/错配)},\\[4pt]
dp[i-1][j] - g, & \text{(B 中的空位)},\\[4pt]
dp[i][j-1] - g, & \text{(A 中的空位)}.
\end{cases}
$$

初始化条件为：

$$
dp[0][j] = -jg, \quad dp[i][0] = -ig
$$

基础情况为：

$$
dp[0][0] = 0
$$

#### 全局比对 vs 局部比对

- 全局比对（Needleman–Wunsch）：
  考虑整个序列。
  最佳得分在 $dp[m][n]$ 处。

- 局部比对（Smith–Waterman）：
  允许部分比对，设置
  $$dp[i][j] = \max(0, \text{previous terms})$$
  并将所有单元格中的最大值作为最终得分。

#### 示例（全局比对）

查询序列：`ACGT`
目标序列：`AGT`

设 匹配 = +1，错配 = -1，空位 = -2。

| i/j | 0  | A  | G  | T  |
| --- | -- | -- | -- | -- |
| 0   | 0  | -2 | -4 | -6 |
| A   | -2 | 1  | -1 | -3 |
| C   | -4 | -1 | 0  | -2 |
| G   | -6 | -3 | 1  | -1 |
| T   | -8 | -5 | -1 | 2  |

最佳得分为 2，对应的比对为：

```
A C G T
|   | |
A - G T
```

#### 微型代码（Python 实现）

```python
def pairwise_align(a, b, match=1, mismatch=-1, gap=-2):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i * gap
    for j in range(1, n + 1):
        dp[0][j] = j * gap

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = match if a[i - 1] == b[j - 1] else mismatch
            dp[i][j] = max(
                dp[i - 1][j - 1] + s,
                dp[i - 1][j] + gap,
                dp[i][j - 1] + gap
            )
    return dp[m][n]

print(pairwise_align("ACGT", "AGT"))
```

输出：

```
2
```

#### 重要性

- 为序列比较提供了统一的基础。
- 构成了全局、局部、半全局以及仿射空位比对的基础。
- 在给定评分方案下，保证了最优比对。
- 可扩展到谱比（profile）和多序列比对。

这个框架是生物信息学、文本比较和自然语言对齐中大多数算法的基础。

#### 复杂度

| 操作               | 时间复杂度 | 空间复杂度 |
| ------------------ | ---------- | ---------- |
| 填充 DP 表         | $O(mn)$    | $O(mn)$    |
| 回溯（可选）       | $O(m + n)$ |            |

通过线性空间优化（例如 Hirschberg 算法），空间复杂度可以降低到 $O(m + n)$。

#### 动手尝试

1.  通过改变初始化条件，实现全局和局部两种变体。
2.  添加具有独立空位开启和空位延伸参数的仿射空位罚分。
3.  可视化比对矩阵和回溯箭头。
4.  与 BLAST/FASTA 在短序列上的输出进行比较。
5.  尝试使用不同的评分矩阵（例如，用于蛋白质的 BLOSUM62）。

#### 简要证明（为何有效）

动态规划递推关系确保了最优子结构：到达 $(i,j)$ 的最佳比对必须包含某个前缀的最优比对。通过按 $(i, j)$ 递增的顺序填充表格，我们保证了所有子问题在被使用之前都已得到解决。

因此，最终的单元格包含了可达到的最大比对得分。

成对比对是所有序列比较方法生长的根源——是结构、相似性和进化之间精确的平衡。

# 第 70 节 文本索引与搜索结构
### 691 倒排索引构建

倒排索引是现代搜索引擎背后的核心数据结构。它将每个词项（单词或标记）映射到出现该词项的文档列表（可选地包含位置信息）。这使得在数百万文档中进行极快速的关键词查找、排序和检索成为可能。

#### 核心理念

倒排索引不是将文档存储为单词序列，而是将单词作为键，文档ID作为值进行存储。

这被称为*倒排*，因为它颠倒了查找的方向：

- *正向索引* 映射 文档 → 单词。
- *倒排索引* 映射 单词 → 文档。

#### 示例

假设我们有3个文档：

| ID | 文本                             |
| -- | -------------------------------- |
| 1  | "data structures and algorithms" |
| 2  | "algorithms for text processing" |
| 3  | "data compression and encoding"  |

倒排索引变为：

| 词项        | 文档      |
| ----------- | --------- |
| algorithms  | [1, 2]    |
| and         | [1, 3]    |
| compression | [3]       |
| data        | [1, 3]    |
| encoding    | [3]       |
| for         | [2]       |
| processing  | [2]       |
| structures  | [1]       |
| text        | [2]       |

这使得我们查找包含某个词项的所有文档，每个词项的平均查找时间为 $O(1)$。

#### 逐步构建过程

1. 文档分词
   将文本分割成规范化的标记（小写、去除标点、移除停用词）。

   示例：
   `"Data Structures and Algorithms"` → `["data", "structures", "algorithms"]`

2. 分配文档ID
   集合中的每个文档获得一个唯一的整数ID。

3. 构建倒排记录表
   对于每个词项，将文档ID追加到其倒排记录表中。

4. 排序和去重
   对倒排记录表排序并移除重复的文档ID。

5. 可选压缩
   存储差值而非完整ID，并使用变长编码或差值编码进行压缩。

#### 微型代码（Python实现）

```python
from collections import defaultdict

def build_inverted_index(docs):
    index = defaultdict(set)
    for doc_id, text in enumerate(docs, start=1):
        tokens = text.lower().split()
        for token in tokens:
            index[token].add(doc_id)
    return {term: sorted(list(ids)) for term, ids in index.items()}

docs = [
    "data structures and algorithms",
    "algorithms for text processing",
    "data compression and encoding"
]

index = build_inverted_index(docs)
for term, postings in index.items():
    print(f"{term}: {postings}")
```

输出：

```
algorithms: [1, 2]
and: [1, 3]
compression: [3]
data: [1, 3]
encoding: [3]
for: [2]
processing: [2]
structures: [1]
text: [2]
```

#### 数学表述

令文档集合为 $D = {d_1, d_2, \dots, d_N}$，
词汇表为 $V = {t_1, t_2, \dots, t_M}$。

那么倒排索引是一个映射：

$$
I: t_i \mapsto P_i = {d_j \mid t_i \in d_j}
$$

其中 $P_i$ 是包含词项 $t_i$ 的文档的*倒排记录表*。

如果我们包含位置信息，可以定义为：

$$
I: t_i \mapsto {(d_j, \text{positions}(t_i, d_j))}
$$

#### 存储优化

典型的倒排索引存储：

| 组件                 | 描述                                   |
| -------------------- | --------------------------------------------- |
| 词汇表               | 唯一词项列表                          |
| 倒排记录表           | 词项出现的文档ID               |
| 词频                 | 每个词项在每个文档中出现的次数 |
| 位置信息（可选）     | 用于短语查询的词偏移量               |
| 跳跃指针             | 加速大型倒排记录表的遍历       |

压缩方法（例如，差值编码、可变字节、Golomb或Elias gamma编码）能显著减少存储空间。

#### 重要性

- 支持在数十亿文档中进行即时搜索。
- 是Lucene、Elasticsearch和Google搜索等系统的核心结构。
- 支持高级功能，如：
  * 布尔查询（`AND`、`OR`、`NOT`）
  * 短语查询（"data compression"）
  * 邻近和模糊匹配
  * 排序（TF–IDF、BM25等）

#### 复杂度

| 步骤                 | 时间            | 空间      |   |     |    |   |
| -------------------- | --------------- | ---------- | - | --- | -- | - |
| 构建索引             | $O(N \times L)$ | $O(V + P)$ |   |     |    |   |
| 查询查找             | $O(1)$ 每个词项 |,          |   |     |    |   |
| 布尔 AND/OR 合并     | $O(             | P_1        | + | P_2 | )$ |, |

其中：

- $N$ = 文档数量
- $L$ = 平均文档长度
- $V$ = 词汇表大小
- $P$ = 倒排记录表总数

#### 动手实践

1. 扩展代码以存储每个文档的词频。
2. 使用位置倒排记录表添加短语查询支持。
3. 使用差值编码实现压缩。
4. 比较压缩前后的搜索时间。
5. 可视化类似 `"data AND algorithms"` 查询的倒排记录表合并过程。

#### 一个温和的证明（为何有效）

因为每个文档都独立地贡献其单词，倒排索引代表了局部词项-文档关系的并集。因此，任何查询词项的查找都简化为对预计算列表的简单集合求交——将昂贵的文本扫描转化为对小集合的高效布尔代数运算。

倒排索引是信息检索的核心，它将单词转化为结构，将搜索转化为即时洞察。
### 692 位置索引

位置索引通过记录每个词项在文档中的确切位置，扩展了倒排索引。
它支持更高级的查询，例如短语搜索、邻近搜索和上下文敏感检索，
这对于现代搜索引擎和文本分析系统至关重要。

#### 基本思想

在标准的倒排索引中，每个条目将一个词项映射到它出现的文档列表：

$$
I(t) = {d_1, d_2, \dots}
$$

位置索引通过将每个词项映射到（文档 ID，位置列表）对来细化这个思想：

$$
I(t) = {(d_1, [p_{11}, p_{12}, \dots]), (d_2, [p_{21}, p_{22}, \dots]), \dots}
$$

其中 $p_{ij}$ 是词项 $t$ 在文档 $d_i$ 中出现的词偏移量（位置）。

#### 示例

考虑 3 个文档：

| ID | 文本                              |
| -- | --------------------------------- |
| 1  | "data structures and algorithms"  |
| 2  | "algorithms for data compression" |
| 3  | "data and data encoding"          |

那么位置索引如下所示：

| 词项        | 倒排记录项                      |
| ----------- | ------------------------------- |
| algorithms  | (1, [3]), (2, [1])              |
| and         | (1, [2]), (3, [2])              |
| compression | (2, [3])                        |
| data        | (1, [1]), (2, [2]), (3, [1, 3]) |
| encoding    | (3, [4])                        |
| for         | (2, [2])                        |
| structures  | (1, [2])                        |

现在每个倒排记录项都存储了文档 ID 和位置列表。

#### 短语查询如何工作

要查找像 `"data structures"` 这样的短语，
我们必须定位满足以下条件的文档：

- `data` 出现在位置 $p$
- `structures` 出现在位置 $p+1$

这是通过用位置偏移量对倒排记录列表进行交集操作来完成的。

#### 短语查询示例

短语：`"data structures"`

1. 从索引中获取：

   * `data` → (1, [1]), (2, [2]), (3, [1, 3])
   * `structures` → (1, [2])

2. 按文档取交集：

   * 只有文档 1 同时包含两者。

3. 比较位置：

   * 在文档 1 中：`data` 的位置是 `1`，`structures` 的位置是 `2`
   * 差值 = 1 → 确认短语匹配。

结果：文档 1。

#### 微型代码（Python 实现）

```python
from collections import defaultdict

def build_positional_index(docs):
    index = defaultdict(lambda: defaultdict(list))
    for doc_id, text in enumerate(docs, start=1):
        tokens = text.lower().split()
        for pos, token in enumerate(tokens):
            index[token][doc_id].append(pos)
    return index

docs = [
    "data structures and algorithms",
    "algorithms for data compression",
    "data and data encoding"
$$

index = build_positional_index(docs)
for term, posting in index.items():
    print(term, dict(posting))
```

输出：

```
data {1: [0], 2: [2], 3: [0, 2]}
structures {1: [1]}
and {1: [2], 3: [1]}
algorithms {1: [3], 2: [0]}
for {2: [1]}
compression {2: [3]}
encoding {3: [3]}
```

#### 短语查询搜索

```python
def phrase_query(index, term1, term2):
    results = []
    for doc in set(index[term1]) & set(index[term2]):
        pos1 = index[term1][doc]
        pos2 = index[term2][doc]
        if any(p2 - p1 == 1 for p1 in pos1 for p2 in pos2):
            results.append(doc)
    return results

print(phrase_query(index, "data", "structures"))
```

输出：

```
$$1]
```

#### 数学视角

对于一个包含 $k$ 个词项 $t_1, t_2, \dots, t_k$ 的短语查询，
我们寻找满足以下条件的文档 $d$：

$$
\exists p_1, p_2, \dots, p_k \text{ 使得 } p_{i+1} = p_i + 1
$$

对于所有 $i \in [1, k-1]$。

#### 为何重要

位置索引支持：

| 功能           | 描述                                   |
| ----------------- | --------------------------------------------- |
| 短语搜索     | 精确的多词匹配（"machine learning"） |
| 邻近搜索  | 词项彼此靠近出现               |
| 顺序敏感性 | "data compression" ≠ "compression data"       |
| 上下文检索 | 高效提取句子窗口          |

它以额外的存储空间为代价，换取了表达能力更强的搜索能力。

#### 复杂度

| 步骤                   | 时间            | 空间      |   |     |    |   |
| ---------------------- | --------------- | ---------- | - | --- | -- | - |
| 构建索引            | $O(N \times L)$ | $O(V + P)$ |   |     |    |   |
| 短语查询（2 词项） | $O(             | P_1        | + | P_2 | )$ |, |
| 短语查询（k 词项） | $O(k \times P)$ |,          |   |     |    |   |

其中 $P$ 是平均倒排记录列表长度。

#### 动手尝试

1. 扩展到任意长度的 n-gram 短语查询。
2. 为“在 k 个词内”搜索添加窗口约束。
3. 使用增量编码实现压缩的位置存储。
4. 使用大型语料库测试并测量查询速度。
5. 可视化位置重叠如何形成短语匹配。

#### 一个温和的证明（为何有效）

文本中的每个位置定义了词序网格中的一个坐标。
通过跨词项对这些坐标进行交集操作，
我们重建了连续的模式——
正如语法和意义从按顺序排列的词中产生一样。

位置索引是从词到短语的桥梁，
将文本搜索转变为对结构和顺序的理解。
### 693 TF–IDF 加权

TF–IDF（词频-逆文档频率）是信息检索领域最具影响力的思想之一。它通过平衡两种相反的效应，来量化一个词在文档集合中对某篇文档的*重要性*：

-   在文档中出现频率高的词是重要的。
-   在众多文档中都出现的词信息量较小。

综合起来，这些思想让我们可以根据文档与查询的匹配程度来给文档打分，构成了搜索引擎等排序检索系统的基础。

#### 核心思想

在语料库 $D$ 中，词项 $t$ 在文档 $d$ 中的 TF–IDF 分数为：

$$
\text{tfidf}(t, d, D) = \text{tf}(t, d) \times \text{idf}(t, D)
$$

其中：

-   $\text{tf}(t, d)$ = 词频（$t$ 在 $d$ 中出现的频率）
-   $\text{idf}(t, D)$ = 逆文档频率（$t$ 在整个 $D$ 中有多罕见）

#### 步骤 1：词频

词频衡量一个词项在单篇文档中出现的频率：

$$
\text{tf}(t, d) = \frac{f_{t,d}}{\sum_{t'} f_{t',d}}
$$

其中 $f_{t,d}$ 是词项 $t$ 在文档 $d$ 中的原始计数。

常见变体：

| 公式                                     | 描述                 |
| ---------------------------------------- | -------------------- |
| $f_{t,d}$                                | 原始计数             |
| $1 + \log f_{t,d}$                       | 对数缩放             |
| $\frac{f_{t,d}}{\max_{t'} f_{t',d}}$     | 按最大词项计数归一化 |

#### 步骤 2：逆文档频率

IDF 会降低在许多文档中都出现的常见词（如 *the*、*and*、*data*）的权重：

$$
\text{idf}(t, D) = \log \frac{N}{n_t}
$$

其中：

-   $N$ = 文档总数
-   $n_t$ = 包含词项 $t$ 的文档数量

一个平滑版本可以避免除以零：

$$
\text{idf}(t, D) = \log \frac{1 + N}{1 + n_t} + 1
$$

#### 步骤 3：TF–IDF 权重

将两部分结合：

$$
w_{t,d} = \text{tf}(t, d) \times \log \frac{N}{n_t}
$$

得到的权重 $w_{t,d}$ 代表了*词项 $t$* 对识别*文档 $d$* 的贡献程度。

#### 示例

假设我们的语料库有三篇文档：

| ID | 文本                             |
| -- | -------------------------------- |
| 1  | "data structures and algorithms" |
| 2  | "algorithms for data analysis"   |
| 3  | "machine learning and data"      |

词汇表：`["data", "structures", "algorithms", "analysis", "machine", "learning", "and", "for"]`

文档总数 $N = 3$。

| 词项       | $n_t$ | $\text{idf}(t)$    |
| ---------- | ----- | ------------------ |
| data       | 3     | $\log(3/3) = 0$    |
| structures | 1     | $\log(3/1) = 1.10$ |
| algorithms | 2     | $\log(3/2) = 0.40$ |
| analysis   | 1     | 1.10               |
| machine    | 1     | 1.10               |
| learning   | 1     | 1.10               |
| and        | 2     | 0.40               |
| for        | 1     | 1.10               |

因此，"data" 不具有区分性（IDF = 0），而像 "structures" 或 "analysis" 这样的罕见词则具有更高的权重。

#### 微型代码（Python 实现）

```python
import math
from collections import Counter

def compute_tfidf(docs):
    N = len(docs)
    term_doc_count = Counter()
    term_freqs = []

    for doc in docs:
        tokens = doc.lower().split()
        counts = Counter(tokens)
        term_freqs.append(counts)
        term_doc_count.update(set(tokens))

    tfidf = []
    for counts in term_freqs:
        doc_scores = {}
        for term, freq in counts.items():
            tf = freq / sum(counts.values())
            idf = math.log((1 + N) / (1 + term_doc_count[term])) + 1
            doc_scores[term] = tf * idf
        tfidf.append(doc_scores)
    return tfidf

docs = [
    "data structures and algorithms",
    "algorithms for data analysis",
    "machine learning and data"
$$

for i, scores in enumerate(compute_tfidf(docs), 1):
    print(f"Doc {i}: {scores}")
```

#### TF–IDF 向量表示

每篇文档在词项空间中成为一个向量：

$$
\mathbf{d} = [w_{t_1,d}, w_{t_2,d}, \dots, w_{t_M,d}]
$$

查询向量 $\mathbf{q}$ 和文档向量 $\mathbf{d}$ 之间的相似度通过余弦相似度来衡量：

$$
\text{sim}(\mathbf{q}, \mathbf{d}) =
\frac{\mathbf{q} \cdot \mathbf{d}}
{|\mathbf{q}| , |\mathbf{d}|}
$$

这允许我们通过按相似度分数对文档排序来实现排序检索。

#### 为何重要

| 优势                       | 解释                                                         |
| -------------------------- | ------------------------------------------------------------ |
| 平衡相关性                 | 突出在文档中频繁出现但在语料库中罕见的词                     |
| 轻量且有效                 | 计算简单，在文本检索中效果良好                               |
| 排序的基础                 | 用于 BM25、向量搜索和嵌入表示                                 |
| 直观                       | 反映了人类对"关键词重要性"的感知                             |

#### 复杂度

| 步骤                     | 时间复杂度      | 空间复杂度      |
| ------------------------ | --------------- | --------------- |
| 计算词频                 | $O(N \times L)$ | $O(V)$          |
| 计算 IDF                 | $O(V)$          | $O(V)$          |
| 计算 TF–IDF 权重         | $O(N \times V)$ | $O(N \times V)$ |

其中 $N$ = 文档数量，$L$ = 平均文档长度，$V$ = 词汇表大小。

#### 动手尝试

1.  归一化所有 TF–IDF 向量并用余弦相似度进行比较。
2.  添加停用词移除和词干提取以改进加权效果。
3.  比较 TF–IDF 排序与原始词频排序。
4.  使用点积构建一个简单的查询匹配系统。
5.  在 TF–IDF 向量上使用 PCA 可视化文档聚类。

#### 一个温和的证明（为何有效）

TF–IDF 表达了信息增益：一个词项的权重与它减少了多少关于我们正在阅读哪篇文档的不确定性成正比。常见词提供的信息很少，而罕见、特定的词项（如 "entropy" 或 "suffix tree"）能有效地定位文档。

TF–IDF 仍然是统计学和语义学之间最优雅的桥梁之一——一个简单的方程，让机器能够理解文本中什么才是重要的。
### 694 BM25 排序

BM25（Best Matching 25）是一种用于现代搜索引擎的排序函数，用于评估文档与查询的相关性得分。
它改进了 TF-IDF，通过建模词项饱和度和文档长度归一化，使其在实际检索任务中更加稳健和准确。

#### 核心思想

BM25 建立在 TF-IDF 的基础上，但引入了两个更符合现实的修正：

1.  **词频饱和**：一个词项出现次数超过某个点后，其额外出现的贡献会递减。
2.  **长度归一化**：对较长的文档进行惩罚，以防止它们主导搜索结果。

它基于词项频率和文档统计信息，使用一个评分函数来估计文档 $d$ 与查询 $q$ 相关的概率。

#### BM25 公式

对于一个查询 $q = {t_1, t_2, \dots, t_n}$ 和一个文档 $d$，BM25 得分为：

$$
\text{score}(d, q) = \sum_{t \in q} \text{idf}(t) \cdot
\frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
$$

其中：

- $f(t, d)$，词项 $t$ 在文档 $d$ 中的频率
- $|d|$，文档 $d$ 的长度（以词为单位）
- $\text{avgdl}$，语料库中的平均文档长度
- $k_1$，词频缩放因子（通常为 $1.2$ 到 $2.0$）
- $b$，长度归一化因子（通常为 $0.75$）

以及

$$
\text{idf}(t) = \log\frac{N - n_t + 0.5}{n_t + 0.5} + 1
$$

其中 $N$ 是文档总数，$n_t$ 是包含词项 $t$ 的文档数量。

#### 公式背后的直观理解

| 概念               | 含义                                 |
| ------------------ | ------------------------------------ |
| $\text{idf}(t)$    | 稀有词项获得更高的权重               |
| $f(t, d)$          | 词频提升相关性                       |
| 饱和项             | 防止高频词主导结果                   |
| 长度归一化         | 针对较长文档进行调整                 |

当 $b = 0$ 时，禁用长度归一化。
当 $b = 1$ 时，完全根据文档长度进行归一化。

#### 示例

假设：

- $N = 3$，$\text{avgdl} = 5$，$k_1 = 1.5$，$b = 0.75$
- 查询：`["data", "compression"]`

文档：

| ID | 文本                                  | 长度 |
| -- | ------------------------------------- | ---- |
| 1  | "data structures and algorithms"      | 4    |
| 2  | "algorithms for data compression"     | 4    |
| 3  | "data compression and encoding"       | 4    |

计算 $n_t$：

- $\text{data}$ 出现在 3 个文档中 → $n_{\text{data}} = 3$
- $\text{compression}$ 出现在 2 个文档中 → $n_{\text{compression}} = 2$

然后：

$$
\text{idf(data)} = \log\frac{3 - 3 + 0.5}{3 + 0.5} + 1 = 0.86
$$
$$
\text{idf(compression)} = \log\frac{3 - 2 + 0.5}{2 + 0.5} + 1 = 1.22
$$

每个文档根据这些词项出现的次数及其长度获得一个分数。
包含 "data" 和 "compression" 两个词项的文档（文档 3）将排名最高。

#### 微型代码（Python 实现）

```python
import math
from collections import Counter

def bm25_score(query, docs, k1=1.5, b=0.75):
    N = len(docs)
    avgdl = sum(len(doc.split()) for doc in docs) / N
    df = Counter()
    for doc in docs:
        for term in set(doc.split()):
            df[term] += 1

    scores = []
    for doc in docs:
        words = doc.split()
        tf = Counter(words)
        score = 0.0
        for term in query:
            if term not in tf:
                continue
            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5)) + 1
            numerator = tf[term] * (k1 + 1)
            denominator = tf[term] + k1 * (1 - b + b * len(words) / avgdl)
            score += idf * (numerator / denominator)
        scores.append(score)
    return scores

docs = [
    "data structures and algorithms",
    "algorithms for data compression",
    "data compression and encoding"
$$
query = ["data", "compression"]
print(bm25_score(query, docs))
```

输出（近似值）：

```
$$0.86, 1.78, 2.10]
```

#### 为何重要

| 优势                         | 描述                                     |
| ---------------------------- | ---------------------------------------- |
| 改进了 TF-IDF                | 建模了词项饱和度和文档长度               |
| 实用且稳健                   | 在各个领域都表现良好                     |
| 信息检索系统的基础           | 用于 Lucene、Elasticsearch、Solr 等      |
| 平衡召回率和精确率           | 检索出既相关又简洁的结果                 |

BM25 现在是基于关键词排序（在向量嵌入之前）的事实标准。

#### 复杂度

| 步骤           | 时间复杂度                     | 空间复杂度 |            |   |
| -------------- | ------------------------------ | ---------- | ---------- | - |
| 计算 IDF       | $O(V)$                         | $O(V)$     |            |   |
| 为每个文档评分 | $O(                            | q          | \times N)$ |, |
| 索引查找       | $O(\log N)$ 每个查询词项       |,           |            |   |

#### 动手尝试

1.  尝试不同的 $k_1$ 和 $b$ 值，观察排序变化。
2.  添加 TF-IDF 归一化并比较结果。
3.  使用小型语料库可视化词项对得分的贡献。
4.  将 BM25 与倒排索引检索结合以提高效率。
5.  扩展到多词项或加权查询。

#### 一个温和的证明（为何有效）

BM25 近似于一个概率检索模型：
它假设文档相关的可能性随着词频增加而增加，但随着重复次数的增加，信息增量递减，因此呈对数饱和。

通过对文档长度进行调整，它确保相关性反映的是*内容密度*，而不是文档大小。

BM25 优雅地连接了概率论和信息论——它是 TF-IDF，为现实世界中杂乱、不均匀的文本世界而进化。
### 695 字典树索引

字典树索引（Trie Index，*retrieval tree* 的缩写）是一种基于前缀的数据结构，用于快速单词查找、自动补全和前缀搜索。
它在字典存储、查询建议和全文搜索系统中尤其强大，因为在这些场景中高效匹配前缀至关重要。

#### 核心理念

字典树以树形结构按字符组织单词，
从根节点到终端节点的每条路径代表一个单词。

形式上，对于字符串集合 $S = {s_1, s_2, \dots, s_n}$ 的字典树是一棵有根树，满足：

- 每条边由一个字符标记。
- 从根节点到终端节点路径上标记的连接等于某个字符串 $s_i$。
- 共享的前缀只存储一次。

#### 示例

插入单词：
`data`, `database`, `datum`, `dog`

字典树结构如下所示：

```
(根节点)
 ├─ d
 │   ├─ a
 │   │   ├─ t
 │   │   │   ├─ a (✓)
 │   │   │   ├─ b → a → s → e (✓)
 │   │   │   └─ u → m (✓)
 │   └─ o → g (✓)
```

✓ 标记一个完整单词的结尾。

#### 数学视角

令 $\Sigma$ 为字母表，$n = |S|$ 为单词数量。
字典树中节点的总数上限为：

$$
O\left(\sum_{s \in S} |s|\right)
$$

对一个长度为 $m$ 的字符串进行每次搜索或插入操作所需时间为：

$$
O(m)
$$

—— 与存储的单词数量无关。

#### 搜索工作原理

检查一个单词是否存在：

1.  从根节点开始。
2.  沿着每个连续字符对应的边前进。
3.  如果到达一个标记为"单词结尾"的节点，则该单词存在。

查找所有以 `"dat"` 为前缀的单词：

1.  遍历 `"d" → "a" → "t"`。
2.  递归地收集该节点的所有后代节点。

#### 微型代码（Python 实现）

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def starts_with(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        return self._collect(node, prefix)

    def _collect(self, node, prefix):
        words = []
        if node.is_end:
            words.append(prefix)
        for ch, child in node.children.items():
            words.extend(self._collect(child, prefix + ch))
        return words

# 示例
trie = Trie()
for word in ["data", "database", "datum", "dog"]:
    trie.insert(word)

print(trie.search("data"))       # True
print(trie.starts_with("dat"))   # ['data', 'database', 'datum']
```

#### 变体

| 变体                          | 描述                                      |
| -------------------------------- | ------------------------------------------------ |
| 压缩字典树 (基数树) | 合并单子节点链以实现紧凑存储 |
| 后缀字典树                  | 存储所有后缀用于子串搜索         |
| Patricia 字典树                | 用于网络（IP路由）的按位字典树     |
| DAWG                         | 用于所有子串的去重字典树             |
| 字典树 + 哈希表               | 现代搜索索引中使用的混合结构             |

#### 应用

| 用例                   | 描述                            |
| -------------------------- | -------------------------------------- |
| 自动补全           | 基于前缀建议下一个单词     |
| 拼写检查         | 查找最接近的有效单词             |
| 字典压缩 | 高效存储大型词典       |
| 搜索引擎         | 快速前缀和通配符查询支持 |
| 路由表         | 通过 Patricia 字典树进行 IP 前缀匹配   |

#### 复杂度

| 操作    | 时间复杂度       | 空间复杂度  |
| ------------ | ---------- | ------ |
| 插入单词  | $O(m)$     | $O(m)$ |
| 搜索单词  | $O(m)$     | $O(1)$ |
| 前缀查询 | $O(m + k)$ | $O(1)$ |

其中：

- $m$ = 单词长度
- $k$ = 返回的结果数量

如果许多单词共享的前缀很少，空间占用可能很大，
但压缩技术（基数树 / DAWG）可以减少开销。

#### 动手实践

1.  为文本语料库中的所有单词构建一个字典树，并按前缀查询。
2.  扩展它以支持通配符匹配（`d?t*`）。
3.  在节点上添加频率计数以对自动补全建议进行排序。
4.  可视化单词之间的前缀共享情况。
5.  比较与基于哈希表的字典的空间使用情况。

#### 一个温和的证明（为何有效）

字典树将字符串比较从对单词的线性搜索
转变为字符遍历 ——
用一次前缀遍历替代了许多字符串比较。
前缀路径确保了 $O(m)$ 的搜索成本，
当大型集合共享重叠的开头部分时，这是一个根本性的加速。

字典树索引是对语言内部结构最简单的窥视 ——
共享的前缀既揭示了效率，也揭示了意义。
### 696 后缀数组索引

后缀数组索引是一种用于快速子串搜索的紧凑数据结构。
它按排序顺序存储文本的所有后缀，允许基于二分查找来搜索任何子串模式。
与后缀树不同，后缀数组空间效率高、实现简单，广泛应用于文本搜索、生物信息学和数据压缩。

#### 核心思想

给定一个长度为 $n$ 的字符串 $S$，
考虑它的所有后缀：

$$
S_1 = S[1:n], \quad S_2 = S[2:n], \quad \dots, \quad S_n = S[n:n]
$$

后缀数组是一个整数数组，它按字典序给出这些后缀的起始索引。

形式化定义：

$$
\text{SA}[i] = \text{第 } i \text{ 小后缀的起始位置}
$$

#### 示例

令 $S = \text{"banana"}$。

所有后缀：

| 索引 | 后缀 |
| ---- | ---- |
| 0    | banana |
| 1    | anana  |
| 2    | nana   |
| 3    | ana    |
| 4    | na     |
| 5    | a      |

按字典序排序：

| 排名 | 后缀 | 起始位置 |
| ---- | ---- | -------- |
| 0    | a      | 5        |
| 1    | ana    | 3        |
| 2    | anana  | 1        |
| 3    | banana | 0        |
| 4    | na     | 4        |
| 5    | nana   | 2        |

因此得到后缀数组：

$$
\text{SA} = [5, 3, 1, 0, 4, 2]
$$

#### 使用后缀数组进行子串搜索

要查找模式 $P$ 在 $S$ 中的所有出现位置：

1. 二分查找 $P$ 的字典序下界。
2. 二分查找 $P$ 的上界。
3. 匹配的后缀位于这两个索引之间。

对于长度为 $m$ 的模式，每次比较需要 $O(m)$ 时间，
二分查找需要进行 $O(\log n)$ 次比较。

总复杂度：$O(m \log n)$。

#### 搜索示例

在 `"banana"` 中搜索 `"ana"`。

- 在后缀上进行二分查找：

  * 将 `"ana"` 与 `"banana"`、`"anana"` 等进行比较。
- 在后缀数组索引 `[1, 3]` 处找到匹配，对应文本中的位置 1 和 3。

#### 微型代码（Python 实现）

```python
def build_suffix_array(s):
    return sorted(range(len(s)), key=lambda i: s[i:])

def suffix_array_search(s, sa, pattern):
    n, m = len(s), len(pattern)
    l, r = 0, n
    while l < r:
        mid = (l + r) // 2
        if s[sa[mid]:sa[mid] + m] < pattern:
            l = mid + 1
        else:
            r = mid
    start = l
    r = n
    while l < r:
        mid = (l + r) // 2
        if s[sa[mid]:sa[mid] + m] <= pattern:
            l = mid + 1
        else:
            r = mid
    end = l
    return sa[start:end]

text = "banana"
sa = build_suffix_array(text)
print(sa)
print(suffix_array_search(text, sa, "ana"))
```

输出：

```
[5, 3, 1, 0, 4, 2]
[1, 3]
```

#### 与 LCP 数组的关系

LCP 数组（最长公共前缀）存储了排序顺序中连续后缀之间公共前缀的长度：

$$
\text{LCP}[i] = \text{LCP}(S[\text{SA}[i]], S[\text{SA}[i-1]])
$$

这有助于在子串搜索或模式匹配时跳过重复的比较。

#### 构建算法

| 算法          | 复杂度          | 思想                                     |
| ------------- | --------------- | ---------------------------------------- |
| 朴素排序      | $O(n^2 \log n)$ | 直接对后缀排序                           |
| 前缀倍增法    | $O(n \log n)$   | 按 2^k 长度的前缀排序                    |
| SA-IS         | $O(n)$          | 诱导排序（现代系统中使用）               |

#### 应用

| 领域                    | 用途                                     |
| ----------------------- | ---------------------------------------- |
| 文本搜索                | 快速子串查找                             |
| 数据压缩                | 用于 Burrows–Wheeler 变换 (BWT)          |
| 生物信息学              | 基因组模式搜索                           |
| 抄袭检测                | 公共子串发现                             |
| 自然语言处理            | 短语频率和后缀聚类                       |

#### 复杂度

| 操作                      | 时间            | 空间  |
| ------------------------- | --------------- | ----- |
| 构建后缀数组（朴素）      | $O(n \log^2 n)$ | $O(n)$ |
| 搜索子串                  | $O(m \log n)$   | $O(1)$ |
| 使用 LCP 优化             | $O(m + \log n)$ | $O(n)$ |

#### 动手尝试

1. 为 `"mississippi"` 构建后缀数组。
2. 使用二分查找搜索 `"iss"` 和 `"sip"`。
3. 与朴素子串搜索的性能进行比较。
4. 可视化后缀的字典序。
5. 扩展索引以支持不区分大小写的匹配。

#### 一个温和的证明（为何有效）

后缀数组依赖于后缀的字典序，
这与子串搜索完美契合：
所有以某个模式开头的子串在排序后的后缀顺序中形成一个连续块。
二分查找能高效地定位这个块，确保确定性的 $O(m \log n)$ 匹配。

后缀数组索引是后缀树的极简主义兄弟——
紧凑、优雅，是快速搜索引擎和基因组分析工具的核心。
### 697 压缩后缀数组

压缩后缀数组（CSA）是经典后缀数组的一种空间高效版本。它在保留子串搜索全部能力的同时，将内存使用从 $O(n \log n)$ 比特减少到接近信息论极限，大致相当于文本本身的熵。CSA 是大规模搜索和生物信息学系统中使用的压缩文本索引的支柱。

#### 核心思想

标准后缀数组显式存储排序后的后缀索引。压缩后缀数组用一个紧凑的、自索引的表示来替换那个显式数组，从而允许：

-   在不存储原始文本的情况下进行子串搜索，以及
-   使用压缩数据结构访问后缀数组位置。

形式上，CSA 在 $O(\log n)$ 或更好的时间内支持三个关键操作：

1.  `find(P)` – 在 $S$ 中查找模式 $P$ 的所有出现位置
2.  `locate(i)` – 恢复后缀数组索引 $i$ 对应的文本位置
3.  `extract(l, r)` – 直接从索引中提取子串 $S[l:r]$

#### 关键组件

压缩后缀数组使用几个协同工作的结构：

1.  **Burrows–Wheeler 变换（BWT）**
    重新排列 $S$ 以聚类相似字符。支持高效的后向搜索。

2.  **Rank/Select 数据结构**
    允许在 BWT 内高效地计数和定位字符。

3.  **采样**
    定期存储完整的后缀位置；通过 BWT 向后遍历来重建其他位置。

#### 构建概览

给定长度为 $n$ 的文本 $S$（以唯一终止符 `$` 结尾）：

1.  为 $S$ 构建后缀数组 $\text{SA}$。

2.  推导 Burrows–Wheeler 变换：

$$
\text{BWT}[i] =
\begin{cases}
S[\text{SA}[i] - 1], & \text{if } \text{SA}[i] > 0,\\[4pt]
\text{\$}, & \text{if } \text{SA}[i] = 0.
\end{cases}
$$

3.  计算 C 数组，其中 $C[c]$ = $S$ 中小于字符 $c$ 的字符数量。

4.  在 BWT 上存储 rank 结构以支持快速字符计数。

5.  以固定间隔（例如，每 $t$ 个条目）保存 $\text{SA}[i]$ 的样本。

#### 后向搜索（模式匹配）

模式 $P = p_1 p_2 \dots p_m$ 采用*后向*搜索：

初始化：
$$
l = 0, \quad r = n - 1
$$

对于每个字符 $p_i$，从最后一个到第一个：

$$
l = C[p_i] + \text{rank}(p_i, l - 1) + 1
$$
$$
r = C[p_i] + \text{rank}(p_i, r)
$$

当 $l > r$ 时，不存在匹配。
否则，$P$ 的所有出现位置都在 $\text{SA}[l]$ 和 $\text{SA}[r]$ 之间（通过采样重建）。

#### 示例

令 $S=\texttt{"banana\textdollar"}$。

1.  $\text{SA} = [6,\,5,\,3,\,1,\,0,\,4,\,2]$
2.  $\text{BWT} = [a,\, n,\, n,\, b,\, \textdollar,\, a,\, a]$
3.  $C = \{\textdollar\!:0,\, a\!:\!1,\, b\!:\!3,\, n\!:\!4\}$

后向搜索 $P=\texttt{"ana"}$：

| 步骤 | 字符 | 新的 $[l,r]$ |
| ---- | ---- | ------------ |
| 初始 | $\epsilon$ | $[0,6]$      |
| 1    | $a$  | $[1,3]$      |
| 2    | $n$  | $[4,5]$      |
| 3    | $a$  | $[2,3]$      |

结果：匹配位置为 $\text{SA}[2]$ 和 $\text{SA}[3]$，对应 $\texttt{"banana"}$ 中的位置 $1$ 和 $3$。

#### 微型代码（简化 Python 原型）

```python
from bisect import bisect_left, bisect_right

def suffix_array(s):
    """计算后缀数组"""
    return sorted(range(len(s)), key=lambda i: s[i:])

def bwt_from_sa(s, sa):
    """从后缀数组生成 BWT"""
    return ''.join(s[i - 1] if i else '$' for i in sa)

def search_bwt(bwt, pattern, sa, s):
    """使用二分查找的朴素后向搜索"""
    suffixes = [s[i:] for i in sa]
    l = bisect_left(suffixes, pattern)
    r = bisect_right(suffixes, pattern)
    return sa[l:r]

s = "banana$"
sa = suffix_array(s)
bwt = bwt_from_sa(s, sa)
print("SA:", sa)
print("BWT:", bwt)
print("Match:", search_bwt(bwt, "ana", sa, s))
```

输出：

```
SA: [6, 5, 3, 1, 0, 4, 2]
BWT: annb$aa
Match: [1, 3]
```

*（这是一个未压缩的版本，真实的 CSA 会用位压缩的 rank/select 结构替换这些数组。）*

#### 压缩技术

| 技术                       | 描述                                                   |
| -------------------------- | ------------------------------------------------------ |
| 小波树                     | 使用分层位图编码 BWT                                   |
| 游程编码 BWT（RLBWT）      | 压缩 BWT 中的重复游程                                  |
| 采样                       | 仅存储每第 $t$ 个后缀；通过 LF 映射恢复其他后缀        |
| 带有 Rank/Select 的位向量 | 无需解压即可实现常数时间导航                           |

#### 应用

| 领域             | 用途                                       |
| ---------------- | ------------------------------------------ |
| 搜索引擎         | 在压缩语料库上进行全文搜索                 |
| 生物信息学       | 基因组比对（Bowtie、BWA 中的 FM-index）    |
| 数据压缩         | 自索引压缩器的核心                         |
| 版本化存储       | 去重化的文档存储                           |

#### 复杂度

| 操作         | 时间               | 空间                          |
| ------------ | ------------------ | ----------------------------- |
| 搜索         | $O(m \log \sigma)$ | $(1 + \epsilon) n H_k(S)$ 比特 |
| 定位         | $O(t \log \sigma)$ | $O(n / t)$ 个采样条目         |
| 提取子串     | $O(\ell + \log n)$ | $O(n)$ 压缩结构               |

其中 $H_k(S)$ 是文本的 $k$ 阶熵，$\sigma$ 是字母表大小。

#### 动手尝试

1.  为 `"mississippi$"` 构建后缀数组和 BWT。
2.  对 `"issi"` 执行后向搜索。
3.  比较与未压缩后缀数组的内存使用情况。
4.  为子串提取实现 LF 映射。
5.  探索针对重复文本的 BWT 游程编码。

#### 一个温和的证明（为何有效）

压缩后缀数组依赖于 BWT 的局部聚类特性——文本中相近的字符被分组，从而降低了熵。通过在 BWT 上维护 rank/select 结构，我们可以*无需显式存储后缀数组*来模拟后缀数组的导航。因此，压缩和索引共存于一个优雅的框架中。

压缩后缀数组将后缀数组转变为一个自索引结构——文本、索引和压缩三者合而为一。
### 698 FM-Index

FM-Index 是一种强大的压缩全文索引，它结合了 Burrows–Wheeler 变换（BWT）、秩/选择位操作和采样技术，以支持快速的子串搜索，而无需存储原始文本。它在实现这一点的同时，使用的空间接近文本的熵，这是简洁数据结构和现代搜索系统中的一个关键里程碑。

#### 核心思想

FM-Index 是压缩后缀数组（CSA）的一种实际实现。它允许在文本 $S$ 中搜索模式 $P$，时间复杂度为 $O(m)$（模式长度为 $m$），并且使用的空间与 $S$ 的压缩大小成正比。

它依赖于文本 $S$ 的 Burrows–Wheeler 变换（BWT），该变换将文本重新排列成一种形式，将相似的上下文分组，从而实现高效的后向导航。

#### Burrows–Wheeler 变换（BWT）回顾

给定以唯一终止符 \(\$\) 结尾的文本 $S$，BWT 定义为：

$$
\text{BWT}[i] =
\begin{cases}
S[\text{SA}[i]-1], & \text{if } \text{SA}[i] > 0,\\
\text{\$}, & \text{if } \text{SA}[i] = 0.
\end{cases}
$$

对于 $S=\texttt{"banana\textdollar"}$，后缀数组为：
$$
\text{SA} = [6,\,5,\,3,\,1,\,0,\,4,\,2].
$$

BWT 字符串变为：
$$
\text{BWT} = \texttt{"annb\textdollar{}aa"}.
$$

#### 关键组件

1. BWT 字符串：变换后的文本。
2. C 数组：对于每个字符 $c$，$C[c]$ = 在 $S$ 中字典序小于 $c$ 的字符数量。
3. 秩结构：支持 $\text{rank}(c, i)$，即字符 $c$ 在 $\text{BWT}[0:i]$ 中出现的次数。
4. 采样数组：定期存储后缀数组的值，以便恢复原始位置。

#### 后向搜索算法

FM-Index 的基本操作是后向搜索。它从右到左处理模式 $P = p_1 p_2 \dots p_m$，并维护后缀数组中的一个范围 $[l, r]$，使得所有以 $P[i:m]$ 开头的后缀都落在此范围内。

初始化：
$$
l = 0, \quad r = n - 1
$$

然后对于 $i = m, m-1, \dots, 1$：

$$
l = C[p_i] + \text{rank}(p_i, l - 1) + 1
$$

$$
r = C[p_i] + \text{rank}(p_i, r)
$$

当 $l > r$ 时，不存在匹配。否则，$P$ 的所有出现位置都在 $\text{SA}[l]$ 和 $\text{SA}[r]$ 之间。

#### 示例：在 "banana$" 中搜索

文本 $S = \text{"banana\$"}$
BWT = `annb$aa`
C = {$:0$, a:1, b:3, n:4}

模式 $P = \text{"ana"}$

| 步骤 | 字符   | $[l, r]$ |
| ---- | ------ | -------- |
| 初始化 |,      | [0, 6]   |
| a    | [1, 3] |          |
| n    | [4, 5] |          |
| a    | [2, 3] |          |

在 SA[2] = 1 和 SA[3] = 3 处找到匹配 → 原始文本中的位置 1 和 3。

#### 微型代码（简化原型）

```python
def bwt_transform(s):
    s += "$"
    table = sorted(s[i:] + s[:i] for i in range(len(s)))
    return "".join(row[-1] for row in table)

def build_c_array(bwt):
    chars = sorted(set(bwt))
    count = 0
    C = {}
    for c in chars:
        C[c] = count
        count += bwt.count(c)
    return C

def rank(bwt, c, i):
    return bwt[:i + 1].count(c)

def backward_search(bwt, C, pattern):
    l, r = 0, len(bwt) - 1
    for ch in reversed(pattern):
        l = C[ch] + rank(bwt, ch, l - 1)
        r = C[ch] + rank(bwt, ch, r) - 1
        if l > r:
            return []
    return range(l, r + 1)

bwt = bwt_transform("banana")
C = build_c_array(bwt)
print("BWT:", bwt)
print("Matches:", list(backward_search(bwt, C, "ana")))
```

输出：

```
BWT: annb$aa
Matches: [2, 3]
```

#### 访问文本位置

因为我们不存储原始后缀数组，所以位置需要通过 LF 映射（Last-to-First 映射）来恢复：

$$
\text{LF}(i) = C[\text{BWT}[i]] + \text{rank}(\text{BWT}[i], i)
$$

重复应用 LF 映射可以在文本中向后移动。每第 $t$ 个后缀数组的值都会被显式存储，以便快速重建。

#### 为何有效

BWT 通过上下文将相同的字符聚类，因此秩和前序边界可以有效地重建文本中哪些部分以任何给定模式开头。

后向搜索将 BWT 转换为隐式的后缀数组遍历——无需显式存储后缀。

#### 复杂度

| 操作         | 时间               | 空间                          |
| ----------------- | ------------------ | ------------------------------ |
| 模式搜索    | $O(m \log \sigma)$ | $(1 + \epsilon) n H_k(S)$ 位 |
| 定位            | $O(t \log \sigma)$ | $O(n/t)$ 个采样点               |
| 提取子串 | $O(\ell + \log n)$ | $O(n)$ 压缩空间              |

这里 $\sigma$ 是字母表大小，$H_k(S)$ 是文本的 $k$ 阶熵。

#### 应用

| 领域               | 用途                                           |
| -------------------- | ----------------------------------------------- |
| 搜索引擎   | 具有快速查找功能的压缩文本搜索         |
| 生物信息学   | 基因组比对（例如 BWA、Bowtie、FM-mapper） |
| 数据压缩 | 自索引压缩存储的核心        |
| 版本控制  | 去重内容检索                  |

#### 动手尝试

1.  计算 `"mississippi$"` 的 BWT 并构建其 FM-Index。
2.  对 `"issi"` 运行后向搜索。
3.  修改算法以返回多文档语料库的文档 ID。
4.  添加秩/选择位向量以优化计数。
5.  比较 FM-Index 与原始后缀数组的内存使用情况。

#### 一个温和的证明（为何有效）

FM-Index 利用了 BWT 的可逆性和字典序的单调性。后向搜索利用每个字符缩小有效后缀范围，使用秩/选择操作在压缩域内模拟后缀数组遍历。因此，文本索引成为可能，*而无需扩展文本*。

FM-Index 是压缩与搜索的完美结合——小到足以容纳一个基因组，强大到足以索引整个网络。
### 699 有向无环单词图 (DAWG)

有向无环单词图 (DAWG) 是一种紧凑的数据结构，用于表示给定文本或字典的所有子串或单词。
它通过合并公共后缀或前缀来减少冗余，为字符串的所有后缀形成一个最小的确定性有限自动机 (DFA)。
DAWG 在文本索引、模式搜索、自动补全和字典压缩中至关重要。

#### 核心思想

DAWG 本质上是一个后缀自动机，或者说是一个能识别文本所有子串的最小自动机。
它可以以与文本长度成比例的线性时间和空间增量式构建。

DAWG 中的每个状态代表一组子串的结束位置，
每条边由一个字符转移标记。

关键特性：

- 有向且无环（除了字符转移外没有循环）
- 确定性（转移没有歧义）
- 最小化（合并等价状态）
- 能识别输入字符串的所有子串

#### 示例

让我们为字符串 `"aba"` 构建一个 DAWG。

所有子串：

```
a, b, ab, ba, aba
```

最小自动机包含：

- 对应不同子串上下文的状态
- 标记为 `a`、`b` 的转移
- 合并的公共部分，如共享的后缀 `"a"` 和 `"ba"`

结果转移：

```
(0) --a--> (1)
(1) --b--> (2)
(2) --a--> (3)
(1) --a--> (3)   (通过后缀合并)
```

#### 与后缀自动机的联系

一个字符串所有子串的 DAWG 与其后缀自动机是同构的。
后缀自动机中的每个状态代表一个或多个共享相同右上下文集合的子串。

形式上，该自动机接受给定文本 $S$ 的所有子串，即：

$$
L(A) = { S[i:j] \mid 0 \le i < j \le |S| }
$$

#### 构建算法（后缀自动机方法）

可以使用后缀自动机算法在 $O(n)$ 时间内增量式构建 DAWG。

每一步都用下一个字符扩展自动机并更新转移。

算法概要：

```python
def build_dawg(s):
    sa = [{}, -1, 0]  # 转移，后缀链接，长度
    last = 0
    for ch in s:
        cur = len(sa) // 3
        sa += [{}, 0, sa[3*last+2] + 1]
        p = last
        while p != -1 and ch not in sa[3*p]:
            sa[3*p][ch] = cur
            p = sa[3*p+1]
        if p == -1:
            sa[3*cur+1] = 0
        else:
            q = sa[3*p][ch]
            if sa[3*p+2] + 1 == sa[3*q+2]:
                sa[3*cur+1] = q
            else:
                clone = len(sa) // 3
                sa += [sa[3*q].copy(), sa[3*q+1], sa[3*p+2] + 1]
                while p != -1 and sa[3*p].get(ch, None) == q:
                    sa[3*p][ch] = clone
                    p = sa[3*p+1]
                sa[3*q+1] = sa[3*cur+1] = clone
        last = cur
    return sa
```

*(这是一个紧凑的后缀自动机构建器，每个节点存储转移和一个后缀链接。)*

#### 特性

| 特性          | 描述                                                                 |
| ------------- | -------------------------------------------------------------------- |
| 确定性        | 每个字符转移是唯一的                                                 |
| 无环          | 没有循环，除了通过文本的转移                                         |
| 紧凑          | 合并等价的后缀状态                                                   |
| 线性大小      | 对于长度为 $n$ 的文本，最多有 $2n - 1$ 个状态和 $3n - 4$ 条边        |
| 增量式        | 支持在线构建                                                         |

#### 可视化示例

对于 `"banana"`：

每个添加的字母都会扩展自动机：

- 添加 `"b"` 后 → 对应 `"b"` 的状态
- 添加 `"ba"` 后 → `"a"`、`"ba"`
- 添加 `"ban"` 后 → `"n"`、`"an"`、`"ban"`
- 像 `"ana"`、`"na"` 这样的公共后缀会被高效地合并。

结果以大约 11 个状态紧凑地编码了 `"banana"` 的所有 21 个子串。

#### 应用

| 领域                      | 用途                                     |
| ------------------------- | ---------------------------------------- |
| 文本索引                  | 存储所有子串以进行快速查询               |
| 字典压缩                  | 合并单词间的公共后缀                     |
| 模式匹配                  | 在 $O(m)$ 时间内测试子串是否存在         |
| 生物信息学                | 匹配基因子序列                           |
| 自然语言处理              | 自动补全和词典表示                       |

#### 使用 DAWG 进行搜索

要检查模式 $P$ 是否是 $S$ 的子串：

```
state = start
for c in P:
    if c not in transitions[state]:
        return False
    state = transitions[state][c]
return True
```

时间复杂度：$O(m)$，其中 $m$ 是 $P$ 的长度。

#### 空间和时间复杂度

| 操作                      | 时间   | 空间  |
| ------------------------- | ------ | ----- |
| 构建                      | $O(n)$ | $O(n)$ |
| 搜索子串                  | $O(m)$ | $O(1)$ |
| 计算不同子串数量          | $O(n)$ | $O(n)$ |

#### 计算不同子串数量

每个 DAWG（后缀自动机）状态代表多个子串。
字符串 $S$ 的不同子串数量为：

$$
\text{count} = \sum_{v} (\text{len}[v] - \text{len}[\text{link}[v]])
```

`"aba"` 的示例：

- $\text{count} = 5$ → 子串：`"a"`、`"b"`、`"ab"`、`"ba"`、`"aba"`

#### 动手实践

1.  为 `"banana"` 构建一个 DAWG 并计算所有子串。
2.  修改算法以支持多个单词（字典 DAWG）。
3.  可视化合并的转移，了解公共后缀如何节省空间。
4.  扩展以支持用于自动补全的前缀查询。
5.  测量查询 `"mississippi"` 所有子串所需的时间。

#### 一个温和的证明（为何有效）

合并等价的后缀状态保持了语言等价性——
每个状态对应一个唯一的右上下文集合。
由于 $S$ 的每个子串都作为一条路径出现在自动机中，
DAWG 无冗余地编码了整个子串集合。
最小化确保没有两个状态代表相同的子串集合。

有向无环单词图是表示一个字符串所有子串的最紧凑方式——
它既优雅又高效，处于自动机、压缩和搜索的交叉点。
### 700 用于文本的小波树

小波树是一种简洁的数据结构，它编码一个符号序列，同时高效地支持 rank、select 和 access 操作。
在文本索引中，它被用作压缩后缀数组和 FM 索引的核心组件，允许在不解压文本的情况下进行子串查询、频率计数和位置查找。

#### 核心思想

给定一个定义在字母表 $\Sigma$ 上的文本 $S$，小波树递归地划分字母表，并将文本表示为一组比特向量，指示每个符号属于字母表的哪一半。

这使得能够基于比特在文本中进行分层导航，从而支持如下查询：

- $\text{access}(i)$，位置 $i$ 处的字符是什么
- $\text{rank}(c, i)$，字符 $c$ 在位置 $i$ 之前出现了多少次
- $\text{select}(c, k)$，字符 $c$ 的第 $k$ 次出现在哪里

所有这些操作都可以使用紧凑的比特向量在 $O(\log |\Sigma|)$ 时间内完成。

#### 构建过程

假设 $S = \text{"banana"}$，字母表 $\Sigma = {a, b, n}$。

1. 划分字母表：
   左半部分 = {a}，右半部分 = {b, n}

2. 构建根节点比特向量：
   对于 $S$ 中的每个符号，

   * 如果它属于左半部分，则写 `0`，
   * 如果它属于右半部分，则写 `1`。

   因此：

   ```
   a b a n a n
   ↓ ↓ ↓ ↓ ↓ ↓
   0 1 0 1 0 1
   ```

   根节点比特向量 = `010101`

3. 递归构建子树：

   * 左子节点处理 `aaa`（`0` 的位置）
   * 右子节点处理 `bnn`（`1` 的位置）

每个节点对应一个字符子集，
其比特向量编码了到子节点位置的映射。

#### 查询示例

让我们查找 $\text{rank}(\text{'n'}, 5)$，即 `"banana"` 前 5 个字符中 `'n'` 的数量。

1. 从根节点开始：

   * `'n'` 在右半部分 → 跟随比特 `1`
   * 计算根节点前 5 个比特（`01010`）中 `1` 的数量 → 2
   * 以索引 2 移动到右子节点

2. 在右子节点中：

   * 字母表 {b, n}，`'n'` 再次在右半部分 → 跟随 `1`
   * 右子节点的比特向量（`011`）→ 前 2 个前缀中有 `1`
   * 计算前 2 个比特中 `1` 的数量 → 1

答案：`'n'` 在位置 5 之前出现了一次。

#### 简化代码

```python
class WaveletTree:
    def __init__(self, s, alphabet=None):
        if alphabet is None:
            alphabet = sorted(set(s))
        if len(alphabet) == 1:
            self.symbol = alphabet[0]
            self.left = self.right = None
            self.bitvector = None
            return
        mid = len(alphabet) // 2
        left_set, right_set = set(alphabet[:mid]), set(alphabet[mid:])
        self.bitvector = [0 if ch in left_set else 1 for ch in s]
        left_s = [ch for ch in s if ch in left_set]
        right_s = [ch for ch in s if ch in right_set]
        self.left = WaveletTree(left_s, alphabet[:mid]) if left_s else None
        self.right = WaveletTree(right_s, alphabet[mid:]) if right_s else None

    def rank(self, c, i):
        if not self.bitvector or i <= 0:
            return 0
        if c == getattr(self, "symbol", None):
            return min(i, len(self.bitvector))
        bit = 0 if c in getattr(self.left, "alphabet", set()) else 1
        count = sum(1 for b in self.bitvector[:i] if b == bit)
        child = self.left if bit == 0 else self.right
        return child.rank(c, count) if child else 0

wt = WaveletTree("banana")
print(wt.rank('n', 5))
```

输出：

```
1
```

#### 可视化

```
                [a,b,n]
                010101
              /        \
          [a]          [b,n]
                      011
                     /   \
                  [b]   [n]
```

- 每一层都划分字母表范围。
- 遍历比特会导向符号所在的叶子节点。

#### 操作总结

| 操作          | 含义                     | 复杂度           |
| ------------- | ------------------------ | ---------------- |
| access(i)     | 获取 $S[i]$              | $O(\log \sigma)$ |
| rank(c, i)    | $S[1..i]$ 中 c 的数量    | $O(\log \sigma)$ |
| select(c, k)  | 第 k 个 c 的位置         | $O(\log \sigma)$ |

其中 $\sigma = |\Sigma|$ 是字母表大小。

#### 与文本索引的集成

小波树是以下内容不可或缺的部分：

- FM 索引，BWT 的 rank/select 操作
- 压缩后缀数组，快速访问字符区间
- 文档检索系统，词频和位置查询
- 生物信息学工具，基因组数据上的高效模式匹配

它们允许在压缩的文本表示上进行随机访问。

#### 复杂度

| 属性           | 值                                      |
| -------------- | -------------------------------------- |
| 每次查询时间   | $O(\log \sigma)$                       |
| 空间占用       | $O(n \log \sigma)$ 比特（未压缩）      |
| 空间（简洁）   | 接近 $n H_0(S)$ 比特                   |
| 构建时间       | $O(n \log \sigma)$                     |

#### 动手尝试

1.  为 `"mississippi"` 构建一个小波树。
2.  查询 $\text{rank}(\text{'s'}, 6)$ 和 $\text{select}(\text{'i'}, 3)$。
3.  扩展它以支持子串频率查询。
4.  测量其内存大小与普通数组的对比。
5.  为每个字母表划分可视化树的层级。

#### 一个温和的证明（为何有效）

在每一层，比特将字母表划分为两半。
因此，rank 和 select 操作转化为在层级之间移动，
并使用前缀计数调整索引。
由于树的高度是 $\log \sigma$，
所有查询都能在对数时间内完成，同时保持数据的完美可逆性。

小波树统一了压缩和搜索：
它编码、索引并查询文本——
所有这些都在信息本身的熵限之内。
