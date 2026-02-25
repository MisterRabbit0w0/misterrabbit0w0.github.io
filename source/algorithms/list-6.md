---
title: 第六章
date: 2026-02-25
---

# 第六章 算法中的数学

# 第五十一节 数论
### 501 欧几里得算法

欧几里得算法是数学中最古老、最优雅的算法之一。它用于寻找两个整数的最大公约数（gcd），即能同时整除这两个数且不留余数的最大数字。

它快速、简单，并且构成了现代数论和密码学的基础。

#### 我们要解决什么问题？

我们想要计算：

$$
\text{gcd}(a, b)
$$

也就是说，最大的整数$d$，使得$d \mid a$ 且$d \mid b$。

与其检查每一个可能的除数，欧几里得发现了一个巧妙的捷径：

$$
\gcd(a, b) = \gcd(b, a \bmod b)
$$

不断地用余数替换较大的数，直到其中一个变为零。最后一个非零数就是最大公约数。

示例：

$$
\gcd(48, 18) = \gcd(18, 48 \bmod 18) = \gcd(18, 12) = \gcd(12, 6) = \gcd(6, 0) = 6
$$

所以 $gcd(48, 18) = 6$。

#### 它是如何工作的（通俗解释）？

可以把求最大公约数想象成剥离余数的层层外衣。
每一步都移除一部分，直到什么都不剩。那个在所有余数运算后“幸存”下来的数就是最大公约数。

让我们一步步来看：

| 步骤 | a  | b  | a mod b |
| ---- | -- | -- | ------- |
| 1    | 48 | 18 | 12      |
| 2    | 18 | 12 | 6       |
| 3    | 12 | 6  | 0       |

当余数变为 0 时，停止。另一个数（6）就是最大公约数。

每一步都快速地减小了数字，所以它的时间复杂度是 O(log min(a, b))，比尝试所有除数要快得多。

#### 简洁代码（简易版本）

C 语言版本

```c
#include <stdio.h>

int gcd(int a, int b) {
    while (b != 0) {
        int r = a % b;
        a = b;
        b = r;
    }
    return a;
}

int main(void) {
    int a, b;
    printf("输入 a 和 b: ");
    scanf("%d %d", &a, &b);
    printf("gcd(%d, %d) = %d\n", a, b, gcd(a, b));
}
```

Python 版本

```python
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

a, b = map(int, input("输入 a 和 b: ").split())
print("gcd(", a, ",", b, ") =", gcd(a, b))
```

#### 为什么它很重要

- 是模运算、数论和密码学的核心。
- 扩展欧几里得算法（用于求解 ax + by = gcd）的基础。
- 用于模逆元和孙子定理（中国剩余定理）。
- 展示了算法思维：将问题分解为更小的余数问题。

#### 一个温和的证明（为什么它有效）

如果$a = bq + r$，那么任何能整除$a$ 和$b$ 的数也必须能整除$r$。
因此，$(a, b)$ 的公因数集合等于$(b, r)$ 的公因数集合。
所以：

$$
\gcd(a, b) = \gcd(b, r)
$$

反复应用这个等式，直到$r = 0$，就揭示出了最大公约数。

因为余数每一步都在缩小，所以算法最多在$O(\log n)$ 步内停止。

#### 自己动手试试

1.  手动计算 gcd(84, 30)。
2.  追踪 gcd(210, 45) 的步骤。
3.  修改代码以计算步骤数。
4.  尝试用大数字（例如 123456, 789012）。
5.  与朴素的除数检查方法比较运行时间。

#### 测试用例

| a   | b   | 步骤数 | gcd |
| --- | --- | ------ | --- |
| 48  | 18  | 3      | 6   |
| 84  | 30  | 3      | 6   |
| 210 | 45  | 3      | 15  |
| 101 | 10  | 2      | 1   |
| 270 | 192 | 5      | 6   |

#### 复杂度

-   时间复杂度：O(log min(a, b))
-   空间复杂度：O(1)（迭代）或 O(log n)（递归）

欧几里得算法展示了简洁的力量：分割、化简、重复，这是一个永恒的思想，至今仍在现代计算中跳动。
### 502 扩展欧几里得算法

扩展欧几里得算法比经典的求最大公约数（gcd）算法更进一步：它不仅找出 gcd(a, b)，还给出满足下式的 x 和 y：

$$
a \cdot x + b \cdot y = \gcd(a, b)
$$

这些系数 (x, y) 是解决丢番图方程和求模逆元的关键，在密码学、模运算和算法设计中至关重要。

#### 我们要解决什么问题？

我们想求解线性方程：

$$
a x + b y = \gcd(a, b)
$$

给定整数 a 和 b，我们需要满足此方程的整数 x 和 y。
欧几里得算法给出了最大公约数，但我们可以回溯其步骤，将最大公约数表示为 a 和 b 的组合。

示例：
找出满足下式的 x, y：
$$
240x + 46y = \gcd(240, 46)
$$

我们知道 gcd(240, 46) = 2。
扩展欧几里得算法给出：
$$
2 = 240(-9) + 46(47)
$$
所以 (x = -9, y = 47)。

#### 它是如何工作的（通俗解释）？

可以把它看作是“记住”在求最大公约数的过程中每个余数是如何产生的。

在每一步：
$$
a = bq + r \quad \Rightarrow \quad r = a - bq
$$

我们递归地计算 gcd(b, r)，并在回溯时，将每个余数用 a 和 b 的形式重写。

以 (240, 46) 为例逐步说明：

| 步骤 | a   | b  | a % b | 方程           |
| ---- | --- | -- | ----- | -------------- |
| 1    | 240 | 46 | 10    | 240 = 46×5 + 10 |
| 2    | 46  | 10 | 6     | 46 = 10×4 + 6   |
| 3    | 10  | 6  | 4     | 10 = 6×1 + 4    |
| 4    | 6   | 4  | 2     | 6 = 4×1 + 2     |
| 5    | 4   | 2  | 0     | 停止，gcd = 2   |

现在回溯：

- 2 = 6 − 4×1
- 4 = 10 − 6×1
- 6 = 46 − 10×4
- 10 = 240 − 46×5

向上代入，直到 gcd 表示为 240x + 46y 的形式。

#### 简洁代码（简易版本）

C 语言版本

```c
#include <stdio.h>

int extended_gcd(int a, int b, int *x, int *y) {
    if (b == 0) {
        *x = 1;
        *y = 0;
        return a;
    }
    int x1, y1;
    int g = extended_gcd(b, a % b, &x1, &y1);
    *x = y1;
    *y = x1 - (a / b) * y1;
    return g;
}

int main(void) {
    int a, b, x, y;
    printf("输入 a 和 b: ");
    scanf("%d %d", &a, &b);
    int g = extended_gcd(a, b, &x, &y);
    printf("gcd(%d, %d) = %d\n", a, b, g);
    printf("系数: x = %d, y = %d\n", x, y);
}
```

Python 版本

```python
def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return g, x, y

a, b = map(int, input("输入 a 和 b: ").split())
g, x, y = extended_gcd(a, b)
print(f"gcd({a}, {b}) = {g}")
print(f"x = {x}, y = {y}")
```

#### 为什么它很重要

- 构建模逆元：如果 gcd(a, m) = 1，则 $a^{-1} \equiv x \pmod{m}$
- 求解线性丢番图方程
- 用于 RSA 密码学、中国剩余定理和模运算
- 将最大公约数转化为输入的线性组合

#### 一个温和的证明（为什么它有效）

在每一步：
$$
\gcd(a, b) = \gcd(b, a \bmod b)
$$
如果我们知道 $b x' + (a \bmod b) y' = \gcd(a, b)$，
并且由于 $a \bmod b = a - b\lfloor a/b \rfloor$，
我们可以写成：
$$
\gcd(a, b) = a y' + b(x' - \lfloor a/b \rfloor y')
$$

令 $x = y'$, $y = x' - \lfloor a/b \rfloor y'$。

这就给出了系数 (x, y) 的递推关系。

#### 亲自尝试

1.  手动求解 (240x + 46y = 2)。
2.  验证 gcd(99, 78) = 3 并找出 x, y。
3.  使用扩展欧几里得算法求 3 模 11 的模逆元。
4.  修改代码使其只返回模逆元。
5.  追踪 (99, 78) 的递归调用。

#### 测试用例

| a   | b  | gcd | x   | y  | 验证               |
| --- | -- | --- | --- | -- | ------------------ |
| 240 | 46 | 2   | -9  | 47 | 240(-9)+46(47)=2 |
| 99  | 78 | 3   | -11 | 14 | 99(-11)+78(14)=3 |
| 35  | 15 | 5   | 1   | -2 | 35(1)+15(-2)=5   |
| 7   | 5  | 1   | -2  | 3  | 7(-2)+5(3)=1     |

#### 复杂度

- 时间复杂度：O(log min(a, b))
- 空间复杂度：O(log n)（递归深度）

扩展欧几里得算法是最大公约数的“记忆”，它不仅找出除数，还展示了它是如何构成的。
### 503 模加法

模加法就像将数字环绕在一个圆圈上。一旦到达终点，就会循环回到起点。这个操作是模运算的核心，是时钟运算、密码学和哈希运算的基础。

当我们以模 M 进行加法运算时，结果始终保持在范围 [0, M−1] 内。

#### 我们要解决什么问题？

我们想要计算两个数在模 M 下的和：

$$
(a + b) \bmod M
$$

示例：
令$a = 17, b = 12, M = 10$

$$
(17 + 12) \bmod 10 = 29 \bmod 10 = 9
$$

所以每超过 10 我们就"环绕"一次。

#### 它是如何工作的（通俗解释）？

想象一个有 M 个小时的时钟。如果你向前移动 a 小时，然后再移动 b 小时，你会落在哪里？

你只需要将 a 和 b 相加，然后取除以 M 的余数。

| a  | b  | M  | a + b | (a + b) mod M | 结果 |
| -- | -- | -- | ----- | ------------- | ---- |
| 17 | 12 | 10 | 29    | 9             | 9    |
| 8  | 7  | 5  | 15    | 0             | 0    |
| 25 | 25 | 7  | 50    | 1             | 1    |

这就是在超过 M 的倍数时进行"环绕"的加法。

#### 微型代码（简易版本）

C 语言版本

```c
#include <stdio.h>

int mod_add(int a, int b, int M) {
    int res = (a % M + b % M) % M;
    if (res < 0) res += M; // 处理负值
    return res;
}

int main(void) {
    int a, b, M;
    printf("输入 a, b, M: ");
    scanf("%d %d %d", &a, &b, &M);
    printf("(a + b) mod M = %d\n", mod_add(a, b, M));
}
```

Python 版本

```python
def mod_add(a, b, M):
    return (a % M + b % M) % M

a, b, M = map(int, input("输入 a, b, M: ").split())
print("(a + b) mod M =", mod_add(a, b, M))
```

#### 为什么它很重要

- 构成了模运算中的基础操作
- 用于密码学算法（RSA、Diffie-Hellman）
- 确保在固定大小的计算中不会溢出
- 支持哈希运算、环运算和校验和计算

#### 一个温和的证明（为什么它成立）

令$a = Mq_1 + r_1$ 且$b = Mq_2 + r_2$。
那么：
$$
a + b = M(q_1 + q_2) + (r_1 + r_2)
$$
因此，
$$
(a + b) \bmod M = (r_1 + r_2) \bmod M = (a \bmod M + b \bmod M) \bmod M
$$
这就是模加法的性质。

#### 动手试试

1.  计算 (25 + 37) mod 12。
2.  计算 (−3 + 5) mod 7。
3.  修改代码以处理 3 个数：(a + b + c) mod M。
4.  编写模减法：(a − b) mod M。
5.  探索 M = 5 时的模式表。

#### 测试用例

| a  | b  | M  | (a + b) mod M | 结果 |
| -- | -- | -- | ------------- | ---- |
| 17 | 12 | 10 | 9             | 正确 |
| 25 | 25 | 7  | 1             | 正确 |
| -3 | 5  | 7  | 2             | 正确 |
| 8  | 8  | 5  | 1             | 正确 |
| 0  | 9  | 4  | 1             | 正确 |

#### 复杂度

-   时间复杂度：O(1)
-   空间复杂度：O(1)

模加法是在一个循环上的算术运算，每个和都会折返到一个有限的世界中，使数字保持整洁和优雅。
### 504 模乘法

模乘法是在一个“环绕”世界中的算术，就像模加法一样，但它是重复的加法。当你在模 M 下将两个数相乘时，你只保留余数，确保结果保持在 [0, M−1] 的范围内。

它是快速幂、哈希、密码学和数论变换的基石。

#### 我们要解决什么问题？

我们想要计算：

$$
(a \times b) \bmod M
$$

示例：
令 $a = 7, b = 8, M = 5$

$$
(7 \times 8) \bmod 5 = 56 \bmod 5 = 1
$$

所以 7×8 在大小为 5 的“圆环”上“环绕”并落在 1 的位置。

#### 它是如何工作的（通俗解释）？

将模乘法视为重复的模加法：

$$
(a \times b) \bmod M = (a \bmod M + a \bmod M + \dots) \bmod M
$$
（重复 b 次）

我们可以用这个恒等式来简化它：
$$
(a \times b) \bmod M = ((a \bmod M) \times (b \bmod M)) \bmod M
$$

| a  | b  | M  | a×b | (a×b) mod M | 结果 |
| -- | -- | -- | --- | ----------- | ---- |
| 7  | 8  | 5  | 56  | 1           | 1    |
| 25 | 4  | 7  | 100 | 2           | 2    |
| 12 | 13 | 10 | 156 | 6           | 6    |

对于大数值，我们通过在每个步骤应用模约简来避免溢出。

#### 微型代码（简易版本）

C 语言版本

```c
#include <stdio.h>

long long mod_mul(long long a, long long b, long long M) {
    a %= M;
    b %= M;
    long long res = (a * b) % M;
    if (res < 0) res += M; // 处理负数
    return res;
}

int main(void) {
    long long a, b, M;
    printf("输入 a, b, M: ");
    scanf("%lld %lld %lld", &a, &b, &M);
    printf("(a * b) mod M = %lld\n", mod_mul(a, b, M));
}
```

Python 版本

```python
def mod_mul(a, b, M):
    return (a % M * b % M) % M

a, b, M = map(int, input("输入 a, b, M: ").split())
print("(a * b) mod M =", mod_mul(a, b, M))
```

对于非常大的数，使用通过加法实现的模乘法（以避免溢出）：

```python
def mod_mul_safe(a, b, M):
    res = 0
    a %= M
    while b > 0:
        if b % 2 == 1:
            res = (res + a) % M
        a = (2 * a) % M
        b //= 2
    return res
```

#### 为什么它很重要

- 在模幂运算和密码学（RSA、Diffie-Hellman）中必不可少
- 避免模运算下的算术溢出
- 是数论变换（NTT）和哈希函数的核心操作
- 支持构建模逆元、幂函数和多项式算术

#### 一个温和的证明（为什么它成立）

令 $a = q_1 M + r_1$，$b = q_2 M + r_2$。

那么：
$$
a \times b = M(q_1 b + q_2 a - q_1 q_2 M) + r_1 r_2
$$
所以对 M 取模时，所有 M 的倍数都消失：
$$
(a \times b) \bmod M = (r_1 \times r_2) \bmod M
$$
因此，
$$
(a \times b) \bmod M = ((a \bmod M) \times (b \bmod M)) \bmod M
$$

#### 自己动手试试

1.  计算 (25 × 13) mod 7。
2.  尝试 (−3 × 8) mod 5。
3.  修改代码以处理负输入。
4.  使用重复加倍（类似于二进制幂运算）编写模乘法。
5.  为 a, b ∈ [0,5] 创建 (a×b) mod 6 的表格。

#### 测试用例

| a          | b          | M          | (a×b) mod M | 结果 |
| ---------- | ---------- | ---------- | ----------- | ---- |
| 7          | 8          | 5          | 1           | 正确 |
| 25         | 4          | 7          | 2           | 正确 |
| -3         | 8          | 5          | 1           | 正确 |
| 12         | 13         | 10         | 6           | 正确 |
| 1000000000 | 1000000000 | 1000000007 | 49          | 正确 |

#### 复杂度

-   时间复杂度：O(1)（使用内置乘法），O(log b)（使用安全的加倍法）
-   空间复杂度：O(1)

模乘法使算术保持稳定和可预测，无论数字有多大，一切都能整齐地折叠回模环中。
### 505 模幂运算

模幂运算是一种在模数下高效计算一个数的幂的技术。  
我们不是将 (a) 乘以自身 (b) 次（那样太慢），而是一步步进行平方和约减。  
这项技术支撑着密码学、哈希、数论以及处理大指数的高效算法。

#### 我们要解决什么问题？

我们想要计算：

$$
a^b \bmod M
$$

直接计算，不溢出，也不循环 (b) 次。

示例：  
设 $a = 3, b = 13, M = 7$

朴素方法：  
$$
3^{13} = 1594323 \quad \Rightarrow \quad 1594323 \bmod 7 = 5
$$

高效方法，我们可以用平方来计算：  
$$
3^{13} \bmod 7 = ((3^8 \cdot 3^4 \cdot 3^1) \bmod 7) = 5
$$

#### 它是如何工作的（通俗解释）？

我们使用平方求幂法。  
将指数分解为二进制。对于每一位，要么平方，要么平方并乘。

如果 (b) 是偶数：  
$$
a^b = (a^{b/2})^2
$$
如果 (b) 是奇数：  
$$
a^b = a \cdot a^{b-1}
$$

每次乘法后都取模，以保持数字较小。

示例：$(a=3,\ b=13,\ M=7)$

| 步骤 | b (二进制) | a                                 | b  | 结果  |
| ---- | ---------- | --------------------------------- | -- | ------- |
| 1    | 1101       | 3                                 | 13 | res = 1 |
| 2    | 奇数       | res = $(1\times3)\bmod7=3$, $a=(3\times3)\bmod7=2$, $b=6$ |    |         |
| 3    | 偶数       | res = 3, $a=(2\times2)\bmod7=4$, $b=3$ |    |         |
| 4    | 奇数       | res = $(3\times4)\bmod7=5$, $a=(4\times4)\bmod7=2$, $b=1$ |    |         |
| 5    | 奇数       | res = $(5\times2)\bmod7=3$, $a=(2\times2)\bmod7=4$, $b=0$ |    |         |

结果 = 3，即 $3^{13} \bmod 7$。

#### 精简代码（简易版本）

C 版本

```c
#include <stdio.h>

long long mod_pow(long long a, long long b, long long M) {
    long long res = 1;
    a %= M;
    while (b > 0) {
        if (b % 2 == 1)
            res = (res * a) % M;
        a = (a * a) % M;
        b /= 2;
    }
    return res;
}

int main(void) {
    long long a, b, M;
    printf("输入 a, b, M: ");
    scanf("%lld %lld %lld", &a, &b, &M);
    printf("%lld^%lld mod %lld = %lld\n", a, b, M, mod_pow(a, b, M));
}
```

Python 版本

```python
def mod_pow(a, b, M):
    res = 1
    a %= M
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % M
        a = (a * a) % M
        b //= 2
    return res

a, b, M = map(int, input("输入 a, b, M: ").split())
print(f"{a}^{b} mod {M} =", mod_pow(a, b, M))
```

或者更简单地：

```python
pow(a, b, M)
```

（Python 内置的 pow 函数能高效处理此运算。）

#### 为什么它很重要

- RSA、Diffie-Hellman 和 ElGamal 加密算法的核心操作
- 费马小定理和模逆元计算所需
- 支持无溢出的快速幂运算
- 将幂运算的时间复杂度从 O(b) 降至 O(log b)

#### 一个温和的证明（为什么它有效）

任何指数都可以写成二进制形式：  
$$
b = \sum_{i=0}^{k} b_i \cdot 2^i
$$

那么：  
$$
a^b = \prod_{i=0}^{k} (a^{2^i})^{b_i}
$$

我们通过平方来预计算 $a^{2^i}$，并且只在 (b_i = 1) 的地方相乘。  
每次平方和乘法之后都进行模约减，因此数字保持较小。

#### 亲自尝试

1.  计算 $2^{10} \bmod 1000$。
2.  计算 $5^{117} \bmod 19$。
3.  修改代码以打印每一步（跟踪幂运算过程）。
4.  与朴素幂运算比较运行时间。
5.  使用 pow(a, b, M) 并验证结果相同。

#### 测试用例

| a  | b   | M    | 结果 | 检查 |
| -- | --- | ---- | ------ | ----- |
| 3  | 13  | 7    | 5      | 正确     |
| 2  | 10  | 1000 | 24     | 正确     |
| 5  | 117 | 19   | 1      | 正确     |
| 10 | 9   | 6    | 4      | 正确     |
| 7  | 222 | 13   | 9      | 正确     |

#### 复杂度

-   时间复杂度：O(log b)
-   空间复杂度：O(1)（迭代）或 O(log b)（递归）

模幂运算让我们得以驾驭巨大的幂，将指数增长转化为模数下快速、对数的舞蹈。
### 506 模逆元

模逆元是在模运算下能“撤销”乘法的数。
如果 $a \cdot x \equiv 1 \pmod{M}$，那么 (x) 被称为 (a) 模 (M) 的模逆元。

它是模运算中进行除法的关键，因为除法没有被直接定义，我们改为乘以一个逆元。

#### 我们要解决什么问题？

我们想求解：

$$
a \cdot x \equiv 1 \pmod{M}
$$

这意味着找到 $x$ 使得：

$$
(a \times x) \bmod M = 1
$$

这个 $x$ 被称为 $a$ 模 $M$ 的模乘逆元。

示例：

求 $3 \pmod{11}$ 的逆元。

我们需要：

$$
3 \cdot x \equiv 1 \pmod{11}
$$

尝试 $x = 4$：

$$
3 \cdot 4 = 12 \equiv 1 \pmod{11}
$$

因此：

$$
3^{-1} \equiv 4 \pmod{11}
$$


#### 它是如何工作的（通俗解释）？

求模逆元主要有两种方法：

1.  扩展欧几里得算法（当 gcd(a, M) = 1 时对所有情况有效）
2.  费马小定理（当 M 是质数时有效）

##### 1. 扩展欧几里得方法

我们求解：

$$
a x + M y = 1
$$

系数 $x \bmod M$ 就是模逆元。

示例：

求 $3 \bmod 11$ 的逆元。

使用扩展欧几里得算法：

$$
\begin{aligned}
11 &= 3 \times 3 + 2 \\
3  &= 2 \times 1 + 1 \\
2  &= 1 \times 2 + 0
\end{aligned}
$$

回代：

$$
\begin{aligned}
1 &= 3 - 2 \times 1 \\
  &= 3 - (11 - 3 \times 3) \\
  &= 4 \times 3 - 1 \times 11
\end{aligned}
$$

所以 $x = 4$。

因此，

$$
3^{-1} \equiv 4 \pmod{11}
$$


##### 2. 费马小定理（M 为质数）

如果 $M$ 是质数且 $a \not\equiv 0 \pmod{M}$，那么：

$$
a^{M-1} \equiv 1 \pmod{M}
$$

两边同时乘以 $a^{-1}$：

$$
a^{M-2} \equiv a^{-1} \pmod{M}
$$

所以模逆元是：

$$
a^{-1} = a^{M-2} \bmod M
$$

示例：

$$
3^{-1} \pmod{11} = 3^{9} \bmod 11 = 4
$$


#### 简短代码（简易版本）

C 语言版本（扩展 GCD）

```c
#include <stdio.h>

long long extended_gcd(long long a, long long b, long long *x, long long *y) {
    if (b == 0) {
        *x = 1;
        *y = 0;
        return a;
    }
    long long x1, y1;
    long long g = extended_gcd(b, a % b, &x1, &y1);
    *x = y1;
    *y = x1 - (a / b) * y1;
    return g;
}

long long mod_inverse(long long a, long long M) {
    long long x, y;
    long long g = extended_gcd(a, M, &x, &y);
    if (g != 1) return -1; // 如果 gcd ≠ 1，则逆元不存在
    x = (x % M + M) % M;
    return x;
}

int main(void) {
    long long a, M;
    printf("输入 a, M: ");
    scanf("%lld %lld", &a, &M);
    long long inv = mod_inverse(a, M);
    if (inv == -1)
        printf("逆元不存在。\n");
    else
        printf("%lld 模 %lld 的逆元 = %lld\n", a, M, inv);
}
```

Python 版本

```python
def mod_inverse(a, M):
    def extended_gcd(a, b):
        if b == 0:
            return a, 1, 0
        g, x1, y1 = extended_gcd(b, a % b)
        x = y1
        y = x1 - (a // b) * y1
        return g, x, y

    g, x, y = extended_gcd(a, M)
    if g != 1:
        return None
    return x % M

a, M = map(int, input("输入 a, M: ").split())
inv = mod_inverse(a, M)
print("逆元:", inv if inv is not None else "None")
```

Python 版本（使用费马小定理，模数为质数）

```python
def mod_inverse_prime(a, M):
    return pow(a, M - 2, M)
```

#### 为什么它很重要

-   使得模运算中的除法成为可能
-   在 RSA、中国剩余定理（CRT）、椭圆曲线和哈希算法中至关重要
-   用于求解如 $a x \equiv b \pmod{M}$ 的方程
-   是求解线性同余式和模方程组的基础

#### 一个温和的证明（为什么它有效）

如果 $\gcd(a, M) = 1$，
根据裴蜀定理：
$$
a x + M y = 1
$$
对 M 取模：
$$
a x \equiv 1 \pmod{M}
$$
因此，(x) 是 (a) 的模逆元。

#### 亲自尝试

1.  求 $5^{-1} \pmod{7}$。
2.  求 $10^{-1} \pmod{17}$。
3.  检查哪些数模 $8$ 没有逆元。
4.  实现扩展 GCD 和费马小定理两种版本。
5.  使用逆元求解 $7x \equiv 3 \pmod{13}$。

#### 测试用例

| a  | M  | 逆元    | 验证                              |
| -- | -- | ------- | --------------------------------- |
| 3  | 11 | 4       | $3 \times 4 = 12 \equiv 1$        |
| 5  | 7  | 3       | $5 \times 3 = 15 \equiv 1$        |
| 10 | 17 | 12      | $10 \times 12 = 120 \equiv 1$     |
| 2  | 4  | 无      | $\gcd(2,4) \ne 1$                 |
| 7  | 13 | 2       | $7 \times 2 = 14 \equiv 1$        |


#### 复杂度

-   扩展 GCD：O(log M)
-   费马小定理（M 为质数）：O(log M)（通过模幂运算）
-   空间复杂度：O(1)（迭代）

模逆元是打开模世界除法之门的钥匙，在这个世界里，每个有效的数都有其对应的“镜像”乘数，能将你带回到 1。
### 507 中国剩余定理

中国剩余定理（CRT）是同余式之间一座美丽的桥梁。
它让你能够求解模方程组，将多个模世界合并成一个一致的解。
它最初在两千多年前被描述，至今仍是现代数论和密码学的基石。

#### 我们要解决什么问题？

我们想要一个满足以下同余方程组的整数 $x$：

$$
\begin{cases}
x \equiv a_1 \pmod{m_1} \\
x \equiv a_2 \pmod{m_2} \\
\vdots \\
x \equiv a_k \pmod{m_k}
\end{cases}
$$

如果模数 $m_1,m_2,\dots,m_k$ 两两互质，则存在一个模
$$
M = m_1 m_2 \cdots m_k
$$
下的唯一解。

#### 示例

寻找满足以下条件的 $x$：
$$
\begin{cases}
x \equiv 2 \pmod{3} \\
x \equiv 3 \pmod{4} \\
x \equiv 2 \pmod{5}
\end{cases}
$$

计算
$$
M = 3 \cdot 4 \cdot 5 = 60,\quad
M_1 = \frac{M}{3}=20,\quad
M_2 = \frac{M}{4}=15,\quad
M_3 = \frac{M}{5}=12.
$$

寻找模逆元
$$
20^{-1} \pmod{3} = 2,\quad
15^{-1} \pmod{4} = 3,\quad
12^{-1} \pmod{5} = 3.
$$

合并
$$
x = 2\cdot 20\cdot 2 \;+\; 3\cdot 15\cdot 3 \;+\; 2\cdot 12\cdot 3
  = 80 + 135 + 72 = 287.
$$

模 $60$ 约简
$$
x \equiv 287 \bmod 60 = 47.
$$

所以解是
$$
x \equiv 47 \pmod{60}.
$$

#### 它是如何工作的（通俗解释）？

每个同余式给出一个每 $m_i$ 重复一次的"车道"。

CRT 找到交叉点，即所有车道对齐的最小 $x$。

可以把模世界想象成具有不同滴答长度的时钟。
CRT 找到所有时钟同时显示指定指针的时间。

| 模数 | 余数 | 周期 | 对齐点        |
| ------- | --------- | ------ | --------------- |
| 3       | 2         | 3      | 2, 5, 8, 11, …  |
| 4       | 3         | 4      | 3, 7, 11, 15, … |
| 5       | 2         | 5      | 2, 7, 12, 17, … |

它们首次在 47 处对齐，然后每 60 重复一次。

#### 精简代码（简易版本）

Python 版本

```python
def extended_gcd(a, b):
    """扩展欧几里得算法"""
    if b == 0:
        return a, 1, 0
    g, x1, y1 = extended_gcd(b, a % b)
    return g, y1, x1 - (a // b) * y1

def mod_inverse(a, m):
    """计算模逆元"""
    g, x, y = extended_gcd(a, m)
    if g != 1:
        return None
    return x % m

def crt(a, m):
    """中国剩余定理求解"""
    M = 1
    for mod in m:
        M *= mod
    x = 0
    for ai, mi in zip(a, m):
        Mi = M // mi
        inv = mod_inverse(Mi, mi)
        x = (x + ai * Mi * inv) % M
    return x

a = [2, 3, 2]
m = [3, 4, 5]
print("x =", crt(a, m))  # 输出: 47
```

C 版本（简化版）

```c
#include <stdio.h>

long long extended_gcd(long long a, long long b, long long *x, long long *y) {
    if (b == 0) { *x = 1; *y = 0; return a; }
    long long x1, y1;
    long long g = extended_gcd(b, a % b, &x1, &y1);
    *x = y1;
    *y = x1 - (a / b) * y1;
    return g;
}

long long mod_inverse(long long a, long long m) {
    long long x, y;
    long long g = extended_gcd(a, m, &x, &y);
    if (g != 1) return -1;
    return (x % m + m) % m;
}

long long crt(int a[], int m[], int n) {
    long long M = 1, x = 0;
    for (int i = 0; i < n; i++) M *= m[i];
    for (int i = 0; i < n; i++) {
        long long Mi = M / m[i];
        long long inv = mod_inverse(Mi, m[i]);
        x = (x + (long long)a[i] * Mi * inv) % M;
    }
    return x;
}

int main(void) {
    int a[] = {2, 3, 2};
    int m[] = {3, 4, 5};
    int n = 3;
    printf("x = %lld\n", crt(a, m, n)); // 输出: 47
}
```

#### 为什么它很重要

- 将多个模系统组合成一个统一的解
- 在 RSA 中至关重要（CRT 优化）
- 用于农历、哈希、多项式模、快速傅里叶变换
- 是大整数运算中多模算术的基础

#### 一个温和的证明（为什么它有效）

如果模数 $m_i$ 两两互质，那么 $M_i = M/m_i$ 与 $m_i$ 互质。
因此每个 $M_i$ 都有一个模逆 $n_i$，使得
$$
M_i \cdot n_i \equiv 1 \pmod{m_i}.
$$

组合
$$
x = \sum_{i=1}^{k} a_i \, M_i \, n_i
$$
对所有 $i$ 都满足 $x \equiv a_i \pmod{m_i}$。
将 $x$ 对 $M$ 取模得到最小的非负解。

#### 自己动手试试

1.  求解
    $x \equiv 1 \pmod{2}$,
    $x \equiv 2 \pmod{3}$,
    $x \equiv 3 \pmod{5}$.
2.  将一个模数改为非互质（例如 $4, 6$），观察会发生什么。
3.  使用 Garner 算法实现非互质模数的 CRT。
4.  测试大素数（使用 Python 大整数）。
5.  使用 CRT 从模 $10^{9}+7$ 和 $998244353$ 的余数重构 $x$。

#### 测试用例

| 方程组                      | 解   | 模数 | 检查 |
| --------------------------- | -------- | ------- | ----- |
| (2 mod 3, 3 mod 4, 2 mod 5) | 47       | 60      | 通过     |
| (1 mod 2, 2 mod 3, 3 mod 5) | 23       | 30      | 通过     |
| (3 mod 5, 1 mod 7)          | 31       | 35      | 通过     |
| (0 mod 3, 1 mod 4)          | 4        | 12      | 通过     |

#### 复杂度

- 时间：O(k log M)（每一步使用扩展欧几里得算法）
- 空间：O(k)

CRT 是模世界的和谐乐章，它将许多同余式统一成一个优雅的答案，既回响着古老的算术，也呼应着现代的加密技术。
### 508 二进制最大公约数（Stein 算法）

二进制最大公约数算法，也称为 Stein 算法，使用位运算而非除法来计算最大公约数。
它通常比经典的欧几里得算法更快，尤其是在二进制硬件上，非常适合底层、对性能敏感的代码。

#### 我们要解决什么问题？

我们希望计算

$$
\gcd(a,b)
$$

而不使用除法，仅使用移位、减法和奇偶性检查。这就是二进制最大公约数（Stein）算法。

算法

1. 如果 $a=0$，返回 $b$。如果 $b=0$，返回 $a$。
2. 令 $k=\min(v_2(a),\,v_2(b))$，其中 $v_2(x)$ 是 $x$ 的尾部零比特的数量。
3. 设置 $a \gets a/2^{v_2(a)}$ 和 $b \gets b/2^{v_2(b)}$（两者都变为奇数）。
4. 当 $a \ne b$ 时：
   - 如果 $a>b$，设置 $a \gets a-b$；然后移除因子 2：$a \gets a/2^{v_2(a)}$。
   - 否则设置 $b \gets b-a$；然后 $b \gets b/2^{v_2(b)}$。
5. 返回 $a \cdot 2^{k}$。

示例

求 $\gcd(48,18)$。

- 尾部零：$v_2(48)=4$，$v_2(18)=1$，所以 $k=\min(4,1)=1$。
- 变为奇数：
  - $a \gets 48/2^{4}=3$
  - $b \gets 18/2^{1}=9$
- 循环：
  - $b \gets 9-3=6 \Rightarrow b \gets 6/2^{1}=3$
  - 现在 $a=b=3$
- 答案：$3 \cdot 2^{1}=6$

所以 $\gcd(48,18)=6$。

注意

- 仅使用减法和比特移位。
- 复杂度为 $O(\log(\max(a,b)))$，操作非常简单。

#### 它是如何工作的（通俗解释）？

Stein 的洞见：

- 如果两个数都是偶数 → $\gcd(a,b) = 2 \times \gcd(a/2,\, b/2)$
- 如果一个是偶数 → 将其除以 2
- 如果两个都是奇数 → 用（较大者 − 较小者）替换较大者
- 重复直到它们相等

| 步骤 | a  | b  | 操作                     | 备注                          |
| ---- | -- | -- | ----------------------- | ----------------------------- |
| 1    | 48 | 18 | 都是偶数 → 除以 2        | $\gcd = 2 \times \gcd(24,9)$  |
| 2    | 24 | 9  | 一个是偶数 → 将 a 除以 2 | $\gcd = 2 \times \gcd(12,9)$  |
| 3    | 12 | 9  | 一个是偶数 → 将 a 除以 2 | $\gcd = 2 \times \gcd(6,9)$   |
| 4    | 6  | 9  | 一个是偶数 → 将 a 除以 2 | $\gcd = 2 \times \gcd(3,9)$   |
| 5    | 3  | 9  | 都是奇数 → $b-a=6$       | $\gcd = 2 \times \gcd(3,6)$   |
| 6    | 3  | 6  | 一个是偶数 → 将 b 除以 2 | $\gcd = 2 \times \gcd(3,3)$   |
| 7    | 3  | 3  | 相等 → 返回 3            | $\gcd = 2 \times 3 = 6$       |

最终结果：$\gcd(48,18)=6$。

#### 简洁代码（简易版本）

C 语言版本

```c
#include <stdio.h>

int gcd_binary(int a, int b) {
    if (a == 0) return b;
    if (b == 0) return a;

    // 找出 2 的幂因子
    int shift = 0;
    while (((a | b) & 1) == 0) {
        a >>= 1;
        b >>= 1;
        shift++;
    }

    // 使 'a' 变为奇数
    while ((a & 1) == 0) a >>= 1;

    while (b != 0) {
        while ((b & 1) == 0) b >>= 1;
        if (a > b) {
            int temp = a;
            a = b;
            b = temp;
        }
        b = b - a;
    }

    return a << shift;
}

int main(void) {
    int a, b;
    printf("输入 a 和 b: ");
    scanf("%d %d", &a, &b);
    printf("gcd(%d, %d) = %d\n", a, b, gcd_binary(a, b));
}
```

Python 版本

```python
def gcd_binary(a, b):
    if a == 0: return b
    if b == 0: return a
    shift = 0
    while ((a | b) & 1) == 0:
        a >>= 1
        b >>= 1
        shift += 1
    while (a & 1) == 0:
        a >>= 1
    while b != 0:
        while (b & 1) == 0:
            b >>= 1
        if a > b:
            a, b = b, a
        b -= a
    return a << shift

a, b = map(int, input("输入 a, b: ").split())
print("gcd(", a, ",", b, ") =", gcd_binary(a, b))
```

#### 为什么它重要

- 避免除法，仅使用移位、减法和比较
- 在二进制处理器上速度快（对硬件友好）
- 适用于无符号整数，在嵌入式系统中有用
- 展示了位运算解决经典问题的威力

#### 一个温和的证明（为什么它有效）

最大公约数的规则保持不变：

- gcd(2a, 2b) = 2 × gcd(a, b)
- 如果 a 是奇数，则 gcd(a, 2b) = gcd(a, b)
- 如果 a 和 b 都是奇数且 a < b，则 gcd(a, b) = gcd(a, b − a)

每一步都保持最大公约数的性质，同时高效地移除因子 2。

通过归纳法，当 a = b 时，该值就是最大公约数。

#### 自己动手试试

1. 逐步计算 gcd(48, 18)。
2. 尝试计算 gcd(56, 98)。
3. 修改代码以统计操作次数。
4. 与欧几里得版本比较运行时间。
5. 在 64 位整数上使用并测试大输入。

#### 测试用例

| a   | b   | gcd(a, b) | 步骤 |
| --- | --- | --------- | ----- |
| 48  | 18  | 6         | 通过     |
| 56  | 98  | 14        | 通过     |
| 101 | 10  | 1         | 通过     |
| 270 | 192 | 6         | 通过     |
| 0   | 8   | 8         | 通过     |

#### 复杂度

- 时间：O(log min(a, b))
- 空间：O(1)
- 位运算使其在实践中比基于除法的最大公约数算法更快

二进制最大公约数是欧几里得算法的二进制形式，是比特之舞中的减法、移位与对称。
### 509 模约简

模约简是通过取一个数除以 M 的余数，将其带回指定范围的过程。它是模运算中虽小但至关重要的一步，每个模算法都使用它来保持数字有界且稳定。

可以将其想象成将一条无限长的数轴折叠成一个长度为 M 的圆环，然后问："我们落在了哪里？"

#### 我们要解决什么问题？

我们想要计算：

$$
x \bmod M
$$

也就是说，当 (x) 除以 (M) 时的余数，总是映射到规范范围 [0, M−1] 内。

示例：
如果 $M = 10$：

| x  | x mod 10 |
| -- | -------- |
| 23 | 3        |
| 17 | 7        |
| 0  | 0        |
| -3 | 7        |

负数会绕回到一个正余数。

因此我们想要一个规范化的结果，永远不会是负数。

#### 它是如何工作的（通俗解释）？

在大多数编程语言中，取余运算符 `%` 对于负输入可能会产生负结果。为了确保得到正确的模余数，我们通过必要时加回 M 来修正它。

经验法则：

$$
\text{mod}(x, M) = ((x % M) + M) % M
$$

| x  | M  | x % M | 规范化后 |
| -- | -- | ----- | ---------- |
| 23 | 10 | 3     | 3          |
| -3 | 10 | -3    | 7          |
| 15 | 6  | 3     | 3          |
| -8 | 5  | -3    | 2          |

这确保了 x mod M ∈ [0, M−1]。

#### 简短代码（简易版本）

C 语言版本

```c
#include <stdio.h>

int mod_reduce(int x, int M) {
    int r = x % M;
    if (r < 0) r += M;
    return r;
}

int main(void) {
    int x, M;
    printf("输入 x 和 M: ");
    scanf("%d %d", &x, &M);
    printf("%d mod %d = %d\n", x, M, mod_reduce(x, M));
}
```

Python 版本

```python
def mod_reduce(x, M):
    return (x % M + M) % M

x, M = map(int, input("输入 x, M: ").split())
print(f"{x} mod {M} =", mod_reduce(x, M))
```

#### 为什么它很重要

- 即使对于负数也能确保正确的余数
- 保持算术运算的一致性：$(a + b) \bmod M = ((a \bmod M) + (b \bmod M)) \bmod M$
- 在密码学、哈希、多项式模运算中至关重要
- 防止模运算中的溢出和符号错误

#### 一个温和的证明（为什么它有效）

任何整数 (x) 都可以写成：

$$
x = qM + r
$$

其中 (q) 是商，(r) 是余数。我们希望 (r \in [0, M-1])。

如果 `%` 给出了负的 (r)，那么 (r + M) 就是正余数：
$$
(x \bmod M) = (x % M + M) % M
$$

这满足模同余关系：
$$
x \equiv r \pmod{M}
$$

#### 自己动手试试

1.  手动计算 $-7 \bmod 5$。
2.  用负数输入测试代码。
3.  使用规范化约简构建 `mod_add`、`mod_sub`、`mod_mul`。
4.  在循环内部使用约简来防止溢出。
5.  在 C 或 Python 中比较 `%` 与正确的模运算在处理负数时的差异。

#### 测试用例

| x  | M  | 期望值 | 检查 |
| -- | -- | -------- | ----- |
| 23 | 10 | 3        | Ok     |
| -3 | 10 | 7        | Ok     |
| 15 | 6  | 3        | Ok     |
| -8 | 5  | 2        | Ok     |
| 0  | 9  | 0        | Ok     |

#### 复杂度

- 时间复杂度：O(1)
- 空间复杂度：O(1)

模约简是模运算的核心，每次运算都折叠回一个圆环，使数字保持小、正且可预测。
### 510 模线性方程求解器

模线性方程是形如
$$
a x \equiv b \pmod{m}
$$
的方程。我们想要找到所有满足这个同余式的整数 $x$。
这是求解 $ax = b$ 的模运算版本，但运算是在一个以 $m$ 的倍数为周期循环的世界中进行的。

#### 我们要解决什么问题？

给定整数 $a,b,m$，求解线性同余式
$$
a x \equiv b \pmod{m}.
$$

关键事实

- 令 $g=\gcd(a,m)$。
- 当且仅当 $g \mid b$ 时，解存在。
- 如果解存在，则模 $m$ 下恰好有 $g$ 个解，它们之间的间隔为 $m/g$。

求解步骤

1) 计算 $g=\gcd(a,m)$。如果 $g \nmid b$，则无解。
2) 化简：
$$
a'=\frac{a}{g},\quad b'=\frac{b}{g},\quad m'=\frac{m}{g}.
$$
然后求解互质的同余式
$$
a' x \equiv b' \pmod{m'}.
$$
3) 求逆元 $a'^{-1} \pmod{m'}$ 并令
$$
x_0 \equiv a'^{-1} b' \pmod{m'}.
$$
4) 模 $m$ 下的所有解为
$$
x \equiv x_0 + t\cdot \frac{m}{g} \pmod{m},\quad t=0,1,\dots,g-1.
$$

工作示例

求解 $6x \equiv 8 \pmod{14}$。

1) $g=\gcd(6,14)=2$，且 $2 \mid 8$，所以解存在。
2) 化简：$a'=6/2=3$，$b'=8/2=4$，$m'=14/2=7$。求解
$$
3x \equiv 4 \pmod{7}.
$$
3) 逆元：$3^{-1}\equiv 5 \pmod{7}$，所以
$$
x_0 \equiv 5\cdot 4 \equiv 20 \equiv 6 \pmod{7}.
$$
4) 提升到模 14。因为 $m/g=7$，解为
$$
x \equiv 6 + t\cdot 7 \pmod{14},\quad t=0,1.
$$
因此 $x \in \{6,\,13\} \pmod{14}$。

#### 它是如何工作的（通俗解释）？

1. 检查可解性  
   当且仅当 $g=\gcd(a,m)$ 整除 $b$ 时，解存在。

2. 用 $g$ 化简同余式
   $$
   \frac{a}{g}\,x \equiv \frac{b}{g} \pmod{\frac{m}{g}}
   $$

3. 求模逆元  
   计算 $\left(\frac{a}{g}\right)^{-1} \pmod{\frac{m}{g}}$。

4. 求出一个解，然后枚举所有 $g$ 个解
   $$
   x_0 \equiv \left(\frac{a}{g}\right)^{-1}\!\left(\frac{b}{g}\right) \pmod{\frac{m}{g}}
   $$
   $$
   x \equiv x_0 + k\cdot\frac{m}{g} \pmod{m},\quad k=0,1,\ldots,g-1
   $$

#### 分步表示例

| 步骤 | 方程                     | 操作                  | 结果              |
| ---- | ------------------------ | --------------------- | ----------------- |
| 1    | $6x \equiv 8 \pmod{14}$  | $\gcd(6,14)=2 \mid 8$ | 可解              |
| 2    | 除以 $2$                 | $3x \equiv 4 \pmod{7}$| 已简化            |
| 3    | $3 \bmod 7$ 的逆元       | $5$                   | 因为 $3\cdot5\equiv1$ |
| 4    | 两边同乘                 | $x \equiv 4\cdot5 \equiv 20 \equiv 6 \pmod{7}$ | 一个解 |
| 5    | 提升到 $\pmod{14}$       | $x \in \{6,\,13\}$    | 完成              |

#### 微型代码（简易版本）

Python 版本

```python
def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x1, y1 = extended_gcd(b, a % b)
    return g, y1, x1 - (a // b) * y1

def solve_modular_linear(a, b, m):
    g, x, y = extended_gcd(a, m)
    if b % g != 0:
        return []  # 无解
    a1, b1, m1 = a // g, b // g, m // g
    x0 = (x * b1) % m1
    return [(x0 + i * m1) % m for i in range(g)]

a, b, m = map(int, input("输入 a, b, m: ").split())
solutions = solve_modular_linear(a, b, m)
if solutions:
    print("解:", solutions)
else:
    print("无解")
```

C 语言版本（简化版）

```c
#include <stdio.h>

long long extended_gcd(long long a, long long b, long long *x, long long *y) {
    if (b == 0) { *x = 1; *y = 0; return a; }
    long long x1, y1;
    long long g = extended_gcd(b, a % b, &x1, &y1);
    *x = y1;
    *y = x1 - (a / b) * y1;
    return g;
}

int solve_modular(long long a, long long b, long long m, long long sol[]) {
    long long x, y;
    long long g = extended_gcd(a, m, &x, &y);
    if (b % g != 0) return 0;
    a /= g; b /= g; m /= g;
    long long x0 = ((x * b) % m + m) % m;
    for (int i = 0; i < g; i++)
        sol[i] = (x0 + i * m) % (m * g);
    return g;
}

int main(void) {
    long long a, b, m, sol[10];
    printf("输入 a, b, m: ");
    scanf("%lld %lld %lld", &a, &b, &m);
    int n = solve_modular(a, b, m, sol);
    if (n == 0) printf("无解\n");
    else {
        printf("解:");
        for (int i = 0; i < n; i++) printf(" %lld", sol[i]);
        printf("\n");
    }
}
```

#### 为什么它很重要

- 求解模方程，是构建中国剩余定理、RSA 和丢番图方程组的基础
- 推广了模逆元（当 b=1 时）
- 是求解线性同余方程组的基础
- 使得在非素数模数下进行模除法成为可能

#### 一个温和的证明（为什么它有效）

如果 $a x \equiv b \pmod{m}$，那么 $m \mid (a x - b)$。

令 $g=\gcd(a,m)$。用 $g$ 除同余式：
$$
\frac{a}{g}\,x \equiv \frac{b}{g} \pmod{\frac{m}{g}}.
$$
现在 $\gcd\!\left(\frac{a}{g},\frac{m}{g}\right)=1$，所以 $\left(\frac{a}{g}\right)^{-1} \pmod{\frac{m}{g}}$ 存在。两边同时乘以这个逆元，得到模 $\frac{m}{g}$ 下的一个解。加上 $\frac{m}{g}$ 的倍数就生成了模 $m$ 下的所有 $g$ 个解：
$$
x \equiv x_0 + k\cdot \frac{m}{g} \pmod{m}, \quad k=0,1,\dots,g-1.
$$

#### 自己动手试试

1. 求解 $6x \equiv 8 \pmod{14}$
2. 求解 $4x \equiv 2 \pmod{6}$
3. 求解 $3x \equiv 2 \pmod{7}$
4. 尝试一个无解的情况：$4x \equiv 3 \pmod{6}$
5. 修改代码，使其在每一步都打印 $\gcd$ 和逆元

#### 测试用例

| a | b | m  | 解         | 检查 |
| - | - | -- | ---------- | ---- |
| 6 | 8 | 14 | 6, 13      | 通过 |
| 4 | 2 | 6  | 2, 5       | 通过 |
| 3 | 2 | 7  | 3          | 通过 |
| 4 | 3 | 6  | 无         | 通过 |

#### 复杂度

- 时间复杂度：O(log m)
- 空间复杂度：O(1)

模线性求解器将算术转化为圆上的代数，寻找线性直线与模网格相交的位置。

# 第 52 节 素性与因式分解
### 5.1.1 试除法

试除法是判断一个数是否为素数的最简单方法，即检查是否有更小的数能将其整除。
对于较大的 $n$ 来说它很慢，但对于建立直观理解、处理小素数，以及作为更高级的因数分解或素性测试中的子程序来说，它是完美的。

#### 我们要解决什么问题？

我们想判断一个整数 $n > 1$ 是素数还是合数。

素数恰好有两个除数（1 和它自身）。
合数则有额外的除数。

试除法检查所有可能的除数，直到 $\sqrt{n}$。

**示例**
$n = 37$ 是素数吗？

检查除数：
$$
\begin{aligned}
37 \bmod 2 &= 1 \\
37 \bmod 3 &= 1 \\
37 \bmod 4 &= 1 \\
37 \bmod 5 &= 2 \\
37 \bmod 6 &= 1
\end{aligned}
$$

直到 $\sqrt{37}$ 都没有找到除数。因此，是素数。

#### 它是如何工作的（通俗解释）

如果 $n = a \times b$，那么 $a$ 或 $b$ 中必有一个满足 $a, b \leq \sqrt{n}$。
所以，如果 $n$ 有一个除数，它会在 $\sqrt{n}$ 之前出现。
我们检查该范围内的每个整数。

为了优化：

- 首先检查 $2$
- 然后只测试奇数

| 步骤 | 除数             | $n \bmod \text{Divisor}$ | 结果       |
| ---- | ---------------- | -------------------------- | ---------- |
| 1    | 2                | 1                          | 跳过       |
| 2    | 3                | 1                          | 跳过       |
| 3    | 4                | 1                          | 跳过       |
| 4    | 5                | 2                          | 跳过       |
| 5    | 6                | 1                          | 跳过       |
|      | 未找到除数       | 素数                       |            |

#### 微型代码（简易版本）

C 语言版本

```c
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

bool is_prime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    int limit = (int)sqrt(n);
    for (int i = 3; i <= limit; i += 2)
        if (n % i == 0)
            return false;
    return true;
}

int main(void) {
    int n;
    printf("请输入 n: ");
    scanf("%d", &n);
    printf("%d 是 %s\n", n, is_prime(n) ? "素数" : "合数");
}
```

Python 版本

```python
import math

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    limit = int(math.sqrt(n)) + 1
    for i in range(3, limit, 2):
        if n % i == 0:
            return False
    return True

n = int(input("请输入 n: "))
print(n, "是", "素数" if is_prime(n) else "合数")
```

#### 为什么它很重要

- 是所有素性测试的基础
- 用于小的 $n$ 和初始筛选
- 非常适合小数的因数分解
- 有助于建立对 $\sqrt{n}$ 边界和除数对的直观理解

#### 一个温和的证明（为什么它有效）

如果 $n = a \times b$，
那么 $a, b$ 中必有一个 $\leq \sqrt{n}$。

如果两者都大于 $\sqrt{n}$，那么：
$$
a \times b > \sqrt{n} \times \sqrt{n} = n
$$
这是不可能的。
因此，任何因数分解都必须包含一个 ≤ $\sqrt{n}$ 的数。

所以，检查到 $\sqrt{n}$ 就足以确认素性。

#### 自己动手试试

1.  检查 $37, 49, 51$ 是否是素数。
2.  修改代码以打印找到的第一个除数。
3.  扩展代码以列出 $n$ 的所有除数。
4.  比较 $n = 10^6$ 和 $n = 10^9$ 的运行时间。
5.  结合筛法以跳过非素数除数。

#### 测试用例

| $n$  | 预期结果 | 第一个除数 |
| ---- | -------- | ---------- |
| 2    | 素数     |            |
| 3    | 素数     |            |
| 4    | 合数     | 2          |
| 9    | 合数     | 3          |
| 37   | 素数     |            |
| 49   | 合数     | 7          |

#### 复杂度

-   时间复杂度：$O(\sqrt{n})$
-   空间复杂度：$O(1)$

试除法是素性测试的 "hello world"，简单、确定，并且是数论的基础。
### 512 埃拉托斯特尼筛法

埃拉托斯特尼筛法是一种经典且高效的算法，用于找出所有不超过给定上限 $n$ 的素数。
它不单独测试每个数字，而是通过排除已知素数的倍数，只留下素数。

这种筛法是已知最古老的算法之一（已有 2000 多年历史），至今仍是计算数论的基石。

#### 我们要解决什么问题？

我们想要生成所有不超过 $n$ 的素数。

示例
找出所有 $\leq 30$ 的素数：

$$
{2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
$$

#### 它是如何工作的（通俗解释）

想象一个整数列表 $2, 3, \ldots, n$。
我们将反复“划掉”每个素数的倍数。

步骤：

1.  从第一个素数 $p = 2$ 开始。
2.  从 $p^2$ 开始，划掉 $p$ 的所有倍数。
3.  移动到下一个尚未被划掉的数字，那就是下一个素数。
4.  重复直到 $p^2 > n$。

剩下的未被标记的数字就是所有的素数。

示例：$n = 30$

| 步骤 | 素数 $p$ | 移除的倍数                        | 剩余的素数                             |
| ---- | -------- | --------------------------------- | -------------------------------------- |
| 1    | 2        | $4, 6, 8, 10, 12, \ldots$         | $2, 3, 5, 7, 9, 11, 13, 15, \ldots$    |
| 2    | 3        | $9, 12, 15, 18, 21, 24, 27, 30$   | $2, 3, 5, 7, 11, 13, 17, 19, 23, 29$   |
| 3    | 5        | $25, 30$                          | 无变化                                 |
| 4    | 停止     | $5^2 = 25 > \sqrt{30}$            | 完成                                   |

最终不超过 30 的素数：

$$
{2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
$$

#### 精简代码（简易版本）

C 语言版本

```c
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

void sieve(int n) {
    bool is_prime[n + 1];
    for (int i = 0; i <= n; i++) is_prime[i] = true;
    is_prime[0] = is_prime[1] = false;

    for (int p = 2; p * p <= n; p++) {
        if (is_prime[p]) {
            for (int multiple = p * p; multiple <= n; multiple += p)
                is_prime[multiple] = false;
        }
    }

    printf("不超过 %d 的素数：\n", n);
    for (int i = 2; i <= n; i++)
        if (is_prime[i]) printf("%d ", i);
    printf("\n");
}

int main(void) {
    int n;
    printf("输入 n：");
    scanf("%d", &n);
    sieve(n);
}
```

Python 版本

```python
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    p = 2
    while p * p <= n:
        if is_prime[p]:
            for multiple in range(p * p, n + 1, p):
                is_prime[multiple] = False
        p += 1
    return [i for i in range(2, n + 1) if is_prime[i]]

n = int(input("输入 n："))
print("素数：", sieve(n))
```

#### 为什么它很重要

-   高效生成所有不超过 $n$ 的素数
-   用于数论、密码学、因式分解和素数筛法预计算
-   避免了重复除法运算
-   是高级筛法（如线性筛法、分段筛法）的基础

#### 一个温和的证明（为什么它有效）

每个合数 $n$ 都有一个最小的质因数 $p$。
当算法处理到 $p$ 时，这个因数 $p$ 会标记该合数。

因此，通过划掉每个素数的所有倍数，所有合数都被移除，只剩下素数。

因为任何合数 $n = a \times b$ 至少有一个 $a \leq \sqrt{n}$，
所以当 $p^2 > n$ 时，我们可以停止筛选。

#### 亲自尝试

1.  生成 $\leq 50$ 的素数。
2.  修改代码以统计找到了多少个素数。
3.  每行打印 10 个素数。
4.  比较 $n = 10^4$, $10^5$, $10^6$ 时的运行时间。
5.  优化内存：只筛选奇数。

#### 测试用例

| $n$ | 期望输出                          |
| --- | --------------------------------- |
| 10  | $2, 3, 5, 7$                      |
| 20  | $2, 3, 5, 7, 11, 13, 17, 19$      |
| 30  | $2, 3, 5, 7, 11, 13, 17, 19, 23, 29$ |

#### 复杂度

-   时间：$O(n \log \log n)$
-   空间：$O(n)$

埃拉托斯特尼筛法融合了清晰性和高效性，通过划掉每一个合数，它留下了塑造数论的素数。
### 513 阿特金筛法

阿特金筛法是埃拉托斯特尼筛法的一项现代改进。
它不是通过划掉倍数，而是使用二次型和模运算来确定素数候选，然后通过平方倍数来剔除非素数。
它在渐进意义上更快且数学上非常优美，尽管实现起来更复杂。

#### 我们要解决什么问题？

我们想找到所有不超过给定上限 $n$ 的素数，但要比经典筛法更快。

阿特金筛法使用基于二次剩余的合同条件来检测潜在的素数。

#### 它是如何工作的（通俗解释）

对于给定的整数 $n$，我们通过检查特定的模方程来确定它是否是素数候选：

1. 初始化一个数组 `is_prime[0..n]` 为 false。

2. 对于每个整数对 $(x, y)$，其中 $x, y \ge 1$，计算：

   * $n_1 = 4x^2 + y^2$

     * 如果 $n_1 \le N$ 且 $n_1 \bmod 12 \in {1, 5}$，则翻转 `is_prime[n1]`
   * $n_2 = 3x^2 + y^2$

     * 如果 $n_2 \le N$ 且 $n_2 \bmod 12 = 7$，则翻转 `is_prime[n2]`
   * $n_3 = 3x^2 - y^2$

     * 如果 $x > y$，$n_3 \le N$，且 $n_3 \bmod 12 = 11$，则翻转 `is_prime[n3]`

3. 消除平方数的倍数：

   * 对于每个满足 $k^2 \le N$ 的 $k$，
     将所有 $k^2$ 的倍数标记为合数。

4. 最后，将 2 和 3 添加为素数。

所有剩余标记为 true 的数字就是素数。

示例（小 N = 50）：

从 2、3 和 5 开始。
应用模条件检测其他数：

| 条件         | 公式    | 模类      | 候选数                      |
| ------------ | ------- | --------- | --------------------------- |
| $4x^2 + y^2$ | $1, 5$  | $12$      | $5, 13, 17, 29, 37, 41, 49$ |
| $3x^2 + y^2$ | $7$     | $12$      | $7, 19, 31, 43$             |
| $3x^2 - y^2$ | $11$    | $12$      | $11, 23, 47$                |

然后移除平方数的倍数（例如 $25, 49$）。
不超过 50 的剩余素数：

$$
{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
$$

#### 精简代码（简易版本）

Python 版本

```python
import math

def sieve_atkin(limit):
    is_prime = [False] * (limit + 1)
    sqrt_limit = int(math.sqrt(limit)) + 1

    for x in range(1, sqrt_limit):
        for y in range(1, sqrt_limit):
            n = 4 * x * x + y * y
            if n <= limit and n % 12 in (1, 5):
                is_prime[n] = not is_prime[n]

            n = 3 * x * x + y * y
            if n <= limit and n % 12 == 7:
                is_prime[n] = not is_prime[n]

            n = 3 * x * x - y * y
            if x > y and n <= limit and n % 12 == 11:
                is_prime[n] = not is_prime[n]

    for n in range(5, sqrt_limit):
        if is_prime[n]:
            for k in range(n * n, limit + 1, n * n):
                is_prime[k] = False

    primes = [2, 3]
    primes.extend([i for i in range(5, limit + 1) if is_prime[i]])
    return primes

n = int(input("输入 n: "))
print("素数:", sieve_atkin(n))
```

#### 为什么它很重要

- 对于非常大的 $n$，比埃拉托斯特尼筛法更快
- 展示了数论与计算之间的深刻联系
- 使用二次剩余的模模式来检测素数
- 是优化筛法算法的基础

#### 一个温和的证明（为什么它有效）

每个整数都可以用二次型表示。
素数出现在特定的模类中：

- 形如 $4x + 1$ 的素数满足 $4x^2 + y^2$
- 形如 $6x + 1$ 或 $6x + 5$ 的素数满足 $3x^2 + y^2$ 或 $3x^2 - y^2$

通过计算这些模 12 同余的解的个数，可以区分素数和合数。
翻转操作确保每个候选数只有在满足素数条件时才会被切换奇数次。

平方数的倍数被消除，因为任何平方因子都不能属于一个素数。

#### 自己动手试试

1. 生成所有 $\le 100$ 的素数。
2. 将输出结果与埃拉托斯特尼筛法进行比较。
3. 测量 $n = 10^6$ 时的性能。
4. 修改代码，使其仅统计素数个数。
5. 打印候选数的翻转情况，看看切换是如何工作的。

#### 测试用例

| $n$ | 期望输出（素数）                                           |
| --- | -------------------------------------------------------- |
| 10  | $2, 3, 5, 7$                                             |
| 30  | $2, 3, 5, 7, 11, 13, 17, 19, 23, 29$                     |
| 50  | $2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47$ |

#### 复杂度

- 时间：$O(n)$（使用模运算预计算）
- 空间：$O(n)$

阿特金筛法将数论转化为代码，利用二次型的优雅性从无限的整数海洋中筛选出素数。
### 514 Miller–Rabin（米勒-拉宾）素性测试

Miller–Rabin 测试是一种快速的概率性素性测试。
给定一个大于 2 的奇数 $n$，它通过检查 $a$ 是否表现为 $n$ 的合数性见证（witness）来判断 $n$ 是合数还是可能为素数。

#### 我们要解决什么问题？

判断一个大整数 $n$ 的素性，比试除法或完整的筛法快得多，尤其是当 $n$ 可能有数百或数千位时。

输入：奇数 $n > 2$，精度参数 $k$（基数的数量）。
输出：合数 或 可能素数。

#### 它是如何工作的（通俗解释）

将 $n-1$ 写成：
$$
n - 1 = 2^s \cdot d \quad \text{其中 } d \text{ 为奇数}.
$$

对 $k$ 个随机基数 $a \in {2, 3, \ldots, n-2}$ 重复以下步骤：

1. 计算
   $$
   x \equiv a^{,d} \bmod n.
   $$

2. 如果 $x = 1$ 或 $x = n-1$，则该基数通过测试。

3. 否则，进行最多 $s-1$ 次平方运算：
   $$
   x \leftarrow x^2 \bmod n.
   $$
   如果在任何一步 $x = n-1$，则该基数通过。
   如果始终没有出现 $n-1$，则判定为合数。

如果所有 $k$ 个基数都通过，则判定为可能素数。
对于合数 $n$，一个随机基数揭露其合数性的概率至少为 $1/4$，因此错误概率最多为 $(1/4)^k$。

#### 精简代码（简易版本）

C 语言版本

```c
#include <stdio.h>
#include <stdint.h>

static uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t m) {
    __uint128_t z =$__uint128_t$a * b % m;
    return (uint64_t)z;
}

static uint64_t pow_mod(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m;
    a %= m;
    while (e > 0) {
        if (e & 1) r = mul_mod(r, a, m);
        a = mul_mod(a, a, m);
        e >>= 1;
    }
    return r;
}

static int miller_rabin_base(uint64_t n, uint64_t a, uint64_t d, int s) {
    uint64_t x = pow_mod(a, d, n);
    if (x == 1 || x == n - 1) return 1;
    for (int i = 1; i < s; i++) {
        x = mul_mod(x, x, n);
        if (x == n - 1) return 1;
    }
    return 0; // 对于此基数为合数
}

int is_probable_prime(uint64_t n) {
    if (n < 2) return 0;
    for (uint64_t p : (uint64_t[]){2,3,5,7,11,13,17,19,23,0}) {
        if (p == 0) break;
        if (n % p == 0) return n == p;
    }
    // 将 n-1 写成 n-1 = 2^s * d
    uint64_t d = n - 1;
    int s = 0;
    while ((d & 1) == 0) { d >>= 1; s++; }

    // 针对 64 位整数的确定性基数集合
    uint64_t bases[] = {2, 3, 5, 7, 11, 13, 17};
    int nb = sizeof(bases) / sizeof(bases[0]);
    for (int i = 0; i < nb; i++) {
        uint64_t a = bases[i] % n;
        if (a <= 1) continue;
        if (!miller_rabin_base(n, a, d, s)) return 0;
    }
    return 1; // 可能素数
}

int main(void) {
    uint64_t n;
    if (scanf("%llu", &n) != 1) return 0;
    printf("%llu is %s\n", n, is_probable_prime(n) ? "probably prime" : "composite");
    return 0;
}
```

Python 版本

```python
def pow_mod(a, e, m):
    r = 1
    a %= m
    while e > 0:
        if e & 1:
            r = (r * a) % m
        a = (a * a) % m
        e >>= 1
    return r

def miller_rabin(n, bases=None):
    if n < 2:
        return False
    small_primes = [2,3,5,7,11,13,17,19,23]
    for p in small_primes:
        if n % p == 0:
            return n == p
    # 将 n-1 写成 n-1 = 2^s * d
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    if bases is None:
        # 针对 64 位范围的确定性基数
        bases = [2, 3, 5, 7, 11, 13, 17]
    for a in bases:
        a %= n
        if a <= 1:
            continue
        x = pow_mod(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

n = int(input().strip())
print("probably prime" if miller_rabin(n) else "composite")
```

#### 为什么它很重要

- 对于大整数是非常快速的筛选测试。
- 是密码学密钥生成流程中的标准步骤。
- 使用固定的基数集合时，对于有界范围（对于 64 位整数，给定的基数集合就足够了）可以变为确定性测试。

#### 一个温和的证明（为什么它有效）

费马小定理式的动机：如果 $n$ 是素数且 $\gcd(a,n)=1$，那么根据欧拉定理
$$
a^{n-1} \equiv 1 \pmod n.
$$
更强地，通过写成 $n-1 = 2^s d$，序列
$$
a^{d},\ a^{2d},\ a^{4d},\ \ldots,\ a^{2^{s-1}d} \pmod n
$$
必须最终到达 $1$，并且在这个过程中，只有在紧邻 $1$ 之前才可能出现值 $-1 \equiv n-1$。如果这个链错过了 $n-1$，那么 $a$ 就证明了 $n$ 是合数。对于合数 $n$，至少四分之三的 $a$ 是见证数，因此在 $k$ 个独立的基数测试后，错误概率最多为 $(1/4)^k$。

#### 自己动手试试

1.  对 $n = 561$ 将 $n-1$ 分解为 $2^s d$，并用基数 $a = 2, 3, 5$ 进行测试。
2.  生成随机的 64 位奇数 $n$，并将 Miller–Rabin 测试与试除法（直到 $10^6$）的结果进行比较。
3.  将基数替换为随机选择，并在卡迈克尔数（Carmichael numbers）上测量错误频率。
4.  扩展基数集合，使其覆盖你的目标范围。

#### 测试用例

| $n$                 | 结果                      | 备注                   |
| ------------------- | --------------------------- | ----------------------- |
| $37$                | 可能素数              | 素数                   |
| $221 = 13 \cdot 17$ | 合数                   | 有小因子            |
| $561$               | 合数                   | 卡迈克尔数       |
| $2^{61}-1$          | 可能素数              | 梅森素数候选数      |
| $10^{18}+3$         | 合数 或 可能素数 | 取决于实际值 |

#### 复杂度

-   时间：使用普通模乘法时为 $O(k \log^3 n)$，其中 $k$ 是基数数量。
-   空间：$O(1)$ 的迭代状态。

Miller–Rabin 提供了一个快速可靠的素性筛查：分解 $n-1$，测试几个基数，然后要么证明是合数，要么返回一个非常强的“可能素数”结果。
### 515 费马素性测试

费马素性测试是判断一个数是否为可能素数的最简单的概率性测试之一。
它基于费马小定理，该定理指出：如果 $n$ 是素数且 $a$ 不能被 $n$ 整除，那么

$$
a^{n-1} \equiv 1 \pmod{n}.
$$

如果对于某个底数 $a$，这个同余式不成立，那么 $n$ 肯定是合数。
如果它对几个随机选择的底数都成立，那么 $n$ 很可能是素数。

#### 我们要解决什么问题？

我们需要一种快速检查 $n$ 是否为素数的方法，特别是对于大 $n$，此时试除法或筛法太慢。

我们测试数字是否满足费马同余条件：

$$
a^{n-1} \bmod n = 1
$$

其中底数 $a \in [2, n-2]$ 是随机选择的。

#### 它是如何工作的（通俗解释）

1.  选择一个随机整数 $a$，满足 $2 \le a \le n-2$。
2.  计算 $x = a^{n-1} \bmod n$。
3.  如果 $x \ne 1$，$n$ 是合数。
4.  如果 $x = 1$，$n$ 可能是素数。
5.  用不同的 $a$ 重复几次以提高置信度。

如果 $n$ 通过了所有选定底数的测试，我们称其为可能素数。
如果它在任何底数上失败，则为合数。

示例：测试 $n = 561$（一个卡迈克尔数）。

-   选择 $a = 2$：$2^{560} \bmod 561 = 1$
-   选择 $a = 3$：$3^{560} \bmod 561 = 1$
-   选择 $a = 5$：$5^{560} \bmod 561 = 1$

全部通过，但 $561 = 3 \cdot 11 \cdot 17$ 是合数。
因此，费马测试可能被卡迈克尔数欺骗。

#### 简易代码（简单版本）

Python 版本

```python
import random

def pow_mod(a, e, m):
    r = 1
    a %= m
    while e > 0:
        if e & 1:
            r = (r * a) % m
        a = (a * a) % m
        e >>= 1
    return r

def fermat_test(n, k=5):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    for _ in range(k):
        a = random.randint(2, n - 2)
        if pow_mod(a, n - 1, n) != 1:
            return False
    return True

n = int(input("输入 n: "))
print("可能素数" if fermat_test(n) else "合数")
```

C 语言版本

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

long long pow_mod(long long a, long long e, long long m) {
    long long r = 1;
    a %= m;
    while (e > 0) {
        if (e & 1) r = (r * a) % m;
        a = (a * a) % m;
        e >>= 1;
    }
    return r;
}

int fermat_test(long long n, int k) {
    if (n < 2) return 0;
    if (n == 2 || n == 3) return 1;
    if (n % 2 == 0) return 0;
    srand(time(NULL));
    for (int i = 0; i < k; i++) {
        long long a = 2 + rand() % (n - 3);
        if (pow_mod(a, n - 1, n) != 1)
            return 0;
    }
    return 1;
}

int main(void) {
    long long n;
    printf("输入 n: ");
    scanf("%lld", &n);
    printf("%lld 是 %s\n", n, fermat_test(n, 5) ? "可能素数" : "合数");
}
```

#### 为什么它重要

-   实现极其简单
-   可作为更强测试（如米勒-拉宾测试）之前的初步筛选
-   是数论中许多概率性算法的基础
-   有助于以计算形式阐释费马小定理

#### 温和的证明（为什么它有效）

如果 $n$ 是素数且 $\gcd(a, n) = 1$，费马小定理保证：

$$
a^{n-1} \equiv 1 \pmod{n}.
$$

如果这不成立，$n$ 就不可能是素数。
然而，一些合数（卡迈克尔数）对于所有与 $n$ 互质的 $a$ 都满足这个条件。
因此，该测试是概率性的，而非确定性的。

#### 亲自尝试

1.  用底数 2, 3, 5 测试 $n = 37$。
2.  尝试 $n = 561$ 并观察其失败情况。
3.  增加 $k$（试验次数）以观察稳定性。
4.  与米勒-拉宾测试比较运行时间。
5.  构造一个能通过一个底数测试但通不过另一个底数测试的合数。

#### 测试用例

| $n$ | 底数     | 结果                     |
| --- | ------- | ---------------------- |
| 37  | 2, 3, 5 | 通过 → 可能素数             |
| 15  | 2       | 失败 → 合数                |
| 561 | 2, 3, 5 | 通过 → 假阳性（误判为素数） |
| 97  | 2, 3    | 通过 → 可能素数             |

#### 复杂度

-   时间：$O(k \log^3 n)$（由于模幂运算）
-   空间：$O(1)$

费马测试是闪电般的素性测试，但有一个骗子的缺陷：它可能被那些被称为卡迈克尔数的狡猾合数所欺骗。
### 516 Pollard's Rho 算法

Pollard's Rho 算法是一种巧妙的随机化整数分解方法。
它使用一个简单的迭代函数和生日悖论来快速找到非平凡因子，而无需试除法。
虽然是概率性的，但它在寻找大数的小因子方面极其有效。

#### 我们要解决什么问题？

给定一个合数 $n$，找到一个非平凡因子 $d$，使得 $1 < d < n$。

Pollard's Rho 算法不是穷举检查可除性，而是使用模 $n$ 的伪随机序列，并检测两个值何时对一个隐藏因子模同余。

#### 它是如何工作的（通俗解释）

我们定义一个迭代：

$$
x_{i+1} = f(x_i) \bmod n, \quad \text{通常 } f(x) = (x^2 + c) \bmod n
$$

两个以不同速度运行的序列（就像“乌龟和兔子”）最终会模 $n$ 的一个因子发生碰撞。
当它们碰撞时，它们的差与 $n$ 的最大公约数（gcd）会给出一个因子。

**算法大纲**

1.  选择一个随机函数 $f(x) = (x^2 + c) \bmod n$，其中 $x_0$ 和 $c$ 是随机的。
2.  设置 $x = y = x_0$，$d = 1$。
3.  当 $d = 1$ 时：
    *   $x = f(x)$
    *   $y = f(f(y))$
    *   $d = \gcd(|x - y|, n)$
4.  如果 $d = n$，则用新函数重新开始。
5.  如果 $1 < d < n$，输出 $d$。

**示例：**
令 $n = 8051 = 83 \times 97$。
选择 $f(x) = (x^2 + 1) \bmod 8051$，$x_0 = 2$。

| 步骤 | $x$ | $y$ | $|x-y|$ | $\gcd(|x-y|, 8051)$ |
|------|------|------|------|------------------|
| 1 | 5 | 26 | 21 | 1 |
| 2 | 26 | 7474 | 7448 | 83 |

好的！找到因子 $83$。

#### 精简代码（简易版本）

**Python 版本**

```python
import math
import random

def f(x, c, n):
    return (x * x + c) % n

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    x = random.randint(2, n - 1)
    y = x
    c = random.randint(1, n - 1)
    d = 1
    while d == 1:
        x = f(x, c, n)
        y = f(f(y, c, n), c, n)
        d = math.gcd(abs(x - y), n)
        if d == n:
            return pollard_rho(n)
    return d

n = int(input("输入 n: "))
factor = pollard_rho(n)
print(f"{n} 的非平凡因子: {factor}")
```

**C 版本**

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

long long gcd(long long a, long long b) {
    while (b != 0) {
        long long t = b;
        b = a % b;
        a = t;
    }
    return a;
}

long long f(long long x, long long c, long long n) {
    return (x * x + c) % n;
}

long long pollard_rho(long long n) {
    if (n % 2 == 0) return 2;
    long long x = rand() % (n - 2) + 2;
    long long y = x;
    long long c = rand() % (n - 1) + 1;
    long long d = 1;

    while (d == 1) {
        x = f(x, c, n);
        y = f(f(y, c, n), c, n);
        long long diff = x > y ? x - y : y - x;
        d = gcd(diff, n);
        if (d == n) return pollard_rho(n);
    }
    return d;
}

int main(void) {
    srand(time(NULL));
    long long n;
    printf("输入 n: ");
    scanf("%lld", &n);
    long long factor = pollard_rho(n);
    printf("非平凡因子: %lld\n", factor);
}
```

#### 为什么它很重要

-   比试除法快得多地找到因子
-   在整数分解、密码分析和 RSA 密钥测试中至关重要
-   是高级算法的基础（例如 Pollard's p–1、Brent's Rho）
-   虽然是概率性的，但对于寻找小因子非常高效

#### 一个温和的证明（为什么它有效）

如果两个值模 $n$ 的一个因子 $p$ 同余：
$$
x_i \equiv x_j \pmod{p}, \quad i \ne j
$$
那么 $p \mid (x_i - x_j)$，因此
$$
\gcd(|x_i - x_j|, n) = p.
$$
因为 $p$ 能整除 $n$，但不能整除 $n$ 的全部，所以它揭示了一个非平凡因子。

"Rho" 形状指的是模 $p$ 重复平方所形成的循环。

#### 亲自尝试

1.  分解 $n = 8051$。
2.  尝试不同的函数 $f(x) = x^2 + c$。
3.  测试 $n = 91, 187, 589$。
4.  与试除法的运行时间进行比较。
5.  结合递归来完全分解 $n$。

#### 测试用例

| $n$  | 因子 | 找到的因子 |
| ---- | ------- | ----- |
| 91   | 7 × 13  | 7     |
| 187  | 11 × 17 | 17    |
| 8051 | 83 × 97 | 83    |
| 2047 | 23 × 89 | 23    |

#### 复杂度

-   期望时间：$O(n^{1/4})$
-   空间：$O(1)$

Pollard's Rho 就像追逐自己的尾巴，但最终，循环会给出一个隐藏的因子。
### 517 Pollard 的 p−1 算法

Pollard 的 p−1 算法是一种专门的因数分解方法，当合数$n$的某个素因子$p$满足$p - 1$是光滑的（即$p-1$能完全分解为小素数）时效果最佳。它是试除法之后最早的实际改进方法之一，对于具有光滑素因子的数来说，它简单、优雅且有效。

#### 我们要解决什么问题？

给定一个合数$n$，我们希望找到一个非平凡因子$d$，满足$1 < d < n$。

这个方法利用了费马小定理：

$$
a^{p-1} \equiv 1 \pmod{p}
$$

如果$p \mid n$，那么$p$整除$a^{p-1} - 1$，即使$p$是未知的。

通过为合适的$M$计算$\gcd(a^M - 1, n)$，我们就有可能找到这样的$p$。

#### 它是如何工作的（通俗解释）

如果$p$是$n$的一个素因子，并且$p-1$整除$M$，
那么$a^{M} \equiv 1 \pmod{p}$。
因此$p$整除$a^{M} - 1$。
计算其与$n$的最大公约数就能揭示出$p$。

算法步骤：

1.  选择一个基数$a$（通常为 2）。
2.  选择一个光滑度边界$B$。
3.  计算：
    $$
    M = \text{lcm}(1, 2, 3, \ldots, B)
    $$
    以及
    $$
    g = \gcd(a^M - 1, n)
    $$
4.  如果$1 < g < n$，返回$g$（一个非平凡因子）。
    如果$g = 1$，增大$B$。
    如果$g = n$，选择另一个$a$。

示例：
令$n = 91 = 7 \times 13$。

取$a = 2$，$B = 5$。
则$M = \text{lcm}(1, 2, 3, 4, 5) = 60$。

计算$g = \gcd(2^{60} - 1, 91)$。

$$
2^{60} - 1 \bmod 91 = 0 \implies g = 7
$$

好的，找到了因子$7$。

#### 微型代码（简易版本）

Python 版本

```python
import math
from math import gcd

def pow_mod(a, e, n):
    r = 1
    a %= n
    while e > 0:
        if e & 1:
            r = (r * a) % n
        a = (a * a) % n
        e >>= 1
    return r

def pollard_p_minus_1(n, B=10, a=2):
    M = 1
    for i in range(2, B + 1):
        M *= i // math.gcd(M, i)
    x = pow_mod(a, M, n)
    g = gcd(x - 1, n)
    if 1 < g < n:
        return g
    return None

n = int(input("请输入 n: "))
factor = pollard_p_minus_1(n, B=10, a=2)
if factor:
    print(f"{n} 的非平凡因子: {factor}")
else:
    print("未找到因子，尝试增大 B")
```

C 语言版本

```c
#include <stdio.h>
#include <stdlib.h>

long long gcd(long long a, long long b) {
    while (b != 0) {
        long long t = b;
        b = a % b;
        a = t;
    }
    return a;
}

long long pow_mod(long long a, long long e, long long n) {
    long long r = 1 % n;
    a %= n;
    while (e > 0) {
        if (e & 1) r = (r * a) % n;
        a = (a * a) % n;
        e >>= 1;
    }
    return r;
}

long long pollard_p_minus_1(long long n, int B, long long a) {
    long long M = 1;
    for (int i = 2; i <= B; i++) {
        long long g = gcd(M, i);
        M = M / g * i;
    }
    long long x = pow_mod(a, M, n);
    long long g = gcd(x - 1, n);
    if (g > 1 && g < n) return g;
    return 0;
}

int main(void) {
    long long n;
    printf("请输入 n: ");
    scanf("%lld", &n);
    long long factor = pollard_p_minus_1(n, 10, 2);
    if (factor)
        printf("非平凡因子: %lld\n", factor);
    else
        printf("未找到因子。尝试增大 B。\n");
}
```

#### 为何重要

-   对于分解具有光滑素因子的数非常出色
-   易于实现
-   与试除法相比速度更快
-   有助于建立对群阶和费马小定理的直观理解

应用于：

-   RSA 密钥验证
-   椭圆曲线方法（ECM）的概念基础
-   教育领域的数论和密码学

#### 一个温和的证明（为何有效）

如果$p \mid n$ 且 $p - 1 \mid M$，
那么根据费马小定理：
$$
a^{p-1} \equiv 1 \pmod{p}
$$
所以
$$
a^{M} \equiv 1 \pmod{p}.
$$

因此$p \mid a^M - 1$，所以
$$
\gcd(a^M - 1, n) \ge p.
$$

如果$p \ne n$，这个最大公约数就给出了一个非平凡因子。

如果$p-1$不是光滑的，则$M$必须更大以包含其因子。

#### 亲自尝试

1.  用$B = 5$分解$91 = 7 \times 13$。
2.  尝试$8051 = 83 \times 97$；逐渐增大$B$。
3.  尝试不同的基数$a$。
4.  与 Pollard Rho 算法比较运行时间。
5.  观察当$p-1$有大素因子时算法的失败情况。

#### 测试用例

| $n$  | 因子       | 边界 $B$ | 找到的因子 |
| ---- | ---------- | -------- | ---------- |
| 91   | 7 × 13     | 5        | 7          |
| 187  | 11 × 17    | 10       | 11         |
| 589  | 19 × 31    | 15       | 19         |
| 8051 | 83 × 97    | 20       | 83         |

#### 复杂度

-   时间：$O(B \log^2 n)$，取决于$p-1$的光滑度
-   空间：$O(1)$

Pollard 的 p−1 方法是一个数学锁孔，当一个因子的阶是由小素数构成时，它能打开合数这把锁。
### 518 轮式筛法

轮式筛法是一种用于试除法的确定性优化方法。它通过构造一个与较小素数互质的候选偏移量的重复模式（即“轮子”），系统地跳过明显的合数。这减少了整除性检查的次数，使得基本的素性测试和因数分解测试快得多。

#### 我们要解决什么问题？

我们想测试一个数 $n$ 是否为素数或对其进行因数分解，但不想检查 $\sqrt{n}$ 以下的每一个整数。

我们不是测试所有数字，而是跳过那些明显能被小素数（如 $2,3,5,7,\ldots$）整除的数。轮式模式编码了这些跳过规则。

#### 它是如何工作的（通俗解释）

1.  选择一组小素数（基础素数），例如 $\{2,3,5\}$。
2.  计算轮子大小为它们的乘积：
    $$
    W = 2 \times 3 \times 5 = 30
    $$
3.  确定与 $W$ 互质的模 $W$ 余数：
    $$
    \{1,7,11,13,17,19,23,29\}
    $$
4.  要测试到 $n$ 的数，只检查形式为
    $$
    kW + r \quad \text{其中 } r \in \text{余数集}.
    $$
    的候选数。
5.  对于每个候选数 $m$，测试到 $\sqrt{m}$ 的整除性。

当使用 $2\times3\times5$ 轮子时，这大约跳过了 73% 的整数。

示例

使用 $\{2,3,5\}$ 轮子找出 $\le 50$ 的素数。

模 30 的轮子余数：
$$
1, 7, 11, 13, 17, 19, 23, 29
$$

候选数：
$$
1, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 49
$$

通过到 $\sqrt{50}$ 的整除性进行筛选：

素数：
$$
7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
$$


#### 微型代码（简易版本）

Python 版本

```python
import math

def wheel_candidates(limit, base_primes=[2, 3, 5]):
    W = 1
    for p in base_primes:
        W *= p
    residues = [r for r in range(1, W) if all(r % p != 0 for p in base_primes)]

    candidates = []
    k = 0
    while k * W <= limit:
        for r in residues:
            num = k * W + r
            if num <= limit:
                candidates.append(num)
        k += 1
    return candidates

def is_prime(n):
    if n < 2: return False
    if n in (2, 3, 5): return True
    for p in [2, 3, 5]:
        if n % p == 0:
            return False
    for candidate in wheel_candidates(int(math.sqrt(n)) + 1):
        if n % candidate == 0:
            return False
    return True

n = int(input("输入 n: "))
print(n, "是", "素数" if is_prime(n) else "合数")
```

C 语言版本

```c
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

bool is_prime(int n) {
    if (n < 2) return false;
    if (n == 2 || n == 3 || n == 5) return true;
    if (n % 2 == 0 || n % 3 == 0 || n % 5 == 0) return false;

    int residues[] = {1, 7, 11, 13, 17, 19, 23, 29};
    int wheel = 30;
    int limit = (int)sqrt(n);

    for (int k = 0; k * wheel <= limit; k++) {
        for (int i = 0; i < 8; i++) {
            int d = k * wheel + residues[i];
            if (d > 1 && d <= limit && n % d == 0)
                return false;
        }
    }
    return true;
}

int main(void) {
    int n;
    printf("输入 n: ");
    scanf("%d", &n);
    printf("%d 是 %s\n", n, is_prime(n) ? "素数" : "合数");
}
```

#### 为什么它很重要

-   通过跳过明显的合数来减少试除检查
-   加速试除法和筛法
-   可在轮式筛法中复用（例如 2×3×5×7 轮子）
-   建立了模运算和素性测试之间的概念联系

常用于：

-   优化的素性测试
-   素数筛法（轮式埃拉托斯特尼筛法）
-   混合因数分解例程

#### 一个温和的证明（为什么它有效）

任何与任意基础素数 $p$ 有公因数的数 $n$，都会出现在模 $W$ 的非互质剩余类中。通过只检查与 $W$ 互质的余数，我们移除了所有这些合数。

因此，每个剩余的候选数都与所有基础素数互质，我们只需要检查是否能被更大的素数整除。

#### 自己动手试试

1.  构建 $\{2, 3\}$、$\{2, 3, 5\}$ 和 $\{2, 3, 5, 7\}$ 的轮子。
2.  计算每个轮子到 100 为止跳过了多少个整数。
3.  与普通试除法比较运行时间。
4.  使用轮子来加速筛法实现。
5.  打印余数并可视化模式。

#### 测试用例

| 基础素数     | 轮子大小 | 余数集                                 | 跳过百分比 |
| ------------ | -------- | -------------------------------------- | ---------- |
| {2, 3}       | 6        | 1, 5                                   | 66%        |
| {2, 3, 5}    | 30       | 1, 7, 11, 13, 17, 19, 23, 29           | 73%        |
| {2, 3, 5, 7} | 210      | 48 个余数                              | 77%        |

| $n$ | 结果       |
| --- | ---------- |
| 7   | 素数       |
| 49  | 合数       |
| 97  | 素数       |
| 121 | 合数       |

#### 复杂度

-   时间复杂度：$O(\frac{\sqrt{n}}{\phi(W)})$，比 $O(\sqrt{n})$ 检查次数少
-   空间复杂度：$O(1)$

轮式筛法就像构建一个只接触有希望的数字的齿轮，一个简单的模运算模式，可以高效地滚动遍历候选数。
### 519 AKS 素性测试

AKS 素性测试是首个确定性的、多项式时间的素性测试算法，不依赖于未经证明的假设。它回答了计算数论中最古老的问题之一：

> 我们能否确定性地在多项式时间内判断一个数是否为素数？

与概率性测试（费马测试、米勒-拉宾测试）不同，AKS 给出确定的答案，没有随机性，也没有误报。

#### 我们要解决什么问题？

我们希望确定性地在多项式时间内判断一个数 $n$ 是素数还是合数，而不依赖于黎曼猜想等假设。

#### 核心思想

一个数 $n$ 是素数，当且仅当它满足以下同余式对所有整数 $a$ 成立：

$$
(x + a)^n \equiv x^n + a \pmod{n}
$$

这个同余式抓住了素数的二项式性质：在素数模下，所有二项式系数 $\binom{n}{k}$（其中 $0 < k < n$）对 $n$ 取模后都为零。

AKS 算法将这个条件精炼成一个可计算的素性测试。

#### 算法（简化版）

步骤 1. 检查 $n$ 是否是完全幂。  
如果存在整数 $a, b > 1$ 使得 $n = a^b$，那么 $n$ 是合数。

步骤 2. 找到最小的整数 $r$，使得

$$
\text{ord}_r(n) > (\log_2 n)^2
$$

其中 $\text{ord}_r(n)$ 是 $n$ 模 $r$ 的乘法阶。

步骤 3. 对每个 $a = 2, 3, \ldots, r$：  
如果 $1 < \gcd(a, n) < n$，那么 $n$ 是合数。

步骤 4. 如果 $n \le r$，那么 $n$ 是素数。

步骤 5. 对所有整数 $a = 1, 2, \ldots, \lfloor \sqrt{\phi(r)} \log n \rfloor$，检查是否满足

$$
(x + a)^n \equiv x^n + a \pmod{(x^r - 1, n)}.
$$

如果任何测试失败，$n$ 是合数。  
否则，$n$ 是素数。

#### 它是如何工作的（通俗解释）

1.  完全幂不可能是素数。
2.  阶条件确保 $r$ 足够大，以区分非素数。
3.  小的最大公约数能捕获平凡的因子。
4.  多项式同余确保 $n$ 在二项式展开下表现得像素数。

这些步骤共同作用，排除了所有合数并确认了所有素数。

#### 示例（概念性）

令 $n = 7$。

1.  $7$ 不是完全幂。
2.  满足 $\text{ord}_r(7) > (\log 7)^2 \approx 5.3$ 的最小 $r$ 是 $r = 5$。
3.  没有找到小的 $\gcd$ 值。
4.  $n > r$。
5.  检查
    $$
    (x + a)^7 \bmod (x^5 - 1, 7) = x^7 + a.
    $$
    所有测试通过，所以 $n$ 是素数。

#### 微型代码（仅作说明）

完整的 AKS 测试在数学上很复杂。下面是一个简化原型，展示了其结构，但未针对大 $n$ 进行优化。

Python 版本

```python
import math
from math import gcd

def is_perfect_power(n):
    for b in range(2, int(math.log2(n)) + 2):
        a = round(n  (1 / b))
        if a  b == n:
            return True
    return False

def multiplicative_order(n, r):
    if gcd(n, r) != 1:
        return 0
    order = 1
    value = n % r
    while value != 1:
        value = (value * n) % r
        order += 1
        if order > r:
            return 0
    return order

def aks_is_prime(n):
    if n < 2:
        return False
    if is_perfect_power(n):
        return False

    logn2 = (math.log2(n))  2
    r = 2
    while True:
        if gcd(n, r) == 1 and multiplicative_order(n, r) > logn2:
            break
        r += 1

    for a in range(2, r + 1):
        g = gcd(a, n)
        if 1 < g < n:
            return False

    if n <= r:
        return True

    limit = int(math.sqrt(r) * math.log2(n))
    for a in range(1, limit + 1):
        # 简化占位符：省略了完整的多项式同余检查
        if pow(a, n, n) != a % n:
            return False
    return True

n = int(input("输入 n: "))
print("素数" if aks_is_prime(n) else "合数")
```

#### 为什么它很重要

-   首个通用的、确定性的、多项式时间的测试
-   计算数论领域的里程碑（Agrawal–Kayal–Saxena，2002年）
-   所有现代素性测试的理论基础
-   证明了素数可以在没有随机性的情况下被识别

#### 一个温和的证明（为什么它有效）

如果 $n$ 是素数，那么根据二项式定理：

$$
(x + a)^n = \sum_{k=0}^{n} \binom{n}{k} x^k a^{n-k} \equiv x^n + a \pmod{n}
$$

因为所有中间的二项式系数都能被 $n$ 整除。  
对于合数，这个恒等式对某些 $a$ 不成立，除非 $n$ 具有特殊的光滑结构（这会被前面的步骤捕获）。

因此，多项式测试对于素性既是必要的也是充分的。

#### 亲自尝试

1.  测试 $n = 37$、$n = 97$、$n = 121$。
2.  与米勒-拉宾测试比较运行时间。
3.  观察对于大 $n$ 的指数级减速。
4.  验证完全幂的排除。
5.  探索小的 $r$ 和阶函数。

#### 测试用例

| $n$ | 结果      | 备注                       |
| --- | --------- | -------------------------- |
| 2   | 素数      | 基本情况                   |
| 7   | 素数      | 通过多项式检查             |
| 37  | 素数      | 正确                       |
| 121 | 合数      | 不满足二项式恒等式         |

#### 复杂度

-   时间：$O((\log n)^6)$（原始版本），改进到 $O((\log n)^3)$
-   空间：$\log n$ 的多项式

AKS 素性测试将素性检查从启发式的艺术转变为一门确定的科学，证明了素数可以在多项式时间内被判定。
### 520 分段筛法

分段筛法是埃拉托斯特尼筛法的一种内存高效变体，旨在生成大范围 $[L, R]$ 内的素数，而无需存储所有直到 $R$ 的数字。  
当 $R$ 非常大（例如 $10^{12}$），但段宽度 $R-L$ 足够小可以放入内存时，这种方法非常理想。

#### 我们要解决什么问题？

我们想要找到范围 $[L, R]$ 内的所有素数，其中 $R$ 可能极大。

一个直到 $R$ 的标准筛法需要 $O(R)$ 的空间，这对于 $R \gg 10^8$ 是不可行的。  
分段筛法通过将范围划分为更小的块，并使用直到 $\sqrt{R}$ 的基素数来标记合数，从而解决了这个问题。

#### 它是如何工作的（通俗解释）

1.  使用标准筛法预计算直到 $\sqrt{R}$ 的基素数。

2.  对于每个段 $[L, R]$：
    *   将所有数字标记为潜在的素数。
    *   对于每个基素数 $p$：
        *   找到 $[L, R]$ 中 $p$ 的第一个倍数：
            $$
            \text{start} = \max\left(p^2,\; \left\lceil \frac{L}{p} \right\rceil \cdot p \right)
            $$
        *   将所有 $p$ 的倍数标记为合数。

3.  剩余未标记的数字就是素数。

如果整个范围太大无法放入内存，则对每个段重复此过程。

示例：查找 [100, 120] 内的素数

1.  计算直到 $\sqrt{120} = 10.9$ 的基素数：  
    $\{2, 3, 5, 7\}$

2.  开始标记：
    *   对于 $p = 2$：标记 100, 102, 104, …
    *   对于 $p = 3$：标记 102, 105, 108, …
    *   对于 $p = 5$：标记 100, 105, 110, 115, 120
    *   对于 $p = 7$：标记 105, 112, 119

未标记的数字：
$$
\boxed{101, 103, 107, 109, 113}
$$
（这些就是 [100, 120] 内的素数）


#### 精简代码（简易版本）

Python 版本

```python
import math

def simple_sieve(limit):
    mark = [True] * (limit + 1)
    mark[0] = mark[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if mark[i]:
            for j in range(i * i, limit + 1, i):
                mark[j] = False
    return [i for i, is_prime in enumerate(mark) if is_prime]

def segmented_sieve(L, R):
    base_primes = simple_sieve(int(math.sqrt(R)) + 1)
    mark = [True] * (R - L + 1)
    for p in base_primes:
        start = max(p * p, ((L + p - 1) // p) * p)
        for j in range(start, R + 1, p):
            mark[j - L] = False
    if L == 1:
        mark[0] = False
    return [L + i for i, is_prime in enumerate(mark) if is_prime]

L, R = map(int, input("输入 L R: ").split())
print("素数:", segmented_sieve(L, R))
```

C 语言版本

```c
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

void simple_sieve(int limit, int primes, int *count) {
    bool *mark = calloc(limit + 1, sizeof(bool));
    for (int i = 2; i <= limit; i++) mark[i] = true;
    for (int i = 2; i * i <= limit; i++)
        if (mark[i])
            for (int j = i * i; j <= limit; j += i)
                mark[j] = false;

    *count = 0;
    for (int i = 2; i <= limit; i++)
        if (mark[i]) (*count)++;

    *primes = malloc(*count * sizeof(int));
    int idx = 0;
    for (int i = 2; i <= limit; i++)
        if (mark[i]) (*primes)[idx++] = i;

    free(mark);
}

void segmented_sieve(long long L, long long R) {
    int limit = sqrt(R) + 1;
    int *primes, count;
    simple_sieve(limit, &primes, &count);

    bool *mark = calloc(R - L + 1, sizeof(bool));
    for (int i = 0; i <= R - L; i++) mark[i] = true;

    for (int i = 0; i < count; i++) {
        int p = primes[i];
        long long start = (long long)p * p;
        if (start < L)
            start = ((L + p - 1) / p) * p;
        for (long long j = start; j <= R; j += p)
            mark[j - L] = false;
    }
    if (L == 1) mark[0] = false;

    for (int i = 0; i <= R - L; i++)
        if (mark[i]) printf("%lld ", L + i);
    printf("\n");

    free(primes);
    free(mark);
}

int main(void) {
    long long L, R;
    printf("输入 L R: ");
    scanf("%lld %lld", &L, &R);
    segmented_sieve(L, R);
}
```

#### 为什么它很重要

-   内存高效：无需完整筛法存储即可处理大范围
-   对于大区间内的素数生成至关重要（竞赛编程、密码学）
-   用于因式分解和概率性素数测试
-   展示了分治筛法思想

#### 一个温和的证明（为什么它有效）

$[L, R]$ 中的每个合数必定有一个 $\le \sqrt{R}$ 的质因数。
通过标记所有这些基素数的倍数，所有合数都被移除，只剩下素数。
每个段重复相同的逻辑，因此每个块上的正确性得以保持。

#### 亲自尝试

1.  生成 $[100, 200]$ 内的素数。
2.  测试 $[10^{12}, 10^{12} + 1000]$。
3.  与完整筛法比较内存使用情况。
4.  为非常大的范围实现动态分段。
5.  在处理每个段时将其打印出来。

#### 测试用例

| 范围 $[L, R]$ | 输出                  |
| -------------- | ----------------------- |
| [10, 30]       | 11, 13, 17, 19, 23, 29  |
| [100, 120]     | 101, 103, 107, 109, 113 |
| [1, 10]        | 2, 3, 5, 7              |

#### 复杂度

-   时间：$O((R - L + 1)\log \log R)$
-   空间：$O(\sqrt{R})$ 用于基素数，$O(R - L)$ 用于段

分段筛法就像淘金一样——一次处理一个托盘（段），以揭示隐藏在广阔数字范围内的素数。


# 第 53 节 组合数学
### 521 阶乘预计算

阶乘预计算是组合数学和模运算中最有用的技术之一。它允许你快速计算诸如 $n!$、二项式系数 $\binom{n}{k}$ 或排列数等值，尤其是在模 $M$ 下，无需每次都从头开始计算。

#### 我们要解决什么问题？

我们经常需要 $n!$（阶乘）或组合数，例如：

$$
\binom{n}{k} = \frac{n!}{k!(n-k)!}
$$

每次直接计算阶乘的时间成本为 $O(n)$，当 $n$ 很大或者我们需要进行多次查询时，这太慢了。通过预先计算一次直到 $n$ 的所有阶乘，我们可以在 O(1) 时间内回答查询。

这在模 $M$ 下工作时尤其重要，因为除法需要模逆元。

#### 它是如何工作的（通俗解释）

我们构建一个数组 `fact[]`，使得：

$$
\text{fact}[i] = (i!) \bmod M
$$

使用递推关系：

$$
\text{fact}[0] = 1, \quad \text{fact}[i] = (i \times \text{fact}[i - 1]) \bmod M
$$

如果我们也需要逆阶乘：

$$
\text{invfact}[n] = (\text{fact}[n])^{-1} \bmod M
$$

那么我们可以反向计算所有逆元：

$$
\text{invfact}[i - 1] = (i \times \text{invfact}[i]) \bmod M
$$

这让我们可以快速计算二项式系数：

$$
\binom{n}{k} = \text{fact}[n] \times \text{invfact}[k] \times \text{invfact}[n - k] \bmod M
$$

#### 示例

设 $M = 10^9 + 7$，$n = 5$：

| $i$ | $i!$ | $i! \bmod M$ |
| ----- | ------ | -------------- |
| 0     | 1      | 1              |
| 1     | 1      | 1              |
| 2     | 2      | 2              |
| 3     | 6      | 6              |
| 4     | 24     | 24             |
| 5     | 120    | 120            |

$$
\binom{5}{2} = \frac{5!}{2! \cdot 3!} = \frac{120}{2 \cdot 6} = 10
$$

使用模逆元，在 $M$ 下同样成立。

#### 精简代码（简易版本）

Python 版本

```python
M = 109 + 7

def precompute_factorials(n):
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = (fact[i - 1] * i) % M
    return fact

def modinv(a, m=M):
    return pow(a, m - 2, m)  # 费马小定理

def precompute_inverses(fact):
    n = len(fact) - 1
    invfact = [1] * (n + 1)
    invfact[n] = modinv(fact[n])
    for i in range(n, 0, -1):
        invfact[i - 1] = (invfact[i] * i) % M
    return invfact

n = 10
fact = precompute_factorials(n)
invfact = precompute_inverses(fact)

def nCr(n, r):
    if r < 0 or r > n: return 0
    return fact[n] * invfact[r] % M * invfact[n - r] % M

print("5C2 =", nCr(5, 2))
```

C 版本

```c
#include <stdio.h>
#define M 1000000007
#define MAXN 1000000

long long fact[MAXN + 1], invfact[MAXN + 1];

long long modpow(long long a, long long e) {
    long long r = 1;
    while (e > 0) {
        if (e & 1) r = r * a % M;
        a = a * a % M;
        e >>= 1;
    }
    return r;
}

void precompute_factorials(int n) {
    fact[0] = 1;
    for (int i = 1; i <= n; i++)
        fact[i] = fact[i - 1] * i % M;

    invfact[n] = modpow(fact[n], M - 2);
    for (int i = n; i >= 1; i--)
        invfact[i - 1] = invfact[i] * i % M;
}

long long nCr(int n, int r) {
    if (r < 0 || r > n) return 0;
    return fact[n] * invfact[r] % M * invfact[n - r] % M;
}

int main(void) {
    precompute_factorials(1000000);
    printf("5C2 = %lld\n", nCr(5, 2));
}
```

#### 为什么它很重要

- 将 $O(n)$ 的重新计算转换为 O(1) 的查询
- 是组合数学、动态规划和模计数的基础
- 支持快速计算：
  * 二项式系数
  * 多重集组合
  * 概率计算
  * 卡特兰数

应用于：

- 组合动态规划
- 概率和期望问题
- 模组合数学

#### 一个温和的证明（为什么它有效）

阶乘以递归方式增长：
$$
n! = n \cdot (n-1)!
$$
因此预计算会存储每一步的结果。
模运算保留了乘法结构：
$$
(ab) \bmod M = ((a \bmod M) \cdot (b \bmod M)) \bmod M
$$
并且当 $M$ 是素数时，根据费马小定理，模逆元存在：
$$
a^{M-1} \equiv 1 \pmod{M} \implies a^{-1} \equiv a^{M-2} \pmod{M}
$$

#### 动手试试

1.  预计算到 $n = 10^6$ 并打印阶乘。
2.  计算 $1000! \bmod 10^9+7$。
3.  验证 $\binom{n}{k} = \binom{n}{n-k}$。
4.  为变化的 $M$ 添加记忆化。
5.  扩展到双阶乘或多重集系数。

#### 测试用例

| $n$ | $k$ | $n! \bmod M$ | $\binom{n}{k} \bmod M$ |
| ----- | ----- | -------------- | ------------------------ |
| 5     | 2     | 120            | 10                       |
| 10    | 3     | 3628800        | 120                      |
| 100   | 50    |,              | 538992043                |

#### 复杂度

- 预计算：$O(n)$
- 查询：$O(1)$
- 空间：$O(n)$

阶乘预计算是你的组合数学查找表，准备一次，即时计算。
### 522 nCr 计算

nCr 计算（二项式系数计算）是组合数学的核心，它计算从一个包含 $n$ 个元素的集合中，不考虑顺序地选择 $r$ 个元素的方法数。
它出现在计数、概率、动态规划和组合恒等式中。

#### 我们要解决什么问题？

我们想要高效地直接计算或模一个大质数 $M$ 计算：

$$
\binom{n}{r} = \frac{n!}{r!(n-r)!}
$$

对于较小的 $n$，可以通过直接乘法和除法来完成。
对于较大的 $n$，由于模运算下除法未定义，我们必须使用模运算和模逆元。

#### 它是如何工作的（通俗解释）

我们可以用几种方式计算 $\binom{n}{r}$：

1.  乘法公式（直接法）：

    $$
    \binom{n}{r} = \prod_{i=1}^{r} \frac{n - r + i}{i}
    $$

    当 $n$ 和 $r$ 适中（< 10^6）时效果很好。

2.  动态规划（帕斯卡三角形）：

    $$
    \binom{n}{r} = \binom{n-1}{r-1} + \binom{n-1}{r}
    $$

    边界条件为 $\binom{n}{0} = 1$，$\binom{n}{n} = 1$。

3.  阶乘预计算（用于模运算）：

    使用预计算的数组：
    $$
    \binom{n}{r} = \text{fact}[n] \cdot \text{invfact}[r] \cdot \text{invfact}[n-r] \bmod M
    $$

#### 示例

计算 $\binom{5}{2}$：

$$
\binom{5}{2} = \frac{5!}{2! \cdot 3!} = \frac{120}{12} = 10
$$

用帕斯卡恒等式验证：

$$
\binom{5}{2} = \binom{4}{1} + \binom{4}{2} = 4 + 6 = 10
$$

#### 简洁代码（简易版本）

1.  乘法公式（无模运算）

```python
def nCr(n, r):
    if r < 0 or r > n:
        return 0
    r = min(r, n - r)
    res = 1
    for i in range(1, r + 1):
        res = res * (n - r + i) // i
    return res

print(nCr(5, 2))  # 10
```

2.  阶乘 + 模逆元

```python
M = 109 + 7

def modpow(a, e, m=M):
    r = 1
    while e > 0:
        if e & 1:
            r = (r * a) % m
        a = (a * a) % m
        e >>= 1
    return r

def nCr_mod(n, r):
    if r < 0 or r > n:
        return 0
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = (fact[i - 1] * i) % M
    invfact = [1] * (n + 1)
    invfact[n] = modpow(fact[n], M - 2)
    for i in range(n, 0, -1):
        invfact[i - 1] = (invfact[i] * i) % M
    return fact[n] * invfact[r] % M * invfact[n - r] % M

print(nCr_mod(5, 2))  # 10
```

3.  帕斯卡三角形（动态规划）

```python
def build_pascal(n):
    C = [[0]*(n+1) for _ in range(n+1)]
    for i in range(n+1):
        C[i][0] = C[i][i] = 1
        for j in range(1, i):
            C[i][j] = C[i-1][j-1] + C[i-1][j]
    return C

pascal = build_pascal(10)
print(pascal[5][2])  # 10
```

#### 为什么它很重要

- 组合数学的基础
- 用于：
    *   二项式展开
    *   概率（例如，超几何分布）
    *   动态规划（例如，计数路径）
    *   数论（卢卡斯定理）
    *   模运算组合学
- 以下内容的核心：
    *   卡特兰数
    *   帕斯卡三角形
    *   容斥原理

#### 一个温和的证明（为什么它有效）

组合解释：

- 要从 $n$ 个物品中选择 $r$ 个，要么包含某个特定物品，要么排除它。

因此：
$$
\binom{n}{r} = \binom{n-1}{r-1} + \binom{n-1}{r}
$$
边界条件：
$$
\binom{n}{0} = \binom{n}{n} = 1
$$

乘法解释：
$$
\frac{n!}{r!(n-r)!} = \frac{n}{1} \times \frac{n-1}{2} \times \cdots \times \frac{n-r+1}{r}
$$

#### 动手试试

1.  手动并使用代码计算 $\binom{10}{3}$。
2.  生成帕斯卡三角形的第 6 行。
3.  验证对称性 $\binom{n}{r} = \binom{n}{n-r}$。
4.  为 $n = 10^6$ 实现模运算版本。
5.  使用 nCr 计算卡特兰数：
    $$
    C_n = \frac{1}{n+1}\binom{2n}{n}
    $$

#### 测试用例

| (n) | (r) | 结果      | 方法       |
| --- | --- | --------- | ---------- |
| 5   | 2   | 10        | 阶乘法     |
| 10  | 3   | 120       | 动态规划   |
| 100 | 50  | 538992043 | 模运算     |

#### 复杂度

| 方法                | 时间复杂度       | 空间复杂度 |
| ------------------- | ---------------- | ---------- |
| 乘法公式            | $O(r)$           | $O(1)$     |
| 动态规划（帕斯卡）  | $O(n^2)$         | $O(n^2)$   |
| 预计算阶乘          | 每次查询 $O(1)$  | $O(n)$     |

nCr 是算法的计数透镜，每一个子集、组合和选择都通过它。
### 523 杨辉三角

杨辉三角是二项式系数的几何排列。
每一项代表 $\binom{n}{r}$，每一行都基于前一行构建。
它不仅仅是一个漂亮的三角形，它是组合数学、二项式展开和动态规划的生命结构。

#### 我们要解决什么问题？

我们希望高效且递归地计算二项式系数，而不使用阶乘或模逆元。

我们使用递归恒等式：

$$
\binom{n}{r}=\binom{n-1}{r-1}+\binom{n-1}{r}
$$

以及基本情况：

$$
\binom{n}{0}=\binom{n}{n}=1
$$

这个递推关系在一个简单的三角形中构建了所有组合，每个数字都是其上方两个数字之和。

#### 它是如何工作的（通俗解释）

从第 0 行开始：`[1]`
每一新行都以 1 开始和结束，中间的每个元素都是其上方两个相邻元素之和。

示例：

| 行号 | 值               |
| --- | ---------------- |
| 0   | 1                |
| 1   | 1 1              |
| 2   | 1 2 1            |
| 3   | 1 3 3 1          |
| 4   | 1 4 6 4 1        |
| 5   | 1 5 10 10 5 1    |

第 $n$ 行、第 $r$ 列的值等于 $\binom{n}{r}$。

#### 示例

从杨辉三角计算 $\binom{5}{2}$：

第 5 行：1 5 10 10 5 1
→ $\binom{5}{2}=10$

验证递推关系：

$$
\binom{5}{2}=\binom{4}{1}+\binom{4}{2}=4+6=10
$$

#### 微型代码（简易版本）

Python 版本

```python
def pascal_triangle(n):
    C = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        C[i][0] = C[i][i] = 1
        for j in range(1, i):
            C[i][j] = C[i - 1][j - 1] + C[i - 1][j]
    return C

C = pascal_triangle(6)
for i in range(6):
    print(C[i][:i + 1])

print("C(5,2) =", C[5][2])
```

C 版本

```c
#include <stdio.h>

void pascal_triangle(int n) {
    int C[n + 1][n + 1];
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= i; j++) {
            if (j == 0 || j == i)
                C[i][j] = 1;
            else
                C[i][j] = C[i - 1][j - 1] + C[i - 1][j];
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }
}

int main(void) {
    pascal_triangle(6);
    return 0;
}
```

#### 为什么它很重要

- 高效地 $O(n^2)$ 构造所有 $\binom{n}{r}$
- 对于较小的 $n$，无需阶乘，无大数溢出问题
- 是以下内容的基础：

  * 二项式展开：
    $(a+b)^n=\sum_{r=0}^n\binom{n}{r}a^{n-r}b^r$
  * 组合动态规划：计数子集、路径、划分
  * 概率计算

还与以下内容相关：

- 斐波那契数列（对角线之和）
- 2 的幂次（行之和）
- 谢尔宾斯基三角形（模 2 模式）

#### 一个温和的证明（为什么它有效）

每一项都计算从 $n$ 个物品中选择 $r$ 个的方法数：
要么包含某个特定元素，要么排除它。

所以：

$$
\binom{n}{r}=\binom{n-1}{r-1}+\binom{n-1}{r}
$$

其中：

- $\binom{n-1}{r-1}$：包含该元素后，从剩余物品中选择 $r-1$ 个的方法数
- $\binom{n-1}{r}$：排除该元素后，从剩余物品中选择 $r$ 个的方法数

这个递推关系一层一层地构建了整个三角形。

#### 自己动手试试

1.  打印杨辉三角的前 10 行。
2.  验证第 $n$ 行的和等于 $2^n$。
3.  使用三角形中的值展开 $(a+b)^5$。
4.  可视化模 2 的模式，得到谢尔宾斯基三角形。
5.  使用对角线构建斐波那契数列。

#### 测试用例

| $n$ | $r$ | $\binom{n}{r}$ | 三角形行                     |
| --- | --- | -------------- | -------------------------- |
| 5   | 2   | 10             | [1, 5, 10, 10, 5, 1]       |
| 6   | 3   | 20             | [1, 6, 15, 20, 15, 6, 1]   |

#### 复杂度

- 时间：$O(n^2)$
- 空间：$O(n^2)$，或者使用行压缩为 $O(n)$

杨辉三角是动态的组合数学，一个不断生长、递归的景观，其中每个数字都记得它的祖先。
### 524 多重集组合

多重集组合计算允许重复的选择，从 $n$ 种类型中选择 $r$ 个元素，其中每种类型可以出现多次。
它是组合、整数划分、星条法（stars and bars）以及袋组合（bag combinations）等问题的组合学基础。

#### 我们要解决什么问题？

我们想要计算从 $n$ 种类型中选择 $r$ 个项目（允许重复）的方式数量。

例如，类型为 ${A,B,C}$ 且 $r=2$ 时，有效的选择是：
$$
{AA,AB,AC,BB,BC,CC}
$$
这是 6 种组合，而不是 $\binom{3}{2}=3$。

所以公式是：

$$
\text{多重集组合}(n,r)=\binom{n+r-1}{r}
$$

这就是“星条法”公式。

#### 它是如何工作的（通俗解释）

想象 $r$ 个相同的星星（项目）被 $(n-1)$ 个分隔符（条）分成 $n$ 组。

示例：$n=3$, $r=4$

我们需要排列 4 个星星和 2 个条：

```
- * | * | *
```

每种排列对应一种组合。

总排列数：

$$
\binom{n+r-1}{r}=\binom{6}{4}=15
$$

所以，从 3 种类型中选取 4 个项目（允许重复）有 15 种方式。

#### 示例

令 $n=3$（类型 A, B, C），$r=2$。
公式：

$$
\binom{3+2-1}{2}=\binom{4}{2}=6
$$

列出所有组合：
AA, AB, AC, BB, BC, CC

好的，与公式匹配。

#### 微型代码（简易版本）

Python 版本

```python
from math import comb

def multiset_combination(n, r):
    return comb(n + r - 1, r)

print("组合数 (n=3, r=2):", multiset_combination(3, 2))
```

模运算版本（使用预计算的阶乘）

```python
M = 109 + 7

def modpow(a, e, m=M):
    r = 1
    while e > 0:
        if e & 1:
            r = (r * a) % m
        a = (a * a) % m
        e >>= 1
    return r

def modinv(a):
    return modpow(a, M - 2)

def nCr_mod(n, r):
    if r < 0 or r > n:
        return 0
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % M
    return fact[n] * modinv(fact[r]) % M * modinv(fact[n - r]) % M

def multiset_combination_mod(n, r):
    return nCr_mod(n + r - 1, r)

print("多重集组合 (n=3, r=2):", multiset_combination_mod(3, 2))
```

#### 为什么它重要

- 为有放回的组合建模
- 出现在以下场景中：

  * 计算多重集 / 袋
  * 整数划分
  * 多项式系数枚举
  * 将相同的球分配到盒子中
- 在多重集上的动态规划、生成函数和概率空间中至关重要

#### 一个温和的证明（为什么它有效）

将 $r$ 个相同的项目表示为星星 `*`
并将 $n-1$ 个分隔符表示为条 `|`。

示例：$n=4$, $r=3$
我们有 $r+(n-1)=6$ 个符号。
选择 $r$ 个星星的位置：

$$
\binom{r+n-1}{r}
$$

每种唯一的排列对应一个多重集。

因此，总组合数 = $\binom{n+r-1}{r}$。

#### 自己动手试试

1.  计算从 {苹果, 香蕉, 樱桃} 中选择 3 个水果的方式数量。
2.  计算 $\text{多重集组合}(5,2)$ 并列出一些例子。
3.  使用帕斯卡三角递推关系构建一个动态规划表：
    $$
    f(n,r)=f(n,r-1)+f(n-1,r)
    $$
4.  对于大的 $n$，使用模运算。
5.  可视化 $n=4, r=3$ 的“星条法”布局。

#### 测试用例

| $n$ | $r$ | $\binom{n+r-1}{r}$ | 结果 |
| --- | --- | ------------------ | ---- |
| 3   | 2   | $\binom{4}{2}=6$   | 6    |
| 4   | 3   | $\binom{6}{3}=20$  | 20   |
| 2   | 5   | $\binom{6}{5}=6$   | 6    |

#### 复杂度

- 时间复杂度：$O(1)$（使用预计算的阶乘时）
- 空间复杂度：$O(n+r)$（用于存储阶乘）

多重集组合开启了超越唯一性的计数，当重复是一种特性而非缺陷时。
### 525 排列生成

排列生成是列出集合中所有可能元素排列的过程，此时顺序至关重要。它是组合数学、递归和搜索算法中最基础的操作之一，为暴力求解器、字典序枚举和回溯框架提供支持。

#### 我们要解决什么问题？

我们希望生成一个大小为 $n$ 的集合的所有排列，即所有可能的顺序。

例如，对于 ${1, 2, 3}$，其排列为：

$$
{1,2,3},{1,3,2},{2,1,3},{2,3,1},{3,1,2},{3,2,1}
$$

总共有 $n!$ 种排列。

#### 它是如何工作的（通俗解释）

有多种策略：

1. 递归回溯法
   * 选择一个元素
   * 排列剩余元素
   * 组合结果

2. 字典序法（下一个排列）
   * 通过寻找字典序中的下一个后继来按排序顺序生成

3. 堆算法
   * 基于交换的迭代生成，时间复杂度 $O(n!)$，空间复杂度 $O(n)$

递归方法示例

要生成 ${1,2,3}$ 的所有排列：

- 固定 1 → 排列 ${2,3}$ → ${1,2,3},{1,3,2}$
- 固定 2 → 排列 ${1,3}$ → ${2,1,3},{2,3,1}$
- 固定 3 → 排列 ${1,2}$ → ${3,1,2},{3,2,1}$

#### 微型代码（简易版本）

Python 版本（递归回溯法）

```python
def permute(arr, path=[]):
    if not arr:
        print(path)
        return
    for i in range(len(arr)):
        permute(arr[:i] + arr[i+1:], path + [arr[i]])

permute([1, 2, 3])
```

Python 版本（使用内置函数）

```python
from itertools import permutations

for p in permutations([1, 2, 3]):
    print(p)
```

C 版本（回溯法）

```c
#include <stdio.h>

void swap(int *a, int *b) {
    int t = *a; *a = *b; *b = t;
}

void permute(int *arr, int l, int r) {
    if (l == r) {
        for (int i = 0; i <= r; i++) printf("%d ", arr[i]);
        printf("\n");
        return;
    }
    for (int i = l; i <= r; i++) {
        swap(&arr[l], &arr[i]);
        permute(arr, l + 1, r);
        swap(&arr[l], &arr[i]); // 回溯
    }
}

int main(void) {
    int arr[] = {1, 2, 3};
    permute(arr, 0, 2);
}
```

#### 为什么它很重要

- 是暴力搜索、回溯和枚举的基础
- 支撑着：
  * 旅行商问题的暴力求解
  * 统计学中的排列检验
  * 人工智能/组合优化中的基于顺序的搜索
- 在以下领域很有用：
  * 组合数学
  * 字符串生成
  * 约束求解

#### 一个温和的证明（为什么它有效）

排列中的每个位置可以容纳一个尚未使用的元素。

递推关系：

$$
P(n)=n\cdot P(n-1)
$$

基本情况 $P(1)=1$，因此 $P(n)=n!$。

递归中的每个分支代表一种排列，总分支数 = $n!$。

#### 动手试试

1.  生成 `[1,2,3]` 的所有排列。
2.  计算 `n=4` 时的排列数量（应为 $24$）。
3.  修改代码以存储排列，而不是打印。
4.  实现字典序的下一个排列方法。
5.  使用排列来测试所有可能的密码顺序。

#### 测试用例

| 输入       | 输出（排列）                                           | 数量 |
| ---------- | ------------------------------------------------------ | ---- |
| [1,2]      | [1,2], [2,1]                                           | 2    |
| [1,2,3]    | [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1] | 6    |
| [‘A',‘B'] | AB, BA                                                 | 2    |

#### 复杂度

- 时间复杂度：$O(n!)$，总排列数
- 空间复杂度：$O(n)$，递归深度

排列生成是穷尽的创造力，探索每一种可能的顺序，每一条可能的路径。
### 526 下一个排列

下一个排列算法用于找出给定序列在字典序中的下一个更大排列。
它是字典序枚举的基础构件，让你能够按排序顺序逐步遍历所有排列，而无需一次性生成全部。

如果不存在更大的排列（序列已按降序排列），则重置为最小排列（升序排列）。

#### 我们要解决什么问题？

给定一个排列（有序序列），找出字典序中下一个更大的排列。

示例：
从 `[1, 2, 3]` 开始，下一个排列是 `[1, 3, 2]`。
从 `[3, 2, 1]` 开始，没有更大的排列，所以我们重置为 `[1, 2, 3]`。

我们希望有一个方法，能够在 O(n) 时间内就地转换序列。

#### 它是如何工作的（通俗解释）

将你的序列想象成一个数字，例如 `[1, 2, 3]` 代表 123。
你想要用相同的数字组成下一个更大的数字。

算法步骤：

1.  **寻找枢轴点**：从右向左扫描，找到第一个索引 `i`，使得
    $a[i] < a[i+1]$
    （这是下一个排列可以增加的点。）

2.  **寻找后继元素**：从右侧开始，找到最小的 $a[j] > a[i]$。

3.  **交换** $a[i]$ 和 $a[j]$。

4.  **反转** 从 $i+1$ 开始的后缀（将降序尾部变为升序）。

如果找不到枢轴点，则反转整个数组（最后一个排列 → 第一个排列）。

示例

从 `[1, 2, 3]` 开始：

1.  枢轴点在 `2` (`a[1]`)，因为 `2 < 3`。
2.  后继元素 = `3`。
3.  交换 → `[1, 3, 2]`。
4.  反转后缀 `[2]` → 保持不变。
    好的，下一个排列 = `[1, 3, 2]`。

从 `[3, 2, 1]` 开始：
没有枢轴点（完全降序）→ 反转为 `[1, 2, 3]`。

#### 精简代码（简易版本）

Python 版本

```python
def next_permutation(arr):
    n = len(arr)
    i = n - 2
    while i >= 0 and arr[i] >= arr[i + 1]:
        i -= 1
    if i == -1:
        arr.reverse()
        return False  # 最后一个排列
    j = n - 1
    while arr[j] <= arr[i]:
        j -= 1
    arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1:] = reversed(arr[i + 1:])
    return True

# 示例
arr = [1, 2, 3]
next_permutation(arr)
print(arr)  # [1, 3, 2]
```

C 语言版本

```c
#include <stdio.h>
#include <stdbool.h>

void reverse(int *a, int l, int r) {
    while (l < r) {
        int t = a[l];
        a[l++] = a[r];
        a[r--] = t;
    }
}

bool next_permutation(int *a, int n) {
    int i = n - 2;
    while (i >= 0 && a[i] >= a[i + 1]) i--;
    if (i < 0) {
        reverse(a, 0, n - 1);
        return false;
    }
    int j = n - 1;
    while (a[j] <= a[i]) j--;
    int t = a[i]; a[i] = a[j]; a[j] = t;
    reverse(a, i + 1, n - 1);
    return true;
}

int main(void) {
    int a[] = {1, 2, 3};
    int n = 3;
    next_permutation(a, n);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
}
```

#### 为什么它很重要

- 字典序枚举的核心工具
- 赋能于：
  * 组合迭代
  * 搜索问题（例如 TSP 暴力破解）
  * 字符串/序列的下一步生成
- 用于 C++ STL (`std::next_permutation`)

#### 一个温和的证明（为什么它有效）

枢轴点 $a[i]$ 之后的后缀是严格递减的，这意味着它是该后缀的最大排列。
为了得到下一个更大的排列：

- 最小化地增加 $a[i]$（与下一个更大的元素 $a[j]$ 交换）
- 然后将后缀重新排列为尽可能小的排列（升序）

从而确保了下一个字典序。

#### 亲自尝试

1.  逐步执行 `[1, 2, 3]` → `[1, 3, 2]` → `[2, 1, 3]` → …
2.  编写循环以按顺序打印所有排列。
3.  尝试字符：`['A', 'B', 'C']`。
4.  测试 `[3, 2, 1]`，应该重置为 `[1, 2, 3]`。
5.  修改代码以生成前一个排列。

#### 测试用例

| 输入      | 输出（下一个） | 备注         |
| --------- | -------------- | ------------ |
| [1,2,3]   | [1,3,2]        | 简单情况     |
| [1,3,2]   | [2,1,3]        | 中间情况     |
| [3,2,1]   | [1,2,3]        | 循环回绕     |
| [1,1,5]   | [1,5,1]        | 包含重复元素 |

#### 复杂度

- 时间复杂度：$O(n)$
- 空间复杂度：$O(1)$

下一个排列是你的字典序步进器，一次小小的交换，一次整洁的反转，一次顺序上的巨大飞跃。
### 527 子集生成

子集生成是列出给定集合所有可能子集的过程，包括空集和全集。
它是组合枚举、幂集构造以及许多回溯和位掩码算法的基石。

#### 我们要解决什么问题？

给定一个包含 $n$ 个元素的集合，生成其所有子集。

例如，对于集合 ${1,2,3}$，其子集（即幂集）为：

$$
\varnothing,{1},{2},{3},{1,2},{1,3},{2,3},{1,2,3}
$$

子集总数 = $2^n$。

#### 它是如何工作的（通俗解释）

有两种经典方法：

1.  递归 / 回溯法
    对每个元素选择包含或不包含。
    以深度优先的方式构建子集。

2.  位掩码枚举法
    使用长度为 $n$ 的二进制掩码来表示子集，
    其中第 $i$ 位表示是否包含第 $i$ 个元素。

以集合 ${1,2,3}$ 为例：

| 掩码 | 二进制 | 子集          |
| ---- | ------ | ------------- |
| 0    | 000    | $\varnothing$ |
| 1    | 001    | ${3}$         |
| 2    | 010    | ${2}$         |
| 3    | 011    | ${2,3}$       |
| 4    | 100    | ${1}$         |
| 5    | 101    | ${1,3}$       |
| 6    | 110    | ${1,2}$       |
| 7    | 111    | ${1,2,3}$     |

#### 简洁代码（简易版本）

Python 版本（递归）

```python
def subsets(arr, path=[], i=0):
    if i == len(arr):
        print(path)
        return
    # 不包含当前元素
    subsets(arr, path, i + 1)
    # 包含当前元素
    subsets(arr, path + [arr[i]], i + 1)

subsets([1, 2, 3])
```

Python 版本（位掩码）

```python
def subsets_bitmask(arr):
    n = len(arr)
    for mask in range(1 << n):
        subset = [arr[i] for i in range(n) if mask & (1 << i)]
        print(subset)

subsets_bitmask([1, 2, 3])
```

C 语言版本（位掩码）

```c
#include <stdio.h>

void subsets(int *arr, int n) {
    int total = 1 << n;
    for (int mask = 0; mask < total; mask++) {
        printf("{ ");
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i))
                printf("%d ", arr[i]);
        }
        printf("}\n");
    }
}

int main(void) {
    int arr[] = {1, 2, 3};
    subsets(arr, 3);
}
```

#### 为何重要

-   功能强大：
    *   在组合问题中枚举所有可能性
    *   子集和、背包问题、位掩码动态规划
    *   搜索与优化
-   是以下内容的基础：
    *   幂集
    *   容斥原理
    *   状态空间探索

每个子集代表组合树中的一个状态、一种选择模式、一个分支。

#### 一个温和的证明（为何有效）

每个元素都可以独立地被包含或排除。
这为每个元素提供了 $2$ 种选择，因此：

$$
\text{子集总数} = 2 \times 2 \times \cdots \times 2 = 2^n
$$

每个二进制掩码唯一地编码了一个子集。
因此，递归法和位掩码法都能恰好访问所有 $2^n$ 个子集各一次。

#### 动手试试

1.  生成 `[1,2,3,4]` 的所有子集。
2.  仅计算大小为 2 的子集数量。
3.  按字典序打印子集。
4.  筛选出和为 5 的子集。
5.  结合动态规划解决子集和问题。

#### 测试用例

| 输入     | 输出（子集）              | 数量 |
| -------- | ------------------------- | ---- |
| [1,2]    | [], [1], [2], [1,2]       | 4    |
| [1,2,3]  | 8 个子集                  | 8    |
| []       | []                        | 1    |

#### 复杂度

-   时间复杂度：$O(2^n \cdot n)$（每个子集的生成需要 $O(n)$ 时间）
-   空间复杂度：$O(n)$（递归栈或临时列表）

子集生成是组合的合唱，每一个是/否的选择都汇入了所有可能性的旋律之中。
### 528 格雷码生成

格雷码按顺序列出所有长度为 $n$ 的 $2^n$ 个二进制字符串，使得相邻的码字恰好相差一位（汉明距离为 $1$）。它们非常适合那些需要微小步进变化以减少错误或避免昂贵重新计算的枚举场景。

#### 我们要解决什么问题？

生成所有 $n$ 位字符串的一个顺序排列
$$
g_0,g_1,\dots,g_{2^n-1}
$$
使得对于每个 $k\ge 1$，汉明距离 $\mathrm{dist}(g_{k-1},g_k)=1$。

#### 它是如何工作的？（通俗解释）

有两种经典的构造方法：

1. 反射与前缀法（递归）

- 基础情况：$G_1=[0,1]$
- 从 $G_{n-1}$ 构建 $G_{n}$：

  * 取 $G_{n-1}$，给每个码字前缀加上 $0$
  * 取 $G_{n-1}$ 的反转顺序，给每个码字前缀加上 $1$
- 将两个列表连接起来

2. 位操作技巧（迭代，索引转格雷码）

- 第 $k$ 个格雷码是
  $$
  g(k)=k\oplus (k!!\gg!1)
  $$
  其中 $\oplus$ 是按位异或，$\gg$ 是右移

两种方法产生的序列本质相同，只是命名可能不同。

#### 示例

对于 $n=3$，反射与前缀法给出：

- 从 $G_1$ 开始：$0,1$
- $G_2$：给 $G_1$ 加前缀 $0$ 得到 $00,01$，给反转的 $G_1$ 加前缀 $1$ 得到 $11,10$
- $G_2=[00,01,11,10]$
- $G_3$：给 $G_2$ 加前缀 $0$ $\to$ $000,001,011,010$
  给反转的 $G_2$ 加前缀 $1$ $\to$ $110,111,101,100$

最终的 $G_3$：
$$
000,001,011,010,110,111,101,100
$$
每一对相邻的码字都恰好相差一位。

#### 简洁代码（简易版本）

Python 版本：反射与前缀法

```python
def gray_reflect(n):
    codes = ["0", "1"]
    if n == 1:
        return codes
    for _ in range(2, n + 1):
        left = ["0" + c for c in codes]
        right = ["1" + c for c in reversed(codes)]
        codes = left + right
    return codes

print(gray_reflect(3))  # ['000','001','011','010','110','111','101','100']
```

Python 版本：位操作技巧 $g(k)=k\oplus(k>>1)$

```python
def gray_bit(n):
    return [k ^ (k >> 1) for k in range(1 << n)]

def to_bits(x, n):
    return format(x, f"0{n}b")

n = 3
codes = [to_bits(v, n) for v in gray_bit(n)]
print(codes)  # ['000','001','011','010','110','111','101','100']
```

C 语言版本：位操作技巧

```c
#include <stdio.h>

unsigned gray(unsigned k) {
    return k ^ (k >> 1);
}

void print_binary(unsigned x, int n) {
    for (int i = n - 1; i >= 0; --i)
        putchar((x & (1u << i)) ? '1' : '0');
}

int main(void) {
    int n = 3;
    unsigned total = 1u << n;
    for (unsigned k = 0; k < total; ++k) {
        unsigned g = gray(k);
        print_binary(g, n);
        putchar('\n');
    }
    return 0;
}
```

#### 为什么它很重要？

- 相邻状态仅相差一位，可以减少硬件编码器和模数转换器中的切换错误和毛刺
- 在格雷码计数器、超立方体上的哈密顿路径以及增量更新成本低的子集搜索中很有用
- 常用于组合生成，以最小化输出之间的变化

#### 一个温和的证明（为什么它有效）

对于反射与前缀法：

- 前半部分是 $0\cdot G_{n-1}$，其中相邻码字已经相差一位
- 后半部分是 $1\cdot \mathrm{rev}(G_{n-1})$，其中相邻码字仍然相差一位
- 边界处的两个码字仅在第一比特位不同，因为我们从 $0\cdot g_0$ 过渡到 $1\cdot g_0$
  通过对 $n$ 进行归纳，相邻码字恰好相差一位。

对于位操作技巧：

- 记 $g(k)=k\oplus(k>>1)$ 和 $g(k+1)=(k+1)\oplus((k+1)>>1)$
- 可以验证 $g(k)$ 和 $g(k+1)$ 仅在 $k$ 递增时发生改变的最低比特位不同
  因此汉明距离为 $1$。

#### 动手尝试

1. 使用两种方法生成 $n=1\ldots 5$ 的格雷码并进行比较。
2. 将格雷码映射回二进制：逆运算是 $b_0=g_0$，对于 $i\ge 1$，$b_i=b_{i-1}\oplus g_i$。
3. 按照格雷码顺序迭代 ${0,\dots,n-1}$ 的所有子集，并通过每一步翻转一个元素来维护一个增量总和。
4. 使用格雷码顺序遍历 $n$ 维立方体的顶点。

#### 测试用例

| $n$ | 预期的序列前缀                     |
| --- | --------------------------------- |
| 1   | $0,1$                             |
| 2   | $00,01,11,10$                     |
| 3   | $000,001,011,010,110,111,101,100$ |

同时验证每一对相邻码字是否恰好相差一位。

#### 复杂度

- 时间：输出所有码字需要 $O(2^n)$
- 空间：每个码字 $O(n)$（或使用位操作技巧时额外空间为 $O(1)$）

格雷码提供了在超立方体上一次只改变一位的遍历方式，非常适合算法和硬件中的平滑过渡。
### 529 卡特兰数动态规划

卡特兰数（Catalan numbers）计数了多种递归结构，从有效的括号字符串到二叉树、多边形三角剖分以及非交叉路径。它们是组合递归、动态规划和上下文无关文法的核心。

#### 我们要解决什么问题？

我们想要计算第 $n$ 个卡特兰数 $C_n$，它计数了以下内容：

- 长度为 $2n$ 的有效括号序列
- 具有 $n$ 个节点的二叉搜索树
- $(n+2)$-边形的三角剖分
- 非交叉划分、Dyck 路径等

递归公式为：

$$
C_0 = 1
$$

$$
C_n = \sum_{i=0}^{n-1} C_i \cdot C_{n-1-i}
$$

或者，使用二项式系数的闭式表达式为：

$$
C_n = \frac{1}{n+1}\binom{2n}{n}
$$

#### 它是如何工作的（通俗解释）

将 $C_n$ 视为将一个结构分割成两个平衡部分的方法数。

示例（有效括号）：
每个序列都以 `(` 开头，并与一个匹配的 `)` 配对，该 `)` 将序列分割成两个较小的有效部分。
如果左边部分有 $C_i$ 种方式，右边部分有 $C_{n-1-i}$ 种方式，则总数为它们的乘积。
对所有可能的分割求和就得到了完整的 $C_n$。

#### 示例

让我们计算前几个：

| $n$ | $C_n$ |
| --- | ----- |
| 0   | 1     |
| 1   | 1     |
| 2   | 2     |
| 3   | 5     |
| 4   | 14    |
| 5   | 42    |

#### 微型代码（简易版本）

Python（动态规划方法）

```python
def catalan(n):
    C = [0] * (n + 1)
    C[0] = 1
    for i in range(1, n + 1):
        for j in range(i):
            C[i] += C[j] * C[i - 1 - j]
    return C[n]

for i in range(6):
    print(f"C({i}) = {catalan(i)}")
```

Python（二项式公式）

```python
from math import comb

def catalan_binom(n):
    return comb(2 * n, n) // (n + 1)

print([catalan_binom(i) for i in range(6)])
```

C 版本（动态规划方法）

```c
#include <stdio.h>

unsigned long catalan(int n) {
    unsigned long C[n + 1];
    C[0] = 1;
    for (int i = 1; i <= n; i++) {
        C[i] = 0;
        for (int j = 0; j < i; j++)
            C[i] += C[j] * C[i - 1 - j];
    }
    return C[n];
}

int main(void) {
    for (int i = 0; i <= 5; i++)
        printf("C(%d) = %lu\n", i, catalan(i));
}
```

#### 为什么它很重要

卡特兰数出现在许多基本的组合和算法上下文中：

- 组合数学：Dyck 路径、格点路径、栈可排序排列
- 动态规划：计数树结构
- 解析理论：有效解析树的数量
- 几何学：凸多边形的三角剖分

它们是平衡递归结构的通用计数。

#### 一个温和的证明（为什么它有效）

每个卡特兰对象围绕一个根节点分割成两个较小的对象：

$$
C_n = \sum_{i=0}^{n-1} C_i \cdot C_{n-1-i}
$$

这个递推关系反映了二叉树的形成，左子树大小为 $i$，右子树大小为 $n-1-i$。

闭式源于二项式恒等式：

$$
C_n = \frac{1}{n+1}\binom{2n}{n}
$$

这是通过使用生成函数求解递推关系得出的。

#### 自己动手试试

1.  使用递推关系手动计算 $C_3$。
2.  验证 $C_4 = 14$。
3.  打印 $n=3$ 的所有有效括号（应该有 5 个）。
4.  比较动态规划和二项式实现。
5.  使用卡特兰数动态规划来计算大小为 $n$ 的二叉搜索树的数量。

#### 测试用例

| $n$ | 期望的 $C_n$ |
| --- | -------------- |
| 0   | 1              |
| 1   | 1              |
| 2   | 2              |
| 3   | 5              |
| 4   | 14             |

#### 复杂度

- 时间复杂度：$O(n^2)$（动态规划），$O(n)$（二项式）
- 空间复杂度：$O(n)$

卡特兰数是许多递归计数问题的核心，捕捉了平衡、结构和对称性的本质。
### 530 斯特林数

斯特林数计算在特定约束下划分或排列元素的方式。
它们连接组合数学、递推关系和生成函数，架起了计数、代数和概率之间的桥梁。

主要有两类：

1. 第一类斯特林数 $c(n, k)$，计算恰好包含 $k$ 个循环的 $n$ 个元素的排列数。
2. 第二类斯特林数 $S(n, k)$，计算将 $n$ 个元素划分成 $k$ 个非空、无标号子集的方式数。

#### 我们要解决什么问题？

我们想要计算：

1. 第一类斯特林数（带符号或不带符号）：

$$
c(n,k) = c(n-1,k-1) + (n-1),c(n-1,k)
$$

2. 第二类斯特林数：

$$
S(n,k) = S(n-1,k-1) + k,S(n-1,k)
$$

基本情况：

$$
c(0,0)=1,\quad S(0,0)=1
$$

$$
c(n,0)=S(n,0)=0 \text{ 当 } n>0
$$

$$
c(0,k)=S(0,k)=0 \text{ 当 } k>0
$$

#### 工作原理（通俗解释）

- 第一类：构建排列

  * 将元素 $n$ 单独作为一个新循环（有 $c(n-1,k-1)$ 种方式）
  * 或者将 $n$ 插入到 $(n-1)$ 个现有循环之一（有 $(n-1),c(n-1,k)$ 种方式）

- 第二类：构建划分

  * 将元素 $n$ 单独放置（新子集）→ $S(n-1,k-1)$
  * 或者将 $n$ 放入 $k$ 个现有子集之一 → $k,S(n-1,k)$

每个递推关系都反映了“添加新项”的逻辑：创建新组或加入旧组。

#### 示例

对于第二类斯特林数 $S(3,k)$：

| $n$ | $k$ | $S(n,k)$ |
| --- | --- | -------- |
| 3   | 1   | 1        |
| 3   | 2   | 3        |
| 3   | 3   | 1        |

所以 $S(3,2)=3$ 表示将 {1,2,3} 分成 2 个非空集合有 3 种方式：

- {1,2},{3}
- {1,3},{2}
- {2,3},{1}

#### 简单代码（简易版本）

Python（第二类斯特林数）

```python
def stirling2(n, k):
    if n == k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    dp = [[0]*(k+1) for _ in range(n+1)]
    dp[0][0] = 1
    for i in range(1, n+1):
        for j in range(1, k+1):
            dp[i][j] = dp[i-1][j-1] + j * dp[i-1][j]
    return dp[n][k]

for k in range(1,4):
    print(f"S(3,{k}) =", stirling2(3,k))
```

Python（第一类斯特林数）

```python
def stirling1(n, k):
    if n == k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    dp = [[0]*(k+1) for _ in range(n+1)]
    dp[0][0] = 1
    for i in range(1, n+1):
        for j in range(1, k+1):
            dp[i][j] = dp[i-1][j-1] + (i-1) * dp[i-1][j]
    return dp[n][k]
```

#### 重要性

斯特林数统一了组合数学、代数和分析：

- $S(n,k)$：集合划分数
- $c(n,k)$：排列循环数
- 出现在：

  * 贝尔数：$B_n = \sum_{k=0}^n S(n,k)$
  * 阶乘展开式
  * 概率论和统计学中的矩
  * 多项式基（下降/上升阶乘）

它们是在不同基中表示阶乘或幂时的系数：
$$
x^n = \sum_k S(n,k),(x)_k, \quad (x)_n = \sum_k c(n,k),x^k
$$

#### 一个温和的证明（为什么有效）

对于 $S(n,k)$，考虑向一个大小为 $n-1$ 的划分中添加一个元素：

- 创建新子集（不选择任何现有子集 → $S(n-1,k-1)$）
- 加入现有子集（从 $k$ 个中选择一个 → $kS(n-1,k)$）

将它们相加 → 递推关系得证。

对于 $c(n,k)$，每个新元素：

- 开始一个新循环（$c(n-1,k-1)$）
- 加入一个现有循环（$(n-1)c(n-1,k)$）

#### 动手尝试

1. 手动计算 $S(4,2)$。
2. 验证 $S(4,2)=7$ 和 $S(4,3)=6$。
3. 打印完整的斯特林三角形。
4. 比较 $S(n,k)$ 与二项式系数。
5. 推导贝尔数：$B_n=\sum_k S(n,k)$。

#### 测试用例

| $n$ | $k$ | $S(n,k)$ | $c(n,k)$ |
| --- | --- | -------- | -------- |
| 3   | 1   | 1        | 2        |
| 3   | 2   | 3        | 3        |
| 3   | 3   | 1        | 1        |
| 4   | 2   | 7        | 11       |

#### 复杂度

- 时间：$O(nk)$
- 空间：$O(nk)$

斯特林数构成了组合计数的语法，表达了划分、循环以及多项式世界之间的转换。

# 第 54 节 概率与随机算法
### 531 蒙特卡洛模拟

蒙特卡洛模拟是一种数值方法，它利用随机抽样来逼近确定性或概率性问题的解。当精确公式难以计算或无法计算时，蒙特卡洛方法通过模拟大量随机实验并对结果取平均值来估计结果。

#### 我们要解决什么问题？

我们希望通过随机抽样来估计一个量（如面积、积分或概率）。

核心原理：

$$
\text{期望值} \approx \frac{1}{N}\sum_{i=1}^{N} f(X_i)
$$

其中 $X_i$ 是随机样本，$f(X_i)$ 衡量每次试验的贡献。

当 $N \to \infty$ 时，根据大数定律，平均值收敛于真实值。

#### 示例：估算 π

想象一个单位正方形，其内包含一个半径为 1 的四分之一圆。落在圆内的点数与总点数的比值近似于 $\pi/4$。

$$
\frac{\text{圆内点数}}{\text{总点数}} \approx \frac{\pi}{4}
$$

因此：

$$
\pi \approx 4 \times \frac{\text{圆内点数}}{\text{总点数}}
$$

#### 它是如何工作的（通俗解释）

1.  定义一个随机实验（例如，在 $[0,1]\times[0,1]$ 区域内采样 $(x,y)$）
2.  检查样本是否满足某个条件（例如，$x^2 + y^2 \le 1$）
3.  重复多次
4.  使用比值（命中次数 / 总次数）来估计所需的概率或值

采样次数越多 → 误差越低（方差 $\propto 1/\sqrt{N}$）

#### 微型代码（简易版本）

Python 版本（估算 π）

```python
import random

def monte_carlo_pi(samples=1000000):
    inside = 0
    for _ in range(samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / samples

print("估算的 π =", monte_carlo_pi())
```

C 语言版本

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int samples = 1000000, inside = 0;
    for (int i = 0; i < samples; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        if (x*x + y*y <= 1.0) inside++;
    }
    double pi = 4.0 * inside / samples;
    printf("估算的 π = %f\n", pi);
    return 0;
}
```

#### 为何重要

蒙特卡洛方法对于以下领域至关重要：

-   高维积分
-   概率建模
-   优化与模拟
-   金融建模（例如，期权定价）
-   物理与工程（例如，粒子输运、统计力学）

它以牺牲部分精度为代价换取通用性，在确定性方法失效时大显身手。

#### 一个温和的证明（为何有效）

令 $X_1, X_2, \dots, X_N$ 为代表结果的独立同分布随机变量。根据大数定律：

$$
\frac{1}{N}\sum_{i=1}^{N} X_i \to \mathbb{E}[X]
$$

因此样本均值收敛于期望值，提供了一个一致估计量。中心极限定理确保了误差以 $1/\sqrt{N}$ 的速度减小。

#### 动手尝试

1.  使用不同的样本数量估算 $\pi$。
2.  使用蒙特卡洛方法计算 $\int_0^1 x^2,dx$。
3.  估算两个骰子点数之和 ≥ 10 的概率。
4.  模拟抛硬币，并与精确概率进行比较。
5.  测量随着 $N$ 增长时的收敛情况（误差 vs 样本数）。

#### 测试用例

| 样本数    | 估算的 π | 预期误差 |
| --------- | -------- | -------- |
| 1000      | ~3.1     | ±0.05    |
| 10,000    | ~3.14    | ±0.02    |
| 1,000,000 | ~3.1415  | ±0.001   |

#### 复杂度

-   时间：$O(N)$
-   空间：$O(1)$

蒙特卡洛模拟是将统计学作为计算手段，通过重复和平均，让随机性揭示结构。
### 532 拉斯维加斯算法

拉斯维加斯算法总是返回正确答案，但其运行时间是随机的，可能因运气好坏而耗时更长或更短。
与蒙特卡洛方法（以精度换取速度）不同，拉斯维加斯算法从不牺牲正确性，只牺牲时间。

#### 我们要解决什么问题？

我们需要随机化算法，它们保证结果正确，但其执行时间取决于随机事件。

换句话说：

- 输出：总是正确的
- 时间：随机变量

我们希望设计这样的算法：随机性有助于避免最坏情况的行为或简化逻辑，同时不破坏正确性。

#### 它是如何工作的（通俗解释）

拉斯维加斯算法使用随机性来指导搜索、选择枢轴或采样数据，但在返回结果前会验证正确性。

如果随机选择不佳，它会重试。

示例：随机化快速排序

快速排序选择一个随机枢轴，围绕它分割数据，然后递归排序。
随机枢轴选择确保了期望的 $O(n \log n)$ 时间复杂度，尽管最坏情况的 $O(n^2)$ 仍然存在。

但输出总是有序的，正确性从不依赖于运气。

另一个示例：随机化快速选择

使用随机枢轴查找第 $k$ 小的元素。
如果枢轴选得不好，运行时间会变差，但结果仍然是正确的。

#### 微型代码（简易版本）

Python（随机化快速排序）

```python
import random

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)

print(quicksort([3, 1, 4, 1, 5, 9, 2, 6]))
```

Python（快速选择）

```python
import random

def quickselect(arr, k):
    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    if k < len(left):
        return quickselect(left, k)
    elif k < len(left) + len(mid):
        return pivot
    else:
        return quickselect(right, k - len(left) - len(mid))

print(quickselect([7, 10, 4, 3, 20, 15], 2))  # 第 3 小的元素
```

#### 为什么它很重要

拉斯维加斯算法结合了正确性的确定性和期望的效率：

- 排序和选择（快速排序，快速选择）
- 计算几何（随机化增量算法）
- 图算法（随机化最小生成树，平面分割）
- 数据结构（跳表，哈希表）

它们通常能简化设计并避免对抗性输入。

#### 一个温和的证明（为什么它有效）

设 $T$ 为运行时间随机变量。
期望时间：

$$
\mathbb{E}[T] = \sum_{t} t \cdot P(T = t)
$$

当"好的"随机事件发生时（例如，平衡的枢轴），算法终止。
通过限制每一步的期望成本，我们可以证明 $\mathbb{E}[T] = O(f(n))$。

正确性通过在随机选择后进行确定性验证来保证。

#### 自己动手试试

1.  比较使用和不使用随机枢轴的快速排序。
2.  测量多次运行的运行时间，观察变化。
3.  统计小数组的递归调用分布。
4.  在跳表插入中使用随机化。
5.  编写一个基于重试的算法（例如，随机哈希探测）。

#### 测试用例

| 输入                      | 算法        | 输出            | 正确？ | 运行时间 |
| ------------------------- | ----------- | --------------- | ------ | -------- |
| [3,1,4,1,5,9]             | QuickSort   | [1,1,3,4,5,9]   | 是     | 变化     |
| [7,10,4,3,20,15], k=2     | QuickSelect | 7               | 是     | 变化     |

#### 复杂度

- 期望时间：$O(f(n))$（通常是 $O(n)$ 或 $O(n\log n)$）
- 最坏时间：仍然可能，但罕见
- 空间：取决于递归或重试次数

拉斯维加斯算法赌的是时间，而不是真相，它们总能找到正确答案，只是每次运行的路径不同。
### 533 蓄水池抽样

蓄水池抽样是一种巧妙的算法，用于从一个未知或非常大的数据流中选取均匀随机样本，仅使用 $O(k)$ 的空间。它是流式算法的基石，适用于无法存储所有数据但仍希望进行公平随机选择的情况。

#### 我们要解决什么问题？

给定一个包含 $n$ 个项目的流（可能非常大或无限），均匀随机地选择 $k$ 个项目，即每个项目被选中的概率相等，均为 $\frac{k}{n}$，且事先不知道 $n$ 的值。

我们希望逐个处理项目，只保留一个包含 $k$ 个样本的蓄水池。

#### 它是如何工作的？（通俗解释）

核心思想是增量公平性：

1.  **初始化**：用前 $k$ 个元素填充蓄水池。
2.  **处理流**：对于第 $k$ 个之后的每个项目 $i$（索引从 1 开始），
    *   生成一个随机整数 $j \in [1, i]$
    *   如果 $j \le k$，则用项目 $i$ 替换 `reservoir[$j$]`

这确保了每个元素在第 $i$ 步被选中的概率 = $\frac{k}{i}$，最终整体概率为 $\frac{k}{n}$。

#### 示例 (k = 2)

流: [A, B, C, D]

1.  蓄水池 = [A, B]
2.  $i=3$ → 随机 $j\in[1,3]$ → 假设 $j=2$ → 替换 B → [A, C]
3.  $i=4$ → $j\in[1,4]$ → 假设 $j=4$ → 跳过（因为 >2）→ [A, C]

处理完整个流后，每一对组合被选中的概率相等。

#### 均匀性证明

在第 $i$ 步：

*   被选中的概率：$\frac{k}{i}$
*   在后续替换中幸存下来的概率：
    $$
    \prod_{t=i+1}^{n}\left(1 - \frac{1}{t}\right)=\frac{i}{n}
    $$
    因此最终概率 = $\frac{k}{i} \cdot \frac{i}{n} = \frac{k}{n}$，对所有元素均相等。

#### 精简代码（简易版本）

Python 版本 (k = 1)

```python
import random

def reservoir_sample(stream):
    result = None
    for i, item in enumerate(stream, start=1):
        if random.randint(1, i) == 1:
            result = item
    return result

data = [10, 20, 30, 40, 50]
print("随机选取:", reservoir_sample(data))
```

Python 版本 (k = 3)

```python
import random

def reservoir_sample_k(stream, k):
    reservoir = []
    for i, item in enumerate(stream, start=1):
        if i <= k:
            reservoir.append(item)
        else:
            j = random.randint(1, i)
            if j <= k:
                reservoir[j - 1] = item
    return reservoir

data = range(1, 11)
print(reservoir_sample_k(data, 3))
```

C 语言版本 (k = 1)

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int stream[] = {10, 20, 30, 40, 50};
    int n = 5;
    int result = stream[0];
    for (int i = 1; i < n; i++) {
        int j = rand() % (i + 1);
        if (j == 0)
            result = stream[i];
    }
    printf("随机选取: %d\n", result);
}
```

#### 为什么它很重要？

蓄水池抽样对于以下情况至关重要：

*   **大数据**：当 $n$ 太大而无法放入内存时
*   **流式 API、日志、遥测数据**
*   **数据库和分布式系统中的随机抽样**
*   **机器学习**：随机小批量、无偏选择

它可以在事先不知道 $n$ 的情况下，给出精确的均匀概率。

#### 一个温和的证明（为什么它有效）

对于任意元素 $x_i$：

*   在出现时被选中的概率 = $\frac{k}{i}$
*   之后不被替换的概率 = $\frac{i}{i+1}\cdot \frac{i+1}{i+2}\cdots\frac{n-1}{n}=\frac{i}{n}$
*   综合概率：$\frac{k}{i} \cdot \frac{i}{n} = \frac{k}{n}$
    对所有 $i$ 都均匀。

#### 亲自尝试

1.  用递增的 $n$ 运行并跟踪频率。
2.  通过 10,000 次试验验证近似均匀性。
3.  用 $k>1$ 进行测试。
4.  应用于来自文件或 API 的数据流。
5.  修改以实现加权抽样。

#### 测试用例

| 流          | k  | 可能的蓄水池       | 概率       |
| ----------- | -- | ------------------ | ---------- |
| [A, B, C]   | 1  | A, B, 或 C         | 各 1/3     |
| [1, 2, 3, 4] | 2  | 任意 2 元素组合    | 均等       |

#### 复杂度

*   时间：$O(n)$（一次遍历）
*   空间：$O(k)$

蓄水池抽样是在约束下的随机性，是从无尽流中进行公平选择，一次一个元素。
### 534 随机快速排序

随机快速排序是经典的“分而治之”排序算法，但加入了随机性的巧妙设计。它不选择固定的枢轴（如第一个或最后一个元素），而是随机选取一个枢轴，从而确保无论输入顺序如何，期望运行时间都能保持在 $O(n \log n)$。

这种简单的随机化处理，优雅地规避了最坏情况。

#### 我们要解决什么问题？

我们需要一种快速的原地排序算法，以避免病态输入。如果枢轴选择不当，标准的快速排序在已排序数据上可能会退化到 $O(n^2)$ 的时间复杂度。通过均匀随机地选择枢轴，我们保证了期望的平衡性和期望的 $O(n \log n)$ 性能。

#### 它是如何工作的（通俗解释）

快速排序围绕一个枢轴对数组进行划分——较小的元素放在左边，较大的放在右边——然后递归地对两边进行排序。

随机化确保枢轴在平均情况下接近中位数，从而保持递归树的平衡。

步骤：

1.  选择一个随机的枢轴索引 `p`。
2.  划分数组，使得：
    *   左边：元素 < 枢轴
    *   右边：元素 > 枢轴
3.  递归地对两个分区进行排序。

当随机化是公平的时，每个枢轴大致将数组均匀分割，使得递归树高度 $\approx \log n$，总工作量 $O(n \log n)$。

#### 示例

对 `[3, 6, 2, 1, 4]` 排序

1.  随机选择枢轴 → 假设是 `3`
2.  划分 → `[2,1] [3] [6,4]`
3.  递归处理 `[2,1]` 和 `[6,4]`
4.  继续直到排序完成：`[1,2,3,4,6]`

不同的随机种子 → 不同的递归路径，相同的最终结果。

#### 精简代码（简易版本）

Python 版本

```python
import random

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)

arr = [3, 6, 2, 1, 4]
print(quicksort(arr))
```

C 语言版本

```c
#include <stdio.h>
#include <stdlib.h>

void swap(int *a, int *b) {
    int t = *a; *a = *b; *b = t;
}

int partition(int a[], int low, int high) {
    int pivot_idx = low + rand() % (high - low + 1);
    swap(&a[pivot_idx], &a[high]);
    int pivot = a[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (a[j] < pivot) swap(&a[++i], &a[j]);
    }
    swap(&a[i + 1], &a[high]);
    return i + 1;
}

void quicksort(int a[], int low, int high) {
    if (low < high) {
        int pi = partition(a, low, high);
        quicksort(a, low, pi - 1);
        quicksort(a, pi + 1, high);
    }
}

int main(void) {
    int a[] = {3, 6, 2, 1, 4};
    int n = 5;
    quicksort(a, 0, n - 1);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
}
```

#### 为什么它很重要

随机快速排序具有以下特点：

-   实际运行速度快（常数因子低，缓存友好）
-   原地排序（无需额外数组）
-   期望时间复杂度为 $O(n \log n)$，与输入无关
-   对对抗性输入和预排序陷阱具有免疫力

广泛应用于：

-   标准库（例如 Python 的 `sort()` 使用了包含随机枢轴的混合算法）
-   数据库系统和数据处理管道
-   教授分治算法和随机化算法

#### 一个温和的证明（为什么它有效）

令 $T(n)$ 为期望成本。每个划分步骤需要 $O(n)$ 次比较。枢轴将数组分割为大小为 $i$ 和 $n-i-1$ 的两部分，每个 $i$ 出现的概率相等。

$$
\mathbb{E}[T(n)] = \frac{1}{n}\sum_{i=0}^{n-1} (\mathbb{E}[T(i)] + \mathbb{E}[T(n-i-1)]) + O(n)
$$

求解可得 $\mathbb{E}[T(n)] = O(n \log n)$。

随着随机选择被平均化，方差会减小。

#### 动手尝试

1.  多次运行算法，记录枢轴序列。
2.  对已排序列表进行排序，期望不会出现速度减慢。
3.  与归并排序比较运行时间。
4.  可视化递归树的深度。
5.  实现一个处理重复元素的三路划分版本。

#### 测试用例

| 输入          | 输出          | 备注         |
| ------------- | ------------- | ------------ |
| [3,6,2,1,4]   | [1,2,3,4,6]   | 基础情况     |
| [1,2,3,4,5]   | [1,2,3,4,5]   | 预排序       |
| [5,4,3,2,1]   | [1,2,3,4,5]   | 逆序         |
| [2,2,2,2]     | [2,2,2,2]     | 重复元素     |

#### 复杂度

-   期望时间：$O(n \log n)$
-   最坏情况：$O(n^2)$（罕见）
-   空间：$O(\log n)$ 递归栈空间

随机快速排序将运气转化为平衡，每一次随机选择都是一重保障，每一个枢轴都是一次达成和谐的机会。
### 535 随机化快速选择

随机化快速选择是一种分治算法，用于在未排序数组中查找第 k 小的元素，其期望时间复杂度为线性。它是快速排序的"选择双胞胎"，使用了相同的划分思想，但在每一步只探索数组的一侧。

#### 我们要解决什么问题？

给定一个未排序数组 `arr` 和一个秩 `k`（从 1 开始计数），我们想要找到如果数组被排序后，会出现在第 `k` 个位置上的元素——而不需要完全排序整个数组。

示例：
在 `[7, 10, 4, 3, 20, 15]` 中，
第 3 小的元素是 `7`。

我们希望以期望 O(n) 的时间复杂度找到它，比排序（$O(n \log n)$）更快。

#### 它是如何工作的？（通俗解释）

1.  从数组中随机选择一个枢轴。
2.  将元素划分为三组：
    *   左侧：小于枢轴
    *   中间：等于枢轴
    *   右侧：大于枢轴
3.  将 `k` 与各组大小进行比较：
    *   如果 `k` ≤ 左侧长度：在左侧递归
    *   否则如果 `k` ≤ 左侧长度 + 中间长度：枢轴就是答案
    *   否则：在右侧递归，并调整秩 `k`

通过只关注包含第 k 个元素的那一侧，我们每次都将问题规模大致减半。

#### 示例

在 `[7, 10, 4, 3, 20, 15]` 中查找第 3 小的元素

1.  随机枢轴 = `10`
    *   左侧 = `[7, 4, 3]`, 中间 = `[10]`, 右侧 = `[20, 15]`
2.  左侧长度 = 3
    由于 `k=3` ≤ 3，递归进入 `[7, 4, 3]`
3.  随机枢轴 = `4`
    *   左侧 = `[3]`, 中间 = `[4]`, 右侧 = `[7]`
4.  左侧长度=1，中间长度=1
    `k=3` > 1+1 → 递归进入 `[7]`，且 `k=1`
    → 答案 = `7`

#### 精简代码（简易版本）

Python 版本

```python
import random

def quickselect(arr, k):
    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    if k <= len(left):
        return quickselect(left, k)
    elif k <= len(left) + len(mid):
        return pivot
    else:
        return quickselect(right, k - len(left) - len(mid))

data = [7, 10, 4, 3, 20, 15]
print(quickselect(data, 3))  # 第 3 小的元素 → 7
```

C 版本（原地划分）

```c
#include <stdio.h>
#include <stdlib.h>

void swap(int *a, int *b) { int t = *a; *a = *b; *b = t; }

int partition(int a[], int l, int r, int pivot_idx) {
    int pivot = a[pivot_idx];
    swap(&a[pivot_idx], &a[r]);
    int store = l;
    for (int i = l; i < r; i++)
        if (a[i] < pivot) swap(&a[i], &a[store++]);
    swap(&a[store], &a[r]);
    return store;
}

int quickselect(int a[], int l, int r, int k) {
    if (l == r) return a[l];
    int pivot_idx = l + rand() % (r - l + 1);
    int idx = partition(a, l, r, pivot_idx);
    int rank = idx - l + 1;
    if (k == rank) return a[idx];
    if (k < rank) return quickselect(a, l, idx - 1, k);
    return quickselect(a, idx + 1, r, k - rank);
}

int main(void) {
    int arr[] = {7, 10, 4, 3, 20, 15};
    int n = 6, k = 3;
    printf("%d\n", quickselect(arr, 0, n - 1, k));
}
```

#### 为什么它很重要

-   期望 O(n) 时间，O(1) 空间
-   是中位数选择和顺序统计的基础
-   应用于：
    *   中位数的中位数
    *   随机化算法
    *   快速排序优化
    *   数据摘要和采样

实际上，快速选择是许多库（如 `numpy.partition`）高效查找中位数和百分位数的方式。

#### 一个温和的证明（为什么它有效）

每一步以 $O(n)$ 的时间划分数组，并在其中一侧递归。
期望的划分比例约为 1:2 → 递归式：

$$
T(n) = T\left(\frac{n}{2}\right) + O(n)
$$

求解得到 $T(n) = O(n)$。
最坏情况（$O(n^2)$）仅当枢轴反复处于极端值时发生，其概率呈指数级减小。

#### 亲自尝试

1.  查找一个大型随机数组的中位数（`k = n//2`）。
2.  比较快速选择与排序的时间。
3.  运行多次试验；注意递归深度的变化。
4.  修改以查找第 k 大的元素。
5.  实现确定性枢轴选择（中位数的中位数）。

#### 测试用例

| 输入               | k | 输出 | 备注           |
| ------------------ | - | ---- | -------------- |
| [7,10,4,3,20,15]   | 3 | 7    | 第 3 小的元素  |
| [5,4,3,2,1]        | 1 | 1    | 最小的元素     |
| [2,2,2,2]          | 2 | 2    | 包含重复元素   |
| [10]               | 1 | 10   | 单个元素       |

#### 复杂度

-   期望时间复杂度：$O(n)$
-   最坏情况：$O(n^2)$（罕见）
-   空间复杂度：$O(1)$（原地）

随机化快速选择是通过概率实现的精确选择，是通往第 k 个元素的一条直接、优雅的路径，由概率驱动。
### 536 生日悖论模拟

生日悖论是一个著名的概率谜题，展示了随机样本中碰撞发生的速度之快。
令人惊讶的是，即使只有 23 个人，两个人共享同一个生日的概率也超过 50%，尽管一年有 365 天。

模拟有助于揭示直觉为何常常失效。

#### 我们要解决什么问题？

我们想要估计在一个大小为 $n$ 的群体中，至少有两个人生日相同的概率。

有两种方法：

1.  解析公式（精确）
2.  蒙特卡洛模拟（经验）

#### 解析概率

首先计算 $P(\text{无碰撞})$：

-   第一个人：365 种选择
-   第二个人：364 种
-   第三个人：363 种
-   ...
-   第 $n$ 个人：$(365 - n + 1)$ 种选择

因此：

$$
P(\text{无匹配}) = \frac{365}{365} \times \frac{364}{365} \times \cdots \times \frac{365-n+1}{365}
$$

那么：

$$
P(\text{碰撞}) = 1 - P(\text{无匹配})
$$

对于 $n=23$，$P(\text{碰撞}) \approx 0.507$

#### 工作原理（通俗解释）

每增加一个人，匹配的概率就会增加，这种匹配不是与所有人，而是与之前任何一个人。
配对的数量增长很快：

$$
\text{配对数} = \binom{n}{2}
$$

这种比较次数的二次增长解释了碰撞为何发生得如此之快。

蒙特卡洛模拟只是多次运行实验，并统计出现重复的频率。

#### 示例

尝试 $n = 23$ 人，10,000 次试验：

-   约 50% 的试验中至少有一个共享生日
-   约 50% 的试验中没有

随机性会收敛到解析结果。

#### 微型代码（简易版本）

Python 版本

```python
import random

def birthday_collision_prob(n, trials=10000):
    count = 0
    for _ in range(trials):
        birthdays = [random.randint(1, 365) for _ in range(n)]
        if len(birthdays) != len(set(birthdays)):
            count += 1
    return count / trials

for n in [10, 20, 23, 30, 40]:
    print(f"n={n}, P(collision)≈{birthday_collision_prob(n):.3f}")
```

C 版本

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

bool has_collision(int n) {
    bool seen[366] = {false};
    for (int i = 0; i < n; i++) {
        int b = rand() % 365 + 1;
        if (seen[b]) return true;
        seen[b] = true;
    }
    return false;
}

double birthday_collision_prob(int n, int trials) {
    int count = 0;
    for (int i = 0; i < trials; i++)
        if (has_collision(n)) count++;
    return (double)count / trials;
}

int main(void) {
    srand(time(NULL));
    int ns[] = {10, 20, 23, 30, 40};
    for (int i = 0; i < 5; i++) {
        int n = ns[i];
        printf("n=%d, P≈%.3f\n", n, birthday_collision_prob(n, 10000));
    }
}
```

#### 为何重要

生日悖论说明了碰撞概率，这对于以下方面至关重要：

-   哈希函数（碰撞分析）
-   密码学（生日攻击）
-   随机 ID 生成（UUID、指纹）
-   模拟和概率推理

它表明，关于随机性的直觉常常低估了碰撞的可能性。

#### 一个温和的证明（为何有效）

对于小的 $n$：

$$
P(\text{无匹配}) = \prod_{i=0}^{n-1} \frac{365 - i}{365}
$$

展开 $n=23$ 的情况：

$$
P(\text{无匹配}) \approx 0.493 \
P(\text{匹配}) = 1 - 0.493 = 0.507
$$

因此，只需 23 个人，两个人共享生日的可能性就大于不共享的可能性。

#### 亲自尝试

1.  绘制 $P(\text{碰撞})$ 与 $n$ 的关系图。
2.  找出概率 > 0.9 的最小 $n$。
3.  尝试不同的“年份长度”（例如 500 或 1000）。
4.  测试非均匀生日分布。
5.  模拟哈希桶情况（$m=2^{16}$，$n=500$）。

#### 测试用例

| n  | 期望 P(collision) | 模拟（近似） |
| -- | ----------------- | ------------ |
| 10 | 0.117             | 0.12         |
| 20 | 0.411             | 0.41         |
| 23 | 0.507             | 0.50         |
| 30 | 0.706             | 0.71         |
| 40 | 0.891             | 0.89         |

#### 复杂度

-   时间：$O(n \cdot \text{trials})$
-   空间：$O(n)$

生日悖论是一个碰撞透镜，一个窗口，让我们看到随机性累积的速度如何快于直觉的预期。
### 537 随机哈希

随机哈希利用随机性来最小化冲突，并将键均匀地分布到各个桶中。
它是哈希表、布隆过滤器和概率数据结构中的核心思想，在这些场景中，公平性和独立性比确定性更重要。

#### 我们要解决什么问题？

我们需要将任意键映射到桶（例如哈希表中的槽位），并且要均匀分布，即使输入键遵循非均匀或对抗性模式时也是如此。

确定性哈希函数可能导致键聚集或泄露结构信息。
在哈希函数中加入随机性，可以确保每个键的行为都像一个随机变量，因此冲突变得罕见，并且仅在期望值上可预测。

#### 它是如何工作的（通俗解释）

随机哈希函数是从一个函数*族* $H = {h_1, h_2, \ldots, h_m}$ 中选出的，在使用前，我们在运行时随机确定一个。

形式上，如果对于所有 $x \ne y$，一个哈希族满足以下条件，则它是通用的：

$$
P(h(x) = h(y)) \le \frac{1}{M}
$$

其中 $M$ 是桶的数量。
这保证了期望的 $O(1)$ 查找时间。

核心思想：每次新的程序运行都使用不同的随机哈希种子，因此攻击者或病态数据集无法强制造成冲突。

#### 示例（通用哈希）

令 $U = {0, 1, \dots, p-1}$，其中 $p$ 是素数。
对于从 $1, \dots, p-1$ 中均匀选择的参数 $a, b$，定义：

$$
h_{a,b}(x) = ((a \cdot x + b) \bmod p) \bmod M
$$

这就得到了一个通用哈希族，具有较低的冲突概率和简单的计算。

#### 实际示例

假设 $p = 17$, $M = 10$, $a = 5$, $b = 3$：

| x | h(x) = (5x+3) mod 17 mod 10 |
| - | --------------------------- |
| 1 | 8                           |
| 2 | 3                           |
| 3 | 8                           |
| 4 | 3                           |

每个键被分配到一个伪随机的桶。
冲突仍然可能发生，但不可预测，由概率而非结构控制。

#### 微型代码（简易版本）

Python 版本

```python
import random

def random_hash(a, b, p, M, x):
    return ((a * x + b) % p) % M

def make_hash(p, M):
    a = random.randint(1, p - 1)
    b = random.randint(0, p - 1)
    return lambda x: ((a * x + b) % p) % M

# 示例
p, M = 17, 10
h = make_hash(p, M)
data = [1, 2, 3, 4, 5]
print([h(x) for x in data])
```

C 版本

```c
#include <stdio.h>
#include <stdlib.h>

int random_hash(int a, int b, int p, int M, int x) {
    return ((a * x + b) % p) % M;
}

int main(void) {
    int p = 17, M = 10;
    int a = rand() % (p - 1) + 1;
    int b = rand() % p;
    int data[] = {1, 2, 3, 4, 5};
    int n = 5;
    for (int i = 0; i < n; i++) {
        printf("x=%d -> h=%d\n", data[i], random_hash(a, b, p, M, data[i]));
    }
}
```

#### 为什么它很重要

- **公平性**：键被均匀分布，降低了冲突风险。
- **安全性**：防止对抗性键选择（对于防御哈希洪水攻击很重要）。
- **性能**：即使在最坏的输入情况下，也能确保期望的 $O(1)$ 查找时间。
- **应用**：
  * 哈希表
  * 布谷鸟哈希
  * 布隆过滤器
  * 一致性哈希
  * 加密方案（非加密随机性）

像 Python、Java 和 Go 这样的语言会随机化哈希种子，以防御基于输入的攻击。

#### 一个温和的证明（为什么它有效）

对于一个定义在 $U$ 上、大小为 $|H| = m$ 的通用哈希族 $H$：

$$
P(h(x) = h(y)) = \frac{1}{M}, \quad x \ne y
$$

因此，对于 $n$ 个键，期望的冲突数量为：

$$
E[\text{collisions}] = \binom{n}{2} \cdot \frac{1}{M}
$$

如果 $M \approx n$，期望冲突数 ≈ 常数。

因此，平均查找和插入时间 = $O(1)$。

#### 动手试试

1.  比较随机哈希与朴素哈希 $(x \bmod M)$。
2.  绘制多次随机运行中桶的频率分布。
3.  测试不同随机种子下的冲突计数。
4.  实现二选一哈希（$h_1$, $h_2$ 选择负载较轻的桶）。
5.  构建一个小的通用哈希表。

#### 测试用例

| 键               | M  | 哈希类型          | 期望冲突数         |
| ---------------- | -- | ----------------- | ------------------ |
| 0–99             | 10 | 朴素 $x \bmod M$  | 聚集               |
| 0–99             | 10 | 随机 $ax+b$       | 平衡               |
| 对抗性键         | 10 | 随机种子          | 不可预测           |

#### 复杂度

- 时间：每次哈希 $O(1)$
- 空间：存储参数 $O(1)$

随机哈希是结构化的不可预测性，在代码中是确定性的，但在行为上是概率公平的。
### 538 随机游走模拟

随机游走是由一系列随机步骤定义的路径。它模拟了无数自然和计算过程，从液体中分子的漂移到股票价格的波动，从物理学中的扩散到人工智能中的探索。

通过模拟随机游走，我们可以捕捉随机性如何随时间展开。

#### 我们要解决什么问题？

我们想要研究当一个过程的每一步都依赖于随机选择，而非确定性规则时，该过程如何演化。

随机游走出现在：

- 物理学：布朗运动
- 金融学：股票运动模型
- 图论：马尔可夫链，PageRank
- 算法：随机搜索和采样

模拟让我们可以看到期望位移、返回概率和空间扩散。

#### 它是如何工作的（通俗解释）

随机游走从一个原点开始（例如，$(0,0)$）。在每一步，随机选择一个方向并移动一个单位。

例子：

- 一维游走：步长 +1 或 −1
- 二维游走：向北、南、东或西移动
- 三维游走：沿 x、y 或 z 轴随机移动

重复 $n$ 步，跟踪位置，并分析结果。

#### 示例（一维）

从 $x=0$ 开始。对于每一步：

- 以概率 $1/2$：$x \gets x+1$
- 以概率 $1/2$：$x \gets x-1$

经过 $n$ 步后，位置是随机变量 $X_n$。
期望值：$E[X_n] = 0$
方差：$Var[X_n] = n$
期望距原点的距离 ≈ $\sqrt{n}$

#### 示例（二维）

从 $(0,0)$ 开始
每一步：随机向上、下、左或右移动。
绘制路径会产生一条蜿蜒的轨迹，这是扩散的图景。

#### 微型代码（简易版本）

Python（一维随机游走）

```python
import random

def random_walk_1d(steps):
    x = 0
    path = [x]
    for _ in range(steps):
        x += random.choice([-1, 1])
        path.append(x)
    return path

walk = random_walk_1d(100)
print("最终位置:", walk[-1])
```

Python（二维随机游走）

```python
import random

def random_walk_2d(steps):
    x, y = 0, 0
    path = [(x, y)]
    for _ in range(steps):
        dx, dy = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
        x, y = x + dx, y + dy
        path.append((x, y))
    return path

walk = random_walk_2d(50)
print("最终位置:", walk[-1])
```

C 语言版本（一维）

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    srand(time(NULL));
    int steps = 100;
    int x = 0;
    for (int i = 0; i < steps; i++) {
        x += (rand() % 2) ? 1 : -1;
    }
    printf("最终位置: %d\n", x);
}
```

#### 为什么它重要

随机游走构成了许多系统的数学基础：

- 物理学：扩散，布朗运动
- 金融学：随机价格波动
- 人工智能与强化学习：探索策略
- 图算法：PageRank，覆盖时间
- 统计学：蒙特卡洛方法

它们展示了秩序如何从随机性中产生，以及为什么许多自然过程会随时间扩散。

#### 一个温和的证明（为什么它有效）

每一步 $S_i$ 是独立的，且 $E[S_i]=0$，$Var[S_i]=1$。经过 $n$ 步后：

$$
X_n = \sum_{i=1}^n S_i, \quad E[X_n]=0, \quad Var[X_n]=n
$$

期望平方距离：$E[X_n^2] = n$
期望绝对位移：$\sqrt{n}$
这种缩放关系解释了扩散的 $\sqrt{t}$ 行为。

在二维或三维中，类似的逻辑可以扩展：
$$
E[|X_n|^2] = n \cdot \text{步长}^2
$$

#### 亲自尝试

1.  绘制 $n=1000$ 时在一维和二维中的路径。
2.  比较多次运行，注意其变异性。
3.  跟踪多次试验中距原点的平均距离。
4.  改变步长概率（有偏游走）。
5.  添加吸收边界（例如，在 $x=10$ 处停止）。

#### 测试用例

| 步数 | 维度 | 期望 $E[X_n]$ | 期望距离 |
| ---- | ---- | ------------- | -------- |
| 10   | 1D   | 0             | ~3.16    |
| 100  | 1D   | 0             | ~10      |
| 100  | 2D   | (0,0)         | ~10      |
| 1000 | 1D   | 0             | ~31.6    |

#### 复杂度

- 时间：$O(n)$
- 空间：如果存储路径则为 $O(n)$；如果只跟踪位置则为 $O(1)$

随机游走是无计划的运动，一步一步，方向随机，模式只在长期运行中显现。
### 539 优惠券收集者问题估算

优惠券收集者问题（Coupon Collector Problem）提出：
从一个大小为 $n$ 的集合中，需要多少次随机抽取（有放回）才能收集到所有不同的物品？

这是概率分析的一个基石，用于模拟从收集交易卡到随机算法测试覆盖率等各种场景。

#### 我们解决的是什么问题？

假设有 $n$ 种不同的优惠券（或宝可梦，或测试用例）。
每次，你均匀随机地抽取一张。

问题：
平均需要抽取多少次 $T$ 才能收集齐所有 $n$ 种？

#### 核心思想

每次新的抽取，发现**新**物品的机会都在变小。
开始时，很容易找到新优惠券；接近结束时，你抽到的大部分都是重复的。

期望的抽取次数为：

$$
E[T] = n \cdot H_n = n \left(1 + \frac{1}{2} + \frac{1}{3} + \cdots + \frac{1}{n}\right)
$$

渐近地看：

$$
E[T] \approx n \ln n + \gamma n + \frac{1}{2}
$$

其中 $\gamma \approx 0.57721$ 是欧拉-马斯刻若尼常数。

#### 工作原理（通俗解释）

你可以将收集过程视为几个阶段：

- 阶段 1：获得第 1 张新优惠券 → 期望 1 次抽取
- 阶段 2：获得第 2 张新优惠券 → 期望 $\frac{n}{n-1}$ 次抽取
- 阶段 3：获得第 3 张新优惠券 → 期望 $\frac{n}{n-2}$ 次抽取
- …
- 阶段 $n$：最后一张优惠券 → 期望 $\frac{n}{1}$ 次抽取

将它们求和得到 $E[T] = n H_n$。

#### 示例

对于 $n = 5$：

$$
E[T] = 5 \cdot (1 + \frac{1}{2} + \frac{1}{3} + \frac{1}{4} + \frac{1}{5}) = 5 \times 2.283 = 11.415
$$

所以平均需要大约 11–12 次抽取才能集齐 5 张。

#### 简易代码示例

Python（模拟）

```python
import random

def coupon_collector(n, trials=10000):
    total = 0
    for _ in range(trials):
        collected = set()
        count = 0
        while len(collected) < n:
            coupon = random.randint(1, n)
            collected.add(coupon)
            count += 1
        total += count
    return total / trials

for n in [5, 10, 20]:
    print(f"n={n}, Expected≈{coupon_collector(n):.2f}, Theory≈{n * sum(1/i for i in range(1, n+1)):.2f}")
```

C（简单模拟）

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

double coupon_collector(int n, int trials) {
    int total = 0;
    bool seen[n+1];
    for (int t = 0; t < trials; t++) {
        for (int i = 1; i <= n; i++) seen[i] = false;
        int count = 0, collected = 0;
        while (collected < n) {
            int c = rand() % n + 1;
            count++;
            if (!seen[c]) { seen[c] = true; collected++; }
        }
        total += count;
    }
    return (double)total / trials;
}

int main(void) {
    srand(time(NULL));
    int ns[] = {5, 10, 20};
    for (int i = 0; i < 3; i++) {
        int n = ns[i];
        printf("n=%d, E≈%.2f\n", n, coupon_collector(n, 10000));
    }
}
```

#### 为何重要

优惠券收集者模型无处不在：

- 算法覆盖率（例如，哈希所有桶）
- 测试与采样（确保所有情况都出现）
- 网络（数据包收集）
- 分布式系统（流言协议）
- 收集类游戏（完成一套的期望成本）

它捕捉了随机性的收益递减规律，最后的几件物品总是需要最长的时间。

#### 一个温和的证明（为何有效）

令 $T_i$ = 当你已经拥有 $i-1$ 张时，获得一张新优惠券所需的抽取次数。

成功概率 = $\frac{n - i + 1}{n}$

所以阶段 $i$ 的期望抽取次数 = $\frac{1}{p_i} = \frac{n}{n - i + 1}$

对所有阶段求和：

$$
E[T] = \sum_{i=1}^n \frac{n}{n - i + 1} = n \sum_{k=1}^n \frac{1}{k} = n H_n
$$

#### 动手尝试

1.  对不同 $n$ 值进行模拟；与理论值比较。
2.  绘制 $E[T]/n$ 的图，应呈 $\ln n$ 增长。
3.  修改为有偏概率（非均匀抽取）。
4.  估算在 $t$ 步内收集齐所有优惠券的概率。
5.  应用于随机负载均衡（球入箱问题）。

#### 测试用例

| n  | 理论值 $E[T]$ | 模拟值（近似） |
| -- | ------------------ | ------------------- |
| 5  | 11.42              | 11.40               |
| 10 | 29.29              | 29.20               |
| 20 | 71.94              | 72.10               |
| 50 | 224.96             | 225.10              |

#### 复杂度

- 时间：$O(n \cdot \text{trials})$
- 空间：$O(n)$

优惠券收集者问题是耐心的数学隐喻：越接近完成，进展变得越罕见。
### 540 马尔可夫链模拟

马尔可夫链模拟一个系统在状态之间按照固定的转移概率进行跳转。未来仅取决于当前状态，而非整个过去历史。当分析困难时，模拟让我们能够估计长期行为、命中时间和稳态分布。

#### 我们要解决什么问题？

给定一个有限状态空间 $\mathcal{S}={1,\dots,m}$ 和一个行随机的转移矩阵 $P\in\mathbb{R}^{m\times m}$，其中
$$
P_{ij}=P(X_{t+1}=j \mid X_t=i),\quad \sum_{j=1}^m P_{ij}=1,
$$
我们想要生成一条轨迹 $X_0,X_1,\dots,X_T$ 并估计以下量：

- 平稳分布 $\pi$，满足 $\pi P=\pi$
- 期望奖励 $\mathbb{E}[f(X_t)]$
- 命中时间或返回时间

#### 它是如何工作的（通俗解释）

1. 选择一个初始状态 $X_0$（或一个初始分布 $\mu$）。
2. 对于 $t=0,1,\dots,T-1$
   * 根据 $P$ 的第 $i$ 行，从当前状态 $i=X_t$ 抽取下一个状态 $j$。
   * 设置 $X_{t+1}=j$。
3. （可选）丢弃一个预热前缀，然后对剩余部分计算统计量的平均值。

如果链是不可约且非周期的，经验频率将收敛到平稳分布。

#### 示例

两状态天气模型，状态 ${S,R}$ 分别代表晴天、雨天：
$$
P=\begin{pmatrix}
0.8 & 0.2\
0.4 & 0.6
\end{pmatrix}.
$$
从晴天开始，模拟 10,000 步，估计晴天的比例。
理论：求解 $\pi=\pi P$，得到 $\pi_S=\tfrac{2}{3}$，$\pi_R=\tfrac{1}{3}$。

#### 微型代码（简易版本）

Python 版本

```python
import random

def simulate_markov(P, start, steps):
    # P: 列表的列表，每行和为 1
    # start: 整数状态索引
    x = start
    traj = [x]
    for _ in range(steps):
        r = random.random()
        cdf, nxt = 0.0, 0
        for j, p in enumerate(P[x]):
            cdf += p
            if r <= cdf:
                nxt = j
                break
        x = nxt
        traj.append(x)
    return traj

# 示例：Sunny=0, Rainy=1
P = [[0.8, 0.2],
     [0.4, 0.6]]
traj = simulate_markov(P, start=0, steps=10000)
burn = 1000
pi_hat_S = sum(1 for s in traj[burn:] if s == 0) / (len(traj) - burn)
print("Estimated pi(Sunny) =", pi_hat_S)
```

C 版本

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int next_state(double *row, int m) {
    double r = (double)rand() / RAND_MAX, cdf = 0.0;
    for (int j = 0; j < m; j++) {
        cdf += row[j];
        if (r <= cdf) return j;
    }
    return m - 1;
}

int main(void) {
    srand((unsigned)time(NULL));
    int m = 2, steps = 10000, x = 0;
    double P[2][2] = {{0.8, 0.2}, {0.4, 0.6}};
    int sunny = 0, burn = 1000;
    for (int t = 0; t < steps; t++) {
        if (t >= burn && x == 0) sunny++;
        x = next_state(P[x], m);
    }
    double pi_hat = (double)sunny / (steps - burn);
    printf("Estimated pi(Sunny) = %.4f\n", pi_hat);
    return 0;
}
```

#### 为什么它很重要

- 统计学和 MCMC：通过从复杂分布中采样来估计积分
- 运筹学：排队系统、可靠性、库存
- 强化学习：建模环境动态和回报
- 网络和图：PageRank、随机游走、混合时间

当闭式解不可用或成本过高时，模拟提供了估计值。

#### 一个温和的证明（为什么它有效）

如果链是不可约、非周期的，且具有平稳分布 $\pi$，那么根据遍历定理
$$
\frac{1}{T}\sum_{t=1}^T f(X_t)\xrightarrow[]{a.s.}\sum_{i\in\mathcal{S}} \pi_i f(i).
$$
特别地，经验状态频率收敛到 $\pi$。预热可以减少来自初始分布的偏差。

#### 亲自尝试

1. 验证两状态示例，将 $\pi_{\text{Sunny}}$ 的估计值与 $2/3$ 进行比较。
2. 构建一个具有已知 $\pi$ 的 5 状态链，并确认收敛性。
3. 添加一个奖励函数 $r(i)$，估计长期平均奖励。
4. 探索预热和样本量对方差的影响。
5. 模拟图上的随机游走，估计访问频率。

#### 测试用例

| 链                                     | 属性                           | 理论         | 模拟（近似）            |
| ---------------------------------------- | ---------------------------------- | ------------ | ----------------------- |
| 2 状态天气模型                           | $\pi_S$                            | $2/3$        | $0.66$ 到 $0.67$        |
| 带阻尼的 3 状态循环链                    | 遍历性                             | 唯一 $\pi$   | 频率匹配 $\pi$          |
| 在具有反射端的直线上随机游走             | 平稳质量在中间附近更高             | 定性         | 匹配直方图              |

#### 复杂度

- 时间：$O(T \cdot \deg)$，其中 $\deg$ 是 $P$ 每行非零项的平均数
- 空间：对于在线统计是 $O(1)$，如果存储完整轨迹则是 $O(T)$

马尔可夫链模拟将矩阵概率转化为经验行为，一次一步，然后平均值揭示了长期的真相。

# 第 55 节 筛法与模运算
### 541 埃拉托斯特尼筛法

埃拉托斯特尼筛法是数学中最古老、最优雅的算法之一，旨在生成所有不超过给定上限 $n$ 的素数。它系统地标记出素数的倍数，只留下未被标记的素数。

#### 我们要解决什么问题？

我们需要一种高效的方法来找到所有 $\le n$ 的素数。一种简单的方法是对每个数测试可除性，时间复杂度为 $O(n\sqrt{n})$。筛法通过批量标记合数，将复杂度显著降低到 $O(n \log\log n)$。

#### 它是如何工作的（通俗解释）

想象一下写下从 $2$ 到 $n$ 的所有数字。从第一个未被标记的数字 $2$ 开始，你：

1.  宣布它为素数。
2.  标记所有 $2$ 的倍数为合数。
3.  移动到下一个未被标记的数字（它必定是素数）。
4.  重复此过程，直到 $p^2 > n$。

剩下的未被标记的数字就是素数。

#### 示例

令 $n = 30$
从 $2$ 开始：

| 步骤 | 素数 | 标记的倍数                                            |
| ---- | ---- | ----------------------------------------------------- |
| 1    | 2    | 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 |
| 2    | 3    | 6, 9, 12, 15, 18, 21, 24, 27, 30                    |
| 3    | 5    | 10, 15, 20, 25, 30                                  |
| 4    | 7    | $7^2 = 49 > 30$ → 停止                               |

剩下的素数：2, 3, 5, 7, 11, 13, 17, 19, 23, 29

#### 精简代码（简易版本）

Python 版本

```python
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    p = 2
    while p * p <= n:
        if is_prime[p]:
            for i in range(p * p, n + 1, p):
                is_prime[i] = False
        p += 1
    return [i for i in range(2, n + 1) if is_prime[i]]

print(sieve(30))
```

C 语言版本

```c
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

void sieve(int n) {
    bool is_prime[n+1];
    for (int i = 0; i <= n; i++) is_prime[i] = true;
    is_prime[0] = is_prime[1] = false;
    for (int p = 2; p * p <= n; p++) {
        if (is_prime[p]) {
            for (int i = p * p; i <= n; i += p)
                is_prime[i] = false;
        }
    }
    for (int i = 2; i <= n; i++)
        if (is_prime[i]) printf("%d ", i);
}

int main(void) {
    sieve(30);
}
```

#### 为什么它很重要

-   高效的素数生成，是数论、密码学和因数分解的基础。
-   核心构建模块，用于：
    *   模运算中的素数表
    *   欧拉函数计算
    *   基于筛法的因数分解
    *   组合数学中的预计算（nCr mod p）
-   直观，是通过模式进行消除的绝佳演示。

#### 一个温和的证明（为什么它有效）

每个合数 $n$ 都有一个最小的素因数 $p \le \sqrt{n}$。当筛法标记每个直到 $\sqrt{n}$ 的素数的倍数时，每个合数恰好被其最小的素因数标记一次。因此，所有未被标记的数都是素数。

#### 亲自尝试

1.  对 $n=50$ 运行筛法，统计素数个数（应为 15）。
2.  修改代码以返回计数而不是列表。
3.  绘制素数密度与 $n$ 的关系图。
4.  扩展到分段筛法以处理大的 $n$。
5.  比较与朴素素数测试的运行时间。

#### 测试用例

| n  | 找到的素数                           | 数量 |
| -- | ------------------------------------ | ---- |
| 10 | 2, 3, 5, 7                           | 4    |
| 20 | 2, 3, 5, 7, 11, 13, 17, 19           | 8    |
| 30 | 2, 3, 5, 7, 11, 13, 17, 19, 23, 29   | 10   |

#### 复杂度

-   时间：$O(n \log\log n)$
-   空间：$O(n)$

埃拉托斯特尼筛法是洞察力磨砺出的简洁，标记倍数，揭示素数。
### 542 线性筛法

线性筛法，也称为欧拉筛法，是埃拉托斯特尼筛法的一种优化版本，它能在 $O(n)$ 时间内计算出所有不超过 $n$ 的素数，并且每个合数只被标记一次。

它通过将素数与其最小质因数相结合，避免了冗余标记。

#### 我们要解决什么问题？

经典筛法会为每个合数的每一个质因数标记一次，导致每个合数被标记多次。
在线性筛法中，每个合数仅由其最小质因数（SPF）生成，确保总工作量与 $n$ 成正比。

我们想要：

- 所有 $\le n$ 的素数
- （可选）每个 $x$ 的最小质因数 `spf[x]`

#### 它是如何工作的（通俗解释）

维护：

- 一个布尔数组 `is_prime[]`
- 一个列表 `primes[]`

算法：

1. 将所有数初始化为素数（`True`）。
2. 对于每个从 2 到 $n$ 的整数 $i$：

   * 如果 `is_prime[i]` 是 `True`，则将 $i$ 添加到素数列表 `primes` 中。
   * 对于 `primes` 中的每一个素数 $p$：

     * 如果 $i \cdot p > n$，则中断循环。
     * 标记 `is_prime[i*p] = False`。
     * 如果 $p$ 能整除 $i$，则停止（以确保每个合数只被标记一次）。

这确保了每个合数恰好被其最小质因数标记一次。

#### 示例

设 $n=10$：

| i  | primes           | 标记的合数 |
| -- | ---------------- | ----------------- |
| 2  | [2]              | 4, 6, 8, 10       |
| 3  | [2,3]            | 6, 9              |
| 4  | 跳过（非素数） |                   |
| 5  | [2,3,5]          | 10                |
| 6  | 跳过             |                   |
| 7  | [2,3,5,7]        |                   |
| 8  | 跳过             |                   |
| 9  | 跳过             |                   |
| 10 | 跳过             |                   |

素数：2, 3, 5, 7

#### 精简代码（简易版本）

Python 版本

```python
def linear_sieve(n):
    is_prime = [True] * (n + 1)
    primes = []
    spf = [0] * (n + 1)  # 最小质因数
    is_prime[0] = is_prime[1] = False

    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            spf[i] = i
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            spf[i * p] = p
            if i % p == 0:
                break
    return primes, spf

pr, spf = linear_sieve(30)
print("Primes:", pr)
```

C 语言版本

```c
#include <stdio.h>
#include <stdbool.h>

void linear_sieve(int n) {
    bool is_prime[n+1];
    int primes[n+1], spf[n+1];
    int count = 0;

    for (int i = 0; i <= n; i++) is_prime[i] = true;
    is_prime[0] = is_prime[1] = false;

    for (int i = 2; i <= n; i++) {
        if (is_prime[i]) {
            primes[count++] = i;
            spf[i] = i;
        }
        for (int j = 0; j < count; j++) {
            int p = primes[j];
            if (i * p > n) break;
            is_prime[i * p] = false;
            spf[i * p] = p;
            if (i % p == 0) break;
        }
    }

    printf("Primes: ");
    for (int i = 0; i < count; i++) printf("%d ", primes[i]);
}

int main(void) {
    linear_sieve(30);
}
```

#### 为什么它很重要

线性筛法是一个强大的改进：

- 每个数只被处理一次
- 最快的素数筛法（紧渐近界）
- 有用的副产品：

  * 用于分解的最小质因数（SPF）表
  * 用于算术函数（例如欧拉函数）的素数列表

广泛应用于：

- 竞赛编程
- 模运算的预计算
- 因数分解和除数枚举

#### 一个温和的证明（为什么它有效）

每个合数 $n = p \cdot m$ 恰好被标记一次，由其最小质因数 $p$ 标记。
当 $i = m$ 时，算法将 $i$ 与 $p$ 配对：

- 如果 $p$ 能整除 $i$，则中断循环以防止进一步的标记。

因此总操作数约为 $O(n)$。

#### 亲自尝试

1.  对于 $n=10^6$，与经典筛法比较运行时间。
2.  打印 $x=2$ 到 $20$ 的 `spf[x]`。
3.  编写一个使用 `spf` 分解 $x$ 的函数。
4.  修改代码以计算质因数个数。
5.  扩展代码以 $O(n)$ 复杂度计算欧拉函数。

#### 测试用例

| n   | 找到的素数                       | 数量 |
| --- | ---------------------------------- | ----- |
| 10  | 2, 3, 5, 7                         | 4     |
| 30  | 2, 3, 5, 7, 11, 13, 17, 19, 23, 29 | 10    |
| 100 | 25 个素数                          | 25    |

#### 复杂度

- 时间：$O(n)$
- 空间：$O(n)$

线性筛法完善了埃拉托斯特尼的智慧，每个合数只标记一次，不多不少。
### 543 分段筛法

分段筛法（Segmented Sieve）扩展了经典筛法，能够高效处理大范围问题，例如生成 $L$ 到 $R$ 之间的素数，其中 $R$ 可能非常大（例如 $10^{12}$），而无需在内存中存储所有直到 $R$ 的数字。

它将范围划分为足够小的段，使其能够放入内存，并使用预先计算的小素数对每一段进行筛选。

#### 我们要解决什么问题？

普通的埃拉托斯特尼筛法（Eratosthenes）需要与 $n$ 成比例的内存，这使得对于大的上限（如 $10^{12}$）变得不可行。
分段筛法通过一个两阶段过程来解决这个问题：

1.  使用标准筛法预先计算到 $\sqrt{R}$ 的小素数。
2.  使用这些素数筛选每个段 $[L, R]$。

这样，即使对于非常大的 $R$，内存使用量也保持在 $O(\sqrt{R}) + O(R-L+1)$。

#### 它是如何工作的（通俗解释）

为了生成 $[L, R]$ 中的素数：

1.  预先筛选小素数：
    $$ \text{small\_primes} = \text{sieve}(\sqrt{R}) $$
2.  初始化一个大小为 $R-L+1$ 的布尔型段数组（全部设为 True）。
3.  对于每个小素数 $p$：
    *   找到 $[L, R]$ 中 $p$ 的第一个倍数：
        $$
        \text{start} = \max(p^2, \lceil \frac{L}{p} \rceil \cdot p)
        $$
    *   将 $p$ 的所有倍数标记为合数。
4.  剩下的未标记数字就是 $[L, R]$ 中的素数。

#### 示例

找出 $[10, 30]$ 中的素数

1.  计算到 $\sqrt{30} = 5$ 的素数：${2, 3, 5}$
2.  段 = `[10..30]`，标记倍数：
    *   $2$：标记 10,12,14,…,30
    *   $3$：标记 12,15,18,21,24,27,30
    *   $5$：标记 10,15,20,25,30
3.  未标记 → 11, 13, 17, 19, 23, 29

#### 简洁代码（简易版本）

Python 版本

```python
import math

def simple_sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[p]:
            for i in range(p * p, limit + 1, p):
                is_prime[i] = False
    return [p for p in range(2, limit + 1) if is_prime[p]]

def segmented_sieve(L, R):
    limit = int(math.sqrt(R)) + 1
    primes = simple_sieve(limit)
    is_prime = [True] * (R - L + 1)

    for p in primes:
        start = max(p * p, ((L + p - 1) // p) * p)
        for i in range(start, R + 1, p):
            is_prime[i - L] = False

    if L == 1:
        is_prime[0] = False

    return [L + i for i, prime in enumerate(is_prime) if prime]

print(segmented_sieve(10, 30))
```

C 版本

```c
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

void simple_sieve(int limit, int primes[], int *count) {
    bool mark[limit+1];
    for (int i = 0; i <= limit; i++) mark[i] = true;
    mark[0] = mark[1] = false;
    for (int p = 2; p * p <= limit; p++)
        if (mark[p])
            for (int i = p * p; i <= limit; i += p)
                mark[i] = false;
    *count = 0;
    for (int i = 2; i <= limit; i++)
        if (mark[i]) primes[(*count)++] = i;
}

void segmented_sieve(long long L, long long R) {
    int primes[100000], count;
    int limit = (int)sqrt(R) + 1;
    simple_sieve(limit, primes, &count);
    int size = R - L + 1;
    bool is_prime[size];
    for (int i = 0; i < size; i++) is_prime[i] = true;

    for (int i = 0; i < count; i++) {
        long long p = primes[i];
        long long start = p * p;
        if (start < L) start = ((L + p - 1) / p) * p;
        for (long long j = start; j <= R; j += p)
            is_prime[j - L] = false;
    }

    if (L == 1) is_prime[0] = false;

    for (int i = 0; i < size; i++)
        if (is_prime[i]) printf("%lld ", L + i);
}

int main(void) {
    segmented_sieve(10, 30);
}
```

#### 为什么它很重要

分段筛法在以下情况下至关重要：

-   $R$ 太大，无法容纳整个数组
-   您需要在范围内生成素数，例如 [10⁹, 10⁹ + 10⁶]
-   为大数算法（RSA、素性测试）构建素数列表

它结合了空间效率和速度，利用了预先计算的小素数。

#### 一个温和的证明（为什么它有效）

$[L,R]$ 中的每个合数都有一个最小的质因数 $p \le \sqrt{R}$。
因此，通过标记所有 $\le \sqrt{R}$ 的素数的倍数，我们消除了所有合数。
剩下的未标记数字必定是素数。

无需存储所有 ≤ $R$ 的数字，只需存储当前段。

#### 亲自尝试

1.  生成 $10^6$ 到 $10^6+1000$ 之间的素数。
2.  测量内存使用情况，与完整筛法进行比较。
3.  尝试非正方形的段大小。
4.  与经典筛法比较性能。
5.  结合轮式分解（wheel factorization）以获得额外速度。

#### 测试用例

| 范围        | 输出素数            | 数量 |
| ----------- | ------------------- | ---- |
| [10, 30]    | 11,13,17,19,23,29   | 6    |
| [1, 10]     | 2,3,5,7             | 4    |
| [100, 120]  | 101,103,107,109,113 | 5    |

#### 复杂度

-   时间：$O((R-L+1)\log\log R)$
-   空间：$O(\sqrt{R}) + O(R-L+1)$

分段筛法将埃拉托斯特尼的思想扩展到无限，逐段筛选，内存不再是障碍。
### 544 SPF（最小质因数）表

最小质因数（SPF）表预先计算每个整数 $1 \le n \le N$ 的最小质因数。
它是快速分解、除数函数和积性算术函数的强大基础，所有这些操作都能在 $O(N)$ 时间内完成。

#### 我们要解决什么问题？

我们经常需要：

- 快速分解多个数字
- 计算算术函数（例如 $\varphi(n)$, $\tau(n)$, $\sigma(n)$）
- 高效检查素数性

试除法每次查询是 $O(\sqrt{n})$，对于大量查询来说太慢。
通过 SPF 预计算，任何数字都可以在 $O(\log n)$ 时间内分解。

#### 它是如何工作的（通俗解释）

我们构建一个表 `spf[i]`，使得：

- 如果 $i$ 是质数 → `spf[i] = i`
- 如果 $i$ 是合数 → `spf[i]` = 能整除 $i$ 的最小质数

我们使用线性筛法填充它：

1.  从 `spf[i] = 0`（未设置）开始。
2.  当访问到质数 $p$ 时，设置 `spf[p] = p`。
3.  对于每个 $p$，如果 `spf[i*p]` 未设置，则用 `spf[i*p] = p` 标记 $i \cdot p$。

每个合数由其最小质因数处理一次。

#### 示例

计算 $1 \le i \le 10$ 的 `spf`：

| i  | spf[i] | 质因数分解 |
| -- | ------ | ---------- |
| 1  | 1      |,           |
| 2  | 2      | 2          |
| 3  | 3      | 3          |
| 4  | 2      | 2×2        |
| 5  | 5      | 5          |
| 6  | 2      | 2×3        |
| 7  | 7      | 7          |
| 8  | 2      | 2×2×2      |
| 9  | 3      | 3×3        |
| 10 | 2      | 2×5        |

分解变得非常简单：

```
n = 84
spf[84] = 2 → 42
spf[42] = 2 → 21
spf[21] = 3 → 7
spf[7] = 7 → 1
→ 2 × 2 × 3 × 7
```

#### 简洁代码（简易版本）

Python 版本

```python
def spf_sieve(n):
    spf = [0] * (n + 1)
    spf[1] = 1
    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i] = i
            for j in range(i * i, n + 1, i):
                if spf[j] == 0:
                    spf[j] = i
    return spf

def factorize(n, spf):
    factors = []
    while n != 1:
        factors.append(spf[n])
        n //= spf[n]
    return factors

spf = spf_sieve(100)
print(factorize(84, spf))
```

C 版本

```c
#include <stdio.h>

void spf_sieve(int n, int spf[]) {
    for (int i = 0; i <= n; i++) spf[i] = 0;
    spf[1] = 1;
    for (int i = 2; i <= n; i++) {
        if (spf[i] == 0) {
            spf[i] = i;
            for (long long j = (long long)i * i; j <= n; j += i)
                if (spf[j] == 0) spf[j] = i;
        }
    }
}

void factorize(int n, int spf[]) {
    while (n != 1) {
        printf("%d ", spf[n]);
        n /= spf[n];
    }
}

int main(void) {
    int n = 100, spf[101];
    spf_sieve(n, spf);
    factorize(84, spf);
}
```

#### 为什么它很重要

SPF 表是数论算法的瑞士军刀：

- 在 $O(\log n)$ 时间内进行质因数分解
- 计算除数个数：使用 SPF 中的指数
- 计算欧拉函数：通过质因数计算 $\varphi(n)$
- 检测无平方因子数

应用于：

- 模运算系统
- 涉及大量分解的算法
- 竞赛编程中的预计算

#### 一个温和的证明（为什么它有效）

每个合数 $n = p \cdot m$ 都有最小质因数 $p$。
当筛法到达 $p$ 时，它设置 `spf[n] = p`。
如果存在更小的质数，它会更早标记 $n$。
因此 `spf[n]` 确实是能整除 $n$ 的最小质数。

每个数字被标记一次 → 总时间 $O(n)$。

#### 亲自尝试

1.  为 $n=50$ 构建 `spf[]` 并打印所有分解式。
2.  编写 `is_prime(i)` = `(spf[i]==i)`。
3.  修改代码以计算不同质因数的个数。
4.  使用 SPF 因子计算 $\varphi(i)$。
5.  可视化质因数频率。

#### 测试用例

| n  | 通过 SPF 的质因数分解 |
| -- | --------------------- |
| 10 | 2 × 5                 |
| 12 | 2 × 2 × 3             |
| 36 | 2 × 2 × 3 × 3         |
| 84 | 2 × 2 × 3 × 7         |

#### 复杂度

-   预计算：$O(n)$
-   每次查询分解：$O(\log n)$
-   空间：$O(n)$

SPF 表将分解从除法操作转变为查找操作，一次预计算，永久复用。
### 545 莫比乌斯函数筛法

莫比乌斯函数 $\mu(n)$ 是一个乘性算术函数，在反演公式和检测平方因子中处于核心地位。
其值定义如下：

- $\mu(1)=1$
- 如果 $n$ 包含任何平方质因子，则 $\mu(n)=0$
- 如果 $n$ 是 $k$ 个不同质数的乘积，则 $\mu(n)=(-1)^k$

莫比乌斯筛法可以高效地计算大 $N$ 范围内的 $\mu(1),\dots,\mu(N)$。

#### 我们要解决什么问题？

我们希望计算所有 $1 \le n \le N$ 的 $\mu(n)$，并且比单独对每个 $n$ 进行因式分解更快。

目标：

- 以接近线性的时间计算所有直到 $N$ 的 $\mu$ 值
- 通常同时生成质数列表和最小质因子

#### 它是如何工作的（通俗解释）

使用线性筛法，并维护以下不变量：

- 维护一个动态的质数列表 `primes`
- 在过程中存储 `mu[i]`
- 对于每个从 $2$ 到 $N$ 的 $i$：

  * 如果 $i$ 是质数，设置 `mu[i] = -1` 并将其加入 `primes`
  * 对于 `primes` 中的每个质数 $p$：

    * 如果 $i p > N$，停止
    * 如果 $p \mid i$，那么：

      * `mu[i*p] = 0`，因为 $p^2 \mid i p$
      * 中断循环以保持线性复杂度
    * 否则：

      * `mu[i*p] = -mu[i]`，因为我们增加了一个新的不同质因子

初始化 `mu[1] = 1`。

这种方法通过其最小质因子的关系，恰好标记每个合数一次。

#### 示例

计算 $1 \le n \le 12$ 的 $\mu(n)$：

| $n$ | 质因子 | 无平方因子？ | $\mu(n)$ |
| --- | ------------- | ---------- | -------- |
| 1   |,             | 是        | 1        |
| 2   | $2$           | 是        | $-1$     |
| 3   | $3$           | 是        | $-1$     |
| 4   | $2^2$         | 否         | 0        |
| 5   | $5$           | 是        | $-1$     |
| 6   | $2\cdot 3$    | 是        | $+1$     |
| 7   | $7$           | 是        | $-1$     |
| 8   | $2^3$         | 否         | 0        |
| 9   | $3^2$         | 否         | 0        |
| 10  | $2\cdot 5$    | 是        | $+1$     |
| 11  | $11$          | 是        | $-1$     |
| 12  | $2^2\cdot 3$  | 否         | 0        |

#### 微型代码（简易版本）

Python 版本（用于 $\mu$ 的线性筛法）

```python
def mobius_sieve(n):
    mu = [0] * (n + 1)
    mu[1] = 1
    primes = []
    is_comp = [False] * (n + 1)

    for i in range(2, n + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            ip = i * p
            if ip > n:
                break
            is_comp[ip] = True
            if i % p == 0:
                mu[ip] = 0           # p^2 整除 ip
                break
            else:
                mu[ip] = -mu[i]      # 增加一个新的不同质因子
    return mu

# 示例
mu = mobius_sieve(50)
print([ (i, mu[i]) for i in range(1, 13) ])
```

C 语言版本

```c
#include <stdio.h>
#include <stdbool.h>

void mobius_sieve(int n, int mu[]) {
    bool comp[n + 1];
    for (int i = 0; i <= n; i++) { comp[i] = false; mu[i] = 0; }
    mu[1] = 1;

    int primes[n + 1], pc = 0;

    for (int i = 2; i <= n; i++) {
        if (!comp[i]) {
            primes[pc++] = i;
            mu[i] = -1;
        }
        for (int j = 0; j < pc; j++) {
            long long ip = 1LL * i * primes[j];
            if (ip > n) break;
            comp[ip] = true;
            if (i % primes[j] == 0) {
                mu[ip] = 0;              // 平方因子
                break;
            } else {
                mu[ip] = -mu[i];         // 增加一个新的不同质因子
            }
        }
    }
}

int main(void) {
    int N = 50;
    int mu[51];
    mobius_sieve(N, mu);
    for (int i = 1; i <= 12; i++)
        printf("mu(%d) = %d\n", i, mu[i]);
    return 0;
}
```

#### 为什么它很重要

- 数论中的莫比乌斯反演：
  $$
  f(n)=\sum_{d\mid n} g(d)
  \quad \Longleftrightarrow \quad
  g(n)=\sum_{d\mid n} \mu(d) f!\left(\frac{n}{d}\right)
  $$
- 检测无平方因子数：$\mu(n)\ne 0$ 当且仅当 $n$ 是无平方因子数
- 在以下方面处于核心地位：

  * 狄利克雷卷积和乘性函数
  * 关于因子的容斥原理
  * 计算像 $\sum_{n\le N}\mu(n)$ 这样的和
  * 涉及最大公约数或互质约束的计数问题

#### 一个温和的证明（为什么它有效）

基于递增的整数进行归纳，同时维护：

- 如果 $i$ 是质数，那么 $\mu(i)=-1$
- 对于任意质数 $p$：

  * 如果 $p \mid i$，那么 $\mu(i p)=0$，因为 $p^2 \mid i p$
  * 如果 $p \nmid i$，那么 $\mu(i p)=-\mu(i)$，因为 $i p$ 是无平方因子数且多了一个不同的质因子

线性循环在遇到第一个能整除 $i$ 的质数时停止，因此每个合数恰好被处理一次。
因此总工作量与 $N$ 成正比，并且符号的递推关系与 $\mu$ 的定义相符。

#### 亲自尝试

1.  验证对于 $n>1$，$\sum_{d\mid n} \mu(d) = 0$；对于 $n=1$，其值为 $1$。
2.  使用 $\sum_{n\le N} [\mu(n)\ne 0]$ 计算直到 $N$ 的无平方因子整数的数量。
3.  使用预先计算的 $\mu$ 实现狄利克雷卷积，以反演因子和。
4.  比较线性筛法与单独对每个 $n$ 进行因式分解的运行时间。
5.  扩展筛法，使其同时存储最小质因子并复用。

#### 测试用例

| $n$ | $\mu(n)$ |
| --- | -------- |
| 1   | 1        |
| 2   | $-1$     |
| 3   | $-1$     |
| 4   | 0        |
| 6   | $+1$     |
| 10  | $+1$     |
| 12  | 0        |
| 30  | $-1$     |

#### 复杂度

- 时间复杂度：使用线性筛法为 $O(N)$
- 空间复杂度：用于数组为 $O(N)$

莫比乌斯筛法在一次遍历中为整个范围提供 $\mu$ 值，使得大规模的反演和无平方因子分析变得切实可行。
### 546 欧拉函数筛法

欧拉函数 $\varphi(n)$ 统计有多少个整数 $1 \le k \le n$ 与 $n$ 互质。即满足 $\gcd(k,n)=1$ 的数字个数。

我们可以使用线性筛法在 $O(N)$ 时间内计算所有 $\varphi(1), \dots, \varphi(N)$。

#### 我们要解决什么问题？

朴素地计算 $\varphi(n)$ 需要质因数分解：
$$
\varphi(n) = n \prod_{p|n}\left(1 - \frac{1}{p}\right)
$$
对每个 $n$ 单独进行此操作太慢了。

我们想要一种快速的筛法，在一次遍历中计算每个不超过 $N$ 的 $n$ 的 $\varphi$ 值。

#### 它是如何工作的（通俗解释）

我们使用类似于生成质数的线性筛法：

1.  从 `phi[1] = 1` 开始。
2.  对于每个数字 $i$：
    *   如果 $i$ 是质数，那么 $\varphi(i)=i-1$。
    *   对于每个质数 $p$：
        *   如果 $i \cdot p > N$，停止。
        *   如果 $p \mid i$（即 $p$ 整除 $i$）：
            *   $\varphi(i \cdot p) = \varphi(i) \cdot p$
            *   break（以确保线性时间复杂度）
        *   否则：
            *   $\varphi(i \cdot p) = \varphi(i) \cdot (p-1)$

每个数字恰好被处理一次，保持了 $O(N)$ 的复杂度。

#### 示例

让我们计算 $n=1$ 到 $10$ 的 $\varphi(n)$：

| $n$ | 质因数       | 公式                                   | $\varphi(n)$ |
| --- | ------------ | -------------------------------------- | ------------ |
| 1   | ,            | 1                                      | 1            |
| 2   | $2$          | $2(1-\frac{1}{2})$                     | 1            |
| 3   | $3$          | $3(1-\frac{1}{3})$                     | 2            |
| 4   | $2^2$        | $4(1-\frac{1}{2})$                     | 2            |
| 5   | $5$          | $5(1-\frac{1}{5})$                     | 4            |
| 6   | $2\cdot3$    | $6(1-\frac{1}{2})(1-\frac{1}{3})$      | 2            |
| 7   | $7$          | $7(1-\frac{1}{7})$                     | 6            |
| 8   | $2^3$        | $8(1-\frac{1}{2})$                     | 4            |
| 9   | $3^2$        | $9(1-\frac{1}{3})$                     | 6            |
| 10  | $2\cdot5$    | $10(1-\frac{1}{2})(1-\frac{1}{5})$     | 4            |

#### 精简代码（简易版本）

Python 版本

```python
def totient_sieve(n):
    phi = [0] * (n + 1)
    primes = []
    phi[1] = 1
    is_comp = [False] * (n + 1)

    for i in range(2, n + 1):
        if not is_comp[i]:
            primes.append(i)
            phi[i] = i - 1
        for p in primes:
            ip = i * p
            if ip > n:
                break
            is_comp[ip] = True
            if i % p == 0:
                phi[ip] = phi[i] * p
                break
            else:
                phi[ip] = phi[i] * (p - 1)
    return phi

# 示例
phi = totient_sieve(20)
for i in range(1, 11):
    print(f"phi({i}) = {phi[i]}")
```

C 版本

```c
#include <stdio.h>
#include <stdbool.h>

void totient_sieve(int n, int phi[]) {
    bool comp[n + 1];
    int primes[n + 1], pc = 0;
    for (int i = 0; i <= n; i++) { comp[i] = false; phi[i] = 0; }
    phi[1] = 1;

    for (int i = 2; i <= n; i++) {
        if (!comp[i]) {
            primes[pc++] = i;
            phi[i] = i - 1;
        }
        for (int j = 0; j < pc; j++) {
            int p = primes[j];
            long long ip = 1LL * i * p;
            if (ip > n) break;
            comp[ip] = true;
            if (i % p == 0) {
                phi[ip] = phi[i] * p;
                break;
            } else {
                phi[ip] = phi[i] * (p - 1);
            }
        }
    }
}

int main(void) {
    int n = 20, phi[21];
    totient_sieve(n, phi);
    for (int i = 1; i <= 10; i++)
        printf("phi(%d) = %d\n", i, phi[i]);
}
```

#### 为什么它很重要

欧拉函数 $\varphi(n)$ 是以下领域的基础：

-   欧拉定理：如果 $\gcd(a,n)=1$，则 $a^{\varphi(n)} \equiv 1 \pmod n$
-   RSA 加密：$\varphi(n)$ 定义了密钥的模逆元
-   计数既约分数：互质对的数量
-   群论：乘法群 $(\mathbb{Z}/n\mathbb{Z})^\times$ 的大小

并且在以下方面很有用：

-   模运算
-   密码学
-   数论组合学

#### 一个温和的证明（为什么它有效）

令 $i$ 为当前数字，$p$ 为一个质数：

-   如果 $p\nmid i$，则 $\varphi(i p) = \varphi(i) \cdot (p-1)$，因为我们乘以了一个新的不同的质数。
-   如果 $p\mid i$，则 $\varphi(i p) = \varphi(i) \cdot p$，因为我们扩展了现有质数的幂。

每个数字都恰好通过其最小质因数的一种方式构建 → 线性时间。

#### 亲自尝试

1.  计算 $n=1$ 到 $20$ 的 $\varphi(n)$。
2.  验证 $\sum_{d|n} \varphi(d) = n$。
3.  绘制 $\varphi(n)/n$ 的图，可视化互质数的密度。
4.  使用 $\varphi(n)$ 通过欧拉定理计算模逆元。
5.  调整筛法以同时存储质数和最小质因数。

#### 测试用例

| $n$ | $\varphi(n)$ |
| --- | ------------ |
| 1   | 1            |
| 2   | 1            |
| 3   | 2            |
| 4   | 2            |
| 5   | 4            |
| 6   | 2            |
| 10  | 4            |
| 12  | 4            |
| 15  | 8            |
| 20  | 8            |

#### 复杂度

-   时间：$O(N)$
-   空间：$O(N)$

欧拉函数筛法是算术、密码学和模运算推理的支柱，一次遍历，所有 $\varphi(n)$ 准备就绪。
### 547 约数个数筛法

约数个数筛法（Divisor Count Sieve）预先计算所有整数 $1 \le n \le N$ 的正约数个数 $d(n)$（也写作 $\tau(n)$）。
它是数论和组合数学中的一个强大工具，非常适合在 $O(N\log N)$ 时间内高效地统计因子个数。

#### 我们要解决什么问题？

我们想要计算每个不大于 $N$ 的整数的约数个数：
$$
d(n) = \sum_{i \mid n} 1
$$
或者等价地，如果 $n$ 的质因数分解为
$$
n = p_1^{a_1} p_2^{a_2} \dots p_k^{a_k},
$$
那么
$$
d(n) = (a_1 + 1)(a_2 + 1)\dots(a_k + 1).
$$

朴素地对每个数进行因式分解太慢。
筛法在一次统一的遍历中完成计算。

#### 它是如何工作的（通俗解释）

我们使用一个约数累加筛：

对于每个从 $1$ 到 $N$ 的 $i$：

- 对 $i$ 的每个倍数加 $1$（因为 $i$ 整除它的每个倍数）
- 代码表示：

  ```python
  for i in range(1, N+1):
      for j in range(i, N+1, i):
          div[j] += 1
  ```

每个整数 $n$ 会为其每个约数 $i \mid n$ 被递增一次。
总操作次数 $\sim N\log N$。

#### 示例

对于 $N = 10$：

| $n$ | 约数         | $d(n)$ |
| --- | ------------ | ------ |
| 1   | 1            | 1      |
| 2   | 1, 2         | 2      |
| 3   | 1, 3         | 2      |
| 4   | 1, 2, 4      | 3      |
| 5   | 1, 5         | 2      |
| 6   | 1, 2, 3, 6   | 4      |
| 7   | 1, 7         | 2      |
| 8   | 1, 2, 4, 8   | 4      |
| 9   | 1, 3, 9      | 3      |
| 10  | 1, 2, 5, 10  | 4      |

#### 简洁代码（简易版本）

Python 版本

```python
def divisor_count_sieve(n):
    div = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            div[j] += 1
    return div

# 示例
N = 10
div = divisor_count_sieve(N)
for i in range(1, N + 1):
    print(f"d({i}) = {div[i]}")
```

C 语言版本

```c
#include <stdio.h>

void divisor_count_sieve(int n, int div[]) {
    for (int i = 0; i <= n; i++) div[i] = 0;
    for (int i = 1; i <= n; i++)
        for (int j = i; j <= n; j += i)
            div[j]++;
}

int main(void) {
    int N = 10, div[11];
    divisor_count_sieve(N, div);
    for (int i = 1; i <= N; i++)
        printf("d(%d) = %d\n", i, div[i]);
}
```

#### 为什么它很重要

约数个数函数 $d(n)$ 在以下方面至关重要：

- 约数和问题
- 高合成数
- 格点计数
- 关于约数的求和（例如 $\sum_{i=1}^N d(i)$）
- 动态规划和组合枚举
- 数论变换

也出现在以下公式中：
$$
\sigma_0(n) = d(n), \quad \sigma_1(n) = \text{约数之和}
$$

#### 一个温和的证明（为什么它有效）

每个整数 $i$ 恰好整除 $\lfloor N/i \rfloor$ 个不大于 $N$ 的数。
所以在嵌套循环中，$i$ 对 $\lfloor N/i \rfloor$ 个条目贡献了 $+1$。
对 $i$ 求和得到总操作次数：
$$
\sum_{i=1}^{N} \frac{N}{i} \approx N \log N
$$

这种方法高效且直接。

#### 动手尝试

1.  打印 $n = 1$ 到 $30$ 的 $d(n)$。
2.  绘制 $d(n)$ 的图形，观察约数个数的波动情况。
3.  修改代码以计算约数之和：

    ```python
    divsum[j] += i
    ```
4.  与欧拉函数筛结合，研究约数分布。
5.  统计恰好有 $k$ 个约数的数的个数。

#### 测试用例

| $n$ | 约数                         | $d(n)$ |
| --- | ---------------------------- | ------ |
| 1   | 1                            | 1      |
| 2   | 1, 2                         | 2      |
| 4   | 1, 2, 4                      | 3      |
| 6   | 1, 2, 3, 6                   | 4      |
| 8   | 1, 2, 4, 8                   | 4      |
| 12  | 1, 2, 3, 4, 6, 12            | 6      |
| 30  | 1, 2, 3, 5, 6, 10, 15, 30    | 8      |

#### 复杂度

-   时间复杂度：$O(N\log N)$
-   空间复杂度：$O(N)$

约数个数筛法简单而强大，通过几个嵌套循环为每个数预先计算了因子结构。
### 548 模运算预计算

模运算预计算会预先构建一些表格，例如阶乘、逆元以及模 $M$ 下的幂，这样在 $O(N)$ 或 $O(N \log M)$ 的初始化之后，后续的查询就可以在 $O(1)$ 时间内完成。
这是在模数下进行快速组合数学、动态规划和数论变换的基础。

#### 我们要解决什么问题？

我们经常需要重复计算：

- $a+b$、$a-b$、$a\cdot b$、$a^k \bmod M$
- 模逆元 $a^{-1} \bmod M$
- 二项式系数 $\binom{n}{r} \bmod M$

如果每次查询都从头计算，通过幂运算会花费 $O(\log M)$ 的时间。
通过预计算，我们可以在一次线性遍历后，以每次查询 $O(1)$ 的时间来回答。

#### 我们预计算什么？

对于一个质数模数 $M$ 和一个选定的上限 $N$：

- `fact[i] = i! mod M`，其中 $0 \le i \le N$
- `inv[i] = i^{-1} mod M`，其中 $1 \le i \le N$
- `invfact[i] = (i!)^{-1} mod M`，其中 $0 \le i \le N$
- 可选：`powA[i] = A^i mod M`，用于固定的底数 $A$

那么
$$
\binom{n}{r} \bmod M = \text{fact}[n]\cdot \text{invfact}[r]\cdot \text{invfact}[n-r] \bmod M
$$
可以在 $O(1)$ 时间内计算。

#### 它是如何工作的（通俗解释）

1.  **阶乘**：一次正向遍历。
2.  **阶乘逆元**：先用费马小定理计算 $\text{invfact}[N]=\text{fact}[N]^{M-2}\bmod M$，然后反向遍历。
3.  **逐元素逆元**：可以从 `invfact` 和 `fact` 推导，或者利用恒等式线性计算：
   $$
   \text{inv}[1]=1,\qquad
   \text{inv}[i]=M-\left(\left\lfloor \frac{M}{i}\right\rfloor\cdot \text{inv}[M\bmod i]\right)\bmod M
   $$
   这可以在 $O(N)$ 时间内完成。

这些方法依赖于 $M$ 是质数，这样每个 $1\le i<M$ 都是可逆的。

#### 边界情况

- 如果 $M$ 不是质数：使用扩展欧几里得算法求与 $M$ 互质的数的逆元，或者只对不会遇到不可逆因子的索引使用阶乘表。对于复合模数 $M$ 的组合数学，可以使用 $M$ 的质因数分解加上中国剩余定理（CRT），或者在适用时使用 Lucas 或 Garner 方法。
- 范围限制：选择的 $N$ 至少应大于或等于你将查询的最大 $n$。

#### 简短代码（简易版本）

Python 版本（质数模数）

```python
M = 109 + 7

def modpow(a, e, m=M):
    r = 1
    while e:
        if e & 1: r = r * a % m
        a = a * a % m
        e >>= 1
    return r

def build_tables(N, M=109+7):
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = fact[i - 1] * i % M

    invfact = [1] * (N + 1)
    invfact[N] = modpow(fact[N], M - 2, M)  # 费马小定理
    for i in range(N, 0, -1):
        invfact[i - 1] = invfact[i] * i % M

    inv = [0] * (N + 1)
    inv[1] = 1
    for i in range(2, N + 1):
        inv[i] = (M - (M // i) * inv[M % i] % M) % M

    return fact, invfact, inv

def nCr_mod(n, r, fact, invfact, M=109+7):
    if r < 0 or r > n: return 0
    return fact[n] * invfact[r] % M * invfact[n - r] % M

# 示例
N = 1_000_000
fact, invfact, inv = build_tables(N, M)
print(nCr_mod(10, 3, fact, invfact, M))  # 120
```

C 语言版本（质数模数）

```c
#include <stdio.h>
#include <stdint.h>

const int MOD = 1000000007;

long long modpow(long long a, long long e) {
    long long r = 1 % MOD;
    while (e) {
        if (e & 1) r = (r * a) % MOD;
        a = (a * a) % MOD;
        e >>= 1;
    }
    return r;
}

void build_tables(int N, int fact[], int invfact[], int inv[]) {
    fact[0] = 1;
    for (int i = 1; i <= N; i++) fact[i] = (long long)fact[i-1] * i % MOD;

    invfact[N] = modpow(fact[N], MOD - 2);
    for (int i = N; i >= 1; i--) invfact[i-1] = (long long)invfact[i] * i % MOD;

    inv[1] = 1;
    for (int i = 2; i <= N; i++)
        inv[i] = (int)((MOD - (long long)(MOD / i) * inv[MOD % i] % MOD) % MOD);
}

int nCr_mod(int n, int r, int fact[], int invfact[]) {
    if (r < 0 || r > n) return 0;
    return (int)((long long)fact[n] * invfact[r] % MOD * invfact[n - r] % MOD);
}

int main(void) {
    int N = 1000000;
    static int fact[1000001], invfact[1000001], inv[1000001];
    build_tables(N, fact, invfact, inv);
    printf("%d\n", nCr_mod(10, 3, fact, invfact)); // 120
    return 0;
}
```

#### 为什么这很重要

- 快速组合数学：以 $O(1)$ 时间计算 $\binom{n}{r}$、排列、多项式系数。
- 模运算下的动态规划：卷积类状态转移、路径计数。
- 数论：按需提供模逆元和幂。
- 在需要频繁进行模查询的竞技编程和密码学原型开发中非常有用。

#### 一个温和的证明（为什么它有效）

对于质数 $M$，$\mathbb{Z}_M^\times$ 是一个域。费马小定理给出对于 $a \not\equiv 0$，有 $a^{M-2}\equiv a^{-1}\pmod M$。
反向填充得到 $\text{invfact}[i-1]=\text{invfact}[i]\cdot i \bmod M$，因此得到 $(i-1)!^{-1}$。
那么
$$
\binom{n}{r} = \frac{n!}{r!,(n-r)!} \equiv \text{fact}[n]\cdot \text{invfact}[r]\cdot \text{invfact}[n-r] \pmod M.
$$
线性逆元恒等式来源于写出 $i\cdot \text{inv}[i]\equiv 1$ 并对 $M\bmod i$ 进行递归。

#### 亲自尝试

1.  预计算到 $N=10^7$，调整内存使用，并验证 $\sum_{r=0}^n \binom{n}{r}\equiv 2^n \pmod M$。
2.  为固定底数 $A$ 添加幂表 `powA[i]`，以便在 $O(1)$ 时间内回答 $A^k \bmod M$。
3.  实现多项式系数：$\frac{n!}{a_1!\cdots a_k!}$，使用 `fact` 和 `invfact`。
4.  对于复合模数 $M$，分解 $M=\prod p_i^{e_i}$，对每个 $p_i^{e_i}$ 计算模数，然后用中国剩余定理（CRT）组合。
5.  对一次性预计算与每次查询按需进行幂运算进行基准测试。

#### 测试用例

| 查询                                | 答案                                 |
| ----------------------------------- | ------------------------------------ |
| $\binom{10}{3}\bmod 10^9+7$         | 120                                  |
| $\binom{1000}{500}\bmod 10^9+7$     | 从表中以 $O(1)$ 计算得出             |
| $a^{-1}\bmod M$，其中 $a=123456$    | `inv[a]`                             |
| $A^k\bmod M$，用于多个 $k$          | 如果已预计算，则为 `powA[k]`         |

#### 复杂度

- 预计算：$O(N)$ 时间，$O(N)$ 空间
- 每次查询：$O(1)$
- 一次幂运算：如果你选择反向方法，需要一次 $O(\log M)$ 的幂运算来初始化 `invfact[N]`

模运算预计算将繁重的算术运算转化为查表操作。前期一次性投入，之后即可即时回答。
### 549 费马小定理

费马小定理是模运算的基石。它指出，如果 $p$ 是一个素数且 $a$ 不能被 $p$ 整除，那么：

$$
a^{p-1} \equiv 1 \pmod p
$$

这个强大的关系是模逆元、素数测试和幂运算优化的基础。

#### 我们要解决什么问题？

我们经常需要简化或求逆大型模表达式：

- 计算 $a^{-1} \bmod p$
- 简化像 $a^k \bmod p$ 这样的大指数
- 验证素数（费马测试、米勒-拉宾测试）

费马定理使我们无需进行昂贵的除法运算，而是给出：

$$
a^{-1} \equiv a^{p-2} \pmod p
$$

因此，求逆变成了模幂运算，可以在 $O(\log p)$ 时间内完成。

#### 它是如何工作的（通俗解释）

当 $p$ 是素数时，模 $p$ 乘法构成了一个包含 $p-1$ 个元素（不包括 $0$）的群。
根据拉格朗日定理，每个元素自乘到群的大小等于单位元：

$$
a^{p-1} \equiv 1 \pmod p
$$

重新整理得到模逆元：

$$
a \cdot a^{p-2} \equiv 1 \pmod p
$$

因此，$a^{p-2}$ 就是 $a$ 在模 $p$ 下的逆元。

#### 示例

设 $a=3$，$p=7$（一个素数）：

$$
3^{6} = 729 \equiv 1 \pmod 7
$$

确实有：

$$
3^{5} = 243 \equiv 5 \pmod 7
$$

因为 $3 \times 5 = 15 \equiv 1 \pmod 7$，
所以 $5$ 是 $3$ 模 $7$ 的模逆元。

#### 微型代码（简易版本）

Python 版本

```python
def modpow(a, e, m):
    r = 1
    a %= m
    while e:
        if e & 1:
            r = r * a % m
        a = a * a % m
        e >>= 1
    return r

def modinv(a, p):
    return modpow(a, p - 2, p)  # 费马小定理

# 示例
p = 7
a = 3
print(modpow(a, p - 1, p))  # 应该是 1
print(modinv(a, p))         # 应该是 5
```

C 版本

```c
#include <stdio.h>

long long modpow(long long a, long long e, long long m) {
    long long r = 1 % m;
    a %= m;
    while (e) {
        if (e & 1) r = r * a % m;
        a = a * a % m;
        e >>= 1;
    }
    return r;
}

long long modinv(long long a, long long p) {
    return modpow(a, p - 2, p); // 费马小定理
}

int main(void) {
    long long a = 3, p = 7;
    printf("a^(p-1) mod p = %lld\n", modpow(a, p - 1, p));
    printf("Inverse = %lld\n", modinv(a, p));
}
```

#### 为什么它很重要

- **模逆元**：在模运算下进行除法的关键（例如，组合数 $\binom{n}{r}$ mod $p$）。
- **指数约简**：对于大指数，利用模 $p-1$ 的周期性。
- **素数测试**：构成了费马测试和米勒-拉宾测试的基础。
- **RSA 与密码学**：涉及素数的模运算的核心。

#### 一个温和的证明（为什么它成立）

考虑模素数 $p$ 下的所有剩余 ${1,2,\ldots,p-1}$。
将每个元素乘以 $a$（其中 $\gcd(a,p)=1$）会得到它们的排列。因此：

$$
1\cdot2\cdots(p-1) \equiv (a\cdot1)(a\cdot2)\cdots(a\cdot(p-1)) \pmod p
$$

消去 $(p-1)!$（根据威尔逊定理，它在模 $p$ 下非零）得到：

$$
a^{p-1} \equiv 1 \pmod p
$$

#### 亲自尝试

1.  验证对于不同的素数 $p$ 和底数 $a$，$a^{p-1}\equiv 1$ 是否成立。
2.  用它来计算模逆元：测试不同 $a$ 对应的 $a^{p-2}$。
3.  结合模幂运算来加速组合公式的计算。
4.  探索当 $p$ 是合数时会发生什么（费马伪素数）。
5.  使用 $a^{p-1}\bmod p$ 实现费马素数测试。

#### 测试用例

| $a$ | $p$ | $a^{p-1}\bmod p$ | $a^{p-2}\bmod p$ (逆元) |
| --- | --- | ---------------- | ----------------------- |
| 2   | 5   | 1                | 3                       |
| 3   | 7   | 1                | 5                       |
| 4   | 11  | 1                | 3                       |
| 10  | 13  | 1                | 4                       |

#### 复杂度

- **模幂运算**：$O(\log p)$
- **空间**：$O(1)$

费马小定理将除法转化为幂运算，为计算算术引入了代数结构。
### 550 威尔逊定理

威尔逊定理使用阶乘给出了素数的一个显著特征：

$$
(p-1)! \equiv -1 \pmod p
$$

也就是说，对于一个素数 $p$，$(p-1)$ 的阶乘除以 $p$ 的余数是 $p-1$（或者等价地，$-1$）。

反之，如果这个同余式成立，那么 $p$ 必定是素数。

#### 我们要解决什么问题？

我们希望通过阶乘来测试素数性或理解模逆元。

虽然由于阶乘的增长，威尔逊定理对于大素数并不实用，但它在概念上非常优雅，并将阶乘、逆元和模运算美妙地联系起来。

它展示了模 $p$ 的乘法结构是如何循环且对称的。

#### 它是如何工作的（通俗解释）

对于一个素数 $p$，每个数 $1,2,\dots,p-1$ 在模 $p$ 下都有一个唯一的逆元，并且只有 $1$ 和 $p-1$ 是自逆元。

当我们将它们全部相乘时：

- 每一对 $a \cdot a^{-1}$ 贡献 $1$
- $1$ 和 $(p-1)$ 贡献 $1$ 和 $(p-1)$

所以整个乘积变为：

$$
(p-1)! \equiv 1 \cdot (p-1) \equiv -1 \pmod p
$$

#### 示例

我们来测试一些小素数：

| $p$ | $(p-1)!$      | $(p-1)! \bmod p$    | 检查 |
| --- | ------------- | ------------------- | ----- |
| 2   | 1             | 1 ≡ -1 mod 2        | ✔     |
| 3   | 2             | 2 ≡ -1 mod 3        | ✔     |
| 5   | 24            | 24 ≡ -1 mod 5       | ✔     |
| 7   | 720           | 720 ≡ -1 mod 7      | ✔     |
| 11  | 10! = 3628800 | 3628800 ≡ -1 mod 11 | ✔     |

尝试一个合数 $p=6$：
$5! = 120$，并且 $120 \bmod 6 = 0$ → 失败。

所以威尔逊条件是素数性的充要条件。

#### 微型代码（简易版本）

Python 版本

```python
def factorial_mod(n, m):
    res = 1
    for i in range(2, n + 1):
        res = (res * i) % m
    return res

def is_prime_wilson(p):
    if p < 2:
        return False
    return factorial_mod(p - 1, p) == p - 1

# 示例
for p in [2, 3, 5, 6, 7, 11]:
    print(p, is_prime_wilson(p))
```

C 版本

```c
#include <stdio.h>
#include <stdbool.h>

long long factorial_mod(int n, int mod) {
    long long res = 1;
    for (int i = 2; i <= n; i++)
        res = (res * i) % mod;
    return res;
}

bool is_prime_wilson(int p) {
    if (p < 2) return false;
    return factorial_mod(p - 1, p) == p - 1;
}

int main(void) {
    int ps[] = {2, 3, 5, 6, 7, 11};
    for (int i = 0; i < 6; i++)
        printf("%d %s\n", ps[i], is_prime_wilson(ps[i]) ? "prime" : "composite");
}
```

#### 为什么它重要

威尔逊定理连接了组合数学、模运算和素数性：

- 素数性表征：$p$ 是素数 $\iff (p-1)! \equiv -1 \pmod p$
- 群结构的证明：$(\mathbb{Z}/p\mathbb{Z})^\times$ 是一个乘法群
- 阶乘逆元：$(p-1)!$ 充当 $-1$，使得某些模证明成为可能

尽管对于大的 $p$ 效率不高，但它在数论中具有重要的概念意义。

#### 一个温和的证明（为什么它成立）

设 $p$ 为素数。
集合 ${1, 2, \dots, p-1}$ 在模 $p$ 乘法下构成一个群。

每个元素 $a$ 都有一个逆元 $a^{-1}$。
将所有元素相乘：

$$
(p-1)! \equiv \prod_{a=1}^{p-1} a \equiv \prod_{a=1}^{p-1} a^{-1} \equiv (p-1)!^{-1} \pmod p
$$

因此：

$$
((p-1)!)^2 \equiv 1 \pmod p
$$

所以 $(p-1)! \equiv \pm 1$。
如果 $(p-1)! \equiv 1$，那么所有数都是自逆元，这只可能发生在 $p=2$。
对于 $p>2$，它必须是 $-1$。

反之，如果 $(p-1)! \equiv -1$，$p$ 不可能是合数（因为合数的阶乘模 $p$ 为 $0$）。

#### 自己动手试试

1.  验证小素数的 $(p-1)! \bmod p$。
2.  检查为什么对于 $p=4,6,8,9$ 它会失败。
3.  探索模一个合数时会发生什么（阶乘将包含 $p$ 的因子）。
4.  用它来证明 $(p-1)! + 1$ 能被 $p$ 整除。
5.  尝试为小范围优化阶乘模 $p$。

#### 测试用例

| $p$ | $(p-1)!$ | $(p-1)! \bmod p$ | 素数? |
| --- | -------- | ---------------- | ------ |
| 2   | 1        | 1                | ✔      |
| 3   | 2        | 2                | ✔      |
| 4   | 6        | 2                | ✖      |
| 5   | 24       | 4                | ✔      |
| 6   | 120      | 0                | ✖      |
| 7   | 720      | 6                | ✔      |

#### 复杂度

- 时间：$O(p)$（阶乘模计算）
- 空间：$O(1)$

尽管对于素数测试效率不高，但威尔逊定理美妙地连接了阶乘、逆元和素数，是初等数论中的一颗明珠。

# 第 56 节 线性代数
### 551 高斯消元法

高斯消元法是求解线性方程组、计算行列式和求矩阵秩的基本算法。
它通过行操作系统地化简约定矩阵为上三角形式，之后可以通过回代求解。

#### 我们要解决什么问题？

我们想求解一个包含 $n$ 个变量、$n$ 个线性方程组的系统：

$$
A\mathbf{x} = \mathbf{b}
$$

其中
$A$ 是一个 $n \times n$ 矩阵，
$\mathbf{x}$ 是未知数向量，
$\mathbf{b}$ 是常数向量。

与猜测或手动代入不同，高斯消元法提供了一种系统的、确定性的、多项式时间的方法。

#### 它是如何工作的（通俗解释）

我们对增广矩阵 $[A | b]$ 执行初等行操作以简化它：

1. 前向消元

   * 对于每一列，选择一个主元（非零元素）。
   * 如果需要，交换行（部分主元法）。
   * 消去主元下方的所有元素使其为零。

2. 回代

   * 一旦化为上三角形式，从最后一个方程开始向上求解。

这将系统转换为：
$$
U\mathbf{x} = \mathbf{c}
$$
其中 $U$ 是上三角矩阵，易于求解。

#### 示例

求解：
$$
\begin{cases}
2x + y - z = 8 \\
-3x - y + 2z = -11 \\
-2x + y + 2z = -3
\end{cases}
$$

步骤 1：写出增广矩阵

$$
\left[
\begin{array}{rrr|r}
2 & 1 & -1 & 8 \\
-3 & -1 & 2 & -11 \\
-2 & 1 & 2 & -3
\end{array}
\right]
$$


步骤 2：消去第一个主元下方的元素

使用主元 = 2（第 1 行）。

$$
R_2 \gets R_2 + \tfrac{3}{2}R_1,\qquad
R_3 \gets R_3 + R_1
$$

$$
\left[
\begin{array}{rrr|r}
2 & 1 & -1 & 8 \\
0 & \tfrac{1}{2} & \tfrac{1}{2} & 1 \\
0 & 2 & 1 & 5
\end{array}
\right]
$$

步骤 3：消去第二个主元下方的元素

主元 = $\tfrac{1}{2}$（第 2 行）。

$$
R_3 \gets R_3 - 4R_2
$$

$$
\left[
\begin{array}{rrr|r}
2 & 1 & -1 & 8 \\
0 & \tfrac{1}{2} & \tfrac{1}{2} & 1 \\
0 & 0 & -1 & 1
\end{array}
\right]
$$


步骤 4：回代

从下往上：

- $-z = 1 \implies z = -1$
- $0.5y + 0.5z = 1 \implies y = 3$
- $2x + y - z = 8 \implies 2x + 3 + 1 = 8 \implies x = 2$

解：$(x, y, z) = (2, 3, -1)$

#### 微型代码（简易版本）

Python 版本

```python
def gaussian_elimination(a, b):
    n = len(a)
    for i in range(n):
        # 选主元
        max_row = max(range(i, n), key=lambda r: abs(a[r][i]))
        a[i], a[max_row] = a[max_row], a[i]
        b[i], b[max_row] = b[max_row], b[i]

        # 消去下方元素
        for j in range(i + 1, n):
            factor = a[j][i] / a[i][i]
            for k in range(i, n):
                a[j][k] -= factor * a[i][k]
            b[j] -= factor * b[i]

    # 回代
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(a[i][j] * x[j] for j in range(i + 1, n))) / a[i][i]
    return x

A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
B = [8, -11, -3]
print(gaussian_elimination(A, B))  # [2.0, 3.0, -1.0]
```

C 语言版本

```c
#include <stdio.h>
#include <math.h>

#define N 3

void gaussian_elimination(double a[N][N], double b[N], double x[N]) {
    for (int i = 0; i < N; i++) {
        // 选主元
        int max_row = i;
        for (int r = i + 1; r < N; r++)
            if (fabs(a[r][i]) > fabs(a[max_row][i]))
                max_row = r;
        for (int c = 0; c < N; c++) {
            double tmp = a[i][c];
            a[i][c] = a[max_row][c];
            a[max_row][c] = tmp;
        }
        double tmp = b[i]; b[i] = b[max_row]; b[max_row] = tmp;

        // 消去
        for (int j = i + 1; j < N; j++) {
            double factor = a[j][i] / a[i][i];
            for (int k = i; k < N; k++)
                a[j][k] -= factor * a[i][k];
            b[j] -= factor * b[i];
        }
    }

    // 回代
    for (int i = N - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < N; j++)
            sum += a[i][j] * x[j];
        x[i] = (b[i] - sum) / a[i][i];
    }
}

int main() {
    double A[N][N] = {{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}};
    double B[N] = {8, -11, -3}, X[N];
    gaussian_elimination(A, B, X);
    printf("x = %.2f, y = %.2f, z = %.2f\n", X[0], X[1], X[2]);
}
```

#### 为什么它很重要

高斯消元法是以下领域的基础：

- 求解 $A\mathbf{x}=\mathbf{b}$
- 求行列式（$\det(A)$ 是主元的乘积）
- 求逆矩阵（通过对 $[A|I]$ 应用消元法）
- 计算矩阵的秩（消元后统计非零行的数量）

它是线性代数库（BLAS/LAPACK）、数值求解器和符号计算的基础。

#### 一个温和的证明（为什么它有效）

每个初等行操作都对应于乘以一个可逆矩阵 $E_i$。
经过一系列操作后：
$$
E_k \cdots E_1 A = U
$$
其中 $U$ 是上三角矩阵。
那么：
$$
A = E_1^{-1}\cdots E_k^{-1}U
$$
因此系统 $A\mathbf{x}=\mathbf{b}$ 变为 $U\mathbf{x} = (E_k\cdots E_1)\mathbf{b}$，可以通过回代求解。

每一步都保持解空间不变，确保了正确性。

#### 亲自尝试

1.  手动使用高斯消元法求解一个 3×3 方程组。
2.  修改算法以返回行列式 = 主元的乘积。
3.  扩展到增广矩阵以计算逆矩阵。
4.  添加部分主元法以处理零主元。
5.  与矩阵分解（LU）进行比较。

#### 测试用例

| 方程组                                  | 解          |
| --------------------------------------- | ----------- |
| $2x+y-z=8,\ -3x-y+2z=-11,\ -2x+y+2z=-3$ | $(2,3,-1)$  |
| $x+y=2,\ 2x-y=0$                        | $(2/3,4/3)$ |
| $x-y=1,\ 2x+y=4$                        | $(5/3,2/3)$ |

#### 复杂度

- 时间：$O(n^3)$
- 空间：$O(n^2)$

高斯消元法是线性代数的核心工具，所有高级求解器都建立在它的基础之上。
### 552 高斯-约当消元法

高斯-约当消元法扩展了高斯消元法，它持续进行化简过程，直到矩阵化为**简化行阶梯形**，而不仅仅是上三角形式。
这使得它非常适合求逆矩阵、测试线性相关性，以及无需回代直接求解线性方程组。

#### 我们要解决什么问题？

我们需要一个完整、系统的方法来：

- 求解 $A\mathbf{x} = \mathbf{b}$
- 求 $A^{-1}$（逆矩阵）
- 确定秩、零空间和主元

我们不会像高斯消元法那样止步于上三角系统，而是更进一步，使每个主元变为 $1$，并清除其上方和下方的所有元素。

#### 它是如何工作的（通俗解释）

1.  **构造增广矩阵**
    将 $A$ 和 $\mathbf{b}$ 组合：
    $[A | \mathbf{b}]$

2.  **前向消元**
    对于每个主元列：

    *   选择主元（必要时交换行）
    *   缩放该行使主元 = 1
    *   向下消元（使主元下方元素为零）

3.  **后向消元**
    对于每个主元列（从最后一个开始）：

    *   向上消元（使主元上方元素为零）

最终，$A$ 变为单位矩阵，右侧则给出解向量。

如果增广的是 $I$，则右侧变为 $A^{-1}$。

#### 示例

求解：

$$
\begin{cases}
x + y + z = 6 \\
2y + 5z = -4 \\
2x + 5y - z = 27
\end{cases}
$$

**步骤 1：** 写出增广矩阵

$$
\left[
\begin{array}{rrr|r}
1 & 1 & 1 & 6 \\
0 & 2 & 5 & -4 \\
2 & 5 & -1 & 27
\end{array}
\right]
$$

**步骤 2：** 在第一个主元下方消元

$R_3 \gets R_3 - 2R_1$

$$
\left[
\begin{array}{rrr|r}
1 & 1 & 1 & 6 \\
0 & 2 & 5 & -4 \\
0 & 3 & -3 & 15
\end{array}
\right]
$$

**步骤 3：** 以第 2 行为主元行

$R_2 \gets \tfrac{1}{2}R_2$

$$
\left[
\begin{array}{rrr|r}
1 & 1 & 1 & 6 \\
0 & 1 & \tfrac{5}{2} & -2 \\
0 & 3 & -3 & 15
\end{array}
\right]
$$

**步骤 4：** 在第二个主元下方和上方消元

$R_3 \gets R_3 - 3R_2,\quad R_1 \gets R_1 - R_2$

$$
\left[
\begin{array}{rrr|r}
1 & 0 & -\tfrac{3}{2} & 8 \\
0 & 1 & \tfrac{5}{2} & -2 \\
0 & 0 & -\tfrac{21}{2} & 21
\end{array}
\right]
$$

**归一化第三个主元**

$R_3 \gets -\tfrac{2}{21} R_3$

$$
\left[
\begin{array}{rrr|r}
1 & 0 & -\tfrac{3}{2} & 8 \\
0 & 1 & \tfrac{5}{2} & -2 \\
0 & 0 & 1 & -2
\end{array}
\right]
$$

**在第三个主元上方消元**

$R_1 \gets R_1 + \tfrac{3}{2}R_3,\quad R_2 \gets R_2 - \tfrac{5}{2}R_3$

$$
\left[
\begin{array}{rrr|r}
1 & 0 & 0 & 5 \\
0 & 1 & 0 & 3 \\
0 & 0 & 1 & -2
\end{array}
\right]
$$

**解：**

$$
x=5,\quad y=3,\quad z=-2
$$

#### 简易代码

**Python 版本**

```python
def gauss_jordan(a, b):
    n = len(a)
    # 用 b 增广 A
    for i in range(n):
        a[i].append(b[i])

    for i in range(n):
        # 主元选择
        max_row = max(range(i, n), key=lambda r: abs(a[r][i]))
        a[i], a[max_row] = a[max_row], a[i]

        # 归一化主元行
        pivot = a[i][i]
        for j in range(i, n + 1):
            a[i][j] /= pivot

        # 消去所有其他行
        for k in range(n):
            if k != i:
                factor = a[k][i]
                for j in range(i, n + 1):
                    a[k][j] -= factor * a[i][j]

    # 提取解
    return [a[i][n] for i in range(n)]

A = [[1, 1, 1],
     [0, 2, 5],
     [2, 5, -1]]
B = [6, -4, 27]
print(gauss_jordan(A, B))  # [5.0, 3.0, -2.0]
```

**C 版本**

```c
#include <stdio.h>
#include <math.h>

#define N 3

void gauss_jordan(double a[N][N+1]) {
    for (int i = 0; i < N; i++) {
        // 主元选择
        int max_row = i;
        for (int r = i + 1; r < N; r++)
            if (fabs(a[r][i]) > fabs(a[max_row][i]))
                max_row = r;
        for (int c = 0; c <= N; c++) {
            double tmp = a[i][c];
            a[i][c] = a[max_row][c];
            a[max_row][c] = tmp;
        }

        // 归一化主元行
        double pivot = a[i][i];
        for (int c = 0; c <= N; c++)
            a[i][c] /= pivot;

        // 消去其他行
        for (int r = 0; r < N; r++) {
            if (r != i) {
                double factor = a[r][i];
                for (int c = 0; c <= N; c++)
                    a[r][c] -= factor * a[i][c];
            }
        }
    }
}

int main() {
    double A[N][N+1] = {
        {1, 1, 1, 6},
        {0, 2, 5, -4},
        {2, 5, -1, 27}
    };
    gauss_jordan(A);
    for (int i = 0; i < N; i++)
        printf("x%d = %.2f\n", i + 1, A[i][N]);
}
```

#### 为何重要

高斯-约当消元法用途广泛：

- 无需回代直接求解
- 通过应用于 $[A|I]$ 来求矩阵逆
- 计算秩（主元个数）
- 测试线性相关性

它概念清晰，是高级线性代数例程的基础。

#### 一个温和的证明（为何有效）

每个行操作都对应于乘以一个可逆矩阵 $E_i$。
经过完全化简后：
$$
E_k \cdots E_1 [A | I] = [I | A^{-1}]
$$
因此，$A^{-1} = E_k \cdots E_1$。
该方法通过可逆操作将 $A$ 转化为 $I$，所以右侧演变为 $A^{-1}$。

对于 $A\mathbf{x} = \mathbf{b}$，用 $\mathbf{b}$ 增广 $A$ 即可得到唯一解向量。

#### 动手试试

1.  使用完全 RREF 求解 $A\mathbf{x}=\mathbf{b}$。
2.  用 $I$ 增广 $A$ 并计算 $A^{-1}$。
3.  计算非零行数以求秩。
4.  实现部分主元法以提高稳定性。
5.  与 LU 分解比较性能。

#### 测试用例

| 方程组                            | 解           |
| --------------------------------- | ------------ |
| $x+y+z=6,\ 2y+5z=-4,\ 2x+5y-z=27$ | $(5,3,-2)$   |
| $x+y=2,\ 3x-2y=1$                 | $(1,1)$      |
| $2x+y=5,\ 4x-2y=6$                | $(2,1)$      |

#### 复杂度

-   时间：$O(n^3)$
-   空间：$O(n^2)$

高斯-约当消元法是一个完整的求解器，在一次处理中，左侧产生单位矩阵，右侧产生解或逆矩阵。
### 553 LU 分解

LU 分解将一个矩阵 $A$ 分解为一个下三角矩阵 $L$ 和一个上三角矩阵 $U$ 的乘积：

$$
A = L \cdot U
$$

这种分解是数值线性代数的核心工具。一旦 $A$ 被分解，我们就可以通过前向替换和后向替换，针对多个右侧向量 $\mathbf{b}$ 快速求解方程组 $A\mathbf{x}=\mathbf{b}$。

#### 我们要解决什么问题？

我们希望高效且重复地求解 $A\mathbf{x}=\mathbf{b}$。

高斯消元法只能工作一次，但 LU 分解可以复用分解结果：

- 首先求解 $L\mathbf{y}=\mathbf{b}$（前向替换）
- 然后求解 $U\mathbf{x}=\mathbf{y}$（后向替换）

同样适用于：

- 行列式计算（$\det(A)=\det(L)\det(U)$）
- 矩阵求逆
- 带选主元的数值稳定性（$PA=LU$）

#### 它是如何工作的（通俗解释）

LU 分解执行与高斯消元法相同的操作，但将乘数记录在 $L$ 中。

1. 初始化

   * $L$ 初始化为单位矩阵
   * $U$ 初始化为 $A$ 的副本

2. 消去每个主元下方的元素

   * 对于 $i$ 从 $0$ 到 $n-1$：

     * 对于 $j>i$：
       $L[j][i] = U[j][i] / U[i][i]$
       从 $U$ 的第 $j$ 行减去 $L[j][i] \times$（第 $i$ 行）

结束时：

- $L$ 的对角线元素为 1，下方为乘数。
- $U$ 是上三角矩阵。

如果 $A$ 为了稳定性需要进行行交换，我们引入一个置换矩阵 $P$：
$$
PA = LU
$$

#### 示例

分解
$$
A = \begin{bmatrix}
2 & 3 & 1\
4 & 7 & 7\
-2 & 4 & 5
\end{bmatrix}
$$

步骤 1：主元 $a_{11}=2$

消去下方元素：

- 第 2 行：$L_{21}=4/2=2$ → Row2 = Row2 - 2*Row1
- 第 3 行：$L_{31}=-2/2=-1$ → Row3 = Row3 + Row1

$L = \begin{bmatrix}
1 & 0 & 0\
2 & 1 & 0\
-1 & 0 & 1
\end{bmatrix},\
U = \begin{bmatrix}
2 & 3 & 1\
0 & 1 & 5\
0 & 7 & 6
\end{bmatrix}$

步骤 2：主元 $U_{22}=1$

消去下方元素：

- 第 3 行：$L_{32}=7/1=7$ → Row3 = Row3 - 7*Row2

$L = \begin{bmatrix}
1 & 0 & 0\
2 & 1 & 0\
-1 & 7 & 1
\end{bmatrix},\
U = \begin{bmatrix}
2 & 3 & 1\
0 & 1 & 5\
0 & 0 & -29
\end{bmatrix}$

检查：$A = L \cdot U$

#### 精简代码（简易版本）

Python 版本

```python
def lu_decompose(A):
    n = len(A)
    L = [[0]*n for _ in range(n)]
    U = [[0]*n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1

    for i in range(n):
        # 上三角部分
        for k in range(i, n):
            U[i][k] = A[i][k] - sum(L[i][j] * U[j][k] for j in range(i))
        # 下三角部分
        for k in range(i + 1, n):
            L[k][i] = (A[k][i] - sum(L[k][j] * U[j][i] for j in range(i))) / U[i][i]
    return L, U

A = [
    [2, 3, 1],
    [4, 7, 7],
    [-2, 4, 5]
]
L, U = lu_decompose(A)
print("L =", L)
print("U =", U)
```

C 版本

```c
#include <stdio.h>

#define N 3

void lu_decompose(double A[N][N], double L[N][N], double U[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            L[i][j] = (i == j) ? 1 : 0;
            U[i][j] = 0;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int k = i; k < N; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += L[i][j] * U[j][k];
            U[i][k] = A[i][k] - sum;
        }
        for (int k = i + 1; k < N; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += L[k][j] * U[j][i];
            L[k][i] = (A[k][i] - sum) / U[i][i];
        }
    }
}

int main(void) {
    double A[N][N] = {{2,3,1},{4,7,7},{-2,4,5}}, L[N][N], U[N][N];
    lu_decompose(A, L, U);

    printf("L:\n");
    for (int i=0;i<N;i++){ for(int j=0;j<N;j++) printf("%6.2f ",L[i][j]); printf("\n"); }
    printf("U:\n");
    for (int i=0;i<N;i++){ for(int j=0;j<N;j++) printf("%6.2f ",U[i][j]); printf("\n"); }
}
```

#### 为什么它很重要

- 快速求解：复用 $LU$ 来求解多个 $\mathbf{b}$
- 行列式：$\det(A)=\prod_i U_{ii}$
- 逆矩阵：对每个 $i$ 求解 $LU\mathbf{x}=\mathbf{e}_i$
- 基础：是 Cholesky、Crout 和 Doolittle 变体的基础
- 稳定性：与选主元结合以提高鲁棒性（$PA=LU$）

#### 一个温和的证明（为什么它有效）

每个消元步骤对应于用初等下三角矩阵 $E_i$ 乘以 $A$。
所有步骤之后：
$$
U = E_k E_{k-1} \cdots E_1 A
$$
那么
$$
A = E_1^{-1} E_2^{-1} \cdots E_k^{-1} U
$$
令
$$
L = E_1^{-1}E_2^{-1}\cdots E_k^{-1}
$$
于是 $A = L U$，其中 $L$ 是下三角矩阵（单位对角线），$U$ 是上三角矩阵。

#### 亲自尝试

1.  手动对一个 $3\times3$ 矩阵进行 LU 分解。
2.  通过乘法验证 $A = L \cdot U$。
3.  通过前向替换 + 后向替换求解 $A\mathbf{x}=\mathbf{b}$。
4.  通过 $\prod U_{ii}$ 实现行列式计算。
5.  添加部分选主元（计算 $P,L,U$）。

#### 测试用例

| $A$                                                     | $L$                                                     | $U$                                                     |
| ------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------- |
| $\begin{bmatrix} 2 & 3 \\ 4 & 7 \end{bmatrix}$          | $\begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix}$          | $\begin{bmatrix} 2 & 3 \\ 0 & 1 \end{bmatrix}$          |
| $\begin{bmatrix} 1 & 1 \\ 2 & 3 \end{bmatrix}$          | $\begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix}$          | $\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$          |

#### 复杂度

- 时间：$O(n^3)$（分解）
- 求解：每个右侧向量 $O(n^2)$
- 空间：$O(n^2)$

LU 分解是数值线性代数的支柱，它将高斯消元法转变为一个可复用、模块化的工具。
### 554 乔里斯基分解

乔里斯基分解（Cholesky Decomposition）是对称正定矩阵的 LU 分解的一种特殊情况。
它将矩阵 $A$ 分解为一个下三角矩阵 $L$ 与其转置的乘积：

$$
A = L \cdot L^{T}
$$

对于对称正定矩阵，此方法的效率是 LU 分解的两倍，并且在数值上更加稳定，是优化、机器学习和统计学中的常用方法。

#### 我们要解决什么问题？

当 $A$ 是对称矩阵（$A=A^T$）且正定矩阵（对所有 $\mathbf{x}\neq0$ 有 $\mathbf{x}^T A \mathbf{x} > 0$）时，我们希望高效地求解 $A\mathbf{x}=\mathbf{b}$。

我们利用矩阵的对称性，将工作量减半，而不是使用一般的消元法。

一旦我们找到 $L$，就可以通过以下两步求解：

- 前代：$L\mathbf{y}=\mathbf{b}$
- 回代：$L^{T}\mathbf{x}=\mathbf{y}$

#### 它是如何工作的（通俗解释）

我们逐行（或逐列）构建 $L$，使用以下公式：

对于对角元素：
$$
L_{ii} = \sqrt{A_{ii} - \sum_{k=1}^{i-1}L_{ik}^2}
$$

对于非对角元素：
$$
L_{ij} = \frac{1}{L_{jj}}\Big(A_{ij} - \sum_{k=1}^{j-1}L_{ik}L_{jk}\Big), \quad i>j
$$

矩阵的上半部分就是 $L$ 的转置。

#### 示例

给定
$$
A =
\begin{bmatrix}
4 & 12 & -16\\
12 & 37 & -43\\
-16 & -43 & 98
\end{bmatrix}
$$

步骤 1: $L_{11} = \sqrt{4} = 2$

步骤 2: $L_{21} = 12/2 = 6$, $L_{31} = -16/2 = -8$

步骤 3: $L_{22} = \sqrt{37 - 6^2} = \sqrt{1} = 1$

步骤 4: $L_{32} = \frac{-43 - (-8)(6)}{1} = 5$

步骤 5: $L_{33} = \sqrt{98 - (-8)^2 - 5^2} = \sqrt{9} = 3$

因此：
$$
L =
\begin{bmatrix}
2 & 0 & 0\\
6 & 1 & 0\\
-8 & 5 & 3
\end{bmatrix}
,\quad
A = L \cdot L^{T}
$$

#### 简易代码实现

Python

```python
import math

def cholesky(A):
    n = len(A)
    L = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k]*L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(A[i][i] - s)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L

A = [
    [4, 12, -16],
    [12, 37, -43],
    [-16, -43, 98]
]

L = cholesky(A)
for row in L:
    print(row)
```

C

```c
#include <stdio.h>
#include <math.h>

#define N 3

void cholesky(double A[N][N], double L[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0;
            for (int k = 0; k < j; k++)
                sum += L[i][k] * L[j][k];
            if (i == j)
                L[i][j] = sqrt(A[i][i] - sum);
            else
                L[i][j] = (A[i][j] - sum) / L[j][j];
        }
    }
}

int main(void) {
    double A[N][N] = {
        {4, 12, -16},
        {12, 37, -43},
        {-16, -43, 98}
    }, L[N][N] = {0};

    cholesky(A, L);

    printf("L:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%8.3f ", L[i][j]);
        printf("\n");
    }
}
```

#### 为什么它很重要

- 工作量是 LU 分解的一半。
- 对于对称正定矩阵数值稳定。
- 是以下领域的基础：

  * 最小二乘回归
  * 卡尔曼滤波器
  * 高斯过程
  * 协方差矩阵分解

#### 一个温和的证明（为什么它有效）

对于一个对称正定矩阵 $A$，其所有顺序主子式都是正的。
这保证了每个对角主元（$A_{ii} - \sum L_{ik}^2$）都是正的，因此我们可以取平方根。

因此 $L$ 存在且唯一。

我们可以看到 $A = L L^T$，因为每个元素 $A_{ij}$ 都可以通过 $L$ 的第 $i$ 行和第 $j$ 行的点积求和来重现。

#### 亲自尝试

1.  手动分解一个 $3\times3$ 的对称正定矩阵。
2.  计算 $L L^T$ 进行验证。
3.  用它来求解 $A\mathbf{x}=\mathbf{b}$。
4.  与 LU 分解比较运行时间。
5.  在协方差矩阵（例如对称正定矩阵）上测试。

#### 测试用例

$$
A =
\begin{bmatrix}
25 & 15 & -5\\
15 & 18 &  0\\
-5 &  0 & 11
\end{bmatrix}
\Rightarrow
L =
\begin{bmatrix}
5 & 0 & 0\\
3 & 3 & 0\\
-1 & 1 & 3
\end{bmatrix}
$$

检查：$L L^T = A$

#### 复杂度

- 时间：$O(n^3/3)$
- 求解：$O(n^2)$
- 空间：$O(n^2)$

乔里斯基分解是处理快速、稳定、对称系统的首选方法，是数值分析和机器学习的基石。
### 555 QR 分解

QR 分解将一个矩阵 $A$ 分解为两个正交因子：

$$
A = Q \cdot R
$$

其中 $Q$ 是正交矩阵（$Q^T Q = I$），$R$ 是上三角矩阵。
这种分解是解决最小二乘问题、特征值计算和正交化任务的关键。

#### 我们要解决什么问题？

我们经常需要在 $A$ 不是方阵（例如 $m > n$）时求解 $A\mathbf{x} = \mathbf{b}$。
与使用正规方程 $(A^TA)\mathbf{x}=A^T\mathbf{b}$ 不同，QR 分解提供了更稳定的解法：

$$
A = Q R \implies R \mathbf{x} = Q^T \mathbf{b}
$$

无需构造 $A^T A$，这可以放大数值误差。

#### 它是如何工作的（通俗解释）

QR 分解逐步正交化 $A$ 的列：

1.  从 $A$ 的列 $a_1, a_2, \ldots, a_n$ 开始
2.  使用 Gram–Schmidt 过程构建正交基 $q_1, q_2, \ldots, q_n$
3.  归一化得到 $Q$ 的标准正交列
4.  计算 $R$ 作为投影系数

对于每个 $i$：
$$
r_{ii} = |a_i - \sum_{j=1}^{i-1}r_{ji}q_j|, \quad q_i = \frac{a_i - \sum_{j=1}^{i-1}r_{ji}q_j}{r_{ii}}
$$

简洁表示：

-   $Q$：标准正交基
-   $R$：上三角系数矩阵

变体：

-   经典 Gram–Schmidt (CGS)：简单但不稳定
-   修正 Gram–Schmidt (MGS)：更稳定
-   Householder 反射：数值精度最佳

#### 示例

令
$$
A =
\begin{bmatrix}
1 & 1 \
1 & -1 \
1 & 1
\end{bmatrix}
$$

步骤 1：取 $a_1 = (1, 1, 1)^T$

$$
q_1 = \frac{a_1}{|a_1|} = \frac{1}{\sqrt{3}}(1, 1, 1)
$$

步骤 2：从 $a_2$ 中移除投影

$$
r_{12} = q_1^T a_2 = \frac{1}{\sqrt{3}}(1 + (-1) + 1) = \frac{1}{\sqrt{3}}
$$

$$
u_2 = a_2 - r_{12} q_1 = (1, -1, 1) - \frac{1}{\sqrt{3}}\cdot\frac{1}{\sqrt{3}}(1,1,1) = (1-\tfrac{1}{3}, -1-\tfrac{1}{3}, 1-\tfrac{1}{3})
$$

$$
u_2 = \left(\frac{2}{3}, -\frac{4}{3}, \frac{2}{3}\right)
$$

归一化：

$$
q_2 = \frac{u_2}{|u_2|} = \frac{1}{\sqrt{8/3}} \left(\frac{2}{3}, -\frac{4}{3}, \frac{2}{3}\right) = \frac{1}{\sqrt{6}}(1, -2, 1)
$$

于是 $Q = [q_1\ q_2]$, $R = Q^T A$

#### 微型代码（简易版本）

Python

```python
import numpy as np

def qr_decompose(A):
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for i in range(n):
        v = A[:, i]
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], A[:, i])
            v = v - R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(v)
        Q[:, i] = v / R[i, i]
    return Q, R

A = [[1, 1], [1, -1], [1, 1]]
Q, R = qr_decompose(A)
print("Q =\n", Q)
print("R =\n", R)
```

C (简化版 Gram–Schmidt)

```c
#include <stdio.h>
#include <math.h>

#define M 3
#define N 2

void qr_decompose(double A[M][N], double Q[M][N], double R[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < M; k++)
            Q[k][i] = A[k][i];
        for (int j = 0; j < i; j++) {
            R[j][i] = 0;
            for (int k = 0; k < M; k++)
                R[j][i] += Q[k][j] * A[k][i];
            for (int k = 0; k < M; k++)
                Q[k][i] -= R[j][i] * Q[k][j];
        }
        R[i][i] = 0;
        for (int k = 0; k < M; k++)
            R[i][i] += Q[k][i] * Q[k][i];
        R[i][i] = sqrt(R[i][i]);
        for (int k = 0; k < M; k++)
            Q[k][i] /= R[i][i];
    }
}

int main(void) {
    double A[M][N] = {{1,1},{1,-1},{1,1}}, Q[M][N], R[N][N];
    qr_decompose(A, Q, R);

    printf("Q:\n");
    for(int i=0;i<M;i++){ for(int j=0;j<N;j++) printf("%8.3f ", Q[i][j]); printf("\n"); }

    printf("R:\n");
    for(int i=0;i<N;i++){ for(int j=0;j<N;j++) printf("%8.3f ", R[i][j]); printf("\n"); }
}
```

#### 为什么它很重要

-   最小二乘问题的数值稳定性
-   列空间的正交基
-   特征值算法（QR 迭代）
-   机器学习：回归、主成分分析 (PCA)
-   信号处理：正交化、投影

#### 一个温和的证明（为什么它有效）

每个列满秩且列独立的矩阵 $A$ 都可以写成：
$$
A = [a_1, a_2, \ldots, a_n] = [q_1, q_2, \ldots, q_n] R
$$

每个 $a_i$ 都可以表示为正交基向量 $q_j$ 的线性组合：
$$
a_i = \sum_{j=1}^{i} r_{ji} q_j
$$

将 $q_j$ 作为 $Q$ 的列收集起来，就得到 $A=QR$。

#### 亲自尝试

1.  手动正交化两个 3D 向量。
2.  验证 $Q^T Q = I$。
3.  计算 $R = Q^T A$。
4.  使用 $QR$ 求解一个超定方程组。
5.  比较经典 Gram–Schmidt 与修正 Gram–Schmidt。

#### 测试用例

$$
A =
\begin{bmatrix}
1 & 1\
1 & 0\
0 & 1
\end{bmatrix}
,\quad
Q =
\begin{bmatrix}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}}\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{6}}\
0 & \frac{2}{\sqrt{6}}
\end{bmatrix}
,\quad
R =
\begin{bmatrix}
\sqrt{2} & \frac{1}{\sqrt{2}}\
0 & \sqrt{\frac{3}{2}}
\end{bmatrix}
$$

检查：$A = Q R$

#### 复杂度

-   时间：$O(mn^2)$
-   空间：$O(mn)$
-   稳定性：高（尤其是使用 Householder 反射时）

QR 分解是你的正交罗盘，在最小二乘法、主成分分析和谱算法中，引导你找到稳定、几何化的解决方案。
### 556 矩阵求逆（高斯-约当法）

矩阵求逆旨在找到一个矩阵 $A^{-1}$，使得

$$
A \cdot A^{-1} = I
$$

此操作是求解方程组、变换空间和表达线性映射的基础。虽然在实际中很少直接使用求逆（求解 $A\mathbf{x}=\mathbf{b}$ 成本更低），但学习如何计算它揭示了线性代数本身的结构。

#### 我们要解决什么问题？

我们希望找到 $A^{-1}$，即能抵消 $A$ 作用的矩阵。
一旦我们有了 $A^{-1}$，任何方程组 $A\mathbf{x}=\mathbf{b}$ 都可以简单地通过下式求解：

$$
\mathbf{x} = A^{-1} \mathbf{b}
$$

但求逆运算仅当 $A$ 是方阵且非奇异（即 $\det(A)\neq0$）时才被定义。

#### 它是如何工作的（通俗解释）

高斯-约当法将单位矩阵 $I$ 增广到 $A$ 上，然后通过行操作将 $A$ 变换为 $I$。
无论 $I$ 在右侧变成了什么，那就是 $A^{-1}$。

逐步过程：

1.  构造增广矩阵 $[A | I]$
2.  应用行操作将 $A$ 变为 $I$
3.  右侧部分变为 $A^{-1}$

#### 示例

令
$$
A=
\begin{bmatrix}
2 & 1\\
5 & 3
\end{bmatrix}.
$$

用单位矩阵增广：
$$
\left[
\begin{array}{cc|cc}
2 & 1 & 1 & 0\\
5 & 3 & 0 & 1
\end{array}
\right].
$$

步骤 1: $R_1 \gets \tfrac{1}{2}R_1$
$$
\left[
\begin{array}{cc|cc}
1 & \tfrac{1}{2} & \tfrac{1}{2} & 0\\
5 & 3 & 0 & 1
\end{array}
\right].
$$

步骤 2: $R_2 \gets R_2 - 5R_1$
$$
\left[
\begin{array}{cc|cc}
1 & \tfrac{1}{2} & \tfrac{1}{2} & 0\\
0 & \tfrac{1}{2} & -\tfrac{5}{2} & 1
\end{array}
\right].
$$

步骤 3: $R_2 \gets 2R_2$
$$
\left[
\begin{array}{cc|cc}
1 & \tfrac{1}{2} & \tfrac{1}{2} & 0\\
0 & 1 & -5 & 2
\end{array}
\right].
$$

步骤 4: $R_1 \gets R_1 - \tfrac{1}{2}R_2$
$$
\left[
\begin{array}{cc|cc}
1 & 0 & 3 & -1\\
0 & 1 & -5 & 2
\end{array}
\right].
$$

因此
$$
A^{-1}=
\begin{bmatrix}
3 & -1\\
-5 & 2
\end{bmatrix}.
$$

验证：
$$
A\,A^{-1}=
\begin{bmatrix}
2 & 1\\
5 & 3
\end{bmatrix}
\begin{bmatrix}
3 & -1\\
-5 & 2
\end{bmatrix}
=
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}.
$$

#### 微型代码（简易版本）

Python

```python
def invert_matrix(A):
    n = len(A)
    # 用单位矩阵增广
    aug = [row + [int(i == j) for j in range(n)] for i, row in enumerate(A)]

    for i in range(n):
        # 使主元为 1
        pivot = aug[i][i]
        for j in range(2*n):
            aug[i][j] /= pivot

        # 消去其他行
        for k in range(n):
            if k != i:
                factor = aug[k][i]
                for j in range(2*n):
                    aug[k][j] -= factor * aug[i][j]

    # 提取逆矩阵
    return [row[n:] for row in aug]

A = [[2,1],[5,3]]
A_inv = invert_matrix(A)
for row in A_inv:
    print(row)
```

C

```c
#include <stdio.h>

#define N 2

void invert_matrix(double A[N][N], double I[N][N]) {
    double aug[N][2*N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            aug[i][j] = A[i][j];
            aug[i][j+N] = (i == j) ? 1 : 0;
        }
    }

    for (int i = 0; i < N; i++) {
        double pivot = aug[i][i];
        for (int j = 0; j < 2*N; j++)
            aug[i][j] /= pivot;

        for (int k = 0; k < N; k++) {
            if (k == i) continue;
            double factor = aug[k][i];
            for (int j = 0; j < 2*N; j++)
                aug[k][j] -= factor * aug[i][j];
        }
    }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            I[i][j] = aug[i][j+N];
}

int main(void) {
    double A[N][N] = {{2,1},{5,3}}, Inv[N][N];
    invert_matrix(A, Inv);
    for (int i=0;i<N;i++){ for(int j=0;j<N;j++) printf("%6.2f ",Inv[i][j]); printf("\n"); }
}
```

#### 为何重要

-   概念清晰：定义了"逆"的含义
-   求解方程组：$\mathbf{x}=A^{-1}\mathbf{b}$
-   几何意义：撤销线性变换
-   符号代数：例如变换、坐标变换

在实践中，你很少显式计算 $A^{-1}$，而是进行分解和求解。

#### 一个温和的证明（为何有效）

初等行操作对应于乘以初等矩阵 $E_i$。
如果

$$
E_k \cdots E_2 E_1 A = I
$$

那么

$$
A^{-1} = E_k \cdots E_2 E_1
$$

每个 $E_i$ 都是可逆的，因此它们的乘积也是可逆的。

#### 亲自尝试

1.  手动求一个 $3 \times 3$ 矩阵的逆。
2.  验证 $A \cdot A^{-1} = I$。
3.  观察如果 $\det(A)=0$ 会发生什么。
4.  与基于 LU 分解的求逆方法进行比较。
5.  计时并与求解 $A\mathbf{x}=\mathbf{b}$ 进行比较。

#### 测试用例

$$
A =
\begin{bmatrix}
1 & 2 & 3\
0 & 1 & 4\
5 & 6 & 0
\end{bmatrix}
\implies
A^{-1} =
\begin{bmatrix}
-24 & 18 & 5\
20 & -15 & -4\
-5 & 4 & 1
\end{bmatrix}
$$

验证：$A A^{-1} = I$

#### 复杂度

-   时间：$O(n^3)$
-   空间：$O(n^2)$

矩阵求逆是一面镜子：将变换反转回自身，并教导我们"求解"和"求逆"是同一操作的两个方面。
### 557 消元法求行列式

矩阵的行列式衡量其缩放因子和可逆性。通过高斯消元法，我们可以将矩阵转换为上三角形式，从而高效地计算行列式。在上三角形式中，行列式等于对角线元素的乘积，并根据行交换进行调整。

$$
\det(A) = (\text{符号}) \times \prod_{i=1}^{n} U_{ii}
$$

#### 我们要解决什么问题？

我们希望计算 $\det(A)$，而不使用递归展开（其复杂度为 $O(n!)$）。基于消元的方法可以在 $O(n^3)$ 时间内完成，与 LU 分解相同。

行列式告诉我们：

- 如果 $\det(A)=0$：$A$ 是奇异的（不可逆）
- 如果 $\det(A)\ne0$：$A$ 是可逆的
- 由 $A$ 定义的线性变换的体积缩放因子
- 方向（正 = 保持，负 = 翻转）

#### 它是如何工作的（通俗解释）

我们执行消元以形成上三角矩阵 $U$，同时记录行交换和缩放。

步骤：

1. 从 $A$ 开始
2. 对于每个主元行 $i$：

   * 如果主元为零，则交换行（每次交换翻转行列式符号）
   * 使用行操作消除下方的元素
     （加上倍数不会改变行列式）
3. 一旦 $U$ 成为上三角矩阵：
   $$
   \det(A) = (\pm 1) \times \prod_{i=1}^n U_{ii}
   $$

只有行交换会影响符号。

#### 示例

令
$$
A =
\begin{bmatrix}
2 & 1 & 3\
4 & 2 & 6\
1 & -1 & 1
\end{bmatrix}
$$

执行消元：

- 行2 = 行2 - 2×行1 → $[0, 0, 0]$
- 行3 = 行3 - ½×行1 → $[0, -1.5, -0.5]$

现在 $U =
\begin{bmatrix}
2 & 1 & 3\
0 & 0 & 0\
0 & -1.5 & -0.5
\end{bmatrix}$

出现零行 → $\det(A)=0$。

现在测试一个非奇异情况：

$$
B =
\begin{bmatrix}
2 & 1 & 1\
1 & 3 & 2\
1 & 0 & 0
\end{bmatrix}
$$

消元：

- 行2 = 行2 - ½×行1 → $[0, 2.5, 1.5]$
- 行3 = 行3 - ½×行1 → $[0, -0.5, -0.5]$
- 行3 = 行3 + 0.2×行2 → $[0, 0, -0.2]$

现在 $U_{11}=2$, $U_{22}=2.5$, $U_{33}=-0.2$

$$
\det(B) = 2 \times 2.5 \times (-0.2) = -1
$$

#### 精简代码（简易版本）

Python

```python
def determinant(A):
    n = len(A)
    A = [row[:] for row in A]  # 创建副本
    det = 1
    sign = 1

    for i in range(n):
        # 选主元
        if A[i][i] == 0:
            for j in range(i+1, n):
                if A[j][i] != 0:
                    A[i], A[j] = A[j], A[i]
                    sign *= -1
                    break
        pivot = A[i][i]
        if pivot == 0:
            return 0
        det *= pivot
        for j in range(i+1, n):
            factor = A[j][i] / pivot
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
    return sign * det

A = [[2,1,1],[1,3,2],[1,0,0]]
print("det =", determinant(A))
```

C

```c
#include <stdio.h>
#include <math.h>

#define N 3

double determinant(double A[N][N]) {
    double det = 1;
    int sign = 1;
    for (int i = 0; i < N; i++) {
        if (fabs(A[i][i]) < 1e-9) {
            int swap = -1;
            for (int j = i+1; j < N; j++) {
                if (fabs(A[j][i]) > 1e-9) { swap = j; break; }
            }
            if (swap == -1) return 0;
            for (int k = 0; k < N; k++) {
                double tmp = A[i][k];
                A[i][k] = A[swap][k];
                A[swap][k] = tmp;
            }
            sign *= -1;
        }
        det *= A[i][i];
        for (int j = i+1; j < N; j++) {
            double factor = A[j][i] / A[i][i];
            for (int k = i; k < N; k++)
                A[j][k] -= factor * A[i][k];
        }
    }
    return det * sign;
}

int main(void) {
    double A[N][N] = {{2,1,1},{1,3,2},{1,0,0}};
    printf("det = %.2f\n", determinant(A));
}
```

#### 为什么它很重要

- 可逆性检查（$\det(A)\ne0$ 意味着可逆）
- 线性变换下的体积缩放
- 方向检测（行列式的符号）
- 对以下方面至关重要：

  * 微积分中的雅可比行列式
  * 变量替换
  * 特征值计算

#### 一个温和的证明（为什么它有效）

高斯消元法将 $A$ 表示为：

$$
A = L U
$$

其中 $L$ 是单位下三角矩阵。
那么：
$$
\det(A) = \det(L)\det(U) = 1 \times \prod_{i=1}^n U_{ii}
$$

行交换每次将 $\det(A)$ 乘以 $-1$。
加上行的倍数不会改变 $\det(A)$。

#### 亲自尝试

1. 通过余子式展开和消元法计算 $\det(A)$，并进行比较。
2. 跟踪符号如何随交换而变化。
3. 用三角矩阵验证。
4. 观察奇异矩阵的行列式（应为 0）。
5. 比较基于 LU 的行列式计算。

#### 测试用例

| 矩阵 $A$                                            | 行列式 |
| ----------------------------------------------------- | ----------- |
| $\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$        | $-2$        |
| $\begin{bmatrix} 2 & 1 & 1 \\ 1 & 3 & 2 \\ 1 & 0 & 0 \end{bmatrix}$ | $-1$        |
| $\begin{bmatrix} 2 & 1 & 3 \\ 4 & 2 & 6 \\ 1 & -1 & 1 \end{bmatrix}$ | $0$         |


#### 复杂度

- 时间：$O(n^3)$
- 空间：$O(1)$（原地操作）

通过消元法求行列式将代数混沌转化为结构，每个主元都是一个体积因子，每次交换都是一次方向的翻转。
### 558 矩阵的秩

矩阵的秩告诉我们它有多少个独立的行或列，换句话说，就是其像（列空间）的维数。它是线性独立性、可解性和维数之间的桥梁。

我们可以使用高斯消元法高效地计算它：将矩阵变换为行阶梯形（REF），然后计算非零行的数量。

#### 我们要解决什么问题？

我们想要衡量矩阵中包含多少信息。
秩可以回答诸如以下问题：

- 列是否独立？
- $A\mathbf{x}=\mathbf{b}$ 是否有解？
- 行/列张成空间的维数是多少？

对于 $m\times n$ 矩阵 $A$：
$$
\text{rank}(A) = \text{REF 中主元（leading pivots）的数量}
$$

#### 它是如何工作的（通俗解释）

1.  应用高斯消元法将 $A$ 化为行阶梯形（REF）
2.  每个非零行代表一个主元（独立方向）
3.  计算主元的数量 → 这就是秩

如果是满秩：

- $\text{rank}(A)=n$ → 列独立
- $\text{rank}(A)<n$ → 有些列是相关的

在简化行阶梯形（RREF）中，主元是显式的 1，其上方和下方都是零。

#### 示例

令
$$
A =
\begin{bmatrix}
2 & 1 & 3\
4 & 2 & 6\
1 & -1 & 1
\end{bmatrix}
$$

执行消元：

行2 = 行2 - 2×行1 → $[0, 0, 0]$
行3 = 行3 - ½×行1 → $[0, -1.5, -0.5]$

结果：
$$
\begin{bmatrix}
2 & 1 & 3\
0 & -1.5 & -0.5\
0 & 0 & 0
\end{bmatrix}
$$

两个非零行 → 秩 = 2

所以这些行张成一个 2 维平面，而不是整个 $\mathbb{R}^3$。

#### 微型代码（简易版本）

Python

```python
def matrix_rank(A):
    n, m = len(A), len(A[0])
    A = [row[:] for row in A] # 创建副本
    rank = 0

    for col in range(m):
        # 寻找主元
        pivot_row = None
        for row in range(rank, n):
            if abs(A[row][col]) > 1e-9:
                pivot_row = row
                break
        if pivot_row is None:
            continue

        # 交换到当前秩所在的行
        A[rank], A[pivot_row] = A[pivot_row], A[rank]

        # 归一化主元行
        pivot = A[rank][col]
        A[rank] = [x / pivot for x in A[rank]]

        # 向下消元
        for r in range(rank+1, n):
            factor = A[r][col]
            A[r] = [A[r][c] - factor*A[rank][c] for c in range(m)]

        rank += 1
    return rank

A = [[2,1,3],[4,2,6],[1,-1,1]]
print("rank =", matrix_rank(A))
```

C (高斯消元法)

```c
#include <stdio.h>
#include <math.h>

#define N 3
#define M 3

int matrix_rank(double A[N][M]) {
    int rank = 0;
    for (int col = 0; col < M; col++) {
        int pivot = -1;
        for (int r = rank; r < N; r++) {
            if (fabs(A[r][col]) > 1e-9) { pivot = r; break; }
        }
        if (pivot == -1) continue;

        // 交换
        if (pivot != rank) {
            for (int c = 0; c < M; c++) {
                double tmp = A[rank][c];
                A[rank][c] = A[pivot][c];
                A[pivot][c] = tmp;
            }
        }

        // 归一化
        double div = A[rank][col];
        for (int c = 0; c < M; c++)
            A[rank][c] /= div;

        // 消元
        for (int r = rank + 1; r < N; r++) {
            double factor = A[r][col];
            for (int c = 0; c < M; c++)
                A[r][c] -= factor * A[rank][c];
        }

        rank++;
    }
    return rank;
}

int main(void) {
    double A[N][M] = {{2,1,3},{4,2,6},{1,-1,1}};
    printf("rank = %d\n", matrix_rank(A));
}
```

#### 为什么它很重要

- 张成空间的维数：秩 = 独立方向的数量
- 可解性：

  * 如果 $\text{rank}(A)=\text{rank}([A|\mathbf{b}])$，则方程组是相容的
  * 如果 $\text{rank}(A)<n$，则有无穷多解或无解
- 列空间：秩 = 像的维数
- 行空间：与列空间维数相同（秩 = 秩$^T$）

#### 一个温和的证明（为什么它有效）

行操作不改变行的张成空间，因此不改变秩。
一旦化为阶梯形，每个非零行都给行空间增加一个线性无关的向量。

因此：
$$
\text{rank}(A) = \text{主元位置的数量}
$$

#### 亲自尝试

1.  将一个 $3\times3$ 矩阵化为 REF。
2.  计算主元行的数量 → 秩。
3.  与列独立性检查进行比较。
4.  计算 rank($A$) 和 rank($A^T$)。
5.  用奇异矩阵（秩 < n）进行测试。

#### 测试用例

| 矩阵                                           | 秩 |
| ------------------------------------------------ | ---- |
| $\begin{bmatrix} 1 & 2 \\ 3 & 6 \end{bmatrix}$   | 1    |
| $\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$   | 2    |
| $\begin{bmatrix} 2 & 1 & 3 \\ 4 & 2 & 6 \\ 1 & -1 & 1 \end{bmatrix}$ | 2    |


#### 复杂度

- 时间复杂度：$O(n^3)$
- 空间复杂度：$O(n^2)$

秩是矩阵的灵魂，它告诉你其行和列中有多少个真正独立的声音在歌唱。
### 559 特征值幂法

幂法是一种简单的迭代算法，用于近似矩阵 $A$ 的主特征值（模最大的特征值）及其对应的特征向量。

它是“倾听”矩阵并找到其最强方向（在重复变换下保持稳定的方向）的最早、最直观的方法之一。

#### 我们要解决什么问题？

我们想要找到 $\lambda_{\max}$ 和 $\mathbf{v}_{\max}$，使得：

$$
A \mathbf{v}*{\max} = \lambda*{\max} \mathbf{v}_{\max}
$$

当 $A$ 很大或稀疏时，求解特征多项式是不可行的。
幂法提供了一种迭代的、低内存的近似方法，这在数值线性代数、PageRank 和 PCA 中至关重要。

#### 它是如何工作的（通俗解释）

1.  从一个随机向量 $\mathbf{x}_0$ 开始
2.  重复应用 $A$：$\mathbf{x}_{k+1} = A \mathbf{x}_k$
3.  每一步进行归一化以防止溢出
4.  当 $\mathbf{x}_k$ 稳定时，它就与主特征向量对齐
5.  对应的特征值由瑞利商近似给出：

$$
\lambda_k \approx \frac{\mathbf{x}_k^T A \mathbf{x}_k}{\mathbf{x}_k^T \mathbf{x}_k}
$$

因为 $A^k \mathbf{x}*0$ 放大了 $\mathbf{v}*{\max}$ 方向上的分量。

#### 算法（逐步说明）

给定 $A$ 和容差 $\varepsilon$：

1.  选择 $\mathbf{x}_0$（非零向量）
2.  重复：

    *   $\mathbf{y} = A \mathbf{x}$
    *   $\lambda = \max(|y_i|)$ 或 $\mathbf{x}^T A \mathbf{x}$
    *   $\mathbf{x} = \mathbf{y} / \lambda$
3.  当 $|\mathbf{x}_{k+1} - \mathbf{x}_k| < \varepsilon$ 时停止

返回 $\lambda, \mathbf{x}$。

收敛要求 $A$ 具有唯一的最大特征值。

#### 示例

令
$$
A =
\begin{bmatrix}
2 & 1\
1 & 3
\end{bmatrix}
$$

开始
$$
\mathbf{x}_0 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

1.  $\mathbf{y} = A\mathbf{x}_0 = [3, 4]^T$, 归一化: $\mathbf{x}_1 = [0.6, 0.8]$
2.  $\mathbf{y} = A\mathbf{x}_1 = [2.0, 3.0]^T$, 归一化: $\mathbf{x}_2 = [0.5547, 0.8321]$
3.  重复，$\mathbf{x}_k$ 收敛到特征向量 $[0.447, 0.894]$
4.  $\lambda \approx 3.618$（主特征值）

#### 微型代码（简易版本）

Python

```python
import numpy as np

def power_method(A, tol=1e-6, max_iter=1000):
    n = len(A)
    x = np.ones(n)
    lambda_old = 0

    for _ in range(max_iter):
        y = np.dot(A, x)
        lambda_new = np.max(np.abs(y))
        x = y / lambda_new
        if np.linalg.norm(x - y/np.linalg.norm(y)) < tol:
            break
        lambda_old = lambda_new
    return lambda_new, x / np.linalg.norm(x)

A = np.array([[2,1],[1,3]], dtype=float)
lam, v = power_method(A)
print("lambda ≈", lam)
print("v ≈", v)
```

C

```c
#include <stdio.h>
#include <math.h>

#define N 2
#define MAX_ITER 1000
#define TOL 1e-6

void power_method(double A[N][N], double v[N]) {
    double y[N], lambda = 0, lambda_old;
    for (int i=0;i<N;i++) v[i]=1.0;

    for (int iter=0;iter<MAX_ITER;iter++) {
        // y = A * v
        for (int i=0;i<N;i++) {
            y[i]=0;
            for (int j=0;j<N;j++) y[i]+=A[i][j]*v[j];
        }
        // 估计特征值
        lambda_old = lambda;
        lambda = fabs(y[0]);
        for (int i=1;i<N;i++) if (fabs(y[i])>lambda) lambda=fabs(y[i]);
        // 归一化
        for (int i=0;i<N;i++) v[i]=y[i]/lambda;

        // 检查收敛性
        double diff=0;
        for (int i=0;i<N;i++) diff+=fabs(y[i]-lambda*v[i]);
        if (diff<TOL) break;
    }

    printf("lambda ≈ %.6f\n", lambda);
    printf("v ≈ [%.3f, %.3f]\n", v[0], v[1]);
}

int main(void) {
    double A[N][N] = {{2,1},{1,3}}, v[N];
    power_method(A, v);
}
```

#### 为什么它很重要

-   找到主特征值/向量
-   适用于大型稀疏矩阵
-   是以下方法的基础：

    *   PageRank（谷歌）
    *   PCA（主成分分析）
    *   谱方法
    *   马尔可夫链稳态

#### 一个温和的证明（为什么它有效）

如果 $A$ 具有特征分解
$$
A = V \Lambda V^{-1}
$$
且 $\mathbf{x}_0 = c_1\mathbf{v}_1 + \dots + c_n\mathbf{v}_n$,

那么
$$
A^k \mathbf{x}_0 = c_1\lambda_1^k\mathbf{v}_1 + \cdots + c_n\lambda_n^k\mathbf{v}_n
$$

当 $k \to \infty$ 时，$\lambda_1^k$ 占主导地位，因此方向 $\to \mathbf{v}_1$。
归一化移除了尺度，留下了主特征向量。

#### 自己尝试

1.  应用于一个 $3\times3$ 对称矩阵。
2.  将结果与 `numpy.linalg.eig` 进行比较。
3.  在对角矩阵上尝试，验证它找到的是最大的对角线元素。
4.  观察如果特征值模相等时算法发散的情况。
5.  修改为反幂法以找到最小特征值。

#### 测试用例

$$
A =
\begin{bmatrix}
4 & 1\
2 & 3
\end{bmatrix},
\quad
\lambda_{\max} \approx 4.561,
\quad
\mathbf{v}_{\max} \approx
\begin{bmatrix}
0.788\
0.615
\end{bmatrix}
$$

#### 复杂度

-   时间：$O(kn^2)$（对于 $k$ 次迭代）
-   空间：$O(n^2)$

幂法是了解特征值的最简单窗口，每次迭代都使你的向量更接近矩阵最强的回响方向。
### 560 奇异值分解 (SVD)

奇异值分解 (SVD) 是线性代数中最强大、最通用的分解方法之一。它将任何矩阵$A$（方阵或矩形矩阵）表示为三个特殊矩阵的乘积：

$$
A = U \Sigma V^T
$$

-$U$：左奇异向量的正交矩阵
-$\Sigma$：奇异值（非负）的对角矩阵
-$V$：右奇异向量的正交矩阵

SVD 推广了特征分解，适用于任何矩阵（甚至非方阵），并揭示了从几何到数据压缩的深层结构。

#### 我们要解决什么问题？

我们希望将$A$分解为更简单、可解释的部分：

- 拉伸方向（通过$V$）
- 拉伸量（通过$\Sigma$）
- 最终的正交方向（通过$U$）

SVD 用于：

- 计算秩、零空间和值域
- 执行降维（PCA）
- 解决最小二乘问题
- 计算伪逆
- 在信号和图像中进行降噪

#### 它是如何工作的（通俗解释）

对于任意$A \in \mathbb{R}^{m \times n}$：

1. 计算$A^T A$（对称且半正定）
2. 求特征值$\lambda_i$和特征向量$v_i$
3. 奇异值$\sigma_i = \sqrt{\lambda_i}$
4. 构造$V = [v_1, \ldots, v_n]$
5. 对于非零$\sigma_i$，构造$U = \frac{1}{\sigma_i} A v_i$
6. 将$\sigma_i$放在对角线上组装$\Sigma$

结果：
$$
A = U \Sigma V^T
$$

如果$A$是$m \times n$：

-$U$：$m \times m$
-$\Sigma$：$m \times n$
-$V$：$n \times n$

#### 示例

令
$$
A=
\begin{bmatrix}
3 & 1\\
1 & 3
\end{bmatrix}.
$$

1. 计算 $A^{\mathsf T}A$：
$$
A^{\mathsf T}A
=
\begin{bmatrix}
3 & 1\\
1 & 3
\end{bmatrix}
\begin{bmatrix}
3 & 1\\
1 & 3
\end{bmatrix}
=
\begin{bmatrix}
10 & 6\\
6 & 10
\end{bmatrix}.
$$

2. 特征值： $16,\,4$  
   奇异值： $\sigma_1=4,\ \sigma_2=2$.

3. 右奇异向量：
$$
v_1=\frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix}, \quad
v_2=\frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}, \quad
V=[v_1,v_2].
$$

4. 计算 $u_i=\frac{1}{\sigma_i}A v_i$：
$$
u_1=\frac{1}{4}A v_1
=\frac{1}{4}
\begin{bmatrix}3 & 1\\1 & 3\end{bmatrix}
\frac{1}{\sqrt{2}}
\begin{bmatrix}1\\1\end{bmatrix}
=\frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix},
$$
$$
u_2=\frac{1}{2}A v_2
=\frac{1}{2}
\begin{bmatrix}3 & 1\\1 & 3\end{bmatrix}
\frac{1}{\sqrt{2}}
\begin{bmatrix}1\\-1\end{bmatrix}
=\frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}.
$$

因此
$$
U=\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1\\
1 & -1
\end{bmatrix}, \quad
\Sigma=
\begin{bmatrix}
4 & 0\\
0 & 2
\end{bmatrix}, \quad
V=U.
$$

验证： $A = U\,\Sigma\,V^{\mathsf T}$.


#### 微型代码（简易版本）

Python (NumPy 内置)

```python
import numpy as np

A = np.array([[3, 1], [1, 3]], dtype=float)
U, S, Vt = np.linalg.svd(A)

print("U =\n", U)
print("S =\n", np.diag(S))
print("V^T =\n", Vt)
```

Python (2×2 手动近似)

```python
import numpy as np

def svd_2x2(A):
    ATA = A.T @ A
    eigvals, V = np.linalg.eig(ATA)
    idx = np.argsort(-eigvals)
    eigvals, V = eigvals[idx], V[:, idx]
    S = np.sqrt(eigvals)
    U = (A @ V) / S
    return U, S, V

A = np.array([[3, 1], [1, 3]], float)
U, S, V = svd_2x2(A)
print("U =", U)
print("S =", S)
print("V =", V)
```

C (概念性框架)

```c
#include <stdio.h>
#include <math.h>

// 用于 2x2 SVD 演示
void svd_2x2(double A[2][2]) {
    double a=A[0][0], b=A[0][1], c=A[1][0], d=A[1][1];
    double ATA[2][2] = {
        {a*a + c*c, a*b + c*d},
        {a*b + c*d, b*b + d*d}
    };

    double trace = ATA[0][0] + ATA[1][1];
    double det = ATA[0][0]*ATA[1][1] - ATA[0][1]*ATA[1][0];
    double s = sqrt(trace*trace/4 - det);

    double sigma1 = sqrt(trace/2 + s);
    double sigma2 = sqrt(trace/2 - s);

    printf("奇异值: %.3f, %.3f\n", sigma1, sigma2);
}

int main(void) {
    double A[2][2] = {{3,1},{1,3}};
    svd_2x2(A);
}
```

#### 为何重要

- 通用分解：适用于任何矩阵
- 几何：描述拉伸和旋转
- 数据科学：PCA，低秩近似
- 数值分析：稳定的伪逆
- 信号处理：降噪，压缩

SVD 为诸如潜在语义分析 (LSA) 和用于降维的截断 SVD 等工具提供支持。

#### 一个温和的证明（为何有效）

由$A^T A = V \Lambda V^T$，其中$\Lambda$是对角矩阵，特征值$\lambda_i \ge 0$。
定义$\Sigma = \sqrt{\Lambda}$。
那么：

$$
A = (A V \Sigma^{-1}) \Sigma V^T
$$

令$U = A V \Sigma^{-1}$，它是正交的。
因此$A = U \Sigma V^T$。

#### 亲自尝试

1. 手动分解一个$3\times2$矩阵。
2. 验证$U^T U = I$,$V^T V = I$。
3. 将奇异值与$\sqrt{\text{$A^T A$ 的特征值}}$进行比较。
4. 使用 SVD 求秩（非零奇异值的个数）。
5. 从前一个奇异值重构$A$以实现压缩。

#### 测试用例

$$
A =
\begin{bmatrix}
3 & 1\\
1 & 3
\end{bmatrix},
\quad
U =
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1\\
1 & -1
\end{bmatrix},
\quad
\Sigma =
\begin{bmatrix}
4 & 0\\
0 & 2
\end{bmatrix},
\quad
V = U
$$

验证：$A = U \Sigma V^T$

#### 复杂度

- 时间：对于稠密矩阵为$O(mn^2)$
- 空间：$O(mn)$

SVD 是线性代数的万能钥匙，同时解锁了秩、特征结构、几何和数据。

# 第 57 节. FFT 与 NTT（快速变换）
### 561 离散傅里叶变换 (DFT)

离散傅里叶变换 (DFT) 是连接时域（或空间域）与频域的数学桥梁。它将一个有限数字序列分解为一系列复正弦波之和，从而揭示构成信号的隐藏频率。

DFT 是信号处理、音频压缩、图像滤波和多项式乘法的基础。它优美、精确且基础。

#### 我们要解决什么问题？

给定一个包含 $n$ 个复数的序列：

$$
x = (x_0, x_1, \ldots, x_{n-1})
$$

我们想要计算一个新的序列 $X = (X_0, X_1, \ldots, X_{n-1})$，用于描述 $x$ 中每个频率成分的含量。

该变换定义为：

$$
X_k = \sum_{j=0}^{n-1} x_j \cdot e^{-2\pi i \frac{jk}{n}}
\quad \text{其中 } k = 0, 1, \ldots, n-1
$$

每个 $X_k$ 衡量了频率为 $k/n$ 的复正弦波的振幅。

#### 它是如何工作的（通俗解释）

将你的输入序列想象成在钢琴上弹奏的一个和弦，它是多个频率的混合。
DFT "聆听" 这个信号，并告诉你存在哪些音符（频率）以及它们的强度如何。

其核心在于，它将信号与复正弦波 $e^{-2\pi i jk/n}$ 相乘，并对结果求和，以找出每个正弦波的贡献强度。

#### 示例

令 $n = 4$，且 $x = (1, 2, 3, 4)$

那么：

$$
X_k = \sum_{j=0}^{3} x_j \cdot e^{-2\pi i \frac{jk}{4}}
$$

计算每一项：

- $X_0 = 1 + 2 + 3 + 4 = 10$
- $X_1 = 1 + 2i - 3 - 4i = -2 - 2i$
- $X_2 = 1 - 2 + 3 - 4 = -2$
- $X_3 = 1 - 2i - 3 + 4i = -2 + 2i$

因此 DFT 结果为：

$$
X = [10, -2-2i, -2, -2+2i]
$$

#### 逆离散傅里叶变换

要从频率成分重建原始序列：

$$
x_j = \frac{1}{n} \sum_{k=0}^{n-1} X_k \cdot e^{2\pi i \frac{jk}{n}}
$$

正向变换和逆变换构成完美的一对，信息不会丢失。

#### 微型代码（简易版本）

Python (朴素 DFT)

```python
import cmath

def dft(x):
    n = len(x)
    X = []
    for k in range(n):
        s = 0
        for j in range(n):
            angle = -2j * cmath.pi * j * k / n
            s += x[j] * cmath.exp(angle)
        X.append(s)
    return X

# 示例
x = [1, 2, 3, 4]
X = dft(x)
print("DFT:", X)
```

Python (逆 DFT)

```python
def idft(X):
    n = len(X)
    x = []
    for j in range(n):
        s = 0
        for k in range(n):
            angle = 2j * cmath.pi * j * k / n
            s += X[k] * cmath.exp(angle)
        x.append(s / n)
    return x
```

C (朴素实现)

```c
#include <stdio.h>
#include <complex.h>
#include <math.h>

#define PI 3.14159265358979323846

void dft(int n, double complex x[], double complex X[]) {
    for (int k = 0; k < n; k++) {
        X[k] = 0;
        for (int j = 0; j < n; j++) {
            double angle = -2.0 * PI * j * k / n;
            X[k] += x[j] * cexp(I * angle);
        }
    }
}

int main(void) {
    int n = 4;
    double complex x[4] = {1, 2, 3, 4};
    double complex X[4];
    dft(n, x, X);

    for (int k = 0; k < n; k++)
        printf("X[%d] = %.2f + %.2fi\n", k, creal(X[k]), cimag(X[k]));
}
```

#### 为什么它很重要

- 将时域信号转换为频域洞察
- 实现滤波、压缩和模式检测
- 在信号处理、音频/视频、密码学、机器学习和基于 FFT 的算法中具有基础性地位

#### 一个温和的证明（为什么它有效）

DFT 利用了复指数的正交性：

$$
\sum_{j=0}^{n-1} e^{-2\pi i (k-l) j / n} =
\begin{cases}
n, & \text{如果 } k = l, \\
0, & \text{如果 } k \ne l.
\end{cases}
$$

这个性质确保我们可以唯一地分离出每个频率分量，并精确地逆变换。

#### 亲自尝试

1.  计算 $[1, 0, 1, 0]$ 的 DFT。
2.  验证应用 IDFT 是否能恢复原始序列。
3.  绘制 $X_k$ 的实部和虚部。
4.  观察 $X_0$ 如何表示输入值的平均值。
5.  对于较大的 $n$，比较其运行时间与 FFT 的运行时间。

#### 测试用例

| 输入 $x$      | 输出 $X$               |
| ------------- | ---------------------- |
| [1, 1, 1, 1]  | [4, 0, 0, 0]           |
| [1, 0, 1, 0]  | [2, 0, 2, 0]           |
| [1, 2, 3, 4]  | [10, -2-2i, -2, -2+2i] |

#### 复杂度

-   时间：$O(n^2)$
-   空间：$O(n)$

DFT 是揭示数据内部隐藏和谐性的数学显微镜，每个信号都变成了一曲频率之歌。
### 562 快速傅里叶变换 (FFT)

快速傅里叶变换 (FFT) 是计算数学中最重要的算法之一。它将原本是 $O(n^2)$ 操作的离散傅里叶变换 (DFT)，通过利用对称性和递归，减少到 $O(n \log n)$。它是数字信号处理、卷积、多项式乘法和频谱分析背后的核心。

#### 我们要解决什么问题？

我们想要计算与 DFT 相同的变换：

$$
X_k = \sum_{j=0}^{n-1} x_j \cdot e^{-2\pi i \frac{jk}{n}}
$$

但要更快。

直接实现需要循环遍历 $j$ 和 $k$，需要 $n^2$ 次操作。FFT 巧妙地将序列分成偶数和奇数部分来重新组织计算，每次将工作量减半。

#### 它是如何工作的（通俗解释）

FFT 使用了分治思想：

1.  将序列按索引分成偶数和奇数元素。
2.  计算每一半的 DFT（递归地）。
3.  利用单位复根的对称性组合结果。

当 $n$ 是 2 的幂时效果最好，因为我们可以每次都均匀分割，直到达到单个元素。

数学上：

令 $n = 2m$。则：

$$
X_k = E_k + \omega_n^k O_k \
X_{k+m} = E_k - \omega_n^k O_k
$$

其中：

-   $E_k$ 是偶数索引项的 DFT，
-   $O_k$ 是奇数索引项的 DFT，
-   $\omega_n = e^{-2\pi i / n}$ 是单位原根。

#### 示例

令 $n = 4$，$x = [1, 2, 3, 4]$

分割：

-   偶数：$[1, 3]$
-   奇数：$[2, 4]$

计算每个的 DFT（大小为 2）：

-   $E = [4, -2]$
-   $O = [6, -2]$

然后组合：

-   $X_0 = E_0 + O_0 = 10$
-   $X_1 = E_1 + \omega_4^1 O_1 = -2 + i(-2) = -2 - 2i$
-   $X_2 = E_0 - O_0 = -2$
-   $X_3 = E_1 - \omega_4^1 O_1 = -2 + 2i$

所以 $X = [10, -2-2i, -2, -2+2i]$，与 DFT 结果相同但计算更快。

#### 微型代码（递归 FFT）

Python (Cooley–Tukey FFT)

```python
import cmath

def fft(x):
    n = len(x)
    if n == 1:
        return x
    w_n = cmath.exp(-2j * cmath.pi / n)
    w = 1
    x_even = fft(x[0::2])
    x_odd = fft(x[1::2])
    X = [0] * n
    for k in range(n // 2):
        t = w * x_odd[k]
        X[k] = x_even[k] + t
        X[k + n // 2] = x_even[k] - t
        w *= w_n
    return X

# 示例
x = [1, 2, 3, 4]
X = fft(x)
print("FFT:", X)
```

C (递归 FFT)

```c
#include <stdio.h>
#include <complex.h>
#include <math.h>

#define PI 3.14159265358979323846

void fft(int n, double complex *x) {
    if (n <= 1) return;

    double complex even[n/2], odd[n/2];
    for (int i = 0; i < n/2; i++) {
        even[i] = x[2*i];
        odd[i] = x[2*i + 1];
    }

    fft(n/2, even);
    fft(n/2, odd);

    for (int k = 0; k < n/2; k++) {
        double complex w = cexp(-2.0 * I * PI * k / n);
        double complex t = w * odd[k];
        x[k] = even[k] + t;
        x[k + n/2] = even[k] - t;
    }
}

int main() {
    double complex x[4] = {1, 2, 3, 4};
    fft(4, x);
    for (int i = 0; i < 4; i++)
        printf("X[%d] = %.2f + %.2fi\n", i, creal(x[i]), cimag(x[i]));
}
```

#### 为什么它很重要

-   在毫秒级别将信号、图像和时间序列转换到频域。
-   是数字滤波器、卷积和压缩的基础。
-   是机器学习（谱方法）和物理模拟的核心。
-   使得多项式乘法能在 $O(n \log n)$ 时间内完成。

#### 一个温和的证明（为什么它有效）

诀窍在于认识到 DFT 矩阵由于 $\omega_n$ 的幂次而具有重复模式：

$$
W_n =
\begin{bmatrix}
1 & 1 & 1 & 1 \\
1 & \omega_n & \omega_n^{2} & \omega_n^{3} \\
1 & \omega_n^{2} & \omega_n^{4} & \omega_n^{6} \\
1 & \omega_n^{3} & \omega_n^{6} & \omega_n^{9}
\end{bmatrix}
$$

通过分割成偶数列和奇数列，我们递归地重用计算。这在每一层将问题规模减半，导致递归深度为 $\log n$，每层工作量为 $n$。

总计：$O(n \log n)$

#### 自己试试

1.  为 $n = 8$ 的随机值实现 FFT。
2.  与朴素 DFT 比较运行时间。
3.  绘制运行时间与 $n$ 的关系图（对数-对数坐标）。
4.  对正弦波应用 FFT 并可视化频率中的峰值。
5.  使用 FFT 卷积乘法两个多项式。

#### 测试用例

| 输入 $x$     | FFT $X$              |
| ------------ | -------------------- |
| [1, 1, 1, 1] | [4, 0, 0, 0]         |
| [1, 2, 3, 4] | [10, -2-2i, -2, -2+2i] |

#### 复杂度

-   时间：$O(n \log n)$
-   空间：$O(n)$（递归）或 $O(1)$（迭代）

FFT 将二次计算变成了近乎线性的计算，这一飞跃如此深远，永远地重塑了科学、工程和计算。
### 563 Cooley–Tukey FFT

Cooley–Tukey 算法是快速傅里叶变换（FFT）最广泛使用的实现。正是这个算法使得 FFT 变得实用、优雅且高效——通过递归地将 $O(n^2)$ 的离散傅里叶变换分解为更小的变换，将其复杂度降至 $O(n\log n)$。

该方法利用了分治原则和单位复根的对称性，使其成为几乎所有 FFT 库的核心。

#### 我们要解决什么问题？

我们希望高效地计算离散傅里叶变换：

$$
X_k=\sum_{j=0}^{n-1}x_j\cdot e^{-2\pi i\frac{jk}{n}}
$$

我们不直接计算所有项，而是将序列分割成更小的片段并重用结果——从而显著减少了冗余计算。

#### 它是如何工作的（通俗解释）

Cooley–Tukey 算法通过递归地将 DFT 分解为更小的 DFT 来工作：

1. 将输入序列按索引的奇偶性分割：
   * $x_{\text{偶数}}=[x_0,x_2,x_4,\ldots]$
   * $x_{\text{奇数}}=[x_1,x_3,x_5,\ldots]$

2. 计算两个大小为 $n/2$ 的较小 DFT：
   * $E_k=\text{DFT}(x_{\text{偶数}})$
   * $O_k=\text{DFT}(x_{\text{奇数}})$

3. 使用旋转因子将它们组合起来：
   * $\omega_n=e^{-2\pi i/n}$

组合步骤：

$$
X_k=E_k+\omega_n^kO_k
$$

$$
X_{k+n/2}=E_k-\omega_n^kO_k
$$

#### 示例（$n=8$）

假设 $x=[x_0,x_1,\ldots,x_7]$

1. 分割为偶数和奇数索引元素：
   * $[x_0,x_2,x_4,x_6]$
   * $[x_1,x_3,x_5,x_7]$

2. 递归计算每个子序列的 4 点 FFT。

3. 组合：
   * $X_k=E_k+\omega_8^kO_k$
   * $X_{k+4}=E_k-\omega_8^kO_k$

每一层递归都将问题规模减半，总共有 $\log_2 n$ 层。

#### 微型代码（递归实现）

Python

```python
import cmath

def cooley_tukey_fft(x):
    n = len(x)
    if n == 1:
        return x
    w_n = cmath.exp(-2j * cmath.pi / n)
    w = 1
    X_even = cooley_tukey_fft(x[0::2])
    X_odd = cooley_tukey_fft(x[1::2])
    X = [0] * n
    for k in range(n // 2):
        t = w * X_odd[k]
        X[k] = X_even[k] + t
        X[k + n // 2] = X_even[k] - t
        w *= w_n
    return X

# 示例
x = [1, 2, 3, 4, 5, 6, 7, 8]
X = cooley_tukey_fft(x)
print("FFT:", X)
```

C (递归)

```c
#include <stdio.h>
#include <complex.h>
#include <math.h>

#define PI 3.14159265358979323846

void fft(int n, double complex *x) {
    if (n <= 1) return;

    double complex even[n/2], odd[n/2];
    for (int i = 0; i < n/2; i++) {
        even[i] = x[2*i];
        odd[i] = x[2*i + 1];
    }

    fft(n/2, even);
    fft(n/2, odd);

    for (int k = 0; k < n/2; k++) {
        double complex w = cexp(-2.0 * I * PI * k / n);
        double complex t = w * odd[k];
        x[k] = even[k] + t;
        x[k + n/2] = even[k] - t;
    }
}

int main() {
    double complex x[8] = {1,2,3,4,5,6,7,8};
    fft(8, x);
    for (int i = 0; i < 8; i++)
        printf("X[%d] = %.2f + %.2fi\n", i, creal(x[i]), cimag(x[i]));
}
```

#### 为什么它很重要

- 将运行时间从 $O(n^2)$ 减少到 $O(n\log n)$
- 是音频/视频处理、图像变换和数字信号处理的基础
- 对于快速多项式乘法、信号滤波和频谱分析至关重要
- 被 FFT 库（如 FFTW、NumPy、cuFFT）普遍采用

#### 一个温和的证明（为什么它有效）

DFT 矩阵 $W_n$ 由单位复根构成：

$$
W_n =
\begin{bmatrix}
1 & 1 & 1 & \dots & 1 \\
1 & \omega_n & \omega_n^2 & \dots & \omega_n^{n-1} \\
1 & \omega_n^2 & \omega_n^4 & \dots & \omega_n^{2(n-1)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \omega_n^{n-1} & \omega_n^{2(n-1)} & \dots & \omega_n^{(n-1)^2}
\end{bmatrix}
$$

其中 $\omega_n = e^{-2\pi i / n}$ 是 $n$ 次单位根。

通过对偶数列和奇数列进行重新排序和分组，我们得到两个较小的 $W_{n/2}$ 块，再乘以旋转因子。
因此，每个递归步骤都将问题规模减半——从而产生 $\log_2 n$ 层计算。

#### 亲自尝试

1. 手动计算 $[1,2,3,4,5,6,7,8]$ 的 FFT（两层递归）。
2. 验证对称性 $X_{k+n/2}=E_k-\omega_n^kO_k$。
3. 比较 $n=1024$ 时与 DFT 的运行时间。
4. 可视化递归树。
5. 在正弦波输入上测试——检查频域的峰值。

#### 测试用例

| 输入             | 输出（幅度）                                      |
| ----------------- | ------------------------------------------------------- |
| [1,1,1,1,1,1,1,1] | [8,0,0,0,0,0,0,0]                                       |
| [1,2,3,4,5,6,7,8] | [36,-4+9.66i,-4+4i,-4+1.66i,-4,-4-1.66i,-4-4i,-4-9.66i] |

#### 复杂度

- 时间：$O(n\log n)$
- 空间：$O(n)$（递归版本）或 $O(1)$（迭代版本）

Cooley–Tukey FFT 不仅仅是一个巧妙的技巧——它是对对称性和结构的深刻洞察，改变了我们在科学和工程各个领域中计算、分析和理解信号的方式。
### 564 迭代式 FFT

迭代式 FFT 是 Cooley–Tukey 快速傅里叶变换的一种高效、非递归的实现方式。它不进行递归调用，而是通过原位计算完成变换，使用位反转置换对元素重新排序，并逐层迭代地合并结果。

这种方法被广泛用于高性能 FFT 库（如 FFTW 或 cuFFT），因为它对缓存友好、栈安全且可并行化。

#### 我们要解决什么问题？

递归式 FFT 很优雅，但函数调用和内存分配的开销会降低其速度。迭代式 FFT 消除了递归，直接在循环中执行所有相同的计算。

我们仍然希望计算：

$$
X_k=\sum_{j=0}^{n-1}x_j\cdot e^{-2\pi i\frac{jk}{n}}
$$

但我们将以迭代方式重用分治模式。

#### 它是如何工作的（通俗解释）

迭代式 FFT 以对数级阶段运行，每个阶段将子问题规模加倍：

1. 位反转置换
   重新排序输入，使索引遵循位反转顺序（镜像二进制数字）。
   例如：对于 $n=8$，索引 $[0,1,2,3,4,5,6,7]$ 变为 $[0,4,2,6,1,5,3,7]$。

2. 蝶形运算
   使用旋转因子组合成对的元素（像蝴蝶的翅膀）：
   $$
   t=\omega_n^k\cdot X_{\text{odd}}
   $$
   然后更新：
   $$
   X_{\text{even}}'=X_{\text{even}}+t
   $$
   $$
   X_{\text{odd}}'=X_{\text{even}}-t
   $$

3. 遍历各个阶段
   每个阶段将较小的 DFT 合并为较大的 DFT，不断将块大小加倍直到达到全长。

最终，$x$ 被原位变换为 $X$。

#### 示例（$n=8$）

1. 输入：$x=[x_0,x_1,x_2,x_3,x_4,x_5,x_6,x_7]$
2. 通过位反转重新排序：$[x_0,x_4,x_2,x_6,x_1,x_5,x_3,x_7]$
3. 阶段 1：组合对 $(x_0,x_1),(x_2,x_3),\ldots$
4. 阶段 2：组合大小为 4 的块
5. 阶段 3：组合大小为 8 的块

每个阶段乘以旋转因子 $\omega_n^k=e^{-2\pi i k/n}$，迭代执行蝶形运算。

#### 微型代码（迭代式 FFT）

Python（原位迭代式 FFT）

```python
import cmath

def bit_reverse(x):
    n = len(x)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            x[i], x[j] = x[j], x[i]

def iterative_fft(x):
    n = len(x)
    bit_reverse(x)
    size = 2
    while size <= n:
        w_m = cmath.exp(-2j * cmath.pi / size)
        for k in range(0, n, size):
            w = 1
            for j in range(size // 2):
                t = w * x[k + j + size // 2]
                u = x[k + j]
                x[k + j] = u + t
                x[k + j + size // 2] = u - t
                w *= w_m
        size *= 2
    return x

# 示例
x = [1, 2, 3, 4, 5, 6, 7, 8]
X = iterative_fft([complex(a, 0) for a in x])
print("FFT:", X)
```

C（原位迭代式 FFT）

```c
#include <stdio.h>
#include <complex.h>
#include <math.h>

#define PI 3.14159265358979323846

void bit_reverse(double complex *x, int n) {
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) {
            double complex temp = x[i];
            x[i] = x[j];
            x[j] = temp;
        }
    }
}

void iterative_fft(double complex *x, int n) {
    bit_reverse(x, n);
    for (int size = 2; size <= n; size <<= 1) {
        double angle = -2 * PI / size;
        double complex w_m = cos(angle) + I * sin(angle);
        for (int k = 0; k < n; k += size) {
            double complex w = 1;
            for (int j = 0; j < size/2; j++) {
                double complex t = w * x[k + j + size/2];
                double complex u = x[k + j];
                x[k + j] = u + t;
                x[k + j + size/2] = u - t;
                w *= w_m;
            }
        }
    }
}
```

#### 为什么它很重要

- 消除了递归开销，实际运行更快
- 原位操作，无需额外数组
- 是 GPU FFT、DSP 硬件和实时系统的基础
- 支持高效批量 FFT（可并行化的循环）

#### 一个温和的证明（为什么它有效）

迭代式 FFT 只是自底向上地遍历同一棵递归树。
位反转确保输入按递归顺序排列。
每个阶段将大小为 $2^k$ 的 DFT 合并为 $2^{k+1}$，使用相同的蝶形方程：

$$
X_k=E_k+\omega_n^kO_k,\quad X_{k+n/2}=E_k-\omega_n^kO_k
$$

通过重复 $\log_2 n$ 个阶段，我们计算出完整的 FFT。

#### 亲自尝试

1. 对 $[1,2,3,4,5,6,7,8]$ 应用迭代式 FFT。
2. 打印位反转后的数组，确认顺序。
3. 将结果与递归式 FFT 进行比较。
4. 绘制 $n=2^k$ 时的运行时间图。
5. 扩展到逆 FFT（改变指数符号，除以 $n$）。

#### 测试用例

| 输入               | 输出（幅度）                                          |
| ------------------ | ----------------------------------------------------- |
| [1,1,1,1,1,1,1,1] | [8,0,0,0,0,0,0,0]                                     |
| [1,2,3,4,5,6,7,8] | [36,-4+9.66i,-4+4i,-4+1.66i,-4,-4-1.66i,-4-4i,-4-9.66i] |

#### 复杂度

- 时间：$O(n\log n)$
- 空间：$O(1)$（原位）

迭代式 FFT 用原始效率取代了优雅的递归——将同样的数学之美带入紧凑、极速的循环中。
### 565 逆快速傅里叶变换 (IFFT)

逆快速傅里叶变换 (IFFT) 是 FFT 的镜像操作。它将信号从频域转换回时域，完美地重建原始序列。
FFT 将信号分解为频率成分，而 IFFT 则利用这些相同的成分重新组装信号，完成一个完整的往返变换。

#### 我们要解决什么问题？

给定傅里叶系数 $X_0, X_1, \ldots, X_{n-1}$，我们希望重建原始信号 $x_0, x_1, \ldots, x_{n-1}$。

逆 DFT 的定义是：

$$
x_j=\frac{1}{n}\sum_{k=0}^{n-1}X_k\cdot e^{2\pi i\frac{jk}{n}}
$$

IFFT 允许我们在频域执行操作（如滤波或卷积）后，恢复时域数据。

#### 它是如何工作的（通俗解释）

IFFT 的工作方式与 FFT 类似，具有相同的蝶形结构，相同的递归或迭代过程，但使用了共轭的旋转因子，并在最后进行 $1/n$ 的缩放。

计算 IFFT 的步骤：

1.  对所有频率分量取复共轭。
2.  运行一次正向 FFT。
3.  再次取复共轭。
4.  将每个结果除以 $n$。

这利用了以下事实：
$$
\text{IFFT}(X)=\frac{1}{n}\cdot\overline{\text{FFT}(\overline{X})}
$$

#### 示例

令 $X=[10,-2-2i,-2,-2+2i]$

1.  取共轭：$[10,-2+2i,-2,-2-2i]$
2.  对共轭值进行 FFT $\to [4,8,12,16]$
3.  再次取共轭：$[4,8,12,16]$
4.  除以 $n=4$：$[1,2,3,4]$

恢复的原始信号：$x=[1,2,3,4]$

确认完美重建。

#### 微型代码（简易版本）

Python (通过 FFT 实现 IFFT)

```python
import cmath

def fft(x):
    n = len(x)
    if n == 1:
        return x
    w_n = cmath.exp(-2j * cmath.pi / n)
    w = 1
    X_even = fft(x[0::2])
    X_odd = fft(x[1::2])
    X = [0] * n
    for k in range(n // 2):
        t = w * X_odd[k]
        X[k] = X_even[k] + t
        X[k + n // 2] = X_even[k] - t
        w *= w_n
    return X

def ifft(X):
    n = len(X)
    X_conj = [x.conjugate() for x in X]
    x = fft(X_conj)
    x = [val.conjugate() / n for val in x]
    return x

# 示例
X = [10, -2-2j, -2, -2+2j]
print("IFFT:", ifft(X))
```

C (使用 FFT 结构实现 IFFT)

```c
#include <stdio.h>
#include <complex.h>
#include <math.h>

#define PI 3.14159265358979323846

void fft(int n, double complex *x, int inverse) {
    if (n <= 1) return;

    double complex even[n/2], odd[n/2];
    for (int i = 0; i < n/2; i++) {
        even[i] = x[2*i];
        odd[i] = x[2*i + 1];
    }

    fft(n/2, even, inverse);
    fft(n/2, odd, inverse);

    double sign = inverse ? 2.0 * PI : -2.0 * PI;
    for (int k = 0; k < n/2; k++) {
        double complex w = cexp(I * sign * k / n);
        double complex t = w * odd[k];
        x[k] = even[k] + t;
        x[k + n/2] = even[k] - t;
        if (inverse) {
            x[k] /= 2;
            x[k + n/2] /= 2;
        }
    }
}

int main() {
    double complex X[4] = {10, -2-2*I, -2, -2+2*I};
    fft(4, X, 1); // 逆 FFT
    for (int i = 0; i < 4; i++)
        printf("x[%d] = %.2f + %.2fi\n", i, creal(X[i]), cimag(X[i]));
}
```

#### 为什么它很重要

-   从频率分量恢复时域数据
-   用于信号重建、卷积和滤波器设计
-   保证与 FFT 的完美可逆性
-   是压缩算法、图像恢复和物理模拟的核心

#### 一个温和的证明（为什么它有效）

DFT 和 IFFT 矩阵是 Hermitian 逆矩阵：

$$
F_{n}^{-1}=\frac{1}{n}\overline{F_n}^T
$$

因此，先应用 FFT 再应用 IFFT 得到恒等变换：

$$
\text{IFFT}(\text{FFT}(x))=x
$$

取共轭会翻转指数的符号，而乘以 $1/n$ 则确保了归一化。

#### 自己动手试试

1.  计算 $[1,2,3,4]$ 的 FFT，然后应用 IFFT。
2.  将重建结果与原始信号进行比较。
3.  通过将 $X$ 的高频分量置零来修改，观察平滑效果。
4.  使用 IFFT 处理多项式乘法的结果。
5.  可视化 IFFT 前后的幅度和相位。

#### 测试用例

| 输入 $X$           | 输出 $x$        |
| ------------------ | --------------- |
| [10,-2-2i,-2,-2+2i] | [1,2,3,4]       |
| [4,0,0,0]          | [1,1,1,1]       |
| [8,0,0,0,0,0,0,0]  | [1,1,1,1,1,1,1,1] |

#### 复杂度

-   时间：$O(n\log n)$
-   空间：$O(n)$

IFFT 是 FFT 的镜像步骤，结构相同，流程相反。它完成了从时域到频域再返回的循环，不丢失任何一点信息。
### 566 通过 FFT 进行卷积

卷积是数学、信号处理和计算机科学中最基本的运算之一。它将两个序列组合成一个，通过移位混合信息。
但直接计算卷积需要 $O(n^2)$ 的时间。FFT 卷积定理为我们提供了一个捷径：通过转换到频域，我们可以在 $O(n\log n)$ 的时间内完成。

#### 我们要解决什么问题？

给定长度为 $n$ 和 $m$ 的两个序列 $a$ 和 $b$，我们想要它们的卷积 $c = a * b$：

$$
c_k=\sum_{i=0}^{k}a_i\cdot b_{k-i}
$$

其中 $k=0,\ldots,n+m-2$。

这出现在：

- 信号处理（滤波、相关）
- 多项式乘法
- 模式匹配
- 概率分布（随机变量之和）

直接计算是 $O(nm)$。使用 FFT，我们可以将其减少到 $O(n\log n)$。

#### 它是如何工作的（通俗解释）

卷积定理指出：

$$
\text{DFT}(a*b)=\text{DFT}(a)\cdot\text{DFT}(b)
$$

也就是说，时域中的卷积等于频域中的逐点乘法。

因此，要快速计算卷积：

1. 将两个序列填充到大小 $N\ge n+m-1$（2 的幂次）。
2. 计算两者的 FFT：$A=\text{FFT}(a)$, $B=\text{FFT}(b)$。
3. 逐点相乘：$C_k=A_k\cdot B_k$。
4. 应用逆 FFT：$c=\text{IFFT}(C)$。
5. 取实部（舍入小误差）。

#### 示例

令 $a=[1,2,3]$, $b=[4,5,6]$

预期的卷积（手动计算）：

$$
c=[1\cdot4,1\cdot5+2\cdot4,1\cdot6+2\cdot5+3\cdot4,2\cdot6+3\cdot5,3\cdot6]
$$

$$
c=[4,13,28,27,18]
$$

FFT 方法：

1. 将 $a,b$ 填充到长度 8
2. $A=\text{FFT}(a)$, $B=\text{FFT}(b)$
3. $C=A\cdot B$
4. $c=\text{IFFT}(C)$
5. 舍入到整数 $\Rightarrow [4,13,28,27,18]$

完全匹配。

#### 微型代码（简易版本）

Python (FFT 卷积)

```python
import cmath

def fft(x):
    n = len(x)
    if n == 1:
        return x
    w_n = cmath.exp(-2j * cmath.pi / n)
    w = 1
    X_even = fft(x[0::2])
    X_odd = fft(x[1::2])
    X = [0] * n
    for k in range(n // 2):
        t = w * X_odd[k]
        X[k] = X_even[k] + t
        X[k + n // 2] = X_even[k] - t
        w *= w_n
    return X

def ifft(X):
    n = len(X)
    X_conj = [x.conjugate() for x in X]
    x = fft(X_conj)
    return [v.conjugate()/n for v in x]

def convolution(a, b):
    n = 1
    while n < len(a) + len(b) - 1:
        n *= 2
    a += [0]*(n - len(a))
    b += [0]*(n - len(b))
    A = fft(a)
    B = fft(b)
    C = [A[i]*B[i] for i in range(n)]
    c = ifft(C)
    return [round(v.real) for v in c]

# 示例
a = [1, 2, 3]
b = [4, 5, 6]
print(convolution(a, b))  # [4, 13, 28, 27, 18]
```

C (FFT 卷积框架)

```c
// 假设 fft() 和 ifft() 函数已实现
void convolution(int n, double complex *a, double complex *b, double complex *c) {
    int size = 1;
    while (size < 2 * n) size <<= 1;

    // 填充数组
    for (int i = n; i < size; i++) {
        a[i] = 0;
        b[i] = 0;
    }

    fft(size, a, 0);
    fft(size, b, 0);
    for (int i = 0; i < size; i++)
        c[i] = a[i] * b[i];
    fft(size, c, 1); // 逆 FFT（已缩放）
}
```

#### 为什么它很重要

- 将缓慢的 $O(n^2)$ 卷积转变为 $O(n\log n)$
- 是多项式乘法、数字滤波器、信号相关、神经网络层、概率和中的核心技术
- 用于大整数乘法（Karatsuba、Schönhage–Strassen）

#### 一个温和的证明（为什么它有效）

DFT 将卷积转换为乘法：

$$
C=\text{DFT}(a*b)=\text{DFT}(a)\cdot\text{DFT}(b)
$$

因为 DFT 矩阵对角化了循环移位，它将卷积（涉及移位和求和）转换为频率的独立乘法。

然后应用 $\text{IDFT}$ 恢复精确结果。

#### 自己动手试试

1.  手动卷积 $[1,2,3]$ 和 $[4,5,6]$，然后通过 FFT 验证。
2.  将输入填充到 2 的幂次，并观察中间数组。
3.  尝试更长的多项式（长度 1000），测量速度差异。
4.  绘制输入、它们的频谱和输出。
5.  实现模卷积（例如使用数论变换）。

#### 测试用例

| $a$     | $b$     | $a*b$           |
| ------- | ------- | --------------- |
| [1,2,3] | [4,5,6] | [4,13,28,27,18] |
| [1,1,1] | [1,2,3] | [1,3,6,5,3]     |
| [2,0,1] | [3,4]   | [6,8,3,4]       |

#### 复杂度

- 时间：$O(n\log n)$
- 空间：$O(n)$

FFT 卷积是计算机如何快速相乘长数字、组合信号和合并模式的方法，比你手动操作快得多。
### 567 数论变换 (NTT)

数论变换 (NTT) 是快速傅里叶变换 (FFT) 在模算术下的版本。
它用有限模数下的整数取代了复数，使得所有计算都能精确完成（无浮点误差），非常适合密码学、组合数学和竞技编程。

FFT 使用单位复根 $e^{2\pi i/n}$，而 NTT 使用模素数下的本原单位根。

#### 我们要解决什么问题？

我们希望使用模算术而非浮点数，来精确执行多项式乘法（或卷积）。

给定两个多项式：

$$
A(x)=\sum_{i=0}^{n-1}a_i x^i,\quad B(x)=\sum_{i=0}^{m-1}b_i x^i
$$

它们的乘积：

$$
C(x)=A(x)\cdot B(x)=\sum_{k=0}^{n+m-2}c_kx^k
$$

其中

$$
c_k=\sum_{i=0}^{k}a_i\cdot b_{k-i}
$$

我们希望在模 $M$ 下高效地计算所有 $c_k$。

#### 核心思想

如果我们能找到一个本原 $n$ 次单位根 $g$，使得：

$$
g^n\equiv1\pmod M
$$

且

$$
g^k\not\equiv1\pmod M,\ \text{对于 } 0<k<n
$$

那么卷积定理在模算术下同样成立。

然后，我们可以像定义 FFT 一样定义 NTT：

$$
A_k=\sum_{j=0}^{n-1}a_j\cdot g^{jk}\pmod M
$$

并使用 $g^{-1}$ 和 $n^{-1}\pmod M$ 来求逆变换。

#### 工作原理（通俗解释）

1.  选择一个模数 $M$，其中 $M=k\cdot 2^m+1$（一个对 FFT 友好的素数）。
    例如：$M=998244353=119\cdot2^{23}+1$
2.  找到本原根 $g$（例如 $g=3$）。
3.  执行类似 FFT 的蝶形运算，但使用模乘法。
4.  对于逆变换，使用 $g^{-1}$ 并通过模逆元除以 $n$。

这确保了精确的结果，没有舍入误差。

#### 示例

令 $M=17$, $n=4$, $g=4$，因为 $4^4\equiv1\pmod{17}$。

输入：$a=[1,2,3,4]$

计算 NTT：

$$
A_k=\sum_{j=0}^{3}a_j\cdot g^{jk}\pmod{17}
$$

结果：$A=[10,2,16,15]$

逆 NTT（使用 $g^{-1}=13$, $n^{-1}=13$ mod 17）：

$$
a_j=\frac{1}{n}\sum_{k=0}^{3}A_k\cdot g^{-jk}\pmod{17}
$$

恢复得到：$[1,2,3,4]$

#### 微型代码（NTT 模板）

Python

```python
MOD = 998244353
ROOT = 3  # 本原根

def modpow(a, e, m):
    res = 1
    while e:
        if e & 1:
            res = res * a % m
        a = a * a % m
        e >>= 1
    return res

def ntt(a, invert):
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    len_ = 2
    while len_ <= n:
        wlen = modpow(ROOT, (MOD - 1) // len_, MOD)
        if invert:
            wlen = modpow(wlen, MOD - 2, MOD)
        for i in range(0, n, len_):
            w = 1
            for j in range(len_ // 2):
                u = a[i + j]
                v = a[i + j + len_ // 2] * w % MOD
                a[i + j] = (u + v) % MOD
                a[i + j + len_ // 2] = (u - v + MOD) % MOD
                w = w * wlen % MOD
        len_ <<= 1
    if invert:
        inv_n = modpow(n, MOD - 2, MOD)
        for i in range(n):
            a[i] = a[i] * inv_n % MOD

def multiply(a, b):
    n = 1
    while n < len(a) + len(b):
        n <<= 1
    a += [0]*(n - len(a))
    b += [0]*(n - len(b))
    ntt(a, False)
    ntt(b, False)
    for i in range(n):
        a[i] = a[i] * b[i] % MOD
    ntt(a, True)
    return a
```

#### 为何重要

-   精确的模运算结果，无浮点舍入
-   支持多项式乘法、组合变换、大整数乘法
-   在密码学、格算法和竞技编程中至关重要
-   用于现代方案（如基于 NTT 的同态加密）

#### 一个温和的证明（为何有效）

NTT 矩阵 $W_n$ 满足 $W_{jk}=g^{jk}\pmod M$，并且：

$$
W_n^{-1}=\frac{1}{n}\overline{W_n}
$$

因为 $g$ 是一个本原 $n$ 次单位根，$W_n$ 的列在模 $M$ 下正交，且模逆元存在（因为 $M$ 是素数）。
因此，正变换和逆变换完美地互为逆运算。

#### 动手尝试

1.  使用 $M=17$, $g=4$, $a=[1,2,3,4]$，手动计算 NTT。
2.  在模 $17$ 下，使用 NTT 计算 $[1,2,3]$ 和 $[4,5,6]$ 的乘积。
3.  测试模 $998244353$，长度 $8$。
4.  比较 FFT（浮点）与 NTT（模运算）的结果。
5.  观察无舍入的精确性。

#### 测试用例

| $a$     | $b$     | $a*b$ (mod 17)                          |
| ------- | ------- | --------------------------------------- |
| [1,2,3] | [4,5,6] | [4,13,28,27,18] mod 17 = [4,13,11,10,1] |
| [1,1]   | [1,1]   | [1,2,1]                                 |

#### 复杂度

-   时间：$O(n\log n)$
-   空间：$O(n)$

NTT 是整数域上的 FFT，同样的优雅，不同的世界。它将代数、数论和计算融合成一个无缝的精确性引擎。
### 568 逆数论变换 (INTT)

逆数论变换 (INTT) 是 NTT 的逆操作。它将数据从频域带回系数域，完全重建原始序列。
就像 IFFT 之于 FFT 一样，INTT 使用单位根的逆元和一个模缩放因子来撤销模变换。

#### 我们要解决什么问题？

给定一个经过 NTT 变换的序列 $A=[A_0,A_1,\ldots,A_{n-1}]$，我们希望在模 $M$ 下恢复原始序列 $a=[a_0,a_1,\ldots,a_{n-1}]$。

其定义与逆 DFT 类似：

$$
a_j = n^{-1} \sum_{k=0}^{n-1} A_k \cdot g^{-jk} \pmod M
$$

其中：

- $g$ 是模 $M$ 下的一个 $n$ 次本原单位根
- $n^{-1}$ 是 $n$ 在模 $M$ 下的模逆元

#### 它是如何工作的（通俗解释）

INTT 遵循与 NTT 相同的蝶形结构，但是：

1. 使用逆旋转因子 $g^{-1}$ 代替 $g$
2. 在所有阶段完成后，将每个元素乘以 $n^{-1} \pmod M$

这确保了完美的逆变换：

$$
\text{INTT}(\text{NTT}(a)) = a
$$

#### 示例

令 $M=17$, $n=4$, $g=4$
则 $g^{-1}=13$（因为 $4\cdot13\equiv1\pmod{17}$），$n^{-1}=13$（因为 $4\cdot13\equiv1\pmod{17}$）

假设 $A=[10,2,16,15]$

计算：

$$
a_j=13\cdot\sum_{k=0}^{3}A_k\cdot(13)^{jk}\pmod{17}
$$

计算后得到：
$a=[1,2,3,4]$

完美重建。

#### 微型代码（逆数论变换）

Python (逆数论变换)

```python
MOD = 998244353
ROOT = 3  # 本原根

def modpow(a, e, m):
    res = 1
    while e:
        if e & 1:
            res = res * a % m
        a = a * a % m
        e >>= 1
    return res

def ntt(a, invert=False):
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    len_ = 2
    while len_ <= n:
        wlen = modpow(ROOT, (MOD - 1) // len_, MOD)
        if invert:
            wlen = modpow(wlen, MOD - 2, MOD)
        for i in range(0, n, len_):
            w = 1
            for j in range(len_ // 2):
                u = a[i + j]
                v = a[i + j + len_ // 2] * w % MOD
                a[i + j] = (u + v) % MOD
                a[i + j + len_ // 2] = (u - v + MOD) % MOD
                w = w * wlen % MOD
        len_ <<= 1

    if invert:
        inv_n = modpow(n, MOD - 2, MOD)
        for i in range(n):
            a[i] = a[i] * inv_n % MOD

# 示例
A = [10, 2, 16, 15]
ntt(A, invert=True)
print("逆数论变换:", A)  # [1,2,3,4]
```

C (逆数论变换框架)

```c
void ntt(double complex *a, int n, int invert) {
    // ... (位反转与 NTT 相同)
    for (int len = 2; len <= n; len <<= 1) {
        long long wlen = modpow(ROOT, (MOD - 1) / len, MOD);
        if (invert) wlen = modinv(wlen, MOD);
        for (int i = 0; i < n; i += len) {
            long long w = 1;
            for (int j = 0; j < len / 2; j++) {
                long long u = a[i + j];
                long long v = a[i + j + len / 2] * w % MOD;
                a[i + j] = (u + v) % MOD;
                a[i + j + len / 2] = (u - v + MOD) % MOD;
                w = w * wlen % MOD;
            }
        }
    }
    if (invert) {
        long long inv_n = modinv(n, MOD);
        for (int i = 0; i < n; i++)
            a[i] = a[i] * inv_n % MOD;
    }
}
```

#### 为什么它很重要

- 完善了模 FFT 流程
- 确保系数的精确重建
- 是多项式乘法、密码学变换、纠错码的核心
- 支持精确的模运算，没有浮点误差

#### 一个温和的证明（为什么它有效）

NTT 和 INTT 互为模逆：

$$
\text{NTT}(a)=W_n\cdot a,\quad \text{INTT}(A)=n^{-1}\cdot W_n^{-1}\cdot A
$$

其中 $W_n$ 是关于 $g$ 的范德蒙德矩阵。
因为 $W_nW_n^{-1}=I$，且 $g$ 是本原根，所以这两个变换构成一个双射。

#### 自己动手试试

1.  计算 $[1,2,3,4]$ 在模 $17$ 下的 NTT，然后应用 INTT。
2.  验证 $\text{INTT}(\text{NTT}(a))=a$。
3.  检查不同 $n=8,16$ 时的逆变换性质。
4.  使用 NTT + INTT 流程进行多项式乘法。
5.  比较模运算与浮点 FFT。

#### 测试用例

| 输入 $A$            | 输出 $a$ (模 17) |
| -------------------- | ------------------- |
| [10,2,16,15]         | [1,2,3,4]           |
| [4,13,11,10,1,0,0,0] | [1,2,3,4,5]         |

#### 复杂度

- 时间：$O(n\log n)$
- 空间：$O(n)$

INTT 完成了这个循环：现在每个模变换都有一个完美的逆变换。从数论到代码，它在每一步都保持了精确性。
### 569 Bluestein 算法

Bluestein 算法，也称为线性调频 z 变换，是一种巧妙的方法，它利用长度为 $m \ge 2n-1$ 的快速傅里叶变换（FFT）来计算任意长度 $n$ 的离散傅里叶变换（DFT）。与 Cooley–Tukey 算法不同，即使 $n$ 不是 2 的幂次，它也能工作。

#### 我们要解决什么问题？

标准的 FFT（如 Cooley–Tukey）要求 $n$ 是 2 的幂次才能进行快速计算。如果我们想计算任意 $n$（例如 $n=12$、$30$ 或一个质数）的 DFT，该怎么办？

Bluestein 算法将 DFT 重写为卷积形式，然后可以使用任何足够大小的 FFT 来高效计算。

给定一个序列 $a=[a_0,a_1,\ldots,a_{n-1}]$，我们想要：

$$
A_k = \sum_{j=0}^{n-1} a_j \cdot \omega_n^{jk}, \quad 0 \le k < n
$$

其中 $\omega_n = e^{-2\pi i / n}$。

#### 它是如何工作的（通俗解释）

Bluestein 算法将 DFT 转化为一个卷积问题：

$$
A_k = \sum_{j=0}^{n-1} a_j \cdot \omega_n^{j^2/2} \cdot \omega_n^{(k-j)^2/2}
$$

定义：

- $b_j = a_j \cdot \omega_n^{j^2/2}$
- $c_j = \omega_n^{-j^2/2}$

那么 $A_k$ 就是 $b$ 和 $c$ 的卷积，再乘以缩放因子 $\omega_n^{k^2/2}$。

即使 $n$ 是任意的，我们也可以使用基于 FFT 的多项式乘法来计算这个卷积。

#### 算法步骤

1.  预计算线性调频因子 $\omega_n^{j^2/2}$ 及其逆。
2.  构建序列 $b_j$ 和 $c_j$。
3.  将两者填充到长度 $m \ge 2n-1$。
4.  执行基于 FFT 的卷积：
    $$
    d = \text{IFFT}(\text{FFT}(b) \cdot \text{FFT}(c))
    $$
5.  提取 $A_k = d_k \cdot \omega_n^{-k^2/2}$，其中 $k=0,\ldots,n-1$。

#### 微型代码（Python）

```python
import cmath
import math

def next_power_of_two(x):
    return 1 << (x - 1).bit_length()

def fft(a, invert=False):
    n = len(a)
    if n == 1:
        return a
    a_even = fft(a[0::2], invert)
    a_odd = fft(a[1::2], invert)
    ang = 2 * math.pi / n * (-1 if not invert else 1)
    w, wn = 1, cmath.exp(1j * ang)
    y = [0] * n
    for k in range(n // 2):
        t = w * a_odd[k]
        y[k] = a_even[k] + t
        y[k + n // 2] = a_even[k] - t
        w *= wn
    if invert:
        for i in range(n):
            y[i] /= 2
    return y

def bluestein_dft(a):
    n = len(a)
    m = next_power_of_two(2 * n - 1)
    ang = math.pi / n
    w = [cmath.exp(-1j * ang * j * j) for j in range(n)]
    b = [a[j] * w[j] for j in range(n)] + [0] * (m - n)
    c = [w[j].conjugate() for j in range(n)] + [0] * (m - n)

    B = fft(b)
    C = fft(c)
    D = [B[i] * C[i] for i in range(m)]
    d = fft(D, invert=True)

    return [d[k] * w[k] for k in range(n)]

# 示例
a = [1, 2, 3]
print("DFT via Bluestein:", bluestein_dft(a))
```

#### 为什么它很重要

-   处理任意长度的 DFT（质数 $n$，非 2 的幂次）
-   广泛应用于信号处理、多项式算术、NTT 推广
-   实现无长度限制的统一 FFT 处理流程
-   核心思想：线性调频 z 变换 → 卷积 → FFT

#### 一个温和的证明（为什么它有效）

我们重写 DFT 项：

$$
A_k = \sum_{j=0}^{n-1} a_j \cdot \omega_n^{jk}
$$

乘以并除以 $\omega_n^{(j^2 + k^2)/2}$：

$$
A_k = \omega_n^{-k^2/2} \sum_{j=0}^{n-1} (a_j \cdot \omega_n^{j^2/2}) \cdot \omega_n^{(k-j)^2/2}
$$

这是两个序列的离散卷积，可以通过 FFT 计算。因此，DFT 被简化为卷积，从而使得对于任意 $n$ 都能实现 $O(m\log m)$ 的性能。

#### 亲自尝试

1.  计算 $n=6$，$a=[1,2,3,4,5,6]$ 的 DFT。
2.  比较 Bluestein 和 Cooley–Tukey 的结果（如果 $n$ 是 2 的幂次）。
3.  尝试 $n=7$（质数），只有 Bluestein 能高效工作。
4.  验证应用逆 DFT 是否返回原始序列。
5.  探索当 $m>2n-1$ 时零填充的效果。

#### 测试用例

| 输入 $a$    | $n$ | 输出（近似值）                                   |
| ----------- | --- | ------------------------------------------------ |
| [1,2,3]     | 3   | [6,(-1.5+0.866i),(-1.5-0.866i)]                  |
| [1,2,3,4,5] | 5   | [15,-2.5+3.44i,-2.5+0.81i,-2.5-0.81i,-2.5-3.44i] |

#### 复杂度

-   时间：$O(m\log m)$，其中 $m\ge2n-1$
-   空间：$O(m)$

Bluestein 算法弥合了 2 的幂次 FFT 与通用长度 DFT 之间的差距，将每个序列都转化为可通过快速变换计算的卷积旋律。
### 570 基于 FFT 的乘法

基于 FFT 的乘法利用快速傅里叶变换，通过将乘法转换为频域中的逐点乘积，来高效地计算大整数或多项式的乘法。

我们不采用逐项相乘的方法（其复杂度为 $O(n^2)$），而是利用 FFT 在 $O(n\log n)$ 时间内计算卷积。

#### 我们要解决什么问题？

当计算两个大多项式（或整数）的乘法时，朴素方法需要 $O(n^2)$ 次运算。对于大的 $n$，这变得不切实际。

我们想要计算：

$$
C(x) = A(x) \cdot B(x)
$$

其中

$$
A(x) = \sum_{i=0}^{n-1} a_i x^i,\quad B(x) = \sum_{j=0}^{n-1} b_j x^j
$$

我们需要 $C(x)$ 的系数 $c_k$，使得

$$
c_k = \sum_{i+j=k} a_i b_j
$$

这正是一个卷积。FFT 可以高效地计算它。

#### 它是如何工作的（通俗解释）

1.  对序列进行填充，使其长度为 2 的幂，并且足够大以容纳完整结果。
2.  将两个序列通过 FFT 变换到频域。
3.  逐元素相乘结果。
4.  进行逆 FFT 以恢复系数。
5.  四舍五入到最接近的整数（对于整数乘法）。

这就像将两段旋律调到频域，相乘它们的谐波，然后再变换回时域。

#### 算法步骤

1.  令 $A$ 和 $B$ 为系数数组。
2.  选择 $n$ 为最小的 2 的幂，满足 $n \ge 2 \cdot \max(\text{len}(A), \text{len}(B))$。
3.  将 $A$ 和 $B$ 填充到长度 $n$。
4.  计算 FFT$(A)$ 和 FFT$(B)$。
5.  计算 $C'[k] = A'[k] \cdot B'[k]$。
6.  计算 $C = \text{IFFT}(C')$。
7.  将 $C$ 的实部四舍五入到最接近的整数。

#### 微型代码（Python）

```python
import cmath
import math

def fft(a, invert=False):
    n = len(a)
    if n == 1:
        return a
    a_even = fft(a[0::2], invert)
    a_odd = fft(a[1::2], invert)
    ang = 2 * math.pi / n * (-1 if not invert else 1)
    w, wn = 1, cmath.exp(1j * ang)
    y = [0] * n
    for k in range(n // 2):
        t = w * a_odd[k]
        y[k] = a_even[k] + t
        y[k + n // 2] = a_even[k] - t
        w *= wn
    if invert:
        for i in range(n):
            y[i] /= 2
    return y

def multiply(a, b):
    n = 1
    while n < len(a) + len(b):
        n <<= 1
    fa = a + [0] * (n - len(a))
    fb = b + [0] * (n - len(b))
    FA = fft(fa)
    FB = fft(fb)
    FC = [FA[i] * FB[i] for i in range(n)]
    C = fft(FC, invert=True)
    return [round(c.real) for c in C]

# 示例：计算 (1 + 2x + 3x^2) * (4 + 5x + 6x^2)
print(multiply([1, 2, 3], [4, 5, 6]))
# 输出: [4, 13, 28, 27, 18]
```

#### 为什么它很重要

-   大整数算术的基础（用于 GMP 等库）。
-   实现快速多项式乘法。
-   用于密码学、信号处理和基于 FFT 的卷积。
-   可以高效地扩展到数百万项。

#### 一个温和的证明（为什么它有效）

令

$$
C(x) = A(x)B(x)
$$

在频域中（使用 DFT）：

$$
\text{DFT}(C) = \text{DFT}(A) \cdot \text{DFT}(B)
$$

根据卷积定理，频域中的乘法对应于时域中的卷积：

$$
C = \text{IFFT}(\text{FFT}(A) \cdot \text{FFT}(B))
$$

因此，在逆变换之后，我们自动得到 $c_k = \sum_{i+j=k} a_i b_j$。

#### 亲自尝试

1.  手动计算 $(1+2x+3x^2)$ 和 $(4+5x+6x^2)$ 的乘积。
2.  使用十进制数字作为系数，实现整数乘法。
3.  比较 $n=1024$ 时 FFT 与朴素乘法的性能。
4.  尝试使用复数或模运算。
5.  探究四舍五入问题和精度。

#### 测试用例

| A(x)    | B(x)    | 结果              |
| ------- | ------- | ----------------- |
| [1,2,3] | [4,5,6] | [4,13,28,27,18]   |
| [3,2,1] | [1,0,1] | [3,2,4,2,1]       |

#### 复杂度

-   时间：$O(n\log n)$
-   空间：$O(n)$

基于 FFT 的乘法将一个缓慢的二次过程转变为频域操作的协奏曲，快速、优雅且可扩展。

# 第 58 章 数值方法
### 571 牛顿-拉夫森法

牛顿-拉夫森法是一种快速的求根方法，它通过沿着函数 $f$ 在 $x$ 处的切线来改进对 $f(x)=0$ 的猜测值 $x$。如果导数不为零，它在简单根附近具有二次收敛性。

#### 我们要解决什么问题？

给定一个可微函数 $f:\mathbb{R}\to\mathbb{R}$，找到 $x^*$ 使得 $f(x^*)=0$。典型的用例包括求解非线性方程、通过 $f^\prime(x)=0$ 优化一维函数，以及作为更大数值方法中的内部步骤。

#### 它是如何工作的（通俗解释）

在当前猜测值 $x_k$ 处，用其切线来近似 $f$：
$$
f(x)\approx f(x_k)+f^\prime(x_k)(x-x_k).
$$
将这个线性模型设为零并求解截距：
$$
x_{k+1}=x_k-\frac{f(x_k)}{f^\prime(x_k)}.
$$
重复此过程，直到变化很小或 $|f(x_k)|$ 很小。

更新规则
$$
x_{k+1}=x_k-\frac{f(x_k)}{f^\prime(x_k)}.
$$

#### 示例

求解 $x^2-2=0$（求 $2$ 的平方根）。

令 $f(x)=x^2-2$，$f^\prime(x)=2x$，初始值 $x_0=1$。

- $x_1=1-\frac{-1}{2}=1.5$
- $x_2=1.5-\frac{1.5^2-2}{3}=1.416\overline{6}$
- $x_3\approx1.4142157$

快速逼近 $\sqrt{2}\approx1.41421356$。

#### 微型代码

Python

```python
def newton(f, df, x0, tol=1e-10, max_iter=100):
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError("导数变为零")
        x_new = x - fx / dfx
        if abs(x_new - x) <= tol:
            return x_new
        x = x_new
    return x

# 示例：sqrt(2)
root = newton(lambda x: x*x - 2, lambda x: 2*x, 1.0)
print(root)
```

C

```c
#include <stdio.h>
#include <math.h>

double newton(double (*f)(double), double (*df)(double),
              double x0, double tol, int max_iter) {
    double x = x0;
    for (int i = 0; i < max_iter; i++) {
        double fx = f(x), dfx = df(x);
        if (dfx == 0.0) break;
        double x_new = x - fx / dfx;
        if (fabs(x_new - x) <= tol) return x_new;
        x = x_new;
    }
    return x;
}

double f(double x){ return x*x - 2.0; }
double df(double x){ return 2.0*x; }

int main(void){
    double r = newton(f, df, 1.0, 1e-10, 100);
    printf("%.12f\n", r);
    return 0;
}
```

#### 为什么它很重要

- 在简单根附近非常快：误差大约每一步平方一次
- 广泛用于非线性方程组求解器、优化和隐式 ODE 步进中
- 当 $f^\prime$ 已知或易于近似时，易于实现

#### 一个温和的证明思路

如果 $f\in C^2$ 且 $x^*$ 是一个简单根，满足 $f(x^*)=0$ 和 $f^\prime(x^*)\ne0$，在 $x^*$ 附近进行泰勒展开得到
$$
f(x)=f^\prime(x^*)(x-x^*)+\tfrac12 f^{\prime\prime}(\xi)(x-x^*)^2.
$$
代入牛顿更新公式，表明新的误差 $e_{k+1}=x_{k+1}-x^*$ 满足
$$
e_{k+1}\approx -\frac{f^{\prime\prime}(x^*)}{2f^\prime(x^*)}e_k^2,
$$
当 $e_k$ 很小时，这就是二次收敛。

#### 实用技巧

- 选择一个好的初始猜测值 $x_0$ 以避免发散
- 防范 $f^\prime(x_k)=0$ 或导数极小的情况
- 使用阻尼：如果步长过大，使用 $x_{k+1}=x_k-\alpha\frac{f(x_k)}{f^\prime(x_k)}$，其中 $0<\alpha\le1$
- 停止准则：$|f(x_k)|\le\varepsilon$ 或 $|x_{k+1}-x_k|\le\varepsilon$

#### 亲自尝试

1.  从 $x_0=0.5$ 开始求解 $\cos x - x=0$。
2.  通过 $f(x)=x^3-5$ 求 $5$ 的立方根。
3.  将牛顿法应用于 $g^\prime(x)=0$ 来最小化 $g(x)=(x-3)^2$。
4.  在 $\pi/2$ 附近对 $f(x)=\tan x - x$ 使用一个糟糕的初始猜测值，观察失败模式。
5.  添加线搜索或阻尼并比较鲁棒性。

#### 测试用例

| $f(x)$       | $f^\prime(x)$  | 根              | 起始 $x_0$ | 迭代至 $10^{-10}$ |
| ------------ | -------------- | ----------------- | ----------- | ------------------ |
| $x^2-2$      | $2x$           | $\sqrt{2}$        | $1.0$       | 5                  |
| $\cos x - x$ | $-,\sin x - 1$ | $\approx0.739085$ | $0.5$       | 5–6                |
| $x^3-5$      | $3x^2$         | $\sqrt[3]{5}$     | $1.5$       | 6–7                |

*（迭代次数仅供参考，取决于容差。）*

#### 复杂度

- 每次迭代：一次 $f$ 和一次 $f^\prime$ 求值，加上 $O(1)$ 的算术运算
- 收敛性：通常在简单根附近为二次收敛
- 总成本：对于行为良好的问题，迭代次数很少

当导数可用且初始猜测值合理时，牛顿-拉夫森法是标准的一维求根器。它简单、快速，并且构成了许多高维方法的基础。
### 572 二分法

二分法是寻找连续函数根的最简单、最可靠的方法之一。它是一种“分割并缩小”的搜索方法，用于在函数变号的区间内找到满足 $f(x^*)=0$ 的 $x^*$。

#### 我们要解决什么问题？

对于一个连续函数 $f$，当我们知道两个点 $a$ 和 $b$ 满足：
$$
f(a)\cdot f(b) < 0
$$
时，我们希望求解 $f(x)=0$。

根据介值定理，这意味着函数在 $a$ 和 $b$ 之间穿过零点。

#### 它是如何工作的（通俗解释）

每一步：
1.  计算中点 $m=\frac{a+b}{2}$
2.  检查 $f(m)$ 的符号
    *   如果 $f(a)\cdot f(m) < 0$，根在 $a$ 和 $m$ 之间
    *   否则，根在 $m$ 和 $b$ 之间
3.  用新的区间替换原区间，并重复此过程，直到区间足够小

它每一步都将搜索空间减半，就像为根进行二分搜索一样。

#### 算法步骤

1.  从满足 $f(a)\cdot f(b)<0$ 的区间 $[a,b]$ 开始
2.  当 $|b-a|>\varepsilon$ 时：
    *   $m=\frac{a+b}{2}$
    *   如果 $f(m)=0$（或足够接近），停止
    *   否则，选择符号发生变化的那个半区间
3.  返回中点作为近似根

#### 微型代码

Python

```python
def bisection(f, a, b, tol=1e-10, max_iter=100):
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise ValueError("f(a) 和 f(b) 必须符号相反")
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol or (b - a) / 2 < tol:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)

# 示例：求解 x^3 - x - 2 = 0 的根
root = bisection(lambda x: x**3 - x - 2, 1, 2)
print(root)  # ~1.5213797
```

C

```c
#include <stdio.h>
#include <math.h>

double f(double x) { return x*x*x - x - 2; }

double bisection(double a, double b, double tol, int max_iter) {
    double fa = f(a), fb = f(b);
    if (fa * fb >= 0) return NAN;
    for (int i = 0; i < max_iter; i++) {
        double m = 0.5 * (a + b);
        double fm = f(m);
        if (fabs(fm) < tol || (b - a) / 2 < tol) return m;
        if (fa * fm < 0) {
            b = m; fb = fm;
        } else {
            a = m; fa = fm;
        }
    }
    return 0.5 * (a + b);
}

int main(void) {
    printf("%.10f\n", bisection(1, 2, 1e-10, 100));
    return 0;
}
```

#### 为什么它很重要

-   如果 $f$ 连续且符号改变，则保证收敛
-   不需要导数（与牛顿-拉夫逊法不同）
-   鲁棒性强且易于实现
-   是混合方法（如 Brent 方法）的基础

#### 一个温和的证明（为什么它有效）

根据介值定理，如果 $f(a)\cdot f(b)<0$ 且 $f$ 连续，则存在 $x^*\in[a,b]$ 使得 $f(x^*)=0$。

每次迭代将区间减半：
$$
|b_{k+1}-a_{k+1}|=\frac{1}{2}|b_k-a_k|
$$
经过 $k$ 次迭代后：
$$
|b_k-a_k|=\frac{1}{2^k}|b_0-a_0|
$$
要达到容差 $\varepsilon$：
$$
k \ge \log_2\frac{|b_0-a_0|}{\varepsilon}
$$

#### 自己动手试试

1.  在 $[1,2]$ 上求 $f(x)=x^3-x-2$ 的根。
2.  在 $[0,1]$ 上尝试 $f(x)=\cos x - x$。
3.  与牛顿-拉夫逊法比较迭代次数。
4.  测试如果 $f(a)$ 和 $f(b)$ 符号相同会发生什么失败情况。
5.  观察容差如何影响精度和迭代次数。

#### 测试用例

| 函数         | 区间   | 根（近似值） | 迭代次数 (ε=1e-6) |
| ------------ | ------ | ------------ | ----------------- |
| $x^2-2$      | [1,2]  | 1.414214     | 20                |
| $x^3-x-2$    | [1,2]  | 1.521380     | 20                |
| $\cos x - x$ | [0,1]  | 0.739085     | 20                |

#### 复杂度

-   时间：$O(\log_2(\frac{b-a}{\varepsilon}))$
-   空间：$O(1)$

二分法以速度为代价换取确定性，当符号条件成立时它永远不会失败，这使其成为可靠求根方法的基石。
### 573 割线法

割线法是一种求根算法，它使用两个初始猜测值，并通过割线（穿过函数最近两个点的直线）反复优化它们。它类似于无需导数的牛顿-拉弗森法：我们不使用 $f'(x)$，而是通过割线的斜率来估计它。

#### 我们要解决什么问题？

我们想求解 $x^*$，使得 $f(x^*)=0$，但有时我们没有导数 $f'(x)$。
割线法通过数值方法近似它：

$$
f'(x_k)\approx \frac{f(x_k)-f(x_{k-1})}{x_k-x_{k-1}}
$$

通过用这个差商替换精确的导数，我们仍然遵循类似牛顿法的更新规则。

#### 它是如何工作的（通俗解释）

想象一下，通过两个点 $(x_{k-1},f(x_{k-1}))$ 和 $(x_k,f(x_k))$ 画一条直线。
这条直线与 $x$ 轴相交于一个新的猜测值 $x_{k+1}$。
重复这个过程直到收敛。

更新公式为：

$$
x_{k+1}=x_k-f(x_k)\frac{x_k-x_{k-1}}{f(x_k)-f(x_{k-1})}
$$

每次迭代使用最后两个估计值，而不是像牛顿法那样只用一个。

#### 算法步骤

1.  从两个初始猜测值 $x_0$ 和 $x_1$ 开始，且满足 $f(x_0)\ne f(x_1)$。

2.  重复直到收敛：

    $$
    x_{k+1}=x_k-f(x_k)\frac{x_k-x_{k-1}}{f(x_k)-f(x_{k-1})}
    $$

3.  当 $|x_{k+1}-x_k|<\varepsilon$ 或 $|f(x_{k+1})|<\varepsilon$ 时停止。

#### 微型代码

Python

```python
def secant(f, x0, x1, tol=1e-10, max_iter=100):
    f0, f1 = f(x0), f(x1)
    for _ in range(max_iter):
        if f1 == f0:
            raise ZeroDivisionError("割线法中除以零")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
        f0, f1 = f1, f(x1)
    return x1

# 示例：求解 x^3 - x - 2 = 0 的根
root = secant(lambda x: x3 - x - 2, 1, 2)
print(root)  # ~1.5213797
```

C

```c
#include <stdio.h>
#include <math.h>

double f(double x) { return x*x*x - x - 2; }

double secant(double x0, double x1, double tol, int max_iter) {
    double f0 = f(x0), f1 = f(x1);
    for (int i = 0; i < max_iter; i++) {
        if (f1 == f0) break;
        double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        if (fabs(x2 - x1) < tol) return x2;
        x0 = x1; f0 = f1;
        x1 = x2; f1 = f(x1);
    }
    return x1;
}

int main(void) {
    printf("%.10f\n", secant(1, 2, 1e-10, 100));
    return 0;
}
```

#### 为什么它很重要

-   无需导数（使用有限差分）
-   比二分法更快（超线性收敛，阶数 ≈1.618）
-   当 $f'(x)$ 计算成本高或无法获得时常用
-   是通往混合方法（例如 Brent 方法）的垫脚石

#### 一个温和的证明（为什么它有效）

牛顿法的更新公式：

$$
x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)}
$$

通过差商近似 $f'(x_k)$：

$$
f'(x_k)\approx\frac{f(x_k)-f(x_{k-1})}{x_k-x_{k-1}}
$$

代入牛顿法的更新公式：

$$
x_{k+1}=x_k-f(x_k)\frac{x_k-x_{k-1}}{f(x_k)-f(x_{k-1})}
$$

因此，它是不需要显式导数的牛顿法，在接近根时仍保持超线性收敛性。

#### 亲自尝试

1.  用 $(x_0,x_1)=(1,2)$ 求解 $x^3-x-2=0$。
2.  与牛顿法（$x_0=1$）比较迭代次数。
3.  用 $(0,1)$ 尝试 $\cos x - x=0$。
4.  观察当 $f(x_k)=f(x_{k-1})$ 时失败的情况。
5.  为了鲁棒性，添加回退到二分法的机制。

#### 测试用例

| 函数         | 初始值 $(x_0,x_1)$ | 根（近似值） | 迭代次数 (ε=1e-10) |
| ------------ | ------------------ | ------------ | ------------------ |
| $x^2-2$      | (1,2)              | 1.41421356   | 6                  |
| $x^3-x-2$    | (1,2)              | 1.5213797    | 7                  |
| $\cos x - x$ | (0,1)              | 0.73908513   | 6                  |

#### 复杂度

-   时间：$O(k)$ 次迭代，每次 $O(1)$（每步计算 1 次函数值）
-   收敛性：超线性（阶数 $\approx1.618$）
-   空间：$O(1)$

割线法融合了牛顿法的速度和二分法的简单性，是一座连接理论与实践的、无需导数的桥梁。
### 574 不动点迭代

不动点迭代是一种求解形如 $x=g(x)$ 方程的通法。我们不是直接寻找 $f(x)=0$ 的根，而是反复应用一个变换，期望它能收敛到一个稳定点，即输入等于输出的不动点。

#### 我们要解决什么问题？

我们想找到 $x^*$，使得

$$
x^* = g(x^*)
$$

如果我们能把问题 $f(x)=0$ 表达为 $x=g(x)$，那么前者的解就是后者的不动点。
例如，求解 $x^2-2=0$ 等价于求解

$$
x = \sqrt{2}
$$

或者用迭代形式表示为，

$$
x_{k+1} = g(x_k) = \frac{1}{2}\left(x_k + \frac{2}{x_k}\right)
$$

#### 它是如何工作的（通俗解释）

从一个初始猜测 $x_0$ 开始，然后不断应用函数 $g$：

$$
x_{k+1} = g(x_k)
$$

如果 $g$ 性质良好（在 $x^*$ 附近是压缩映射），这个序列将越来越接近不动点。

你可以把它想象成反复向平衡点弹跳，每次弹跳幅度越来越小，直到落在 $x^*$ 上。

#### 收敛条件

如果 $g$ 在 $x^*$ 附近连续，并且满足

$$
|g'(x^*)| < 1
$$

则不动点迭代收敛。

如果 $|g'(x^*)| > 1$，迭代发散。

这意味着函数 $g$ 在不动点处的斜率不能太陡，它应该将你“拉”向不动点，而不是推离。

#### 算法步骤

1.  选择初始猜测 $x_0$。
2.  计算 $x_{k+1} = g(x_k)$。
3.  如果 $|x_{k+1} - x_k| < \varepsilon$ 或达到最大迭代次数，则停止。
4.  返回 $x_{k+1}$ 作为近似根。

#### 微型代码

Python

```python
def fixed_point(g, x0, tol=1e-10, max_iter=100):
    x = x0
    for _ in range(max_iter):
        x_new = g(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

# 示例：求解 x = cos(x)
root = fixed_point(lambda x: math.cos(x), 0.5)
print(root)  # ~0.739085
```

C

```c
#include <stdio.h>
#include <math.h>

double g(double x) { return cos(x); }

double fixed_point(double x0, double tol, int max_iter) {
    double x = x0, x_new;
    for (int i = 0; i < max_iter; i++) {
        x_new = g(x);
        if (fabs(x_new - x) < tol) return x_new;
        x = x_new;
    }
    return x;
}

int main(void) {
    printf("%.10f\n", fixed_point(0.5, 1e-10, 100));
    return 0;
}
```

#### 为什么它重要

-   是 Newton–Raphson（牛顿-拉夫森）、Gauss–Seidel（高斯-赛德尔）以及许多非线性求解器的基础。
-   概念简单且通用。
-   展示了收敛性如何依赖于变换的设计，而不仅仅是函数本身。

#### 一个温和的证明（为什么它有效）

假设 $x^*$ 是一个不动点，即 $g(x^*)=x^*$。
在 $x^*$ 附近对 $g$ 进行泰勒展开近似：

$$
g(x) = g(x^*) + g'(x^*)(x - x^*) + O((x-x^*)^2)
$$

两边减去 $x^*$：

$$
x_{k+1}-x^* = g'(x^*)(x_k-x^*) + O((x_k-x^*)^2)
$$

因此，对于小误差，
$$
|x_{k+1}-x^*| \approx |g'(x^*)||x_k-x^*|
$$

如果 $|g'(x^*)|<1$，误差在每次迭代中都会缩小，呈几何级数收敛。

#### 自己动手试试

1.  从 $x_0=0.5$ 开始，求解 $x=\cos(x)$。
2.  使用 $x_{k+1}=\frac{1}{2}(x_k+\frac{2}{x_k})$ 求解 $x=\sqrt{2}$。
3.  尝试 $x=g(x)=1+\frac{1}{x}$，观察它如何发散。
4.  尝试将 $f(x)=x^3-x-2$ 变换成不同的 $g(x)$ 进行实验。
5.  观察对初始猜测和斜率的敏感性。

#### 测试用例

| $g(x)$                       | 起始点 $x_0$ | 结果（近似） | 收敛？ |
| ---------------------------- | ------------ | ------------ | ------ |
| $\cos x$                     | 0.5          | 0.739085     | 是     |
| $\frac{1}{2}(x+\frac{2}{x})$ | 1            | 1.41421356   | 是     |
| $1+\frac{1}{x}$              | 1            | 发散         | 否     |

#### 复杂度

-   每次迭代：$O(1)$（一次函数求值）
-   收敛性：线性（如果 $|g'(x^*)|<1$）
-   空间：$O(1)$

不动点迭代是数值求解的温和心跳，它简单、几何直观，并且是现代求根和优化算法的基础。
### 575 高斯求积法

高斯求积法是一种高精度的数值积分方法，它通过在最优选择的点（节点）处计算 $f(x)$ 并仔细加权来近似
$$
I=\int_a^b f(x),dx
$$
对于给定次数的函数求值，它能达到最大精度，对于光滑函数来说，远比梯形法则或辛普森法则精确得多。

#### 我们要解决什么问题？

我们想用数值方法近似一个积分，但不是使用等间距的采样点，而是选择最佳可能的点来最小化误差。

传统方法均匀采样，但高斯求积法使用正交多项式（如勒让德、切比雪夫、拉盖尔或埃尔米特多项式）来确定理想的节点 $x_i$ 和权重 $w_i$，使得该法则对于所有次数不超过 $2n-1$ 的多项式都能精确积分。

形式化地，

$$
\int_a^b f(x),dx \approx \sum_{i=1}^n w_i,f(x_i)
$$

#### 它是如何工作的（通俗解释）

1.  在 $[a,b]$ 上选择一组正交多项式（对于标准积分，通常使用勒让德多项式）。
2.  第 $n$ 个多项式的根成为采样点 $x_i$。
3.  计算相应的权重 $w_i$，使得该公式能精确积分所有次数不超过 $2n-1$ 的多项式。
4.  计算 $f(x_i)$，乘以权重，然后求和。

每个 $x_i$ 和 $w_i$ 都是预先计算好的，你只需代入你的函数即可。

#### 示例：$[-1,1]$ 上的高斯-勒让德求积法

对于 $n=2$：
$$
x_1=-\frac{1}{\sqrt{3}}, \quad x_2=\frac{1}{\sqrt{3}}, \quad w_1=w_2=1.
$$

因此，
$$
\int_{-1}^{1} f(x),dx \approx f(-1/\sqrt{3})+f(1/\sqrt{3}).
$$

对于任意的 $[a,b]$，通过以下映射
$$
t=\frac{b-a}{2}x+\frac{a+b}{2},
$$
并将结果乘以 $\frac{b-a}{2}$ 进行缩放。

#### 微型代码

Python

```python
import numpy as np

def gauss_legendre(f, a, b, n=2):
    # n=2 时的节点和权重，可扩展至更大的 n
    x = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    w = np.array([1.0, 1.0])
    # 变换到 [a,b]
    t = 0.5*(b - a)*x + 0.5*(b + a)
    return 0.5*(b - a)*np.sum(w * f(t))

# 示例：在 [0, 1] 上积分 f(x)=x^2
result = gauss_legendre(lambda x: x2, 0, 1)
print(result)  # ~0.333333
```

C

```c
#include <stdio.h>
#include <math.h>

double f(double x) { return x*x; }

double gauss_legendre(double a, double b) {
    double x1 = -1.0/sqrt(3.0), x2 = 1.0/sqrt(3.0);
    double w1 = 1.0, w2 = 1.0;
    double t1 = 0.5*(b - a)*x1 + 0.5*(b + a);
    double t2 = 0.5*(b - a)*x2 + 0.5*(b + a);
    return 0.5*(b - a)*(w1*f(t1) + w2*f(t2));
}

int main(void) {
    printf("%.6f\n", gauss_legendre(0, 1)); // 0.333333
    return 0;
}
```

#### 为何重要

-   用很少的点就能达到高精度
-   对所有次数不超过 $2n-1$ 的多项式精确
-   对于光滑的被积函数效果极佳
-   构成了谱方法、有限元法和概率积分的基础

#### 一个温和的证明（为何有效）

令 $p_n(x)$ 为 $[a,b]$ 上关于权函数 $w(x)$ 的 $n$ 次正交多项式。
它的 $n$ 个根 $x_i$ 满足正交性：

$$
\int_a^b p_n(x),p_m(x),w(x),dx=0 \quad (m<n).
$$

如果 $f$ 是一个次数 $\le2n-1$ 的多项式，它可以分解为直到 $p_{2n-1}(x)$ 的项，而使用这些根的求积法则进行积分会得到精确值。

因此，高斯求积法在多项式空间内最小化了积分误差。

#### 亲自尝试

1.  使用 2 点和 3 点高斯-勒让德求积法在 $[0, \pi/2]$ 上积分 $\sin x$。
2.  与辛普森法则进行比较。
3.  使用预先计算的节点和权重扩展到 $n=3$。
4.  在 $[0,1]$ 上尝试 $f(x)=e^x$。
5.  尝试缩放到非标准区间 $[a,b]$。

#### 测试用例

| 函数     | 区间      | $n$ | 结果（近似） | 真实值     |
| -------- | --------- | --- | ------------ | ---------- |
| $x^2$    | [0,1]     | 2   | 0.333333     | 1/3        |
| $\sin x$ | [0,π/2]   | 2   | 0.99984      | 1.00000    |
| $e^x$    | [0,1]     | 2   | 1.71828      | 1.71828    |

#### 复杂度

-   时间：$O(n)$ 次 $f$ 的求值
-   空间：$O(n)$ 用于存储节点和权重
-   精度：对所有次数 ≤ $2n-1$ 的多项式精确

高斯求积法展示了纯数学与计算如何交织，正交多项式引导我们以手术般的精度进行积分。
### 576 辛普森法则

辛普森法则是一种经典的数值积分方法，它使用抛物线而非直线来近似光滑函数的积分。它结合了梯形法则的简单性和高阶精度，使其成为处理等间距数据最实用的方法之一。

#### 我们要解决什么问题？

我们想要近似计算定积分

$$
I = \int_a^b f(x),dx
$$

而我们只知道 $f(x)$ 在离散点上的值，或者无法找到解析的原函数。

辛普森法则不是用直线来近似每对点之间的 $f$，而是通过每两个子区间进行二次插值，从而获得更高的精度。

#### 它是如何工作的（通俗解释）

想象你取三个等间距 $h$ 的点 $(x_0,f(x_0))$, $(x_1,f(x_1))$, $(x_2,f(x_2))$。
通过这些点拟合一条抛物线，对该抛物线进行积分，然后重复此过程。

对于单个抛物线弧段：

$$
\int_{x_0}^{x_2} f(x),dx \approx \frac{h}{3}\big[f(x_0) + 4f(x_1) + f(x_2)\big]
$$

对于 $n$ 个子区间（其中 $n$ 为偶数）：

$$
I \approx \frac{h}{3}\Big[f(x_0) + 4\sum_{i=1,3,5,\ldots}^{n-1} f(x_i) + 2\sum_{i=2,4,6,\ldots}^{n-2} f(x_i) + f(x_n)\Big]
$$

其中 $h = \frac{b - a}{n}$。

#### 微型代码

Python

```python
def simpson(f, a, b, n=100):
    if n % 2 == 1:
        n += 1  # 必须为偶数
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        s += 4 * f(x) if i % 2 else 2 * f(x)
    return s * h / 3

# 示例：从 0 到 π 积分 sin(x)
import math
res = simpson(math.sin, 0, math.pi, n=100)
print(res)  # ~2.000000
```

C

```c
#include <stdio.h>
#include <math.h>

double f(double x) { return sin(x); }

double simpson(double a, double b, int n) {
    if (n % 2 == 1) n++; // 必须为偶数
    double h = (b - a) / n;
    double s = f(a) + f(b);
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        s += (i % 2 ? 4 : 2) * f(x);
    }
    return s * h / 3.0;
}

int main(void) {
    printf("%.6f\n", simpson(0, M_PI, 100)); // 2.000000
    return 0;
}
```

#### 为什么它很重要

- 对于光滑函数，既精确又简单
- 对于三次多项式是精确的
- 对于列表数据，通常是精度和计算量之间的最佳平衡
- 构成了自适应和复合积分方案的基础

#### 一个温和的证明（为什么它有效）

令 $f(x)$ 用一个二次多项式近似
$$
p(x) = ax^2 + bx + c
$$
该多项式通过点 $(x_0,f_0)$, $(x_1,f_1)$, $(x_2,f_2)$。
在 $[x_0,x_2]$ 上对 $p(x)$ 积分得到

$$
\int_{x_0}^{x_2} p(x),dx = \frac{h}{3}\big[f_0 + 4f_1 + f_2\big].
$$

因为误差项依赖于 $f^{(4)}(\xi)$，所以辛普森法则具有四阶精度，即

$$
E = -\frac{(b-a)}{180}h^4f^{(4)}(\xi)
$$

其中 $\xi\in[a,b]$。

#### 亲自尝试

1.  从 $0$ 到 $\pi$ 积分 $\sin x$（结果应为 $2$）。
2.  从 $0$ 到 $1$ 积分 $x^4$，检查对于次数 ≤ 3 的多项式的精确性。
3.  与相同 $n$ 下的梯形法则进行比较。
4.  尝试使用非偶数 $n$ 并验证收敛性。
5.  实现自适应辛普森法则以进行自动细化。

#### 测试用例

| 函数     | 区间     | $n$  | 辛普森结果 | 真实值     |
| -------- | -------- | ---- | ---------- | ---------- |
| $\sin x$ | [0, π]   | 100  | 1.999999   | 2.000000   |
| $x^2$    | [0, 1]   | 10   | 0.333333   | 1/3        |
| $e^x$    | [0, 1]   | 20   | 1.718282   | 1.718282   |

#### 复杂度

- 时间：$O(n)$ 次 $f(x)$ 求值
- 精度：$O(h^4)$
- 空间：$O(1)$

辛普森法则是简单性和精确性的完美结合，是超越直线近似的抛物线式飞跃。
### 577 梯形法则

梯形法则是最简单的数值积分方法之一。它通过将曲线下的面积划分为梯形而非矩形来近似，从而在采样点之间提供线性插值。

#### 我们要解决什么问题？

我们想要估计

$$
I = \int_a^b f(x),dx
$$

当我们只知道 $f(x)$ 在离散点处的值，或者当积分没有简单的解析形式时。
其思想是在每对相邻点之间将 $f(x)$ 近似为分段线性函数。

#### 它是如何工作的（通俗解释）

如果你知道 $f(a)$ 和 $f(b)$，最简单的估计就是一个梯形的面积：

$$
I \approx \frac{b-a}{2},[f(a)+f(b)].
$$

对于等宽 $h=(b-a)/n$ 的多个子区间：

$$
I \approx \frac{h}{2}\Big[f(x_0)+2\sum_{i=1}^{n-1}f(x_i)+f(x_n)\Big].
$$

每对连续的点定义一个梯形，我们求和这些梯形的面积来近似总面积。

#### 微型代码

Python

```python
def trapezoidal(f, a, b, n=100):
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s += f(a + i * h)
    return s * h

# 示例：对 e^x 从 0 到 1 积分
import math
res = trapezoidal(math.exp, 0, 1, 100)
print(res)  # ~1.718282
```

C

```c
#include <stdio.h>
#include <math.h>

double f(double x) { return exp(x); }

double trapezoidal(double a, double b, int n) {
    double h = (b - a) / n;
    double s = 0.5 * (f(a) + f(b));
    for (int i = 1; i < n; i++) {
        s += f(a + i * h);
    }
    return s * h;
}

int main(void) {
    printf("%.6f\n", trapezoidal(0, 1, 100)); // 1.718282
    return 0;
}
```

#### 为何重要

- 简单且广泛用作第一种数值积分方法
- 对于光滑函数具有鲁棒性
- 是辛普森法则和龙贝格积分的基础构件
- 可直接用于表格数据

#### 一个温和的证明（为何有效）

在一个子区间 $[x_i,x_{i+1}]$ 上，用一条直线近似 $f(x)$：

$$
f(x)\approx f(x_i)+\frac{f(x_{i+1})-f(x_i)}{h}(x-x_i).
$$

精确积分这条直线得到

$$
\int_{x_i}^{x_{i+1}} f(x),dx \approx \frac{h}{2}[f(x_i)+f(x_{i+1})].
$$

对所有区间求和就得到了复合梯形法则。

一个区间的误差项与 $f$ 的曲率成正比：

$$
E = -\frac{(b-a)}{12}h^2f''(\xi)
$$

其中 $\xi\in[a,b]$。因此，它具有二阶精度。

#### 亲自尝试

1.  对 $\sin x$ 从 $0$ 到 $\pi$ 积分（结果 $\approx2$）。
2.  对 $x^2$ 从 $0$ 到 $1$ 积分，并与精确值 $1/3$ 比较。
3.  对相同的 $n$，比较其误差与辛普森法则的误差。
4.  尝试 $f(x)=1/x$ 在 $[1,2]$ 区间上。
5.  探索将 $h$ 减半如何影响精度。

#### 测试用例

| 函数     | 区间      | $n$ | 梯形法则    | 真值       |
| -------- | --------- | --- | ----------- | ---------- |
| $\sin x$ | [0, π]    | 100 | 1.9998      | 2.0000     |
| $x^2$    | [0,1]     | 100 | 0.33335     | 1/3        |
| $e^x$    | [0,1]     | 100 | 1.71828     | 1.71828    |

#### 复杂度

-   时间：$O(n)$ 次函数求值
-   精度：$O(h^2)$
-   空间：$O(1)$

梯形法则是数值积分的入门方法，直观、可靠，并且在精细离散化或处理光滑函数时出人意料地有效。
### 578 龙格-库塔（RK4）方法

龙格-库塔（RK4）方法是数值求解常微分方程（ODEs）最广泛使用的技术之一。它在每个步长内使用多个斜率评估，以在精度和计算简单性之间取得了完美的平衡，从而实现了四阶精度。

#### 我们要解决什么问题？

我们想求解一个初值问题（IVP）：

$$
\frac{dy}{dx}=f(x,y), \quad y(x_0)=y_0
$$

当不存在解析解（或难以找到）时，RK4 可以高精度地逐步逼近 $y(x)$。

#### 它是如何工作的（通俗解释）

与每步只取一个斜率（如欧拉法）不同，RK4 采样四个斜率并将它们组合起来：

$$
\begin{aligned}
k_1 &= f(x_n, y_n),\
k_2 &= f(x_n + h/2,, y_n + h k_1/2),\
k_3 &= f(x_n + h/2,, y_n + h k_2/2),\
k_4 &= f(x_n + h,, y_n + h k_3),\
y_{n+1} &= y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4).
\end{aligned}
$$

这个加权平均值以惊人的精度捕捉了曲率和局部行为。

#### 微型代码

Python

```python
def rk4(f, x0, y0, h, n):
    x, y = x0, y0
    for _ in range(n):
        k1 = f(x, y)
        k2 = f(x + h/2, y + h*k1/2)
        k3 = f(x + h/2, y + h*k2/2)
        k4 = f(x + h, y + h*k3)
        y += h*(k1 + 2*k2 + 2*k3 + k4)/6
        x += h
    return x, y

# 示例：dy/dx = y, y(0) = 1 → 真实解 y = e^x
import math
f = lambda x, y: y
x, y = rk4(f, 0, 1, 0.1, 10)
print(y, "vs", math.e)
```

C

```c
#include <stdio.h>
#include <math.h>

double f(double x, double y) { return y; } // dy/dx = y

void rk4(double x0, double y0, double h, int n) {
    double x = x0, y = y0;
    for (int i = 0; i < n; i++) {
        double k1 = f(x, y);
        double k2 = f(x + h/2, y + h*k1/2);
        double k3 = f(x + h/2, y + h*k2/2);
        double k4 = f(x + h, y + h*k3);
        y += h*(k1 + 2*k2 + 2*k3 + k4)/6.0;
        x += h;
    }
    printf("x=%.2f, y=%.6f\n", x, y);
}

int main(void) {
    rk4(0, 1, 0.1, 10); // y(1) ≈ e ≈ 2.718282
    return 0;
}
```

#### 为什么它很重要

- 无需复杂导数即可获得四阶精度
- 用于物理学、工程学和机器学习（ODE 求解器）
- 自适应步长求解器和神经 ODE 的基础
- 比欧拉法或中点法稳定得多且精确得多

#### 一个温和的证明（为什么它有效）

用泰勒级数展开真实解：

$$
y(x+h)=y(x)+h y'(x)+\frac{h^2}{2}y''(x)+\frac{h^3}{6}y^{(3)}(x)+O(h^4)
$$

RK4 对 $k_1,k_2,k_3,k_4$ 的组合重现了直到 $h^4$ 的所有项，给出了全局误差 $O(h^4)$。

每个 $k_i$ 估计中间点处的导数，形成一个加权平均值，捕捉了局部曲率。

#### 亲自尝试

1.  用 $h=0.1$ 求解 $dy/dx = y$，从 $x=0$ 到 $x=1$。
2.  与欧拉法比较。
3.  尝试 $dy/dx = -2y$，$y(0)=1$。
4.  在斜率场上可视化 $k_1,k_2,k_3,k_4$。
5.  用非线性 $f(x,y)=x^2+y^2$ 进行测试。

#### 测试用例

| 微分方程        | 区间   | 步长 h | 结果     | 真实值          |
| --------------- | ------ | ------ | -------- | --------------- |
| $dy/dx=y$       | [0,1]  | 0.1    | 2.71828  | $e$             |
| $dy/dx=-2y$     | [0,1]  | 0.1    | 0.13534  | $e^{-2}$        |
| $dy/dx=x+y$     | [0,1]  | 0.1    | 2.7183   | 解析解 $2e-1$ |

#### 复杂度

- 时间：$O(n)$ 次 $f$ 的求值（每步 4 次）
- 精度：全局误差 $O(h^4)$
- 空间：$O(1)$

龙格-库塔方法是数值设计的杰作，简单到几分钟就能编写代码，强大到足以驱动现代模拟和控制系统。
### 579 欧拉方法

欧拉方法是求解常微分方程（ODEs）最简单的数值方法。它是数值积分领域的 "hello world"，概念清晰，易于实现，并且为更高级的方法（如龙格-库塔法）奠定了基础。

#### 我们要解决什么问题？

我们想要近似求解一个初值问题：

$$
\frac{dy}{dx}=f(x,y), \quad y(x_0)=y_0.
$$

如果我们无法解析求解，我们可以使用小步长 $h$ 在离散点上近似 $y(x)$。

#### 它是如何工作的（通俗解释）

核心思想：利用当前点的斜率来预测下一个点。

在每一步：

$$
y_{n+1} = y_n + h f(x_n, y_n),
$$

以及

$$
x_{n+1} = x_n + h.
$$

这其实就是"沿着切线走一小步"。$h$ 越小，近似效果越好。

#### 微型代码

Python

```python
def euler(f, x0, y0, h, n):
    x, y = x0, y0
    for _ in range(n):
        y += h * f(x, y)
        x += h
    return x, y

# 示例：dy/dx = y, y(0)=1 -> y=e^x
import math
f = lambda x, y: y
x, y = euler(f, 0, 1, 0.1, 10)
print(y, "vs", math.e)
```

C

```c
#include <stdio.h>
#include <math.h>

double f(double x, double y) { return y; }

void euler(double x0, double y0, double h, int n) {
    double x = x0, y = y0;
    for (int i = 0; i < n; i++) {
        y += h * f(x, y);
        x += h;
    }
    printf("x=%.2f, y=%.6f\n", x, y);
}

int main(void) {
    euler(0, 1, 0.1, 10); // y(1) ≈ 2.5937 vs e ≈ 2.7183
    return 0;
}
```

#### 为什么它很重要

-   进入数值常微分方程求解的第一步
-   简单、直观且具有教育意义
-   展示了步长与精度之间的权衡
-   用作龙格-库塔法、休恩法和预测-校正方法的基础构件

#### 一个温和的证明（为什么它有效）

使用泰勒展开：

$$
y(x+h) = y(x) + h y'(x) + \frac{h^2}{2} y''(\xi)
$$

由于 $y'(x)=f(x,y)$，欧拉方法通过忽略高阶项来近似：

$$
y_{n+1} \approx y_n + h f(x_n, y_n).
$$

局部误差为 $O(h^2)$，全局误差为 $O(h)$。

#### 动手尝试

1.  用 $h=0.1$ 求解 $dy/dx=y$，区间从 $0$ 到 $1$。
2.  减小 $h$，观察其向 $e$ 收敛。
3.  尝试 $dy/dx=-2y$ 并绘制指数衰减图。
4.  与相同步长的龙格-库塔法进行比较。
5.  实现一个存储并绘制所有 $(x,y)$ 点对的版本。

#### 测试用例

| 微分方程 | 区间 | 步长 $h$ | 欧拉结果 | 真实值 |
| -------- | ---- | -------- | -------- | ------ |
| $dy/dx=y$ | [0,1] | 0.1 | 2.5937 | 2.7183 |
| $dy/dx=-2y$ | [0,1] | 0.1 | 0.1615 | 0.1353 |
| $dy/dx=x+y$ | [0,1] | 0.1 | 2.65 | 2.7183 |

#### 复杂度

-   时间：$O(n)$
-   精度：全局 $O(h)$
-   空间：$O(1)$

欧拉方法是踏入数值动力学领域最简单的一步，是在微分方程这个弯曲世界里画出的一条直线。
### 580 梯度下降（一维数值优化）

梯度下降是一种简单而强大的迭代算法，用于寻找可微函数的最小值。在一维情况下，它是一个直观的过程：沿着斜率相反的方向移动，直到函数停止下降。

#### 我们要解决什么问题？

我们想找到一个实值函数 $f(x)$ 的局部最小值，即：

$$
\min_x f(x)
$$

如果 $f'(x)$ 存在，但解析地求解 $f'(x)=0$ 很困难，我们可以使用梯度（斜率）逐步逼近最小值。

#### 它是如何工作的（通俗解释）

在每次迭代中，沿着导数的相反方向移动，移动距离由学习率 $\eta$ 缩放：

$$
x_{t+1} = x_t - \eta f'(x_t)
$$

导数 $f'(x_t)$ 指向上坡方向；减去它就向下坡方向移动。步长 $\eta$ 决定了我们移动的距离。

- 如果 $\eta$ 太小，收敛速度慢。
- 如果 $\eta$ 太大，可能会越过最小值或发散。

这个过程重复进行，直到 $|f'(x_t)|$ 变得非常小。

#### 微型代码

Python

```python
def gradient_descent_1d(df, x0, eta=0.1, tol=1e-6, max_iter=1000):
    x = x0
    for _ in range(max_iter):
        grad = df(x)
        if abs(grad) < tol:
            break
        x -= eta * grad
    return x

# 示例：最小化 f(x) = x^2 -> df/dx = 2x
f_prime = lambda x: 2*x
x_min = gradient_descent_1d(f_prime, x0=5)
print(x_min)  # ≈ 0
```

C

```c
#include <stdio.h>
#include <math.h>

double df(double x) { return 2*x; } // x^2 的导数

double gradient_descent(double x0, double eta, double tol, int max_iter) {
    double x = x0;
    for (int i = 0; i < max_iter; i++) {
        double grad = df(x);
        if (fabs(grad) < tol) break;
        x -= eta * grad;
    }
    return x;
}

int main(void) {
    double xmin = gradient_descent(5.0, 0.1, 1e-6, 1000);
    printf("x_min = %.6f\n", xmin);
    return 0;
}
```

#### 为什么它很重要

- 是优化、机器学习和深度学习的基础
- 自然地可以从一维扩展到高维
- 有助于可视化能量景观、收敛性和学习动态

#### 一个温和的证明（为什么它有效）

对于凸且光滑的函数 $f(x)$，泰勒展开给出：

$$
f(x-\eta f'(x)) \approx f(x) - \eta (f'(x))^2 + \frac{\eta^2}{2} f''(\xi) (f'(x))^2
$$

对于小的 $\eta>0$ 且 $f''(\xi)>0$，项 $-\eta (f'(x))^2$ 占主导地位，确保 $f(x)$ 每一步都减小。

这表明算法会收敛到 $f'(x)=0$ 的驻点。

#### 亲自尝试

1.  $f(x)=x^2$，$\eta=0.1$，从 $x_0=5$ 开始。
2.  尝试 $\eta=0.01$ 和 $\eta=1.0$，观察收敛或发散情况。
3.  最小化 $f(x)=(x-3)^2$；检查 $x\to3$ 的速度。
4.  在 $f(x)$ 的图上可视化每次迭代。
5.  修改为随机或带动量的变体。

#### 测试用例

| 函数      | 导数        | 起始点 $x_0$ | $\eta$ | 结果    | 真实最小值 |
| --------- | ----------- | ------------ | ------ | ------- | ---------- |
| $x^2$     | $2x$        | 5            | 0.1    | 0.0000  | 0          |
| $(x-3)^2$ | $2(x-3)$    | 0            | 0.1    | 3.0000  | 3          |
| $\cos x$  | $-\sin x$   | 2            | 0.05   | 1.57    | $\pi/2$    |

#### 复杂度

- 时间：$O(k)$ 次迭代（取决于步长和容差）
- 精度：取决于 $\eta$ 和 $f$ 的光滑性
- 空间：$O(1)$

梯度下降是最简单的学习形式，是下山的温和一步，体现了优化和机器智能的核心。

# 第 59 节 数学优化
### 581 单纯形法

单纯形法是数学优化的基石，是一种高效求解线性规划（LP）问题的几何算法。它沿着可行域（一个凸多面体）的顶点移动，直到到达最优顶点，此时目标函数达到最大值或最小值。

#### 我们要解决什么问题？

我们希望在线性约束条件下优化一个线性目标函数：

$$
\text{最大化 } z = c_1x_1 + c_2x_2 + \dots + c_nx_n
$$

约束条件为

$$
\begin{aligned}
a_{11}x_1 + a_{12}x_2 + \dots + a_{1n}x_n &\le b_1, \\
a_{21}x_1 + a_{22}x_2 + \dots + a_{2n}x_n &\le b_2, \\
&\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \dots + a_{mn}x_n &\le b_m, \\
x_i &\ge 0.
\end{aligned}
$$

这是线性规划的标准形式。

目标是找到能使 $z$ 取得最大（或最小）值的 $(x_1,\dots,x_n)$。

#### 它是如何工作的（通俗解释）

想象所有约束条件在二维空间中形成一个多边形，或在更高维空间中形成一个多面体。
可行域是凸的，因此最优点总是位于某个顶点上。

单纯形法：

1.  从一个顶点（可行基解）开始。
2.  沿着边移动到能改善目标函数的相邻顶点。
3.  重复此过程，直到无法进一步改善，该顶点即为最优解。

#### 代数视角

1.  使用松弛变量将不等式转换为等式。
    例如：
    $x_1 + x_2 \le 4$ → $x_1 + x_2 + s_1 = 4$，其中 $s_1 \ge 0$。

2.  将方程组表示为单纯形表形式。

3.  执行主元操作（类似于高斯消元法），从一个基本可行解移动到另一个。

4.  当目标行中的所有检验数都为非负时（对于最大化问题）停止。

#### 微型代码（简化演示）

Python（教学版本）

```python
import numpy as np

def simplex(c, A, b):
    m, n = A.shape
    tableau = np.zeros((m+1, n+m+1))
    tableau[:m, :n] = A
    tableau[:m, n:n+m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[-1, :n] = -c

    while True:
        col = np.argmin(tableau[-1, :-1])
        if tableau[-1, col] >= 0:
            break  # 达到最优
        ratios = [tableau[i, -1] / tableau[i, col] if tableau[i, col] > 0 else np.inf for i in range(m)]
        row = np.argmin(ratios)
        pivot = tableau[row, col]
        tableau[row, :] /= pivot
        for i in range(m+1):
            if i != row:
                tableau[i, :] -= tableau[i, col] * tableau[row, :]
    return tableau[-1, -1]

# 示例：最大化 z = 3x1 + 2x2
# 约束条件：x1 + x2 ≤ 4, x1 ≤ 2, x2 ≤ 3
c = np.array([3, 2])
A = np.array([[1, 1], [1, 0], [0, 1]])
b = np.array([4, 2, 3])
print("最大 z =", simplex(c, A, b))
```

#### 为何重要

-   是运筹学、经济学和优化理论的基础。
-   应用于物流、金融、资源分配和机器学习（例如支持向量机）。
-   尽管最坏情况复杂度是指数级的，但在实践中速度极快。

#### 一个温和的证明（为何有效）

因为线性规划是凸的，最优解（如果存在）必然出现在可行域的某个顶点上。
单纯形算法以严格改善目标函数的方式探索顶点，直到到达一个没有更优相邻顶点的顶点。

这对应于单纯形表中所有检验数都为非负的条件，表明已达到最优。

#### 动手尝试

1.  求解线性规划：最大化 $z = 3x_1 + 5x_2$
    约束条件：
    $$
    \begin{cases}
    2x_1 + x_2 \le 8 \\
    x_1 + 2x_2 \le 8 \\
    x_1, x_2 \ge 0
    \end{cases}
    $$

2.  画出可行域并验证最优顶点。

3.  修改约束条件，观察解如何移动。

4.  通过取负目标函数来尝试最小化问题。

#### 测试用例

| 目标函数     | 约束条件                               | 结果 $(x_1,x_2)$ | 最大 $z$ |
| ------------ | -------------------------------------- | ---------------- | -------- |
| $3x_1+2x_2$ | $x_1+x_2\le4,\ x_1\le2,\ x_2\le3$ | (1,3)            | 9        |
| $2x_1+3x_2$ | $2x_1+x_2\le8,\ x_1+2x_2\le8$     | (2,3)            | 13       |
| $x_1+x_2$   | $x_1,x_2\le5$                     | (5,5)            | 10       |

#### 复杂度

-   时间：实践中为多项式时间（尽管最坏情况是指数级）
-   空间：存储单纯形表需要 $O(mn)$

单纯形法仍然是有史以来最优雅的算法之一，是在凸形地貌上的一场几何舞蹈，总能找到价值最高的角落。
### 582 对偶单纯形法

对偶单纯形法是单纯形算法的近亲。原始单纯形法保持解的可行性并朝着最优性移动，而对偶单纯形法则恰恰相反，它保持解的最优性并朝着可行性移动。

当约束条件发生变化，或者当我们从一个已经满足最优性条件但不可行的解开始时，这种方法尤其有用。

#### 我们要解决什么问题？

我们仍然解决标准形式的线性规划（LP）问题，但我们从一个对偶可行（目标函数最优）但原始不可行（某些右侧项为负）的单纯形表开始。

最大化

$$
z = c^T x
$$

约束条件为

$$
A x \le b, \quad x \ge 0.
$$

对偶单纯形法致力于在保持检验数最优性的同时，恢复可行性。

#### 它是如何工作的（通俗解释）

可以把单纯形法和对偶单纯形法看作是镜像关系：

| 步骤         | 单纯形法                     | 对偶单纯形法                   |
| ------------ | -------------------------- | ----------------------------- |
| 保持         | 可行解                     | 最优检验数                     |
| 修复         | 最优性                     | 可行性                         |
| 主元选择     | 最负的检验数               | 最负的右侧项                   |

在对偶单纯形法中，每次迭代：

1.  识别一个右侧项（RHS）为负的行（违反可行性）。
2.  在该行的系数中，选择一个主元列，使得旋转后检验数保持非负。
3.  执行旋转操作，使该约束变得可行。
4.  重复上述步骤，直到所有右侧项均为非负（完全可行）。

#### 代数公式

如果我们维护的单纯形表为：

$$
\begin{bmatrix}
A & I & b \\
c^T & 0 & z
\end{bmatrix}
$$

那么主元选择条件变为：

-   选择满足 $b_r < 0$ 的行 $r$。
-   选择满足 $a_{rs} < 0$ 且 $\frac{c_s}{a_{rs}}$ 最小的列 $s$。
-   在 $(r,s)$ 处进行旋转。

这逐步恢复了原始可行性，同时保持了对偶最优性。

#### 微型代码（示例）

Python

```python
import numpy as np

def dual_simplex(A, b, c):
    m, n = A.shape
    tableau = np.zeros((m+1, n+m+1))
    tableau[:m, :n] = A
    tableau[:m, n:n+m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[-1, :n] = -c

    while np.any(tableau[:-1, -1] < 0):
        r = np.argmin(tableau[:-1, -1])
        ratios = []
        for j in range(n+m):
            if tableau[r, j] < 0:
                ratios.append(tableau[-1, j] / tableau[r, j])
            else:
                ratios.append(np.inf)
        s = np.argmin(ratios)
        tableau[r, :] /= tableau[r, s]
        for i in range(m+1):
            if i != r:
                tableau[i, :] -= tableau[i, s] * tableau[r, :]
    return tableau[-1, -1]

# 示例
A = np.array([[1, 1], [2, 1]])
b = np.array([-2, 2])  # 不可行起点
c = np.array([3, 2])
print("最优值:", dual_simplex(A, b, c))
```

#### 为什么它很重要

-   对于约束条件或右侧项发生微小变化后的问题重新优化非常高效。
-   常用于整数规划的分支定界法和割平面法。
-   当先前的单纯形解变得不可行时，避免了重新计算。

#### 一个温和的证明（为什么它有效）

在每次迭代中，旋转操作保持了对偶可行性，这意味着检验数保持非负，并且减少了目标函数值（对于最大化问题）。
当所有基变量都变得可行（右侧项非负）时，该解既是可行的也是最优的。

因此，在非退化情况下，保证在有限步内收敛。

#### 亲自尝试

1.  从一个右侧项为负的单纯形表开始。
2.  应用对偶单纯形法的主元规则来修复可行性。
3.  观察目标函数值从不增加（对于最大化问题）。
4.  与标准单纯形法所采取的路径进行比较。
5.  使用它来重新求解一个修改后的线性规划问题，而无需从头开始。

#### 测试用例

| 目标函数   | 约束条件                      | 方法起点        | 结果                    |
| ----------- | ---------------------------- | ----------------- | ------------------------- |
| $3x_1+2x_2$ | $x_1+x_2\le2$, $x_1-x_2\ge1$ | 原始不可行 | $x_1=1.5, x_2=0.5, z=5.5$ |
| $2x_1+x_2$  | $x_1-x_2\ge2$, $x_1+x_2\le6$ | 原始不可行 | $x_1=2, x_2=4, z=8$       |

#### 复杂度

-   时间：与单纯形法类似，实践中高效
-   空间：单纯形表需要 $O(mn)$

对偶单纯形法是一面平衡不可行性与最优性的镜子，当问题环境发生变化但解必须优雅适应时，它是一种实用的算法。
### 583 内点法

内点法是求解线性和凸优化问题的现代方法，是单纯形法的一种替代方案。它并非沿着可行区域的边界移动，而是在可行区域内部沿着光滑曲线向最优点移动。

#### 我们要解决什么问题？

我们想求解一个标准线性规划问题：

$$
\text{最小化 } c^T x
$$

约束条件为

$$
A x = b, \quad x \ge 0.
$$

这定义了一个凸可行区域，即半空间和等式约束的交集。内点法在该区域内部而非边界上搜索最优点。

#### 它是如何工作的（通俗解释）

将可行区域想象成一个多面体。与单纯形法“沿着棱边行走”不同，我们沿着一条由目标函数和约束共同引导的光滑中心路径，在内部“滑动”。

该方法使用一个障碍函数来防止解越过边界。例如，为了保持 $x_i \ge 0$，我们在目标函数中添加一个惩罚项 $-\mu \sum_i \ln(x_i)$。

因此，我们求解：

$$
\text{最小化 } c^T x - \mu \sum_i \ln(x_i)
$$

其中 $\mu > 0$ 控制我们与边界保持多近的距离。当 $\mu \to 0$ 时，解趋近于真正的最优顶点。

#### 数学步骤

1.  障碍问题形式化：
    $$
    \min_x ; c^T x - \mu \sum_{i=1}^n \ln(x_i)
    $$
    约束条件为 $A x = b$。

2.  一阶条件：
    $$
    A x = b, \quad X s = \mu e, \quad s = c - A^T y,
    $$
    其中 $X = \text{diag}(x_1, \dots, x_n)$，$s$ 是松弛变量。

3.  牛顿更新：
    求解线性化后的 KKT (Karush–Kuhn–Tucker) 方程组，得到 $\Delta x, \Delta y, \Delta s$。

4.  步长和更新：
    $$
    x \leftarrow x + \alpha \Delta x, \quad y \leftarrow y + \alpha \Delta y, \quad s \leftarrow s + \alpha \Delta s,
    $$
    选择步长 $\alpha$ 以保持正性。

5.  减小 $\mu$ 并重复直到收敛。

#### 微型代码（概念示例）

Python（说明性版本）

```python
import numpy as np

def interior_point(A, b, c, mu=1.0, tol=1e-8, max_iter=50):
    m, n = A.shape
    x = np.ones(n)
    y = np.zeros(m)
    s = np.ones(n)

    for _ in range(max_iter):
        r1 = A @ x - b
        r2 = A.T @ y + s - c
        r3 = x * s - mu * np.ones(n)

        if np.linalg.norm(r1) < tol and np.linalg.norm(r2) < tol and np.linalg.norm(r3) < tol:
            break

        # 构造牛顿系统
        diagX = np.diag(x)
        diagS = np.diag(s)
        M = A @ np.linalg.inv(diagS) @ diagX @ A.T
        rhs = -r1 + A @ np.linalg.inv(diagS) @ (r3 - diagX @ r2)
        dy = np.linalg.solve(M, rhs)
        ds = -r2 - A.T @ dy
        dx = (r3 - diagX @ ds) / s

        # 步长
        alpha = 0.99 * min(1, min(-x[dx < 0] / dx[dx < 0], default=1))
        x += alpha * dx
        y += alpha * dy
        s += alpha * ds
        mu *= 0.5
    return x, y, c @ x

# 示例
A = np.array([[1, 1], [1, -1]])
b = np.array([1, 0])
c = np.array([1, 2])
x, y, val = interior_point(A, b, c)
print("最优解 x:", x, "目标值:", val)
```

#### 为何重要

-   对于超大规模线性规划和二次规划问题，可与单纯形法竞争或超越之。
-   光滑且鲁棒，避免了逐顶点遍历。
-   构成了现代凸优化和机器学习求解器（例如支持向量机、逻辑回归）的基础。

#### 一个温和的证明（为何有效）

对数障碍函数确保迭代点始终保持严格正性。每个牛顿步都最小化障碍增强目标函数的局部二次近似。当 $\mu \to 0$ 时，障碍项消失，解收敛到原线性规划问题的真实 KKT 最优点。

#### 亲自尝试

1.  最小化 $x_1 + x_2$，约束条件为
    $$
    \begin{cases}
    x_1 + 2x_2 \ge 2, \
    3x_1 + x_2 \ge 3, \
    x_1, x_2 \ge 0.
    \end{cases}
    $$
2.  与单纯形法的解比较收敛性。
3.  尝试不同的 $\mu$ 缩减策略。
4.  绘制轨迹图，注意穿过内部的光滑曲线。

#### 测试用例

| 目标函数    | 约束条件                       | 最优解 $(x_1,x_2)$ | 最小值 $c^T x$ |
| ----------- | ------------------------------ | ------------------ | -------------- |
| $x_1+x_2$   | $x_1+x_2\ge2$                  | (1,1)              | 2              |
| $x_1+2x_2$  | $x_1+2x_2\ge4$, $x_1,x_2\ge0$ | (0,2)              | 4              |
| $2x_1+x_2$  | $x_1+x_2\ge3$                  | (2,1)              | 5              |

#### 复杂度

-   时间：对于线性规划为 $O(n^{3.5}L)$（多项式时间）
-   空间：由于矩阵分解，为 $O(n^2)$

内点法是优化领域中优雅的光滑旅行者，它以数学的优雅滑过可行空间的中心，同时向全局最优解靠拢。
### 584 梯度下降法（无约束优化）

梯度下降法是最简单、最基础的优化算法之一。它通过反复沿着梯度的反方向（即最速下降方向）移动，来寻找可微函数的局部最小值。

#### 我们要解决什么问题？

我们希望最小化一个光滑函数 $f(x)$，其中 $x \in \mathbb{R}^n$：

$$
\min_x f(x)
$$

函数 $f(x)$ 可以表示成本、损失或误差，我们的目标是找到一个梯度（斜率）接近于零的点：

$$
\nabla f(x^*) = 0
$$

#### 它是如何工作的（通俗解释）

在每一步，我们通过*逆着*梯度的方向更新 $x$，因为这是 $f(x)$ 下降最快的方向。

$$
x_{t+1} = x_t - \eta \nabla f(x_t)
$$

其中：

- $\eta > 0$ 是学习率（步长）。
- $\nabla f(x_t)$ 是当前点的梯度向量。

算法持续进行，直到梯度变得非常小，或者 $x$ 或 $f(x)$ 的变化可以忽略不计为止。

#### 逐步示例

让我们最小化 $f(x) = x^2$。
那么 $\nabla f(x) = 2x$。

$$
x_{t+1} = x_t - \eta (2x_t) = (1 - 2\eta)x_t
$$

如果 $0 < \eta < 1$，序列将收敛到全局最小值 $x=0$。

#### 微型代码（简单实现）

Python

```python
import numpy as np

def gradient_descent(fprime, x0, eta=0.1, tol=1e-6, max_iter=1000):
    x = x0
    for _ in range(max_iter):
        grad = fprime(x)
        if np.linalg.norm(grad) < tol:
            break
        x -= eta * grad
    return x

# 示例：最小化 f(x) = x^2 + y^2
fprime = lambda v: np.array([2*v[0], 2*v[1]])
x_min = gradient_descent(fprime, np.array([5.0, -3.0]))
print("最小值位于:", x_min)
```

C

```c
#include <stdio.h>
#include <math.h>

void grad(double x[], double g[]) {
    g[0] = 2*x[0];
    g[1] = 2*x[1];
}

void gradient_descent(double x[], double eta, double tol, int max_iter) {
    double g[2];
    for (int i = 0; i < max_iter; i++) {
        grad(x, g);
        double norm = sqrt(g[0]*g[0] + g[1]*g[1]);
        if (norm < tol) break;
        x[0] -= eta * g[0];
        x[1] -= eta * g[1];
    }
}

int main(void) {
    double x[2] = {5.0, -3.0};
    gradient_descent(x, 0.1, 1e-6, 1000);
    printf("最小值位于 (%.4f, %.4f)\n", x[0], x[1]);
    return 0;
}
```

#### 为什么它很重要

- 是机器学习、深度学习和优化理论的基础。
- 在高维空间中有效，且每步计算简单。
- 是更高级算法（如 SGD、动量法、Adam）的基础。

#### 一个温和的证明（为什么它有效）

对于具有 Lipschitz 连续梯度（$L$-光滑）的凸可微函数 $f(x)$，更新规则保证：

$$
f(x_{t+1}) \le f(x_t) - \frac{\eta}{2}|\nabla f(x_t)|^2
$$

如果 $0 < \eta \le \frac{1}{L}$。

这意味着每一步都会使 $f(x)$ 减少一个与梯度幅度的平方成正比的量，从而保证收敛。

#### 亲自尝试

1.  从 $(5,-3)$ 开始，最小化 $f(x)=x^2+y^2$。
2.  尝试 $\eta=0.1$、$\eta=0.5$ 和 $\eta=1.0$，看看哪个收敛最快。
3.  添加一个基于 $|f(x_{t+1}) - f(x_t)|$ 的停止条件。
4.  在 $f(x,y)$ 的等高线图上可视化路径。
5.  扩展到非凸函数，如 $f(x)=x^4 - 3x^3 + 2$。

#### 测试用例

| 函数        | 梯度          | 起始点 $x_0$ | $\eta$ | 结果         | 真实最小值   |
| ----------- | ------------- | ------------ | ------ | ------------ | ------------ |
| $x^2$       | $2x$          | 5            | 0.1    | 0.0000       | 0            |
| $x^2+y^2$   | $(2x,2y)$     | (5,-3)       | 0.1    | (0,0)        | (0,0)        |
| $(x-2)^2$   | $2(x-2)$      | 0            | 0.1    | 2.0000       | 2            |

#### 复杂度

- 时间：$O(k)$ 次迭代（取决于学习率和容差）
- 空间：$O(n)$

梯度下降法是通用的下降路径，是一种简单、优雅的方法，构成了整个现代计算领域中优化和学习的基础。
### 585 随机梯度下降 (SGD)

随机梯度下降 (SGD) 是现代机器学习的核心工具。它通过使用*随机样本*（或小批量样本）来估计每一步的梯度，从而扩展了普通梯度下降法，使其能够高效地扩展到海量数据集。

#### 我们要解决什么问题？

我们的目标是最小化一个定义为许多基于样本的损失函数平均值的函数：

$$
f(x) = \frac{1}{N} \sum_{i=1}^N f_i(x)
$$

当 $N$ 很大时，在每次迭代中计算完整梯度 $\nabla f(x) = \frac{1}{N}\sum_i \nabla f_i(x)$ 可能非常昂贵。

SGD 通过每一步仅使用一个（或几个）随机样本来避免这种情况：

$$
x_{t+1} = x_t - \eta \nabla f_{i_t}(x_t)
$$

其中 $i_t$ 是从 ${1,2,\dots,N}$ 中随机选择的。

#### 它是如何工作的（通俗解释）

SGD 不是计算整个"地形"的*精确*斜率，而是采用一个噪声较大但计算成本低得多的斜率估计。它曲折地向下行进，有时会走过头，有时会修正，但总体趋势是朝着最小值前进。

这种随机性就像"内置的探索"，有助于 SGD 在非凸问题中逃离浅层局部最小值。

#### 算法步骤

1.  初始化 $x_0$（随机或零）。
2.  对于每次迭代 $t$：
    *   随机采样 $i_t \in {1,\dots,N}$
    *   计算梯度估计 $g_t = \nabla f_{i_t}(x_t)$
    *   更新参数：$x_{t+1} = x_t - \eta g_t$
3.  可选地，随时间衰减学习率 $\eta_t$。

常见的衰减策略：
$$
\eta_t = \frac{\eta_0}{1 + \lambda t}
$$

#### 微型代码（简单示例）

Python

```python
import numpy as np

def sgd(fprime, x0, data, eta=0.1, epochs=1000):
    x = x0
    N = len(data)
    for t in range(epochs):
        i = np.random.randint(N)
        grad = fprime(x, data[i])
        x -= eta * grad
    return x

# 示例：最小化样本上的平均 (x - y)^2
data = np.array([1.0, 2.0, 3.0, 4.0])
def grad(x, y): return 2*(x - y)
x_min = sgd(grad, x0=0.0, data=data)
print("估计的最小值:", x_min)
```

C

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double grad(double x, double y) {
    return 2*(x - y);
}

double sgd(double *data, int N, double x0, double eta, int epochs) {
    double x = x0;
    srand(time(NULL));
    for (int t = 0; t < epochs; t++) {
        int i = rand() % N;
        double g = grad(x, data[i]);
        x -= eta * g;
    }
    return x;
}

int main(void) {
    double data[] = {1.0, 2.0, 3.0, 4.0};
    int N = 4;
    double xmin = sgd(data, N, 0.0, 0.1, 1000);
    printf("估计的最小值: %.4f\n", xmin);
    return 0;
}
```

#### 为什么它很重要

-   对于训练神经网络和大规模模型至关重要。
-   能高效处理数十亿数据点。
-   天然适合流式或在线学习场景。
-   随机性有助于避免非凸"地形"中的不良局部最小值。

#### 一个温和的证明（为什么它有效）

如果学习率 $\eta_t$ 满足
$$
\sum_t \eta_t = \infty \quad \text{且} \quad \sum_t \eta_t^2 < \infty,
$$
那么在温和的凸性和平滑性假设下，SGD 在期望上收敛到真实最小值 $x^*$。

随机性引入了*方差*，但通过平均或减小 $\eta_t$ 可以控制它。

#### 亲自尝试

1.  针对随机样本 $y_i$ 最小化 $f(x) = \frac{1}{N}\sum_i (x - y_i)^2$。
2.  比较完整梯度下降与 SGD 的收敛速度。
3.  添加学习率衰减：$\eta_t = \eta_0/(1+0.01t)$。
4.  尝试小批量 SGD（每步使用多个样本）。
5.  绘制 $f(x_t)$ 随迭代变化的曲线，注意其虽有噪声但呈下降趋势。

#### 测试用例

| 函数       | 梯度       | 数据集         | 结果   | 真实最小值 |
| ---------- | ---------- | -------------- | ------ | ---------- |
| $(x-y)^2$  | $2(x-y)$   | [1,2,3,4]      | ≈ 2.5  | 2.5        |
| $(x-5)^2$  | $2(x-5)$   | [5]*100        | 5.0    | 5.0        |
| $(x-y)^2$  | [1,1000]   | 大方差         | ~500   | 500        |

#### 复杂度

-   时间：$O(k)$ 次迭代（每次使用一个或少量样本）
-   空间：$O(1)$ 或 $O(\text{小批量大小})$
-   收敛性：次线性但可扩展

SGD 是现代学习的核心，是一个简单而强大的思想，它以精度换取速度，让大规模系统能够高效地从数据海洋中学习。
### 586 牛顿法（多元优化）

多维牛顿法将一维求根方法推广到高效定位光滑函数驻点的问题上。它同时利用梯度（一阶导数）和海森矩阵（二阶导数矩阵）来向最优点进行二次步进。

#### 我们正在解决什么问题？

我们希望最小化一个光滑函数 $f(x)$，其中 $x \in \mathbb{R}^n$：

$$
\min_x f(x)
$$

在最小值点，梯度为零：

$$
\nabla f(x^*) = 0.
$$

牛顿法通过用二阶泰勒展开局部逼近 $f(x)$ 来改进猜测 $x_t$：

$$
f(x+\Delta x) \approx f(x) + \nabla f(x)^T \Delta x + \frac{1}{2} \Delta x^T H(x) \Delta x,
$$

其中 $H(x)$ 是海森矩阵。

令此逼近的梯度为零，得到：

$$
H(x) \Delta x = -\nabla f(x),
$$

从而导出更新规则：

$$
x_{t+1} = x_t - H(x_t)^{-1} \nabla f(x_t).
$$

#### 它是如何工作的（通俗解释）

想象你站在代表 $f(x)$ 的曲面上。梯度告诉你斜率，但海森矩阵告诉你斜率本身是如何弯曲的。通过结合两者，牛顿法直接跳向该曲线二次逼近的局部最小值点，当曲面性质良好时，通常只需几步即可收敛。

#### 微型代码（示例）

Python

```python
import numpy as np

def newton_multivariate(fprime, hessian, x0, tol=1e-6, max_iter=100):
    x = x0
    for _ in range(max_iter):
        g = fprime(x)
        H = hessian(x)
        if np.linalg.norm(g) < tol:
            break
        dx = np.linalg.solve(H, g)
        x -= dx
    return x

# 示例：f(x, y) = x^2 + y^2
fprime = lambda v: np.array([2*v[0], 2*v[1]])
hessian = lambda v: np.array([[2, 0], [0, 2]])
x_min = newton_multivariate(fprime, hessian, np.array([5.0, -3.0]))
print("最小值点位于：", x_min)
```

C (简化版)

```c
#include <stdio.h>
#include <math.h>

void gradient(double x[], double g[]) {
    g[0] = 2*x[0];
    g[1] = 2*x[1];
}

void hessian(double H[2][2]) {
    H[0][0] = 2; H[0][1] = 0;
    H[1][0] = 0; H[1][1] = 2;
}

void newton(double x[], double tol, int max_iter) {
    double g[2], H[2][2];
    for (int k = 0; k < max_iter; k++) {
        gradient(x, g);
        if (sqrt(g[0]*g[0] + g[1]*g[1]) < tol) break;
        hessian(H);
        x[0] -= g[0] / H[0][0];
        x[1] -= g[1] / H[1][1];
    }
}

int main(void) {
    double x[2] = {5.0, -3.0};
    newton(x, 1e-6, 100);
    printf("最小值点位于 (%.4f, %.4f)\n", x[0], x[1]);
    return 0;
}
```

#### 为什么它很重要

-   在最优点附近极快（二次收敛）。
-   是许多高级求解器（如 BFGS、Newton-CG 和信赖域方法）的基础。
-   是优化、机器学习和数值分析的核心。

#### 一个温和的证明（为什么它有效）

在真实最小值点 $x^*$ 附近，函数行为几乎是二次的：

$$
f(x) \approx f(x^*) + \frac{1}{2}(x-x^*)^T H(x^*)(x-x^*).
$$

因此，牛顿更新 $x_{t+1}=x_t-H^{-1}\nabla f(x_t)$ 有效地精确求解了这个局部二次模型。当 $H$ 是正定的且 $x_t$ 足够接近 $x^*$ 时，收敛是二次的，误差大致按先前误差的平方缩小。

#### 亲自尝试

1.  从 $(5,-3)$ 开始最小化 $f(x,y)=x^2+y^2$。
2.  尝试一个非对角海森矩阵：$f(x,y)=x^2+xy+y^2$。
3.  与梯度下降法比较收敛速度。
4.  观察当海森矩阵不是正定时的行为。
5.  添加线搜索以提高鲁棒性。

#### 测试用例

| 函数                  | 梯度                  | 海森矩阵                                           | 起始点      | 结果     | 真实最小值 |
| --------------------- | --------------------- | ------------------------------------------------- | ---------- | ------ | ------------ |
| $x^2 + y^2$           | $(2x,\,2y)$           | $\begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$    | $(5,-3)$   | $(0,0)$ | $(0,0)$      |
| $(x-2)^2 + (y+1)^2$   | $(2(x-2),\,2(y+1))$   | $\operatorname{diag}(2,2)$                        | $(0,0)$    | $(2,-1)$ | $(2,-1)$     |
| $x^2 + xy + y^2$      | $(2x+y,\,x+2y)$       | $\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$    | $(3,-2)$   | $(0,0)$ | $(0,0)$      |

#### 复杂度

-   时间：每次迭代 $O(n^3)$（矩阵求逆或线性求解）
-   空间：$O(n^2)$（存储海森矩阵）

牛顿法是数学家的手术刀，精确、优雅且快速——当地形光滑时，只需几个谨慎的步骤就能直击最优点的核心。
### 587 共轭梯度法

共轭梯度（CG）法是一种用于求解大型线性方程组的迭代算法，方程组形式为

$$
A x = b
$$

其中 $A$ 是对称正定（SPD）矩阵。
它之所以特别强大，是因为它不需要矩阵求逆，甚至不需要显式存储 $A$，只需要能够计算矩阵-向量乘积。

CG 也可以看作是在高维空间中高效最小化二次函数的一种方法。

#### 我们要解决什么问题？

我们希望最小化二次型

$$
f(x) = \frac{1}{2}x^T A x - b^T x,
$$

其梯度为

$$
\nabla f(x) = A x - b.
$$

令 $\nabla f(x)=0$ 得到相同的方程 $A x = b$。

因此，求解 $A x = b$ 和最小化 $f(x)$ 是等价的。

#### 它是如何工作的（通俗解释）

当 $f(x)$ 的等高线被拉长时，普通的梯度下降法可能会曲折前进且收敛缓慢。
共轭梯度法通过确保每个搜索方向都与之前的所有方向 A-正交（共轭）来修正这个问题，这意味着每一步都在一个新的维度上消除误差，而不会抵消先前步骤取得的进展。

在精确算术中，对于 $n$ 个变量，它最多可以在 $n$ 步内找到精确解。

#### 算法步骤

1. 初始化 $x_0$（例如，全零向量）。
2. 计算初始残差 $r_0 = b - A x_0$ 并设置方向 $p_0 = r_0$。
3. 对于每次迭代 $k=0,1,2,\dots$：
   $$
   \alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}
   $$
   $$
   x_{k+1} = x_k + \alpha_k p_k
   $$
   $$
   r_{k+1} = r_k - \alpha_k A p_k
   $$
   如果 $|r_{k+1}|$ 足够小，则停止。
   否则：
   $$
   \beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}, \quad p_{k+1} = r_{k+1} + \beta_k p_k.
   $$

每个新的 $p_k$ 都相对于 $A$ 与之前的所有方向“共轭”。

#### 微型代码（最小实现）

Python

```python
import numpy as np

def conjugate_gradient(A, b, x0=None, tol=1e-6, max_iter=1000):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0
    r = b - A @ x
    p = r.copy()
    for _ in range(max_iter):
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x += alpha * p
        r_new = r - alpha * Ap
        if np.linalg.norm(r_new) < tol:
            break
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
    return x

# 示例：求解 A x = b
A = np.array([[4, 1], [1, 3]], dtype=float)
b = np.array([1, 2], dtype=float)
x = conjugate_gradient(A, b)
print("解:", x)
```

C

```c
#include <stdio.h>
#include <math.h>

void matvec(double A[2][2], double x[2], double y[2]) {
    y[0] = A[0][0]*x[0] + A[0][1]*x[1];
    y[1] = A[1][0]*x[0] + A[1][1]*x[1];
}

void conjugate_gradient(double A[2][2], double b[2], double x[2], int max_iter, double tol) {
    double r[2], p[2], Ap[2];
    matvec(A, x, r);
    for (int i = 0; i < 2; i++) {
        r[i] = b[i] - r[i];
        p[i] = r[i];
    }
    for (int k = 0; k < max_iter; k++) {
        matvec(A, p, Ap);
        double rr = r[0]*r[0] + r[1]*r[1];
        double alpha = rr / (p[0]*Ap[0] + p[1]*Ap[1]);
        x[0] += alpha * p[0];
        x[1] += alpha * p[1];
        for (int i = 0; i < 2; i++) r[i] -= alpha * Ap[i];
        double rr_new = r[0]*r[0] + r[1]*r[1];
        if (sqrt(rr_new) < tol) break;
        double beta = rr_new / rr;
        for (int i = 0; i < 2; i++) p[i] = r[i] + beta * p[i];
    }
}

int main(void) {
    double A[2][2] = {{4, 1}, {1, 3}};
    double b[2] = {1, 2};
    double x[2] = {0, 0};
    conjugate_gradient(A, b, x, 1000, 1e-6);
    printf("解: (%.4f, %.4f)\n", x[0], x[1]);
    return 0;
}
```

#### 为什么它很重要

- 非常适合大型稀疏系统，尤其是来自数值偏微分方程和有限元方法的系统。
- 避免显式矩阵求逆，只需要 $A p$ 乘积。
- 是机器学习、物理模拟和科学计算中的核心构建模块。

#### 一个温和的证明（为什么它有效）

每个方向 $p_k$ 的选择都满足

$$
p_i^T A p_j = 0 \quad \text{对于 } i \ne j,
$$

这确保了在 $A$-内积下的正交性。
这意味着每一步都沿着一个共轭方向消除误差，永远不会重新访问该方向。
对于一个 $n$ 维系统，最多 $n$ 步后，所有误差分量都会被消除。

#### 自己动手试试

1. 用
$$
A = 
\begin{bmatrix}
4 & 1\\
1 & 3
\end{bmatrix}, 
\quad 
b = 
\begin{bmatrix}
1\\
2
\end{bmatrix}.
$$
求解 $A x = b$。

2. 将结果与高斯消元法进行比较。

3. 将 $A$ 修改为非对称矩阵，观察失败或振荡现象。

4. 添加一个预处理器 $M^{-1}$ 来改善收敛性。

5. 绘制 $|r_k|$ 随迭代次数的变化图，并注意其几何衰减。

#### 测试用例

| $A$                                               | $b$         | 解 $x^*$          | 迭代次数 |
| ------------------------------------------------- | ----------- | ----------------------- | ----------- |
| $\begin{bmatrix} 4 & 1 \\ 1 & 3 \end{bmatrix}$    | $[1, 2]^T$  | $(0.0909,\, 0.6364)$    | 3           |
| $\operatorname{diag}(2, 5, 10)$                   | $[1, 1, 1]^T$ | $(0.5,\, 0.2,\, 0.1)$    | 3           |
| 随机 SPD $(5 \times 5)$                           | 随机 $b$    | 精确解                  | $< n$       |

#### 复杂度

- 时间：$O(k n)$（每次迭代需要一次矩阵-向量乘法）
- 空间：$O(n)$
- 收敛性：线性收敛，但对于条件良好的 $A$ 非常高效

共轭梯度法是数值优化中默默无闻的力量，它将几何与代数携手并用，以优雅的效率求解庞大的系统。
### 588 拉格朗日乘子法（约束优化）

拉格朗日乘子法提供了一种系统化的方法，用于寻找在等式约束条件下函数的极值（最小值或最大值）。它引入了称为*拉格朗日乘子*的辅助变量，这些变量以代数方式强制执行约束，从而将一个约束问题转化为无约束问题。

#### 我们要解决什么问题？

我们希望在满足一个或多个等式约束的条件下，最小化（或最大化）一个函数：
$$
f(x_1, x_2, \dots, x_n)
$$
约束条件为：
$$
g_i(x_1, x_2, \dots, x_n) = 0, \quad i = 1, 2, \dots, m.
$$

为简单起见，从一个约束 $g(x) = 0$ 开始。

#### 核心思想

在最优点，$f$ 的梯度必须与约束 $g$ 的梯度方向相同：

$$
\nabla f(x^*) = \lambda \nabla g(x^*),
$$

其中 $\lambda$ 是拉格朗日乘子。这捕捉了这样一个思想：沿着约束曲面的任何微小移动都无法进一步减小 $f$。

我们引入拉格朗日函数：

$$
\mathcal{L}(x, \lambda) = f(x) - \lambda g(x),
$$

并通过将所有导数设为零来寻找驻点：

$$
\nabla_x \mathcal{L} = 0, \quad g(x) = 0.
$$

#### 示例：经典的双变量情况

最小化
$$
f(x, y) = x^2 + y^2
$$
约束条件为
$$
x + y = 1.
$$

1.  构造拉格朗日函数：
    $$
    \mathcal{L}(x, y, \lambda) = x^2 + y^2 - \lambda(x + y - 1)
    $$

2.  求偏导数并令其为零：
    $$
    \frac{\partial \mathcal{L}}{\partial x} = 2x - \lambda = 0
    $$
    $$
    \frac{\partial \mathcal{L}}{\partial y} = 2y - \lambda = 0
    $$
    $$
    \frac{\partial \mathcal{L}}{\partial \lambda} = -(x + y - 1) = 0
    $$

3.  求解：从前两个方程得到 $2x = 2y = \lambda$ → $x = y$。
    代入约束 $x + y = 1$ → $x = y = 0.5$。

因此，最小值出现在 $(x, y) = (0.5, 0.5)$ 处，$\lambda = 1$。

#### 微型代码（简单示例）

Python

```python
import sympy as sp

x, y, lam = sp.symbols('x y lam')
f = x2 + y2
g = x + y - 1
L = f - lam * g

sol = sp.solve([sp.diff(L, x), sp.diff(L, y), sp.diff(L, lam)], [x, y, lam])
print(sol)
```

输出：

```
{x: 0.5, y: 0.5, lam: 1}
```

C（概念性数值）

```c
#include <stdio.h>

int main(void) {
    double x = 0.5, y = 0.5, lam = 1.0;
    printf("Minimum at (%.2f, %.2f), lambda = %.2f\n", x, y, lam);
    return 0;
}
```

#### 为何重要

-   是微积分、经济学和机器学习中约束优化的核心。
-   易于推广到多个约束和高维情况。
-   构成了用于凸优化和支持向量机的 KKT 条件（见下一节）的基础。

#### 一个温和的证明（几何视角）

在最优点，任何沿着约束曲面切向的移动都不能改变 $f(x)$。因此，$\nabla f$ 必须垂直于该曲面，即平行于 $\nabla g$。引入 $\lambda$ 允许我们在代数上等同这些方向，从而创建可解的方程。

#### 动手尝试

1.  在 $x + y = 1$ 约束下最小化 $f(x, y) = x^2 + y^2$。
2.  尝试在 $x - y = 0$ 约束下最小化 $f(x, y) = x^2 + 2y^2$。
3.  求解具有两个约束的问题：
    $$
    g_1(x, y) = x + y - 1 = 0, \quad g_2(x, y) = x - 2y = 0.
    $$
4.  观察 $\lambda_1, \lambda_2$ 如何作为强制执行约束的"权重"。

#### 测试用例

| 函数        | 约束条件     | 结果 $(x^*, y^*)$                     | $\lambda$ |
| ----------- | ------------ | ------------------------------------- | --------- |
| $x^2+y^2$   | $x+y=1$      | (0.5, 0.5)                            | 1         |
| $x^2+2y^2$  | $x-y=0$      | (0, 0)                                | 0         |
| $x^2+y^2$   | $x^2+y^2=4$  | 圆形约束 → 圆上任意点                 | 变量      |

#### 复杂度

-   符号解：求解方程的复杂度为 $O(n^3)$。
-   数值解：对于大型系统，使用迭代方法（牛顿-拉夫逊法，SQP）。

拉格朗日乘子法是自由与约束之间的数学桥梁，指导着在自然、设计或逻辑本身定义的边界上进行优化。
### 589 Karush–Kuhn–Tucker (KKT) 条件

Karush–Kuhn–Tucker (KKT) 条件将拉格朗日乘子法推广到处理非线性优化中的不等式和等式约束。
它们构成了现代约束优化，特别是凸优化、机器学习（支持向量机）和经济学中的基石。

#### 我们要解决什么问题？

我们希望最小化

$$
f(x)
$$

约束条件为

$$
g_i(x) \le 0 \quad (i = 1, \dots, m)
$$

和

$$
h_j(x) = 0 \quad (j = 1, \dots, p).
$$

其中：

- $g_i(x)$ 是不等式约束，
- $h_j(x)$ 是等式约束。

#### 拉格朗日函数

我们扩展拉格朗日函数的思想：

$$
\mathcal{L}(x, \lambda, \mu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \mu_j h_j(x),
$$

其中

- $\lambda_i \ge 0$ 是不等式约束的乘子，
- $\mu_j$ 是等式约束的乘子。

#### KKT 条件

对于一个最优点 $x^*$，必须存在 $\lambda_i$ 和 $\mu_j$ 满足以下四个条件：

1. 平稳性
   $$
   \nabla f(x^*) + \sum_{i=1}^m \lambda_i \nabla g_i(x^*) + \sum_{j=1}^p \mu_j \nabla h_j(x^*) = 0
   $$

2. 原始可行性
   $$
   g_i(x^*) \le 0, \quad h_j(x^*) = 0
   $$

3. 对偶可行性
   $$
   \lambda_i \ge 0 \quad \text{对所有 } i
   $$

4. 互补松弛性
   $$
   \lambda_i g_i(x^*) = 0 \quad \text{对所有 } i
   $$

互补松弛性意味着，如果一个约束不是"紧的"（非活跃的），其对应的 $\lambda_i$ 必须为零，它对解不施加任何"力"。

#### 示例：带约束的二次优化

最小化
$$
f(x) = x^2
$$
约束条件为
$$
g(x) = 1 - x \le 0.
$$

步骤 1：写出拉格朗日函数
$$
\mathcal{L}(x, \lambda) = x^2 + \lambda(1 - x)
$$

步骤 2：KKT 条件

- 平稳性：
  $$
  \frac{d\mathcal{L}}{dx} = 2x - \lambda = 0
  $$

- 原始可行性：
  $$
  1 - x \le 0
  $$

- 对偶可行性：
  $$
  \lambda \ge 0
  $$

- 互补松弛性：
  $$
  \lambda(1 - x) = 0
  $$

求解：
由平稳性得，$\lambda = 2x$。
由互补松弛性得，要么 $\lambda=0$，要么 $1-x=0$。

- 如果 $\lambda=0$，则 $x=0$。但 $1-x=1>0$，违反可行性。
- 如果 $1-x=0$，则 $x=1$ 且 $\lambda=2$。

解：$x^* = 1$，$\lambda^* = 2$。

#### 微型代码（符号计算示例）

Python

```python
import sympy as sp

x, lam = sp.symbols('x lam')
f = x2
g = 1 - x
L = f + lam * g

sol = sp.solve([sp.diff(L, x), g, lam >= 0, lam * g], [x, lam], dict=True)
print(sol)
```

输出：

```
$${x: 1, lam: 2}]
```

#### 为什么它重要

- 将拉格朗日乘子法推广到处理不等式约束。
- 是凸优化、机器学习（支持向量机）、计量经济学和工程设计的基础。
- 为最优性提供了必要条件（对于凸问题也是充分条件）。

#### 一个温和的证明（直观解释）

在最优点，任何可行的扰动 $\Delta x$ 都不能降低 $f(x)$。
只有当 $f$ 的梯度位于由活跃约束的梯度所形成的锥体内部时，这才可能。
KKT 乘子 $\lambda_i$ 在数学上表达了这种组合。

#### 亲自尝试

1. 最小化 $f(x)=x^2+y^2$，约束条件为 $x+y\ge1$。
2. 求解 $f(x)=x^2$，约束条件为 $x\ge1$。
3. 比较无约束解和约束解。
4. 使用符号微分实现一个 KKT 求解器。

#### 测试用例

| 函数       | 约束条件     | 结果 $(x^*)$ | $\lambda^*$ |
| ---------- | ------------ | ------------ | ----------- |
| $x^2$      | $1-x\le0$    | 1            | 2           |
| $x^2+y^2$  | $x+y-1=0$    | $(0.5, 0.5)$ | 1           |
| $x^2$      | $x\ge0$      | 0            | 0           |

#### 复杂度

- 符号求解：对于小型系统为 $O(n^3)$。
- 数值 KKT 求解器（例如序列二次规划）：每次迭代 $O(n^3)$。

KKT 条件是优化的语法，表达了目标和约束在可能性的边界上如何协商平衡。
### 590 坐标下降法

坐标下降法是最简单却出奇强大的优化算法之一。
它不是一次性调整所有变量，而是每次更新一个坐标，循环遍历变量直至收敛。
该方法广泛应用于 LASSO 回归、矩阵分解和稀疏优化。

#### 我们要解决什么问题？

我们想要最小化一个函数

$$
f(x_1, x_2, \dots, x_n)
$$

可能还需要满足简单的约束条件（例如 $x_i \ge 0$）。

#### 核心思想

我们不一次性处理完整的梯度 $\nabla f(x)$，而是固定除一个变量外的所有变量，并针对该变量进行最小化。

例如，在二维情况下：

$$
f(x, y) \rightarrow \text{首先固定 } y，\text{ 对 } x \text{ 最小化}；\text{ 然后固定 } x，\text{ 对 } y \text{ 最小化}。
$$

每次更新步骤都会降低目标函数值，对于凸函数而言，这会导致收敛。

#### 算法步骤

给定初始向量 $x^{(0)}$：

1. 对于每个坐标 $i = 1, \dots, n$：

   * 定义*偏函数* $f_i(x_i) = f(x_1, \dots, x_i, \dots, x_n)$，保持其他变量固定。
   * 找到
     $$
     x_i^{(k+1)} = \arg \min_{x_i} f_i(x_i)
     $$
2. 重复直到收敛（当 $f(x)$ 变化可忽略不计或 $|x^{(k+1)} - x^{(k)}| < \varepsilon$ 时）。

#### 示例：简单的二次函数

最小化
$$
f(x, y) = (x - 2)^2 + (y - 3)^2.
$$

1. 初始化 $(x, y) = (0, 0)$。
2. 固定 $y=0$：最小化 $(x-2)^2$ → $x=2$。
3. 固定 $x=2$：最小化 $(y-3)^2$ → $y=3$。
4. 完成，一次扫描即达到 $(2, 3)$。

对于凸二次函数，坐标下降法线性收敛到全局最小值。

#### 微型代码

Python

```python
import numpy as np

def coordinate_descent(f, grad, x0, tol=1e-6, max_iter=1000):
    x = x0.copy()
    n = len(x)
    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            g = grad(x)
            x[i] -= 0.1 * g[i]  # 沿着坐标梯度的简单步长
        if np.linalg.norm(x - x_old) < tol:
            break
    return x

# 示例：f(x,y) = (x-2)^2 + (y-3)^2
f = lambda v: (v[0]-2)2 + (v[1]-3)2
grad = lambda v: np.array([2*(v[0]-2), 2*(v[1]-3)])
x = coordinate_descent(f, grad, np.array([0.0, 0.0]))
print("最小值:", x)
```

C (简单版本)

```c
#include <stdio.h>
#include <math.h>

int main(void) {
    double x = 0, y = 0;
    double alpha = 0.1;
    for (int iter = 0; iter < 100; iter++) {
        double x_old = x, y_old = y;
        x -= alpha * 2 * (x - 2);
        y -= alpha * 2 * (y - 3);
        if (fabs(x - x_old) < 1e-6 && fabs(y - y_old) < 1e-6) break;
    }
    printf("最小值在 (%.6f, %.6f)\n", x, y);
    return 0;
}
```

#### 为何重要

- 实现极其简单。
- 当 $f(x)$ 是可分离的或按坐标凸时效果很好。
- 可扩展到非常高维，因为每一步只更新一个变量。
- 用于 LASSO、岭回归、支持向量机和矩阵分解。

#### 一个温和的证明（直观解释）

在每次迭代中，对一个坐标进行最小化不会增加 $f(x)$。
因此，$f(x^{(k)})$ 是非递增的并且收敛。
对于凸函数 $f$，这个极限就是全局最小值。

#### 亲自尝试

1.  最小化 $f(x, y) = (x-1)^2 + (2y-3)^2$。
2.  添加约束 $x, y \ge 0$。
3.  将梯度更新替换为每个坐标的精确一维最小化。
4.  尝试一个非凸的例子，观察收敛到局部最小值的情况。

#### 测试用例

| 函数                  | 起点   | 结果       | 备注                         |
| --------------------- | ------ | ---------- | ---------------------------- |
| $(x-2)^2+(y-3)^2$     | (0,0)  | (2,3)      | 二次函数，精确最小值         |
| $(x-1)^2+(y-1)^2$     | (5,5)  | (1,1)      | 线性收敛                     |
| $\sin(x)+\sin(y)$     | (2,2)  | (π/2, π/2) | 局部最小值                   |

#### 复杂度

-   每次坐标更新：$O(1)$（如果梯度计算成本低）。
-   一个完整循环：$O(n)$。
-   总复杂度：$k$ 次迭代为 $O(nk)$。

坐标下降法体现了优化的极简主义精神，一次只改进一小部分，却始终朝着整体解决方案迈进。

# 第 60 节. 代数技巧与变换技术
### 591 多项式乘法（FFT）

通过快速傅里叶变换将系数向量转换为点值，进行逐点相乘，然后再变换回来，从而快速计算两个多项式的乘积。这能以 $O(n\log n)$ 的时间复杂度计算卷积，而不是 $O(n^2)$。

#### 我们要解决什么问题？

给定
$$
A(x)=\sum_{i=0}^{n-1}a_i x^i,\quad B(x)=\sum_{j=0}^{m-1}b_j x^j,
$$
计算
$$
C(x)=A(x)B(x)=\sum_{k=0}^{n+m-2}c_k x^k,
\quad c_k=\sum_{i+j=k}a_i b_j.
$$
的系数。

朴素卷积的时间复杂度是 $O(nm)$。FFT 利用卷积定理，在 $O(N\log N)$ 的时间内完成，其中 $N$ 是大于等于 $n+m-1$ 的 2 的幂。

#### 工作原理

1.  选择大小：$N=\text{大于等于 } n+m-1 \text{ 的下一个 2 的幂}$。
2.  将 $a$ 和 $b$ 零填充到长度 $N$。
3.  对两个序列进行 FFT：$\hat a=\operatorname{FFT}(a)$, $\hat b=\operatorname{FFT}(b)$。
4.  逐点相乘：$\hat c_k=\hat a_k\hat b_k$。
5.  逆 FFT：$c=\operatorname{IFFT}(\hat c)$。
6.  如果输入是整数，则将实部四舍五入到最接近的整数。

卷积定理：
$$
\mathcal{F}(a*b)=\mathcal{F}(a)\odot \mathcal{F}(b).
$$

#### 精简代码

Python (NumPy, 实系数)

```python
import numpy as np

def poly_mul_fft(a, b):
    n = len(a) + len(b) - 1
    N = 1 << (n - 1).bit_length()
    fa = np.fft.rfft(a, N)
    fb = np.fft.rfft(b, N)
    fc = fa * fb
    c = np.fft.irfft(fc, N)[:n]
    # 如果输入是整数，四舍五入到最接近的整数
    return np.rint(c).astype(int)

# 示例
print(poly_mul_fft([1,2,3], [4,5]))  # [4,13,22,15]
```

C++17 (迭代 Cooley–Tukey 算法，使用 std::complex)

```cpp
#include <bits/stdc++.h>
using namespace std;

using cd = complex<double>;
const double PI = acos(-1);

void fft(vector<cd>& a, bool inv){
    int n = (int)a.size();
    static vector<int> rev;
    static vector<cd> roots{0,1};
    if ((int)rev.size() != n){
        int k = __builtin_ctz(n);
        rev.assign(n,0);
        for (int i=0;i<n;i++)
            rev[i] = (rev[i>>1]>>1) | ((i&1)<<(k-1));
    }
    if ((int)roots.size() < n){
        int k = __builtin_ctz(roots.size());
        roots.resize(n);
        while ((1<<k) < n){
            double ang = 2*PI/(1<<(k+1));
            for (int i=1<<(k-1); i<(1<<k); i++){
                roots[2*i]   = roots[i];
                roots[2*i+1] = cd(cos(ang*(2*i+1-(1<<k))), sin(ang*(2*i+1-(1<<k))));
            }
            k++;
        }
    }
    for (int i=0;i<n;i++) if (i<rev[i]) swap(a[i],a[rev[i]]);
    for (int len=1; len<n; len<<=1){
        for (int i=0;i<n;i+=2*len){
            for (int j=0;j<len;j++){
                cd u=a[i+j];
                cd v=a[i+j+len]*roots[len+j];
                a[i+j]=u+v;
                a[i+j+len]=u-v;
            }
        }
    }
    if (inv){
        reverse(a.begin()+1, a.end());
        for (auto& x:a) x/=n;
    }
}

vector<long long> multiply(const vector<long long>& A, const vector<long long>& B){
    int n = 1;
    int need = (int)A.size() + (int)B.size() - 1;
    while (n < need) n <<= 1;
    vector<cd> fa(n), fb(n);
    for (size_t i=0;i<A.size();i++) fa[i] = A[i];
    for (size_t i=0;i<B.size();i++) fb[i] = B[i];
    fft(fa,false); fft(fb,false);
    for (int i=0;i<n;i++) fa[i] *= fb[i];
    fft(fa,true);
    vector<long long> C(need);
    for (int i=0;i<need;i++) C[i] = llround(fa[i].real());
    return C;
}

int main(){
    vector<long long> a={1,2,3}, b={4,5};
    auto c = multiply(a,b); // 4 13 22 15
    for (auto x:c) cout<<x<<" ";
    cout<<"\n";
}
```

#### 为什么重要

-   将多项式乘法的时间复杂度从二次降低到拟线性。
-   是大整数算术、信号处理、通过卷积进行字符串匹配以及组合计数的核心。
-   可扩展到多维卷积。

#### 一个简要的证明思路

选择 $N$ 次单位根 $\omega_N^k$。DFT 在 $N$ 个不同的点上计算一个次数小于 $N$ 的多项式：
$$
\hat a_k=\sum_{t=0}^{N-1} a_t \omega_N^{kt}.
$$
逐点相乘得到 $C(x)=A(x)B(x)$ 在这些相同点上的值：
$\hat c_k=\hat a_k\hat b_k$。由于求值点是不同的，逆 DFT 可以唯一地重构出 $c_0,\dots,c_{N-1}$。

#### 实用技巧

-   选择 $N$ 为 2 的幂以获得最快的 FFT。
-   对于整数输入，在 IFFT 后进行四舍五入可以恢复精确系数。
-   大系数存在浮点误差风险。两种解决方法：
    1.  将系数分块，并使用两次或三次 FFT 配合中国剩余定理重构，
    2.  在素数模下使用 NTT 进行精确的模卷积。
-   修剪尾随的零。

#### 动手尝试

1.  手动计算 $(1+2x+3x^2)$ 乘以 $(4+5x)$，并与 FFT 结果比较。
2.  对两个长度为 10 的随机向量进行卷积，并与朴素的 $O(n^2)$ 算法验证。
3.  测量当规模翻倍时，运行时间与朴素算法的对比。
4.  实现循环卷积，并通过零填充与线性卷积进行比较。
5.  通过缩放输入来探索双精度的极限。

#### 测试用例

| 输入 A     | 输入 B     | 输出 C               |
| ---------- | ---------- | -------------------- |
| $[1,2,3]$  | $[4,5]$    | $[4,13,22,15]$       |
| $[1,0,1]$  | $[1,1]$    | $[1,1,1,1]$          |
| $[2,3,4]$  | $[5,6,7]$  | $[10,27,52,45,28]$   |

#### 复杂度

-   时间：$O(N\log N)$，其中 $N\ge n+m-1$
-   空间：$O(N)$

基于 FFT 的乘法是处理大型多项式和整数乘积的标准快速方法。
### 592 多项式求逆（牛顿迭代）

多项式求逆旨在找到一个级数 $B(x)$，使得

$$
A(x)B(x)\equiv1\pmod{x^n}.
$$

这意味着 $A$ 和 $B$ 相乘的结果，在 $n-1$ 次项之前应为 $1$。这是使用迭代求精计算 $1/a$ 的多项式版本，建立在相同的牛顿-拉弗森原理之上。

#### 我们要解决什么问题？

给定一个形式幂级数

$$
A(x)=a_0+a_1x+a_2x^2+\dots,
$$

我们希望找到另一个级数

$$
B(x)=b_0+b_1x+b_2x^2+\dots
$$

使得

$$
A(x)B(x)=1.
$$

这仅在 $a_0\ne0$ 时才有可能。

#### 级数的牛顿法

我们使用牛顿迭代，类似于数值求逆。令 $B_k(x)$ 表示我们当前对模 $x^k$ 的近似值。然后我们使用以下公式进行改进：

$$
B_{2k}(x)=B_k(x),(2-A(x)B_k(x)) \pmod{x^{2k}}.
$$

每次迭代都会使正确的项数翻倍。

#### 算法步骤

1.  从 $B_1(x)=1/a_0$ 开始。
2.  对于 $k=1,2,4,8,\dots$ 直到 $k\ge n$：
    *   计算 $C(x)=A(x)B_k(x)\pmod{x^{2k}}$。
    *   更新 $B_{2k}(x)=B_k(x),(2-C(x))\pmod{x^{2k}}$。
3.  将 $B(x)$ 截断到 $n-1$ 次。

所有多项式乘法都使用 FFT 或 NTT 以提高速度。

#### 示例

假设

$$
A(x)=1+x.
$$

我们期望

$$
B(x)=1-x+x^2-x^3+\dots.
$$

步骤 1：$B_1=1$。
步骤 2：$A(x)B_1=1+x$，所以

$$
B_2=B_1(2-(1+x))=1-x.
$$

步骤 3：
$A(x)B_2=(1+x)(1-x)=1-x^2$，

$$
B_4=B_2(2-(1-x^2))=(1-x)(1+x^2)=1-x+x^2-x^3.
$$

依此类推。每一步都使精度翻倍。

#### 微型代码

Python（使用 NumPy FFT）

```python
import numpy as np

def poly_mul(a, b):
    n = len(a) + len(b) - 1
    N = 1 << (n - 1).bit_length()
    fa = np.fft.rfft(a, N)
    fb = np.fft.rfft(b, N)
    fc = fa * fb
    c = np.fft.irfft(fc, N)[:n]
    return c

def poly_inv(a, n):
    b = np.array([1 / a[0]])
    k = 1
    while k < n:
        k *= 2
        ab = poly_mul(a[:k], b)[:k]
        b = (b * 2 - poly_mul(b, ab)[:k])[:k]
    return b[:n]

# 示例：求 1 + x 的逆
a = np.array([1.0, 1.0])
b = poly_inv(a, 8)
print(np.round(b, 3))
# [1. -1. 1. -1. 1. -1. 1. -1.]
```

#### 为何重要

-   用于基于 FFT 的算术中的级数除法、模逆元和多项式除法。
-   是形式幂级数计算中的核心原语。
-   出现在组合数学、符号代数和计算机代数系统中。

#### 牛顿更新背后的直觉

如果我们希望 $AB=1$，定义
$$
F(B)=A(x)B(x)-1.
$$
牛顿迭代给出

$$
B_{k+1}=B_k-F(B_k)/A(x)=B_k-(A(x)B_k-1)B_k=B_k(2-A(x)B_k),
$$

这与我们的多项式更新规则一致。

#### 动手尝试

1.  求 $A(x)=1+x+x^2$ 的逆，直到 8 次项。
2.  通过计算 $A(x)B(x)$ 并确认常数项以上的所有项都为零来验证。
3.  在素数 $p$ 下使用模算术实现。
4.  比较基于 FFT 的方法与朴素方法的性能。

#### 测试用例

| $A(x)$ | $B(x)$ (前几项)             | 验证                          |
| ------ | -------------------------- | --------------------------- |
| $1+x$  | $1-x+x^2-x^3+\dots$        | $(1+x)(1-x+x^2-\dots)=1$    |
| $1-2x$ | $1+2x+4x^2+8x^3+\dots$     | $(1-2x)(1+2x+4x^2+\dots)=1$ |
| $2+x$  | $0.5-0.25x+0.125x^2-\dots$ | $(2+x)(B)=1$                |

#### 复杂度

-   每次迭代精度翻倍。
-   使用 FFT 乘法 → $O(n\log n)$。
-   内存：$O(n)$。

通过牛顿迭代进行多项式求逆是代数效率的典范，一个简单的更新每次都能使精度翻倍。
### 593 多项式求导

多项式求导是最简单的符号运算之一，但它广泛出现在优化、求根、级数处理和数值分析的算法中。

给定多项式
$$
A(x)=a_0+a_1x+a_2x^2+\dots+a_nx^n,
$$
其导数为
$$
A'(x)=a_1+2a_2x+3a_3x^2+\dots+n a_nx^{n-1}.
$$

#### 我们要解决什么问题？

我们希望高效地计算导数系数，并以与 $A(x)$ 相同的系数形式表示 $A'(x)$。
如果
$$
A(x)=[a_0,a_1,\dots,a_n],
$$
那么
$$
A'(x)=[a_1,2a_2,3a_3,\dots,n a_n].
$$

此操作在 $O(n)$ 时间内运行，是多项式代数和形式幂级数微积分中的关键子程序。

#### 算法步骤

1.  给定系数 $a_0,\dots,a_n$。
2.  对于每个从 $1$ 到 $n$ 的 $i$：
    *   计算 $b_{i-1}=i\times a_i$。
3.  返回 $A'(x)$ 的系数 $[b_0,b_1,\dots,b_{n-1}]$。

#### 示例

令
$$
A(x)=3+2x+5x^2+4x^3.
$$

那么
$$
A'(x)=2+10x+12x^2.
$$

以系数形式表示：

| 项    | 系数 | 新系数          |
| ----- | ---- | --------------- |
| $x^0$ | $3$  | ,               |
| $x^1$ | $2$  | $2\times1=2$    |
| $x^2$ | $5$  | $5\times2=10$   |
| $x^3$ | $4$  | $4\times3=12$   |

#### 微型代码

Python

```python
def poly_derivative(a):
    n = len(a)
    return [i * a[i] for i in range(1, n)]

# 示例
A = [3, 2, 5, 4]
print(poly_derivative(A))  # [2, 10, 12]
```

C

```c
#include <stdio.h>

int main(void) {
    double a[] = {3, 2, 5, 4};
    int n = 4;
    double d[n - 1];
    for (int i = 1; i < n; i++)
        d[i - 1] = i * a[i];
    for (int i = 0; i < n - 1; i++)
        printf("%.0f ", d[i]);
    printf("\n");
    return 0;
}
```

#### 为何重要

-   是 Newton–Raphson（牛顿-拉夫逊）求根法、梯度下降法和优化中的核心操作。
-   出现在多项式除法、泰勒级数、微分方程和符号代数中。
-   作为代数基础，用于自动微分和反向传播。

#### 一个温和的证明

根据微积分：
$$
\frac{d}{dx}(x^i)=i x^{i-1}.
$$

由于 $A(x)=\sum_i a_i x^i$，根据线性性质可得
$$
A'(x)=\sum_i a_i i x^{i-1}.
$$
因此，$A'(x)$ 中 $x^{i-1}$ 的系数是 $i a_i$。

#### 亲自尝试

1.  对 $A(x)=5x^4+2x^3-x^2+3x-7$ 求导。
2.  计算两次导数（二阶导数）。
3.  为大型整数多项式实现模 $p$ 求导。
4.  在 Newton–Raphson（牛顿-拉夫逊）根更新中使用导数：
    $x_{k+1}=x_k-\frac{A(x_k)}{A'(x_k)}.$

#### 测试用例

| 多项式             | 导数           | 结果          |
| ------------------ | -------------- | ------------- |
| $3+2x+5x^2+4x^3$ | $2+10x+12x^2$ | `[2,10,12]` |
| $x^4$              | $4x^3$         | `[0,0,0,4]` |
| $1+x+x^2+x^3$    | $1+2x+3x^2$   | `[1,2,3]`   |

#### 复杂度

-   时间复杂度：$O(n)$
-   空间复杂度：$O(n)$

多项式求导是一个单行操作，但它构成了算法微积分的很大一部分，是推动重大变革的一小步。
### 594 多项式积分

多项式积分是微分的逆运算：我们寻找一个多项式 $B(x)$，使得 $B'(x)=A(x)$。
这是符号代数、数值积分和生成函数中一个简单但至关重要的工具。

#### 我们要解决什么问题？

给定
$$
A(x)=a_0+a_1x+a_2x^2+\dots+a_{n-1}x^{n-1},
$$
我们想要
$$
B(x)=\int A(x),dx=C+b_1x+b_2x^2+\dots+b_nx^n,
$$
其中每个系数满足
$$
b_{i+1}=\frac{a_i}{i+1}.
$$

通常，出于计算目的，我们设积分常数 $C=0$。

#### 工作原理

如果 $A(x)=[a_0,a_1,a_2,\dots,a_{n-1}]$，
那么
$$
B(x)=[0,\frac{a_0}{1},\frac{a_1}{2},\frac{a_2}{3},\dots,\frac{a_{n-1}}{n}].
$$

每一项都除以其新的指数索引。

#### 示例

令
$$
A(x)=2+10x+12x^2.
$$

那么
$$
B(x)=2x+\frac{10}{2}x^2+\frac{12}{3}x^3=2x+5x^2+4x^3.
$$

以系数形式表示：

| 项    | $A(x)$ 中的系数 | 积分后的项       | $B(x)$ 中的系数 |
| ----- | --------------- | ---------------- | --------------- |
| $x^0$ | 2               | $2x$             | 2               |
| $x^1$ | 10              | $10x^2/2$        | 5               |
| $x^2$ | 12              | $12x^3/3$        | 4               |

所以 $B(x)=[0,2,5,4]$。

#### 微型代码

Python

```python
def poly_integrate(a):
    n = len(a)
    return [0.0] + [a[i] / (i + 1) for i in range(n)]

# 示例
A = [2, 10, 12]
print(poly_integrate(A))  # [0.0, 2.0, 5.0, 4.0]
```

C

```c
#include <stdio.h>

int main(void) {
    double a[] = {2, 10, 12};
    int n = 3;
    double b[n + 1];
    b[0] = 0.0;
    for (int i = 0; i < n; i++)
        b[i + 1] = a[i] / (i + 1);
    for (int i = 0; i <= n; i++)
        printf("%.2f ", b[i]);
    printf("\n");
    return 0;
}
```

#### 为何重要

- 将微分方程转换为积分形式。
- 是符号微积分和自动积分的核心。
- 用于泰勒级数重构、原函数计算和形式幂级数分析。
- 在计算上下文中，支持模 $p$ 积分，用于精确的代数操作。

#### 一个温和的证明

我们从微积分中知道：
$$
\frac{d}{dx}(x^{i+1})=(i+1)x^i.
$$

因此，为了"撤销"微分，每一项的系数都要除以 $(i+1)$：
$$
\int a_i x^i dx = \frac{a_i}{i+1}x^{i+1}.
$$

积分的线性性保证了整个多项式遵循相同的规则。

#### 动手试试

1.  对 $A(x)=3+4x+5x^2$ 进行积分，设 $C=0$。
2.  添加一个非零积分常数 $C=7$。
3.  通过微分你的结果来验证。
4.  实现模运算下的积分（$\text{mod }p$）。
5.  使用积分计算多项式曲线下从 $x=0$ 到 $x=1$ 的面积。

#### 测试用例

| $A(x)$        | $B(x)$                          | 验证                     |
| ------------- | ------------------------------- | ------------------------ |
| $2+10x+12x^2$ | $2x+5x^2+4x^3$                  | $B'(x)=A(x)$             |
| $1+x+x^2$     | $x+\frac{x^2}{2}+\frac{x^3}{3}$ | 求导可恢复 $A(x)$        |
| $5x^2$        | $\frac{5x^3}{3}$                | 求导得到 $5x^2$          |

#### 复杂度

- 时间：$O(n)$
- 空间：$O(n)$

多项式积分虽然简单但很基础，它一次一个分数地将离散系数转化为连续曲线。
### 595 形式幂级数复合

形式幂级数（FPS）复合是指将一个级数代入另一个级数的操作，
$$
C(x)=A(B(x)),
$$
其中 $A(x)$ 和 $B(x)$ 都是形式幂级数，无需考虑收敛性，我们只关心到选定阶数 $n$ 的系数。

这是代数组合学、符号计算和生成函数分析中的基本操作。

#### 我们要解决什么问题？

给定
$$
A(x)=a_0+a_1x+a_2x^2+\dots,
$$
和
$$
B(x)=b_0+b_1x+b_2x^2+\dots,
$$
我们想要
$$
C(x)=A(B(x))=a_0+a_1B(x)+a_2(B(x))^2+a_3(B(x))^3+\dots.
$$

我们只计算到 $n-1$ 阶的项。

#### 关键假设

1. 通常 $b_0=0$（即 $B(x)$ 没有常数项），否则 $A(B(x))$ 会涉及常数平移。
2. 目标阶数 $n$ 限制了所有计算，每一步都进行截断。

#### 工作原理

1. 初始化 $C(x)=[a_0]$。
2. 对于每个 $k\ge1$：

   * 使用重复卷积或预计算的幂次来计算 $(B(x))^k$。
   * 乘以 $a_k$。
   * 加到 $C(x)$ 上，并在 $n-1$ 阶后截断。

数学上表示为：
$$
C(x)=\sum_{k=0}^{n-1} a_k (B(x))^k \pmod{x^n}.
$$

高效的算法使用分治法和基于 FFT 的多项式乘法。

#### 示例

令
$$
A(x)=1+x+x^2, \quad B(x)=x+x^2.
$$

那么：
$$
A(B(x))=1+(x+x^2)+(x+x^2)^2=1+x+x^2+x^2+2x^3+x^4.
$$
简化后：
$$
A(B(x))=1+x+2x^2+2x^3+x^4.
$$

系数形式：$[1,1,2,2,1]$。

#### 微型代码

Python

```python
import numpy as np

def poly_mul(a, b, n):
    m = len(a) + len(b) - 1
    N = 1 << (m - 1).bit_length()
    fa = np.fft.rfft(a, N)
    fb = np.fft.rfft(b, N)
    fc = fa * fb
    c = np.fft.irfft(fc, N)[:n]
    return np.rint(c).astype(int)

def poly_compose(a, b, n):
    res = np.zeros(n, dtype=int)
    term = np.ones(1, dtype=int)
    for i in range(len(a)):
        if i > 0:
            term = poly_mul(term, b, n)
        res[:len(term)] += a[i] * term[:n]
    return res[:n]

# 示例
A = [1, 1, 1]
B = [0, 1, 1]
print(poly_compose(A, B, 5))  # [1, 1, 2, 2, 1]
```

#### 为什么它很重要

- 复合是组合类生成函数的核心。
- 出现在幂级数求逆、函数迭代和微分方程中。
- 用于复合函数的泰勒展开和符号代数系统。
- 构建指数生成函数和级数变换所必需。

#### 一个温和的证明

使用形式定义：
$$
A(x)=\sum_{k=0}^\infty a_kx^k.
$$

然后将 $B(x)$ 代入得到
$$
A(B(x))=\sum_{k=0}^\infty a_k (B(x))^k.
$$

由于我们只保留到 $x^{n-1}$ 的系数，更高阶的项在模 $x^n$ 下消失。
每个 $(B(x))^k$ 贡献的项阶数 $\ge k$，这确保了计算是有限的。

#### 亲自尝试

1. 计算 $A(x)=1+x+x^2$, $B(x)=x+x^2$ 时的 $A(B(x))$。
2. 与符号展开进行比较。
3. 测试 $A(x)=\exp(x)$ 和 $B(x)=\sin(x)$ 到 6 阶。
4. 为较大的 $n$ 实现基于 FFT 的截断复合。
5. 探索求逆：找到 $B(x)$ 使得 $A(B(x))=x$。

#### 测试用例

| $A(x)$    | $B(x)$  | 到 $x^4$ 的结果       |
| --------- | ------- | --------------------- |
| $1+x+x^2$ | $x+x^2$ | $1+x+2x^2+2x^3+x^4$ |
| $1+2x$    | $x+x^2$ | $1+2x+2x^2$         |
| $1+x^2$   | $x+x^2$ | $1+x^2+2x^3+x^4$    |

#### 复杂度

- 朴素方法：$O(n^2)$
- 基于 FFT 的方法：每次乘法 $O(n\log n)$
- 分治法复合：使用高级算法可达 $O(n^{1.5})$

形式幂级数复合是代数与结构相遇之处，将一个无穷级数代入另一个，以构建新的函数、模式和生成法则。
### 596 平方取幂法

平方取幂法是一种高效计算幂的快速方法。
它采用分治策略，通过每一步将指数减半，而不是重复相乘，从而将乘法次数从 $O(n)$ 减少到 $O(\log n)$。

它是模幂运算、快速矩阵幂运算和多项式幂运算背后的核心方法。

#### 我们要解决什么问题？

我们想要计算
$$
a^n
$$
其中 $a$ 为整数（或多项式、矩阵），$n$ 为非负整数。

朴素的方法是 $a$ 自乘 $n$ 次，这很慢。
平方取幂法利用 $n$ 的二进制展开来跳过不必要的乘法。

#### 核心思想

使用以下规则：

$$
a^n =
\begin{cases}
1, & n = 0, \\[6pt]
\left(a^{\,n/2}\right)^2, & n \text{ 为偶数}, \\[6pt]
a \cdot \left(a^{\,(n-1)/2}\right)^2, & n \text{ 为奇数}.
\end{cases}
$$


每一步都将 $n$ 减半，因此只需要 $\log_2 n$ 层递归。

#### 示例

计算 $a^9$：

| 步骤  | 表达式          | 说明           |
| ----- | --------------- | -------------- |
| $a^9$ | $a\cdot(a^4)^2$ | 使用奇数规则   |
| $a^4$ | $(a^2)^2$       | 使用偶数规则   |
| $a^2$ | $(a^1)^2$       | 基础递归       |
| $a^1$ | $a$             | 基本情况       |

总计：4 次乘法，而不是 8 次。

#### 精简代码

Python

```python
def power(a, n):
    if n == 0:
        return 1
    if n % 2 == 0:
        half = power(a, n // 2)
        return half * half
    else:
        half = power(a, (n - 1) // 2)
        return a * half * half

print(power(3, 9))  # 19683
```

C

```c
#include <stdio.h>

long long power(long long a, long long n) {
    if (n == 0) return 1;
    if (n % 2 == 0) {
        long long half = power(a, n / 2);
        return half * half;
    } else {
        long long half = power(a, n / 2);
        return a * half * half;
    }
}

int main(void) {
    printf("%lld\n", power(3, 9)); // 19683
    return 0;
}
```

模运算版本 (Python)

```python
def modpow(a, n, m):
    res = 1
    a %= m
    while n > 0:
        if n & 1:
            res = (res * a) % m
        a = (a * a) % m
        n >>= 1
    return res

print(modpow(3, 200, 1000000007))  # 快速
```

#### 为何重要

- 用于模运算、密码学、RSA 和离散指数运算。
- 对于动态规划中的矩阵幂运算和斐波那契数列计算至关重要。
- 是多项式幂运算、快速倍增法和快速模逆运算的基础。

#### 一个温和的证明

每一步都将指数减半，同时保持等价性：

对于偶数 $n$：
$$
a^n=(a^{n/2})^2.
$$
对于奇数 $n$：
$$
a^n=a\cdot(a^{(n-1)/2})^2.
$$
由于每次递归都使用 $n/2$，总调用次数为 $O(\log n)$。

#### 动手尝试

1.  使用重复平方法手动计算 $2^{31}$。
2.  与朴素的 $O(n)$ 方法比较乘法次数。
3.  修改算法以适用于 $2\times2$ 矩阵。
4.  使用卷积进行乘法，将其扩展到多项式。
5.  实现一个迭代版本并对两者进行基准测试。

#### 测试用例

| 底数 $a$ | 指数 $n$ | 结果          |
| -------- | ------------ | ------------- |
| $3$      | $9$          | $19683$       |
| $2$      | $10$         | $1024$        |
| $5$      | $0$          | $1$           |
| $7$      | $13$         | $96889010407$ |

#### 复杂度

- 时间复杂度：$O(\log n)$ 次乘法
- 空间复杂度：$O(\log n)$（递归）或 $O(1)$（迭代）

平方取幂法是典型的分治求幂技巧，完美融合了简洁性、速度和数学优雅性。
### 597 模幂运算

模幂运算能够高效地计算
$$
a^b \bmod m
$$
而不会使中间结果溢出。
它是密码学、素数测试以及诸如 RSA 和 Diffie–Hellman 等模算术系统的基础。

#### 我们要解决什么问题？

我们希望计算
$$
r=(a^b)\bmod m,
$$
其中 $a$, $b$, 和 $m$ 是整数，且 $b$ 可能非常大。

直接计算 $a^b$ 是不切实际的，因为中间结果会呈指数级增长。
我们改为在每次乘法步骤中应用模约简，使用以下规则：

$$
(xy)\bmod m=((x\bmod m)(y\bmod m))\bmod m.
$$

#### 核心思想

将模算术与平方取幂法结合：

$$
a^b \bmod m =
\begin{cases}
1, & b = 0, \\[6pt]
\left((a^{\,b/2} \bmod m)^2\right) \bmod m, & b \text{ 为偶数}, \\[6pt]
\left(a \times (a^{\,(b-1)/2} \bmod m)^2\right) \bmod m, & b \text{ 为奇数}.
\end{cases}
$$

每一步都减少了 $b$ 的值，并将中间值对 $m$ 取模。

#### 示例

计算 $3^{13}\bmod 17$：

| 步骤 | $b$  | 操作                                           | 结果     |
| ---- | ---- | --------------------------------------------- | -------- |
| 13   | 奇数 | $res=3$                                       | $res=3$  |
| 6    | 偶数 | 平方 $3\to9$                                  | $a=9$    |
| 3    | 奇数 | $res=res*a=3*9=27\Rightarrow27\bmod17=10$     | $res=10$ |
| 1    | 奇数 | $res=res*a=10*13=130\Rightarrow130\bmod17=11$ | 最终结果 |

结果：$3^{13}\bmod17=11$

#### 简洁代码

Python

```python
def modexp(a, b, m):
    res = 1
    a %= m
    while b > 0:
        if b & 1:
            res = (res * a) % m
        a = (a * a) % m
        b >>= 1
    return res

print(modexp(3, 13, 17))  # 11
```

C

```c
#include <stdio.h>

long long modexp(long long a, long long b, long long m) {
    long long res = 1;
    a %= m;
    while (b > 0) {
        if (b & 1)
            res = (res * a) % m;
        a = (a * a) % m;
        b >>= 1;
    }
    return res;
}

int main(void) {
    printf("%lld\n", modexp(3, 13, 17)); // 11
    return 0;
}
```

#### 重要性

- RSA 加密/解密、Diffie–Hellman 密钥交换和 ElGamal 系统的核心。
- 在模逆元、哈希和原根计算中至关重要。
- 使得像 $a^{10^{18}}\bmod m$ 这样的大规模计算能够在微秒级完成。
- 用于费马、米勒-拉宾和卡迈克尔素数测试。

#### 一个温和的证明

对于任意模数 $m$，
$$
(a\times b)\bmod m=((a\bmod m)(b\bmod m))\bmod m.
$$
这个性质允许我们在每次乘法后进行约简。

使用二进制指数法，
$$
a^b=\prod_{i=0}^{k-1} a^{2^i\cdot d_i},
$$
其中 $b=\sum_i d_i2^i$。
我们只对 $d_i=1$ 的项进行乘法运算，并保持所有结果对 $m$ 取模以保持有界。

#### 动手试试

1.  计算 $5^{117}\bmod19$。
2.  与 Python 内置的 `pow(5, 117, 19)` 进行比较。
3.  修改代码以使用模逆元处理 $b<0$ 的情况。
4.  扩展到模 $m$ 的矩阵运算。
5.  证明对于素数 $p$，有 $a^{p-1}\bmod p=1$（费马小定理）。

#### 测试用例

| $a$ | $b$ | $m$  | $a^b\bmod m$ |
| --- | --- | ---- | ------------ |
| 3   | 13  | 17   | 11           |
| 2   | 10  | 1000 | 24           |
| 7   | 256 | 13   | 9            |
| 5   | 117 | 19   | 1            |

#### 复杂度

-   时间复杂度：$O(\log b)$ 次乘法
-   空间复杂度：$O(1)$

模幂运算是计算数论中最优雅、最基础的例程之一，它小巧、精确且足够强大，足以保障互联网的安全。
### 598 快速沃尔什-哈达玛变换 (FWHT)

快速沃尔什-哈达玛变换 (FWHT) 是一种分治算法，用于高效计算基于按位异或 (XOR) 的卷积。它是快速傅里叶变换 (FFT) 的离散模拟，但不同于加法下的乘法，它工作在按位异或运算下。

#### 我们要解决什么问题？

给定两个序列
$$
A=(a_0,a_1,\dots,a_{n-1}), \quad B=(b_0,b_1,\dots,b_{n-1}),
$$
我们想要它们的 XOR 卷积，定义为

$$
C[k]=\sum_{i\oplus j=k}a_i b_j,
$$
其中 $\oplus$ 是按位异或。

朴素方法需要 $O(n^2)$ 的工作量。FWHT 将其减少到 $O(n\log n)$。

#### 核心思想

沃尔什-哈达玛变换 (WHT) 将一个向量映射到其 XOR 域形式。我们可以通过变换两个序列，逐点相乘，然后逆变换来计算 XOR 卷积。

令 $\text{FWT}$ 表示变换。那么

$$
C=\text{FWT}^{-1}(\text{FWT}(A)\circ\text{FWT}(B)),
$$
其中 $\circ$ 是逐元素乘法。

#### 变换定义

对于 $n=2^k$，递归定义快速沃尔什-哈达玛变换 (FWT)。

基本情况：
$$
\text{FWT}([a_0]) = [a_0].
$$

递归步骤：将 $A$ 分割为 $A_1$（前半部分）和 $A_2$（后半部分）。计算 $\text{FWT}(A_1)$ 和 $\text{FWT}(A_2)$ 后，按如下方式组合：
$$
\forall\, i=0,\dots,\tfrac{n}{2}-1:\quad
A'[i] = A_1[i] + A_2[i],\qquad
A'[i+\tfrac{n}{2}] = A_1[i] - A_2[i].
$$

逆变换：
$$
\text{IFWT}(A') = \frac{1}{n}\,\text{FWT}(A').
$$

#### 示例

令 $A=[1,2,3,4]$。分割为 $A_1=[1,2]$ 和 $A_2=[3,4]$，然后组合
$$
[\,1+3,\ 2+4,\ 1-3,\ 2-4\,] = [\,4,6,-2,-2\,].
$$
对于更长的向量，应用相同的分割-递归-组合模式，直到长度为 $1$。

#### 精简代码

Python

```python
def fwht(a, inverse=False):
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(h):
                x = a[i + j]
                y = a[i + j + h]
                a[i + j] = x + y
                a[i + j + h] = x - y
        h *= 2
    if inverse:
        for i in range(n):
            a[i] //= n
    return a

def xor_convolution(a, b):
    n = 1
    while n < max(len(a), len(b)):
        n *= 2
    a = a + [0] * (n - len(a))
    b = b + [0] * (n - len(b))
    A = fwht(a[:])
    B = fwht(b[:])
    C = [A[i] * B[i] for i in range(n)]
    C = fwht(C, inverse=True)
    return C

print(xor_convolution([1,2,3,4],[4,3,2,1]))
```

#### 为什么它很重要

- XOR 卷积出现在子集变换、位掩码动态规划和布尔代数问题中。
- 用于信号处理、纠错码和 GF(2) 上的多项式变换。
- 是在超立方体和按位域上进行快速计算的关键。

#### 一个温和的证明

递归的每一层计算成对的和与差——这是使用哈达玛矩阵 $H_n$ 的线性变换：

$$
H_n=
\begin{bmatrix}
H_{n/2} & H_{n/2}\
H_{n/2} & -H_{n/2}
\end{bmatrix}.
$$

由于 $H_nH_n^T=nI$，其逆为 $\frac{1}{n}H_n^T$，这保证了正确性和可逆性。

#### 动手试试

1.  计算 $[1,1,0,0]$ 的 FWHT。
2.  执行 $[1,2,3,4]$ 和 $[4,3,2,1]$ 的 XOR 卷积。
3.  修改代码以处理浮点数或模运算。
4.  显式实现逆变换并验证恢复结果。
5.  与朴素的 $O(n^2)$ 方法比较运行时间。

#### 测试用例

| 输入 A     | 输入 B     | XOR 卷积结果        |
| ---------- | ---------- | ------------------- |
| [1,2]      | [3,4]      | [10, -2]            |
| [1,1,0,0]  | [0,1,1,0]  | [2,0,0,2]           |
| [1,2,3,4]  | [4,3,2,1]  | [20, 0, 0, 0]       |

#### 复杂度

-   时间：$O(n\log n)$
-   空间：$O(n)$

快速沃尔什-哈达玛变换是 FFT 在 XOR 世界的孪生兄弟，更小巧，更锐利，同样优雅。
### 598 快速沃尔什-哈达玛变换 (FWHT)

快速沃尔什-哈达玛变换 (FWHT) 是一种用于高效计算基于按位异或的卷积的分治算法。它是快速傅里叶变换的离散模拟，但其运算基于按位异或操作，而非加法下的乘法。

#### 我们要解决什么问题？

给定两个序列
$$
A=(a_0,a_1,\dots,a_{n-1}), \quad B=(b_0,b_1,\dots,b_{n-1}),
$$
我们想要它们的异或卷积，定义为

$$
C[k]=\sum_{i\oplus j=k}a_i b_j,
$$
其中 $\oplus$ 是按位异或。

朴素方法需要 $O(n^2)$ 的工作量。FWHT 将其减少到 $O(n\log n)$。

#### 核心思想

沃尔什-哈达玛变换 (WHT) 将一个向量映射到其异或域形式。我们可以通过变换两个序列，逐点相乘，然后逆变换来计算异或卷积。

令 $\text{FWT}$ 表示变换。那么

$$
C=\text{FWT}^{-1}(\text{FWT}(A)\circ\text{FWT}(B)),
$$
其中 $\circ$ 是逐元素乘法。

#### 变换定义

对于 $n = 2^k$，递归定义：

$$
\text{FWT}(A) =
\begin{cases}
[a_0], & n = 1, \\[6pt]
\text{合并 } \text{FWT}(A_1) \text{ 和 } \text{FWT}(A_2)
 & \text{对于 } n > 1.
\end{cases}
$$

合并步骤：
$$
A'[i] = A_1[i] + A_2[i], \qquad
A'[i + n/2] = A_1[i] - A_2[i],
\quad i = 0, \dots, n/2 - 1.
$$

为了求逆，在应用相同过程后，将所有结果除以 $n$。

#### 示例

令 $A = [1, 2, 3, 4]$。

1. 分割成 $[1, 2]$ 和 $[3, 4]$  
   合并：
   $$
   [1 + 3,\ 2 + 4,\ 1 - 3,\ 2 - 4] = [4, 6, -2, -2].
   $$

2. 对更长的向量递归应用。

#### 精简代码

Python

```python
def fwht(a, inverse=False):
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(h):
                x = a[i + j]
                y = a[i + j + h]
                a[i + j] = x + y
                a[i + j + h] = x - y
        h *= 2
    if inverse:
        for i in range(n):
            a[i] //= n
    return a

def xor_convolution(a, b):
    n = 1
    while n < max(len(a), len(b)):
        n *= 2
    a = a + [0] * (n - len(a))
    b = b + [0] * (n - len(b))
    A = fwht(a[:])
    B = fwht(b[:])
    C = [A[i] * B[i] for i in range(n)]
    C = fwht(C, inverse=True)
    return C

print(xor_convolution([1,2,3,4],[4,3,2,1]))
```

#### 为什么它重要

- 异或卷积出现在子集变换、位掩码动态规划和布尔代数问题中。
- 用于信号处理、纠错码和 GF(2) 上的多项式变换。
- 是在超立方体和按位域上进行快速计算的关键。

#### 一个温和的证明

递归的每一层计算成对的和与差——这是使用哈达玛矩阵 $H_n$ 的线性变换：

$$
H_n=
\begin{bmatrix}
H_{n/2} & H_{n/2}\
H_{n/2} & -H_{n/2}
\end{bmatrix}.
$$

由于 $H_nH_n^T=nI$，其逆矩阵为 $\frac{1}{n}H_n^T$，这给出了正确性和可逆性。

#### 动手试试

1.  计算 $[1,1,0,0]$ 的 FWHT。
2.  执行 $[1,2,3,4]$ 和 $[4,3,2,1]$ 的异或卷积。
3.  修改代码以处理浮点数或模运算。
4.  显式实现逆变换并验证恢复。
5.  与朴素的 $O(n^2)$ 方法比较运行时间。

#### 测试用例

| 输入 A     | 输入 B     | 异或卷积结果      |
| ---------- | ---------- | ----------------- |
| [1,2]      | [3,4]      | [10, -2]          |
| [1,1,0,0]  | [0,1,1,0]  | [2,0,0,2]         |
| [1,2,3,4]  | [4,3,2,1]  | [20, 0, 0, 0]     |

#### 复杂度

-   时间：$O(n\log n)$
-   空间：$O(n)$

快速沃尔什-哈达玛变换是 FFT 在异或世界的孪生兄弟，更小巧、更犀利，同样优雅。
### 599 Zeta 变换

Zeta 变换是一种在子集或超集上累积值的组合变换。它在子集动态规划（基于位掩码的动态规划）、容斥原理和快速子集卷积中特别有用。

#### 我们要解决什么问题？

给定一个在全集 $U$（大小为 $n$）的子集 $S$ 上定义的函数 $f(S)$，*Zeta 变换*会产生一个新的函数 $F(S)$，该函数对 $S$ 的所有子集（或超集）求和：

- 子集版本
  $$
  F(S)=\sum_{T\subseteq S}f(T)
  $$

- 超集版本
  $$
  F(S)=\sum_{T\supseteq S}f(T)
  $$

朴素实现下，该变换的运行时间为 $O(n2^n)$，但可以使用位动态规划高效地以 $O(n2^n)$ 的时间复杂度计算。

#### 核心思想

对于大小为 $n$ 的位掩码进行子集 Zeta 变换：

对于每个比特位 $i$，从 $0$ 到 $n-1$：

- 对于每个掩码 $m$，从 $0$ 到 $2^n-1$：

  * 如果掩码 $m$ 中设置了比特位 $i$，则将 $f[m\text{ 去掉比特位 }i]$ 加到 $f[m]$ 上。

用代码表示：
$$
f[m]+=f[m\setminus{i}].
$$

这有效地累积了所有子集的贡献。

#### 示例

设 $f$ 在 3 比特子集上的值为：

| 掩码 | 子集    | $f(S)$ |
| ---- | ------- | ------ |
| 000  | ∅       | 1      |
| 001  | {0}     | 2      |
| 010  | {1}     | 3      |
| 011  | {0,1}   | 4      |
| 100  | {2}     | 5      |
| 101  | {0,2}   | 6      |
| 110  | {1,2}   | 7      |
| 111  | {0,1,2} | 8      |

经过子集 Zeta 变换后，$F(S)=\sum_{T\subseteq S}f(T)$。

对于 $S={0,1}$（掩码 011）：
$$
F(011)=f(000)+f(001)+f(010)+f(011)=1+2+3+4=10.
$$

#### 微型代码

Python

```python
def subset_zeta_transform(f):
    n = len(f).bit_length() - 1
    F = f[:]
    for i in range(n):
        for mask in range(1 << n):
            if mask & (1 << i):
                F[mask] += F[mask ^ (1 << i)]
    return F

# 示例：大小为 3 的子集的 f
f = [1,2,3,4,5,6,7,8]
print(subset_zeta_transform(f))
```

C

```c
#include <stdio.h>

void subset_zeta_transform(int *f, int n) {
    for (int i = 0; i < n; i++) {
        for (int mask = 0; mask < (1 << n); mask++) {
            if (mask & (1 << i))
                f[mask] += f[mask ^ (1 << i)];
        }
    }
}

int main(void) {
    int f[8] = {1,2,3,4,5,6,7,8};
    subset_zeta_transform(f, 3);
    for (int i = 0; i < 8; i++) printf("%d ", f[i]);
    return 0;
}
```

#### 为什么它很重要

- 是子集动态规划、SOS 动态规划和位掩码卷积的核心。
- 用于快速容斥原理和莫比乌斯反演。
- 出现在计数问题、图子集枚举和布尔函数变换中。
- 与莫比乌斯变换协同工作，以反转累积结果。

#### 一个温和的证明

每次对比特位 $i$ 的迭代确保每个不包含比特位 $i$ 的子集都会贡献给包含比特位 $i$ 的超集。通过归纳法，所有子集 $T\subseteq S$ 都累积到 $F(S)$ 中。

#### 亲自尝试

1.  手动计算 $n=3$ 时的 Zeta 变换。
2.  使用莫比乌斯反演（下一节）验证反转。
3.  修改代码以实现超集版本（翻转条件）。
4.  实现模运算（例如，$\bmod 10^9+7$）。
5.  用它来计算具有给定属性的子集数量。

#### 测试用例

| $f(S)$            | 比特位数 | $F(S)$            |
| ----------------- | -------- | ----------------- |
| [1,2,3,4]         | 2        | [1,3,4,10]        |
| [0,1,1,0,1,0,0,0] | 3        | [0,1,1,2,1,2,2,4] |

#### 复杂度

- 时间复杂度：$O(n2^n)$
- 空间复杂度：$O(2^n)$

Zeta 变换是子集动态规划的求和核心，是一种优雅而高效地同时观察集合所有部分的方式。
### 600 莫比乌斯反演

莫比乌斯反演是泽塔变换的数学逆运算。它允许我们从其累积形式 $F(S)$ 中恢复出函数 $f(S)$，其中
$$F(S)=\sum_{T\subseteq S}f(T).$$
它是组合数学、数论和子集动态规划中的基石。

#### 我们要解决什么问题？

假设我们知道 $F(S)$，即所有子集 $T\subseteq S$ 的总贡献。我们想要反转这种累积，得到原始的 $f(S)$。

反演公式为：

$$
f(S)=\sum_{T\subseteq S}(-1)^{|S\setminus T|}F(T).
$$

这是子集求和上微分的组合类比。

#### 核心思想

莫比乌斯反演反转了子集泽塔变换的累积效应。在迭代（按位）形式中，我们可以通过“减去贡献”而非添加它们来高效地执行。

对于每个从 $0$ 到 $n-1$ 的位 $i$：

如果位 $i$ 在 $S$ 中被设置，则从 $f[S]$ 中减去 $f[S\setminus{i}]$：

$$
f[S]-=f[S\setminus{i}].
$$

这撤销了泽塔变换中所做的包含步骤。

#### 示例

让我们从
$$
F(S)=\sum_{T\subseteq S}f(T)
$$
开始，并且我们知道：

| 掩码 | 子集 | $F(S)$ |
| ---- | ------ | ------ |
| 00   | ∅      | 1      |
| 01   | {0}    | 3      |
| 10   | {1}    | 4      |
| 11   | {0,1}  | 10     |

我们应用莫比乌斯反演：

| 步骤  | 子集                                   | 计算   | 结果 |
| ----- | ---------------------------------------- | ------------- | ------ |
| ∅     |,                                        | $f(∅)=F(∅)=1$ | 1      |
| {0}   | $f(01)=F(01)-F(00)=3-1$                  | 2             |        |
| {1}   | $f(10)=F(10)-F(00)=4-1$                  | 3             |        |
| {0,1} | $f(11)=F(11)-F(01)-F(10)+F(00)=10-3-4+1$ | 4             |        |

因此我们恢复了 $f=[1,2,3,4]$。

#### 微型代码

Python

```python
def mobius_inversion(F):
    n = len(F).bit_length() - 1
    f = F[:]
    for i in range(n):
        for mask in range(1 << n):
            if mask & (1 << i):
                f[mask] -= f[mask ^ (1 << i)]
    return f

# 示例
F = [1,3,4,10]
print(mobius_inversion(F))  # [1,2,3,4]
```

C

```c
#include <stdio.h>

void mobius_inversion(int *F, int n) {
    for (int i = 0; i < n; i++) {
        for (int mask = 0; mask < (1 << n); mask++) {
            if (mask & (1 << i))
                F[mask] -= F[mask ^ (1 << i)];
        }
    }
}

int main(void) {
    int F[4] = {1,3,4,10};
    mobius_inversion(F, 2);
    for (int i = 0; i < 4; i++) printf("%d ", F[i]);
    return 0;
}
```

#### 为何重要

- 是容斥原理和子集动态规划的基础。
- 用于数论反演，例如经典的整数莫比乌斯函数 μ(n)。
- 与泽塔变换完美配对，可在累积表示和逐点表示之间切换。
- 出现在快速子集卷积、多项式变换和组合计数中。

#### 一个温和的证明

由
$$
F(S)=\sum_{T\subseteq S}f(T),
$$
我们可以将系统视为一个三角矩阵，当 $T\subseteq S$ 时元素为 $1$。其逆矩阵的元素为 $(-1)^{|S\setminus T|}$（当 $T\subseteq S$ 时），从而得到

$$
f(S)=\sum_{T\subseteq S}(-1)^{|S\setminus T|}F(T).
$$

因此，莫比乌斯反演恰好抵消了泽塔变换中的重复计数。

#### 亲自尝试

1.  从 $f=[1,2,3,4]$ 开始，使用泽塔变换计算其 $F$。
2.  应用莫比乌斯反演以恢复 $f$。
3.  扩展到 3 位子集（$n=3$）。
4.  使用它来计算容斥计数。
5.  修改它以在模 $10^9+7$ 下工作。

#### 测试用例

| $F(S)$                | 期望的 $f(S)$   |
| --------------------- | ----------------- |
| [1,3,4,10]            | [1,2,3,4]         |
| [0,1,1,2]             | [0,1,1,0]         |
| [2,4,6,12,8,16,20,40] | [2,2,2,4,2,4,4,8] |

#### 复杂度

-   时间：$O(n2^n)$
-   空间：$O(2^n)$

莫比乌斯反演完成了组合变换的循环，它是泽塔变换的镜像，将求和变回其源头。
