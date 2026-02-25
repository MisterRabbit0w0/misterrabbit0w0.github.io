---
title: 第三章
date: 2026-02-25
disableNunjucks: true
---

# 第 3 章 数据结构实战

## 第 21 节 数组、链表、栈、队列
### 201 动态数组扩容

动态数组扩容是我们在需要时扩展存储空间，同时保持 `push` 操作快速的方法。其思想很简单：当数组已满时，分配一个更大的数组，复制元素，然后继续操作。如果每次扩容时将容量加倍，那么每次 `push` 的平均成本将保持恒定。

#### 我们要解决什么问题？

固定数组有固定的大小。实际程序并不总是能提前知道 n 的大小。我们想要一个数组，它支持在末尾进行接近常数时间的 `push` 操作，能够自动增长，并且保持元素在连续内存中以利于缓存。

目标：在一个可调整大小的数组上提供 `push`、`pop`、`get`、`set` 操作，并使 `push` 操作的摊还时间复杂度为 O(1)。

#### 它是如何工作的？（通俗解释）

维护两个数字：`size`（当前元素数量）和 `capacity`（当前容量）。

1.  从一个较小的容量开始（例如，1 或 8）。
2.  执行 `push` 时，如果 `size < capacity`，则写入元素并增加 `size`。
3.  如果 `size == capacity`，则分配一个容量加倍的新存储空间，复制旧元素，释放旧内存块，然后执行 `push`。
4.  （可选）如果多次 `pop` 操作导致 `size` 远低于 `capacity`，则通过将容量减半来收缩数组，以避免空间浪费。

为什么是加倍？
加倍操作使得昂贵的扩容次数很少。大多数 `push` 操作只是廉价的写入。只有偶尔我们才需要为复制操作付出代价。

示例步骤（增长模拟）

| 步骤 | 扩容前 Size | 扩容前 Capacity | 操作                                         | 扩容后 Size | 扩容后 Capacity |
| ---- | ----------- | --------------- | -------------------------------------------- | ----------- | --------------- |
| 1    | 0           | 1               | push(1)                                      | 1           | 1               |
| 2    | 1           | 1               | 已满 → 扩容至 2，复制 1 个元素，push(2)      | 2           | 2               |
| 3    | 2           | 2               | 已满 → 扩容至 4，复制 2 个元素，push(3)      | 3           | 4               |
| 4    | 3           | 4               | push(4)                                      | 4           | 4               |
| 5    | 4           | 4               | 已满 → 扩容至 8，复制 4 个元素，push(5)      | 5           | 8               |

注意容量只是偶尔加倍，而大多数 `push` 操作的成本是 O(1)。

#### 微型代码（简易版本）

C

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t size;
    size_t capacity;
    int *data;
} DynArray;

int da_init(DynArray *a, size_t init_cap) {
    if (init_cap == 0) init_cap = 1;
    a->data = (int *)malloc(init_cap * sizeof(int));
    if (!a->data) return 0;
    a->size = 0;
    a->capacity = init_cap;
    return 1;
}

int da_resize(DynArray *a, size_t new_cap) {
    int *p = (int *)realloc(a->data, new_cap * sizeof(int));
    if (!p) return 0;
    a->data = p;
    a->capacity = new_cap;
    return 1;
}

int da_push(DynArray *a, int x) {
    if (a->size == a->capacity) {
        size_t new_cap = a->capacity * 2;
        if (!da_resize(a, new_cap)) return 0;
    }
    a->data[a->size++] = x;
    return 1;
}

int da_pop(DynArray *a, int *out) {
    if (a->size == 0) return 0;
    *out = a->data[--a->size];
    if (a->capacity > 1 && a->size <= a->capacity / 4) {
        size_t new_cap = a->capacity / 2;
        if (new_cap == 0) new_cap = 1;
        da_resize(a, new_cap);
    }
    return 1;
}

int main(void) {
    DynArray a;
    if (!da_init(&a, 1)) return 1;
    for (int i = 0; i < 10; ++i) da_push(&a, i);
    for (size_t i = 0; i < a.size; ++i) printf("%d ", a.data[i]);
    printf("\nsize=%zu cap=%zu\n", a.size, a.capacity);
    free(a.data);
    return 0;
}
```

Python

```python
class DynArray:
    def __init__(self, init_cap=1):
        self._cap = max(1, init_cap)
        self._n = 0
        self._data = [None] * self._cap

    def _resize(self, new_cap):
        new = [None] * new_cap
        for i in range(self._n):
            new[i] = self._data[i]
        self._data = new
        self._cap = new_cap

    def push(self, x):
        if self._n == self._cap:
            self._resize(self._cap * 2)
        self._data[self._n] = x
        self._n += 1

    def pop(self):
        if self._n == 0:
            raise IndexError("pop from empty DynArray")
        self._n -= 1
        x = self._data[self._n]
        self._data[self._n] = None
        if self._cap > 1 and self._n <= self._cap // 4:
            self._resize(max(1, self._cap // 2))
        return x

    def __getitem__(self, i):
        if not 0 <= i < self._n:
            raise IndexError("index out of range")
        return self._data[i]
```

#### 为什么它很重要

-   为动态增长提供摊还 O(1) 的 `push` 操作
-   连续内存布局有利于缓存性能
-   简单的 API，是向量（vector）、数组列表（array list）和许多脚本语言数组的基础
-   通过可选的收缩操作平衡内存使用

#### 一个温和的证明（为什么它有效）

加倍策略的记账视角
为每次 `push` 操作收取一个固定的信用（例如 3 个单位）。

1.  一次普通的 `push` 消耗 1 个单位成本，并为该元素存储 2 个单位的信用。
2.  当从容量 k 扩容到 2k 时，复制 k 个元素消耗 k 个单位成本，由之前存储的信用支付。
3.  新元素支付自己的 1 个单位成本。

因此，每次 `push` 的摊还成本是 O(1)。

增长因子的选择
任何大于 1 的因子都能提供摊还 O(1) 的复杂度。

-   加倍 → 扩容次数更少，但内存闲置更多
-   1.5 倍 → 闲置内存更少，但复制更频繁
-   太小（<1.2）→ 会破坏摊还界限

#### 动手试试

1.  模拟因子为 2.0 和 1.5 时，从 1 到 32 的 `push` 操作。统计扩容次数。
2.  添加 `reserve(n)` 函数来预分配容量。
3.  实现当 `size <= capacity / 4` 时的收缩操作。
4.  将 `int` 替换为一个结构体，测量复制成本。
5.  使用势能法：Φ = 2·size − capacity。

#### 测试用例

| 操作序列          | 容量变化轨迹（因子 2） | 备注                     |
| ----------------- | ---------------------- | ------------------------ |
| push 1..8         | 1 → 2 → 4 → 8          | 在 1, 2, 4 时扩容        |
| push 9            | 8 → 16                 | 在写入前扩容             |
| push 10..16       | 16                     | 无扩容                   |
| pop 16..9         | 16                     | 收缩是可选的             |
| pop 8             | 16 → 8                 | 在负载为 25% 时收缩      |

边界情况

-   在满数组上执行 `push` 会在写入前触发一次扩容
-   在空数组上执行 `pop` 应返回错误或 false
-   `reserve(n)` 如果大于当前容量，必须保留原有数据
-   收缩操作永远不会使容量低于当前大小

#### 复杂度

-   时间复杂度：
    *   `push` 摊还 O(1)
    *   `push` 最坏情况 O(n)（在扩容时）
    *   `pop` 摊还 O(1)
    *   `get`/`set` O(1)
-   空间复杂度：O(n)，容量 ≤ 常数 × 大小

动态数组扩容将僵硬的数组变成了灵活的容器。在需要时增长，偶尔复制，就能享受到平均意义上的常数时间 `push` 操作。
### 202 循环数组实现

循环数组（或环形缓冲区）将元素存储在固定大小的数组中，同时允许环绕索引。它非常适合实现队列和缓冲区，其中旧数据以先进先出的方式被覆盖或处理。

#### 我们要解决什么问题？

如果我们只让队首和队尾指针向前移动，常规数组会浪费空间。循环数组通过在指针到达数组末尾时环绕索引来解决这个问题，这样所有槽位都可以被重用，无需移动元素。

目标：
使用具有环绕索引的固定大小缓冲区，高效支持 O(1) 时间复杂度的入队、出队、查看队首元素和获取大小操作。

#### 它是如何工作的（通俗解释）？

维护以下变量：

- `front`：第一个元素的索引
- `rear`：下一个空闲槽位的索引
- `count`：缓冲区中的元素数量
- `capacity`：最大元素数量

环绕索引使用模运算：

```text
next_index = (current_index + 1) % capacity
```

当添加或移除元素时，总是按照这个规则递增 `front` 或 `rear`。

示例步骤（环绕模拟）

| 步骤 | 操作         | Front | Rear | Count | 数组状态         | 说明           |
| ---- | ------------ | ----- | ---- | ----- | ---------------- | -------------- |
| 1    | enqueue(10)  | 0     | 1    | 1     | [10, _, _, _]    | rear 前进      |
| 2    | enqueue(20)  | 0     | 2    | 2     | [10, 20, _, _]   |                |
| 3    | enqueue(30)  | 0     | 3    | 3     | [10, 20, 30, _]  |                |
| 4    | dequeue()    | 1     | 3    | 2     | [_, 20, 30, _]   | front 前进     |
| 5    | enqueue(40)  | 1     | 0    | 3     | [40, 20, 30, _]  | 环绕           |
| 6    | enqueue(50)  | 1     | 1    | 4     | [40, 20, 30, 50] | 队列满         |

#### 精简代码（简易版本）

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    int front;
    int rear;
    int count;
    int capacity;
} CircularQueue;

int cq_init(CircularQueue *q, int capacity) {
    q->data = malloc(capacity * sizeof(int));
    if (!q->data) return 0;
    q->capacity = capacity;
    q->front = 0;
    q->rear = 0;
    q->count = 0;
    return 1;
}

int cq_enqueue(CircularQueue *q, int x) {
    if (q->count == q->capacity) return 0; // 队列满
    q->data[q->rear] = x;
    q->rear = (q->rear + 1) % q->capacity;
    q->count++;
    return 1;
}

int cq_dequeue(CircularQueue *q, int *out) {
    if (q->count == 0) return 0; // 队列空
    *out = q->data[q->front];
    q->front = (q->front + 1) % q->capacity;
    q->count--;
    return 1;
}

int main(void) {
    CircularQueue q;
    cq_init(&q, 4);
    cq_enqueue(&q, 10);
    cq_enqueue(&q, 20);
    cq_enqueue(&q, 30);
    int val;
    cq_dequeue(&q, &val);
    cq_enqueue(&q, 40);
    cq_enqueue(&q, 50); // 如果容量=4，应该失败
    printf("Front value: %d\n", q.data[q.front]);
    free(q.data);
    return 0;
}
```

Python

```python
class CircularQueue:
    def __init__(self, capacity):
        self._cap = capacity
        self._data = [None] * capacity
        self._front = 0
        self._rear = 0
        self._count = 0

    def enqueue(self, x):
        if self._count == self._cap:
            raise OverflowError("queue full")
        self._data[self._rear] = x
        self._rear = (self._rear + 1) % self._cap
        self._count += 1

    def dequeue(self):
        if self._count == 0:
            raise IndexError("queue empty")
        x = self._data[self._front]
        self._front = (self._front + 1) % self._cap
        self._count -= 1
        return x

    def peek(self):
        if self._count == 0:
            raise IndexError("queue empty")
        return self._data[self._front]
```

#### 为什么重要

- 实现常数时间的队列操作
- 无需移动元素
- 高效利用固定内存
- 是循环缓冲区、任务调度器、流处理管道以及音视频缓冲区的核心

#### 一个温和的证明（为什么可行）

模运算索引确保了当位置到达数组末尾时循环回来。
如果容量是 n，那么所有索引都保持在 `[0, n-1]` 范围内。
不会发生溢出，因为 `front` 和 `rear` 总是会环绕。
条件 `count == capacity` 检测队列满，`count == 0` 检测队列空。

因此，每个操作只涉及一个元素并更新 O(1) 个变量。

#### 动手实践

1.  实现一个带覆盖功能的循环缓冲区（旧数据被替换）。
2.  添加 `is_full()` 和 `is_empty()` 辅助函数。
3.  模拟生产者-消费者模型，每个生产者节拍入队一次，每个消费者节拍出队一次。
4.  扩展以存储结构体而不是整数。

#### 测试用例

| 操作序列         | Front | Rear | Count | 数组状态         | 说明       |
| ---------------- | ----- | ---- | ----- | ---------------- | ---------- |
| enqueue(10)      | 0     | 1    | 1     | [10, _, _, _]    | 正常入队   |
| enqueue(20)      | 0     | 2    | 2     | [10, 20, _, _]   |            |
| enqueue(30)      | 0     | 3    | 3     | [10, 20, 30, _]  |            |
| dequeue()        | 1     | 3    | 2     | [_, 20, 30, _]   |            |
| enqueue(40)      | 1     | 0    | 3     | [40, 20, 30, _]  | 环绕       |
| enqueue(50)      | 1     | 1    | 4     | [40, 20, 30, 50] | 队列满     |

边界情况

- 队列满时入队 → 错误或覆盖
- 队列空时出队 → 错误
- 模运算索引避免溢出
- 适用于任何容量 ≥ 1 的情况

#### 复杂度

- 时间复杂度：入队、出队、查看队首元素为 O(1)
- 空间复杂度：O(n) 固定缓冲区

循环数组通过环绕索引提供了优雅的常数时间队列，这个小技巧支撑着大型系统。
### 203 单链表插入/删除

单链表是由一系列节点组成的链，每个节点指向下一个节点。它能够动态地增长和收缩，无需预先分配内存。插入和删除等操作依赖于指针的调整，而不是元素的移动。

#### 我们要解决什么问题？

静态数组是固定大小的，在中间进行插入或删除操作时需要移动元素。单链表通过指针连接元素来解决这个问题，允许在给定节点引用的情况下实现高效的 O(1) 插入和删除。

目标：
支持在一种可以动态增长而无需重新分配内存或移动元素的结构上进行插入、删除和遍历操作。

#### 它是如何工作的（通俗解释）？

每个节点存储两个字段：

- `data`（值）
- `next`（指向下一个节点的指针）

链表有一个指向第一个节点的 `head` 指针。
插入和删除操作通过更新 `next` 指针来实现。

示例步骤（在指定位置插入）

| 步骤 | 操作                 | 更新的节点   | 之前           | 之后             | 备注                   |
| ---- | -------------------- | ------------ | -------------- | ---------------- | ---------------------- |
| 1    | 创建链表             | -            | 空             | head = NULL      | 链表初始为空           |
| 2    | 在头部插入(10)       | 新节点       | NULL           | [10]             | head → 10              |
| 3    | 在头部插入(20)       | 新节点       | [10]           | [20 → 10]        | 更新 head              |
| 4    | 在 20 之后插入(30)   | 新节点       | [20 → 10]      | [20 → 30 → 10]   | 重定向指针             |
| 5    | 删除(30)             | 节点 20      | [20 → 30 → 10] | [20 → 10]        | 绕过被移除的节点       |

核心思想：
要插入，将新节点的 `next` 指向目标节点的下一个节点，然后将目标节点的 `next` 指向新节点。
要删除，通过将前一个节点的 `next` 指向目标节点之后的节点来绕过目标节点。

#### 微型代码（简易版本）

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node *next;
} Node;

Node* insert_head(Node* head, int value) {
    Node* new_node = malloc(sizeof(Node));
    new_node->data = value;
    new_node->next = head;
    return new_node; // 新的头节点
}

Node* insert_after(Node* node, int value) {
    if (!node) return NULL;
    Node* new_node = malloc(sizeof(Node));
    new_node->data = value;
    new_node->next = node->next;
    node->next = new_node;
    return new_node;
}

Node* delete_value(Node* head, int value) {
    if (!head) return NULL;
    if (head->data == value) {
        Node* tmp = head->next;
        free(head);
        return tmp;
    }
    Node* prev = head;
    Node* cur = head->next;
    while (cur) {
        if (cur->data == value) {
            prev->next = cur->next;
            free(cur);
            break;
        }
        prev = cur;
        cur = cur->next;
    }
    return head;
}

void print_list(Node* head) {
    for (Node* p = head; p; p = p->next)
        printf("%d -> ", p->data);
    printf("NULL\n");
}

int main(void) {
    Node* head = NULL;
    head = insert_head(head, 10);
    head = insert_head(head, 20);
    insert_after(head, 30);
    print_list(head);
    head = delete_value(head, 30);
    print_list(head);
    return 0;
}
```

Python

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def insert_head(self, data):
        node = Node(data)
        node.next = self.head
        self.head = node

    def insert_after(self, prev, data):
        if prev is None:
            return
        node = Node(data)
        node.next = prev.next
        prev.next = node

    def delete_value(self, value):
        cur = self.head
        prev = None
        while cur:
            if cur.data == value:
                if prev:
                    prev.next = cur.next
                else:
                    self.head = cur.next
                return
            prev = cur
            cur = cur.next

    def print_list(self):
        cur = self.head
        while cur:
            print(cur.data, end=" -> ")
            cur = cur.next
        print("NULL")
```

#### 为什么它重要

- 提供动态增长，无需预先分配
- 支持在头部或给定位置进行 O(1) 的插入和删除
- 可用于栈、队列、哈希表桶和邻接表
- 是更复杂数据结构（树、图）的基础

#### 一个温和的证明（为什么它有效）

每个操作只改变固定数量的指针。

- 插入：2 次赋值 → O(1)
- 删除：查找节点 O(n)，然后 1 次重新赋值
  因为指针是局部更新的，链表的其余部分保持有效。

#### 亲自尝试

1.  实现 `insert_tail` 以实现 O(n) 的尾部插入。
2.  编写 `reverse()` 以原地翻转链表。
3.  添加 `size()` 以统计节点数。
4.  尝试从头部和中间删除。

#### 测试用例

| 操作                   | 输入           | 结果             | 备注               |
| ---------------------- | -------------- | ---------------- | ------------------ |
| insert_head(10)        | []             | [10]             | 创建头节点         |
| insert_head(20)        | [10]           | [20 → 10]        | 新的头节点         |
| insert_after(head, 30) | [20 → 10]      | [20 → 30 → 10]   | 指针重定向         |
| delete_value(30)       | [20 → 30 → 10] | [20 → 10]        | 节点被移除         |
| delete_value(40)       | [20 → 10]      | [20 → 10]        | 无变化             |

边界情况

- 从空链表删除 → 无效果
- 在 NULL 之后插入 → 无效果
- 删除头节点 → 更新头指针

#### 复杂度

- 时间：插入 O(1)，删除 O(n)（如果按值搜索），遍历 O(n)
- 空间：对于 n 个节点为 O(n)

单链表教授指针操作，其结构一次增长一个节点，每个链接都很重要。
### 204 双向链表的插入与删除

双向链表通过增加后向链接扩展了单向链表。每个节点既指向其前驱节点，也指向其后继节点，这使得在两个方向上的插入和删除操作都更加容易。

#### 我们要解决什么问题？

单向链表无法向后遍历，并且删除一个节点需要访问其前驱节点。双向链表通过在每个节点存储两个指针来解决这个问题，当拥有节点引用时，允许在任意位置进行常数时间的插入和删除操作。

目标：
支持双向遍历和高效的局部插入/删除操作，指针更新的时间复杂度为 O(1)。

#### 它是如何工作的（通俗解释）？

每个节点存储：

- `data`：节点的值
- `prev`：指向前一个节点的指针
- `next`：指向后一个节点的指针

链表跟踪两端：

- `head` 指向第一个节点
- `tail` 指向最后一个节点

操作步骤示例（插入和删除）

| 步骤 | 操作             | 目标位置 | 操作前          | 操作后              | 说明               |
| ---- | ---------------- | -------- | --------------- | ------------------- | ------------------ |
| 1    | 创建链表         | -        | 空              | [10]                | head=tail=10       |
| 2    | insert_front(20) | 头部     | [10]            | [20 ⇄ 10]           | 更新 head          |
| 3    | insert_back(30)  | 尾部     | [20 ⇄ 10]       | [20 ⇄ 10 ⇄ 30]      | 更新 tail          |
| 4    | delete(10)       | 中间     | [20 ⇄ 10 ⇄ 30]  | [20 ⇄ 30]           | 指针绕过节点 10    |

每个操作只涉及附近的节点，无需移动元素。

#### 精简代码（简易版本）

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node* prev;
    struct Node* next;
} Node;

Node* insert_front(Node* head, int value) {
    Node* node = malloc(sizeof(Node));
    node->data = value;
    node->prev = NULL;
    node->next = head;
    if (head) head->prev = node;
    return node;
}

Node* insert_back(Node* head, int value) {
    Node* node = malloc(sizeof(Node));
    node->data = value;
    node->next = NULL;
    if (!head) {
        node->prev = NULL;
        return node;
    }
    Node* tail = head;
    while (tail->next) tail = tail->next;
    tail->next = node;
    node->prev = tail;
    return head;
}

Node* delete_value(Node* head, int value) {
    Node* cur = head;
    while (cur && cur->data != value) cur = cur->next;
    if (!cur) return head; // 未找到
    if (cur->prev) cur->prev->next = cur->next;
    else head = cur->next; // 删除头节点
    if (cur->next) cur->next->prev = cur->prev;
    free(cur);
    return head;
}

void print_forward(Node* head) {
    for (Node* p = head; p; p = p->next) printf("%d ⇄ ", p->data);
    printf("NULL\n");
}
```

Python

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None

    def insert_front(self, data):
        node = Node(data)
        node.next = self.head
        if self.head:
            self.head.prev = node
        self.head = node

    def insert_back(self, data):
        node = Node(data)
        if not self.head:
            self.head = node
            return
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = node
        node.prev = cur

    def delete_value(self, value):
        cur = self.head
        while cur:
            if cur.data == value:
                if cur.prev:
                    cur.prev.next = cur.next
                else:
                    self.head = cur.next
                if cur.next:
                    cur.next.prev = cur.prev
                return
            cur = cur.next

    def print_forward(self):
        cur = self.head
        while cur:
            print(cur.data, end=" ⇄ ")
            cur = cur.next
        print("NULL")
```

#### 为什么它很重要

- 双向遍历（向前/向后）
- 当已知节点时，插入和删除操作时间复杂度为 O(1)
- 是双端队列（deque）、LRU 缓存和文本编辑器缓冲区的基础
- 支持简洁的链表反转和拼接操作

#### 一个温和的证明（为什么它有效）

每个节点的更新最多修改四个指针：

- 插入：新节点的 `next`、`prev` 以及邻居节点的链接
- 删除：前驱节点的 `next`、后继节点的 `prev`
  由于每个步骤都是常数工作量，因此在给定节点引用的情况下，操作是 O(1) 的。

遍历仍然是 O(n) 的代价，但局部编辑是高效的。

#### 动手试试

1. 通过交换 `prev` 和 `next` 指针来实现 reverse()。
2. 添加 tail 指针以支持 O(1) 的 insert_back 操作。
3. 支持双向迭代。
4. 实现 `pop_front()` 和 `pop_back()`。

#### 测试用例

| 操作             | 输入             | 输出               | 说明           |
| ---------------- | ---------------- | ------------------ | -------------- |
| insert_front(10) | []               | [10]               | head=tail=10   |
| insert_front(20) | [10]             | [20 ⇄ 10]          | 新的 head      |
| insert_back(30)  | [20 ⇄ 10]        | [20 ⇄ 10 ⇄ 30]     | 新的 tail      |
| delete_value(10) | [20 ⇄ 10 ⇄ 30]   | [20 ⇄ 30]          | 删除中间节点   |
| delete_value(20) | [20 ⇄ 30]        | [30]               | 删除头节点     |

边界情况

- 从空链表删除 → 无效果
- 在空链表上插入会同时设置 head 和 tail
- 每次更新后正确维护两个方向

#### 复杂度分析

- 时间复杂度：
  * 插入/删除（给定节点）：O(1)
  * 搜索：O(n)
  * 遍历：O(n)
- 空间复杂度：O(n)（每个节点 2 个指针）

双向链表带来了对称性，可以向前移动、向后移动，并在常数时间内进行编辑。
### 205 栈的入栈与出栈

栈是一种简单但强大的数据结构，遵循 LIFO（后进先出）规则。最近添加的元素会最先被移除，就像叠盘子一样，你只能从顶部取走。

#### 我们要解决什么问题？

我们经常需要反转顺序或跟踪嵌套操作，例如函数调用、括号匹配、撤销/重做以及表达式求值。栈通过入栈（添加）和出栈（移除）为我们提供了一种清晰的方式来管理这种行为。

目标：
支持在 O(1) 时间内进行入栈、出栈、查看栈顶和判断是否为空的操作，并保持 LIFO 顺序。

#### 它是如何工作的（通俗解释）？

想象一个垂直的堆叠。

- 入栈(x)：将 x 放在顶部。
- 出栈()：移除顶部元素。
- 查看栈顶()：查看顶部元素但不移除。

实现可以使用数组（固定或动态）或链表。

示例步骤（数组栈模拟）

| 步骤 | 操作     | 栈（顶部 → 底部） | 备注           |
| ---- | -------- | ----------------- | -------------- |
| 1    | 入栈(10) | [10]              | 第一个元素     |
| 2    | 入栈(20) | [20, 10]          | 顶部是 20      |
| 3    | 入栈(30) | [30, 20, 10]      |                |
| 4    | 出栈()   | [20, 10]          | 30 被移除      |
| 5    | 查看栈顶() | 顶部 = 20         | 顶部保持不变   |

#### 微型代码（简易版本）

C（基于数组的栈）

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    int top;
    int capacity;
} Stack;

int stack_init(Stack *s, int cap) {
    s->data = malloc(cap * sizeof(int));
    if (!s->data) return 0;
    s->capacity = cap;
    s->top = -1;
    return 1;
}

int stack_push(Stack *s, int x) {
    if (s->top + 1 == s->capacity) return 0; // 栈满
    s->data[++s->top] = x;
    return 1;
}

int stack_pop(Stack *s, int *out) {
    if (s->top == -1) return 0; // 栈空
    *out = s->data[s->top--];
    return 1;
}

int stack_peek(Stack *s, int *out) {
    if (s->top == -1) return 0;
    *out = s->data[s->top];
    return 1;
}

int main(void) {
    Stack s;
    stack_init(&s, 5);
    stack_push(&s, 10);
    stack_push(&s, 20);
    int val;
    stack_pop(&s, &val);
    printf("出栈: %d\n", val);
    stack_peek(&s, &val);
    printf("顶部: %d\n", val);
    free(s.data);
    return 0;
}
```

Python（使用列表作为栈）

```python
class Stack:
    def __init__(self):
        self._data = []

    def push(self, x):
        self._data.append(x)

    def pop(self):
        if not self._data:
            raise IndexError("从空栈弹出")
        return self._data.pop()

    def peek(self):
        if not self._data:
            raise IndexError("查看空栈顶部")
        return self._data[-1]

    def is_empty(self):
        return len(self._data) == 0

# 示例
s = Stack()
s.push(10)
s.push(20)
print(s.pop())  # 20
print(s.peek()) # 10
```

#### 为什么它很重要

- 递归、解析和回溯的核心
- 支撑编程语言中的函数调用栈
- 是撤销、反转和括号匹配的自然结构
- 简单性，在算法中应用广泛

#### 一个温和的证明（为什么它有效）

每个操作只涉及顶部元素或索引。

- `push`：递增 top，写入值
- `pop`：读取 top，递减 top
  所有操作都是 O(1) 时间。
  LIFO 属性确保了函数调用和嵌套作用域的正确逆序。

#### 亲自尝试

1.  使用链表节点实现栈。
2.  栈满时动态扩展容量。
3.  使用栈检查字符串中的括号是否匹配。
4.  使用栈操作反转列表。

#### 测试用例

| 操作     | 输入     | 输出                   | 备注             |
| -------- | -------- | ---------------------- | ---------------- |
| 入栈(10) | []       | [10]                   | 顶部 = 10        |
| 入栈(20) | [10]     | [20, 10]               | 顶部 = 20        |
| 出栈()   | [20, 10] | 返回 20, 栈变为 [10]   |                  |
| 查看栈顶() | [10]     | 返回 10                |                  |
| 出栈()   | [10]     | 返回 10, 栈变为 []     | 出栈后栈空       |

边界情况

- 从空栈出栈 → 错误或返回 false
- 查看空栈顶部 → 错误
- 如果数组已满则溢出（除非是动态的）

#### 复杂度

- 时间：入栈 O(1)，出栈 O(1)，查看栈顶 O(1)
- 空间：对于 n 个元素为 O(n)

栈体现了有序的内存管理，后进先出，是控制流和历史记录的一个极简模型。
### 206 队列的入队/出队

队列是栈的镜像双胞胎。它遵循 FIFO（先进先出）原则，最先添加的元素最先被移除，就像排队等候的人群。

#### 我们要解决什么问题？

我们需要一种结构，其中的项目按到达顺序进行处理：调度任务、缓冲数据、广度优先搜索或管理打印作业。队列允许我们在后端入队，从前端出队，简单而公平。

目标：
使用循环布局或链表，在 O(1) 时间内支持入队、出队、查看队首和判空操作。

#### 它是如何工作的（通俗解释）？

队列有两端：

- 前端 → 项目被移除的地方
- 后端 → 项目被添加的地方

操作：

- `enqueue(x)` 在 `rear` 处添加
- `dequeue()` 从 `front` 处移除
- `peek()` 查看 `front` 处的项目

如果使用循环数组实现，则使用模运算来循环索引。

示例步骤（FIFO 模拟）

| 步骤 | 操作        | 队列（前端 → 后端） | 前端 | 后端 | 备注           |
| ---- | ----------- | -------------------- | ----- | ---- | -------------- |
| 1    | enqueue(10) | [10]                 | 0     | 1    | 第一个元素     |
| 2    | enqueue(20) | [10, 20]             | 0     | 2    | 后端前进       |
| 3    | enqueue(30) | [10, 20, 30]         | 0     | 3    |                |
| 4    | dequeue()   | [20, 30]             | 1     | 3    | 10 被移除      |
| 5    | enqueue(40) | [20, 30, 40]         | 1     | 0    | 循环回绕       |

#### 微型代码（简易版本）

C 语言（循环数组队列）

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    int front;
    int rear;
    int count;
    int capacity;
} Queue;

int queue_init(Queue *q, int cap) {
    q->data = malloc(cap * sizeof(int));
    if (!q->data) return 0;
    q->capacity = cap;
    q->front = 0;
    q->rear = 0;
    q->count = 0;
    return 1;
}

int enqueue(Queue *q, int x) {
    if (q->count == q->capacity) return 0; // 队列已满
    q->data[q->rear] = x;
    q->rear = (q->rear + 1) % q->capacity;
    q->count++;
    return 1;
}

int dequeue(Queue *q, int *out) {
    if (q->count == 0) return 0; // 队列为空
    *out = q->data[q->front];
    q->front = (q->front + 1) % q->capacity;
    q->count--;
    return 1;
}

int queue_peek(Queue *q, int *out) {
    if (q->count == 0) return 0;
    *out = q->data[q->front];
    return 1;
}

int main(void) {
    Queue q;
    queue_init(&q, 4);
    enqueue(&q, 10);
    enqueue(&q, 20);
    enqueue(&q, 30);
    int val;
    dequeue(&q, &val);
    printf("出队元素: %d\n", val);
    enqueue(&q, 40);
    enqueue(&q, 50); // 如果容量为 4，此操作将失败
    queue_peek(&q, &val);
    printf("队首元素: %d\n", val);
    free(q.data);
    return 0;
}
```

Python 语言（使用 deque 将列表作为队列）

```python
from collections import deque

class Queue:
    def __init__(self):
        self._data = deque()

    def enqueue(self, x):
        self._data.append(x)

    def dequeue(self):
        if not self._data:
            raise IndexError("从空队列出队")
        return self._data.popleft()

    def peek(self):
        if not self._data:
            raise IndexError("查看空队列的队首")
        return self._data[0]

    def is_empty(self):
        return len(self._data) == 0

# 示例
q = Queue()
q.enqueue(10)
q.enqueue(20)
print(q.dequeue())  # 10
print(q.peek())     # 20
```

#### 为什么它很重要

- 强制公平性（先到先服务）
- 是 BFS、调度器、缓冲区、管道的基础
- 易于实现和理解
- 是栈的自然对应物

#### 一个温和的证明（为什么它有效）

每个操作只更新前端或后端索引，而不是整个数组。
循环索引确保常数时间的回绕：

```text
rear = (rear + 1) % capacity
front = (front + 1) % capacity
```

所有操作都涉及 O(1) 的数据和字段，因此运行时间保持为 O(1)。

#### 亲自尝试

1.  实现一个基于链表的队列。
2.  添加 `is_full()` 和 `is_empty()` 检查。
3.  在一个简单图上编写基于队列的 BFS。
4.  比较线性队列与循环队列的行为。

#### 测试用例

| 操作        | 队列（前端 → 后端） | 前端 | 后端 | 计数 | 备注         |
| ----------- | -------------------- | ----- | ---- | ---- | ------------ |
| enqueue(10) | [10]                 | 0     | 1    | 1    |              |
| enqueue(20) | [10, 20]             | 0     | 2    | 2    |              |
| enqueue(30) | [10, 20, 30]         | 0     | 3    | 3    |              |
| dequeue()   | [20, 30]             | 1     | 3    | 2    | 移除 10      |
| enqueue(40) | [20, 30, 40]         | 1     | 0    | 3    | 循环回绕     |
| peek()      | front=20             | 1     | 0    | 3    | 检查队首元素 |

边界情况

- 从空队列出队 → 错误
- 向满队列入队 → 溢出
- 与循环回绕无缝协作

#### 复杂度

- 时间：入队 O(1)，出队 O(1)，查看队首 O(1)
- 空间：固定缓冲区或动态增长时为 O(n)

队列为数据带来了公平性，先进入的先出来，稳定且可预测。
### 207 双端队列实现

双端队列是一种灵活的容器，允许从两端添加和移除元素，是栈和队列行为的结合。

#### 我们要解决什么问题？

栈限制你只能在一端操作，队列则限制为两个固定的角色。有时我们需要两者兼备：在头部或尾部插入，从任意一侧弹出。双端队列为滑动窗口算法、回文检查、撤销-重做系统和任务调度器提供动力。

目标：
支持 `push_front`、`push_back`、`pop_front`、`pop_back` 以及 `peek_front/back`，时间复杂度为 O(1)。

#### 它是如何工作的（通俗解释）？

双端队列可以使用以下方式构建：

- 循环数组（使用环绕索引）
- 双向链表（双向指针）

操作：

- `push_front(x)` → 在头部之前插入
- `push_back(x)` → 在尾部之后插入
- `pop_front()` → 移除头部元素
- `pop_back()` → 移除尾部元素

示例步骤（循环数组模拟）

| 步骤 | 操作          | 头部 | 尾部 | 双端队列（头部 → 尾部） | 备注               |
| ---- | ------------- | ---- | ---- | ----------------------- | ------------------ |
| 1    | push_back(10) | 0    | 1    | [10]                    | 第一个元素         |
| 2    | push_back(20) | 0    | 2    | [10, 20]                | 尾部增长           |
| 3    | push_front(5) | 3    | 2    | [5, 10, 20]             | 头部环绕           |
| 4    | pop_back()    | 3    | 1    | [5, 10]                 | 移除 20            |
| 5    | pop_front()   | 0    | 1    | [10]                    | 移除 5             |

#### 微型代码（简易版本）

C 语言（循环数组双端队列）

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    int front;
    int rear;
    int count;
    int capacity;
} Deque;

int dq_init(Deque *d, int cap) {
    d->data = malloc(cap * sizeof(int));
    if (!d->data) return 0;
    d->capacity = cap;
    d->front = 0;
    d->rear = 0;
    d->count = 0;
    return 1;
}

int dq_push_front(Deque *d, int x) {
    if (d->count == d->capacity) return 0; // 队列已满
    d->front = (d->front - 1 + d->capacity) % d->capacity;
    d->data[d->front] = x;
    d->count++;
    return 1;
}

int dq_push_back(Deque *d, int x) {
    if (d->count == d->capacity) return 0;
    d->data[d->rear] = x;
    d->rear = (d->rear + 1) % d->capacity;
    d->count++;
    return 1;
}

int dq_pop_front(Deque *d, int *out) {
    if (d->count == 0) return 0;
    *out = d->data[d->front];
    d->front = (d->front + 1) % d->capacity;
    d->count--;
    return 1;
}

int dq_pop_back(Deque *d, int *out) {
    if (d->count == 0) return 0;
    d->rear = (d->rear - 1 + d->capacity) % d->capacity;
    *out = d->data[d->rear];
    d->count--;
    return 1;
}

int main(void) {
    Deque d;
    dq_init(&d, 4);
    dq_push_back(&d, 10);
    dq_push_back(&d, 20);
    dq_push_front(&d, 5);
    int val;
    dq_pop_back(&d, &val);
    printf("从尾部弹出: %d\n", val);
    dq_pop_front(&d, &val);
    printf("从头部弹出: %d\n", val);
    free(d.data);
    return 0;
}
```

Python（使用 collections 的双端队列）

```python
from collections import deque

d = deque()
d.append(10)       # push_back
d.appendleft(5)    # push_front
d.append(20)
print(d.pop())     # pop_back -> 20
print(d.popleft()) # pop_front -> 5
print(d)           # deque([10])
```

#### 为什么它很重要

- 泛化了栈和队列
- 是滑动窗口最大值、回文检查、带状态的 BFS 和任务缓冲区的核心工具
- 具有双端灵活性，操作时间复杂度为常数
- 非常适合需要对称访问的系统

#### 一个温和的证明（为什么它有效）

循环索引确保了 O(1) 的环绕操作。
每个操作移动一个索引并改变一个值：

- push → 设置值 + 调整索引
- pop → 调整索引 + 读取值

无需移动或重新分配内存，因此所有操作都保持 O(1)。

#### 自己动手试试

1.  使用双向链表实现双端队列。
2.  添加 `peek_front()` 和 `peek_back()`。
3.  模拟滑动窗口最大值算法。
4.  比较双端队列与列表在类似队列任务中的性能。

#### 测试用例

| 操作          | 头部 | 尾部 | 计数 | 双端队列（头部 → 尾部） | 备注       |
| ------------- | ---- | ---- | ---- | ----------------------- | ---------- |
| push_back(10) | 0    | 1    | 1    | [10]                    | 初始化     |
| push_back(20) | 0    | 2    | 2    | [10, 20]                |            |
| push_front(5) | 3    | 2    | 3    | [5, 10, 20]             | 头部环绕   |
| pop_back()    | 3    | 1    | 2    | [5, 10]                 | 移除 20    |
| pop_front()   | 0    | 1    | 1    | [10]                    | 移除 5     |

边界情况

- 在已满的双端队列上推入 → 错误或调整大小
- 在空的双端队列上弹出 → 错误
- 环绕的正确性至关重要

#### 复杂度

- 时间：push/pop front/back O(1)
- 空间：O(n)

双端队列是敏捷的队列，你可以从任意一侧快速而公平地操作。
### 208 循环队列

循环队列是一种针对固定大小缓冲区优化的队列，其索引会自动回绕。它广泛应用于实时系统、网络数据包缓冲区和流处理管道中，以高效地复用空间。

#### 我们要解决什么问题？

线性队列在多次出队操作后会浪费空间，因为队首索引会向前移动。循环队列通过回绕索引来解决这个问题，使得每个槽位都可重复使用。

目标：
实现一个具有固定容量的队列，其中入队和出队操作的时间复杂度均为 O(1)，并且空间是循环使用的。

#### 它是如何工作的（通俗解释）？

循环队列跟踪以下信息：

- `front`：第一个元素的索引
- `rear`：下一个要插入元素的位置索引
- `count`：元素数量

使用模运算进行回绕：

```text
next_index = (current_index + 1) % capacity
```

关键条件

- 满队列：当 `count == capacity`
- 空队列：当 `count == 0`

示例步骤（回绕模拟）

| 步骤 | 操作         | Front | Rear | Count | 队列状态         | 说明           |
| ---- | ------------ | ----- | ---- | ----- | ---------------- | -------------- |
| 1    | enqueue(10)  | 0     | 1    | 1     | [10, _, _, _]    |                |
| 2    | enqueue(20)  | 0     | 2    | 2     | [10, 20, _, _]   |                |
| 3    | enqueue(30)  | 0     | 3    | 3     | [10, 20, 30, _]  |                |
| 4    | dequeue()    | 1     | 3    | 2     | [_, 20, 30, _]   | 队首前进       |
| 5    | enqueue(40)  | 1     | 0    | 3     | [40, 20, 30, _]  | 回绕           |
| 6    | enqueue(50)  | 1     | 1    | 4     | [40, 20, 30, 50] | 满队列         |

#### 微型代码（简易版本）

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    int front;
    int rear;
    int count;
    int capacity;
} CircularQueue;

int cq_init(CircularQueue *q, int cap) {
    q->data = malloc(cap * sizeof(int));
    if (!q->data) return 0;
    q->capacity = cap;
    q->front = 0;
    q->rear = 0;
    q->count = 0;
    return 1;
}

int cq_enqueue(CircularQueue *q, int x) {
    if (q->count == q->capacity) return 0; // 队列已满
    q->data[q->rear] = x;
    q->rear = (q->rear + 1) % q->capacity;
    q->count++;
    return 1;
}

int cq_dequeue(CircularQueue *q, int *out) {
    if (q->count == 0) return 0; // 队列为空
    *out = q->data[q->front];
    q->front = (q->front + 1) % q->capacity;
    q->count--;
    return 1;
}

int cq_peek(CircularQueue *q, int *out) {
    if (q->count == 0) return 0;
    *out = q->data[q->front];
    return 1;
}

int main(void) {
    CircularQueue q;
    cq_init(&q, 4);
    cq_enqueue(&q, 10);
    cq_enqueue(&q, 20);
    cq_enqueue(&q, 30);
    int val;
    cq_dequeue(&q, &val);
    printf("出队: %d\n", val);
    cq_enqueue(&q, 40);
    cq_enqueue(&q, 50); // 如果队列已满，此操作应失败
    cq_peek(&q, &val);
    printf("队首: %d\n", val);
    free(q.data);
    return 0;
}
```

Python

```python
class CircularQueue:
    def __init__(self, capacity):
        self._cap = capacity
        self._data = [None] * capacity
        self._front = 0
        self._rear = 0
        self._count = 0

    def enqueue(self, x):
        if self._count == self._cap:
            raise OverflowError("队列已满")
        self._data[self._rear] = x
        self._rear = (self._rear + 1) % self._cap
        self._count += 1

    def dequeue(self):
        if self._count == 0:
            raise IndexError("队列为空")
        x = self._data[self._front]
        self._front = (self._front + 1) % self._cap
        self._count -= 1
        return x

    def peek(self):
        if self._count == 0:
            raise IndexError("队列为空")
        return self._data[self._front]

# 示例
q = CircularQueue(4)
q.enqueue(10)
q.enqueue(20)
q.enqueue(30)
print(q.dequeue())  # 10
q.enqueue(40)
print(q.peek())     # 20
```

#### 为什么它很重要

- 高效的空间复用，没有浪费的槽位
- 为实时系统提供可预测的内存使用
- 缓冲系统（音频、网络、流处理）的支柱
- 快速的 O(1) 操作，无需移动元素

#### 一个温和的证明（为什么它有效）

由于 `front` 和 `rear` 都使用模运算进行回绕，所有操作都保持在 `[0, capacity-1]` 范围内。每个操作修改固定数量的变量。因此：

- 入队：1 次写入 + 2 次更新
- 出队：1 次读取 + 2 次更新
  无需移动元素，无需调整大小 → 所有操作都是 O(1) 时间复杂度。

#### 亲自尝试

1.  添加 `is_full()` 和 `is_empty()` 辅助函数。
2.  实现一个覆盖模式，新的入队操作会覆盖最旧的数据。
3.  可视化多次回绕时的索引移动。
4.  使用循环队列模拟生产者-消费者缓冲区。

#### 测试用例

| 操作         | Front | Rear | Count | 队列 (Front → Rear) | 说明         |
| ------------ | ----- | ---- | ----- | ------------------- | ------------ |
| enqueue(10)  | 0     | 1    | 1     | [10, _, _, _]       | 第一个元素   |
| enqueue(20)  | 0     | 2    | 2     | [10, 20, _, _]      |              |
| enqueue(30)  | 0     | 3    | 3     | [10, 20, 30, _]     |              |
| dequeue()    | 1     | 3    | 2     | [_, 20, 30, _]      | 10 被移除    |
| enqueue(40)  | 1     | 0    | 3     | [40, 20, 30, _]     | 回绕         |
| enqueue(50)  | 1     | 1    | 4     | [40, 20, 30, 50]    | 满队列       |

边界情况

- 满队列时入队 → 拒绝或覆盖
- 空队列时出队 → 错误
- 回绕索引必须正确处理 0

#### 复杂度

- 时间复杂度：入队 O(1)，出队 O(1)，查看队首 O(1)
- 空间复杂度：O(n) 固定缓冲区

循环队列是实时数据流的心跳，稳定、循环，从不浪费一个字节。
### 209 用队列实现栈

用队列实现栈是一种有趣的转换，它使用 FIFO（先进先出）工具来实现 LIFO（后进先出）行为。它展示了如何通过巧妙地组合基本操作，让一种数据结构模拟另一种数据结构。

#### 我们要解决什么问题？

有时我们只能使用队列操作（`enqueue`、`dequeue`），但仍然希望获得栈的后进先出顺序。我们可以使用一个或两个队列来模拟 `push` 和 `pop`。

目标：
构建一个栈，仅使用队列操作，支持 `push`、`pop` 和 `peek` 操作，时间复杂度为 O(1) 或 O(n)（取决于策略）。

#### 它是如何工作的（通俗解释）？

主要有两种策略：

1.  **入栈代价高**：每次入栈后旋转元素，使得队首始终是栈顶。
2.  **出栈代价高**：正常入队，但在出栈时进行旋转。

我们将展示入栈代价高的版本，概念上更简单。

思路：
每次 `push` 操作将新元素入队，然后将所有旧元素旋转到它后面，这样最后入栈的元素始终在队首（准备出栈）。

示例步骤（入栈代价高）

| 步骤 | 操作      | 队列（队首 → 队尾） | 备注                       |
| ---- | --------- | -------------------- | ------------------------ |
| 1    | push(10)  | [10]                 | 只有一个元素             |
| 2    | push(20)  | [20, 10]             | 旋转后 20 在队首         |
| 3    | push(30)  | [30, 20, 10]         | 再次旋转                 |
| 4    | pop()     | [20, 10]             | 移除 30                  |
| 5    | push(40)  | [40, 20, 10]         | 旋转保持顺序             |

#### 微型代码（简易版本）

C 语言（使用两个队列）

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    int front, rear, count, capacity;
} Queue;

int q_init(Queue *q, int cap) {
    q->data = malloc(cap * sizeof(int));
    if (!q->data) return 0;
    q->front = 0;
    q->rear = 0;
    q->count = 0;
    q->capacity = cap;
    return 1;
}

int q_enqueue(Queue *q, int x) {
    if (q->count == q->capacity) return 0;
    q->data[q->rear] = x;
    q->rear = (q->rear + 1) % q->capacity;
    q->count++;
    return 1;
}

int q_dequeue(Queue *q, int *out) {
    if (q->count == 0) return 0;
    *out = q->data[q->front];
    q->front = (q->front + 1) % q->capacity;
    q->count--;
    return 1;
}

typedef struct {
    Queue q1, q2;
} StackViaQueue;

int svq_init(StackViaQueue *s, int cap) {
    return q_init(&s->q1, cap) && q_init(&s->q2, cap);
}

int svq_push(StackViaQueue *s, int x) {
    q_enqueue(&s->q2, x);
    int val;
    while (s->q1.count) {
        q_dequeue(&s->q1, &val);
        q_enqueue(&s->q2, val);
    }
    // 交换 q1 和 q2
    Queue tmp = s->q1;
    s->q1 = s->q2;
    s->q2 = tmp;
    return 1;
}

int svq_pop(StackViaQueue *s, int *out) {
    return q_dequeue(&s->q1, out);
}

int svq_peek(StackViaQueue *s, int *out) {
    if (s->q1.count == 0) return 0;
    *out = s->q1.data[s->q1.front];
    return 1;
}

int main(void) {
    StackViaQueue s;
    svq_init(&s, 10);
    svq_push(&s, 10);
    svq_push(&s, 20);
    svq_push(&s, 30);
    int val;
    svq_pop(&s, &val);
    printf("Popped: %d\n", val);
    svq_peek(&s, &val);
    printf("Top: %d\n", val);
    free(s.q1.data);
    free(s.q2.data);
    return 0;
}
```

Python

```python
from collections import deque

class StackViaQueue:
    def __init__(self):
        self.q = deque()

    def push(self, x):
        n = len(self.q)
        self.q.append(x)
        # 旋转所有旧元素
        for _ in range(n):
            self.q.append(self.q.popleft())

    def pop(self):
        if not self.q:
            raise IndexError("pop from empty stack")
        return self.q.popleft()

    def peek(self):
        if not self.q:
            raise IndexError("peek from empty stack")
        return self.q[0]

# 示例
s = StackViaQueue()
s.push(10)
s.push(20)
s.push(30)
print(s.pop())  # 30
print(s.peek()) # 20
```

#### 为什么这很重要

- 展示了数据结构的对偶性（在队列上构建栈）
- 强化了对操作权衡（入栈代价高 vs 出栈代价高）的理解
- 是算法模拟的绝佳教学示例
- 有助于深入理解复杂度和资源使用

#### 一个温和的证明（为什么它有效）

在入栈代价高的方法中：

- 每次入栈将所有先前的元素旋转到新元素后面 → 新元素成为队首。
- 出栈只是简单地从队首出队 → 正确的 LIFO 顺序。

因此顺序得以保持：最新的元素总是最先离开。

#### 自己动手试试

1.  实现出栈代价高的变体（push O(1), pop O(n)）。
2.  添加 `is_empty()` 辅助函数。
3.  比较 n 次入栈和出栈的总操作次数。
4.  使用结构体或模板扩展为泛型类型。

#### 测试用例

| 操作      | 队列（队首 → 队尾） | 备注           |
| --------- | -------------------- | -------------- |
| push(10)  | [10]                 | 单个元素       |
| push(20)  | [20, 10]             | 旋转           |
| push(30)  | [30, 20, 10]         | 旋转           |
| pop()     | [20, 10]             | 返回 30        |
| peek()    | [20, 10]             | 返回 20        |

边界情况

- 从空栈 pop/peek → 错误
- 达到容量限制 → 拒绝 push
- 即使使用单个队列也能工作（入栈后旋转）

#### 复杂度

- 入栈：O(n)
- 出栈/查看栈顶：O(1)
- 空间：O(n)

用队列实现栈证明了约束催生创造力，相同的数据，不同的舞蹈。
### 210 用栈实现队列

用栈实现队列反转了故事：使用 LIFO（后进先出）工具构建 FIFO（先进先出）行为。这是一个经典的算法反转练习，展示了如何通过巧妙的顺序操作让基本操作相互模拟。

#### 我们要解决什么问题？

假设你只有栈（具有 `push`、`pop`、`peek` 操作），但需要队列行为（具有 `enqueue`、`dequeue` 操作）。我们希望按到达顺序处理项目，即先进先出，尽管栈的操作是后进先出。

目标：
仅使用栈操作实现队列的 `enqueue`、`dequeue` 和 `peek` 方法。

#### 它是如何工作的（通俗解释）？

两个栈就足够了：

- 收件箱（inbox）：我们在此压入新项目（入队）
- 发件箱（outbox）：我们在此弹出旧项目（出队）

当出队时，如果 `发件箱` 为空，我们将所有项目从 `收件箱` 移动到 `发件箱`，反转它们的顺序，使得最旧的项目位于顶部。

这个反转步骤恢复了 FIFO 行为。

示例步骤（双栈法）

| 步骤 | 操作         | 收件箱（顶 → 底） | 发件箱（顶 → 底） | 备注                 |
| ---- | ------------ | ----------------- | ----------------- | -------------------- |
| 1    | enqueue(10)  | [10]              | []                |                      |
| 2    | enqueue(20)  | [20, 10]          | []                |                      |
| 3    | enqueue(30)  | [30, 20, 10]      | []                |                      |
| 4    | dequeue()    | []                | [10, 20, 30]      | 转移 + 弹出(10)      |
| 5    | enqueue(40)  | [40]              | [20, 30]          | 混合状态             |
| 6    | dequeue()    | [40]              | [30]              | 从发件箱弹出(20)     |

#### 微型代码（简易版本）

C 语言（使用两个栈）

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    int top;
    int capacity;
} Stack;

int stack_init(Stack *s, int cap) {
    s->data = malloc(cap * sizeof(int));
    if (!s->data) return 0;
    s->top = -1;
    s->capacity = cap;
    return 1;
}

int stack_push(Stack *s, int x) {
    if (s->top + 1 == s->capacity) return 0;
    s->data[++s->top] = x;
    return 1;
}

int stack_pop(Stack *s, int *out) {
    if (s->top == -1) return 0;
    *out = s->data[s->top--];
    return 1;
}

int stack_peek(Stack *s, int *out) {
    if (s->top == -1) return 0;
    *out = s->data[s->top];
    return 1;
}

int stack_empty(Stack *s) {
    return s->top == -1;
}

typedef struct {
    Stack in, out;
} QueueViaStack;

int qvs_init(QueueViaStack *q, int cap) {
    return stack_init(&q->in, cap) && stack_init(&q->out, cap);
}

int qvs_enqueue(QueueViaStack *q, int x) {
    return stack_push(&q->in, x);
}

int qvs_shift(QueueViaStack *q) {
    int val;
    while (!stack_empty(&q->in)) {
        stack_pop(&q->in, &val);
        stack_push(&q->out, val);
    }
    return 1;
}

int qvs_dequeue(QueueViaStack *q, int *out) {
    if (stack_empty(&q->out)) qvs_shift(q);
    return stack_pop(&q->out, out);
}

int qvs_peek(QueueViaStack *q, int *out) {
    if (stack_empty(&q->out)) qvs_shift(q);
    return stack_peek(&q->out, out);
}

int main(void) {
    QueueViaStack q;
    qvs_init(&q, 10);
    qvs_enqueue(&q, 10);
    qvs_enqueue(&q, 20);
    qvs_enqueue(&q, 30);
    int val;
    qvs_dequeue(&q, &val);
    printf("Dequeued: %d\n", val);
    qvs_enqueue(&q, 40);
    qvs_peek(&q, &val);
    printf("Front: %d\n", val);
    free(q.in.data);
    free(q.out.data);
    return 0;
}
```

Python（双栈队列）

```python
class QueueViaStack:
    def __init__(self):
        self.inbox = []
        self.outbox = []

    def enqueue(self, x):
        self.inbox.append(x)

    def dequeue(self):
        if not self.outbox:
            while self.inbox:
                self.outbox.append(self.inbox.pop())
        if not self.outbox:
            raise IndexError("dequeue from empty queue")
        return self.outbox.pop()

    def peek(self):
        if not self.outbox:
            while self.inbox:
                self.outbox.append(self.inbox.pop())
        if not self.outbox:
            raise IndexError("peek from empty queue")
        return self.outbox[-1]

# 示例
q = QueueViaStack()
q.enqueue(10)
q.enqueue(20)
q.enqueue(30)
print(q.dequeue())  # 10
q.enqueue(40)
print(q.peek())     # 20
```

#### 为什么这很重要

- 展示了用栈操作模拟队列
- 数据结构对偶性的核心教学示例
- 有助于在约束条件下设计抽象接口
- 支撑了一些流处理和缓冲系统

#### 一个温和的证明（为什么它有效）

每次转移（`收件箱 → 发件箱`）都会反转一次顺序，从而恢复 FIFO 序列。

- 入队将项目压入收件箱（LIFO）
- 出队从发件箱弹出项目（反转后的 LIFO）
  因此整体效果 = FIFO

转移仅在发件箱为空时发生，因此每个操作的摊还成本是 O(1)。

#### 亲自尝试

1.  实现一个单栈递归版本。
2.  添加 `is_empty()` 辅助方法。
3.  测量摊还复杂度与最坏情况复杂度。
4.  扩展到泛型数据类型。

#### 测试用例

| 操作         | 收件箱（顶→底） | 发件箱（顶→底） | 结果        | 备注             |
| ------------ | --------------- | --------------- | ----------- | ---------------- |
| enqueue(10)  | [10]            | []              |             |                  |
| enqueue(20)  | [20, 10]        | []              |             |                  |
| enqueue(30)  | [30, 20, 10]    | []              |             |                  |
| dequeue()    | []              | [10, 20, 30]    | 返回 10     | 转移 + 弹出      |
| enqueue(40)  | [40]            | [20, 30]        |             |                  |
| dequeue()    | [40]            | [30]            | 返回 20     |                  |

边界情况

- 从空队列出队 → 错误
- 多次出队触发一次转移
- 发件箱被高效复用

#### 复杂度

- 时间：每个操作摊还 O(1)（转移时最坏情况 O(n)）
- 空间：O(n)

用栈实现队列展示了对称性，通过一次巧妙的反转将 LIFO 变为 FIFO。

## 第 22 章 哈希表及其变体
### 211 哈希表插入

哈希表存储键值对，以实现闪电般的快速查找。它使用哈希函数将键映射到数组索引，让我们能以接近常数的时间访问数据。

#### 我们要解决什么问题？

我们需要一种数据结构，能够通过键高效地插入、搜索和删除，而无需扫描每个元素。数组可以通过索引进行随机访问；哈希表将这种能力扩展到了任意键。

目标：
通过哈希函数将每个键映射到一个槽位，并优雅地解决任何冲突。

#### 它如何工作（通俗解释）？

哈希表使用哈希函数将键转换为索引：

```text
index = hash(key) % capacity
```

当插入一个新的（键，值）对时：

1.  计算哈希索引。
2.  如果槽位为空 → 将键值对放入该槽位。
3.  如果槽位被占用 → 处理冲突（链地址法或开放寻址法）。

我们将使用最简单的链地址法（每个槽位一个链表）。

示例步骤（链地址法）

| 步骤 | 键       | Hash(key) | 索引 | 操作                           |
| ---- | -------- | --------- | ---- | ------------------------------ |
| 1    | "apple"  | 42        | 2    | 插入 ("apple", 10)             |
| 2    | "banana" | 15        | 3    | 插入 ("banana", 20)            |
| 3    | "pear"   | 18        | 2    | 冲突 → 在索引 2 处链入         |
| 4    | "peach"  | 21        | 1    | 插入新键值对                   |

插入后的表：

| 索引 | 链                               |
| ---- | -------------------------------- |
| 0    | -                                |
| 1    | ("peach", 40)                    |
| 2    | ("apple", 10) → ("pear", 30)     |
| 3    | ("banana", 20)                   |

#### 微型代码（简易版本）

C 语言（链地址法示例）

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TABLE_SIZE 5

typedef struct Node {
    char *key;
    int value;
    struct Node *next;
} Node;

typedef struct {
    Node *buckets[TABLE_SIZE];
} HashTable;

unsigned int hash(const char *key) {
    unsigned int h = 0;
    while (*key) h = h * 31 + *key++;
    return h % TABLE_SIZE;
}

HashTable* ht_create() {
    HashTable *ht = malloc(sizeof(HashTable));
    for (int i = 0; i < TABLE_SIZE; i++) ht->buckets[i] = NULL;
    return ht;
}

void ht_insert(HashTable *ht, const char *key, int value) {
    unsigned int idx = hash(key);
    Node *node = ht->buckets[idx];
    while (node) {
        if (strcmp(node->key, key) == 0) { node->value = value; return; }
        node = node->next;
    }
    Node *new_node = malloc(sizeof(Node));
    new_node->key = strdup(key);
    new_node->value = value;
    new_node->next = ht->buckets[idx];
    ht->buckets[idx] = new_node;
}

int ht_search(HashTable *ht, const char *key, int *out) {
    unsigned int idx = hash(key);
    Node *node = ht->buckets[idx];
    while (node) {
        if (strcmp(node->key, key) == 0) { *out = node->value; return 1; }
        node = node->next;
    }
    return 0;
}

int main(void) {
    HashTable *ht = ht_create();
    ht_insert(ht, "apple", 10);
    ht_insert(ht, "pear", 30);
    ht_insert(ht, "banana", 20);
    int val;
    if (ht_search(ht, "pear", &val))
        printf("pear: %d\n", val);
    return 0;
}
```

Python（字典模拟）

```python
class HashTable:
    def __init__(self, size=5):
        self.size = size
        self.table = [[] for _ in range(size)]

    def _hash(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        idx = self._hash(key)
        for pair in self.table[idx]:
            if pair[0] == key:
                pair[1] = value
                return
        self.table[idx].append([key, value])

    def search(self, key):
        idx = self._hash(key)
        for k, v in self.table[idx]:
            if k == key:
                return v
        return None

# 示例
ht = HashTable()
ht.insert("apple", 10)
ht.insert("pear", 30)
ht.insert("banana", 20)
print(ht.search("pear"))  # 30
```

#### 为什么重要

-   提供平均 O(1) 的访问、插入、删除操作
-   是符号表、映射、集合和字典的基石
-   用于缓存、编译器和索引系统
-   引入了核心思想：哈希函数 + 冲突处理

#### 一个温和的证明（为什么它有效）

设表大小 = m，键的数量 = n。
如果哈希函数均匀分布键，则期望的链长度 = α = n/m（负载因子）。

-   平均查找时间：O(1 + α)
-   保持 α ≤ 1 → 接近常数时间
  如果冲突最小化，操作就能保持快速。

#### 动手试试

1.  实现更新操作以修改现有键的值。
2.  添加 delete(key) 函数以从链中删除条目。
3.  尝试不同的哈希函数（例如 djb2、FNV-1a）。
4.  测量时间与负载因子的关系。

#### 测试用例

| 操作     | 键       | 值   | 结果     | 备注         |
| -------- | -------- | ---- | -------- | ------------ |
| insert   | "apple"  | 10   | success  | 新键         |
| insert   | "banana" | 20   | success  | 新键         |
| insert   | "apple"  | 15   | update   | 键已存在     |
| search   | "banana" |      | 20       | 找到         |
| search   | "grape"  |      | None     | 未找到       |

边界情况

-   插入重复键 → 更新值
-   搜索不存在的键 → 返回 None
-   表满（开放寻址法） → 需要重新哈希

#### 复杂度

-   时间：平均 O(1)，最坏 O(n)（全部冲突）
-   空间：O(n + m)（键 + 表槽位）

哈希表插入是将混乱转化为秩序的艺术：哈希、映射、解决、存储。
### 212 线性探测

线性探测是开放寻址哈希表中最简单的冲突解决策略之一。当发生冲突时，它会一步一步地在表中查找下一个空槽，如果需要则回绕到表头。

#### 我们要解决什么问题？

当两个键哈希到同一个索引时，我们把新的键存储在哪里？
线性探测不使用链式节点，而是搜索下一个可用的槽，将所有数据都保存在数组内部。

目标：
通过从冲突点开始线性扫描，直到找到一个空槽，来解决冲突。

#### 它是如何工作的（通俗解释）？

插入一个键时：

1. 计算索引 = `hash(key) % capacity`。
2. 如果槽为空 → 将键放在那里。
3. 如果被占用 → 移动到 `(index + 1) % capacity`。
4. 重复直到找到一个空槽或表已满。

查找和删除遵循相同的探测序列，直到找到键或遇到空槽。

示例步骤（容量 = 7）

| 步骤 | 键  | Hash(key) | 索引 | 操作                       |
| ---- | --- | --------- | ---- | -------------------------- |
| 1    | 10  | 3         | 3    | 放置在索引 3               |
| 2    | 24  | 3         | 4    | 冲突 → 移动到 4            |
| 3    | 31  | 3         | 5    | 冲突 → 移动到 5            |
| 4    | 17  | 3         | 6    | 冲突 → 移动到 6            |
| 5    | 38  | 3         | 0    | 回绕 → 放置在 0            |

最终表

| 索引 | 值  |
| ---- | --- |
| 0    | 38  |
| 1    | -   |
| 2    | -   |
| 3    | 10  |
| 4    | 24  |
| 5    | 31  |
| 6    | 17  |

#### 微型代码（简易版本）

C

```c
#include <stdio.h>
#include <stdlib.h>

#define CAPACITY 7
#define EMPTY -1
#define DELETED -2

typedef struct {
    int *table;
} HashTable;

int hash(int key) { return key % CAPACITY; }

HashTable* ht_create() {
    HashTable *ht = malloc(sizeof(HashTable));
    ht->table = malloc(sizeof(int) * CAPACITY);
    for (int i = 0; i < CAPACITY; i++) ht->table[i] = EMPTY;
    return ht;
}

void ht_insert(HashTable *ht, int key) {
    int idx = hash(key);
    for (int i = 0; i < CAPACITY; i++) {
        int pos = (idx + i) % CAPACITY;
        if (ht->table[pos] == EMPTY || ht->table[pos] == DELETED) {
            ht->table[pos] = key;
            return;
        }
    }
    printf("表已满，无法插入 %d\n", key);
}

int ht_search(HashTable *ht, int key) {
    int idx = hash(key);
    for (int i = 0; i < CAPACITY; i++) {
        int pos = (idx + i) % CAPACITY;
        if (ht->table[pos] == EMPTY) return -1;
        if (ht->table[pos] == key) return pos;
    }
    return -1;
}

void ht_delete(HashTable *ht, int key) {
    int pos = ht_search(ht, key);
    if (pos != -1) ht->table[pos] = DELETED;
}

int main(void) {
    HashTable *ht = ht_create();
    ht_insert(ht, 10);
    ht_insert(ht, 24);
    ht_insert(ht, 31);
    ht_insert(ht, 17);
    ht_insert(ht, 38);
    for (int i = 0; i < CAPACITY; i++)
        printf("[%d] = %d\n", i, ht->table[i]);
    return 0;
}
```

Python

```python
class LinearProbingHash:
    def __init__(self, size=7):
        self.size = size
        self.table = [None] * size

    def _hash(self, key):
        return key % self.size

    def insert(self, key):
        idx = self._hash(key)
        for i in range(self.size):
            pos = (idx + i) % self.size
            if self.table[pos] is None or self.table[pos] == "DELETED":
                self.table[pos] = key
                return
        raise OverflowError("哈希表已满")

    def search(self, key):
        idx = self._hash(key)
        for i in range(self.size):
            pos = (idx + i) % self.size
            if self.table[pos] is None:
                return None
            if self.table[pos] == key:
                return pos
        return None

    def delete(self, key):
        pos = self.search(key)
        if pos is not None:
            self.table[pos] = "DELETED"

# 示例
ht = LinearProbingHash()
for k in [10, 24, 31, 17, 38]:
    ht.insert(k)
print(ht.table)
print("Search 17 at:", ht.search(17))
```

#### 为什么它很重要

- 开放寻址的最简单形式
- 将所有条目保存在一个数组内（无需额外内存）
- 出色的缓存性能
- 构成现代原地哈希映射的基础

#### 一个温和的证明（为什么它有效）

每个键在插入、查找和删除期间都遵循相同的探测序列。
因此，如果一个键在表中，查找会找到它；如果不在，查找会遇到一个空槽并停止。
均匀哈希确保平均探测长度 ≈ 1 / (1 - α)，其中 α = n / m 是负载因子。

#### 自己动手试试

1. 当负载因子 > 0.7 时，实现 `resize()`。
2. 测试插入顺序和回绕行为。
3. 比较线性探测与链式法的性能。
4. 可视化负载增加时的聚集现象。

#### 测试用例

| 操作     | 键  | 结果         | 备注           |
| -------- | --- | ------------ | -------------- |
| insert   | 10  | index 3      | 无冲突         |
| insert   | 24  | index 4      | 移动 1 个槽    |
| insert   | 31  | index 5      | 移动 2 个槽    |
| insert   | 17  | index 6      | 移动 3 个槽    |
| insert   | 38  | index 0      | 回绕           |
| search   | 17  | found at 6   | 线性搜索       |
| delete   | 24  | mark deleted | 槽可重用       |

边界情况

- 表已满 → 插入失败
- 已删除的槽被重用
- 必须在 EMPTY 处停止，而不是 DELETED

#### 复杂度

- 时间：

  * 如果 α 小，平均 O(1)
  * 如果聚集严重，最坏 O(n)
- 空间：O(n)

线性探测通过冲突点走直线，简单、局部化，在负载低时速度快。
### 213 二次探测

二次探测通过减少主聚集（primary clustering）改进了线性探测。它不是逐个槽位地遍历，而是以二次增量跳跃，使冲突的键更均匀地散布在表中。

#### 我们要解决什么问题？

在线性探测中，连续的已占用槽位会导致聚集（clustering），从而产生长的探测链并降低性能。
二次探测通过使用非线性探测序列来打破这些连续序列。

目标：
通过检查偏移量为二次方值（`+1², +2², +3², …`）的索引来解决冲突，在保持可预测探测顺序的同时减少聚集。

#### 它是如何工作的（通俗解释）？

插入键时：

1.  计算 `index = hash(key) % capacity`。
2.  如果槽位为空 → 插入。
3.  如果槽位被占用 → 尝试 `(index + 1²) % capacity`，`(index + 2²) % capacity`，等等。
4.  继续直到找到空槽位或表满为止。

查找和删除遵循相同的探测序列。

示例步骤（容量 = 7）

| 步骤 | 键  | Hash(key) | 探测（序列） | 最终槽位 |
| ---- | --- | --------- | ------------ | -------- |
| 1    | 10  | 3         | 3            | 3        |
| 2    | 24  | 3         | 3, 4         | 4        |
| 3    | 31  | 3         | 3, 4, 0      | 0        |
| 4    | 17  | 3         | 3, 4, 0, 2   | 2        |

插入后的表：

| 索引 | 值  |
| ---- | --- |
| 0    | 31  |
| 1    | -   |
| 2    | 17  |
| 3    | 10  |
| 4    | 24  |
| 5    | -   |
| 6    | -   |

#### 微型代码（简易版本）

C

```c
#include <stdio.h>
#include <stdlib.h>

#define CAPACITY 7
#define EMPTY -1
#define DELETED -2

typedef struct {
    int *table;
} HashTable;

int hash(int key) { return key % CAPACITY; }

HashTable* ht_create() {
    HashTable *ht = malloc(sizeof(HashTable));
    ht->table = malloc(sizeof(int) * CAPACITY);
    for (int i = 0; i < CAPACITY; i++) ht->table[i] = EMPTY;
    return ht;
}

void ht_insert(HashTable *ht, int key) {
    int idx = hash(key);
    for (int i = 0; i < CAPACITY; i++) {
        int pos = (idx + i * i) % CAPACITY;
        if (ht->table[pos] == EMPTY || ht->table[pos] == DELETED) {
            ht->table[pos] = key;
            return;
        }
    }
    printf("表已满，无法插入 %d\n", key);
}

int ht_search(HashTable *ht, int key) {
    int idx = hash(key);
    for (int i = 0; i < CAPACITY; i++) {
        int pos = (idx + i * i) % CAPACITY;
        if (ht->table[pos] == EMPTY) return -1;
        if (ht->table[pos] == key) return pos;
    }
    return -1;
}

int main(void) {
    HashTable *ht = ht_create();
    ht_insert(ht, 10);
    ht_insert(ht, 24);
    ht_insert(ht, 31);
    ht_insert(ht, 17);
    for (int i = 0; i < CAPACITY; i++)
        printf("[%d] = %d\n", i, ht->table[i]);
    return 0;
}
```

Python

```python
class QuadraticProbingHash:
    def __init__(self, size=7):
        self.size = size
        self.table = [None] * size

    def _hash(self, key):
        return key % self.size

    def insert(self, key):
        idx = self._hash(key)
        for i in range(self.size):
            pos = (idx + i * i) % self.size
            if self.table[pos] is None or self.table[pos] == "DELETED":
                self.table[pos] = key
                return
        raise OverflowError("哈希表已满")

    def search(self, key):
        idx = self._hash(key)
        for i in range(self.size):
            pos = (idx + i * i) % self.size
            if self.table[pos] is None:
                return None
            if self.table[pos] == key:
                return pos
        return None

# 示例
ht = QuadraticProbingHash()
for k in [10, 24, 31, 17]:
    ht.insert(k)
print(ht.table)
print("搜索 24 在位置:", ht.search(24))
```

#### 为什么它重要

-   减少了线性探测中出现的主聚集
-   使键的分布更均匀
-   避免了额外的指针（所有内容都保存在一个数组中）
-   适用于空间紧张且局部性有帮助的哈希表

#### 一个温和的证明（为什么它有效）

探测序列：
$$
i = 0, 1, 2, 3, \ldots
$$
第 *i* 步的索引是
$$
(index + i^2) \bmod m
$$
如果表大小 *m* 是质数，该序列在重复之前最多访问 ⌈m/2⌉ 个不同的槽位，从而保证当负载因子 < 0.5 时能找到空槽位。

因此，所有操作都遵循可预测的、有限的序列。

#### 自己动手试试

1.  比较相同键下与线性探测的聚集情况。
2.  尝试不同的表大小（质数 vs 合数）。
3.  正确实现删除标记。
4.  用小表可视化探测路径。

#### 测试用例

| 操作   | 键  | 探测序列     | 槽位 | 备注           |
| ------ | --- | ------------ | ---- | -------------- |
| insert | 10  | 3            | 3    | 无冲突         |
| insert | 24  | 3, 4         | 4    | 1 步           |
| insert | 31  | 3, 4, 0      | 0    | 2 步           |
| insert | 17  | 3, 4, 0, 2   | 2    | 3 步           |
| search | 31  | 3 → 4 → 0    | 找到 | 二次探测路径 |

边界情况

-   表满 → 插入失败
-   需要表大小为质数以实现完全覆盖
-   需要负载因子 < 0.5 以避免无限循环

#### 复杂度

-   时间：平均 O(1)，最坏 O(n)
-   空间：O(n)

二次探测用曲线代替直线，将冲突平滑地散布在整个表中。
### 214 双重哈希

双重哈希使用两个独立的哈希函数来最小化冲突。当发生冲突时，它通过第二个哈希值向前跳跃，为每个键创建唯一的探测序列，从而大大减少了聚集现象。

#### 我们要解决什么问题？

线性探测和二次探测都存在聚集模式的问题，特别是当键具有相似的初始索引时。双重哈希通过引入第二个哈希函数来打破这种模式，该函数定义了每个键的步长。

目标：
使用两个哈希函数来确定探测序列：
$$
\text{index}_i = (h_1(key) + i \cdot h_2(key)) \bmod m
$$
这会产生独立的探测路径，并避免键之间的重叠。

#### 它是如何工作的（通俗解释）？

插入键时：

1.  计算主哈希：`h1 = key % capacity`。
2.  计算步长：`h2 = 1 + (key % (capacity - 1))`（永不为零）。
3.  尝试 `h1`；如果被占用，则尝试 `(h1 + h2) % m`、`(h1 + 2*h2) % m` 等。
4.  重复直到找到空槽。

搜索和删除操作遵循相同的模式。

示例步骤（容量 = 7）

| 步骤 | 键  | h₁(key) | h₂(key) | 探测序列       | 最终槽位 |
| ---- | --- | ------- | ------- | -------------- | -------- |
| 1    | 10  | 3       | 4       | 3              | 3        |
| 2    | 24  | 3       | 4       | 3 → 0          | 0        |
| 3    | 31  | 3       | 4       | 3 → 0 → 4      | 4        |
| 4    | 17  | 3       | 3       | 3 → 6          | 6        |

最终表

| 索引 | 值   |
| ---- | ---- |
| 0    | 24   |
| 1    | -    |
| 2    | -    |
| 3    | 10   |
| 4    | 31   |
| 5    | -    |
| 6    | 17   |

#### 微型代码（简易版本）

C

```c
#include <stdio.h>
#include <stdlib.h>

#define CAPACITY 7
#define EMPTY -1
#define DELETED -2

typedef struct {
    int *table;
} HashTable;

int h1(int key) { return key % CAPACITY; }
int h2(int key) { return 1 + (key % (CAPACITY - 1)); }

HashTable* ht_create() {
    HashTable *ht = malloc(sizeof(HashTable));
    ht->table = malloc(sizeof(int) * CAPACITY);
    for (int i = 0; i < CAPACITY; i++) ht->table[i] = EMPTY;
    return ht;
}

void ht_insert(HashTable *ht, int key) {
    int idx1 = h1(key);
    int step = h2(key);
    for (int i = 0; i < CAPACITY; i++) {
        int pos = (idx1 + i * step) % CAPACITY;
        if (ht->table[pos] == EMPTY || ht->table[pos] == DELETED) {
            ht->table[pos] = key;
            return;
        }
    }
    printf("表已满，无法插入 %d\n", key);
}

int ht_search(HashTable *ht, int key) {
    int idx1 = h1(key), step = h2(key);
    for (int i = 0; i < CAPACITY; i++) {
        int pos = (idx1 + i * step) % CAPACITY;
        if (ht->table[pos] == EMPTY) return -1;
        if (ht->table[pos] == key) return pos;
    }
    return -1;
}

int main(void) {
    HashTable *ht = ht_create();
    ht_insert(ht, 10);
    ht_insert(ht, 24);
    ht_insert(ht, 31);
    ht_insert(ht, 17);
    for (int i = 0; i < CAPACITY; i++)
        printf("[%d] = %d\n", i, ht->table[i]);
    return 0;
}
```

Python

```python
class DoubleHash:
    def __init__(self, size=7):
        self.size = size
        self.table = [None] * size

    def _h1(self, key):
        return key % self.size

    def _h2(self, key):
        return 1 + (key % (self.size - 1))

    def insert(self, key):
        h1 = self._h1(key)
        h2 = self._h2(key)
        for i in range(self.size):
            pos = (h1 + i * h2) % self.size
            if self.table[pos] is None or self.table[pos] == "DELETED":
                self.table[pos] = key
                return
        raise OverflowError("哈希表已满")

    def search(self, key):
        h1 = self._h1(key)
        h2 = self._h2(key)
        for i in range(self.size):
            pos = (h1 + i * h2) % self.size
            if self.table[pos] is None:
                return None
            if self.table[pos] == key:
                return pos
        return None

# 示例
ht = DoubleHash()
for k in [10, 24, 31, 17]:
    ht.insert(k)
print(ht.table)
print("搜索 24 在位置:", ht.search(24))
```

#### 为什么它很重要

-   最小化主要和次要聚集
-   探测序列依赖于键，而不是在冲突键之间共享
-   当两个哈希函数都良好时，实现均匀分布
-   构成高性能开放寻址映射的基础

#### 一个温和的证明（为什么它有效）

如果容量 m 是质数且 h₂(key) 永不为 0，
那么每个键都会生成一个覆盖所有槽位的唯一探测序列：
$$
\text{indices} = {h_1, h_1 + h_2, h_1 + 2h_2, \ldots} \bmod m
$$
因此，空槽总是可达的，并且搜索能找到所有候选位置。

预期探测次数 ≈ $\frac{1}{1 - \alpha}$，与其他开放寻址法相同，但聚集程度更低。

#### 自己动手试试

1.  尝试不同的 h₂ 函数（例如 `7 - key % 7`）。
2.  与线性和二次探测比较探测长度。
3.  可视化小容量表的探测路径。
4.  使用合数与质数容量进行测试。

#### 测试用例

| 操作     | 键  | h₁   | h₂    | 探测序列       | 最终槽位 |
| -------- | --- | ---- | ----- | -------------- | -------- |
| 插入     | 10  | 3    | 4     | 3              | 3        |
| 插入     | 24  | 3    | 4     | 3, 0           | 0        |
| 插入     | 31  | 3    | 4     | 3, 0, 4        | 4        |
| 插入     | 17  | 3    | 3     | 3, 6           | 6        |
| 搜索     | 24  | 3, 0 | 找到  |                |          |

边界情况

-   h₂(key) 必须非零
-   m 应为质数以实现完全覆盖
-   哈希函数选择不当 → 覆盖不完全

#### 复杂度

-   时间：平均 O(1)，最坏 O(n)
-   空间：O(n)

双重哈希将冲突转化为一场优雅的舞蹈，两个哈希函数编织出很少交叉的路径。
### 215 布谷鸟哈希

布谷鸟哈希从自然界汲取灵感：就像布谷鸟在多个巢穴中产卵一样，每个键都有不止一个可能的家。如果一个位置被占用，它会*踢出*当前的占用者，然后该占用者会移动到它的备用位置，从而确保快速且可预测的查找。

#### 我们要解决什么问题？

传统的开放寻址方法（线性探测、二次探测、双重哈希）在高负载因子下性能可能会下降，导致长的探测序列。布谷鸟哈希通过为每个键提供多个可能的位置，保证了常数时间的查找。

目标：
使用两个哈希函数，并在发生冲突时重新定位键，以维持 O(1) 的搜索和插入时间。

#### 它是如何工作的（通俗解释）？

每个键有两个候选槽位，由两个哈希函数决定：
$$
h_1(k), \quad h_2(k)
$$

插入一个键时：

1. 尝试 `h1(k)` → 如果为空，则放入。
2. 如果被占用 → *踢出*现有的键。
3. 将被踢出的键重新插入到它的备用槽位。
4. 重复此过程，直到所有键都放置好或检测到循环（此时需要重新哈希）。

示例步骤（容量 = 7）

| 步骤 | 键  | h₁(键) | h₂(键) | 操作                       |
| ---- | --- | ------ | ------ | -------------------------- |
| 1    | 10  | 3      | 5      | 槽位 3 空 → 放置 10        |
| 2    | 24  | 3      | 4      | 槽位 3 被占用 → 移动 10    |
| 3    | 10  | 5      | 3      | 槽位 5 空 → 放置 10        |
| 4    | 31  | 3      | 6      | 槽位 3 空 → 放置 31        |

最终表

| 索引 | 值  |
| ---- | --- |
| 0    | -   |
| 1    | -   |
| 2    | -   |
| 3    | 31  |
| 4    | 24  |
| 5    | 10  |
| 6    | -   |

每个键只需检查两个位置，即可在 O(1) 时间内访问。

#### 微型代码（简易版本）

C 语言（双表布谷鸟哈希）

```c
#include <stdio.h>
#include <stdlib.h>

#define CAPACITY 7
#define EMPTY -1
#define MAX_RELOCATIONS 10

typedef struct {
    int table1[CAPACITY];
    int table2[CAPACITY];
} CuckooHash;

int h1(int key) { return key % CAPACITY; }
int h2(int key) { return (key / CAPACITY) % CAPACITY; }

void init(CuckooHash *ht) {
    for (int i = 0; i < CAPACITY; i++) {
        ht->table1[i] = EMPTY;
        ht->table2[i] = EMPTY;
    }
}

int insert(CuckooHash *ht, int key) {
    int pos, tmp, loop_guard = 0;
    for (int i = 0; i < MAX_RELOCATIONS; i++) {
        pos = h1(key);
        if (ht->table1[pos] == EMPTY) {
            ht->table1[pos] = key;
            return 1;
        }
        // 踢出
        tmp = ht->table1[pos];
        ht->table1[pos] = key;
        key = tmp;
        pos = h2(key);
        if (ht->table2[pos] == EMPTY) {
            ht->table2[pos] = key;
            return 1;
        }
        tmp = ht->table2[pos];
        ht->table2[pos] = key;
        key = tmp;
    }
    printf("检测到循环，需要重新哈希\n");
    return 0;
}

int search(CuckooHash *ht, int key) {
    int pos1 = h1(key);
    int pos2 = h2(key);
    if (ht->table1[pos1] == key || ht->table2[pos2] == key) return 1;
    return 0;
}

int main(void) {
    CuckooHash ht;
    init(&ht);
    insert(&ht, 10);
    insert(&ht, 24);
    insert(&ht, 31);
    for (int i = 0; i < CAPACITY; i++)
        printf("[%d] T1=%d T2=%d\n", i, ht.table1[i], ht.table2[i]);
    return 0;
}
```

Python

```python
class CuckooHash:
    def __init__(self, size=7):
        self.size = size
        self.table1 = [None] * size
        self.table2 = [None] * size
        self.max_reloc = 10

    def _h1(self, key): return key % self.size
    def _h2(self, key): return (key // self.size) % self.size

    def insert(self, key):
        for _ in range(self.max_reloc):
            idx1 = self._h1(key)
            if self.table1[idx1] is None:
                self.table1[idx1] = key
                return
            key, self.table1[idx1] = self.table1[idx1], key  # 交换

            idx2 = self._h2(key)
            if self.table2[idx2] is None:
                self.table2[idx2] = key
                return
            key, self.table2[idx2] = self.table2[idx2], key  # 交换
        raise RuntimeError("检测到循环，需要重新哈希")

    def search(self, key):
        return key in self.table1 or key in self.table2

# 示例
ht = CuckooHash()
for k in [10, 24, 31]:
    ht.insert(k)
print("表1:", ht.table1)
print("表2:", ht.table2)
print("查找 24:", ht.search(24))
```

#### 为什么它很重要

- O(1) 查找，每个键始终只有两个槽位
- 完全避免了聚集
- 对于高负载因子（可达 0.5–0.9）表现优异
- 简单可预测的探测路径
- 硬件表（例如网络路由）的绝佳选择

#### 一个温和的证明（为什么它有效）

每个键最多有两个可能的家。

- 如果两个都被占用，置换操作确保最终收敛（或检测到循环）。
- 循环长度有界 → 很少需要重新哈希。

期望的插入时间 = O(1) 摊还；搜索始终只需 2 次检查。

#### 亲自尝试

1.  实现检测到循环时的重新哈希。
2.  添加删除键功能并测试重新插入。
3.  可视化插入时的置换链。
4.  与双重哈希进行性能比较。

#### 测试用例

| 操作     | 键  | h₁   | h₂    | 操作                     |
| -------- | --- | ---- | ----- | ------------------------ |
| 插入     | 10  | 3    | 5     | 放置在 3                 |
| 插入     | 24  | 3    | 4     | 置换 10 → 移动到 5       |
| 插入     | 31  | 3    | 6     | 放置在 3                 |
| 查找     | 10  | 3, 5 | 找到  |                          |

边界情况

- 检测到循环 → 需要重新哈希
- 两个表都满 → 调整大小
- 必须限制重新定位尝试次数

#### 复杂度

- 时间：
  *   查找：O(1)
  *   插入：O(1) 摊还
- 空间：O(2n)

布谷鸟哈希维持着巢穴中的秩序，每个键都能找到一个家，否则表会学会重建它的世界。
### 216 罗宾汉哈希

罗宾汉哈希（Robin Hood hashing）是对开放寻址法的一种巧妙改进：当新键发生冲突时，它会比较自己"距离家（原始哈希槽）的远近"与当前占据者的距离。如果新键的"旅行距离"更远，它就*抢占*这个槽位，从而更均匀地重新分配探测距离，保持较低的方差。

#### 我们解决什么问题？

在线性探测中，运气不好的键可能需要移动很长的距离，而其他键则紧挨着它们的"家"。这导致了探测序列的不平衡，某些键的搜索路径很长，而另一些则很短。
罗宾汉哈希"劫富济贫"，通过交换键来帮助那些远离"家"的键，从而最小化最大探测距离。

目标：
通过交换键来均衡探测距离，使得没有键"落后"其他键太多。

#### 它是如何工作的？（通俗解释）

每个条目记录其探测距离 = 从其原始哈希槽出发的步数。
插入新键时：

1. 计算 `index = hash(key) % capacity`。
2. 如果槽位为空 → 插入。
3. 如果被占用 → 比较探测距离。

   * 如果新来者的距离 > 占据者的距离 → 交换它们。
   * 对被置换出的键继续执行插入过程。

示例步骤（容量 = 7）

| 步骤 | 键  | Hash(key) | 探测距离 | 操作                                                                   |
| ---- | --- | --------- | -------------- | ------------------------------------------------------------------------ |
| 1    | 10  | 3         | 0              | 放置在索引 3                                                               |
| 2    | 24  | 3         | 0              | 冲突 → 移动到索引 4 (距离=1)                                           |
| 3    | 31  | 3         | 0              | 冲突 → 距离=0 < 0? 否 → 移动 → 距离=1 < 1? 否 → 距离=2 → 放置在索引 5 |
| 4    | 17  | 3         | 0              | 冲突链 → 比较并在距离更远时交换                            |

这确保了所有键都保持在接近其原始索引的位置，对所有键的访问更加公平。

结果表：

| 索引 | 键  | 距离 |
| ----- | --- | ---- |
| 3     | 10  | 0    |
| 4     | 24  | 1    |
| 5     | 31  | 2    |

#### 简易代码实现

C

```c
#include <stdio.h>
#include <stdlib.h>

#define CAPACITY 7
#define EMPTY -1

typedef struct {
    int key;
    int dist; // 探测距离
} Slot;

typedef struct {
    Slot *table;
} HashTable;

int hash(int key) { return key % CAPACITY; }

HashTable* ht_create() {
    HashTable *ht = malloc(sizeof(HashTable));
    ht->table = malloc(sizeof(Slot) * CAPACITY);
    for (int i = 0; i < CAPACITY; i++) {
        ht->table[i].key = EMPTY;
        ht->table[i].dist = 0;
    }
    return ht;
}

void ht_insert(HashTable *ht, int key) {
    int idx = hash(key);
    int dist = 0;

    while (1) {
        if (ht->table[idx].key == EMPTY) {
            ht->table[idx].key = key;
            ht->table[idx].dist = dist;
            return;
        }

        if (dist > ht->table[idx].dist) {
            // 交换键
            int tmp_key = ht->table[idx].key;
            int tmp_dist = ht->table[idx].dist;
            ht->table[idx].key = key;
            ht->table[idx].dist = dist;
            key = tmp_key;
            dist = tmp_dist;
        }

        idx = (idx + 1) % CAPACITY;
        dist++;
        if (dist >= CAPACITY) {
            printf("表已满\n");
            return;
        }
    }
}

int ht_search(HashTable *ht, int key) {
    int idx = hash(key);
    int dist = 0;
    while (ht->table[idx].key != EMPTY && dist <= ht->table[idx].dist) {
        if (ht->table[idx].key == key) return idx;
        idx = (idx + 1) % CAPACITY;
        dist++;
    }
    return -1;
}

int main(void) {
    HashTable *ht = ht_create();
    ht_insert(ht, 10);
    ht_insert(ht, 24);
    ht_insert(ht, 31);
    for (int i = 0; i < CAPACITY; i++) {
        if (ht->table[i].key != EMPTY)
            printf("[%d] key=%d dist=%d\n", i, ht->table[i].key, ht->table[i].dist);
    }
    return 0;
}
```

Python

```python
class RobinHoodHash:
    def __init__(self, size=7):
        self.size = size
        self.table = [None] * size
        self.dist = [0] * size

    def _hash(self, key):
        return key % self.size

    def insert(self, key):
        idx = self._hash(key)
        d = 0
        while True:
            if self.table[idx] is None:
                self.table[idx] = key
                self.dist[idx] = d
                return
            # 如果新来者距离更远，则进行罗宾汉交换
            if d > self.dist[idx]:
                key, self.table[idx] = self.table[idx], key
                d, self.dist[idx] = self.dist[idx], d
            idx = (idx + 1) % self.size
            d += 1
            if d >= self.size:
                raise OverflowError("表已满")

    def search(self, key):
        idx = self._hash(key)
        d = 0
        while self.table[idx] is not None and d <= self.dist[idx]:
            if self.table[idx] == key:
                return idx
            idx = (idx + 1) % self.size
            d += 1
        return None

# 示例
ht = RobinHoodHash()
for k in [10, 24, 31]:
    ht.insert(k)
print(list(zip(range(ht.size), ht.table, ht.dist)))
print("搜索 24:", ht.search(24))
```

#### 为什么它重要

- 平衡所有键的访问时间
- 最小化探测长度的方差
- 在高负载下性能优于线性探测
- 优雅的公平性原则，旅行距离长的键获得优先权

#### 一个温和的证明（为什么它有效）

通过确保所有探测距离大致相等，最坏情况下的搜索成本 ≈ 平均搜索成本。
键永远不会被"困"在长簇的后面，并且当探测距离超过现有槽位的距离时，搜索会提前终止。

平均搜索成本 ≈ O(1 + α)，但比标准线性探测具有*更小的方差*。

#### 动手尝试

1.  以不同顺序插入键并比较探测距离。
2.  实现删除操作（标记为已删除并调整邻居位置）。
3.  跟踪负载增长时的平均探测距离。
4.  与标准线性探测比较公平性。

#### 测试用例

| 操作     | 键  | 原始位置 | 最终槽位 | 距离 | 说明             |
| --------- | --- | ---- | ---------- | ---- | ----------------- |
| insert    | 10  | 3    | 3          | 0    | 首次插入      |
| insert    | 24  | 3    | 4          | 1    | 冲突         |
| insert    | 31  | 3    | 5          | 2    | 进一步冲突 |
| search    | 24  | 3→4  | 找到      |      |                   |

边界情况

- 表已满 → 停止插入
- 必须限制距离以防止无限循环
- 删除操作需要重新平衡邻居

#### 复杂度

- 时间：平均 O(1)，最坏 O(n)
- 空间：O(n)

罗宾汉哈希为冲突带来了正义，没有键会离家太远而漂泊无依。
### 217 链式哈希表

链式哈希表是处理碰撞的经典解决方案，它不是将每个键都塞进数组，而是让每个桶持有一个链表（或称链），用于存放共享相同哈希索引的条目。

#### 我们解决什么问题？

使用开放寻址法时，碰撞会迫使你在数组中探查新的槽位。
链式法在外部解决碰撞，每个索引指向一个小的动态列表，因此多个键可以共享同一个槽位而不会造成拥挤。

目标：
使用链表（链）将碰撞的键存储在相同的哈希索引处，从而在平均情况下保持插入、搜索和删除操作的简单高效。

#### 它是如何工作的（通俗解释）？

每个数组索引存储一个指向键值对链表的指针。
插入时：

1.  计算索引 = `hash(key) % capacity`。
2.  遍历链表检查键是否存在。
3.  如果不存在，将新节点添加到链表前端（或后端）。

搜索和删除遵循相同的索引和链表操作。

示例（容量 = 5）

| 步骤 | 键    | Hash(key) | 索引 | 操作                 |
| ---- | ----- | --------- | ---- | -------------------- |
| 1    | "cat" | 2         | 2    | 放入 chain[2]        |
| 2    | "dog" | 4         | 4    | 放入 chain[4]        |
| 3    | "bat" | 2         | 2    | 追加到 chain[2]      |
| 4    | "ant" | 2         | 2    | 追加到 chain[2]      |

表结构

| 索引 | 链表                     |
| ---- | ------------------------ |
| 0    | -                        |
| 1    | -                        |
| 2    | "cat" → "bat" → "ant"    |
| 3    | -                        |
| 4    | "dog"                    |

#### 微型代码（简易版本）

C

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAPACITY 5

typedef struct Node {
    char *key;
    int value;
    struct Node *next;
} Node;

typedef struct {
    Node *buckets[CAPACITY];
} HashTable;

unsigned int hash(const char *key) {
    unsigned int h = 0;
    while (*key) h = h * 31 + *key++;
    return h % CAPACITY;
}

HashTable* ht_create() {
    HashTable *ht = malloc(sizeof(HashTable));
    for (int i = 0; i < CAPACITY; i++) ht->buckets[i] = NULL;
    return ht;
}

void ht_insert(HashTable *ht, const char *key, int value) {
    unsigned int idx = hash(key);
    Node *node = ht->buckets[idx];
    while (node) {
        if (strcmp(node->key, key) == 0) { node->value = value; return; }
        node = node->next;
    }
    Node *new_node = malloc(sizeof(Node));
    new_node->key = strdup(key);
    new_node->value = value;
    new_node->next = ht->buckets[idx];
    ht->buckets[idx] = new_node;
}

int ht_search(HashTable *ht, const char *key, int *out) {
    unsigned int idx = hash(key);
    Node *node = ht->buckets[idx];
    while (node) {
        if (strcmp(node->key, key) == 0) { *out = node->value; return 1; }
        node = node->next;
    }
    return 0;
}

void ht_delete(HashTable *ht, const char *key) {
    unsigned int idx = hash(key);
    Node curr = &ht->buckets[idx];
    while (*curr) {
        if (strcmp((*curr)->key, key) == 0) {
            Node *tmp = *curr;
            *curr = (*curr)->next;
            free(tmp->key);
            free(tmp);
            return;
        }
        curr = &(*curr)->next;
    }
}

int main(void) {
    HashTable *ht = ht_create();
    ht_insert(ht, "cat", 1);
    ht_insert(ht, "bat", 2);
    ht_insert(ht, "ant", 3);
    int val;
    if (ht_search(ht, "bat", &val))
        printf("bat: %d\n", val);
    ht_delete(ht, "bat");
    if (!ht_search(ht, "bat", &val))
        printf("bat deleted\n");
    return 0;
}
```

Python

```python
class ChainedHash:
    def __init__(self, size=5):
        self.size = size
        self.table = [[] for _ in range(size)]

    def _hash(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        idx = self._hash(key)
        for pair in self.table[idx]:
            if pair[0] == key:
                pair[1] = value
                return
        self.table[idx].append([key, value])

    def search(self, key):
        idx = self._hash(key)
        for k, v in self.table[idx]:
            if k == key:
                return v
        return None

    def delete(self, key):
        idx = self._hash(key)
        self.table[idx] = [p for p in self.table[idx] if p[0] != key]

# 示例
ht = ChainedHash()
ht.insert("cat", 1)
ht.insert("bat", 2)
ht.insert("ant", 3)
print(ht.table)
print("Search bat:", ht.search("bat"))
ht.delete("bat")
print(ht.table)
```

#### 为什么它重要

-   简单可靠的碰撞处理
-   负载因子可以超过 1（链表吸收溢出）
-   删除操作直接明了（只需移除节点）
-   即使在高负载下性能也稳定（如果哈希分布均匀）

#### 一个温和的证明（为什么它有效）

期望链表长度 = 负载因子 $\alpha = \frac{n}{m}$。
每个操作遍历单个链表，所以平均成本 = O(1 + α)。
均匀的哈希分布确保 α 保持较小 → 操作 ≈ O(1)。

#### 亲自尝试

1.  当平均链表长度增长时，实现动态调整大小。
2.  比较前插与后插策略。
3.  测量表填充时的平均搜索步数。
4.  用平衡树替换链表（针对高 α 情况）。

#### 测试用例

| 操作   | 键    | 索引 | 链表结果              | 备注         |
| ------ | ----- | ---- | --------------------- | ------------ |
| insert | "cat" | 2    | ["cat"]               |              |
| insert | "bat" | 2    | ["bat", "cat"]        | 碰撞         |
| insert | "ant" | 2    | ["ant", "bat", "cat"] | 链表增长     |
| search | "bat" | 2    | found                 |              |
| delete | "bat" | 2    | ["ant", "cat"]        | 已移除       |

边界情况

-   许多键具有相同索引 → 长链表
-   哈希函数不佳 → 分布不均
-   需要为指针/节点分配内存

#### 复杂度

-   时间：平均 O(1)，最坏 O(n)（所有键都在一个链表中）
-   空间：O(n + m)

链式哈希将碰撞转化为对话，如果一个桶满了，它只是将它们整齐地排列在一个列表中。
### 218 完美哈希

完美哈希是哈希表的理想场景，完全没有冲突。每个键都映射到唯一的槽位，因此查找、插入和删除操作在最坏情况下都只需要 O(1) 时间，而不仅仅是平均情况。

#### 我们要解决什么问题？

大多数哈希策略（线性探测、链地址法、布谷鸟哈希）都是在冲突发生后处理它们。
完美哈希通过为固定的键集合设计一个无冲突的哈希函数，从而完全消除冲突。

目标：
构造一个哈希函数 $h(k)$，使得所有键都映射到不同的索引。

#### 它是如何工作的（通俗解释）？

如果键集合是预先已知的（静态集合），我们可以精心选择或构建一个哈希函数，为每个键分配一个唯一的槽位。

主要有两种类型：

1.  完美哈希，无冲突。
2.  最小完美哈希，无冲突且表大小 = 键的数量。

简单示例（键 = {10, 24, 31, 17}, 容量 = 7）

让我们找一个函数：
$$
h(k) = (a \cdot k + b) \bmod 7
$$
我们可以寻找能产生唯一索引的系数 a 和 b：

| 键  | h(k) = (2k + 1) mod 7 | 索引        |
| --- | --------------------- | ----------- |
| 10  | (21) % 7 = 0          | 0           |
| 24  | (49) % 7 = 0          | ❌ 冲突     |
| 31  | (63) % 7 = 0          | ❌ 冲突     |

所以我们尝试另一对系数 (a=3, b=2)：

| 键  | (3k + 2) mod 7 | 索引        |
| --- | -------------- | ----------- |
| 10  | 5              | 5           |
| 24  | 4              | 4           |
| 31  | 4              | ❌ 冲突     |
| 17  | 6              | 6           |

最终，通过调整参数或使用两级构造，我们找到一个没有重复的映射。

#### 两级完美哈希

实用的完美哈希通常使用两级方案：

1.  顶层：将键哈希到桶中。
2.  第二层：每个桶获得自己的小型哈希表，并使用自己的完美哈希函数。

这确保了整体上零冲突，总空间约为 O(n)。

过程：

-   大小为 *b* 的每个桶获得一个大小为 *b²* 的二级表。
-   对该桶使用第二个哈希函数 $h_i$，以唯一地放置所有键。

#### 微型代码（简易版本）

Python（两级静态完美哈希）

```python
import random

class PerfectHash:
    def __init__(self, keys):
        self.n = len(keys)
        self.size = self.n
        self.buckets = [[] for _ in range(self.size)]
        self.secondary = [None] * self.size

        # 第一级哈希
        a, b = 3, 5  # 固定的小型哈希参数
        def h1(k): return (a * k + b) % self.size

        # 将键分配到桶中
        for k in keys:
            self.buckets[h1(k)].append(k)

        # 构建第二级表
        for i, bucket in enumerate(self.buckets):
            if not bucket:
                continue
            m = len(bucket)  2
            table = [None] * m
            found = False
            while not found:
                found = True
                a2, b2 = random.randint(1, m - 1), random.randint(0, m - 1)
                def h2(k): return (a2 * k + b2) % m
                table = [None] * m
                for k in bucket:
                    pos = h2(k)
                    if table[pos] is not None:
                        found = False
                        break
                    table[pos] = k
            self.secondary[i] = (table, a2, b2)

        self.h1 = h1

    def search(self, key):
        i = self.h1(key)
        table, a2, b2 = self.secondary[i]
        m = len(table)
        pos = (a2 * key + b2) % m
        return table[pos] == key

# 示例
keys = [10, 24, 31, 17]
ph = PerfectHash(keys)
print([len(b) for b in ph.buckets])
print("查找 24:", ph.search(24))
print("查找 11:", ph.search(11))
```

#### 为什么它重要

-   保证 O(1) 最坏情况查找
-   无聚集、无冲突、无链
-   适用于静态键集合（例如编译器中的保留关键字、路由表）
-   内存可预测，访问速度极快

#### 一个温和的证明（为什么它有效）

令 $n=|S|$ 为键的数量。
使用一个真正随机的哈希函数族映射到 $m$ 个桶中，两个不同键的冲突概率是 $1/m$。

选择 $m=n^2$。那么期望的冲突次数为
$$
\mathbb{E}[C]=\binom{n}{2}\cdot \frac{1}{m}
=\frac{n(n-1)}{2n^2}<\tfrac12.
$$
根据马尔可夫不等式，$\Pr[C\ge1]\le \mathbb{E}[C]<\tfrac12$，所以 $\Pr[C=0]>\tfrac12$。
因此，存在一个无冲突的哈希函数。在实践中，可以尝试随机种子直到找到一个 $C=0$ 的，或者使用确定性构造来获得完美哈希。

#### 亲自尝试

1.  为小型静态集合 {"if", "else", "for", "while"} 生成完美哈希。
2.  构建最小完美哈希（表大小 = n）。
3.  与标准字典比较查找时间。
4.  可视化第二级哈希表的大小。

#### 测试用例

| 操作       | 键              | 结果    | 备注                 |
| ---------- | --------------- | ------- | -------------------- |
| 构建       | [10,24,31,17]   | 成功    | 每个槽位唯一         |
| 查找       | 24              | True    | 找到                 |
| 查找       | 11              | False   | 不在表中             |
| 冲突       | 无              |         | 完美映射             |

边界情况

-   仅适用于静态集合（不能动态插入）
-   构建可能需要多次重哈希尝试
-   内存随着二级表的平方级增长而增加

#### 复杂度

-   构建：O(n²)（寻找无冲突的映射）
-   查找：O(1)
-   空间：O(n) 到 O(n²)，取决于方法

完美哈希就像为每把锁找到完美的钥匙，一次构建，瞬间开启，永不冲突。
### 219 一致性哈希

一致性哈希是一种为*分布式系统*而非单内存表设计的冲突处理策略。它确保当节点（服务器、缓存或分片）加入或离开时，只有一小部分键需要重新映射，这使其成为可扩展、容错架构的支柱。

#### 我们要解决什么问题？

在传统哈希（例如 `hash(key) % n`）中，当服务器数量 `n` 发生变化时，几乎每个键的位置都会改变。这对于缓存、数据库或负载均衡来说是灾难性的。

一致性哈希通过将键和服务器映射到同一个哈希空间来解决这个问题，并将键顺时针放置在离其最近的服务器附近，从而在系统变化时最大限度地减少重新分配。

目标：
在动态变化的服务器数量下，以最小的移动和良好的平衡性实现稳定的键分布。

#### 它是如何工作的（通俗解释）？

1.  想象一个哈希环，数字 0 到 (2^{m}-1) 排列成一个圆圈。
2.  每个节点（服务器）和键都被哈希到环上的一个位置。
3.  一个键被分配给它哈希位置顺时针方向的下一个服务器。
4.  当一个节点加入/离开时，只有其紧邻区域的键会移动。

示例（容量 = 2¹⁶）

| 项     | 哈希值 | 在环上的位置             | 所有者 |
| ------ | ------ | ------------------------ | ------ |
| 节点 A | 1000   | •                        |        |
| 节点 B | 4000   | •                        |        |
| 节点 C | 8000   | •                        |        |
| 键 K₁  | 1200   | → 节点 B                 |        |
| 键 K₂  | 8500   | → 节点 A（环绕）         |        |

如果节点 B 离开，只有其段（1000–4000）内的键会移动，其他所有键保持不变。

#### 改进负载均衡

为了防止分布不均，每个节点由多个虚拟节点（vnode）表示，每个虚拟节点都有自己的哈希值。
这使得键的分布平滑地扩展到所有节点。

示例：

-   节点 A → 哈希到 1000, 6000
-   节点 B → 哈希到 3000, 9000

键被分配给顺时针方向最近的虚拟节点。

#### 精简代码（简易版本）

Python（带虚拟节点的简单一致性哈希）

```python
import bisect
import hashlib

def hash_fn(key):
    return int(hashlib.md5(str(key).encode()).hexdigest(), 16)

class ConsistentHash:
    def __init__(self, nodes=None, vnodes=3):
        self.ring = []
        self.map = {}
        self.vnodes = vnodes
        if nodes:
            for n in nodes:
                self.add_node(n)

    def add_node(self, node):
        for i in range(self.vnodes):
            h = hash_fn(f"{node}-{i}")
            self.map[h] = node
            bisect.insort(self.ring, h)

    def remove_node(self, node):
        for i in range(self.vnodes):
            h = hash_fn(f"{node}-{i}")
            self.ring.remove(h)
            del self.map[h]

    def get_node(self, key):
        if not self.ring:
            return None
        h = hash_fn(key)
        idx = bisect.bisect(self.ring, h) % len(self.ring)
        return self.map[self.ring[idx]]

# 示例
ch = ConsistentHash(["A", "B", "C"], vnodes=2)
print("Key 42 ->", ch.get_node(42))
ch.remove_node("B")
print("After removing B, Key 42 ->", ch.get_node(42))
```

#### 为什么它很重要

-   当节点变化时，最小化键的重新映射（大约移动 1/n 的键）
-   为缓存、数据库分片和分布式存储实现弹性扩展
-   用于 Amazon Dynamo、Cassandra、Riak 和 memcached 客户端等系统
-   使用虚拟节点平衡负载
-   将哈希函数与节点数量解耦

#### 一个温和的证明（为什么它有效）

设 $N$ 为节点数，$K$ 为键数。  
每个节点负责环的一部分，比例为 $\frac{1}{N}$。

当一个节点离开时，只有其段内的键会移动，大约 $K/N$ 个键。  
因此，预期的重新映射比例是 $\frac{1}{N}$。

添加虚拟节点（vnode）增加了均匀性。  
每个物理节点有 $V$ 个虚拟节点时，每个节点负载比例的方差大约按 $\approx \frac{1}{V}$ 缩放。

#### 亲自尝试

1.  添加和移除节点，跟踪有多少键移动。
2.  尝试不同的虚拟节点数量（1, 10, 100）。
3.  可视化哈希环，标记节点和键的位置。
4.  模拟缓存：分配 1000 个键，移除一个节点，计算移动的键数。

#### 测试用例

| 操作       | 节点      | 键          | 结果                 | 备注             |
| ---------- | --------- | ----------- | -------------------- | ---------------- |
| add_nodes  | A, B, C   | [1..1000]   | 均匀分布             |                  |
| remove_node| B         | [1..1000]   | 约 1/3 的键移动      | 稳定性检查       |
| add_node   | D         | [1..1000]   | 约 1/4 的键重新映射  |                  |
| lookup     | 42        | -> 节点 C   | 一致的映射           |                  |

边界情况

-   空环 → 返回 None
-   重复节点 → 通过唯一的虚拟节点 ID 处理
-   没有虚拟节点 → 负载不均

#### 复杂度

-   查找：$O(\log n)$（在环中进行二分查找）
-   插入/删除节点：$O(v \log n)$
-   空间：$O(n \times v)$

一致性哈希在风暴中维持秩序 —— 服务器可能来来去去，但大多数键都留在它们所属的位置。
### 220 动态重哈希

动态重哈希是哈希表在数据增长或收缩时优雅适应的方法。它不再受限于固定大小的数组，而是*自行调整大小*，重建其布局，从而保持负载平衡，查找操作依然快速。

#### 我们要解决什么问题？

当哈希表填满时，冲突变得频繁，性能会下降到 O(n)。
我们需要一种机制，通过自动调整大小和重哈希，来维持一个较低的负载因子（元素数量与容量的比值）。

目标：
检测负载因子何时超过阈值，分配一个更大的数组，并高效地将所有键重新哈希到它们的新位置。

#### 它如何工作（通俗解释）？

1.  监控负载因子

    $$
    \alpha = \frac{n}{m}
    $$
    其中 $n$ 是元素数量，$m$ 是表的大小。

2.  触发重哈希

    - 如果 $\alpha > 0.75$，扩展表（例如，容量翻倍）。
    - 如果 $\alpha < 0.25$，收缩它（可选）。

3.  重建

    - 创建一个具有更新后容量的新表。
    - 使用新的哈希函数（模新容量）重新插入每个键。

示例步骤

| 步骤 | 容量 | 项目数 | 负载因子 | 操作          |
| ---- | ---- | ------ | -------- | ------------- |
| 1    | 4    | 2      | 0.5      | 正常          |
| 2    | 4    | 3      | 0.75     | 正常          |
| 3    | 4    | 4      | 1.0      | 调整大小为 8  |
| 4    | 8    | 4      | 0.5      | 已重哈希      |

每个键都会获得新位置，因为 `hash(key) % new_capacity` 发生了变化。

#### 增量重哈希

为了避免一次性重哈希所有键（代价高昂的峰值），增量重哈希将工作分摊到多个操作中：

- 同时维护旧表和新表。
- 每次插入或搜索时重哈希少量条目，直到旧表为空。

即使在调整大小期间，这也能保持摊还 O(1) 的性能。

#### 微型代码（简易版本）

C 语言（简单翻倍重哈希）

```c
#include <stdio.h>
#include <stdlib.h>

#define INIT_CAP 4

typedef struct {
    int *keys;
    int size;
    int count;
} HashTable;

int hash(int key, int size) { return key % size; }

HashTable* ht_create(int size) {
    HashTable *ht = malloc(sizeof(HashTable));
    ht->keys = malloc(sizeof(int) * size);
    for (int i = 0; i < size; i++) ht->keys[i] = -1;
    ht->size = size;
    ht->count = 0;
    return ht;
}

void ht_resize(HashTable *ht, int new_size) {
    printf("从 %d 调整大小到 %d\n", ht->size, new_size);
    int *old_keys = ht->keys;
    int old_size = ht->size;

    ht->keys = malloc(sizeof(int) * new_size);
    for (int i = 0; i < new_size; i++) ht->keys[i] = -1;

    ht->size = new_size;
    ht->count = 0;
    for (int i = 0; i < old_size; i++) {
        if (old_keys[i] != -1) {
            int key = old_keys[i];
            int idx = hash(key, new_size);
            while (ht->keys[idx] != -1) idx = (idx + 1) % new_size;
            ht->keys[idx] = key;
            ht->count++;
        }
    }
    free(old_keys);
}

void ht_insert(HashTable *ht, int key) {
    float load = (float)ht->count / ht->size;
    if (load > 0.75) ht_resize(ht, ht->size * 2);

    int idx = hash(key, ht->size);
    while (ht->keys[idx] != -1) idx = (idx + 1) % ht->size;
    ht->keys[idx] = key;
    ht->count++;
}

int main(void) {
    HashTable *ht = ht_create(INIT_CAP);
    for (int i = 0; i < 10; i++) ht_insert(ht, i * 3);
    for (int i = 0; i < ht->size; i++)
        printf("[%d] = %d\n", i, ht->keys[i]);
    return 0;
}
```

Python

```python
class DynamicHash:
    def __init__(self, cap=4):
        self.cap = cap
        self.size = 0
        self.table = [None] * cap

    def _hash(self, key):
        return hash(key) % self.cap

    def _rehash(self, new_cap):
        old_table = self.table
        self.table = [None] * new_cap
        self.cap = new_cap
        self.size = 0
        for key in old_table:
            if key is not None:
                self.insert(key)

    def insert(self, key):
        if self.size / self.cap > 0.75:
            self._rehash(self.cap * 2)
        idx = self._hash(key)
        while self.table[idx] is not None:
            idx = (idx + 1) % self.cap
        self.table[idx] = key
        self.size += 1

# 示例
ht = DynamicHash()
for k in [10, 24, 31, 17, 19, 42, 56, 77]:
    ht.insert(k)
print(ht.table)
```

#### 为什么它很重要

-   保持负载因子稳定，以实现 O(1) 操作
-   防止开放寻址中的聚集现象
-   支持无限制增长
-   是动态字典、映射、缓存的基础

#### 一个温和的证明（为什么它有效）

如果每次重哈希都使容量翻倍，那么 N 次插入的总成本 = O(N)。
每个元素被移动 O(1) 次（每次翻倍移动一次），因此每次插入的摊还成本是 O(1)。

增量重哈希进一步确保没有任何单一操作是昂贵的，工作被均匀分摊。

#### 动手试试

1.  添加打印输出，观察每次插入时的负载因子。
2.  当负载因子 < 0.25 时实现收缩。
3.  使用两个表实现增量重哈希。
4.  比较容量翻倍增长与质数容量增长。

#### 测试用例

| 步骤 | 容量 | 项目数 | 负载因子 | 操作       |
| ---- | ---- | ------ | -------- | ---------- |
| 1    | 4    | 2      | 0.5      | 无         |
| 2    | 4    | 3      | 0.75     | 正常       |
| 3    | 4    | 4      | 1.0      | 调整大小   |
| 4    | 8    | 4      | 0.5      | 已重哈希   |

边界情况

-   重哈希必须正确处理已删除的槽位
-   避免过于频繁地调整大小（滞后性）
-   在调整大小期间保持哈希函数的一致性

#### 复杂度

-   平均插入/搜索/删除：O(1)
-   摊还插入：O(1)
-   最坏情况调整大小：O(n)

动态重哈希是哈希表的心跳，它在满时扩展，在空闲时收缩，始终确保操作平稳流畅。

## 第 23 章 堆
### 221 二叉堆插入

二叉堆是一种存储在数组中的完全二叉树，它维护堆的性质：每个父节点比其子节点小（最小堆）或大（最大堆）。
插入操作通过将新元素*上浮*直到找到正确位置，来保持堆的有序性。

#### 我们要解决什么问题？

我们需要一种数据结构，能够高效地访问最小（或最大）元素，同时支持快速插入。

二叉堆提供：

- `O(1)` 访问最小/最大值
- `O(log n)` 插入和删除
- `O(n)` 初始建堆时间

目标：
插入一个新元素，同时保持堆的性质和完全性。

#### 它是如何工作的（通俗解释）？

堆以数组形式存储，表示一棵完全二叉树。
索引为 `i` 的每个节点具有：

- 父节点：`(i - 1) / 2`
- 左子节点：`2i + 1`
- 右子节点：`2i + 2`

插入步骤（最小堆）

1.  将新元素追加到末尾（最底层，最右侧）。
2.  将其与其父节点比较。
3.  如果更小（最小堆）或更大（最大堆），则交换。
4.  重复直到堆的性质恢复。

示例（最小堆）

插入序列：[10, 24, 5, 31]

| 步骤       | 数组              | 操作                               |
| ---------- | ----------------- | ---------------------------------- |
| 初始       | [ ]               | 空                                 |
| 插入 10    | [10]              | 只有根节点                         |
| 插入 24    | [10, 24]          | 24 > 10，不交换                    |
| 插入 5     | [10, 24, 5]       | 5 < 10 → 交换 → [5, 24, 10]        |
| 插入 31    | [5, 24, 10, 31]   | 31 > 24，不交换                    |

最终堆：[5, 24, 10, 31]

树形视图：

```
        5
      /   \
    24     10
   /
 31
```

#### 微型代码（简易版本）

C

```c
#include <stdio.h>
#define MAX 100

typedef struct {
    int arr[MAX];
    int size;
} MinHeap;

void swap(int *a, int *b) {
    int tmp = *a; *a = *b; *b = tmp;
}

void heap_insert(MinHeap *h, int val) {
    int i = h->size++;
    h->arr[i] = val;

    // 上浮
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (h->arr[i] >= h->arr[parent]) break;
        swap(&h->arr[i], &h->arr[parent]);
        i = parent;
    }
}

void heap_print(MinHeap *h) {
    for (int i = 0; i < h->size; i++) printf("%d ", h->arr[i]);
    printf("\n");
}

int main(void) {
    MinHeap h = {.size = 0};
    int vals[] = {10, 24, 5, 31};
    for (int i = 0; i < 4; i++) heap_insert(&h, vals[i]);
    heap_print(&h);
}
```

Python

```python
class MinHeap:
    def __init__(self):
        self.arr = []

    def _parent(self, i): return (i - 1) // 2

    def insert(self, val):
        self.arr.append(val)
        i = len(self.arr) - 1
        while i > 0:
            p = self._parent(i)
            if self.arr[i] >= self.arr[p]:
                break
            self.arr[i], self.arr[p] = self.arr[p], self.arr[i]
            i = p

    def __repr__(self):
        return str(self.arr)

# 示例
h = MinHeap()
for x in [10, 24, 5, 31]:
    h.insert(x)
print(h)
```

#### 为什么它重要

- 优先级队列的基础
- Dijkstra 最短路径算法、Prim 最小生成树算法和调度器的核心
- 支持高效的 `insert`、`extract-min/max` 和 `peek` 操作
- 用于堆排序和事件驱动模拟

#### 温和的证明（为什么它有效）

每次插入最多上浮堆的高度次 = $\log_2 n$。
由于堆始终保持完全性，结构是平衡的，确保了操作是对数级的。

堆的性质得以保持，因为每次交换都确保了父节点 ≤ 子节点（最小堆）。

#### 亲自尝试

1.  按降序插入元素 → 观察上浮过程。
2.  修改比较逻辑以实现最大堆。
3.  每次插入后逐层打印树。
4.  实现 `extract_min()` 以移除根节点并恢复堆。

#### 测试用例

| 操作     | 输入              | 输出              | 备注                 |
| -------- | ----------------- | ----------------- | -------------------- |
| 插入     | [10, 24, 5, 31]   | [5, 24, 10, 31]   | 最小堆性质           |
| 插入     | [3, 2, 1]         | [1, 3, 2]         | 交换链               |
| 插入     | [10]              | [10]              | 单个元素             |
| 插入     | []                | [x]               | 从空开始             |

边界情况

- 数组已满（静态堆）→ 需要调整大小
- 负数值 → 处理方式相同
- 重复键值 → 顺序得以保持

#### 复杂度

- 插入：O(log n)
- 搜索：O(n)（除了堆性质外，未排序）
- 空间：O(n)

二叉堆插入是优先级队列的心跳，每个元素都一步步地、通过一次温和的交换，攀升到它应有的位置。
### 222 二叉堆删除

从二叉堆中删除意味着移除根元素（最小堆中的最小值或最大堆中的最大值），同时保持堆性质和完全二叉树结构不变。
为此，我们将根元素与最后一个元素交换，移除最后一个元素，然后将新的根元素向下冒泡（下沉），直到堆再次有效。

#### 我们要解决什么问题？

我们想从堆中快速移除最高优先级的元素（最小值或最大值），而不破坏其结构。

在最小堆中，最小值始终位于索引 0 处。
在最大堆中，最大值位于索引 0 处。

目标：
高效地移除根元素并在 O(log n) 时间内恢复堆序。

#### 它是如何工作的（通俗解释）？

1.  将根元素（索引 0）与最后一个元素交换。
2.  移除最后一个元素（现在根元素的值已移除）。
3.  从根开始向下堆化（下沉）：
    *   与子节点比较。
    *   与较小的（最小堆）或较大的（最大堆）子节点交换。
    *   重复直到堆性质恢复。

示例（最小堆）

初始：`[5, 24, 10, 31]`
移除最小值（5）：

| 步骤 | 操作                                             | 数组            |              |
| ---- | ------------------------------------------------ | --------------- | ------------ |
| 1    | 交换根 (5) 与最后一个元素 (31)                   | [31, 24, 10, 5] |              |
| 2    | 移除最后一个元素                                 | [31, 24, 10]    |              |
| 3    | 比较 31 与子节点 (24, 10) → 最小值 = 10          | 交换            | [10, 24, 31] |
| 4    | 停止 (31 > 无子节点)                             | [10, 24, 31]    |              |

结果：`[10, 24, 31]`，仍然是一个有效的最小堆。

#### 微型代码（简易版本）

C

```c
#include <stdio.h>
#define MAX 100

typedef struct {
    int arr[MAX];
    int size;
} MinHeap;

void swap(int *a, int *b) {
    int tmp = *a; *a = *b; *b = tmp;
}

void heapify_down(MinHeap *h, int i) {
    int smallest = i;
    int left = 2*i + 1;
    int right = 2*i + 2;

    if (left < h->size && h->arr[left] < h->arr[smallest])
        smallest = left;
    if (right < h->size && h->arr[right] < h->arr[smallest])
        smallest = right;

    if (smallest != i) {
        swap(&h->arr[i], &h->arr[smallest]);
        heapify_down(h, smallest);
    }
}

int heap_delete_min(MinHeap *h) {
    if (h->size == 0) return -1;
    int root = h->arr[0];
    h->arr[0] = h->arr[h->size - 1];
    h->size--;
    heapify_down(h, 0);
    return root;
}

int main(void) {
    MinHeap h = {.arr = {5, 24, 10, 31}, .size = 4};
    int val = heap_delete_min(&h);
    printf("Deleted: %d\n", val);
    for (int i = 0; i < h.size; i++) printf("%d ", h.arr[i]);
    printf("\n");
}
```

Python

```python
class MinHeap:
    def __init__(self):
        self.arr = []

    def _parent(self, i): return (i - 1) // 2
    def _left(self, i): return 2 * i + 1
    def _right(self, i): return 2 * i + 2

    def insert(self, val):
        self.arr.append(val)
        i = len(self.arr) - 1
        while i > 0 and self.arr[i] < self.arr[self._parent(i)]:
            p = self._parent(i)
            self.arr[i], self.arr[p] = self.arr[p], self.arr[i]
            i = p

    def _heapify_down(self, i):
        smallest = i
        left, right = self._left(i), self._right(i)
        n = len(self.arr)
        if left < n and self.arr[left] < self.arr[smallest]:
            smallest = left
        if right < n and self.arr[right] < self.arr[smallest]:
            smallest = right
        if smallest != i:
            self.arr[i], self.arr[smallest] = self.arr[smallest], self.arr[i]
            self._heapify_down(smallest)

    def delete_min(self):
        if not self.arr:
            return None
        root = self.arr[0]
        last = self.arr.pop()
        if self.arr:
            self.arr[0] = last
            self._heapify_down(0)
        return root

# 示例
h = MinHeap()
for x in [5, 24, 10, 31]:
    h.insert(x)
print("Before:", h.arr)
print("Deleted:", h.delete_min())
print("After:", h.arr)
```

#### 为什么它很重要

-   优先队列中的关键操作
-   Dijkstra（迪杰斯特拉）和 Prim（普里姆）算法的核心
-   堆排序的基础（重复执行 delete-min）
-   确保高效提取极值元素

#### 一个温和的证明（为什么它有效）

每次删除操作：

-   常数时间移除根元素
-   对数时间的向下堆化（树的高度 = log n）
    → 总成本：O(log n)

堆性质得以保持，因为每次交换都将一个较大的（最小堆）或较小的（最大堆）元素向下移动到子节点，确保了每一步的局部有序性。

#### 亲自尝试

1.  重复执行删除操作来对数组排序（堆排序）。
2.  尝试最大堆删除（反转比较逻辑）。
3.  可视化每次删除后的交换过程。
4.  在升序/降序输入序列上进行测试。

#### 测试用例

| 输入           | 操作     | 输出 | 删除后堆状态 | 备注       |
| -------------- | -------- | ---- | ------------ | ---------- |
| [5,24,10,31]   | delete   | 5    | [10,24,31]   | 有效       |
| [1,3,2]        | delete   | 1    | [2,3]        | 正常       |
| [10]           | delete   | 10   | []           | 空堆       |
| []             | delete   | None | []           | 安全处理   |

边界情况

-   空堆 → 返回哨兵值
-   单个元素 → 清空堆
-   重复元素自然处理

#### 复杂度

-   删除根元素：O(log n)
-   空间：O(n)

从堆中删除就像从整齐的牌堆顶部取走一张牌，替换它，将其向下筛选，然后恢复平衡。
### 223 建堆（堆化）

堆化（建堆）是指从无序数组在 O(n) 时间内构建一个有效二叉堆的过程。我们不是逐个插入元素，而是*原地*重组数组，使得每个父节点都满足堆的性质。

#### 我们要解决什么问题？

如果我们将每个元素单独插入到一个空堆中，总时间是 O(n log n)。
但我们可以做得更好。
通过自底向上地进行堆化，我们可以在 O(n) 时间内构建整个堆，这对于堆排序和高效初始化优先队列至关重要。

目标：
快速将任意数组转换为有效的最小堆或最大堆。

#### 它是如何工作的？（通俗解释）

存储在数组中的堆表示一棵完全二叉树：

- 对于节点 `i`，其子节点是 `2i + 1` 和 `2i + 2`

构建堆的步骤：

1.  从最后一个非叶子节点开始 = `(n / 2) - 1`。
2.  应用向下堆化（下沉）操作，确保以 `i` 为根的子树满足堆性质。
3.  向上移动到根节点，重复此过程。

每个子树都成为一个有效的堆，完成后，整个数组就是一个堆。

示例（最小堆）

初始数组：`[31, 10, 24, 5, 12, 7]`

| 步骤 | i  | 子树           | 操作               | 结果                     |
| ---- | -- | -------------- | ------------------ | ------------------------ |
| 初始 | -  | 整个数组       | -                  | [31, 10, 24, 5, 12, 7]   |
| 1    | 2  | (24, 7)        | 24 > 7 → 交换      | [31, 10, 7, 5, 12, 24]   |
| 2    | 1  | (10, 5, 12)    | 10 > 5 → 交换      | [31, 5, 7, 10, 12, 24]   |
| 3    | 0  | (31, 5, 7)     | 31 > 5 → 交换      | [5, 31, 7, 10, 12, 24]   |
| 4    | 1  | (31, 10, 12)   | 31 > 10 → 交换     | [5, 10, 7, 31, 12, 24]   |

最终堆：[5, 10, 7, 31, 12, 24]

#### 精简代码（简易版本）

C 语言（自底向上建堆）

```c
#include <stdio.h>

#define MAX 100

typedef struct {
    int arr[MAX];
    int size;
} MinHeap;

void swap(int *a, int *b) {
    int tmp = *a; *a = *b; *b = tmp;
}

void heapify_down(MinHeap *h, int i) {
    int smallest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < h->size && h->arr[left] < h->arr[smallest])
        smallest = left;
    if (right < h->size && h->arr[right] < h->arr[smallest])
        smallest = right;
    if (smallest != i) {
        swap(&h->arr[i], &h->arr[smallest]);
        heapify_down(h, smallest);
    }
}

void build_heap(MinHeap *h) {
    for (int i = h->size / 2 - 1; i >= 0; i--)
        heapify_down(h, i);
}

int main(void) {
    MinHeap h = {.arr = {31, 10, 24, 5, 12, 7}, .size = 6};
    build_heap(&h);
    for (int i = 0; i < h.size; i++) printf("%d ", h.arr[i]);
    printf("\n");
}
```

Python

```python
class MinHeap:
    def __init__(self, arr):
        self.arr = arr
        self.size = len(arr)
        self.build_heap()

    def _left(self, i): return 2 * i + 1
    def _right(self, i): return 2 * i + 2

    def _heapify_down(self, i):
        smallest = i
        l, r = self._left(i), self._right(i)
        if l < self.size and self.arr[l] < self.arr[smallest]:
            smallest = l
        if r < self.size and self.arr[r] < self.arr[smallest]:
            smallest = r
        if smallest != i:
            self.arr[i], self.arr[smallest] = self.arr[smallest], self.arr[i]
            self._heapify_down(smallest)

    def build_heap(self):
        for i in range(self.size // 2 - 1, -1, -1):
            self._heapify_down(i)

# 示例
arr = [31, 10, 24, 5, 12, 7]
h = MinHeap(arr)
print(h.arr)
```

#### 为什么它很重要

-   在 O(n) 时间内构建有效堆（而不是 O(n log n)）
-   用于堆排序的初始化
-   高效地从批量数据构建优先队列
-   自动保证平衡的树结构

#### 一个温和的证明（为什么它有效）

深度为 `d` 的每个节点需要 O(高度) = O(log(n/d)) 的工作量。
较低深度有更多节点（每个节点工作量较少），顶部节点较少（每个节点工作量较多）。
对各层求和得到 O(n) 的总时间，而不是 O(n log n)。

因此，自底向上的堆化在渐进意义上是最优的。

#### 动手尝试

1.  使用不同大小的数组、随机顺序运行。
2.  比较与逐个插入的时间。
3.  翻转比较操作以实现最大堆。
4.  将交换过程可视化为树。

#### 测试用例

| 输入                | 输出                | 说明             |
| ------------------- | ------------------- | ---------------- |
| [31,10,24,5,12,7]   | [5,10,7,31,12,24]   | 有效最小堆       |
| [3,2,1]             | [1,2,3]             | 已堆化           |
| [10]                | [10]                | 单个元素         |
| []                  | []                  | 空数组安全处理   |

边界情况

-   已经是堆 → 无变化
-   逆序输入 → 多次交换
-   正确处理重复元素

#### 复杂度

-   时间复杂度：O(n)
-   空间复杂度：O(1)（原地操作）

堆化是一位安静的工匠，通过自底向上的温和交换，将混乱塑造成秩序。
### 224 堆排序

堆排序是一种经典的基于比较的排序算法，它使用二叉堆来组织数据并按顺序提取元素。通过构建一个最大堆（或最小堆）并反复移除根节点，我们可以在 O(n log n) 时间内实现完全排序的数组，且无需额外空间。

#### 我们要解决什么问题？

我们想要一个快速、原地排序的算法，它：

- 不需要递归（像归并排序那样）
- 具有可预测的 O(n log n) 行为
- 避免最坏情况的二次时间（像快速排序那样）

目标：
利用堆的结构来高效地反复选择下一个最大（或最小）元素。

#### 它是如何工作的（通俗解释）？

1.  从未排序的数组构建一个最大堆。
2.  重复直到堆为空：
    *   将根节点（最大元素）与最后一个元素交换。
    *   将堆的大小减一。
    *   对新的根节点进行向下堆化以恢复堆性质。

数组将按升序排序（对于最大堆）。

示例（升序排序）

起始：`[5, 31, 10, 24, 7]`

| 步骤 | 操作               | 数组              |
| ---- | ------------------ | ----------------- |
| 1    | 构建最大堆         | [31, 24, 10, 5, 7] |
| 2    | 交换 31 ↔ 7，堆化  | [24, 7, 10, 5, 31] |
| 3    | 交换 24 ↔ 5，堆化  | [10, 7, 5, 24, 31] |
| 4    | 交换 10 ↔ 5，堆化  | [5, 7, 10, 24, 31] |
| 5    | 已排序             | [5, 7, 10, 24, 31] |

#### 精简代码（简易版本）

C 语言（原地堆排序）

```c
#include <stdio.h>

void swap(int *a, int *b) {
    int tmp = *a; *a = *b; *b = tmp;
}

void heapify(int arr[], int n, int i) {
    int largest = i;
    int left = 2*i + 1;
    int right = 2*i + 2;

    if (left < n && arr[left] > arr[largest])
        largest = left;
    if (right < n && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        swap(&arr[i], &arr[largest]);
        heapify(arr, n, largest);
    }
}

void heap_sort(int arr[], int n) {
    // 构建最大堆
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    // 提取元素
    for (int i = n - 1; i > 0; i--) {
        swap(&arr[0], &arr[i]);
        heapify(arr, i, 0);
    }
}

int main(void) {
    int arr[] = {5, 31, 10, 24, 7};
    int n = sizeof(arr)/sizeof(arr[0]);
    heap_sort(arr, n);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}
```

Python

```python
def heapify(arr, n, i):
    largest = i
    l, r = 2*i + 1, 2*i + 2

    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    # 构建最大堆
    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i)
    # 提取
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

# 示例
arr = [5, 31, 10, 24, 7]
heap_sort(arr)
print(arr)
```

#### 为什么它重要

- 保证 O(n log n) 运行时间（最坏情况安全）
- 原地排序，无需额外数组
- 是展示堆结构运作的绝佳教学示例
- 用于空间限制严格的系统

#### 一个温和的证明（为什么它有效）

1.  构建堆 = O(n)（自底向上堆化）。
2.  每次提取 = O(log n)，重复 n 次 → O(n log n)。
3.  堆性质确保根节点始终持有最大元素。

因此，经过反复的根节点提取后，最终数组是排序好的。

#### 亲自尝试

1.  切换比较方式以实现最小堆排序（降序）。
2.  每次交换后打印数组以观察过程。
3.  与快速排序和归并排序进行比较。
4.  测试大型随机数组。

#### 测试用例

| 输入          | 输出         | 备注         |
| ------------- | ------------ | ------------ |
| [5,31,10,24,7] | [5,7,10,24,31] | 升序         |
| [10,9,8,7]     | [7,8,9,10]     | 反向输入     |
| [3]            | [3]            | 单个元素     |
| []             | []             | 空数组       |

边界情况

- 重复值处理良好
- 已排序的输入仍为 O(n log n)
- 稳定？ ❌（不保持顺序）

#### 复杂度

- 时间：O(n log n)
- 空间：O(1)
- 稳定：否

堆排序是登山者的算法，一次一步地将最大的元素拉到顶部，直到秩序从顶峰蔓延到底部。
### 225 最小堆实现

最小堆是一种二叉树，其中每个父节点都小于或等于其子节点。它以紧凑的方式存储在数组中，保证能以 O(1) 的时间访问最小元素，并以 O(log n) 的时间进行插入和删除，非常适合优先级队列和调度系统。

#### 我们要解决什么问题？

我们需要一种数据结构，它能够：

- 快速给出最小元素
- 支持高效的插入和删除操作
- 动态地维持顺序

最小堆实现了这种平衡，它始终将最小元素保持在根节点，同时保持结构紧凑且完全。

#### 它是如何工作的（通俗解释）？

二叉最小堆以数组形式存储，其中：

- `parent(i) = (i - 1) / 2`
- `left(i) = 2i + 1`
- `right(i) = 2i + 2`

操作：

1. 插入(x)：

   * 将 `x` 添加到末尾。
   * 当 `x < parent(x)` 时，进行"上浮"操作。

2. 提取最小值()：

   * 移除根节点。
   * 将最后一个元素移到根节点位置。
   * 当父节点大于最小的子节点时，进行"下沉"操作。

3. 查看()：

   * 返回 `arr[0]`。

示例

初始：`[ ]`
插入序列：10, 24, 5, 31, 7

| 步骤 | 操作       | 数组                       |             |
| ---- | ---------- | -------------------------- | ----------- |
| 1    | 插入 10    | [10]                       |             |
| 2    | 插入 24    | [10, 24]                   |             |
| 3    | 插入 5     | 上浮 → 与 10 交换          | [5, 24, 10] |
| 4    | 插入 31    | [5, 24, 10, 31]            |             |
| 5    | 插入 7     | [5, 7, 10, 31, 24]         |             |

提取最小值：

- 交换根节点与最后一个元素 (5 ↔ 24) → `[24, 7, 10, 31, 5]`
- 移除最后一个元素 → `[24, 7, 10, 31]`
- 下沉 → `[7, 24, 10, 31]`

#### 精简代码（简易版本）

C

```c
#include <stdio.h>
#define MAX 100

typedef struct {
    int arr[MAX];
    int size;
} MinHeap;

void swap(int *a, int *b) {
    int tmp = *a; *a = *b; *b = tmp;
}

void heapify_up(MinHeap *h, int i) {
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (h->arr[i] >= h->arr[parent]) break;
        swap(&h->arr[i], &h->arr[parent]);
        i = parent;
    }
}

void heapify_down(MinHeap *h, int i) {
    int smallest = i;
    int left = 2*i + 1;
    int right = 2*i + 2;

    if (left < h->size && h->arr[left] < h->arr[smallest]) smallest = left;
    if (right < h->size && h->arr[right] < h->arr[smallest]) smallest = right;

    if (smallest != i) {
        swap(&h->arr[i], &h->arr[smallest]);
        heapify_down(h, smallest);
    }
}

void insert(MinHeap *h, int val) {
    h->arr[h->size] = val;
    heapify_up(h, h->size);
    h->size++;
}

int extract_min(MinHeap *h) {
    if (h->size == 0) return -1;
    int root = h->arr[0];
    h->arr[0] = h->arr[--h->size];
    heapify_down(h, 0);
    return root;
}

int peek(MinHeap *h) {
    return h->size > 0 ? h->arr[0] : -1;
}

int main(void) {
    MinHeap h = {.size = 0};
    int vals[] = {10, 24, 5, 31, 7};
    for (int i = 0; i < 5; i++) insert(&h, vals[i]);
    printf("最小值: %d\n", peek(&h));
    printf("已提取: %d\n", extract_min(&h));
    for (int i = 0; i < h.size; i++) printf("%d ", h.arr[i]);
    printf("\n");
}
```

Python

```python
class MinHeap:
    def __init__(self):
        self.arr = []

    def _parent(self, i): return (i - 1) // 2
    def _left(self, i): return 2 * i + 1
    def _right(self, i): return 2 * i + 2

    def insert(self, val):
        self.arr.append(val)
        i = len(self.arr) - 1
        while i > 0 and self.arr[i] < self.arr[self._parent(i)]:
            p = self._parent(i)
            self.arr[i], self.arr[p] = self.arr[p], self.arr[i]
            i = p

    def extract_min(self):
        if not self.arr:
            return None
        root = self.arr[0]
        last = self.arr.pop()
        if self.arr:
            self.arr[0] = last
            self._heapify_down(0)
        return root

    def _heapify_down(self, i):
        smallest = i
        l, r = self._left(i), self._right(i)
        if l < len(self.arr) and self.arr[l] < self.arr[smallest]:
            smallest = l
        if r < len(self.arr) and self.arr[r] < self.arr[smallest]:
            smallest = r
        if smallest != i:
            self.arr[i], self.arr[smallest] = self.arr[smallest], self.arr[i]
            self._heapify_down(smallest)

    def peek(self):
        return self.arr[0] if self.arr else None

# 示例
h = MinHeap()
for x in [10, 24, 5, 31, 7]:
    h.insert(x)
print("堆:", h.arr)
print("最小值:", h.peek())
print("已提取:", h.extract_min())
print("之后:", h.arr)
```

#### 为什么它很重要

- 是优先级队列、Dijkstra（迪杰斯特拉）、Prim（普里姆）、A* 等算法的基础
- 始终能以 O(1) 的时间给出最小元素
- 基于数组的紧凑结构（无需指针）
- 非常适合动态、有序的集合

#### 一个温和的证明（为什么它有效）

每次插入或删除只影响单一路径（高度 = log n）。
由于每次交换都改善了局部顺序，堆的性质在 O(log n) 时间内得以恢复。
在任何时刻，父节点 ≤ 子节点 → 全局最小值位于根节点。

#### 自己动手试试

1.  实现一个最大堆变体。
2.  跟踪每次插入的交换次数。
3.  通过重复提取最小值与堆排序结合。
4.  将堆可视化为树状图。

#### 测试用例

| 操作          | 输入           | 输出   | 操作后堆         | 备注       |
| ------------- | -------------- | ------ | ---------------- | ---------- |
| 插入          | [10,24,5,31,7] | -      | [5,7,10,31,24]   | 有效       |
| 查看          | -              | 5      | [5,7,10,31,24]   | 最小值     |
| 提取最小值    | -              | 5      | [7,24,10,31]     | 重新排序   |
| 空堆提取      | []             | None   | []               | 安全       |

边界情况

- 重复元素 → 处理良好
- 空堆 → 返回哨兵值
- 负数 → 没有问题

#### 复杂度

| 操作          | 时间复杂度 | 空间复杂度 |
| ------------- | ---------- | ---------- |
| 插入          | O(log n)   | O(1)       |
| 提取最小值    | O(log n)   | O(1)       |
| 查看          | O(1)       | O(1)       |

最小堆是安静的整理者，始终将最小的任务保持在顶部，随时待命。
### 226 最大堆实现

最大堆是一种二叉树，其中每个父节点都大于或等于其子节点。它紧凑地存储在数组中，保证了 O(1) 的时间访问最大元素，以及 O(log n) 的时间进行插入和删除，这使其成为调度系统、排行榜和基于优先级的系统的理想选择。

#### 我们要解决什么问题？

我们需要一种数据结构，能够高效地维护一个动态元素集合，同时允许我们：

- 快速访问最大元素
- 高效地插入和删除元素
- 自动维持顺序

最大堆正是为此而生，它总是将最大的元素"冒泡"到顶部。

#### 它是如何工作的（通俗解释）？

二叉最大堆以数组形式存储，具有以下关系：

- `parent(i) = (i - 1) / 2`
- `left(i) = 2i + 1`
- `right(i) = 2i + 2`

操作：

1. 插入(x)：
   * 将 `x` 添加到末尾。
   * 当 `x > parent(x)` 时，执行"上浮"操作。

2. 提取最大值()：
   * 移除根节点（最大值）。
   * 将最后一个元素移到根节点位置。
   * 当父节点小于较大的子节点时，执行"下沉"操作。

3. 查看最大值()：
   * 返回 `arr[0]`（最大元素）。

示例

初始状态：`[ ]`
插入序列：10, 24, 5, 31, 7

| 步骤 | 操作       | 数组                         |
| ---- | ---------- | ---------------------------- |
| 1    | 插入 10    | [10]                         |
| 2    | 插入 24    | 上浮 → [24, 10]              |
| 3    | 插入 5     | [24, 10, 5]                  |
| 4    | 插入 31    | 上浮 → [31, 24, 5, 10]       |
| 5    | 插入 7     | [31, 24, 5, 10, 7]           |

提取最大值：

- 交换根节点与最后一个节点 (31 ↔ 7)：`[7, 24, 5, 10, 31]`
- 移除最后一个节点：`[7, 24, 5, 10]`
- 下沉：交换 7 ↔ 24 → `[24, 10, 5, 7]`

#### 精简代码（简易版本）

C

```c
#include <stdio.h>
#define MAX 100

typedef struct {
    int arr[MAX];
    int size;
} MaxHeap;

void swap(int *a, int *b) {
    int tmp = *a; *a = *b; *b = tmp;
}

void heapify_up(MaxHeap *h, int i) {
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (h->arr[i] <= h->arr[parent]) break;
        swap(&h->arr[i], &h->arr[parent]);
        i = parent;
    }
}

void heapify_down(MaxHeap *h, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < h->size && h->arr[left] > h->arr[largest]) largest = left;
    if (right < h->size && h->arr[right] > h->arr[largest]) largest = right;

    if (largest != i) {
        swap(&h->arr[i], &h->arr[largest]);
        heapify_down(h, largest);
    }
}

void insert(MaxHeap *h, int val) {
    h->arr[h->size] = val;
    heapify_up(h, h->size);
    h->size++;
}

int extract_max(MaxHeap *h) {
    if (h->size == 0) return -1;
    int root = h->arr[0];
    h->arr[0] = h->arr[--h->size];
    heapify_down(h, 0);
    return root;
}

int peek(MaxHeap *h) {
    return h->size > 0 ? h->arr[0] : -1;
}

int main(void) {
    MaxHeap h = {.size = 0};
    int vals[] = {10, 24, 5, 31, 7};
    for (int i = 0; i < 5; i++) insert(&h, vals[i]);
    printf("最大值: %d\n", peek(&h));
    printf("已提取: %d\n", extract_max(&h));
    for (int i = 0; i < h.size; i++) printf("%d ", h.arr[i]);
    printf("\n");
}
```

Python

```python
class MaxHeap:
    def __init__(self):
        self.arr = []

    def _parent(self, i): return (i - 1) // 2
    def _left(self, i): return 2 * i + 1
    def _right(self, i): return 2 * i + 2

    def insert(self, val):
        self.arr.append(val)
        i = len(self.arr) - 1
        while i > 0 and self.arr[i] > self.arr[self._parent(i)]:
            p = self._parent(i)
            self.arr[i], self.arr[p] = self.arr[p], self.arr[i]
            i = p

    def extract_max(self):
        if not self.arr:
            return None
        root = self.arr[0]
        last = self.arr.pop()
        if self.arr:
            self.arr[0] = last
            self._heapify_down(0)
        return root

    def _heapify_down(self, i):
        largest = i
        l, r = self._left(i), self._right(i)
        if l < len(self.arr) and self.arr[l] > self.arr[largest]:
            largest = l
        if r < len(self.arr) and self.arr[r] > self.arr[largest]:
            largest = r
        if largest != i:
            self.arr[i], self.arr[largest] = self.arr[largest], self.arr[i]
            self._heapify_down(largest)

    def peek(self):
        return self.arr[0] if self.arr else None

# 示例
h = MaxHeap()
for x in [10, 24, 5, 31, 7]:
    h.insert(x)
print("堆:", h.arr)
print("最大值:", h.peek())
print("已提取:", h.extract_max())
print("之后:", h.arr)
```

#### 为什么它很重要

- 需要快速访问最大值的优先队列
- 调度最高优先级任务
- 动态跟踪最大元素
- 用于堆排序、选择问题、Top-K 查询

#### 一个温和的证明（为什么它有效）

每个操作最多调整高度 = log n 层。
由于所有交换都将较大的元素向上移动，堆属性（父节点 ≥ 子节点）在对数时间内得以恢复。
因此，根节点始终保存着最大值。

#### 亲自尝试

1.  通过翻转比较操作，将其转换为最小堆。
2.  使用堆来实现查找第 k 大元素的函数。
3.  追踪每次插入/删除后的交换过程。
4.  将堆可视化为树形图。

#### 测试用例

| 操作          | 输入           | 输出 | 操作后堆         | 备注               |
| ------------- | -------------- | ---- | ---------------- | ------------------ |
| 插入          | [10,24,5,31,7] | -    | [31,24,5,10,7]   | 最大值在根节点     |
| 查看最大值    | -              | 31   | [31,24,5,10,7]   | 最大值             |
| 提取最大值    | -              | 31   | [24,10,5,7]      | 堆已修复           |
| 空堆提取      | []             | None | []               | 安全处理           |

边界情况

- 重复元素 → 处理良好
- 空堆 → 哨兵值
- 负数 → 有效

#### 复杂度

| 操作        | 时间复杂度 | 空间复杂度 |
| ----------- | ---------- | ---------- |
| 插入        | O(log n)   | O(1)       |
| 提取最大值  | O(log n)   | O(1)       |
| 查看最大值  | O(1)       | O(1)       |

最大堆是顶峰的守护者，总是加冕最伟大的元素，确保从根到叶的层级结构始终如一。
### 227 斐波那契堆的插入/删除

斐波那契堆是一种可合并堆，针对非常快速的摊还操作进行了优化。它维护着一组堆序树，大多数操作只进行最小的结构调整，将工作积攒起来用于偶尔的合并操作。经典结论：`insert`、`find-min` 和 `decrease-key` 的摊还时间复杂度为 O(1)，而 `extract-min` 的摊还时间复杂度为 O(log n)。

#### 我们要解决什么问题？

我们希望在大量调用 `decrease-key` 的算法（如 Dijkstra（迪杰斯特拉）和 Prim（普里姆）算法）中实现超快速的优先队列操作。二叉堆和配对堆的 `decrease-key` 操作时间复杂度为 O(log n) 或具有较好的常数因子，但斐波那契堆实现了 `insert` 和 `decrease-key` 的摊还 O(1) 时间复杂度，从而改善了理论界限。

目标：
维护一组堆序树，使得大多数更新操作只涉及少量指针修改，将合并操作延迟到 `extract-min` 时才进行。

#### 它是如何工作的（通俗解释）？

结构要点

- 一个由多棵树组成的根链表，每棵树都遵循最小堆序。
- 一个指向全局最小根的指针。
- 每个节点存储度数、父节点、子节点以及一个由 `decrease-key` 使用的标记位。
- 根链表是一个循环双向链表，子链表也类似。

核心思想

- 插入操作将一个单节点树添加到根链表中，并在需要时更新 `min`。
- 删除操作通常实现为 `decrease-key(x, -inf)` 然后 `extract-min`。
- 提取最小值操作移除最小根，将其子节点提升到根链表，然后通过链接度数相同的树来合并根节点，直到所有根的度数都唯一。

#### 插入操作如何工作

`insert(x)` 的步骤

1.  创建一个单节点 `x`。
2.  将 `x` 拼接到根链表中。
3.  如果 `x.key < min.key`，则更新 `min`。

示例步骤（根链表视图）

| 步骤 | 操作       | 根链表          | 最小值 |
| ---- | ---------- | --------------- | ------ |
| 1    | 插入 12    | [12]            | 12     |
| 2    | 插入 7     | [12, 7]         | 7      |
| 3    | 插入 25    | [12, 7, 25]     | 7      |
| 4    | 插入 3     | [12, 7, 25, 3]  | 3      |

所有插入操作都是 O(1) 的指针拼接操作。

#### 删除操作如何工作

要删除任意节点 `x`，标准方法是

1.  `decrease-key(x, -inf)`，使 `x` 变为最小值。
2.  `extract-min()` 将其移除。

删除操作继承了 `extract-min` 的成本。

#### 微型代码（教学骨架）

这是一个最小化的框架，用于说明结构和所要求的两种操作。它省略了完整的 `decrease-key` 和级联切断操作以保持重点。实际上，一个完整的斐波那契堆还需要实现带有合并数组的 `decrease-key` 和 `extract-min`。

Python（用于通过提取最小值路径进行插入和删除的教学骨架）

```python
class FibNode:
    def __init__(self, key):
        self.key = key
        self.degree = 0
        self.mark = False
        self.parent = None
        self.child = None
        # 循环双向链表指针
        self.left = self
        self.right = self

def _splice(a, b):
    # 在循环链表中将节点 b 插入到节点 a 的右侧
    b.right = a.right
    b.left = a
    a.right.left = b
    a.right = b

def _remove_from_list(x):
    x.left.right = x.right
    x.right.left = x.left
    x.left = x.right = x

class FibHeap:
    def __init__(self):
        self.min = None
        self.n = 0

    def insert(self, key):
        x = FibNode(key)
        if self.min is None:
            self.min = x
        else:
            _splice(self.min, x)
            if x.key < self.min.key:
                self.min = x
        self.n += 1
        return x

    def _merge_root_list(self, other_min):
        if other_min is None:
            return
        # 连接循环链表：self.min 和 other_min
        a = self.min
        b = other_min
        a_right = a.right
        b_left = b.left
        a.right = b
        b.left = a
        a_right.left = b_left
        b_left.right = a_right
        if b.key < self.min.key:
            self.min = b

    def extract_min(self):
        z = self.min
        if z is None:
            return None
        # 将子节点提升到根链表
        if z.child:
            c = z.child
            nodes = []
            cur = c
            while True:
                nodes.append(cur)
                cur = cur.right
                if cur == c:
                    break
            for x in nodes:
                x.parent = None
                _remove_from_list(x)
                _splice(z, x)  # 插入到根链表中 z 附近
        # 从根链表中移除 z
        if z.right == z:
            self.min = None
        else:
            nxt = z.right
            _remove_from_list(z)
            self.min = nxt
            self._consolidate()  # 实际实现会链接度数相同的树
        self.n -= 1
        return z.key

    def _consolidate(self):
        # 占位符：完整实现使用数组 A[0..floor(log_phi n)]
        # 来链接度数相同的根节点，直到所有度数唯一。
        pass

    def delete(self, node):
        # 在完整的斐波那契堆中：
        # decrease_key(node, -inf)，然后 extract_min
        # 这里我们通过手动将节点设为最小值（如果节点是最小值）来近似实现，
        # 并且为了简洁省略了级联切断。
        # 为了在实际使用中保证正确性，需要实现 decrease_key 和切断操作。
        node.key = float("-inf")
        if self.min and node.key < self.min.key:
            self.min = node
        return self.extract_min()

# 示例
H = FibHeap()
n1 = H.insert(12)
n2 = H.insert(7)
n3 = H.insert(25)
n4 = H.insert(3)
print("提取的最小值:", H.extract_min())  # 3
```

注意 这个骨架展示了 `insert` 如何拼接到根链表以及 `extract_min` 如何提升子节点。生产版本必须实现带有级联切断的 `decrease-key` 和链接度数相同树的 `_consolidate`。

#### 为什么它很重要

- 对于大量使用 `decrease-key` 的图算法，提供了理论上的加速。
- 合并操作可以通过连接根链表实现 O(1) 时间复杂度。
- 摊还保证由势能函数分析支持。

#### 一个温和的证明（为什么它有效）

摊还分析使用一个基于根链表中树的数量和标记节点数量的势能函数。

- `insert` 只添加一个根节点并可能更新 `min`，这会减少或略微增加势能，因此摊还 O(1)。
- `decrease-key` 切断一个节点并可能级联切断其父节点，但标记限制了整个序列中级联切断的总数，从而得到摊还 O(1)。
- `extract-min` 触发合并。不同度数的数量是 O(log n)，因此链接操作的摊还成本是 O(log n)。

#### 动手试试

1.  使用一个按度数索引的数组完成 `_consolidate`，链接度数相同的根节点直到度数唯一。
2.  实现带有切断和级联切断规则的 `decrease_key(x, new_key)`。
3.  添加 `union(H1, H2)`，它连接两个根链表并选择较小的 `min`。
4.  在包含大量 `decrease-key` 操作的工作负载上，与二叉堆和配对堆进行基准测试。

#### 测试用例

| 操作         | 输入               | 预期结果            | 备注                                       |
| ------------ | ------------------ | ------------------- | ------------------------------------------ |
| insert       | 12, 7, 25, 3       | min = 3             | 简单的根链表更新                           |
| extract_min  | 上述操作之后       | 返回 3              | 子节点被提升，合并                         |
| delete(node) | 删除 7             | 7 被移除            | 通过减小到负无穷然后提取                   |
| meld         | 两个堆的合并       | 新 min = min(m1, m2) | O(1) 连接                                  |

边界情况

- 从空堆中提取返回 None
- 重复键值是可以的
- 在没有提取操作的情况下进行大量插入会构建许多小树，直到合并

#### 复杂度

| 操作         | 摊还时间                         |
| ------------ | -------------------------------- |
| insert       | O(1)                             |
| find-min     | O(1)                             |
| decrease-key | O(1)                             |
| extract-min  | O(log n)                         |
| delete       | O(log n)（通过减小然后提取）     |

斐波那契堆以严格的顺序换取惰性的优雅。大多数操作都是微小的指针调整，繁重的工作只在合并期间偶尔发生。
### 228 配对堆合并

配对堆是一种简单的、基于指针的可合并堆，在实践中以速度快而闻名。它的秘密武器是合并操作：通过将较大键值的根节点作为较小键值根节点的子节点，来连接两个堆的根节点。许多其他操作都可以归结为少量的合并操作。

#### 我们要解决什么问题？

我们想要一个具有极小常数因子和非常简单的代码的优先队列，同时保持理论保证接近斐波那契堆。配对堆提供了极快的 `insert`、`meld` 和通常的 `decrease-key`，而 `delete-min` 则由轻量级的多路合并提供支持。

目标：
将堆表示为一棵节点树，并实现合并操作，使得所有更高级别的操作都可以表示为一系列合并操作。

#### 它是如何工作的（通俗解释）？

每个节点包含：键值、第一个孩子节点和下一个兄弟节点。堆只是一个指向根节点的指针。
要合并两个堆 `A` 和 `B`：

1. 如果其中一个为空，则返回另一个。
2. 比较根节点。
3. 通过将较大根节点的堆链接为较小根节点的第一个孩子，使其成为较小根节点堆的新孩子。

通过合并实现的其他操作

- 插入(x)：创建一个单节点堆，然后 `merge(root, x)`。
- 查找最小值：根节点的键值。
- 删除最小值：移除根节点，然后分两趟合并其子节点

  1. 从左到右，成对合并相邻的兄弟节点。
  2. 从右到左，将结果堆合并回一个堆。
- 减小键值(x, new)：将 `x` 从其位置切断，设置 `x.key = new`，然后 `merge(root, x)`。

示例步骤（仅合并）

| 步骤 | 堆 A (根节点) | 堆 B (根节点) | 操作           | 新根节点 |
| ---- | ------------- | ------------- | -------------- | -------- |
| 1    | 7             | 12            | 将 12 链接到 7 下 | 7        |
| 2    | 7             | 3             | 将 7 链接到 3 下  | 3        |
| 3    | 3             | 9             | 将 9 链接到 3 下  | 3        |

最终根节点是所有合并堆中的最小值。

#### 精简代码（简易版本）

C 语言（合并、插入、查找最小值、两趟删除最小值）

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int key;
    struct Node *child;
    struct Node *sibling;
} Node;

Node* make_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key; n->child = NULL; n->sibling = NULL;
    return n;
}

Node* merge(Node* a, Node* b) {
    if (!a) return b;
    if (!b) return a;
    if (b->key < a->key) { Node* t = a; a = b; b = t; }
    // 使 b 成为 a 的第一个孩子
    b->sibling = a->child;
    a->child = b;
    return a;
}

Node* insert(Node* root, int key) {
    return merge(root, make_node(key));
}

Node* merge_pairs(Node* first) {
    if (!first || !first->sibling) return first;
    Node* a = first;
    Node* b = first->sibling;
    Node* rest = b->sibling;
    a->sibling = b->sibling = NULL;
    return merge(merge(a, b), merge_pairs(rest));
}

Node* delete_min(Node* root, int* out) {
    if (!root) return NULL;
    *out = root->key;
    Node* new_root = merge_pairs(root->child);
    free(root);
    return new_root;
}

int main(void) {
    Node* h = NULL;
    h = insert(h, 7);
    h = insert(h, 12);
    h = insert(h, 3);
    h = insert(h, 9);
    int m;
    h = delete_min(h, &m);
    printf("Deleted min: %d\n", m); // 3
    return 0;
}
```

Python（简洁的配对堆）

```python
class Node:
    __slots__ = ("key", "child", "sibling")
    def __init__(self, key):
        self.key = key
        self.child = None
        self.sibling = None

def merge(a, b):
    if not a: return b
    if not b: return a
    if b.key < a.key:
        a, b = b, a
    b.sibling = a.child
    a.child = b
    return a

def merge_pairs(first):
    if not first or not first.sibling:
        return first
    a, b, rest = first, first.sibling, first.sibling.sibling
    a.sibling = b.sibling = None
    return merge(merge(a, b), merge_pairs(rest))

class PairingHeap:
    def __init__(self): self.root = None
    def find_min(self): return None if not self.root else self.root.key
    def insert(self, x):
        self.root = merge(self.root, Node(x))
    def meld(self, other):
        self.root = merge(self.root, other.root)
    def delete_min(self):
        if not self.root: return None
        m = self.root.key
        self.root = merge_pairs(self.root.child)
        return m

# 示例
h = PairingHeap()
for x in [7, 12, 3, 9]:
    h.insert(x)
print(h.find_min())      # 3
print(h.delete_min())    # 3
print(h.find_min())      # 7
```

#### 为什么它很重要

- 代码极其简单，但在实践中性能很高
- 合并是常数时间的指针操作
- 非常适合混合频繁插入和减小键值的工作负载
- 是斐波那契堆的一个强大的实用替代方案

#### 一个温和的证明（为什么它有效）

合并的正确性

- 链接后，较小的根节点仍然是父节点，因此堆序性质在根节点和新附加的子树路径上成立。

删除最小值的两趟操作

- 成对合并减少了树的数量，同时保持根节点较小。
- 第二趟从右向左的折叠将较大的部分堆合并成一个堆。
- 分析表明 `delete-min` 的摊还时间复杂度为 O(log n)；`insert` 和 `meld` 为 O(1) 摊还时间。
- `decrease-key` 在实践中被认为是 O(1) 摊还时间，在常见模型下理论上也接近于此。

#### 动手尝试

1. 实现 `decrease_key(node, new_key)`：将节点从其父节点切断，并在降低其键值后 `merge(root, node)`。
2. 添加一个句柄表以快速访问节点，实现快速的减小键值操作。
3. 在 Dijkstra 工作负载上，与二叉堆和斐波那契堆进行基准测试。
4. 在随机树上可视化删除最小值的两趟配对过程。

#### 测试用例

| 操作       | 输入           | 输出                 | 备注                   |
| ---------- | -------------- | -------------------- | ---------------------- |
| 插入       | 7, 12, 3, 9    | 最小值 = 3           | 根节点跟踪全局最小值   |
| 合并       | 合并两个堆     | 新最小值是两者的最小 | 常数时间链接           |
| 删除最小值 | 在上述操作之后 | 3                    | 两趟配对               |
| 删除最小值 | 下一次         | 7                    | 堆重新组织             |

边界情况

- 与空堆合并返回另一个堆
- 重复键值自然处理
- 单节点堆安全删除为空堆

#### 复杂度

| 操作         | 摊还时间复杂度                          | 备注           |
| ------------ | --------------------------------------- | -------------- |
| 合并         | O(1)                                    | 核心原语       |
| 插入         | O(1)                                    | 合并单节点堆   |
| 查找最小值   | O(1)                                    | 根节点键值     |
| 删除最小值   | O(log n)                                | 两趟合并       |
| 减小键值     | 实践中 O(1)，模型中接近 O(1) 摊还时间 | 切断+合并      |

配对堆让合并变得毫不费力：一次比较，几个指针操作，就完成了。
### 229 二项堆合并

二项堆是一组二项树的集合，其行为类似于优先级队列的二进制计数器。核心操作是合并：合并两个堆就像将两个二进制数相加。相同度数的树会发生碰撞，你将其中一棵链接到另一棵下方，并进位到下一个度数。

#### 我们要解决什么问题？

我们想要一个支持快速合并（并集）操作，同时保持简单、可证明性能边界的优先级队列。二项堆提供了：

- `meld` 操作时间复杂度为 O(log n)
- `insert` 操作通过合并一个单节点堆实现，摊还时间复杂度为 O(1)
- `find-min` 操作时间复杂度为 O(log n)
- `delete-min` 操作通过将子节点列表反转后合并实现，时间复杂度为 O(log n)

目标：
将堆表示为一个按度数排序的二项树列表，通过遍历这些列表并链接度数相同的根节点来合并两个堆。

#### 它是如何工作的（通俗解释）？

二项树特性

- 度数为 `k` 的二项树有 `2^k` 个节点。
- 在一个二项堆中，每个度数最多出现一次。
- 根节点按度数递增顺序排列。

合并两个堆 H1 和 H2

1.  按度数合并根列表，类似于合并有序列表。
2.  遍历合并后的列表。每当遇到两棵连续树度数相同时，链接它们：将键值较大的根节点作为键值较小的根节点的子节点，度数增加 1。
3.  使用类似于二进制加法的进位思想。

Link(u, v)

- 前提条件：`degree(u) == degree(v)`。
- 链接后，`min(u.key, v.key)` 成为父节点，度数增加 1。

示例步骤（括号内为度数）

初始状态

- H1 根列表：`[2(0), 7(1), 12(3)]`
- H2 根列表：`[3(0), 9(2)]`

1. 按度数合并列表

- 合并后：`[2(0), 3(0), 7(1), 9(2), 12(3)]`

2. 解决度数相同的情况

- 在最小根节点下链接 2(0) 和 3(0) → 2 成为父节点：2(1)
- 现在列表：`[2(1), 7(1), 9(2), 12(3)]`
- 链接 2(1) 和 7(1) → 2 成为父节点：2(2)
- 现在列表：`[2(2), 9(2), 12(3)]`
- 链接 2(2) 和 9(2) → 2 成为父节点：2(3)
- 现在列表：`[2(3), 12(3)]`
- 链接 2(3) 和 12(3) → 2 成为父节点：2(4)

最终堆的根列表为 `[2(4)]`，2 是全局最小值。

#### 精简代码（简易版本）

C 语言（合并加链接，最小骨架）

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int key;
    int degree;
    struct Node* parent;
    struct Node* child;
    struct Node* sibling; // 下一个根节点或子节点列表中的下一个兄弟节点
} Node;

Node* make_node(int key){
    Node* x = calloc(1, sizeof(Node));
    x->key = key;
    return x;
}

// 假设 x->key <= y->key 且度数相等，使 'y' 成为 'x' 的子节点
static Node* link_tree(Node* x, Node* y){
    y->parent = x;
    y->sibling = x->child;
    x->child = y;
    x->degree += 1;
    return x;
}

// 按度数合并根列表（尚未链接）
static Node* merge_root_lists(Node* a, Node* b){
    if(!a) return b;
    if(!b) return a;
    Node dummy = {0};
    Node* tail = &dummy;
    while(a && b){
        if(a->degree <= b->degree){
            tail->sibling = a; a = a->sibling;
        }else{
            tail->sibling = b; b = b->sibling;
        }
        tail = tail->sibling;
    }
    tail->sibling = a ? a : b;
    return dummy.sibling;
}

// 带进位逻辑的并集操作
Node* binomial_union(Node* h1, Node* h2){
    Node* head = merge_root_lists(h1, h2);
    if(!head) return NULL;

    Node* prev = NULL;
    Node* curr = head;
    Node* next = curr->sibling;

    while(next){
        if(curr->degree != next->degree || (next->sibling && next->sibling->degree == curr->degree)){
            prev = curr;
            curr = next;
        }else{
            if(curr->key <= next->key){
                curr->sibling = next->sibling;
                curr = link_tree(curr, next);
            }else{
                if(prev) prev->sibling = next;
                else head = next;
                curr = link_tree(next, curr);
            }
        }
        next = curr->sibling;
    }
    return head;
}

// 便捷函数：通过与单节点堆合并来插入
Node* insert(Node* heap, int key){
    return binomial_union(heap, make_node(key));
}

int main(void){
    Node* h1 = NULL;
    Node* h2 = NULL;
    h1 = insert(h1, 2);
    h1 = insert(h1, 7);
    h1 = insert(h1, 12); // 在合并后度数会规范化
    h2 = insert(h2, 3);
    h2 = insert(h2, 9);
    Node* h = binomial_union(h1, h2);
    // h 现在保存合并后的堆；查找最小值需要扫描根列表
    for(Node* r = h; r; r = r->sibling)
        printf("根节点 key=%d deg=%d\n", r->key, r->degree);
    return 0;
}
```

Python 语言（简洁的并集和链接）

```python
class Node:
    __slots__ = ("key", "degree", "parent", "child", "sibling")
    def __init__(self, key):
        self.key = key
        self.degree = 0
        self.parent = None
        self.child = None
        self.sibling = None

def link_tree(x, y):
    # 假设 x.key <= y.key 且度数相等
    y.parent = x
    y.sibling = x.child
    x.child = y
    x.degree += 1
    return x

def merge_root_lists(a, b):
    if not a: return b
    if not b: return a
    dummy = Node(-1)
    t = dummy
    while a and b:
        if a.degree <= b.degree:
            t.sibling, a = a, a.sibling
        else:
            t.sibling, b = b, b.sibling
        t = t.sibling
    t.sibling = a if a else b
    return dummy.sibling

def binomial_union(h1, h2):
    head = merge_root_lists(h1, h2)
    if not head: return None
    prev, curr, nxt = None, head, head.sibling
    while nxt:
        if curr.degree != nxt.degree or (nxt.sibling and nxt.sibling.degree == curr.degree):
            prev, curr, nxt = curr, nxt, nxt.sibling
        else:
            if curr.key <= nxt.key:
                curr.sibling = nxt.sibling
                curr = link_tree(curr, nxt)
                nxt = curr.sibling
            else:
                if prev: prev.sibling = nxt
                else: head = nxt
                curr = link_tree(nxt, curr)
                nxt = curr.sibling
    return head

def insert(heap, key):
    return binomial_union(heap, Node(key))

# 示例
h1 = None
for x in [2, 7, 12]:
    h1 = insert(h1, x)
h2 = None
for x in [3, 9]:
    h2 = insert(h2, x)
h = binomial_union(h1, h2)
roots = []
r = h
while r:
    roots.append((r.key, r.degree))
    r = r.sibling
print(roots)  # 例如：[(2, 4)] 或具有最小根节点的一组唯一度数的小集合
```

#### 为什么它很重要

- 易于合并：合并堆是首要操作，而非事后考虑。
- 具有二进制计数器直觉的清晰、可证明的性能边界。
- 是斐波那契堆及其变体的基础。
- 在算法或多队列系统需要频繁合并时非常有用。

#### 一个温和的证明（为什么它有效）

合并根列表会产生非递减顺序的度数。链接只发生在相邻的、度数相同的根节点之间，在所有进位解决后，恰好产生每个度数的一棵树。这反映了二进制加法：每次链接对应于将 1 进位到下一位。由于最大度数为 O(log n)，合并操作执行 O(log n) 次链接和扫描，从而给出 O(log n) 的时间复杂度。

#### 动手尝试

1.  通过扫描根列表并保持一个指向最小根节点的指针来实现 `find_min`。
2.  实现 `delete_min`：移除最小根节点，将其子节点列表反转成一个独立的堆，然后执行 `union`。
3.  通过将节点剪切并重新插入到根列表，然后修复父节点顺序来添加 `decrease_key` 操作。
4.  在大规模随机工作负载上，比较二项堆与二叉堆、配对堆的合并时间。

#### 测试用例

| 用例             | H1 根节点 (度数)      | H2 根节点 (度数) | 结果                   | 备注                 |
| ---------------- | ------------------- | -------------- | ------------------------ | --------------------- |
| 简单合并         | `[2(0), 7(1)]`        | `[3(0)]`         | 链接后根节点唯一  | 2 成为 3 的父节点 |
| 进位链           | `[2(0), 7(1), 12(3)]` | `[3(0), 9(2)]`   | 单个根节点 2(4)         | 级联链接       |
| 通过合并插入     | 具有根节点的 H        | 加上 `[x(0)]`    | 以 O(1) 摊还时间合并 | 可能发生单次进位 |

边界情况

- 与空堆合并返回另一个堆。
- 重复键有效；任意打破平局。
- 链接时保持稳定的兄弟节点指针。

#### 复杂度

| 操作         | 时间复杂度                                        |
| ------------ | ------------------------------------------- |
| meld (union) | O(log n)                                    |
| insert       | 通过与单节点堆合并实现，摊还时间复杂度 O(1)   |
| find-min     | O(log n) 扫描，或保持指针实现 O(1) 查看 |
| delete-min   | O(log n)                                    |
| decrease-key | 典型实现为 O(log n)             |

合并二项堆就像做二进制加法：相同度数的树发生碰撞、链接并向前进位，直到结构整洁，最小值位于某个根节点。
### 230 左倾堆合并

左倾堆是一种为高效合并而优化的二叉树堆。其结构向左倾斜，使得合并两个堆可以在 O(log n) 时间内递归完成。其巧妙之处在于存储每个节点的空路径长度（npl），确保到空子节点的最短路径总是在右侧，这使得合并操作保持较浅的深度。

#### 我们要解决什么问题？

我们想要一种对合并友好的堆，其结构比斐波那契堆或配对堆更简单，但合并速度比标准二叉堆更快。左倾堆是一个理想的折中点：合并速度快、代码简单，并且仍然支持所有关键的堆操作。

目标：
设计一种堆，使其最短子树保持在右侧，从而确保递归合并保持对数复杂度。

#### 它是如何工作的？（通俗解释）

每个节点存储：

- `key` – 用于排序的值
- `left`, `right` – 子节点指针
- `npl` – 空路径长度（到最近空节点的距离）

规则：

1.  堆序：父节点的键 ≤ 子节点的键（最小堆）
2.  左倾性质：`npl(left) ≥ npl(right)`

合并(a, b)：

1.  如果其中一个为空，返回另一个。
2.  比较根节点，键值较小的根成为新根。
3.  递归合并 `a.right` 和 `b`。
4.  合并后，如果需要，交换子节点以保持左倾性质。
5.  更新 `npl`。

其他操作通过合并实现

-   插入(x)：将堆与单节点堆 `x` 合并。
-   删除最小元素()：合并根的左子树和右子树。

示例（最小堆合并）

堆 A：根为 3

```
   3
  / \
 5   9
```

堆 B：根为 4

```
  4
 / \
 8  10
```

合并(3, 4)：

-   3 < 4 → 新根为 3
-   合并右子树(9) 与堆 4
-   合并后：

```
     3
    / \
   5   4
      / \
     8  10
        /
       9
```

如果右侧的 npl > 左侧，则通过交换进行重新平衡 → 确保左倾形状。

#### 精简代码（简易版本）

C 语言（合并、插入、删除最小元素）

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int key;
    int npl;
    struct Node *left, *right;
} Node;

Node* make_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key;
    n->npl = 0;
    n->left = n->right = NULL;
    return n;
}

int npl(Node* x) { return x ? x->npl : -1; }

Node* merge(Node* a, Node* b) {
    if (!a) return b;
    if (!b) return a;
    if (b->key < a->key) { Node* t = a; a = b; b = t; }
    a->right = merge(a->right, b);
    // 维护左倾性质
    if (npl(a->left) < npl(a->right)) {
        Node* t = a->left;
        a->left = a->right;
        a->right = t;
    }
    a->npl = npl(a->right) + 1;
    return a;
}

Node* insert(Node* h, int key) {
    return merge(h, make_node(key));
}

Node* delete_min(Node* h, int* out) {
    if (!h) return NULL;
    *out = h->key;
    Node* new_root = merge(h->left, h->right);
    free(h);
    return new_root;
}

int main(void) {
    Node* h1 = NULL;
    int vals[] = {5, 3, 9, 7, 4};
    for (int i = 0; i < 5; i++)
        h1 = insert(h1, vals[i]);

    int m;
    h1 = delete_min(h1, &m);
    printf("删除的最小值: %d\n", m);
    return 0;
}
```

Python

```python
class Node:
    __slots__ = ("key", "npl", "left", "right")
    def __init__(self, key):
        self.key = key
        self.npl = 0
        self.left = None
        self.right = None

def npl(x): return x.npl if x else -1

def merge(a, b):
    if not a: return b
    if not b: return a
    if b.key < a.key:
        a, b = b, a
    a.right = merge(a.right, b)
    # 强制执行左倾性质
    if npl(a.left) < npl(a.right):
        a.left, a.right = a.right, a.left
    a.npl = npl(a.right) + 1
    return a

def insert(h, key):
    return merge(h, Node(key))

def delete_min(h):
    if not h: return None, None
    m = h.key
    h = merge(h.left, h.right)
    return m, h

# 示例
h = None
for x in [5, 3, 9, 7, 4]:
    h = insert(h, x)
m, h = delete_min(h)
print("删除的最小值:", m)
```

#### 为什么它重要

-   使用简单递归实现快速合并
-   适用于需要频繁合并的优先队列
-   实现比斐波那契堆更清晰
-   保证对数复杂度的合并和删除最小元素操作

#### 一个温和的证明（为什么它有效）

空路径长度（npl）确保右侧路径总是最短的。因此，每个合并步骤只递归遍历一条右侧路径，而不是两条。这将递归深度限制在 O(log n) 以内。
所有其他操作（插入、删除最小元素）都被定义为少量合并操作，因此也是 O(log n)。

#### 亲自尝试

1.  追踪两个 3 节点堆的合并过程。
2.  可视化每一步之后的 `npl` 值。
3.  实现 `find_min()`（只需返回根节点的键）。
4.  尝试通过翻转比较操作来创建一个最大堆变体。

#### 测试用例

| 操作        | 输入          | 输出      | 备注                           |
| ----------- | ------------- | --------- | ------------------------------ |
| 插入        | [5,3,9,7,4]   | 根节点=3  | 最小堆性质成立                 |
| 删除最小元素 | 移除 3       | 新根节点=4 | 左倾性质得以保持               |
| 合并        | [3,5] + [4,6] | 根节点=3  | 右侧路径长度 ≤ log n           |
| 空合并      | None + [5]    | [5]       | 安全处理                       |

边界情况

-   与空堆合并 → 返回另一个堆
-   重复键值 → 处理正常
-   单节点 → npl = 0

#### 复杂度

| 操作        | 时间复杂度 | 空间复杂度 |
| ----------- | ---------- | ---------- |
| 合并        | O(log n)   | O(1)       |
| 插入        | O(log n)   | O(1)       |
| 删除最小元素 | O(log n)   | O(1)       |
| 查找最小元素 | O(1)       | O(1)       |

左倾堆就像一条总是向左弯曲的河流，塑造自身以使合并操作流畅而自然地进行。

## 第 24 节 平衡树
### 231 AVL 树插入

AVL 树是一种自平衡二叉搜索树，其中任何节点的左右子树的高度差（平衡因子）最多为 1。这个不变性保证了 O(log n) 的查找、插入和删除操作，使 AVL 树成为动态维护顺序的经典示例。

#### 我们要解决什么问题？

普通的二叉搜索树在不幸的插入操作后可能会变得倾斜（像链表一样），导致性能下降到 O(n)。AVL 树在每次插入后自动恢复平衡，确保搜索和更新操作保持快速。

目标：
通过在插入后旋转节点来维护一个平衡的搜索树，使得各处的高度差 ≤ 1。

#### 它是如何工作的（通俗解释）？

1.  像在普通 BST 中一样插入键。
2.  递归返回时更新高度。
3.  检查平衡因子 = `height(left) - height(right)`。
4.  如果它超出 {−1, 0, +1} 的范围，执行四种旋转之一来恢复平衡：

| 情况              | 模式                                 | 修复方法                         |
| ----------------- | ------------------------------------ | -------------------------------- |
| 左-左 (LL)        | 插入到左-左子树                      | 右旋                             |
| 右-右 (RR)        | 插入到右-右子树                      | 左旋                             |
| 左-右 (LR)        | 插入到左-右子树                      | 先在子节点左旋，然后右旋         |
| 右-左 (RL)        | 插入到右-左子树                      | 先在子节点右旋，然后左旋         |

示例

插入 30, 20, 10

-   30 → 根节点
-   20 → 30 的左子节点
-   10 → 20 的左子节点
    → 在 30 处不平衡：平衡因子 = 2
    → LL 情况 → 对 30 进行右旋

平衡后的树：

```
     20
    /  \
   10   30
```

#### 逐步示例

插入序列：10, 20, 30, 40, 50

| 步骤 | 插入 | 树（中序遍历）     | 不平衡位置 | 旋转   | 旋转后根节点 |
| ---- | ---- | ------------------ | ---------- | ------ | ------------ |
| 1    | 10   | [10]               | -          | -      | 10           |
| 2    | 20   | [10,20]            | 平衡       | -      | 10           |
| 3    | 30   | [10,20,30]         | 在 10 (RR) | 左旋   | 20           |
| 4    | 40   | [10,20,30,40]      | 平衡       | -      | 20           |
| 5    | 50   | [10,20,30,40,50]   | 在 20 (RR) | 左旋   | 30           |

最终的平衡树：

```
     30
    /  \
   20   40
  /       \
 10        50
```

#### 精简代码（简易版本）

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int key, height;
    struct Node *left, *right;
} Node;

int height(Node* n) { return n ? n->height : 0; }
int max(int a, int b) { return a > b ? a : b; }

Node* new_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key;
    n->height = 1;
    n->left = n->right = NULL;
    return n;
}

Node* rotate_right(Node* y) {
    Node* x = y->left;
    Node* T2 = x->right;
    x->right = y;
    y->left = T2;
    y->height = 1 + max(height(y->left), height(y->right));
    x->height = 1 + max(height(x->left), height(x->right));
    return x;
}

Node* rotate_left(Node* x) {
    Node* y = x->right;
    Node* T2 = y->left;
    y->left = x;
    x->right = T2;
    x->height = 1 + max(height(x->left), height(x->right));
    y->height = 1 + max(height(y->left), height(y->right));
    return y;
}

int balance(Node* n) { return n ? height(n->left) - height(n->right) : 0; }

Node* insert(Node* node, int key) {
    if (!node) return new_node(key);
    if (key < node->key) node->left = insert(node->left, key);
    else if (key > node->key) node->right = insert(node->right, key);
    else return node; // 不允许重复键

    node->height = 1 + max(height(node->left), height(node->right));
    int bf = balance(node);

    // LL
    if (bf > 1 && key < node->left->key)
        return rotate_right(node);
    // RR
    if (bf < -1 && key > node->right->key)
        return rotate_left(node);
    // LR
    if (bf > 1 && key > node->left->key) {
        node->left = rotate_left(node->left);
        return rotate_right(node);
    }
    // RL
    if (bf < -1 && key < node->right->key) {
        node->right = rotate_right(node->right);
        return rotate_left(node);
    }
    return node;
}

void inorder(Node* root) {
    if (!root) return;
    inorder(root->left);
    printf("%d ", root->key);
    inorder(root->right);
}

int main(void) {
    Node* root = NULL;
    int keys[] = {10, 20, 30, 40, 50};
    for (int i = 0; i < 5; i++)
        root = insert(root, keys[i]);
    inorder(root);
    printf("\n");
}
```

Python

```python
class Node:
    def __init__(self, key):
        self.key = key
        self.left = self.right = None
        self.height = 1

def height(n): return n.height if n else 0
def balance(n): return height(n.left) - height(n.right) if n else 0

def rotate_right(y):
    x, T2 = y.left, y.left.right
    x.right, y.left = y, T2
    y.height = 1 + max(height(y.left), height(y.right))
    x.height = 1 + max(height(x.left), height(x.right))
    return x

def rotate_left(x):
    y, T2 = x.right, x.right.left
    y.left, x.right = x, T2
    x.height = 1 + max(height(x.left), height(x.right))
    y.height = 1 + max(height(y.left), height(y.right))
    return y

def insert(node, key):
    if not node: return Node(key)
    if key < node.key: node.left = insert(node.left, key)
    elif key > node.key: node.right = insert(node.right, key)
    else: return node

    node.height = 1 + max(height(node.left), height(node.right))
    bf = balance(node)

    if bf > 1 and key < node.left.key:  # LL
        return rotate_right(node)
    if bf < -1 and key > node.right.key:  # RR
        return rotate_left(node)
    if bf > 1 and key > node.left.key:  # LR
        node.left = rotate_left(node.left)
        return rotate_right(node)
    if bf < -1 and key < node.right.key:  # RL
        node.right = rotate_right(node.right)
        return rotate_left(node)
    return node

def inorder(root):
    if not root: return
    inorder(root.left)
    print(root.key, end=' ')
    inorder(root.right)

# 示例
root = None
for k in [10, 20, 30, 40, 50]:
    root = insert(root, k)
inorder(root)
```

#### 为什么它很重要

-   保证 O(log n) 的操作复杂度
-   防止退化为线性链
-   基于旋转的清晰平衡逻辑
-   是其他平衡树（例如红黑树）的基础

#### 一个温和的证明（为什么它有效）

每次插入最多可能使一个节点不平衡，即插入节点的最低祖先节点。
一次单旋转（或双旋转）可以将该节点的平衡因子恢复到 {−1, 0, +1}，
并在常数时间内更新所有受影响的高度。
因此，每次插入执行 O(1) 次旋转，O(log n) 次递归更新。

#### 自己动手试试

1.  插入 30, 20, 10 → LL 情况
2.  插入 10, 30, 20 → LR 情况
3.  插入 30, 10, 20 → RL 情况
4.  插入 10, 20, 30 → RR 情况
5.  画出每次旋转前后的树

#### 测试用例

| 序列          | 旋转类型 | 最终根节点 | 高度 |
| ------------- | -------- | ---------- | ---- |
| [30, 20, 10]  | LL       | 20         | 2    |
| [10, 30, 20]  | LR       | 20         | 2    |
| [30, 10, 20]  | RL       | 20         | 2    |
| [10, 20, 30]  | RR       | 20         | 2    |

边界情况

-   忽略重复键
-   插入到空树 → 新节点
-   所有升序或降序插入 → 通过旋转保持平衡

#### 复杂度

| 操作   | 时间复杂度 | 空间复杂度     |
| ------ | ---------- | -------------- |
| 插入   | O(log n)   | O(h) 递归      |
| 搜索   | O(log n)   | O(h)           |
| 删除   | O(log n)   | O(h)           |

AVL 树是一位细心的园丁，随时修剪任何生长出的不平衡，因此你的搜索总能找到一条笔直的道路。
### 232 AVL 树删除

在 AVL 树中删除一个节点，就像从一座精心平衡的塔中抽出一块积木：你把它拿出来，然后执行旋转以恢复平衡。关键在于将二叉搜索树（BST）的删除规则与平衡因子检查以及沿路径的重新平衡结合起来。

#### 我们要解决什么问题？

在普通二叉搜索树中进行删除操作可能会破坏树的形状，使其变得倾斜且低效。在 AVL 树中，我们希望：

1.  删除一个节点（使用标准的二叉搜索树删除方法）
2.  重新计算高度和平衡因子
3.  通过旋转恢复平衡

这确保了树始终保持高度平衡，使操作保持在 O(log n) 的时间复杂度。

#### 它是如何工作的？（通俗解释）

1.  像在普通二叉搜索树中一样找到目标节点。
2.  删除它：
    *   如果是叶子节点 → 直接移除。
    *   如果只有一个子节点 → 用该子节点替换。
    *   如果有两个子节点 → 找到中序后继节点，复制其值，然后递归地删除该后继节点。
3.  向上回溯，更新每个祖先节点的高度和平衡因子。
4.  如果发现不平衡，应用四种旋转情况之一：

| 情况 | 条件                                 | 修复方法                         |
| ---- | ------------------------------------ | -------------------------------- |
| LL   | 平衡因子 > 1 且 左子树平衡因子 ≥ 0   | 右旋                             |
| LR   | 平衡因子 > 1 且 左子树平衡因子 < 0   | 先对左子节点左旋，然后整体右旋   |
| RR   | 平衡因子 < -1 且 右子树平衡因子 ≤ 0  | 左旋                             |
| RL   | 平衡因子 < -1 且 右子树平衡因子 > 0  | 先对右子节点右旋，然后整体左旋   |

#### 示例

从以下树中删除 10：

```
     20
    /  \
   10   30
```

-   删除叶子节点 10
-   节点 20：平衡因子 = 0 → 平衡

现在删除 30：

```
    20
   /
 10
```

-   平衡因子(20) = +1 → 仍然平衡

接着删除 10：

```
20
```

树变为单节点，仍然是 AVL 树。

#### 分步示例

插入序列 [10, 20, 30, 40, 50, 25]
然后删除 40

| 步骤 | 操作       | 不平衡节点（平衡因子） | 旋转 | 根节点 |
| ---- | ---------- | ---------------------- | ---- | ------ |
| 1    | 删除 40    | 在 30 处 (平衡因子 = -2) | RL   | 30     |
| 2    | 旋转后     | 平衡                   | -    | 30     |

#### 精简代码（简易版本）

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int key, height;
    struct Node *left, *right;
} Node;

int height(Node* n) { return n ? n->height : 0; }
int max(int a, int b) { return a > b ? a : b; }

Node* new_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key; n->height = 1;
    n->left = n->right = NULL;
    return n;
}

Node* rotate_right(Node* y) {
    Node* x = y->left;
    Node* T2 = x->right;
    x->right = y;
    y->left = T2;
    y->height = 1 + max(height(y->left), height(y->right));
    x->height = 1 + max(height(x->left), height(x->right));
    return x;
}

Node* rotate_left(Node* x) {
    Node* y = x->right;
    Node* T2 = y->left;
    y->left = x;
    x->right = T2;
    x->height = 1 + max(height(x->left), height(x->right));
    y->height = 1 + max(height(y->left), height(y->right));
    return y;
}

int balance(Node* n) { return n ? height(n->left) - height(n->right) : 0; }

Node* min_node(Node* n) {
    Node* cur = n;
    while (cur->left) cur = cur->left;
    return cur;
}

Node* insert(Node* root, int key) {
    if (!root) return new_node(key);
    if (key < root->key) root->left = insert(root->left, key);
    else if (key > root->key) root->right = insert(root->right, key);
    else return root;

    root->height = 1 + max(height(root->left), height(root->right));
    int bf = balance(root);

    // 重新平衡
    if (bf > 1 && key < root->left->key) return rotate_right(root);
    if (bf < -1 && key > root->right->key) return rotate_left(root);
    if (bf > 1 && key > root->left->key) {
        root->left = rotate_left(root->left);
        return rotate_right(root);
    }
    if (bf < -1 && key < root->right->key) {
        root->right = rotate_right(root->right);
        return rotate_left(root);
    }
    return root;
}

Node* delete(Node* root, int key) {
    if (!root) return root;

    if (key < root->key) root->left = delete(root->left, key);
    else if (key > root->key) root->right = delete(root->right, key);
    else {
        // 节点有一个子节点或无子节点
        if (!root->left || !root->right) {
            Node* tmp = root->left ? root->left : root->right;
            if (!tmp) { tmp = root; root = NULL; }
            else *root = *tmp;
            free(tmp);
        } else {
            Node* tmp = min_node(root->right);
            root->key = tmp->key;
            root->right = delete(root->right, tmp->key);
        }
    }
    if (!root) return root;

    root->height = 1 + max(height(root->left), height(root->right));
    int bf = balance(root);

    // 重新平衡
    if (bf > 1 && balance(root->left) >= 0)
        return rotate_right(root);
    if (bf > 1 && balance(root->left) < 0) {
        root->left = rotate_left(root->left);
        return rotate_right(root);
    }
    if (bf < -1 && balance(root->right) <= 0)
        return rotate_left(root);
    if (bf < -1 && balance(root->right) > 0) {
        root->right = rotate_right(root->right);
        return rotate_left(root);
    }
    return root;
}

void inorder(Node* r) {
    if (!r) return;
    inorder(r->left);
    printf("%d ", r->key);
    inorder(r->right);
}

int main(void) {
    Node* root = NULL;
    int keys[] = {10, 20, 30, 40, 50, 25};
    for (int i = 0; i < 6; i++) root = insert(root, keys[i]);
    root = delete(root, 40);
    inorder(root);
    printf("\n");
}
```

Python

```python
class Node:
    def __init__(self, key):
        self.key = key
        self.height = 1
        self.left = self.right = None

def height(n): return n.height if n else 0
def balance(n): return height(n.left) - height(n.right) if n else 0

def rotate_right(y):
    x, T2 = y.left, y.left.right
    x.right, y.left = y, T2
    y.height = 1 + max(height(y.left), height(y.right))
    x.height = 1 + max(height(x.left), height(x.right))
    return x

def rotate_left(x):
    y, T2 = x.right, x.right.left
    y.left, x.right = x, T2
    x.height = 1 + max(height(x.left), height(x.right))
    y.height = 1 + max(height(y.left), height(y.right))
    return y

def min_node(n):
    while n.left: n = n.left
    return n

def insert(r, k):
    if not r: return Node(k)
    if k < r.key: r.left = insert(r.left, k)
    elif k > r.key: r.right = insert(r.right, k)
    else: return r
    r.height = 1 + max(height(r.left), height(r.right))
    bf = balance(r)
    if bf > 1 and k < r.left.key: return rotate_right(r)
    if bf < -1 and k > r.right.key: return rotate_left(r)
    if bf > 1 and k > r.left.key:
        r.left = rotate_left(r.left); return rotate_right(r)
    if bf < -1 and k < r.right.key:
        r.right = rotate_right(r.right); return rotate_left(r)
    return r

def delete(r, k):
    if not r: return r
    if k < r.key: r.left = delete(r.left, k)
    elif k > r.key: r.right = delete(r.right, k)
    else:
        if not r.left: return r.right
        elif not r.right: return r.left
        temp = min_node(r.right)
        r.key = temp.key
        r.right = delete(r.right, temp.key)
    r.height = 1 + max(height(r.left), height(r.right))
    bf = balance(r)
    if bf > 1 and balance(r.left) >= 0: return rotate_right(r)
    if bf > 1 and balance(r.left) < 0:
        r.left = rotate_left(r.left); return rotate_right(r)
    if bf < -1 and balance(r.right) <= 0: return rotate_left(r)
    if bf < -1 and balance(r.right) > 0:
        r.right = rotate_right(r.right); return rotate_left(r)
    return r

def inorder(r):
    if not r: return
    inorder(r.left); print(r.key, end=' '); inorder(r.right)

root = None
for k in [10,20,30,40,50,25]:
    root = insert(root, k)
root = delete(root, 40)
inorder(root)
```

#### 为什么这很重要？

-   保持搜索、插入、删除操作均为 O(log n)
-   删除后自动重新平衡
-   展示了旋转如何维持结构一致性
-   是所有平衡树（如红黑树、AVL 变体）的基础

#### 一个温和的证明（为什么它有效）

每次删除最多改变子树高度 1。
每个祖先节点的平衡因子都会被重新计算；如果发现不平衡，一次（或两次）旋转即可恢复平衡。
最多访问 O(log n) 个节点，且每次修复操作是 O(1) 的。

#### 自己动手试试

1.  构建 [10, 20, 30, 40, 50, 25]，然后删除 50
2.  观察在 30 处发生的 RR 旋转
3.  删除 10，检查在 20 处的重新平衡
4.  依次删除所有节点，确认中序遍历的有序性

#### 测试用例

| 插入序列             | 删除 | 旋转 | 删除后根节点 | 是否平衡 |
| -------------------- | ---- | ---- | ------------ | -------- |
| [10,20,30,40,50,25] | 40   | RL   | 30           | ✅        |
| [10,20,30]          | 10   | RR   | 20           | ✅        |
| [30,20,10]          | 30   | LL   | 20           | ✅        |

边界情况

-   从空树中删除 → 安全
-   单节点树 → 变为 NULL
-   重复键 → 忽略

#### 复杂度

| 操作   | 时间复杂度 | 空间复杂度       |
| ------ | ---------- | ---------------- |
| 删除   | O(log n)   | O(h) 递归栈空间  |
| 搜索   | O(log n)   | O(h)             |
| 插入   | O(log n)   | O(h)             |

AVL 树的删除是一门温和的艺术：移除键值，重新平衡分支，和谐便得以恢复。
### 233 红黑树插入

红黑树（RBT）是一种自平衡二叉搜索树，它使用颜色位（红色或黑色）来间接控制平衡。与通过高度平衡的 AVL 树不同，红黑树通过颜色规则进行平衡，允许更灵活、更快速的插入，且旋转次数更少。

#### 我们要解决什么问题？

普通的二叉搜索树可能会退化为链表，导致操作复杂度为 O(n)。
红黑树保持接近平衡的高度，确保搜索、插入和删除操作的时间复杂度为 O(log n)。
与 AVL 树精确的高度平衡不同，红黑树强制执行颜色不变量，使路径长度大致相等。

#### 红黑树性质

1.  每个节点是红色或黑色。
2.  根节点始终是黑色。
3.  空（NIL）节点被视为黑色。
4.  没有两个连续的红色节点（即红色父节点不能有红色子节点）。
5.  从任意节点到其后代 NIL 节点的每条路径都包含相同数量的黑色节点。

这些规则保证了树的高度 ≤ 2 × log₂(n + 1)。

#### 它是如何工作的（通俗解释）？

1.  像普通二叉搜索树一样插入节点（将其颜色设为红色）。
2.  如果任何性质被破坏，则修复违规。
3.  根据红色节点出现的位置，使用旋转和重新着色。

| 情况   | 条件                                 | 修复方法                                           |
| ------ | ------------------------------------ | -------------------------------------------------- |
| 情况 1 | 新节点是根节点                       | 重新着色为黑色                                     |
| 情况 2 | 父节点是黑色                         | 无需修复                                           |
| 情况 3 | 父节点红色，叔节点红色               | 将父节点和叔节点重新着色为黑色，祖父节点重新着色为红色 |
| 情况 4 | 父节点红色，叔节点黑色，三角形（LR/RL） | 旋转以对齐                                         |
| 情况 5 | 父节点红色，叔节点黑色，直线（LL/RR） | 旋转并重新着色祖父节点                             |

#### 示例

插入序列：10, 20, 30

1.  插入 10 → 根节点 → 黑色
2.  插入 20 → 红色子节点 → 平衡
3.  插入 30 → 父节点（20）红色 → 情况 5（RR）→ 对节点 10 进行左旋
    → 将根节点重新着色为黑色，子节点重新着色为红色

结果：

```
    20(B)
   /    \
10(R)   30(R)
```

#### 逐步示例

插入 [7, 3, 18, 10, 22, 8, 11, 26]

| 步骤 | 插入 | 违规情况                   | 修复方法           | 根节点 |
| ---- | ---- | -------------------------- | ------------------ | ------ |
| 1    | 7    | 根节点                     | 设为黑色           | 7(B)   |
| 2    | 3    | 父节点黑色                 | 无                 | 7(B)   |
| 3    | 18   | 父节点黑色                 | 无                 | 7(B)   |
| 4    | 10   | 父节点红色，叔节点红色     | 重新着色，向上传递 | 7(B)   |
| 5    | 22   | 父节点黑色                 | 无                 | 7(B)   |
| 6    | 8    | 父节点红色，叔节点红色     | 重新着色           | 7(B)   |
| 7    | 11   | 父节点红色，叔节点黑色，LR | 旋转 + 重新着色    | 7(B)   |
| 8    | 26   | 父节点黑色                 | 无                 | 7(B)   |

最终树通过颜色不变量保持平衡。

#### 精简代码（简易版本）

C 语言（简化版）

```c
#include <stdio.h>
#include <stdlib.h>

typedef enum { RED, BLACK } Color;

typedef struct Node {
    int key;
    Color color;
    struct Node *left, *right, *parent;
} Node;

Node* new_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key;
    n->color = RED;
    n->left = n->right = n->parent = NULL;
    return n;
}

void rotate_left(Node root, Node* x) {
    Node* y = x->right;
    x->right = y->left;
    if (y->left) y->left->parent = x;
    y->parent = x->parent;
    if (!x->parent) *root = y;
    else if (x == x->parent->left) x->parent->left = y;
    else x->parent->right = y;
    y->left = x;
    x->parent = y;
}

void rotate_right(Node root, Node* y) {
    Node* x = y->left;
    y->left = x->right;
    if (x->right) x->right->parent = y;
    x->parent = y->parent;
    if (!y->parent) *root = x;
    else if (y == y->parent->left) y->parent->left = x;
    else y->parent->right = x;
    x->right = y;
    y->parent = x;
}

void fix_violation(Node root, Node* z) {
    while (z->parent && z->parent->color == RED) {
        Node* gp = z->parent->parent;
        if (z->parent == gp->left) {
            Node* uncle = gp->right;
            if (uncle && uncle->color == RED) {
                z->parent->color = BLACK;
                uncle->color = BLACK;
                gp->color = RED;
                z = gp;
            } else {
                if (z == z->parent->right) {
                    z = z->parent;
                    rotate_left(root, z);
                }
                z->parent->color = BLACK;
                gp->color = RED;
                rotate_right(root, gp);
            }
        } else {
            Node* uncle = gp->left;
            if (uncle && uncle->color == RED) {
                z->parent->color = BLACK;
                uncle->color = BLACK;
                gp->color = RED;
                z = gp;
            } else {
                if (z == z->parent->left) {
                    z = z->parent;
                    rotate_right(root, z);
                }
                z->parent->color = BLACK;
                gp->color = RED;
                rotate_left(root, gp);
            }
        }
    }
    (*root)->color = BLACK;
}

Node* bst_insert(Node* root, Node* z) {
    if (!root) return z;
    if (z->key < root->key) {
        root->left = bst_insert(root->left, z);
        root->left->parent = root;
    } else if (z->key > root->key) {
        root->right = bst_insert(root->right, z);
        root->right->parent = root;
    }
    return root;
}

void insert(Node root, int key) {
    Node* z = new_node(key);
    *root = bst_insert(*root, z);
    fix_violation(root, z);
}

void inorder(Node* r) {
    if (!r) return;
    inorder(r->left);
    printf("%d(%c) ", r->key, r->color == RED ? 'R' : 'B');
    inorder(r->right);
}

int main(void) {
    Node* root = NULL;
    int keys[] = {10, 20, 30};
    for (int i = 0; i < 3; i++) insert(&root, keys[i]);
    inorder(root);
    printf("\n");
}
```

Python（简化版）

```python
class Node:
    def __init__(self, key, color="R", parent=None):
        self.key = key
        self.color = color
        self.left = self.right = None
        self.parent = parent

def rotate_left(root, x):
    y = x.right
    x.right = y.left
    if y.left: y.left.parent = x
    y.parent = x.parent
    if not x.parent: root = y
    elif x == x.parent.left: x.parent.left = y
    else: x.parent.right = y
    y.left = x
    x.parent = y
    return root

def rotate_right(root, y):
    x = y.left
    y.left = x.right
    if x.right: x.right.parent = y
    x.parent = y.parent
    if not y.parent: root = x
    elif y == y.parent.left: y.parent.left = x
    else: y.parent.right = x
    x.right = y
    y.parent = x
    return root

def fix_violation(root, z):
    while z.parent and z.parent.color == "R":
        gp = z.parent.parent
        if z.parent == gp.left:
            uncle = gp.right
            if uncle and uncle.color == "R":
                z.parent.color = uncle.color = "B"
                gp.color = "R"
                z = gp
            else:
                if z == z.parent.right:
                    z = z.parent
                    root = rotate_left(root, z)
                z.parent.color = "B"
                gp.color = "R"
                root = rotate_right(root, gp)
        else:
            uncle = gp.left
            if uncle and uncle.color == "R":
                z.parent.color = uncle.color = "B"
                gp.color = "R"
                z = gp
            else:
                if z == z.parent.left:
                    z = z.parent
                    root = rotate_right(root, z)
                z.parent.color = "B"
                gp.color = "R"
                root = rotate_left(root, gp)
    root.color = "B"
    return root

def bst_insert(root, z):
    if not root: return z
    if z.key < root.key:
        root.left = bst_insert(root.left, z)
        root.left.parent = root
    elif z.key > root.key:
        root.right = bst_insert(root.right, z)
        root.right.parent = root
    return root

def insert(root, key):
    z = Node(key)
    root = bst_insert(root, z)
    return fix_violation(root, z)

def inorder(r):
    if not r: return
    inorder(r.left)
    print(f"{r.key}({r.color})", end=" ")
    inorder(r.right)

root = None
for k in [10,20,30]:
    root = insert(root, k)
inorder(root)
```

#### 为什么它很重要

-   比 AVL 树旋转次数更少
-   保证所有操作的时间复杂度为 O(log n)
-   应用于实际系统：Linux 内核、Java `TreeMap`、C++ `map`
-   使用着色 + 旋转的插入逻辑简单

#### 一个温和的证明（为什么它有效）

黑高不变量确保每条路径长度在 `h` 和 `2h` 之间。
通过重新着色 + 单次旋转进行平衡，确保了对数高度。
每次插入最多需要 2 次旋转，遍历复杂度为 O(log n)。

#### 自己动手试试

1.  插入 [10, 20, 30] → RR 旋转
2.  插入 [30, 15, 10] → LL 旋转
3.  插入 [10, 15, 5] → 重新着色，无旋转
4.  绘制颜色，确认不变量

#### 测试用例

| 序列                   | 旋转次数             | 根节点 | 黑高 |
| ---------------------- | -------------------- | ------ | ---- |
| [10,20,30]             | 左旋                 | 20(B)  | 2    |
| [7,3,18,10,22,8,11,26] | 混合                 | 7(B)   | 3    |
| [1,2,3,4,5]            | 多次重新着色+旋转    | 2(B)   | 3    |

#### 复杂度

| 操作   | 时间复杂度 | 空间复杂度       |
| ------ | ---------- | ---------------- |
| 插入   | O(log n)   | O(1) 次旋转      |
| 搜索   | O(log n)   | O(h)             |
| 删除   | O(log n)   | O(1) 次旋转      |

红黑树将秩序绘入混沌，每一抹红色的火花都由沉稳的黑色阴影所平衡。
### 234 红黑树删除

红黑树（RBT）中的删除是一项精细操作：移除一个节点，然后通过调整颜色和旋转来恢复平衡，以维持所有五个 RBT 不变式。插入操作修复的是“红色过多”的问题，而删除操作则常常修复“黑色过多”的问题。

#### 我们要解决什么问题？

从红黑树中删除一个节点后，黑高属性可能会被破坏（某些路径失去了一个黑色节点）。
我们的目标是：在保持时间复杂度为 O(log n) 的同时，恢复所有红黑不变式。

#### 红黑树性质（回顾）

1.  根节点是黑色的。
2.  每个节点要么是红色，要么是黑色。
3.  所有 NIL 叶子节点都是黑色的。
4.  红色节点不能有红色子节点。
5.  从任意节点到其所有 NIL 后代叶节点的路径上，包含相同数量的黑色节点。

#### 它是如何工作的（通俗解释）？

1.  执行标准的二叉搜索树（BST）删除操作。
2.  跟踪被删除的节点是否为黑色（这可能导致“双重黑”问题）。
3.  向上修复违规情况，直到树重新恢复平衡。

| 情况 | 条件                                   | 修复方法                         |
| ---- | -------------------------------------- | -------------------------------- |
| 1    | 节点是红色的                           | 直接移除（无需重新平衡）         |
| 2    | 节点是黑色的，其子节点是红色的         | 替换节点并将子节点重新着色为黑色 |
| 3    | 节点是黑色的，其子节点是黑色的（“双重黑”） | 使用下面的兄弟节点情况           |

双重黑修复情况

| 情况 | 描述                                     | 操作                                                         |
| ---- | ---------------------------------------- | ------------------------------------------------------------ |
| 1    | 兄弟节点是红色的                         | 旋转并重新着色，使兄弟节点变为黑色                           |
| 2    | 兄弟节点是黑色的，且其两个子节点都是黑色的 | 将兄弟节点重新着色为红色，将双重黑问题向上移动               |
| 3    | 兄弟节点是黑色的，近端子节点是红色的，远端子节点是黑色的 | 将兄弟节点向节点方向旋转，交换颜色                           |
| 4    | 兄弟节点是黑色的，远端子节点是红色的     | 旋转父节点，重新着色兄弟节点和父节点，将远端子节点设为黑色   |

#### 示例

从以下树中删除 `30`：

```
     20(B)
    /    \
 10(R)   30(R)
```

- 30 是红色 → 直接移除
- 没有违反任何性质

删除 `10`：

```
     20(B)
    /
   NIL
```

10 是红色 → 简单移除 → 树有效

删除 `20`：
树变为空 → 没问题

#### 分步示例

插入 [10, 20, 30, 15, 25]
删除 20

| 步骤 | 操作       | 违规情况               | 修复方法                                      |        |
| ---- | ---------- | ---------------------- | --------------------------------------------- | ------ |
| 1    | 删除 20    | 在 25 处出现双重黑     | 兄弟节点 10 为黑色，远端子节点为红色 → 情况 4 | 旋转   |
| 2    | 旋转后     | 所有性质恢复           | -                                             |        |

#### 简化代码（概念性）

C 语言（概念性）

```c
// 为简洁起见，此代码片段省略了完整的 BST 插入逻辑。

typedef enum { RED, BLACK } Color;
typedef struct Node {
    int key;
    Color color;
    struct Node *left, *right, *parent;
} Node;

// 工具函数：获取兄弟节点
Node* sibling(Node* n) {
    if (!n->parent) return NULL;
    return n == n->parent->left ? n->parent->right : n->parent->left;
}

void fix_delete(Node root, Node* x) {
    while (x != *root && (!x || x->color == BLACK)) {
        Node* s = sibling(x);
        if (x == x->parent->left) {
            if (s->color == RED) {
                s->color = BLACK;
                x->parent->color = RED;
                rotate_left(root, x->parent);
                s = x->parent->right;
            }
            if ((!s->left || s->left->color == BLACK) &&
                (!s->right || s->right->color == BLACK)) {
                s->color = RED;
                x = x->parent;
            } else {
                if (!s->right || s->right->color == BLACK) {
                    if (s->left) s->left->color = BLACK;
                    s->color = RED;
                    rotate_right(root, s);
                    s = x->parent->right;
                }
                s->color = x->parent->color;
                x->parent->color = BLACK;
                if (s->right) s->right->color = BLACK;
                rotate_left(root, x->parent);
                x = *root;
            }
        } else {
            // 右子节点的镜像逻辑
            if (s->color == RED) {
                s->color = BLACK;
                x->parent->color = RED;
                rotate_right(root, x->parent);
                s = x->parent->left;
            }
            if ((!s->left || s->left->color == BLACK) &&
                (!s->right || s->right->color == BLACK)) {
                s->color = RED;
                x = x->parent;
            } else {
                if (!s->left || s->left->color == BLACK) {
                    if (s->right) s->right->color = BLACK;
                    s->color = RED;
                    rotate_left(root, s);
                    s = x->parent->left;
                }
                s->color = x->parent->color;
                x->parent->color = BLACK;
                if (s->left) s->left->color = BLACK;
                rotate_right(root, x->parent);
                x = *root;
            }
        }
    }
    if (x) x->color = BLACK;
}
```

Python（简化伪代码）

```python
def fix_delete(root, x):
    while x != root and (not x or x.color == "B"):
        if x == x.parent.left:
            s = x.parent.right
            if s.color == "R":
                s.color, x.parent.color = "B", "R"
                root = rotate_left(root, x.parent)
                s = x.parent.right
            if all(c is None or c.color == "B" for c in [s.left, s.right]):
                s.color = "R"
                x = x.parent
            else:
                if not s.right or s.right.color == "B":
                    if s.left: s.left.color = "B"
                    s.color = "R"
                    root = rotate_right(root, s)
                    s = x.parent.right
                s.color, x.parent.color = x.parent.color, "B"
                if s.right: s.right.color = "B"
                root = rotate_left(root, x.parent)
                x = root
        else:
            # 镜像逻辑
            ...
    if x: x.color = "B"
    return root
```

#### 为什么这很重要

-   在删除后保持对数级高度
-   用于核心数据结构（`std::map`、`TreeSet`）
-   展示了作为软平衡方案的颜色逻辑
-   通过双重黑修复优雅地处理边界情况

#### 一个温和的证明（为什么它有效）

当一个黑色节点被移除时，某条路径会失去一个黑色计数。
修复情况通过旋转和重新着色来重新分配黑高。
每次循环迭代都将双重黑问题向上移动，最多 O(log n) 步。

#### 自己动手试试

1.  构建树 [10, 20, 30, 15, 25, 5]。删除 10（叶子节点）。
2.  删除 30（红色叶子节点） → 无需修复。
3.  删除 20（有一个子节点的黑色节点） → 重新着色修复。
4.  可视化每种情况：兄弟节点为红色，兄弟节点为黑色且有红色子节点。

#### 测试用例

| 插入序列                | 删除 | 触发情况 | 修复方法           |
| ----------------------- | ---- | -------- | ------------------ |
| [10,20,30]              | 20   | 情况 4   | 旋转并重新着色     |
| [7,3,18,10,22,8,11,26] | 18   | 情况 2   | 重新着色兄弟节点   |
| [10,5,1]                | 5    | 情况 1   | 旋转父节点         |

#### 复杂度

| 操作   | 时间复杂度 | 空间复杂度       |
| ------ | ---------- | ---------------- |
| 删除   | O(log n)   | O(1) 次旋转      |
| 插入   | O(log n)   | O(1)             |
| 搜索   | O(log n)   | O(h)（树的高度） |

红黑树的删除就像一次精心的调校，一次调整一种颜色，和谐在每条路径上得以恢复。
### 235 伸展树访问

伸展树（Splay Tree）是一种自调整的二叉搜索树，它通过一系列树旋转（即伸展操作）将频繁访问的元素移到靠近根节点的位置。与 AVL 树或红黑树不同，它不保持严格的平衡，但保证了访问、插入和删除操作的摊还时间复杂度为 O(log n)。

#### 我们要解决什么问题？

在许多工作负载中，某些元素的访问频率远高于其他元素。普通的二叉搜索树（BST）不会给"热点"键带来任何优势，而平衡树虽然能保持树的形状，但不会考虑访问的最近性。伸展树针对时间局部性进行了优化：最近访问过的项目会被移到靠近根节点的位置，从而使后续的重复访问更快。

#### 它是如何工作的（通俗解释）？

每当你访问一个节点（通过搜索、插入或删除操作）时，都会执行一次伸展操作，根据该节点的位置及其与父节点的关系，反复将其向根节点旋转。

三种旋转模式：

| 情况 | 结构 | 操作 |
| :--- | :--- | :--- |
| Zig（单旋） | 节点是根节点的子节点 | 单次旋转 |
| Zig-Zig（一字型） | 节点和父节点都是左孩子或都是右孩子 | 双旋转（先旋转父节点，再旋转祖父节点） |
| Zig-Zag（之字形） | 节点和父节点分别是左孩子和右孩子（或相反） | 双旋转（将节点向上旋转两次） |

伸展操作后，被访问的节点成为新的根节点。

#### 示例

访问序列：10, 20, 30

1.  插入 10（根节点）
2.  插入 20 → 20 成为 10 的右孩子
3.  访问 20 → Zig 旋转 → 20 成为根节点
4.  插入 30 → 30 成为 20 的右孩子
5.  访问 30 → Zig 旋转 → 30 成为根节点

最终树形（所有访问后）：

```
   30
  /
20
/
10
```

频繁使用的节点会自动上升到顶部。

#### 逐步示例

从键值 [5, 3, 8, 1, 4, 7, 9] 开始。
访问键值 1。

| 步骤 | 操作 | 情况 | 旋转 | 新根节点 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 访问 1 | Zig-Zig（左-左） | 先旋转 3，再旋转 5 | 1 |
| 2 | 伸展后 | - | - | 1 |

最终树：1 成为根节点，为下一次访问缩短了路径。

#### 微型代码（简易版本）

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int key;
    struct Node *left, *right;
} Node;

Node* new_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key;
    n->left = n->right = NULL;
    return n;
}

Node* rotate_right(Node* x) {
    Node* y = x->left;
    x->left = y->right;
    y->right = x;
    return y;
}

Node* rotate_left(Node* x) {
    Node* y = x->right;
    x->right = y->left;
    y->left = x;
    return y;
}

Node* splay(Node* root, int key) {
    if (!root || root->key == key) return root;

    // 键值在左子树中
    if (key < root->key) {
        if (!root->left) return root;
        if (key < root->left->key) {
            root->left->left = splay(root->left->left, key);
            root = rotate_right(root);
        } else if (key > root->left->key) {
            root->left->right = splay(root->left->right, key);
            if (root->left->right)
                root->left = rotate_left(root->left);
        }
        return root->left ? rotate_right(root) : root;
    }
    // 键值在右子树中
    else {
        if (!root->right) return root;
        if (key > root->right->key) {
            root->right->right = splay(root->right->right, key);
            root = rotate_left(root);
        } else if (key < root->right->key) {
            root->right->left = splay(root->right->left, key);
            if (root->right->left)
                root->right = rotate_right(root->right);
        }
        return root->right ? rotate_left(root) : root;
    }
}

Node* insert(Node* root, int key) {
    if (!root) return new_node(key);
    root = splay(root, key);
    if (root->key == key) return root;
    Node* n = new_node(key);
    if (key < root->key) {
        n->right = root;
        n->left = root->left;
        root->left = NULL;
    } else {
        n->left = root;
        n->right = root->right;
        root->right = NULL;
    }
    return n;
}

void inorder(Node* r) {
    if (!r) return;
    inorder(r->left);
    printf("%d ", r->key);
    inorder(r->right);
}

int main(void) {
    Node* root = NULL;
    int keys[] = {5, 3, 8, 1, 4, 7, 9};
    for (int i = 0; i < 7; i++) root = insert(root, keys[i]);
    root = splay(root, 1);
    inorder(root);
    printf("\n");
}
```

Python

```python
class Node:
    def __init__(self, key):
        self.key = key
        self.left = self.right = None

def rotate_right(x):
    y = x.left
    x.left = y.right
    y.right = x
    return y

def rotate_left(x):
    y = x.right
    x.right = y.left
    y.left = x
    return y

def splay(root, key):
    if not root or root.key == key:
        return root
    if key < root.key:
        if not root.left:
            return root
        if key < root.left.key:
            root.left.left = splay(root.left.left, key)
            root = rotate_right(root)
        elif key > root.left.key:
            root.left.right = splay(root.left.right, key)
            if root.left.right:
                root.left = rotate_left(root.left)
        return rotate_right(root) if root.left else root
    else:
        if not root.right:
            return root
        if key > root.right.key:
            root.right.right = splay(root.right.right, key)
            root = rotate_left(root)
        elif key < root.right.key:
            root.right.left = splay(root.right.left, key)
            if root.right.left:
                root.right = rotate_right(root.right)
        return rotate_left(root) if root.right else root

def insert(root, key):
    if not root:
        return Node(key)
    root = splay(root, key)
    if root.key == key:
        return root
    n = Node(key)
    if key < root.key:
        n.right = root
        n.left = root.left
        root.left = None
    else:
        n.left = root
        n.right = root.right
        root.right = None
    return n

def inorder(r):
    if not r: return
    inorder(r.left)
    print(r.key, end=" ")
    inorder(r.right)

root = None
for k in [5,3,8,1,4,7,9]:
    root = insert(root, k)
root = splay(root, 1)
inorder(root)
```

#### 为什么它很重要

-   摊还 O(log n) 的性能
-   动态适应访问模式
-   非常适合缓存、文本编辑器、网络路由表
-   逻辑简单：无需跟踪高度或颜色

#### 一个温和的证明（为什么它有效）

每次伸展操作可能需要 O(h) 时间，但在多次操作中，其摊还成本是 O(log n)。频繁访问的元素会被保持在靠近根节点的位置，从而改善未来的操作性能。

#### 自己动手试试

1.  构建树 [5, 3, 8, 1, 4, 7, 9]
2.  访问 1 → 观察 Zig-Zig 旋转
3.  访问 8 → 观察 Zig-Zag 旋转
4.  插入 10 → 检查重新平衡
5.  重复访问 7 → 移动到根节点

#### 测试用例

| 序列 | 访问 | 情况 | 操作后根节点 |
| :--- | :--- | :--- | :--- |
| [5,3,8,1,4] | 1 | Zig-Zig | 1 |
| [10,20,30] | 30 | Zig | 30 |
| [5,3,8,1,4] | 4 | Zig-Zag | 4 |

#### 复杂度

| 操作 | 时间复杂度 | 空间复杂度 |
| :--- | :--- | :--- |
| 访问 | 摊还 O(log n) | O(h) 递归 |
| 插入 | 摊还 O(log n) | O(h) |
| 删除 | 摊还 O(log n) | O(h) |

伸展树就像记忆一样，你接触得越多的东西，它离你就越近。
### 236 Treap 插入

Treap（Tree + Heap）是一种巧妙的混合结构：它既是按键排序的二叉搜索树，又是按优先级排序的堆。每个节点包含一个键和一个随机优先级。通过将键的有序性与随机的堆优先级相结合，Treap 能够*在平均情况下保持平衡*，无需像 AVL 树那样进行严格的旋转，也无需像红黑树那样维护颜色规则。

#### 我们要解决什么问题？

我们想要一种简单的平衡搜索树，期望性能为 O(log n)，但无需显式地维护高度或颜色属性。Treap 通过分配随机优先级来解决这个问题，从而*在期望上*保持结构平衡。

#### 工作原理（通俗解释）

每个节点 `(键, 优先级)` 必须满足：

- BST 属性：`键(左子树) < 键 < 键(右子树)`
- 堆属性：`优先级(父节点) < 优先级(子节点)`（最小堆或最大堆约定均可）

首先像普通 BST 一样按键插入节点，如果优先级违反堆属性，则通过旋转来修复。

| 步骤 | 规则                                                   |
| ---- | ------------------------------------------------------ |
| 1    | 按键插入节点（类似 BST）                               |
| 2    | 分配一个随机优先级                                     |
| 3    | 当父节点优先级 > 节点优先级时 → 将节点向上旋转         |

这种随机化确保了期望的对数高度。

#### 示例

插入序列（键, 优先级）：

| 步骤 | 键       | 优先级                        | 结构                       |
| ---- | -------- | ----------------------------- | -------------------------- |
| 1    | (50, 15) | 根节点                        | 50                         |
| 2    | (30, 10) | 优先级更小 → 右旋              | 30(根) → 50(右子节点)      |
| 3    | (70, 25) | 保持在 50 的右侧              | 平衡                       |

结果（按键是 BST，按优先级是最小堆）：

```
     (30,10)
         \
         (50,15)
             \
             (70,25)
```

#### 分步示例

插入键： [40, 20, 60, 10, 30, 50, 70]
优先级： [80, 90, 70, 100, 85, 60, 75]

| 键  | 优先级 | 旋转操作               | 结果       |
| --- | ------ | ---------------------- | ---------- |
| 40  | 80     | 根节点                 | 40         |
| 20  | 90     | 无                     | 左子节点   |
| 60  | 70     | 左旋（修复堆属性）     | 60 为根    |
| 10  | 100    | 无                     | 叶子节点   |
| 30  | 85     | 无                     | 在 20 下方 |
| 50  | 60     | 左旋                   | 50 为根    |
| 70  | 75     | 无                     | 叶子节点   |

#### 精简代码（简易版本）

C

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct Node {
    int key, priority;
    struct Node *left, *right;
} Node;

Node* new_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key;
    n->priority = rand() % 100; // 随机优先级
    n->left = n->right = NULL;
    return n;
}

Node* rotate_right(Node* y) {
    Node* x = y->left;
    y->left = x->right;
    x->right = y;
    return x;
}

Node* rotate_left(Node* x) {
    Node* y = x->right;
    x->right = y->left;
    y->left = x;
    return y;
}

Node* insert(Node* root, int key) {
    if (!root) return new_node(key);

    if (key < root->key) {
        root->left = insert(root->left, key);
        if (root->left->priority < root->priority)
            root = rotate_right(root);
    } else if (key > root->key) {
        root->right = insert(root->right, key);
        if (root->right->priority < root->priority)
            root = rotate_left(root);
    }
    return root;
}

void inorder(Node* root) {
    if (!root) return;
    inorder(root->left);
    printf("(%d,%d) ", root->key, root->priority);
    inorder(root->right);
}

int main(void) {
    srand(time(NULL));
    Node* root = NULL;
    int keys[] = {40, 20, 60, 10, 30, 50, 70};
    for (int i = 0; i < 7; i++)
        root = insert(root, keys[i]);
    inorder(root);
    printf("\n");
}
```

Python

```python
import random

class Node:
    def __init__(self, key):
        self.key = key
        self.priority = random.randint(1, 100)
        self.left = self.right = None

def rotate_right(y):
    x = y.left
    y.left = x.right
    x.right = y
    return x

def rotate_left(x):
    y = x.right
    x.right = y.left
    y.left = x
    return y

def insert(root, key):
    if not root:
        return Node(key)
    if key < root.key:
        root.left = insert(root.left, key)
        if root.left.priority < root.priority:
            root = rotate_right(root)
    elif key > root.key:
        root.right = insert(root.right, key)
        if root.right.priority < root.priority:
            root = rotate_left(root)
    return root

def inorder(root):
    if not root: return
    inorder(root.left)
    print(f"({root.key},{root.priority})", end=" ")
    inorder(root.right)

root = None
for k in [40, 20, 60, 10, 30, 50, 70]:
    root = insert(root, k)
inorder(root)
```

#### 为什么它重要

- 实现简单，且期望上是平衡的
- 是 AVL / 红黑树的概率性替代方案
- 适用于随机化数据结构和笛卡尔树应用
- 无需显式跟踪高度或颜色

#### 一个温和的证明（为什么它有效）

随机优先级意味着随机结构；节点位于树高层的概率随深度呈指数级下降。因此，期望高度以高概率为 O(log n)。

#### 自己动手试试

1.  插入键 [10, 20, 30, 40] 并分配随机优先级。
2.  观察随机结构（并非按插入顺序排序）。
3.  更改优先级生成器 → 测试偏斜度。
4.  可视化 BST 和堆不变式。

#### 测试用例

| 键           | 优先级      | 最终根节点 | 高度     |
| ----------- | ---------- | ---------- | -------- |
| [10,20,30]  | [5,3,4]    | 20         | 2        |
| [5,2,8,1,3] | 随机       | 变化       | ~log n   |
| [1..100]    | 随机       | 平衡       | O(log n) |

#### 复杂度

| 操作   | 时间              | 空间          |
| ------ | ----------------- | ------------- |
| 插入   | O(log n) 期望     | O(h) 递归     |
| 搜索   | O(log n) 期望     | O(h)          |
| 删除   | O(log n) 期望     | O(h)          |

Treap 随着机会的节奏起舞，将有序性与随机性相结合，从而保持灵活与公平。
### 237 Treap 删除

从 Treap 中删除融合了两种方法的优点：使用 BST 搜索定位节点，并基于堆的旋转来移除节点同时保持平衡。我们不是显式地重新平衡，而是将目标节点向下旋转，直到它变成叶子节点，然后将其移除。

#### 我们要解决什么问题？

我们希望在删除一个键的同时保持以下两个属性：

- BST 属性：键有序
- 堆属性：优先级遵循最小堆/最大堆规则

Treap 通过将目标节点向下旋转，直到它最多只有一个子节点，然后直接移除它来处理这个问题。

#### 它是如何工作的？（通俗解释）

1.  通过键搜索节点（BST 搜索）。
2.  找到后，将其向下旋转，直到它拥有 ≤ 1 个子节点：
   *   如果右子节点的优先级 < 左子节点的优先级，则左旋
   *   否则右旋
3.  一旦它成为叶子节点或单子节点，就将其移除。

由于优先级是随机的，树*在期望上*保持平衡。

#### 示例

Treap（键，优先级）：

```
      (40,20)
     /      \
 (30,10)   (50,25)
```

删除 40：

- 比较子节点：(30,10) < (50,25) → 对 40 进行右旋
- 新根节点 (30,10)
- 40 现在成为右子节点 → 如果需要则继续旋转 → 最终成为叶子节点 → 删除

新的结构保持了 BST + 堆属性。

#### 分步示例

起始节点：

| 键  | 优先级 |
| --- | ------ |
| 40  | 50     |
| 20  | 60     |
| 60  | 70     |
| 10  | 90     |
| 30  | 80     |
| 50  | 65     |
| 70  | 75     |

删除 40：

| 步骤 | 节点 | 操作                         | 旋转类型       | 根节点           |
| ---- | ---- | ---------------------------- | -------------- | ---------------- |
| 1    | 40   | 比较子节点优先级             | 左子节点优先级更高 | 右旋             |
| 2    | 30   | 现在是 40 的父节点           | 40 被降级      | 如果需要则继续   |
| 3    | 40   | 成为叶子节点                 | 移除           | 30               |

最终根节点：(30,80)
所有属性均保持。

#### 精简代码（简易版本）

C

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct Node {
    int key, priority;
    struct Node *left, *right;
} Node;

Node* new_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key;
    n->priority = rand() % 100;
    n->left = n->right = NULL;
    return n;
}

Node* rotate_right(Node* y) {
    Node* x = y->left;
    y->left = x->right;
    x->right = y;
    return x;
}

Node* rotate_left(Node* x) {
    Node* y = x->right;
    x->right = y->left;
    y->left = x;
    return y;
}

Node* insert(Node* root, int key) {
    if (!root) return new_node(key);
    if (key < root->key) {
        root->left = insert(root->left, key);
        if (root->left->priority < root->priority)
            root = rotate_right(root);
    } else if (key > root->key) {
        root->right = insert(root->right, key);
        if (root->right->priority < root->priority)
            root = rotate_left(root);
    }
    return root;
}

Node* delete(Node* root, int key) {
    if (!root) return NULL;

    if (key < root->key)
        root->left = delete(root->left, key);
    else if (key > root->key)
        root->right = delete(root->right, key);
    else {
        // 找到要删除的节点
        if (!root->left && !root->right) {
            free(root);
            return NULL;
        } else if (!root->left)
            root = rotate_left(root);
        else if (!root->right)
            root = rotate_right(root);
        else if (root->left->priority < root->right->priority)
            root = rotate_right(root);
        else
            root = rotate_left(root);

        root = delete(root, key);
    }
    return root;
}

void inorder(Node* root) {
    if (!root) return;
    inorder(root->left);
    printf("(%d,%d) ", root->key, root->priority);
    inorder(root->right);
}

int main(void) {
    srand(time(NULL));
    Node* root = NULL;
    int keys[] = {40, 20, 60, 10, 30, 50, 70};
    for (int i = 0; i < 7; i++)
        root = insert(root, keys[i]);

    printf("删除前:\n");
    inorder(root);
    printf("\n");

    root = delete(root, 40);

    printf("删除 40 后:\n");
    inorder(root);
    printf("\n");
}
```

Python

```python
import random

class Node:
    def __init__(self, key):
        self.key = key
        self.priority = random.randint(1, 100)
        self.left = self.right = None

def rotate_right(y):
    x = y.left
    y.left = x.right
    x.right = y
    return x

def rotate_left(x):
    y = x.right
    x.right = y.left
    y.left = x
    return y

def insert(root, key):
    if not root:
        return Node(key)
    if key < root.key:
        root.left = insert(root.left, key)
        if root.left.priority < root.priority:
            root = rotate_right(root)
    elif key > root.key:
        root.right = insert(root.right, key)
        if root.right.priority < root.priority:
            root = rotate_left(root)
    return root

def delete(root, key):
    if not root:
        return None
    if key < root.key:
        root.left = delete(root.left, key)
    elif key > root.key:
        root.right = delete(root.right, key)
    else:
        if not root.left and not root.right:
            return None
        elif not root.left:
            root = rotate_left(root)
        elif not root.right:
            root = rotate_right(root)
        elif root.left.priority < root.right.priority:
            root = rotate_right(root)
        else:
            root = rotate_left(root)
        root = delete(root, key)
    return root

def inorder(root):
    if not root: return
    inorder(root.left)
    print(f"({root.key},{root.priority})", end=" ")
    inorder(root.right)

root = None
for k in [40,20,60,10,30,50,70]:
    root = insert(root, k)
print("删除前:", end=" ")
inorder(root)
print()
root = delete(root, 40)
print("删除后:", end=" ")
inorder(root)
```

#### 为什么这很重要

- 无需显式重新平衡，旋转由堆优先级自然产生
- 期望 O(log n) 的删除复杂度
- 随机化搜索树的自然、简单结构
- 用于随机化算法、动态集合和顺序统计

#### 一个温和的证明（为什么它有效）

在每一步，节点都向优先级较小的子节点方向旋转，从而保持堆属性。
由于独立的随机优先级，期望高度保持为 O(log n)。

#### 自己动手试试

1.  用 [10, 20, 30, 40, 50] 构建一个 treap。
2.  删除 30 → 观察旋转到叶子的过程。
3.  删除 10（根节点）→ 旋转会选择优先级较小的子节点。
4.  重复删除操作，验证 BST + 堆不变量是否保持。

#### 测试用例

| 键序列                     | 删除 | 旋转类型   | 是否平衡 | 删除后根节点 |
| ------------------------ | ---- | ---------- | -------- | ------------ |
| [40,20,60,10,30,50,70]   | 40   | 右旋, 左旋 | ✅        | 30           |
| [10,20,30]               | 20   | 左旋       | ✅        | 10           |
| [50,30,70,20,40,60,80]   | 30   | 右旋       | ✅        | 50           |

#### 复杂度

| 操作   | 时间复杂度         | 空间复杂度       |
| ------ | ------------------ | ---------------- |
| 删除   | 期望 O(log n)      | O(h) 递归        |
| 插入   | 期望 O(log n)      | O(h)             |
| 搜索   | 期望 O(log n)      | O(h)             |

Treap 优雅地执行删除，通过旋转让目标节点逐渐下移，直到像一片落叶般消失，留下平衡的结构。
### 238 重量平衡树

重量平衡树（Weight Balanced Tree，WBT）不是通过高度或颜色来维持平衡，而是通过子树的大小（或称“重量”）来维持平衡。每个节点都会追踪其左子树和右子树中包含的元素数量，并通过要求它们的重量保持在一个恒定比例内来强制维持平衡。

#### 我们要解决什么问题？

在普通的二叉搜索树（BST）中，如果按键的排序顺序插入，不平衡性可能会加剧，导致性能下降至 O(n)。重量平衡树通过将子树大小保持在一个安全范围内来解决这个问题。当你需要依赖于大小而非高度的顺序统计（例如“查找第 k 个元素”）或分裂/合并操作时，它们特别有用。

#### 工作原理（通俗解释）

每个节点维护一个重量，即其子树中的节点总数。
插入或删除时，更新重量并检查平衡条件：

如果对于某个平衡常数 α（例如 0.7），满足：

```
weight(left) ≤ α × weight(node)
weight(right) ≤ α × weight(node)
```

则认为树是平衡的。

如果不满足，则执行旋转（类似于 AVL 树）以恢复平衡。

| 操作     | 步骤                                                                 |
| -------- | -------------------------------------------------------------------- |
| 插入     | 像 BST 一样插入 → 更新重量 → 如果比例被破坏 → 旋转                    |
| 删除     | 移除节点 → 更新重量 → 如果比例被破坏 → 重建或旋转                     |
| 搜索     | BST 搜索（重量在排名查询中指导决策）                                  |

因为重量反映了实际大小，所以树能确保期望高度为 O(log n)。

#### 示例

让我们使用 α = 0.7

插入 [10, 20, 30, 40, 50]

| 步骤 | 插入 | 左子树重量 | 右子树重量 | 平衡？ | 修复         |
| ---- | ---- | ---------- | ---------- | ------ | ------------ |
| 1    | 10   | 0          | 0          | ✅      | -            |
| 2    | 20   | 1          | 0          | ✅      | -            |
| 3    | 30   | 2          | 0          | ❌      | 左旋         |
| 4    | 40   | 3          | 0          | ❌      | 左旋         |
| 5    | 50   | 4          | 0          | ❌      | 左旋         |

一旦左右比例超过 α，树通过旋转保持平衡。

#### 分步示例

使用 α = 0.7 插入 [1, 2, 3, 4, 5]

| 步骤 | 操作     | 重量   | 平衡？ | 旋转       |
| ---- | -------- | ------ | ------ | ---------- |
| 1    | 插入 1   | (0,0)  | ✅      | -          |
| 2    | 插入 2   | (1,0)  | ✅      | -          |
| 3    | 插入 3   | (2,0)  | ❌      | 左旋       |
| 4    | 插入 4   | (3,0)  | ❌      | 左旋       |
| 5    | 插入 5   | (4,0)  | ❌      | 左旋       |

基于重量，树保持高度平衡。

#### 简易代码（简化版）

C 语言（简化版）

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int key;
    int size; // 重量 = 子树大小
    struct Node *left, *right;
} Node;

int size(Node* n) { return n ? n->size : 0; }

Node* new_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key;
    n->size = 1;
    n->left = n->right = NULL;
    return n;
}

Node* rotate_left(Node* x) {
    Node* y = x->right;
    x->right = y->left;
    y->left = x;
    x->size = 1 + size(x->left) + size(x->right);
    y->size = 1 + size(y->left) + size(y->right);
    return y;
}

Node* rotate_right(Node* y) {
    Node* x = y->left;
    y->left = x->right;
    x->right = y;
    y->size = 1 + size(y->left) + size(y->right);
    x->size = 1 + size(x->left) + size(x->right);
    return x;
}

double alpha = 0.7;

int balanced(Node* n) {
    if (!n) return 1;
    int l = size(n->left), r = size(n->right);
    return l <= alpha * n->size && r <= alpha * n->size;
}

Node* insert(Node* root, int key) {
    if (!root) return new_node(key);
    if (key < root->key) root->left = insert(root->left, key);
    else if (key > root->key) root->right = insert(root->right, key);
    root->size = 1 + size(root->left) + size(root->right);
    if (!balanced(root)) {
        if (size(root->left) > size(root->right))
            root = rotate_right(root);
        else
            root = rotate_left(root);
    }
    return root;
}

void inorder(Node* n) {
    if (!n) return;
    inorder(n->left);
    printf("%d(%d) ", n->key, n->size);
    inorder(n->right);
}

int main(void) {
    Node* root = NULL;
    int keys[] = {10, 20, 30, 40, 50};
    for (int i = 0; i < 5; i++)
        root = insert(root, keys[i]);
    inorder(root);
    printf("\n");
}
```

Python（简化版）

```python
class Node:
    def __init__(self, key):
        self.key = key
        self.size = 1
        self.left = self.right = None

def size(n): return n.size if n else 0

def update(n):
    if n:
        n.size = 1 + size(n.left) + size(n.right)

def rotate_left(x):
    y = x.right
    x.right = y.left
    y.left = x
    update(x)
    update(y)
    return y

def rotate_right(y):
    x = y.left
    y.left = x.right
    x.right = y
    update(y)
    update(x)
    return x

alpha = 0.7

def balanced(n):
    return not n or (size(n.left) <= alpha * n.size and size(n.right) <= alpha * n.size)

def insert(root, key):
    if not root: return Node(key)
    if key < root.key: root.left = insert(root.left, key)
    elif key > root.key: root.right = insert(root.right, key)
    update(root)
    if not balanced(root):
        if size(root.left) > size(root.right):
            root = rotate_right(root)
        else:
            root = rotate_left(root)
    return root

def inorder(n):
    if not n: return
    inorder(n.left)
    print(f"{n.key}({n.size})", end=" ")
    inorder(n.right)

root = None
for k in [10,20,30,40,50]:
    root = insert(root, k)
inorder(root)
```

#### 为什么它很重要？

- 基于真实的子树大小而非高度进行平衡
- 非常适合顺序统计（`kth`、`rank`）
- 自然地支持分裂、合并和范围查询
- 无需随机性的确定性平衡

#### 一个温和的证明（为什么它有效）

保持重量比例保证了树高为对数级别。
如果 `α` < 1，每个子树拥有 ≤ α × n 个节点，
因此高度 ≤ log₁/α(n) → O(log n)。

#### 自己动手试试

1.  使用 α = 0.7 构建 [10, 20, 30, 40, 50]。
2.  打印每个节点的大小。
3.  删除 20 → 通过旋转重新平衡。
4.  尝试 α = 0.6 和 α = 0.8 → 比较树形。

#### 测试用例

| 序列               | α   | 平衡？ | 高度     |
| ------------------ | --- | ------ | -------- |
| [10,20,30,40,50]   | 0.7 | ✅      | log n    |
| [1..100]           | 0.7 | ✅      | O(log n) |
| [sorted 1..10]     | 0.6 | ✅      | ~4       |

#### 复杂度

| 操作   | 时间复杂度 | 空间复杂度 |
| ------ | ---------- | ---------- |
| 插入   | O(log n)   | O(h)       |
| 删除   | O(log n)   | O(h)       |
| 搜索   | O(log n)   | O(h)       |

重量平衡树就像一个走钢丝的人，无论序列如何展开，它总是调整自己的姿态以保持完美的平衡。
### 239 替罪羊树重建

替罪羊树是一种通过偶尔重建整棵子树来维持平衡的二叉搜索树。  
它不会在每次插入时执行旋转操作，而是监控节点的深度。  
当某个节点的深度超过允许的限制时，算法会定位一个*替罪羊*祖先节点，该节点的子树过于不平衡，然后将该子树展平为一个有序列表，并将其重建为一棵完美平衡的树。

给定平衡参数 $\alpha$ 在 $(0.5, 1)$ 范围内，替罪羊树保证：
- 插入和删除的**摊还时间复杂度**为 $O(\log n)$
- **最坏情况高度**为 $O(\log n)$

#### 我们要解决什么问题？

标准的二叉搜索树可能会变得倾斜，性能退化到 O(n)。像 AVL 树和红黑树这样的旋转树维持严格的局部不变量，但增加了每次更新的开销。替罪羊树选择了一条中间道路：

- 大多数时候什么都不做
- 偶尔在超过全局界限时重建整棵子树

**目标**  
在保持代码简单且避免在关键路径上进行旋转的同时，将高度维持在接近 $\log_{1/\alpha} n$。

#### 工作原理（通俗解释）

**参数和不变量**

- 选择 $\alpha \in (0.5, 1)$，例如 $\alpha = \tfrac{2}{3}$
- 维护 `n` = 当前大小 和 `n_max` = 自上次全局重建以来的最大大小
- 高度界限：如果一次插入的深度超过 $\lfloor \log_{1/\alpha} n \rfloor$，则树太高

**插入算法**

1.  像普通二叉搜索树一样插入，并跟踪路径长度 `depth`。
2.  如果 `depth` 在允许的界限内，则停止。
3.  否则，向上遍历以找到替罪羊节点 `s`，使得
    $$
    \max\big(\text{size}(s.\text{left}),\ \text{size}(s.\text{right})\big) > \alpha \cdot \text{size}(s)
    $$
4.  以该子树大小线性的时间复杂度，将根节点为 `s` 的子树重建为一棵完美平衡的二叉搜索树。

**删除算法**

- 执行普通的二叉搜索树删除。
- 递减 `n`。
- 如果 $n < \alpha \cdot n_{\text{max}}$，则重建整棵树并设置 $n_{\text{max}} = n$。

**为什么重建有效**

- 重建过程执行中序遍历，将节点收集到有序数组中，然后从该数组构造平衡的二叉搜索树。
- 重建操作不频繁，其成本被分摊到许多廉价的插入和删除操作中，从而确保平均对数时间复杂度。

#### 示例演练

令 $\alpha = \tfrac{2}{3}$。

按升序插入键 `[10, 20, 30, 40, 50, 60]`。

| 步骤 | 操作       | 深度 vs 界限         | 找到的替罪羊       | 重建               |
| ---- | ---------- | -------------------- | ------------------ | ------------------ |
| 1    | 插入 10    | 深度 0 在界限内      | 无                 | 否                 |
| 2    | 插入 20    | 深度 1 在界限内      | 无                 | 否                 |
| 3    | 插入 30    | 深度 2 在界限内      | 无                 | 否                 |
| 4    | 插入 40    | 深度 3 超过界限      | 祖先违反 α         | 重建该子树         |
| 5    | 插入 50    | 可能在界限内         | 无                 | 否                 |
| 6    | 插入 60    | 如果深度超过界限     | 找到祖先           | 重建子树           |

尽管输入是有序的，但偶尔的重建会产生一棵近乎完美的二叉搜索树。

#### 精简代码（简易版本）

Python（简洁参考实现）

```python
from math import log, floor

class Node:
    __slots__ = ("key", "left", "right", "size")
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.size = 1

def size(x): return x.size if x else 0
def update(x): 
    if x: x.size = 1 + size(x.left) + size(x.right)

def flatten_inorder(x, arr):
    if not x: return
    flatten_inorder(x.left, arr)
    arr.append(x)
    flatten_inorder(x.right, arr)

def build_balanced(nodes, lo, hi):
    if lo >= hi: return None
    mid = (lo + hi) // 2
    root = nodes[mid]
    root.left = build_balanced(nodes, lo, mid)
    root.right = build_balanced(nodes, mid + 1, hi)
    update(root)
    return root

class ScapegoatTree:
    def __init__(self, alpha=2/3):
        self.alpha = alpha
        self.root = None
        self.n = 0
        self.n_max = 0

    def _log_alpha(self, n):
        # 高度界限 floor(log_{1/alpha} n)
        if n <= 1: return 0
        return floor(log(n, 1 / self.alpha))

    def _find_and_rebuild(self, path):
        # path 是从根节点到插入节点的节点列表
        for i in range(len(path) - 1, -1, -1):
            x = path[i]
            l, r = size(x.left), size(x.right)
            if max(l, r) > self.alpha * size(x):
                # 重建以 x 为根的子树并重新连接到父节点
                nodes = []
                flatten_inorder(x, nodes)
                new_sub = build_balanced(nodes, 0, len(nodes))
                if i == 0:
                    self.root = new_sub
                else:
                    p = path[i - 1]
                    if p.left is x: p.left = new_sub
                    else: p.right = new_sub
                    # 从父节点开始向上修正大小
                    for j in range(i - 1, -1, -1):
                        update(path[j])
                return

    def insert(self, key):
        self.n += 1
        self.n_max = max(self.n_max, self.n)
        if not self.root:
            self.root = Node(key)
            return

        # 标准二叉搜索树插入，同时跟踪路径
        path = []
        cur = self.root
        while cur:
            path.append(cur)
            if key < cur.key:
                if cur.left: cur = cur.left
                else:
                    cur.left = Node(key)
                    path.append(cur.left)
                    break
            elif key > cur.key:
                if cur.right: cur = cur.right
                else:
                    cur.right = Node(key)
                    path.append(cur.right)
                    break
            else:
                # 忽略重复键
                self.n -= 1
                return
        # 更新大小
        for node in reversed(path[:-1]):
            update(node)

        # 深度检查及可能的重建
        if len(path) - 1 > self._log_alpha(self.n):
            self._find_and_rebuild(path)

    def _join_left_max(self, t):
        # 从子树 t 中移除并返回最大节点，以及新的子树
        if not t.right:
            return t, t.left
        m, t.right = self._join_left_max(t.right)
        update(t)
        return m, t

    def delete(self, key):
        # 标准二叉搜索树删除
        def _del(x, key):
            if not x: return None, False
            if key < x.key:
                x.left, removed = _del(x.left, key)
            elif key > x.key:
                x.right, removed = _del(x.right, key)
            else:
                removed = True
                if not x.left: return x.right, True
                if not x.right: return x.left, True
                # 用前驱节点替换
                m, x.left = self._join_left_max(x.left)
                m.left, m.right = x.left, x.right
                x = m
            update(x)
            return x, removed

        self.root, removed = _del(self.root, key)
        if not removed: return
        self.n -= 1
        if self.n < self.alpha * self.n_max:
            # 全局重建
            nodes = []
            flatten_inorder(self.root, nodes)
            self.root = build_balanced(nodes, 0, len(nodes))
            self.n_max = self.n
```

C（重建思路概述）

```c
// 仅作草图：展示重建路径
// 1) 执行二叉搜索树插入，同时记录路径栈
// 2) 如果深度 > 界限，沿路径向上查找替罪羊节点
// 3) 中序遍历复制子树节点到数组
// 4) 递归地从该数组重建平衡子树
// 5) 将重建的子树重新连接到父节点，并沿路径修正大小
```

#### 为什么它重要

- 采用简单平衡策略，偶尔但强力地重建子树
- 非常适合插入操作成批到达且不希望频繁旋转的工作负载
- 保证高度为 $O(\log n)$，且可通过 $\alpha$ 调节常数因子
- 基于中序遍历的重建过程使得有序操作高效且易于实现

#### 温和的证明（为什么它有效）

令 $\alpha \in (0.5, 1)$。

- **高度界限**：如果高度超过 $\log_{1/\alpha} n$，则必然存在替罪羊节点，因为某个祖先节点必定违反 $\max\{|L|,\ |R|\} \le \alpha\,|T|$。
- **摊还分析**：每个节点参与 $O(1)$ 次重建，每次重建工作量为 $\Theta(\text{其子树大小})$，因此 $m$ 次操作的总成本为 $O(m \log n)$。
- **删除规则**：当 $n < \alpha\,n_{\max}$ 时进行重建，防止松弛度累积。

#### 自己动手试试

1.  插入升序键以触发重建，设置 $\alpha = \tfrac{2}{3}$
2.  删除许多键，使得 $n < \alpha\,n_{\max}$，观察全局重建的触发
3.  尝试不同的 $\alpha$ 值
4.  使用存储的子树大小添加顺序统计查询

#### 测试用例

| 操作         | 输入                           | α    | 预期结果                                     |
| ------------ | ------------------------------ | ---- | -------------------------------------------- |
| 插入升序序列 | [1..1000]                      | 0.66 | 高度保持 O(log n)，并定期重建                |
| 混合操作     | 随机插入和删除                 | 0.7  | 每次操作摊还 O(log n)                        |
| 收缩检查     | 插入 1..200，删除 101..200     | 0.7  | 当大小降至 α n_max 以下时触发全局重建        |
| 重复插入     | 两次插入 42                    | 0.66 | 第二次插入后大小不变                         |

**边界情况**

- 重建空子树或单节点子树是无操作
- 重复键根据策略被忽略或处理
- 选择 $\alpha > 0.5$ 以保证替罪羊节点的存在

#### 复杂度

| 操作         | 时间复杂度                     | 空间复杂度                     |
| ------------ | ------------------------------ | ------------------------------ |
| 搜索         | 最坏 O(h)，摊还 O(log n)       | O(1) 额外空间                  |
| 插入         | 摊还 O(log n)                  | O(1) 额外空间加上重建缓冲区     |
| 删除         | 摊还 O(log n)                  | O(1) 额外空间加上重建缓冲区     |
| 重建子树     | O(k)，k 为子树节点数           | O(k) 临时数组                  |

替罪羊树冷静地等待，直到不平衡性无可否认，然后果断地进行重建。其结果是一棵保持精干且无需过多繁琐操作的二叉搜索树。
### 240 AA 树

AA 树是红黑树的一种简化版本，它通过每个节点使用一个单一的层级值（而非颜色）来维持平衡。它通过一组易于编码的规则，使用 `skew` 和 `split` 两种操作来强制平衡。AA 树的简单性使其成为一个流行的教学和实际实现选择，当你想要红黑树的性能而又不想处理复杂情况时。

#### 我们要解决什么问题？

我们想要一个平衡二叉搜索树，具有：

- 插入/搜索/删除的 O(log n) 时间复杂度
- 比红黑树或 AVL 树更简单的代码
- 更少的旋转情况

AA 树通过使用层级（类似于红黑树中的黑高）来强制结构，确保每次修复只需一次旋转即可达到平衡形状。

#### 工作原理（通俗解释）

每个节点包含：

- `key`：存储的值
- `level`：类似于黑高（根节点 = 1）

AA 树的不变式：

1. 左子节点层级 < 节点层级
2. 右子节点层级 ≤ 节点层级
3. 右-右孙节点层级 < 节点层级
4. 每个叶节点的层级为 1

在插入/删除后恢复平衡时，应用 skew 和 split 操作：

| 操作     | 规则                                 | 动作               |
| -------- | ------------------------------------ | ------------------ |
| Skew  | 如果 left.level == node.level        | 右旋               |
| Split | 如果 right.right.level == node.level | 左旋，节点层级加一 |

#### 示例

插入键值 [10, 20, 30]

| 步骤 | 树             | 修复操作                                         |
| ---- | -------------- | ------------------------------------------------ |
| 1    | 10 (层级 1)    | -                                                |
| 2    | 10 → 20        | 无需 Skew                                        |
| 3    | 10 → 20 → 30   | 违反右-右规则 → Split → 对 10 进行左旋 |

结果：

```
    20 (2)
   /  \
 10(1) 30(1)
```

#### 分步示例

插入 [30, 20, 10, 25]

| 步骤 | 插入 | 违反的规则                     | 修复操作           |          |
| ---- | ---- | ------------------------------ | ------------------ | -------- |
| 1    | 30   | 无                             | -                  |          |
| 2    | 20   | 左子节点层级 = 节点层级        | Skew → 右旋        |          |
| 3    | 10   | 左子树更深                     | Skew & Split       | 20 为根  |
| 4    | 25   | 插入到 20 的右侧               | 如果需要则 Split   | 平衡     |

#### 精简代码（简易版本）

C

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int key;
    int level;
    struct Node *left, *right;
} Node;

Node* new_node(int key) {
    Node* n = malloc(sizeof(Node));
    n->key = key;
    n->level = 1;
    n->left = n->right = NULL;
    return n;
}

Node* rotate_left(Node* x) {
    Node* y = x->right;
    x->right = y->left;
    y->left = x;
    return y;
}

Node* rotate_right(Node* y) {
    Node* x = y->left;
    y->left = x->right;
    x->right = y;
    return x;
}

Node* skew(Node* x) {
    if (x && x->left && x->left->level == x->level)
        x = rotate_right(x);
    return x;
}

Node* split(Node* x) {
    if (x && x->right && x->right->right && x->right->right->level == x->level) {
        x = rotate_left(x);
        x->level++;
    }
    return x;
}

Node* insert(Node* root, int key) {
    if (!root) return new_node(key);
    if (key < root->key)
        root->left = insert(root->left, key);
    else if (key > root->key)
        root->right = insert(root->right, key);
    root = skew(root);
    root = split(root);
    return root;
}

void inorder(Node* r) {
    if (!r) return;
    inorder(r->left);
    printf("(%d, %d) ", r->key, r->level);
    inorder(r->right);
}

int main(void) {
    Node* root = NULL;
    int keys[] = {10, 20, 30, 15, 25};
    for (int i = 0; i < 5; i++)
        root = insert(root, keys[i]);
    inorder(root);
    printf("\n");
}
```

Python

```python
class Node:
    def __init__(self, key):
        self.key = key
        self.level = 1
        self.left = None
        self.right = None

def rotate_left(x):
    y = x.right
    x.right = y.left
    y.left = x
    return y

def rotate_right(y):
    x = y.left
    y.left = x.right
    x.right = y
    return x

def skew(x):
    if x and x.left and x.left.level == x.level:
        x = rotate_right(x)
    return x

def split(x):
    if x and x.right and x.right.right and x.right.right.level == x.level:
        x = rotate_left(x)
        x.level += 1
    return x

def insert(root, key):
    if not root:
        return Node(key)
    if key < root.key:
        root.left = insert(root.left, key)
    elif key > root.key:
        root.right = insert(root.right, key)
    root = skew(root)
    root = split(root)
    return root

def inorder(r):
    if not r: return
    inorder(r.left)
    print(f"({r.key},{r.level})", end=" ")
    inorder(r.right)

root = None
for k in [10, 20, 30, 15, 25]:
    root = insert(root, k)
inorder(root)
```

#### 为什么它很重要

- 比红黑树更简单
- 只有两个平衡函数（skew, split）
- 保持 O(log n) 的性能
- 是自平衡二叉搜索树极佳的教学和参考结构

#### 一个温和的证明（为什么它有效）

AA 树是红黑树的一种特殊情况，其中：

- 右倾的红链接代表层级
- Skew 操作修复左倾的红链接
- Split 操作修复两个连续的右倾链接

这些规则确保了黑高不变式和树高的对数级别。

#### 自己动手试试

1.  插入升序键值 → 观察周期性的右旋
2.  画出每一步后的树 → 检查层级
3.  与等价的红黑树结构进行比较
4.  修改 split 条件 → 观察不平衡的出现

#### 测试用例

| 输入键值       | 期望的根节点 | 树高 | 是否平衡？ |
| -------------- | ------------ | ---- | ---------- |
| [10,20,30]     | 20           | 2    | ✅          |
| [1..10]        | 4–5          | ~log n | ✅          |
| [30,20,10,25]  | 20           | 3    | ✅          |

#### 复杂度

| 操作   | 时间复杂度 | 空间复杂度 |
| ------ | ---------- | ---------- |
| 插入   | O(log n)   | O(h)       |
| 搜索   | O(log n)   | O(1)       |
| 删除   | O(log n)   | O(h)       |

AA 树融合了红黑树的效率和 AVL 树的简单性，平衡、优雅且易于推理。

## 第 25 节. 线段树与树状数组
### 241 构建线段树

线段树是一种基于数组的二叉树，用于存储区间信息以实现快速查询和更新。构建线段树意味着从基础数组构造出这棵树，使得每个节点汇总一个区间的信息，例如求和、最小值、最大值或最大公约数。经典的构建时间复杂度为 O(n)，并且它支持 O(log n) 的查询和更新操作。

#### 我们要解决什么问题？

我们想要快速回答区间查询问题：

- 区间 [l, r] 的和
- 区间 [l, r] 的最小值或最大值
- 区间 [l, r] 的计数或按位运算

如果我们预先计算一个线段树，每个内部节点存储其两个子节点信息的合并结果。那么查询和更新操作只会涉及 O(log n) 个节点。

目标
将一个数组 `A[0..n-1]` 转换成一棵树 `T`，其中 `T[v]` 汇总了区间 `A[L..R]` 的信息。

#### 工作原理（通俗解释）

将树表示在一个数组中（为简单起见使用 1 起始索引）

- 节点 `v` 覆盖区间 `[L, R]`
- 左子节点 `2v` 覆盖 `[L, mid]`
- 右子节点 `2v+1` 覆盖 `[mid+1, R]`
- `T[v] = merge(T[2v], T[2v+1])`，其中 `merge` 是求和、最小值、最大值等操作。

构建步骤

1. 如果 `L == R`，将叶子值 `A[L]` 存入 `T[v]`。
2. 否则在 `mid` 处分割，递归构建左子树和右子树，然后 `T[v] = merge(left, right)`。

构建示例（求和）

数组 `A = [2, 1, 3, 4]`

| 节点 v | 区间 [L,R] | 值 |
| ------ | ------------- | ----- |
| 1      | [0,3]         | 10    |
| 2      | [0,1]         | 3     |
| 3      | [2,3]         | 7     |
| 4      | [0,0]         | 2     |
| 5      | [1,1]         | 1     |
| 6      | [2,2]         | 3     |
| 7      | [3,3]         | 4     |

`merge = sum`，所以 `T[1] = 3 + 7 = 10`，叶子节点存储原始值。

另一个构建示例（最小值）

数组 `A = [5, 2, 6, 1]`

| 节点 v | 区间 [L,R] | 值 (最小值) |
| ------ | ------------- | ----------- |
| 1      | [0,3]         | 1           |
| 2      | [0,1]         | 2           |
| 3      | [2,3]         | 1           |
| 4      | [0,0]         | 5           |
| 5      | [1,1]         | 2           |
| 6      | [2,2]         | 6           |
| 7      | [3,3]         | 1           |

#### 简洁代码（简易版本）

C 语言（迭代存储大小 4n，递归构建求和树）

```c
#include <stdio.h>

#define MAXN 100000
int A[MAXN];
long long T[4*MAXN];

long long merge(long long a, long long b) { return a + b; }

void build(int v, int L, int R) {
    if (L == R) {
        T[v] = A[L];
        return;
    }
    int mid = (L + R) / 2;
    build(2*v, L, mid);
    build(2*v+1, mid+1, R);
    T[v] = merge(T[2*v], T[2*v+1]);
}

int main(void) {
    int n = 4;
    A[0]=2; A[1]=1; A[2]=3; A[3]=4;
    build(1, 0, n-1);
    for (int v = 1; v < 8; v++) printf("T[%d]=%lld\n", v, T[v]);
    return 0;
}
```

Python（求和线段树，递归构建）

```python
def build(arr):
    n = len(arr)
    size = 1
    while size < n:
        size <<= 1
    T = [0] * (2 * size)

    # 叶子节点
    for i in range(n):
        T[size + i] = arr[i]
    # 内部节点
    for v in range(size - 1, 0, -1):
        T[v] = T[2*v] + T[2*v + 1]
    return T, size  # T 的索引是 1..size*2-1，叶子节点从索引 size 开始

# 示例
A = [2, 1, 3, 4]
T, base = build(A)
# T[1] 存储所有元素的和，T[base+i] 存储 A[i]
print("根节点和:", T[1])
```

说明

- C 语言版本展示了经典的递归自顶向下构建方法。
- Python 版本展示了使用 2 的幂次基底的迭代自底向上构建方法，时间复杂度也是 O(n)。

#### 为什么这很重要？

- O(n) 的预处理时间使得以下操作成为可能：

  * O(log n) 的区间查询
  * O(log n) 的单点更新
- 选择不同的 `merge` 操作可以使线段树适应多种任务

  * 求和、最小值、最大值、最大公约数、按位与、自定义结构体等

#### 一个温和的证明（为什么它有效）

每个元素恰好出现在一个叶子节点中，并且在每一层最多贡献给 O(1) 个节点。树有 O(log n) 层。构建的节点总数最多为 2n，所以总的工作量是 O(n)。

#### 自己动手试试

1. 将 `merge` 改为 `min` 或 `max` 并重新构建。
2. 构建后验证 `T[1]` 是否等于 `sum(A)`。
3. 逐层打印树以可视化各个区间。
4. 扩展该结构以支持单点更新和区间查询。

#### 测试用例

| 数组 A   | 合并操作 | 期望的根节点 T[1] | 备注                        |
| --------- | ----- | ------------------ | ---------------------------- |
| [2,1,3,4] | sum   | 10                 | 基本的求和树               |
| [5,2,6,1] | min   | 1                  | 将合并操作切换为 min          |
| [7]       | sum   | 7                  | 单个元素               |
| []        | sum   | 0 或无树       | 将空数组作为特殊情况处理 |

#### 复杂度

| 阶段        | 时间复杂度     | 空间复杂度      |
| ------------ | -------- | ---------- |
| 构建        | O(n)     | O(n)       |
| 查询        | O(log n) | O(1) 额外空间 |
| 单点更新 | O(log n) | O(1) 额外空间 |

一次构建，永远快速回答。线段树将数组变成了一个区间应答机。
### 242 区间求和查询

区间求和查询（RSQ）使用线段树来获取子数组 `[L, R]` 中元素的总和。一旦树构建完成，每个查询通过组合覆盖 `[L, R]` 的最小线段集的结果，在 O(log n) 时间内运行。

#### 我们要解决什么问题？

给定一个数组 `A[0..n-1]`，我们希望快速回答类似
```
sum(A[L..R])
```
的查询，而不需要每次都从头重新计算。

朴素方法：每次查询 O(R–L+1)
线段树方法：在 O(n) 构建后，每次查询 O(log n)。

#### 工作原理（通俗解释）

我们维护一个如前所述构建的线段树 `T`（`T[v]` = 线段的和）。
为了回答 `query(v, L, R, qL, qR)`

1. 如果线段 `[L, R]` 完全在 `[qL, qR]` 之外，返回 0。
2. 如果线段 `[L, R]` 完全在 `[qL, qR]` 之内，返回 `T[v]`。
3. 否则，在 `mid` 处分割，并组合左、右子节点的结果。

因此，每个查询只访问 O(log n) 个节点，每个节点恰好覆盖查询范围的一部分。

#### 示例

数组：`A = [2, 1, 3, 4, 5]`

查询：求和 [1,3]（基于1的索引：元素 1..3 = 1+3+4=8）

| 节点 v | 线段 [L,R] | 值 | 与 [1,3] 的关系 | 贡献 |
| ------ | ------------- | ----- | --------------------- | ------------ |
| 1      | [0,4]         | 15    | 重叠                  | 递归         |
| 2      | [0,2]         | 6     | 重叠                  | 递归         |
| 3      | [3,4]         | 9     | 重叠                  | 递归         |
| 4      | [0,1]         | 3     | 部分重叠              | 递归         |
| 5      | [2,2]         | 3     | 内部                  | +3           |
| 8      | [0,0]         | 2     | 外部                  | 跳过         |
| 9      | [1,1]         | 1     | 内部                  | +1           |
| 6      | [3,3]         | 4     | 内部                  | +4           |
| 7      | [4,4]         | 5     | 外部                  | 跳过         |

总和 = 1 + 3 + 4 = 8 ✅

#### 分步表格（为了清晰）

| 步骤 | 当前线段 | 是否覆盖？       | 操作   |
| ---- | --------------- | -------------- | -------- |
| 1    | [0,4]           | 与 [1,3] 重叠  | 分割     |
| 2    | [0,2]           | 与 [1,3] 重叠  | 分割     |
| 3    | [3,4]           | 与 [1,3] 重叠  | 分割     |
| 4    | [0,1]           | 重叠           | 分割     |
| 5    | [0,0]           | 外部           | 返回 0   |
| 6    | [1,1]           | 内部           | 返回 1   |
| 7    | [2,2]           | 内部           | 返回 3   |
| 8    | [3,3]           | 内部           | 返回 4   |
| 9    | [4,4]           | 外部           | 返回 0   |

总计 = 1+3+4 = 8

#### 精简代码（简易版本）

C 语言（递归 RSQ）

```c
#include <stdio.h>

#define MAXN 100000
int A[MAXN];
long long T[4*MAXN];

long long merge(long long a, long long b) { return a + b; }

void build(int v, int L, int R) {
    if (L == R) {
        T[v] = A[L];
        return;
    }
    int mid = (L + R) / 2;
    build(2*v, L, mid);
    build(2*v+1, mid+1, R);
    T[v] = merge(T[2*v], T[2*v+1]);
}

long long query(int v, int L, int R, int qL, int qR) {
    if (qR < L || R < qL) return 0; // 不相交
    if (qL <= L && R <= qR) return T[v]; // 完全覆盖
    int mid = (L + R) / 2;
    long long left = query(2*v, L, mid, qL, qR);
    long long right = query(2*v+1, mid+1, R, qL, qR);
    return merge(left, right);
}

int main(void) {
    int n = 5;
    int Avals[5] = {2,1,3,4,5};
    for (int i=0;i<n;i++) A[i]=Avals[i];
    build(1, 0, n-1);
    printf("Sum [1,3] = %lld\n", query(1,0,n-1,1,3)); // 期望 8
}
```

Python（迭代）

```python
def build(arr):
    n = len(arr)
    size = 1
    while size < n:
        size <<= 1
    T = [0]*(2*size)
    # 叶子节点
    for i in range(n):
        T[size+i] = arr[i]
    # 父节点
    for v in range(size-1, 0, -1):
        T[v] = T[2*v] + T[2*v+1]
    return T, size

def query(T, base, l, r):
    l += base
    r += base
    res = 0
    while l <= r:
        if l % 2 == 1:
            res += T[l]
            l += 1
        if r % 2 == 0:
            res += T[r]
            r -= 1
        l //= 2
        r //= 2
    return res

A = [2,1,3,4,5]
T, base = build(A)
print("Sum [1,3] =", query(T, base, 1, 3))  # 期望 8
```

#### 为什么这很重要

- 支持对静态或动态数组进行快速区间查询。
- 许多扩展的基础：
  * 区间最小/最大值查询（改变合并函数）
  * 延迟传播（用于区间更新）
  * 二维和可持久化线段树

#### 一个温和的证明（为什么它有效）

递归的每一层都分割成不相交的线段。
每层最多有 2 个线段被加入到结果中。
深度 = O(log n)，所以总工作量 = O(log n)。

#### 亲自尝试

1. 构建后查询不同的 [L, R] 区间。
2. 将 `merge` 替换为 `min()` 或 `max()` 并验证结果。
3. 组合查询以验证重叠的线段只被计算一次。
4. 添加更新操作来更改元素并重新查询。

#### 测试用例

| 数组 A     | 查询 [L,R] | 期望值 | 备注          |
| ----------- | ----------- | -------- | -------------- |
| [2,1,3,4,5] | [1,3]       | 8        | 1+3+4          |
| [5,5,5,5]   | [0,3]       | 20       | 均匀数组       |
| [1,2,3,4,5] | [2,4]       | 12       | 3+4+5          |
| [7]         | [0,0]       | 7        | 单个元素       |

#### 复杂度

| 操作   | 时间复杂度 | 空间复杂度 |
| --------- | -------- | ----- |
| 查询     | O(log n) | O(1)  |
| 构建     | O(n)     | O(n)  |
| 更新    | O(log n) | O(1)  |

线段树让你可以问 *"这个区间里有什么？"*，并且无论数组多大，都能快速得到答案。
### 243 区间更新（懒惰传播技术）

区间更新高效地修改区间 `[L, R]` 内的所有元素。如果不进行优化，你需要处理每个元素，时间复杂度为 O(n)。使用懒惰传播技术，你可以推迟工作，将待处理的更新存储在一个单独的数组中，从而实现每次更新和查询的 O(log n) 时间复杂度。

当许多更新操作重叠时（例如反复对区间 [2, 6] 内的每个元素加 5），这种模式至关重要。

#### 我们要解决什么问题？

我们希望高效地同时支持以下两种操作：

1.  **区间更新**：在区间 `[L, R]` 上增加或设置一个值。
2.  **区间查询**：获取区间 `[L, R]` 上的和/最小值/最大值。

朴素解法：每次更新 O(R–L+1)。
懒惰传播：每次更新/查询 O(log n)。

示例目标：

```
对 A[2..5] 加 +3
然后查询 sum(A[0..7])
```

#### 工作原理（通俗解释）

每个线段树节点 `T[v]` 存储汇总信息（例如和）。
每个节点 `lazy[v]` 存储待处理的更新（尚未下推到子节点）。

当更新区间 `[L, R]` 时：

- 如果节点的线段完全包含在更新区间内，则直接将更新应用到 `T[v]` 并标记 `lazy[v]`（无需递归）。
- 如果部分重叠，则先将待处理的更新下推，然后递归处理。

当查询区间 `[L, R]` 时：

- 首先下推任何待处理的更新
- 像往常一样合并子节点的结果

这样，每个节点在每条路径上最多被更新一次，从而得到 O(log n) 的复杂度。

#### 示例

数组：`A = [1, 2, 3, 4, 5, 6, 7, 8]`
构建求和线段树。

步骤 1：区间更新 `[2, 5]` += 3

| 节点 | 线段   | 操作                     | lazy[]    | T[] 和    |
| ---- | ------ | ------------------------ | --------- | --------- |
| 1    | [0,7]  | 重叠 → 递归              | 0         | 不变      |
| 2    | [0,3]  | 重叠 → 递归              | 0         | 不变      |
| 3    | [4,7]  | 重叠 → 递归              | 0         | 不变      |
| 4    | [0,1]  | 在区间外                 | -         | -         |
| 5    | [2,3]  | 在区间内 → 添加 +3×2=6   | lazy[5]=3 | T[5]+=6   |
| 6    | [4,5]  | 在区间内 → 添加 +3×2=6   | lazy[6]=3 | T[6]+=6   |

后续的查询会自动将 +3 应用到受影响的子区间。

#### 逐步示例

让我们跟踪两个操作：

1.  `update(2,5,+3)`
2.  `query(0,7)`

| 步骤   | 操作                                | 结果              |
| ------ | ----------------------------------- | ----------------- |
| 构建   | sum = [1,2,3,4,5,6,7,8] → 36        | T[1]=36           |
| 更新   | 标记完全在 [2,5] 内的线段的 lazy 值 | T[v]+=3×len       |
| 查询   | 在使用 T[v] 前传播 lazy 值          | sum = 36 + 3×4 = 48 |

更新后的结果：总和 = 48 ✅

#### 精简代码（简易版本）

C 语言（递归，带懒惰加法功能的求和线段树）

```c
#include <stdio.h>

#define MAXN 100000
int A[MAXN];
long long T[4*MAXN], lazy[4*MAXN];

long long merge(long long a, long long b) { return a + b; }

void build(int v, int L, int R) {
    if (L == R) { T[v] = A[L]; return; }
    int mid = (L + R) / 2;
    build(2*v, L, mid);
    build(2*v+1, mid+1, R);
    T[v] = merge(T[2*v], T[2*v+1]);
}

void push(int v, int L, int R) {
    if (lazy[v] != 0) {
        T[v] += lazy[v] * (R - L + 1);
        if (L != R) {
            lazy[2*v] += lazy[v];
            lazy[2*v+1] += lazy[v];
        }
        lazy[v] = 0;
    }
}

void update(int v, int L, int R, int qL, int qR, int val) {
    push(v, L, R);
    if (qR < L || R < qL) return;
    if (qL <= L && R <= qR) {
        lazy[v] += val;
        push(v, L, R);
        return;
    }
    int mid = (L + R) / 2;
    update(2*v, L, mid, qL, qR, val);
    update(2*v+1, mid+1, R, qL, qR, val);
    T[v] = merge(T[2*v], T[2*v+1]);
}

long long query(int v, int L, int R, int qL, int qR) {
    push(v, L, R);
    if (qR < L || R < qL) return 0;
    if (qL <= L && R <= qR) return T[v];
    int mid = (L + R) / 2;
    return merge(query(2*v, L, mid, qL, qR), query(2*v+1, mid+1, R, qL, qR));
}

int main(void) {
    int n = 8;
    int vals[] = {1,2,3,4,5,6,7,8};
    for (int i=0;i<n;i++) A[i]=vals[i];
    build(1,0,n-1);
    update(1,0,n-1,2,5,3);
    printf("Sum [0,7] = %lld\n", query(1,0,n-1,0,7)); // 期望 48
}
```

Python（迭代版本）

```python
class LazySegTree:
    def __init__(self, arr):
        n = len(arr)
        size = 1
        while size < n: size <<= 1
        self.n = n; self.size = size
        self.T = [0]*(2*size)
        self.lazy = [0]*(2*size)
        for i in range(n): self.T[size+i] = arr[i]
        for i in range(size-1, 0, -1):
            self.T[i] = self.T[2*i] + self.T[2*i+1]
    
    def _apply(self, v, val, length):
        self.T[v] += val * length
        if v < self.size:
            self.lazy[v] += val

    def _push(self, v, length):
        if self.lazy[v]:
            self._apply(2*v, self.lazy[v], length//2)
            self._apply(2*v+1, self.lazy[v], length//2)
            self.lazy[v] = 0

    def update(self, l, r, val):
        def _upd(v, L, R):
            if r < L or R < l: return
            if l <= L and R <= r:
                self._apply(v, val, R-L+1)
                return
            self._push(v, R-L+1)
            mid = (L+R)//2
            _upd(2*v, L, mid)
            _upd(2*v+1, mid+1, R)
            self.T[v] = self.T[2*v] + self.T[2*v+1]
        _upd(1, 0, self.size-1)

    def query(self, l, r):
        def _qry(v, L, R):
            if r < L or R < l: return 0
            if l <= L and R <= r:
                return self.T[v]
            self._push(v, R-L+1)
            mid = (L+R)//2
            return _qry(2*v, L, mid) + _qry(2*v+1, mid+1, R)
        return _qry(1, 0, self.size-1)

A = [1,2,3,4,5,6,7,8]
st = LazySegTree(A)
st.update(2,5,3)
print("Sum [0,7] =", st.query(0,7))  # 期望 48
```

#### 为什么它很重要

- 对于有许多重叠更新操作的问题至关重要。
- 用于区间加法、区间赋值、区间覆盖以及二维扩展。
- 是 Segment Tree Beats 的基础。

#### 一个温和的证明（为什么它有效）

每次更新最多在每一层标记一个节点为“懒惰”。
当后续查询时，我们“下推”这些更新一次。
每个节点的待处理更新应用 O(1) 次 → 总成本 O(log n)。

#### 自己动手试试

1.  构建 `[1,2,3,4,5]` → 更新 [1,3] += 2 → 查询和 [0,4] → 期望 21
2.  更新 [0,4] += 1 → 查询 [2,4] → 期望 3+2+2+1+1 = 15
3.  组合多个更新 → 验证累积结果

#### 测试用例

| 操作             | 数组               | 查询               | 期望值       |
| ---------------- | ------------------ | ------------------ | ------------ |
| update [2,5] +=3 | [1,2,3,4,5,6,7,8]  | sum [0,7]          | 48           |
| update [0,3] +=2 | [5,5,5,5]          | sum [1,2]          | 14           |
| 两次更新         | [1,1,1,1,1]        | [1,3] +1, [2,4] +2 | [1,3] 和 = 10 |

#### 复杂度

| 操作         | 时间复杂度 | 空间复杂度 |
| ------------ | ---------- | ---------- |
| 区间更新     | O(log n)   | O(n)       |
| 区间查询     | O(log n)   | O(n)       |
| 构建         | O(n)       | O(n)       |

懒惰传播，更聪明地工作，而不是更努力。只在需要的时候，应用需要的内容。
### 244 单点更新

单点更新会改变数组中的一个元素，并更新从该元素到根节点路径上所有相关的线段树节点。此操作确保线段树在未来的区间查询中保持一致。

与区间更新（一次性标记多个元素）不同，单点更新仅涉及 O(log n) 个节点，每一层一个。

#### 我们要解决什么问题？

给定一个构建在 `A[0..n-1]` 上的线段树，我们想要：

- 改变一个元素：`A[pos] = new_value`
- 在线段树中反映这一改变，使得所有区间查询保持正确

目标：高效更新，无需完全重建

朴素方法：重建树 → O(n)
线段树单点更新：O(log n)

#### 工作原理（通俗解释）

每个树节点 `T[v]` 代表一个区间 `[L, R]`。
当 `pos` 位于 `[L, R]` 内时，该节点的值可能需要调整。

算法（递归）：

1. 如果 `L == R == pos`，则赋值 `A[pos] = val`，设置 `T[v] = val`。
2. 否则，计算 `mid`。
   * 根据 `pos` 递归进入左孩子或右孩子。
   * 在孩子更新后，重新计算 `T[v] = merge(T[2v], T[2v+1])`。

不需要懒惰传播，这是一个直接向下的路径。

#### 示例

数组：`A = [2, 1, 3, 4]`
树存储和。

| 节点 | 区间   | 值   |
| ---- | ------ | ---- |
| 1    | [0,3]  | 10   |
| 2    | [0,1]  | 3    |
| 3    | [2,3]  | 7    |
| 4    | [0,0]  | 2    |
| 5    | [1,1]  | 1    |
| 6    | [2,2]  | 3    |
| 7    | [3,3]  | 4    |

操作：`A[1] = 5`

路径：[1, 2, 5]

| 步骤   | 节点 | 旧值 | 新值 | 更新操作       |
| ------ | ---- | ---- | ---- | -------------- |
| 叶子   | 5    | 1    | 5    | T[5]=5         |
| 父节点 | 2    | 3    | 7    | T[2]=2+5=7     |
| 根节点 | 1    | 10   | 14   | T[1]=7+7=14    |

新数组：`[2, 5, 3, 4]`
新和：14 ✅

#### 逐步跟踪

```
update(v=1, L=0, R=3, pos=1, val=5)
  mid=1
  -> pos<=mid → 左孩子 (v=2)
    update(v=2, L=0, R=1)
      mid=0
      -> pos>mid → 右孩子 (v=5)
        update(v=5, L=1, R=1)
        T[5]=5
      T[2]=merge(T[4]=2, T[5]=5)=7
  T[1]=merge(T[2]=7, T[3]=7)=14
```

#### 精简代码（简易版本）

C (递归，求和合并)

```c
#include <stdio.h>

#define MAXN 100000
int A[MAXN];
long long T[4*MAXN];

long long merge(long long a, long long b) { return a + b; }

void build(int v, int L, int R) {
    if (L == R) { T[v] = A[L]; return; }
    int mid = (L + R)/2;
    build(2*v, L, mid);
    build(2*v+1, mid+1, R);
    T[v] = merge(T[2*v], T[2*v+1]);
}

void point_update(int v, int L, int R, int pos, int val) {
    if (L == R) {
        T[v] = val;
        A[pos] = val;
        return;
    }
    int mid = (L + R)/2;
    if (pos <= mid) point_update(2*v, L, mid, pos, val);
    else point_update(2*v+1, mid+1, R, pos, val);
    T[v] = merge(T[2*v], T[2*v+1]);
}

long long query(int v, int L, int R, int qL, int qR) {
    if (qR < L || R < qL) return 0;
    if (qL <= L && R <= qR) return T[v];
    int mid = (L + R)/2;
    return merge(query(2*v, L, mid, qL, qR), query(2*v+1, mid+1, R, qL, qR));
}

int main(void) {
    int n = 4;
    int vals[] = {2,1,3,4};
    for (int i=0;i<n;i++) A[i]=vals[i];
    build(1,0,n-1);
    printf("更新前: 区间和 [0,3] = %lld\n", query(1,0,n-1,0,3)); // 10
    point_update(1,0,n-1,1,5);
    printf("更新后: 区间和 [0,3] = %lld\n", query(1,0,n-1,0,3)); // 14
}
```

Python (迭代)

```python
def build(arr):
    n = len(arr)
    size = 1
    while size < n: size <<= 1
    T = [0]*(2*size)
    for i in range(n):
        T[size+i] = arr[i]
    for v in range(size-1, 0, -1):
        T[v] = T[2*v] + T[2*v+1]
    return T, size

def update(T, base, pos, val):
    v = base + pos
    T[v] = val
    v //= 2
    while v >= 1:
        T[v] = T[2*v] + T[2*v+1]
        v //= 2

def query(T, base, l, r):
    l += base; r += base
    res = 0
    while l <= r:
        if l%2==1: res += T[l]; l+=1
        if r%2==0: res += T[r]; r-=1
        l//=2; r//=2
    return res

A = [2,1,3,4]
T, base = build(A)
print("更新前:", query(T, base, 0, 3))  # 10
update(T, base, 1, 5)
print("更新后:", query(T, base, 0, 3))   # 14
```

#### 为什么重要

- 动态数据的基础，支持快速局部编辑
- 用于线段树、Fenwick 树和 BIT
- 非常适合动态评分、累积和、实时数据更新等应用

#### 一个温和的证明（为什么有效）

每一层都有一个节点受到位置 `pos` 的影响。
树的高度 = O(log n)。
因此，恰好重新计算 O(log n) 个节点。

#### 自己动手试试

1.  构建 `[2,1,3,4]`，更新 `A[2]=10` → 区间和 [0,3]=17
2.  将合并操作替换为 `min` → 更新元素 → 测试查询
3.  对于大的 n，比较与完全重建的时间

#### 测试用例

| 输入 A     | 更新       | 查询         | 预期结果 |
| ---------- | ---------- | ------------ | -------- |
| [2,1,3,4]  | A[1]=5     | 区间和[0,3]  | 14       |
| [5,5,5]    | A[2]=2     | 区间和[0,2]  | 12       |
| [1]        | A[0]=9     | 区间和[0,0]  | 9        |

#### 复杂度

| 操作       | 时间复杂度 | 空间复杂度 |
| ---------- | ---------- | ---------- |
| 单点更新   | O(log n)   | O(1)       |
| 查询       | O(log n)   | O(1)       |
| 构建       | O(n)       | O(n)       |

单点更新就像池塘中的涟漪，一次改变向上传播，使整个结构保持和谐。
### 245 Fenwick Tree（树状数组）的构建

Fenwick Tree，或称 Binary Indexed Tree (BIT)，是一种用于前缀查询和点更新的紧凑数据结构。它非常适合处理累积和、频率统计或任何可结合的操作。高效地构建它是实现 O(log n) 查询和更新的基础。

与线段树不同，Fenwick Tree 使用巧妙的索引运算，在单个数组中表示重叠的区间。

#### 我们要解决什么问题？

我们希望预计算一个结构，以便：

- 回答前缀查询：`sum(0..i)`
- 支持更新：`A[i] += delta`

朴素的前缀和数组：

- 查询：O(1)
- 更新：O(n)

Fenwick Tree：

- 查询：O(log n)
- 更新：O(log n)
- 构建：O(n)

#### 工作原理（通俗解释）

Fenwick Tree 利用索引的最低有效位 (LSB) 来确定区间长度。

每个节点 `BIT[i]` 存储以下区间的和：

```
(i - LSB(i) + 1) .. i
```

因此：

- `BIT[1]` 存储 A[1]
- `BIT[2]` 存储 A[1..2]
- `BIT[3]` 存储 A[3]
- `BIT[4]` 存储 A[1..4]
- `BIT[5]` 存储 A[5]
- `BIT[6]` 存储 A[5..6]

前缀和 = 这些重叠区间的和。

#### 示例

数组 `A = [2, 1, 3, 4, 5]` (为清晰起见，使用 1-索引)

| i | A[i] | LSB(i) | 存储的区间 | BIT[i] (和) |
| - | ---- | ------ | ------------ | ------------ |
| 1 | 2    | 1      | [1]          | 2            |
| 2 | 1    | 2      | [1..2]       | 3            |
| 3 | 3    | 1      | [3]          | 3            |
| 4 | 4    | 4      | [1..4]       | 10           |
| 5 | 5    | 1      | [5]          | 5            |

#### 逐步构建 (O(n))

从 i = 1 到 n 迭代：

1. `BIT[i] += A[i]`
2. 将 `BIT[i]` 加到其父节点：`BIT[i + LSB(i)] += BIT[i]`

| 步骤 | i | LSB(i) | 更新 BIT[i+LSB(i)]             | 结果 BIT  |
| ---- | - | ------ | -------------------------------- | ----------- |
| 1    | 1 | 1      | BIT[2]+=2                        | [2,3,0,0,0] |
| 2    | 2 | 2      | BIT[4]+=3                        | [2,3,0,3,0] |
| 3    | 3 | 1      | BIT[4]+=3                        | [2,3,3,6,0] |
| 4    | 4 | 4      | BIT[8]+=6 (忽略，超出范围) | [2,3,3,6,0] |
| 5    | 5 | 1      | BIT[6]+=5 (忽略，超出范围) | [2,3,3,6,5] |

构建好的 BIT：`[2, 3, 3, 6, 5]` ✅

#### 简洁代码（简易版本）

C (O(n) 构建)

```c
#include <stdio.h>

#define MAXN 100005
int A[MAXN];
long long BIT[MAXN];
int n;

int lsb(int i) { return i & -i; }

void build() {
    for (int i = 1; i <= n; i++) {
        BIT[i] += A[i];
        int parent = i + lsb(i);
        if (parent <= n)
            BIT[parent] += BIT[i];
    }
}

long long prefix_sum(int i) {
    long long s = 0;
    while (i > 0) {
        s += BIT[i];
        i -= lsb(i);
    }
    return s;
}

int main(void) {
    n = 5;
    int vals[6] = {0, 2, 1, 3, 4, 5}; // 1-索引
    for (int i=1;i<=n;i++) A[i]=vals[i];
    build();
    printf("Prefix sum [1..3] = %lld\n", prefix_sum(3)); // 期望 6
}
```

Python (1-索引)

```python
def build(A):
    n = len(A) - 1
    BIT = [0]*(n+1)
    for i in range(1, n+1):
        BIT[i] += A[i]
        parent = i + (i & -i)
        if parent <= n:
            BIT[parent] += BIT[i]
    return BIT

def prefix_sum(BIT, i):
    s = 0
    while i > 0:
        s += BIT[i]
        i -= (i & -i)
    return s

A = [0,2,1,3,4,5]  # 1-索引
BIT = build(A)
print("BIT =", BIT[1:])
print("Sum[1..3] =", prefix_sum(BIT, 3))  # 期望 6
```

#### 为什么它很重要

- 线段树的轻量级替代方案
- O(n) 构建，O(log n) 查询，O(log n) 更新
- 用于频率表、逆序数计数、前缀查询、累积直方图

#### 一个温和的证明（为什么它有效）

每个索引 i 最多贡献给 log n 个 BIT 条目。
每个 BIT[i] 存储由 LSB 定义的一个不相交区间的和。
通过向父节点传播来构建，确保了正确的重叠覆盖。

#### 亲自尝试

1.  从 [2, 1, 3, 4, 5] 构建 BIT → 查询 sum(3)=6
2.  更新 A[2]+=2 → prefix(3)=8
3.  与累积和数组比较以验证正确性

#### 测试用例

| A (1-索引) | 查询     | 期望值 |
| ------------- | --------- | -------- |
| [2,1,3,4,5]   | prefix(3) | 6        |
| [5,5,5,5]     | prefix(4) | 20       |
| [1]           | prefix(1) | 1        |

#### 复杂度

| 操作 | 时间复杂度 | 空间复杂度 |
| --------- | -------- | ----- |
| 构建     | O(n)     | O(n)  |
| 查询     | O(log n) | O(1)  |
| 更新    | O(log n) | O(1)  |

Fenwick Tree 的艺术在于用更少的存储，做足够的工作，使前缀和计算变得极快。
### 246 树状数组更新

树状数组（Binary Indexed Tree）支持点更新，即调整单个元素，并高效地将该变化反映在所有相关的累积和中。更新操作通过由最低有效位（Least Significant Bit, LSB）确定的父索引向上传播。

#### 我们要解决什么问题？

给定一个已构建的树状数组，我们想要执行如下操作：

```
A[pos] += delta
```

并保持所有前缀和 `sum(0..i)` 的一致性。

朴素方法：更新每个前缀 → O(n)
树状数组：通过选定的索引传播 → O(log n)

#### 工作原理（通俗解释）

在树状数组中，`BIT[i]` 覆盖范围 `(i - LSB(i) + 1) .. i`。
因此，当我们更新 `A[pos]` 时，我们必须将 delta 加到所有其覆盖范围包含 `pos` 的 `BIT[i]` 上。

规则：

```
for (i = pos; i <= n; i += LSB(i))
    BIT[i] += delta
```

- LSB(i) 跳转到下一个覆盖 `pos` 的索引
- 当索引超过 `n` 时停止

#### 示例

数组 `A = [2, 1, 3, 4, 5]`（1-索引）
BIT 构建为 `[2, 3, 3, 10, 5]`

现在执行 update(2, +2) → A[2] = 3

| 步骤 | i | LSB(i) | BIT[i] 变化 | 新 BIT      |
| ---- | - | ------ | ----------- | ----------- |
| 1    | 2 | 2      | BIT[2]+=2   | [2,5,3,10,5] |
| 2    | 4 | 4      | BIT[4]+=2   | [2,5,3,12,5] |
| 3    | 8 | 停止   | -           | 完成        |

新的前缀和反映了更新后的数组 `[2,3,3,4,5]`

检查 prefix(3) = 2+3+3=8 ✅

#### 逐步表格

| 前缀 | 旧和 | 新和 |
| ---- | ---- | ---- |
| 1    | 2    | 2    |
| 2    | 3    | 5    |
| 3    | 6    | 8    |
| 4    | 10   | 12   |
| 5    | 15   | 17   |

#### 微型代码（简易版本）

C 语言（树状数组更新）

```c
#include <stdio.h>

#define MAXN 100005
long long BIT[MAXN];
int n;

int lsb(int i) { return i & -i; }

void update(int pos, int delta) {
    for (int i = pos; i <= n; i += lsb(i))
        BIT[i] += delta;
}

long long prefix_sum(int i) {
    long long s = 0;
    for (; i > 0; i -= lsb(i))
        s += BIT[i];
    return s;
}

int main(void) {
    n = 5;
    // 从 A = [2, 1, 3, 4, 5] 构建
    BIT[1]=2; BIT[2]=3; BIT[3]=3; BIT[4]=10; BIT[5]=5;
    update(2, 2); // A[2]+=2
    printf("Sum [1..3] = %lld\n", prefix_sum(3)); // 期望 8
}
```

Python（1-索引）

```python
def update(BIT, n, pos, delta):
    while pos <= n:
        BIT[pos] += delta
        pos += (pos & -pos)

def prefix_sum(BIT, pos):
    s = 0
    while pos > 0:
        s += BIT[pos]
        pos -= (pos & -pos)
    return s

# 示例
BIT = [0,2,3,3,10,5]  # 从 [2,1,3,4,5] 构建
update(BIT, 5, 2, 2)
print("Sum [1..3] =", prefix_sum(BIT, 3))  # 期望 8
```

#### 为什么这很重要

- 支持实时数据更新和快速前缀查询
- 对于求和操作，比线段树更简单且空间效率更高
- 是许多算法的核心：逆序对计数、频率累积、顺序统计

#### 一个温和的证明（为什么它有效）

每个 `BIT[i]` 覆盖一个固定范围 `(i - LSB(i) + 1 .. i)`。
如果 `pos` 位于该范围内，则递增 `BIT[i]` 可确保未来的前缀和正确。
由于每次更新都按 LSB(i) 移动，最多发生 log₂(n) 步。

#### 自己动手试试

1.  从 `[2,1,3,4,5]` 构建 BIT → update(2,+2)
2.  查询 prefix(3) → 期望 8
3.  Update(5, +5) → prefix(5) = 22
4.  链式执行多个更新 → 验证增量求和

#### 测试用例

| A (1-索引) | 更新   | 查询       | 期望值 |
| ---------- | ------ | ---------- | ------ |
| [2,1,3,4,5] | (2,+2) | sum[1..3] | 8      |
| [5,5,5,5]   | (4,+1) | sum[1..4] | 21     |
| [1,2,3,4,5] | (5,-2) | sum[1..5] | 13     |

#### 复杂度

| 操作   | 时间复杂度 | 空间复杂度 |
| ------ | ---------- | ---------- |
| 更新   | O(log n)   | O(1)       |
| 查询   | O(log n)   | O(1)       |
| 构建   | O(n)       | O(n)       |

每次更新都是一次涟漪，沿着树向上攀升，使所有前缀和保持完美同步。
### 247 树状数组查询

一旦你构建或更新了树状数组（Binary Indexed Tree），你通常会想要提取有用的信息，最常见的是前缀和。查询操作通过索引运算向下遍历树，沿途收集部分和。

这个简单而强大的例程将线性的前缀和扫描转变为优雅的 O(log n) 解决方案。

#### 我们要解决什么问题？

我们需要在更新后高效地计算：

```
sum(1..i) = A[1] + A[2] + ... + A[i]
```

在树状数组中，每个索引存储由其最低有效位（LSB）决定的区段和。
通过向下移动（每一步减去 LSB），我们收集互不相交的区间，这些区间共同覆盖了 `[1..i]`。

#### 工作原理（通俗解释）

每个 `BIT[i]` 存储范围 `(i - LSB(i) + 1 .. i)` 的和。
因此，为了得到前缀和，我们通过向下遍历来组合所有这些区段：

```
sum = 0
while i > 0:
    sum += BIT[i]
    i -= LSB(i)
```

关键洞察：

- 更新：向上移动 (`i += LSB(i)`)
- 查询：向下移动 (`i -= LSB(i)`)

它们相辅相成，就像累积逻辑的阴阳两面。

#### 示例

假设我们有 `A = [2, 3, 3, 4, 5]`（1-索引），
构建的 BIT = [2, 5, 3, 12, 5]

让我们计算 `prefix_sum(5)`

| 步骤 | i | BIT[i] | LSB(i) | 累计和 | 解释                 |
| ---- | - | ------ | ------ | --------------- | -------------------- |
| 1    | 5 | 5      | 1      | 5               | 加 BIT[5] (A[5])    |
| 2    | 4 | 12     | 4      | 17              | 加 BIT[4] (A[1..4]) |
| 3    | 0 | -      | -      | 停止            | 完成                 |

✅ `prefix_sum(5) = 17` 匹配 `2+3+3+4+5=17`

现在试试 `prefix_sum(3)`

| 步骤 | i | BIT[i] | LSB(i) | 和   |
| ---- | - | ------ | ------ | ---- |
| 1    | 3 | 3      | 1      | 3    |
| 2    | 2 | 5      | 2      | 8    |
| 3    | 0 | -      | -      | 停止 |

✅ `prefix_sum(3) = 8`

#### 精简代码（简易版本）

C 语言（树状数组查询）

```c
#include <stdio.h>

#define MAXN 100005
long long BIT[MAXN];
int n;

int lsb(int i) { return i & -i; }

long long prefix_sum(int i) {
    long long s = 0;
    while (i > 0) {
        s += BIT[i];
        i -= lsb(i);
    }
    return s;
}

int main(void) {
    n = 5;
    long long B[6] = {0,2,5,3,12,5}; // 已构建的 BIT
    for (int i=1;i<=n;i++) BIT[i] = B[i];
    printf("前缀和 [1..3] = %lld\n", prefix_sum(3)); // 期望 8
    printf("前缀和 [1..5] = %lld\n", prefix_sum(5)); // 期望 17
}
```

Python（1-索引）

```python
def prefix_sum(BIT, i):
    s = 0
    while i > 0:
        s += BIT[i]
        i -= (i & -i)
    return s

BIT = [0,2,5,3,12,5]
print("sum[1..3] =", prefix_sum(BIT, 3))  # 8
print("sum[1..5] =", prefix_sum(BIT, 5))  # 17
```

#### 为什么重要

- 动态更新后快速计算前缀和
- 对频率表、顺序统计、逆序计数至关重要
- 是许多竞赛编程技巧的核心（例如“计数小于 k 的数”）

#### 一个温和的证明（为什么有效）

每个索引 `i` 贡献给一组固定的 BIT 节点。
查询时，我们收集所有 BIT 区段，它们共同构成 `[1..i]`。
LSB 确保没有重叠，每个范围都是不相交的。
总步数 = `i` 中设置的位数 = O(log n)。

#### 亲自尝试

1.  从 `[2,3,3,4,5]` 构建 BIT
2.  查询 prefix_sum(3) → 期望 8
3.  更新(2,+2)，查询(3) → 期望 10
4.  查询 prefix_sum(5) → 验证正确性

#### 测试用例

| A (1-索引)    | 查询    | 期望值 |
| ------------- | ------ | -------- |
| [2,3,3,4,5]   | sum(3) | 8        |
| [2,3,3,4,5]   | sum(5) | 17       |
| [1,2,3,4,5]   | sum(4) | 10       |

#### 复杂度

| 操作   | 时间复杂度 | 空间复杂度 |
| --------- | -------- | ----- |
| 查询     | O(log n) | O(1)  |
| 更新    | O(log n) | O(1)  |
| 构建     | O(n)     | O(n)  |

树状数组查询是优雅的下降过程，沿着比特位向下，收集求和谜题的每一块碎片。
### 248 线段树合并

线段树不仅能处理求和，它是一种多功能的数据结构，可以使用任何结合性操作（求和、最小值、最大值、最大公约数等）来合并一个区间两半的结果。合并函数是线段树的核心：它告诉我们如何将子节点的结果合并到父节点中。

#### 我们要解决什么问题？

我们想要将左、右子区间的结果合并成一个父节点结果。
如果没有合并逻辑，树就无法聚合数据或回答查询。

例如：

- 对于求和查询 → merge(a, b) = a + b
- 对于最小值查询 → merge(a, b) = min(a, b)
- 对于最大公约数查询 → merge(a, b) = gcd(a, b)

因此，合并函数定义了*这棵树的意义*。

#### 工作原理（通俗解释）

每个节点代表一个区间 `[L, R]`。
它的值由其两个子节点推导而来：

```
节点 = merge(左子节点, 右子节点)
```

当你进行以下操作时：

- 构建：递归地从子节点计算节点值
- 查询：合并部分重叠的区间结果
- 更新：使用合并函数重新计算受影响的节点

所以，合并是将整棵树粘合在一起的统一规则。

#### 示例（求和线段树）

数组 `A = [2, 1, 3, 4]`

| 节点   | 区间    | 左子节点 | 右子节点 | 合并（求和） |
| ------ | ------- | -------- | -------- | ------------ |
| 根节点 | [1..4]  | 6        | 4        | 10           |
| 左节点 | [1..2]  | 2        | 1        | 3            |
| 右节点 | [3..4]  | 3        | 4        | 7            |

每个父节点 = 左子节点 + 右子节点

树结构：

```
          [1..4]=10
         /         \
   [1..2]=3       [3..4]=7
   /    \         /     \
$$1]=2 [2]=1   [3]=3   [4]=4
```

合并规则：`merge(a, b) = a + b`

#### 示例（最小值线段树）

数组 `A = [5, 2, 7, 1]`
合并规则：`min(a, b)`

| 节点   | 区间    | 左子节点 | 右子节点 | 合并（最小值） |
| ------ | ------- | -------- | -------- | -------------- |
| 根节点 | [1..4]  | 2        | 1        | 1              |
| 左节点 | [1..2]  | 5        | 2        | 2              |
| 右节点 | [3..4]  | 7        | 1        | 1              |

结果：根节点存储 1，即全局最小值。

#### 精简代码（简易版本）

C 语言（求和线段树合并）

```c
#include <stdio.h>
#define MAXN 100005

int A[MAXN], tree[4*MAXN];
int n;

int merge(int left, int right) {
    return left + right;
}

void build(int node, int l, int r) {
    if (l == r) {
        tree[node] = A[l];
        return;
    }
    int mid = (l + r) / 2;
    build(node*2, l, mid);
    build(node*2+1, mid+1, r);
    tree[node] = merge(tree[node*2], tree[node*2+1]);
}

int query(int node, int l, int r, int ql, int qr) {
    if (qr < l || ql > r) return 0; // 中性元素
    if (ql <= l && r <= qr) return tree[node];
    int mid = (l + r) / 2;
    int left = query(node*2, l, mid, ql, qr);
    int right = query(node*2+1, mid+1, r, ql, qr);
    return merge(left, right);
}

int main(void) {
    n = 4;
    int vals[5] = {0, 2, 1, 3, 4};
    for (int i=1; i<=n; i++) A[i] = vals[i];
    build(1,1,n);
    printf("Sum [1..4] = %d\n", query(1,1,n,1,4)); // 10
    printf("Sum [2..3] = %d\n", query(1,1,n,2,3)); // 4
}
```

Python

```python
def merge(a, b):
    return a + b  # 在此定义操作

def build(arr, tree, node, l, r):
    if l == r:
        tree[node] = arr[l]
        return
    mid = (l + r) // 2
    build(arr, tree, 2*node, l, mid)
    build(arr, tree, 2*node+1, mid+1, r)
    tree[node] = merge(tree[2*node], tree[2*node+1])

def query(tree, node, l, r, ql, qr):
    if qr < l or ql > r:
        return 0  # 求和的中性元素
    if ql <= l and r <= qr:
        return tree[node]
    mid = (l + r)//2
    left = query(tree, 2*node, l, mid, ql, qr)
    right = query(tree, 2*node+1, mid+1, r, ql, qr)
    return merge(left, right)

A = [0, 2, 1, 3, 4]
n = 4
tree = [0]*(4*n)
build(A, tree, 1, 1, n)
print("Sum[1..4] =", query(tree, 1, 1, n, 1, 4))  # 10
```

#### 为什么这很重要

- 合并是线段树的“灵魂”，定义合并函数，就定义了树的目的。
- 适用于多种任务：求和、最小值/最大值、最大公约数、异或、矩阵乘法。
- 统一的模式：构建、查询、更新都依赖于相同的操作。

#### 一个温和的证明（为什么它有效）

线段树通过分治法工作。
如果一个操作是结合性的（如 +、min、max、gcd），合并部分结果得到的答案与在整个区间上计算的结果相同。
因此，只要 `merge(a, b)` = `merge(b, a)` 且具有结合性，正确性就随之而来。

#### 自己动手试试

1.  将合并函数替换为 `min(a,b)` → 构建最小值线段树
2.  将合并函数替换为 `max(a,b)` → 构建最大值线段树
3.  将合并函数替换为 `__gcd(a,b)` → 构建最大公约数树
4.  更改合并规则后测试 `query(2,3)`

#### 测试用例

| A            | 操作     | 查询    | 预期结果 |
| ------------ | -------- | ------- | -------- |
| [2,1,3,4]    | 求和     | [1..4]  | 10       |
| [5,2,7,1]    | 最小值   | [1..4]  | 1        |
| [3,6,9,12]   | 最大公约数 | [2..4]  | 3        |

#### 复杂度

| 操作   | 时间复杂度 | 空间复杂度 |
| ------ | ---------- | ---------- |
| 构建   | O(n)       | O(n)       |
| 查询   | O(log n)   | O(n)       |
| 更新   | O(log n)   | O(n)       |

合并函数是每个线段树的心跳，一旦你定义了它，你的树就明白了“合并”意味着什么。
### 249 可持久化线段树

可持久化线段树是一种神奇的数据结构，它让你能够进行时间旅行。在更新时，它不会覆盖节点，而是创建新版本，同时保留旧版本。每次更新都会返回一个新的根节点，让你能够访问完整的历史记录，并且每个版本都能以 O(log n) 的时间复杂度进行访问。

#### 我们要解决什么问题？

我们需要一个支持以下操作的数据结构：

- 进行点更新而不丢失过去的状态
- 在任何历史版本上进行区间查询

应用场景：

- 撤销/回滚系统
- 版本化数据库
- 离线查询（"第二次更新后，区间 [1..3] 的和是多少？"）

每个版本都是不可变的，非常适合函数式编程或审计需求。

#### 工作原理（通俗解释）

可持久化线段树只克隆从根节点到被更新叶子节点路径上的节点。
所有其他节点都是共享的。

对于每次更新：

1. 创建一个新的根节点。
2. 复制从根节点到被更改索引路径上的节点。
3. 重用未更改的子树。

结果：每个版本占用 O(log n) 的内存，而不是 O(n)。

#### 示例

数组 `A = [1, 2, 3, 4]`

版本 0：为 `[1, 2, 3, 4]` 构建的树

- 总和 = 10

更新 A[2] = 5

- 创建版本 1，复制到 A[2] 的路径
- 新的总和 = 1 + 5 + 3 + 4 = 13

| 版本 | A         | 区间和(1..4) | 区间和(1..2) |
| ---- | --------- | ------------ | ------------ |
| 0    | [1,2,3,4] | 10           | 3            |
| 1    | [1,5,3,4] | 13           | 6            |

两个版本同时存在。
版本 0 保持不变。版本 1 反映了新的值。

#### 可视化

```
版本 0: root0
          /       \
      [1..2]=3   [3..4]=7
      /    \      /     \
   [1]=1 [2]=2 [3]=3 [4]=4

版本 1: root1 (新)
          /       \
  [1..2]=6*       [3..4]=7 (共享)
    /     \
$$1]=1   [2]=5*
```

星号标记新创建的节点。

#### 精简代码（简易版本）

C 语言（基于指针，求和树）

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int val;
    struct Node *left, *right;
} Node;

Node* build(int arr[], int l, int r) {
    Node* node = malloc(sizeof(Node));
    if (l == r) {
        node->val = arr[l];
        node->left = node->right = NULL;
        return node;
    }
    int mid = (l + r) / 2;
    node->left = build(arr, l, mid);
    node->right = build(arr, mid+1, r);
    node->val = node->left->val + node->right->val;
    return node;
}

Node* update(Node* prev, int l, int r, int pos, int new_val) {
    Node* node = malloc(sizeof(Node));
    if (l == r) {
        node->val = new_val;
        node->left = node->right = NULL;
        return node;
    }
    int mid = (l + r) / 2;
    if (pos <= mid) {
        node->left = update(prev->left, l, mid, pos, new_val);
        node->right = prev->right;
    } else {
        node->left = prev->left;
        node->right = update(prev->right, mid+1, r, pos, new_val);
    }
    node->val = node->left->val + node->right->val;
    return node;
}

int query(Node* node, int l, int r, int ql, int qr) {
    if (qr < l || ql > r) return 0;
    if (ql <= l && r <= qr) return node->val;
    int mid = (l + r)/2;
    return query(node->left, l, mid, ql, qr)
         + query(node->right, mid+1, r, ql, qr);
}

int main(void) {
    int A[5] = {0,1,2,3,4}; // 1-索引
    Node* root0 = build(A, 1, 4);
    Node* root1 = update(root0, 1, 4, 2, 5);
    printf("v0 sum[1..2]=%d\n", query(root0,1,4,1,2)); // 3
    printf("v1 sum[1..2]=%d\n", query(root1,1,4,1,2)); // 6
}
```

Python（递归，求和树）

```python
class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build(arr, l, r):
    if l == r:
        return Node(arr[l])
    mid = (l + r) // 2
    left = build(arr, l, mid)
    right = build(arr, mid+1, r)
    return Node(left.val + right.val, left, right)

def update(prev, l, r, pos, val):
    if l == r:
        return Node(val)
    mid = (l + r) // 2
    if pos <= mid:
        return Node(prev.val - prev.left.val + val,
                    update(prev.left, l, mid, pos, val),
                    prev.right)
    else:
        return Node(prev.val - prev.right.val + val,
                    prev.left,
                    update(prev.right, mid+1, r, pos, val))

def query(node, l, r, ql, qr):
    if qr < l or ql > r: return 0
    if ql <= l and r <= qr: return node.val
    mid = (l + r)//2
    return query(node.left, l, mid, ql, qr) + query(node.right, mid+1, r, ql, qr)

A = [0,1,2,3,4]
root0 = build(A, 1, 4)
root1 = update(root0, 1, 4, 2, 5)
print("v0 sum[1..2] =", query(root0, 1, 4, 1, 2))  # 3
print("v1 sum[1..2] =", query(root1, 1, 4, 1, 2))  # 6
```

#### 为什么它很重要

- 不可变版本 → 非常适合撤销系统、快照、持久化数据库
- 节省内存（每个版本 O(log n)）
- 每个版本都是完全功能独立且独立的

#### 一个温和的证明（为什么它有效）

每次更新影响 O(log n) 个节点。通过仅复制这些节点，我们维持了 O(log n) 的新内存开销。
因为所有旧节点仍然被引用，所以没有数据丢失。
因此，每个版本都是一致的且不可变的。

#### 自己动手试试

1.  用 `[1,2,3,4]` 构建版本 0
2.  更新(2,5) → 版本 1
3.  在两个版本中查询区间和(1,4) → 10, 13
4.  通过更新(3,1) 创建版本 2
5.  在每个版本中查询区间和(1,3)

#### 测试用例

| 版本 | A         | 查询       | 预期结果 |
| ---- | --------- | ---------- | -------- |
| 0    | [1,2,3,4] | 区间和[1..2] | 3        |
| 1    | [1,5,3,4] | 区间和[1..2] | 6        |
| 2    | [1,5,1,4] | 区间和[1..3] | 7        |

#### 复杂度

| 操作   | 时间复杂度 | 空间复杂度（每个版本） |
| ------ | ---------- | ---------------------- |
| 构建   | O(n)       | O(n)                   |
| 更新   | O(log n)   | O(log n)               |
| 查询   | O(log n)   | O(1)                   |

可持久化线段树的每个版本都是时间中的一个快照，完美地记住了每一个过去，而无需付出遗忘的代价。
### 250 二维线段树

二维线段树将经典的线段树扩展到二维，非常适合处理矩阵上的区间查询，例如子矩形内的求和、最小值或最大值。

这种数据结构让你能够提问：

> "矩形 (x1, y1) 到 (x2, y2) 内元素的总和是多少？"
> 并且仍然能在 O(log² n) 时间内得到答案。

#### 我们要解决什么问题？

我们希望在二维网格上高效地执行两种操作：

1. 区间查询：子矩阵内的求和/最小值/最大值
2. 点更新：修改一个元素并反映变化

朴素方法 → 每次查询 O(n²)
二维线段树 → 每次查询和更新 O(log² n)

#### 工作原理（通俗解释）

可以把二维线段树看作一个由线段树组成的线段树。

1. 外部树对行进行划分。
2. 外部树的每个节点都持有一个针对其行的内部线段树。

因此，每个节点都代表矩阵中的一个矩形。

构建时：

- 使用合并规则（如求和）在水平和垂直方向上合并子节点。

查询时：

- 合并与查询矩形重叠的节点的答案。

#### 示例

矩阵 A (3×3)：

|     | y=1 | y=2 | y=3 |
| --- | --- | --- | --- |
| x=1 | 2   | 1   | 3   |
| x=2 | 4   | 5   | 6   |
| x=3 | 7   | 8   | 9   |

查询矩形 `(1,1)` 到 `(2,2)`
→ 2 + 1 + 4 + 5 = 12

#### 核心思想

为每个行区间构建一个列线段树。
对于每个父节点（行区间），合并子节点的列树：

```
tree[x][y] = merge(tree[2*x][y], tree[2*x+1][y])
```

#### 示例详解（求和树）

1. 构建行线段树
2. 在每个行节点内部，构建列线段树
3. 查询矩形 `(x1,y1)` 到 `(x2,y2)`：

   * 在 x 区间上查询 → 合并垂直结果
   * 在每个 x 节点内部，在 y 区间上查询 → 合并水平结果

#### 精简代码（简易版本）

Python (求和二维线段树)

```python
class SegmentTree2D:
    def __init__(self, mat):
        self.n = len(mat)
        self.m = len(mat[0])
        self.tree = [[0]*(4*self.m) for _ in range(4*self.n)]
        self.mat = mat
        self.build_x(1, 0, self.n-1)

    def merge(self, a, b):
        return a + b  # 求和合并

    # 为固定的行区间构建列树
    def build_y(self, nodex, lx, rx, nodey, ly, ry):
        if ly == ry:
            if lx == rx:
                self.tree[nodex][nodey] = self.mat[lx][ly]
            else:
                self.tree[nodex][nodey] = self.merge(
                    self.tree[2*nodex][nodey], self.tree[2*nodex+1][nodey]
                )
            return
        midy = (ly + ry)//2
        self.build_y(nodex, lx, rx, 2*nodey, ly, midy)
        self.build_y(nodex, lx, rx, 2*nodey+1, midy+1, ry)
        self.tree[nodex][nodey] = self.merge(
            self.tree[nodex][2*nodey], self.tree[nodex][2*nodey+1]
        )

    # 构建行树
    def build_x(self, nodex, lx, rx):
        if lx != rx:
            midx = (lx + rx)//2
            self.build_x(2*nodex, lx, midx)
            self.build_x(2*nodex+1, midx+1, rx)
        self.build_y(nodex, lx, rx, 1, 0, self.m-1)

    def query_y(self, nodex, nodey, ly, ry, qly, qry):
        if qry < ly or qly > ry: return 0
        if qly <= ly and ry <= qry:
            return self.tree[nodex][nodey]
        midy = (ly + ry)//2
        return self.merge(
            self.query_y(nodex, 2*nodey, ly, midy, qly, qry),
            self.query_y(nodex, 2*nodey+1, midy+1, ry, qly, qry)
        )

    def query_x(self, nodex, lx, rx, qlx, qrx, qly, qry):
        if qrx < lx or qlx > rx: return 0
        if qlx <= lx and rx <= qrx:
            return self.query_y(nodex, 1, 0, self.m-1, qly, qry)
        midx = (lx + rx)//2
        return self.merge(
            self.query_x(2*nodex, lx, midx, qlx, qrx, qly, qry),
            self.query_x(2*nodex+1, midx+1, rx, qlx, qrx, qly, qry)
        )

    def query(self, x1, y1, x2, y2):
        return self.query_x(1, 0, self.n-1, x1, x2, y1, y2)

# 示例
A = [
    [2, 1, 3],
    [4, 5, 6],
    [7, 8, 9]
$$
seg2d = SegmentTree2D(A)
print(seg2d.query(0, 0, 1, 1))  # 期望输出 12
```

#### 为何重要

- 以 log² 复杂度支持二维查询
- 适用于求和、最小值、最大值、最大公约数、异或等操作
- 是高级二维数据结构（二维树状数组、KD 树等）的基础

#### 一个温和的证明（为何有效）

线段树的正确性依赖于结合律。
在二维中，我们将此属性扩展到两个维度。
每个节点代表一个矩形区域；合并子节点即可得到父节点的正确聚合值。

#### 动手尝试

1. 从 3×3 矩阵构建
2. 查询 `(0,0)-(2,2)` → 总和
3. 查询 `(1,1)-(2,2)` → 右下角 5+6+8+9=28
4. 修改 A[1][2]=10，重建，重新检查总和

#### 测试用例

| 矩阵                      | 查询         | 结果 |
| ------------------------- | ----------- | ---- |
| [[2,1,3],[4,5,6],[7,8,9]] | (0,0)-(1,1) | 12   |
| [[2,1,3],[4,5,6],[7,8,9]] | (1,1)-(2,2) | 28   |
| [[1,2],[3,4]]             | (0,0)-(1,1) | 10   |

#### 复杂度

| 操作     | 时间               | 空间  |
| -------- | ------------------ | ----- |
| 构建     | O(n·m·log n·log m) | O(n·m) |
| 查询     | O(log² n)          | O(1)   |
| 更新     | O(log² n)          | O(1)   |

二维线段树是你的网格超能力，像拼图一样合并矩形，一次一个对数时间。

## 第 26 节. 并查集
### 251 Make-Set

Make-Set 操作是并查集（Disjoint Set Union，简称 DSU，也称为 Union-Find）的起点。并查集是一种用于管理将元素划分为不相交集合的基本数据结构。

每个元素最初都位于自己的集合中，并作为自己的父节点。后续的操作（`Find` 和 `Union`）将高效地合并它们并跟踪它们之间的关系。

#### 我们要解决什么问题？

我们需要一种方式来表示一组不相交的集合（即互不重叠的组），同时支持以下操作：

1. Make-Set(x)：创建一个仅包含 `x` 的新集合
2. Find(x)：找到 x 所在集合的代表（领导者）
3. Union(x, y)：合并包含 x 和 y 的集合

这种数据结构是许多图算法（如 Kruskal 最小生成树、连通分量和聚类）的支柱。

#### 工作原理（通俗解释）

最初，每个元素都是自己的父节点，形成一个自环。
我们存储两个数组（或映射）：

- `parent[x]` → 指向 x 的父节点（初始为自身）
- `rank[x]` 或 `size[x]` → 有助于后续平衡合并操作

因此：

```
parent[x] = x  
rank[x] = 0
```

每个元素都是一棵孤立的树。随着时间的推移，合并操作会将树连接起来。

#### 示例

初始化 `n = 5` 个元素：`{1, 2, 3, 4, 5}`

| x | parent[x] | rank[x] | 含义       |
| - | --------- | ------- | ---------- |
| 1 | 1         | 0       | 自己的领导者 |
| 2 | 2         | 0       | 自己的领导者 |
| 3 | 3         | 0       | 自己的领导者 |
| 4 | 4         | 0       | 自己的领导者 |
| 5 | 5         | 0       | 自己的领导者 |

所有 `Make-Set` 操作后，每个元素都在自己的组中：

```
{1}, {2}, {3}, {4}, {5}
```

#### 可视化

```
1   2   3   4   5
↑   ↑   ↑   ↑   ↑
|   |   |   |   |
self self self self self
```

每个节点指向自身，形成五棵独立的树。

#### 微型代码（简易版本）

C 语言实现

```c
#include <stdio.h>

#define MAXN 1000

int parent[MAXN];
int rank_[MAXN];

void make_set(int v) {
    parent[v] = v;   // 自己的父节点
    rank_[v] = 0;    // 初始秩
}

int main(void) {
    int n = 5;
    for (int i = 1; i <= n; i++)
        make_set(i);
    
    printf("初始集合：\n");
    for (int i = 1; i <= n; i++)
        printf("元素 %d: parent=%d, rank=%d\n", i, parent[i], rank_[i]);
    return 0;
}
```

Python 实现

```python
def make_set(x, parent, rank):
    parent[x] = x
    rank[x] = 0

n = 5
parent = {}
rank = {}

for i in range(1, n+1):
    make_set(i, parent, rank)

print("Parent:", parent)
print("Rank:", rank)
# Parent: {1:1, 2:2, 3:3, 4:4, 5:5}
```

#### 为什么重要

- 是并查集的基础
- 支持高效的图算法
- 是按秩合并和路径压缩的构建模块
- 每个并查集都从 `Make-Set` 开始

#### 一个温和的证明（为什么它有效）

每个元素开始时都是单元素集合。
由于没有指针在不同元素之间交叉，所以集合是不相交的。
后续的 `Union` 操作通过合并树来保持这一属性，从不复制节点。

因此，`Make-Set` 保证了：

- 每个新节点都是独立的
- 父指针形成有效的森林

#### 自己动手试试

1. 使用 `Make-Set` 初始化 `{1..5}`
2. 打印 parent 和 rank 数组
3. 添加 `Union(1,2)` 并检查 parent 的变化
4. 在合并前验证所有 parent[x] = x

#### 测试用例

| 输入 | 操作           | 预期输出           |
| ---- | -------------- | ------------------ |
| n=3  | Make-Set(1..3) | parent = [1,2,3]   |
| n=5  | Make-Set(1..5) | rank = [0,0,0,0,0] |

#### 复杂度

| 操作     | 时间复杂度                     | 空间复杂度 |
| -------- | ------------------------------ | ---------- |
| Make-Set | O(1)                           | O(1)       |
| Find     | O(α(n)) 带路径压缩             | O(1)       |
| Union    | O(α(n)) 带按秩合并             | O(1)       |

`Make-Set` 步骤是你在并查集舞蹈中的第一步，简单、常数时间，并且对后续操作至关重要。
### 252 查找

查找操作是并查集（Disjoint Set Union，DSU），也称为 Union-Find 的核心。它用于定位包含给定元素的集合的代表元（领导者）。同一集合中的每个元素共享同一个领导者，这就是 DSU 识别哪些元素属于同一集合的方式。

为了使查找高效，`Find` 使用了一种称为路径压缩的巧妙优化，它能够扁平化树的结构，使得未来的查询操作几乎达到常数时间。

#### 我们要解决什么问题？

给定一个元素 `x`，我们想要确定它属于哪个集合。
每个集合由一个根节点（领导者）表示。

我们维护一个 `parent[]` 数组，使得：

- 如果 `x` 是根节点（领导者），则 `parent[x] = x`
- 否则 `parent[x] = x 的父节点`

`Find(x)` 操作递归地跟随 `parent[x]` 指针，直到到达根节点。

#### 工作原理（通俗解释）

将每个集合想象成一棵树，其中根节点是代表元。

例如：

```
1 ← 2 ← 3    4 ← 5
```

表示 `{1,2,3}` 是一个集合，`{4,5}` 是另一个集合。
Find(3) 操作跟随 `3→2→1`，发现 1 是根节点。

路径压缩通过将每个节点直接指向根节点来扁平化树，从而减少深度并加速未来的查找操作。

压缩后：

```
1 ← 2   1 ← 3
4 ← 5
```

现在 `Find(3)` 是 O(1) 时间复杂度。

#### 示例

初始状态：

| x | parent[x] | rank[x] |
| - | --------- | ------- |
| 1 | 1         | 1       |
| 2 | 1         | 0       |
| 3 | 2         | 0       |
| 4 | 4         | 0       |
| 5 | 4         | 0       |

执行 `Find(3)`：

1. 3 → parent[3] = 2
2. 2 → parent[2] = 1
3. 1 → 找到根节点

路径压缩重设 `parent[3] = 1`

结果：

| x | parent[x] |
| - | --------- |
| 1 | 1         |
| 2 | 1         |
| 3 | 1         |
| 4 | 4         |
| 5 | 4         |

#### 可视化

压缩前：

```
1  
↑  
2  
↑  
3
```

压缩后：

```
1  
↑ ↑  
2 3
```

现在所有节点都直接指向根节点 `1`。

#### 精简代码（简易版本）

C 语言实现

```c
#include <stdio.h>

#define MAXN 1000
int parent[MAXN];
int rank_[MAXN];

void make_set(int v) {
    parent[v] = v;
    rank_[v] = 0;
}

int find_set(int v) {
    if (v == parent[v])
        return v;
    // 路径压缩
    parent[v] = find_set(parent[v]);
    return parent[v];
}

int main(void) {
    int n = 5;
    for (int i = 1; i <= n; i++) make_set(i);
    parent[2] = 1;
    parent[3] = 2;
    printf("Find(3) 前: %d\n", parent[3]);
    printf("3 的根节点: %d\n", find_set(3));
    printf("Find(3) 后: %d\n", parent[3]);
}
```

Python 实现

```python
def make_set(x, parent, rank):
    parent[x] = x
    rank[x] = 0

def find_set(x, parent):
    if parent[x] != x:
        parent[x] = find_set(parent[x], parent)  # 路径压缩
    return parent[x]

n = 5
parent = {}
rank = {}
for i in range(1, n+1):
    make_set(i, parent, rank)

parent[2] = 1
parent[3] = 2

print("Find(3) 前:", parent)
print("3 的根节点:", find_set(3, parent))
print("Find(3) 后:", parent)
```

#### 为什么它很重要

- 实现 O(α(n)) 几乎常数时间的查询
- 是 Union-Find 效率的基础
- 显著降低树的高度
- 用于图算法（Kruskal、连通分量等）

#### 一个温和的证明（为什么它有效）

每个集合都是一棵有根树。
如果没有压缩，一系列合并操作可能会构建出一条很深的链。
通过路径压缩，每次 `Find` 都会扁平化路径，确保每次操作的摊还时间为常数（阿克曼反函数时间）。

因此，经过多次操作后，DSU 树会变得非常浅。

#### 自己动手试试

1.  初始化 `{1..5}`
2.  设置 `parent[2]=1, parent[3]=2`
3.  调用 `Find(3)` 并观察压缩效果
4.  检查压缩后 `parent[3] == 1`

#### 测试用例

| 压缩前的 parent | Find(x) | 压缩后的 parent |
| --------------- | ------- | --------------- |
| [1,1,2]         | Find(3) | [1,1,1]         |
| [1,2,3,4,5]     | Find(5) | [1,2,3,4,5]     |
| [1,1,1,1,1]     | Find(3) | [1,1,1,1,1]     |

#### 复杂度

| 操作      | 时间（摊还） | 空间 |
| --------- | ------------ | ---- |
| Find      | O(α(n))      | O(1) |
| Make-Set  | O(1)         | O(1) |
| Union     | O(α(n))      | O(1) |

查找操作是你的指南针，它总能将你引向你所在集合的领导者，并且通过路径压缩，每一步都会变得更快。
### 253 合并操作

合并操作是将并查集（Disjoint Set Union，DSU）联系在一起的纽带。在用 `Make-Set` 初始化集合并用 `Find` 查找代表元之后，`Union` 操作将两个不相交的集合合并为一个，这是动态连通性问题的基石。

为了保持树结构浅层且高效，我们将其与“按秩合并”或“按大小合并”等启发式策略结合使用。

#### 我们要解决什么问题？

我们希望合并包含元素 `a` 和 `b` 的两个集合。

如果 `a` 和 `b` 属于不同的集合，那么它们的代表元（`Find(a)` 和 `Find(b)`）就不同。
合并操作会连接这两个代表元，确保它们现在共享同一个代表元。

我们必须高效地完成此操作，避免不必要的树高度增加。这就是“按秩合并”发挥作用的地方。

#### 工作原理（通俗解释）

1. 找到两个元素的根节点：

   ```
   rootA = Find(a)
   rootB = Find(b)
   ```
2. 如果它们已经相等 → 属于同一集合，无需操作。
3. 否则，将较矮的树连接到较高的树下：

   * 如果 `rank[rootA] < rank[rootB]`：`parent[rootA] = rootB`
   * 否则如果 `rank[rootA] > rank[rootB]`：`parent[rootB] = rootA`
   * 否则（秩相等）：`parent[rootB] = rootA`，并且 `rank[rootA]++`

这样能保持合并后的森林平衡，确保操作的时间复杂度为 O(α(n))。

#### 示例

初始集合：`{1}, {2}, {3}, {4}`

| x | parent[x] | rank[x] |
| - | --------- | ------- |
| 1 | 1         | 0       |
| 2 | 2         | 0       |
| 3 | 3         | 0       |
| 4 | 4         | 0       |

执行 `Union(1,2)` → 将 2 连接到 1 下

```
parent[2] = 1  
rank[1] = 1
```

现在集合：`{1,2}, {3}, {4}`

执行 `Union(3,4)` → 将 4 连接到 3 下
集合：`{1,2}, {3,4}`

然后执行 `Union(2,3)` → 合并代表元（1 和 3）
将较低秩的树连接到较高秩的树下（两者秩均为 1 → 平局）
→ 将 3 连接到 1 下，`rank[1] = 2`

最终的父节点表：

| x | parent[x] | rank[x] |
| - | --------- | ------- |
| 1 | 1         | 2       |
| 2 | 1         | 0       |
| 3 | 1         | 1       |
| 4 | 3         | 0       |

现在所有元素都在根节点 1 下连通。✅

#### 可视化

初始：

```
1   2   3   4
```

Union(1,2):

```
  1
  |
  2
```

Union(3,4):

```
  3
  |
  4
```

Union(2,3):

```
    1
  / | \
 2  3  4
```

所有元素在 1 下连通。

#### 精简代码（简易版本）

C 语言实现

```c
#include <stdio.h>

#define MAXN 1000
int parent[MAXN];
int rank_[MAXN];

void make_set(int v) {
    parent[v] = v;
    rank_[v] = 0;
}

int find_set(int v) {
    if (v == parent[v]) return v;
    return parent[v] = find_set(parent[v]); // 路径压缩
}

void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b) {
        if (rank_[a] < rank_[b])
            parent[a] = b;
        else if (rank_[a] > rank_[b])
            parent[b] = a;
        else {
            parent[b] = a;
            rank_[a]++;
        }
    }
}

int main(void) {
    for (int i=1;i<=4;i++) make_set(i);
    union_sets(1,2);
    union_sets(3,4);
    union_sets(2,3);
    for (int i=1;i<=4;i++)
        printf("元素 %d: 父节点=%d, 秩=%d\n", i, parent[i], rank_[i]);
}
```

Python 实现

```python
def make_set(x, parent, rank):
    parent[x] = x
    rank[x] = 0

def find_set(x, parent):
    if parent[x] != x:
        parent[x] = find_set(parent[x], parent)
    return parent[x]

def union_sets(a, b, parent, rank):
    a = find_set(a, parent)
    b = find_set(b, parent)
    if a != b:
        if rank[a] < rank[b]:
            parent[a] = b
        elif rank[a] > rank[b]:
            parent[b] = a
        else:
            parent[b] = a
            rank[a] += 1

n = 4
parent = {}
rank = {}
for i in range(1, n+1): make_set(i, parent, rank)
union_sets(1,2,parent,rank)
union_sets(3,4,parent,rank)
union_sets(2,3,parent,rank)
print("父节点表:", parent)
print("秩表:", rank)
```

#### 为什么它很重要

- 是并查集（Union-Find）中的核心操作
- 能够高效地实现动态集合合并
- 对图算法至关重要：

  * Kruskal 最小生成树算法
  * 连通分量
  * 环检测

将“按秩合并”与“路径压缩”结合，对于海量数据集能实现接近常数的性能。

#### 一个温和的证明（为什么它有效）

`Union` 操作确保不相交性：

- 只合并具有不同根节点的集合
- 为每个集合维护一个代表元
  `按秩合并` 保持树平衡 → 树高度 ≤ log n
  结合路径压缩，每个操作的摊还成本是 α(n)，对于所有实际应用的 n 值，这基本上是常数。

#### 自己动手试试

1. 创建 5 个单元素集合
2. 执行 Union(1,2), Union(3,4), Union(2,3)
3. 验证所有元素共享同一个根节点
4. 检查合并前后的秩值

#### 测试用例

| 操作序列       | 期望的集合         | 代表元数组 |
| -------------- | ------------------ | ---------- |
| Make-Set(1..4) | {1},{2},{3},{4}    | [1,2,3,4]  |
| Union(1,2)     | {1,2}              | [1,1,3,4]  |
| Union(3,4)     | {3,4}              | [1,1,3,3]  |
| Union(2,3)     | {1,2,3,4}          | [1,1,1,1]  |

#### 复杂度

| 操作     | 时间（摊还） | 空间 |
| -------- | ------------ | ---- |
| Union    | O(α(n))      | O(1) |
| Find     | O(α(n))      | O(1) |
| Make-Set | O(1)         | O(1) |

`Union` 是集合之间的“握手”，当与智能优化结合时，它变得谨慎、平衡且极其迅速。
### 254 按秩合并

按秩合并是用于并查集（DSU）结构的一种平衡策略，旨在保持树结构的低深度。当合并两个集合时，我们不是随意地将一个根节点附加到另一个根节点之下，而是将较矮的树（秩较低）附加到较高的树（秩较高）之下。

这个简单技巧，结合路径压缩，赋予了 DSU 近乎常数的传奇性能，每次操作几乎为 O(1)。

#### 我们要解决什么问题？

如果不进行平衡，重复的 `Union` 操作可能会创建出长长的链，例如：

```
1 ← 2 ← 3 ← 4 ← 5
```

在最坏情况下，`Find(x)` 会变成 O(n)。

按秩合并通过维护树高的粗略度量来防止这种情况。
每个 `rank[root]` 记录其树的*近似高度*。

合并时：
- 将秩较低的树附加到秩较高的树之下。
- 如果秩相等，则选择一个作为父节点，并将其秩加 1。

#### 工作原理（通俗解释）

每个节点 `x` 拥有：
- `parent[x]` → 指向其所属集合的代表（首领）节点
- `rank[x]` → 对树高的估计值

执行 `Union(a, b)` 时：
1. `rootA = Find(a)`
2. `rootB = Find(b)`
3. 如果秩不同，将较小的树附加到较大的树下：
   ```
   if rank[rootA] < rank[rootB]:
       parent[rootA] = rootB
   else if rank[rootA] > rank[rootB]:
       parent[rootB] = rootA
   else:
       parent[rootB] = rootA
       rank[rootA] += 1
   ```
这确保了树只在必要时才增长。

#### 示例

初始状态：`{1}, {2}, {3}, {4}`
所有节点的 `rank = 0`

执行 `Union(1,2)` → 秩相同 → 将 2 附加到 1 下，`rank[1]` 增加为 1
```
1
|
2
```

执行 `Union(3,4)` → 秩相同 → 将 4 附加到 3 下，`rank[3]` 增加为 1
```
3
|
4
```

现在执行 `Union(1,3)` → 两个根的秩都为 1 → 平局 → 将 3 附加到 1 下，`rank[1]` 增加为 2
```
   1(rank=2)
  / \
 2   3
     |
     4
```
树的高度保持较小，平衡良好。

| 元素 | 父节点 | 秩 |
| ---- | ------ | -- |
| 1    | 1      | 2  |
| 2    | 1      | 0  |
| 3    | 1      | 1  |
| 4    | 3      | 0  |

#### 可视化

平衡前：
```
1 ← 2 ← 3 ← 4
```

按秩合并后：
```
    1
   / \
  2   3
      |
      4
```
平衡且高效 ✅

#### 精简代码（简易版本）

C 语言实现

```c
#include <stdio.h>

#define MAXN 1000
int parent[MAXN];
int rank_[MAXN];

void make_set(int v) {
    parent[v] = v;
    rank_[v] = 0;
}

int find_set(int v) {
    if (v == parent[v]) return v;
    return parent[v] = find_set(parent[v]); // 路径压缩
}

void union_by_rank(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b) {
        if (rank_[a] < rank_[b]) parent[a] = b;
        else if (rank_[a] > rank_[b]) parent[b] = a;
        else {
            parent[b] = a;
            rank_[a]++;
        }
    }
}

int main(void) {
    for (int i=1;i<=4;i++) make_set(i);
    union_by_rank(1,2);
    union_by_rank(3,4);
    union_by_rank(2,3);
    for (int i=1;i<=4;i++)
        printf("元素 %d: 父节点=%d 秩=%d\n", i, parent[i], rank_[i]);
}
```

Python 实现

```python
def make_set(x, parent, rank):
    parent[x] = x
    rank[x] = 0

def find_set(x, parent):
    if parent[x] != x:
        parent[x] = find_set(parent[x], parent)
    return parent[x]

def union_by_rank(a, b, parent, rank):
    a = find_set(a, parent)
    b = find_set(b, parent)
    if a != b:
        if rank[a] < rank[b]:
            parent[a] = b
        elif rank[a] > rank[b]:
            parent[b] = a
        else:
            parent[b] = a
            rank[a] += 1

n = 4
parent = {}
rank = {}
for i in range(1, n+1): make_set(i, parent, rank)
union_by_rank(1,2,parent,rank)
union_by_rank(3,4,parent,rank)
union_by_rank(2,3,parent,rank)
print("父节点字典:", parent)
print("秩字典:", rank)
```

#### 为什么这很重要

- 保持树结构平衡且深度浅
- 确保摊还时间复杂度为 O(α(n))
- 对于大规模连通性问题至关重要
- 用于 Kruskal 最小生成树算法、支持回滚的并查集以及网络聚类

#### 一个温和的证明（为什么它有效）

秩仅在两个高度相等的树合并时才增长。
因此，任何树的高度都以 log₂ n 为上界。
将此与路径压缩结合，每次操作的摊还复杂度变为 O(α(n))，其中 α(n)（反阿克曼函数）对于所有实际应用中的 n 值都小于 5。

#### 自己动手试试

1.  创建 8 个单元素集合
2.  按顺序执行合并操作：(1,2), (3,4), (1,3), (5,6), (7,8), (5,7), (1,5)
3.  观察秩和父节点结构的变化

#### 测试用例

| 操作       | 结果（父节点数组） | 秩          |
| ---------- | ------------------ | ----------- |
| Union(1,2) | [1,1,3,4]          | [1,0,0,0]   |
| Union(3,4) | [1,1,3,3]          | [1,0,1,0]   |
| Union(2,3) | [1,1,1,3]          | [2,0,1,0]   |

#### 复杂度

| 操作         | 时间（摊还） | 空间 |
| ------------ | ------------ | ---- |
| 按秩合并     | O(α(n))      | O(1) |
| 查找         | O(α(n))      | O(1) |
| 创建集合     | O(1)         | O(1) |

按秩合并是一门优雅合并的艺术，总是托起较小的树，让你的"森林"保持轻盈、平坦和快速。
### 255 路径压缩

路径压缩是让并查集（DSU）变得极速的秘密武器。每次执行 `Find` 操作时，它都会通过让每个被访问的节点直接指向根节点来*扁平化*结构。随着时间的推移，这会将深树转变为几乎扁平的结构，将昂贵的查找操作转变为接近常数时间。

#### 我们要解决什么问题？

在基础的 DSU 中，`Find(x)` 沿着父指针向上遍历树，直到到达根节点。如果没有压缩，频繁的合并操作可能会形成长链：

```
1 ← 2 ← 3 ← 4 ← 5
```

一次 `Find(5)` 将需要 5 步。在多次查询中重复这种情况，性能就会急剧下降。

路径压缩通过*重新连接*搜索路径上的所有节点，使其直接指向根节点，从而有效地扁平化树，解决了这个低效问题。

#### 它是如何工作的（通俗解释）

每当我们调用 `Find(x)` 时，我们递归地找到根节点，然后让路径上的每个节点都直接指向那个根节点。

伪代码：

```
Find(x):
    if parent[x] != x:
        parent[x] = Find(parent[x])
    return parent[x]
```

现在，未来对 `x` 及其后代的查找就变得瞬间完成了。

#### 示例

从一个链开始：

```
1 ← 2 ← 3 ← 4 ← 5
```

执行 `Find(5)`：

1. `Find(5)` 调用 `Find(4)`
2. `Find(4)` 调用 `Find(3)`
3. `Find(3)` 调用 `Find(2)`
4. `Find(2)` 调用 `Find(1)`（根节点）
5. 在返回的过程中，每个节点都被更新：

   ```
   parent[5] = 1
   parent[4] = 1
   parent[3] = 1
   parent[2] = 1
   ```

压缩后，结构变得扁平：

```
1
├── 2
├── 3
├── 4
└── 5
```

| 元素 | 压缩前父节点 | 压缩后父节点 |
| ---- | ------------ | ------------ |
| 1    | 1            | 1            |
| 2    | 1            | 1            |
| 3    | 2            | 1            |
| 4    | 3            | 1            |
| 5    | 4            | 1            |

下次调用 `Find(5)` → 只需一步。

#### 可视化

压缩前：

```
1 ← 2 ← 3 ← 4 ← 5
```

压缩后：

```
1
├─2
├─3
├─4
└─5
```

#### 精简代码（简易版本）

C 语言实现

```c
#include <stdio.h>

#define MAXN 100
int parent[MAXN];

void make_set(int v) {
    parent[v] = v;
}

int find_set(int v) {
    if (v != parent[v])
        parent[v] = find_set(parent[v]); // 路径压缩
    return parent[v];
}

void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b)
        parent[b] = a;
}

int main(void) {
    for (int i = 1; i <= 5; i++) make_set(i);
    union_sets(1, 2);
    union_sets(2, 3);
    union_sets(3, 4);
    union_sets(4, 5);
    find_set(5); // 压缩路径
    for (int i = 1; i <= 5; i++)
        printf("元素 %d → 父节点 %d\n", i, parent[i]);
}
```

Python 实现

```python
def make_set(x, parent):
    parent[x] = x

def find_set(x, parent):
    if parent[x] != x:
        parent[x] = find_set(parent[x], parent)
    return parent[x]

def union_sets(a, b, parent):
    a = find_set(a, parent)
    b = find_set(b, parent)
    if a != b:
        parent[b] = a

parent = {}
for i in range(1, 6):
    make_set(i, parent)
union_sets(1, 2, parent)
union_sets(2, 3, parent)
union_sets(3, 4, parent)
union_sets(4, 5, parent)
find_set(5, parent)
print("压缩后的父节点映射:", parent)
```

#### 为什么它很重要

- 通过扁平化树，极大地加速了 Find 操作。
- 与按秩合并（Union by Rank）完美搭配。
- 实现了摊还 O(α(n)) 的性能。
- 在图算法中至关重要，如 Kruskal 最小生成树算法、连通性检查和动态聚类。

#### 一个温和的证明（为什么它有效）

路径压缩确保每个节点的父节点直接跳转到根节点。
每个节点的深度随着每次 `Find` 操作呈指数级减少。
经过几次操作后，树变得几乎扁平，随后的每次 `Find` 操作都变为 O(1)。

结合按秩合并：

> 每次操作（Find 或 Union）都变为 O(α(n))，
> 其中 α(n) 是反阿克曼函数（对于任何实际输入都小于 5）。

#### 自己动手试试

1.  创建 6 个单元素集合。
2.  执行合并操作：(1,2), (2,3), (3,4), (4,5), (5,6)。
3.  调用 `Find(6)` 并打印压缩前后的父节点映射。
4.  观察链是如何扁平化的。
5.  测量压缩前后调用次数的差异。

#### 测试用例

| 操作序列         | 操作后的父节点映射          | 树深度 |
| ---------------- | --------------------------- | ------ |
| 无压缩           | `{1:1, 2:1, 3:2, 4:3, 5:4}` | 4      |
| 执行 Find(5) 后  | `{1:1, 2:1, 3:1, 4:1, 5:1}` | 1      |
| 执行 Find(3) 后  | `{1:1, 2:1, 3:1, 4:1, 5:1}` | 1      |

#### 复杂度

| 操作                     | 时间（摊还） | 空间 |
| ------------------------ | ------------ | ---- |
| 带路径压缩的 Find        | O(α(n))      | O(1) |
| 合并（带按秩合并）       | O(α(n))      | O(1) |
| 创建集合（Make-Set）     | O(1)         | O(1) |

路径压缩是 DSU 的扁平化魔法，一旦施展，你的集合会变得简洁，查找变得迅速，合并变得势不可挡。
### 256 支持回滚的并查集

支持回滚的并查集扩展了经典的不相交集合并（DSU）功能，使其能够*撤销*最近的操作。这在需要探索多种状态的场景中至关重要，例如回溯算法、动态连通性查询，或者可能需要撤销合并操作的离线问题。

这种版本不是破坏过去的状态，而是*记住*发生了什么变化，并且可以在常数时间内回滚到之前的版本。

#### 我们要解决什么问题？

标准的并查集操作（`Find`、`Union`）会改变数据结构，父指针和秩会被更新，因此你无法轻易地回退。

但是，如果你正在探索一个搜索树，或者处理需要以下操作的离线查询呢？
- 添加一条边（Union），
- 探索一条路径，
- 然后恢复到之前的数据结构？

支持回滚的并查集允许你撤销更改，非常适合基于时间的分治算法、树上的莫队算法以及离线动态图问题。

#### 工作原理（通俗解释）

原理很简单：
- 每次修改并查集时，将更改的内容记录到一个栈中。
- 当需要撤销时，从栈中弹出并撤销最后一次操作。

需要跟踪的操作：
- 当 `parent[b]` 改变时，存储 `(b, old_parent)`
- 当 `rank[a]` 改变时，存储 `(a, old_rank)`

你永远不执行路径压缩，因为它不容易撤销。相反，依靠按秩合并来保证效率。

#### 示例

让我们构建集合 `{1}, {2}, {3}, {4}`

执行：

1. `Union(1,2)` → 将 2 连接到 1 下
   * 压栈 `(2, parent=2, rank=None)`
2. `Union(3,4)` → 将 4 连接到 3 下
   * 压栈 `(4, parent=4, rank=None)`
3. `Union(1,3)` → 将 3 连接到 1 下
   * 压栈 `(3, parent=3, rank=None)`
   * 1 的秩增加 → 压栈 `(1, rank=0)`

回滚一次 → 撤销最后一次合并：
- 恢复 `parent[3] = 3`
- 恢复 `rank[1] = 0`

现在并查集回到步骤 2 之后的状态。

| 步骤             | 父节点映射 | 秩        | 栈                                         |
| ---------------- | ---------- | --------- | --------------------------------------------- |
| 初始化           | [1,2,3,4]  | [0,0,0,0] | []                                            |
| Union(1,2) 之后 | [1,1,3,4]  | [1,0,0,0] | [(2,2,None)]                                  |
| Union(3,4) 之后 | [1,1,3,3]  | [1,0,1,0] | [(2,2,None),(4,4,None)]                       |
| Union(1,3) 之后 | [1,1,1,3]  | [2,0,1,0] | [(2,2,None),(4,4,None),(3,3,None),(1,None,0)] |
| 回滚之后         | [1,1,3,3]  | [1,0,1,0] | [(2,2,None),(4,4,None)]                       |

#### 精简代码（简易版本）

C 语言实现（概念性）

```c
#include <stdio.h>

#define MAXN 1000
int parent[MAXN], rank_[MAXN];
typedef struct { int node, parent, rank_val, rank_changed; } Change;
Change stack[MAXN * 10];
int top = 0;

void make_set(int v) {
    parent[v] = v;
    rank_[v] = 0;
}

int find_set(int v) {
    while (v != parent[v]) v = parent[v];
    return v; // 无路径压缩
}

void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a == b) return;
    if (rank_[a] < rank_[b]) { int tmp = a; a = b; b = tmp; }
    stack[top++] = (Change){b, parent[b], 0, 0};
    parent[b] = a;
    if (rank_[a] == rank_[b]) {
        stack[top++] = (Change){a, 0, rank_[a], 1};
        rank_[a]++;
    }
}

void rollback() {
    if (top == 0) return;
    Change ch = stack[--top];
    if (ch.rank_changed) rank_[ch.node] = ch.rank_val;
    else parent[ch.node] = ch.parent;
}

int main() {
    for (int i=1;i<=4;i++) make_set(i);
    union_sets(1,2);
    union_sets(3,4);
    union_sets(1,3);
    printf("回滚前: parent[3]=%d\n", parent[3]);
    rollback();
    printf("回滚后: parent[3]=%d\n", parent[3]);
}
```

Python 实现

```python
class RollbackDSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n
        self.stack = []

    def find(self, x):
        while x != self.parent[x]:
            x = self.parent[x]
        return x  # 无路径压缩

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.stack.append(('p', b, self.parent[b]))
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.stack.append(('r', a, self.rank[a]))
            self.rank[a] += 1
        return True

    def rollback(self):
        if not self.stack:
            return
        typ, node, val = self.stack.pop()
        if typ == 'r':
            self.rank[node] = val
        else:
            self.parent[node] = val

dsu = RollbackDSU(5)
dsu.union(1,2)
dsu.union(3,4)
dsu.union(1,3)
print("回滚前:", dsu.parent)
dsu.rollback()
print("回滚后:", dsu.parent)
```

#### 为什么它很重要

- 支持可逆的合并操作
- 非常适合离线动态连通性问题
- 是基于时间的分治算法的核心
- 用于树上的莫队算法
- 有助于探索回溯（例如，递归上的 DSU）

#### 一个温和的证明（为什么它有效）

支持回滚的并查集之所以高效，是因为：
- 每次合并修改 O(1) 个字段
- 每次回滚恢复 O(1) 个字段
- `Find` 操作在 O(log n) 时间内运行（无路径压缩）
  因此，每个操作都是 O(log n) 或更好，并且完全可逆。

#### 自己动手试试

1. 创建 6 个集合 `{1}..{6}`
2. 执行合并：(1,2), (3,4), (2,3)
3. 回滚一次，验证集合 `{1,2}` 和 `{3,4}` 保持分离
4. 再次回滚，检查是否恢复到单个集合
5. 在每一步打印父节点和秩

#### 测试用例

| 步骤 | 操作       | 父节点     | 秩        | 栈大小 |
| ---- | ---------- | ---------- | --------- | ------ |
| 1    | Union(1,2) | [1,1,3,4] | [1,0,0,0] | 1      |
| 2    | Union(3,4) | [1,1,3,3] | [1,0,1,0] | 2      |
| 3    | Union(1,3) | [1,1,1,3] | [2,0,1,0] | 4      |
| 4    | 回滚       | [1,1,3,3] | [1,0,1,0] | 2      |

#### 复杂度

| 操作     | 时间复杂度 | 空间复杂度 |
| -------- | ---------- | ---------- |
| Make-Set | O(1)       | O(n)       |
| Union    | O(log n)   | O(1)       |
| Rollback | O(1)       | O(1)       |

支持回滚的并查集是你的时间机器，合并、探索、撤销。在持久性和性能之间取得了完美的平衡。
### 257 树上启发式合并

树上启发式合并是一种结合了并查集（DSU）和深度优先搜索（DFS）的混合技术，用于高效处理子树查询。它通常被称为"小到大"合并技术，在需要回答如下查询的问题中表现出色：

> "对于每个节点，统计其子树内的某种信息。"

我们不是在每个节点都从头重新计算，而是利用并查集的逻辑来复用已计算的数据，将较小的子树合并到较大的子树中，从而实现接近线性的复杂度。

#### 我们要解决什么问题？

许多树问题要求计算子树的聚合属性：

- 子树中不同颜色的数量
- 标签、值或权重的频率
- 子树的和、计数或众数

朴素的 DFS 在每个节点都重新计算 → O(n²) 时间复杂度。

树上启发式合并通过巧妙地合并子子树的信息来避免重复计算。

#### 工作原理（通俗解释）

将每个节点的子树视为一个信息包（例如颜色的多重集合）。
我们按 DFS 顺序处理子树，在每一步：

1. 先处理所有*小子节点*，并在使用后丢弃它们的数据。
2. 最后处理*重儿子*，并保留其数据（复用）。
3. 将小子树的数据增量地合并到大的子树中。

这种"小到大"的合并确保了每个元素最多移动 O(log n) 次，从而实现 O(n log n) 的总复杂度。

#### 示例

假设我们有一棵树：

```
     1
   / | \
  2  3  4
 / \    \
5   6    7
```

每个节点有一个颜色：

```
颜色 = [1, 2, 2, 1, 3, 3, 2]
```

目标：对于每个节点，统计其子树中不同颜色的数量。

朴素方法：为每个子树从头重新计算，O(n²)。
树上启发式合并方法：复用重儿子的颜色集合，合并小子树。

| 节点 | 子树         | 颜色      | 不同计数 |
| ---- | --------------- | ------- | -------------- |
| 5    | [5]             | {3}     | 1              |
| 6    | [6]             | {3}     | 1              |
| 2    | [2,5,6]         | {2,3}   | 2              |
| 3    | [3]             | {2}     | 1              |
| 7    | [7]             | {2}     | 1              |
| 4    | [4,7]           | {1,2}   | 2              |
| 1    | [1,2,3,4,5,6,7] | {1,2,3} | 3              |

每个子树只将其小的颜色集合合并到大的集合中一次 → 高效。

#### 逐步思路

1.  DFS 计算子树大小
2.  识别重儿子（最大的子树）
3.  再次 DFS：
    *   处理所有*轻儿子*（小子树），丢弃结果
    *   处理*重儿子*，保留其结果
    *   将所有轻儿子的数据合并到重儿子的数据中
4.  合并后记录每个节点的答案

#### 精简代码（简易版本）

C 语言实现（概念性）

```c
#include <stdio.h>
#include <vector>
#include <set>

#define MAXN 100005
using namespace std;

vector<int> tree[MAXN];
int color[MAXN];
int subtree_size[MAXN];
int answer[MAXN];
int freq[MAXN];
int n;

void dfs_size(int u, int p) {
    subtree_size[u] = 1;
    for (int v : tree[u])
        if (v != p) {
            dfs_size(v, u);
            subtree_size[u] += subtree_size[v];
        }
}

void add_color(int u, int p, int val) {
    freq[color[u]] += val;
    for (int v : tree[u])
        if (v != p) add_color(v, u, val);
}

void dfs(int u, int p, bool keep) {
    int bigChild = -1, maxSize = -1;
    for (int v : tree[u])
        if (v != p && subtree_size[v] > maxSize)
            maxSize = subtree_size[v], bigChild = v;

    // 处理小子节点
    for (int v : tree[u])
        if (v != p && v != bigChild)
            dfs(v, u, false);

    // 处理重儿子
    if (bigChild != -1) dfs(bigChild, u, true);

    // 合并小子节点的信息
    for (int v : tree[u])
        if (v != p && v != bigChild)
            add_color(v, u, +1);

    freq[color[u]]++;
    // 示例查询：统计不同颜色数量
    answer[u] = 0;
    for (int i = 1; i <= n; i++)
        if (freq[i] > 0) answer[u]++;

    if (!keep) add_color(u, p, -1);
}

int main() {
    n = 7;
    // 构建树，设置颜色...
    // dfs_size(1,0); dfs(1,0,true);
}
```

Python 实现（简化版）

```python
from collections import defaultdict

def dfs_size(u, p, tree, size):
    size[u] = 1
    for v in tree[u]:
        if v != p:
            dfs_size(v, u, tree, size)
            size[u] += size[v]

def add_color(u, p, tree, color, freq, val):
    freq[color[u]] += val
    for v in tree[u]:
        if v != p:
            add_color(v, u, tree, color, freq, val)

def dfs(u, p, tree, size, color, freq, ans, keep):
    bigChild, maxSize = -1, -1
    for v in tree[u]:
        if v != p and size[v] > maxSize:
            maxSize, bigChild = size[v], v

    for v in tree[u]:
        if v != p and v != bigChild:
            dfs(v, u, tree, size, color, freq, ans, False)

    if bigChild != -1:
        dfs(bigChild, u, tree, size, color, freq, ans, True)

    for v in tree[u]:
        if v != p and v != bigChild:
            add_color(v, u, tree, color, freq, 1)

    freq[color[u]] += 1
    ans[u] = len([c for c in freq if freq[c] > 0])

    if not keep:
        add_color(u, p, tree, color, freq, -1)

# 示例用法
n = 7
tree = {1:[2,3,4], 2:[1,5,6], 3:[1], 4:[1,7], 5:[2], 6:[2], 7:[4]}
color = {1:1,2:2,3:2,4:1,5:3,6:3,7:2}
size = {}
ans = {}
freq = defaultdict(int)
dfs_size(1,0,tree,size)
dfs(1,0,tree,size,color,freq,ans,True)
print(ans)
```

#### 为什么它重要

- 以 O(n log n) 时间复杂度处理子树查询
- 适用于静态树和离线查询
- 复用重子树的计算结果
- 是树上莫队算法和颜色频率问题的基础

#### 一个温和的证明（为什么它有效）

每个节点的颜色（或元素）最多被合并 O(log n) 次：

- 每次合并，它都从一个较小的集合移动到一个较大的集合。
- 因此总合并次数 = O(n log n)。

`keep` 标志确保我们只保留大的子树，丢弃轻子树以节省内存和时间。

#### 亲自尝试

1.  给一棵 8 个节点的树分配随机颜色。
2.  使用树上启发式合并计算每个子树的不同颜色数量。
3.  与暴力 DFS 的结果进行比较。
4.  验证两者结果一致，但树上启发式合并运行更快。

#### 测试用例

| 节点 | 子树颜色      | 不同计数 |
| ---- | -------------- | -------------- |
| 5    | {3}            | 1              |
| 6    | {3}            | 1              |
| 2    | {2,3}          | 2              |
| 3    | {2}            | 1              |
| 7    | {2}            | 1              |
| 4    | {1,2}          | 2              |
| 1    | {1,2,3}        | 3              |

#### 复杂度

| 操作         | 时间复杂度 | 空间复杂度 |
| ------------ | ---------- | ---------- |
| DFS 遍历     | O(n)       | O(n)       |
| DSU 合并     | O(n log n) | O(n)       |
| 总体         | O(n log n) | O(n)       |

树上启发式合并是你的子树超能力，聪明地合并，丢弃轻的，优雅地征服查询。
### 258 Kruskal 最小生成树算法（使用 DSU）

Kruskal 算法是一种经典的贪心方法，用于构建最小生成树（MST）。最小生成树是连接所有顶点的一个边集子集，其总权重最小且不包含环。该算法依赖于并查集（DSU）来高效地检查添加一条边是否会形成环。

使用并查集后，Kruskal 的最小生成树算法变得简洁、快速且概念优雅，它按排序顺序逐条边地构建生成树。

#### 我们要解决什么问题？

给定一个具有 $n$ 个顶点和 $m$ 条边的连通加权图，我们希望找到一棵满足以下条件的树：

- 连接所有顶点（生成树）
- 没有环（树）
- 最小化总边权重

一种朴素的方法是测试边的每个子集，复杂度为 $O(2^m)$。
Kruskal 算法使用边排序和并查集（DSU）结构，将复杂度降低到 $O(m \log m)$。

#### 工作原理（通俗解释）

该算法遵循三个步骤：

1. 将所有边按权重升序排序。
2. 将每个顶点初始化为其自身的集合（`Make-Set`）。
3. 按顺序处理每条边 $(u, v, w)$：
   - 如果 $\text{Find}(u) \ne \text{Find}(v)$，说明顶点位于不同的连通分量中 → 将该边加入 MST 并使用 `Union` 合并集合。
   - 否则，跳过该边，因为它会形成环。

重复此过程，直到 MST 包含 $n - 1$ 条边。

#### 示例

图：

| 边   | 权重 |
| ---- | ---- |
| A–B  | 1    |
| B–C  | 4    |
| A–C  | 3    |
| C–D  | 2    |

排序后的边：(A–B, 1), (C–D, 2), (A–C, 3), (B–C, 4)

逐步过程：

| 步骤 | 边        | 操作         | MST 边                    | MST 权重 | 父节点映射               |
| ---- | --------- | ------------ | ------------------------- | -------- | ------------------------ |
| 1    | (A,B,1)   | 添加         | [(A,B)]                   | 1        | A→A, B→A                 |
| 2    | (C,D,2)   | 添加         | [(A,B), (C,D)]            | 3        | C→C, D→C                 |
| 3    | (A,C,3)   | 添加         | [(A,B), (C,D), (A,C)]     | 6        | A→A, B→A, C→A, D→C       |
| 4    | (B,C,4)   | 跳过（会成环） | –                         | –        | –                        |

✅ MST 总权重 = 6
✅ 边数 = 3 = n − 1

#### 可视化

之前：

```
A --1-- B
|      /
3    4
|  /
C --2-- D
```

之后：

```
A
| \
1  3
B   C
    |
    2
    D
```

#### 精简代码（简易版本）

C 语言实现

```c
#include <stdio.h>
#include <stdlib.h>

#define MAXN 100
#define MAXM 1000

typedef struct {
    int u, v, w;
} Edge;

int parent[MAXN], rank_[MAXN];
Edge edges[MAXM];

int cmp(const void *a, const void *b) {
    return ((Edge*)a)->w - ((Edge*)b)->w;
}

void make_set(int v) {
    parent[v] = v;
    rank_[v] = 0;
}

int find_set(int v) {
    if (v != parent[v]) parent[v] = find_set(parent[v]);
    return parent[v];
}

void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b) {
        if (rank_[a] < rank_[b]) parent[a] = b;
        else if (rank_[a] > rank_[b]) parent[b] = a;
        else { parent[b] = a; rank_[a]++; }
    }
}

int main() {
    int n = 4, m = 4;
    edges[0] = (Edge){0,1,1};
    edges[1] = (Edge){1,2,4};
    edges[2] = (Edge){0,2,3};
    edges[3] = (Edge){2,3,2};
    qsort(edges, m, sizeof(Edge), cmp);

    for (int i = 0; i < n; i++) make_set(i);

    int total = 0;
    printf("Edges in MST:\n");
    for (int i = 0; i < m; i++) {
        int u = edges[i].u, v = edges[i].v, w = edges[i].w;
        if (find_set(u) != find_set(v)) {
            union_sets(u, v);
            total += w;
            printf("%d - %d (w=%d)\n", u, v, w);
        }
    }
    printf("Total Weight = %d\n", total);
}
```

Python 实现

```python
def make_set(parent, rank, v):
    parent[v] = v
    rank[v] = 0

def find_set(parent, v):
    if parent[v] != v:
        parent[v] = find_set(parent, parent[v])
    return parent[v]

def union_sets(parent, rank, a, b):
    a, b = find_set(parent, a), find_set(parent, b)
    if a != b:
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

def kruskal(n, edges):
    parent, rank = {}, {}
    for i in range(n):
        make_set(parent, rank, i)
    mst, total = [], 0
    for u, v, w in sorted(edges, key=lambda e: e[2]):
        if find_set(parent, u) != find_set(parent, v):
            union_sets(parent, rank, u, v)
            mst.append((u, v, w))
            total += w
    return mst, total

edges = [(0,1,1),(1,2,4),(0,2,3),(2,3,2)]
mst, total = kruskal(4, edges)
print("MST:", mst)
print("Total Weight:", total)
```

#### 为什么它很重要？

- 贪心且优雅：简单的排序 + DSU 逻辑
- 高级主题的基础：
  * 最小生成森林
  * 动态连通性
  * MST 变体（最大生成树、次优生成树等）
- 非常适合边列表输入

#### 一个温和的证明（为什么它有效）

根据割性质：
跨越任何割的最小边都属于 MST。
由于 Kruskal 算法总是选择最小的不会形成环的边，因此它构建了一个有效的 MST。

每次合并操作都在不形成环的情况下合并连通分量 → 最终得到一棵生成树。

#### 自己动手试试

1.  构建一个包含 5 个节点、随机边和权重的图。
2.  对边进行排序，逐步跟踪合并操作。
3.  画出 MST。
4.  与 Prim 算法的结果进行比较，它们应该一致。

#### 测试用例

| 图                                   | MST 边         | 权重 |
| ------------------------------------ | -------------- | ---- |
| 三角形 (1–2:1, 2–3:2, 1–3:3)         | (1–2, 2–3)     | 3    |
| 正方形（4 条边，权重分别为 1,2,3,4） | 3 条最小的边   | 6    |

#### 复杂度

| 步骤           | 时间复杂度   |
| -------------- | ------------ |
| 边排序         | O(m log m)   |
| DSU 操作       | O(m α(n))    |
| 总计           | O(m log m)   |

| 空间复杂度 | O(n + m) |

Kruskal 的最小生成树算法是贪心与并查集之间优雅的握手，它总是以最轻的方式连接，永不回头绕圈。
### 259 连通分量（使用并查集）

连通分量是指一组顶点，其中每个节点都可以通过一系列边到达其他任何节点。使用并查集（DSU），我们可以高效地识别和标记图中的这些连通分量，即使对于海量数据集也是如此。

与通过深度优先搜索（DFS）或广度优先搜索（BFS）探索每个区域不同，并查集以增量方式构建连通关系，在边出现时合并节点。

#### 我们要解决什么问题？

给定一个图（有向或无向），我们想要回答：

- 存在多少个连通分量？
- 哪些顶点属于同一个连通分量？
- 在 `u` 和 `v` 之间是否存在路径？

一种朴素的方法（对每个节点进行 DFS）运行时间为 O(n + m)，但可能需要递归或邻接遍历。
使用并查集，我们可以直接处理边列表，时间接近常数级的摊还时间。

#### 工作原理（通俗解释）

每个顶点最初都在自己的分量中。
对于每条边 `(u, v)`：

- 如果 `Find(u) != Find(v)`，它们在不同的分量中 → 执行 Union(u, v)
- 否则，跳过（已经连通）

处理完所有边后，共享同一个根节点的所有顶点都属于同一个连通分量。

#### 示例

图：

```
1, 2     3, 4
      \   /
        5
```

边： (1–2), (2–5), (3–5), (3–4)

逐步执行：

| 步骤 | 边    | 操作   | 分量                  |
| ---- | ----- | ------ | --------------------- |
| 1    | (1,2) | Union  | {1,2}, {3}, {4}, {5} |
| 2    | (2,5) | Union  | {1,2,5}, {3}, {4}    |
| 3    | (3,5) | Union  | {1,2,3,5}, {4}       |
| 4    | (3,4) | Union  | {1,2,3,4,5}          |

✅ 全部连通 → 1 个分量

#### 可视化

处理前：

```
1   2   3   4   5
```

执行合并后：

```
1—2—5—3—4
```

一个大的连通分量。

#### 精简代码（简易版本）

C 语言实现

```c
#include <stdio.h>

#define MAXN 100
int parent[MAXN], rank_[MAXN];

void make_set(int v) {
    parent[v] = v;
    rank_[v] = 0;
}

int find_set(int v) {
    if (v != parent[v])
        parent[v] = find_set(parent[v]);
    return parent[v];
}

void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b) {
        if (rank_[a] < rank_[b]) parent[a] = b;
        else if (rank_[a] > rank_[b]) parent[b] = a;
        else { parent[b] = a; rank_[a]++; }
    }
}

int main() {
    int n = 5;
    int edges[][2] = {{1,2},{2,5},{3,5},{3,4}};
    for (int i=1; i<=n; i++) make_set(i);
    for (int i=0; i<4; i++)
        union_sets(edges[i][0], edges[i][1]);
    
    int count = 0;
    for (int i=1; i<=n; i++)
        if (find_set(i) == i) count++;
    printf("连通分量数量: %d\n", count);
}
```

Python 实现

```python
def make_set(parent, rank, v):
    parent[v] = v
    rank[v] = 0

def find_set(parent, v):
    if parent[v] != v:
        parent[v] = find_set(parent, parent[v])
    return parent[v]

def union_sets(parent, rank, a, b):
    a, b = find_set(parent, a), find_set(parent, b)
    if a != b:
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

def connected_components(n, edges):
    parent, rank = {}, {}
    for i in range(1, n+1):
        make_set(parent, rank, i)
    for u, v in edges:
        union_sets(parent, rank, u, v)
    roots = {find_set(parent, i) for i in parent}
    components = {}
    for i in range(1, n+1):
        root = find_set(parent, i)
        components.setdefault(root, []).append(i)
    return components

edges = [(1,2),(2,5),(3,5),(3,4)]
components = connected_components(5, edges)
print("连通分量:", components)
print("数量:", len(components))
```

输出：

```
连通分量: {1: [1, 2, 3, 4, 5]}
数量: 1
```

#### 为什么它很重要

- 快速回答连通性问题
- 直接处理边列表（无需邻接矩阵）
- 构成 Kruskal 最小生成树等算法的核心
- 可扩展至动态连通性和离线查询

#### 一个温和的证明（为什么它有效）

并查集形成一片森林，每个分量对应一棵树。
当且仅当存在一条边连接两棵树时，每次合并操作才会合并它们。
不会引入环；最终的根节点标记了不同的连通分量。

每个顶点最终都恰好链接到一个代表元。

#### 自己动手试试

1.  构建一个有 6 个节点和 2 个不连通子图的图。
2.  对边执行并查集合并操作。
3.  统计唯一根节点的数量。
4.  打印分组 `{根节点: [成员列表]}`。

#### 测试用例

| 图                 | 边                | 连通分量           |
| ------------------ | ----------------- | ------------------ |
| 1–2–3, 4–5         | (1,2),(2,3),(4,5) | {1,2,3}, {4,5}, {6} |
| 完全图 (1–n)       | 所有节点对        | 1                  |
| 空图               | 无                | n                  |

#### 复杂度

| 操作       | 时间（摊还） | 空间 |
| ---------- | ------------ | ---- |
| Make-Set   | O(1)         | O(n) |
| Union      | O(α(n))      | O(1) |
| Find       | O(α(n))      | O(1) |
| 总计（m 条边） | O(m α(n))    | O(n) |

连通分量（使用并查集），一种清晰且可扩展的方法，用于揭示任何图中隐藏的簇。
### 260 离线查询并查集

离线查询并查集是对标准并查集的一种巧妙变体，用于处理图中随时间变化的连通性查询，特别是当边被添加或删除时。

我们不实时处理更新（在线处理），而是先收集所有查询，然后**反向**处理它们，利用并查集在“撤销”删除或反向模拟时间线的过程中高效地追踪连接关系。

#### 我们要解决什么问题？

我们经常遇到这样的问题：

*   “删除这些边后，节点 `u` 和 `v` 是否仍然连通？”
*   “如果我们随时间添加边，`u` 和 `v` 在何时会变得连通？”

在线处理很困难，因为并查集不直接支持删除操作。
诀窍在于：**反转时间**，将删除视为反向的添加，并离线处理查询。

#### 工作原理（通俗解释）

1.  按发生顺序记录所有事件：
    *   边的添加或删除
    *   连通性查询
2.  反转时间线：
    *   从最后一个事件开始反向处理
    *   每个“删除边”事件变成“添加边”事件
    *   查询按反向顺序回答
3.  使用并查集：
    *   当边（在反向过程中）出现时，每次合并操作都会合并连通分量
    *   处理查询时，检查 `Find(u) == Find(v)`
4.  最后，反转答案顺序以匹配原始顺序。

#### 示例

想象一个图：

```
1, 2, 3
```

事件（按时间顺序）：

1.  Query(1,3)?
2.  Remove edge (2,3)
3.  Query(1,3)?

我们无法轻松在线处理删除操作，所以我们反转：

```
反向顺序：
1. Query(1,3)?
2. Add (2,3)
3. Query(1,3)?
```

逐步（反向）处理：

| 步骤 | 操作         | 动作                     | 答案 |
| ---- | ------------ | ------------------------ | ---- |
| 1    | Query(1,3)   | 1 和 3 不连通            | No   |
| 2    | Add(2,3)     | Union(2,3)               | –    |
| 3    | Query(1,3)   | 1–2–3 连通               | Yes  |

反转答案：[Yes, No]

✅ 最终输出：

*   查询 1：Yes
*   查询 2：No

#### 可视化

正向时间：

```
1—2—3
```

→ 移除 (2,3) →

```
1—2   3
```

反向时间：
从 `1—2   3` 开始
→ 添加 (2,3) → `1—2—3`

我们通过在反向过程中合并边来随时间重建连通性。

#### 精简代码（简易版本）

Python 实现

```python
def make_set(parent, rank, v):
    parent[v] = v
    rank[v] = 0

def find_set(parent, v):
    if parent[v] != v:
        parent[v] = find_set(parent, parent[v])
    return parent[v]

def union_sets(parent, rank, a, b):
    a, b = find_set(parent, a), find_set(parent, b)
    if a != b:
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

# 示例
n = 3
edges = {(1,2), (2,3)}
queries = [
    ("?", 1, 3),
    ("-", 2, 3),
    ("?", 1, 3)
$$

# 反转事件
events = list(reversed(queries))

parent, rank = {}, {}
for i in range(1, n+1):
    make_set(parent, rank, i)

active_edges = set(edges)
answers = []

for e in events:
    if e[0] == "?":
        _, u, v = e
        answers.append("YES" if find_set(parent, u) == find_set(parent, v) else "NO")
    elif e[0] == "-":
        _, u, v = e
        union_sets(parent, rank, u, v)

answers.reverse()
for ans in answers:
    print(ans)
```

输出：

```
YES
NO
```

#### 为什么它很重要

*   无需回滚即可处理边删除
*   非常适合离线动态连通性问题
*   用于解决如下问题：
    *   “在 k 次删除后，u 和 v 是否连通？”
    *   “u 和 v 最早在何时变得连通？”
*   是动态树和基于时间的分治算法背后的核心思想

#### 一个温和的证明（为什么它有效）

并查集是单调的，它支持添加边，但不支持删除边。
通过反转时间，所有删除都变成了添加。
因此，我们可以在时间上反向维护有效的连通性信息，
并且能够正确回答仅依赖于图连通性的查询。

之后反转答案即可恢复其原始顺序。

#### 自己动手试试

1.  创建一个包含 5 个节点和边 (1–2, 2–3, 3–4, 4–5) 的图
2.  依次移除 (3–4), (2–3)
3.  在每次移除后询问 (1,5) 之间的连通性
4.  反转时间线，用并查集模拟

#### 测试用例

| 事件序列                       | 结果       |
| ------------------------------ | ---------- |
| [?, 1–3], [–, 2–3], [?, 1–3]   | [YES, NO]  |
| [–, 1–2], [?, 1–3]             | [NO]       |
| [?, 4–5] (无边)                | [NO]       |

#### 复杂度

| 操作                     | 时间（均摊）   | 空间    |
| ------------------------ | -------------- | ------- |
| Make-Set                 | O(1)           | O(n)    |
| Union                    | O(α(n))        | O(1)    |
| Find                     | O(α(n))        | O(1)    |
| 总计（Q 个查询，E 条边） | O((Q+E) α(n))  | O(n+E)  |

离线查询并查集是你的时间反转工具，翻转故事，将边加回，揭示跨越历史的连通性。

## 第 27 章 概率数据结构
### 261 布隆过滤器插入

布隆过滤器是一种紧凑的概率型数据结构，用于成员资格测试。它可以告诉你一个元素**肯定不存在**或**可能存在**，但绝不会给出假阴性结果。

在布隆过滤器中的插入操作简单而优雅：使用多个哈希函数对元素进行哈希，并在一个位数组中把对应的位设置为 `1`。

#### 我们要解决什么问题？

你有一个庞大的数据集，可能有数百万甚至数十亿个键，而你只想问：

> “我之前见过这个吗？”

一个普通的哈希集合会占用巨大的内存。
布隆过滤器提供了一个轻量级的替代方案：

- 没有假阴性（可以安全地跳过）
- 内存占用小
- 固定大小的位数组

应用于以下系统：

- 数据库（缓存、去重）
- 网络爬虫（已访问的 URL）
- 分布式系统（HBase、Cassandra、Bigtable）

#### 工作原理（通俗解释）

布隆过滤器只是一个长度为 `m` 的位数组（初始全为 0），外加 k 个独立的哈希函数。

要插入一个元素 `x`：

1.  计算 `k` 个哈希值：`h1(x), h2(x), ..., hk(x)`
2.  将每个哈希值映射到 `[0, m-1]` 范围内的一个索引
3.  将所有 `bit[h_i(x)]` 设置为 1

因此，每个元素会点亮多个位。之后，要检查成员资格，我们查看这些相同的位，如果其中任何一位是 0，则该元素从未被插入过。

#### 示例

让我们构建一个布隆过滤器，参数为：

- `m = 10` 位
- `k = 3` 个哈希函数

插入 "cat"：

```
h1(cat) = 2
h2(cat) = 5
h3(cat) = 7
```

将第 2、5、7 位设置为 1：

| 索引 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| ---- | - | - | - | - | - | - | - | - | - | - |
| 值   | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |

插入 "dog"：

```
h1(dog) = 1
h2(dog) = 5
h3(dog) = 9
```

现在的位数组：

| 索引 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| ---- | - | - | - | - | - | - | - | - | - | - |
| 值   | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 1 |

#### 可视化

每次插入都在多个位置留下“足迹”：

```
Insert(x):
  for i in [1..k]:
     bit[ h_i(x) ] = 1
```

位的重叠允许巨大的压缩，但当不相关的键共享相同的位时，也会导致假阳性。

#### 微型代码（简易版本）

C 语言实现（概念性）

```c
#include <stdio.h>
#include <string.h>

#define M 10
#define K 3

int bitset[M];

int hash1(int x) { return x % M; }
int hash2(int x) { return (x * 3 + 1) % M; }
int hash3(int x) { return (x * 7 + 5) % M; }

void insert(int x) {
    int h[K] = {hash1(x), hash2(x), hash3(x)};
    for (int i = 0; i < K; i++)
        bitset[h[i]] = 1;
}

void print_bits() {
    for (int i = 0; i < M; i++) printf("%d ", bitset[i]);
    printf("\n");
}

int main() {
    memset(bitset, 0, sizeof(bitset));
    insert(42);
    insert(23);
    print_bits();
}
```

Python 实现

```python
m, k = 10, 3
bitset = [0] * m

def hash_functions(x):
    return [(hash(x) + i * i) % m for i in range(k)]

def insert(x):
    for h in hash_functions(x):
        bitset[h] = 1

def display():
    print("Bit array:", bitset)

insert("cat")
insert("dog")
display()
```

#### 为什么重要

- 极其节省空间
- 无需存储实际数据
- 适用于成员资格过滤器、重复检测以及在昂贵查找之前的预检查
- 近似数据结构的基石

#### 一个温和的证明（为什么它有效）

每个位初始为 0。
每次插入将 `k` 个位翻转为 1。
查询时，如果所有 `k` 个位都为 1，则返回“可能存在”，否则返回“不存在”。
因此：

- 假阴性：不可能（永远不会将位重置为 0）
- 假阳性：可能，由于哈希冲突

假阳性的概率 ≈ $((1 - e^{-kn/m})^k)$

选择合适的 `(m)` 和 `(k)` 可以在准确性和内存之间取得平衡。

#### 亲自尝试

1.  选择 `m = 20`, `k = 3`
2.  插入 {"apple", "banana", "grape"}
3.  打印位数组
4.  查询 "mango" → 很可能返回“可能存在”（假阳性）

#### 测试用例

| 插入的元素 | 查询 | 结果                               |
| ---------- | ---- | ---------------------------------- |
| {cat, dog} | cat  | 可能存在（真阳性）                 |
| {cat, dog} | dog  | 可能存在（真阳性）                 |
| {cat, dog} | fox  | 可能存在 / 不存在（可能出现假阳性） |

#### 复杂度

| 操作         | 时间                | 空间 |
| ------------ | ------------------- | ---- |
| 插入         | O(k)                | O(m) |
| 查询         | O(k)                | O(m) |
| 假阳性概率 | ≈ $(1 - e^{-kn/m})^k$ | –    |

布隆过滤器插入，一次写入，可能永久有效。紧凑、快速且概率强大。
### 262 布隆过滤器查询

布隆过滤器查询用于检查一个元素是否*可能*存在于集合中。它使用与插入时相同的 k 个哈希函数和位数组，但并非设置位，而是简单地测试它们。

其神奇之处在于：

- 如果任何一位是 `0`，则该元素从未被插入过（肯定不存在）。
- 如果所有位都是 `1`，则该元素可能存在（也许是）。

没有假阴性，如果布隆过滤器说"否"，它总是正确的。

#### 我们要解决什么问题？

在处理海量数据集（网络爬虫、缓存、键值存储）时，我们需要一种快速且内存高效的方式来回答：

> "我之前见过这个吗？"

但完全存储的代价很高。
布隆过滤器允许我们通过早期自信地排除项目，来跳过昂贵的查找操作。

典型用例：

- 数据库：避免对缺失的键进行磁盘查找
- 网络爬虫：跳过重新访问已知的 URL
- 网络：缓存成员资格检查

#### 工作原理（通俗解释）

一个布隆过滤器包含：

- 位数组 `bits[0..m-1]`
- `k` 个哈希函数

查询元素 `x` 时：

1.  计算所有哈希值：`h1(x), h2(x), ..., hk(x)`
2.  检查这些位置上的位

    *   如果任何 `bit[h_i(x)] == 0`，返回"否"（肯定不存在）
    *   如果所有位都是 `1`，返回"可能"（可能出现假阳性）

关键规则：位只能被置为开（1），永远不会被关闭（0），因此"否"的回答是可靠的。

#### 示例

让我们使用一个大小为 `m = 10`、`k = 3` 的过滤器：

插入 "cat" 和 "dog" 后的位数组：

```
索引:  0 1 2 3 4 5 6 7 8 9
位:    0 1 1 0 0 1 0 1 0 1
```

现在查询 "cat"：

```
h1(cat)=2, h2(cat)=5, h3(cat)=7
bits[2]=1, bits[5]=1, bits[7]=1 → 可能存在 ✅
```

查询 "fox"：

```
h1(fox)=3, h2(fox)=5, h3(fox)=8
bits[3]=0 → 肯定不存在 ❌
```

#### 可视化

查询(x)：

```
for i in 1..k:
  if bit[ h_i(x) ] == 0:
     return "NO"
return "MAYBE"
```

布隆过滤器为了安全起见会说"可能"，但从不会用"否"来撒谎。

#### 微型代码（简易版本）

C 语言实现（概念性）

```c
#include <stdio.h>

#define M 10
#define K 3
int bitset[M];

int hash1(int x) { return x % M; }
int hash2(int x) { return (x * 3 + 1) % M; }
int hash3(int x) { return (x * 7 + 5) % M; }

int query(int x) {
    int h[K] = {hash1(x), hash2(x), hash3(x)};
    for (int i = 0; i < K; i++)
        if (bitset[h[i]] == 0)
            return 0; // 肯定不存在
    return 1; // 可能存在
}

int main() {
    bitset[2] = bitset[5] = bitset[7] = 1; // 插入 "cat"
    printf("查询 42: %s\n", query(42) ? "可能" : "否");
    printf("查询 23: %s\n", query(23) ? "可能" : "否");
}
```

Python 实现

```python
m, k = 10, 3
bitset = [0] * m

def hash_functions(x):
    return [(hash(x) + i * i) % m for i in range(k)]

def insert(x):
    for h in hash_functions(x):
        bitset[h] = 1

def query(x):
    for h in hash_functions(x):
        if bitset[h] == 0:
            return "NO"
    return "MAYBE"

insert("cat")
insert("dog")
print("查询 cat:", query("cat"))
print("查询 dog:", query("dog"))
print("查询 fox:", query("fox"))
```

#### 为什么它很重要

- 常数时间的成员资格测试
- 没有假阴性，只有罕见的假阳性
- 非常适合在繁重的查找操作之前进行预过滤
- 常见于分布式系统和缓存层

#### 一个温和的证明（为什么它有效）

每个插入的元素将位数组中的 $k$ 位设置为 1。
对于一个查询，如果所有 $k$ 位都是 1，则该元素*可能*存在（或者哈希冲突导致了这些位被设置）。
如果任何一位是 0，则该元素肯定从未被插入过。

假阳性概率为：

$$
p = \left(1 - e^{-kn/m}\right)^k
$$

其中：

- $m$：位数组的大小
- $n$：已插入元素的数量
- $k$：哈希函数的数量

选择 $m$ 和 $k$ 以最小化目标假阳性率下的 $p$。

#### 自己动手试试

1.  创建一个 `m=20`、`k=3` 的过滤器
2.  插入 {"apple", "banana"}
3.  查询 {"apple", "grape"}
4.  观察 "grape" 如何可能返回 "maybe"

#### 测试用例

| 已插入元素 | 查询     | 结果   |
| ---------- | -------- | ------ |
| {cat, dog} | cat      | maybe  |
| {cat, dog} | dog      | maybe  |
| {cat, dog} | fox      | no     |
| {}         | anything | no     |

#### 复杂度

| 操作     | 时间复杂度 | 空间复杂度 | 假阴性 | 假阳性     |
| -------- | ---------- | ---------- | ------ | ---------- |
| 查询     | O(k)       | O(m)       | 无     | 可能出现   |

布隆过滤器查询，快速、内存占用少，并且在说"否"时是可信的。
### 263 计数布隆过滤器

计数布隆过滤器（CBF）通过允许删除操作扩展了经典的布隆过滤器。
它不使用简单的位数组，而是使用一个整数计数器数组，因此每个位变成了一个小的计数器，用于追踪有多少个元素映射到了该位置。

插入时，递增计数器；删除时，递减计数器。如果所有必需的计数器都大于零，则该元素可能存在。

#### 我们要解决什么问题？

常规的布隆过滤器是*只写*的：你可以插入，但不能移除。一旦某个位被设置为 `1`，它将永远保持为 `1`。

但如果遇到以下情况呢？

- 你正在追踪*活跃会话*？
- 你需要移除*过期的缓存键*？
- 你想要维护一个数据的*滑动窗口*？

那么你就需要一个计数布隆过滤器，它支持安全的删除操作。

#### 工作原理（通俗解释）

我们将位数组替换为一个大小为 $m$ 的计数器数组。

对于每个元素 $x$：

插入(x)  
- 对于每个哈希值 $h_i(x)$：递增 `count[h_i(x)]++`

查询(x)  
- 检查是否所有 `count[h_i(x)] > 0` → "可能存在"  
- 如果任何一个为 `0`，→ "不存在"

删除(x)  
- 对于每个哈希值 $h_i(x)$：递减 `count[h_i(x)]--`  
- 确保计数器永远不会变为负数

#### 示例

设 $m = 10$, $k = 3$

初始状态  
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

插入 "cat"  

$h_1(\text{cat}) = 2$  
$h_2(\text{cat}) = 5$  
$h_3(\text{cat}) = 7$  

→ 递增 $count[2]$, $count[5]$, $count[7]$  

插入后的数组  
[0, 0, 1, 0, 0, 1, 0, 1, 0, 0]

插入 "dog"  

$h_1(\text{dog}) = 1$  
$h_2(\text{dog}) = 5$  
$h_3(\text{dog}) = 9$  

→ 递增 $count[1]$, $count[5]$, $count[9]$  

插入后的数组  
[0, 1, 1, 0, 0, 2, 0, 1, 0, 1]

删除 "cat"  

$h_1(\text{cat}) = 2$  
$h_2(\text{cat}) = 5$  
$h_3(\text{cat}) = 7$  

→ 递减 $count[2]$, $count[5]$, $count[7]$  

删除后的数组  
[0, 1, 0, 0, 0, 1, 0, 0, 0, 1]

删除后，"cat" 被移除，而 "dog" 仍然存在。

#### 可视化

计数器随着每次插入/删除而变化：

| 操作        | 索引 1 | 2 | 5 | 7 | 9 |
| ----------- | ------- | - | - | - | - |
| 插入(cat)   | –       | 1 | 1 | 1 | – |
| 插入(dog)   | 1       | 1 | 2 | 1 | 1 |
| 删除(cat)   | 1       | 0 | 1 | 0 | 1 |

#### 微型代码（简易版本）

Python 实现

```python
m, k = 10, 3
counts = [0] * m

def hash_functions(x):
    return [(hash(x) + i * i) % m for i in range(k)]

def insert(x):
    for h in hash_functions(x):
        counts[h] += 1

def query(x):
    return all(counts[h] > 0 for h in hash_functions(x))

def delete(x):
    for h in hash_functions(x):
        if counts[h] > 0:
            counts[h] -= 1

# 示例
insert("cat")
insert("dog")
print("插入后:", counts)
delete("cat")
print("删除(cat)后:", counts)
print("查询 cat:", "可能存在" if query("cat") else "不存在")
print("查询 dog:", "可能存在" if query("dog") else "不存在")
```

输出

```
插入后: [0,1,1,0,0,2,0,1,0,1]
删除(cat)后: [0,1,0,0,0,1,0,0,0,1]
查询 cat: 不存在
查询 dog: 可能存在
```

#### 为什么它很重要

- 支持安全删除，无需完全重置
- 对缓存失效、会话跟踪、流式窗口很有用
- 是动态哈希集的内存高效替代方案

#### 一个温和的证明（为什么它有效）

每个计数器近似地表示有多少项映射到了它。
删除操作只递减计数器，如果多个元素共享一个哈希值，计数器会保持 ≥1。
因此：

- 除非过度递减（错误），否则不会出现假阴性
- 假阳性仍然可能发生，就像在经典布隆过滤器中一样

假阳性的概率仍然是：
$$
p = \left(1 - e^{-kn/m}\right)^k
$$

#### 亲自尝试

1.  创建一个过滤器，参数为 `m=20, k=3`
2.  插入 5 个单词
3.  删除其中的 2 个
4.  查询所有 5 个，被删除的应显示不存在，其他的显示可能存在

#### 测试用例

| 操作        | 数组快照                | 查询结果                |
| ----------- | ----------------------- | ----------------------- |
| 插入(cat)   | [0,0,1,0,0,1,0,1,0,0]   | –                       |
| 插入(dog)   | [0,1,1,0,0,2,0,1,0,1]   | –                       |
| 删除(cat)   | [0,1,0,0,0,1,0,0,0,1]   | cat → 不存在, dog → 可能存在 |

#### 复杂度

| 操作   | 时间复杂度 | 空间复杂度 |
| ------ | ---------- | ---------- |
| 插入   | O(k)       | O(m)       |
| 查询   | O(k)       | O(m)       |
| 删除   | O(k)       | O(m)       |

计数布隆过滤器，灵活且可逆，保持内存精简，删除操作清晰。
### 264 布谷鸟过滤器

布谷鸟过滤器是一种空间效率高的布隆过滤器替代方案，它支持插入和删除操作，同时保持较低的错误肯定率。它不使用位数组，而是在哈希桶中存储键的短小*指纹*，并使用布谷鸟哈希来解决冲突。

它就像一个更聪明、更整洁的室友，当空间变得拥挤时，总是通过移动别人来腾出空间。

#### 我们要解决什么问题？

布隆过滤器速度快且紧凑，但它们无法高效地删除元素。计数布隆过滤器解决了这个问题，但它们更消耗内存。

我们想要：

-   快速的成员查询 (`O(1)`)
-   支持插入 + 删除
-   高负载因子 (~95%)
-   紧凑的内存占用

布谷鸟过滤器通过使用布谷鸟哈希 + 短小指纹解决了所有这三个问题。

#### 工作原理（通俗解释）

每个元素由一个短指纹（例如，8 位）表示。
每个指纹可以放在两个可能的桶中，由两个哈希函数决定。

如果一个桶已满，我们就*驱逐*一个现有的指纹，并将其重新定位到它的备用桶（布谷鸟风格）。

操作：

1.  插入(x)

    *   计算指纹 `f = hash_fingerprint(x)`
    *   计算 `i1 = hash(x) % m`
    *   计算 `i2 = i1 ⊕ hash(f)` (备用索引)
    *   尝试将 `f` 放入 `i1` 或 `i2`
    *   如果两者都满，则驱逐一个指纹并重新定位

2.  查询(x)

    *   检查 `f` 是否存在于桶 `i1` 或 `i2` 中

3.  删除(x)

    *   如果在任一桶中找到 `f`，则将其移除

#### 示例（逐步说明）

假设：

-   桶数：`m = 4`
-   桶大小：`b = 2`
-   指纹大小：4 位

初始（空）：

| Bucket 0 | Bucket 1 | Bucket 2 | Bucket 3 |
| -------- | -------- | -------- | -------- |
|          |          |          |          |

插入 A

```
f(A) = 1010
i1 = 1, i2 = 1 ⊕ hash(1010) = 3
→ 放入桶 1
```

| B0 | B1   | B2 | B3 |
| -- | ---- | -- | -- |
|    | 1010 |    |    |

插入 B

```
f(B) = 0111
i1 = 3, i2 = 3 ⊕ hash(0111) = 0
→ 放入桶 3
```

| B0 | B1   | B2 | B3   |
| -- | ---- | -- | ---- |
|    | 1010 |    | 0111 |

插入 C

```
f(C) = 0100
i1 = 1, i2 = 1 ⊕ hash(0100) = 2
→ 桶 1 满了吗？如果需要则移动（布谷鸟）
→ 放入桶 2
```

| B0 | B1   | B2   | B3   |
| -- | ---- | ---- | ---- |
|    | 1010 | 0100 | 0111 |

查询 C → 在桶 2 中找到 ✅
删除 A → 从桶 1 中移除

#### 微型代码（简易版）

Python 示例

```python
import random

class CuckooFilter:
    def __init__(self, size=4, bucket_size=2, fingerprint_bits=4):
        self.size = size
        self.bucket_size = bucket_size
        self.buckets = [[] for _ in range(size)]
        self.mask = (1 << fingerprint_bits) - 1

    def _fingerprint(self, item):
        return hash(item) & self.mask

    def _alt_index(self, i, fp):
        return (i ^ hash(fp)) % self.size

    def insert(self, item, max_kicks=4):
        fp = self._fingerprint(item)
        i1 = hash(item) % self.size
        i2 = self._alt_index(i1, fp)
        for i in (i1, i2):
            if len(self.buckets[i]) < self.bucket_size:
                self.buckets[i].append(fp)
                return True
        # 布谷鸟驱逐
        i = random.choice([i1, i2])
        for _ in range(max_kicks):
            fp, self.buckets[i][0] = self.buckets[i][0], fp
            i = self._alt_index(i, fp)
            if len(self.buckets[i]) < self.bucket_size:
                self.buckets[i].append(fp)
                return True
        return False  # 插入失败

    def contains(self, item):
        fp = self._fingerprint(item)
        i1 = hash(item) % self.size
        i2 = self._alt_index(i1, fp)
        return fp in self.buckets[i1] or fp in self.buckets[i2]

    def delete(self, item):
        fp = self._fingerprint(item)
        i1 = hash(item) % self.size
        i2 = self._alt_index(i1, fp)
        for i in (i1, i2):
            if fp in self.buckets[i]:
                self.buckets[i].remove(fp)
                return True
        return False
```

#### 为什么它重要

-   高效支持删除
-   失败前具有高负载因子 (~95%)
-   在相同错误率下比计数布隆过滤器更小
-   适用于缓存、成员检查、去重

#### 一个温和的证明（为什么它有效）

每个元素有两个潜在位置 → 高灵活性
布谷鸟驱逐确保表保持紧凑
短指纹在保持低冲突的同时节省内存

错误肯定率：
$$
p \approx \frac{2b}{2^f}
$$
其中 `b` 是桶大小，`f` 是指纹位数

#### 自己试试

1.  构建一个过滤器，包含 8 个桶，每个桶有 2 个槽位
2.  插入 5 个单词
3.  删除 1 个单词
4.  查询所有 5 个单词，被删除的那个应该返回 "否"

#### 测试用例

| 操作      | Bucket 0 | Bucket 1 | Bucket 2 | Bucket 3 | 结果      |
| --------- | -------- | -------- | -------- | -------- | --------- |
| Insert(A) | –        | 1010     | –        | –        | OK        |
| Insert(B) | –        | 1010     | –        | 0111     | OK        |
| Insert(C) | –        | 1010     | 0100     | 0111     | OK        |
| Query(C)  | –        | 1010     | 0100     | 0111     | Maybe     |
| Delete(A) | –        | –        | 0100     | 0111     | Deleted ✅ |

#### 复杂度

| 操作   | 时间           | 空间          |
| ------ | -------------- | ------------- |
| Insert | O(1) 摊销      | O(m × b × f)  |
| Query  | O(1)           | O(m × b × f)  |
| Delete | O(1)           | O(m × b × f)  |

布谷鸟过滤器，一个基于巧妙驱逐机制构建的灵活、紧凑且可删除的成员数据结构。
### 265 计数最小草图

计数最小草图（CMS）是一种紧凑的数据结构，用于估计数据流中元素的频率。
它不存储每个项目，而是维护一个由多个哈希函数更新的小型二维计数器数组。
它永远不会低估计数，但由于哈希冲突可能会轻微高估。

可以把它想象成一个内存高效的雷达，它看不到每辆车，但大致知道每条车道上有多少辆车。

#### 我们要解决什么问题？

在流式或海量数据集中，我们无法精确存储每个键值对的计数。
我们希望用有限的内存来跟踪近似频率，支持：

- 流式更新：在项目到达时进行计数
- 近似查询：估计项目频率
- 内存效率：亚线性空间

应用于网络监控、自然语言处理（词频统计）、频繁项检测和在线分析。

#### 工作原理（通俗解释）

CMS 使用一个具有 `d` 行和 `w` 列的二维数组。
每一行使用一个不同的哈希函数。
每个项目更新每一行中的一个计数器，位置由其哈希索引决定。

插入(x)：
对于每一行 `i`：

```
index = hash_i(x) % w
count[i][index] += 1
```

查询(x)：
对于每一行 `i`：

```
estimate = min(count[i][hash_i(x) % w])
```

我们取所有行中的*最小值*，因此得名“计数最小”。

#### 示例（逐步说明）

设 `d = 3` 个哈希函数，`w = 10` 宽度。

初始化表格：

```
行1: [0 0 0 0 0 0 0 0 0 0]
行2: [0 0 0 0 0 0 0 0 0 0]
行3: [0 0 0 0 0 0 0 0 0 0]
```

插入 "apple"

```
h1(apple)=2, h2(apple)=5, h3(apple)=9
→ 增加位置 (1,2), (2,5), (3,9)
```

插入 "banana"

```
h1(banana)=2, h2(banana)=3, h3(banana)=1
→ 增加 (1,2), (2,3), (3,1)
```

现在：

```
行1: [0 0 2 0 0 0 0 0 0 0]
行2: [0 0 0 1 0 1 0 0 0 0]
行3: [0 1 0 0 0 0 0 0 0 1]
```

查询 "apple"：

```
min(count[1][2], count[2][5], count[3][9]) = min(2,1,1) = 1
```

估计频率 ≈ 1（如果冲突重叠，可能会略高）。

#### 表格可视化

| 项目   | h₁(x) | h₂(x) | h₃(x) | 估计计数 |
| ------ | ----- | ----- | ----- | -------- |
| apple  | 2     | 5     | 9     | 1        |
| banana | 2     | 3     | 1     | 1        |

#### 微型代码（简易版本）

Python 示例

```python
import mmh3

class CountMinSketch:
    def __init__(self, width=10, depth=3):
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]
        self.seeds = [i * 17 for i in range(depth)]  # 不同的哈希种子

    def _hash(self, item, seed):
        return mmh3.hash(str(item), seed) % self.width

    def add(self, item, count=1):
        for i, seed in enumerate(self.seeds):
            idx = self._hash(item, seed)
            self.table[i][idx] += count

    def query(self, item):
        estimates = []
        for i, seed in enumerate(self.seeds):
            idx = self._hash(item, seed)
            estimates.append(self.table[i][idx])
        return min(estimates)

# 使用示例
cms = CountMinSketch(width=10, depth=3)
cms.add("apple")
cms.add("banana")
cms.add("apple")
print("apple:", cms.query("apple"))
print("banana:", cms.query("banana"))
```

输出

```
apple: 2
banana: 1
```

#### 为什么它很重要

- 紧凑：O(w × d) 内存
- 快速：O(1) 更新和查询
- 可扩展：适用于无界数据流
- 确定性上界：永远不会低估

应用于：

- 词频估计
- 网络流量计数
- 点击流分析
- 近似直方图

#### 一个温和的证明（为什么它有效）

每个项目被哈希到 `d` 个位置。
冲突可能导致高估，但永远不会低估。
通过取最小值，我们得到了最佳的上界估计。

误差界：
$$
\text{error} \le \epsilon N, \quad \text{with probability } 1 - \delta
$$
选择：
$$
w = \lceil e / \epsilon \rceil,\quad d = \lceil \ln(1/\delta) \rceil
$$

#### 亲自尝试

1.  创建一个 CMS，参数为 `(w=20, d=4)`
2.  流式处理 1000 个随机项目
3.  比较估计计数与实际计数
4.  观察高估模式

#### 测试用例

| 操作           | 动作 | 查询结果 |
| -------------- | ---- | -------- |
| Insert(apple)  | +1   | –        |
| Insert(apple)  | +1   | –        |
| Insert(banana) | +1   | –        |
| Query(apple)   | –    | 2        |
| Query(banana)  | –    | 1        |

#### 复杂度

| 操作   | 时间 | 空间     |
| ------ | ---- | -------- |
| 插入   | O(d) | O(w × d) |
| 查询   | O(d) | O(w × d) |

计数最小草图，轻量级，足够准确，专为永不停止的流而构建。
### 266 HyperLogLog

HyperLogLog（HLL）是一种用于基数估计的概率数据结构，即使用极少的内存来估计海量数据集或数据流中不同元素的数量。

它不记录你*看到过哪些*项目，只记录*大概有多少个不同的项目*。可以把它想象成一个内存高效的客流计数器：它不认识人脸，但它知道体育场有多满。

#### 我们要解决什么问题？

在处理大规模数据流（网络分析、日志、独立访客等）时，精确计算不同元素（使用集合或哈希表）会消耗过多内存。

我们需要一个解决方案，能够：

- 近似跟踪不同元素的数量
- 使用恒定内存
- 支持可合并性（轻松合并两个草图）

HyperLogLog 以仅几千字节的内存，实现了每次更新 O(1) 的时间复杂度和约 1.04/√m 的误差率。

#### 工作原理（通俗解释）

每个输入元素被哈希成一个大的二进制数。
HLL 利用哈希值中最高位 1 的位置来估计元素有多么罕见（从而推断出有多少个不同元素）。

它维护一个包含 `m` 个*寄存器*（桶）的数组。每个桶存储其哈希值范围内迄今观察到的最大前导零计数。

步骤：

1.  将元素 `x` 哈希为 64 位：`h = hash(x)`
2.  桶索引：使用 `h` 的前 `p` 位来选择 `m = 2^p` 个桶中的一个
3.  秩：计算剩余位中的前导零数量 + 1
4.  更新：在该桶中存储 `max(现有值, 秩)`

最终计数 = `2^rank` 值的调和平均值，再乘以一个经过偏差校正的常数进行缩放。

#### 示例（逐步说明）

设 `p = 2` → `m = 4` 个桶。
初始化寄存器：`[0,0,0,0]`

插入 "apple"：

```
hash("apple") = 110010100…
桶 = 前 2 位 = 11 (3)
秩 = 前缀之后第一个 1 的位置 = 2
→ bucket[3] = max(0, 2) = 2
```

插入 "banana"：

```
hash("banana") = 01000100…
桶 = 01 (1)
秩 = 3
→ bucket[1] = 3
```

插入 "pear"：

```
hash("pear") = 10010000…
桶 = 10 (2)
秩 = 4
→ bucket[2] = 4
```

寄存器：`[0, 3, 4, 2]`

估计值：
$$
E = \alpha_m \cdot m^2 / \sum 2^{-M[i]}
$$
其中 `α_m` 是偏差校正常数（对于小的 m 约为 0.673）。

#### 可视化

| 桶   | 看到的数值 | 前导零 | 存储的秩 |
| ---- | ---------- | ------ | -------- |
| 0    | 无         | –      | 0        |
| 1    | banana     | 2      | 3        |
| 2    | pear       | 3      | 4        |
| 3    | apple      | 1      | 2        |

#### 微型代码（简易版）

Python 实现

```python
import mmh3
import math

class HyperLogLog:
    def __init__(self, p=4):
        self.p = p
        self.m = 1 << p
        self.registers = [0] * self.m
        self.alpha = 0.673 if self.m == 16 else 0.709 if self.m == 32 else 0.7213 / (1 + 1.079 / self.m)

    def _hash(self, x):
        return mmh3.hash(str(x), 42) & 0xffffffff

    def add(self, x):
        h = self._hash(x)
        idx = h >> (32 - self.p)
        w = (h << self.p) & 0xffffffff
        rank = self._rank(w, 32 - self.p)
        self.registers[idx] = max(self.registers[idx], rank)

    def _rank(self, w, bits):
        r = 1
        while w & (1 << (bits - 1)) == 0 and r <= bits:
            r += 1
            w <<= 1
        return r

    def count(self):
        Z = sum([2.0  -v for v in self.registers])
        E = self.alpha * self.m * self.m / Z
        return round(E)

# 示例
hll = HyperLogLog(p=4)
for x in ["apple", "banana", "pear", "apple"]:
    hll.add(x)
print("估计的不同元素数量：", hll.count())
```

输出

```
估计的不同元素数量： 3
```

#### 为什么它很重要

-   内存占用极小（数十亿元素仅需几千字节）
-   可合并：HLL(A∪B) = 逐桶取 max(HLL(A), HLL(B))
-   应用于：Redis、Google BigQuery、Apache DataSketches、分析系统

#### 温和的证明（为什么有效）

最高位 1 的位置遵循几何分布，罕见的长零串意味着更多唯一元素。
通过保持每个桶观察到的*最大*秩，HLL 有效地捕捉了全局的稀有性。
跨桶组合得到一个调和平均值，平衡了少计和多计。

误差 ≈ 1.04 / √m，因此将 `m` 加倍会使误差减半。

#### 自己动手试试

1.  创建 `p = 10` (`m = 1024`) 的 HLL
2.  添加 100 万个随机数
3.  比较 HLL 估计值与真实计数
4.  测试合并两个草图

#### 测试用例

| 项目数 | 真实计数 | 估计值    | 误差   |
| ------ | -------- | --------- | ------ |
| 10     | 10       | 10        | 0%     |
| 1000   | 1000     | 995       | ~0.5%  |
| 1e6    | 1,000,000 | 1,010,000 | ~1%    |

#### 复杂度

| 操作   | 时间   | 空间   |
| ------ | ------ | ------ |
| 添加   | O(1)   | O(m)   |
| 计数   | O(m)   | O(m)   |
| 合并   | O(m)   | O(m)   |

HyperLogLog，一次一个前导零，数着数不清的数。
### 267 Flajolet–Martin 算法

Flajolet–Martin（FM）算法是最早、最简单的概率计数方法之一，它使用极小的内存来估计数据流中不同元素的数量。

它是 HyperLogLog 的概念先驱，展示了一个精妙的想法：*哈希值中第一个 1 比特的位置能告诉我们关于稀有度的信息*。

#### 我们要解决什么问题？

当元素以数据流的形式到达，数据量太大而无法精确存储（例如网络请求、IP 地址、单词）时，我们希望在不存储所有元素的情况下估计不同元素的数量。

一种朴素的方法（哈希集合）需要 O(n) 的内存。Flajolet–Martin 算法则实现了 O(1) 的内存和 O(1) 的更新开销。

我们需要：

- 一个流式算法
- 具有常数空间复杂度
- 用于不同元素计数估计

#### 工作原理（通俗解释）

每个元素被哈希成一个大的二进制数（均匀随机）。然后我们找到最低有效位 1 的位置——即末尾零的个数。一个末尾零很多的值是罕见的，这暗示了底层总体数量较大。

我们跟踪观察到的最大末尾零个数，记为 `R`。不同元素数量的估计值为：

$$
\hat{N} = \phi \times 2^{R}
$$

其中 $\phi \approx 0.77351$ 是一个校正常数。

步骤：

1. 初始化 $R = 0$
2. 对于每个元素 $x$：
   - $h = \text{hash}(x)$
   - $r = \text{count\_trailing\_zeros}(h)$
   - $R = \max(R, r)$
3. 估计不同元素数量为 $\hat{N} \approx \phi \times 2^{R}$

#### 示例（逐步演示）

数据流：`[apple, banana, apple, cherry, date]`

| 元素   | 哈希值（二进制） | 末尾零个数 | R |
| ------ | ---------------- | ---------- | - |
| apple  | 10110            | 1          | 1 |
| banana | 10000            | 4          | 4 |
| apple  | 10110            | 1          | 4 |
| cherry | 11000            | 3          | 4 |
| date   | 01100            | 2          | 4 |

最终值：$R = 4$

估计：
$$
\hat{N} = 0.77351 \times 2^{4} = 0.77351 \times 16 \approx 12.38
$$

所以估计的不同元素数量约为 12（由于样本量小，存在高估）。

在实践中，会使用多个独立的哈希函数或寄存器，并对它们的结果取平均以减少方差并提高准确性。

#### 可视化

| 哈希值 | 二进制形式 | 末尾零个数 | 含义                                       |
| ------ | ---------- | ---------- | ------------------------------------------ |
| 10000  | 16         | 4          | 非常罕见的模式 → 暗示总体数量很大          |
| 11000  | 24         | 3          | 比较罕见                                   |
| 10110  | 22         | 1          | 常见                                       |

*最长的*零串给出了稀有度的尺度。

#### 微型代码（简易版本）

Python 示例

```python
import mmh3
import math

def trailing_zeros(x):
    if x == 0:
        return 32
    tz = 0
    while (x & 1) == 0:
        tz += 1
        x >>= 1
    return tz

def flajolet_martin(stream, seed=42):
    R = 0
    for x in stream:
        h = mmh3.hash(str(x), seed) & 0xffffffff
        r = trailing_zeros(h)
        R = max(R, r)
    phi = 0.77351
    return int(phi * (2  R))

# 示例
stream = ["apple", "banana", "apple", "cherry", "date"]
print("Estimated distinct count:", flajolet_martin(stream))
```

输出

```
Estimated distinct count: 12
```

#### 为什么它重要？

- 是现代流式算法的基础
- 启发了 LogLog 和 HyperLogLog 算法
- 内存占用极轻：只需几个整数
- 在近似分析、网络遥测、数据仓库中很有用

#### 一个温和的证明（为什么它有效）

每个哈希输出都是均匀随机的。观察到具有 $r$ 个末尾零的值的概率是：

$$
P(r) = \frac{1}{2^{r+1}}
$$

如果这样的值出现，则暗示流的大小大约为 $2^{r}$。因此，估计值 $2^{R}$ 自然地随着不同元素的数量而缩放。

通过使用多个独立的估计器并对它们的结果取平均，可以将方差从大约 50% 降低到 10% 左右。

#### 亲自尝试

1.  生成 100 个随机数，输入给 FM 算法
2.  比较估计值与真实计数
3.  重复运行 10 次，计算平均误差
4.  尝试组合多个估计器（均值的中位数）

#### 测试用例

| 流大小 | 真实不同元素数 | R  | 估计值 | 误差 |
| ------ | -------------- | -- | ------ | ---- |
| 10     | 10             | 3  | 6      | -40% |
| 100    | 100            | 7  | 99     | -1%  |
| 1000   | 1000           | 10 | 791    | -21% |

使用多个寄存器可以显著提高准确性。

#### 复杂度

| 操作   | 时间 | 空间 |
| ------ | ---- | ---- |
| 插入   | O(1) | O(1) |
| 查询   | O(1) | O(1) |

Flajolet–Martin 算法，概率计数的最初火花，将随机性转化为估计的魔法。
### 268 MinHash

MinHash 是一种用于估计集合相似度（特别是 Jaccard 相似度）的概率算法，无需直接比较所有元素。
它不存储每个元素，而是构建一个由紧凑哈希值组成的签名。签名重叠越多，集合就越相似。

MinHash 是大规模相似度估计的基础，它快速、内存高效且数学上优雅。

#### 我们要解决什么问题？

要计算两个集合 $A$ 和 $B$ 之间的精确 Jaccard 相似度：

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

这需要比较所有元素，对于大型数据集来说是不可行的。

我们需要的方法应提供：

- 使用小内存的紧凑概要
- 快速的近似比较
- 可通过哈希函数数量调整的准确度

MinHash 通过应用多个随机哈希函数，并将每个函数的最小哈希值记录为集合的签名，来满足这些要求。

#### 工作原理（通俗解释）

对于每个集合，我们应用几个独立的哈希函数。
对于每个哈希函数，我们记录所有元素中的最小哈希值。

共享许多元素的两个集合，在这些哈希函数中往往共享相同的最小值。
因此，匹配的最小值的比例可以估计它们的 Jaccard 相似度。

算法：

1. 选择 $k$ 个哈希函数 $h_1, h_2, \ldots, h_k$。
2. 对于每个集合 $S$，计算其签名：

$$
\text{sig}(S)[i] = \min_{x \in S} h_i(x)
$$

3. 通过下式估计 Jaccard 相似度：

$$
\widehat{J}(A, B) = \frac{1}{k} \sum_{i=1}^{k} \mathbf{1}\{\text{sig}(A)[i] = \text{sig}(B)[i]\}
$$

$\text{sig}(A)$ 和 $\text{sig}(B)$ 之间的每个匹配位置对应一个一致的哈希函数，匹配的比例近似于 $J(A,B)$。

#### 示例（逐步说明）

令：

$$
A = {\text{apple}, \text{banana}, \text{cherry}}, \quad
B = {\text{banana}, \text{cherry}, \text{date}}
$$

假设我们有 3 个哈希函数：

| 条目   | $h_1$ | $h_2$ | $h_3$ |
| ------ | ----- | ----- | ----- |
| apple  | 5     | 1     | 7     |
| banana | 2     | 4     | 3     |
| cherry | 3     | 2     | 1     |
| date   | 4     | 3     | 2     |

签名(A)：
$$
\text{sig}(A) = [\min(5,2,3), \min(1,4,2), \min(7,3,1)] = [2,1,1]
$$

签名(B)：
$$
\text{sig}(B) = [\min(2,3,4), \min(4,2,3), \min(3,1,2)] = [2,2,1]
$$

逐元素比较：
匹配出现在位置 1 和 3 → $\tfrac{2}{3} \approx 0.67$

实际 Jaccard 相似度：

$$
J(A, B) = \frac{|\{\text{banana}, \text{cherry}\}|}{|\{\text{apple}, \text{banana}, \text{cherry}, \text{date}\}|}
= \frac{2}{4} = 0.5
$$

MinHash 估计值 $0.67$ 与真实值 $0.5$ 相当接近，
这表明即使哈希函数数量很少，也能得到良好的近似。

#### 可视化

| 哈希函数 | $\text{sig}(A)$ | $\text{sig}(B)$ | 匹配 |
| -------------- | --------------- | --------------- | ------ |
| $h_1$ | 2 | 2 | ✓ |
| $h_2$ | 1 | 2 | ✗ |
| $h_3$ | 1 | 1 | ✓ |
| 相似度 | – | – | $(2/3 = 0.67)$ |

#### 微型代码（简易版本）

```python
import mmh3
import math

def minhash_signature(elements, num_hashes=5, seed=42):
    """计算集合的 MinHash 签名"""
    sig = [math.inf] * num_hashes
    for x in elements:
        for i in range(num_hashes):
            h = mmh3.hash(str(x), seed + i)
            if h < sig[i]:
                sig[i] = h
    return sig

def jaccard_minhash(sigA, sigB):
    """通过 MinHash 签名估计 Jaccard 相似度"""
    matches = sum(1 for a, b in zip(sigA, sigB) if a == b)
    return matches / len(sigA)

# 示例
A = {"apple", "banana", "cherry"}
B = {"banana", "cherry", "date"}

sigA = minhash_signature(A, 10)
sigB = minhash_signature(B, 10)
print("近似相似度:", jaccard_minhash(sigA, sigB))
```

输出

```
近似相似度: 0.6
```

#### 为什么它很重要

- 可扩展的相似度：支持快速比较非常大的集合
- 紧凑的表示：每个集合仅存储 $k$ 个整数
- 可组合：支持使用分量最小值进行集合并集操作
- 常见应用：
  - 文档去重
  - 网络爬虫和搜索索引
  - 推荐系统
  - 大规模聚类

#### 一个温和的证明（为什么它有效）

对于一个随机排列 $h$：

$$
P[\min(h(A)) = \min(h(B))] = J(A, B)
$$

每个哈希函数的行为都像一个成功概率为 $J(A, B)$ 的伯努利试验。
MinHash 估计量是无偏的：

$$
E[\widehat{J}] = J(A, B)
$$

方差随着 $\tfrac{1}{k}$ 减小，
因此增加哈希函数的数量可以提高准确度。

#### 自己动手试试

1. 选择两个有部分重叠的集合。
2. 使用 $k = 20$ 个哈希函数生成 MinHash 签名。
3. 计算估计的和真实的 Jaccard 相似度。
4. 增加 $k$ 并观察估计的相似度如何收敛到真实值——更大的 $k$ 会减少方差并提高准确度。

#### 测试用例

| 集合                 | 真实 $J(A,B)$ | $k$ | 估计值 | 误差 |
| -------------------- | ------------- | --- | ---------- | ----- |
| $A=\{1,2,3\}, B=\{2,3,4\}$ | 0.5 | 10 | 0.6  | +0.1  |
| $A=\{1,2\}, B=\{1,2,3,4\}$ | 0.5 | 20 | 0.45 | -0.05 |
| $A=B$                | 1.0 | 10 | 1.0 | 0.0  |

#### 复杂度

| 操作       | 时间             | 空间 |
| --------------- | ---------------- | ----- |
| 构建签名 | $O(n \times k)$  | $O(k)$ |
| 比较         | $O(k)$           | $O(k)$ |

MinHash 将集合相似度转化为紧凑的签名——
一个以统计的优雅捕捉大型集合本质的小型概要。
### 269 蓄水池抽样

蓄水池抽样是一种经典算法，用于从未知或极大容量的数据流中随机抽取 k 个元素，确保每个元素被选中的概率均等。

当你无法存储所有数据时，它是最佳工具，就像从一条无尽的河流中，一个接一个、无偏地捕捉几条鱼。

#### 我们要解决什么问题？

当数据以流的形式到达，其规模大到无法全部存入内存时，我们无法预先知道其总大小。  
然而，我们常常需要维护一个固定大小为 $k$ 的均匀随机样本。

一种简单的方法是存储所有项目然后抽样，  
但对于大规模或无限的数据，这种方法变得不可行。

蓄水池抽样提供了一种单次遍历的解决方案，并保证：

- 流中的每个项目被包含的概率均为 $\tfrac{k}{n}$
- 仅使用 $O(k)$ 的内存
- 单次遍历处理数据

#### 工作原理（通俗解释）

我们维护一个大小为 $k$ 的蓄水池（数组）。  
当每个新元素到达时，我们以概率方式决定是否用它替换现有项目中的一个。

步骤：

1. 用前 $k$ 个元素填充蓄水池。
2. 对于索引为 $i$ 的每个元素（从 $i = k + 1$ 开始）：

   - 生成一个随机整数 $j \in [1, i]$
   - 如果 $j \le k$，则用新元素替换 $\text{reservoir}[j]$

这确保了每个元素都有相等的概率 $\tfrac{k}{n}$ 留在蓄水池中。

#### 示例（逐步演示）

数据流：[A, B, C, D, E]  
目标：$k = 2$

1. 从前 2 个元素开始 → [A, B]
2. $i = 3$，项目 = C
   - 随机选取 $j \in [1, 3]$
   - 假设 $j = 2$ → 替换 B → [A, C]
3. $i = 4$，项目 = D
   - 随机选取 $j \in [1, 4]$
   - 假设 $j = 4$ → 不执行任何操作 → [A, C]
4. $i = 5$，项目 = E
   - 随机选取 $j \in [1, 5]$
   - 假设 $j = 1$ → 替换 A → [E, C]

最终样本：[E, C]  
每个项目 A–E 出现在最终蓄水池中的概率均等。
### 数学直觉

位置 $i$ 处的每个元素 $x_i$ 被选入最终样本的概率为

$$
P(x_i \text{ 在最终样本中}) = \frac{k}{i} \cdot \prod_{j=i+1}^{n} \left(1 - \frac{1}{j}\right) = \frac{k}{n}
$$

因此，每个项目被选中的可能性均等，确保了最终样本的完美均匀性。

#### 可视化

| 步骤 | 项目 | 随机数 $j$ | 动作       | 蓄水池 |
| ---- | ---- | ---------- | ---------- | ------ |
| 1    | A    | –          | 添加       | [A]    |
| 2    | B    | –          | 添加       | [A, B] |
| 3    | C    | 2          | 替换 B     | [A, C] |
| 4    | D    | 4          | 不替换     | [A, C] |
| 5    | E    | 1          | 替换 A     | [E, C] |

#### 精简代码（简易版本）

Python 实现

```python
import random

def reservoir_sample(stream, k):
    reservoir = []
    for i, item in enumerate(stream, 1):
        if i <= k:
            reservoir.append(item)
        else:
            j = random.randint(1, i)
            if j <= k:
                reservoir[j - 1] = item
    return reservoir

# 示例
stream = ["A", "B", "C", "D", "E"]
sample = reservoir_sample(stream, 2)
print("蓄水池样本:", sample)
```

输出（随机）：

```
蓄水池样本: ['E', 'C']
```

每次运行都会产生一个不同的均匀随机样本。

#### 为何重要

- 适用于流式数据
- 仅需 O(k) 内存
- 提供均匀无偏采样
- 应用于：

  * 大数据分析
  * 随机化算法
  * 在线学习
  * 网络监控

#### 温和的证明（为何有效）

- 前 $k$ 个元素：初始进入蓄水池的概率 $= 1$。
- 索引 $i$ 处的新元素：以概率 $\tfrac{k}{i}$ 替换现有项目之一。
- 较早的项目可能被替换，但每个项目存留的概率为

$$
P(\text{存留}) = \prod_{j=i+1}^{n} \left(1 - \frac{1}{j}\right)
$$

将这些项相乘，得到最终的包含概率 $\tfrac{k}{n}$。

通过归纳法保证了选择的均匀性。

#### 亲自尝试

1.  使用 $k = 3$ 处理 10 个数字的流。
2.  多次运行算法 —— 所有 3 元素子集出现的频率大致相等。
3.  增加 $k$ 并观察样本变得更加稳定，不同运行间的差异减小。

#### 测试用例

| 流           | k  | 样本大小 | 说明                 |
| ------------ | -- | -------- | -------------------- |
| [1,2,3,4,5]  | 2  | 2        | 均匀随机对           |
| [A,B,C,D]    | 1  | 1        | 每个项目 25% 概率    |
| Range(1000)  | 10 | 10       | 单次遍历即可完成     |

#### 复杂度

| 操作     | 时间   | 空间   |
| -------- | ------ | ------ |
| 插入     | $O(1)$ | $O(k)$ |
| 查询     | $O(1)$ | $O(k)$ |

蓄水池采样，优雅公平，极其简单，为无限流数据做好准备。
### 270 跳跃布隆过滤器

跳跃布隆过滤器是一种概率数据结构，它扩展了布隆过滤器以支持范围查询，能够判断给定区间内是否存在任何元素，而不仅仅是检查单个元素。

它结合了布隆过滤器和分层范围分割，在保持空间使用紧凑的同时，允许进行近似范围查找。

#### 我们要解决什么问题？

经典的布隆过滤器只回答点查询：

$$
\text{"元素 } x \text{ 在集合中吗？"}
$$

然而，许多实际应用需要范围查询，例如：

- 数据库："在 10 到 20 之间有任何键吗？"
- 时间序列："在这个时间间隔内发生过任何事件吗？"
- 网络："这个子网中有任何 IP 吗？"

我们需要一种空间高效、适合流式处理且概率性的数据结构，能够：

- 处理范围成员检查，
- 保持较低的错误肯定率，
- 随论域大小呈对数级扩展。

跳跃布隆过滤器通过在不同大小的对齐范围上分层布隆过滤器来解决这个问题。

#### 工作原理（通俗解释）

跳跃布隆过滤器维护多个布隆过滤器，每个过滤器对应一个层级，覆盖大小为 $2^0, 2^1, 2^2, \ldots$ 的区间（桶）。

每个元素被插入到所有代表包含它的范围的布隆过滤器中。  
当查询一个范围时，查询被分解为对应于这些层级的对齐子范围，并在其各自的过滤器中检查每个子范围。

算法：

1. 将论域按每个层级 $\ell$ 划分为大小为 $2^\ell$ 的区间。
2. 每个层级 $\ell$ 维护一个代表这些桶的布隆过滤器。
3. 插入键 $x$：标记所有包含 $x$ 的跨层级桶。
4. 查询范围 $[a,b]$：将其分解为一组不相交的对齐区间，并检查相应的布隆过滤器。

#### 示例（逐步说明）

假设我们在论域 ([0, 15]) 中存储键

$$
S = {3, 7, 14}
$$

我们构建范围大小为 (1, 2, 4, 8) 的层级过滤器：

| 层级 | 桶大小 | 桶（范围）                          |
| ---- | ------ | ----------------------------------- |
| 0    | 1      | [0], [1], [2], ..., [15]            |
| 1    | 2      | [0–1], [2–3], [4–5], ..., [14–15]   |
| 2    | 4      | [0–3], [4–7], [8–11], [12–15]       |
| 3    | 8      | [0–7], [8–15]                       |

插入键 3：

- 层级 0: [3]
- 层级 1: [2–3]
- 层级 2: [0–3]
- 层级 3: [0–7]

插入键 7：

- 层级 0: [7]
- 层级 1: [6–7]
- 层级 2: [4–7]
- 层级 3: [0–7]

插入键 14：

- 层级 0: [14]
- 层级 1: [14–15]
- 层级 2: [12–15]
- 层级 3: [8–15]

查询范围 [2, 6]：

1. 分解为对齐区间：([2–3], [4–5], [6])
2. 检查过滤器：

   * [2–3] → 命中
   * [4–5] → 未命中
   * [6] → 未命中

结果：可能非空，因为 [2–3] 包含 3。

#### 可视化

| 层级 | 桶     | 包含键 | 布隆条目 |
| ---- | ------ | ------ | -------- |
| 0    | [3]    | 是     | 1        |
| 1    | [2–3]  | 是     | 1        |
| 2    | [0–3]  | 是     | 1        |
| 3    | [0–7]  | 是     | 1        |

每个键在多个层级中都有表示，实现了多尺度范围覆盖。

#### 微型代码（简化 Python）

```python
import math, mmh3

class Bloom:
    def __init__(self, size=64, hash_count=3):
        self.size = size
        self.hash_count = hash_count
        self.bits = [0] * size

    def _hashes(self, key):
        return [mmh3.hash(str(key), i) % self.size for i in range(self.hash_count)]

    def add(self, key):
        for h in self._hashes(key):
            self.bits[h] = 1

    def query(self, key):
        return all(self.bits[h] for h in self._hashes(key))

class SkipBloom:
    def __init__(self, levels=4, size=64, hash_count=3):
        self.levels = [Bloom(size, hash_count) for _ in range(levels)]

    def add(self, key):
        level = 0
        while (1 << level) <= key:
            bucket = key // (1 << level)
            self.levels[level].add(bucket)
            level += 1

    def query_range(self, start, end):
        l = int(math.log2(end - start + 1))
        bucket = start // (1 << l)
        return self.levels[l].query(bucket)

# 示例
sb = SkipBloom(levels=4)
for x in [3, 7, 14]:
    sb.add(x)

print("查询 [2,6]:", sb.query_range(2,6))
```

输出：

```
查询 [2,6]: True
```

#### 为什么它重要

- 能以概率方式支持范围查询
- 紧凑且分层
- 没有错误否定（对于正确配置的过滤器）
- 广泛应用于：

  * 近似数据库索引
  * 网络前缀搜索
  * 时间序列事件检测

#### 一个温和的证明（为什么它有效）

每个插入的键参与 $O(\log U)$ 个布隆过滤器，每个层级一个。  
范围查询 $[a,b]$ 被分解为 $O(\log U)$ 个对齐的子范围。

一个具有 $m$ 位、$k$ 个哈希函数和 $n$ 个插入元素的布隆过滤器具有错误肯定概率：

$$
p = \left(1 - e^{-kn/m}\right)^k
$$

对于跳跃布隆过滤器，总错误肯定率的上限为：

$$
P_{fp} \le O(\log U) \cdot p
$$

每个层级保证没有错误否定，因为包含元素的每个范围都被标记了。  
因此，正确性得到保证，只有高估（错误肯定）可能发生。

空间复杂度  
每个层级有 $m$ 位，共有 $\log U$ 个层级：

$$
\text{总空间} = O(m \log U)
$$

时间复杂度  
每个查询检查 $O(\log U)$ 个桶，每个桶需要 $O(k)$ 的时间：

$$
T_{\text{查询}} = O(k \log U)
$$

#### 自己动手试试

1. 插入键 $\{3, 7, 14\}$
2. 查询范围 $[2,6]$, $[8,12]$, $[0,15]$
3. 将真实内容与结果进行比较
4. 调整参数 $m$, $k$ 或层级数量，观察错误肯定率如何变化

#### 测试用例

| 查询范围 | 结果  | 真实内容 |
| -------- | ----- | -------- |
| [2,6]    | True  | {3}      |
| [8,12]   | False | ∅        |
| [12,15]  | True  | {14}     |

#### 复杂度

| 操作   | 时间             | 空间            |
| ------ | ---------------- | --------------- |
| 插入   | $O(\log U)$      | $O(m \log U)$   |
| 查询   | $O(\log U)$      | $O(m \log U)$   |

跳跃布隆过滤器是标准布隆过滤器的范围感知扩展。  
通过将分层分解与标准哈希相结合，它能够在非常大的论域上实现快速、内存高效和近似的范围查询。

## 第 28 章 跳表和 B 树
### 271 跳表插入

跳表是平衡树的一种概率性替代方案。它维护多层有序链表，其中每一层都比它下面的一层跳过更多的元素。插入操作依靠随机化来实现期望的 O(log n) 搜索和更新时间，而无需像 AVL 树或红黑树那样进行严格的重平衡。

#### 我们要解决什么问题？

我们希望以有序方式存储元素并支持：

-   快速的搜索、插入和删除操作。
-   结构简单且易于实现。
-   无需复杂的旋转即可获得期望的对数性能。

平衡二叉搜索树（BST）可以达到 $O(\log n)$ 的时间复杂度，但需要复杂的旋转操作。
跳表通过*随机提升*来解决这个问题：每个插入的节点以递减的概率被提升到更高层，形成一个塔状结构。

#### 工作原理（通俗解释）

插入值 $x$ 到跳表的步骤：

1.  从顶层开始。
2.  当下一个节点的键值 < $x$ 时，向右移动。
3.  如果无法向右移动，则下降一层。
4.  重复上述步骤，直到到达第 0 层。
5.  在第 0 层的排序位置插入新节点。
6.  随机选择节点的高度 $h$（例如，每层抛一次公平硬币直到出现反面，因此 $P(h \ge t) = 2^{-t}$）。
7.  对于每一层 $1 \dots h$，通过重复使用下降过程中保存的更新指针进行“先右移再下降”的搜索，将新节点链接到该层。

注意
-   重复项的处理取决于策略。通常，如果已存在键值为 $x$ 的节点，则跳过插入。
-   典型高度为 $O(\log n)$，期望的搜索和插入时间为 $O(\log n)$，空间复杂度为 $O(n)$。

#### 逐步示例

让我们将 $x = 17$ 插入到当前包含 [ 5, 10, 15, 20, 25 ] 的跳表中。

| 层数 | 节点（插入前）         | 遍历路径                               |
| :--- | :--------------------- | :------------------------------------- |
| 3    | 5 → 15 → 25            | 从 5 移动到 15，然后下降               |
| 2    | 5 → 10 → 15 → 25       | 从 5 移动到 10 再到 15，然后下降       |
| 1    | 5 → 10 → 15 → 20 → 25  | 从 15 移动到 20，然后下降              |
| 0    | 5 → 10 → 15 → 20 → 25  | 在 15 之后插入                         |

假设 17 的随机层数为 2。
我们在第 0 层和第 1 层插入 17。

| 层数 | 节点（插入后）             |
| :--- | :------------------------- |
| 3    | 5 → 15 → 25                |
| 2    | 5 → 10 → 15 → 25           |
| 1    | 5 → 10 → 15 → 17 → 20 → 25 |
| 0    | 5 → 10 → 15 → 17 → 20 → 25 |

#### 微型代码（简化 Python 版）

```python
import random

class Node:
    def __init__(self, key, level):
        self.key = key
        self.forward = [None] * (level + 1)

class SkipList:
    def __init__(self, max_level=4, p=0.5):
        self.max_level = max_level
        self.p = p
        self.header = Node(-1, max_level)
        self.level = 0

    def random_level(self):
        lvl = 0
        while random.random() < self.p and lvl < self.max_level:
            lvl += 1
        return lvl

    def insert(self, key):
        update = [None] * (self.max_level + 1)
        current = self.header

        # 向右并向下移动
        for i in reversed(range(self.level + 1)):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]
        if current is None or current.key != key:
            lvl = self.random_level()
            if lvl > self.level:
                for i in range(self.level + 1, lvl + 1):
                    update[i] = self.header
                self.level = lvl
            new_node = Node(key, lvl)
            for i in range(lvl + 1):
                new_node.forward[i] = update[i].forward[i]
                update[i].forward[i] = new_node
```

#### 重要性

-   搜索、插入和删除的期望时间复杂度为 $O(\log n)$
-   比 AVL 树或红黑树更简单
-   概率性平衡避免了严格的旋转操作
-   常用于数据库和键值存储，如 LevelDB 和 Redis

#### 一个温和的证明（为什么它有效）

每个节点出现在第 $i$ 层的概率为 $p^i$。
每层的期望节点数为 $n p^i$。
总层数为 $O(\log_{1/p} n)$。

期望搜索路径长度：

$$
E[\text{步数}] = \frac{1}{1 - p} \log_{1/p} n = O(\log n)
$$

期望空间使用量：

$$
O(n) \text{ 个节点} \times O\!\left(\frac{1}{1 - p}\right) \text{ 每个节点的指针数}
$$

因此，跳表实现了期望的对数性能和线性空间。

#### 动手试试

1.  构建一个跳表并插入 $\{5, 10, 15, 20, 25\}$。
2.  插入 $17$ 并追踪每一层更新了哪些指针。
3.  尝试 $p = 0.25, 0.5, 0.75$。
4.  观察随机高度如何影响整体平衡。

#### 测试用例

| 操作   | 输入 | 期望结构（第 0 层）     |
| :----- | :--- | :---------------------- |
| Insert | 10   | 10                      |
| Insert | 5    | 5 → 10                  |
| Insert | 15   | 5 → 10 → 15             |
| Insert | 17   | 5 → 10 → 15 → 17        |
| Search | 15   | 找到                    |
| Search | 12   | 未找到                  |

#### 复杂度

| 操作   | 时间（期望） | 空间    |
| :----- | :----------- | :------ |
| Search | $O(\log n)$  | $O(n)$  |
| Insert | $O(\log n)$  | $O(n)$  |
| Delete | $O(\log n)$  | $O(n)$  |

跳表是一种简单而强大的数据结构。
它以随机性作为平衡力量，实现了树的优雅和链表的灵活性。
### 272 跳表删除

跳表中的删除操作与插入操作类似：我们从顶层向下遍历，记录每一层的前驱节点，并在目标节点出现的所有层级中解除其链接。该结构保持概率平衡，因此无需重新平衡，删除操作的期望时间复杂度为 $O(\log n)$。

#### 我们要解决什么问题？

我们希望从一个有序的、概率平衡的结构中高效地移除一个元素。朴素的链表需要 $O(n)$ 的遍历时间；平衡二叉搜索树（BST）则需要复杂的旋转操作。跳表为我们提供了一种折中方案，只需简单的指针更新，并具有期望的对数时间复杂度。

#### 工作原理（通俗解释）

跳表中的每个节点可以出现在多个层级。要删除键值 $x$：

1. 从顶层开始。
2. 当下一个节点的键值 < $x$ 时，向右移动。
3. 如果下一个节点的键值 == $x$，则将当前节点记录在 `update` 数组中。
4. 下降一层并重复上述步骤。
5. 到达底层后，从 `update` 数组中移除所有指向该节点的前向引用。
6. 如果最顶层变为空，则降低列表的层级。

#### 逐步示例

从此跳表中删除 $x = 17$：

| 层级 | 节点（删除前）             |
| ---- | -------------------------- |
| 3    | 5 → 15 → 25                |
| 2    | 5 → 10 → 15 → 25           |
| 1    | 5 → 10 → 15 → 17 → 20 → 25 |
| 0    | 5 → 10 → 15 → 17 → 20 → 25 |

遍历过程：

- 从第 3 层开始：15 < 17 → 向右移动 → 25 > 17 → 下降一层
- 第 2 层：15 < 17 → 向右移动 → 25 > 17 → 下降一层
- 第 1 层：15 < 17 → 向右移动 → 找到 17 → 记录前驱节点
- 第 0 层：15 < 17 → 向右移动 → 找到 17 → 记录前驱节点

从记录的前驱节点中移除所有指向 17 的前向指针。

| 层级 | 节点（删除后）         |
| ---- | --------------------- |
| 3    | 5 → 15 → 25           |
| 2    | 5 → 10 → 15 → 25      |
| 1    | 5 → 10 → 15 → 20 → 25 |
| 0    | 5 → 10 → 15 → 20 → 25 |

#### 微型代码（简化版 Python）

```python
import random

class Node:
    def __init__(self, key, level):
        self.key = key
        self.forward = [None] * (level + 1)

class SkipList:
    def __init__(self, max_level=4, p=0.5):
        self.max_level = max_level
        self.p = p
        self.header = Node(-1, max_level)
        self.level = 0

    def delete(self, key):
        update = [None] * (self.max_level + 1)
        current = self.header

        # 从上到下遍历
        for i in reversed(range(self.level + 1)):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        # 找到节点
        if current and current.key == key:
            for i in range(self.level + 1):
                if update[i].forward[i] != current:
                    continue
                update[i].forward[i] = current.forward[i]
            # 如果最高层为空，则降低层级
            while self.level > 0 and self.header.forward[self.level] is None:
                self.level -= 1
```

#### 为什么重要

- 与插入操作对称
- 无需旋转或重新平衡
- 期望 $O(\log n)$ 的性能
- 非常适合有序映射、数据库、键值存储

#### 一个温和的证明（为什么有效）

每一层包含比例为 $p^i$ 的节点。期望遍历的层级数为 $O(\log_{1/p} n)$。在每一层，我们平均水平移动 $O(1)$ 次。

因此期望成本为：

$$
E[T_{\text{delete}}] = O(\log n)
$$

#### 自己动手试试

1. 插入 ${5, 10, 15, 17, 20, 25}$。
2. 删除 $17$。
3. 逐层跟踪所有指针变化。
4. 与 AVL 树删除的复杂度进行比较。

#### 测试用例

| 操作     | 输入           | 期望的第 0 层结果 |
| -------- | -------------- | ----------------- |
| 插入     | 5,10,15,17,20  | 5 → 10 → 15 → 17 → 20 |
| 删除     | 17             | 5 → 10 → 15 → 20      |
| 删除     | 10             | 5 → 15 → 20           |
| 删除     | 5              | 15 → 20               |

#### 复杂度

| 操作     | 时间（期望）   | 空间      |
| -------- | -------------- | --------- |
| 查找     | $O(\log n)$    | $O(n)$    |
| 插入     | $O(\log n)$    | $O(n)$    |
| 删除     | $O(\log n)$    | $O(n)$    |

跳表删除通过简洁性保持了优雅，它使用清晰的指针调整代替了复杂的树结构调整。
### 273 跳表搜索

在跳表中搜索就像一场跨层级的舞蹈：我们向右移动直到无法继续，然后向下移动，如此重复，直到找到目标键值或确定其不存在。
得益于随机化的层级结构，其期望时间复杂度为 $O(\log n)$，与平衡二叉搜索树相当，但指针逻辑更简单。

#### 我们要解决什么问题？

我们需要在一个有序集合中实现快速搜索，并且该结构能优雅地适应动态的插入和删除操作。
平衡树能保证 $O(\log n)$ 的搜索时间，但需要进行旋转操作。
跳表通过随机化而非严格的平衡规则，实现了相同的期望时间复杂度。

#### 工作原理（通俗解释）

跳表包含多个层级的链表。
每一层都充当一条“快车道”，可以跳过多个节点。
搜索键值 $x$ 的步骤如下：

1.  从左上角的头节点开始。
2.  在每一层，当 `next.key < x` 时，向右移动。
3.  当 `next.key ≥ x` 时，向下移动一层。
4.  重复步骤 2 和 3，直到第 0 层。
5.  如果 `current.forward[0].key == x`，则找到；否则，未找到。

搜索路径在层级间“曲折”前进，平均大约访问 $\log n$ 个节点。

#### 逐步示例

在以下跳表中搜索 $x = 17$：

| 层级 | 节点序列                     |
| ---- | ---------------------------- |
| 3    | 5 → 15 → 25                  |
| 2    | 5 → 10 → 15 → 25             |
| 1    | 5 → 10 → 15 → 17 → 20 → 25   |
| 0    | 5 → 10 → 15 → 17 → 20 → 25   |

遍历过程：

-   第 3 层：5 → 15 → (下一个 = 25 > 17) → 下降
-   第 2 层：15 → (下一个 = 25 > 17) → 下降
-   第 1 层：15 → 17 找到 → 成功
-   第 0 层：确认 17 存在

路径：5 → 15 → (下降) → 15 → (下降) → 15 → 17

#### 可视化

跳表搜索遵循阶梯模式：

```
Level 3:  5 --------> 15 -----↓
Level 2:  5 ----> 10 --> 15 --↓
Level 1:  5 -> 10 -> 15 -> 17 -> 20
Level 0:  5 -> 10 -> 15 -> 17 -> 20
```

每个 "↓" 表示当下一个节点过大时下降一层。

#### 精简代码（简化版 Python）

```python
class SkipList:
    def __init__(self, max_level=4, p=0.5):
        self.max_level = max_level
        self.p = p
        self.header = Node(-1, max_level)
        self.level = 0

    def search(self, key):
        current = self.header
        # 从上到下遍历
        for i in reversed(range(self.level + 1)):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        current = current.forward[0]
        return current and current.key == key
```

#### 为什么重要

-   简单高效：期望 $O(\log n)$ 时间复杂度
-   概率平衡：避免了树的旋转操作
-   是有序映射、索引和数据库的基础
-   搜索路径长度平均是对数级的

#### 一个温和的证明（为什么有效）

每一层大约包含下一层节点数的 $p$ 倍。
期望的层数：$O(\log_{1/p} n)$。

在每一层，期望的水平移动次数：$O(1)$。

因此，总的期望搜索时间：

$$
E[T_{\text{search}}] = O(\log n)
$$

#### 动手试试

1.  用 ${5, 10, 15, 17, 20, 25}$ 构建一个跳表。
2.  搜索 $17$ 并追踪每一层的路径。
3.  搜索 $13$，你在哪里停止？
4.  将路径长度与相同大小的二叉搜索树进行比较。

#### 测试用例

| 操作   | 输入 | 预期输出     | 路径           |
| ------ | ---- | ------------ | -------------- |
| 搜索   | 17   | 找到         | 5 → 15 → 17    |
| 搜索   | 10   | 找到         | 5 → 10         |
| 搜索   | 13   | 未找到       | 5 → 10 → 15    |
| 搜索   | 5    | 找到         | 5              |

#### 复杂度

| 操作   | 时间（期望） | 空间   |
| ------ | ------------ | ------ |
| 搜索   | $O(\log n)$  | $O(n)$ |
| 插入   | $O(\log n)$  | $O(n)$ |
| 删除   | $O(\log n)$  | $O(n)$ |

跳表搜索展示了概率结构如何产生类似确定性的效率，沿着随机性的阶梯走向确定性。
### 274 B树插入

B树是一种为外部存储系统（如磁盘或SSD）设计的平衡搜索树。与二叉树不同，每个节点可以存储多个键和多个子节点，通过将更多数据打包到单个节点中来最小化磁盘I/O。
向B树中插入操作通过按需分裂已满的节点来保持排序顺序和平衡。

#### 我们要解决什么问题？

当数据太大而无法装入内存时，标准的二叉树性能会很差，因为每次节点访问都可能触发一次磁盘读取。
我们需要一种结构，能够：

- 减少I/O操作次数
- 保持树的高度较小
- 保持键的排序顺序
- 在 $O(\log n)$ 时间内支持搜索、插入和删除

B树通过在每个节点存储多个键并通过受控分裂进行自我平衡来解决这个问题。

#### 工作原理（通俗解释）

每个B树节点最多可以包含 $2t - 1$ 个键和 $2t$ 个子节点，其中 $t$ 是最小度数。

插入步骤：

1.  从根节点开始，像二分搜索一样向下遍历。
2.  如果子节点已满（有 $2t - 1$ 个键），则在下降之前将其分裂。
3.  将新键插入到合适的非满节点中。

分裂一个已满的节点：

- 中间的键向上移动到父节点
- 左半部分和右半部分成为独立的子节点

这确保了每个节点都保持在允许的大小范围内，从而将高度保持在 $O(\log_t n)$。

#### 逐步示例

设 $t = 2$（每个节点最多3个键）。
按顺序插入键：$[10, 20, 5, 6, 12, 30, 7, 17]$

步骤1：插入 10 → 根节点 = [10]
步骤2：插入 20 → [10, 20]
步骤3：插入 5 → [5, 10, 20]
步骤4：插入 6 → 节点已满 → 分裂

分裂 [5, 6, 10, 20]：

- 中间键 10 向上移动
- 左子节点 [5, 6]，右子节点 [20]
  树：

```
      [10]
     /    \
 [5, 6]   [20]
```

步骤5：插入 12 → 向右走 → [12, 20]
步骤6：插入 30 → [12, 20, 30]
步骤7：插入 7 → 向左走 → [5, 6, 7] → 已满 → 分裂

- 中间键 6 向上移动

现在树为：

```
        [6, 10]
       /   |    \
 [5]  [7]  [12, 20, 30]
```

步骤8：插入 17 → 走到 [12, 20, 30] → 插入后为 [12, 17, 20, 30] → 分裂

- 中间键 20 向上移动

最终树：

```
          [6, 10, 20]
         /   |   |   \
       [5] [7] [12,17] [30]
```

#### 可视化

B树在每一层都维护排序的键，并通过在插入时分裂节点来保证最小高度。

```
根节点: [6, 10, 20]
子节点: [5], [7], [12, 17], [30]
```

#### 微型代码（简化版 Python）

```python
class BTreeNode:
    def __init__(self, t, leaf=False):
        self.keys = [] # 键列表
        self.children = [] # 子节点列表
        self.leaf = leaf # 是否为叶节点
        self.t = t # 最小度数

    def insert_non_full(self, key):
        i = len(self.keys) - 1
        if self.leaf:
            self.keys.append(key)
            self.keys.sort()
        else:
            while i >= 0 and key < self.keys[i]:
                i -= 1
            i += 1
            if len(self.children[i].keys) == 2 * self.t - 1:
                self.split_child(i)
                if key > self.keys[i]:
                    i += 1
            self.children[i].insert_non_full(key)

    def split_child(self, i):
        t = self.t
        y = self.children[i]
        z = BTreeNode(t, y.leaf)
        mid = y.keys[t - 1]
        z.keys = y.keys[t:]
        y.keys = y.keys[:t - 1]
        if not y.leaf:
            z.children = y.children[t:]
            y.children = y.children[:t]
        self.children.insert(i + 1, z)
        self.keys.insert(i, mid)

class BTree:
    def __init__(self, t):
        self.root = BTreeNode(t, True)
        self.t = t

    def insert(self, key):
        r = self.root
        if len(r.keys) == 2 * self.t - 1:
            s = BTreeNode(self.t)
            s.children.insert(0, r)
            s.split_child(0)
            i = 0
            if key > s.keys[0]:
                i += 1
            s.children[i].insert_non_full(key)
            self.root = s
        else:
            r.insert_non_full(key)
```

#### 为什么重要

- 磁盘友好：每个节点恰好装入一个页面
- 高度浅：$O(\log_t n)$ 层 → 磁盘读取次数少
- 确定性平衡：没有随机性，始终保持平衡
- 文件系统、数据库、索引的基础（例如 NTFS、MySQL、PostgreSQL）

#### 一个温和的证明（为什么有效）

每个节点（根节点除外）的键数在 $t-1$ 到 $2t-1$ 之间。

只有当根节点分裂时，树的高度才会增加。

因此，高度 $h$ 满足：

$$
t^h \le n \le (2t)^h
$$

取对数：

$$
h = O(\log_t n)
$$

所以插入和搜索的时间复杂度为 $O(t \cdot \log_t n)$，当 $t$ 为常数时通常简化为 $O(\log n)$。

#### 自己动手试试

1.  构建一个 $t=2$ 的B树。
2.  插入 $[10, 20, 5, 6, 12, 30, 7, 17]$。
3.  在每次插入后绘制树的结构。
4.  观察分裂何时发生以及哪些键被提升。

#### 测试用例

| 输入键序列             | t | 最终根节点 | 高度 |
| ---------------------- | - | ---------- | ---- |
| [10,20,5,6,12,30,7,17] | 2 | [6,10,20]  | 2    |
| [1,2,3,4,5,6,7,8,9]    | 2 | [4]        | 3    |

#### 复杂度

| 操作     | 时间复杂度   | 空间复杂度 |
| -------- | ------------ | ---------- |
| 搜索     | $O(\log n)$  | $O(n)$     |
| 插入     | $O(\log n)$  | $O(n)$     |
| 删除     | $O(\log n)$  | $O(n)$     |

B树插入是外部存储算法的心跳，分裂、提升、平衡，确保数据保持紧密、浅层和有序。
### 275 B 树删除

B 树中的删除操作比插入操作更为复杂，我们必须小心地移除一个键，同时保持 B 树的平衡特性。每个节点必须至少包含 $t - 1$ 个键（根节点除外），因此删除操作可能涉及从兄弟节点借用键或合并节点。

目标是维护 B 树的不变式：

- 每个节点内的键有序排列
- 节点键数在 $t - 1$ 到 $2t - 1$ 之间
- 高度平衡

#### 我们要解决什么问题？

我们希望从 B 树中移除一个键，而不违反平衡性或占用约束。
与二叉搜索树不同，在二叉搜索树中我们可以简单地替换或修剪节点，而 B 树必须保持最小度数，以确保高度一致和 I/O 效率。

#### 工作原理（通俗解释）

要从 B 树中删除键 $k$：

1. 如果 $k$ 在叶子节点中：

   * 直接移除它。

2. 如果 $k$ 在内部节点中：

   * 情况 A：如果左子节点有 ≥ $t$ 个键 → 用前驱键替换 $k$。
   * 情况 B：否则，如果右子节点有 ≥ $t$ 个键 → 用后继键替换 $k$。
   * 情况 C：否则，两个子节点都只有 $t - 1$ 个键 → 合并它们并递归处理。

3. 如果 $k$ 不在当前节点中：

   * 移动到正确的子节点。
   * 在下降之前，确保该子节点至少有 $t$ 个键。如果没有，

     * 从一个有 ≥ $t$ 个键的兄弟节点借用，或者
     * 与一个兄弟节点合并以保证占用率。

这确保了在遍历过程中不会发生下溢。

#### 逐步示例

令 $t = 2$（每个节点最多 3 个键）。
删除前的 B 树：

```
          [6, 10, 20]
         /    |     |    \
       [5]  [7]  [12,17] [30]
```

#### 删除键 17

- 17 在叶子节点 [12, 17] 中 → 直接移除它

结果：

```
          [6, 10, 20]
         /    |     |    \
       [5]  [7]   [12]   [30]
```

#### 删除键 10

- 10 在内部节点中
- 左子节点 [7] 有 $t - 1 = 1$ 个键
- 右子节点 [12] 也有 1 个键
  → 合并 [7], 10, [12] → [7, 10, 12]

树变为：

```
       [6, 20]
      /    |    \
    [5] [7,10,12] [30]
```

#### 删除键 6

- 6 在内部节点中
- 左子节点 [5] 有 1 个键，右子节点 [7,10,12] 有 ≥ 2 个键 → 从右子节点借用
- 用后继键 7 替换 6

重新平衡后的树：

```
       [7, 20]
      /    |    \
    [5,6] [10,12] [30]
```

#### 可视化

每次删除都通过确保所有节点（根节点除外）保持 ≥ $t - 1$ 的占用率来维持树的平衡。

```
删除前：             删除 10 后：
$$6,10,20]           [6,20]
 / | | \             / | \
$$5][7][12,17][30]   [5][7,10,12][30]
```

#### 简化代码（简化版 Python）

```python
class BTreeNode:
    def __init__(self, t, leaf=False):
        self.t = t
        self.keys = []
        self.children = []
        self.leaf = leaf

    def find_key(self, k):
        for i, key in enumerate(self.keys):
            if key >= k:
                return i
        return len(self.keys)

class BTree:
    def __init__(self, t):
        self.root = BTreeNode(t, True)
        self.t = t

    def delete(self, node, k):
        t = self.t
        i = node.find_key(k)

        # 情况 1：键在节点中
        if i < len(node.keys) and node.keys[i] == k:
            if node.leaf:
                node.keys.pop(i)
            else:
                if len(node.children[i].keys) >= t:
                    pred = self.get_predecessor(node, i)
                    node.keys[i] = pred
                    self.delete(node.children[i], pred)
                elif len(node.children[i+1].keys) >= t:
                    succ = self.get_successor(node, i)
                    node.keys[i] = succ
                    self.delete(node.children[i+1], succ)
                else:
                    self.merge(node, i)
                    self.delete(node.children[i], k)
        else:
            # 情况 2：键不在节点中
            if node.leaf:
                return  # 未找到
            if len(node.children[i].keys) < t:
                self.fill(node, i)
            self.delete(node.children[i], k)
```

*（为简洁起见，省略了辅助方法 `merge`、`fill`、`borrow_from_prev` 和 `borrow_from_next`）*

#### 为什么重要

- 在删除后保持平衡的高度
- 防止子节点发生下溢
- 确保 $O(\log n)$ 的复杂度
- 广泛应用于数据库和文件系统，这些场景对稳定性能要求极高

#### 一个温和的证明（为什么它有效）

B 树节点始终满足：

$$
t - 1 \le \text{每个节点的键数} \le 2t - 1
$$

合并或借用操作确保所有节点保持在界限内。
高度 $h$ 满足：

$$
h \le \log_t n
$$

因此，删除操作最多需要访问 $O(\log_t n)$ 个节点，并在每层执行常数时间的合并/借用操作。

因此：

$$
T_{\text{delete}} = O(\log n)
$$

#### 自己动手试试

1. 构建一个 $t = 2$ 的 B 树。
2. 插入 $[10, 20, 5, 6, 12, 30, 7, 17]$。
3. 按顺序删除键：$17, 10, 6$。
4. 在每次删除后绘制树，并观察合并/借用操作。

#### 测试用例

| 输入键                 | 删除键 | 结果（第 0 层）     |
| ---------------------- | ------ | ------------------- |
| [5,6,7,10,12,17,20,30] | 17     | [5,6,7,10,12,20,30] |
| [5,6,7,10,12,20,30]    | 10     | [5,6,7,12,20,30]    |
| [5,6,7,12,20,30]       | 6      | [5,7,12,20,30]      |

#### 复杂度

| 操作     | 时间        | 空间  |
| -------- | ----------- | ------ |
| 搜索     | $O(\log n)$ | $O(n)$ |
| 插入     | $O(\log n)$ | $O(n)$ |
| 删除     | $O(\log n)$ | $O(n)$ |

B 树删除是一种精密的平衡操作，通过合并、借用和提升键，恰到好处地保持树的紧凑、低矮和有序。
### 276 B+ 树搜索

B+ 树是 B 树的扩展，针对范围查询和顺序访问进行了优化。所有实际数据（记录或值）都存储在叶子节点中，这些叶子节点通过链表连接在一起形成一个有序列表。内部节点仅包含用于引导搜索的键。

在 B+ 树中进行搜索遵循与 B 树相同的原理，即基于键比较进行自上而下的遍历，但最终会到达存储实际数据的叶子层。

#### 我们要解决什么问题？

我们需要一种对磁盘友好的搜索结构，它能够：

- 保持较低的高度（减少磁盘 I/O 次数）
- 支持快速的范围扫描
- 将索引键（内部节点）与记录（叶子节点）分离

B+ 树通过以下特性满足这些需求：

- 高扇出：每个节点包含许多键
- 链接的叶子节点：用于高效的顺序遍历
- 确定性平衡：高度始终为 $O(\log n)$

#### 工作原理（通俗解释）

每个内部节点充当一个路由器。每个叶子节点包含键和数据指针。

要搜索键 $k$：

1.  从根节点开始。
2.  在每个内部节点，找到其键范围包含 $k$ 的子节点。
3.  沿着该指针进入下一层。
4.  继续直到到达叶子节点。
5.  在叶子节点内执行线性扫描以找到 $k$。

如果在叶子节点中未找到，则 $k$ 不在树中。

#### 逐步示例

设 $t = 2$（每个节点最多容纳 3 个键）。
B+ 树：

```
          [10 | 20]
         /     |     \
   [1 5 8]  [12 15 18]  [22 25 30]
```

搜索 15：

- 根节点 [10 | 20]：15 > 10 且 < 20 → 跟随中间指针
- 节点 [12 15 18]：找到 15

搜索 17：

- 根节点 [10 | 20]：17 > 10 且 < 20 → 中间指针
- 节点 [12 15 18]：未找到 → 不在树中

#### 可视化

```
         [10 | 20]
        /     |     \
 [1 5 8] [12 15 18] [22 25 30]
```

- 内部节点引导路径
- 叶子节点保存数据（并链接到下一个叶子节点）
- 搜索总是终止于叶子节点

#### 简化代码（简化版 Python）

```python
class BPlusNode:
    def __init__(self, t, leaf=False):
        self.t = t
        self.leaf = leaf
        self.keys = []
        self.children = []
        self.next = None  # 指向下一个叶子节点的链接

class BPlusTree:
    def __init__(self, t):
        self.root = BPlusNode(t, True)
        self.t = t

    def search(self, node, key):
        if node.leaf:
            return key in node.keys
        i = 0
        while i < len(node.keys) and key >= node.keys[i]:
            i += 1
        return self.search(node.children[i], key)

    def find(self, key):
        return self.search(self.root, key)
```

#### 为什么它很重要

- 高效的磁盘 I/O：高分支因子保持较低的高度
- 所有数据在叶子节点中：简化了范围查询
- 链接的叶子节点：支持顺序遍历（按排序顺序）
- 应用场景：数据库、文件系统、键值存储（例如 MySQL、InnoDB、NTFS）

#### 一个温和的证明（为什么它有效）

每个内部节点有 $t$ 到 $2t$ 个子节点。
每个叶子节点保存 $t - 1$ 到 $2t - 1$ 个键。
因此，高度 $h$ 满足：

$$
t^h \le n \le (2t)^h
$$

取对数：

$$
h = O(\log_t n)
$$

所以搜索时间是：

$$
T_{\text{search}} = O(\log n)
$$

并且由于叶子节点内的最终扫描是常数时间（很小），总成本仍保持对数级别。

#### 动手试试

1.  用 $t = 2$ 构建一个 B+ 树，并插入键 $[1, 5, 8, 10, 12, 15, 18, 20, 22, 25, 30]$。
2.  搜索 15、17 和 8。
3.  追踪你从根节点 → 内部节点 → 叶子节点的路径。
4.  观察所有搜索都终止于叶子节点。

#### 测试用例

| 搜索键 | 预期结果 | 路径                     |
| ------ | -------- | ------------------------ |
| 15     | 找到     | 根节点 → 中间节点 → 叶子节点 |
| 8      | 找到     | 根节点 → 左节点 → 叶子节点   |
| 17     | 未找到   | 根节点 → 中间节点 → 叶子节点 |
| 25     | 找到     | 根节点 → 右节点 → 叶子节点   |

#### 复杂度

| 操作       | 时间            | 空间    |
| ---------- | --------------- | ------- |
| 搜索       | $O(\log n)$     | $O(n)$  |
| 插入       | $O(\log n)$     | $O(n)$  |
| 范围查询   | $O(\log n + k)$ | $O(n)$  |

B+ 树搜索体现了 I/O 感知设计：每个被跟随的指针对应一个磁盘页，每次叶子节点扫描都对缓存友好，并且每个键都恰好位于范围查询需要它的地方。
### 278 B* 树

B* 树是 B 树的一个改进版本，旨在实现更高的节点占用率和更少的分裂。它强制要求每个节点（除了根节点）必须至少填充三分之二，相比之下，标准 B 树只保证半满。

为了实现这一点，B* 树在分裂之前使用兄弟节点之间的键值重分配，这提高了空间利用率和 I/O 效率，使其成为数据库和文件系统索引的理想选择。

#### 我们要解决什么问题？

在标准 B 树中，每个节点至少维护 $t - 1$ 个键（占用率 50%）。但频繁的分裂会导致碎片化和空间浪费。

我们希望：
- 提高空间效率（减少空槽）
- 尽可能推迟分裂
- 保持平衡和排序顺序

B* 树通过在分裂前在兄弟节点之间借用和重新分配键来解决这个问题，确保占用率 $\ge 2/3$。

#### 工作原理（通俗解释）

B* 树的工作方式类似于 B 树，但采用了更智能的分裂逻辑：

1. 插入路径：
   * 自上而下遍历以找到目标叶节点。
   * 如果目标节点已满，则检查其兄弟节点。

2. 重分配步骤：
   * 如果兄弟节点有空位，则在它们和父节点键之间重新分配键。

3. 双重分裂步骤：
   * 如果两个兄弟节点都已满，则将它们分裂成三个节点（两个满节点 + 一个新节点），并在它们之间均匀地重新分配键。

这确保了每个节点（除了根节点）至少填充了 2/3，从而带来更好的磁盘利用率。

#### 逐步示例

设 $t = 2$（每个节点最多 3 个键）。
插入键：$[5, 10, 15, 20, 25, 30, 35]$

步骤 1–4：像 B 树一样构建，直到根节点为 [10, 20]。

```
        [10 | 20]
       /    |    \
   [5]   [15]   [25, 30, 35]
```

现在插入 40 → 最右边的节点 [25, 30, 35] 已满。
- 检查兄弟节点：左兄弟节点 [15] 有空位 → 重新分配键
  合并 [15]、[25, 30, 35] 和父节点键 20 → [15, 20, 25, 30, 35]
  均匀地分裂成三个节点：[15, 20]、[25, 30]、[35]
  使用新的分隔符更新父节点。

结果：

```
         [20 | 30]
        /     |     \
    [5,10,15] [20,25] [30,35,40]
```

每个节点 ≥ 2/3 满，没有浪费空间。

#### 可视化

```
          [20 | 30]
         /     |     \
 [5 10 15] [20 25] [30 35 40]
```

重分配确保了在分裂之前的平衡和密度。

#### 微型代码（简化伪代码）

```python
def insert_bstar(tree, key):
    node = find_leaf(tree.root, key)
    if node.full():
        sibling = node.get_sibling()
        if sibling and not sibling.full():
            redistribute(node, sibling, tree.parent(node))
        else:
            split_three(node, sibling, tree.parent(node))
    insert_into_leaf(node, key)
```

*（实际实现更复杂，涉及父节点更新和兄弟节点指针。）*

#### 为什么重要

- 更好的空间利用率：节点 ≥ 66% 满
- 更少的分裂：在大量插入下性能更稳定
- 改进的 I/O 局部性：访问的磁盘块更少
- 用于：数据库系统（IBM DB2）、文件系统（ReiserFS）、基于 B* 的缓存结构

#### 一个温和的证明（为什么它有效）

在 B* 树中：
- 每个非根节点包含 $\ge \frac{2}{3} (2t - 1)$ 个键。
- 高度 $h$ 满足：

$$
\left( \frac{3}{2} t \right)^h \le n
$$

取对数：

$$
h = O(\log_t n)
$$

因此，高度保持对数级，但每层节点打包了更多数据。

更少的层数 → 更少的 I/O → 更好的性能。

#### 自己动手试试

1. 构建一个 $t = 2$ 的 B* 树。
2. 插入键：$[5, 10, 15, 20, 25, 30, 35, 40]$。
3. 观察在分裂之前如何发生重分配。
4. 与相同序列的 B 树分裂进行比较。

#### 测试用例

| 输入键              | t | 结果                     | 备注                             |
| ------------------- | - | ------------------------ | -------------------------------- |
| [5,10,15,20,25]     | 2 | 根节点 [15]              | 完全利用                         |
| [5,10,15,20,25,30,35] | 2 | 根节点 [20,30]           | 分裂前重分配                     |
| [1..15]             | 2 | 平衡，节点 2/3 满        | 高密度                           |

#### 复杂度

| 操作     | 时间          | 空间    | 占用率      |
| -------- | ------------- | ------- | ----------- |
| 搜索     | $O(\log n)$   | $O(n)$  | $\ge 66%$   |
| 插入     | $O(\log n)$   | $O(n)$  | $\ge 66%$   |
| 删除     | $O(\log n)$   | $O(n)$  | $\ge 66%$   |

B* 树汲取了 B 树的优雅之处，并将其推向更完美的境界：更少的分裂、更密集的节点，以及对大型数据集更平滑的扩展。
### 279 自适应基数树

自适应基数树（ART）是一种空间高效、缓存友好的数据结构，它结合了字典树和基数树的理念。它根据子节点数量动态调整节点表示方式，从而优化内存使用和查找速度。

与固定大小的基数树（稀疏节点会浪费空间）不同，ART 根据占用情况选择紧凑的节点类型（如 Node4、Node16、Node48、Node256），并按需增长。

#### 我们要解决什么问题？

标准的字典树和基数树速度快但内存占用大。
如果键共享长前缀，许多节点只持有一个子节点，浪费内存。

我们想要一种结构，能够：

- 保持 O(L) 的查找时间（L = 键长度）
- 根据占用情况调整节点大小
- 最小化指针开销
- 利用缓存局部性

ART 通过动态切换节点类型（随着子节点数量增长）来实现这一点。

#### 工作原理（通俗解释）

ART 中的每个内部节点可以是以下四种类型之一：

| 节点类型 | 容量       | 描述                       |
| -------- | ---------- | -------------------------- |
| Node4    | 4 个子节点 | 最小，使用线性搜索         |
| Node16   | 16 个子节点 | 小数组，向量化搜索         |
| Node48   | 48 个子节点 | 索引映射，存储子节点指针   |
| Node256  | 256 个子节点 | 按字节值直接寻址           |

键按字节逐个处理，在每一层进行分支。
当一个节点超出其容量时，它会升级到下一个节点类型。

#### 示例

插入键：`["A", "AB", "AC", "AD", "AE"]`

1.  从根节点 Node4 开始（可存储 4 个子节点）。
2.  插入 "AE" 后，Node4 超出容量 → 升级为 Node16。
3.  子节点按键字节保持排序顺序。

这种自适应升级保持了节点的紧凑和高效。

#### 逐步示例

| 步骤 | 操作        | 节点类型          | 存储的键           | 备注             |
| ---- | ----------- | ----------------- | ------------------ | ---------------- |
| 1    | 插入 "A"    | Node4             | A                  | 创建根节点       |
| 2    | 插入 "AB"   | Node4             | A, AB              | 添加分支         |
| 3    | 插入 "AC"   | Node4             | A, AB, AC          | 仍少于 4 个      |
| 4    | 插入 "AD"   | Node4             | A, AB, AC, AD      | 已满             |
| 5    | 插入 "AE"   | 升级为 Node16     | A, AB, AC, AD, AE  | 自适应增长       |

#### 可视化

```
根节点 (Node16)
 ├── 'A' → 节点
      ├── 'B' (叶子节点)
      ├── 'C' (叶子节点)
      ├── 'D' (叶子节点)
      └── 'E' (叶子节点)
```

每种节点类型都调整其布局以获得最佳性能。

#### 微型代码（简化伪代码）

```python
class Node:
    def __init__(self):
        self.children = {}

def insert_art(root, key):
    node = root
    for byte in key:
        if byte not in node.children:
            node.children[byte] = Node()
        node = node.children[byte]
    node.value = True
```

*(真实的 ART 会在 Node4、Node16、Node48、Node256 表示之间动态切换。)*

#### 为什么它很重要

- 自适应内存使用，稀疏节点不浪费空间
- 缓存友好，连续内存布局
- 查找速度快，Node16 支持向量化搜索
- 用于现代数据库（例如 HyPer、Umbra、DuckDB）

#### 一个温和的证明（为什么它有效）

令 $L$ = 键长度，$b$ = 分支因子（每个字节最多 256）。
在朴素的字典树中，每个节点分配 $O(b)$ 个槽位，许多未使用。

在 ART 中：

- 每个节点只存储实际的子节点，因此
  $$
  \text{空间} \approx O(n + L)
  $$
- 查找时间保持 $O(L)$，因为我们每个字节遍历一个节点。
- 空间效率的提升与稀疏程度成正比。

因此，ART 保持了类似字典树的性能，同时具有类似哈希表的紧凑性。

#### 亲自尝试

1.  插入 `["dog", "dot", "door", "dorm"]`
2.  观察 Node4 → Node16 的转换是如何发生的
3.  计算节点数量，与朴素字典树进行比较
4.  测量内存使用情况和访问速度

#### 测试用例

| 键                             | 生成的根节点类型 | 备注                       |
| ------------------------------ | ---------------- | -------------------------- |
| `["a", "b"]`                   | Node4            | 2 个子节点                 |
| `["a", "b", "c", "d", "e"]`    | Node16           | 第 5 次插入后升级          |
| `["aa", "ab", "ac"... "az"]`   | Node48 或 Node256 | 密集分支                   |

#### 复杂度

| 操作   | 时间   | 空间      | 自适应行为         |
| ------ | ------ | --------- | ------------------ |
| 搜索   | $O(L)$ | $O(n + L)$ | 节点增长/收缩      |
| 插入   | $O(L)$ | $O(n)$     | 节点类型升级       |
| 删除   | $O(L)$ | $O(n)$     | 如果稀疏则降级     |

自适应基数树结合了两者的优点：字典树的前缀压缩和哈希表的空间效率，是现代高性能索引的强大工具。
### 280 字典树压缩

压缩字典树（也称为基数树或 Patricia 树）是字典树的一种优化形式，它将单子节点链合并为单一边。每个边不再存储一个字符，而是可以存储整个子字符串。

这降低了字典树的高度，最小化了内存使用，并加速了搜索，非常适合前缀查找、路由表和字典存储等应用。

#### 我们要解决什么问题？

当许多节点只有一个子节点时，朴素的字典树会浪费空间。

例如，将 `["cat", "car", "dog"]` 插入一个朴素字典树会产生长而细的路径：

```
c → a → t
c → a → r
d → o → g
```

我们可以将这些线性链压缩为带有子字符串标签的边：

```
c → a → "t"
c → a → "r"
d → "og"
```

这节省了内存并减少了遍历深度。

#### 工作原理（通俗解释）

核心思想是路径压缩：每当一个节点只有一个子节点时，就将它们合并为一条包含组合子字符串的边。

| 步骤 | 操作                           | 结果                       |
| ---- | ------------------------------ | -------------------------- |
| 1    | 构建一个普通的字典树           | 每条边一个字符             |
| 2    | 遍历每条路径                   | 如果节点只有一个子节点，合并 |
| 3    | 用子字符串边替换链             | 节点更少，高度更短         |

压缩字典树将边标签存储为子字符串，而不是单个字符。

#### 示例

插入 `["bear", "bell", "bid", "bull", "buy"]`

1.  从朴素字典树开始。
2.  识别单子节点路径。
3.  合并路径：

```
b
 ├── e → "ar"
 │    └── "ll"
 └── u → "ll"
      └── "y"
```

现在每条边都承载一个子字符串，而不是单个字母。

#### 逐步示例

| 步骤 | 插入                         | 操作                         | 结果       |
| ---- | ---------------------------- | ---------------------------- | ---------- |
| 1    | "bear"                       | 创建路径 b-e-a-r             | 4 个节点   |
| 2    | "bell"                       | 共享前缀 "be"                | 合并前缀   |
| 3    | "bid"                        | 在 "b" 处新建分支            | 添加新边   |
| 4    | 压缩单子节点路径             | 用子字符串替换边             |            |

#### 微型代码（简化伪代码）

```python
class Node:
    def __init__(self):
        self.children = {}
        self.is_end = False

def insert_trie_compressed(root, word):
    node = root
    i = 0
    while i < len(word):
        for edge, child in node.children.items():
            prefix_len = common_prefix(edge, word[i:])
            if prefix_len > 0:
                if prefix_len < len(edge):
                    # 分割边
                    remainder = edge[prefix_len:]
                    new_node = Node()
                    new_node.children[remainder] = child
                    node.children[word[i:i+prefix_len]] = new_node
                    del node.children[edge]
                node = node.children[word[i:i+prefix_len]]
                i += prefix_len
                break
        else:
            node.children[word[i:]] = Node()
            node.children[word[i:]].is_end = True
            break
    node.is_end = True
```

这个简化版本在可能的情况下合并边。

#### 为什么重要

-   通过合并链来节省内存
-   搜索更快（每次查找跳转更少）
-   非常适合基于前缀的查询
-   用于路由表、自动补全系统和字典

#### 一个温和的证明（为什么有效）

令 $n$ 为所有键的总长度，$k$ 为键的数量。

-   一个朴素的字典树最多可以有 $O(n)$ 个节点。
-   一个压缩字典树最多有 $k - 1$ 个内部节点和 $k$ 个叶子节点，因为每个分支点对应至少两个键共享的唯一前缀。

因此，压缩字典树同时减少了高度和节点数量：

$$
O(n) \text{ 个节点（朴素）} \quad \to \quad O(k) \text{ 个节点（压缩）}
$$

搜索和插入操作仍然是 $O(L)$，其中 $L$ 是键的长度，但步骤更少。

#### 自己动手试试

1.  插入 `["car", "cat", "cart", "dog"]`
2.  绘制朴素字典树和压缩字典树
3.  计算压缩前后的节点数
4.  验证边标签是否为子字符串

#### 测试用例

| 键                         | 朴素字典树节点数 | 压缩字典树节点数 |
| -------------------------- | ---------------- | ---------------- |
| `["a", "b"]`               | 3                | 3                |
| `["apple", "app"]`         | 6                | 4                |
| `["abc", "abd", "aef"]`    | 8                | 6                |
| `["car", "cart", "cat"]`   | 9                | 6                |

#### 复杂度

| 操作   | 时间   | 空间   | 注释                       |
| ------ | ------ | ------ | -------------------------- |
| 搜索   | $O(L)$ | $O(k)$ | $L$ = 键的长度             |
| 插入   | $O(L)$ | $O(k)$ | 需要时分割边               |
| 删除   | $O(L)$ | $O(k)$ | 如果路径缩短则合并边       |

压缩字典树优雅地融合了字典树结构和路径压缩，将长链转化为紧凑的边，这是迈向高效前缀树、路由表和用于文本索引的字典树的关键一步。

## 第 29 节 持久化和函数式数据结构
### 281 持久化栈

持久化栈是一种版本化的数据结构，能够记住其所有过去的状态。每次 `push` 或 `pop` 操作都不会覆盖数据，而是创建一个栈的新版本，同时仍可访问旧版本。

这一概念属于函数式数据结构范畴，其中不变性和版本历史是一等公民。

#### 我们要解决什么问题？

在传统的栈中，每次操作都会改变（mutate）结构，旧版本会丢失。

持久化栈通过允许以下功能来解决这个问题：
- 随时访问之前的状态
- 撤销或时间旅行功能
- 纯粹的函数式程序，其中数据永不改变

应用于编译器、回溯系统和函数式编程语言。

#### 工作原理（通俗解释）

栈是一个链表：
- `push(x)` 添加一个新的头节点
- `pop()` 返回下一个节点

为了实现持久化，我们从不修改节点。相反，每次操作都会创建一个新的头节点，指向现有的尾部。

| 版本 | 操作 | 顶部元素 | 结构 |
| ------- | --------- | ----------- | ----------- |
| v0 | 空 |, | ∅ |
| v1 | push(10) | 10 | 10 → ∅ |
| v2 | push(20) | 20 | 20 → 10 → ∅ |
| v3 | pop() | 10 | 10 → ∅ |

每个版本都复用之前的节点，无需复制数据。

#### 示例

从一个空栈 `v0` 开始：

1. `v1 = push(v0, 10)` → 栈 `[10]`
2. `v2 = push(v1, 20)` → 栈 `[20, 10]`
3. `v3 = pop(v2)` → 返回 20，新栈 `[10]`

现在我们有了三个可访问的版本：

```
v0: ∅
v1: 10
v2: 20 → 10
v3: 10
```

#### 微型代码（Python）

```python
class Node:
    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node

class PersistentStack:
    def __init__(self, top=None):
        self.top = top

    def push(self, value):
        # 新节点指向当前栈顶
        return PersistentStack(Node(value, self.top))

    def pop(self):
        if not self.top:
            return self, None
        return PersistentStack(self.top.next), self.top.value

    def peek(self):
        return None if not self.top else self.top.value

# 示例
v0 = PersistentStack()
v1 = v0.push(10)
v2 = v1.push(20)
v3, popped = v2.pop()

print(v2.peek())  # 20
print(v3.peek())  # 10
```

这种方法复用节点，在不改变原有版本的情况下创建新版本。

#### 微型代码（C，概念性）

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int value;
    struct Node* next;
} Node;

typedef struct {
    Node* top;
} Stack;

Stack* push(Stack* s, int value) {
    Node* node = malloc(sizeof(Node));
    node->value = value;
    node->next = s->top;
    Stack* new_stack = malloc(sizeof(Stack));
    new_stack->top = node;
    return new_stack;
}

Stack* pop(Stack* s, int* popped_value) {
    if (!s->top) return s;
    *popped_value = s->top->value;
    Stack* new_stack = malloc(sizeof(Stack));
    new_stack->top = s->top->next;
    return new_stack;
}
```

每次 `push` 或 `pop` 都会创建一个新的 `Stack*`，指向之前的结构。

#### 为什么重要

- 不变性确保了数据安全性和并发友好的设计
- 版本化允许回溯、撤销或分支计算
- 是函数式编程和持久化数据存储的基础

#### 一个温和的证明（为什么可行）

栈的每个版本都与之前的版本共享未改变的节点。因为 `push` 和 `pop` 只修改头节点的引用，所以旧版本保持完整。

如果 $n$ 是操作总数：
- 每个新版本增加 $O(1)$ 的空间
- 共享尾部确保总空间 = $O(n)$

数学上：

$$
S_n = S_{n-1} + O(1)
$$

并且旧版本永远不会被覆盖，确保了持久性。

#### 动手试试

1. 用值 [1, 2, 3] 构建一个持久化栈
2. 执行一次 pop，并确认早期版本仍然保留其值
3. 与可变栈实现进行比较
4. 可视化不同版本之间共享的链表节点

#### 测试用例

| 操作 | 结果 | 备注 |
| ------------------- | ---------------- | ------------------------ |
| `v1 = push(v0, 10)` | `[10]` | 新版本 |
| `v2 = push(v1, 20)` | `[20, 10]` | 共享 10 节点 |
| `v3, val = pop(v2)` | val = 20, `[10]` | 旧的 v2 保持原样 |
| `v1.peek()` | `10` | 不受后续 pop 影响 |

#### 复杂度

| 操作 | 时间 | 空间 | 备注 |
| ------------------ | ------ | ------ | --------------- |
| Push | $O(1)$ | $O(1)$ | 新头节点 |
| Pop | $O(1)$ | $O(1)$ | 新的顶部指针 |
| 访问旧版本 | $O(1)$ |, | 存储引用 |

持久化栈优雅地结合了不变性、共享和时间旅行，是迈向函数式数据结构世界的一个小而强大的步骤。
### 282 持久化数组

持久化数组是一种不可变的、带版本号的数据结构，允许访问所有历史状态。每次更新不是覆盖元素，而是创建一个新版本，该版本与先前版本共享大部分结构。

这使得"时间旅行"成为可能，可以在常数或对数时间内查看或恢复任何早期版本，而无需复制整个数组。

#### 我们解决什么问题？

普通数组是可变的，每次 `arr[i] = x` 都会破坏旧值。如果我们想要历史记录、撤销或分支计算，这是不可接受的。

持久化数组保留所有版本：

| 版本 | 操作         | 状态                     |
| ---- | ------------ | ------------------------ |
| v0   | `[]`         | 空                       |
| v1   | `set(0, 10)` | `[10]`                   |
| v2   | `set(0, 20)` | `[20]` (v1 仍为 `[10]`) |

每个版本都重用数组中未修改的部分，避免了完全复制。

#### 工作原理（通俗解释）

持久化数组可以使用写时复制或基于树的结构来实现。

1.  **写时复制（小数组）**
    *   仅当元素更改时才创建新的数组副本。
    *   简单，但更新成本为 $O(n)$。

2.  **基于树的路径复制（大数组）**
    *   将数组表示为平衡二叉树（如线段树）。
    *   每次更新仅复制到被更改叶节点的路径。
    *   每次更新的空间 = $O(\log n)$

因此，每个版本都指向一个根节点。当修改索引 $i$ 时，会创建一条通向树底部的新路径，而未触及的子树则被共享。

#### 示例

让我们构建一个大小为 4 的持久化数组。

#### 步骤 1：初始版本

```
v0 = [0, 0, 0, 0]
```

#### 步骤 2：更新索引 2

```
v1 = set(v0, 2, 5)  → [0, 0, 5, 0]
```

#### 步骤 3：更新索引 1

```
v2 = set(v1, 1, 9)  → [0, 9, 5, 0]
```

`v0`、`v1` 和 `v2` 都独立共存。

#### 逐步示例（树形表示）

每个节点覆盖一个范围：

```
根节点: [0..3]
  ├── 左子树 [0..1]
  │     ├── [0] → 0
  │     └── [1] → 9
  └── 右子树 [2..3]
        ├── [2] → 5
        └── [3] → 0
```

更新索引 1 仅复制路径 `[0..3] → [0..1] → [1]`，而不是整棵树。

#### 微型代码（Python，基于树）

```python
class Node:
    def __init__(self, left=None, right=None, value=0):
        self.left = left
        self.right = right
        self.value = value

def build(l, r):
    if l == r:
        return Node()
    m = (l + r) // 2
    return Node(build(l, m), build(m+1, r))

def update(node, l, r, idx, val):
    if l == r:
        return Node(value=val)
    m = (l + r) // 2
    if idx <= m:
        return Node(update(node.left, l, m, idx, val), node.right)
    else:
        return Node(node.left, update(node.right, m+1, r, idx, val))

def query(node, l, r, idx):
    if l == r:
        return node.value
    m = (l + r) // 2
    return query(node.left, l, m, idx) if idx <= m else query(node.right, m+1, r, idx)

# 示例
n = 4
v0 = build(0, n-1)
v1 = update(v0, 0, n-1, 2, 5)
v2 = update(v1, 0, n-1, 1, 9)

print(query(v2, 0, n-1, 2))  # 5
print(query(v1, 0, n-1, 1))  # 0
```

每次更新都会创建一个新的版本根节点。

#### 为什么重要

-   时间旅行调试：检索旧状态
-   编辑器中的撤销/重做系统
-   持久化算法中的分支计算
-   无需可变性的函数式编程

#### 一个温和的证明（为什么有效）

设 $n$ 为数组大小，$u$ 为更新次数。

每次更新复制 $O(\log n)$ 个节点。
因此总空间为：

$$
O(n + u \log n)
$$

每次查询遍历一条路径 $O(\log n)$。
没有版本会使另一个版本失效，所有根节点都保持可访问。

持久性得以保持，因为我们从不修改现有节点，只分配新节点并重用子树。

#### 自己动手试试

1.  构建一个大小为 8 的数组，全部初始化为 0。
2.  创建 v1 = 设置索引 4 → 7
3.  创建 v2 = 设置索引 2 → 9
4.  打印 v0、v1、v2 中的值
5.  确认旧版本保持不变。

#### 测试用例

| 操作         | 输入       | 输出         | 备注       |
| ------------ | ---------- | ------------ | ---------- |
| build(4)     | `[0,0,0,0]` | v0           | 基础版本   |
| set(v0,2,5)  |            | `[0,0,5,0]`  | 新版本     |
| set(v1,1,9)  |            | `[0,9,5,0]`  | 重用 v1    |

#### 复杂度

| 操作             | 时间        | 空间        | 备注           |
| ---------------- | ----------- | ----------- | -------------- |
| 构建             | $O(n)$      | $O(n)$      | 初始化         |
| 更新             | $O(\log n)$ | $O(\log n)$ | 路径复制       |
| 查询             | $O(\log n)$ |,           | 一条路径       |
| 访问旧版本       | $O(1)$      |,           | 根节点引用     |

持久化数组将短暂的内存转变为带版本的时间线，每次更改都是一个分支，每个版本都是永恒的。非常适合函数式编程、调试和算法历史记录。
### 283 可持久化线段树

可持久化线段树是一种支持版本管理的数据结构，它能够进行区间查询和单点更新，同时保留所有历史版本以供访问。

它巧妙地结合了线段树和可持久化技术，使你能够查询历史状态、执行撤销操作，甚至高效地比较过去和现在的结果。

#### 我们要解决什么问题？

标准的线段树支持：

- 单点更新：`arr[i] = x`
- 区间查询：`sum(l, r)` 或 `min(l, r)`

但每次更新都会覆盖旧值。可持久化线段树通过每次更新时创建一个新版本，并复用未更改的节点来解决这个问题。

| 版本 | 操作         | 状态             |
| ---- | ------------ | ---------------- |
| v0   | 构建         | `[1, 2, 3, 4]`   |
| v1   | update(2, 5) | `[1, 2, 5, 4]`   |
| v2   | update(1, 7) | `[1, 7, 5, 4]`   |

现在你可以查询任何版本：

- `query(v0, 1, 3)` → 旧的区间和
- `query(v2, 1, 3)` → 更新后的区间和

#### 工作原理（通俗解释）

线段树是一种二叉树，其中每个节点存储一个区间上的聚合值（如和、最小值、最大值）。

可持久化通过路径复制实现：

- 当更新索引 `i` 时，只有从根节点到叶子节点的路径上的节点会被替换。
- 所有其他节点在不同版本间共享。

因此，每个新版本需要 $O(\log n)$ 个新节点和空间。

| 步骤 | 操作         | 受影响的节点        |
| ---- | ------------ | ------------------- |
| 1    | 构建         | $O(n)$ 个节点       |
| 2    | Update(2, 5) | $O(\log n)$ 个新节点 |
| 3    | Update(1, 7) | $O(\log n)$ 个新节点 |

#### 示例

假设初始数组为 `[1, 2, 3, 4]`

1.  构建树 (v0)
2.  v1 = update(v0, 2 → 5)
3.  v2 = update(v1, 1 → 7)

现在：

- `query(v0, 1, 4) = 10`
- `query(v1, 1, 4) = 12`
- `query(v2, 1, 4) = 17`

所有版本共享大部分节点，节省了内存。

#### 逐步示例

#### 在索引 2 处更新 v0 → v1

| 版本 | 复制的树节点               | 共享部分           |
| ---- | -------------------------- | ------------------ |
| v1   | 路径 [根节点 → 左子节点 → 右子节点] | 其他节点保持不变 |

所以 v1 仅沿一条路径与 v0 不同。

#### 精简代码 (Python)

```python
class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build(arr, l, r):
    if l == r:
        return Node(arr[l])
    m = (l + r) // 2
    left = build(arr, l, m)
    right = build(arr, m + 1, r)
    return Node(left.val + right.val, left, right)

def update(node, l, r, idx, val):
    if l == r:
        return Node(val)
    m = (l + r) // 2
    if idx <= m:
        left = update(node.left, l, m, idx, val)
        return Node(left.val + node.right.val, left, node.right)
    else:
        right = update(node.right, m + 1, r, idx, val)
        return Node(node.left.val + right.val, node.left, right)

def query(node, l, r, ql, qr):
    if qr < l or ql > r:
        return 0
    if ql <= l and r <= qr:
        return node.val
    m = (l + r) // 2
    return query(node.left, l, m, ql, qr) + query(node.right, m + 1, r, ql, qr)

# 示例
arr = [1, 2, 3, 4]
v0 = build(arr, 0, 3)
v1 = update(v0, 0, 3, 2, 5)
v2 = update(v1, 0, 3, 1, 7)

print(query(v0, 0, 3, 0, 3))  # 10
print(query(v1, 0, 3, 0, 3))  # 12
print(query(v2, 0, 3, 0, 3))  # 17
```

#### 为什么它很重要

- 即时访问任何版本
- 支持“时间旅行”查询
- 支持不可变数据分析
- 应用于离线查询、算法竞赛和函数式数据库

#### 一个温和的证明（为什么它有效）

每次更新只修改 $O(\log n)$ 个节点。
所有其他子树是共享的，因此总空间为：

$$
S(u) = O(n + u \log n)
$$

查询任何版本的时间复杂度为 $O(\log n)$，因为只需要遍历一条路径。

可持久化得以保持，因为没有节点被修改，只有被替换。

#### 自己动手试试

1.  构建 `[1, 2, 3, 4]`
2.  更新索引 2 → 5 (v1)
3.  更新索引 1 → 7 (v2)
4.  查询 v0, v1, v2 中的 sum(1, 4)
5.  通过可视化验证共享的子树

#### 测试用例

| 版本 | 操作        | Query(0,3) | 说明             |
| ---- | ----------- | ---------- | ---------------- |
| v0   | `[1,2,3,4]` | 10         | 基础版本         |
| v1   | set(2,5)    | 12         | 更改了一个叶子节点 |
| v2   | set(1,7)    | 17         | 另一次更新       |

#### 复杂度

| 操作         | 时间        | 空间        | 说明     |
| ------------ | ----------- | ----------- | -------- |
| 构建         | $O(n)$      | $O(n)$      | 完整树   |
| 更新         | $O(\log n)$ | $O(\log n)$ | 路径复制 |
| 查询         | $O(\log n)$ |,           | 一条路径 |
| 版本访问     | $O(1)$      |,           | 通过根节点 |

可持久化线段树是你不可变的预言家，每个版本都是时间的一个快照，永远可查询，永远完好无损。
### 284 持久化链表

持久化链表是经典单向链表的版本化变体，其中每次插入或删除操作都会产生一个新版本，而不会破坏旧版本。

每个版本代表链表的一个不同状态，所有版本通过共享结构共存：未更改的节点被重用，只有被修改的路径会被复制。

这种技术是函数式编程、撤销系统和不可变数据结构的核心。

#### 我们要解决什么问题？

一个可变的链表在每次更改后都会丢失其历史记录。

通过持久化，我们保留了所有过去的版本：

| 版本 | 操作           | 链表      |
| ---- | -------------- | --------- |
| v0   | 空             | ∅         |
| v1   | push_front(10) | [10]      |
| v2   | push_front(20) | [20, 10]  |
| v3   | pop_front()    | [10]      |

每个版本都是一等公民，你可以随时遍历、查询或比较任何版本。

#### 工作原理（通俗解释）

单向链表中的每个节点包含：
- `value`（值）
- `next` 指针

为了实现持久化，我们从不修改节点。相反，操作返回一个新的头节点：
- `push_front(x)`：创建一个新节点 `n = Node(x, old_head)`
- `pop_front()`：返回 `old_head.next` 作为新的头节点

所有旧节点保持原样并被共享。

| 操作           | 新节点           | 共享结构           |
| -------------- | ---------------- | ------------------ |
| push_front(20) | 新的头节点       | 尾部被重用         |
| pop_front()    | 新的头节点(next) | 旧的头节点仍然存在 |

#### 示例

#### 逐步版本化

1.  `v0 = []`
2.  `v1 = push_front(v0, 10)` → `[10]`
3.  `v2 = push_front(v1, 20)` → `[20, 10]`
4.  `v3 = pop_front(v2)` → `[10]`

版本：

```
v0: ∅
v1: 10
v2: 20 → 10
v3: 10
```

所有版本共存并共享结构。

#### 示例（图形视图）

```
v0: ∅
v1: 10 → ∅
v2: 20 → 10 → ∅
v3: 10 → ∅
```

注意：`v2.tail` 是从 `v1` 重用的。

#### 微型代码（Python）

```python
class Node:
    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node

class PersistentList:
    def __init__(self, head=None):
        self.head = head

    def push_front(self, value):
        new_node = Node(value, self.head)
        return PersistentList(new_node)

    def pop_front(self):
        if not self.head:
            return self, None
        return PersistentList(self.head.next), self.head.value

    def to_list(self):
        result, curr = [], self.head
        while curr:
            result.append(curr.value)
            curr = curr.next
        return result

# 示例
v0 = PersistentList()
v1 = v0.push_front(10)
v2 = v1.push_front(20)
v3, popped = v2.pop_front()

print(v1.to_list())  # [10]
print(v2.to_list())  # [20, 10]
print(v3.to_list())  # [10]
```

#### 微型代码（C，概念性）

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int value;
    struct Node* next;
} Node;

typedef struct {
    Node* head;
} PList;

PList push_front(PList list, int value) {
    Node* new_node = malloc(sizeof(Node));
    new_node->value = value;
    new_node->next = list.head;
    PList new_list = { new_node };
    return new_list;
}

PList pop_front(PList list, int* popped) {
    if (!list.head) return list;
    *popped = list.head->value;
    PList new_list = { list.head->next };
    return new_list;
}
```

没有修改操作，只分配新节点。

#### 为什么这很重要

- **不可变性**，非常适合函数式程序
- **撤销/时间旅行**，可以重新访问旧版本
- **安全的并发性**，没有数据竞争
- **内存高效**，尾部共享重用旧结构

#### 一个温和的证明（为什么它有效）

设 $n$ 为操作次数。
每次 `push_front` 或 `pop_front` 最多创建一个新节点。

因此：
- $n$ 次操作后的总空间：$O(n)$
- 每次操作的时间：$O(1)$

持久性得到保证，因为没有节点被原地修改。

所有版本共享未更改的后缀：

$$
v_k = \text{Node}(x_k, v_{k-1})
$$

因此，结构共享是线性的且安全的。

#### 自己动手试试

1.  从空链表开始
2.  Push 3 → Push 2 → Push 1
3.  Pop 一次
4.  打印所有版本
5.  观察尾部是如何被共享的

#### 测试用例

| 操作        | 输入 | 输出      | 版本 |
| ----------- | ---- | --------- | ---- |
| push_front  | 10   | [10]      | v1   |
| push_front  | 20   | [20, 10]  | v2   |
| pop_front   |,    | [10]      | v3   |
| to_list     | v1   | [10]      |,    |

#### 复杂度

| 操作        | 时间     | 空间     | 注释                 |
| ----------- | -------- | -------- | -------------------- |
| push_front  | $O(1)$   | $O(1)$   | 仅新头节点           |
| pop_front   | $O(1)$   | $O(1)$   | 重用 next            |
| 访问        | $O(n)$   |,        | 与链表相同           |

持久化链表是通往持久化的最简单途径，每次操作都是 $O(1)$，每个版本都是不朽的。
它是函数式栈、队列和不可变集合的支柱。
### 286 手指树

手指树是一种多用途、持久化的数据结构，它提供了在两端（前端和后端）的均摊 $O(1)$ 访问，以及在中间部分的 $O(\log n)$ 访问或更新。

它是一种函数式、不可变的序列结构，是一种平衡树，并增加了指向其两端的*手指*（快速访问点）。手指树构成了许多持久化数据类型的基础，例如队列、双端队列、优先级序列，甚至是类似绳索（rope）的文本编辑器。

#### 我们要解决什么问题？

不可变列表在前端操作很快，但在后端操作很慢。不可变数组则相反。支持持久化的双端队列很难高效地维护。

我们想要：

- $O(1)$ 的前端访问
- $O(1)$ 的后端访问
- $O(\log n)$ 的中间访问
- 持久化和不可变性

手指树通过在边缘使用浅层的数字缓冲区，并在中间使用平衡节点来实现这一点。

#### 工作原理（通俗解释）

手指树是递归构建的：

```
FingerTree = Empty
           | Single(a)
           | Deep(prefix, deeper_tree, suffix)
```

- prefix 和 suffix：包含 1-4 个元素的小数组（数字）
- deeper_tree：递归地保存更高层级的节点

*手指*（prefix/suffix）提供了对两端的常数时间访问。
插入操作将元素推入数字缓冲区；当缓冲区满时，元素会“滚入”更深层的树中。

#### 示例

插入 1, 2, 3, 4, 5：

```
Deep [1,2] (Node [3,4]) [5]
```

你可以：

- 前端插入 → 将元素添加到 prefix
- 后端插入 → 将元素追加到 suffix
- 以 O(1) 访问两端
- 中间插入 → 递归进入更深层的树（O(log n)）

每个操作都会返回一个新版本，该版本共享未改变的子树。

#### 示例状态

| 操作           | 结构                     | 说明             |
| -------------- | ------------------------ | ---------------- |
| empty          | `Empty`                  | 基础状态         |
| push_front(1)  | `Single(1)`              | 一个元素         |
| push_front(2)  | `Deep [2] Empty [1]`     | 两端             |
| push_back(3)   | `Deep [2] Empty [1,3]`   | 添加后缀         |
| push_back(4)   | `Deep [2,4] Empty [1,3]` | 平衡增长         |

每个版本都重用了其大部分结构，确保了持久化。

#### 微型代码（Python - 概念性）

这是一个简化的模型，并非完整实现（真正的手指树依赖于更多类型层面的机制）。

```python
class Empty:
    pass

class Single:
    def __init__(self, value):
        self.value = value

class Deep:
    def __init__(self, prefix, middle, suffix):
        self.prefix = prefix
        self.middle = middle
        self.suffix = suffix

def push_front(tree, x):
    if isinstance(tree, Empty):
        return Single(x)
    if isinstance(tree, Single):
        return Deep([x], Empty(), [tree.value])
    if len(tree.prefix) < 4:
        return Deep([x] + tree.prefix, tree.middle, tree.suffix)
    else:
        # 将 prefix 滚入更深层的树
        new_middle = push_front(tree.middle, tree.prefix)
        return Deep([x], new_middle, tree.suffix)

def to_list(tree):
    if isinstance(tree, Empty):
        return []
    if isinstance(tree, Single):
        return [tree.value]
    return tree.prefix + to_list(tree.middle) + tree.suffix
```

这段代码捕捉了核心的递归风格、常数时间的手指操作和对数递归。

#### 为什么它很重要

- 序列的通用框架
- 在两端插入/删除的均摊 $O(1)$ 时间复杂度
- $O(\log n)$ 的连接、分割或搜索
- 作为以下结构的基础：

  * 函数式双端队列
  * 优先级队列
  * 有序序列（如 RRB 树）
  * 增量式编辑器

#### 一个温和的证明（为什么它有效）

数字缓冲区最多存储 4 个元素，保证了有界开销。
每个递归步骤都将规模减少一个常数因子，确保深度为 $O(\log n)$。

对于每个操作：

- 在 $O(1)$ 时间内触及两端
- 在 $O(\log n)$ 深度进行结构更改

因此，总成本为：

$$
T(n) = O(\log n), \quad \text{在两端为均摊 O(1)}
$$

持久化得以保证，因为所有更新都创建新节点而不修改现有节点。

#### 自己动手试试

1.  从空树开始。
2.  插入 1, 2, 3, 4。
3.  弹出前端元素，观察结构。
4.  插入 5，检查版本之间的共享情况。
5.  将每个版本转换为列表，比较结果。

#### 测试用例

| 操作           | 结果      | 说明             |
| -------------- | --------- | ---------------- |
| push_front(1)  | [1]       | 基础             |
| push_front(2)  | [2, 1]    | prefix 增长      |
| push_back(3)   | [2, 1, 3] | suffix 添加      |
| pop_front()    | [1, 3]    | 从 prefix 中移除 |

#### 复杂度

| 操作            | 时间               | 空间       | 说明             |
| --------------- | ------------------ | ---------- | ---------------- |
| push_front/back | $O(1)$ 均摊        | $O(1)$     | 使用数字缓冲区   |
| pop_front/back  | $O(1)$ 均摊        | $O(1)$     | 常数时间手指操作 |
| 随机访问        | $O(\log n)$        | $O(\log n)$ | 递归             |
| 连接/分割       | $O(\log n)$        | $O(\log n)$ | 高效分割         |

手指树是持久化序列的瑞士军刀，两端操作快，内部平衡，并且完美地不可变。它是无数函数式数据结构背后的蓝图。
### 287 拉链结构

拉链是一种强大的技术，它能让不可变数据结构表现得像可变数据结构一样。它在持久化结构（列表、树等）内部提供一个焦点——类似指针的位置——允许进行局部更新、导航和编辑，而无需进行修改。

可以把它看作是纯函数式世界中的一个光标。每一次移动或编辑都会产生一个新版本，同时共享未修改的部分。

#### 我们要解决什么问题？

不可变数据结构不能“就地修改”。如果不重建整个结构，你就不能简单地“移动光标”或“替换元素”。

拉链通过维护以下内容来解决这个问题：

1.  焦点元素，以及
2.  上下文（左边/右边或上面/下面有什么）。

然后，你就可以高效地移动、更新或重建，并重用其他所有部分。

#### 工作原理（通俗解释）

拉链将一个结构分离为：

-   焦点：当前被关注的元素
-   上下文：你所走路径的可逆描述

当你移动焦点时，你更新上下文。
当你改变焦点元素时，你创建一个新节点并从上下文重建。

对于列表：

```
拉链 = (左部, 焦点, 右部)
```

对于树：

```
拉链 = (父节点上下文, 焦点节点)
```

你可以把它想象成图灵机中的一条纸带——左边和右边的一切都被保留。

#### 示例（列表拉链）

我们用一个光标在 `c` 上的列表 `[a, b, c, d]` 来表示：

```
左部: [b, a]   焦点: c   右部: [d]
```

从这里开始：

-   `move_left` → 焦点 = `b`
-   `move_right` → 焦点 = `d`
-   `update(x)` → 用 `x` 替换 `c`

所有这些操作都是 O(1) 时间复杂度，并返回一个新的拉链版本。

#### 示例操作

| 操作                     | 结果               | 描述           |
| ------------------------ | ------------------ | -------------- |
| `from_list([a,b,c,d])`   | ([ ], a, [b,c,d])  | 在头部初始化   |
| `move_right`             | ([a], b, [c,d])    | 焦点右移       |
| `update('X')`            | ([a], X, [c,d])    | 替换焦点       |
| `to_list`                | [a,X,c,d]          | 重建完整列表   |

#### 微型代码（Python – 列表拉链）

```python
class Zipper:
    def __init__(self, left=None, focus=None, right=None):
        self.left = left or []
        self.focus = focus
        self.right = right or []

    @staticmethod
    def from_list(lst):
        if not lst: return Zipper([], None, [])
        return Zipper([], lst[0], lst[1:])

    def move_left(self):
        if not self.left: return self
        return Zipper(self.left[:-1], self.left[-1], [self.focus] + self.right)

    def move_right(self):
        if not self.right: return self
        return Zipper(self.left + [self.focus], self.right[0], self.right[1:])

    def update(self, value):
        return Zipper(self.left, value, self.right)

    def to_list(self):
        return self.left + [self.focus] + self.right

# 示例
z = Zipper.from_list(['a', 'b', 'c', 'd'])
z1 = z.move_right().move_right()   # 焦点在 'c' 上
z2 = z1.update('X')
print(z2.to_list())  # ['a', 'b', 'X', 'd']
```

每个操作都返回一个新的拉链，使得持久化编辑变得简单。

#### 示例（树拉链 – 概念性）

树拉链将到达根节点的路径作为上下文存储：

```
拉链 = (父节点路径, 焦点节点)
```

路径中的每个父节点都记得你来自哪一侧，这样你就可以在编辑后向上重建。

例如，编辑叶子节点 `L` 会创建一个新的 `L'`，并且只重建路径上的节点，其他子树保持不变。

#### 为什么它很重要

-   支持在不可变结构中进行局部更新
-   用于函数式编辑器、解析器、导航系统
-   提供 O(1) 的局部移动，O(深度) 的重建
-   Huet 拉链中的核心概念，函数式编程的基础思想之一

#### 一个温和的证明（为什么它有效）

每次移动或编辑只影响局部上下文：

-   在列表中向左/右移动 → $O(1)$
-   在树中向上/下移动 → $O(1)$
-   重建完整结构 → $O(\text{深度})$

没有发生修改；每个版本都重用所有未触及的子结构。

形式上，如果 $S$ 是原始结构，$f$ 是焦点，
$$
\text{zip}(S) = (\text{context}, f)
$$
并且
$$
\text{unzip}(\text{context}, f) = S
$$
确保了可逆性。

#### 自己动手试试

1.  从 `[1,2,3,4]` 创建一个拉链
2.  将焦点移动到 3
3.  将焦点更新为 99
4.  重建完整列表
5.  验证旧的拉链仍然具有旧值

#### 测试用例

| 步骤 | 操作                     | 结果               | 备注           |
| ---- | ------------------------ | ------------------ | -------------- |
| 1    | `from_list([a,b,c,d])`   | ([ ], a, [b,c,d])  | 初始化         |
| 2    | `move_right()`           | ([a], b, [c,d])    | 移动焦点       |
| 3    | `update('X')`            | ([a], X, [c,d])    | 编辑           |
| 4    | `to_list()`              | [a, X, c, d]       | 重建           |

#### 复杂度

| 操作               | 时间     | 空间     | 备注               |
| ------------------ | -------- | -------- | ------------------ |
| 向左/右移动        | $O(1)$   | $O(1)$   | 移动焦点           |
| 更新               | $O(1)$   | $O(1)$   | 局部替换           |
| 重建               | $O(n)$   | $O(1)$   | 当解压时           |
| 访问旧版本         | $O(1)$   |,         | 持久化             |

拉链将不可变性转化为交互性。有了拉链，你可以移动、聚焦和编辑，所有这些都不会破坏持久性。它是静态结构和动态导航之间的桥梁。
### 289 支持版本化的字典树

支持版本化的字典树是一种持久化数据结构，用于跨多个历史版本存储字符串（或序列）。每次新的更新——插入、删除或修改——都会创建一个新的字典树版本，而不会改变之前的版本，这是通过路径复制实现结构共享的。

这使得时间旅行查询成为可能：你可以查找历史上任意时间点存在的键。

#### 我们正在解决什么问题？

我们希望维护一个支持版本化的字符串或序列字典，支持：

- 快速前缀搜索（$O(\text{长度})$）
- 无需修改原结构的高效更新
- 访问历史版本（例如，快照、撤销/重做、历史记录）

版本化字典树通过仅复制从根节点到被修改节点的路径，同时共享所有其他子树，实现了以上三点。

常见用例：

- 版本化的符号表
- 历史字典
- 支持回滚的自动补全
- 函数式语言中的持久化字典树

#### 工作原理（通俗解释）

每个字典树节点包含：

- 一个从字符到子节点的映射
- 一个标记单词结束的标志位

为了实现持久化：

- 当你插入或删除时，仅复制受影响路径上的节点。
- 旧节点保持不变，并被旧版本共享。

因此，版本 $v_{k+1}$ 与 $v_k$ 仅在修改路径上有所不同。

#### 微型代码（概念性 Python 实现）

```python
class TrieNode:
    def __init__(self, children=None, is_end=False):
        self.children = dict(children or {})
        self.is_end = is_end

def insert(root, word):
    def _insert(node, i):
        node = TrieNode(node.children, node.is_end)
        if i == len(word):
            node.is_end = True
            return node
        ch = word[i]
        node.children[ch] = _insert(node.children.get(ch, TrieNode()), i + 1)
        return node
    return _insert(root, 0)

def search(root, word):
    node = root
    for ch in word:
        if ch not in node.children:
            return False
        node = node.children[ch]
    return node.is_end
```

每次调用 `insert` 都会返回一个新的根节点（新版本），共享所有未修改的分支。

#### 示例

| 版本   | 操作             | 字典树内容                 |
| ------ | ---------------- | -------------------------- |
| v1     | insert("cat")    | { "cat" }                  |
| v2     | insert("car")    | { "cat", "car" }           |
| v3     | insert("dog")    | { "cat", "car", "dog" }    |
| v4     | delete("car")    | { "cat", "dog" }           |

在所有早期版本之间，版本共享前缀 `"c"` 的节点。

#### 为何重要

- **不可变且安全**：没有原地修改，非常适合函数式系统
- **高效回滚**：以 $O(1)$ 时间复杂度访问任何先前版本
- **前缀共享**：通过结构重用节省内存
- **历史记录实用**：非常适合版本化字典、IDE、搜索索引

#### 一个温和的证明（为何有效）

每次插入复制一条长度为 $L$（单词长度）的路径。
总时间复杂度：
$$
T_{\text{插入}} = O(L)
$$

新路径上的每个节点共享所有其他未更改的子树。
如果 $N$ 是所有版本中存储的总字符数，
$$
\text{空间} = O(N)
$$

每个版本的根节点是一个单独的指针，支持 $O(1)$ 访问：
$$
\text{版本}_i = \text{根节点}_i
$$

旧版本保持完全可用，因为它们永远不会被修改。

#### 亲自尝试

1.  将 "cat", "car", "dog" 插入版本化字典树
2.  删除 "car" 以形成新版本
3.  在所有版本中查询前缀 "ca"
4.  检查 "car" 是否仅在删除前存在
5.  打印跨版本的共享节点数量
### 测试用例

| 步骤 | 操作          | 版本 | 存在于版本中       | 结果   |
| ---- | ------------- | ---- | ------------------ | ------ |
| 1    | insert("cat") | v1   | cat                | True   |
| 2    | insert("car") | v2   | car, cat           | True   |
| 3    | insert("dog") | v3   | cat, car, dog      | True   |
| 4    | delete("car") | v4   | car (否), cat, dog | False  |

#### 复杂度

| 操作               | 时间复杂度 | 空间复杂度 | 备注                     |
| ------------------ | ---------- | ---------- | ------------------------ |
| 插入               | $O(L)$     | $O(L)$     | 路径复制                 |
| 删除               | $O(L)$     | $O(L)$     | 路径复制                 |
| 查找               | $O(L)$     | $O(1)$     | 遵循共享结构             |
| 访问旧版本         | $O(1)$     |,          | 版本指针访问             |

支持版本控制的字典树融合了结构共享与前缀索引。每个版本都是一个冻结的快照——紧凑、可查询且不可变——非常适合用于版本化的单词历史记录。
### 290 可持久化并查集

可持久化并查集扩展了经典的不相交集合并（DSU）结构，以支持*时间旅行查询*。它不会直接修改父节点数组和秩数组，而是每次合并操作都会产生一个新版本，从而支持诸如以下的查询：

- “在版本 $v$ 中，$x$ 和 $y$ 是否连通？”
- “在最后一次合并之前，集合是什么样子的？”

这种结构对于需要关注合并历史的动态连通性问题至关重要。

#### 我们要解决什么问题？

经典的 DSU 能以接近常数的时间复杂度高效地支持 `find` 和 `union` 操作，但仅限于单个不断演化的状态。一旦合并了两个集合，旧版本就消失了。

我们需要一个版本化的 DSU，它能保持所有先前状态不变，并支持：

- 操作的撤销/回滚
- 对过去连通性的查询
- 离线动态连通性分析

#### 工作原理（通俗解释）

可持久化并查集使用路径复制（类似于可持久化数组）来维护多个版本：

- 每次 `union` 操作创建一个新版本
- 只有受影响的父节点和秩条目会在新结构中被更新
- 所有其他节点与先前版本共享结构

主要有两种设计：

1. 使用函数式风格的路径复制实现完全持久化
2. 使用回滚栈（撤销操作）实现部分持久化

我们这里重点关注*完全持久化*。

#### 微型代码（概念性 Python）

```python
class PersistentDSU:
    def __init__(self, n):
        self.versions = []
        parent = list(range(n))
        rank = [0] * n
        self.versions.append((parent, rank))
    
    def find(self, parent, x):
        if parent[x] == x:
            return x
        return self.find(parent, parent[x])
    
    def union(self, ver, x, y):
        parent, rank = [*ver[0]], [*ver[1]]  # 复制数组
        rx, ry = self.find(parent, x), self.find(parent, y)
        if rx != ry:
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
        self.versions.append((parent, rank))
        return len(self.versions) - 1  # 新版本索引
    
    def connected(self, ver, x, y):
        parent, _ = self.versions[ver]
        return self.find(parent, x) == self.find(parent, y)
```

每次合并操作返回一个新的版本索引。你可以随时查询 `connected(version, a, b)`。

#### 示例

| 步骤 | 操作               | 版本  | 连通关系               |
| ---- | ------------------ | ----- | ---------------------- |
| 1    | make-set(5)        | v0    | 0 1 2 3 4              |
| 2    | union(0,1)         | v1    | {0,1}, 2, 3, 4         |
| 3    | union(2,3)         | v2    | {0,1}, {2,3}, 4        |
| 4    | union(1,2)         | v3    | {0,1,2,3}, 4           |
| 5    | query connected(0,1) | v1  | True                   |
| 6    | query connected(1,3) | v1  | False (尚未合并)       |

你可以在任何版本检查连通性。

#### 为什么重要

- 跨历史版本的时间旅行查询
- 非破坏性更新允许安全回滚
- 对于离线动态连通性（例如，随时间推移的边插入）至关重要
- 简化了调试、模拟和版本跟踪

#### 一个温和的证明（为什么它有效）

设 $n$ 为元素数量，$q$ 为版本数量。

每个版本仅在少数父节点/秩条目上存在差异。
如果每次 `union` 复制 $O(\alpha(n))$ 个元素，则 $q$ 次操作后的总空间为：

$$
O(n + q \alpha(n))
$$

每次查询在固定版本上操作，时间复杂度为：

$$
O(\alpha(n))
$$

路径压缩通常被*部分压缩*替代或省略，以确保持久性（完全的路径压缩会破坏不可变性）。

#### 动手试试

1.  用 5 个元素初始化 DSU
2.  逐步执行合并操作，保存每个版本
3.  跨多个版本查询连通性
4.  通过恢复到旧版本来撤销合并
5.  可视化父节点树的演变

#### 测试用例

| 版本  | 查询             | 结果   |
| ----- | ---------------- | ------ |
| v0    | connected(0,1)   | False  |
| v1    | connected(0,1)   | True   |
| v2    | connected(2,3)   | True   |
| v3    | connected(1,3)   | True   |
| v1    | connected(1,3)   | False  |

#### 复杂度

| 操作             | 时间复杂度     | 空间复杂度   | 备注                     |
| ---------------- | -------------- | ------------ | ------------------------ |
| Find             | $O(\alpha(n))$ | $O(1)$       | 每个版本                 |
| Union            | $O(\alpha(n))$ | $O(n)$ 复制  | 写时复制路径             |
| Connected        | $O(\alpha(n))$ | $O(1)$       | 版本化查询               |
| 访问旧版本       | $O(1)$         | $O(1)$       | 版本指针查找             |

可持久化并查集是连通性的历史地图。每个版本捕获了关系的快照——不可变、可查询且高效——非常适合演化中的图和支持回滚的算法。

## 第 30 章. 高级树结构与区间查询
### 291 稀疏表构建

稀疏表是一种静态数据结构，在 O(n log n) 的预处理后，能以 O(1) 的时间回答幂等区间查询。它非常适用于区间最小/最大值查询、最大公约数查询，以及任何允许答案重叠合并的操作，例如 `min`、`max`、`gcd`、`lcm`（需注意）和按位 `and/or`。如果你需要 O(1) 的查询，它不适用于求和或其他非幂等操作。

#### 我们要解决什么问题？

给定一个数组 `A[0..n-1]`，我们想要回答以下类型的查询：

- RMQ：区间 `[L, R]` 的最小值
- RMaxQ：区间 `[L, R]` 的最大值
  每次查询的时间复杂度为 O(1)，且不支持更新。

#### 工作原理（通俗解释）

预先计算所有长度为 2 的幂次的区间的答案。
令 `st[k][i]` 存储从索引 `i` 开始、长度为 `2^k` 的区间（即 `[i, i + 2^k - 1]`）的答案。

构建递推关系：

- 基础层 `k = 0`：长度为 1 的区间
  `st[0][i] = A[i]`
- 更高层：合并两个长度为 `2^{k-1}` 的半区间
  `st[k][i] = op(st[k-1][i], st[k-1][i + 2^{k-1}])`

要回答区间 `[L, R]` 的查询，令 `len = R - L + 1`，`k = floor(log2(len))`。
对于像 `min` 或 `max` 这样的幂等操作，我们可以用两个重叠的块覆盖该区间：

- 块 1：`[L, L + 2^k - 1]`
- 块 2：`[R - 2^k + 1, R]`
  然后
  $$
  \text{ans} = \operatorname{op}\big(\text{st}[k][L],\ \text{st}[k][R - 2^k + 1]\big)
  $$

#### 逐步示例

数组 `A = [7, 2, 3, 0, 5, 10, 3, 12, 18]`，`op = min`。

1. 构建 `st[0]`（长度 1）：
   `st[0] = [7, 2, 3, 0, 5, 10, 3, 12, 18]`

2. 构建 `st[1]`（长度 2）：
   `st[1][i] = min(st[0][i], st[0][i+1])`
   `st[1] = [2, 2, 0, 0, 5, 3, 3, 12]`

3. 构建 `st[2]`（长度 4）：
   `st[2][i] = min(st[1][i], st[1][i+2])`
   `st[2] = [0, 0, 0, 0, 3, 3]`

4. 构建 `st[3]`（长度 8）：
   `st[3][i] = min(st[2][i], st[2][i+4])`
   `st[3] = [0, 0]`

查询示例：区间 `[3, 8]` 的 RMQ
`len = 6`，`k = floor(log2(6)) = 2`，`2^k = 4`

- 块 1：`[3, 6]` 使用 `st[2][3]`
- 块 2：`[5, 8]` 使用 `st[2][5]`
  答案
  $$
  \min\big(\text{st}[2][3], \text{st}[2][5]\big) = \min(0, 3) = 0
  $$

#### 精简代码（Python，使用 min 的 RMQ）

```python
import math

def build_sparse_table(arr, op=min):
    n = len(arr)
    K = math.floor(math.log2(n)) + 1
    st = [[0] * n for _ in range(K)]
    for i in range(n):
        st[0][i] = arr[i]
    j = 1
    while (1 << j) <= n:
        step = 1 << (j - 1)
        for i in range(n - (1 << j) + 1):
            st[j][i] = op(st[j - 1][i], st[j - 1][i + step])
        j += 1
    # 预计算对数以实现 O(1) 查询
    lg = [0] * (n + 1)
    for i in range(2, n + 1):
        lg[i] = lg[i // 2] + 1
    return st, lg

def query(st, lg, L, R, op=min):
    length = R - L + 1
    k = lg[length]
    return op(st[k][L], st[k][R - (1 << k) + 1])

# 示例
A = [7, 2, 3, 0, 5, 10, 3, 12, 18]
st, lg = build_sparse_table(A, op=min)
print(query(st, lg, 3, 8, op=min))  # 0
print(query(st, lg, 0, 2, op=min))  # 2
```

对于 `max`，只需传入 `op=max`。对于 `gcd`，传入 `math.gcd`。

#### 为什么它很重要

- 对于静态数组，查询时间为 O(1)
- 预处理时间为 O(n log n)，且状态转移简单
- 非常适用于 RMQ 类任务、基于欧拉序 RMQ 的 LCA 以及许多竞赛编程问题
- 在不需要更新的情况下，与线段树相比，缓存友好且实现简单

#### 一个温和的证明（为什么它有效）

该表存储了所有长度为 `2^k` 的区间的答案。任何区间 `[L, R]` 都可以被两个长度为 `2^k` 的、等长的、重叠的 2 的幂次块覆盖，其中 `k = floor(log2(R - L + 1))`。
对于幂等操作 `op`，重叠不会影响正确性，因此
$$
\text{op}\big([L, R]\big) = \text{op}\big([L, L + 2^k - 1],\ [R - 2^k + 1, R]\big)
$$
这两个块都是预先计算好的，因此查询是常数时间。

#### 自己动手试试

1.  为 `A = [5, 4, 3, 6, 1, 2]` 构建一个 `op = min` 的表。
2.  回答区间 `[1, 4]` 和 `[0, 5]` 的 RMQ。
3.  将 `op` 切换为 `max` 并重新检查。
4.  使用 `op = gcd` 并在几个区间上验证结果。

#### 测试用例

| 数组                     | op  | 查询    | 期望值 |
| ------------------------ | --- | ------- | ------ |
| [7, 2, 3, 0, 5, 10, 3]   | min | [0, 2]  | 2      |
| [7, 2, 3, 0, 5, 10, 3]   | min | [3, 6]  | 0      |
| [1, 5, 2, 4, 6, 1, 3]    | max | [2, 5]  | 6      |
| [12, 18, 6, 9, 3]        | gcd | [1, 4]  | 3      |

#### 复杂度

| 阶段       | 时间          | 空间         |
| ---------- | ------------- | ------------ |
| 预处理     | $O(n \log n)$ | $O(n \log n)$ |
| 查询       | $O(1)$        |              |

注意
稀疏表支持静态数据。如果你需要更新，请考虑使用线段树或树状数组。当数组固定且需要非常快速的查询时，稀疏表表现出色。
### 292 笛卡尔树

笛卡尔树是一种由数组构建的二叉树，满足以下条件：

1. 树的中序遍历结果与原数组一致，并且
2. 树满足关于数组值的堆性质（最小堆或最大堆）。

这种结构优雅地连接了数组和二叉树，并在范围最小值查询（RMQ）、最近公共祖先（LCA）和序列分解等算法中扮演着关键角色。

#### 我们要解决什么问题？

我们希望将数组 $A[0..n-1]$ 表示为一棵能编码范围关系的树。对于 RMQ 问题，如果树是一棵最小堆笛卡尔树，那么节点 $i$ 和 $j$ 的 LCA 就对应着范围 $[i, j]$ 中最小元素的索引。

因此，构建一棵笛卡尔树，在 $O(n)$ 的预处理之后，为我们提供了一条从 RMQ 到 LCA 的优雅路径，查询时间复杂度为 $O(1)$。

#### 工作原理（通俗解释）

笛卡尔树是递归构建的：
- 根节点是数组中的最小元素（对于最小堆）或最大元素（对于最大堆）。
- 左子树由根节点左侧的元素构建。
- 右子树由根节点右侧的元素构建。

一种更高效的线性时间构造方法使用栈：
1. 从左到右遍历数组。
2. 维护一个节点值递增的栈。
3. 对于每个新元素，当栈顶元素值大于新元素值时，持续弹出栈顶元素。然后将新节点作为最后一个弹出节点的右孩子，或者作为当前栈顶节点的左孩子。

#### 示例

设 $A = [3, 2, 6, 1, 9]$

1. 从空栈开始
2. 插入 3 → 栈 = [3]
3. 插入 2 → 弹出 3（因为 3 > 2）
   * 2 成为 3 的父节点
   * 栈 = [2]
4. 插入 6 → 6 > 2 → 6 成为 2 的右孩子
   * 栈 = [2, 6]
5. 插入 1 → 弹出 6，弹出 2 → 1 成为新的根节点
   * 2 成为 1 的右孩子
6. 插入 9 → 9 成为 6 的右孩子

树结构（最小堆）：

```
       1
      / \
     2   9
    / \
   3   6
```

中序遍历：`[3, 2, 6, 1, 9]`
堆性质：每个父节点都小于其子节点 ✅

#### 微型代码（Python，最小堆笛卡尔树）

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def build_cartesian_tree(arr):
    stack = []
    root = None
    for val in arr:
        node = Node(val)
        last = None
        while stack and stack[-1].val > val:
            last = stack.pop()
        node.left = last
        if stack:
            stack[-1].right = node
        else:
            root = node
        stack.append(node)
    return root

def inorder(node):
    return inorder(node.left) + [node.val] + inorder(node.right) if node else []
```

使用示例：

```python
A = [3, 2, 6, 1, 9]
root = build_cartesian_tree(A)
print(inorder(root))  # [3, 2, 6, 1, 9]
```

#### 为何重要

- **O(1) 的 RMQ**：在笛卡尔树中，RMQ 转化为 LCA 问题（经过欧拉环游 + 稀疏表预处理后）。
- **与单调栈的联系**：线性构造过程与基于栈的范围问题（下一个更大元素、直方图）逻辑相通。
- **分治分解**：代表了数组的递归结构。
- **高效构建**：使用栈实现线性时间。

#### 一个温和的证明（为何有效）

每个元素最多被压入和弹出栈一次，因此总操作次数为 $O(n)$。
堆性质保证了 RMQ 的正确性：
- 在最小堆树中，任何子树的根节点都是该段的最小值。
- 因此，LCA(i, j) 给出了 $\min(A[i..j])$ 的索引。

所以，通过将 RMQ 归约为 LCA，我们实现了：

$$
\text{RMQ}(i, j) = \text{index}( \text{LCA}(i, j) )
$$

#### 动手试试

1.  为 $A = [4, 5, 2, 3, 1]$ 构建一棵笛卡尔树（最小堆）。
2.  验证中序遍历结果是否等于原数组。
3.  标记出父节点小于子节点的关系。
4.  利用 LCA 从树中找出 RMQ(1, 3)。

#### 测试用例

| 数组              | 树类型     | 根节点 | RMQ(1, 3) | 中序遍历匹配 |
| ----------------- | ---------- | ------ | --------- | ------------ |
| [3, 2, 6, 1, 9]   | 最小堆     | 1      | 2         | ✅            |
| [5, 4, 3, 2, 1]   | 最小堆     | 1      | 2         | ✅            |
| [1, 2, 3, 4, 5]   | 最小堆     | 1      | 2         | ✅            |
| [2, 7, 5, 9]      | 最小堆     | 2      | 5         | ✅            |

#### 复杂度

| 操作             | 时间复杂度      | 空间复杂度      | 备注                         |
| ---------------- | --------------- | --------------- | ---------------------------- |
| 构建             | $O(n)$          | $O(n)$          | 基于栈的构造                 |
| 查询（RMQ）      | $O(1)$          |,               | 欧拉环游 + 稀疏表预处理后    |
| LCA 预处理       | $O(n \log n)$   | $O(n \log n)$   | 稀疏表方法                   |

笛卡尔树编织了顺序与层级：中序遍历代表序列，堆性质代表支配关系，它是数组与树之间一座无声的桥梁。
### 293 线段树 Beats

线段树 Beats 是经典线段树的一种高级变体，能够处理超出求和、最小值或最大值范围的非平凡区间查询和更新操作。它专为那些操作*非线性*或*不可逆*的问题而设计，例如区间 chmin/chmax、带最小值追踪的区间加法，或区间次小值查询。

它通过存储额外状态（如次小值、次大值）来决定何时可以提前停止更新，从而“击败”了经典懒惰传播的限制。

#### 我们正在解决什么问题？

标准线段树无法高效处理复杂的更新操作，例如：

- “将 $[L, R]$ 区间内所有 $A[i]$ 设为 `min(A[i], x)`”
- “将 $[L, R]$ 区间内所有 $A[i]$ 设为 `max(A[i], x)`”

因为线段中不同元素的行为可能因其相对于 `x` 的值而异。

线段树 Beats 通过在每个节点中维护额外的约束来解决这个问题，这样当条件满足时，我们就可以“击败”递归并提前跳过分支。

#### 工作原理（通俗解释）

每个节点不仅存储聚合信息，还存储*足够的信息以了解操作何时可以完全应用*。

对于区间 Chmin 操作（A[i] = min(A[i], x)）：

- 如果 `max <= x`：无变化
- 如果 `second_max < x < max`：仅更新值为 `max` 的元素
- 如果 `x < second_max`：递归到子节点

每个节点存储：

- `max`（区间内的最大值）
- `second_max`（区间内的第二大值）
- `count_max`（`max` 值出现的次数）
- `sum`（区间和）

这使我们能够在不触及所有元素的情况下决定更新逻辑。

#### 示例（区间 Chmin）

设 $A = [4, 7, 6, 3]$

我们应用 `chmin(L=0, R=3, x=5)` → 每个大于 5 的元素被设为 5。

步骤：

1. 节点 `[0, 3]`：`max = 7`，`second_max = 6`
2. 由于 `x = 5 < second_max = 6`，递归
3. 更新左子节点 `[0,1]`：包含 `7` → `7` 变为 `5`
4. 更新右子节点 `[2,3]`：`max = 6` → `6` 变为 `5`

新数组：`[4, 5, 5, 3]`

#### 简化代码（简化的区间 Chmin）

```python
class Node:
    def __init__(self):
        self.sum = 0
        self.max = 0
        self.smax = -float('inf')
        self.cnt = 0
        self.l = None
        self.r = None

def merge(a, b):
    node = Node()
    node.sum = a.sum + b.sum
    if a.max == b.max:
        node.max = a.max
        node.smax = max(a.smax, b.smax)
        node.cnt = a.cnt + b.cnt
    elif a.max > b.max:
        node.max = a.max
        node.smax = max(a.smax, b.max)
        node.cnt = a.cnt
    else:
        node.max = b.max
        node.smax = max(a.max, b.smax)
        node.cnt = b.cnt
    return node

def push_chmin(node, x, length):
    if node.max <= x:
        return
    node.sum -= (node.max - x) * node.cnt
    node.max = x

def update_chmin(node, l, r, ql, qr, x):
    if r < ql or qr < l or node.max <= x:
        return
    if ql <= l and r <= qr and node.smax < x:
        push_chmin(node, x, r - l + 1)
        return
    m = (l + r) // 2
    update_chmin(node.l, l, m, ql, qr, x)
    update_chmin(node.r, m+1, r, ql, qr, x)
    new = merge(node.l, node.r)
    node.max, node.smax, node.cnt, node.sum = new.max, new.smax, new.cnt, new.sum
```

其核心在于：尽可能跳过更新，必要时才分裂。

#### 为何重要

- 高效处理非线性更新
- 通过减少不必要的递归保持对数复杂度
- 用于许多带有区间上限操作的竞争性编程 RMQ 类挑战
- 将线段树推广到“困难”问题（区间 `min` 上限、区间 `max` 上限、条件和）

#### 一个温和的证明（为何有效）

诀窍在于：通过存储最大值、次大值和计数，如果操作仅影响等于最大值的元素，我们就可以停止向下遍历。
每次更新最多涉及 O(log n) 个节点，因为：

- 每次更新都会降低某些最大值
- 每个元素的值在稳定之前对数级地减少

因此总复杂度摊还后为：
$$
O((n + q) \log n)
$$

#### 动手尝试

1.  为 $A = [4, 7, 6, 3]$ 构建一个线段树 Beats。
2.  应用 `chmin(0,3,5)` → 验证得到 `[4,5,5,3]`。
3.  应用 `chmin(0,3,4)` → 验证得到 `[4,4,4,3]`。
4.  跟踪每次操作后的 `sum`。

#### 测试用例

| 数组         | 操作           | 结果         |
| ------------ | -------------- | ------------ |
| [4,7,6,3]    | chmin(0,3,5)   | [4,5,5,3]    |
| [4,5,5,3]    | chmin(1,2,4)   | [4,4,4,3]    |
| [1,10,5,2]   | chmin(0,3,6)   | [1,6,5,2]    |
| [5,5,5]      | chmin(0,2,4)   | [4,4,4]      |

#### 复杂度

| 操作         | 时间（摊还）     | 空间     | 备注                         |
| ------------ | ---------------- | -------- | ---------------------------- |
| 构建         | $O(n)$           | $O(n)$   | 与普通线段树相同             |
| 区间 Chmin   | $O(\log n)$      | $O(n)$   | 在操作上摊还                 |
| 查询（求和） | $O(\log n)$      | ,        | 像往常一样组合               |

线段树 Beats 优雅与强大并存，在保持 $O(\log n)$ 直觉的同时，处理了经典线段树无法触及的操作类型。
### 294 归并排序树

归并排序树是一种线段树，其中每个节点存储其区间内元素的有序列表。它允许高效地执行依赖于顺序统计的查询，例如统计有多少元素落在某个范围内，或者查找子数组中的第 $k$ 小元素。

它之所以被称为"归并排序树"，是因为其构建方式与归并排序完全相同：分治、合并已排序的两半。

#### 我们要解决什么问题？

经典的线段树处理求和、最小值或最大值，但不处理*基于值*的查询。归并排序树支持以下操作：

- 统计 $A[L..R]$ 中有多少个数 $\le x$
- 统计值在范围 $[a,b]$ 内的元素数量
- 查找 $A[L..R]$ 中的第 $k$ 小元素

这些问题经常出现在区间频率查询、逆序对计数和离线查询中。

#### 工作原理（通俗解释）

树的每个节点覆盖数组的一个区间 `[l, r]`。它不存储单个数字，而是存储该区间内所有元素的有序列表。

构建过程：

1. 如果 `l == r`，则存储 `[A[l]]`。
2. 否则，递归构建左子节点和右子节点。
3. 合并两个已排序的列表，形成此节点的列表。

查询过程：
要统计区间 `[L, R]` 中 $\le x$ 的数字数量：

- 访问所有与 `[L, R]` 完全或部分重叠的线段树节点。
- 在每个节点中，二分查找 `x` 在该节点有序列表中的位置。

#### 示例

令 $A = [2, 5, 1, 4, 3]$

#### 步骤 1：构建树

每个叶子节点存储一个值：

| 节点区间 | 存储的列表 |
| ---------- | ----------- |
| [0,0]      | [2]         |
| [1,1]      | [5]         |
| [2,2]      | [1]         |
| [3,3]      | [4]         |
| [4,4]      | [3]         |

现在进行合并：

| 节点区间 | 存储的列表 |
| ---------- | ----------- |
| [0,1]      | [2,5]       |
| [2,3]      | [1,4]       |
| [2,4]      | [1,3,4]     |
| [0,4]      | [1,2,3,4,5] |

#### 步骤 2：查询示例

统计区间 `[1, 4]` 中 $\le 3$ 的元素数量。

我们查询覆盖 `[1,4]` 的节点：`[1,1]`、`[2,3]`、`[4,4]`。

- `[1,1]` → 列表 = `[5]` → 数量 = 0
- `[2,3]` → 列表 = `[1,4]` → 数量 = 1
- `[4,4]` → 列表 = `[3]` → 数量 = 1

总计 = 2 个元素 ≤ 3。

#### 精简代码（Python，统计 ≤ x）

```python
import bisect

class MergeSortTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [[] for _ in range(4 * self.n)]
        self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr, node, l, r):
        if l == r:
            self.tree[node] = [arr[l]]
            return
        m = (l + r) // 2
        self._build(arr, node * 2, l, m)
        self._build(arr, node * 2 + 1, m + 1, r)
        self.tree[node] = sorted(self.tree[node * 2] + self.tree[node * 2 + 1])

    def query_leq(self, node, l, r, ql, qr, x):
        if r < ql or qr < l:
            return 0
        if ql <= l and r <= qr:
            return bisect.bisect_right(self.tree[node], x)
        m = (l + r) // 2
        return (self.query_leq(node * 2, l, m, ql, qr, x) +
                self.query_leq(node * 2 + 1, m + 1, r, ql, qr, x))

# 示例
A = [2, 5, 1, 4, 3]
mst = MergeSortTree(A)
print(mst.query_leq(1, 0, 4, 1, 4, 3))  # 统计 [1,4] 中 ≤ 3 的数量 = 2
```

#### 为什么它很重要

- 支持基于顺序的查询（≤, ≥, 计数, 排名）
- 对于离线区间计数、第 k 小查询和逆序对查询很有用
- 将分治排序与区间分解相结合

#### 一个温和的证明（为什么它有效）

树的每一层都合并来自子节点的已排序列表。

- 每个元素出现在 $O(\log n)$ 个节点中。
- 每次合并的时间复杂度与子数组大小成线性关系。

因此，总构建时间：
$$
O(n \log n)
$$

每次查询访问 $O(\log n)$ 个节点，每个节点进行 $O(\log n)$ 的二分查找：
$$
O(\log^2 n)
$$

#### 亲自尝试

1.  为 $A = [5, 1, 4, 2, 3]$ 构建树。
2.  查询 `[0, 4]` 中 ≤ 3 的数量。
3.  查询 `[1, 3]` 中 ≤ 2 的数量。
4.  使用两次 `query_leq` 调用来实现"统计 $[a,b]$ 之间"的查询。

#### 测试用例

| 数组         | 查询        | 条件 | 答案 |
| ----------- | --------- | --------- | ------ |
| [2,5,1,4,3] | [1,4], ≤3 | 计数     | 2      |
| [2,5,1,4,3] | [0,2], ≤2 | 计数     | 2      |
| [1,2,3,4,5] | [2,4], ≤4 | 计数     | 3      |
| [5,4,3,2,1] | [0,4], ≤3 | 计数     | 3      |

#### 复杂度

| 操作           | 时间复杂度          | 空间复杂度         |
| ------------------- | ------------- | ------------- |
| 构建               | $O(n \log n)$ | $O(n \log n)$ |
| 查询 (≤ x)         | $O(\log^2 n)$ |,             |
| 查询 (区间计数) | $O(\log^2 n)$ |,             |

归并排序树优雅地连接了排序和分段，使得区间查询不仅依赖于聚合值，还依赖于值本身的分布。
### 295 小波树（Wavelet Tree）

小波树是一种紧凑、可索引的结构，构建在序列之上，支持在 $O(\log \sigma)$ 时间内进行按值的秩（rank）、选择（select）和范围查询，空间复杂度为 $O(n \log \sigma)$，其中 $(\sigma)$ 是字母表大小。可以将其视为构建在位向量（bitvector）之上、具有值感知能力的线段树，允许你使用秩计数在不同层级之间跳转。

#### 我们要解决什么问题？

给定一个值域在 $[1..\sigma]$ 内的数组 $A[1..n]$，定义以下查询：

- $\text{rank}(x, r)$：值 $x$ 在 $A[1..r]$ 中出现的次数
- $\text{select}(x, k)$：值 $x$ 在 $A$ 中第 $k$ 次出现的位置
- $\text{kth}(l, r, k)$：子数组 $A[l..r]$ 中第 $k$ 小的值
- $\text{range\_count}(l, r, a, b)$：$A[l..r]$ 中值在 $[a, b]$ 范围内的元素数量

#### 工作原理

1. **按值域对半划分**
   递归地将值域 $[v_{\min}, v_{\max}]$ 在中点 $m$ 处划分为两半。
   - 左子节点存储所有 $\le m$ 的元素
   - 右子节点存储所有 $> m$ 的元素

2. **稳定划分与位向量**
   在每个节点，保持原始顺序，并记录一个长度等于到达该节点元素数量的位向量 $B$：
   - 如果第 $i$ 个元素进入左子节点，则 $B[i] = 0$
   - 如果第 $i$ 个元素进入右子节点，则 $B[i] = 1$
   支持在 $B$ 上进行快速秩查询：$\mathrm{rank}_0(B,i)$ 和 $\mathrm{rank}_1(B,i)$。

3. **查询导航**
   - **基于位置的下探**：使用 $B$ 的前缀计数来转换位置。
     如果在此节点的查询区间是 $[l,r]$，那么左子节点中对应的区间是
     $[\mathrm{rank}_0(B,l-1)+1,\ \mathrm{rank}_0(B,r)]$，
     右子节点中对应的区间是
     $[\mathrm{rank}_1(B,l-1)+1,\ \mathrm{rank}_1(B,r)]$。
   - **基于值的下探**：通过比较值与 $m$ 的大小来选择左子节点或右子节点。

树高为 $O(\log \sigma)$。每一步使用 $O(1)$ 次位向量秩操作。

#### 示例

数组：$A = [3, 1, 4, 1, 5, 9, 2, 6]$，值域 $[1..9]$

根节点按中点 $m = 5$ 划分
- 左子节点接收 $\le 5$ 的元素：$\{3, 1, 4, 1, 5, 2\}$
- 右子节点接收 $> 5$ 的元素：$\{9, 6\}$

根节点位向量标记元素是向左（0）还是向右（1）路由，保持顺序：
- 节点处的 $A$： [3, 1, 4, 1, 5, 9, 2, 6]
- $B$： [0, 0, 0, 0, 0, 1, 0, 1]

左子节点在 $[1..5]$ 范围内按 $m = 3$ 划分
- 左-左子节点（值 $\le 3$）：在根节点 $B=0$ 的位置，然后按 $m=3$ 路由
  到达的序列： [3, 1, 4, 1, 5, 2]
  此节点的位向量 $B_L$： [0, 0, 1, 0, 1, 0]
  子节点：
  - $\le 3$： [3, 1, 1, 2]
  - $> 3$： [4, 5]
- 根节点的右-右子树包含 [9, 6]（未展示进一步划分）

根节点处的秩转换示例

对于根节点的一个区间 $[l,r]$，对应的区间为：
- 左子节点： $[\,\mathrm{rank}_0(B,l-1)+1,\ \mathrm{rank}_0(B,r)\,]$
- 右子节点： $[\,\mathrm{rank}_1(B,l-1)+1,\ \mathrm{rank}_1(B,r)\,]$

示例：取根节点处 $[l,r]=[2,7]$，$B=[0,0,0,0,0,1,0,1]$
- $\mathrm{rank}_0(B,1)=1$， $\mathrm{rank}_0(B,7)=6$ → 左区间 $[2,6]$
- $\mathrm{rank}_1(B,1)=0$， $\mathrm{rank}_1(B,7)=1$ → 右区间 $[1,1]$

树高为 $O(\log \sigma)$，每次下探步骤使用 $O(1)$ 次局部位向量的秩操作。
### 核心操作

1. rank(x, r)  
   自顶向下遍历。在分割值为 m 的节点处：

   - 如果 $x \le m$，设置 $r \leftarrow \mathrm{rank}_0(B, r)$ 并向左走
   - 否则，设置 $r \leftarrow \mathrm{rank}_1(B, r)$ 并向右走

   当到达值 $x$ 对应的叶子节点时，当前的 $r$ 就是答案。

2. select(x, k)  
   从值 $x$ 对应的叶子节点开始，其局部索引为 $k$。  
   向上移动到根节点，在每个父节点处反转位置映射：
   - 如果来自左子节点，设置 $k \leftarrow \mathrm{select\_pos\_0}(B, k)$
   - 如果来自右子节点，设置 $k \leftarrow \mathrm{select\_pos\_1}(B, k)$  
   根节点处最终的 $k$ 就是第 $k$ 个 $x$ 的全局位置。

3. kth(l, r, k)  
   在分割值为 $m$ 的节点处，令
   $$
   c \;=\; \mathrm{rank}_0(B, r)\;-\;\mathrm{rank}_0(B, l-1)
   $$
   这是 $[l, r]$ 区间内被路由到左侧的元素数量。

   - 如果 $k \le c$，通过以下方式将区间映射到左子节点：
     $$
     l' \;=\; \mathrm{rank}_0(B, l-1) + 1,\quad
     r' \;=\; \mathrm{rank}_0(B, r)
     $$
     并在 $(l', r', k)$ 上递归。
   - 否则，向右走，使用：
     $$
     k \leftarrow k - c,\quad
     l' \;=\; \mathrm{rank}_1(B, l-1) + 1,\quad
     r' \;=\; \mathrm{rank}_1(B, r)
     $$
     并在 $(l', r', k)$ 上递归。

4. range_count(l, r, a, b)  
   仅递归进入与 $[a, b]$ 相交的值区间。  
   在每个访问的节点处，使用 $B$ 映射位置区间：
   - 左子节点区间
     $$
     [\,\mathrm{rank}_0(B, l-1)+1,\ \mathrm{rank}_0(B, r)\,]
     $$
   - 右子节点区间
     $$
     [\,\mathrm{rank}_1(B, l-1)+1,\ \mathrm{rank}_1(B, r)\,]
     $$
   当节点的值范围完全在 $[a, b]$ 内部或外部时停止：
   - 完全在内部：加上其区间长度
   - 不相交：加上 0

#### Python 微型代码草图

此草图展示了结构和 kth 查询。生产版本需要为 (B) 提供简洁的 rank 结构以保证 (O(1)) 的 rank 操作。

```python
import bisect

class WaveletTree:
    def __init__(self, arr, lo=None, hi=None):
        self.lo = min(arr) if lo is None else lo
        self.hi = max(arr) if hi is None else hi
        self.b = []          # 位向量，存储为 0-1 列表
        self.pref = [0]      # 1 的前缀和，用于 O(1) 的 rank1
        if self.lo == self.hi or not arr:
            self.left = self.right = None
            return
        mid = (self.lo + self.hi) // 2
        left_part, right_part = [], []
        for x in arr:
            go_right = 1 if x > mid else 0
            self.b.append(go_right)
            self.pref.append(self.pref[-1] + go_right)
            if go_right:
                right_part.append(x)
            else:
                left_part.append(x)
        self.left = WaveletTree(left_part, self.lo, mid)
        self.right = WaveletTree(right_part, mid + 1, self.hi)

    def rank1(self, idx):  # b[1..idx] 中 1 的数量
        return self.pref[idx]

    def rank0(self, idx):  # b[1..idx] 中 0 的数量
        return idx - self.rank1(idx)

    def kth(self, l, r, k):
        # 1-索引位置
        if self.lo == self.hi:
            return self.lo
        mid = (self.lo + self.hi) // 2
        cnt_left = self.rank0(r) - self.rank0(l - 1)
        if k <= cnt_left:
            nl = self.rank0(l - 1) + 1
            nr = self.rank0(r)
            return self.left.kth(nl, nr, k)
        else:
            nl = self.rank1(l - 1) + 1
            nr = self.rank1(r)
            return self.right.kth(nl, nr, k - cnt_left)
```

使用示例：

```python
A = [3,1,4,1,5,9,2,6]
wt = WaveletTree(A)
print(wt.kth(1, 8, 3))  # 整个数组中第 3 小的元素
```

#### 为何重要

- 将值分区与位置稳定性相结合，支持子区间上的顺序统计
- 是简洁索引、FM 索引、rank select 字典和快速离线区间查询的基础
- 当 (\sigma) 适中或可压缩时效率很高

#### 复杂度界限的温和证明

由于每一层都将值域减半，树的高度为 $O(\log \sigma)$。每个查询下降一层，并在位向量上执行 $O(1)$ 次 rank 操作。因此
$$
T_{\text{query}} = O(\log \sigma)
$$
空间上平均每层为每个元素存储一个比特，所以
$$
S = O(n \log \sigma)
$$
使用支持常数时间 rank 和 select 的压缩位向量，这些界限在实践中成立。

#### 动手尝试

1.  为 $A = [2, 7, 1, 8, 2, 8, 1]$ 构建一棵小波树。
2.  计算 $\text{kth}(2, 6, 2)$。
3.  计算 $\text{range\_count}(2, 7, 2, 8)$。
4.  与对 $A[l..r]$ 进行朴素排序的结果进行比较。

#### 测试用例

| 数组               | 查询                 | 答案 |
| ------------------ | -------------------- | ---- |
| [3,1,4,1,5,9,2,6]  | kth(1,8,3)           | 3    |
| [3,1,4,1,5,9,2,6]  | range_count(3,7,2,5) | 3    |
| [1,1,1,1,1]        | rank(1,5)            | 5    |
| [5,4,3,2,1]        | kth(2,5,2)           | 3    |

#### 复杂度

| 操作         | 时间                | 空间               |
| ------------ | ------------------- | ------------------ |
| 构建         | $O(n \log \sigma)$  | $O(n \log \sigma)$ |
| rank, select | $O(\log \sigma)$    | –                  |
| kth, range_count | $O(\log \sigma)$ | –                  |

小波树是处理顺序感知区间查询的利器。通过将位向量与稳定分区交织，它们能在原始序列之上提供简洁、对数时间的答案。
### 296 KD 树

KD 树（k 维树）是一种用于组织 k 维空间中点的二叉空间分割数据结构。它支持快速的范围搜索、最近邻查询和空间索引，常用于几何、图形和机器学习（如 k-NN）领域。

#### 我们要解决什么问题？

我们需要存储和查询 k 维空间中的 n 个点，以便：

- 最近邻查询：找到距离查询点 q 最近的点。
- 范围查询：找到给定区域（矩形或超球体）内的所有点。
- k 近邻查询：找到距离 q 最近的 k 个点。

朴素的搜索会检查所有 n 个点，时间复杂度为 O(n)。
对于平衡的 KD 树，可以将查询时间降低到 O(log n) 的期望时间复杂度。

#### 工作原理

#### 1. 按维度递归分割

在每一层，KD 树沿一个维度分割点集，并在所有维度间循环。

如果当前深度为 d，则使用轴 a = d mod k：

- 按点的第 a 个坐标对点进行排序。
- 选择中位数点作为根节点以保持平衡。
- 左子节点：第 a 个坐标较小的点。
- 右子节点：第 a 个坐标较大的点。

#### 2. 搜索（最近邻）

要查找查询点 q 的最近邻：

1.  沿着分割平面（类似于二叉搜索树）向下遍历树。
2.  跟踪当前最佳结果（迄今为止找到的最近点）。
3.  如果分割平面另一侧可能存在更近的点（到平面的距离 < 当前最佳距离），则进行回溯。

这确保了可以剪枝掉不可能包含更近点的子树。

#### 示例

假设有 2D 点：
$$
P = { (2,3), (5,4), (9,6), (4,7), (8,1), (7,2) }
$$

步骤 1：根节点按 x 轴（轴 0）分割。
按 x 排序：((2,3), (4,7), (5,4), (7,2), (8,1), (9,6))。
中位数 = ((7,2))。根节点 = (7,2)。

步骤 2：

-   左子树（x < 7 的点）按 y 轴分割。
-   右子树（x > 7 的点）按 y 轴分割。

这创建了交替按 x 和 y 轴进行的分割，形成了轴对齐的矩形。

#### 微型代码（Python）

```python
class Node:
    def __init__(self, point, axis):
        self.point = point
        self.axis = axis
        self.left = None
        self.right = None

def build_kdtree(points, depth=0):
    if not points:
        return None
    k = len(points[0])
    axis = depth % k
    points.sort(key=lambda p: p[axis])
    median = len(points) // 2
    node = Node(points[median], axis)
    node.left = build_kdtree(points[:median], depth + 1)
    node.right = build_kdtree(points[median + 1:], depth + 1)
    return node
```
### 最近邻搜索的工作原理

给定查询点 (q)，维护最佳距离 (d_{\text{best}})。  
对于每个访问的节点：

- 计算距离 (d = |q - p|)
- 如果 (d < d_{\text{best}})，则更新最佳结果
- 仅当 (|q[a] - p[a]| < d_{\text{best}}) 时，才检查对侧分支

#### 示例表格

| 步骤 | 访问的节点 | 轴 | 当前最佳结果 | 到分割平面的距离 | 下一步搜索 |
| ---- | ---------- | -- | ------------ | ---------------- | ---------- |
| 1    | (7,2)      | x  | (7,2), 0.0   | 0.0              | 左子树     |
| 2    | (5,4)      | y  | (5,4), 2.8   | 2.0              | 左子树     |
| 3    | (2,3)      | x  | (5,4), 2.8   | 3.0              | 停止       |

#### 为何重要

- 在多维数据中进行高效的空间查询。
- 常用于 k-最近邻分类、计算机图形学和机器人路径规划。
- 是 `scipy.spatial.KDTree` 等库的基础。

#### 一个温和的证明（为何有效）

每一层都将点集分成两半，形成 $O(\log n)$ 的高度。  
每次查询访问的节点数量是有限的（取决于维度）。  
期望的最近邻查询成本：

$$
T_{\text{query}} = O(\log n)
$$

构建时在每一层对点进行排序：

$$
T_{\text{build}} = O(n \log n)
$$

#### 动手尝试

1.  为二维点 ([(2,3),(5,4),(9,6),(4,7),(8,1),(7,2)]) 构建 KD 树。
2.  查询 (q = (9,2)) 的最近邻。
3.  追踪访问的节点，并在可能时剪枝子树。

#### 测试用例

| 点集                                  | 查询点 | 最近邻 | 期望路径          |
| ------------------------------------- | ------ | ------ | ----------------- |
| [(2,3),(5,4),(9,6),(4,7),(8,1),(7,2)] | (9,2)  | (8,1)  | 根节点 → 右子树 → 叶节点 |
| [(1,1),(2,2),(3,3)]                   | (2,3)  | (2,2)  | 根节点 → 右子树       |
| [(0,0),(10,10)]                       | (5,5)  | (10,10)| 根节点 → 右子树       |

#### 复杂度

| 操作           | 时间                      | 空间        |
| -------------- | ------------------------- | ----------- |
| 构建           | $O(n \log n)$             | $O(n)$      |
| 最近邻查询     | $O(\log n)$ （期望）      | $O(\log n)$ |
| 范围查询       | $O(n^{1 - 1/k} + m)$      | –           |

KD 树将几何与二分搜索相结合，通过维度切割空间，从而比暴力搜索更快地回答问题。
### 297 范围树

范围树是一种多级搜索结构，用于回答多维空间中的正交范围查询，例如查找轴对齐矩形内的所有点。它通过投影上的递归树，将一维平衡搜索树扩展到更高维度。

#### 我们要解决什么问题？

给定 $k$ 维空间中的 $n$ 个点，我们希望高效地回答如下查询：

> "列出所有满足 $x_1 \le x \le x_2$ 且 $y_1 \le y \le y_2$ 的点 $(x, y)$。"

朴素扫描每次查询需要 $O(n)$ 时间。范围树将此降低到 $O(\log^k n + m)$，其中 $m$ 是报告的点数。

#### 工作原理

#### 1. 一维情况（基线）

在 $x$ 坐标上构建一个简单的平衡二叉搜索树（例如 AVL 树），通过遍历路径并收集范围内的节点来支持范围查询。

#### 2. 二维情况（扩展）

-   在 $x$ 坐标上构建一棵主树。
-   在每个节点处，存储一棵基于其子树中点的 $y$ 坐标构建的次级树。

每一层都递归地维护着沿其他轴的排序视图。

#### 3. 范围查询

1.  在主树中搜索分裂节点 $s$，即通往 $x_1$ 和 $x_2$ 的路径在此分叉。
2.  对于完全在 $[x_1, x_2]$ 内的节点，查询它们关联的 $y$ 树以获取 $[y_1, y_2]$ 内的点。
3.  合并结果。

#### 示例

给定点：

$$
P = {(2,3), (4,7), (5,1), (7,2), (8,5)}
$$

查询：$[3,7] \times [1,5]$

-   在 $x$ 上的主树：中位数点 $(5,1)$ 作为根。
-   每个节点处都有在 $y$ 上的次级树。

搜索路径：

-   分裂节点 $(5,1)$ 覆盖 $x \in [3,7]$。
-   访问 $x \in [4,7]$ 的子树。
-   在次级树内部查询 $y$ 在 $[1,5]$ 内的点。

返回的点：$(5,1), (7,2), (4,7)$ → 过滤 $y \le 5$ → $(5,1),(7,2)$。

#### 微型代码（类 Python 伪代码）

```python
class RangeTree:
    def __init__(self, points, depth=0):
        if not points:
            self.node = None
            return
        axis = depth % 2
        points.sort(key=lambda p: p[axis])
        mid = len(points) // 2
        self.node = points[mid]
        self.left = RangeTree(points[:mid], depth + 1)
        self.right = RangeTree(points[mid + 1:], depth + 1)
        self.sorted_y = sorted(points, key=lambda p: p[1])
```

递归查询：

-   按 $x$ 过滤节点。
-   在 $y$ 列表上进行二分搜索。
### 逐步操作表（二维查询）

| 步骤 | 操作             | 轴   | 条件            | 动作               |
| ---- | ---------------- | ---- | --------------- | ------------------ |
| 1    | 在 (5,1) 处分割 | x    | $3 \le 5 \le 7$ | 递归处理两侧       |
| 2    | 左 (2,3),(4,7)   | x    | $x < 5$         | 访问 (4,7) 子树    |
| 3    | 右 (7,2),(8,5)   | x    | $x \le 7$       | 访问 (7,2) 子树    |
| 4    | 按 $y$ 过滤      | y    | $1 \le y \le 5$ | 保留 (5,1),(7,2)   |

#### 为何重要

与 kd 树（性能可能退化）不同，范围树为多维查询提供了确定性的性能。它们非常适合：

- 正交范围计数
- 数据库索引
- 计算几何问题

它们是静态结构，适用于数据不经常变化的情况。

#### 一个温和的证明（为何有效）

每个维度都会增加一个对数因子。
在二维中，构建时间：

$$
T(n) = O(n \log n)
$$

查询在主树中访问 $O(\log n)$ 个节点，每个节点查询 $O(\log n)$ 个辅助树：

$$
Q(n) = O(\log^2 n + m)
$$

由于辅助树的存在，空间复杂度为 $O(n \log n)$。

#### 动手尝试

1.  为点集 $P = {(1,2),(2,3),(3,4),(4,5),(5,6)}$ 构建一个二维范围树。
2.  查询矩形 $[2,4] \times [3,5]$。
3.  追踪访问过的节点并验证输出。

#### 测试用例

| 点集                               | 查询矩形        | 结果               |
| ---------------------------------- | --------------- | ------------------ |
| [(2,3),(4,7),(5,1),(7,2),(8,5)]    | [3,7] × [1,5]   | (5,1),(7,2)        |
| [(1,2),(2,4),(3,6),(4,8),(5,10)]   | [2,4] × [4,8]   | (2,4),(3,6),(4,8)  |
| [(1,1),(2,2),(3,3)]                | [1,2] × [1,3]   | (1,1),(2,2)        |

#### 复杂度

| 操作     | 时间               | 空间           |
| -------- | ------------------ | -------------- |
| 构建     | $O(n \log n)$      | $O(n \log n)$  |
| 查询(2D) | $O(\log^2 n + m)$  |,               |
| 更新     |, (需要重建)        |,               |

范围树是精确的几何索引：每个轴划分空间，嵌套的树提供了对任意轴对齐框内所有点的快速访问。
### 298 二维 Fenwick 树

二维 Fenwick 树（也称为二维二叉索引树）将一维 Fenwick 树扩展到二维网格（如矩阵）上，以处理区间查询和点更新。它能在 $O(\log^2 n)$ 时间内高效计算前缀和并支持动态更新。

#### 我们要解决什么问题？

给定一个 $n \times m$ 的矩阵 $A$，我们希望高效支持两种操作：

1.  **更新**：将值 $v$ 加到元素 $A[x][y]$ 上。
2.  **查询**：计算子矩阵 $[1..x][1..y]$ 中所有元素的和。

朴素方法每次查询需要 $O(nm)$ 时间。二维 Fenwick 树将更新和查询都降低到 $O(\log n \cdot \log m)$。

#### 工作原理

树中的每个节点 $(i, j)$ 存储一个子矩阵区域的和，该区域由其索引的二进制表示决定：

$$
T[i][j] = \sum_{x = i - 2^{r_i} + 1}^{i} \sum_{y = j - 2^{r_j} + 1}^{j} A[x][y]
$$

其中 $r_i$ 和 $r_j$ 表示 $i$ 和 $j$ 的最低有效位（LSB）。

#### 更新规则

当通过 $v$ 更新 $(x, y)$ 时：

```python
for i in range(x, n+1, i & -i):
    for j in range(y, m+1, j & -j):
        tree[i][j] += v
```

#### 查询规则

计算前缀和 $(1,1)$ 到 $(x,y)$：

```python
res = 0
for i in range(x, 0, -i & -i):
    for j in range(y, 0, -j & -j):
        res += tree[i][j]
return res
```

#### 区间查询

子矩阵 $[(x_1,y_1),(x_2,y_2)]$ 的和：

$$
S = Q(x_2, y_2) - Q(x_1-1, y_2) - Q(x_2, y_1-1) + Q(x_1-1, y_1-1)
$$

#### 示例

给定一个 $4 \times 4$ 矩阵：

| $x/y$ | 1 | 2 | 3 | 4 |
| ----- | - | - | - | - |
| 1     | 2 | 1 | 0 | 3 |
| 2     | 1 | 2 | 3 | 1 |
| 3     | 0 | 1 | 2 | 0 |
| 4     | 4 | 0 | 1 | 2 |

构建二维 Fenwick 树，然后查询子矩阵 $[2,2]$ 到 $[3,3]$ 的和。

预期结果：

$$
A[2][2] + A[2][3] + A[3][2] + A[3][3] = 2 + 3 + 1 + 2 = 8
$$
### 逐步更新示例

假设我们在 $(2, 3)$ 处添加 $v = 5$：

| 步骤 | 更新的 $(i,j)$ | 增加值 | 原因                         |
| ---- | -------------- | ------ | ---------------------------- |
| 1    | $(2,3)$        | +5     | 基础位置                     |
| 2    | $(2,4)$        | +5     | 通过 $j += j \& -j$ 下一个   |
| 3    | $(4,3)$        | +5     | 通过 $i += i \& -i$ 下一个   |
| 4    | $(4,4)$        | +5     | 两个索引都向上传播           |

#### 精简代码（类 Python 伪代码）

```python
class Fenwick2D:
    def __init__(self, n, m):
        self.n, self.m = n, m
        self.tree = [[0]*(m+1) for _ in range(n+1)]

    def update(self, x, y, val):
        i = x
        while i <= self.n:
            j = y
            while j <= self.m:
                self.tree[i][j] += val
                j += j & -j
            i += i & -i

    def query(self, x, y):
        res = 0
        i = x
        while i > 0:
            j = y
            while j > 0:
                res += self.tree[i][j]
                j -= j & -j
            i -= i & -i
        return res

    def range_sum(self, x1, y1, x2, y2):
        return (self.query(x2, y2) - self.query(x1-1, y2)
                - self.query(x2, y1-1) + self.query(x1-1, y1-1))
```

#### 为何重要

二维 Fenwick 树是用于前缀和与子矩阵查询的轻量级、动态数据结构。它们广泛应用于：

- 图像处理（积分图像更新）
- 基于网格的动态规划
- 竞赛编程中的二维范围查询

它们以稍高的代码复杂度为代价，换来了卓越的更新-查询效率。

#### 一个温和的证明（为何有效）

每个更新/查询操作在 $x$ 上执行 $\log n$ 步，在 $y$ 上执行 $\log m$ 步：

$$
T(n,m) = O(\log n \cdot \log m)
$$

每一层都添加由最低有效位分解确定的子区域的贡献，确保每个单元格在查询和中恰好贡献一次。

#### 亲自尝试

1.  初始化一个 $4 \times 4$ 的二维 Fenwick 树。
2.  在 $(2,3)$ 处添加 $5$。
3.  查询从 $(1,1)$ 到 $(2,3)$ 的和。
4.  验证其是否与手动计算结果一致。

#### 测试用例

| 矩阵（部分）                  | 更新       | 查询矩形       | 结果 |
| ----------------------------- | ---------- | -------------- | ---- |
| $[ [2,1,0],[1,2,3],[0,1,2] ]$ | $(2,3)+5$  | $[1,1]$–$[2,3]$ | 14   |
| $[ [1,1,1],[1,1,1],[1,1,1] ]$ | $(3,3)+2$  | $[2,2]$–$[3,3]$ | 5    |
| $[ [4,0],[0,4] ]$             |            | $[1,1]$–$[2,2]$ | 8    |

#### 复杂度

| 操作     | 时间                     | 空间   |
| -------- | ------------------------ | ------ |
| 更新     | $O(\log n \cdot \log m)$ | $O(nm)$ |
| 查询     | $O(\log n \cdot \log m)$ |        |

二维 Fenwick 树是前缀和与空间查询之间一座优雅的桥梁，对于动态二维网格来说，它简单、强大且高效。
### 299 Treap 分裂/合并

Treap 分裂/合并算法允许你高效地分割和组合 Treap（随机化平衡二叉搜索树），它利用基于优先级的旋转和基于键值的排序。这是对隐式 Treap 进行范围操作（如分裂、合并、范围更新和区间查询）的基础。

#### 我们要解决什么问题？

我们经常需要：

1. 将一个 Treap 分成两部分：
   * 所有键值 $\le k$ 的节点进入左侧 Treap
   * 所有键值 $> k$ 的节点进入右侧 Treap

2. 合并两个 Treap $T_1$ 和 $T_2$，其中 $T_1$ 中的所有键值都小于 $T_2$ 中的键值

这些操作使得在保持平衡高度的同时，能够高效地进行区间查询、持久化编辑和顺序统计。

#### 工作原理（通俗解释）

Treap 结合了两种属性：
- 二叉搜索树属性：左子树 < 根节点 < 右子树
- 堆属性：节点优先级 > 子节点优先级

分裂和合并依赖于由键值和优先级引导的递归下降。

#### 分裂操作

按键值 $k$ 分裂 Treap $T$：
- 如果 $T.key \le k$，将 $T.right$ 分裂为 $(t2a, t2b)$ 并设置 $T.right = t2a$
- 否则，将 $T.left$ 分裂为 $(t1a, t1b)$ 并设置 $T.left = t1b$

返回 $(T.left, T.right)$

#### 合并操作

合并 $T_1$ 和 $T_2$：
- 如果 $T_1.priority > T_2.priority$，设置 $T_1.right = \text{merge}(T_1.right, T_2)$
- 否则，设置 $T_2.left = \text{merge}(T_1, T_2.left)$

返回新的根节点。

#### 示例

假设我们有一个包含以下键值的 Treap：
$$ [1, 2, 3, 4, 5, 6, 7] $$

#### 按 $k = 4$ 分裂：

左侧 Treap：$[1, 2, 3, 4]$
右侧 Treap：$[5, 6, 7]$

现在，将它们合并可以恢复原始顺序。
### 逐步拆分示例

| 步骤 | 节点键值 | 与 $k=4$ 比较 | 操作             |
| ---- | -------- | ------------- | ---------------- |
| 1    | 4        | $\le$ 4       | 向右走           |
| 2    | 5        | $>$ 4         | 拆分左子树       |
| 3    | 5.left=∅$ | 返回 (null,5) | 组合返回         |

结果：左子树 = $[1,2,3,4]$，右子树 = $[5,6,7]$

#### 精简代码（类 Python 伪代码）

```python
import random

class Node:
    def __init__(self, key):
        self.key = key
        self.priority = random.random()
        self.left = None
        self.right = None

def split(root, key):
    if not root:
        return (None, None)
    if root.key <= key:
        left, right = split(root.right, key)
        root.right = left
        return (root, right)
    else:
        left, right = split(root.left, key)
        root.left = right
        return (left, root)

def merge(t1, t2):
    if not t1 or not t2:
        return t1 or t2
    if t1.priority > t2.priority:
        t1.right = merge(t1.right, t2)
        return t1
    else:
        t2.left = merge(t1, t2.left)
        return t2
```

#### 为何重要

Treap 的拆分/合并操作解锁了灵活的序列操作和基于范围的操作：

- 区间求和 / 最小值 / 最大值查询
- 以 $O(\log n)$ 时间复杂度插入或删除
- 用于列表的持久化或隐式 Treap
- 区间的惰性传播

它是函数式编程和竞赛编程数据结构中的关键构建模块。

#### 一个温和的证明（为何有效）

每次拆分或合并操作都会遍历 Treap 的高度。由于 Treap 是期望平衡的，其高度为 $O(\log n)$。

- 拆分的正确性：
  每次递归调用都保持了二叉搜索树（BST）的排序性质。
- 合并的正确性：
  由于优先级最高的节点成为根节点，因此保持了堆性质。

因此，两者都返回有效的 Treap。

#### 动手尝试

1.  用键值 $[1..7]$ 构建一个 Treap。
2.  以 $k=4$ 进行拆分。
3.  打印两个子 Treap 的中序遍历。
4.  合并回去。确认结构与原始结构匹配。

#### 测试用例

| 输入键值        | 拆分键值 | 左子树 Treap 键值 | 右子树 Treap 键值 |
| --------------- | -------- | ----------------- | ----------------- |
| [1,2,3,4,5,6,7] | 4        | [1,2,3,4]         | [5,6,7]           |
| [10,20,30,40]   | 25       | [10,20]           | [30,40]           |
| [5,10,15,20]    | 5        | [5]               | [10,15,20]        |

#### 复杂度

| 操作   | 时间复杂度 | 空间复杂度 |
| ------ | ---------- | ---------- |
| 拆分   | $O(\log n)$ | $O(1)$     |
| 合并   | $O(\log n)$ | $O(1)$     |

Treap 的拆分/合并是许多动态集合和序列结构优雅的核心：一个键值，一个随机优先级，两个简单操作，无限灵活性。
### 300 树上的莫队算法

树上的莫队算法是经典莫队算法在数组上的扩展。它通过将树转换为线性顺序（欧拉序），然后应用分块策略，能够高效处理树上的离线查询，特别是涉及子树或路径的查询。

#### 我们要解决什么问题？

当你需要回答多个类似以下的查询时：

- "节点 $u$ 的子树中有多少个不同的值？"
- "从 $u$ 到 $v$ 的路径上的和是多少？"

朴素的方法可能需要对每个查询进行 $O(n)$ 的遍历，导致总复杂度为 $O(nq)$。
树上的莫队算法通过复用邻近查询的结果，将复杂度降低到大约 $O((n + q)\sqrt{n})$。

#### 工作原理（通俗解释）

1.  **欧拉序扁平化**
    使用欧拉遍历将树转换为线性数组。每个节点的首次出现标记了它在线性化序列中的位置。

2.  **查询转换**

    *   对于子树查询，一个子树在欧拉数组中变成一个连续的范围。
    *   对于路径查询，将其拆分为两个子范围，并单独处理最近公共祖先（LCA）。

3.  **莫队排序**
    按以下规则对查询进行排序：

    *   左端点所在块（使用 $\text{block} = \lfloor L / \sqrt{N} \rfloor$）
    *   右端点（升序或按块交替）

4.  **添加/移除函数**
    随着窗口移动，维护一个频率映射或运行结果。

#### 示例

给定一棵树：

```
1
├── 2
│   ├── 4
│   └── 5
└── 3
```

欧拉序：`[1, 2, 4, 4, 5, 5, 2, 3, 3, 1]`

子树(2)：覆盖范围 `[2, 4, 4, 5, 5, 2]`

每个子树查询都变成了对欧拉数组的一个范围查询。
莫队算法按排序后的顺序高效地处理这些范围。
### 逐步示例

| 步骤 | 查询 (L,R) | 当前区间 | 添加/移除 | 结果  |
| ---- | ----------- | ------------- | ---------- | ------- |
| 1    | (2,6)       | [2,6]         | +4,+5      | 计数=2 |
| 2    | (2,8)       | [2,8]         | +3         | 计数=3 |
| 3    | (1,6)       | [1,6]         | -3,+1      | 计数=3 |

每个查询都通过增量调整来回答，而非重新计算。

#### 精简代码（类 Python 伪代码）

```python
import math

# 预处理
def euler_tour(u, p, g, order):
    order.append(u)
    for v in g[u]:
        if v != p:
            euler_tour(v, u, g, order)
            order.append(u)

# Mo 的结构
class Query:
    def __init__(self, l, r, idx):
        self.l, self.r, self.idx = l, r, idx

def mo_on_tree(n, queries, order, value):
    block = int(math.sqrt(len(order)))
    queries.sort(key=lambda q: (q.l // block, q.r))

    freq = [0]*(n+1)
    answer = [0]*len(queries)
    cur = 0
    L, R = 0, -1

    def add(pos):
        nonlocal cur
        node = order[pos]
        freq[node] += 1
        if freq[node] == 1:
            cur += value[node]

    def remove(pos):
        nonlocal cur
        node = order[pos]
        freq[node] -= 1
        if freq[node] == 0:
            cur -= value[node]

    for q in queries:
        while L > q.l:
            L -= 1
            add(L)
        while R < q.r:
            R += 1
            add(R)
        while L < q.l:
            remove(L)
            L += 1
        while R > q.r:
            remove(R)
            R -= 1
        answer[q.idx] = cur
    return answer
```

#### 为何重要

- 高效地将树查询转换为区间查询
- 通过滑动窗口技术重用计算
- 适用于频率、求和或去重计数查询
- 支持子树查询、路径查询（需处理 LCA）和基于颜色的查询

#### 一个温和的证明（为何有效）

1. 欧拉序保证每个子树都是一个连续区间。
2. Mo 算法确保添加/移除操作的总数为 $O((n + q)\sqrt{n})$。
3. 结合两者，每个查询都能在对数摊还成本内增量处理。

因此，离线处理的复杂度对每个查询是次线性的。

#### 动手尝试

1. 为包含 7 个节点的树构建欧拉序。
2. 为节点 $2,3,4$ 编写子树查询。
3. 按块顺序对查询进行排序。
4. 实现添加/移除逻辑以计数不同颜色或求和。
5. 与每次查询都进行朴素 DFS 的性能进行比较。

#### 测试用例

| 节点结构               | 查询          | 预期结果                     |
| ------------------- | ---------- | --------------------------- |
| [1-2-3-4-5]         | 子树(2)    | 节点 2–5 的和或计数          |
| [1: {2,3}, 2:{4,5}] | 子树(1)    | 所有节点                    |
| [1-2,1-3]           | 路径(2,3)  | LCA=1 需单独处理            |

#### 复杂度

| 操作             | 时间                 | 空间  |
| --------------------- | -------------------- | ------ |
| 预处理（欧拉序） | $O(n)$               | $O(n)$ |
| 查询排序         | $O(q \log q)$        | $O(q)$ |
| 处理            | $O((n + q)\sqrt{n})$ | $O(n)$ |

树上的 Mo 算法是图遍历、离线区间查询和摊还优化的优雅结合点，为复杂的分层数据带来了次线性的查询处理能力。
