# 附录1：Python 语法速查（写给 C / Java 程序员）

> 定位：你已熟悉 C 或 Java，本附录**不讲"编程是什么"**，只讲"Python 哪里不一样"。学完应能看懂 PyTorch 章节里的 `class`、`@`、`with`、`yield`、`lambda`、列表推导式这些写法，以及 §9 的 NumPy / SciPy 数组（张量的前置）。
>
> 一条主线贯穿全文：**C/Java 是"编译期强类型 + 值/引用分离"，Python 是"运行时动态类型 + 一切皆对象"。** 绝大多数差异都从这"不同构建逻辑"推出来。

## 1. 三个最大的直觉差异

### 1.1 代码块：缩进替代花括号

C/Java 用 `{}` 分块；Python 用**冒号 + 缩进**（约定 4 空格）：

```python
if x > 0:
    print(x)      # 缩进 = 属于 if 块
else:
    print(-x)
```

* 缩进不一致直接报错（混用 Tab/空格也是）
* 续行：行尾 `\`；在 `[]` `{}` `()` 内部换行不用 `\`
* **注意 `{}` 在 Python 里不是代码块，是字典字面量**——这是最容易眼花的差异

**注释**：`#` 是单行注释；`'''...'''` 或 `"""..."""` 是多行注释（严格说是"没被使用的字符串"，运行时被丢弃，但惯用法上就当注释/文档用）：

```python
# 单行注释
'''
多行注释：中间可以随便换行
'''
```

> 顺带认识一个编辑器记号：`#%%` 是 Spyder / VS Code 的"单元格分隔符"——把脚本切成一段段单独运行。它只是编辑器的记号，不是 Python 语法，跑脚本时不起作用。

### 1.2 变量：动态类型 + 名字绑定对象

| | C / Java | Python |
|---|---|---|
| 类型 | 声明时固定（`int a`） | 运行时推断，可随时换类型 |
| `a = b` | 拷贝值（或赋值引用） | **让名字 `a` 指向 `b` 那个对象** |
| 数组下标越界 | 未定义行为 / 异常 | 抛 `IndexError` |

Python 变量是**名字 → 对象**的绑定，赋值不复制数据：

```python
a = [1, 2, 3]
b = a          # 不是拷贝！b 和 a 指向同一个 list
b.append(4)
print(a)       # [1, 2, 3, 4]   a 也变了
c = a.copy()   # 真要复制用 .copy() / 深拷贝 copy.deepcopy()
```

**深拷贝 vs 浅拷贝**（C/Java 程序员最容易混）：Python 把"复制"分成两档，且 `b = a` 根本不算复制——它只是让 `b` 也指向 `a` 那个对象：

| 写法 | 类型 | 行为 |
|---|---|---|
| `b = a` | 引用绑定 | **不拷贝**，`b` 和 `a` 是同一个对象 |
| `b = a.copy()` / `copy.copy(a)` | **浅拷贝** | 新外层对象，但内部的元素仍是共享引用 |
| `b = a[:]`（切片） | 浅拷贝（简写） | 等价于 `a.copy()`：新外层、内层共享 |
| `b = copy.deepcopy(a)` | **深拷贝** | 递归复制所有内容，完全独立 |

```python
import copy
lst = [[1], [2]]
shallow = copy.copy(lst)          # 新外层，但内层 [1]、[2] 还是共享的
deep = copy.deepcopy(lst)         # 内层也复制，完全独立
shallow[0].append(9)
print(lst)        # [[1, 9], [2]]   ← 浅拷贝内层共享，原列表跟着变
deep[0].append(9)
print(lst)        # [[1, 9], [2]]   ← 深拷贝早已独立，不受影响
```

> 记忆：C 里 `=` 是拷贝值，Java 里 `=` 是拷贝引用，**Python 的 `=` 和 Java 一样是引用**——但 Java 的 `clone()` 默认浅拷贝且用起来麻烦，Python 的 `copy.copy()` / `copy.deepcopy()` 一清二楚。嵌套容器（list 套 list）必须用深拷贝才彻底独立。

### 1.3 函数参数：传"对象引用"

* **可变对象**（list/dict/set）：函数内修改会体现在外面（和 Java 传对象引用一样）
* **不可变对象**（int/str/tuple）：改的是新对象，外面看不到（看起来像 C 的值传递）

```python
def add_one(x):
    x += 1        # int 不可变 → 外面不受影响
    return x

def append_item(lst):
    lst.append(9) # list 可变 → 外面能看到

n = 1;  add_one(n);   print(n)   # 1
m = [1]; append_item(m); print(m) # [1, 9]
```

> 常见坑：默认参数别用可变对象——`def f(a=[])` 会**共享同一个 list**。

## 2. 类似的表达，不同的语义

| C/Java 写法 | Python 写法 | 语义差异 |
|---|---|---|
| `a = b` | `a = b` | 绑定引用，不拷贝 |
| `a == b` | `a == b` | Python 默认比**内容**（Java 比引用，要 `.equals`）；比"是不是同一个对象"用 `is` |
| `&&` \| `\|\|` | `and` \| `or` | 关键字，不是符号 |
| `++i` | 没有 | 用 `i += 1` |
| `a ? b : c` | `b if a else c` | 三元顺序不同 |
| `switch` | 没有 | 用 `if/elif/elif` 或字典 |
| `for(i=0;i<n;i++)` | `for i in range(n)` | 迭代式，不是计数式 |
| `a[]` | `a[0]`, `a[-1]` | 负索引从末尾数；切片 `a[1:3]` |
| `{ }` | `{}` = 字典 | Python 的块是"缩进"，`{}` 不是块 |
| `+` | `+` | 数字加法；`[1]+[2]` 拼 list、`"a"+"b"` 拼字符串（重载） |
| `*` | `*` | 乘法；`[0]*3` 重复、`*args` 解包 |
| `%` | `%` | 取模；还能做字符串格式化（`"%d" % x`） |
| `;` | 可省 | 不需要分号（写了也不算错） |
| `NULL` / `null` | `None` | 唯一空值；判断用 `if x is None:` |

## 3. 数据容器

| 容器 | 写法 | 类似 C/Java | 特点 |
|---|---|---|---|
| list 列表 | `[1, 2, 3]` | C 数组 / Java `ArrayList` | 可增删、元素可不同类型、切片、负索引 |
| tuple 元组 | `(1, 2)` | Java 无直接对应 | 不可变，可作字典键 |
| dict 字典 | `{'a': 1}` | Java `HashMap` | 键值对，键可以是任意不可变对象 |
| set 集合 | `{1, 2}` | Java `HashSet` | 去重、快速成员判断 |

**易错**：`(9)` 是 int；`(9,)` 才是单元素元组。

list 常用操作：

```python
lst = ['Google', 1997, 2000]
del lst[0]            # 删元素
lst.append('Baidu')   # 追加
lst[1] = 'New'        # 改元素
lst[1:3]              # 切片：含 1 不含 3
lst + [5]             # 拼接
lst * 2               # 重复
'Baidu' in lst        # 成员判断
a, b = b, a           # 交换（同时赋值）
```

**list 还能当"栈"用**（后进先出）：`append` 进栈、`pop` 出栈（删掉最后一个并返回它）：

```python
stack = [1, 2, 3]
stack.append(4)
stack.pop()        # 4 —— 删掉最后一个并返回
stack              # [1, 2, 3]
```

**`zip`：把多个序列"按位配对"成 tuple**，`zip(*...)` 是它的逆运算，能再拆回原列：

```python
a = [1, 2, 3]
b = [4, 5, 6]
list(zip(a, b))       # [(1, 4), (2, 5), (3, 6)]
zip(*zip(a, b))       # 逆运算：拆回两列 [(1, 2, 3), (4, 5, 6)]
```

> `zip()` 返回**惰性迭代器**，要 `list()` 才真正算出来。

**列表推导式**（C/Java 没有的直觉，一行顶一个循环）：

```python
squares = [x * x for x in range(10) if x % 2 == 0]
# 等价于：
squares = []
for x in range(10):
    if x % 2 == 0:
        squares.append(x * x)
```

## 4. 控制流

```python
if x > 0:
    ...
elif x == 0:      # "else if" 的简写
    ...
else:
    ...

for i in range(n):      # 0..n-1
    ...
for item in lst:        # 遍历任意可迭代对象
    ...
for idx, val in enumerate(lst):   # 要下标：enumerate
    ...
while condition:
    ...
```

## 5. 函数

### 5.1 def

```python
def add(a, b):
    return a + b
```

* 参数可有默认值（`def f(a, b=1)`）；调用可带名字（`f(b=2, a=1)`）
* 可返回多个值——其实返回一个 tuple：`def f(): return 1, 2`

返回的 tuple 可以**拆包**到多个变量（同一语法，§3 已见）：

```python
def load_data():
    return 50000, 10000, 10000     # 其实返回一个 tuple

train_n, valid_n, test_n = load_data()   # 三个变量各拿一个
```

> §3 的 `a, b = b, a` 交换、`train_set, valid_set, test_set = ds` 拆开数据集，都是同一种"元组解包"：`=` 右边先算成一个 tuple，再按位置拆给左边各变量。

### 5.2 lambda：匿名函数

**从一个需求开始**：你想给 `sorted` 传一个"按第 2 列排序"的规则。`sorted` 需要一个函数作为参数（它会对每个元素调用这个函数来取排序键）。怎么办？

**写法一（直观写法）**：先定义一个函数，再传给 `sorted`：

```python
def key_func(item):
    return item[1]        # 取第 2 列作为排序键

sorted(lst, key=key_func)
```

痛点：这个 `key_func` 只用一次、只有一行，却要 `def` 一整段、还得起个名字。

**写法二（用 lambda）**：`lambda` 就是"没有名字的一行函数"，专门用来写这种"用完即走"的小函数：

```python
sorted(lst, key=lambda item: item[1])
#                 ↑lambda 参数: 返回值        ↑ 等价于写法一的 key_func
```

**解释**：`lambda x, y: x + y` 等价于：

```python
def add(x, y):
    return x + y
```

区别只有两点：① 没有名字（匿名）；② 只能写**一个表达式**，不能有语句。所以：

| | 写法一 `def` | 写法二 `lambda` |
|---|---|---|
| 有没有名字 | 有（`key_func`） | 无（匿名） |
| 能干的事 | 多行、带语句 | 只能一个表达式 |
| 典型用途 | 复杂逻辑 | 传给 `sorted`/`map`/`filter` 的小规则 |

> 不推荐 `add = lambda x, y: x + y` 这样把 lambda 赋给变量——既然要起名字，直接用 `def` 更清晰。

### 5.3 yield：生成器

**从一个需求开始**：你要按块读取一个大文件（比如每次 1024 字节），处理完一块再读下一块。直观做法是一次性全读进内存，但文件太大时会爆内存。怎么办？

**写法一（直观写法）**：用列表收集所有块，读完再逐个处理：

```python
def read_all(fpath):
    blocks = []
    with open(fpath, 'rb') as f:
        while True:
            block = f.read(1024)
            if not block:
                break
            blocks.append(block)     # 全读进内存，大文件就爆了
    return blocks

for block in read_all('data.txt'):
    process(block)                   # 等到全部读完才开始处理
```

痛点：所有块一次性占内存，且要等全部读完才能开始处理。

**写法二（用 yield）**：`yield` 让函数**边产出边暂停**——每产出一块就暂停，等你要下一块时再继续读。内存里始终只有一块：

```python
def read_block(fpath):
    with open(fpath, 'rb') as f:
        while True:
            block = f.read(1024)
            if not block:
                return
            yield block              # 产出当前块，暂停，等下次再要

for block in read_block('data.txt'): # 边读边处理，内存里只有一块
    process(block)
```

**解释**：带 `yield` 的函数**不再是普通函数**——调用它不会执行函数体，而是返回一个**生成器对象**（类似一个迭代器）：

```python
it = read_block('data.txt')   # 此刻什么都没读，只是拿到生成器
next(it)                      # 执行到第一个 yield，拿到第一个块并暂停
next(it)                      # 从暂停处继续，读到第二个块
```

| | 写法一 `def` + 列表 | 写法二 `def` + `yield` |
|---|---|---|
| 何时算 | 一次全算完，存列表 | 逐个产出，用完即弃 |
| 内存 | 所有块同时在内存 | 任意时刻只有一块 |
| 何时可开始处理 | 全部读完 | 第一个块出来就能处理 |

> 写法二里调用 `read_block()` 只是"创建生成器"，真正执行要靠 `for` 循环或 `next()` 一步步"拉"它。Python 3 用 `next(it)`，不是 `it.next()`。

### 5.4 装饰器（记号 `@` + 装饰器函数）

**从一个需求开始**：你写了个 `my_func()`，现在想让它每次被调用时自动打印耗时，但**不改动 `my_func` 内部**。怎么办？

**写法一（直观写法）**：写一个"加工函数" `timer`，它接收一个函数、返回一个"加了计时的新函数"，然后把这个新函数**重新赋给 `my_func` 这个名字**：

```python
def timer(f):                       # 加工函数：参数 f = 待加工的函数
    def wrapper(*args, **kwargs):   # 新函数 wrapper：内部先计时
        import time
        t = time.time()
        result = f(*args, **kwargs) # 这里才真正调用原函数
        print(f"耗时 {time.time() - t:.4f}s")
        return result
    return wrapper                  # 返回新函数

def my_func():                      # 先正常定义原函数
    ...

my_func = timer(my_func)            # 关键：用加工后的新函数覆盖原名字
#         ↑ 传原函数给 timer        ↑ timer 返回的新函数（wrapper）赋给 my_func
```

从此调用 `my_func()`，跑的是新函数 `wrapper`（计时 → 调原函数 → 打印）。原函数本体没被改动，只是"名字 `my_func` 被换了指向"。

**写法二（用装饰器语法）**：把"`def` 定义 + 事后赋值覆盖"两步合并成一行。在函数定义上方加 `@timer`，意思是"**把这个函数（就是紧挨在下面那行 `def my_func`）交给 `@` 后的加工函数去处理，再用它返回的新函数覆盖原名字**"：

```python
@timer                                # 这里就是装饰器：@ + 加工函数 timer
def my_func():                        # 紧挨在 @ 下面的这个 def = 被装饰函数（原函数）
    ...
```

> **`timer` 的实现在哪里？** 不在写法二里。`timer` 是**在别处先定义好的**（它就是写法一的那个加工函数，实现完全一样），写法二只负责"用 `@` 应用它"。所以写法二完整程序 = 写法一的 `def timer(...)` + 上面的 `@timer` + `def my_func`。写法一里"手动写 `my_func = timer(my_func)`"这一步，写法二用 `@` 自动完成。

**两种写法完全等价**，`@` 纯粹是缩写记号——它替你做了写法一里的 `my_func = timer(my_func)`。现在对照一下三个角色：

* **`@` 后面的 `timer`** = 装饰器（加工函数）
* **`@` 下面的 `def my_func`** = 被装饰函数（原函数）——写法一里它被传给了 `timer`，写法二的 `@` 就是"自动替你做这个传递"
* **`wrapper`** = 增强版函数（成品，覆盖 `my_func`）

> **为什么 `@` 叫"语法糖"**："语法糖"= 让代码更好写的简化记号，不增加新功能。`@timer` 与 `my_func = timer(my_func)` 效果完全相同，`@` 纯属"甜甜的语法"。你觉得绕就不用记这个词，记住"等价于 `my_func = timer(my_func)`"即可。

**C 语言里没有**（`@` 语法糖是 Python 独有），但概念上对应三种 C 手法：

| Python 装饰器 | C 的对应物 |
|---|---|
| `f = decorator(f)` | **函数指针 + 包装函数**：把函数指针传给包装函数，返回新的函数指针 |
| 编译期改写函数（如 `@cuda.jit`） | **宏**：`#define WRAP(x) ...` 在预处理期展开 |
| 给函数"贴标签"（如 CUDA 的 `@cuda.jit`） | **CUDA 函数修饰符**：`__global__` / `__device__` / `__host__`——同样是"声明这个函数有特殊性质"，编译期改写 |

```c
/* C 的"包装函数"：接收函数指针，返回增强后的函数指针 */
typedef int (*fn_t)(int);
int with_timer(fn_t f, int x) {     /* 等价于 Python 的 wrapper */
    double t = clock();
    int r = f(x);
    printf("耗时 %f\n", (clock() - t) / CLOCKS_PER_SEC);
    return r;
}
```

```c
/* CUDA 的"贴标签"修饰符——和 Python 装饰器思路同源 */
__global__ void vec_add_kernel(float *a, float *b, float *c) { ... }
```

### 5.5 with：上下文管理器

**从一个需求开始**：你想把一句话写进文件，写完**必须关闭文件**（忘了关会占资源、内容可能没落盘）。但写的过程中可能报错——一报错就跳过了 `f.close()`。怎么办？

**写法一（直观写法）**：用 try/finally 保证"无论正常还是报错，最后都执行 `f.close()`"：

```python
f = open('t.txt', 'w')
try:
    f.write('hello')      # 这里可能报错
finally:
    f.close()             # 不管报不报错，一定执行到这里
```

写法一逻辑清楚，但有个痛点：**每开一次文件，都要手动写一套 try/finally**。而这个"打开→用完→关闭"的配对，是极其常见的需求。

**写法二（用 `with` 语法）**：把"进入时准备、退出时清理"这个配对封装起来，一行代替整套 try/finally：

```python
with open('t.txt', 'w') as f:
    f.write('hello')
# with 自动完成：进入时打开文件并给 f；退出时（正常/报错都）调用 f.close()
```

> **`open()` 为什么会自动关闭？** `open()` 返回的文件对象实现了"上下文管理器协议"——它内部定义了 `__enter__`（进入时打开）和 `__exit__`（退出时关闭），所以能被 `with` 用。`as f` 就是把 `__enter__` 的返回值（文件对象）赋给 `f`。

**`with` 背后的通用机制**：不只是文件，**任何实现了 `__enter__` 和 `__exit__` 的对象**都能被 `with` 管理。自己写一个：

```python
class ManagedFile:
    def __init__(self, name):
        self.name = name
    def __enter__(self):                  # 进入 with 块时调用
        self.file = open(self.name, 'w')
        return self.file                  # as 后面的变量拿到它
    def __exit__(self, exc_type, exc_val, exc_tb):  # 退出时（正常/异常都调用）
        self.file.close()                 # 异常时 exc_type/exc_val/exc_tb 非 None，可吞掉或继续抛

with ManagedFile('hello.txt') as f:
    f.write('hi')
```

**对照两种写法**：写法二就是写法一的"封装版"——`with` 替你写好了 try/finally 和 `f.close()`，你要做的只是：给对象实现 `__enter__`/`__exit__`（或直接用现成的文件对象）。

PyTorch 里常见：`with torch.inference_mode():`——`__enter__` 关掉 Autograd 记账，`__exit__` 恢复现场（PyTorch 篇讲过）。

**C 语言里没有**（C++ 的 RAII 才等价），C 只能手动管理，容易漏收尾：

| Python `with` | C 的对应物 |
|---|---|
| 自动调用 `__exit__` 收尾 | **手动 `fclose()`**，忘了就泄漏；不保证异常路径执行 |
| 保证清理一定执行 | **`goto cleanup` 错误处理模式**——所有错误路径跳到一个统一收尾点 |
| 自动"进入准备" | 手动 `fopen()` / 初始化 |

```c
/* C 的 goto cleanup 模式：with 想解决的正是这个问题 */
FILE *f = fopen("t.txt", "w");
if (!f) goto error;
if (fwrite("hello", 1, 5, f) < 5) goto cleanup;   /* 中途出错也走到 cleanup */
cleanup:
    fclose(f);
    return 0;
error:
    return -1;
```

> 注意：C++ 用**构造函数/析构函数（RAII）**实现同样的"离开作用域自动清理"，是 Python `with` 最接近的等价物；C 语言只能靠 `goto cleanup` 这类手动约定。

## 6. 面向对象

### 6.1 先建立直觉：Python 面向对象的直观特点

如果你从 Java 过来，**先接受一件事：Java 的面向对象是"学院派"——严谨、强制、条条都按教科书来；Python 的面向对象是"自由派"——能省则省、少管闲事**。同一个概念，Java 有一整套规则约束你，Python 往往只有一条最简规则。

**Python 直观的特点（和 Java 一对比就懂）：**

| 特点 | Python | Java |
|---|---|---|
| 万物皆对象 | 连函数、类本身都是对象 | 有 8 种基本类型不是对象 |
| 建对象不用 `new` | `Animal("Dog")` 直接构造 | `new Animal("Dog")` |
| 成员访问不加修饰 | 所有属性/方法都 public，随便 `obj.x` | 要 `private`/`getter`/`setter` 层层管 |
| 实例方法带 `self` | 第一个参数必须显式写 `self` | `this` 是隐式的，不用写 |
| 类型只在运行时查 | `a.speak()` 编译期不检查，跑起来才知道 | 编译期强类型检查 |

**Python 反直觉的地方（Java 程序员最容易踩）：**

1. **`self` 显式却"没值"**：调用 `a.speak()` 时 Python 自动把 `a` 塞进第一个参数，**不需要你传**。所以 `speak` 定义时要有 `self`，调用时却不要——同一个 `self`，定义里写、调用里不写。

```python
class Animal:
    def speak(self):        # 定义时：第一个参数必须是 self
        print("hi")

a = Animal()
a.speak()                   # 调用时：不要传 self，Python 自动塞 a
```

2. **没有真正的 `private`**：Java 的 `private` 是硬规则，Python 只是"约定"——下划线 `_x` 表示"你别碰"，但真要碰没人拦你。

3. **没有接口关键字**：Java 用 `interface` 强制实现契约，Python 靠"鸭子类型"——长得像鸭子就叫鸭子，运行时只要对象有你要的方法就行，编译期不检查。

4. **属性可以后补**：Java 的字段在类里声明好；Python 的对象可以**随时往身上挂新属性**，类里没写也能加。

```python
a = Animal("Dog")
a.age = 3        # Java 里这行是编译错误；Python 允许，直接给对象加了个属性
```

**一句话建立心智模型**：Java 面向对象是"警察管得严"——类型、权限、接口处处把关，编译期就把错堵死；Python 面向对象是"无政府"——写起来自由、出错了运行时才报。这就是为什么 Java 严谨直观、Python 偶尔反直觉：**Python 把"省事"放在"严格"前面**。

### 6.2 类与构造函数

```python
class Animal:
    def __init__(self, name):   # 构造函数
        self.name = name        # self = 这个对象（Java 的 this，但要显式写）
    def speak(self):
        print(self.name)

a = Animal("Dog")               # 不需要 new
```

* 实例方法第一个参数**必须**是 `self`（C++/Java 的 `this` 是隐式的）
* `__init__` 是构造器，**不是静态函数**；`__xxx__` 是 Python 预留的特殊方法（`__len__`、`__repr__` 等）

### 6.3 运算符重载：靠 `__xxx__` 特殊方法

**从一个需求开始**：你定义了自己的 `Vec` 类，想让 `Vec + Vec`、`Vec == Vec` 直接能算，而不必每次写 `v.add(u)`。怎么办？

**直观思路**：给 `Vec` 定义一个"怎么相加"的方法，然后在 `+` 时自动调用它。这正是 Python 做的——**运算符背后就是特殊方法**：

```python
class Vec:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __add__(self, other):          # 重载 +：a + b 等价于 a.__add__(b)
        return Vec(self.x + other.x, self.y + other.y)

    def __eq__(self, other):           # 重载 ==：a == b 等价于 a.__eq__(b)
        return self.x == other.x and self.y == other.y

    def __repr__(self):                # 重载 print()：直接打印出内容
        return f"Vec({self.x}, {self.y})"

v = Vec(1, 2) + Vec(3, 4)   # 实际调用 Vec(1,2).__add__(Vec(3,4))
print(v)                    # Vec(4, 6)
print(Vec(1, 2) == Vec(1, 2))   # True，调用 __eq__
```

**常用特殊方法**：

| 运算符/功能 | 特殊方法 |
|---|---|
| `+` `-` `*` `/` | `__add__` `__sub__` `__mul__` `__truediv__` |
| `==` `!=` `<` `>` | `__eq__` `__ne__` `__lt__` `__gt__` |
| `len(x)` | `__len__` |
| `print(x)` | `__repr__` / `__str__` |
| `x[i]` | `__getitem__` |
| `str(x)` | `__str__` |

**和 C++/Java 的区别**：C++ 用 `operator+` 关键字重载，Java **不提供**运算符重载（`a + b` 只能算基本类型）；Python 靠"特殊方法"——**运算符是语法糖，`a + b` 本质就是 `a.__add__(b)`**。

> **为什么 PyTorch 的张量能直接 `a + b`、`a * b`？** 因为 `torch.Tensor` 重载了全套运算符（`__add__`、`__mul__`…），所以 `x + y` 这个"像普通数字"的写法，实际会走张量的运算逻辑（PyTorch 篇讲过）。这就是运算符重载在真实库里的价值。

### 6.4 继承

```python
class Dog(Animal):      # 括号 = 继承（Java 用 extends）
    ...

class X(A, B):          # 多重继承
    ...
```

> 没有 `interface` 关键字；多重继承 + 抽象基类（`abc`）承担类似职责。

### 6.5 与 C++ / Java 对照表

| Python | C++ | Java |
|---|---|---|
| `def __init__(self)` | 构造函数 | 构造函数 |
| `self`（显式） | `this` 指针 | `this`（隐式） |
| 缩进 = 代码块 | `{}` | `{}` |
| `class Dog(Animal):` | `class Dog : public Animal` | `class Dog extends Animal` |
| 自动 GC | 手动 `new`/`delete` | GC |
| 一切皆对象 | 值/引用分离 | 基本类型 + 引用 |
| 模块级函数（`os.path.xxx`） | 全局函数/库 | 静态方法 |

## 7. 模块与导入

```python
import os                     # 引入模块名，用 os.xxx 访问
import os as operating_system # 取别名
from os import path           # 只引某个名字，直接用 path
from os import *              # 引入全部（不推荐）
```

* 类比：`#include <os.h>` / `import java.util.*`，但 Python 模块就是 `.py` 文件本身
* 推荐 `import os` 或 `import os as xxx`，少用 `from ... import *`（污染命名空间）
* 科学计算有固定别名约定：`import numpy as np`——看到 `np.xxx` 就是 NumPy（本仓库案例、PyTorch 篇大量出现）

## 8. 标准库速查

**内置函数**（无需 import）：`help()` `type()` `len()` `range()` `sum()` `sorted()` `min()` `max()` `enumerate()` `zip()` `map()` `filter()` `eval()` `hash()` `id()`

> `reduce()` 在 Python 3 移到了 `functools`；`apply()` 已移除。

```python
import os
os.path.isdir('./.data')     # 判断目录
os.mkdir('./.data')          # 建目录
os.getcwd()                  # 当前目录

import shutil
shutil.copyfile('a.db', 'b.db')   # 复制
shutil.move('/src', '/dst')       # 移动/重命名

import sys
sys.path                       # 模块搜索路径
```

**对象序列化：把任意 Python 对象存进文件再读回来**——`pickle`（存的是字节流，必须用二进制模式打开）：

```python
import pickle
with open('data.pkl', 'wb') as f:
    pickle.dump([1, 2, 3], f)      # 存
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)          # 读
```

* 配 `gzip` 压缩着存：`gzip.open('data.pkl.gz', 'rb')` 打开压缩包，再 `pickle.load`（读 MNIST 这类数据集就是这套流程）
* `'wb'` / `'rb'` 里的 `b` 表示二进制模式——pickle 的数据是字节流，不能开文本模式

## 9. NumPy：从 Python 列表到"数组"（读懂张量的前置）

> 深度学习的"数据"是**多维数值数组**：图片是 `[高, 宽, 通道]` 的数字、权重是一张数字表。Python 原生 `list` 能用，但没有"形状"的概念、运算也慢。**NumPy** 提供 `ndarray`（n 维数组），PyTorch 的张量（PyTorch 篇）就是它的"GPU 亲戚"——`torch.Tensor` 和 `ndarray` 长得几乎一样。先把 NumPy 的基本直觉建立起来，PyTorch 篇上手会非常顺。

### 9.1 ndarray 是什么：有"形状"的数组

`list` 没有"行列"的概念；`ndarray` 有 `shape`（形状，各维大小的 tuple）：

```python
import numpy as np

d1 = np.array([1, 2, 3])        # 一维数组
d1.shape                        # (3,)  —— 注意是"一个元素的元组"
c2 = np.array([[1], [2], [3]])  # 二维，只有一列
c2.shape                        # (3, 1)
c3 = np.array([[1, 2, 3]])      # 二维，只有一行
c3.shape                        # (1, 3)

np.arange(15)       # 0..14 的数组（对比：range(15) 是惰性序列，不是数组）
np.ones((2, 3))     # 全 1 数组；np.zeros(...) 全 0
```

* `(3,)` 和 `(3, 1)` 都是"3 个数"，但**形状不同**：前者是一维数组，后者是 3 行 1 列的"矩阵"。张量编程里形状写错最常见就栽在这——PyTorch 篇的图片张量 `[3, 224, 224]` 是"通道在前"的写法
* 打印时一维数组往往竖着排，容易误当成列向量——**`shape` 才是真相**（`d1.shape` 是 `(3,)` 不是 `(3, 1)`）

**多维数组 ≠ 矩阵**：缺省 `ndarray` 做 `*` 是**逐元素**相乘，不是矩阵乘；要矩阵运算用 `@`：

```python
a = np.array([[1, 2], [3, 4]])
a * a       # 逐元素平方：[[1, 4], [9, 16]]
a @ a       # 矩阵乘：[[7, 10], [15, 22]]
```

> 这个区别一路带到 PyTorch：`x * y` 是逐元素、`x @ y` 是矩阵乘（PyTorch 篇讲过）。

### 9.2 广播（自动扩展）：不同形状也能相加

数组加标量、矩阵加向量，NumPy 会自动把小的"扩展"成大的——叫**广播（broadcasting）**。规则一句话：**从最后一维往前对齐，要么维度相等、要么有一方是 1**：

```python
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
row = np.array([1, 0, 1])      # 形状 (3,)——广播成"每行都加这一行"
a + row                        # [[2, 2, 4], [5, 5, 7], [8, 8, 10]]
a + 10                         # 标量广播到每个元素
```

* 向量 `(3,)` 默认按"行"扩展；要按"列"得先改成 `(3, 1)`（如 `np.array([[1], [2], [3]])`）
* 形状对不上（如 `(3, 2)` 加 `(3,)`）会**报错**——报错比悄悄算错好，这正是它和 C 数组的区别：越界、形状错都明确抛异常

**把一维数组变成"列/行"：`np.expand_dims`**——`(m,)` 只有一维、没有行/列之分（显示时像列向量但不是）。`np.expand_dims(a, 1)` 加一维成 `(m, 1)`（"列"），`np.expand_dims(a, 0)` 成 `(1, m)`（"行"）：

```python
v = np.array([10, 20, 30])       # 形状 (3,)：一维数组
np.expand_dims(v, 1).shape       # (3, 1) —— 变成"列"
np.expand_dims(v, 0).shape       # (1, 3) —— 变成"行"

m = np.arange(9).reshape(3, 3)   # 3×3 矩阵
m + np.expand_dims(v, 1)         # 按列扩展：每一列都加 [10, 20, 30]
```

**双向传播**：一边按行、一边按列，同时各自扩展成矩阵：

```python
row = np.array([10, 20, 30])     # (3,)   → 按行扩展成 3×3
col = np.array([[1], [2], [3]])  # (3, 1) → 按列扩展成 3×3
row + col                        # [[11 21 31],
                                 #  [12 22 32],
                                 #  [13 23 33]]
```

**一维数组用 `*` 没有"转置"概念**：长度相同就逐元素乘，不必像矩阵乘那样关心方向：

```python
v = np.array([1, 2, 3])
v * v                            # [1, 4, 9] 逐元素平方，不是矩阵乘（`@` 才是）
```

> 一句话：`*`（逐元素）只看**形状能不能对上**，不看行/列方向；`@`（矩阵乘）才关心方向。这就是为什么 `(m,)` 一维数组"没有向量的概念"。

### 9.3 重塑 reshape：换一种看法

```python
a = np.arange(784)
a.reshape(28, 28)        # 一维 784 个数 → 28×28 矩阵（图片常用）
np.reshape(a, (28, 28))  # 和上面等价
```

* `reshape` 只改"怎么看"，元素顺序和总数不变；`-1` 表示"让 numpy 自己算这一维"：`a.reshape(-1, 28)` 自动算出 28
* 图片处理常要"换轴"：`img.swapaxes(0, 2)` 交换第 0 和第 2 维（把 RGB 从最后一维挪到最前），再 `reshape(1, 3, 256, 256)` 凑出"批 × 通道 × 高 × 宽"的四维张量——PyTorch 篇数据加载就是这套

### 9.4 随机数与"种子"（可复现）

```python
rng = np.random.RandomState(23455)                      # 带种子的随机数发生器
rng.uniform(low=-0.5, high=0.5, size=(2, 3))            # 均匀分布
```

* **种子（seed）让随机序列固定**：同样种子，每次跑出来的随机数一模一样。深度学习里初始化、数据划分都要"可复现"，所以常看到固定种子
* 老写法 `np.random.seed(23455)` 设的是"全局种子"；`RandomState` 是更推荐的"独立发生器"，互不干扰

### 9.5 数据存取与图片

```python
np.save('a.npy', arr)     # 二进制存
arr2 = np.load('a.npy')   # 二进制读
np.savetxt('a.txt', arr)  # 文本存；np.loadtxt 文本读
```

图片（PIL 读入）转成数组，再喂给深度学习：

```python
from PIL import Image
img = Image.open('lenna.jpg')
img.size, img.mode        # (256, 256), 'RGB' —— 尺寸和颜色模式
imga = np.asarray(img, dtype='float32') / 255.   # 转数组并归一化到 0~1
```

> `np.asarray(x, dtype=...)` 把已有的 list/图片"变成"数组，`dtype` 指定元素类型——`float32` 是深度学习最常用的精度（对应 GPU 基础篇讲过的 FP32）。

### 9.6 SciPy：NumPy 之上的"科学计算工具箱"

> NumPy 管"数组"，SciPy 管"拿数组做科学计算"——线性代数、傅里叶变换、信号处理、图像滤波等都有现成函数。这里只讲和深度学习关系最直接的一个：**二维卷积**。

**从一个需求开始**：你想对一张图像做卷积（比如边缘检测），直观做法是自己写循环，但图像是几百×几百的数组，循环又慢又容易错。怎么办？

**直接调 `scipy.signal.convolve2d`**——一行处理整个二维数组的卷积：

```python
import numpy as np
from scipy import signal

a = np.array([[1, 2, 3],
              [3, 4, 5]])
b = np.array([[2, 3, 4],
              [4, 5, 6]])
c = signal.convolve2d(a, b, mode='full', boundary='wrap')
```

`convolve2d(a, b)` 把核 `b` **旋转 180°**（卷积的定义）后在 `a` 上滑动。两个参数控制行为：

| 参数 | 取值 | 含义 |
|---|---|---|
| `mode`（结果尺寸） | `'full'` | 结果 = a 的尺寸 +（b 的尺寸 − 1），边缘也完整算 |
| | `'valid'` | 只算核完全落入 a 内的部分，结果比 a 小 |
| | `'same'` | 结果和 a 一样大 |
| `boundary`（a 边缘外怎么补） | `'wrap'` | 把 a 平铺（卷绕） |
| | `'fill'` | 用 0 填充 |
| | `'symm'` | 对称平铺 |

**验证结果**：`np.testing.assert_array_equal(x, y)` 断言两个数组完全相等，不一样直接抛异常——验证手算的卷积结果很方便：

```python
d = np.array([[...]])                        # 手算的期望结果
np.testing.assert_array_equal(c, d)          # 不一致就抛 AssertionError
```

**和深度学习的关系**：CNN 的"卷积"就是这个操作的缩小版——图像和一个小核（如 3×3）做卷积来提取特征。经典的 **Laplacian 边缘检测核**，卷积后数值大的地方就是边缘：

```python
laplacian = np.array([[0,  1, 0],
                      [1, -4, 1],
                      [0,  1, 0]], dtype=np.float32)   # 3×3 二阶导核

# 造一张 64×64 灰度图：中间一个白色方块
image = np.zeros((64, 64))
image[20:44, 20:44] = 255

edges = signal.convolve2d(image, laplacian, mode='same', boundary='fill')
import matplotlib.pyplot as plt
plt.imshow(edges)              # 亮的地方就是方块边缘
```

> PyTorch 里 `nn.Conv2d` 做的也是"图像 × 核 = 特征图"，只是核是**训练学出来的**而不是手写的。PyTorch 篇你会看到：卷积这类高频算子最终交给 cuDNN 优化（autotuner 选最快算法），数据也常摆成对卷积更友好的 `NCHW` 布局（对应 §9.3 的 CHW 四维张量）——这条直觉读 PyTorch 篇会顺很多。

## 10. 学习资源

* 中文教程：https://github.com/jackfrued/Python-100-Days
* 中文书：https://github.com/ethan-funny/explore-python
* 官方中文指南：https://pythonguidecn.readthedocs.io/zh/latest/
* Python Cookbook：https://python3-cookbook.readthedocs.io/zh_CN/latest/
* 英文小抄：https://github.com/gto76/python-cheatsheet

## 11. 延伸阅读：语法糖（Syntactic Sugar）——一个词的前世今生

> 文档里反复出现"语法糖"这个词（如 5.4 装饰器、5.5 with），这里专门讲清楚它的来历、含义和家族。看完以后，任何文档里的"语法糖"你都能自动翻译成"省事的写法记号"。

### 11.1 来历：一个英国老头，1964 年

这个词是 **Peter J. Landin**（英国计算机科学家）在 **1964 年**的论文《The Mechanical Evaluation of Expressions》中首次提出的。原文大意是：

> ...**syntactic sugar** ... a phrase to describe the sweetenings of the surface syntax ...

翻译：一个用来形容"表层语法的甜味剂"的说法。他说的就是：语言表面上那层"甜味剂"——为了让代码更好写、更好看而加的东西，**不改变语言的本质能力，只是让表达更舒服**。

有意思的是，Landin 是 **Lisp 世界**的人。Lisp 的语法极简（就是括号 + 列表），他常年生活在"语法最少"的语言里，所以格外能看出其他语言里哪些是"锦上添花的甜味"、哪些是"本质能力"——就像天天喝白水的人，最清楚什么是糖。

### 11.2 含义一句话

**语法糖 = 只让代码更好写/更好看的简化记号，本身不增加任何新功能。**

Python 里全是例子（左边是语法糖写法，右边是剥掉糖衣后的等价写法——**对比看，左边明显更短**）：

| 语法糖写法 | 还原后的等价写法 | 本质是什么 |
|---|---|---|
| `@timer` | `my_func = timer(my_func)` | 就是普通函数调用 + 赋值 |
| `a + b`（自定义类） | `a.__add__(b)` | 就是一次方法调用 |
| `x[i]` | `x.__getitem__(i)` | 也是方法调用 |
| `with open(...) as f:` | 手动 try/finally + `f.close()` | 就是异常处理 |
| `a if c else b` | `if c: ... else: ...` | 就是 if 语句 |

`@`、`+`、`[]` 这些"看起来是全新语法"的东西，剥开全是"普通函数调用"的糖衣。**语法糖的威力在于：糖衣让代码读起来像自然语言，但语言的能力并没有扩大。**

### 11.3 为什么"糖"是表面的、"盐"是必需的？

因为用的是**调味/防腐**的隐喻，不是"好吃"那个角度：

* **糖 = 可选的美味**：菜里加糖纯粹让味道更好，不加糖菜照样能吃。语法糖的特征是——**去掉它，程序功能一丁点不变**，只是写法变啰嗦。`@timer` 换成 `my_func = timer(my_func)`，跑的代码完全一样。糖是"锦上添花"，可有可无。

* **盐 = 必需的保障**：菜里加盐（防腐）是为了防止东西坏掉；身体缺盐会出问题。语法盐的特征是——**故意强制你写啰嗦/麻烦的东西，逼你想清楚，防止出错**。比如 Java 的 `throws IOException`，写起来麻烦，但强制你"声明这个函数可能抛异常"、逼你考虑异常路径；C 的 `break`，不写就 fall-through 进下一个 case，故意放个坑让你填。

一句话：**糖——不加也能用，加了更甜；盐——不加就出错，加了才安全。**

### 11.4 衍生词：语法糖的家族

有糖就有盐、有衣，这个家族都源自 Landin 那句造词：

| 词 | 出处 | 意思 |
|---|---|---|
| 语法糖（syntactic sugar） | Landin 1964 | 可写可不写，写了更舒服 |
| 语法盐（syntactic salt） | Landin 1965《The Next 700 Programming Languages》 | **强制你写**的东西，故意让人"不舒服"，逼你考虑周全 |
| 语法糖衣（syntactic saccharine） | 后来的说法 | "甜过头、几乎没人用"的糖 |

### 11.5 它引起的争论：糖吃多了蛀牙

"语法糖"一开始就带着褒贬两极：

* **赞**：让代码接近人类语言，表达力强。Python 社区特别喜欢这词——`@`、`with`、列表推导式都是"甜而不腻"的典范。Python 之禅说"Readability counts"（可读性优先），语法糖正是为可读性服务的。

* **贬**：1975 年就有人写文章反问"Is 'syntactic sugar' always sweet?"（语法糖总是甜的吗？）——反对者说"糖吃多了蛀牙"：过度堆砌糖，代码华丽但难懂，新人看 `@decorator` 一头雾水。这也是为什么 Go、Rust 等语言刻意少放糖。

### 11.6 一个冷知识

Landin 还发明了 **J 操作符**和 **SECD 机**（最早的概念性函数式机器），对后来的 **ML、Haskell** 影响深远。他是"函数式编程"的教父级人物——所以"语法糖"这个词带着浓厚的函数式圈子气质：**他们推崇"最小语法 + 强大本质"，把花哨写法一律叫糖。**

### 11.7 对你的实用价值

以后在任何文档看到"语法糖"，自动翻译成"**省事的写法记号，等价于某种更啰嗦的常规写法**"即可。在 Python 里学会"剥糖衣"很有用：遇到看不懂的 `@`、`with`、推导式，先在脑子里把它还原成普通写法，就懂了。