# csp 刷题记录

[TOC]

## 2014-12-4 最优灌溉 100

Kruskal，最小生成树。

## 2017-03-4 地铁修建 100

Kruskal。

将所有边从小到大排序，然后根据Kruskal算法，构建最小生成树。过程中，记录最小生成树中边的最大值，当第n个点加入集合时，输出记录的最大值。

## 2017-09-3 JSON查询 90

采用树的数据结构对数据进行储存。

## 2017-12-4 行车路线 70

Dijkstra + Floyd。

## 2018-03-1 跳一跳 100

仔细审题。

发现一处错误时，还需要寻找有没有另一处**连带错误**。

## 2018-03-4 棋局评估 75

博弈论 + DFS

参考：[https://blog.csdn.net/IoT_fast/article/details/82556785](https://blog.csdn.net/IoT_fast/article/details/82556785)

问题关键在于：

> 两人都以最优策略行棋

定义一个`int dfs(int cur)`函数，它返回当前棋手(cur)所能得到的最好分值。终止条件是没有空位置，或遍历完所有空位置。

对于当前棋手，考虑所有空位置能得到的分值，并取其中的最好的返回。

确定一个空位置下棋后，若已经胜利则直接与最大值比较取最大值；若还未胜出，则交给下一个棋手(`dfs(下一个棋手)`)，返回他能得到的最好分值，再与最大值比较。

## 2018-09-4 再买菜 70

DFS。

对于第`i>=2`(从0开始)家店，根据第`i-1`、`i-2`家店第一二天的价格，可以确定一个范围。遍历范围中的可能值，类似于八皇后。

## 2019-03-2 二十四点 100

栈。

这道题由于情况较少，可以直接暴力破解，三个运算符优先级全排列即可。

我所用的方法，是比较蠢的通用表达式求值方法。

建立两个栈：操作数栈和运算符栈。运算符栈的规则是，栈底运算符优先级 `<` 栈顶优先级；每次运算符入栈时，如果栈顶元素优先级高于要入栈的运算符，则先将栈顶运算符弹出进行运算。

## 2019-09-4 推荐系统 100

考试时用链表0分，正解应该是**集合(set)**。

集合的优势在于：

- 集合内部由**红黑树（平衡二叉树）**实现，可自定义排序规则。

- 插入即排序。

- 查询时间为`O(商品总数)`。

同时，为了方便删除，还需要使用**映射(map)**，`key`为商品类别和编号，`value`为商品的评分，若删除则设为`-1`。

参考博客：[https://blog.csdn.net/qq_39475280/article/details/101642741](https://blog.csdn.net/qq_39475280/article/details/101642741)

## 2019-12-3 化学方程式 70

编译原理递归下降。

编译原理类型题，给出化学方程式文法，进行元素配平。
