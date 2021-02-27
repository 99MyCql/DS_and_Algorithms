/*
n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。



上图为 8 皇后问题的一种解法。

给定一个整数 n，返回 n 皇后不同的解决方案的数量。

示例:

输入: 4
输出: 2
解释: 4 皇后问题存在如下两个不同的解法。
[
 [".Q..",  // 解法 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // 解法 2
  "Q...",
  "...Q",
  ".Q.."]
]
*/
#include <cstdio>
#include <iostream>
#include <stack>

using namespace std;

class Solution {
public:
    int g_count = 0;    // 解法种数

    // 打印棋盘
    // 1 ---> 0 4 7 5 2 6 1 3
    // 第1中情况 ---> 第0行第0列 第1行第4列 第2行第7列 ······
    void show(int *chess, int n) {
        g_count++;
        printf("%d ---> ", g_count);
        for(int i=0; i<n; i++) {
            printf("%d ", chess[i]);
        }
        printf("\n");
    }

    // 判断是否满足八皇后要求：同一行、同一列、同一斜线上只能有一个皇后
    bool check(int *chess, int row) {
        for (int i = 0; i <= row; i++) {
            for (int j = i+1; j <= row; j++) {
                if (chess[i] == chess[j] || chess[j]-chess[i] == j-i || chess[i]-chess[j] == j-i) {
                    return false;
                }
            }
        }
        return true;
    }

    int totalNQueens(int n) {
        int chess[n+1]; // 棋盘。i: 表示第i行；chess[i]: 表示第i行第chess[i]列（都是从0开始）
        stack<int> s;   // 栈上储存着每行选第几列
        int row = 0;    // 记录现在是第几行

        chess[row] = -1;        // -1表示这一行还未开始摆放皇后
        s.push(chess[row]);
        while (!s.empty()) {
            int top = s.top();  // 取出栈顶元素，即当前行的第top列

            bool flag = false;  // 当前行第top+1列至第n-1列，是否有满足要求的列

            // 从第top+1列到第n-1列，依次尝试摆放皇后
            for (int i=top+1; i<n; i++) {
                chess[row] = i;

                // 如果在当前列摆放皇后能满足要求
                if (check(chess, row)) {
                    // cout << row << ": " << chess[row] << endl;
                    // 更新栈顶元素
                    s.pop();
                    s.push(chess[row]);

                    // 如果当前行已经是第n-1行，则打印棋盘
                    if (row == n-1) {
                        show(chess, n);
                    }
                    // 否则将下一行压入栈。并在下一次while中，从下一行的第0列开始尝试
                    else {
                        row++;
                        chess[row] = -1;
                        s.push(chess[row]);
                    }

                    flag = true;
                    break;  // 跳出循环
                }
            }

            // 当前行第top+1列至第n-1列，不存在满足要求的列，则退回到上一行
            if (!flag) {
                s.pop();
                row--;
            }
        }
        return this->g_count;
    }
};