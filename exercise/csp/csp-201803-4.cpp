/*
Alice和Bob正在玩井字棋游戏。
　　井字棋游戏的规则很简单：两人轮流往3*3的棋盘中放棋子，Alice放的是“X”，Bob放的是“O”，Alice执先。当同一种棋子占据一行、一列或一条对角线的三个格子时，游戏结束，该种棋子的持有者获胜。当棋盘被填满的时候，游戏结束，双方平手。
　　Alice设计了一种对棋局评分的方法：
　　- 对于Alice已经获胜的局面，评估得分为(棋盘上的空格子数+1)；
　　- 对于Bob已经获胜的局面，评估得分为 -(棋盘上的空格子数+1)；
　　- 对于平局的局面，评估得分为0；


　　例如上图中的局面，Alice已经获胜，同时棋盘上有2个空格，所以局面得分为2+1=3。
　　由于Alice并不喜欢计算，所以他请教擅长编程的你，如果两人都以最优策略行棋，那么当前局面的最终得分会是多少？
输入格式
　　输入的第一行包含一个正整数T，表示数据的组数。
　　每组数据输入有3行，每行有3个整数，用空格分隔，分别表示棋盘每个格子的状态。0表示格子为空，1表示格子中为“X”，2表示格子中为“O”。保证不会出现其他状态。
　　保证输入的局面合法。(即保证输入的局面可以通过行棋到达，且保证没有双方同时获胜的情况)
　　保证输入的局面轮到Alice行棋。
输出格式
　　对于每组数据，输出一行一个整数，表示当前局面的得分。
样例输入
3
1 2 1
2 1 2
0 0 0
2 1 1
0 2 1
0 0 2
0 0 0
0 0 0
0 0 0
样例输出
3
-4
0
样例说明
　　第一组数据：
　　Alice将棋子放在左下角(或右下角)后，可以到达问题描述中的局面，得分为3。
　　3为Alice行棋后能到达的局面中得分的最大值。
　　第二组数据：


　　Bob已经获胜(如图)，此局面得分为-(3+1)=-4。
　　第三组数据：
　　井字棋中若双方都采用最优策略，游戏平局，最终得分为0。
数据规模和约定
　　对于所有评测用例，1 ≤ T ≤ 5。
*/
#include <cstdio>
#include <cmath>
#include <vector>

using namespace std;

int chessboard[9];
vector<int> zero_pos;
int len;
int zero_count;

// 检查是否有人已经获得了胜利，若有则返回胜者编号
int check(int *chessboard) {
    // 三横
    for (int i = 0; i < 3; i++) {
        if (chessboard[3*i] != 0 && chessboard[3*i] == chessboard[1+3*i] && chessboard[1+3*i] == chessboard[2+3*i]) {
            return chessboard[3*i];
        }
    }
    // 三竖
    for (int i = 0; i < 3; i++) {
        if (chessboard[i] != 0 && chessboard[i] == chessboard[3+i] && chessboard[3+i] == chessboard[6+i]) {
            return chessboard[i];
        }
    }
    // 交叉
    if (chessboard[0] != 0 && chessboard[0] == chessboard[4] && chessboard[4] == chessboard[8]) {
        return chessboard[0];
    }
    if (chessboard[2] != 0 && chessboard[2] == chessboard[4] && chessboard[4] == chessboard[6]) {
        return chessboard[2];
    }

    return 0;
}

// dfs(): 返回当前棋手能得到的最好分值
// cur: 1 Alice | 2 Bob
int dfs(int cur) {
    int max_score = -10;
    int min_score = 10;
    if (zero_count == 0) return 0; // 平手
    // 考虑所有空位置
    for (int i=0; i < len; i++) {
        int p = zero_pos[i];
        if (chessboard[p] == 0) {
            chessboard[p] = cur;
            zero_count--;
            int win = check(chessboard);
            if (win) {
                int score = win == 1 ? (zero_count+1) : -(zero_count+1);
                cur == 1 ? (max_score = max(max_score, score)) : (min_score = min(min_score, score));
            }
            else {
                cur == 1 ? (max_score = max(max_score, dfs(2))) : (min_score = min(min_score, dfs(1))); // 对抗。两人都以最优策略行棋
            }
            chessboard[p] = 0;
            zero_count++;
        }
    }
    return cur == 1 ? max_score : min_score;
}

int main() {
    int n;
    scanf("%d", &n);
    for (int i=0; i < n; i++) {
        for (int i=0; i < 9; i++) {
            chessboard[i] = 0;
            scanf("%d", &chessboard[i]);
            if (chessboard[i] == 0) zero_pos.push_back(i);
        }

        // 先检查是不是已经决出了胜负
        int win = check(chessboard);
        if (win) {
            int score = win == 1 ? (zero_count+1) : -(zero_count+1);
            printf("%d\n", score);
        }
        else {
            zero_count = len = zero_pos.size();
            // 全0则平局
            if (zero_count == 9) printf("0\n");
            else printf("%d\n", dfs(1));
        }
        zero_pos.clear();
    }
}