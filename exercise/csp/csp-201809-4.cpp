/*
问题描述
　　在一条街上有n个卖菜的商店，按1至n的顺序排成一排，这些商店都卖一种蔬菜。
　　第一天，每个商店都自己定了一个正整数的价格。店主们希望自己的菜价和其他商店的一致，第二天，每一家商店都会根据他自己和相邻商店的价格调整自己的价格。具体的，每家商店都会将第二天的菜价设置为自己和相邻商店第一天菜价的平均值（用去尾法取整）。
　　注意，编号为1的商店只有一个相邻的商店2，编号为n的商店只有一个相邻的商店n-1，其他编号为i的商店有两个相邻的商店i-1和i+1。
　　给定第二天各个商店的菜价，可能存在不同的符合要求的第一天的菜价，请找到符合要求的第一天菜价中字典序最小的一种。
　　字典序大小的定义：对于两个不同的价格序列(a1, a2, ..., an)和(b1, b2, b3, ..., bn)，若存在i (i>=1), 使得ai<bi，且对于所有j<i，aj=bj，则认为第一个序列的字典序小于第二个序列。
输入格式
　　输入的第一行包含一个整数n，表示商店的数量。
　　第二行包含n个正整数，依次表示每个商店第二天的菜价。
输出格式
　　输出一行，包含n个正整数，依次表示每个商店第一天的菜价。
样例输入
8
2 2 1 3 4 9 10 13
样例输出
2 2 2 1 6 5 16 10
数据规模和约定
　　对于30%的评测用例，2<=n<=5，第二天每个商店的菜价为不超过10的正整数；
　　对于60%的评测用例，2<=n<=20，第二天每个商店的菜价为不超过100的正整数；
　　对于所有评测用例，2<=n<=300，第二天每个商店的菜价为不超过100的正整数。
　　请注意，以上都是给的第二天菜价的范围，第一天菜价可能会超过此范围。
*/
#include <cstdio>
#include <cstdlib>

using namespace std;

int second[300]; // 第二天菜价
int first[300]; // 第一天菜价
int count; // 商店的个数

void dfs(int n) {
    int lower, upper, a, b;
    if (n == count) {
        // printf("%d %d\n", n, first[n-1]);
        for (int i=0; i < count; i++) {
            printf("%d ", first[i]);
        }
        exit(0);
    }
    else if (n == 0) {
        lower = 1;
        upper = second[0]*2 - 1;
    }
    else if (n == 1) {
        // printf("%d %d\n", n, first[n-1]);
        a = second[0]*2 - first[0];
        b = 1;
        lower = a > b ? a : b;

        a = (second[0]+1)*2 - first[0];
        b = (second[1]+1)*3 - first[0] - 1;
        upper = a < b ? a : b;
    }
    else if (n == count-1) {
        // printf("%d %d\n", n, first[n-1]);
        a = second[n-1]*3 - first[n-1] - first[n-2];
        b = second[n]*2 - first[n-1];
        lower = a > b ? a : b;

        a = (second[n-1]+1)*3 - first[n-1] - first[n-2];
        b = (second[n]+1)*2 - first[n-1];
        upper = a < b ? a : b;
    }
    else {
        // printf("%d %d\n", n, first[n-1]);
        a = second[n-1]*3 - first[n-1] - first[n-2];
        b = 1;
        lower = a > b ? a : b;

        a = (second[n-1]+1)*3 - first[n-1] - first[n-2];
        b = (second[n]+1)*3 - first[n-1] - 1;
        upper = a < b ? a : b;
    }
    // printf("lower: %d, upper: %d\n", lower, upper);
    for (int i=lower; i < upper; i++) {
        first[n] = i;
        dfs(n+1);
    }
}

int main() {
    scanf("%d", &count);
    for (int i=0; i < count; i++) {
        scanf("%d", &second[i]);
    }
    dfs(0);
}