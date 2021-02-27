/*
地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

 

示例 1：

输入：m = 2, n = 3, k = 1
输出：3
示例 1：

输入：m = 3, n = 1, k = 0
输出：1
提示：

1 <= n,m <= 100
0 <= k <= 20

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/
#include <queue>
#include <cstdio>

using namespace std;

typedef struct {
    int i;
    int j;
} Pos;

class Solution {
public:
    int g_m;    // 行长
    int g_n;    // 列长
    int g_k;    // 行列坐标位数之和的最大值
    vector<vector<bool>> flag;  // 标志位，为true表示已经走过

    // 初始化标志位
    void initFlag() {
        for (int i=0; i < g_m; i++) {
            vector<bool> temp;
            for (int j=0; j < g_n; j++) {
                temp.push_back(false);
            }
            flag.push_back(temp);
        }
    }

    // 判断当前坐标是否符合标准
    bool check(int i, int j) {
        if (i >= 0 && i < g_m && j >=0 && j < g_n && !flag[i][j]) {
            int sum = 0;
            while (i) {
                sum += i%10;
                i /= 10;
            }
            while (j) {
                sum += j%10;
                j /= 10;
            }
            if (sum <= g_k) {
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }

    // 解题接口
    int movingCount(int m, int n, int k) {
        g_m = m;
        g_n = n;
        g_k = k;
        initFlag();

        // BFS
        int ans = 0;
        queue<Pos> q;
        q.push(Pos{0, 0});
        flag[0][0] = true;
        ans++;
        while (!q.empty()) {
            Pos temp = q.front();
            q.pop();
            printf("i:%d, j:%d\n", temp.i, temp.j);
            if (check(temp.i-1, temp.j)) {
                q.push(Pos{temp.i-1, temp.j});
                flag[temp.i-1][temp.j] = true;
                ans++;
            }
            if (check(temp.i+1, temp.j)) {
                q.push(Pos{temp.i+1, temp.j});
                flag[temp.i+1][temp.j] = true;
                ans++;
            }
            if (check(temp.i, temp.j-1)) {
                q.push(Pos{temp.i, temp.j-1});
                flag[temp.i][temp.j-1] = true;
                ans++;
            }
            if (check(temp.i, temp.j+1)) {
                q.push(Pos{temp.i, temp.j+1});
                flag[temp.i][temp.j+1] = true;
                ans++;
            }
        }
        return ans;
    }
};