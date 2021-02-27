/*
给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

说明：每次只能向下或者向右移动一步。

示例:

输入:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 7
解释: 因为路径 1→3→1→1→1 的总和最小。
*/
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;

#define min(a, b) (a < b ? a : b)

class Solution {
public:
    int **dp;
    int m;
    int n;

    int minPathSum(vector<vector<int>>& grid) {
        m = grid.size();
        n = grid[0].size();
        dp = (int**)malloc(sizeof(int*) * 2); // 此题动态规划只需要两行即可
        for (int i=0; i < 2; i++) {
            dp[i] = (int*)malloc(sizeof(int) * n);
        }

        dp[0][0] = grid[0][0];
        for (int i=0; i < m; i++) {
            for (int j=0; j < n; j++) {
                if (i == 0 && j == 0) continue;
                if (i == 0) {
                    dp[i%2][j] = grid[i][j] + dp[i%2][j-1];
                    continue;
                }
                if (j == 0) {
                    dp[i%2][j] = grid[i][j] + dp[(i-1)%2][j];
                    continue;
                }

                dp[i%2][j] = grid[i][j] + min(dp[(i-1)%2][j], dp[i%2][j-1]);
            }
        }

        printf("%d\n", dp[(m-1)%2][n-1]);
        return dp[(m-1)%2][n-1];
    }
};