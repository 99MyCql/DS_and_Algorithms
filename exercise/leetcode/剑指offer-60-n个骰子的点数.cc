/*
把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。

 

你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率。

 

示例 1:

输入: 1
输出: [0.16667,0.16667,0.16667,0.16667,0.16667,0.16667]
示例 2:

输入: 2
输出: [0.02778,0.05556,0.08333,0.11111,0.13889,0.16667,0.13889,0.11111,0.08333,0.05556,0.02778]
 

限制：

1 <= n <= 11

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/
#include <vector>
#include <cstdio>

using namespace std;

class Solution {
public:
    vector<double> dicesProbability(int n) {
        vector<double> ans;

        int dp[n+1][6*n+1];
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= 6*n; j++) {
                dp[i][j] = 0;
            }
        }
        dp[0][0] = 1;

        for (int i = 1; i <= n; i++) {
            for (int j = i; j <= 6*n; j++) {
                for (int k = 1; k <= 6; k++) {
                    if (k > j) break;
                    dp[i][j] += dp[i-1][j-k];
                }
                printf("d[%d][%d]:%d\n", i, j, dp[i][j]);
            }
        }

        int sum = 1;
        for (int i = 0; i < n; i++) sum *= 6;
        for (int i = n; i <= 6*n; i++) {
            ans.push_back((double)dp[n][i] / sum);
        }
        return ans;
    }
};