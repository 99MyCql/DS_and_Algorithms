/*
给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m] 。请问 k[0]*k[1]*...*k[m] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

示例 1：

输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1
示例 2:

输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
提示：

2 <= n <= 58

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/jian-sheng-zi-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/
#include <vector>
#include <cstdio>

using namespace std;

class Solution {
public:
    vector<int> dp; // 下标i处的值表示：i~n 段绳子最大乘积

    // 获取 [start, end) 段绳子的最大乘积（绳子至少被剪一刀）
    int getMaxMul(int start, int end) {
        if (dp[start]) {
            return dp[start];
        }

        // i 表示剪一刀的位置，其区间为[start+1, end)。start+1 是因为不可能剪头部
        // 绳子的最大乘积可以分解为：
        // 1. (i-start) * getMaxMul(i, end)
        // 2. 右半部分不剪：(i-start) * (end-i)
        // 注意：不能转换成 getMaxMul(start, i)*getMaxMul(i, end)
        int max = 0;
        for (int i=start+1; i<end; i++) {
            int temp = getMaxMul(i, end);
            int mul = (i-start) * (temp > (end-i) ? temp : (end-i));
            if (mul > max) {
                max = mul;
            }
        }

        dp[start] = max;
        printf("start:%d, end-1:%d, max:%d\n", start, end-1, max);
        return max;
    }

    // 解题接口
    int cuttingRope(int n) {
        for (int i=0; i<n; i++) {
            dp.push_back(0);
        }

        return getMaxMul(0, n);
    }
};