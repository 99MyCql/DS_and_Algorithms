/*
实现函数double Power(double base, int exponent)，求base的exponent次方。不得使用库函数，同时不需要考虑大数问题。

 

示例 1:

输入: 2.00000, 10
输出: 1024.00000
示例 2:

输入: 2.10000, 3
输出: 9.26100
示例 3:

输入: 2.00000, -2
输出: 0.25000
解释: 2-2 = 1/22 = 1/4 = 0.25
 

说明:

-100.0 < x < 100.0
n 是 32 位有符号整数，其数值范围是 [−2^31, 2^31 − 1] 。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/
#include <cstdio>

class Solution {
public:
    double myPosFun(double x, int n) {
        printf("n:%d\n", n);
        if (n == 0) return 1;

        int m = n/2;
        double ans = myPosFun(x, m);
        // 幂n为偶数时，pow(x,n) = pow(x,n/2) * pow(x,n/2) 一直递归，可以将时间复杂度n缩短为logn
        if (n%2 == 0) {
            ans = ans*ans;
        }
        // 幂n为奇数时，pow(x,n) = pow(x,n/2) * pow(x,n/2) * x
        else {
            ans = ans*ans*x;
        }

        printf("ans:%f\n", ans);
        return ans;
    }

    double myPow(double x, int n) {
        if (n >= 0) {
            return myPosFun(x, n);
        } else {
            return 1/myPosFun(x, n);
        }
    }
};