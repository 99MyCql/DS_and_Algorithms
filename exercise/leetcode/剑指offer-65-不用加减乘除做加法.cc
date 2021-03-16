/*
写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。

 

示例:

输入: a = 1, b = 1
输出: 2
 

提示：

a, b 均可能是负数或 0
结果不会溢出 32 位整数

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/
#include <cstdio>

class Solution {
public:
    // int add(int a, int b) {
    //     int ans = 0;
    //     int c = 0;
    //     for (int i = 0; i < 32; i++) {
    //         int temp = 1 << i;
    //         ans = ans | ((a & temp) ^ (b & temp) ^ (c & temp));
    //         printf("%d: %d %d ", i, ans, (a & temp) ^ (b & temp) ^ (c & temp));
    //         if ((((a & temp) == temp) && ((b & temp) == temp))
    //         || (((a & temp) == temp) && ((c & temp) == temp))
    //         || (((b & temp) == temp) && ((c & temp) == temp))) {
    //             if (i < 31) c = temp << 1;
    //         }
    //         printf("%d\n", c);
    //     }
    //     return ans;
    // }

    int add(int a, int b) {
        int sum = a^b, c; // sum = a^b 防止 b 为 0
        while (b != 0) {
            sum = a ^ b;
            c = (a & b & 0x7fffffff) << 1;
            a = sum;
            b = c;
        }
        return sum;
    }
};