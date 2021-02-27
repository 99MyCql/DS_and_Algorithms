/*
数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从下标0开始计数）是5，第13位是1，第19位是4，等等。

请写一个函数，求任意第n位对应的数字。

 

示例 1：

输入：n = 3
输出：3
示例 2：

输入：n = 11
输出：0
 

限制：

0 <= n < 2^31

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/
#include <cstdio>

class Solution {
public:
    int getNumCount(int num) {
        if (num == 0) return 1;

        int count = 0;
        while (num) {
            num = num / 10;
            count++;
        }
        return count;
    }
    int findNthDigit(int n) {
        long long i = 0;
        int num = 0;        // 当前数
        int num_count = 0;  // 当前数的位数
        int pre_num = 0;    // 上一个数
        while (i <= n) {
            pre_num = num;
            if (num % 10 == 0  && num_count < getNumCount(num)) {
                num_count++;
            }
            i += num_count;
            num++;
        }
        printf("i:%d, pre_num:%d\n", i, pre_num);
        while (i-n-1) {
            pre_num = pre_num/10;
            i--;
        }
        return pre_num%10;
    }
};