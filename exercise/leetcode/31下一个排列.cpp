/*
实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。

如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

必须原地修改，只允许使用额外常数空间。

以下是一些例子，输入位于左侧列，其相应输出位于右侧列。
1,2,3 → 1,3,2
3,2,1 → 1,2,3
1,1,5 → 1,5,1
*/
#include <cstdio>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        unsigned int len = nums.size();

        bool isExist = false;
        int min_pos = -1;
        int swap_pos = 0;
        for (int i = len-1; i >= 0; i--) {
            // 向前找比nums[i]小的数
            for (int j = i-1; j >= 0; j--) {
                if (nums[j] < nums[i] && j > min_pos) {
                    min_pos = j;
                    swap_pos = i;
                    break;
                }
            }
        }

        // 如果找到，则交换，并将less_pos后的数升序排序
        if (min_pos != -1) {
            int temp = nums[swap_pos];
            nums[swap_pos] = nums[min_pos];
            nums[min_pos] = temp;
            sort(nums.begin()+min_pos+1, nums.end());
            isExist = true;
        }

        // 如果不存在更大的字典序，则变成降序
        if (!isExist) {
            for (int i = 0; i < len/2; i++) {
                int temp = nums[i];
                nums[i] = nums[len-i-1];
                nums[len-i-1] = temp;
            }
        }
    }
};
