/*
给定一个数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

返回滑动窗口中的最大值。

 

进阶：

你能在线性时间复杂度内解决此题吗？

 

示例:

输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
输出: [3,3,5,5,6,7]
解释:

  滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
 

提示：

1 <= nums.length <= 10^5
-10^4 <= nums[i] <= 10^4
1 <= k <= nums.length

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/sliding-window-maximum
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

#include <vector>
#include <deque>

using namespace std;

class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int len = nums.size();

        vector<int> ans;      // 结果数组
        deque<int> pos_deque; // 保存 nums 数组下标的双向队列，队头到队尾单调递减，即 nums[pos_deque[i-1]] > nums[pos_deque[i]]

        // 先处理第一个窗口（前K个数）
        for (int i=0; i<k; i++) {
            // 如果 双向队列不为空 且 队尾下标指向的元素小于当前元素，则弹出队尾元素
            while (!pos_deque.empty() && nums[pos_deque.back()] < nums[i]) {
                pos_deque.pop_back();
            }
            pos_deque.push_back(i); // 最后将当前元素的下标压入队尾
        }
        ans.push_back(nums[pos_deque.front()]); // 向结果数组中压入第一个窗口的最大值

        // 依次处理后续窗口
        for (int i=k; i<len; i++) {
            // 如果队头不在当前窗口中，则弹出队头元素
            if (pos_deque.front() == i-k) {
                pos_deque.pop_front();
            }

            // 如果 双向队列不为空 且 队尾下标指向的元素小于当前元素，则弹出队尾元素
            while (!pos_deque.empty() && nums[pos_deque.back()] < nums[i]) {
                pos_deque.pop_back();
            }
            pos_deque.push_back(i);                 // 将当前元素的下标压入队尾
            ans.push_back(nums[pos_deque.front()]); // 向结果数组中压入当前窗口的最大值
        }

        return ans;
    }
};