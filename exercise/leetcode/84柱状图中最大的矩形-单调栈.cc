/*
给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

 



以上是柱状图的示例，其中每个柱子的宽度为 1，给定的高度为 [2,1,5,6,2,3]。

 



图中阴影部分为所能勾勒出的最大矩形面积，其面积为 10 个单位。

 

示例:

输入: [2,1,5,6,2,3]
输出: 10

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/largest-rectangle-in-histogram
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

#include <iostream>
#include <vector>
#include <stack>

using namespace std;

class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        heights.push_back(-1);  // 方便最后处理栈中剩余数据
        int len = heights.size();

        stack<int> s;   // 单调栈：栈中第i个元素与第i-1个元素的关系为 heights[i] > heights[i-1]
        int max = 0;    // 最大矩形的面积
        for (int i=0; i<len; i++) {
            // 如果栈为空，或者当前值大于栈顶值，则直接入栈
            if (s.empty() || heights[i] >= heights[s.top()]) {
                s.push(i);
            } else {
                // 当前值小于栈顶值，则不断pop栈顶，直到栈为空，或当前值大于栈顶值
                while (!s.empty() && heights[i] < heights[s.top()]) {
                    // 每弹出一次栈顶，就计算以 heights[s.top()] 为高的矩形的最大值。
                    // 进而，我们需要知道矩形宽的最大值是：左边第一个小于 heights[s.top()] 的柱子，
                    // 和右边第一个小于 heights[s.top()] 的柱子，之间所夹的宽值。
                    int temp = s.top();
                    s.pop();
                    int area = 0;
                    if (s.empty()) {
                        area = i * heights[temp];
                    } else {
                        area = (i - s.top() - 1) * heights[temp];
                    }

                    // 以 heights[s.top()] 为高的矩形的最大值是否大于柱状图中最大的矩形面积
                    if (area > max) {
                        max = area;
                    }
                }
                s.push(i);
            }
        }

        return max;
    }
};