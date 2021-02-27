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

using namespace std;

class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        int len = heights.size();
        int max = 0;
        for (int i=0; i<len; i++) {
            int min = heights[i];   // i ~ j 间高的最小值
            for (int j=i; j<len; j++) {
                if (heights[j] < min) min = heights[j];
                if ((j-i+1)*min > max) max = (j-i+1)*min;   // 计算宽为 j-i，高为 min 的矩形面积
            }
        }
        return max;
    }
};