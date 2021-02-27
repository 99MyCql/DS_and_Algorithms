/*
给定一个仅包含 0 和 1 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。

示例:

输入:
[
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
输出: 6

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/maximal-rectangle
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

#include <vector>
#include <stack>
#include <iostream>

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

    int maximalRectangle(vector<vector<char>>& matrix) {
        int row_len = matrix.size();
        if (row_len == 0) return 0;
        int col_len = matrix[0].size();

        vector<vector<int>> matrix_count;
        for (int i=0; i < row_len; i++) {
            vector<int> temp;
            for (int j=0; j < col_len; j++) {
                if (matrix[i][j] == '0') temp.push_back(0);
                else if (matrix[i][j] == '1') {
                    if (i == 0) temp.push_back(1);
                    else temp.push_back(matrix_count[i-1][j] + 1);
                }
                cout << temp[j] << " ";
            }
            matrix_count.push_back(temp);
            cout << endl;
        }

        int max = 0;
        for (int i=0; i < row_len; i++) {
            int temp = largestRectangleArea(matrix_count[i]);
            if (temp > max) max = temp;
        }

        return max;
    }
};