/*
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

 

示例 1：

输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
示例 2：

输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]
 

限制：

0 <= matrix.length <= 100
0 <= matrix[i].length <= 100

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/
#include <vector>

using namespace std;

class Solution {
public:
    // 参考题解后的解法
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> ans;
        int n = matrix.size();
        if (n == 0) return ans;
        int m = matrix[0].size();

        int l = 0, r = m-1, t = 0, b = n-1;
        while (1) {
            for (int i = l; i <= r; i++) ans.push_back(matrix[t][i]);
            t += 1;
            if (t > b) break;

            for (int i = t; i <= b; i++) ans.push_back(matrix[i][r]);
            r -= 1;
            if (r < l) break;

            for (int i = r; i >= l; i--) ans.push_back(matrix[b][i]);
            b -= 1;
            if (b < t) break;

            for (int i = b; i >= t; i--) ans.push_back(matrix[i][l]);
            l += 1;
            if (l > r) break;
        }
        return ans;
    }

    // 我的解法
    // vector<int> spiralOrder(vector<vector<int>>& matrix) {
    //     vector<int> ans;
    //     int n = matrix.size();
    //     if (n == 0) return ans;
    //     int m = matrix[0].size();

    //     for (int k = 0, i = 0, j = 0; k < n-k && k < m-k; k++) {
    //         i = k;
    //         j = k;
    //         int tempi = i, tempj = j;
    //         while (j < m-1-k) {
    //             ans.push_back(matrix[i][j]);
    //             j++;
    //         }
    //         while (i < n-1-k) {
    //             ans.push_back(matrix[i][j]);
    //             i++;
    //         }

    //         if (tempi == n-1-k) {
    //             ans.push_back(matrix[i][j]);
    //             break;
    //         }
    //         while (j > k) {
    //             ans.push_back(matrix[i][j]);
    //             j--;
    //         }

    //         if (tempj == m-1-k) {
    //             ans.push_back(matrix[i][j]);
    //             break;
    //         }
    //         while (i > k) {
    //             ans.push_back(matrix[i][j]);
    //             i--;
    //         }

    //         if (tempi == n-1-k && tempj == m-1-k) ans.push_back(matrix[i][j]);
    //     }
    //     return ans;
    // }
};
