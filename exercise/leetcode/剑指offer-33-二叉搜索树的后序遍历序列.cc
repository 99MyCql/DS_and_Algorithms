/*
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。

 

参考以下这颗二叉搜索树：

     5
    / \
   2   6
  / \
 1   3
示例 1：

输入: [1,6,3,2,5]
输出: false
示例 2：

输入: [1,3,2,6,5]
输出: true
 

提示：

数组长度 <= 1000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/
#include <vector>

using namespace std;

class Solution {
public:
    bool verifyPostorder(vector<int>& postorder) {
        int n = postorder.size();
        if (n == 0 || n == 1) return true;

        int root = postorder[n-1];
        vector<int> front, back;
        int i = 0;
        while (i < n-1 && postorder[i] < root) {
            front.push_back(postorder[i]);
            i++;
        }
        while (i < n-1 && postorder[i] > root) {
            back.push_back(postorder[i]);
            i++;
        }

        if (i != n-1) return false;
        return verifyPostorder(front) && verifyPostorder(back);
    }
};