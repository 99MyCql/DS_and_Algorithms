/*
输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

 

例如，给出

前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
返回如下的二叉树：

    3
   / \
  9  20
    /  \
   15   7
 

限制：

0 <= 节点个数 <= 5000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/
#include <vector>
#include <cstdio>
#include <cstdlib>

using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int len = preorder.size();
        for (int i=0; i < len; i++) {
            printf("%d, ", preorder[i]);
        }
        printf("\n");
        for (int i=0; i < len; i++) {
            printf("%d, ", inorder[i]);
        }
        printf("\n");
        if (len == 0) {
            return NULL;
        }

        int root_val = preorder[0];
        int i = 0;
        for (; i < len; i++) {
            if (root_val == inorder[i]) {
                break;
            }
        }

        int left_len = i;
        int right_len = len-i-1; // 0 1  2  3 4
        vector<int> left_preorder(preorder.begin()+1, preorder.begin()+1+left_len);
        vector<int> left_inorder(inorder.begin(), inorder.begin()+left_len);
        vector<int> right_preorder(preorder.begin()+1+left_len, preorder.end());
        vector<int> right_inorder(inorder.begin()+left_len+1, inorder.end());

        TreeNode *root = (TreeNode*)malloc(sizeof(TreeNode));
        root->val = root_val;
        root->left = buildTree(left_preorder, left_inorder);
        root->right = buildTree(right_preorder, right_inorder);
        return root;
    }
};