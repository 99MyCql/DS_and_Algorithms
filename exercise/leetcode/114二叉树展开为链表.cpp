/**
给定一个二叉树，原地将它展开为链表。

例如，给定二叉树

    1
   / \
  2   5
 / \   \
3   4   6
将其展开为：

1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6


 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
#include <cstdio>

using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    // 将树原地转换成链表，并返回最后一个元素
    TreeNode* changeToList(TreeNode* root) {
        if (root == NULL) return NULL;
        if (root->left == NULL && root->right == NULL) return root;

        TreeNode* right = root->right, *left = root->left;
        root->left = NULL;
        root->right = left;
        TreeNode* left_end = changeToList(left);
        if (left_end != NULL) {
            left_end->right = right;
        }
        else {
            root->right = right;
        }

        if (right != NULL) {
            return changeToList(right);
        }
        else {
            return left_end;
        }
    }

    void flatten(TreeNode* root) {
        changeToList(root);
        printf("ok\n");
        TreeNode* p = root;
        while (p) {
            printf("%d %d %d\n", p->val, p->left, p->right);
            p = p->right;
        }
    }
};