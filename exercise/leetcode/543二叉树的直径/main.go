/*
给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过根结点。

示例 :
给定二叉树

          1
         / \
        2   3
       / \
      4   5
返回 3, 它的长度是路径 [4,2,1,3] 或者 [5,2,1,3]。

注意：两结点之间的路径长度是以它们之间边的数目表示。
*/
package main


type TreeNode struct {
    Val int
    Left *TreeNode
    Right *TreeNode
}

func diameterOfBinaryTree(root *TreeNode) int {
    if root == nil {
        return 0
    } else {
        _, ans := getDeepAndDia(root)
        return ans
    }
}

func getDeepAndDia(root *TreeNode) (int, int) {
    if root == nil {
        return -1, -1
    }

    leftDeep, leftDia := getDeepAndDia(root.Left)
    rightDeep, rightDia := getDeepAndDia(root.Right)

    return max(leftDeep, rightDeep)+1, max(leftDeep+rightDeep+2, max(leftDia, rightDia))
}

func max(a int, b int) int {
    if a >= b {
        return a
    } else {
        return b
    }
}