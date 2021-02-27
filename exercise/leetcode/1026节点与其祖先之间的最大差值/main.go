/*
给定二叉树的根节点 root，找出存在于不同节点 A 和 B 之间的最大值 V，其中 V = |A.val - B.val|，且 A 是 B 的祖先。

（如果 A 的任何子节点之一为 B，或者 A 的任何子节点是 B 的祖先，那么我们认为 A 是 B 的祖先）



示例：



输入：[8,3,10,1,6,null,14,null,null,4,7,13]
输出：7
解释：
我们有大量的节点与其祖先的差值，其中一些如下：
|8 - 3| = 5
|3 - 7| = 4
|8 - 1| = 7
|10 - 13| = 3
在所有可能的差值中，最大值 7 由 |8 - 1| = 7 得出。


提示：

树中的节点数在 2 到 5000 之间。
每个节点的值介于 0 到 100000 之间。
*/
package main

import "fmt"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func main() {}

func maxAncestorDiff(root *TreeNode) int {
	_, _, ans := fun(root)
	return ans
}

// 返回该树各结点中的最大值、最小值和该树的最大差值
func fun(root *TreeNode) (int, int, int) {
	var maxVal = 0
	var minVal = 100000
	var diff = 0

	if root == nil {
		return maxVal, minVal, diff
	}

	fmt.Printf("Val:%d\n", root.Val)

	leftMax, leftMin, leftDiff := fun(root.Left)
	rightMax, rightMin, rightDiff := fun(root.Right)

	maxVal = max(root.Val, max(leftMax, rightMax))
	minVal = min(root.Val, min(leftMin, rightMin))

	if root.Left != nil {
		diff = max(leftDiff, max(abs(root.Val-leftMax), abs(root.Val-leftMin)))
	}

	if root.Right != nil {
		diff = max(diff, max(rightDiff, max(abs(root.Val-rightMax), abs(root.Val-rightMin))))
	}

	fmt.Printf("Val:%d--->max:%d, min:%d, diff:%d\n", root.Val, maxVal, minVal, diff)
	return maxVal, minVal, diff
}

// 取绝对值
func abs(x int) int {
	if x < 0 {
		x = -x
	}
	return x
}

func min(a int, b int) int {
	if a <= b {
		return a
	}
	return b

}

func max(a int, b int) int {
	if a >= b {
		return a
	}
	return b
}
