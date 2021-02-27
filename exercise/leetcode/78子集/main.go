/*
给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。

说明：解集不能包含重复的子集。

示例:

输入: nums = [1,2,3]
输出:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]
*/
package main

func fun(nums []int, n int) [][]int {
	if nums == nil {
		return nil
	} else if n == 0 {
		return [][]int{{}}
	}

	var ans [][]int
	for i, v := range nums {
		ret := fun(nums[i+1:], n-1)
		if ret != nil {
			for _, set := range ret {
				ans = append(ans, append([]int{v}, set...))
			}
		}
	}

	return ans
}

func subsets(nums []int) [][]int {
	ans := [][]int {{}, nums}

	length := len(nums)
	for i := 1; i < length; i++ {
		ans = append(ans, fun(nums, i)...)
	}

	return ans
}