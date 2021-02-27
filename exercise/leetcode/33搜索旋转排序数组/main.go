/*
假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。

搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。

你可以假设数组中不存在重复的元素。

你的算法时间复杂度必须是 O(log n) 级别。

示例 1:

输入: nums = [4,5,6,7,0,1,2], target = 0
输出: 4

示例 2:

输入: nums = [4,5,6,7,0,1,2], target = 3
输出: -1
*/
package main

import "fmt"

func search(nums []int, target int) int {

	fmt.Println(nums)

    length := len(nums)
    if length == 1 && nums[0] == target {
    	return 0
    } else if length <= 1 {
        return -1
    }

    mid := length/2 // 中间
    temp := -1      // 暂存下一层递归的返回值
    if nums[0] == target {
        return 0
    } else if target > nums[0] {
        if target < nums[mid] || nums[mid] < nums[0] {
            temp = search(nums[0:mid], target)
        } else {
            temp = search(nums[mid:], target)
            if temp != -1 {
            	temp += mid
            }
        }
    } else {
        if target == nums[length-1] {
            return length-1
        } else if target < nums[length-1] {
        	// 必须使用 >= ，因为递归调用中，传入的切片包含 nums[mid]
            if target >= nums[mid] || nums[mid] > nums[length-1] {
                temp = search(nums[mid:], target)
                if temp != -1 {
                	temp += mid
                }
            } else {
                temp = search(nums[0:mid], target)
            }
        } else {
            return -1
        }
    }

    if temp == -1 {
        return -1
    } else {
        return temp
    }
}
