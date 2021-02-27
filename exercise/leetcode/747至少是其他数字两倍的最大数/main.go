package main

func dominantIndex(nums []int) int {
    var (
        max = 0
        max_p = -1
        sec_max = 0
    )
    for i := 0; i < len(nums); i++ {
        if nums[i] > max {
            sec_max = max
            max = nums[i]
            max_p = i
        } else if nums[i] > sec_max {
            sec_max = nums[i]
        }
    }

    if max >= sec_max*2 {
        return max_p
    } else {
        return -1
    }
}