/*
给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。

请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。

你可以假设 nums1 和 nums2 不会同时为空。

示例 1:

nums1 = [1, 3]
nums2 = [2]

则中位数是 2.0
示例 2:

nums1 = [1, 2]
nums2 = [3, 4]

则中位数是 (2 + 3)/2 = 2.5
*/
#include <cstdio>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> n1;
    vector<int> n2;
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        n1 = nums1;
        n2 = nums2;
        int n1_start = 0, n1_end = n1.size(), n2_start = 0, n2_end = n2.size();
        int n1_mid = 0, n2_mid = 0;
        bool n1_mid_find = false, n2_mid_find = false; // n1, n2是否找到中位数
        if (n1_start == n1_end) {
            return n2_end%2 == 0? (n2[n2_end/2-1]+n2[n2_end/2])/2.0 : n2[n2_end/2];
        }
        if (n2_start == n2_end) {
            return n1_end%2 == 0? (n1[n1_end/2-1]+n1[n1_end/2])/2.0 : n1[n1_end/2];
        }
        while (!n1_mid_find || !n2_mid_find) {
            if (!n1_mid_find) n1_mid = (n1_start + n1_end) / 2;
            if (!n2_mid_find) n2_mid = (n2_start + n2_end) / 2;
            if (n1[n1_mid] < n2[n2_mid]) {
                n1_start = n1_mid;
                n2_end = n2_mid;
            }
            else {
                n1_end = n1_mid;
                n2_start = n2_mid;
            }
            if (n1_end - n1_start <= 1) {
                n1_mid_find = true;
            }
            if (n2_end - n2_start <= 1) {
                n2_mid_find = true;
            }
        }
        printf("%d %d", n1[n1_mid], n2[n2_mid]);

        if ((n1.size() + n2.size())%2 == 0) {
            return (n1[n1_mid] + n2[n2_mid]) / 2.0;
        }
        else {
            return n1[n1_mid] < n2[n2_mid] ? n1[n1_mid] : n2[n2_mid];
        }
    }
};