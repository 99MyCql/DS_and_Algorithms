/*
请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。

 

示例 1:

输入: "abcabcbb"
输出: 3
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:

输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
示例 3:

输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
 

提示：

s.length <= 40000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

#include <cstdio>
#include <string>
#include <map>

using namespace std;

class Solution {
public:

    // 滑动窗口
    int lengthOfLongestSubstring(string s) {
        int len = s.size();

        int i = 0;  // 窗口开始指针
        int j = 0;  // 窗口结束指针
        int max = 0;
        map<char, bool> ch_map; // true: 出现过；false: 出现过，但不在当前字符串中
        while (j < len) {
            // 如果当前字符不在窗口字符串中
            if (ch_map.count(s[j]) == 0 || ch_map[s[j]] == false) {
                ch_map[s[j]] = true;
                j++;
            } else {
                // 移动前指针，直至遇到重复字符
                while (s[i] != s[j]) {
                    ch_map[s[i]] = false;
                    i++;
                }
                ch_map[s[i]] = false;
                i++;
            }
            if (j-i > max) {
                max = j-i;
            }
        }
        return max;
    }
};