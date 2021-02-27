/*
给你一个仅包含小写字母的字符串，请你去除字符串中重复的字母，使得每个字母只出现一次。需保证返回结果的字典序最小（要求不能打乱其他字符的相对位置）。

 

示例 1:

输入: "bcabc"
输出: "abc"
示例 2:

输入: "cbacdcbc"
输出: "acdb"

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/remove-duplicate-letters
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

#include <string>
#include <stack>
#include <iostream>

using namespace std;

// 字符信息结构体
struct CharInfo {
    bool in_result;     // 是否存在于结果串中
    int repeat_count;   // 重复次数

    // 必须定义。ch_map[s[i]-'a'] 时，会自动创建 CharInfo 结构体，此时需要无参构造函数
    CharInfo() {
        in_result = false;
        repeat_count = 0;
    }

    CharInfo(bool i, int r) : in_result(i), repeat_count(r) {}
};

class Solution {
public:
    string removeDuplicateLetters(string s) {
        int len = s.size();
        CharInfo ch_map[26];    // 由于只有26个小写字母，所以可以用数组实现map，而不需要用红黑树实现的map

        // 统计每个字符的重复次数
        for (int i=0; i<len; i++) {
            ch_map[s[i]-'a'].repeat_count++;
        }

        stack<char> ch_stack;   // 单调栈
        ch_stack.push('a'-1);   // 先压入一个最小字符
        for (int i=0; i<len;) {
            char ch_top = ch_stack.top();

            // 如果当前字符 未存在于 栈中
            if (ch_map[s[i]-'a'].in_result == false) {
                // 如果，栈顶字符小于当前字符 或 剩余字符串中没有栈顶字符，则压入当前字符
                if (ch_top < s[i] || ch_map[ch_top-'a'].repeat_count == 0) {
                    cout << "push: " << s[i] << endl;
                    ch_stack.push(s[i]);
                    ch_map[s[i]-'a'].repeat_count--;
                    ch_map[s[i]-'a'].in_result = true;
                    i++;
                } else { // 反之，则循环弹出 大于当前字符 且 剩余字符串中存在重复字符 的栈顶字符
                    while (ch_stack.top() > s[i]
                        && ch_map[ch_stack.top()-'a'].repeat_count > 0) {
                        cout << "pop: " << ch_stack.top() << endl;
                        ch_map[ch_stack.top()-'a'].in_result = false;
                        ch_stack.pop();
                    }
                }
            } else { // 如果当前字符 已经存在于 栈中，则 跳过 当前字符
                ch_map[s[i]-'a'].repeat_count--;
                i++;
            }
        }

        // 将栈中字符倒序
        string ans;
        int ans_len = ch_stack.size() - 1;
        ans.resize(ans_len, '0');
        for (int i=0; i<ans_len; i++) {
            ans[ans_len-i-1] = ch_stack.top();
            ch_stack.pop();
        }

        return ans;
    }
};