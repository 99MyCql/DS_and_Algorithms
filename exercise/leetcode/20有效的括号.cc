/*
20. 有效的括号
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
注意空字符串可被认为是有效字符串。

示例 1:

输入: "()"
输出: true
示例 2:

输入: "()[]{}"
输出: true
示例 3:

输入: "(]"
输出: false
示例 4:

输入: "([)]"
输出: false
示例 5:

输入: "{[]}"
输出: true
*/
#include <iostream>
#include <string>
#include <stack>

using namespace std;

class Solution {
public:
    bool isValid(string s) {
        int len = s.size();
        stack<char> ch_stack;
        for (int i=0; i<len; i++) {
            if (ch_stack.empty()) {
                ch_stack.push(s[i]);
            } else {
                char ch = ch_stack.top();   // 取出栈顶字符
                // 如果栈顶字符与当前字符配对，则弹出栈顶字符
                if ((ch == '(' && s[i] == ')')
                    || (ch == '[' && s[i] == ']')
                    || (ch == '{' && s[i] == '}')) {
                    ch_stack.pop();
                }
                // 如果不匹配，则将新字符压栈
                else {
                    ch_stack.push(s[i]);
                }
            }
        }
        if (ch_stack.empty()) return true;
        else return false;
    }
};