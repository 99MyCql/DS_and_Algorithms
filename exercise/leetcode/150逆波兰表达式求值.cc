/*
根据逆波兰表示法，求表达式的值。

有效的运算符包括 +, -, *, / 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。

说明：

整数除法只保留整数部分。
给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。
示例 1：

输入: ["2", "1", "+", "3", "*"]
输出: 9
解释: ((2 + 1) * 3) = 9
示例 2：

输入: ["4", "13", "5", "/", "+"]
输出: 6
解释: (4 + (13 / 5)) = 6
示例 3：

输入: ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
输出: 22
解释:
  ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
*/
#include <vector>
#include <string>
#include <iostream>
#include <stack>

using namespace std;

class Solution {
public:
    int evalRPN(vector<string>& tokens) {
        int len = tokens.size();

        stack<int> num_s;
        for (int i=0; i<len; i++) {
            // 如果当前是运算符
            if (tokens[i].size() == 1 && !(tokens[i][0] >= '0' && tokens[i][0] <= '9')) {
                int b = num_s.top();
                num_s.pop();
                int a = num_s.top();
                num_s.pop();
                switch (tokens[i][0]) {
                    case '+':
                        num_s.push(a+b);
                        break;
                    case '-':
                        num_s.push(a-b);
                        break;
                    case '*':
                        num_s.push(a*b);
                        break;
                    case '/':
                        num_s.push(a/b);
                        break;
                    default:
                        cout << "ERROR!!!\n";
                        num_s.push(a);
                        num_s.push(b);
                        break;
                }
            }
            // 如果是数字
            else {
                int num = 0;
                int j = 0;
                if (tokens[i][0] == '-') j++;   // 如果有负号，跳过
                // 字符串转整数
                while (j < tokens[i].size()) {
                    num = num*10 + (tokens[i][j] - '0');
                    j++;
                }
                if (tokens[i][0] == '-') num = -num;    // 如果有负号，转负数
                num_s.push(num);
            }
        }

        return num_s.top();
    }
};