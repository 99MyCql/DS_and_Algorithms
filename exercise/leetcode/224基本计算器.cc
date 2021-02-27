/*
实现一个基本的计算器来计算一个简单的字符串表达式的值。

字符串表达式可以包含左括号 ( ，右括号 )，加号 + ，减号 -，非负整数和空格  。

示例 1:

输入: "1 + 1"
输出: 2
示例 2:

输入: " 2-1 + 2 "
输出: 3
示例 3:

输入: "(1+(4+5+2)-3)+(6+8)"
输出: 23
说明：

你可以假设所给定的表达式都是有效的。
请不要使用内置的库函数 eval。
*/
#include <iostream>
#include <string>
#include <stack>

using namespace std;

enum Token {
    Well = 0,           // #
    LEFT_PARENTHESIS,   // (
    RIGHT_PARENTHESIS,  // )
    ADD,                // +
    SUB,                // -
};

class Solution {
public:
    // i代表栈顶运算符 与 j代表当前运算符 比较。' ': 不可能出现
    char priority[5][5] = {
        //  #,   (,   ),   +,  -
        { '=', '<', '<', '<', '<'  },    // #
        { ' ', '<', '=', '<', '<'  },    // (
        { ' ', ' ', ' ', ' ', ' '  },    // )
        { '>', '<', '>', '>', '>'  },    // +
        { '>', '<', '>', '>', '>'  },    // -
    };

    // 运算符转对应整型
    int op2int(char op) {
        switch (op) {
            case '#':
                return Well;
            case '(':
                return LEFT_PARENTHESIS;
            case ')':
                return RIGHT_PARENTHESIS;
            case '+':
                return ADD;
            case '-':
                return SUB;
            default:
                return -1;
        }
    }

    // 比较a与b的优先级大小
    char compare(char a, char b) {
        return priority[op2int(a)][op2int(b)];
    }

    // 运算，返回 a op b
    int operate(char op, int a, int b) {
        switch (op) {
            case '+':
                return a+b;
            case '-':
                return a-b;
            default:
                cout << "ERROR!!!\n";
                return -1;
        }
    }

    int calculate(string s) {
        s.push_back('#');   // 字符串最后追加 #
        int len = s.size();

        stack<int> num_s;
        stack<char> op_s;
        op_s.push('#');     // 栈顶压入 # ，与字符串结尾 # 配对

        for (int i=0; i<len; ) {
            // 跳过空格
            if (s[i] == ' ') {
                i++;
                continue;
            }

            // 如果是数字，则转成整型
            if (s[i] >= '0' && s[i] <= '9') {
                int temp = 0;
                while (s[i] >= '0' && s[i] <= '9') {
                    temp = temp * 10 + (s[i] - '0');
                    i++;
                }
                num_s.push(temp);
            } else {
                // 栈顶运算符与当前运算符比较优先级
                switch (compare(op_s.top(), s[i])) {
                    // 相等：弹出栈顶运算符
                    case '=': {
                        op_s.pop();
                        i++;
                        break;
                    }
                    // 栈顶运算符优先级小：压入当前运算符
                    case '<': {
                        op_s.push(s[i]);
                        i++;
                        break;
                    }
                    // 栈顶运算符优先级大：弹出栈顶运算符进行运算
                    case '>': {
                        char op = op_s.top();
                        op_s.pop();
                        int b = num_s.top();
                        num_s.pop();
                        int a = num_s.top();
                        num_s.pop();
                        num_s.push(operate(op, a, b));
                        break;
                    }
                    default: {
                        cout << "ERROR!!!\n";
                        break;
                    }
                }
            }
        }

        return num_s.top();
    }
};