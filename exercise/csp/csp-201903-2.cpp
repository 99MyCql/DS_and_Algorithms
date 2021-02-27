/*
10
9+3+4x3
5+4x5x5
7-9-9+8
5x6/5x4
3+5+7+9
1x1+9-9
1x9-5/9
8/5+6x9
6x7-3x6
6x4+4/5
*/
#include<cstdio>
#include<stack>

using namespace std;

// 运算
int operate(int a, int b, char op) {
    // printf("%d %d %c\n", a, b, op);
    int ret = 0;
    switch (op) {
        case 'x' :
            ret = a*b;
            break;
        case '/' :
            ret = a/b;
            break;
        case '+' :
            ret = a+b;
            break;
        case '-' :
            ret = a-b;
            break;
    }
    return ret;
}

// 运算符优先级比较，若a>b则返回true
bool compare(char a, char b) {
    switch (a) {
        case '+':
        case '-':
            if (b == '-' || b == '+') return true;
            if (b == 'x' || b == '/') return false;
            break;
        case 'x':
        case '/':
            return true;
    }
}

int main() {
    int n=0;
    scanf("%d", &n);

    for (int i=0; i<n; i++) {
        stack<int> numS;
        stack<char> charS;
        char temp[8];
        scanf("%s", temp);
        for (int i=0; i<7; i++) {
            // 如果是数字
            if (temp[i] >= '0' && temp[i] <= '9') {
                numS.push(temp[i]-'0');
            }
            // 如果是运算符
            else {
                if (!charS.empty()) {
                    char op_a = charS.top();
                    char op_b = temp[i];
                    // 如果运算符栈顶元素优先级高，则先弹出栈顶运算符进行运算
                    while (compare(op_a, op_b)) {
                        int b = numS.top();
                        numS.pop();
                        int a = numS.top();
                        numS.pop();
                        char op = charS.top();
                        charS.pop();
                        numS.push(operate(a,b,op));

                        if (charS.empty()) break;
                        op_a = charS.top();
                    }
                    charS.push(op_b);
                }
                // 如果运算符栈为空，则直接入栈
                else {
                    charS.push(temp[i]);
                }
            }
        }
        while (!charS.empty()) {
            int b = numS.top();
            numS.pop();
            int a = numS.top();
            numS.pop();
            char op = charS.top();
            charS.pop();
            numS.push(operate(a,b,op));
        }
        if (numS.top() == 24) {
            printf("Yes\n");
        }
        else {
            // printf("No: %d\n", numS.top());
            printf("No\n");
        }
    }
}
