/*
设计一个支持 push，pop，top 操作，并能在常数时间内检索到最小元素的栈。

push(x) -- 将元素 x 推入栈中。
pop() -- 删除栈顶的元素。
top() -- 获取栈顶元素。
getMin() -- 检索栈中的最小元素。
示例:

MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2.

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/min-stack
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

#include <iostream>
#include <stack>

using namespace std;

class MinStack {
public:
    stack<int> *data_stack;
    stack<int> *min_stack;

    /** initialize your data structure here. */
    MinStack() {
        this->data_stack = new stack<int>();
        this->min_stack = new stack<int>();
    }

    void push(int x) {
        data_stack->push(x);
        // 如果最小栈为空，或当前元素小于最小栈栈顶值，则将当前值压入最小栈中
        if (min_stack->empty() || min_stack->top() > x) {
            min_stack->push(x);
        } else {
            min_stack->push(min_stack->top());
        }
    }

    void pop() {
        data_stack->pop();
        min_stack->pop();
    }

    int top() {
        return data_stack->top();
    }

    int getMin() {
        return min_stack->top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */