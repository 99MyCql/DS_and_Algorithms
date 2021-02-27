/*
使用队列实现栈的下列操作：

push(x) -- 元素 x 入栈
pop() -- 移除栈顶元素
top() -- 获取栈顶元素
empty() -- 返回栈是否为空
注意:

你只能使用队列的基本操作-- 也就是 push to back, peek/pop from front, size, 和 is empty 这些操作是合法的。
你所使用的语言也许不支持队列。 你可以使用 list 或者 deque（双端队列）来模拟一个队列 , 只要是标准的队列操作即可。
你可以假设所有操作都是有效的（例如, 对一个空的栈不会调用 pop 或者 top 操作）。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/implement-stack-using-queues
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

#include <queue>

using namespace std;

class MyStack {
private:
    int stack_top;  // 记录栈顶元素
    queue<int> *qA; // 入栈时，操作的队列
    queue<int> *qB; // 出栈时，操作的队列

public:
    /** Initialize your data structure here. */
    MyStack() {
        qA = new queue<int>();
        qB = new queue<int>();
    }

    /** Push element x onto stack. */
    void push(int x) {
        this->stack_top = x;
        if (!qA->empty()) qA->push(x);
        else qB->push(x);
    }

    /** Removes the element on top of the stack and returns that element. */
    int pop() {
        int top = 0;
        if (qB->empty()) {
            while (qA->size() > 1) {
                this->stack_top = qA->front();
                qB->push(this->stack_top);
                qA->pop();
            }
            top = qA->front();
            qA->pop();
        } else {
            while (qB->size() > 1) {
                this->stack_top = qB->front();
                qA->push(this->stack_top);
                qB->pop();
            }
            top = qB->front();
            qB->pop();
        }
        return top;
    }

    /** Get the top element. */
    int top() {
        return this->stack_top;
    }

    /** Returns whether the stack is empty. */
    bool empty() {
        return qA->empty() && qB->empty();
    }
};

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack* obj = new MyStack();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->top();
 * bool param_4 = obj->empty();
 */