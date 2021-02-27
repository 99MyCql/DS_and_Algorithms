/*
使用栈实现队列的下列操作：

push(x) -- 将一个元素放入队列的尾部。
pop() -- 从队列首部移除元素。
peek() -- 返回队列首部的元素。
empty() -- 返回队列是否为空。
示例:

MyQueue queue = new MyQueue();

queue.push(1);
queue.push(2);
queue.peek();  // 返回 1
queue.pop();   // 返回 1
queue.empty(); // 返回 false
说明:

你只能使用标准的栈操作 -- 也就是只有 push to top, peek/pop from top, size, 和 is empty 操作是合法的。
你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。
假设所有操作都是有效的 （例如，一个空的队列不会调用 pop 或者 peek 操作）。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/implement-queue-using-stacks
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

#include <stack>

using namespace std;

class MyQueue {
public:
    stack<int> *in_stack;
    stack<int> *out_stack;

    /** Initialize your data structure here. */
    MyQueue() {
        this->in_stack = new stack<int>();
        this->out_stack = new stack<int>();
    }

    /** Push element x to the back of queue. */
    void push(int x) {
        in_stack->push(x);
    }

    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        // 如果出栈为空，则将入栈的值依次弹出压入出栈中
        if (out_stack->empty()) {
            while (!in_stack->empty()) {
                int temp = in_stack->top();
                in_stack->pop();
                out_stack->push(temp);
            }
        }
        int temp = out_stack->top();
        out_stack->pop();
        return temp;
    }

    /** Get the front element. */
    int peek() {
        if (out_stack->empty()) {
            while (!in_stack->empty()) {
                int temp = in_stack->top();
                in_stack->pop();
                out_stack->push(temp);
            }
        }
        return out_stack->top();
    }

    /** Returns whether the queue is empty. */
    bool empty() {
        return in_stack->empty() && out_stack->empty();
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue* obj = new MyQueue();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->peek();
 * bool param_4 = obj->empty();
 */