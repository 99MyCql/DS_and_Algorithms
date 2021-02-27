/*
给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。

说明：不允许修改给定的链表。
 

示例 1：

输入：head = [3,2,0,-4], pos = 1
输出：tail connects to node index 1
解释：链表中有一个环，其尾部连接到第二个节点。


示例 2：

输入：head = [1,2], pos = 0
输出：tail connects to node index 0
解释：链表中有一个环，其尾部连接到第一个节点。


示例 3：

输入：head = [1], pos = -1
输出：no cycle
解释：链表中没有环。


进阶：
你是否可以不用额外空间解决此题？

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/linked-list-cycle-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/
#include <iostream>

using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        // 习惯性添加头结点
        ListNode *temp = new ListNode(0);
        temp->next = head;
        head = temp;

        // 第一阶段：判断是否存在环
        ListNode *slow_p = head->next;
        ListNode *fast_p = head->next;
        while (fast_p) {
            slow_p = slow_p->next;
            fast_p = fast_p->next;
            if (fast_p) {
                fast_p = fast_p->next;
                if (fast_p == slow_p) break;
            }
        }

        if (fast_p == NULL) return NULL;

        // 第二阶段：找环的入口
        ListNode *p = head->next;
        while (p != slow_p) {
            p = p->next;
            slow_p = slow_p->next;
        }
        return p;
    }
};
