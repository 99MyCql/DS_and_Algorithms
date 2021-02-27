/*
给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。

示例：

给定一个链表: 1->2->3->4->5, 和 n = 2.

当删除了倒数第二个节点后，链表变为 1->2->3->5.
说明：

给定的 n 保证是有效的。

进阶：

你能尝试使用一趟扫描实现吗？

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

#include <iostream>

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        // 由于原链表中不存在头结点，添加头结点（方便删除：假如删除第一个结点，前一个结点则为头结点）
        ListNode *temp = new ListNode(0);
        temp->next = head;
        head = temp;

        ListNode *p1 = head->next;  // 第一个指针，最终指向NULL结点
        ListNode *p2_pred = head;   // 第二个指针，最终指向倒数第N+1个结点。指向倒数第N+1个结点，是为了方便删除倒数第N个结点

        int i = 1;  // 记录第一个指针遍历到第几个结点
        while (p1) {
            p1 = p1->next;
            i++;

            // 当第一个指针遍历到第N+2个结点时，第二个指针从第1个结点开始往后遍历。两者一直相隔N+1个结点
            if (i >= n+2) {
                p2_pred = p2_pred->next;
            }

        }

        // 删除倒数第N个结点
        p2_pred->next = p2_pred->next->next;

        // 返回无头结点链表
        return head->next;
    }
};