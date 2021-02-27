/*
92. 反转链表 II
反转从位置 m 到 n 的链表。请使用一趟扫描完成反转。

说明:
1 ≤ m ≤ n ≤ 链表长度。

示例:

输入: 1->2->3->4->5->NULL, m = 2, n = 4
输出: 1->4->3->2->5->NULL
*/
#include <iostream>

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int m, int n) {
        // 添加头结点
        ListNode *temp = new ListNode(0);
        temp->next = head;
        head = temp;

        int i = 1;                  // 记录当前第几个结点
        ListNode *p = head->next;   // 指向当前结点
        ListNode *p_m_pred = head;  // 指向第m结点的前一个结点。初值为 head，防止 m==1 的情况
        ListNode *p_n_next = NULL;  // 指向第n结点的后一个结点
        while (p && i <= n) {
            if (i == n) {
                p_n_next = p->next; // 第 n+1 个结点
            }
            if (i == m-1) {
                p_m_pred = p;       // 第 m-1 个结点
            }
            p = p->next;
            i++;
        }

        p = p_m_pred->next;
        p_m_pred->next = p_n_next;  // 先将第 m-1 个结点的 next 指针指向第 n+1 个结点
        // 从第m个结点遍历到第n个结点
        while (p != p_n_next) {
            ListNode *temp = p;
            p = p->next;    // 先指向下一个结点

            // 用头插法。将当前结点，插入第 m-1 个结点之后。第 m-1 个结点相当于头结点。
            temp->next = p_m_pred->next;
            p_m_pred->next = temp;
        }

        return head->next;
    }
};