/*
编写一个程序，找到两个单链表相交的起始节点。

如下面的两个链表：



在节点 c1 开始相交。



示例 1：



输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出：Reference of the node with value = 8
输入解释：相交节点的值为 8 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。


示例 2：



输入：intersectVal = 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Reference of the node with value = 2
输入解释：相交节点的值为 2 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [0,9,1,2,4]，链表 B 为 [3,2,4]。在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。


示例 3：



输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
输出：null
输入解释：从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。
解释：这两个链表不相交，因此返回 null。


注意：

如果两个链表没有交点，返回 null.
在返回结果后，两个链表仍须保持原有的结构。
可假定整个链表结构中没有循环。
程序尽量满足 O(n) 时间复杂度，且仅用 O(1) 内存。
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
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        // 计算链表A的长度
        int len_a = 0;
        ListNode *p_a = headA;
        while (p_a) {
            len_a++;
            p_a = p_a->next;
        }

        // 计算链表B的长度
        int len_b = 0;
        ListNode *p_b = headB;
        while (p_b) {
            len_b++;
            p_b = p_b->next;
        }

        cout << len_a << ", " << len_b << endl;

        // 计算链表长度之差 sub，并使 p_long 指向长链表、p_short 指向短链表
        int sub;
        ListNode *p_long, *p_short;
        if (len_a >= len_b) {
            sub = len_a - len_b;
            p_long = headA;
            p_short = headB;
        } else {
            sub = len_b - len_a;
            p_long = headB;
            p_short = headA;
        }

        // 寻找交点：长链表先走 sub 步，然后短链表开始遍历
        int i = 1;
        while (p_long) {
            // 如果指向同一结点，则说明链表相交
            if (p_long == p_short) {
                return p_long;
            }
            p_long = p_long->next;
            if (i > sub) {
                p_short = p_short->next;
            }
            i++;
        }

        return NULL;
    }
};