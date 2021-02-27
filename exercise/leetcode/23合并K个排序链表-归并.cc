/*
合并 k 个排序链表，返回合并后的排序链表。请分析和描述算法的复杂度。

示例:

输入:
[
  1->4->5,
  1->3->4,
  2->6
]
输出: 1->1->2->3->4->4->5->6

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/merge-k-sorted-lists
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/
#include <iostream>
#include <vector>

using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class Solution {
public:
    // 合并两个链表
    ListNode* my_merge2Lists(ListNode *list_a, ListNode *list_b) {
        // 习惯性地给两个链表添加头结点
        ListNode *temp = new ListNode(0);
        temp->next = list_a;
        list_a = temp;

        temp = new ListNode(0);
        temp->next = list_b;
        list_b = temp;

        ListNode *pre_a = list_a;
        ListNode *p_a = list_a->next;
        ListNode *p_b = list_b->next;
        while (p_b && p_a) {
            // 如果b链表上的结点 小于 a链表上的结点，就将b链表上的结点 插入 a链表上的结点之前
            if (p_b->val < p_a->val) {
                pre_a->next = p_b;  // 先将 p_a 前一个结点 pre_a 的 next 指向 p_b
                pre_a = p_b;        // 然后 pre_a 指向 p_b
                p_b = p_b->next;    // p_b 先后移
                pre_a->next = p_a;  // 最后将 pre_a 的 next 指向 p_a
            }
            // 反之，则a链表继续向后遍历
            else {
                pre_a = p_a;
                p_a = p_a->next;
            }
        }

        // 如果 a 链表先遍历结束，则说 b 链表剩余元素都大于 a 链表上的元素
        if (p_b) {
            pre_a->next = p_b;
        }

        delete temp;
        return list_a->next;
    }

    ListNode* my_mergeKLists(vector<ListNode*>& lists, int start, int end) {
        cout << "my_mergeKLists() ===> " << start << ", " << end << endl;
        if (end - start == 0) {
            return NULL;
        } else if (end - start == 1) {
            return lists[start];
        } else if (end - start == 2) {
            return my_merge2Lists(lists[start], lists[start+1]);
        } else {
            return my_merge2Lists(
                my_mergeKLists(lists, start, (end+start)/2),    // 注意是 (end+start)/2 !!!
                my_mergeKLists(lists, (end+start)/2, end));
        }
    }

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        return my_mergeKLists(lists, 0, lists.size());
    }
};