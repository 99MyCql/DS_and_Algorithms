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
#include <queue>

using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

// 优先队列中元素的自定义比较
struct cmp {
    bool operator () (ListNode *a, ListNode *b) {
        return a->val > b->val; // 小顶堆
    }
};

class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        int len = lists.size();
        if (len == 0) return NULL;

        priority_queue<ListNode*, vector<ListNode*>, cmp> q;
        for (int i=0; i<len; i++) {
            if (lists[i]) q.push(lists[i]); // 注意：不能将空链表的 NULL 值加入优先队列中
        }

        ListNode *ans = new ListNode(0);
        ListNode *p = ans;
        while (!q.empty()) {
            p->next = q.top();
            q.pop();
            p = p->next;
            if (p->next) q.push(p->next);
        }

        return ans->next;
    }
};