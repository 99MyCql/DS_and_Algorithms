/*
我们有一个非负整数数组 A。

对于每个（连续的）子数组 B = [data[i], data[i+1], ..., data[j]] （ i <= j），我们对 B 中的每个元素进行按位或操作，获得结果 A[i] | data[i+1] | ... | data[j]。

返回可能结果的数量。 （多次出现的结果在最终答案中仅计算一次。）


示例 1：

输入：[0]
输出：1
解释：
只有一个可能的结果 0 。
示例 2：

输入：[1,1,2]
输出：3
解释：
可能的子数组为 [1]，[1]，[2]，[1, 1]，[1, 2]，[1, 1, 2]。
产生的结果为 1，1，2，1，3，3 。
有三个唯一值，所以答案是 3 。
示例 3：

输入：[1,2,4]
输出：6
解释：
可能的结果是 1，2，3，4，6，以及 7 。
 

提示：

1 <= A.length <= 50000
0 <= data[i] <= 10^9
*/
#include<cstdio>
#include<cstdlib>
#include<vector>
#include<list>
#include<unordered_set>

using namespace std;

typedef struct List{
    int data;
    List* next;
};

class Solution {
public:
    // int len;
    // int **dp;
    // unordered_set<int> res_set; // 可能值集合

    // int getDp(int start, int end) {
    //     // printf("start: %d, end: %d\n", start, end);
    //     if (dp[start][end] > -1) return dp[start][end];
    //     if (start == end) {
    //         dp[start][end] = data[start];
    //     }
    //     else {
    //         dp[start][end] = getDp(start, end-1) | getDp(end, end);
    //     }
    //     // printf("dp[start][end]: %d\n", dp[start][end]);
    //     res_set.insert(dp[start][end]);
    //     return dp[start][end];
    // }

    int subarrayBitwiseORs(vector<int>& A) {
        // // 转换成链表
        // List* head = (List*)malloc(sizeof(List));
        // List* p = head;
        // for (int i=0; i < A.size(); i++) {
        //     // printf("---> %d\n", i);
        //     List* temp = (List*)malloc(sizeof(List));
        //     temp->data = A[i];
        //     p->next = temp;
        //     p = temp;
        // }
        // p->next = NULL;

        // // 去重
        // p = head->next;
        // while (p) {
        //     List* temp = p->next;
        //     while (temp != NULL && temp->data == p->data) {
        //         // printf("while\n");
        //         List* next = temp->next;
        //         p->next = next;
        //         free(temp);
        //         temp = next;
        //     }
        //     p = temp;
        // }

        // // 转换成数组
        // vector<int> data;
        // p = head->next;
        // while (p) {
        //     data.push_back(p->data);
        //     // printf("==%d\n", p->data);
        //     p = p->next;
        // }

        // init dp
        // int **dp = (int**)malloc(sizeof(int*) * len);
        // for (int i=0; i < len; i++) {
        //     int* temp = (int*)malloc(sizeof(int) * len);
        //     dp[i] = temp;
        // }

        vector<int> data = A; // 数据集
        int len = data.size(); // 长度
        unordered_set<int> res_set; // 可能值集合

        // 先求上限最大值
        int max = 0;
        for (int i=0; i < len; i++) {
            // dp[i][i] = data[i];
            max |= data[i];
        }
        res_set.insert(max);

        // 求出全部结果，有条件地放入集合中
        for (int i=0; i < len; i++) {
            // res_set.insert(dp[i][i]);
            int temp = 0;
            for (int j=i; j < len; j++) {
                temp |= data[j];
                if (temp == max) break; // 如果等于最大值，之后则不需要再计算了
                res_set.insert(temp);
                // dp[i][j] = dp[i][j-1] | dp[j][j];
                // if (dp[i][j] == max) break;
                // res_set.insert(dp[i][j]);
            }
        }

        return res_set.size();
    }
};

// int main() {
//     int a[] = {1,2,4};
//     vector<int> v(a, a+3);
//     Solution s;
//     s.subarrayBitwiseORs(v);
//     return 0;
// }