/*
5. 发 LeetCoin
力扣决定给一个刷题团队发LeetCoin作为奖励。同时，为了监控给大家发了多少LeetCoin，力扣有时候也会进行查询。

该刷题团队的管理模式可以用一棵树表示：

团队只有一个负责人，编号为1。除了该负责人外，每个人有且仅有一个领导（负责人没有领导）；
不存在循环管理的情况，如A管理B，B管理C，C管理A。

力扣想进行的操作有以下三种：

给团队的一个成员（也可以是负责人）发一定数量的LeetCoin；
给团队的一个成员（也可以是负责人），以及他/她管理的所有人（即他/她的下属、他/她下属的下属，……），发一定数量的LeetCoin；
查询某一个成员（也可以是负责人），以及他/她管理的所有人被发到的LeetCoin之和。

输入：

N表示团队成员的个数（编号为1～N，负责人为1）；
leadership是大小为(N - 1) * 2的二维数组，其中每个元素[a, b]代表b是a的下属；
operations是一个长度为Q的二维数组，代表以时间排序的操作，格式如下：
operations[i][0] = 1: 代表第一种操作，operations[i][1]代表成员的编号，operations[i][2]代表LeetCoin的数量；
operations[i][0] = 2: 代表第二种操作，operations[i][1]代表成员的编号，operations[i][2]代表LeetCoin的数量；
operations[i][0] = 3: 代表第三种操作，operations[i][1]代表成员的编号；
输出：

返回一个数组，数组里是每次查询的返回值（发LeetCoin的操作不需要任何返回值）。由于发的LeetCoin很多，请把每次查询的结果模1e9+7 (1000000007)。

示例 1：

输入：N = 6, leadership = [[1, 2], [1, 6], [2, 3], [2, 5], [1, 4]], operations = [[1, 1, 500], [2, 2, 50], [3, 1], [2, 6, 15], [3, 1]]
输出：[650, 665]
解释：团队的管理关系见下图。
第一次查询时，每个成员得到的LeetCoin的数量分别为（按编号顺序）：500, 50, 50, 0, 50, 0;
第二次查询时，每个成员得到的LeetCoin的数量分别为（按编号顺序）：500, 50, 50, 0, 50, 15.

限制：

1 <= N <= 50000
1 <= Q <= 50000
operations[i][0] != 3 时，1 <= operations[i][2] <= 5000
*/
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;

typedef struct Node {
    int index;
    int coins;
    Node* son;
    Node* brother;
};

class Solution {
public:
    void createNode(Node *tree, int fa, int son) {
        if (tree == NULL) return;
        if (tree->index == fa) {
            Node* temp = (Node*)malloc(sizeof(Node));
            temp->index = son;
            temp->coins = 0;
            temp->son = NULL;
            temp->brother = NULL;

            if (tree->son == NULL) tree->son = temp;
            else {
                Node* p = tree->son;
                while (p->brother) {
                    p = p->brother;
                }
                p->brother = temp;
            }
        }
        else {
            createNode(tree->son, fa, son);
            createNode(tree->brother, fa, son);
        }
    }

    Node* findNode(Node* tree, int index) {
        if (tree == NULL) return NULL;
        if (tree->index == index) {
            return tree;
        }
        Node* temp1 = findNode(tree->son, index);
        Node* temp2 = findNode(tree->brother, index);
        if (temp1 != NULL) return temp1;
        else if (temp2 != NULL) return temp2;
        else return NULL;
    }

    void send2Son(Node* tree, int coin) {
        if (tree == NULL) return;
        tree->coins += coin;
        send2Son(tree->son, coin);
        send2Son(tree->brother, coin);
    }

    int sumCoins(Node* tree) {
        if (tree == NULL) return 0;
        return tree->coins + sumCoins(tree->son) + sumCoins(tree->brother);
    }

    vector<int> bonus(int n, vector<vector<int>>& leadership, vector<vector<int>>& operations) {
        Node* tree = (Node*)malloc(sizeof(Node));
        tree->index = 1;
        tree->coins = 0;
        tree->son = NULL;
        tree->brother = NULL;

        int len = leadership.size();
        for (int i=0; i < len; i++) {
            createNode(tree, leadership[i][0], leadership[i][1]);
        }

        len = operations.size();
        vector<int> ans;
        for (int i=0; i < len; i++) {
            Node *temp = NULL;
            int sum = 0;
            switch (operations[i][0]) {
                // 给团队的一个成员（也可以是负责人）发一定数量的LeetCoin；
                case 1:
                    temp = findNode(tree, operations[i][1]);
                    temp->coins += operations[i][2];
                    break;
                // 给团队的一个成员（也可以是负责人），以及他/她管理的所有人（即他/她的下属、他/她下属的下属，……），发一定数量的LeetCoin；
                case 2:
                    temp = findNode(tree, operations[i][1]);
                    temp->coins += operations[i][2];
                    send2Son(temp->son, operations[i][2]);
                    break;
                // 查询某一个成员（也可以是负责人），以及他/她管理的所有人被发到的LeetCoin之和。
                case 3:
                    temp = findNode(tree, operations[i][1]);
                    sum += temp->coins + sumCoins(temp->son);
                    ans.push_back(sum);
                    break;
                default:
                    break;
            }
        }
        return ans;
    }
};