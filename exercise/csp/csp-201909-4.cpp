/*
Title: 消息传递接口
Example:
2 3
1 3
2 2
3 1
8
3 100 1 1
1 0 4 3
1 0 5 1
3 10 2 2
3 10 1 1
2 0 1
3 2 1 1
3 1 1 1

2 3
1 3
2 2
3 1
7 
1 1 5 4
1 0 2 4
3 2 2 2
1 0 1 5
3 4 2 2
2 0 2
3 5 3 3
*/
#include <cstdio>
#include <set>
#include <map>
#include <vector>

using namespace std;

//#define DEBUG

typedef long long ll;

/* 商品结构体 */
struct Commodity{
    int type;   // 商品类别
    int id;     // 商品编号
    int score;  // 商品评分

    // 构造函数
    Commodity(int type, int id, int score): type(type), id(id), score(score) {}

    // 运算符重载，用于set集合中的默认排序
    bool operator < (const Commodity c) const {
        return score > c.score || (score == c.score && type < c.type) || (score == c.score && type == c.type && id < c.id);
    }
};

set<Commodity> g_commodity_set; // 商品集合
map<int, int> g_type_com_map[51]; // 相应类别中相应编号的商品 -> 该商品的评分


void printInfo() {
    printf("------------------\n");
    for (set<Commodity>::iterator it = g_commodity_set.begin(); it != g_commodity_set.end(); it++) {
        printf("type:%d, id:%d, score:%d\n", it->type, it->id, it->score);
    }
    printf("------------------\n");
}

int main() {
    int n, m; // m代表商品种类个数
    scanf("%d%d", &m, &n);
    for (int i=0; i < n; i++) {
        int id, score;
        scanf("%d%d", &id, &score);
        for (int j=0; j < m; j++) {
            Commodity temp(j, id, score); // 注意顺序！！！
            g_commodity_set.insert(temp);
            g_type_com_map[j][id] = score;
        }
    }
    
    int op_n;
    scanf("%d", &op_n);
    for (int i=0; i < op_n; i++) {
        int op_type;
        scanf("%d", &op_type);
        #ifdef DEBUG
        printf("====>op_type:%d\n", op_type);
        #endif
        switch (op_type) {
            // 增加一个商品
            case 1: {
                int type, id, score;
                scanf("%d%d%d", &type, &id, &score);
                Commodity temp(type, id, score);
                g_commodity_set.insert(temp);
                g_type_com_map[type][id] = score;
                break;
            }
            // 删除一个商品
            case 2: {
                int type, id;
                scanf("%d%d", &type, &id);
                g_type_com_map[type][id] = -1;
                break;
            }
            // 查询
            case 3: {
                int K, ki[51];
                scanf("%d", &K);
                for (int j=0; j < m; j++) {
                    scanf("%d", &ki[j]);
                }

                vector<int> ans[51];
                for (auto it = g_commodity_set.begin(); it != g_commodity_set.end() && K > 0; it++) {
                    if (ki[it->type] > 0 && g_type_com_map[it->type][it->id] == it->score) {
                        #ifdef DEBUG
                        printf("type:%d, id:%d, score:%d\n", it->type, it->id, it->score);
                        #endif
                        ans[it->type].push_back(it->id);
                        K--;
                        ki[it->type]--;
                    }
                }

                for (int j=0; j < m; j++) {
                    #ifdef DEBUG
                    printf("====>type:%d\n", j);
                    #endif
                    if (ans[j].empty()) {
                        printf("-1\n");
                    }
                    else {
                        for (int k=0; k < ans[j].size(); k++) {
                            printf("%d ", ans[j][k]);
                        }
                        printf("\n");
                    }
                }
                break;
            }
            default:
                break;
        }
        #ifdef DEBUG
        printInfo();
        #endif
    }
}
