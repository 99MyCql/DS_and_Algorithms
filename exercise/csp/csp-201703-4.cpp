/*
问题描述
　　A市有n个交通枢纽，其中1号和n号非常重要，为了加强运输能力，A市决定在1号到n号枢纽间修建一条地铁。
　　地铁由很多段隧道组成，每段隧道连接两个交通枢纽。经过勘探，有m段隧道作为候选，两个交通枢纽之间最多只有一条候选的隧道，没有隧道两端连接着同一个交通枢纽。
　　现在有n家隧道施工的公司，每段候选的隧道只能由一个公司施工，每家公司施工需要的天数一致。而每家公司最多只能修建一条候选隧道。所有公司同时开始施工。
　　作为项目负责人，你获得了候选隧道的信息，现在你可以按自己的想法选择一部分隧道进行施工，请问修建整条地铁最少需要多少天。
输入格式
　　输入的第一行包含两个整数n, m，用一个空格分隔，分别表示交通枢纽的数量和候选隧道的数量。
　　第2行到第m+1行，每行包含三个整数a, b, c，表示枢纽a和枢纽b之间可以修建一条隧道，需要的时间为c天。
输出格式
　　输出一个整数，修建整条地铁线路最少需要的天数。
样例输入
6 6
1 2 4
2 3 4
3 6 7
1 4 2
4 5 5
5 6 6
样例输出
6
样例说明
　　可以修建的线路有两种。
　　第一种经过的枢纽依次为1, 2, 3, 6，所需要的时间分别是4, 4, 7，则整条地铁线需要7天修完；
　　第二种经过的枢纽依次为1, 4, 5, 6，所需要的时间分别是2, 5, 6，则整条地铁线需要6天修完。
　　第二种方案所用的天数更少。
评测用例规模与约定
　　对于20%的评测用例，1 ≤ n ≤ 10，1 ≤ m ≤ 20；
　　对于40%的评测用例，1 ≤ n ≤ 100，1 ≤ m ≤ 1000；
　　对于60%的评测用例，1 ≤ n ≤ 1000，1 ≤ m ≤ 10000，1 ≤ c ≤ 1000；
　　对于80%的评测用例，1 ≤ n ≤ 10000，1 ≤ m ≤ 100000；
　　对于100%的评测用例，1 ≤ n ≤ 100000，1 ≤ m ≤ 200000，1 ≤ a, b ≤ n，1 ≤ c ≤ 1000000。

　　所有评测用例保证在所有候选隧道都修通时1号枢纽可以通过隧道到达其他所有枢纽。
*/
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

using namespace std;

typedef struct {
    int from;
    int to;
    int time;
} Edge;

int n, m; // n个点，m条边
int* father; // 并查集数组

// 初始化并查集
void initFa() {
    father = (int*)malloc(sizeof(int) * (n+1));
    for (int i=0; i <= n; i++) {
        father[i] = i;
    }
}

// 并查集：查找元素所属的集合(爸爸)
int findFa(int n) {
    return father[n] == n ? n : father[n] = findFa(father[n]); // 认爷爷做爸爸
}

// 设置a是b的爸爸
void setFa(int a, int b) {
    father[b] = a;
}

bool cmp(Edge a, Edge b) {
    return a.time < b.time;
}

int main() {
    scanf("%d%d", &n, &m);
    vector<Edge> edges;
    for (int i=0; i < m; i++) {
        Edge edge;
        scanf("%d%d%d", &edge.from, &edge.to, &edge.time);
        edges.push_back(edge);
    }

    // 将边从小到大排序
    sort(edges.begin(), edges.end(), cmp);

    // for (int i=0; i < m; i++) {
    //     printf("%d---%d--->%d\n", edges[i].from, edges[i].time, edges[i].to);
    // }

    // 求解最小生成树
    initFa();
    int ans = 0; // 答案
    for (int i=0; i < m; i++) {
        int xFa = findFa(edges[i].from);
        int yFa = findFa(edges[i].to);
        if (xFa == yFa) continue;
        else {
            setFa(xFa, yFa);
            if (edges[i].time > ans) ans = edges[i].time;
        }

        if (findFa(1) == findFa(n)) {
            printf("%d", ans);
            return 0;
        }
    }
}