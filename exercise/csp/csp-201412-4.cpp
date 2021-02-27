/*
csp 201412-4 最优灌溉
*/
#include<cstdio>
#include<algorithm>

using namespace std;


typedef struct {
	int from, to;
	int weight;
}Arc;

Arc arcs[100005];		// 边集合 
int points[1005];		// 并查集 

// 找n点所处等价类的头结点
// 并在此过程中，将线形压缩成树形 
int find(int n) {
	return n == points[n] ? n : points[n] = find(points[n]);
}

// 传入系统sort快排函数的cmp函数参数 
bool cmp(Arc a, Arc b) {
	return a.weight < b.weight;
}

int main() {
	long long n=0, m=0;
	scanf("%d%d", &n, &m);
	
	for (int i=0; i<n; i++) points[i] = i;
	
	for (int i=0; i<m; i++) {
		scanf("%d%d%d", &arcs[i].from, &arcs[i].to, &arcs[i].weight);	// 用cin比scanf慢 
		arcs[i].from--;
		arcs[i].to--;
	}
	
	// 对m条边，进行快速排序
	sort(arcs, arcs+m, cmp);
	
	int sum = 0;
	for (int i=0; i<m; i++) {
		// 找到from点所在连通点集合的头结点 
		int x = find(arcs[i].from);
		// 同理 
		int y = find(arcs[i].to);
		// 如果from点所在连通点集合的头结点与to点的不相等
		// 则说明from、to点不连通 
		if (x != y) {
			// 将from点所在连通点集的头结点，纳入到to点所在连通点集中
			// 即将两个集合连通 
			points[x] = y;
			sum += arcs[i].weight;
		}
	}
	
	printf("%d", sum);
	return 0;
}
