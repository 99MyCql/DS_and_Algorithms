/*
Title: 小明种苹果
Example:
3 3
73 -8 -6 -4
76 -5 -10 -8
80 -6 -15 0

2 2
10 -3 -1
15 -4 0
*/
#include <cstdio>

int main() {
	int n,m;
	scanf("%d%d", &n, &m);
	long long sum = 0;
	long long min = 0;
	int min_index = -1;
	for (int i=0; i < n; i++) {
		int source;
		scanf("%d", &source);
		int count = 0; // 疏果总数，负数 
		for (int j=0; j < m; j++) {
			int temp;
			scanf("%d", &temp);
			count += temp;
		}
		sum += (source + count);
		if (count < min) {
			min = count;
			min_index = i;
		}
	}
	printf("%d %d %d", sum, min_index+1, -min);
}