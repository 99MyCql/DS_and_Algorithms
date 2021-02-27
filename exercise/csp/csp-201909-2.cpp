/*
Title: 小明种苹果（续）
Example:
4
4 74 -7 -12 -5
5 73 -8 -6 59 -4
5 76 -5 -10 60 -2
5 80 -6 -15 59 0

5
4 10 0 9 0
4 10 -2 7 0
2 10 0
4 10 -3 5 0
4 10 -1 8 0
*/
#include <cstdio>

typedef struct Tree{
	int count;
	bool isDrop;
};

Tree trees[1005];

int main() {
	int n;
	scanf("%d", &n);
	for (int i=0; i < n; i++) {
		int m;
		scanf("%d", &m);
		scanf("%d", &trees[i].count);
		trees[i].isDrop = false;
		for (int j=1; j < m; j++) {
			int temp;
			scanf("%d", &temp);
			if (temp > 0) {
				if (temp < trees[i].count) {
					trees[i].isDrop = true;
					trees[i].count = temp;
				}
			}
			else {
				trees[i].count += temp; // temp是负数 
			}
		}
	}
	
	long long sum = 0;
	int drop_sum = 0;
	int e_sum = 0;
	for (int i=0; i < n; i++) {
		sum += trees[i].count;
		if (trees[i].isDrop) drop_sum++;
		if (trees[i%n].isDrop && trees[(i+1)%n].isDrop && trees[(i+2)%n].isDrop) {
			e_sum++;
		}
	}
	
	printf("%d %d %d", sum, drop_sum, e_sum);
} 