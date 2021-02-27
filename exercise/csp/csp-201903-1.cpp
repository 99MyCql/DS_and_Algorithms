/*
3
-1 2 4

4
-2 -1 3 4
*/
#include<cstdio>

int main() {
	int n;
	scanf("%d", &n);
	long num[n];
	for (int i=0; i<n; i++) {
		scanf("%d", &num[i]);
	}
	
	long max=0, min=0;
	if (num[0] > num[n-1]) {
		max = num[0];
		min = num[n-1];
	}
	else {
		max = num[n-1];
		min = num[0];
	}
	
	if (n%2 == 0) {
		double mid = (double)(num[n/2-1] + num[n/2]) / 2.0;
		if (mid > (long)mid)
			printf("%d %.1f %d", max, mid, min);
		else
			printf("%d %d %d", max, (long)mid, min);
	}
	else {
		int mid = num[n/2];
		printf("%d %d %d", max, mid, min);
	}
	
	return 0;
}
