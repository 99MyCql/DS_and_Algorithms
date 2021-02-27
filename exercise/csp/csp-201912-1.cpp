// INFO BEGIN
//
// User = 201911508964(ÀîË«¾Á) 
// Group = C/C++ 
// Problem = ±¨Êý 
// Language = DCPP 
// SubmitTime = 2019-12-15 13:46:40 
//
// INFO END

/*
20

2
1
1
0

66

7
5
11
5
*/
#include <cstdio>

using namespace std;

//#define DEBUG

typedef long long ll;

bool hasSeven(ll num) {
	while (num != 0) {
		if (num%10 == 7) return true;
		else num = num/10;
	}
	return false;
}

int main() {
	int n;
	scanf("%d", &n);
	ll a=0, b=0, c=0, d=0; // ¼×ÒÒ±û¶¡
	for (ll i=0, num=1; i<n; num++) {
		#ifdef DEBUG
		printf("===>i:%d, num:%d\n", i, num);
		#endif
		if (num%7 == 0 || hasSeven(num)) {
			int x = num%4;
			switch (x) {
				case 0:
					d++;
					break;
				case 1:
					a++;
					break;
				case 2:
					b++;
					break;
				case 3:
					c++;
					break;
			}
		}
		else i++;
	}
	
	printf("%d\n%d\n%d\n%d\n", a, b, c, d);
	return 0;
}