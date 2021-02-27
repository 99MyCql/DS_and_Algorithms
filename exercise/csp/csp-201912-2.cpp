// INFO BEGIN
//
// User = 201911508964(李双玖) 
// Group = C/C++ 
// Problem = 回收站选址 
// Language = DCPP 
// SubmitTime = 2019-12-15 14:29:47 
//
// INFO END

/*
7
1 2
2 1
0 0
1 1
1 0
2 0
0 1

2
0 0
-100000 10

11
9 10
10 10
11 10
12 10
13 10
11 9
11 8
12 9
10 9
10 11
12 11

8
0 0
0 1
0 2
1 0
1 1
1 2
2 0
2 1

9
0 0
0 1
0 2
1 0
1 1
1 2
2 0
2 1
2 2

2
1000000000 1000000000
-1000000000 -1000000000
*/
#include <cstdio>
#include <set>

using namespace std;

//#define DEBUG

typedef long long ll;

struct Point {
	ll x;
	ll y;
	Point (ll x, ll y) : x(x), y(y) {}
	bool operator<(const Point p) const {
		return x < p.x || (x == p.x && y < p.y);
	}
};

int main() {
	set<Point> point_set;
	int n;
	scanf("%d", &n);
	for (int i=0; i<n; i++) {
		ll x, y;
		scanf("%d%d", &x, &y);
		point_set.insert(Point(x,y));
	}
	
	int s0_num=0, s1_num=0, s2_num=0, s3_num=0, s4_num=0;
	for (set<Point>::iterator it=point_set.begin(); it!=point_set.end(); it++) {
		ll x=it->x, y=it->y;
		#ifdef DEBUG
		printf("===>x:%d, y:%d\n", x, y);
		#endif
		if (point_set.count(Point(x,y-1)) && point_set.count(Point(x,y+1)) 
		&& point_set.count(Point(x-1,y)) && point_set.count(Point(x+1,y))) {
			#ifdef DEBUG
			printf("===>ok\n");
			#endif
			int score = 0;
			if (point_set.count(Point(x-1,y-1))) score++;
			if (point_set.count(Point(x-1,y+1))) score++;
			if (point_set.count(Point(x+1,y+1))) score++;
			if (point_set.count(Point(x+1,y-1))) score++;
			
			if (score==0) s0_num++;
			else if (score==1) s1_num++;
			else if (score==2) s2_num++;
			else if (score==3) s3_num++;
			else if (score==4) s4_num++;
		}
	}
	
	printf("%d\n%d\n%d\n%d\n%d\n", s0_num, s1_num, s2_num, s3_num, s4_num);
	
	return 0;
}