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
*/
#include <cstdio>
#include <list>
#include <vector>
#include <algorithm>

using namespace std;

typedef struct Good {
	int id;
	int score;
	int type;
};

list<Good> good_types[55];
int m, n;

void insert(int type, Good good) {
	good.type = type;
	list<Good>::iterator it = good_types[type].begin();
	list<Good>::iterator end = good_types[type].end();
	while (it != end && (it->score > good.score || (it->score == good.score && it->id < good.id))) {
		it++;
	}
	good_types[type].insert(it, good);
}

void del(int type, int id) {
//	printf("--->%d %d\n", type, id);
	list<Good>::iterator it = good_types[type].begin();
	list<Good>::iterator end = good_types[type].end();
	for (; it != end; it++) {
//		printf("%d\n", it->id);
		if (it->id == id) {
			if (it == good_types[type].begin()) good_types[type].pop_front();
			else good_types[type].erase(it);
			break;
		}
//		printf("good\n");
	}
}

bool cmp(Good a, Good b) {
	if (a.score > b.score) {
		return true;
	}
	else if (a.score == b.score) {
		if (a.type < b.type) return true;
		else if (a.type == b.type && a.id < b.id) return true;
		else return false;
	}
	else {
		return false;
	}
}

void find(int K, int* ks) {
	vector<Good> v;
	for (int i=0; i < m; i++) {
		list<Good>::iterator it = good_types[i].begin();
		list<Good>::iterator end = good_types[i].end();
		int j = 0;
		while (j < ks[i] && it != end) {
			v.push_back(*it);
			it++, j++;
		}
	}
	
	sort(v.begin(), v.end(), cmp);
	
	int len = v.size();
	vector<int> ans[55];
	for (int i=0; i < K && i < len; i++) {
		ans[v[i].type].push_back(v[i].id);
	}
	
	for (int i=0; i < m; i++) {
		sort(ans[i].begin(), ans[i].end());
	}
	
	for (int i=0; i < m; i++) {
		int count = ans[i].size();
		if (count == 0) {
			printf("-1\n");
			continue;
		}
		for (int j=0; j < count; j++) {
			printf("%d ", ans[i][j]);
		}
		printf("\n");
	}
}

int main() {
	scanf("%d%d", &m, &n);
	for (int i=0; i < n; i++) {
		Good temp;
		scanf("%d%d", &temp.id, &temp.score);
		for (int j=0; j < m; j++) {
			insert(j, temp);
		}
	}
	
//	for (int i=0; i < m; i++) {
//		list<Good>::iterator it = good_types[i].begin();
//		list<Good>::iterator end = good_types[i].end();
//		while (it != end) {
//			printf("=== %d %d %d\n", it->type, it->id, it->score);
//			it++;
//		}
//	}
	
	int op_num;
	scanf("%d", &op_num);
	for (int i=0; i < op_num; i++) {
		int choose, type, id;
		Good temp;
		scanf("%d", &choose);
		switch (choose) {
			case 1:
				scanf("%d%d%d", &type, &temp.id, &temp.score);
				insert(type, temp);
				break;
			case 2:
				scanf("%d%d", &type, &id);
				del(type, id);
				break;
			case 3:
				int K;
				int ks[55];
				scanf("%d", &K);
				for (int j=0; j < m; j++) {
					scanf("%d", &ks[j]);
				}
				find(K, ks);
				break;
		}
	}
}