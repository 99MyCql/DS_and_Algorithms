/*
csp 201712-4	行车路线
[问题描述]
　　小明和小芳出去乡村玩，小明负责开车，小芳来导航。
　　小芳将可能的道路分为大道和小道。大道比较好走，每走1公里小明会增加1的疲劳度。
	小道不好走，如果连续走小道，小明的疲劳值会快速增加，连续走s公里小明会增加s2的疲劳度。
　　例如：有5个路口，1号路口到2号路口为小道，2号路口到3号路口为小道，3号路口到4号路口为大道，
	4号路口到5号路口为小道，相邻路口之间的距离都是2公里。如果小明从1号路口到5号路口，
	则总疲劳值为(2+2)2+2+22=16+2+4=22。
	现在小芳拿到了地图，请帮助她规划一个开车的路线，使得按这个路线开车小明的疲劳度最小。
[基本要求]
输入格式：
　　输入的第一行包含两个整数n, m，分别表示路口的数量和道路的数量。路口由1至n编号，
	小明需要开车从1号路口到n号路口。
接下来m行描述道路，每行包含四个整数t, a, b, c，表示一条类型为t，连接a与b两个路口，
长度为c公里的双向道路。
其中t为0表示大道，t为1表示小道。保证1号路口和n号路口是连通的。
输出格式
输出一个整数，表示最优路线下小明的疲劳度。
样例输入
6 7
1 1 2 3
1 2 3 2
0 1 3 30
0 3 4 20
0 4 5 30
1 3 5 6
1 5 6 1
样例输出
76
样例说明
　　从1走小道到2，再走小道到3，疲劳度为5^2=25；然后从3走大道经过4到达5，疲劳度为20+30=50；
最后从5走小道到6，疲劳度为1。总共为76。
课程设计要求：
（1）要求从文本文件中输入；
（2）采用适当的数据结构存储由输入数据中的道路所形成的图结构；
（3）编写尽可能优的算法，处理好连续走小道造成的疲劳值的指数增长
（提示：基于迪杰斯特拉算法进行改进即可完成本题）；
（4）除严格按题目要求进行输出以外，还要求输出最优路线的路径，以及从出发点到各个点的最小疲劳值。
*/
#include<cstdio>
#include<cstdlib>
#include<iostream>
#include<stack>

using namespace std;

#define LL long long

typedef struct 
{
	char type;	// 1为小道，0为大道 
	LL length;	// 边的权重，0表示不相邻 
}Arc;

typedef struct 
{
	LL pre;		// 最短路径的上一个点 
	LL length;		// 到该点(数组下标为对应序号的点)到起始点的最短路径长度 
	bool final;		// 是否在最小路径中 
}Djs;


LL n = 0, m = 0;	// 点个数和边个数 
Arc **graph;		// 邻接矩阵 
Djs *dj = NULL;		// 即迪杰斯特拉执行数组 
LL **flo = NULL;	// 记住每两个点之间都为小路的通路 
LL *order = NULL;	// order数组按照点到起点最短路径由小到大排序 


// 初始化函数 
void Init()
{
	scanf("%d%d", &n, &m);
	
	// 初始化图和佛洛依德图 
	graph = (Arc**)malloc(sizeof(Arc*) * (n+1));
	flo = (LL**)malloc(sizeof(LL*) * (n+1));
	for (LL i=0; i<=n; i++) 
	{
		graph[i] = (Arc*)malloc(sizeof(Arc) * (n+1));
		flo[i] = (LL*)malloc(sizeof(LL) * (n+1));
	}
	for (LL i=0; i<=n; i++) 
	{
		for (LL j=0; j<=n; j++) 
		{
			graph[i][j].type = 0;
			graph[i][j].length = 0;
			flo[i][j] = 0;
		}
	} 
	
	//初始化order
	order = (LL*)malloc(sizeof(LL) * (n+1));
	for (LL i=1; i<=n; i++)
	{
		order[i] = 0;
	}
	
	// 根据数据给图赋值
	LL x = 0, y = 0, type = 0, length = 0;
	for (LL i=1; i<=m; i++) 
	{
		scanf("%d%d%d%d", &type, &x, &y, &length);
		graph[x][y].type = type;
		graph[x][y].length = length;
		graph[y][x] = graph[x][y];
		if (type)
		{
			flo[x][y] = length;
			flo[y][x] = length;
		}
	}
	
	// 初始化迪杰斯特拉算法数据结构并赋值 
	dj = (Djs*)malloc(sizeof(Djs) * (n+1));
	for (LL i=1; i<=n; i++)
	{
		dj[i].final = false;
		dj[i].pre = 1;
		if (graph[1][i].type) 
		{
			dj[i].length = graph[1][i].length * graph[1][i].length;
		} 
		else 
		{
			dj[i].length = graph[1][i].length;
		}
	}
	dj[1].final = true;
	order[1] = 1;
	
	// 求出flo
	for (LL k=1; k<=n; k++) 
	{
		for (LL i=1; i<=n; i++)
		{
			for (LL j=1; j<=n; j++)
			{
				if (i == j) continue;
				if ((flo[i][j] && flo[i][k] && flo[k][j] && flo[i][k]+flo[k][j]<flo[i][j])
					|| (!flo[i][j] && flo[i][k] && flo[k][j]))
				{
					flo[i][j] = flo[i][k]+flo[k][j];
				}
			}
		}
	}
}

// 结束函数
void Finish() 
{
	for (LL i=0; i<=n; i++) 
	{
		free(graph[i]);
	}
	free(graph);
	
	for (LL i=0; i<=n; i++) 
	{
		free(flo[i]);
	}
	free(flo);
	
	free(dj);
	
	free(order);
} 

// 辅助函数，求某点到起始点的长度 
LL getLength(LL cur, LL pre) 
{
	LL length = INT_MAX;
	//cout<<"====="<<pre<<"===="<<cur<<"===="<<(LL)graph[pre][cur].type<<"====="<<endl;
	 
	// 如果当前点到上一个点之间是小道 
	if (graph[pre][cur].type) 
	{
		LL sum = INT_MAX;
		// 遍历pre点的所有前点中为大路的 
		for (LL i=1; i<=n; i++)
		{
			if (dj[i].final && graph[i][pre].length && !graph[i][pre].type)
			{
				sum = graph[pre][cur].length * graph[pre][cur].length +  graph[i][pre].length + dj[i].length;
				if (sum < length) length = sum;
			}
		}
		//cout<<"1---->"<<length<<endl;
		
		// 遍历所有以pre点为终点，且通路上都为小路的始点。即flo佛洛依德矩阵中pre点的上一个点。遍历次序按照到起点最短路径由小到大的点 
		for (LL i=1; i<=n; i++)
		{
			if (dj[order[i]].final && flo[order[i]][pre])
			{
				LL smallLen = graph[pre][cur].length + flo[order[i]][pre];
				sum = smallLen * smallLen + dj[order[i]].length;
				if (sum < length) length = sum;
				break;
			}
		}
		//cout<<"2---->"<<length<<endl;
	}
	// 如果当前点到上一个点之间是大道 
	else
	{
		length = graph[pre][cur].length + dj[pre].length;
	}
	
	//cout<<"======="<<length<<"======="<<endl;
	return length;
}

// 根据迪杰斯特拉求最短路径 
void FindDjs() 
{
	// 2到n遍历n-1次，因为起始点已经为true 
	for (LL i=2; i<=n; i++) 
	{
		LL min = INT_MAX, minPos = i;
		for (LL j=1; j<=n; j++) 
		{
			if (dj[j].length && !dj[j].final && dj[j].length<min) 
			{
				min = dj[j].length;
				minPos = j;
			}
		}
		dj[minPos].final = true;
		order[i] = minPos;
		
		// 更新每个点的最短路径 
		for (LL j=1; j<=n; j++) 
		{
			// （1）dj[j].final为false，说明这个点还未得到最短路径 
			// （2）graph[minPos][j].length不为0，则说明j位置顶点跟这个新点相邻
			//  满足上述两个条件，则需要判断j点是否需要更新 
			if (!dj[j].final && graph[minPos][j].length) 
			{
				// 当前一个点是newPos时，获取pos位置点的新路径(到起点) 
				LL newLength = getLength(j, minPos);
				// 如果pos位置点还未存在最短路径 
				if (!dj[j].length)
				{
					dj[j].length = newLength;
					dj[j].pre = minPos;
				}
				// 如果pos位置点有最短路径 
				else
				{
					if (newLength < dj[j].length) 
					{
						dj[j].length = newLength;
						dj[j].pre = minPos;
					}
				}
			}
		}
	}
}

int main() {
	Init();
	FindDjs();
	printf("%d\n", dj[n].length);
	Finish();
}
