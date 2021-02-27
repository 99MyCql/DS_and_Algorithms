/*
3. 机器人大冒险
力扣团队买了一个可编程机器人，机器人初始位置在原点(0, 0)。小伙伴事先给机器人输入一串指令command，机器人就会无限循环这条指令的步骤进行移动。指令有两种：

U: 向y轴正方向移动一格
R: 向x轴正方向移动一格。
不幸的是，在 xy 平面上还有一些障碍物，他们的坐标用obstacles表示。机器人一旦碰到障碍物就会被损毁。

给定终点坐标(x, y)，返回机器人能否完好地到达终点。如果能，返回true；否则返回false。

示例 1：

输入：command = "URR", obstacles = [], x = 3, y = 2
输出：true
解释：U(0, 1) -> R(1, 1) -> R(2, 1) -> U(2, 2) -> R(3, 2)。
示例 2：

输入：command = "URR", obstacles = [[2, 2]], x = 3, y = 2
输出：false
解释：机器人在到达终点前会碰到(2, 2)的障碍物。
示例 3：

输入：command = "URR", obstacles = [[4, 2]], x = 3, y = 2
输出：true
解释：到达终点后，再碰到障碍物也不影响返回结果。

限制：

2 <= command的长度 <= 1000
command由U，R构成，且至少有一个U，至少有一个R
0 <= x <= 1e9, 0 <= y <= 1e9
0 <= obstacles的长度 <= 1000
obstacles[i]不为原点或者终点
*/
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <map>

using namespace std;

class Solution {
public:
    bool robot(string command, vector<vector<int>>& obstacles, int x, int y) {
        // 设置陷阱位
        map<vector<int>, bool> Map;
        int obstacles_len = obstacles.size();
        for (int i=0; i < obstacles_len; i++) {
            if (obstacles[i][0] > x || obstacles[i][1] > y) continue;
            Map[obstacles[i]] = true;
        }

        int command_len = command.size();
        vector<int> pos;
        pos.push_back(0); // x
        pos.push_back(0); // y
        for (int i=0; ; i++) {
            // printf("%d\n", i);
            if (command[i%command_len] == 'U') pos[1]++;
            if (command[i%command_len] == 'R') pos[0]++;

            // printf("%d %d\n", pos[0], pos[1]);
            // 如果到终点
            if (pos[0] == x && pos[1] == y) return true;
            // 遇到陷阱或走出去
            if (pos[0] > x || pos[1] > y || Map[pos]) break;
        }
        
        return false;
    }
};