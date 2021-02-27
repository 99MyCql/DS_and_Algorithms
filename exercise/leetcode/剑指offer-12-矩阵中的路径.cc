/*
请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。

[["a","b","c","e"],
["s","f","c","s"],
["a","d","e","e"]]

但矩阵中不包含字符串“abfb”的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。

 

示例 1：

输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
示例 2：

输入：board = [["a","b"],["c","d"]], word = "abcd"
输出：false
提示：

1 <= board.length <= 200
1 <= board[i].length <= 200

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    vector<vector<bool>> flag; // 二维矩阵标志位
    int len;        // 行长度
    int row_len;    // 列长度

    // 初始化标志位
    void initFlag() {
        for (int i=0; i < len; i++) {
            vector<bool> temp;
            for (int j=0; j < row_len; j++) {
                temp.push_back(false);
            }
            flag.push_back(temp);
        }
    }

    // 深度遍历
    bool dfs(vector<vector<char>>& board, int i, int j, string word, int word_pos) {
        if (word_pos >= word.size()) {
            return true;
        }

        // 判断符不符合标准
        if (i >= 0 && i < len && j >= 0 && j < row_len
            && !flag[i][j] && board[i][j] == word[word_pos]) {
            flag[i][j] = true;
            // 将上下左右格子进行深度遍历
            if (dfs(board, i-1, j, word, word_pos+1)
                || dfs(board, i+1, j, word, word_pos+1)
                || dfs(board, i, j-1, word, word_pos+1)
                || dfs(board, i, j+1, word, word_pos+1)) {
                return true;
            } else {
                flag[i][j] = false;
            }
        }

        return false;
    }

    // 解题接口
    bool exist(vector<vector<char>>& board, string word) {
        len = board.size();
        row_len = board[0].size();
        initFlag();

        for (int i=0; i < len; i++) {
            for (int j=0; j < row_len; j++) {
                if (dfs(board, i, j, word, 0)) {
                    return true;
                }
            }
        }
        return false;
    }
};