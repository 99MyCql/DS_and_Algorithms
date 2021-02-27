/*
给定一些标记了宽度和高度的信封，宽度和高度以整数对形式 (w, h) 出现。当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。

请计算最多能有多少个信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。

说明:
不允许旋转信封。

示例:

输入: envelopes = [[5,4],[6,4],[6,7],[2,3]]
输出: 3 
解释: 最多信封的个数为 3, 组合为: [2,3] => [5,4] => [6,7]。
*/
#include<cstdio>
#include<cstdlib>
#include<vector>
#include<algorithm>

using namespace std;


bool cmp(vector<int> a, vector<int> b) {
    if ((a[0] > b[0]) && (a[1] > b[1])) return false;
    else return true;
}

class Solution {
public:
    vector<int> dp;
    int len;

    int getMax(vector<vector<int>>& envelopes, int n) {
        if (dp[n] > -1) return dp[n];

        bool flag = false;
        int i = n-1;
        while (i >= 0) {
            if (envelopes[i][0] < envelopes[n][0] && envelopes[i][1] < envelopes[n][1]
                && (getMax(envelopes, i)+1 > dp[n])) {
                dp[n] = getMax(envelopes, i) + 1;
                flag = true;
            }
            i--;
        }
        if (!flag) dp[n] = 1;
        return dp[n];
    }

    void quick_sort(vector<vector<int>>& envelopes, int start, int end) {
        // printf("start=%d, end=%d\n", start, end);
        if (start >= end) return;

        int i = start+1, j = start+1 ;
        for (; j < end; j++) {
            if (envelopes[j][0] < envelopes[start][0] && envelopes[j][1] < envelopes[start][1]) {
                vector<int> temp;
                temp = envelopes[i];
                envelopes[i] = envelopes[j];
                envelopes[j] = temp;
                i++;
            }
        }
        // printf("i=%d, j=%d\n", i, j);
        vector<int> temp;
        temp = envelopes[i-1];
        envelopes[i-1] = envelopes[start];
        envelopes[start] = temp;
        quick_sort(envelopes, start, i-1);
        quick_sort(envelopes, i, j);
    }

    int maxEnvelopes(vector<vector<int>>& envelopes) {
        len = envelopes.size();

        // sort(envelopes.begin(), envelopes.end(), cmp);
        quick_sort(envelopes, 0, len);
        for (int i = 0; i < len; i++) {
            printf("(%d %d) ", envelopes[i][0], envelopes[i][1]);
        }
        printf("\n");

        for (int i = 0; i < len; i++) {
            dp.push_back(-1);
        }

        int max = 0;
        for (int i = 0; i < len; i++) {
            printf("%d ", getMax(envelopes, i));
            if (dp[i] > max) {
                max = dp[i];
            }
        }

        return max;
    }
};
