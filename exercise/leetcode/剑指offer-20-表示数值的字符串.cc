/*
请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100"、"5e2"、"-123"、"3.1416"、"0123"及"-1E-16"都表示数值，但"12e"、"1a3.14"、"1.2.3"、"+-5"及"12e+5.4"都不是。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

#include <regex>
#include <string>

using namespace std;

class Solution {
public:
    bool isNumber(string s) {
        s = regex_replace(s, regex("(\\s*$)|(^\\s*)"), "");
        printf("\"%s\"", s.c_str());
        if (s.size() == 0) return false;
        if (regex_match(s, regex("(\\+|-)?((\\d+)|(\\.\\d+)|(\\d+\\.\\d*))((e|E)(\\+|-)?\\d+)?"))) {
            return true;
        } else {
            return false;
        }
    }
};