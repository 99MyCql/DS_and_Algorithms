#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>

using namespace std;

class Solution {
public:
    int months[13] = {
        0,
        31,
        28, // 2月，平年28，闰年29
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31
    };

    // 是否闰年
    bool isRun(int year) {
        // 普通年能被4整除且不能被100整除的为闰年。(如2004年就是闰年,1900年不是闰年)
        if ((year % 4 == 0) && (year % 100 != 0)) return true;
        else return false;
    }

    int dayOfYear(string date) {
        char *date_s = date.data();
        cout << date_s << endl;
        char *year_s = strtok(date_s, "-");
        char *month_s = strtok(NULL, "-");
        char *day_s = strtok(NULL, "-");
        cout << year_s << month_s << day_s << endl;
        int year, month, day;
        sscanf(year_s, "%d", &year);
        sscanf(month_s, "%d", &month);
        sscanf(day_s, "%d", &day);
        cout << year << month << day << endl;

        int ans = 0;
        for (int i=1; i < month; i++) {
            if (i == 2 && isRun(year)) {
                ans += 29;
            }
            else {
                ans += months[i];
            }
        }
        ans += day;

        return ans;
    }
};