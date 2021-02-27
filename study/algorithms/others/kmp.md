# 字符串匹配 - KMP算法

KMP是一种模式匹配算法，什么是模式匹配呢？就是，现有一个**模式串（字符串）**，要在另一个主串中查找是否存在与模式串相等的子串（主串中任意连续字符组成的子序列）。

该算法的用处广泛，是正则表达式的基础。

例如，有一个字符串为`ababcabcacbab`，我们想知道该字符串中有没有一个`abcac`的字符串，这就叫模式匹配。

## 1. 普通模式匹配算法

经典的模式匹配，是让模式串的首字符从主串的首字符开始向右匹配。

匹配的规则是：

1. 在主串的第i个位置，让pos=i，如果主串pos位置的字符等于模式串的第一个字符，则pos++，再与模式串的第二个字符进行匹配。
2. 如果又相等则继续向下匹配，直到匹配到模式串的结尾，就说明主串中存在该模式串；如果不相等，则停止匹配，回到主串中，并i++。
3. 然后再让pos=i，重复上述步骤，直到i到达主串结尾。

相应代码如下：

```c
#include<stdio.h>
#include<string.h>

int main() {
    char str[10000],pat[10000];
    printf("please push the main string:\n");
    scanf("%s",str);
    printf("please push the pattern:\n");
    scanf("%s",pat);
    int lenS=strlen(str),lenP=strlen(pat);

    // 从主串第一位置开始向后匹配，直到结尾
    for(int i=0;i<lenS;i++) {
        int pos=i,j=0;
        while(j<lenP) {
            // 如果主串pos位置与模式串的j位置相等，则都向后一位
            if(pat[j]==str[pos])
            {
                j++;pos++;
            }
            // 否则中断与模式串的匹配
            else break;
        }
        // 判断与模式串是否完全匹配，若是则输出i位置
        if(pos-i==lenP) printf("\npattern start in %d",i);
    }
}
```

经典模式匹配算法有一个缺陷就是，每次匹配失败后都要回退到主串中i的位置。

例如：主串为`000000000000000000001`，模式串为`0001`时，匹配正确之前的每次匹配都要到模式串的最后一个字符，才会回退，这其中浪费的时间不容小觑。

那有没有一种算法是不需要回退的呢？

## 2. KMP模式匹配算法

KMP算法的好处就是不用回退，而是以一种**滑动**的方式向下匹配模式串。

### 假设next数组

首先，我们假设对于**模式串**有一个`next`数组。

该数组值的含义是：模式串第`0`位后长度为`next[j]`的字符串，与第`j`位（从0开始，除0外）前长度为`next[j]`的字符串**相等**，同时`next[j]`值要竟可能**大**，但两个字符串又**不能重合**。特例：第`0`位的`next`值为`-1`。

例如：`abcac`模式串的`next`数组值依次为`[-1,0,0,0,1]`。模式串中第4位字符`c`的`next`值为1，表明字符`c`前长度为1的字符串即`a`，与从第0位开始长度为1的字符串即`a`相等。其余位，除第0位外，`next`值都为0。

再例如：`0001`模式串的`next`数组值依次为`[-1,0,1,2]`。

### 利用next数组进行模式匹配

在普通模式匹配中，主串`pos`位与模式串`j`位不匹配时，主串回到`i+1`位重新与模式串`0`位进行匹配。

有了`next`数组，我们便不用回退到`i+1`位重新匹配了。而是用主串`pos`位与模式串`next[j]`位进行匹配，若还是不匹配，则转到`next[next[j]]`。直到`next`的值等于`-1`，模式串滑动到主串`pos+1`位重新匹配。

其实，当不匹配求`next[j]`值时，就是将模式串向下滑动的过程。其原理是因为`next`数组的特殊性：

> 模式串第`0`位后长度为`next[j]`的字符串，与第`j`位(从0开始，除0外)前长度为`next[j]`的字符串**相等**。

### 详细代码

```c
#include<stdio.h>
#include<string.h>
#include<stdlib.h>

// 获取模式串的next数组函数：
// 将模式串既当作“主串”又当作模式串，同样按照KMP的算法进行匹配求next值。
void getNext(char *str,int *next,int n) {

    next[0]=-1;
    int i=1;
    int j=-1;
    while (i<n) {
        // 如果j==-1，则“主串”i位置的next值为0。
        // 前进到“主串”的下一位，并使模式串从头开始匹配
        if (j == -1) {
            next[i]=0;
            i++;
            j=0;
        }
        // 如果“主串”i位置的前一位和模式串j位置相同，则“主串”i位置的next值为j+1。
        // 同时“主串”与模式串继续向下匹配。
        else if (str[i-1] == str[j]) {
            next[i]=j+1;
            i++;
            j++;
        }
        // 如果不匹配，则前进到模式串的next[j]位置
        else j=next[j];
    }
}

int main() {

    char str[10000],pat[10000];
    printf("please push the main string:\n");
    scanf("%s",str);
    printf("please push the pattern:\n");
    scanf("%s",pat);
    int lenS=strlen(str),lenP=strlen(pat);

    // next数组：当模式串第j位与主串当前第i位不匹配时，将模式串向后滑动，
    //          用模式串中第next[j]位再与主串中的第i位比较。
    //          若next[j]==-1，则将模式串整体向后移动一位，与主串中第i+1位继续比较。
    int *next=(int*)malloc(sizeof(int)*lenP);
    getNext(pat, next, lenP);

    int i=0,j=0;    // i为主串中字符位置，j为模式串中字符位置
    while (i<lenS) {
        // 如果j==-1，则模式串整体向后移动一位，与主串中第i+1位比较
        if (j == -1) {
            i++;
            j=0;
        }
        // 如果主串i位置与模式串j位置相等，则i、j都向下继续配对
        else if (str[i] == pat[j]) {
            i++;
            j++;
            if (j == lenP) {
                printf("find pattern in %d\n", i-lenP);
                j=0;
            }
        }
        // 如果模式串j位置与主串i位置不匹配，则将模式串向后滑动，
        // 用模式串的next[j]位置与主串i位置匹配
        else j=next[j];
    }
    free(next);
    return 0;
}
```
