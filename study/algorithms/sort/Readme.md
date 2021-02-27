# Sort

## 概述

在线性结构中，查找一个数的时间复杂度为 $O(n)$ 。

但是，如果线性结构中的数是有序排列的，那么就可以使用**二分查找法**，时间复杂度可缩减至 $O(logn)$ 。

因此，排序成为算法中的一个重要问题。

## 内部排序之八大排序算法

内部排序是指所有数据可放入**存储器（内存）**中进行的排序，不考虑磁盘读写所带来的性能问题。

注：以下都是有小到大排序。

### 1. 插入排序

i从`1`到`n-1`，每次将第i位的数插入到前面已排好序的序列中。

插入的方法为：将i与i-1位置的数比较，i小则交换并i--，i大则停止插入。

```c
// 插入排序
void InsertSort(int *num, int n, int start, int grep)
{
    // 对第i个数往前面已排好序的数组插入
    for (int i=start; i<n; i+=grep)
    {
        // 当j位置数小于j-1位置数时，交换；若大于则不交换
        for (int j=i; j>=0+grep && num[j]<num[j-grep]; j-=grep)
        {
            int temp = num[j];
            num[j] = num[j-grep];
            num[j-grep] = temp;
        }
    }
}
```

### 2. 希尔排序

将以5 3 1为增量的序列依次进行插入排序。例如，增量为5时，对(a0,a5...)、(a1,a6...)、(a2,a7...)、(a3,a8...)、(a4,a9...)序列依次进行插入排序。

```c
// 希尔排序
void ShellSort(int *num, int n)
{
    // 对每隔5个数的序列排序，一共有5个这样序列 ，起始点分别为0,1,2,3,4
    for (int i=0; i<5; i++) InsertSort(num, n, i, 5);
    // 对每隔3个数的序
    for (int i=0; i<3; i++) InsertSort(num, n, i, 3);
    InsertSort(num, n, 0, 1);
}
```

### 3. 冒泡排序

每次从0到i序列的数中挑取最大的数放在i位置，i从`n-1`到`0`。

挑取最大数的过程为：将j位置数与j+1位置比较，j大则交换，小不交换，j++。

```c
// 冒泡排序
void BubbleSort(int *num, int n)
{
    for (int i=0; i<n; i++)
    {
        bool allSort = true;
        for (int j=1; j<n-i; j++)
        {
            if (num[j] < num[j-1])
            {
                int temp = num[j];
                num[j] = num[j-1];
                num[j-1] = temp;
                allSort = false;
            }
        }
        // 如果遍历一次所有已经排好序，则不需要再排序了
        if (allSort) break;
    }
}
```

### 4. 快速排序

遍历一次数组，以第一个数为标准，分成三部分，实现：第一个数在中间，小于它的数在它前面，大于它的数在它后面。然后对前部分和后部分递归该操作，递归终点为数组中个数为1。此处要注意的是，中间部分的数不需要再递归操作。

快排核心：历一遍数组，空间复杂度为`O(1)`，将数组分为三部分的方法：

- 1）取出第一个数暂存，设定`i`和`j`分别指向数组头和尾，`pos`指向空位置；
- 2）`j--`，直到比第一个数小，取出`j`位置数放到`pos`位置，`pos=j`；
- 3）`i++`，直到比第一个数大，取出`i`位置数放到`pos`位置，`pos=i`。
- 其中，要时刻保持`i<j`。

```c
// 快速排序
void QuickSort(int *num, int n)
{
    // 递归终止条件
    if (n <= 1) return;

    int i = 0, j = n-1;
    int temp = num[i];  // 将i位置的数先存起来，i位置现在相当于“空”
    while (i<j)
    {
        while (num[j] >= temp && j>i) j--;
        // 此处直接将j位置不合格的数放在i位置就可以了，因为i位置本来就是“空的”
        num[i] = num[j];
        while (num[i] <= temp && i<j) i++;
        num[j] = num[i];
    }
    num[i] = temp;

    QuickSort(num, i);
    QuickSort(num+i+1, n-i-1);  // 此处不用再把pos位置的数传进去排序了！！
}
```

### 5. 选择排序

每次从i到n-1个数中选最小的数放在i位置，i从0到n-1。

```c
// 选择排序
void SelectSort(int *num, int n)
{
    for (int i=0; i<n; i++)
    {
        int min = num[i], pos;
        for (int j=i+1; j<n; j++)
        {
            if (num[j] < min)
            {
                min = num[j];
                pos = j;
            }
        }
        num[pos] = num[i];
        num[i] = min;
    }
}
```

### 6. 堆排序

堆排序，又称完全二叉树排序。是将数组看作一颗完全二叉树，数组`i`位置的左右孩子分别为`2*i+1`和`2*i+2`位置。

排序过程为：

- 1）建堆：从最后一个往第一个，逐个进行“向下筛选”，即建成最大顶堆；
- 2）排序：将堆顶（堆中的最大值）取出，将最后一个结点放置堆顶，然后再对第一个结点（堆顶）进行“向下筛选”；

向下筛选：当结点两个孩子的值比该结点的值大时，就交换结点与孩子位置。然后，对交换后的结点重复操作，直到两个孩子的值都比该结点小，或孩子为空。

```c
// 堆排序中的筛选算法
void HeapAdjust(int *num, int n, int pos)
{
    int lchild = 2*pos+1, rchild = 2*pos+2; // 左孩子和右孩子的位置
    int maxPos;
    while (pos<n && lchild<n)
    {
        // 找到孩子中最大那个孩子
        if (num[lchild] >= num[rchild]) maxPos = lchild;
        else if (rchild < n) maxPos = rchild;

        // 如果筛选点小于孩子中最大的，则筛下去继续操作
        if (num[pos] < num[maxPos])
        {
            int temp = num[maxPos];
            num[maxPos] = num[pos];
            num[pos] = temp;

            pos = maxPos;
            lchild = 2*pos+1, rchild = 2*pos+2;
        }
        else break;
    }
}

// 堆排序（完全二叉树排序）
void HeapSort(int *num, int n)
{
    // 从底往顶逐个筛选，建成小顶堆
    for (int i=n-1; i>=0; i--)
    {
        HeapAdjust(num, n, i);
    }

    // 交换第一个和最后一个，再对第一个筛选
    for (int i=n-1; i>0; i--)
    {
        int temp = num[0];
        num[0] = num[i];
        num[i] = temp;
        HeapAdjust(num, i, 0);
    }
}
```

### 7. 归并排序

对于一段数组先分两半，各自进行归并排序，再将两半部分内容有序地合并起来，此时只需各遍历一次即可，采用递归实现。递归的终结点是数组中只有一个数时则不需要进行上述操作。

```c
// 归并排序
void MergerSort(int *num, int n)
{
    // 递归终止条件
    if (n == 1) return;

    int mid = n/2;
    // 对前半部分归并排序
    MergerSort(num, mid);
    // 对后半部分归并排序
    MergerSort(num+mid, n-mid);

    // 将排好序两部分合并到temp数组
    int *temp = (int*)malloc(sizeof(int) * n);
    int i = 0, j = mid, k = 0;
    while (i<mid && j<n)
    {
        if (num[i] <= num[j])
        {
            temp[k] = num[i];
            i++;
        } else
        {
            temp[k] = num[j];
            j++;
        }
        k++;
    }
    // i指针未到中间
    while (i<mid)
    {
        temp[k] = num[i];
        k++;
        i++;
    }
    // j指针未到结尾
    while (j<n)
    {
        temp[k] = num[j];
        k++;
        j++;
    }

    // 将排序好的temp数组赋给num
    for (int i=0; i<n; i++) num[i] = temp[i];
    free(temp);
}
```

### 8. 基数排序

在所有要排序的数据中找到最大的数，求出它的位数n。

建立编号从0到9的十个队列。

i从最高位n到最低位1。

遍历数组将第i位为0的依次放入编号为0的队列中，第i位为1的放入编号为1的队列中。。。

所有数入相应队列后，再依次将0到9编号队列中数取出放回数组中。

然后对低一位重复该操作。

```c
// 基数排序
void RadixSort(int *num, int n, int limit)
{
    queue<int> radix[10];

    // i是余数
    for (int i=1; i<limit; i*=10)
    {
        for (int j=0; j<n; j++)
        {
            radix[num[j]/i%10].push(num[j]);
        }
        int j=0;
        for (int k=0; k<10; k++)
        {
            while (!radix[k].empty())
            {
                num[j] = radix[k].front();
                radix[k].pop();
                j++;
            }
        }
    }
}
```

### 9. 八大排序时间复杂度比较

此图取自网上。

总体而言，快排、堆排、归并排、希尔排（shell）、桶排（基数）是相对较快的排序算法。但，其中快排和归并排编写相对简单，而快排又比归并排空间复杂度小。所以，快排应是上上之选。

![八大排序时间复杂度比较](./img/1.png)
