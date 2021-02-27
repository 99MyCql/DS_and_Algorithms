#include<stdio.h>
int main()
{
	int num[1000];
	int res[1000];
	int n=0;
	scanf("%d",&n);
	for(int i=0;i<n;i++)
		scanf("%d",&num[i]);
	for(int i=0;i<n;i++)
	{
		if(i==0) printf("%d ",(num[0]+num[1])/2);
		else if(i==n-1) printf("%d ",(num[n-2]+num[n-1])/2);
		else printf("%d ",(num[i-1]+num[i]+num[i+1])/3);
	}
	return 0;
}
